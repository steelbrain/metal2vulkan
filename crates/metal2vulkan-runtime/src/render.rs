//! `RenderCommandEncoder` (`MTLRenderCommandEncoder`) and `CommandBuffer::render_command_encoder`.
//!
//! - **The deferred begin**. Opening the encoder checks its descriptor (`render_pass.rs`) and
//!   begins a secondary command buffer against a pass compatible with it (`render_target.rs`);
//!   every draw records there. A pass whose first instance may record inline begins none at its
//!   opening, and one only where a command records into that instance or a split keeps it
//!   (`render/inline_first.rs`). `end_encoding`, when every store action is known, makes the pass
//!   and framebuffer with the attachments' load and store operations, and records into the
//!   encoder's own command buffer the pass's begin, the draws, and its end. Separate depth and
//!   stencil textures render through a combined scratch image, copied in before the pass and back
//!   after it (`depth_stencil_scratch.rs`).
//! - **Draws** follow a compute dispatch's steps: Metal's checks (a pipeline state set; the state's
//!   formats and samples were held to the pass's when it was set, as Metal's validation holds
//!   them); the vertex and fragment tables as a `bind::BindState`; the state's plan and variant for
//!   its shape; `instantiate_draw`; the descriptors, uploads and push bytes (`binding.rs`); the
//!   `VkPipeline` of the draw's graphics key (`pipeline/graphics.rs`), against the compatible pass.
//!   Then, and only then, the draw records: the pipeline, sets, push constants, vertex and index
//!   buffers, every state the pipeline makes dynamic at the encoder's value, and the draw. A
//!   refused draw records nothing and changes nothing; a draw of no vertices, indices, patches,
//!   instances or mesh threadgroups records nothing either, as plain Metal draws nothing there,
//!   where its validation stops a draw of no instances or patches (Metal probe cases `draw_zero`,
//!   `zero_instances` and `zero_patches` on an M4). Such an empty draw is checked only as plain
//!   Metal checks one (an M4): without a pipeline state, with a tile pipeline's,
//!   with a state of another kind (patches with a drawing state, primitives with a mesh state) and
//!   a patch draw without a tessellation factor buffer it stops the process, so those stay refused;
//!   past them plain Metal draws nothing whatever the encoder's bindings and amplification count,
//!   and so does the encoder, which checks nothing more (a mesh draw of no threadgroups with a
//!   drawing state included). So does a draw of primitives with a tessellation pipeline, and one
//!   whose amplification count is 0. A primitive type outside the pipeline's input primitive
//!   topology class, which Metal's validation stops, draws as plain Metal draws it: its own
//!   primitives, points rasterizing nothing unless sized.
//! - **Misaligned offsets**. Metal's validation stops a draw whose index buffer offset is not a
//!   multiple of the index size, whose indirect arguments' offset or patch index buffer offset is
//!   not a multiple of 4; plain Metal reads each from the offset rounded down to that multiple
//!   (Metal probe cases `ind_offset_misaligned` and `ind_indexed_offset_misaligned` on an M4; a
//!   probe for control-point indices), as the draws here do.
//! - **Encoder state** starts at Metal's default render encoder state, and its setters take and
//!   refuse what Metal's do (`render_state.rs`, measured): viewports and scissor rectangles (the
//!   viewport flipped, `y + height` and `-height`, so framebuffer rows, `FragCoord` and
//!   derivatives match Metal's y-up clip space; each scissor clipped to the pixels its viewport
//!   overlaps), culling and winding, the depth/stencil state (`depth_stencil.rs`) and stencil
//!   references, depth bias, the depth clip and triangle fill modes, and the blend color. A state
//!   the pipeline keeps static is in its graphics key; one it makes dynamic is recorded at each
//!   draw.
//! - **The hidden depth and stencil**. A draw whose depth/stencil state uses depth or the stencil
//!   the pass has no attachment for, which Metal's validation stops, is tested as plain Metal tests
//!   it, against a hidden `float` depth of 1.0 and stencil of 0 from the pass's start: it gives the
//!   pass an image of its own for them (`attachment_stand_in.rs`), cleared before the pass, the
//!   depth/stencil attachment of the draw's instance and every later one (instances before it have
//!   none). Metal's attachment setters and pipelines still see a pass without one.
//! - **Ordering.** Metal does not order a render encoder's draws against each other's memory
//!   writes without the program's barriers, so the encoder is an unordered one: each draw's uses
//!   (and the attachments', at `end_encoding`) join its summary, and the scheduler orders them
//!   against other encoders. The program's memory barriers split the pass (`fence.rs`): the next
//!   draw records into a new instance, after a barrier from everything before.
//!
//! - **Index positions past the index buffer**. Metal's validation stops a draw whose indices run
//!   past its index buffer's length; Metal's runtime reads an index's bytes up to the length
//!   rounded up to 4 bytes, and draws each position from there as index 0 plus the base vertex,
//!   bounded by the buffer even inside a heap (measured on an M3 Max and an M4). Vulkan states no
//!   index for a fetch past the bound range, robust or not, so the runtime draws such a direct draw
//!   from a zero-padded copy of its indices on every device (`draw_source.rs`, the
//!   `isolated-index-bytes` degradation): Metal's indices, then index 0 at every position past
//!   them, to which Vulkan adds the base vertex. From an index offset at or past the length,
//!   where Metal reads the memory past the buffer, every position draws index 0
//!   (`zeroed-index-fetch`, a recorded near-equivalent). An indexed indirect draw, whose
//!   positions its arguments choose on the GPU, stays one draw bounded by the device's range only
//!   where the device enables `robustBufferAccess2` and its driver is measured to read zero past
//!   the range (`DriverFact::RobustIndexFetchReadsZero`): a range that ends where Metal's fetch
//!   stops where the device enables `maintenance5`, else a copy of the indices. Elsewhere the
//!   index bounds pass draws it (below). The execution record keeps each direct draw's
//!   `index_padding`.
//! - **Indirect draws** read Metal's arguments, which share Vulkan's layouts
//!   (`MTLDrawPrimitivesIndirectArguments` is a `VkDrawIndirectCommand`, the indexed form a
//!   `VkDrawIndexedIndirectCommand`), straight from the buffer, or from a copy where a no-copy
//!   buffer's bytes put them where Vulkan cannot read them (`draw_source.rs`, which reads index
//!   buffers so too), from the offset rounded down to a multiple of 4 (above). Metal's
//!   validation stops arguments that run past the buffer (measured), where plain Metal reads
//!   past it, and so does the draw here. Where the plan asks for the arguments on the CPU
//!   (`DrawCommands::indirect_needs_arguments`) only because nothing bounds an indexed draw's
//!   index fetch (`DrawCommands::index_bounds_pass`), or where it leaves the fetch to the device's
//!   range on a driver not measured to read zero past it, the index bounds pass draws it on the GPU
//!   (`index_bounds.rs`); for any other reason, and where a draw of the encoder wrote what that
//!   pass reads since the last memory barrier, the draw is `NotImplemented(IndirectDrawArguments)`:
//!   the runtime never reads arguments an earlier command may write. The other exact paths are GPU
//!   passes over the arguments too.
//! - **A draw's steps before the pass** (its copies, its index bounds pass) record before the pass,
//!   unless they read what an earlier draw or tile dispatch of the encoder wrote behind a memory
//!   barrier: those record before the draw's own render pass instance (`Before`), after the
//!   instances before it, and read the write as Metal's draw does, in a pass that executes render
//!   commands too (`render/pass_steps.rs`).
//! - **Visibility results** (`visibility.rs`): occlusion queries begun and ended in the draws'
//!   command buffer, reset before the pass, copied to their offsets after it, and resolved to
//!   Metal's 1 or 0 for `Boolean` ones.
//! - **Resolves** (`resolve.rs`) happen at the end of the pass's subpass; a late store action
//!   must resolve exactly where the attachment has a resolve texture (measured). A color resolve
//!   no device's average writes (an extended-range attachment's, one into its format's sRGB
//!   twin) is drawn after the pass instead, by the crate-owned resolve (`color_resolve.rs`). A
//!   resolve into
//!   a texture whose image is no attachment goes through a scratch image, copied after the pass.
//!   The encoder may replace an attachment's resolve target until the pass ends
//!   (`render/late_resolve.rs`).
//! - **Framebuffer fetch** (`framebuffer_fetch.rs`) through input attachments: a draw's input
//!   attachments are the pass's color attachments, which its subpass reads in rasterization
//!   order where the device's fetch does; the fragment shader's reads join the draw's uses.
//!   Through the pixel interlock: a draw's storage attachments are color attachments its subpass
//!   leaves out, bound as storage views of their level and layer, whose fragment shader reads
//!   and writes join the draw's uses. Either way a fetched attachment whose image lacks the
//!   usage its form needs (input attachment usage, or `STORAGE`) renders, from the draw on and
//!   for the whole pass, into a stand-in that has it (`attachment_stand_in.rs`). In a
//!   multisampled pass the interlock form reads and writes the attachment's sample planes on a
//!   device without multisampled storage images (`sample_planes.rs`): made at the first draw that
//!   fetches the attachment, bound to every such draw as a storage view of every layer, and moved
//!   into and out of the attachment between the pass's render pass instances (each draw that
//!   fetches it leaves it out of its subpass, so the instances say where). An attachment whose
//!   image the planes' copies cannot read and write renders into a stand-in that they can. The
//!   exact resolve reads the planes themselves where they hold the samples as the pass ends, which
//!   then merge back only where Metal keeps the samples; a memoryless such attachment needs no
//!   backing unless the pass splits. A pass whose last instance leaves out an attachment the device
//!   resolves runs one more instance, of no draws, after the planes are merged back: it resolves
//!   the pass, and the instances before it do not. On a device with multisampled storage images the
//!   interlock form reads and writes the attachment's own multisampled image, bound as a storage
//!   view of the pass's samples, as at one sample; its resolve likewise runs in one more instance
//!   after the last that leaves it out.
//! - **Instances.** A draw records into the pass's current render pass instance when the
//!   attachments it leaves out of its subpass (its storage attachments) and its views (below)
//!   are the instance's, and no stand-in (nor the hidden depth and stencil) arrives with it;
//!   otherwise into a new instance of its own shape
//!   (`render_target.rs`): a stand-in with input attachment usage makes its attachment an input
//!   attachment of the draws' subpass, which it was not before. The new instance takes
//!   the current one's place when nothing recorded into it and, for the pass's first, every
//!   attachment it leaves out that Metal clears can be cleared before the pass instead (its
//!   level and layer is the whole render area, cleared by `vkCmdClearColorImage` with the value
//!   the load operation would take; Vulkan converts both alike); otherwise the pass splits, and
//!   the new instance follows the current one after a barrier from everything before to
//!   everything after. A split pass renders its memoryless attachments into stand-ins
//!   (`attachment_stand_in.rs`), since its instances keep their contents in memory. A query
//!   begins and ends in one instance, so a split while a visibility result mode is set ends the
//!   active query in the instance before and begins a continuation of it in the next, which the
//!   results combine (`visibility.rs`). A tile dispatch runs between instances: the instance
//!   before it, recorded or not, runs first, the next draw splits the pass, and a pass ending
//!   with a dispatch runs one more instance, of no draws, for its store actions
//!   (`render/tile_dispatch.rs`).
//! - **The first instance inline** (`render/inline_first.rs`): where the pass's layout and its
//!   first draw leave nothing for the pass's end to change in its first instance, that instance's
//!   draws record inline in a primary command buffer the encoder holds for the pass, its render
//!   pass instance begun at that draw, in place of a secondary; the encoder's commands before the
//!   pass record meanwhile into a command buffer that goes before it. So do the instances of the
//!   same plain shape after a split of such a pass, each after the one before, its commands kept
//!   until its copies are known (`render/inline_splits.rs`).
//! - **The shader-side stencil** (`shadow_stencil.rs`, where the device lacks stencil export): a
//!   draw whose fragment function tests and writes the stencil itself records into an instance of
//!   such draws, which a draw of another pipeline leaves for a new one. Before each such instance
//!   the stencil is copied into the pass's shadow, which the draws read and write as a storage
//!   image with the stencil test off, and after it the shadow is copied back; a pass whose last
//!   instance is one and whose stencil resolves runs one more instance, of no draws, after the copy
//!   back. A memoryless stencil renders into its stand-in there.
//! - **The shader-side depth test** (`depth_stand_in.rs`): a draw whose fragment function runs the
//!   depth test itself against a stand-in of the depth (`GraphicsState::shader_tests`) records into
//!   an instance of such draws, as the shader-side stencil's do, under depth `ALWAYS` without
//!   writes and the stencil test off. Before each such instance the stand-in is filled (from the
//!   depth's load action in the instance that runs them), and after it copied back into the depth;
//!   a pass whose last instance is one and whose depth resolves runs one more instance, of no
//!   draws, after the copy back. A memoryless depth renders into its stand-in there. A draw whose
//!   instance needs a copy in the device cannot run, and such a draw among indirect render
//!   commands, are refused.
//! - **Fragment imageblocks** (`fragment_imageblock.rs`): a draw whose fragment function declares
//!   `[[imageblock_data]]` reads and writes the pass's storage, a 2D-array storage image of the
//!   render area's tiles made at the pass's first such draw or tile dispatch, cleared to zero
//!   before the pass and kept across its instances, in the pixel interlock; a tile function's
//!   explicit imageblock is the same storage (`render/tile_dispatch.rs`). A pipeline of more
//!   imageblock bytes per sample than the pass gives is refused as it is set, as Metal's validation
//!   refuses it; a draw of another master of the pass's cell and alignment views the same bytes, as
//!   Metal's does without its validation; a draw of another cell or alignment than the pass's, and
//!   a pass that also dispatches a tile function whose explicit imageblock is aliased past the
//!   attachments, are `NotImplemented`.
//! - **Tile threadgroup memory** (`render/tile_memory.rs`): a draw whose fragment function reads
//!   its threadgroup arguments binds the pass's buffer of every tile's memory, which the pass's
//!   tile dispatches share; a draw that reaches it after another draw of the current instance
//!   records into a new one, so it sees that draw's writes as Metal's does. An execution's commands
//!   reach it too, splitting their instance into runs (`render/executions.rs`).
//! - **Rate-mapped passes** (`rate_mapped_pass.rs`): a pass over a rasterization-rate map reads its
//!   viewports and scissor rectangles in screen pixels, over the map's screen (or, without
//!   attachments, the render target size), and records each draw once per region of the compile
//!   layer's plan, with the region's viewport and scissor: the plan of the map's layer of the slice
//!   the draw (or each amplification drawn alone) renders, as Metal rasterizes it; a pipeline whose
//!   vertex function chooses its slice over a map whose layers differ is
//!   `NotImplemented(RateMapRegions(LayeredPipeline))`.
//! - **Patch draws** (`tessellation.rs`): a tessellation pipeline (its vertex function a
//!   post-tessellation one) draws patches, and only it does. Each such draw binds the encoder's
//!   factor buffer to the pipeline's control stage and pushes the factors' offset, instance stride
//!   and scale, as the binder plans them; until a scale is set, and while the scale is 0, negative
//!   or NaN as a `half`, a pipeline that enables the scale draws nothing, as Metal culls its
//!   patches (measured). A draw takes the control points the function states, whatever it names, as
//!   plain Metal draws it (Metal probe cases `control_points_4` and `_0` on an M4). Refused as
//!   Metal's validation refuses them (`Invalid`, measured): a draw of patches with another
//!   pipeline, no factor buffer, and the indirect arguments' range as for primitives; the factor
//!   setter an odd offset or instance stride (an offset 2 past a multiple of 4, which Metal's
//!   validation stops, is read from that byte as plain Metal reads it, and such a stride steps the
//!   instances by the multiple of 4 below, as plain Metal steps them). A draw's patch index buffer
//!   binds to the control stage, which reads each patch's index from it (its factors stay at the
//!   patch's position, measured), from the offset rounded down to a multiple of 4 (above); a direct
//!   draw's indices must lie inside the buffer, as Metal's validation requires. A pipeline with a
//!   control-point index type reads each control point's index from the draw's control-point index
//!   buffer, bound to the control stage (measured), from the offset rounded down to a multiple of
//!   the index size; a direct draw's `(patchStart + patchCount) · N` indices must lie inside the
//!   buffer, as Metal's validation requires. A plain patch draw on such a pipeline reads every
//!   index as 0, and an indexed one on a pipeline without a type draws the patches' own control
//!   points, as plain Metal draws them (Metal probe cases `plain_with_type` and
//!   `indexed_without_type` on an M4), where Metal's validation stops both, and their indirect
//!   forms likewise (a Metal probe).
//!
//! - **Vertex amplification** (`amplification.rs`). The encoder's count and view mappings are the
//!   bind state's, held to the pipeline's `maxVertexAmplificationCount` at each draw as Metal's
//!   validation holds them, and the binder plans the layers of the pass the draw renders
//!   (`bind::ViewSetup`). An amplified draw records into an instance whose subpasses have its view
//!   mask, over a framebuffer of one layer whose attachments' views start at the layer of its first
//!   view; a draw a view mapping moves to one layer, into an instance without multiview whose
//!   attachments' views are of that layer alone (`render_target::viewed`); its input attachments
//!   are read through those views. Such an instance does not run the pass's clears unless it
//!   renders every layer: the first instance then stays, of no draws, and clears them. Only a
//!   layered pass renders some of its layers: one that resolves and whose last instance renders
//!   some of them ends with an instance of no draws over every layer, which resolves them all. What
//!   multiview does not draw as Metal does (view mappings that offset the viewport or name layers
//!   out of order, more amplifications than the device's views, a pipeline that chooses or reads
//!   its slice) the binder plans one amplification at a time, on a variant of the pipeline state:
//!   the draw records once per amplification (`bind::DrawCommands::amplifications`), in
//!   amplification order, each after that amplification's push writes, into the instance of the
//!   pass's layers, and announces `amplification-draws`. While a visibility result mode is set
//!   every amplified draw is drawn so (`VertexAmplification::one_at_a_time`), since a query counts
//!   the samples of every amplification as Metal's does (a Metal probe on the M3 Max and an M4)
//!   where a multiview instance would take one query per view; a mode set after a draw amplified as
//!   views begins its query in a new instance of the pass's layers. So is every amplified draw in a
//!   rate-mapped pass whose map's layers differ, each amplification through its own slice's
//!   regions.
//!
//! - **Executions of render commands** (`render/executions.rs`): `execute_commands_in_buffer` ends
//!   the current instance's secondary command buffer and goes on in a new one; the commands draw,
//!   as the records stand at commit (or, where the GPU may write them, at a host step), into a
//!   secondary of their own between the two, through the steps of a draw, with the encoder's state
//!   at the call. A pass with such an execution records its pass later, as a segment of the
//!   encoder.
//!
//! - **Primitive restart.** An indexed draw of a strip (`LineStrip`, `TriangleStrip`) restarts it
//!   at the all-ones index of its index type, and a list or points draw the all-ones index as a
//!   vertex (Metal probes on an M4, with and without API validation), as Vulkan's
//!   `primitiveRestartEnable` does, set for strips only (`metal::state::input_assembly`). After
//!   `set_primitive_restart_enabled` a strip restarts at the call's index as Metal takes it, and
//!   only there: elsewhere than at the all-ones index, it is drawn as the list an index rewrite
//!   pass writes before the pass (`render/restart.rs`, `index_rewrite.rs`).
//!
//! - **Textures over buffers** (`render/shadows.rs`). A 2D texture over a buffer renders, as an
//!   attachment or a resolve target, into its aliasing image, the buffer's bytes, or into its
//!   shadow; a shadowed texture or a staged texture buffer a draw binds or reaches, and a shadowed
//!   attachment, is copied in from the buffer before the instance that first uses it and back as
//!   the encoder ends, by `shadow.rs`'s rules with an instance as the command; an execution of
//!   render commands gathers those its commands may use at the call. A texture buffer renders, as
//!   an attachment or a resolve target, through its attachment twin, a 2D texture of one row over
//!   its bytes (`Texture::attachment_twin`).
//!
//! - **Color groups** (`render/color_groups.rs`). A pass of more color attachments than the device
//!   draws at once runs once per group of them: the encoder records the first group's pass, and
//!   as it ends each later group's, from its log, over a twin of the depth and stencil.

use super::address_map::{self, SamplerMaps};
use super::amplification::{Amplification, VertexAmplificationViewMapping};
use super::argument_table::{check_index, ArgumentTable};
use super::attachment_stand_in::{self, Cause, HiddenClear, StandIn, StandInCopies};
use super::binding;
use super::blit::TRANSFER;
use super::buffer::Buffer;
use super::clear_reach::{self, ReachClears};
use super::color_list::ColorList;
use super::color_reformat::{self, Routing};
use super::color_resolve::ShaderResolves;
use super::command_buffer::CommandBuffer;
use super::deferred::Deferred;
use super::degradation::{
    BoundFormats, CompiledPipeline, Degradation, EachWatch, HandedToMetal, SplitCauses, WholePass,
};
use super::depth_stand_in::DepthStandIn;
use super::depth_stencil::{DepthStencilState, FailingDraw};
use super::depth_stencil_resolve::DrawnResolve;
use super::depth_stencil_scratch::{ScratchCopy, Separate};
use super::device::{lock, Device};
use super::draw_source::{CopySteps, DrawSource, IndexBind};
use super::dynamic_state::ExtendedDynamicState;
use super::encoder::{EncoderCore, Replayed};
use super::error::{
    internal, invalid, vulkan, CommandBufferError, EncoderCall, Error, Problem, RuntimeGap,
};
use super::execution::{ExecutedPipeline, Execution};
use super::fragment_imageblock::PassImageblock;
use super::framebuffer_fetch;
use super::fresh_unwritten::FreshUnwritten;
use super::function_table::VisibleFunctionTable;
use super::heap_sets;
use super::index_bounds::{self, Bounded, BoundsPass, BoundsSteps};
use super::index_rewrite::{self, Listed, RewritePass, RewriteSteps, WrittenList};
use super::kept_bytes;
use super::linear_attachment;
use super::linear_merge::LinearMerge;
use super::live_bind::{Emptied, LiveBind, Reshape};
use super::nan_clear::{keeps_nan_sign, NanDepthWrite};
use super::owned_kernel::OwnedModule;
use super::pipeline::Settled;
use super::pipeline::{BufferForms, DrawPipeline};
use super::pipeline::{DrawDynamic, RawPipeline, RenderVariant, VariantObjects};
use super::pipeline::{
    DrawKey, FixedState, GraphicsState as DrawingState, RenderPipelineState, RenderStateKind,
    TileState,
};
use super::rate_mapped_pass::{self, RateMappedPass};
use super::render_pass::{
    self, AspectTarget, Attachment, DepthStencilTarget, PassLayout, RenderPassDescriptor,
    RenderPassProblem, Target, MAX_COLOR_ATTACHMENTS,
};
use super::render_state::{self, DepthBiasRecord, EncoderState, ViewportList, MAX_VIEWPORTS};
use super::render_state::{Compensation, DepthBiasEffect, WrittenDepthClamp};
use super::render_target::{self, covers, Feedback, PassShape, Position, RenderPassObject};
use super::render_values::{
    DrawPrimitiveType, LoadAction, ScissorRect, StoreAction, StoreActionOptions, Viewport,
};
use super::replay::Replay;
use super::residency::Residency;
use super::resolve::{
    self, DepthStencilResolve, Resolve, ResolveCopies, ResolveMode, ResolveSource,
};
use super::resource::ResourceKey;
use super::sample_planes::{self, Planes, PlanesCommands};
use super::sample_positions;
use super::sampler::OwnedSampler;
use super::sampler::{ArgumentSamplers, SamplerState};
use super::shadow::{self, ShadowCopy};
use super::shadow_stencil::ShadowStencil;
use super::sparse::emulated::Scrub;
use super::tessellation::{FactorBuffer, Factors};
use super::texture::held_depth;
use super::texture::Texture;
use super::tracker::{self, Access, Barrier, GeometryWrites, IndirectReads, UnorderedWrites, Use};
use super::used_resources::EncoderKind;
use super::versioned::Versioned;
use super::visibility::{self, Continuation, Visibility, VisibilityResultMode};
use super::written_commands::Writers;
use crate::device::functions::Vk;
use crate::work_counts::Begun;
use crate::work_counts::NotInline;
use crate::work_counts::Split;
use ash::vk;
use draw_shape::{PipelineInputs, ShapeInputs, ShapeMemo, ShapeToken, ShapedState};
pub(super) use executions::DeferredPass;
use executions::{PassInstance, PassStorage, RenderExecution, RunPasses};
use extended_shadow::ExtendedShadows;
pub use late_resolve::ResolveTarget;
use metal2vulkan::bind::BindingPlan;
use metal2vulkan::bind::VertexAmplification;
use metal2vulkan::bind::{
    DrawCall, FragmentTests, HeapPlacement, ImageSource, ImageblockSize, IndexBufferRef,
    IndirectRef, MeshDispatch, MeshGroups, RewriteProgram, TessellationFactors, ViewSetup,
    WriteSource,
};
use metal2vulkan::cache::SpecializationKey;
use metal2vulkan::device::{DriverFact, FormatUsage, VkFeature};
use metal2vulkan::error::BindError;
use metal2vulkan::error::Feature as ErrorFeature;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::format::{ycbcr_renders, ExtendedRange};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CullMode, DepthClipMode, IndexType, PixelFormat, PrimitiveType, RenderStages, TriangleFillMode,
    Winding,
};
use metal2vulkan::pipeline::{
    ColorGroup, GraphicsState, PipelineInterface, PushField, RuntimeFact, Stage,
    StorageAttachmentUse, StorageAttachmentView,
};
use pass_start_reads::{PassStartReads, PassStartTables};
use pass_steps::{InstanceSteps, PassSteps, StagedDispatch, StagedRows};
use pending::{OwnedRecording, PendingDraw};
use provoking::ProvokingForms;
use shadows::InstanceShadows;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};
use tile_dispatch::{ScratchEdges, TileDispatch};

mod arguments;
mod bound;
mod call_checks;
mod color_groups;
mod deferral;
mod draw_shape;
mod executions;
mod extended_shadow;
mod inline_first;
mod inline_splits;
mod late_resolve;
mod lowering_worker;
mod mesh;
mod mesh_arguments;
mod pass_start_reads;
mod pass_steps;
mod pending;
mod pending_tile;
mod provoking;
mod queued_commits;
mod replay;
mod reroute;
mod restart;
mod shadows;
mod snapshot;
mod table_facts;
mod tessellation_scan;
mod tile_dispatch;
mod tile_memory;
mod tile_store;
mod untouched;
mod vertex_reach;
mod vertex_strides;

use bound::{Bound, EqualShapes, Now, Recorded};
use deferral::Logged;
pub(crate) use deferral::{lowering, DeferredEncoders, EndInFlight};
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) use deferral::{BATCH_CALLS, HANDED_BATCHES};
pub(crate) use inline_splits::SpareKept;
use replay::{
    DrawIndexed, DrawIndexedIndirect, DrawIndexedPatches, DrawIndexedPatchesIndirect, DrawPatches,
    DrawPatchesIndirect, Samplers,
};
pub(crate) use replay::{RenderCall, RenderLog, RenderRoom, Table};

/// What a draw, or the render pipeline state or tessellation factors set for the draws, did
/// wrong.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawProblem {
    /// `set_render_pipeline_state`: the state's pixel format of the color attachment differs
    /// from the pass's (measured). Plain Metal packs the state's set colors from byte 0 of a
    /// per-sample record ([`color_record`](metal2vulkan::metal::format::color_record)) and
    /// stores each place into the pass's attachment at that place of the pass's record (Metal
    /// probes), as the runtime draws it where each state color lies exactly on a pass color or on
    /// none: a state lacking color 0 beside its color 1 stores its color 1 into the pass's color 0,
    /// and a pass lacking color 0 beside its color 1 takes the state's color 0 there. Refused where
    /// a state color covers part of a pass color, or more than one (the index is the lowest that
    /// moves), and where such a draw moves a color of a mesh render pipeline's state or one that
    /// fetches or reads a color or declares an imageblock (unmeasured). A color both have in
    /// different formats stays refused but where they differ in channel order alone (`RGBA8Unorm`
    /// and `BGRA8Unorm`, their sRGB forms, `RGB10A2Unorm` and `BGR10A2Unorm`), in a render
    /// pipeline's state with a fragment function that neither reads the color nor declares an
    /// imageblock: plain Metal stores the state's encoding of the output in the pass's texel, cut
    /// or zero-filled to its size, in the pass's channel order where the pass's format puts blue
    /// first, and blended as the state's format over the pass's bytes (Metal probes), which for a
    /// channel-order twin is the pass's own format. A pass color at no place of the state's record
    /// is drawn as Metal stores it: zeroed at the covered samples where Metal's code reads no
    /// attachment, kept where a state color blends reading the destination or writes some of its
    /// channels alone (with the `uncovered-color-kept` notice where Metal's compiler may fold the
    /// output to a constant and zero it); refused for a mesh render pipeline's state and a state
    /// without a fragment function or that rasterizes nothing (unmeasured). A state color at no
    /// place of the pass's record is drawn, its output dropped, but where the fragment function
    /// reads that color (unmeasured). A tile pipeline's state is refused a color format other than
    /// the pass's only up to the last color its function reaches through its implicit imageblock,
    /// or at any color where it declares an explicit one: plain Metal runs the function whatever
    /// the formats, the colors it reaches at their places in the state's packing (a Metal probe).
    ///
    /// Basis: unknown (what stays refused is what the Metal probes did not reach: a state color
    /// covering part of a pass color or more than one, a mesh render pipeline's state, and a
    /// fragment function that fetches a color or declares an imageblock).
    AttachmentFormat(Attachment),
    /// `set_render_pipeline_state`: the state's raster sample count differs from the pass's
    /// (measured). Plain Metal draws a render pipeline's state of more samples in a pass of one
    /// as a state of one sample, as the runtime does (Metal probes), but for alpha to coverage and
    /// a fragment function's `get_num_samples()`, which read the state's count; such a state, a
    /// mesh render pipeline's, one whose fragment function uses an imageblock (unmeasured), and a
    /// state of one sample in a pass of more (plain Metal writes sample 0 alone, under a coverage
    /// rule unmeasured) are refused.
    ///
    /// Basis: unknown (what stays refused is what the Metal probes did not reach: alpha to
    /// coverage, `get_num_samples()`, a mesh render pipeline's state, an imageblock, and a state of
    /// one sample in a pass of more, whose coverage rule is unmeasured).
    SampleCount,
    /// `set_render_pipeline_state`: a drawing state whose fragment function declares an explicit
    /// imageblock, or a tile pipeline's state whose tile function aliases one onto its implicit
    /// imageblock, needs more imageblock bytes per sample (`imageblock_sample_length`) than the
    /// pass gives, the larger of its `imageblock_sample_length` and its attachments' texels
    /// (measured: "the pipelineState's per sample imageBlock usage(32) is greater than the
    /// encoder's perSample imageBlock usage(4)"). A tile state whose explicit imageblock stands
    /// alone runs there, as plain Metal runs it: its dispatches keep the words of the imageblock
    /// below the pass's bytes rounded up to 8 and read every other as zero.
    ///
    /// Basis: metal-documented (Metal asserts "the pipelineState's per sample imageBlock usage(32)
    /// is greater than the encoder's perSample imageBlock usage(4)").
    ImageblockSampleLength {
        /// The state's bytes per sample.
        pipeline: u64,
        /// The pass's bytes per sample.
        pass: u64,
    },
    /// A draw with a tile pipeline's state set (measured: "renderPipelineState must be created
    /// using a MTLRenderPipelineDescriptor.").
    ///
    /// Basis: metal-documented (Metal asserts "renderPipelineState must be created using a
    /// MTLRenderPipelineDescriptor.").
    TileState,
    /// A draw of patches with a pipeline that is not a tessellation one (measured on an M4: Metal's
    /// validation stops it, and Metal without it stops the process); a draw of primitives or
    /// patches with a mesh render pipeline's state, or a mesh draw with any other state (measured,
    /// a probe: "renderPipelineState must be created using a MTLRenderPipelineDescriptor.", and
    /// Metal without its validation stops the process). A draw of primitives with a tessellation
    /// pipeline, which Metal's validation stops too, draws nothing, as plain Metal draws it (the
    /// same probe), and so does a mesh draw of no threadgroups with a drawing state; an empty draw
    /// of patches or primitives of another kind stops plain Metal too, and stays refused.
    ///
    /// Basis: metal-measured (an M4: plain Metal stops its process over a patch draw with a
    /// pipeline that is not a tessellation one, and over a mesh state's draw of primitives or
    /// patches, its validation off).
    DrawKind,
    /// A patch draw while the encoder has no tessellation factor buffer (measured: Metal's
    /// validation stops it, and Metal without it stops the process).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal ends the process, SIGSEGV,
    /// at a quad patch draw with no tessellation factor buffer, where the same draw with one set
    /// runs and paints the target, the API validation layer off).
    NoTessellationFactorBuffer,
    /// An indirect draw's arguments, read from its offset rounded down to a multiple of 4 as Metal
    /// reads them, run past its indirect buffer's length (measured: Metal's validation stops it;
    /// plain Metal draws nothing, or draws from whatever memory follows the buffer, Metal probe
    /// cases `ind_offset_past` and `indirect_past_end` on an M4: undefined, so refused).
    ///
    /// Basis: metal-takes (an M4: plain Metal draws nothing, or draws from whatever memory follows
    /// the buffer, rather than refusing — probe cases `ind_offset_past` and `indirect_past_end`:
    /// undefined).
    IndirectBufferRange,
    /// `dispatch_threads_per_tile` with a drawing state set (measured: "the renderPipelineState
    /// was not created with a tileDescriptor").
    ///
    /// Basis: metal-documented (Metal asserts "the renderPipelineState was not created with a
    /// tileDescriptor").
    NotTileState,
    /// `dispatch_threads_per_tile` with threads per tile wider or taller than the pass's tile
    /// (measured: "threadsPerTile.width(32) must be <= 16"), within Metal's limit for the state:
    /// past that limit the dispatch runs nothing instead, as plain Metal runs it. Plain Metal
    /// takes such a dispatch, and runs the threads past the tile at the pixels of the tiles that
    /// follow it: for every tile of the pass it runs the threads whose pixels lie in the threads
    /// per tile from that tile's origin and inside the render area, each thread reporting the
    /// threadgroup position of the tile holding its own pixel and its position in that tile, and
    /// the threads per threadgroup as the dispatch's own. A pixel within the threads per tile of
    /// more than one tile's origin therefore runs once for each of them, on overlapping
    /// threadgroups the runtime does not model: a dispatch of the tile's own threads would run
    /// each pixel once and store once where Metal stores repeatedly, so this is refused rather
    /// than served by a narrower dispatch.
    ///
    /// Basis: metal-takes (an Apple M3 Max, macOS 27.0.1, probe
    /// `refusals_without_a_named_device` shapes `tilec-*` over a 32 by 32 target in 16 by 16
    /// tiles, four fresh processes each with one reading: plain Metal completes the pass with no
    /// error, and an atomic visit counter per position per tile counts, for threads per tile of 32
    /// by 1, 17 by 1, 24 by 1, 16 by 32, 32 by 32, 64 by 16, 16 by 64 and 33 by 31, exactly the
    /// pixels of the rectangle of that size from each tile's origin cut to the render area -- 96,
    /// 66, 80, 1536, 2304, 1536, 1536 and 2256 invocations -- with the overlapped tiles' positions
    /// visited 2 and 4 times, and the threads per threadgroup reported as the dispatch's own: so
    /// the threads past the tile neither vanish nor clamp onto the tile's own positions; the
    /// refusal reaches past Metal, and what Metal runs is more than a dispatch of the tile's
    /// threads).
    ThreadsPerTilePastTile {
        /// The threads per tile's width and height.
        threads: [u32; 2],
        /// The pass's tile.
        tile: [u32; 2],
    },
    /// A draw whose vertex amplification count is above its pipeline state's
    /// `maxVertexAmplificationCount` of 1, or above Metal's 8 (Metal's validation stops a count
    /// above the pipeline's maximum, a Metal probe on an M4: "Vertex Amplification Count (3) must
    /// be between (inclusive) 1 and the maximum vertex amplification count specified in the
    /// pipeline state (2)"). Plain Metal stored garbage in slice 0 for 2 and 3 on a pipeline of 1
    /// (probe 13 `amp2_pipeline1`, a probe: undefined, so refused); a count above 8 is not
    /// measured. One above a larger maximum draws that many amplifications, and a count of 0 draws
    /// nothing, as plain Metal draws them (probe 13 `amp0`, a probe).
    ///
    /// Basis: metal-takes (an M4: plain Metal stored garbage in slice 0 for counts 2 and 3 on a
    /// pipeline of 1, probe `amp2_pipeline1`, rather than refusing; a count above 8 is unmeasured).
    AmplificationCount {
        /// The encoder's count.
        count: u64,
        /// The pipeline state's `maxVertexAmplificationCount`.
        max: u32,
    },
    /// `set_vertex_amplification_count`: fewer view mappings than the count, past which Metal
    /// would read its pointer (memory the call was not given: undefined, so refused).
    ///
    /// Basis: ours (this crate's slice of view mappings: Metal's call takes a pointer, and reading
    /// the count would pass the memory the caller handed over).
    ViewMappings {
        /// The amplification count.
        count: u64,
        /// The view mappings given.
        given: u64,
    },
    /// A direct patch draw's patch indices run past its patch index buffer: Metal's validation
    /// stops a draw whose offset (rounded down to a multiple of 4, as Metal reads it) plus 4 bytes
    /// for each of `patchStart + patchCount` indices passes the buffer's length (measured on an M4:
    /// `patchIndexBufferOffset(0) + (patchIndicesUsed(4) * 4) must be <= [patchIndexBuffer
    /// length](8)`); plain Metal reads past the buffer (probe case `pib_short`: undefined, so
    /// refused).
    ///
    /// Basis: metal-takes (an M4: plain Metal reads past the patch index buffer rather than
    /// refusing, probe case `pib_short`: undefined).
    PatchIndexBufferRange {
        /// `patchIndexBufferOffset`.
        offset: u64,
        /// `patchStart + patchCount`.
        indices: u64,
        /// The patch index buffer's length.
        length: u64,
    },
    /// A direct indexed patch draw's control-point indices run past its control-point index buffer:
    /// Metal's validation stops a draw whose offset plus `size` bytes for each of `(patchStart +
    /// patchCount) · numberOfPatchControlPoints` indices passes the buffer's length, with or
    /// without a patch index buffer (measured on an M4: `controlPointIndexBufferOffset(0) +
    /// (indexCount(9) * 4) must be <= [controlPointIndexBuffer length](24)`); plain Metal reads
    /// past the buffer (probe cases `cpi_short` and `cpi_start1_count2`: undefined, so refused).
    ///
    /// Basis: metal-takes (an M4: plain Metal reads past the control-point index buffer rather than
    /// refusing, probe cases `cpi_short` and `cpi_start1_count2`: undefined).
    ControlPointIndexBufferRange {
        /// `controlPointIndexBufferOffset`.
        offset: u64,
        /// `(patchStart + patchCount) · numberOfPatchControlPoints`.
        indices: u64,
        /// The index size, in bytes.
        size: u64,
        /// The control-point index buffer's length.
        length: u64,
    },
    /// An indirect render command's indexed draw whose `count` indices from `offset` (rounded
    /// down to a multiple of the index size, as Metal reads it) run past its index buffer's
    /// `length` (measured: `indexBufferOffset(0) + (indexCount(6) * 4)
    /// must be <= [indexBuffer length](16)`); plain Metal draws from whatever memory follows the
    /// buffer, which the command's record does not bound: it holds the index buffer as one address
    /// and no length, where a direct draw's fetch stops at the buffer (the module docs, "Index
    /// positions past the index buffer"). The memory past the buffer is memory the draw was not
    /// given: undefined, so refused.
    ///
    /// Basis: metal-takes (an Apple M3 Max, macOS 27.0.1, probe
    /// `refusals_without_a_named_device` shape `idxc-past`, four fresh processes with one reading:
    /// plain Metal completes an indirect indexed draw of 12 indices from byte 480 of a 512-byte
    /// index buffer, with no error, and its vertex function is given the eight indices inside the
    /// buffer and then the four words that follow it in the heap, each of which holds its own word
    /// number -- so the fetch read the following memory by name, where the same draw made directly
    /// on the encoder was given index 0 at those four positions, `idxv-past`).
    IndexBufferRange {
        /// `indexBufferOffset`.
        offset: u64,
        /// `indexCount`.
        count: u64,
        /// The index buffer's length.
        length: u64,
    },
}

/// The buffers one draw reads besides its bindings: an index buffer and an indirect buffer, each
/// at the draw's offset into it (`draw_source.rs`), and a patch draw's patch index and
/// control-point index buffers at their offsets, which the control stage reads through
/// descriptors.
#[derive(Default)]
struct DrawBuffers<'a> {
    index: Option<DrawSource<'a>>,
    indirect: Option<DrawSource<'a>>,
    patch_indices: Option<(&'a Buffer, u64)>,
    control_point_indices: Option<(&'a Buffer, u64)>,
}

/// An indexed draw's values but its index buffer and offset (`draw_indexed`).
#[derive(Clone, Copy)]
struct IndexedDraw {
    primitive_type: DrawPrimitiveType,
    index_count: u32,
    index_type: IndexType,
    instance_count: u32,
    base_vertex: i32,
    base_instance: u32,
}

/// A patch draw's values but its patch index buffer (`draw_patch_range`).
#[derive(Clone, Copy)]
struct PatchDraw {
    control_points: u32,
    patch_start: u32,
    patch_count: u32,
    instance_count: u32,
    base_instance: u32,
}

/// What a draw assembles: primitives of a type, a tessellation pipeline's patches, or a mesh
/// render pipeline's meshes (`render/mesh.rs`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Assembly {
    Primitives(PrimitiveType),
    /// One of Metal's private primitive types, drawn as the list its program writes
    /// (`render/restart.rs`).
    Private(RewriteProgram),
    Patches,
    Meshes,
}

impl Assembly {
    /// What a draw of primitive type `primitive` assembles; `None` for type 9, which draws
    /// nothing ([`RenderLowering::draw_nine`]).
    fn of(primitive: DrawPrimitiveType) -> Option<Assembly> {
        Some(match primitive {
            DrawPrimitiveType::Public(p) => Assembly::Primitives(p),
            DrawPrimitiveType::LineLoop => Assembly::Private(RewriteProgram::LineLoop),
            DrawPrimitiveType::TriangleFan => Assembly::Private(RewriteProgram::TriangleFan),
            DrawPrimitiveType::Quads => Assembly::Private(RewriteProgram::Quads),
            DrawPrimitiveType::QuadStrip => Assembly::Private(RewriteProgram::QuadStrip),
            DrawPrimitiveType::Nine => return None,
        })
    }

    /// The program that lists the assembly's primitives in Metal's order
    /// (`index_rewrite::RewriteProgram`): a line or triangle type's, or a private type's own.
    /// `None` for points, patches and meshes, which no list holds.
    fn program(self) -> Option<RewriteProgram> {
        Some(match self {
            Assembly::Primitives(primitive) => match primitive {
                PrimitiveType::LineStrip => RewriteProgram::LineStrip,
                PrimitiveType::TriangleStrip => RewriteProgram::TriangleStrip,
                PrimitiveType::Line => RewriteProgram::LineList,
                PrimitiveType::Triangle => RewriteProgram::TriangleList,
                PrimitiveType::Point => return None,
            },
            Assembly::Private(program) => program,
            Assembly::Patches | Assembly::Meshes => return None,
        })
    }
}

/// How a draw of primitive type 9 fails its command buffer, as Metal's GPU fails it
/// (`DrawPrimitiveType::Nine`): it hangs, or faults the encoder with the error.
#[derive(Clone, Copy)]
enum NineFailure {
    Hang,
    Fault(CommandBufferError),
}

/// The index type of `draw` where it is a direct indexed draw of `pipeline` whose vertex function
/// writes the primitive ID's varying (`LoweringInputs::primitive_id_varying`, which reads
/// `PushField::VerticesPerPrimitive`): such a draw runs the variant that draws its indices as
/// vertices from 0 (`BindState::primitive_id_indices`). `None` for every other draw.
fn primitive_id_indices(pipeline: &DrawingState, draw: &DrawCall) -> Option<IndexType> {
    let DrawCall::Indexed { index_type, .. } = *draw else {
        return None;
    };
    if !writes_primitive_ids(pipeline) {
        return None;
    }
    match index_type {
        t if t == metal2vulkan::vk::IndexType::UINT16 => Some(IndexType::UInt16),
        t if t == metal2vulkan::vk::IndexType::UINT32 => Some(IndexType::UInt32),
        _ => None,
    }
}

/// Whether `pipeline`'s vertex function writes the primitive ID's varying
/// (`LoweringInputs::primitive_id_varying`, which reads `PushField::VerticesPerPrimitive`).
fn writes_primitive_ids(pipeline: &DrawingState) -> bool {
    pipeline.base().interface.push.iter().any(|layout| {
        layout.stage == Stage::Vertex
            && layout
                .fields
                .iter()
                .any(|f| f.field == PushField::VerticesPerPrimitive)
    })
}

/// How a draw of one of Metal's private types, drawn as its program's list, binds
/// (`BindState::quad_primitive_ids`, `BindState::private_list`); the default for every other
/// draw.
#[derive(Clone, Copy, Default)]
pub(super) struct PrivateForm {
    /// Quads or a quad strip whose fragment function reads `[[primitive_id]]`: the variant that
    /// reads the quad's, as Metal counts one primitive a quad where the list draws two triangles.
    quad_primitive_ids: bool,
    /// The pulled list form (`VariantKey::private_list`): each list position run once without
    /// indices, its index and Metal's base vertex read at the list's address. Metal runs the vertex
    /// function once per vertex of its list for these types (its probes' counts on the M3 Max,
    /// macOS 27.0.1), where Vulkan's indexed list draw may run a vertex once for every primitive
    /// that shares it: the form for a vertex function that writes memory, and for one that writes
    /// the primitive ID's varying, whose index the list's positions are. Also for a direct draw
    /// without indices whose last vertex ID passes `i32::MAX`: an indexed list's base vertex
    /// (`vertexOffset`, an `int32_t`) cannot carry a start past it, and Vulkan states no sum of an
    /// index and the base past it, where the form adds them in the vertex function, wrapping as
    /// Metal's vertex IDs do (its probes from vertex starts 2^31 - 3, 2^31 + 3 and 2^32 - 4).
    private_list: bool,
}

impl PrivateForm {
    /// The form of `draw` of `pipeline` of a private type, listed by `program`.
    fn of(pipeline: &DrawingState, program: RewriteProgram, draw: &DrawCall) -> PrivateForm {
        let graphics = &pipeline.base().graphics;
        let quads = matches!(program, RewriteProgram::Quads | RewriteProgram::QuadStrip);
        PrivateForm {
            quad_primitive_ids: quads && graphics.reads_primitive_id,
            private_list: writes_primitive_ids(pipeline)
                || graphics.vertex_writes_memory
                || far(draw),
        }
    }

    /// The form of a public type's `draw` drawn as a list (a strip that restarts where Vulkan does
    /// not, a draw under the last provoking vertex): the private list form where its last vertex
    /// ID passes `i32::MAX`, which an indexed list's base vertex cannot carry.
    fn listed(draw: &DrawCall) -> PrivateForm {
        PrivateForm {
            quad_primitive_ids: false,
            private_list: far(draw),
        }
    }
}

/// Whether `draw` is a direct draw without indices whose last vertex ID passes `i32::MAX`.
fn far(draw: &DrawCall) -> bool {
    matches!(*draw, DrawCall::Primitives { vertex_start, vertex_count, .. }
        if u64::from(vertex_start) + u64::from(vertex_count) > 1 << 31)
}

/// The draw the binder instantiates for `draw` of a private type in the private list form, of
/// `program`'s list: a direct draw's list from its first index (the word after Metal's base vertex)
/// as first vertex, with its instances and base instance; an indirect draw as given, whose list
/// the GPU counts.
fn listed_call(program: RewriteProgram, draw: &DrawCall) -> DrawCall {
    let listed = |count: u32, instance_count, base_instance| DrawCall::Primitives {
        vertex_start: 1,
        vertex_count: u32::try_from(program.capacity(u64::from(count))).unwrap_or(u32::MAX),
        instance_count,
        base_instance,
    };
    match *draw {
        DrawCall::Primitives {
            vertex_count,
            instance_count,
            base_instance,
            ..
        } => listed(vertex_count, instance_count, base_instance),
        DrawCall::Indexed {
            index_count,
            instance_count,
            base_instance,
            ..
        } => listed(index_count, instance_count, base_instance),
        other => other,
    }
}

/// What a draw settles from the encoder's state before its bind state
/// (`draw_shape::ShapeInputs::settings`), kept from one draw to the next
/// (`draw_shape::ShapeMemo`).
#[derive(Clone, PartialEq)]
struct DrawSettings {
    /// Points that rasterize nothing: the draw runs its vertex function alone, through empty
    /// scissors.
    unsized_points: bool,
    /// Its input assembly.
    topology: metal2vulkan::vk::PrimitiveTopology,
    primitive_restart: bool,
    /// Whether the draw gives the pass its hidden depth and stencil (made for the draw:
    /// `RenderCommandEncoder::needed_hidden`).
    hidden: bool,
    /// The depth and stencil state of a draw whose compare fails every fragment.
    failing_state: Option<DepthStencilState>,
    /// Its vertex amplification.
    amplification: VertexAmplification,
    /// The pass's tile size and the tiles' area, where the fragment function reads its tile's
    /// threadgroup memory (`render/tile_memory.rs`).
    tiles: Option<([u32; 2], [u32; 2])>,
    /// The device takes depth ranges past `[0, 1]` natively.
    native: bool,
    /// Its viewports' compensation in the vertex stage.
    compensation: Option<Compensation>,
    /// The clamp a fragment function that writes its depth takes itself.
    written_clamp: Option<WrittenDepthClamp>,
    /// The depth bias its primitives take: none for points and lines.
    bias: Option<DepthBiasEffect>,
    /// Where its colors go among the pass's.
    routing: Arc<Routing>,
    /// The pass's colors it zeroes and its own it drops (`ShapeInputs::routed_colors`): the
    /// encoder's, held, where it keeps them for its pipeline state in its pass (`Routed`).
    routed_colors: Arc<RoutedColors>,
    /// Its `VkPipeline` takes the last provoking vertex (`render/provoking.rs`,
    /// `FixedState::provoking_last`).
    provoking_last: bool,
}

/// The render encoder state one draw takes, statically or as dynamic state, in Vulkan's terms
/// (`render_state.rs`).
#[derive(Clone)]
struct DrawState {
    viewports: ViewportList<vk::Viewport>,
    scissors: ViewportList<vk::Rect2D>,
    fixed: FixedState,
    /// The front and back stencil references.
    stencil_reference: [u32; 2],
    /// The depth bias's factors and the command they record by.
    depth_bias: DepthBiasRecord,
    blend_color: [f32; 4],
}

impl PartialEq for DrawState {
    /// Every field equal, each viewport's by its fields (`vk::Viewport` has no `PartialEq`).
    fn eq(&self, other: &DrawState) -> bool {
        let fields = |v: &vk::Viewport| [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth];
        self.viewports.len() == other.viewports.len()
            && (self.viewports.iter().zip(other.viewports.iter()))
                .all(|(a, b)| fields(a) == fields(b))
            && self.scissors == other.scissors
            && self.fixed == other.fixed
            && self.stencil_reference == other.stencil_reference
            && self.depth_bias == other.depth_bias
            && self.blend_color == other.blend_color
    }
}

/// The states a draw records with (`draw_resident`): the whole draw's, or in a rate-mapped pass
/// each region's of each amplification drawn alone, in order (`rate_mapped_pass.rs`).
enum DrawStates {
    Whole([DrawState; 1]),
    Regions(Vec<Vec<DrawState>>),
}

impl DrawStates {
    /// The states amplification `a` records with: the last amplification's for any past it.
    fn of(&self, a: usize) -> &[DrawState] {
        match self {
            DrawStates::Whole(whole) => whole,
            DrawStates::Regions(regions) => &regions[a.min(regions.len() - 1)],
        }
    }

    /// Every state.
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut DrawState> {
        let (whole, regions) = match self {
            DrawStates::Whole(whole) => (&mut whole[..], &mut [][..]),
            DrawStates::Regions(regions) => (&mut [][..], &mut regions[..]),
        };
        whole.iter_mut().chain(regions.iter_mut().flatten())
    }
}

/// The storage the pixel interlock orders that the draws of a render pass instance reach, kind
/// by kind: what a draw reads or writes, or what an instance's draws wrote since its last storage
/// barrier. A draw that reaches a kind an earlier draw of its instance wrote records after the
/// barrier from the fragment shader's writes to its reads and writes
/// (`render_target::between_storage_draws`). The interlock orders the overlapping fragments of a
/// draw, and Vulkan orders those of consecutive draws in a subpass too: in the ordered modes
/// "interlocked operations in a fragment shader must happen before interlocked operations in
/// fragment shader invocations that execute later in rasterization order and cover at least one
/// sample in the same pixel" (Fragment Operations, "Interlocked Operations"), rasterization order
/// follows primitive order through a subpass, submission order first ("Primitive Order"), and
/// every access inside the critical section is coherent (`lower::sync::memory_model`), as the
/// Memory Model appendix has interlock shaders "declare the relevant resources as coherent to get
/// implicit per-instruction availability and visibility operations". RADV on an
/// AMD 780M (Mesa 26.2.4) does not: a later draw, of the same pipeline or another, read texels as
/// they were before an earlier one's writes (for the imageblock and for a color). So every device
/// takes the barrier but where its driver is measured to order such draws' colors by the
/// interlock alone (`DriverFact::InterlockOrdersDraws`, by
/// `runtime_lane::interlocked_draw_order`): there a draw's colors are no kind of it
/// ([`InterlockedStorage::of`]). Another kind of storage a draw reads and writes inside the
/// interlock joins as a field here and in [`InterlockedStorage::of`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InterlockedStorage {
    /// The pass's fragment imageblock storage (`fragment_imageblock.rs`).
    imageblock: bool,
    /// Colors in the interlocked storage form, the fetch's (`framebuffer_fetch.rs`).
    colors: bool,
    /// The depth's stand-in, which a draw that tests the depth in the shader reads and may write
    /// (`depth_stand_in.rs`).
    depth: bool,
}

impl InterlockedStorage {
    const NONE: Self = Self {
        imageblock: false,
        colors: false,
        depth: false,
    };

    /// Every kind: what an execution of render commands, or the encoder's draws before it, may
    /// have written.
    const ALL: Self = Self {
        imageblock: true,
        colors: true,
        depth: true,
    };

    /// What a draw of `graphics` whose colors in the interlocked storage form are `storage`
    /// reaches, and what it writes, on a device whose driver orders interlocked draws' colors by
    /// the interlock alone where `colors_ordered` (then no color is a kind of it). A draw that
    /// reaches the imageblock may write it, and one that tests the depth in the shader may write
    /// the depth's stand-in.
    fn of(
        graphics: &GraphicsState,
        storage: &[StorageAttachmentUse],
        colors_ordered: bool,
    ) -> (Self, Self) {
        let imageblock = graphics.fragment_imageblock.is_some();
        let depth = graphics.shader_tests.is_some();
        let reaches = Self {
            imageblock,
            colors: !colors_ordered && !storage.is_empty(),
            depth,
        };
        let writes = Self {
            imageblock,
            colors: !colors_ordered && storage.iter().any(|s| s.writes),
            depth,
        };
        (reaches, writes)
    }

    /// Whether a kind is in both.
    fn meets(self, other: Self) -> bool {
        (self.imageblock && other.imageblock)
            || (self.colors && other.colors)
            || (self.depth && other.depth)
    }

    /// The kinds of either.
    fn or(self, other: Self) -> Self {
        Self {
            imageblock: self.imageblock || other.imageblock,
            colors: self.colors || other.colors,
            depth: self.depth || other.depth,
        }
    }
}

/// One render pass instance of an encoder's pass (the module docs).
struct Instance {
    /// The secondary command buffer its draws record into, against `compatible`. Null for a
    /// pass's first instance that may record inline until a command records into it or a split
    /// keeps it (`RenderCommandEncoder::begin_current_draws`), for a new instance until the
    /// caller that installs it begins it (`RenderCommandEncoder::begin_new_draws`), and for one
    /// recorded inline after a split, which takes none (`render/inline_splits.rs`): an instance
    /// whose secondary never began recorded nothing, and its pass executes none for it.
    draws: vk::CommandBuffer,
    /// Versioned: a draw's `VkPipeline` is found by it (`render/draw_shape.rs`).
    shape: Versioned<PassShape>,
    compatible: Arc<RenderPassObject>,
    /// Something recorded into `draws`: a draw or a visibility query's begin or end.
    recorded: bool,
    /// Its draws test and write the stencil in the shader (`shadow_stencil.rs`).
    shadow: bool,
    /// Its draws test the depth in the shader (`depth_stand_in.rs`).
    shader_tests: bool,
    /// A draw recorded into `draws` reaches the pass's fragment imageblock storage: its draws
    /// record again no more (`render/reroute.rs`). (The barrier before the next such draw follows
    /// `interlocked_writes`.)
    imageblock: bool,
    /// The interlocked storage the draws recorded into `draws` wrote since its last storage
    /// barrier: a draw that reaches any of it records after a barrier from them
    /// (`InterlockedStorage`).
    interlocked_writes: InterlockedStorage,
    /// The memory the draws recorded into `draws` read as their indirect arguments: a draw that
    /// writes any of it records after a barrier from those reads
    /// (`render_target::after_indirect_reads`).
    indirect_reads: IndirectReads<ResourceKey>,
    /// The memory the draws recorded into `draws` wrote before their fragments: a draw whose
    /// fragment shader reads or writes any of it records after a barrier from those writes
    /// (`render_target::after_geometry_writes`).
    geometry_writes: GeometryWrites<ResourceKey>,
    /// A draw recorded into `draws` reaches the pass's tile threadgroup memory, so the next such
    /// draw records into a new instance (`render/tile_memory.rs`).
    tile_memory: bool,
    /// A memory barrier records inside it, between its draws (`barrier_between_draws`): a draw
    /// after it that samples a color attachment's texture records into a new instance, before
    /// which the texture's copy is taken (`texture/attached_reads.rs`).
    barriers: bool,
    /// The layers of the pass its draws render (`render_target::viewed`): views of them, its
    /// shape's view mask, under multiview.
    views: ViewSetup,
    /// What its subpass runs before `draws`, in order (`render/executions.rs`): the secondary
    /// command buffers of the draws before each execution of render commands, and the
    /// executions'.
    pieces: Vec<Piece>,
    /// The shadowed textures its draws use, and their uses of buffers' bytes
    /// (`render/shadows.rs`).
    shadows: InstanceShadows,
    /// What records before it, after the instances before it, in order (`Before`).
    before: Vec<Before>,
    /// The pass colors its draws render through a view of another format (`Reformat`): its
    /// shape's formats are theirs.
    reformats: Vec<Reformat>,
    /// The pipeline of its first recorded draw where the pass has custom sample positions,
    /// which every pipeline drawn in it takes: bound before the instance begins, so a driver that
    /// sets a pass's positions from the bound pipeline when it begins (MoltenVK) begins with
    /// them (`sample_positions.rs`).
    positioned: Option<vk::Pipeline>,
    /// The draws recorded into `draws`, each with the encoder's state at it, while every
    /// recording of the instance is such a draw (`render/reroute.rs`): what records them again
    /// into an instance that leaves more colors out of its subpass, in place of this one. `None`
    /// once anything else recorded into it, or a draw that cannot record again; and from the
    /// start in a pass that holds no extended-range color's `half` shadow
    /// (`ExtendedShadows::keeps_draws`).
    rerecord: Option<Vec<reroute::Rerecord>>,
    /// What the draws recorded into `draws` bound there that still holds (`render/bound.rs`): a
    /// draw leaves out a state command that sets what it holds.
    bound: Bound,
    /// The pixels its draws may write, as the encoder's `drawn` holds the pass's: within the
    /// pass's, which bounds them only where it bounds the pass's (`render/shadows.rs`).
    drawn: clear_reach::DrawnArea,
    /// Its draws record inline in a primary command buffer the encoder holds for it, in its
    /// render pass instance, begun at its first draw (`render/inline_first.rs`): `draws` is that
    /// command buffer.
    inline: bool,
    /// It follows an instance recorded inline only because that one runs no secondary: a draw
    /// whose pipeline is being made, or an execution of render commands, that the instance
    /// before would have taken (`render/inline_first.rs`). The shadowed textures' copies take
    /// the two as one: its draws' shadowed textures and uses are gathered into the first of
    /// them ([`RenderCommandEncoder::gathering`]), copied in before it, and nothing is copied
    /// between them.
    continues: bool,
    /// It records inline after a split (`render/inline_splits.rs`): its commands, kept until its
    /// copies are known, then recorded into the pass's command buffer after the instance
    /// before it. `None` for any other instance.
    kept: Option<inline_splits::Kept>,
    /// Its copies were planned, and recorded, as its group ended before the pass's end
    /// (`render/inline_splits.rs`): the pass's end plans it no more.
    planned: bool,
    /// The colors a draw recorded into it reads through a snapshot, in the feedback loop layout
    /// for its render pass instance (`render/snapshot.rs`, `render_target::Feedback`). Such an
    /// instance records in a secondary, never inline, and runs as one render pass instance.
    feedback: Feedback,
    /// The colors whose images a draw recorded into it binds as a storage image
    /// (`RenderCommandEncoder::storage_colors`): a draw that reads one of them through a
    /// snapshot records into another instance, as the feedback loop layout serves no storage
    /// access (`render/snapshot.rs`).
    storage_colors: u32,
}

/// A pass color a draw renders in another format than the attachment's, where the state's color
/// is another format whose encoding plain Metal stores into the attachment's bytes
/// (`color_reformat::view_format`): the instance's framebuffer holds `target` in place of the
/// pass's attachment, a view of the attachment's image in that format, or where the image takes
/// no such view a stand-in image of it (`copies`), copied from the attachment before the
/// instance and back after it (`attachment_stand_in::stand_in_as`).
#[derive(Clone)]
struct Reformat {
    index: usize,
    target: Target,
    copies: bool,
}

impl Reformat {
    /// Whether `a` and `b` render the same colors through the same formats, alike.
    fn same(a: &[Reformat], b: &[Reformat]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(a, b)| {
                a.index == b.index && a.target.format == b.target.format && a.copies == b.copies
            })
    }
}

/// How a draw renders a color in another format than its attachment's (`Reformat`): through a
/// view of the attachment's image, or a stand-in image copied to and from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Viewing {
    View,
    StandIn,
}

/// `layout` with each of `reformats`' views in place of its attachment: an instance's
/// framebuffer's (`Instance::reformats`).
fn reformatted<'a>(
    layout: &'a PassLayout,
    reformats: &[Reformat],
) -> std::borrow::Cow<'a, PassLayout> {
    if reformats.is_empty() {
        return std::borrow::Cow::Borrowed(layout);
    }
    let mut layout = layout.clone();
    for reformat in reformats {
        if let Some(Some(color)) = layout.colors.get_mut(reformat.index) {
            color.target = reformat.target.clone();
        }
    }
    std::borrow::Cow::Owned(layout)
}

/// The stages a draw runs in inside its render pass instance, where an emulated mesh draw's
/// dispatches and a draw's copies, which record before the pass, run in none.
pub(super) const PASS_DRAW_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::DRAW_INDIRECT.as_raw()
        | vk::PipelineStageFlags2::VERTEX_INPUT.as_raw()
        | vk::PipelineStageFlags2::INDEX_INPUT.as_raw()
        | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT.as_raw()
        | vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
        | vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER.as_raw()
        | vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER.as_raw()
        | vk::PipelineStageFlags2::GEOMETRY_SHADER.as_raw()
        | vk::PipelineStageFlags2::TASK_SHADER_EXT.as_raw()
        | vk::PipelineStageFlags2::MESH_SHADER_EXT.as_raw()
        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw(),
);

/// The stages of `PASS_DRAW_STAGES` that run a draw's fragments. On Metal's Apple GPUs they run
/// after the object and mesh stages of every later draw of the pass: a mesh draw's object and mesh
/// functions read none of an earlier draw's fragment writes, behind a command's barrier or not
/// (an M4).
pub(super) const FRAGMENT_DRAW_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw(),
);

/// The stages of `PASS_DRAW_STAGES` before a draw's fragments: its indirect arguments, vertex
/// input, vertex, tessellation, geometry, object and mesh stages.
pub(super) const GEOMETRY_DRAW_STAGES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::from_raw(PASS_DRAW_STAGES.as_raw() & !FRAGMENT_DRAW_STAGES.as_raw());

/// What records before a render pass instance, after the instances before it and their steps:
/// the steps a draw takes before its pass (a copy of its bytes, an index bounds or rewrite pass) that read
/// what a draw of an earlier instance wrote behind a memory barrier (the module docs).
enum Before {
    /// A copy of a draw's bytes into a new buffer (`draw_source.rs`).
    Copy(CopySteps),
    /// An index bounds pass (`index_bounds.rs`).
    Bounds(BoundsSteps),
    /// An emulated mesh draw's dispatch (`render/mesh.rs`).
    Mesh(mesh::MeshSteps),
    /// An indirect mesh draw's arguments pass (`render/mesh_arguments.rs`).
    MeshArguments(mesh_arguments::ArgumentsSteps),
    /// An emulated tessellation draw's scan (`render/tessellation_scan.rs`).
    Tessellation(tessellation_scan::TessellationSteps),
    /// An index rewrite pass (`index_rewrite.rs`).
    Rewrite(RewriteSteps),
}

impl Before {
    /// The step's uses.
    fn uses(&self) -> &[Use<ResourceKey>] {
        match self {
            Before::Copy(copy) => copy.uses(),
            Before::Bounds(pass) => pass.uses(),
            Before::Mesh(steps) => steps.uses(),
            Before::MeshArguments(steps) => steps.uses(),
            Before::Tessellation(steps) => steps.uses(),
            Before::Rewrite(pass) => pass.uses(),
        }
    }

    /// Record the step into `command_buffer` between a barrier from everything before and one to
    /// everything after.
    ///
    /// # Safety
    ///
    /// `command_buffer` is the encoder's, recording outside a render pass, and only the encoder
    /// records it.
    unsafe fn record_in(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's.
        unsafe {
            match self {
                Before::Copy(copy) => copy.record_in(raw, command_buffer),
                Before::Bounds(pass) => pass.record_in(raw, command_buffer),
                Before::Mesh(steps) => steps.record_in(raw, command_buffer),
                Before::MeshArguments(steps) => steps.record_in(raw, command_buffer),
                Before::Tessellation(steps) => steps.record_in(raw, command_buffer),
                Before::Rewrite(pass) => pass.record_in(raw, command_buffer),
            }
        }
    }
}

/// What a render pass instance's subpass runs, in order (`render/executions.rs`).
#[derive(Clone, Copy)]
enum Piece {
    /// A secondary command buffer of the encoder's draws, ended at `end_encoding`.
    Draws(vk::CommandBuffer),
    /// The encoder's execution of render commands at this index, recorded with its pass.
    Execution(usize),
    /// The encoder's draw at this index whose pipeline was being made, recorded with its pass
    /// once made (`render/pending.rs`).
    Pending(usize),
}

/// Where a draw records (the module docs).
enum Next {
    /// Into the current instance.
    Current,
    /// Into a new instance, which takes the current one's place or follows it (`split`).
    New {
        instance: Box<Instance>,
        split: bool,
    },
}

/// An attachment of a pass and what stands in for it from a draw on (`attachment_stand_in.rs`),
/// or a color attachment's sample planes, made at the draw (`sample_planes.rs`).
enum Standing {
    Color(usize, Target, StandIn),
    DepthStencil(Target, StandIn),
    Planes(usize, Planes),
    /// The pass's hidden depth and stencil (`attachment_stand_in::hidden`).
    Hidden(Target),
}

/// A render encoder (`MTLRenderCommandEncoder`): `Send`, not `Sync`; `end_encoding` consumes it.
/// Each object a call binds to a stage is held until its command buffer completes (or, never
/// committed, is released), as Metal holds what an encoder was handed: one a later call replaced
/// or unbound before any draw used it, and one bound after the last draw, too.
pub struct RenderCommandEncoder {
    /// Its lowering, which runs its calls, where it lowers each inside the call
    /// (`RenderLoweringPolicy::AtCall`): no slot, no lock (`render/deferral.rs`). Boxed where
    /// it is made ([`RenderLowering::assemble`]), so neither the encoder's return nor a caller's
    /// move of it copies the lowering's state, thousands of bytes.
    own: Option<Box<RenderLowering>>,
    /// Else its slot of the device's, which holds its lowering, its checks at the call and its
    /// calls not yet lowered.
    logged: Option<Logged>,
    /// The index its next call takes (`EncoderCall::index`).
    next: u32,
    /// Not `Sync`, as every encoder.
    unshared: PhantomData<std::cell::Cell<()>>,
}

// A program's encoder is a handle, moved by every return and by each of the program's own moves
// of it, so it holds its lowering in a box: the lowering's state is thousands of bytes.
const _: () = assert!(std::mem::size_of::<RenderCommandEncoder>() <= 128);

impl std::fmt::Debug for RenderCommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderCommandEncoder")
            .field("lowered_at_call", &self.lowers_at_call())
            .field("next", &self.next)
            .finish()
    }
}

/// A render encoder's lowering ([`RenderCommandEncoder`]): the encoder's state, which each of its
/// calls runs on.
pub(crate) struct RenderLowering {
    core: EncoderCore,
    /// The pass's instances before the current one, which the draws record into.
    earlier: Vec<Instance>,
    current: Instance,
    /// Versioned, as the state and the amplification are: a draw's shape is derived from them
    /// (`render/draw_shape.rs`).
    layout: Versioned<PassLayout>,
    /// The compatible pass of each shape the encoder's instances took.
    passes: Vec<(PassShape, Arc<RenderPassObject>)>,
    /// The descriptor's `renderTargetArrayLength`: 0 (Metal's default), 1, or the layers of a
    /// layered pass (`PassLayout::layers`).
    render_target_array_length: u32,
    state: Versioned<EncoderState>,
    pipeline: Option<RenderPipelineState>,
    /// The routing of the drawing state last set (`ShapeInputs::routing`), which its check
    /// and each of its draws (their settings and their coverage) read.
    routed: Option<Routed>,
    vertex: ArgumentTable,
    fragment: ArgumentTable,
    /// The object and mesh stages' tables, which a mesh render pipeline's draws bind
    /// (`render/mesh.rs`).
    object: ArgumentTable,
    mesh: ArgumentTable,
    /// The object function's threadgroup memory lengths by index
    /// (`set_object_threadgroup_memory_length`), which its mesh draws bind.
    object_threadgroup_lengths: BTreeMap<u32, u32>,
    visibility: Visibility,
    /// The tile stage's table, which tile dispatches bind (`tile_dispatch.rs`).
    tile: ArgumentTable,
    /// The pass's tile dispatches, each recorded after the instance before it.
    dispatches: Vec<TileDispatch>,
    /// A tile dispatch follows the current instance: the next draw records into a new one.
    dispatched: bool,
    /// The scratch of each attachment the pass's tile dispatches stage, by color index, in the
    /// staging format its view takes.
    scratch: Vec<(u32, Target, metal2vulkan::vk::Format)>,
    /// The shader-side stencil's shadow, made at the pass's first draw that tests the stencil
    /// in the shader (`shadow_stencil.rs`). (A draw that runs the draw's stencil test itself holds
    /// the stencil in it too, as its stand-in.)
    shadow: Option<ShadowStencil>,
    /// A draw of the pass returned `[[stencil]]` and tested it against the shadow, which the pass
    /// announces as it ends.
    returned_stencil: bool,
    /// The depth stand-in, made at the pass's first draw that tests the depth in the shader
    /// (`depth_stand_in.rs`).
    depth_stand_in: Option<DepthStandIn>,
    /// The imageblock's storage, made at the pass's first draw of a pipeline whose fragment
    /// function declares an explicit imageblock, or tile dispatch of a function whose explicit
    /// imageblock stands alone (`fragment_imageblock.rs`).
    imageblock: Option<PassImageblock>,
    /// A tile dispatch of the pass ran a tile function whose explicit imageblock is aliased onto
    /// its implicit one, the attachments, past the bytes they pack to (`render/tile_dispatch.rs`).
    tile_imageblock: bool,
    /// The tessellation factor buffer and scale (`tessellation.rs`).
    factors: Factors,
    /// The pass's rasterization-rate map, whose regions each draw is recorded in
    /// (`rate_mapped_pass.rs`).
    rate_map: Option<RateMappedPass>,
    /// The vertex amplification the draws take (`amplification.rs`).
    amplification: Versioned<Amplification>,
    /// A memory barrier follows draws of the current instance: the next draw records into a new
    /// instance after a barrier from everything before (`fence.rs`).
    barrier: bool,
    /// The splits such barriers made, for the split's degrade notice.
    barrier_splits: usize,
    /// What MoltenVK restarts its Metal render pass for inside the subpass, storing the
    /// attachments there, was recorded: a barrier between draws whose destination reads input
    /// attachments (`barrier_between_draws`), or a tessellated draw (`Device::degrade_pass`).
    driver_restarts: bool,
    /// The splits between draws that reach the pass's tile threadgroup memory, for the split's
    /// degrade notice (`render/tile_memory.rs`).
    tile_memory_splits: usize,
    /// The tile functions' threadgroup arguments and the pass's tile threadgroup memory
    /// (`render/tile_memory.rs`).
    tile_memory: tile_memory::TileMemory,
    /// What `use_resource(s)` and `use_heap(s)` made resident, each in its render stages
    /// (`render/arguments.rs`).
    residency: Residency,
    /// The indirect command buffers it made resident for writing (`written_commands.rs`).
    writers: Writers,
    /// The executions of render commands, in encoding order, left to record with the pass
    /// (`render/executions.rs`).
    executions: Vec<RenderExecution>,
    /// The draws whose pipelines were being made, in encoding order, left to record with the
    /// pass (`render/pending.rs`).
    pending: Vec<PendingDraw>,
    /// The encoder's draw calls that returned `Ok` so far: the next one's order among them,
    /// which a draw whose pipeline is being made keeps for its failure (`FailedPipeline::draw`).
    draw_calls: usize,
    /// Its tile dispatch calls that returned `Ok` so far, likewise for a tile dispatch
    /// (`render/pending_tile.rs`).
    tile_dispatch_calls: usize,
    /// The encoder records an execution's commands (`render/executions.rs`): a draw that needs
    /// another instance than the execution's is refused.
    replaying: bool,
    /// The place among the command buffer's ended encoders of the encoder whose execution it
    /// replays (`RenderExecution`), which names a draw of the commands it drops
    /// (`CommandBuffer::drop_late_draw`); `None` where it replays none.
    executing_encoder: Option<usize>,
    /// The encoder records the current instance's draws again into an instance that leaves more
    /// colors out of its subpass (`render/reroute.rs`): a draw that would record elsewhere, or
    /// whose variant the interlocked storage form refuses, fails the reroute.
    rerouting: bool,
    /// The buffer forms its draws' variants take (`BufferForms`): the large-buffer form where it
    /// records an execution's commands at its host step, or settles their variants at commit
    /// (`render/executions.rs`), so no buffer a command binds asks for another variant; the
    /// replay form where it records an execution's GPU form (`render/executions/gpu.rs`).
    forms: BufferForms,
    /// Where it records an execution's GPU form, the replay's draws, which its draw records in
    /// place of its own (`render/executions/gpu.rs`).
    replayed: Option<executions::ReplayedDraws>,
    /// Where the encoder records an execution's commands, the tracked uses of the pass's draws
    /// before the execution, in the stages a draw runs in inside its pass (`PASS_DRAW_STAGES`):
    /// which an emulated mesh command's dispatches, recorded before the pass, must not touch but
    /// as Metal orders them.
    drawn_before: Vec<Use<ResourceKey>>,
    /// The encoder records an execution's commands and a command's barrier stands: the next draw
    /// records into the execution's next run, after a barrier from everything before
    /// (`render/executions.rs`).
    run_barrier: bool,
    /// The encoder records an execution's commands and the command it draws holds a barrier,
    /// which orders its draw after every draw before it (`render/executions.rs`).
    command_barrier: bool,
    /// The tracked writes of the draws and tile dispatches since the last memory barrier over
    /// buffers or textures (`fence.rs`), which no barrier orders before the draws after them.
    unordered: UnorderedWrites<ResourceKey>,
    /// The buffer of zero control-point indices a patch draw of a pipeline with a control-point
    /// index type reads where the draw gives none, made at the first such draw.
    zero_indices: Option<Buffer>,
    /// The tracked writes of the pass's tile dispatches, which Metal orders before no later
    /// draw's fetch of its arguments or indices: no barrier clears them (Metal refuses a memory
    /// barrier after the tile stage, and the draw read the bytes from before the write in every
    /// run measured, with a barrier before the dispatch too).
    unordered_dispatches: UnorderedWrites<ResourceKey>,
    /// A memory barrier over buffers or textures was encoded: an indexed indirect draw after it
    /// reads bytes a draw wrote since it in place (`draw_source.rs`'s racing copy is taken only
    /// before any barrier, where Metal's answer was measured fixed).
    barriered: bool,
    /// A memory barrier was encoded inside the pass (before the next draw, or between draws
    /// over the render targets): a command after it may read what one before it wrote of an
    /// attachment (`linear_attachment::choose`).
    barrier_inside: bool,
    /// A command of the pass may reach memory it does not name: a draw or tile dispatch whose
    /// functions load pointers from memory (a root table: dynamic roots, which reach every live
    /// buffer, `live_buffers.rs`) or handles (a handle table, which reaches every texture of
    /// the device's heap set) (`linear_attachment::choose`).
    reaches_unnamed: bool,
    /// The instances (by index among the pass's) that follow a memory barrier: before each, the
    /// colors rendered through a copied stand-in over a buffer's bytes are merged back into them
    /// and copied in again (`linear_merge.rs`), so a draw after the barrier reads what the
    /// pass drew.
    reload_at: BTreeSet<usize>,
    /// The pass has a draw call, whether or not it draws anything, an execution of render
    /// commands or a tile dispatch: a `Clear` load clears over whole tiles (`clear_reach.rs`).
    called: bool,
    /// A depth/stencil state with depth writes was set in the pass, or an indirect command
    /// buffer's commands that set their own ran in it.
    depth_writes: bool,
    /// A draw that records a command (not one of no vertices, indices or instances; an indirect
    /// one whatever its arguments), or such an execution, ran in the pass under a state with
    /// depth writes, which hands the device's pass that state: where it did not though
    /// `depth_writes` holds, a depth whose device's pass is Metal's own keeps its signs, where
    /// Metal's pass stores its words without them (`Texture::degrade_kept_signs`).
    drew_depth_writes: bool,
    /// The pass draws: a draw call drew vertices, indices, patches or meshes through a pipeline
    /// that rasterizes, an execution of render commands or a tile dispatch ran: its draws reach
    /// tiles (`clear_reach.rs`'s `PassReach`).
    drew: bool,
    /// The pixels the pass's draws may write: their viewports' within their scissors
    /// (`clear_reach::DrawnArea`).
    drawn: clear_reach::DrawnArea,
    /// What the pass's draws write of each color, and blend, for the choice of a bound linear
    /// attachment's rendering (`linear_attachment::Coverage`).
    coverage: linear_attachment::Coverage,
    /// The color indices a draw of the pass read as input attachments (framebuffer fetch's input
    /// form): its descriptor sets view the attachment's image, which then stays the pass's
    /// (`kept_bytes.rs`).
    inputs_read: BTreeSet<u32>,
    /// The kept-attachment notices of the layout as it stands (`degrade_kept_attachments`),
    /// decided at the first draw or execution under it; `None` until then, and again each time
    /// the encoder changes its layout: installs another, sets a color's store action or resolve,
    /// loads a memoryless color, or reroutes the pass.
    kept_attachments: Option<tile_store::KeptAttachments>,
    /// The pass's extended-range color attachments an earlier draw or execution of render
    /// commands may have written, or whose clear Metal's tile memory holds as values its codes
    /// do not read back (`cleared_off_codes`), by index (`degrade_extended_range_blends`).
    extended_written: BTreeSet<u32>,
    /// The `half` shadows of the pass's extended-range colors (`render/extended_shadow.rs`).
    extended: ExtendedShadows,
    /// The pass's fragment-stage writes and the copies of them its vertex stages read
    /// (`render/pass_start_reads.rs`).
    pass_start: PassStartReads,
    /// The index of the encoder's last `set_primitive_restart_enabled` call, where its indexed
    /// strips restart (`render/restart.rs`); `None` before any, where they restart at the index
    /// type's all-ones index.
    restart: Option<u64>,
    /// The bind state the encoder's last draw bound, with its shape, which each draw brings to
    /// its own in place (`live_bind.rs`). Shared only so that a draw holds it while it calls the
    /// encoder's other methods.
    live: Arc<Mutex<LiveBind>>,
    /// Its pass's descriptor and its calls so far, where the device had a buffer over host pages
    /// as it opened (`render/replay.rs`): `None` where it logs nothing, as when it is encoded
    /// again.
    log: Option<RenderLog>,
    /// The last draw's variant, plan and specialization under which the bound values' notices
    /// were quiet (`Device::degrade_bound_values`), where it was.
    quiet_bound_values: Option<QuietBoundValues>,
    /// The views and sampler descriptions its draws' binds made or found, taken from the
    /// device's ended render encoders' and given back as it ends (`binding::BindMemo`).
    bind_memo: binding::BindMemo,
    /// The shapes its draws derived, with what from (`render/draw_shape.rs`).
    shape_memo: ShapeMemo,
    /// What its draws retained of their own, each once ([`LastRetained`]).
    last_retained: LastRetained,
    /// The residency's generation as a command last noted the heaps of what it made resident
    /// (`command_encoded`): no command notes them again before it changes.
    heaps_noted: Option<u64>,
    /// Whether the pass's first instance records inline (`render/inline_first.rs`).
    inline: inline_first::Inline,
    /// What the draws recorded inline into the pass's command buffer bound there, in its every
    /// render pass instance (`render/inline_splits.rs`).
    held_bound: Bound,
    /// The pass's color group, where the device draws its colors in groups: the encoder draws
    /// that group's (`render/color_groups.rs`).
    grouped: Option<Box<color_groups::Grouped>>,
}

/// What a render encoder's draws retained of their own (their pipeline states, their variants'
/// objects, their `VkPipeline`s and the samplers their binds named), each held once: a draw
/// retains an object only where no earlier draw of the encoder retained it, by another reference
/// to it (`retained.rs`). What an encoder retains goes to its command buffer, which holds it until
/// it completes, so an object retained once is held for every later draw of the encoder; and as
/// each is held here, an object seen again is the same one. The command buffer lets go of its
/// holds last first, so an object held twice went at its first hold: holding it once there lets
/// it go at the same point.
#[derive(Default)]
struct LastRetained {
    pipelines: Kept<DrawingState>,
    objects: Kept<Arc<VariantObjects>>,
    made: Kept<Arc<RawPipeline>>,
    samplers: Kept<Arc<OwnedSampler>>,
}

/// The objects of each kind a render encoder keeps in place ([`Kept`]): a pass's draws of a few
/// pipeline states retain no more, and keep them with no storage of their own.
const KEPT_INLINE: usize = 8;

/// Objects an encoder's draws retained, each once: the first [`KEPT_INLINE`] in place, the others
/// in a list made at the first of them.
struct Kept<T> {
    inline: [Option<T>; KEPT_INLINE],
    more: Vec<T>,
}

impl<T> Default for Kept<T> {
    fn default() -> Kept<T> {
        Kept {
            inline: [const { None }; KEPT_INLINE],
            more: Vec::new(),
        }
    }
}

impl<T> Kept<T> {
    /// Whether `object` is among them, by `same`, the latest kept compared first.
    fn holds(&self, object: &T, same: impl Fn(&T, &T) -> bool) -> bool {
        let inline = self.inline.iter().rev().flatten();
        self.more
            .iter()
            .rev()
            .chain(inline)
            .any(|kept| same(kept, object))
    }

    /// Keep `object`, which is not among them.
    fn keep(&mut self, object: T) {
        match self.inline.iter_mut().find(|slot| slot.is_none()) {
            Some(slot) => *slot = Some(object),
            None => self.more.push(object),
        }
    }
}

impl LastRetained {
    /// Retain into `core` a draw's pipeline state, variant objects and made `VkPipeline`, each
    /// unless an earlier draw of the encoder retained it.
    fn retain_draw(
        &mut self,
        core: &mut EncoderCore,
        (pipeline, objects, made): (DrawingState, Arc<VariantObjects>, Option<Arc<RawPipeline>>),
    ) {
        if !self.pipelines.holds(&pipeline, DrawingState::same) {
            core.retain_shared(pipeline.held());
            self.pipelines.keep(pipeline);
        }
        if !self.objects.holds(&objects, Arc::ptr_eq) {
            core.retain_shared(objects.clone());
            self.objects.keep(objects);
        }
        if let Some(made) = made {
            if !self.made.holds(&made, Arc::ptr_eq) {
                core.retain_shared(made.clone());
                self.made.keep(made);
            }
        }
    }

    /// Retain into `core` the samplers a draw's bind named, each unless an earlier draw of the
    /// encoder retained it: by another reference to it, the last first, so they go in their
    /// order (`retained.rs`). `samplers` is left empty, its storage kept.
    fn retain_samplers(&mut self, core: &mut EncoderCore, samplers: &mut Vec<Arc<OwnedSampler>>) {
        for sampler in samplers.drain(..).rev() {
            if !self.samplers.holds(&sampler, Arc::ptr_eq) {
                core.retain_shared(sampler.clone());
                self.samplers.keep(sampler);
            }
        }
    }
}

/// A variant, plan and specialization under which `Device::degrade_bound_values` was quiet: no
/// clause of a note it tested held. Which hold is a function of the three alone, so a draw of
/// the same three announces nothing there. The variant and the plan are held, so a pointer seen
/// again is the same object.
struct QuietBoundValues {
    variant: Arc<RenderVariant>,
    plan: Arc<BindingPlan>,
    specialization: SpecializationKey,
}

impl QuietBoundValues {
    /// Whether a draw of `variant`, `plan` and `specialization` is of the three.
    fn holds(
        &self,
        variant: &Arc<RenderVariant>,
        plan: &Arc<BindingPlan>,
        specialization: &SpecializationKey,
    ) -> bool {
        Arc::ptr_eq(&self.variant, variant)
            && Arc::ptr_eq(&self.plan, plan)
            && self.specialization == *specialization
    }
}

/// A drawing state's routing into a pass's colors (`ShapeInputs::routing`), with the state and
/// the pass's color formats it was found for, and the colors a draw of it zeroes and drops by
/// that routing (`ShapeInputs::routed_colors`): each a function of those alone.
struct Routed {
    state: DrawingState,
    pass: [Option<PixelFormat>; MAX_COLOR_ATTACHMENTS],
    routing: Result<Arc<Routing>, u32>,
    /// `None` where the state's routing is refused.
    colors: Option<Arc<RoutedColors>>,
}

/// The pass colors a draw of a drawing state zeroes, and the state's colors it drops, by its
/// routing (`ShapeInputs::zeroed_colors`, `ShapeInputs::dropped_colors`).
#[derive(Clone, PartialEq)]
struct RoutedColors {
    zeroed: BTreeMap<u32, PixelFormat>,
    dropped: BTreeSet<u32>,
}

impl std::fmt::Debug for RenderLowering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderLowering")
            .field("extent", &self.layout.extent)
            .field("samples", &self.layout.samples)
            .finish()
    }
}

impl CommandBuffer {
    /// `renderCommandEncoderWithDescriptor:`: an encoder of the pass `descriptor` describes.
    /// Refused as Metal's validation refuses the descriptor (`render_pass.rs`), and as
    /// [`CommandBuffer::compute_command_encoder`] is: while another encoder of this command
    /// buffer is open, after one was abandoned, after commit.
    pub fn render_command_encoder(
        &self,
        descriptor: &RenderPassDescriptor,
    ) -> Result<RenderCommandEncoder, Error> {
        // The last render encoder's end, where its call left it to the device's lowering thread.
        self.settle_end();
        // Metal's process ends here under some protection options (`command_buffer/protection.rs`).
        self.refuse_protected_encoder("render_command_encoder")?;
        self.open_render(descriptor, None, None)
            .map(RenderCommandEncoder::new)
    }

    /// [`render_command_encoder`](Self::render_command_encoder)'s encoder, or with `again` the
    /// residency an ended encoder of the committed command buffer started with, that encoder
    /// opened again (`render/replay.rs`): over a new `VkCommandBuffer` with no refusal of a
    /// committed command buffer, starting with that residency, logging nothing but where its pass
    /// takes several color groups (`render/color_groups.rs`), and keeping `screen`, the screen
    /// size of the pass's rasterization-rate map as the ended encoder took it, through a reset of
    /// the map (`rate_mapped_pass.rs`). Its layout is the first color group's where the device
    /// draws the pass's colors in groups.
    fn open_render(
        &self,
        given: &RenderPassDescriptor,
        again: Option<Residency>,
        screen: Option<[u32; 2]>,
    ) -> Result<Box<RenderLowering>, Error> {
        const CALL: &str = "render_command_encoder";
        let device = self.device();
        let screen = screen.or_else(|| {
            (given.rasterization_rate_map.as_ref()).map(|map| map.state().screen_size())
        });
        rate_mapped_pass::check_descriptor(CALL, device, given)?;
        let taken = render_pass::surface_stencil_loads(given);
        let descriptor = &*taken;
        let full = render_pass::layout(CALL, self, descriptor)?;
        // A pass of more colors than the device draws at once draws them in groups, the
        // encoder's the first group's (`render/color_groups.rs`).
        let grouping = color_groups::grouping(device, &full);
        let groups = grouping.as_ref().map_or(1, ColorGroup::groups);
        if groups > 1 {
            color_groups::check_twin(&full)?;
        }
        // Each later group runs the encoder's calls again from its log, which such a pass keeps
        // whatever the device's rebinds, as does a pass over a rasterization-rate map, which a
        // reset of the map before its submission encodes again (`rate_mapped_pass.rs`).
        let log = match (&again, groups > 1) {
            (_, true) => Some(device.shared().scheduler.spare_logs().render(given, screen)),
            (None, false) => (device.shared().rebinds.any()
                || given.rasterization_rate_map.is_some())
            .then(|| device.shared().scheduler.spare_logs().render(given, screen)),
            (Some(_), false) => None,
        };
        let projected;
        let (mut layout, mut full, descriptor) = match &grouping {
            Some(group) => {
                projected = color_groups::project_descriptor(descriptor, group);
                (color_groups::project(&full, group), Some(full), &projected)
            }
            None => (full, None, descriptor),
        };
        let rate_map = RateMappedPass::of(CALL, descriptor, &layout, screen)?;
        let shape = PassShape::of(&layout)?;
        let compatible = render_target::shared_compatible_pass(device, &shape)?;
        if again.is_none() {
            if let Some(problem) = self.state().refusal() {
                return Err(invalid(CALL, problem));
            }
        }
        // Its first instance may record inline (`render/inline_first.rs`), not where it encodes
        // an ended encoder's calls again, nor in a pass of several color groups, whose later
        // groups record after the first group's pass: its draws' secondary begins only where a
        // command records into it (`RenderCommandEncoder::begin_current_draws`).
        let extended = ExtendedShadows::of(descriptor, &layout, rate_map.is_some());
        let opens = match again.is_none() && groups == 1 {
            true => inline_first::opens_inline(device, &layout, rate_map.is_some(), &extended),
            false => Err(NotInline::Encoders),
        };
        if let Err(why) = opens {
            device.shared().counts.pass_not_inline(why);
        }
        let inline = opens.is_ok();
        // And its instances after a split, where no attachment is a shadowed texture
        // (`render/inline_splits.rs`).
        let splits = inline && !shadows::attachments_shadowed(&layout);
        let draws = match inline {
            true => vk::CommandBuffer::null(),
            false => self.begin_draws(&compatible)?,
        };
        // A refusal past here leaves the secondary command buffer unused, freed with the pool.
        let mut core = match again {
            None => EncoderCore::open(self, CALL, EncoderKind::Render, false)?,
            Some(_) => EncoderCore::replaying(self, false)?,
        };
        if let Some(pass) = &rate_map {
            let (map, state) = pass.taken();
            core.take_rate_map(map, state);
        }
        // The whole pass, every group's colors.
        let whole = full.as_ref().unwrap_or(&layout);
        if let Some(error) = whole.fails_command_buffer {
            core.fail_command_buffer(error);
        }
        // The pass uses its attachments from its open, as Metal's encoder does: the bytes over
        // host pages they lie in are held from here, as a command that names them holds them
        // (`rebind.rs`). So a rebind of them while it is open is refused, and a detach meanwhile
        // leaves the images that lay in them until its command buffer goes (`detach.rs`): its
        // first instance may record with them before its end (`render/inline_first.rs`).
        core.hold_bytes(render_target::attachment_bytes(whole));
        // The pass's loads read a new texture's unwritten texels (`fresh_unwritten.rs`): the
        // descriptor's loads, before a pass of several groups loads a `DontCare` depth or stencil.
        FreshUnwritten::begin_pass(device, whole)?;
        render_pass::degrade_depth_clear(device, whole);
        if groups > 1 {
            color_groups::settle_depth_stencil(&mut layout);
            if let Some(full) = full.as_mut() {
                color_groups::settle_depth_stencil(full);
            }
        }
        let residency = match again {
            Some(residency) => residency,
            None => self.inherited_residency(true),
        };
        let grouped = match (grouping, full) {
            (Some(group), Some(full)) => Some(Box::new(color_groups::Grouped::first(
                group, full, &residency,
            ))),
            _ => None,
        };
        let opening = Opening::new(
            descriptor,
            (layout, extended),
            (shape, compatible),
            rate_map,
            inline.then_some(splits),
            residency,
            log,
            grouped,
        );
        Ok(RenderLowering::assemble(core, draws, opening))
    }
}

/// What a render encoder opens with but its core and the secondary command buffer of its first
/// instance ([`RenderCommandEncoder::assemble`]): its pass's layout (a color group's, where the
/// device draws the pass's colors in groups), that layout's shape and compatible pass, its
/// rasterization-rate map, its extended-range shadows and the extended-range colors whose clear
/// Metal's tile memory holds off its codes, whether its first instance records inline (and its
/// instances after a split with it), its residency, its log and its color group.
struct Opening {
    layout: PassLayout,
    /// The descriptor's `renderTargetArrayLength`, as the encoder keeps it.
    render_target_array_length: u32,
    shape: PassShape,
    compatible: Arc<RenderPassObject>,
    rate_map: Option<RateMappedPass>,
    extended: ExtendedShadows,
    extended_written: BTreeSet<u32>,
    inline: Option<bool>,
    residency: Residency,
    log: Option<RenderLog>,
    grouped: Option<Box<color_groups::Grouped>>,
}

impl Opening {
    /// The opening of a pass of `descriptor` (projected to the layout's color group where it
    /// has one) and `layout`, with its extended-range shadows (`ExtendedShadows::of` of them, made
    /// once), its shape and compatible pass, its map, its inline first instance (`Some(splits)`),
    /// its residency, its log and its group.
    fn new(
        descriptor: &RenderPassDescriptor,
        (layout, extended): (PassLayout, ExtendedShadows),
        (shape, compatible): (PassShape, Arc<RenderPassObject>),
        rate_map: Option<RateMappedPass>,
        inline: Option<bool>,
        residency: Residency,
        log: Option<RenderLog>,
        grouped: Option<Box<color_groups::Grouped>>,
    ) -> Opening {
        Opening {
            // A length of 1 or more is the layout's layers, a `u32`.
            render_target_array_length: match descriptor.render_target_array_length {
                0 => 0,
                _ => layout.layers,
            },
            extended,
            extended_written: cleared_off_codes(descriptor),
            layout,
            shape,
            compatible,
            rate_map,
            inline,
            residency,
            log,
            grouped,
        }
    }
}

impl RenderLowering {
    /// An encoder over `core` of the pass `opening` describes, its first instance's draws
    /// recording into `draws` (null where it records inline), made in its box.
    fn assemble(core: EncoderCore, draws: vk::CommandBuffer, opening: Opening) -> Box<Self> {
        let Opening {
            layout,
            render_target_array_length,
            shape,
            compatible,
            rate_map,
            extended,
            extended_written,
            inline,
            residency,
            log,
            grouped,
        } = opening;
        let rerecord = extended.keeps_draws();
        // The bind state the device's last ended render encoder left (`live_bind.rs`).
        let live = core.device().shared().live_binds.take();
        // The views and sampler descriptions its binds made or found (`binding::BindMemo`).
        let bind_memo = core.device().shared().bind_memos.take();
        Box::new(RenderLowering {
            core,
            earlier: Vec::new(),
            current: Instance {
                draws,
                shape: Versioned::new(shape.clone()),
                compatible: Arc::clone(&compatible),
                recorded: false,
                shadow: false,
                shader_tests: false,
                imageblock: false,
                interlocked_writes: InterlockedStorage::NONE,
                indirect_reads: IndirectReads::new(),
                geometry_writes: GeometryWrites::new(),
                tile_memory: false,
                barriers: false,
                views: ViewSetup::default(),
                pieces: Vec::new(),
                shadows: InstanceShadows::default(),
                before: Vec::new(),
                reformats: Vec::new(),
                positioned: None,
                rerecord: rerecord.then(Vec::new),
                bound: Bound::default(),
                drawn: clear_reach::DrawnArea::default(),
                inline: false,
                continues: false,
                kept: None,
                planned: false,
                feedback: Feedback::NONE,
                storage_colors: 0,
            },
            state: Versioned::new(EncoderState::metal_defaults(
                rate_map
                    .as_ref()
                    .map_or(layout.extent, RateMappedPass::area),
            )),
            render_target_array_length,
            visibility: Visibility::new(layout.visibility.clone()),
            layout: Versioned::new(layout),
            passes: vec![(shape, compatible)],
            pipeline: None,
            routed: None,
            vertex: ArgumentTable::default(),
            fragment: ArgumentTable::default(),
            object: ArgumentTable::default(),
            mesh: ArgumentTable::default(),
            object_threadgroup_lengths: BTreeMap::new(),
            tile: ArgumentTable::default(),
            dispatches: Vec::new(),
            dispatched: false,
            scratch: Vec::new(),
            shadow: None,
            returned_stencil: false,
            depth_stand_in: None,
            imageblock: None,
            tile_imageblock: false,
            factors: Factors::default(),
            rate_map,
            amplification: Versioned::new(Amplification::default()),
            barrier: false,
            barrier_splits: 0,
            driver_restarts: false,
            tile_memory_splits: 0,
            tile_memory: tile_memory::TileMemory::default(),
            residency,
            writers: Writers::default(),
            executions: Vec::new(),
            pending: Vec::new(),
            draw_calls: 0,
            tile_dispatch_calls: 0,
            replaying: false,
            executing_encoder: None,
            rerouting: false,
            forms: BufferForms::Bound,
            replayed: None,
            drawn_before: Vec::new(),
            run_barrier: false,
            command_barrier: false,
            unordered: UnorderedWrites::new(),
            zero_indices: None,
            unordered_dispatches: UnorderedWrites::new(),
            barriered: false,
            barrier_inside: false,
            reaches_unnamed: false,
            reload_at: BTreeSet::new(),
            called: false,
            depth_writes: false,
            drew_depth_writes: false,
            drew: false,
            drawn: clear_reach::DrawnArea::default(),
            coverage: linear_attachment::Coverage::default(),
            inputs_read: BTreeSet::new(),
            kept_attachments: None,
            extended_written,
            extended,
            pass_start: PassStartReads::default(),
            restart: None,
            live,
            log,
            quiet_bound_values: None,
            bind_memo,
            shape_memo: ShapeMemo::default(),
            last_retained: LastRetained::default(),
            heaps_noted: None,
            inline: match inline {
                Some(splits) => inline_first::Inline::Open { splits },
                None => inline_first::Inline::Never,
            },
            held_bound: Bound::default(),
            grouped,
        })
    }
}

impl CommandBuffer {
    /// A secondary command buffer of this command buffer's pool, begun inside a pass compatible
    /// with `compatible`: the draws of a render pass instance (the module docs).
    fn begin_draws(&self, compatible: &RenderPassObject) -> Result<vk::CommandBuffer, Error> {
        let mut state = self.state();
        let draws = self.allocate_secondary(&mut state)?;
        let inheritance = vk::CommandBufferInheritanceInfo::default()
            .render_pass(compatible.raw)
            .subpass(0);
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                    | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            )
            .inheritance_info(&inheritance);
        // SAFETY: the command buffer was just allocated from the pool, under the state lock;
        // the compatible pass lives as long as the encoder, and its command buffer after.
        unsafe {
            self.device()
                .shared()
                .vk()
                .begin_command_buffer(draws, &begin)
        }
        .map_err(vulkan)?;
        let counts = &self.device().shared().counts;
        counts.encoder_command_buffer_begun(Begun::Secondary);
        Ok(draws)
    }
}

/// Bytes of one index of `ty`.
pub(super) fn index_size(ty: IndexType) -> u64 {
    match ty {
        IndexType::UInt16 => 2,
        IndexType::UInt32 => 4,
    }
}

/// Where a draw reads the indices of `size` bytes of `buffer` from `offset`, a multiple of
/// `size`: the offset, or the first index position past the buffer's length where the offset
/// lies further (every position from either is past the buffer, and the binding layer bounds
/// both alike); and the whole indices from there before where Metal's fetch stops reading the
/// buffer (`IndexBufferRef::fetch_end`: its length rounded up to 4 bytes).
fn indices_inside(buffer: &Buffer, offset: u64, size: u64) -> (u64, u64) {
    let offset = offset.min(buffer.length().next_multiple_of(size));
    let end = IndexBufferRef::new(0, buffer.length()).fetch_end(offset);
    (offset, end.saturating_sub(offset) / size)
}

/// Whether `state` belongs to the fragment shader or fragment output state, which a pipeline
/// with rasterization disabled does not include: it is none of such a pipeline's dynamic states,
/// so setting it after binding one is `VUID-vkCmdDraw-None-08608`, and binding one leaves what an
/// earlier pipeline set it to unknown. The depth and stencil test states are fragment shader state.
fn fragment_state(state: vk::DynamicState) -> bool {
    matches!(
        state,
        vk::DynamicState::BLEND_CONSTANTS
            | vk::DynamicState::STENCIL_COMPARE_MASK
            | vk::DynamicState::STENCIL_WRITE_MASK
            | vk::DynamicState::STENCIL_REFERENCE
            | vk::DynamicState::DEPTH_TEST_ENABLE
            | vk::DynamicState::DEPTH_WRITE_ENABLE
            | vk::DynamicState::DEPTH_COMPARE_OP
            | vk::DynamicState::STENCIL_TEST_ENABLE
            | vk::DynamicState::STENCIL_OP
    )
}

/// The states `VK_EXT_extended_dynamic_state` defines (core in Vulkan 1.3).
fn extended_state(state: vk::DynamicState) -> bool {
    matches!(
        state,
        vk::DynamicState::CULL_MODE
            | vk::DynamicState::FRONT_FACE
            | vk::DynamicState::PRIMITIVE_TOPOLOGY
            | vk::DynamicState::DEPTH_TEST_ENABLE
            | vk::DynamicState::DEPTH_WRITE_ENABLE
            | vk::DynamicState::DEPTH_COMPARE_OP
            | vk::DynamicState::STENCIL_TEST_ENABLE
            | vk::DynamicState::STENCIL_OP
            | vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE
            | vk::DynamicState::PRIMITIVE_RESTART_ENABLE
    )
}

impl DrawDynamic {
    /// What a draw of a variant of `graphics` takes of its dynamic states on a device of
    /// `extended`, with the write mask command where `masks`: the pipeline's dynamic states, but
    /// for those of a part it lacks (the fragment states of a pipeline that discards
    /// rasterization, `fragment_state`), and whether the encoder records each of them there.
    fn of(
        graphics: &GraphicsState,
        extended: Option<ExtendedDynamicState>,
        masks: bool,
    ) -> DrawDynamic {
        let states: Vec<vk::DynamicState> = graphics
            .dynamic_states
            .iter()
            .map(|&state| state.into())
            .filter(|&state| !(graphics.rasterizer_discard && fragment_state(state)))
            .collect();
        // The write masks are recorded with a draw's two draws, on a device with their command
        // (`render/snapshot.rs`).
        let recorded = states.iter().all(|&state| {
            let recorded = RECORDED.contains(&state)
                || state == vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE
                || (state == vk::DynamicState::COLOR_WRITE_MASK_EXT && masks);
            let restart_missing = state == vk::DynamicState::PRIMITIVE_RESTART_ENABLE
                && extended.is_some_and(|e| e.primitive_restart_enable.is_none());
            recorded && !(extended_state(state) && extended.is_none()) && !restart_missing
        });
        DrawDynamic { states, recorded }
    }
}

/// No colors or slots: what a draw's set of them is compared with where it names none
/// ([`RenderCommandEncoder::settle_draw`]).
static NO_INDICES: BTreeSet<u32> = BTreeSet::new();

/// The dynamic states a draw records, beside `VERTEX_INPUT_BINDING_STRIDE` (recorded with the
/// vertex buffers).
const RECORDED: [vk::DynamicState; 16] = [
    vk::DynamicState::VIEWPORT,
    vk::DynamicState::SCISSOR,
    vk::DynamicState::DEPTH_BIAS,
    vk::DynamicState::BLEND_CONSTANTS,
    vk::DynamicState::STENCIL_COMPARE_MASK,
    vk::DynamicState::STENCIL_WRITE_MASK,
    vk::DynamicState::STENCIL_REFERENCE,
    vk::DynamicState::CULL_MODE,
    vk::DynamicState::FRONT_FACE,
    vk::DynamicState::PRIMITIVE_TOPOLOGY,
    vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
    vk::DynamicState::DEPTH_TEST_ENABLE,
    vk::DynamicState::DEPTH_WRITE_ENABLE,
    vk::DynamicState::DEPTH_COMPARE_OP,
    vk::DynamicState::STENCIL_TEST_ENABLE,
    vk::DynamicState::STENCIL_OP,
];

/// The checks of a pipeline state set on an encoder, over the encoder's pass and state as they stand
/// (`ShapeInputs`): the lowering's, and the program's encoder's at the call (`render/call_checks.rs`).
impl ShapeInputs<'_> {
    /// Refuse `pipeline` for this pass in `call` as [`RenderCommandEncoder::set_render_pipeline_state`]
    /// says.
    pub(super) fn check_pipeline(
        &self,
        call: &'static str,
        state: &RenderPipelineState,
    ) -> Result<(), Error> {
        let pipeline = match state.kind() {
            RenderStateKind::Graphics(state) => state,
            RenderStateKind::Tile(state) => return self.check_tile(call, state),
        };
        let descriptor = pipeline.descriptor();
        let graphics = &pipeline.base().graphics;
        let refused = |index: u32| {
            draw_problem(
                call,
                DrawProblem::AttachmentFormat(Attachment::Color(index)),
            )
        };
        // Each state color is drawn into the pass color at its place in Metal's per-sample record
        // (`routing`), where it stores a whole color into one attachment; refused elsewhere. A
        // routed draw that fetches or reads a color, or declares an imageblock, and a mesh render
        // pipeline's state routed at all, are refused (unmeasured). A pass whose draws may fetch
        // through input attachments routes the others, its input attachments at their indices.
        let routing = self.routing(pipeline).map_err(refused)?;
        let mesh = descriptor.vertex().is_none();
        let reads = |index: u32| {
            graphics
                .input_attachments
                .iter()
                .any(|a| a.color_index == index)
                || graphics
                    .storage_attachments
                    .iter()
                    .any(|a| a.color_index == index)
        };
        if let Some(location) = routing.moved() {
            let fetches = !graphics.input_attachments.is_empty()
                || !graphics.storage_attachments.is_empty()
                || graphics.fragment_imageblock.is_some();
            if mesh || fetches {
                return Err(refused(location));
            }
        }
        // A color of the pass at no place of the state's record: zeroed at each draw's covered
        // samples or kept, as plain Metal stores it (`zeroed_colors`); refused for a state that
        // rasterizes and writes no color, and a mesh render pipeline's state (unmeasured). A state
        // that rasterizes nothing covers no sample, so it stores into no attachment: it binds in a
        // pass of any colors it does not name, and its draws leave every one of them as they were
        // (`unrasterized.rs`). A color of the pipeline at no place of the pass's: its output is
        // dropped, as plain Metal drops it, by the variant `dropped_colors` names, unless the
        // fragment function reads that color or the state is a mesh render pipeline's (not built).
        if let Some(&index) = routing.uncovered.values().next() {
            if mesh || (!draw_shape::writes_colors(pipeline) && !graphics.rasterizer_discard) {
                return Err(refused(index));
            }
            // A pass color the draw keeps takes a blend element that writes nothing
            // (`pipeline/graphics.rs`): on a device without `independentBlend` the state's
            // elements must write nothing too.
            let independent = self
                .device
                .enabled()
                .features
                .contains(&VkFeature::IndependentBlend);
            let writes = graphics
                .color_attachments
                .iter()
                .flatten()
                .any(|a| !a.write_mask.is_empty() || a.blend.is_some());
            if !independent && writes && self.zeroed_colors(pipeline, &routing).is_empty() {
                return Err(Error::Unsupported(metal2vulkan::error::Unsupported::needs(
                    metal2vulkan::error::Feature::IndependentBlending,
                    metal2vulkan::error::Capability::Feature(VkFeature::IndependentBlend),
                )));
            }
        }
        if let Some(&index) = routing.dropped.iter().find(|&&index| reads(index) || mesh) {
            return Err(refused(index));
        }
        let wanted = descriptor.color_formats();
        for (&index, &other) in &routing.placed {
            let wanted = wanted
                .get(index as usize)
                .copied()
                .flatten()
                .ok_or_else(|| internal("a placed color the state lacks"))?;
            let pass = self
                .layout
                .colors
                .get(other as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| internal("a placed color the pass lacks"))?;
            if pass.target.texture.pixel_format() != wanted {
                // A color both have in different formats: drawn where the pass's attachment
                // stores the state's encoding as plain Metal stores it in its own format (a
                // channel-order twin, `color_reformat.rs`); refused otherwise.
                // Or through a view of the attachment's image in the state's encoding, where the
                // image takes it, else a stand-in of its own (`viewing`).
                match self.color_view_format(pipeline, index, wanted, pass) {
                    Some(view) if view == pass.target.format => {}
                    Some(view) if self.viewing(pipeline, index, pass, view)?.is_some() => {}
                    Some(_) | None => return Err(refused(index)),
                }
                continue;
            }
            // An attachment the pipeline reads and writes as a storage image in the interlock is
            // no color attachment of its subpass, in the same format (`framebuffer_fetch.rs`).
            let format = graphics
                .color_attachments
                .get(index as usize)
                .and_then(Option::as_ref)
                .map(|a| a.format)
                .or_else(|| {
                    graphics
                        .storage_attachments
                        .iter()
                        .find(|s| s.color_index == index)
                        .map(|s| s.format)
                });
            if Some(pass.target.format) != format {
                return Err(Error::NotImplemented(RuntimeGap::Render));
            }
        }
        // The state's depth and stencil formats decide nothing: plain Metal tests and writes the
        // pass's depth and stencil attachments in their own formats whatever the state names
        // (one the state lacks, or another format), and a draw whose state names one the pass
        // lacks against the pass's hidden ones, as the pass's render pass instance does here (Metal
        // probes). A state of more samples in a pass of one is its single-sample twin here where
        // Metal draws it as one (`drawn_as`). A state that rasterizes nothing has no samples to
        // draw at, and Metal binds it in a pass of any sample count (`unrasterized.rs`).
        if descriptor.raster_sample_count() != self.layout.samples && !graphics.rasterizer_discard {
            return Err(draw_problem(call, DrawProblem::SampleCount));
        }
        // Its formats are the pass's, so a state without an explicit imageblock packs as the
        // pass (measured on an M3 Max and an M4).
        if graphics.fragment_imageblock.is_some() {
            let pipeline = state.imageblock_sample_length()?;
            let pass = self.pass_sample_length()?;
            if pipeline > pass {
                return Err(draw_problem(
                    call,
                    DrawProblem::ImageblockSampleLength { pipeline, pass },
                ));
            }
        }
        Ok(())
    }

    /// How a draw of `pipeline` renders its color `index` into the pass's attachment `color` in
    /// `format`, another format of its texel size (`Reformat`): through a view of the attachment's
    /// image where the image takes it (`PixelFormatView` usage, or a format list holding it,
    /// `image_plan.rs`), else through a stand-in image of `format` where the device copies both
    /// images; `None`, refused, for an attachment of more than one sample, that resolves, is
    /// memoryless or renders into a stand-in or sample planes already, for a 3D or 1D one, and
    /// for a state that fetches a color. `Unsupported` naming `format` where the device lacks it
    /// as a color attachment, or as a blended one where the state's color blends.
    pub(super) fn viewing(
        &self,
        pipeline: &DrawingState,
        index: u32,
        color: &render_pass::ColorTarget,
        format: metal2vulkan::vk::Format,
    ) -> Result<Option<Viewing>, Error> {
        let graphics = &pipeline.base().graphics;
        let fetches =
            !graphics.input_attachments.is_empty() || !graphics.storage_attachments.is_empty();
        let own = color.target.samples == 1
            && color.resolve.is_none()
            && color.copied_resolve.is_none()
            && !color.memoryless
            && color.stand_in.is_none()
            && color.planes.is_none()
            && !render_pass::not_2d(&color.target);
        if fetches || !own {
            return Ok(None);
        }
        let device = self.device;
        let profile = device.compile_device().profile();
        let blends = graphics
            .color_attachments
            .get(index as usize)
            .copied()
            .flatten()
            .is_some_and(|a| a.blend.is_some());
        let usages = [
            Some(FormatUsage::ColorAttachment),
            blends.then_some(FormatUsage::ColorAttachmentBlend),
        ];
        for usage in usages.into_iter().flatten() {
            if !profile.supports_format(format, usage) {
                return Err(Error::Unsupported(metal2vulkan::error::Unsupported::needs(
                    metal2vulkan::error::Feature::Format,
                    metal2vulkan::error::Capability::Format { format, usage },
                )));
            }
        }
        let plan = color.target.texture.image_plan()?;
        if super::binding_view::views_as(plan, format, false) {
            return Ok(Some(Viewing::View));
        }
        let copies = metal2vulkan::vk::ImageUsageFlags::TRANSFER_SRC
            | metal2vulkan::vk::ImageUsageFlags::TRANSFER_DST;
        let transfer = metal2vulkan::vk::FormatFeatureFlags2::TRANSFER_SRC
            | metal2vulkan::vk::FormatFeatureFlags2::TRANSFER_DST;
        let facts = device.shared().textures.facts(profile);
        let copied = plan.usage.contains(copies) && facts.optimal(format).contains(transfer);
        Ok(copied.then_some(Viewing::StandIn))
    }

    /// The format of the view through which a draw of `pipeline` stores its color `index`, of
    /// pixel format `wanted`, into the pass's attachment `color` at its place, of another pixel
    /// format, as plain Metal stores it (`color_reformat::view_format`). `None`, refused, for a
    /// mesh render pipeline's state, a state without a fragment function, that rasterizes
    /// nothing, has a fragment imageblock or reads the color (what Metal stores there is
    /// unmeasured), and a format held in a format other than its own (an extended-range
    /// format's).
    pub(super) fn color_view_format(
        &self,
        pipeline: &DrawingState,
        index: u32,
        wanted: PixelFormat,
        color: &render_pass::ColorTarget,
    ) -> Option<metal2vulkan::vk::Format> {
        let descriptor = pipeline.descriptor();
        let graphics = &pipeline.base().graphics;
        let fragment = descriptor.vertex().and_then(|d| d.fragment.as_ref());
        let reads = graphics
            .input_attachments
            .iter()
            .any(|a| a.color_index == index)
            || graphics
                .storage_attachments
                .iter()
                .any(|a| a.color_index == index);
        if fragment.is_none()
            || graphics.rasterizer_discard
            || graphics.fragment_imageblock.is_some()
            || reads
        {
            return None;
        }
        let profile = self.device.compile_device().profile();
        let state = graphics
            .color_attachments
            .get(index as usize)
            .copied()
            .flatten()?;
        let own = color_reformat::held(wanted, profile) == Some(state.format)
            && color_reformat::held(color.target.texture.pixel_format(), profile)
                == Some(color.target.format);
        if !own {
            return None;
        }
        color_reformat::view_format(state.format, color.target.format)
    }

    /// The imageblock bytes per sample the pass gives a pipeline: the larger of its
    /// descriptor's `imageblock_sample_length` and its attachments' texels
    /// ([`imageblock_sample_length`](metal2vulkan::metal::format::imageblock_sample_length)).
    pub(super) fn pass_sample_length(&self) -> Result<u64, Error> {
        let formats = self.layout.pass_formats();
        let packed = metal2vulkan::metal::format::imageblock_sample_length(&formats)
            .ok_or_else(|| internal("a pass attachment without a texel size"))?;
        Ok(u64::from(packed).max(self.layout.tile.sample_length))
    }

    /// Refuse the tile pipeline's state `state` for this pass in `call`: a color format other
    /// than the pass's up to the last color its function reaches through its implicit imageblock
    /// (every color where it declares an explicit one), the raster sample count other than the
    /// pass's, and, with an explicit imageblock aliased onto the implicit one, an imageblock per
    /// sample larger than the pass gives ([`pass_sample_length`](Self::pass_sample_length);
    /// measured: "the pipelineState's per sample imageBlock usage(32) is greater than the
    /// encoder's perSample imageBlock usage(4)"). An explicit imageblock standing alone runs in
    /// a pass of fewer bytes ([`kept_imageblock_bytes`](Self::kept_imageblock_bytes)).
    pub(super) fn check_tile(&self, call: &'static str, state: &TileState) -> Result<(), Error> {
        let descriptor = state.descriptor();
        // Plain Metal runs a tile function whatever the state's color formats (a Metal probe: one
        // of another format, none, one more or one fewer than the pass's ran every tile, the
        // attachments it does not reach untouched, and one more beside the color it reaches kept
        // that color's texels as the pass's). The function reaches a color at its place in the
        // state's packing, so the formats are checked up to the last color it reaches (every color
        // for an explicit imageblock, which lies after them): a place other than the pass's is
        // unmeasured.
        let kernel = state.kernel();
        let explicit =
            kernel.aliased_imageblock_bytes().is_some() || kernel.tile_imageblock().is_some();
        let checked = if explicit {
            Some(MAX_COLOR_ATTACHMENTS)
        } else {
            kernel
                .tile_attachments_reached()
                .max()
                .map(|last| last as usize + 1)
        };
        for index in 0..checked.unwrap_or(0).min(MAX_COLOR_ATTACHMENTS) {
            let pass = self.layout.colors.get(index).and_then(Option::as_ref);
            let pass = pass.map(|c| c.target.texture.pixel_format());
            let wanted = descriptor
                .color_attachment_formats
                .get(index)
                .copied()
                .flatten()
                .filter(|&format| format != PixelFormat::Invalid);
            if pass != wanted {
                return Err(draw_problem(
                    call,
                    DrawProblem::AttachmentFormat(Attachment::Color(index as u32)),
                ));
            }
        }
        if descriptor.raster_sample_count != self.layout.samples {
            return Err(draw_problem(call, DrawProblem::SampleCount));
        }
        // Without an explicit imageblock the colors the function reaches are the pass's, at the
        // pass's places, whatever bytes the state's other colors would take.
        if !explicit {
            return Ok(());
        }
        let pass = self.pass_sample_length()?;
        // Its formats are the pass's, so a state without an aliased layout packs as the pass.
        let pipeline = state
            .imageblock_sample_length()
            .ok_or_else(|| internal("a tile state's attachment without a texel size"))?;
        // An explicit imageblock standing alone runs in a pass of fewer bytes, as plain Metal
        // runs it (`kept_imageblock_bytes`); an aliased one past them is unmeasured.
        if pipeline > pass && kernel.tile_imageblock().is_none() {
            return Err(draw_problem(
                call,
                DrawProblem::ImageblockSampleLength { pipeline, pass },
            ));
        }
        Ok(())
    }
}

impl RenderLowering {
    /// The lowering of [`RenderCommandEncoder::set_render_pipeline_state`].
    pub(crate) fn set_render_pipeline_state(
        &mut self,
        state: &RenderPipelineState,
    ) -> Result<(), Error> {
        self.log(|log| RenderCall::PipelineState(log.hold_render_state(state)));
        let turn = self.group_turn();
        if let color_groups::Turn::Later(false) = turn {
            return Err(color_groups::gap());
        }
        let set = self.set_state(state);
        self.group_outcome(turn, set.is_ok());
        set
    }

    /// [`set_render_pipeline_state`](Self::set_render_pipeline_state)'s setting, its turn among
    /// the pass's color groups taken.
    fn set_state(&mut self, state: &RenderPipelineState) -> Result<(), Error> {
        const CALL: &str = "set_render_pipeline_state";
        self.core.check_device(CALL, state.device())?;
        let drawn = drawn_as(self.layout.samples, state)?;
        let state = self.grouped_as(drawn.clone())?;
        // Routed once here: the check and each draw read it (`routing`).
        if let RenderStateKind::Graphics(graphics) = state.kind() {
            let pass = self.layout.pass_formats();
            let routing = graphics.routed(&pass, || ShapeInputs::route(graphics, &pass));
            let colors = routing
                .as_ref()
                .ok()
                .map(|routing| Arc::new(self.shape_inputs().colors_by(graphics, routing)));
            self.routed = Some(Routed {
                routing,
                colors,
                state: graphics.clone(),
                pass,
            });
        }
        self.shape_inputs().check_pipeline(CALL, &state)?;
        self.writers.state_set(state.indirect_command_writes());
        self.pipeline = Some(state);
        self.grouped_set(&drawn);
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_buffer`].
    pub(crate) fn set_vertex_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
            RenderCall::Buffer(Table::Vertex, buffer, offset, index)
        });
        self.vertex
            .set_buffer(&mut self.core, "set_vertex_buffer", buffer, offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_buffers`].
    pub(crate) fn set_vertex_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            RenderCall::Buffers(
                Table::Vertex,
                log.buffer_span(buffers),
                log.word_span(offsets),
                first,
            )
        });
        self.vertex.set_buffers(
            &mut self.core,
            "set_vertex_buffers",
            buffers,
            offsets,
            first,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_buffer_offset`].
    pub(crate) fn set_vertex_buffer_offset(
        &mut self,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::BufferOffset(Table::Vertex, offset, index));
        self.vertex
            .set_buffer_offset(&mut self.core, "set_vertex_buffer_offset", offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_bytes`].
    pub(crate) fn set_vertex_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        self.log(|log| RenderCall::Bytes(Table::Vertex, log.byte_span(bytes), index));
        self.vertex
            .set_bytes(&mut self.core, "set_vertex_bytes", bytes, None, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_texture`].
    pub(crate) fn set_vertex_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let texture = texture.map(|texture| log.hold_texture(texture));
            RenderCall::Texture(Table::Vertex, texture, index)
        });
        self.vertex
            .set_textures(&mut self.core, "set_vertex_texture", &[texture], index)
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_textures`].
    pub(crate) fn set_vertex_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| RenderCall::Textures(Table::Vertex, log.texture_span(textures), first));
        self.vertex
            .set_textures(&mut self.core, "set_vertex_textures", textures, first)
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_sampler_state`].
    pub(crate) fn set_vertex_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
            RenderCall::Sampler(Table::Vertex, sampler, index)
        });
        self.vertex.set_samplers(
            &mut self.core,
            "set_vertex_sampler_state",
            &[sampler],
            None,
            index,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_sampler_states`].
    pub(crate) fn set_vertex_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::Samplers(Table::Vertex, Samplers::of(samplers, clamps, first)));
        self.vertex.set_samplers(
            &mut self.core,
            "set_vertex_sampler_states",
            samplers,
            clamps,
            first,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_buffer`].
    pub(crate) fn set_fragment_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
            RenderCall::Buffer(Table::Fragment, buffer, offset, index)
        });
        self.fragment
            .set_buffer(&mut self.core, "set_fragment_buffer", buffer, offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_buffers`].
    pub(crate) fn set_fragment_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            RenderCall::Buffers(
                Table::Fragment,
                log.buffer_span(buffers),
                log.word_span(offsets),
                first,
            )
        });
        self.fragment.set_buffers(
            &mut self.core,
            "set_fragment_buffers",
            buffers,
            offsets,
            first,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_buffer_offset`].
    pub(crate) fn set_fragment_buffer_offset(
        &mut self,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::BufferOffset(Table::Fragment, offset, index));
        self.fragment
            .set_buffer_offset(&mut self.core, "set_fragment_buffer_offset", offset, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_bytes`].
    pub(crate) fn set_fragment_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        self.log(|log| RenderCall::Bytes(Table::Fragment, log.byte_span(bytes), index));
        self.fragment
            .set_bytes(&mut self.core, "set_fragment_bytes", bytes, None, index)
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_texture`].
    pub(crate) fn set_fragment_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let texture = texture.map(|texture| log.hold_texture(texture));
            RenderCall::Texture(Table::Fragment, texture, index)
        });
        self.fragment
            .set_textures(&mut self.core, "set_fragment_texture", &[texture], index)
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_textures`].
    pub(crate) fn set_fragment_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|log| RenderCall::Textures(Table::Fragment, log.texture_span(textures), first));
        self.fragment
            .set_textures(&mut self.core, "set_fragment_textures", textures, first)
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_sampler_state`].
    pub(crate) fn set_fragment_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|log| {
            let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
            RenderCall::Sampler(Table::Fragment, sampler, index)
        });
        self.fragment.set_samplers(
            &mut self.core,
            "set_fragment_sampler_state",
            &[sampler],
            None,
            index,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_sampler_states`].
    pub(crate) fn set_fragment_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::Samplers(Table::Fragment, Samplers::of(samplers, clamps, first)));
        self.fragment.set_samplers(
            &mut self.core,
            "set_fragment_sampler_states",
            samplers,
            clamps,
            first,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_color_store_action`].
    pub(crate) fn set_color_store_action(
        &mut self,
        action: StoreAction,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::ColorStoreAction(action, index));
        const CALL: &str = "set_color_store_action";
        check_index(CALL, index, MAX_COLOR_ATTACHMENTS as u32)?;
        let which = Attachment::Color(index);
        // Its color group's, where the pass draws in groups (`render/color_groups.rs`).
        let Some(Some(color)) = self.color_slot(index) else {
            return Ok(());
        };
        if !color.late_store && action != StoreAction::Unknown {
            return Ok(());
        }
        let named = color.resolve.is_some() || color.copied_resolve.is_some();
        let action = match (named, action) {
            (false, StoreAction::MultisampleResolve) => StoreAction::DontCare,
            (false, StoreAction::StoreAndMultisampleResolve) => StoreAction::Store,
            (_, action) => action,
        };
        // A late `CustomSampleDepthStore` of a resolving attachment stays refused (`late_store`);
        // any other action that does not resolve drops the resolve.
        color.store = late_store(
            CALL,
            which,
            color.late_store,
            named,
            color.memoryless,
            action,
        )?;
        if !color.store.resolves() {
            color.resolve = None;
            color.copied_resolve = None;
        }
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_depth_store_action`].
    pub(crate) fn set_depth_store_action(&mut self, action: StoreAction) -> Result<(), Error> {
        self.log(|_| RenderCall::DepthStoreAction(action));
        self.set_aspect_store_action("set_depth_store_action", Attachment::Depth, action)
    }

    /// The lowering of [`RenderCommandEncoder::set_stencil_store_action`].
    pub(crate) fn set_stencil_store_action(&mut self, action: StoreAction) -> Result<(), Error> {
        self.log(|_| RenderCall::StencilStoreAction(action));
        self.set_aspect_store_action("set_stencil_store_action", Attachment::Stencil, action)
    }

    /// The lowering of [`RenderCommandEncoder::set_color_store_action_options`].
    pub(crate) fn set_color_store_action_options(
        &mut self,
        _options: StoreActionOptions,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::ColorStoreActionOptions(_options, index));
        const CALL: &str = "set_color_store_action_options";
        check_index(CALL, index, MAX_COLOR_ATTACHMENTS as u32)?;
        let which = Attachment::Color(index);
        // The options change nothing, as plain Metal's (`sample_positions.rs`). A later color
        // group's encoder checks only its own colors (`render/color_groups.rs`).
        let later = self.later_group_encoder();
        match self.color_slot(index) {
            Some(Some(_)) => Ok(()),
            None if later => Ok(()),
            _ => Err(store_problem(CALL, RenderPassProblem::NoAttachment(which))),
        }
    }

    /// The lowering of [`RenderCommandEncoder::set_depth_store_action_options`].
    pub(crate) fn set_depth_store_action_options(
        &mut self,
        options: StoreActionOptions,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::DepthStoreActionOptions(options));
        self.set_aspect_store_action_options(
            "set_depth_store_action_options",
            Attachment::Depth,
            options,
        )
    }

    /// The lowering of [`RenderCommandEncoder::set_stencil_store_action_options`].
    pub(crate) fn set_stencil_store_action_options(
        &mut self,
        options: StoreActionOptions,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::StencilStoreActionOptions(options));
        self.set_aspect_store_action_options(
            "set_stencil_store_action_options",
            Attachment::Stencil,
            options,
        )
    }

    /// The late store action options of the depth or stencil attachment (`which`).
    fn set_aspect_store_action_options(
        &mut self,
        call: &'static str,
        which: Attachment,
        options: StoreActionOptions,
    ) -> Result<(), Error> {
        // A later color group's twin of the depth and stencil stores nothing
        // (`render/color_groups.rs`).
        if self.later_group_encoder() {
            return Ok(());
        }
        let aspect = self
            .layout
            .depth_stencil
            .as_mut()
            .filter(|ds| !ds.hidden)
            .and_then(|ds| match which {
                Attachment::Stencil => ds.stencil.as_mut(),
                _ => ds.depth.as_mut(),
            });
        let Some(aspect) = aspect else {
            return Err(store_problem(call, RenderPassProblem::NoAttachment(which)));
        };
        if !aspect.late_store {
            return Err(store_problem(
                call,
                RenderPassProblem::StoreActionKnown(which),
            ));
        }
        aspect.options = options;
        Ok(())
    }

    /// The late store action of the depth or stencil attachment (`which`).
    fn set_aspect_store_action(
        &mut self,
        call: &'static str,
        which: Attachment,
        action: StoreAction,
    ) -> Result<(), Error> {
        // A later color group's twin of the depth and stencil stores nothing
        // (`render/color_groups.rs`).
        if self.later_group_encoder() {
            return Ok(());
        }
        let none = ResolveMode::None;
        let maker = self.core.command_buffer().clone();
        // A depth store action without a depth attachment changes nothing, as plain Metal's (Metal
        // probe case `late_store_depth_no_texture` on an M4); a stencil one, unmeasured, is
        // refused.
        let absent = || match which {
            Attachment::Depth => Ok(()),
            _ => Err(store_problem(call, RenderPassProblem::NoAttachment(which))),
        };
        // The pass's hidden depth and stencil are no attachment of Metal's (`next_instance`).
        let Some(ds) = self.layout.depth_stencil.as_mut().filter(|ds| !ds.hidden) else {
            return absent();
        };
        let modes = ds
            .resolve
            .as_ref()
            .map_or((none, none), |r| (r.depth_mode, r.stencil_mode));
        let (aspect, resolves, other) = match which {
            Attachment::Stencil => (ds.stencil, modes.1 != none, ds.depth.map(|d| d.store)),
            _ => (ds.depth, modes.0 != none, ds.stencil.map(|s| s.store)),
        };
        let Some(aspect) = aspect else {
            return absent();
        };
        let store = late_store(
            call,
            which,
            aspect.late_store,
            resolves,
            aspect.memoryless,
            action,
        )?;
        // One texture's depth and stencil mixing storing and resolving as Metal leaves
        // undetermined (`resolve::mixed`, measured). An aspect without an attachment mixes none.
        let mixed = other.is_some_and(|other| match which {
            Attachment::Stencil => super::resolve::mixed(other, store),
            _ => super::resolve::mixed(store, other),
        });
        if mixed && ds.separate.is_none() {
            return Err(store_problem(
                call,
                RenderPassProblem::DepthStencilResolveMix,
            ));
        }
        // One that does not resolve where the aspect resolves drops that aspect's resolve
        // (`late_store`): the resolve is made again from both aspects' targets, as a late target
        // makes it (`render/late_resolve.rs`).
        let resolve = if resolves && !store.resolves() {
            let stores = match which {
                Attachment::Stencil => (other, Some(store)),
                _ => (Some(store), other),
            };
            Some(late_resolve::remade(call, &maker, ds, stores)?)
        } else {
            None
        };
        let slot = match which {
            Attachment::Stencil => &mut ds.stencil,
            _ => &mut ds.depth,
        };
        if let Some(aspect) = slot {
            aspect.store = store;
        }
        if let Some(resolve) = resolve {
            ds.resolve = resolve;
        }
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_viewport`].
    pub(crate) fn set_viewport(&mut self, viewport: Viewport) -> Result<(), Error> {
        self.log(|_| RenderCall::Viewport(Box::new(viewport)));
        let refused = render_state::depth_ranges_refused(self.core.device());
        self.state
            .set_viewports("set_viewport", &[viewport], refused)
    }

    /// The lowering of [`RenderCommandEncoder::set_viewports`].
    pub(crate) fn set_viewports(&mut self, viewports: &[Viewport]) -> Result<(), Error> {
        self.log(|_| RenderCall::Viewports(viewports.into()));
        let refused = render_state::depth_ranges_refused(self.core.device());
        self.state
            .set_viewports("set_viewports", viewports, refused)
    }

    /// The lowering of [`RenderCommandEncoder::set_scissor_rect`].
    pub(crate) fn set_scissor_rect(&mut self, rect: ScissorRect) -> Result<(), Error> {
        self.log(|_| RenderCall::ScissorRect(rect));
        self.state.set_scissors("set_scissor_rect", &[rect])
    }

    /// The lowering of [`RenderCommandEncoder::set_scissor_rects`].
    pub(crate) fn set_scissor_rects(&mut self, rects: &[ScissorRect]) -> Result<(), Error> {
        self.log(|_| RenderCall::ScissorRects(rects.into()));
        self.state.set_scissors("set_scissor_rects", rects)
    }

    /// The lowering of [`RenderCommandEncoder::set_cull_mode`].
    pub(crate) fn set_cull_mode(&mut self, mode: CullMode) {
        self.log(|_| RenderCall::CullMode(mode));
        self.state.set_cull_mode(mode);
    }

    /// The lowering of [`RenderCommandEncoder::set_front_facing_winding`].
    pub(crate) fn set_front_facing_winding(&mut self, winding: Winding) {
        self.log(|_| RenderCall::FrontFacingWinding(winding));
        self.state.set_winding(winding);
    }

    /// The lowering of [`RenderCommandEncoder::set_depth_stencil_state`].
    pub(crate) fn set_depth_stencil_state<'a>(
        &mut self,
        state: impl Into<Option<&'a DepthStencilState>>,
    ) -> Result<(), Error> {
        let state = state.into();
        self.log(|log| {
            RenderCall::DepthStencilState(state.map(|state| log.hold_depth_stencil(state)))
        });
        if let Some(state) = state {
            self.core
                .check_device("set_depth_stencil_state", state.device())?;
        }
        // One boolean: a pass that sets one but draws under none keeps a depth's signs.
        self.depth_writes |= state.is_some_and(|s| s.descriptor().depth_write_enabled);
        self.state.set_depth_stencil(state);
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_stencil_reference_value`].
    pub(crate) fn set_stencil_reference_value(&mut self, value: u32) {
        self.log(|_| RenderCall::StencilReferenceValue(value));
        self.state.set_stencil_references(value, value);
    }

    /// The lowering of [`RenderCommandEncoder::set_stencil_reference_values`].
    pub(crate) fn set_stencil_reference_values(&mut self, front: u32, back: u32) {
        self.log(|_| RenderCall::StencilReferenceValues(front, back));
        self.state.set_stencil_references(front, back);
    }

    /// The lowering of [`RenderCommandEncoder::set_depth_bias`].
    pub(crate) fn set_depth_bias(
        &mut self,
        depth_bias: f32,
        slope_scale: f32,
        clamp: f32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::DepthBias(depth_bias, slope_scale, clamp));
        self.state.set_depth_bias(depth_bias, slope_scale, clamp)
    }

    /// The lowering of [`RenderCommandEncoder::set_depth_clip_mode`].
    pub(crate) fn set_depth_clip_mode(&mut self, mode: DepthClipMode) -> Result<(), Error> {
        self.log(|_| RenderCall::DepthClipMode(mode));
        let profile = self.core.device().compile_device().profile();
        self.state.set_clip_mode(mode, profile)
    }

    /// The lowering of [`RenderCommandEncoder::set_triangle_fill_mode`].
    pub(crate) fn set_triangle_fill_mode(&mut self, mode: TriangleFillMode) -> Result<(), Error> {
        self.log(|_| RenderCall::TriangleFillMode(mode));
        let profile = self.core.device().compile_device().profile();
        self.state.set_fill_mode(mode, profile)
    }

    /// The lowering of [`RenderCommandEncoder::set_blend_color`].
    pub(crate) fn set_blend_color(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        self.log(|_| RenderCall::BlendColor([red, green, blue, alpha]));
        self.state.set_blend_color([red, green, blue, alpha]);
    }

    /// The lowering of [`RenderCommandEncoder::set_tessellation_factor_buffer`].
    pub(crate) fn set_tessellation_factor_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        instance_stride: u64,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::TessellationFactorBuffer(buffer.cloned(), offset, instance_stride)
        });
        const CALL: &str = "set_tessellation_factor_buffer";
        if let Some(buffer) = buffer {
            self.core.check_device(CALL, buffer.device())?;
        }
        self.factors.buffer = buffer.map(|buffer| FactorBuffer {
            buffer: buffer.clone(),
            offset,
            instance_stride,
        });
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::set_tessellation_factor_scale`].
    pub(crate) fn set_tessellation_factor_scale(&mut self, scale: f32) {
        self.log(|_| RenderCall::TessellationFactorScale(scale));
        self.factors.scale = Some(scale);
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_amplification_count`].
    pub(crate) fn set_vertex_amplification_count(
        &mut self,
        count: u64,
        view_mappings: Option<&[VertexAmplificationViewMapping]>,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::VertexAmplificationCount(count, view_mappings.map(Into::into)));
        *self.amplification = Amplification::new(count, view_mappings)
            .map_err(|problem| draw_problem("set_vertex_amplification_count", problem))?;
        Ok(())
    }

    /// Set the alpha test reference (`setAlphaTestReferenceValue:`, `apple_only.rs`), which a
    /// pipeline with the private alpha test compares against (`BindState::alpha_test_reference`).
    pub(crate) fn set_alpha_test_reference(&mut self, value: f32) {
        self.state.set_alpha_test_reference(value);
    }

    /// Set the amplification count `count` with the view mappings set before kept
    /// (`setVertexAmplificationMode:value:`, `apple_only.rs`); `None`, the amplification kept,
    /// where the count passes them ([`Amplification::with_count`]).
    pub(crate) fn amplify_with_kept_mappings(&mut self, count: u64) -> Option<()> {
        let amplified = self.amplification.with_count(count)?;
        *self.amplification = amplified;
        Some(())
    }

    /// The lowering of [`RenderCommandEncoder::draw_primitives`].
    pub(crate) fn draw_primitives(
        &mut self,
        primitive_type: DrawPrimitiveType,
        vertex_start: u32,
        vertex_count: u32,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::DrawPrimitives(
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            )
        });
        const CALL: &str = "draw_primitives";
        let draw = DrawCall::Primitives {
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        };
        let buffers = DrawBuffers::default();
        let empty = vertex_count == 0 || instance_count == 0;
        let Some(assembly) = Assembly::of(primitive_type) else {
            // Metal's GPU hangs at 3 vertices or more, and draws 1 or 2 as nothing.
            let hangs = vertex_count >= 3 && !empty;
            return self.draw_nine(CALL, empty, hangs.then_some(NineFailure::Hang));
        };
        self.draw(CALL, assembly, draw, buffers, empty)
    }

    /// The lowering of [`RenderCommandEncoder::draw_indexed_primitives`].
    pub(crate) fn draw_indexed_primitives(
        &mut self,
        primitive_type: DrawPrimitiveType,
        index_count: u32,
        index_type: IndexType,
        index_buffer: &Buffer,
        index_buffer_offset: u64,
        instance_count: u32,
        base_vertex: i32,
        base_instance: u32,
    ) -> Result<(), Error> {
        self.log_indexed(|log| DrawIndexed {
            primitive_type,
            index_count,
            index_type,
            index_buffer: log.hold_buffer(index_buffer),
            index_buffer_offset,
            instance_count,
            base_vertex,
            base_instance,
        });
        const CALL: &str = "draw_indexed_primitives";
        self.core.check_device(CALL, index_buffer.device())?;
        // Metal reads from the offset's low 32 bits.
        let index_buffer_offset = u64::from(index_buffer_offset as u32);
        let index_buffer_offset =
            index_buffer_offset - index_buffer_offset % index_size(index_type);
        let draw = IndexedDraw {
            primitive_type,
            index_count,
            index_type,
            instance_count,
            base_vertex,
            base_instance,
        };
        self.draw_indexed(CALL, draw, index_buffer, index_buffer_offset)
    }

    /// Draw `draw`'s indices from `index_buffer_offset` of `index_buffer`, a multiple of the index
    /// size, those past the buffer's length as the module docs say, nothing for no indices or
    /// instances: the steps of `draw_indexed_primitives` after its refusals, which an execution of
    /// indirect render commands takes too (`render/executions.rs`).
    fn draw_indexed(
        &mut self,
        call: &'static str,
        draw: IndexedDraw,
        index_buffer: &Buffer,
        index_buffer_offset: u64,
    ) -> Result<(), Error> {
        let IndexedDraw {
            primitive_type,
            index_count,
            index_type,
            instance_count,
            base_vertex,
            base_instance,
        } = draw;
        let size = index_size(index_type);
        let (offset, inside) = indices_inside(index_buffer, index_buffer_offset, size);
        let count = u64::from(index_count);
        // Vulkan states no index past the bound range: positions past where Metal's fetch stops
        // read a zero-padded copy.
        let index = if count > inside {
            DrawSource::padded(index_buffer, offset, inside * size, count * size, size)?
        } else {
            DrawSource::new(index_buffer, offset, count * size, size, false, false)?
        };
        let draw = DrawCall::Indexed {
            index_type: metal2vulkan::metal::state::index_type(index_type),
            index_buffer: IndexBufferRef::new(0, index.length),
            index_count,
            index_offset: index.at,
            instance_count,
            base_vertex,
            base_instance,
        };
        let buffers = DrawBuffers {
            index: Some(index),
            ..DrawBuffers::default()
        };
        let empty = index_count == 0 || instance_count == 0;
        let Some(assembly) = Assembly::of(primitive_type) else {
            // Metal's GPU faults at it, of any index count, of no index or instance too.
            let fault = NineFailure::Fault(CommandBufferError::PageFault);
            return self.draw_nine(call, empty, Some(fault));
        };
        self.draw(call, assembly, draw, buffers, empty)
    }

    /// The lowering of [`RenderCommandEncoder::draw_primitives_indirect`].
    pub(crate) fn draw_primitives_indirect(
        &mut self,
        primitive_type: DrawPrimitiveType,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::DrawPrimitivesIndirect(
                primitive_type,
                indirect_buffer.clone(),
                indirect_buffer_offset,
            )
        });
        const CALL: &str = "draw_primitives_indirect";
        self.core.check_device(CALL, indirect_buffer.device())?;
        let indirect_buffer_offset =
            check_arguments(CALL, indirect_buffer.length(), indirect_buffer_offset, 16)?;
        let indirect =
            DrawSource::new(indirect_buffer, indirect_buffer_offset, 16, 4, false, false)?;
        let draw = DrawCall::Indirect {
            indirect: IndirectRef::new(0, indirect.at),
            indexed: None,
        };
        let buffers = DrawBuffers {
            indirect: Some(indirect),
            ..DrawBuffers::default()
        };
        let Some(assembly) = Assembly::of(primitive_type) else {
            // Metal fails it as a victim of the GPU's recovery, whatever its arguments.
            let victim = NineFailure::Fault(CommandBufferError::GpuRecoveryVictim);
            return self.draw_nine(CALL, false, Some(victim));
        };
        self.draw(CALL, assembly, draw, buffers, false)
    }

    /// The lowering of [`RenderCommandEncoder::draw_indexed_primitives_indirect`].
    pub(crate) fn draw_indexed_primitives_indirect(
        &mut self,
        primitive_type: DrawPrimitiveType,
        index_type: IndexType,
        index_buffer: &Buffer,
        index_buffer_offset: u64,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::DrawIndexedPrimitivesIndirect(Box::new(DrawIndexedIndirect {
                primitive_type,
                index_type,
                index_buffer: index_buffer.clone(),
                index_buffer_offset,
                indirect_buffer: indirect_buffer.clone(),
                indirect_buffer_offset,
            }))
        });
        const CALL: &str = "draw_indexed_primitives_indirect";
        self.core.check_device(CALL, index_buffer.device())?;
        self.core.check_device(CALL, indirect_buffer.device())?;
        let size = index_size(index_type);
        let index_buffer_offset = index_buffer_offset - index_buffer_offset % size;
        let indirect_buffer_offset =
            check_arguments(CALL, indirect_buffer.length(), indirect_buffer_offset, 20)?;
        // The arguments choose the indices on the GPU: any of the whole ones from the offset,
        // and positions past them.
        let (offset, inside) = indices_inside(index_buffer, index_buffer_offset, size);
        // Bounded by the device's range where the draw stays one draw, and by a copy where the
        // device binds no range of a size (`draw_source.rs`); the index bounds pass bounds it
        // elsewhere (`draw`).
        let isolate =
            self.robust_index_fetch() && self.core.device().shared().sized_index_bind.is_none();
        // Bytes a draw or tile dispatch of the encoder wrote, before any memory barrier, which
        // Metal does not order before this draw's read, are read through a copy taken before the
        // pass: the bytes from before the write, Metal's answer in every run measured (our probes,
        // a tile dispatch's too). After a barrier Metal's answer to a draw's write
        // varied: those are read in place (refused where the index bounds pass reads them,
        // `draw`). A tile dispatch's write after a barrier, which Metal's draw does not read
        // either, is refused: the copy before the pass misses what the barrier ordered.
        let index_range = offset..offset + inside * size;
        let arguments = indirect_buffer_offset..indirect_buffer_offset + 20;
        let read =
            |buffer: &Buffer, range: std::ops::Range<u64>| [super::blit::reads(buffer, range)];
        let tile_written = self
            .unordered_dispatches
            .written(&read(index_buffer, index_range.clone()))
            || self
                .unordered_dispatches
                .written(&read(indirect_buffer, arguments.clone()));
        if !self.replaying && self.barriered && tile_written {
            return Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments));
        }
        let racing = |buffer: &Buffer, range: std::ops::Range<u64>| {
            !self.replaying && !self.barriered && self.unordered.written(&read(buffer, range))
        };
        let index_racing = racing(index_buffer, index_range);
        let indirect_racing = racing(indirect_buffer, arguments);
        let index = DrawSource::new(
            index_buffer,
            offset,
            inside * size,
            size,
            isolate,
            index_racing,
        )?;
        let indirect = DrawSource::new(
            indirect_buffer,
            indirect_buffer_offset,
            20,
            4,
            false,
            indirect_racing,
        )?;
        let draw = DrawCall::Indirect {
            indirect: IndirectRef::new(0, indirect.at),
            indexed: Some((
                metal2vulkan::metal::state::index_type(index_type),
                IndexBufferRef::new(0, index.length),
                index.at,
            )),
        };
        let buffers = DrawBuffers {
            index: Some(index),
            indirect: Some(indirect),
            ..DrawBuffers::default()
        };
        let Some(assembly) = Assembly::of(primitive_type) else {
            // Metal fails it as a victim of the GPU's recovery, whatever its arguments.
            let victim = NineFailure::Fault(CommandBufferError::GpuRecoveryVictim);
            return self.draw_nine(CALL, false, Some(victim));
        };
        self.draw(CALL, assembly, draw, buffers, false)
    }

    /// The lowering of [`RenderCommandEncoder::draw_patches`].
    pub(crate) fn draw_patches(
        &mut self,
        number_of_patch_control_points: u32,
        patch_start: u32,
        patch_count: u32,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::DrawPatches(Box::new(DrawPatches {
                control_points: number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer: patch_index_buffer.cloned(),
                patch_index_buffer_offset,
                instance_count,
                base_instance,
            }))
        });
        const CALL: &str = "draw_patches";
        let patch_index_buffer_offset = self.check_patch_indices(
            CALL,
            patch_index_buffer,
            patch_index_buffer_offset,
            Some((patch_start, patch_count)),
        )?;
        let draw = PatchDraw {
            control_points: number_of_patch_control_points,
            patch_start,
            patch_count,
            instance_count,
            base_instance,
        };
        let patch_indices = patch_index_buffer.map(|b| (b, patch_index_buffer_offset));
        // A pipeline with a control-point index type reads every index as 0 (Metal probe case
        // `plain_with_type` on an M4; a probe for a command): a buffer of zeros, past whose end an
        // index reads 0 too.
        let zeros = match self.control_point_index_size() {
            Some(_) => Some(self.zero_indices()?),
            None => None,
        };
        let points = zeros.as_ref().map(|zeros| (zeros, 0));
        self.draw_patch_range(CALL, draw, patch_indices, points)
    }

    /// Draw `draw`'s patches, each patch's index from `patch_indices` (a buffer and an offset
    /// into it, a multiple of 4) where given, and their control points' indices from
    /// `control_point_indices` (a buffer and an offset into it, a multiple of the pipeline's index
    /// size; an index past the buffer reads 0) for a pipeline with a control-point index type,
    /// nothing for no patches or instances: the steps of `draw_patches` and
    /// `draw_indexed_patches` after their refusals, which an execution of indirect render
    /// commands takes too (`render/executions.rs`).
    fn draw_patch_range(
        &mut self,
        call: &'static str,
        draw: PatchDraw,
        patch_indices: Option<(&Buffer, u64)>,
        control_point_indices: Option<(&Buffer, u64)>,
    ) -> Result<(), Error> {
        let PatchDraw {
            control_points,
            patch_start,
            patch_count,
            instance_count,
            base_instance,
        } = draw;
        let draw = DrawCall::Patches {
            control_points,
            patch_start,
            patch_count,
            patch_index: patch_indices.map(|(b, offset)| index_ref(b, offset)),
            instance_count,
            base_instance,
            control_point_index: control_point_indices.map(|(b, offset)| index_ref(b, offset)),
        };
        let buffers = DrawBuffers {
            patch_indices,
            control_point_indices,
            ..DrawBuffers::default()
        };
        let empty = patch_count == 0 || instance_count == 0;
        self.draw(call, Assembly::Patches, draw, buffers, empty)
    }

    /// The lowering of [`RenderCommandEncoder::draw_patches_indirect`].
    pub(crate) fn draw_patches_indirect(
        &mut self,
        number_of_patch_control_points: u32,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::DrawPatchesIndirect(Box::new(DrawPatchesIndirect {
                control_points: number_of_patch_control_points,
                patch_index_buffer: patch_index_buffer.cloned(),
                patch_index_buffer_offset,
                indirect_buffer: indirect_buffer.clone(),
                indirect_buffer_offset,
            }))
        });
        const CALL: &str = "draw_patches_indirect";
        let patch_index_buffer_offset =
            self.check_patch_indices(CALL, patch_index_buffer, patch_index_buffer_offset, None)?;
        self.core.check_device(CALL, indirect_buffer.device())?;
        let indirect_buffer_offset =
            check_arguments(CALL, indirect_buffer.length(), indirect_buffer_offset, 16)?;
        let indirect =
            DrawSource::new(indirect_buffer, indirect_buffer_offset, 16, 4, false, false)?;
        let zeros = match self.control_point_index_size() {
            Some(_) => Some(self.zero_indices()?),
            None => None,
        };
        let draw = DrawCall::PatchesIndirect {
            control_points: number_of_patch_control_points,
            patch_index: patch_index_buffer.map(|b| index_ref(b, patch_index_buffer_offset)),
            indirect: IndirectRef::new(0, indirect.at),
            control_point_index: zeros.as_ref().map(|zeros| index_ref(zeros, 0)),
        };
        let buffers = DrawBuffers {
            indirect: Some(indirect),
            patch_indices: patch_index_buffer.map(|b| (b, patch_index_buffer_offset)),
            control_point_indices: zeros.as_ref().map(|zeros| (zeros, 0)),
            ..DrawBuffers::default()
        };
        self.draw(CALL, Assembly::Patches, draw, buffers, false)
    }

    /// The lowering of [`RenderCommandEncoder::draw_indexed_patches`].
    pub(crate) fn draw_indexed_patches(
        &mut self,
        number_of_patch_control_points: u32,
        patch_start: u32,
        patch_count: u32,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        control_point_index_buffer: &Buffer,
        control_point_index_buffer_offset: u64,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::DrawIndexedPatches(Box::new(DrawIndexedPatches {
                control_points: number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer: patch_index_buffer.cloned(),
                patch_index_buffer_offset,
                control_point_index_buffer: control_point_index_buffer.clone(),
                control_point_index_buffer_offset,
                instance_count,
                base_instance,
            }))
        });
        const CALL: &str = "draw_indexed_patches";
        let patch_index_buffer_offset = self.check_patch_indices(
            CALL,
            patch_index_buffer,
            patch_index_buffer_offset,
            Some((patch_start, patch_count)),
        )?;
        // The indices the draw reads: of its pipeline's control points where its function
        // states them, as Metal draws them.
        let indices = (u64::from(patch_start) + u64::from(patch_count))
            * u64::from(self.patch_control_points(number_of_patch_control_points));
        let control_point_index_buffer_offset = self.check_control_point_indices(
            CALL,
            control_point_index_buffer,
            control_point_index_buffer_offset,
            Some(indices),
        )?;
        let draw = PatchDraw {
            control_points: number_of_patch_control_points,
            patch_start,
            patch_count,
            instance_count,
            base_instance,
        };
        // A pipeline without a control-point index type draws the patches' own control points, the
        // buffer unread (Metal probe case `indexed_without_type` on an M4).
        let points = self.control_point_index_size().map(|_| {
            (
                control_point_index_buffer,
                control_point_index_buffer_offset,
            )
        });
        self.draw_patch_range(
            CALL,
            draw,
            patch_index_buffer.map(|b| (b, patch_index_buffer_offset)),
            points,
        )
    }

    /// The lowering of [`RenderCommandEncoder::draw_indexed_patches_indirect`].
    pub(crate) fn draw_indexed_patches_indirect(
        &mut self,
        number_of_patch_control_points: u32,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        control_point_index_buffer: &Buffer,
        control_point_index_buffer_offset: u64,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::DrawIndexedPatchesIndirect(Box::new(DrawIndexedPatchesIndirect {
                control_points: number_of_patch_control_points,
                patch_index_buffer: patch_index_buffer.cloned(),
                patch_index_buffer_offset,
                control_point_index_buffer: control_point_index_buffer.clone(),
                control_point_index_buffer_offset,
                indirect_buffer: indirect_buffer.clone(),
                indirect_buffer_offset,
            }))
        });
        const CALL: &str = "draw_indexed_patches_indirect";
        let patch_index_buffer_offset =
            self.check_patch_indices(CALL, patch_index_buffer, patch_index_buffer_offset, None)?;
        let control_point_index_buffer_offset = self.check_control_point_indices(
            CALL,
            control_point_index_buffer,
            control_point_index_buffer_offset,
            None,
        )?;
        self.core.check_device(CALL, indirect_buffer.device())?;
        let indirect_buffer_offset =
            check_arguments(CALL, indirect_buffer.length(), indirect_buffer_offset, 16)?;
        let indirect =
            DrawSource::new(indirect_buffer, indirect_buffer_offset, 16, 4, false, false)?;
        let points = self.control_point_index_size().map(|_| {
            (
                control_point_index_buffer,
                control_point_index_buffer_offset,
            )
        });
        let draw = DrawCall::PatchesIndirect {
            control_points: number_of_patch_control_points,
            patch_index: patch_index_buffer.map(|b| index_ref(b, patch_index_buffer_offset)),
            indirect: IndirectRef::new(0, indirect.at),
            control_point_index: points.map(|(b, offset)| index_ref(b, offset)),
        };
        let buffers = DrawBuffers {
            indirect: Some(indirect),
            patch_indices: patch_index_buffer.map(|b| (b, patch_index_buffer_offset)),
            control_point_indices: points,
            ..DrawBuffers::default()
        };
        self.draw(CALL, Assembly::Patches, draw, buffers, false)
    }

    /// A patch draw's patch index buffer (`None` for `nil`) at `offset`: the offset the draw reads
    /// its patch indices from, rounded down to a multiple of 4 as plain Metal reads it, where its
    /// validation stops the draw (Metal indirect command buffer probe case
    /// `patch_index_offset_misaligned` on an M4, a command's indices 9 and 3 at offset 2 drew
    /// patches 9 and 3; Metal probe case `pib_offset2` on an M4, the encoder's, drew what that read
    /// draws). Refused for another device and as Metal's validation refuses it (`Invalid`,
    /// measured): for a direct draw of `(patch_start, patch_count)`, indices `patch_start +
    /// patch_count` past the buffer from the offset.
    fn check_patch_indices(
        &self,
        call: &'static str,
        buffer: Option<&Buffer>,
        offset: u64,
        direct: Option<(u32, u32)>,
    ) -> Result<u64, Error> {
        let Some(buffer) = buffer else {
            return Ok(offset);
        };
        self.core.check_device(call, buffer.device())?;
        let offset = offset - offset % 4;
        let Some((patch_start, patch_count)) = direct else {
            return Ok(offset);
        };
        let indices = u64::from(patch_start) + u64::from(patch_count);
        let end = offset.checked_add(4 * indices);
        if end.is_none_or(|end| end > buffer.length()) {
            let problem = DrawProblem::PatchIndexBufferRange {
                offset,
                indices,
                length: buffer.length(),
            };
            return Err(draw_problem(call, problem));
        }
        Ok(offset)
    }

    /// The control points a patch draw naming `given` draws: its pipeline state's, where the
    /// state's post-tessellation vertex function states them (`[[patch(_, N)]]`), as Metal draws
    /// them whatever the draw names (Metal probe cases `control_points_4` and `_0` on an M4; for a
    /// command); `given` elsewhere.
    fn patch_control_points(&self, given: u32) -> u32 {
        match self.pipeline.as_ref().map(RenderPipelineState::kind) {
            Some(RenderStateKind::Graphics(state)) => {
                crate::tessellation::patches(&state.base().graphics)
                    .map(|p| p.control_points)
                    .filter(|&n| n != 0)
                    .unwrap_or(given)
            }
            _ => given,
        }
    }

    /// The size of the encoder's pipeline state's control-point indices: `None` for a state that
    /// is not a tessellation one or whose `tessellationControlPointIndexType` is `None`.
    fn control_point_index_size(&self) -> Option<u64> {
        use metal2vulkan::metal::TessellationControlPointIndexType as T;
        let ty = match self.pipeline.as_ref().map(RenderPipelineState::kind) {
            Some(RenderStateKind::Graphics(state)) => {
                crate::tessellation::patches(&state.base().graphics)
                    .map(|p| p.control_point_index_type)
            }
            _ => None,
        };
        match ty {
            Some(T::UInt16) => Some(2),
            Some(T::UInt32) => Some(4),
            _ => None,
        }
    }

    /// The encoder's buffer of zero control-point indices, past whose end an index reads 0 too:
    /// made the first time a draw asks for it, and held by the command buffer.
    fn zero_indices(&mut self) -> Result<Buffer, Error> {
        if let Some(zeros) = &self.zero_indices {
            return Ok(zeros.clone());
        }
        let zeros = self
            .core
            .device()
            .own_buffer(16, super::values::ResourceOptions::shared())?;
        self.core.retain(zeros.clone());
        self.zero_indices = Some(zeros.clone());
        Ok(zeros)
    }

    /// An indexed patch draw's control-point index buffer at `offset`: the offset the draw reads
    /// its indices from. Refused for another device and, where the encoder's pipeline state is a
    /// tessellation one with a control-point index type, as Metal's validation refuses it
    /// (`Invalid`, measured): for a direct draw reading `indices`, indices past the buffer from
    /// the offset. An offset that is not a multiple of the type's size is read from the offset
    /// rounded down to one, as plain Metal reads it (the M3 Max and the M4 alike: 32-bit indices at
    /// offsets 2 and 6, 16-bit at 1 and 3, direct and indirect), where its validation stops the
    /// draw ("controlPointIndexBufferOffset(1) must be a multiple of 2 bytes"). Any other pipeline
    /// state the draw refuses.
    fn check_control_point_indices(
        &self,
        call: &'static str,
        buffer: &Buffer,
        offset: u64,
        indices: Option<u64>,
    ) -> Result<u64, Error> {
        self.core.check_device(call, buffer.device())?;
        let Some(size) = self.control_point_index_size() else {
            return Ok(offset);
        };
        let offset = offset - offset % size;
        let Some(indices) = indices else {
            return Ok(offset);
        };
        let end = indices
            .checked_mul(size)
            .and_then(|bytes| offset.checked_add(bytes));
        if end.is_none_or(|end| end > buffer.length()) {
            let problem = DrawProblem::ControlPointIndexBufferRange {
                offset,
                indices,
                size,
                length: buffer.length(),
            };
            return Err(draw_problem(call, problem));
        }
        Ok(offset)
    }

    /// The lowering of [`RenderCommandEncoder::set_visibility_result_mode`].
    pub(crate) fn set_visibility_result_mode(
        &mut self,
        mode: VisibilityResultMode,
        offset: u64,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::VisibilityResultMode(mode, offset));
        let turn = self.group_turn();
        if let color_groups::Turn::Later(ran) = turn {
            // A later group draws as the first one does under the mode, and counts nothing.
            if !ran {
                return Err(color_groups::gap());
            }
            self.mirror_visibility(mode != VisibilityResultMode::Disabled);
            return Ok(());
        }
        let set = self.set_visibility_mode(mode, offset);
        self.group_outcome(turn, set.is_ok());
        set
    }

    /// [`set_visibility_result_mode`](Self::set_visibility_result_mode)'s setting, its turn
    /// among the pass's color groups taken.
    fn set_visibility_mode(
        &mut self,
        mode: VisibilityResultMode,
        offset: u64,
    ) -> Result<(), Error> {
        let offset = visibility::result_offset(offset);
        let device = self.core.device().clone();
        // A mode begins a query in the current instance, and `Disabled` ends the active one
        // there (`Instance::draws`).
        if mode != VisibilityResultMode::Disabled || self.visibility.active() {
            self.begin_current_draws()?;
        }
        if mode != VisibilityResultMode::Disabled && self.current.views.view_mask.count_ones() > 1 {
            self.visibility
                .check_mode("set_visibility_result_mode", &device, mode, offset)?;
            // No query to continue: no draw under one is drawn as views.
            let (mut next, stood) = self.next_instance(
                (&[], None),
                (false, false),
                ViewSetup::default(),
                false,
                None,
                &[],
                &[],
            )?;
            self.begin_new_draws(&mut next)?;
            self.install(next, stood, None, Split::Visibility);
        }
        let active = self.visibility.active();
        self.visibility.set_mode(
            "set_visibility_result_mode",
            &device,
            self.current.draws,
            mode,
            offset,
        )?;
        if mode != VisibilityResultMode::Disabled {
            device
                .shared()
                .counts
                .pool_compiled(crate::work_counts::PoolCompiled::VisibilityQuery, 1);
        }
        // A mode begins a query; `Disabled` ends the active one, if any.
        if mode != VisibilityResultMode::Disabled || active {
            self.current.recorded = true;
            // The query records in the instance: its draws record again no more.
            self.current.rerecord = None;
        }
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::end_encoding`].
    pub(crate) fn end_encoding(self: Box<Self>) -> Result<(), Error> {
        self.core.device().shared().counts.render_encoder_ended();
        // And the pass's later color groups, where it draws in groups (`render/color_groups.rs`).
        let (mut core, residency, log) = self.end_pass()?.with_later_groups()?;
        // A notice of a kind the device refuses while lowering, announced by the end of an encoder
        // that logged its calls, refuses the end (`render/deferral.rs`).
        if let (Some(call), Some(notice)) = (lowering::current(), lowering::take_refusal()) {
            core.fail_command_buffer(CommandBufferError::CallRefused(call));
            (core.command_buffer().state()).refuse_call(Error::Degraded(Box::new(notice)));
        }
        core.end_with_replay(
            EncoderKind::Render,
            Some(residency),
            log.map(|log| Replay::Render(Box::new(log))),
        )
    }

    /// An encoder made by `CommandBuffer::open_render` to encode an ended one again
    /// (`render/replay.rs`), ended as [`end_encoding`](Self::end_encoding) ends one, into its
    /// place: what it recorded, and what it made resident.
    pub(super) fn finish_replay(self: Box<Self>) -> Result<(Replayed, Option<Residency>), Error> {
        let (core, residency, _) = self.end_pass()?.with_later_groups()?;
        Ok((core.finish_replay()?, Some(residency)))
    }

    /// An encoder of `command_buffer`'s, ended and committed, opened again over `descriptor`
    /// with the `residency` it started with, its rasterization-rate map's `screen` as it took it
    /// (`render/replay.rs`).
    pub(super) fn replaying(
        command_buffer: &CommandBuffer,
        descriptor: &RenderPassDescriptor,
        residency: Residency,
        screen: Option<[u32; 2]>,
    ) -> Result<Box<RenderLowering>, Error> {
        command_buffer.open_render(descriptor, Some(residency), screen)
    }

    /// Refuse `call` as its lowering refused it, after it returned, for `error`
    /// (`render/deferral.rs`): the command buffer fails from this encoder on
    /// (`CommandBufferError::CallRefused`, `EncoderCore::fail_command_buffer`) and keeps `error`
    /// for `CommandBuffer::refusal`, the first such call's.
    pub(super) fn refuse_call(&mut self, call: EncoderCall, error: Error) {
        self.core
            .fail_command_buffer(CommandBufferError::CallRefused(call));
        self.core.command_buffer().state().refuse_call(error);
    }

    /// [`end_encoding`](Self::end_encoding)'s pass, all but its core's end and its later color
    /// groups' passes: the core, what the encoder made resident, its log, and what its later
    /// color groups draw (`render/color_groups.rs`).
    fn end_pass(mut self: Box<Self>) -> Result<color_groups::Ended, Error> {
        const CALL: &str = "end_encoding";
        // Over a mutable rasterization-rate map: Metal's GPU hangs at a pass of no draw call until
        // a command buffer with a draw call over such a map was committed (`rate_mapped_pass.rs`).
        if self.rate_map.as_ref().is_some_and(RateMappedPass::mutable) {
            match self.called {
                true => self.core.command_buffer().state().called_over_mutable_map = true,
                false => {
                    let called = &self.core.device().shared().mutable_rate_map_called;
                    if !called.load(std::sync::atomic::Ordering::Acquire) {
                        self.core.hang();
                    }
                }
            }
        }
        // What its tables bind is held until the command buffer completes, as Metal holds what an
        // encoder was handed (`ArgumentTable::let_go_all`).
        for table in [
            &mut self.vertex,
            &mut self.fragment,
            &mut self.object,
            &mut self.mesh,
            &mut self.tile,
        ] {
            table.let_go_all(&mut self.core);
        }
        if self.dispatched {
            self.instance_after_dispatch()?;
        }
        // The last instance recorded inline after a split, its commands still kept: recorded
        // now, before the pass's plan of the instances after it (`render/inline_splits.rs`).
        self.flush_inline()?;
        // Whether a command of the pass may read an attachment's bytes it did not name.
        let unnamed_reads =
            self.barrier_inside || self.reaches_unnamed || !self.executions.is_empty();
        let RenderLowering {
            mut core,
            mut earlier,
            current,
            mut layout,
            passes,
            mut visibility,
            dispatches,
            scratch: tile_scratch,
            shadow,
            returned_stencil,
            depth_stand_in,
            imageblock,
            barrier_splits,
            driver_restarts,
            tile_memory_splits,
            tile_memory,
            executions,
            pending,
            called,
            drew,
            drawn,
            coverage,
            inputs_read,
            residency,
            extended,
            reload_at,
            log,
            depth_writes,
            drew_depth_writes,
            inline,
            grouped,
            live,
            bind_memo,
            ..
        } = *self;
        // The views and sampler descriptions its binds made or found, which the device's next
        // render encoder takes (`binding::BindMemo`).
        core.device().shared().bind_memos.give(bind_memo);
        // The bind state its last draw left, which the device's next render encoder starts from
        // (`live_bind.rs`): no draw holds it once the encoder ends.
        core.device().shared().live_binds.give(live);
        earlier.push(current);
        let mut instances = earlier;
        // The first instance's render pass and framebuffer, where it recorded inline in the
        // command buffer the encoder holds for the pass (`render/inline_first.rs`), and the
        // instances' after a split that recorded inline there too (`render/inline_splits.rs`):
        // the pass's leading `inlined` instances.
        let inline_objects = match inline {
            inline_first::Inline::Begun(begun) => Some(begun.objects()),
            inline_first::Inline::Never | inline_first::Inline::Open { .. } => None,
        };
        let inline = inline_objects.is_some();
        let inlined = instances
            .iter()
            .take_while(|instance| instance.inline)
            .count();
        if inline && instances.len() > 1 {
            core.device().shared().counts.inline_pass_split();
        }
        // A color attachment's store action still `Unknown` stores nothing, as plain Metal's
        // `DontCare` (Metal probe case `store_unknown` on an M4, which its validation stops). So
        // does one that resolves, and an aspect of the depth/stencil attachment, resolving or not,
        // which leave a resolve texture untouched (color, depth and stencil alike): their resolves
        // go.
        for color in layout.colors.iter_mut().flatten() {
            if color.store == StoreAction::Unknown {
                color.store = StoreAction::DontCare;
                color.resolve = None;
                color.copied_resolve = None;
            }
        }
        if let Some(ds) = layout.depth_stencil.as_mut().filter(|ds| !ds.hidden) {
            let unknown = |aspect: Option<AspectTarget>| {
                aspect.is_some_and(|a| a.store == StoreAction::Unknown)
            };
            if unknown(ds.depth) || unknown(ds.stencil) {
                let settled = |a: AspectTarget| AspectTarget {
                    store: match a.store {
                        StoreAction::Unknown => StoreAction::DontCare,
                        store => store,
                    },
                    ..a
                };
                let depth = ds.depth.map(settled);
                let stencil = ds.stencil.map(settled);
                let stores = (depth.map(|d| d.store), stencil.map(|s| s.store));
                ds.resolve = late_resolve::remade(CALL, core.command_buffer(), ds, stores)?;
                ds.depth = depth;
                ds.stencil = stencil;
            }
        }
        // The stencil attachment's store action options against its store action as it ends
        // (Metal's deferred store action validation, `sample_positions.rs`; a color or depth
        // attachment's change nothing, as plain Metal's).
        if let Some(s) = layout
            .depth_stencil
            .as_ref()
            .and_then(|ds| ds.stencil.as_ref())
        {
            sample_positions::check_options(CALL, Attachment::Stencil, s.options, s.store, None)?;
        }
        // A pass that changes no texel records nothing, but holds its attachments until its
        // command buffer completes (`render/untouched.rs`); not the first group of a pass of
        // several color groups, whose later groups' colors its layout leaves out.
        // A pass that loads and stores its depth and set a state with depth writes, drawing
        // under none, keeps its words' signs on a device whose pass is Metal's own, where Metal
        // stores them without.
        if depth_writes && !drew_depth_writes {
            if let Some(own) = kept_signs(&layout) {
                own.texture.degrade_kept_signs();
            }
        }
        if grouped.is_none() && untouched::changes_nothing(&layout, called, &instances) {
            core.device().shared().counts.untouched_pass_skipped();
            let targets: Vec<Texture> = (layout.colors.iter().flatten())
                .map(|color| color.target.texture.clone())
                .chain(layout.depth_stencil.iter().flat_map(|ds| {
                    let separate = ds.separate.iter().flat_map(Separate::targets);
                    std::iter::once(&ds.target)
                        .chain(separate)
                        .map(|t| t.texture.clone())
                }))
                .collect();
            core.retain((targets, passes, inline_objects));
            // The instance's secondary, where the pass began one as it opened, is given up
            // unrun (`end_unrun`).
            if let [only] = instances.as_slice() {
                if only.draws != vk::CommandBuffer::null() {
                    // SAFETY: the secondary command buffer is recording, only this encoder
                    // records it, nothing recorded into it (`untouched::changes_nothing`), and
                    // nothing executes it.
                    let ended = unsafe { core.command_buffer().give_up_raw(only.draws) };
                    if ended.is_ok() {
                        core.device().shared().counts.unrun_instance_ended();
                    }
                }
            }
            return Ok(color_groups::Ended {
                core,
                residency,
                log,
                later: None,
            });
        }
        // An extended-range attachment's clear takes Metal's straight route where Metal keeps its
        // level compressed and the pass has no draw call, where a `half` NaN clear keeps its sign
        // (`nan_clear::keeps_nan_sign`), and goes through `half` elsewhere (bfloat: the consumer's
        // M3 Max alike; `BGR10_XR` and `BGRA10_XR`).
        for color in layout.colors.iter_mut().flatten() {
            let straight = color.load == LoadAction::Clear && keeps_nan_sign(&color.target, called);
            if let Some(value) = color.straight_clear.filter(|_| straight) {
                color.clear = value;
            }
        }
        // An attachment the pass clears and resolves alone into another pixel format, in a pass
        // without a draw call: Metal keeps its samples' earlier texels, compressed or not, so the
        // pass renders it into a backing its texture keeps for the queue, cleared and resolved
        // there, from which nothing is copied back (`attachment_stand_in::kept_backing`).
        if !called {
            let queue = core.command_buffer().command_queue().id();
            for color in layout.colors.iter_mut().flatten() {
                let across = color.clears_and_resolves_across_formats();
                if across && color.stand_in.is_none() && !color.memoryless {
                    let own = color.target.clone();
                    color.target = attachment_stand_in::kept_backing(
                        core.command_buffer(),
                        &own,
                        queue,
                        false,
                    )?;
                    color.stand_in = Some(StandIn {
                        own,
                        cause: Cause::Backing,
                        area: None,
                    });
                }
            }
        }
        // A texture's linear image renders in place only where every color attachment of each
        // instance is linear: where a draw made one an image of the runtime's (a stand-in of
        // another format, a fetched texture's scratch), the attachments the device renders only
        // so render through images their textures keep, but one a draw took a view of
        // (`linear_attachment.rs`): the first such color, of the first instance with one.
        let other = (instances.iter())
            .flat_map(|instance| instance.reformats.iter())
            .find(|r| r.copies)
            .map(|r| r.index);
        // The colors a draw took a view of: read as input attachments or in the pixel
        // interlock, as sample planes, through a view of another format, or its extended-range
        // shadow. A pass whose draws fetch through input attachments counts every color taken
        // for its bound linear ones (`linear_attachment.rs`).
        let viewed = ColorIndices::of((0..layout.colors.len()).filter(|&index| {
            let planes = layout.colors[index]
                .as_ref()
                .is_some_and(|color| color.planes.is_some());
            inputs_read.contains(&(index as u32))
                || planes
                || extended.made(index as u32)
                || instances.iter().any(|instance| {
                    instance.shape.storage.contains(&(index as u32))
                        || instance
                            .reformats
                            .iter()
                            .any(|r| r.index == index && !r.copies)
                })
        }));
        let taken = match layout.fetch_inputs {
            true => ColorIndices::of(0..layout.colors.len()),
            false => viewed,
        };
        // Whether a draw renders views, and the pass executes render commands, whose commands
        // record later, whatever they record (`render/executions.rs`).
        let views = instances
            .iter()
            .any(|instance| instance.views != ViewSetup::default());
        let executes = !executions.is_empty();
        // Nothing recorded restarts a driver's Metal pass (`WholePass`).
        let unbroken =
            !executes && !driver_restarts && !views && layout.sample_positions.is_empty();
        // The colors a pass of one unbroken instance hands a driver that stores `DONT_CARE` as
        // Metal's own `DontCare` so (`HandedToMetal`): none on another driver.
        let whole = match instances.as_slice() {
            [only] if unbroken => Some(WholePass {
                shape: &only.shape,
                drew,
            }),
            _ => None,
        };
        let handed = HandedToMetal::of(core.device(), &layout, whole);
        let handed = ColorIndices::of(
            (layout.colors.iter().enumerate())
                .filter_map(|(index, color)| Some((index, color.as_ref()?)))
                .filter(|&(index, color)| handed.as_ref().is_some_and(|h| h.color(index, color)))
                .map(|(index, _)| index),
        );
        // A texture over a buffer the pass does not store renders through an image its texture
        // keeps, so the buffer keeps its bytes, as Metal's does, where the pass does not hand it
        // to Metal's own `DontCare` and no draw took a view of its image (`kept_bytes.rs`).
        // Before the linear attachments settle: it is not linear.
        kept_bytes::settle(
            core.command_buffer(),
            &mut layout.colors,
            executes,
            |index| handed.contains(index),
            |index| viewed.contains(index),
            |index| taken.contains(index),
        )?;
        // A multisampled color attachment whose samples a resolve the runtime draws after the
        // pass reads, where the pass stores nothing of it (`MultisampleResolve`): it renders
        // through an image its texture keeps, so the texture keeps the samples it held before the
        // pass, as Metal's does, while the resolve reads what the pass rendered
        // (`resolve::keep_samples`).
        resolve::keep_samples(
            core.command_buffer(),
            &mut layout.colors,
            executes,
            |index| viewed.contains(index),
        )?;
        linear_attachment::settle(core.command_buffer(), &mut layout.colors, other, |index| {
            taken.contains(index)
        })?;
        // One the pass loads renders in place, apart over the box its draws write in; one it
        // clears through such an image, copied back, only where no command of the pass may read
        // its bytes: no barrier inside it, no command reaching memory it does not name, and no
        // command naming the attachment's bytes or a resource over its host pages or in a heap
        // (`linear_attachment::choose`).
        let reached_heaps = core.reached_heaps();
        let rebinds = &core.device().shared().rebinds;
        let may_read = ColorIndices::of((0..layout.colors.len()).filter(|&index| {
            let Some(color) = layout.colors[index].as_ref() else {
                return false;
            };
            if unnamed_reads || reached_heaps {
                return true;
            }
            let none = crate::tracker::Access::NONE;
            let key = color.target.texture.bound_use(none, none).key;
            let named = core.named_keys();
            if named.contains(&key) {
                return true;
            }
            rebinds.host_range(key).is_some_and(|bytes| {
                named.iter().any(|&other| {
                    rebinds
                        .host_range(other)
                        .is_some_and(|other| other.start < bytes.end && bytes.start < other.end)
                })
            })
        }));
        linear_attachment::choose(
            core.command_buffer(),
            &mut layout.colors,
            &coverage,
            |index| taken.contains(index),
            |index| may_read.contains(index),
        )?;
        let device = core.device().clone();
        // The pass's clears write a new texture's unwritten texels where they reach it: through
        // another image only where the pass stores the aspect, whose copy back writes it
        // (`fresh_unwritten.rs`). Before its `DontCare` loads take clears below.
        FreshUnwritten::cleared(&device, &layout)?;
        let count = instances.len();
        // The tiles the pass's draws reach, as far as the host decides them, before the
        // `DontCare` loads below become `Clear` loads (`clear_reach.rs`): where they reach the
        // tiles their primitives touch, at most those their viewports and scissors reach, and
        // none where those reach none.
        let reach = clear_reach::PassReach::of(&layout, called, drew);
        // The color attachments the pass loads as the encoder set them, before its `DontCare`
        // loads and reach clears below give any a clear before the pass and a load
        // (`render/shadows.rs` bounds the others' copies back by the draws' scissors). Not a
        // two-plane YCbCr texture in a pass that reaches every tile: Metal stores every tile of
        // it, each texel's codes re-encoded from what the tile loaded (`ycbcr.rs`).
        let mut loaded = [false; MAX_COLOR_ATTACHMENTS];
        for (loaded, color) in loaded.iter_mut().zip(&layout.colors) {
            *loaded = color.as_ref().is_some_and(|color| {
                color.load == LoadAction::Load
                    && !(reach == clear_reach::PassReach::Every
                        && ycbcr_renders(color.target.texture.pixel_format()))
            });
        }
        let loaded_colors = &loaded[..layout.colors.len()];
        let tiles = match reach {
            clear_reach::PassReach::Draws => drawn.tiles(&layout),
            _ => None,
        };
        let reach = match &tiles {
            Some(tiles) if tiles.is_empty() => clear_reach::PassReach::Nothing,
            _ => reach,
        };
        let tiles = tiles.filter(|tiles| !tiles.is_empty());
        // A `DontCare` load of an attachment the pass stores: loaded in a pass whose draws reach
        // no tile, else cleared to zero or its clear value, Metal's answer in each tile a draw
        // reaches (`clear_reach.rs`), and only in those `tiles` bounds.
        let dont_care = clear_reach::dont_care_loads(&mut layout, reach);
        if let Some(named) = dont_care.named.clone() {
            device.degrade(Degradation::DontCareLoadTiles, || {
                (
                    format!("{named} of a render pass, loaded with `DontCare`"),
                    "Metal clears only the tiles a draw's primitives touch, which depend on where \
                     the GPU places the vertices"
                        .to_owned(),
                )
            });
        }
        // A rotated texture's frame that a pass which draws loads by Metal's map, which is not
        // the inverse of its store: every tile turns, where Metal turns only those a draw
        // reaches (`rotated.rs`), but where the pass's draws reach every tile.
        let draws = reach == clear_reach::PassReach::Draws;
        // A two-plane YCbCr texture the pass loads and stores: its render shadow's copy back
        // stores every tile of the rectangle bounding the draws' scissors, where Metal stores
        // only those a draw's primitives reach (`ycbcr.rs`), but where they reach every tile.
        if let Some(named) = draws.then(|| shadows::ycbcr_loaded(&layout)).flatten() {
            device.degrade(Degradation::YCbCrLoadTiles, || {
                (
                    named,
                    "Metal loads and stores only the tiles a draw's primitives reach, which \
                     depend on where the GPU places the vertices"
                        .to_owned(),
                )
            });
        }
        if let Some(named) = draws.then(|| shadows::loaded_apart(&layout)).flatten() {
            device.degrade(Degradation::RotatedLoadTiles, || {
                (
                    named,
                    "Metal loads and stores only the tiles a draw's primitives reach, which \
                     depend on where the GPU places the vertices"
                        .to_owned(),
                )
            });
        }
        // A color attachment the last instance leaves out does not resolve there: a pass that
        // resolves one runs one more instance, of no draws, after its sample planes are merged
        // back, which resolves the pass (`sample_planes.rs`).
        // Nor does a stencil the last instance tests in the shader: its shadow is copied back
        // after it (`shadow_stencil.rs`). Nor do the layers a last instance of some of a layered
        // pass's layers does not render: a layered pass that resolves then ends with an instance of
        // no draws over every layer. Every instance resolves as the pass ends: into the targets set
        // last (`render/late_resolve.rs`).
        let resolving = PassShape::of(&layout)?;
        late_resolve::settle(
            instances.iter_mut().map(|instance| &mut *instance.shape),
            &resolving,
        );
        let last = instances
            .last()
            .ok_or_else(|| internal("a pass without instances"))?;
        let last_shape = &last.shape;
        let stencil_resolves = render_target::depth_stencil_steps(&layout)
            .any(|step| step.stencil != ResolveMode::None);
        // Nor does a depth the last instance tests in the shader: its stand-in is copied back
        // after it (`depth_stand_in.rs`).
        let depth_resolves =
            render_target::depth_stencil_steps(&layout).any(|step| step.depth != ResolveMode::None);
        let subpass_resolves = layout
            .colors
            .iter()
            .any(|color| render_target::subpass_resolve(color.as_ref()).is_some())
            || render_target::depth_stencil_steps(&layout).next().is_some();
        // Nor does a depth/stencil attachment at another layer or level than its image's first
        // on a device whose render pass resolves from the first alone: it is copied there into
        // a scratch image after the instances, which an instance of no draws resolves from.
        let resolve_source = ResolveSource::plan(core.command_buffer(), &layout)?;
        let resolve_after = layout.colors.iter().enumerate().any(|(index, color)| {
            render_target::subpass_resolve(color.as_ref()).is_some()
                && last_shape.storage.contains(&(index as u32))
        }) || (last.shadow && stencil_resolves)
            || (last.shader_tests && depth_resolves)
            || (subpass_resolves && !covers(last.views, layout.layers))
            || resolve_source.is_some();
        // The exact color resolves, drawn after the pass from the samples its last instance
        // stores (or a copy of them), or from the sample planes that hold them
        // (`color_resolve.rs`).
        let exact_resolves =
            ShaderResolves::plan(core.command_buffer(), &layout, &last_shape.storage)?;
        // The depth and stencil resolves picked by depth, drawn after the pass from the samples
        // its last instance stores (`depth_stencil_resolve.rs`).
        let picked_resolve = DrawnResolve::plan(&device, &layout)?;
        // The instances' shapes, and the resolving instance's after them where the pass has one,
        // kept for the announcements once the instances are taken apart below.
        let mut shapes: Vec<PassShape> = instances
            .iter()
            .map(|instance| instance.shape.clone())
            .collect();
        // The sample planes' steps before each instance and after the last: none in a pass
        // without planes.
        let planes = match layout.colors.iter().flatten().any(|c| c.planes.is_some()) {
            true => PlanesCommands::plan(
                &device,
                &layout,
                &sample_planes::steps(&layout, &shapes.iter().collect::<Vec<_>>()),
            )?,
            false => None,
        };
        if resolve_after {
            shapes.push(resolving.clone());
        }
        let mut uses = render_target::attachment_uses(&layout)?;
        if let Some(planes) = &planes {
            uses.extend(planes.uses.iter().cloned());
        }
        if let Some(source) = &resolve_source {
            uses.extend(source.uses.iter().cloned());
        }
        // The shader-side stencil's copies between the stencil and its shadow, around each
        // instance of such draws; the shadow is cleared in place of the first copy where the
        // first instance is one and clears the stencil.
        let shadow_copies = match (&shadow, &layout.depth_stencil) {
            (Some(shadow), Some(ds)) => {
                uses.extend(shadow.copy_uses(&ds.target)?);
                let clear = ds
                    .stencil
                    .as_ref()
                    .filter(|stencil| stencil.load == LoadAction::Clear)
                    .and_then(|stencil| match stencil.clear {
                        render_pass::AspectClear::Stencil(value) => Some(value),
                        render_pass::AspectClear::Depth(_) => None,
                    });
                Some(shadow.copies(&ds.target, clear)?)
            }
            (Some(_), None) => return Err(internal("a shader-side stencil without a stencil")),
            (None, _) => None,
        };
        // The shader-side depth test's copies between the depth and its stand-in, around each
        // instance of such draws; the instance that runs the pass's load actions fills the
        // stand-in from the depth's.
        let stand_in_copies = match (&depth_stand_in, &layout.depth_stencil) {
            (Some(stand_in), Some(ds)) => {
                uses.extend(stand_in.copy_uses(&ds.target)?);
                let depth = ds.depth.as_ref();
                Some(stand_in.copies(
                    &ds.target,
                    depth.map(|depth| depth.load),
                    depth.map(|depth| depth.clear),
                )?)
            }
            (Some(_), None) => return Err(internal("a shader-side depth test without a depth")),
            (None, _) => None,
        };
        let clear_values = render_target::clear_values(&layout);
        // Separate depth and stencil textures: the copies into the scratch image the pass
        // renders, and back (`depth_stencil_scratch.rs`).
        let (copy_in, copy_out) = match &layout.depth_stencil {
            Some(DepthStencilTarget {
                target,
                depth,
                stencil: Some(stencil),
                separate: Some(separate),
                ..
            }) => {
                let plan = |into_scratch| {
                    ScratchCopy::plan(
                        core.command_buffer(),
                        separate,
                        target,
                        [depth.as_ref(), Some(stencil)],
                        layout.extent,
                        into_scratch,
                    )
                };
                (plan(true)?, plan(false)?)
            }
            _ => (None, None),
        };
        // Fetched textures' scratch images: copied in where Metal loads the texture, and back
        // where it stores it (`attachment_stand_in.rs`).
        // Each over the render area, but the copy back of a stand-in its pass clears past the
        // render area, which covers that clear's reach (`clear_reach.rs`).
        // A `DontCare` load bounded to the tiles the pass's draws reach clears those tiles of the
        // stand-in alone and loads the rest in the pass (`ReachClears::load_in_pass`), whose
        // texels Metal keeps: so that stand-in is copied into too, over the reach its copy back
        // covers, and the copy back returns those texels as they came, not the stand-in's.
        // And a stand-in's over the box its draws write in alone (`StandIn::area`).
        let tiled = |taken: bool| taken && tiles.is_some();
        let area_of = |c: &render_pass::ColorTarget, own: &StandIn| {
            let [width, height] = match c.load == LoadAction::Clear {
                true => clear_reach::cleared(&layout, c.target.extent, called),
                false => layout.extent,
            };
            own.area.unwrap_or([0, 0, width, height])
        };
        let scratch = |copied: &dyn Fn(usize, &render_pass::ColorTarget) -> bool| {
            layout
                .colors
                .iter()
                .enumerate()
                .filter_map(|(index, c)| {
                    let c = c.as_ref()?;
                    let own = c.stand_in.as_ref().filter(|s| s.copies())?;
                    let area = area_of(c, own);
                    copied(index, c).then_some((
                        &c.target,
                        &own.own,
                        area,
                        vk::ImageAspectFlags::COLOR,
                    ))
                })
                .collect::<Vec<_>>()
        };
        // A depth/stencil texture's stand-in made for the pass's custom sample positions, or for
        // a texture without render target usage: each aspect copied in where the pass loads it
        // or keeps it (an aspect no attachment names), and back where it stores or keeps it, over
        // the reach of the pass's clear where an aspect is cleared and stored (`clear_reach.rs`),
        // in and back alike, so what lies past the render area goes back as it came or as the
        // clear left it (`attachment_stand_in.rs`).
        // And copied in where its `DontCare` load is bounded to the draws' tiles, as a color's.
        let depth_stencil_copies = |back: bool| {
            let ds = layout.depth_stencil.as_ref()?;
            let own = (ds.stand_in.as_ref()).filter(|s| s.copies())?;
            let stores = |store: StoreAction| {
                matches!(
                    store,
                    StoreAction::Store
                        | StoreAction::StoreAndMultisampleResolve
                        | StoreAction::CustomSampleDepthStore
                )
            };
            let mut aspects = vk::ImageAspectFlags::empty();
            let mut cleared = false;
            let [depth_bounded, stencil_bounded] = dont_care.aspects().map(tiled);
            for (aspect, kept, loaded, flag) in [
                (
                    ds.depth.as_ref(),
                    ds.keep_depth,
                    depth_bounded,
                    vk::ImageAspectFlags::DEPTH,
                ),
                (
                    ds.stencil.as_ref(),
                    ds.keep_stencil,
                    stencil_bounded,
                    vk::ImageAspectFlags::STENCIL,
                ),
            ] {
                let copied = match aspect {
                    Some(a) if back => stores(a.store),
                    Some(a) => a.load == LoadAction::Load || loaded,
                    None => kept,
                };
                if copied {
                    aspects |= flag;
                }
                cleared |= aspect.is_some_and(|a| a.load == LoadAction::Clear && stores(a.store));
            }
            let [width, height] = match cleared {
                true => clear_reach::cleared(&layout, own.own.extent, called),
                false => layout.extent,
            };
            let area = [0, 0, width, height];
            (!aspects.is_empty()).then_some((&ds.target, &own.own, area, aspects))
        };
        let loaded = |index: usize, c: &render_pass::ColorTarget| {
            c.load == LoadAction::Load || (tiled(dont_care.color(index)) && c.planes.is_none())
        };
        let stored = |c: &render_pass::ColorTarget| {
            matches!(
                c.store,
                StoreAction::Store | StoreAction::StoreAndMultisampleResolve
            )
        };
        // A color over a buffer's bytes rendered through a copied stand-in (a linear attachment
        // rendered apart, a texture without render target usage, a fetched texture's scratch),
        // copied in and stored: the words its pass changed merged back into the bytes, over the
        // area copied in, never a copy back of the area, which would write the pixels no draw
        // wrote with what the copy in read (`linear_merge.rs`).
        let over_bytes = |own: &StandIn| own.copies() && own.own.texture.aliases_buffer();
        let merged: Vec<(usize, Arc<LinearMerge>)> = layout
            .colors
            .iter()
            .enumerate()
            .filter_map(|(index, c)| {
                let c = c.as_ref()?;
                let own = c.stand_in.as_ref().filter(|s| over_bytes(s))?;
                (loaded(index, c) && stored(c)).then_some((index, c, own))
            })
            .map(|(index, c, own)| {
                LinearMerge::new(&device, &c.target, &own.own, area_of(c, own))
                    .map(|merge| (index, Arc::new(merge)))
            })
            .collect::<Result<_, Error>>()?;
        // One such color the pass does not load, and stores, is copied back over the clear's
        // reach before each instance after a memory barrier too, so a draw after it reads what
        // the pass drew (`reload_at`).
        let published = match reload_at.is_empty() {
            true => None,
            false => StandInCopies::plan_aspects(
                &scratch(&|index, c| {
                    stored(c) && !loaded(index, c) && c.stand_in.as_ref().is_some_and(over_bytes)
                }),
                false,
            )?,
        };
        // Where the first instance recorded inline, it rendered the attachments themselves: every
        // stand-in, arriving later, takes the attachment's texels right after it, not before the
        // pass (`render/inline_first.rs`).
        let mut copied_in = match inline {
            true => scratch(&|_, _| true),
            false => scratch(&loaded),
        };
        copied_in.extend(depth_stencil_copies(false));
        let stand_ins_in = StandInCopies::plan_aspects(&copied_in, true)?;
        let (stand_ins_in, stand_ins_after_first) = match inline {
            true => (None, stand_ins_in),
            false => (stand_ins_in, None),
        };
        if let Some(copies) = &stand_ins_after_first {
            uses.extend(copies.uses().iter().cloned());
        }
        // A memoryless color's backing, arriving later, where the first instance recorded inline
        // and rendered its transient image: cleared after it to what that left, the clear its
        // first instance loaded, which no draw of it wrote over (`render/inline_first.rs`). Where
        // the instance after it recorded inline too, the backings arrived with it and were
        // cleared before it (`render/inline_splits.rs`): their uses are these, recorded there.
        let backings_after_first = match inline {
            true => Clears::plan(
                (layout.colors.iter().flatten())
                    .filter(|c| {
                        c.memoryless
                            && c.load == LoadAction::Clear
                            && c.stand_in
                                .as_ref()
                                .is_some_and(|s| s.cause == Cause::Backing)
                    })
                    .map(|c| (&c.target, c.clear)),
            )?,
            false => None,
        };
        if let Some(clears) = &backings_after_first {
            uses.extend(clears.uses.iter().cloned());
            // A backing is cleared after the pass's first instance, not by a load operation: the
            // note that operation does not owe is (`ColorTarget::withheld_clear`).
            for color in layout.colors.iter().flatten().filter(|c| c.memoryless) {
                if let Some(clear) = color.withheld_clear {
                    let pf = color.target.texture.pixel_format();
                    super::render_pass::degrade_clear_conversion(core.device(), pf, clear);
                }
            }
        }
        let backings_after_first = backings_after_first.filter(|_| inlined == 1);
        let mut copied_back =
            scratch(&|index, c| stored(c) && !merged.iter().any(|&(at, _)| at == index));
        copied_back.extend(depth_stencil_copies(true));
        let stand_ins_out = StandInCopies::plan_aspects(&copied_back, false)?;
        // A clear holding a NaN the device's conversion or resolve decides (`nan_clear.rs`).
        super::nan_clear::degrade_nan_clears(core.device(), &layout, drew);
        // A depth whose `Clear` load held a NaN, loaded instead: Metal's word written into its
        // texture over the clear's reach before the pass, before anything copies the depth out
        // of it (`nan_clear.rs`).
        let nan_depth = match layout.depth_stencil.as_ref() {
            Some(
                ds @ DepthStencilTarget {
                    depth: Some(depth), ..
                },
            ) if depth.written_nan => {
                let own = match (&ds.separate, &ds.stand_in) {
                    (Some(separate), _) => separate.depth.as_ref(),
                    (None, Some(stand_in)) => Some(&stand_in.own),
                    (None, None) => Some(&ds.target),
                };
                let own = own.ok_or_else(|| internal("a NaN clear depth without its texture"))?;
                let extent = clear_reach::cleared(&layout, own.extent, called);
                Some(NanDepthWrite::plan(core.command_buffer(), own, extent)?)
            }
            _ => None,
        };
        // The attachments the first instance leaves out that Metal clears: cleared before the
        // pass, with the value the load operation takes (`next_instance`); an attachment's
        // sample planes are cleared in its place (`sample_planes.rs`).
        // Where no whole-subresource clear may stand for the load operation, a render pass of the
        // attachment alone clears it (`clear_reach.rs`), as it clears an attachment past the
        // render area.
        let mut clears = Vec::new();
        let mut left_out = Vec::new();
        for &index in &instances[0].shape.storage {
            let color = layout
                .colors
                .get(index as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| internal("a storage attachment the pass does not have"))?;
            // Cleared before the pass, not by its load operation: the note that operation does
            // not owe is (`ColorTarget::withheld_clear`).
            if let Some(clear) = color
                .withheld_clear
                .filter(|_| color.load == LoadAction::Clear)
            {
                let pf = color.target.texture.pixel_format();
                super::render_pass::degrade_clear_conversion(core.device(), pf, clear);
            }
            if color.load == LoadAction::Clear && color.planes.is_none() {
                match cleared_ahead(&color.target, layout.extent)? {
                    true => clears.push((&color.target, color.clear)),
                    false => left_out.push(index),
                }
            }
        }
        let clears = Clears::plan(clears)?;
        let bounded = tiles.as_ref().map(|tiles| (&dont_care, tiles));
        // Where the first instance recorded inline, its own clears reached as far as Metal's (it
        // rendered over their reach), and it had no other clear to make before it: no depth or
        // stencil, storage color or `DontCare` load but a memoryless color's
        // (`render/inline_first.rs`).
        let reach_clears = match inline {
            true => None,
            false => ReachClears::plan(&device, &layout, called, &left_out, bounded)?,
        };
        // An attachment the reach clear covers loads in the pass: that clear holds the render
        // area too, and a pass's own clear on a device that renders through Metal (MoltenVK)
        // reaches the whole attachment where the pass has no draw (`clear_reach.rs`). The pass
        // objects are made from the layout so loaded.
        if let Some(reach_clears) = &reach_clears {
            reach_clears.load_in_pass(&mut layout);
        }
        // The tile dispatches' cells past their attachments, as Metal's tile memory holds them
        // when the pass begins (`render/tile_dispatch.rs`): of the layout as the pass's clears
        // ahead of it leave its loads.
        let scratch_edges = ScratchEdges::plan(&layout, &tile_scratch)?;
        // The instances before the pass's hidden depth and stencil arrived have none: their
        // passes and framebuffers are the layout's without it.
        let bare = layout
            .depth_stencil
            .as_ref()
            .filter(|ds| ds.hidden)
            .map(|_| PassLayout {
                depth_stencil: None,
                ..layout.clone()
            });
        // The layout of the instances that resolve: the scratch image as the depth/stencil
        // attachment where its copy stands for it.
        let resolving_layout = resolve_source
            .as_ref()
            .map(|source| source.resolving(&layout));
        let resolving_layout = resolving_layout.as_ref().unwrap_or(&layout);
        let resolve_instance = if resolve_after {
            let at = Position {
                first: false,
                last: true,
            };
            let none = Feedback::NONE;
            let general = vk::ImageLayout::GENERAL;
            let pass = render_target::final_pass(
                &device,
                resolving_layout,
                &resolving,
                at,
                none,
                general,
            )?;
            let views = ViewSetup::default();
            let framebuffer = render_target::framebuffer(
                &device,
                &pass,
                resolving_layout,
                &resolving,
                views,
                none,
            )?;
            Some((pass, framebuffer))
        } else {
            None
        };
        // The fragment imageblock storage's clear: Metal's starts undefined
        // (`fragment_imageblock.rs`).
        let imageblock_clear = imageblock.as_ref().map(PassImageblock::clear).transpose()?;
        // The hidden depth and stencil's clear to Metal's 1.0 and 0, which every instance loads.
        let hidden_clear = layout
            .depth_stencil
            .as_ref()
            .filter(|ds| ds.hidden)
            .map(|ds| HiddenClear::plan(&ds.target))
            .transpose()?;
        // A single-sampled attachment's resolve: its texels, from the image the pass rendered
        // (a stand-in's where one stands in), copied into the resolve texture after the pass.
        let copied: Vec<Resolve> = layout
            .colors
            .iter()
            .flatten()
            .filter_map(|c| {
                Some(Resolve {
                    attachment: c.target.clone(),
                    copy_to: Some(c.copied_resolve.clone()?),
                    shader: None,
                    twin: false,
                    clear: None,
                })
            })
            .collect();
        // Resolves into scratch images: copied into their textures after the pass
        // (`resolve.rs`).
        let resolved: Vec<(&Resolve, metal2vulkan::vk::ImageAspectFlags)> = layout
            .colors
            .iter()
            .flatten()
            .filter_map(|c| c.resolve.as_ref())
            .chain(&copied)
            .map(|r| (r, metal2vulkan::vk::ImageAspectFlags::COLOR))
            .chain(
                layout
                    .depth_stencil
                    .iter()
                    .filter_map(|ds| ds.resolve.as_ref())
                    .flat_map(DepthStencilResolve::written),
            )
            .collect();
        // The depth/stencil resolves past the first, each in a pass of its own after the pass's
        // instances (`render_target.rs`).
        let later_resolves = render_target::later_steps(&layout)
            .map(|step| render_target::later_resolve(&device, resolving_layout, step))
            .collect::<Result<Vec<_>, Error>>()?;
        let resolve_copies = ResolveCopies::plan(&device, &resolved, layout.extent)?;
        // The tile dispatches' copies, from the pass's attachments as it ends.
        let mut staging = Vec::with_capacity(dispatches.len());
        for dispatch in &dispatches {
            staging.push(dispatch.copies(&layout)?);
        }
        let raw = device.shared().vk();
        if let Some(last) = instances.last() {
            visibility.end(raw, last.draws);
        }
        let results = visibility.plan(&mut core)?;
        // A shadowed texture the pass renders, resolves into or a draw uses.
        let texels_shadowed = shadows::attachments_shadowed(&layout)
            || instances
                .iter()
                .any(|instance| !instance.shadows.is_empty())
            || dispatches
                .iter()
                .any(|dispatch| !dispatch.shadows.is_empty());
        // Their truths, which an execution's commands may reach past what its call gathered
        // (`render/executions.rs`).
        let pass_shadows = match texels_shadowed {
            true => shadows::pass_shadows(
                &layout,
                (instances.iter().map(|instance| &instance.shadows))
                    .chain(dispatches.iter().map(|dispatch| &dispatch.shadows)),
            ),
            false => Vec::new(),
        };
        // The shadowed textures' copies before each instance: the first's before the pass's
        // other steps, the others' between the instances (`render/shadows.rs`), in the pass's
        // segment where it executes render commands (`render/executions.rs`).
        // Each tile dispatch's go before it, planned in the pass's order: after the instance before
        // it (`render/tile_dispatch.rs`). The dispatches follow the instances in the order they
        // were made, so theirs are in the dispatches' order.
        let (texel_copies, dispatch_copies) = if texels_shadowed {
            let mut order = Vec::with_capacity(instances.len() + dispatches.len());
            // The first instance of the group the instance is in, planned as the group ended
            // (`render/inline_splits.rs`).
            let mut head_planned = false;
            for (index, instance) in instances.iter().enumerate() {
                // An instance that continues the one before (`Instance::continues`) takes no
                // copies: its draws' shadowed textures were gathered into the first it continues,
                // whose copies go before them all; that one's writes are bounded no more.
                if instance.continues {
                    let first = order.iter_mut().rev().find(|(_, _, of)| !of);
                    if let (Some(first), false) = (first, head_planned) {
                        first.1 = None;
                    }
                    continue;
                }
                head_planned = instance.planned;
                // Nor does one recorded inline after a split, or the first instance before
                // one: its copies were planned and recorded as its group ended.
                if !instance.planned {
                    // The rectangle bounding the pixels an instance of draws alone writes,
                    // after the first, whose copies the steps before the pass's instances
                    // follow; none for an instance through views of another format, whose
                    // stand-ins' copies write its colors' every texel.
                    let alone =
                        index > 0 && instance.reformats.is_empty() && instance.pieces.is_empty();
                    let bound = alone.then(|| instance.drawn.bounds()).flatten();
                    order.push((&instance.shadows, bound, false));
                }
                let after = dispatches.iter().filter(|d| d.after == index + 1);
                order.extend(after.map(|dispatch| (&dispatch.shadows, None, true)));
            }
            let reached = reach != clear_reach::PassReach::Nothing;
            // A pass whose draws reach tiles changes the colors the encoder set to load inside
            // its draws' scissors alone (`render/shadows.rs`), whichever tiles it reaches: one
            // that reaches every tile (an attachment it clears and stores) loads and stores
            // their other texels as they were. A clear before the pass leaves them alone too: a
            // reach clear clears only attachments whose load is `Clear`, a `DontCare` load turned
            // into one among them (`clear_reach::ReachClears`), and no load the encoder set to
            // `Load` turns into `Clear`.
            let drawn_bounds = match reach {
                clear_reach::PassReach::Draws | clear_reach::PassReach::Every => drawn.bounds(),
                clear_reach::PassReach::Nothing => None,
            };
            let planned = shadows::plan(
                &mut core,
                order.iter().map(|&(s, bound, _)| (s, bound)),
                &layout,
                (reached, drawn_bounds.map(|area| (area, loaded_colors))),
                drawn_bounds.is_some().then_some(&drawn),
            )?;
            let (mut planned_copies, mut dispatch_copies) = (Vec::new(), Vec::new());
            for (&(_, _, of_dispatch), copies) in order.iter().zip(planned) {
                match of_dispatch {
                    true => dispatch_copies.push(copies),
                    false => planned_copies.push(copies),
                }
            }
            // Each instance's, none for one that continues another or was planned already.
            let mut planned_copies = planned_copies.into_iter();
            let texel_copies: Vec<Vec<ShadowCopy>> = (instances.iter())
                .map(|instance| match instance.continues || instance.planned {
                    true => Vec::new(),
                    false => planned_copies.next().unwrap_or_default(),
                })
                .collect();
            (texel_copies, dispatch_copies)
        } else {
            (Vec::new(), Vec::new())
        };
        let mut texel_copies = texel_copies.into_iter();
        let first_texel_copies = texel_copies.next();
        let later_texel_copies: Vec<Vec<ShadowCopy>> = texel_copies.collect();
        uses.extend(
            later_texel_copies
                .iter()
                .chain(&dispatch_copies)
                .flatten()
                .flat_map(ShadowCopy::uses),
        );
        uses.extend(
            instances
                .iter()
                .flat_map(|instance| &instance.before)
                .flat_map(|step| step.uses().iter().cloned()),
        );
        // The extended-range colors' shadows set before the instances that first hold them
        // (`render/extended_shadow.rs`).
        let (shadow_fills, fill_uses) = extended.settings(&device, &layout, instances.len())?;
        uses.extend(fill_uses);
        let extended_textures = extended.textures();
        // The copies of each color an instance renders through a stand-in of another format:
        // from the attachment before it, and back after it but where it is the pass's last and
        // the color's store discards it. Never the pass's first instance clearing that color
        // (`next_instance`), so no clear is lost. No list where no instance copies: each
        // instance's copies are then none.
        let copying = (instances.iter()).any(|i| i.reformats.iter().any(|r| r.copies));
        let mut reformat_copies = Vec::with_capacity(if copying { count } else { 0 });
        for (index, instance) in instances.iter().enumerate().filter(|_| copying) {
            let last = index + 1 == count && !resolve_after;
            let pairs: Vec<_> = instance
                .reformats
                .iter()
                .filter(|r| r.copies)
                .map(|r| {
                    let color = layout.colors.get(r.index).and_then(Option::as_ref);
                    let color =
                        color.ok_or_else(|| internal("a stand-in of a color the pass lacks"))?;
                    Ok((r, color))
                })
                .collect::<Result<_, Error>>()?;
            let into: Vec<_> = pairs
                .iter()
                .map(|(r, color)| (&r.target, &color.target, layout.extent))
                .collect();
            let back: Vec<_> = pairs
                .iter()
                .filter(|(_, color)| {
                    !last
                        || matches!(
                            color.store,
                            StoreAction::Store | StoreAction::StoreAndMultisampleResolve
                        )
                })
                .map(|(r, color)| (&r.target, &color.target, layout.extent))
                .collect();
            let copies = (
                StandInCopies::plan(&into, true)?,
                StandInCopies::plan(&back, false)?,
            );
            uses.extend(
                copies
                    .0
                    .iter()
                    .chain(&copies.1)
                    .flat_map(|c| c.uses().iter().cloned()),
            );
            reformat_copies.push(copies);
        }
        let stand_ins: Vec<Texture> = instances
            .iter()
            .flat_map(|instance| &instance.reformats)
            .filter(|r| r.copies)
            .map(|r| r.target.texture.clone())
            .collect();
        for instance in &instances {
            let pieces = instance.pieces.iter().filter_map(|piece| match *piece {
                Piece::Draws(draws) => Some(draws),
                // Ended once recorded, with the pass.
                Piece::Execution(_) | Piece::Pending(_) => None,
            });
            // An instance whose secondary never began recorded nothing (`Instance::draws`).
            let own = (!instance.inline && instance.draws != vk::CommandBuffer::null())
                .then_some(instance.draws);
            for draws in pieces.chain(own) {
                // SAFETY: only this encoder records the secondary command buffers.
                unsafe { core.command_buffer().end_raw(draws) }.map_err(vulkan)?;
            }
        }
        let [width, height] = layout.extent;
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        if let Some(first) = &first_texel_copies {
            core.shadow_copies(first);
        }
        if let Some(write) = nan_depth {
            write.record(&mut core);
            core.retain(write);
        }
        if let Some(copy) = copy_in {
            copy.record(&mut core);
        }
        if let Some(copies) = stand_ins_in {
            copies.record(&mut core);
        }
        // What the merged colors hold once copied in, before any step writes them.
        for (_, merge) in &merged {
            merge.record_keep(&mut core);
        }
        if let Some(clears) = reach_clears {
            clears.record(&mut core);
        }
        if let Some(clears) = clears {
            clears.record(&mut core);
        }
        if let Some(clear) = imageblock_clear {
            clear.record(&mut core);
        }
        if let Some(clear) = hidden_clear {
            clear.record(&mut core);
        }
        // After every step that writes the attachments ahead of the pass: a loaded attachment's
        // edges as its first instance loads them.
        if let Some(edges) = scratch_edges {
            edges.record(&mut core);
        }
        // The first color group's pass of several: the twin of the depth and stencil the later
        // groups render against, loaded as its first instance loads them, after every step
        // before it (`render/color_groups.rs`).
        let later = match grouped {
            Some(grouped) => grouped.later(&mut core, &layout)?,
            None => None,
        };
        // The multisampled depth the encoder holds in the attachment layout through all its
        // instances, and between the encoders that hold it (`texture/held_depth.rs`): a held
        // texture's own image as the depth/stencil attachment (no scratch, stand-in or hidden
        // depth stands for it), where nothing else in the encoder touches it. No command of the
        // encoder named it so far (its draws' bindings, the steps recorded ahead of the pass), no
        // step of the pass uses it but as the attachment, and no step after the instances reads or
        // writes it: no depth resolve drawn after them, in a pass of its own or in a resolving
        // instance, no shader-side depth or stencil test, and no execution of
        // render commands or draw whose pipeline is being made, whose pass records later.
        // The tile dispatches' pipelines, where every making finished made: a pass of one still
        // being made, or failed, records later too (`render/pending_tile.rs`).
        let tiles_made: Option<Vec<_>> = dispatches
            .iter()
            .map(|dispatch| dispatch.pipelines.made_now())
            .collect();
        let held_depth = match layout.depth_stencil.as_ref() {
            Some(ds) if ds.separate.is_none() && ds.stand_in.is_none() && !ds.hidden => {
                let none = crate::tracker::Access::NONE;
                let key = ds.target.texture.bound_use(none, none).key;
                let alone = !inline
                    && executions.is_empty()
                    && pending.is_empty()
                    && tiles_made.is_some()
                    && !resolve_after
                    && later_resolves.is_empty()
                    && picked_resolve.is_none()
                    && depth_stand_in.is_none()
                    && shadow.is_none()
                    && !core.named_keys().contains(&key)
                    && (uses.iter())
                        .filter(|used| used.key == key)
                        .all(crate::tracker::Use::is_depth_attachment);
                (alone && device.shared().textures.held.holds(key)).then_some(key)
            }
            _ => None,
        };
        if let Some(key) = held_depth {
            core.uses().hold(key);
        }
        // Those of the instances after the ones recorded inline, and the subresources of the
        // attachments each holds out of `GENERAL`, which none recorded inline does: the colors in
        // the feedback loop layout (`render/snapshot.rs`) and the depth/stencil attachment where
        // the device renders it faster in the attachment layout
        // (`render_target::depth_layout`), moved there before the instance and back after it,
        // but for a held depth, which is there already. An instance an execution of render
        // commands joins holds its depth/stencil in `GENERAL`: the execution's runs begin passes
        // of their own (`render/executions.rs`), each made in that layout.
        let mut objects = Vec::with_capacity(count - inlined);
        let mut later_moved = Vec::with_capacity(count - inlined);
        for (index, instance) in instances.iter().enumerate().skip(inlined) {
            let at = Position {
                first: index == 0,
                last: index + 1 == count && !resolve_after,
            };
            let own = instance_layout(&layout, bare.as_ref(), &instance.shape);
            let feedback = instance.feedback;
            // Its framebuffer holds the views its draws render colors through.
            let viewed = reformatted(own, &instance.reformats);
            let executes = (instance.pieces.iter()).any(|p| matches!(p, Piece::Execution(_)));
            let (depth, moved) = match (held_depth, executes) {
                (Some(_), _) => (held_depth::HELD.into(), vk::ImageLayout::GENERAL),
                (None, true) => (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL),
                (None, false) => {
                    let depth = render_target::depth_layout(&device, &viewed)?;
                    (depth, depth)
                }
            };
            if held_depth.is_some() {
                device.shared().counts.depth_instance_held();
            } else if depth != vk::ImageLayout::GENERAL {
                device.shared().counts.depth_attachment_moved();
            }
            let shape = &instance.shape;
            let pass = render_target::final_pass(&device, own, shape, at, feedback, depth)?;
            let framebuffer = render_target::framebuffer(
                &device,
                &pass,
                &viewed,
                shape,
                instance.views,
                feedback,
            )?;
            later_moved.push(render_target::moved_images(
                &viewed,
                instance.views,
                feedback,
                moved,
            )?);
            objects.push((pass, framebuffer));
        }
        // The draws and pieces of each instance after those recorded inline, and the steps
        // around the instances, which record alike whether the pass records now or later with
        // its executions (`render/pass_steps.rs`). An empty list of the extended-range shadows'
        // settings or of the stand-ins' copies leaves every instance's empty.
        let dispatch_count = dispatches.len();
        let mut drawn = Vec::with_capacity(count - inlined);
        let mut instance_steps = Vec::with_capacity(count);
        let shadow_fills = (shadow_fills.into_iter()).chain(std::iter::repeat_with(Vec::new));
        let none = || (None, None);
        let reformat_copies = (reformat_copies.into_iter()).chain(std::iter::repeat_with(none));
        let fills_and_copies = shadow_fills.zip(reformat_copies);
        let moved_images = (std::iter::repeat_with(Vec::new).take(inlined)).chain(later_moved);
        for (index, ((instance, (shadow_fills, (reformat_in, reformat_out))), moved)) in instances
            .into_iter()
            .zip(fills_and_copies)
            .zip(moved_images)
            .enumerate()
        {
            let Instance {
                draws,
                shape,
                shadow: shadowed,
                shader_tests,
                pieces,
                before,
                positioned,
                kept,
                ..
            } = instance;
            // Its commands, kept as it recorded inline after a split, were recorded at the pass's
            // end (`render/inline_splits.rs`): their room goes back to the device.
            if let Some(kept) = kept {
                device.shared().kept_rooms.give(kept);
            }
            // An instance holding attachments out of `GENERAL` runs in one render pass instance
            // of its own pass: no execution of render commands joins it (`Piece`), whose runs
            // would begin others (`render/snapshot.rs`, the depth's layout above).
            if !moved.is_empty() && pieces.iter().any(|p| matches!(p, Piece::Execution(_))) {
                return Err(internal("an execution in an instance of moved attachments"));
            }
            instance_steps.push(InstanceSteps {
                shadow: shadowed,
                shader_tests,
                shadow_fills,
                before,
                positioned,
                reformat_in,
                reformat_out,
                moved,
            });
            if index >= inlined {
                drawn.push((draws, pieces, shape));
            }
        }
        let mut dispatch_copies = dispatch_copies.into_iter();
        let steps = PassSteps {
            instances: instance_steps,
            family: core.queue_flags(),
            reload_at,
            reloads: merged.iter().map(|(_, merge)| Arc::clone(merge)).collect(),
            published,
            texel_copies: later_texel_copies,
            planes,
            shadow_copies,
            stand_in_copies,
            dispatches: dispatches
                .into_iter()
                .zip(staging)
                .map(|(dispatch, staging)| StagedDispatch {
                    dispatch,
                    staging,
                    copies: dispatch_copies.next().unwrap_or_default(),
                })
                .collect(),
            resolve_source,
            resolve_instance,
            later_resolves,
            clear_values,
            render_area,
            staged_rows: StagedRows {
                device: device.clone(),
                sets: Default::default(),
            },
        };
        // An inline first instance's steps before it go before the pass, into the command buffer
        // before the pass's, as the end records them before the pass's first instance; the
        // encoder then records on in the pass's, after that instance (`render/inline_first.rs`).
        if inline {
            if steps
                .instances
                .first()
                .is_some_and(|i| !i.before.is_empty())
            {
                core.command(vk::PipelineStageFlags2::NONE, &[], |raw, command_buffer| {
                    // SAFETY: the command buffer before the pass is recording outside a render
                    // pass, and only this encoder records it; the steps' objects are live until
                    // the command buffer completes (the encoder retains them with the draws).
                    unsafe { steps.before(raw, command_buffer, 0) }
                });
            }
            core.release_pass()?;
        }
        // The passes of the runs an execution may split an instance into (`render/executions.rs`).
        let mut run_objects = Vec::new();
        // The steps of a pass recorded now, or the pass that records later.
        let (mut recorded_steps, mut deferred) = (None, None);
        // A pass of a draw whose pipeline is being made records later too, once it is made
        // (`render/pending.rs`), and one of a tile dispatch's (`render/pending_tile.rs`).
        let now = match &tiles_made {
            Some(tiles) if executions.is_empty() && pending.is_empty() => Some(tiles),
            Some(_) | None => None,
        };
        if let Some(tiles) = now {
            core.command(
                render_target::ATTACHMENT_STAGES,
                &uses,
                |raw, command_buffer| {
                    // SAFETY: the encoder's command buffer is recording outside a render pass;
                    // the passes, framebuffers, attachments and the instances' command buffers
                    // (each recorded against a pass compatible with its own), the steps' objects
                    // (retained with them below: the shader-side stencil's shadow and staging
                    // buffer, the sample planes' objects, the resolving passes), and the
                    // shadowed textures copied between instances (the draws' and the
                    // attachments', retained with them), are live until the command buffer
                    // completes, as are the draws' copies and index bounds passes before their
                    // instances (retained at the draws); every image rests in `GENERAL` but a
                    // held depth, which only the instances use, in its attachment layout
                    // (`texture/held_depth.rs`), and each copy is recorded between two
                    // instances.
                    unsafe {
                        if let Some(results) = &results {
                            results.reset(raw, command_buffer);
                        }
                        // The instances recorded inline ended in its command buffer, this one
                        // (`render/inline_first.rs`, `render/inline_splits.rs`): a stand-in
                        // arriving later takes the attachment's texels after the last of them,
                        // whose steps after it follow (the others have none).
                        if let Some(last) = inlined.checked_sub(1) {
                            if let Some(copies) = &stand_ins_after_first {
                                copies.record_in(raw, command_buffer);
                            }
                            if let Some(clears) = &backings_after_first {
                                clears.record_in(raw, command_buffer);
                            }
                            steps.after(raw, command_buffer, last, tiles);
                        }
                        let later = drawn.iter().zip(&objects);
                        for (index, ((draws, _, _), (pass, framebuffer))) in (inlined..).zip(later)
                        {
                            // Each instance after the first sees everything the one before wrote.
                            if index > 0 {
                                Barrier::everything().record(raw, command_buffer);
                            }
                            steps.before(raw, command_buffer, index);
                            steps.begin(raw, command_buffer, index, pass.raw, framebuffer.raw);
                            // An instance whose secondary never began recorded nothing: its
                            // render pass instance runs its loads and stores alone.
                            if *draws != vk::CommandBuffer::null() {
                                raw.cmd_execute_commands(command_buffer, &[*draws]);
                            }
                            raw.cmd_end_render_pass(command_buffer);
                            steps.ended(raw, command_buffer, index);
                            steps.after(raw, command_buffer, index, tiles);
                        }
                        steps.end(raw, command_buffer);
                    }
                },
            );
            // Those made since their dispatches, held until the command buffer completes: none
            // in a pass without tile dispatches.
            if !tiles.is_empty() {
                core.retain(tiles.clone());
            }
            recorded_steps = Some(steps);
        } else {
            // The pass records later, as a segment of its own: its visibility results are reset
            // before it here, as query commands run in submission order.
            if let Some(results) = &results {
                core.command(vk::PipelineStageFlags2::NONE, &[], |raw, command_buffer| {
                    // SAFETY: the encoder's command buffer is recording outside a render pass;
                    // the pools live until the command buffer completes (retained with the
                    // results below).
                    unsafe { results.reset(raw, command_buffer) }
                });
            }
            // The instances recorded inline ended in the pass's command buffer, this one: the
            // steps after the last of them go there, before the pass's later instances
            // (`render/inline_first.rs`, `render/inline_splits.rs`).
            if let Some(last) = inlined.checked_sub(1) {
                core.command(
                    render_target::ATTACHMENT_STAGES,
                    &[],
                    |raw, command_buffer| {
                        // SAFETY: the command buffer is recording outside a render pass, and only this
                        // encoder records it; the stand-ins' copies and the steps' objects are live
                        // until the command buffer completes (held with the pass, and the layout's
                        // textures with the encoder).
                        unsafe {
                            if let Some(copies) = &stand_ins_after_first {
                                copies.record_in(raw, command_buffer);
                            }
                            if let Some(clears) = &backings_after_first {
                                clears.record_in(raw, command_buffer);
                            }
                            // The tile dispatches after the last of them record with the later
                            // instances, once their pipelines are made (`render/pending_tile.rs`).
                            steps.after_copies(raw, command_buffer, last);
                        }
                    },
                );
            }
            // The command buffer of the instances recorded inline and the steps after them goes
            // over to the segment, which places it after what it records before the pass.
            let held = match inline {
                true => Some((core.hand_over_pass()?, inlined)),
                false => None,
            };
            let mut pass_instances = Vec::with_capacity(count);
            let later = drawn.into_iter().zip(&objects);
            for (index, ((draws, pieces, shape), (pass, framebuffer))) in (inlined..).zip(later) {
                // An instance whose execution may reach the tile threadgroup memory, or hold a
                // command's barrier, may run as several, each run's draws after the run before
                // (`render/executions.rs`), each in a pass compatible with its own and its
                // framebuffer.
                let splits = pieces.iter().any(|piece| match *piece {
                    Piece::Execution(at) => executions[at].may_split(),
                    Piece::Draws(_) | Piece::Pending(_) => false,
                });
                let runs = match splits {
                    true => {
                        let mut run_pass = |first, last| {
                            let at = Position { first, last };
                            let own = instance_layout(&layout, bare.as_ref(), &shape);
                            let (none, general) = (Feedback::NONE, vk::ImageLayout::GENERAL);
                            let pass =
                                render_target::final_pass(&device, own, &shape, at, none, general)?;
                            let raw = pass.raw;
                            run_objects.push(pass);
                            Ok::<_, Error>(raw)
                        };
                        Some(RunPasses {
                            first: run_pass(index == 0, false)?,
                            between: run_pass(false, false)?,
                            last: run_pass(false, index + 1 == count && !resolve_after)?,
                        })
                    }
                    false => None,
                };
                // A pass that records later began every instance's secondary its executions and
                // pending draws record into (`Instance::draws`): one that never began, as a pass
                // whose first command is a tile dispatch leaves its first, records nothing, and
                // the pass executes an empty one for it.
                let draws = match draws == vk::CommandBuffer::null() {
                    true => {
                        let empty = core.command_buffer().begin_draws(pass)?;
                        // SAFETY: only this encoder records it, begun just now.
                        unsafe { core.command_buffer().end_raw(empty) }.map_err(vulkan)?;
                        empty
                    }
                    false => draws,
                };
                pass_instances.push(PassInstance {
                    pass: pass.raw,
                    framebuffer: framebuffer.raw,
                    pieces: pieces.into_iter().chain([Piece::Draws(draws)]).collect(),
                    runs,
                    shape: shape.into_inner(),
                    position: Position {
                        first: index == 0,
                        last: index + 1 == count && !resolve_after,
                    },
                });
            }
            let pass = Arc::new(DeferredPass::new(
                (pass_instances, held),
                steps,
                uses,
                (executions, pending),
                PassStorage {
                    tile_memory: tile_memory.into_buffer(),
                    imageblock: imageblock.clone(),
                    shadow: shadow.clone(),
                },
                (layout.clone(), bare.clone()),
                pass_shadows,
                core.command_buffer().state().open_encoder_place(),
            ));
            // The steps' objects, held with the pass until the command buffer completes.
            deferred = Some(Arc::clone(&pass));
            core.defer(Deferred::RenderPass(pass))?;
        }
        if let Some(resolves) = exact_resolves {
            resolves.record(&mut core);
        }
        if let Some(resolve) = picked_resolve {
            resolve.record(&mut core);
        }
        if let Some(copy) = copy_out {
            copy.record(&mut core);
        }
        if let Some(copies) = stand_ins_out {
            copies.record(&mut core);
        }
        for (_, merge) in &merged {
            merge.record_merge(&mut core);
        }
        if let Some(copies) = resolve_copies {
            copies.record(&mut core);
        }
        if let Some(results) = results {
            results.record(&mut core);
        }
        // Metal drops the pass's stores into a sparse texture's unmapped tiles: each sparse
        // texture it wrote is scrubbed after it (`sparse/emulated.rs`).
        let written = layout
            .colors
            .iter()
            .flatten()
            .flat_map(|c| std::iter::once(&c.target).chain(c.stand_in.iter().map(|s| &s.own)))
            .chain(layout.depth_stencil.iter().flat_map(|ds| {
                let separate = ds.separate.iter().flat_map(Separate::targets);
                std::iter::once(&ds.target)
                    .chain(separate)
                    .chain(ds.stand_in.iter().map(|s| &s.own))
            }))
            .chain(
                resolved
                    .iter()
                    .flat_map(|(r, _)| std::iter::once(&r.attachment).chain(r.copy_to.as_ref())),
            );
        for scrub in Scrub::of_pass(written, layout.extent) {
            core.defer(Deferred::SparseScrub(Arc::new(scrub)))?;
        }
        // The resolve textures' texels the pass's resolves wrote are no longer a new texture's
        // unwritten pattern (`fresh_unwritten.rs`).
        FreshUnwritten::resolved(&device, &resolved, layout.extent)?;
        // The pass's one instance, where nothing recorded restarts a driver's Metal pass.
        let whole = (shapes.len() == 1 && unbroken).then(|| WholePass {
            shape: &shapes[0],
            drew,
        });
        device.degrade_pass(&layout, whole);
        device.degrade_instances(
            &layout,
            &shapes,
            SplitCauses {
                shadow: shadow.is_some(),
                shader_tests: depth_stand_in.is_some(),
                dispatches: dispatch_count,
                barriers: barrier_splits,
                tile_memory: tile_memory_splits,
                viewed: views,
            },
        );
        device.degrade_planes(&layout);
        device.degrade_shadow_stencil(&layout, returned_stencil);
        device.degrade_shader_tests(&layout, depth_stand_in.is_some());
        if let Some(imageblock) = &imageblock {
            device.degrade_fragment_imageblock(imageblock);
        }
        let resolves = resolved.iter().flat_map(|(r, _)| {
            std::iter::once(&r.attachment)
                .chain(r.copy_to.as_ref())
                .map(|t| t.texture.clone())
        });
        let stood_in = layout
            .colors
            .iter()
            .flatten()
            .filter_map(|c| c.stand_in.as_ref())
            .chain(
                layout
                    .depth_stencil
                    .iter()
                    .filter_map(|ds| ds.stand_in.as_ref()),
            )
            .map(|s| s.own.texture.clone());
        let planes_images = layout
            .colors
            .iter()
            .flatten()
            .filter_map(|c| c.planes.as_ref())
            .map(|p| p.texture.clone());
        let targets: Vec<Texture> = layout
            .colors
            .iter()
            .flatten()
            .map(|c| c.target.texture.clone())
            .chain(layout.depth_stencil.iter().flat_map(|ds| {
                let separate = ds.separate.iter().flat_map(Separate::targets);
                std::iter::once(&ds.target)
                    .chain(separate)
                    .map(|t| t.texture.clone())
            }))
            .chain(resolves)
            .chain(stood_in)
            .chain(planes_images)
            .collect();
        // What the pass's commands use, held until the command buffer completes as one object,
        // which lets go of them in this order: the command buffer lets go of what it holds from
        // the last retained back.
        core.retain((
            imageblock.as_ref().map(PassImageblock::held),
            depth_stand_in.as_ref().map(DepthStandIn::held),
            shadow.as_ref().map(ShadowStencil::held),
            run_objects,
            objects,
            inline_objects,
            passes,
            layout.into_inner().visibility,
            tile_scratch,
            targets,
            merged,
            deferred,
            recorded_steps,
            stand_ins,
            extended_textures,
        ));
        // What it made resident joins its command buffer's (`residency.rs`), and is its own
        // (`CommandBuffer::ended_encoders`).
        Ok(color_groups::Ended {
            core,
            residency,
            log,
            later,
        })
    }

    /// Whether an indexed indirect draw stays one indirect draw whose index fetch the device's
    /// bound range bounds: the device enables `robustBufferAccess2`, and its driver is measured to
    /// read zero past the bound index range (`DriverFact::RobustIndexFetchReadsZero`,
    /// `device/driver_facts.rs`), which Vulkan leaves undefined. Elsewhere the index bounds pass
    /// draws it (`index_bounds.rs`).
    fn robust_index_fetch(&self) -> bool {
        let device = self.core.device();
        device
            .enabled()
            .features
            .contains(&VkFeature::RobustBufferAccess2)
            && device
                .properties()
                .driver_facts
                .contains(&DriverFact::RobustIndexFetchReadsZero)
    }

    /// Bring `live`, the bind state the encoder's last draw bound, to the encoder's tables as a
    /// bind state, in place (`live_bind.rs`), with every viewport the encoder holds: a pipeline
    /// whose vertex function selects a viewport draws an index at or past their count through
    /// viewport 0 (`BindState::viewport_count`), and no other pipeline reads the count. The
    /// tessellation factors are the state's for a tessellation pipeline only. The resources the
    /// encoder made resident are its used resources, each in its render stages, `argument_samplers`
    /// the device's sampler states a draw lists, each at the heap element of its resource ID, and
    /// `function_tables` the live tables its state made (`render/arguments.rs`). Each stage's table
    /// is read as the interface of `pipeline`, the drawing state, reads it: a buffer where a stage
    /// reads a visible-function table reads as a null table; a table read last through the same
    /// interface and maps, that no call changed, whose slots that tell the device's view base
    /// alignment apart the state still holds as it read them, and that no no-copy buffer's move
    /// may have changed since, is not read again (`live_bind::ReadFrom`). The vertex, object and
    /// mesh stages' tables are `pass_start`'s, with the copies of bytes the pass's fragment stages
    /// wrote, or the encoder's (`render/pass_start_reads.rs`; the object and mesh stages'
    /// changed). Each stage's samplers are bound as `maps` binds them, and the buffer slots that
    /// tell the device's view base alignment apart are the state's
    /// (`argument_table::AtBindViews`). The depth/stencil state and the fields the draw's settings
    /// state are the caller's to bring ([`settle_draw`](Self::settle_draw)); the values a draw sets
    /// past its settle, which no shape reads, are Metal's defaults until it sets them.
    fn refresh_bind_state(
        &self,
        live: &mut LiveBind,
        pass_start: &PassStartTables,
        (argument_samplers, function_tables): (&mut ArgumentSamplers, &[VisibleFunctionTable]),
        (pipeline, maps): (&DrawingState, &SamplerMaps),
    ) -> Reshape {
        let mut reshape = Reshape::default();
        let interface = &pipeline.base().interface;
        let state = &mut live.state;
        let views = pipeline.at_bind_views();
        reshape.field(&mut state.view_base_alignment, views.alignment());
        let tables = [
            (Stage::Vertex, pass_start.or(Stage::Vertex, &self.vertex)),
            (Stage::Fragment, &self.fragment),
            (Stage::Object, pass_start.or(Stage::Object, &self.object)),
            (Stage::Mesh, pass_start.or(Stage::Mesh, &self.mesh)),
        ];
        // Read again only where what a table brings in may have changed since it was read
        // (`live_bind::ReadFrom`); a table that keeps its stamp is checked in debug builds alone.
        let moved = crate::buffer::host_cells_moved();
        let same = live.read_from.reads(interface, maps, moved);
        // The state's used resources are what the encoder made resident still where the
        // residency kept its generation since they were made of it (`Residency::generation`), no
        // no-copy buffer moved since (their addresses), and each indirect command buffer made
        // resident holds its ID's slot as they state (`Residency::holds_slots_as`).
        let generation = self.residency.generation();
        let resident = live.read_from.resident(generation, moved);
        let counts = &self.core.device().shared().counts;
        for (at, (stage, table)) in tables.into_iter().enumerate() {
            let kept = state.stages.entry(stage).or_default();
            // Its slots that tell the view base alignment apart are as the state holds them
            // (`AtBindViews::holds`): another table's offsets may have left some out since.
            let views_held = views.holds(stage, kept.at_bind_views);
            if same && views_held && live.read_from.unchanged(at, table.stamp()) {
                #[cfg(debug_assertions)]
                table.check_reading(kept, (stage, interface), (maps, views));
                continue;
            }
            counts.stage_table_read();
            // Only the slots calls changed since it was read, where the table keeps them.
            if let Some(read) = live.read_from.read_under(at).filter(|_| same && views_held) {
                let read = (&mut *kept, read);
                if table.refresh_changed(read, (stage, interface), (maps, views), &mut reshape) {
                    counts.stage_table_read_in_part();
                    continue;
                }
            }
            table.refresh_reading(kept, (stage, interface), (maps, views), &mut reshape);
        }
        let stamps = tables.map(|(_, table)| table.stamp());
        live.read_from
            .read(stamps, (interface, || pipeline.held()), maps, moved);
        // The object function's threadgroup memory lengths, which its draws specialize.
        let object = state.stages.entry(Stage::Object).or_default();
        reshape.field_from(
            &mut object.threadgroup_lengths,
            &self.object_threadgroup_lengths,
        );
        let kept = resident && self.residency.holds_slots_as(&state.used_resources);
        debug_assert!(
            !kept || state.used_resources == self.residency.used_resources(),
            "a draw's kept used resources that are not its residency's"
        );
        if !kept {
            reshape.field(&mut state.used_resources, self.residency.used_resources());
        }
        live.read_from.made_resident(generation);
        // Listed again only where the device's sampler states changed since: not where the
        // listing kept is of every live state at the count of their slots' changes that still
        // holds; else they are taken and compared state by state.
        let device = self.core.device();
        let current = argument_samplers.states().is_none()
            && live.listed_at == Some(device.sampler_slots_changed());
        if !current {
            let states = argument_samplers.take(device);
            if !heap_sets::lists(&state.argument_samplers, states) {
                reshape.field(&mut state.argument_samplers, heap_sets::listed(states));
            }
            live.listed_at = argument_samplers.listed_at();
        }
        let listed = function_tables
            .iter()
            .map(VisibleFunctionTable::listed)
            .collect();
        reshape.field(&mut state.function_tables, listed);
        reshape.field(
            &mut state.render_target_array_length,
            self.render_target_array_length,
        );
        // Values no shape reads (`BindState::shape`), stored as they are: a change of one leaves
        // the shape as it was (`live_bind.rs`).
        state.heap_placement = HeapPlacement::ResourceId;
        state.viewport_count = self.state.viewport_count(true);
        // An extended-range blend computed in the shader reads the blend color as push data.
        state.blend_color = self.state.blend_color();
        // A fragment function that runs the pipeline's private alpha test reads the reference as
        // push data.
        state.alpha_test_reference = self.state.alpha_test_reference();
        // A fragment function that runs the draw's depth test itself clamps its depth as the
        // encoder's depth clip mode says, read as push data (its range: `draw_resident`).
        state.depth_clip_mode_clamp = self.state.depth_clip_mode_clamp();
        // A fragment function that runs the draw's stencil test itself reads the references as
        // push data.
        state.stencil_references = self.state.stencil_reference();
        // Only a tessellation pipeline reads the factors: any other's shape leaves them out. The
        // shape reads whether they name a buffer alone.
        let kind = self.pipeline.as_ref().map(RenderPipelineState::kind);
        let factors = match kind {
            Some(RenderStateKind::Graphics(graphics))
                if crate::tessellation::patches(&graphics.base().graphics).is_some() =>
            {
                self.factors.bind_state()
            }
            _ => TessellationFactors::default(),
        };
        reshape.shaped(&mut state.tessellation_factors, factors, |f| {
            f.buffer.is_some()
        });
        state.primitive_type = None;
        state.depth_range = <[f32; 2]>::default();
        reshape
    }

    /// The color attachments `graphics` reads, and may write, as storage images in the pixel
    /// interlock (framebuffer fetch's storage form, `framebuffer_fetch.rs`), by ascending index:
    /// each a color attachment of the pass of its format (the encoder held the pipeline's formats
    /// to the pass's), viewed in its format's storage view
    /// (`framebuffer_fetch::storage_view_format`), of a pass whose fetch is not through input
    /// attachments. In a multisampled pass each is read as its sample planes
    /// (`StorageAttachmentView::SamplePlanes`, a device without multisampled storage images,
    /// `sample_planes.rs`), of one layer per sample of the pass, or as the attachment's own
    /// multisampled storage image (`StorageAttachmentView::Attachment`, a device with them), of the
    /// pass's samples.
    fn storage_attachments(
        &self,
        graphics: &GraphicsState,
    ) -> Result<Vec<StorageAttachmentUse>, Error> {
        let mut storage = graphics.storage_attachments.clone();
        storage.sort_by_key(|s| s.color_index);
        let profile = self.core.device().compile_device().profile();
        for s in &storage {
            let present = self
                .layout
                .colors
                .get(s.color_index as usize)
                .and_then(Option::as_ref)
                .is_some_and(|c| c.target.format == s.format);
            if !present || self.layout.fetch_inputs {
                return Err(internal("a storage attachment the pass does not have"));
            }
            // The pass makes each fetched attachment's image take the storage view of its format
            // (`framebuffer_fetch::storage_view_format`: an 8-bit sRGB format's unsigned-integer
            // view, a BGRA-order one's RGBA-order twin where the device reads it without a format
            // in neither direction; the compile layer's `view_alias`), and no other.
            if s.view_format() != framebuffer_fetch::storage_view_format(profile, s.format) {
                return Err(internal(
                    "a storage attachment viewed in another format than its pass makes",
                ));
            }
        }
        let samples = self.layout.samples;
        for s in &storage {
            match s.view {
                StorageAttachmentView::Attachment => {}
                StorageAttachmentView::SamplePlanes { layers }
                    if layers == samples && samples > 1 => {}
                StorageAttachmentView::SamplePlanes { .. } => {
                    return Err(internal(
                        "sample planes of another count than the pass's samples",
                    ))
                }
            }
        }
        Ok(storage)
    }

    /// What the tests of `depth_stencil` can do to the fragments of a draw of `pipeline`
    /// (`BindState::fragment_tests`) whose fragment function may store a color through the
    /// interlocked storage form before them: one that states `RuntimeFact::TestedStorageWrites`,
    /// or one whose draw writes colors through that form (`stores`: the pass's extended-range
    /// shadows), whose variant may state it.
    /// `FragmentTests::Any`, which no other plan reads, for every other draw, so that its plans
    /// stay one per shape whatever its tests.
    fn fragment_tests(
        &self,
        pipeline: &DrawingState,
        depth_stencil: &DepthStencilDescriptor,
        stores: bool,
    ) -> FragmentTests {
        let base = pipeline.base();
        let tested = stores
            || base
                .dependencies
                .iter()
                .any(|d| matches!(d.fact, RuntimeFact::TestedStorageWrites { .. }));
        if !tested {
            return FragmentTests::Any;
        }
        FragmentTests::of(
            depth_stencil,
            base.graphics.depth_format.is_some(),
            base.graphics.stencil_format.is_some(),
            self.counts_visibility(),
        )
    }

    /// Whether a draw that reads and writes the colors `storage` (ascending) in the interlocked
    /// storage form, and writes the colors of the mask `written` (`GraphicsState::color_outputs`),
    /// records in the current instance as far as those colors go: the instance leaves out each of
    /// `storage`, and the draw writes none of the others it leaves out, whose writes would go
    /// nowhere.
    fn fits_storage(&self, storage: &[u32], written: u32) -> bool {
        let left_out = &self.current.shape.storage;
        storage.iter().all(|c| left_out.contains(c))
            && left_out.iter().all(|c| {
                storage.contains(c) || 1u32.checked_shl(*c).is_none_or(|bit| written & bit == 0)
            })
    }

    /// Whether a draw of `pipeline`, its colors routed by `routing` (its variant zeroing the pass
    /// colors at no place of its record where `zeroed`), writes a pass color of a memoryless
    /// texture rendered in place, through no stand-in (`linear_attachment::pass_writes`): what no
    /// instance recorded inline does (`render/inline_first.rs`).
    fn writes_memoryless(&self, pipeline: &DrawingState, routing: &Routing, zeroed: bool) -> bool {
        let writes = linear_attachment::writes(&pipeline.base().graphics.color_attachments);
        let writes = linear_attachment::pass_writes(&writes, routing, zeroed);
        (self.layout.colors.iter().zip(writes)).any(|(color, write)| {
            write.is_some()
                && color.as_ref().is_some_and(|color| {
                    color.stand_in.is_none() && (color.memoryless || memoryless(&color.target))
                })
        })
    }

    /// Whether the pass's color `index` renders into its own target for the rest of the pass,
    /// whatever a later draw fetches: a fetch in the interlock of a single-sampled pass that does
    /// not fetch input attachments renders it in place where its texture is not memoryless and its
    /// image has `STORAGE` and takes the storage view, and through a stand-in for the whole pass
    /// otherwise (`next_instance`), which a draw read through a snapshot of the texture would not
    /// see (`render/snapshot.rs`).
    fn keeps_own_attachment(&self, index: usize) -> bool {
        let Some(color) = self.layout.colors.get(index).and_then(Option::as_ref) else {
            return false;
        };
        let own = &color.target;
        let Ok(plan) = own.texture.image_plan() else {
            return false;
        };
        let profile = self.core.device().compile_device().profile();
        !self.layout.fetch_inputs
            && self.layout.samples == 1
            && color.stand_in.is_none()
            && !memoryless(own)
            && plan.usage.contains(STORAGE)
            && framebuffer_fetch::stores_in_place(profile, own.format, plan)
            && !super::render_pass::not_2d(own)
    }

    /// Where a draw that fetches the color attachments `fetched` (ascending indices: its input
    /// attachments, or in the pixel interlock its storage attachments) and renders the pass's
    /// layers `views` records, and the pass's layout from it on where stand-ins or sample planes
    /// arrive with it (the module docs, `attachment_stand_in.rs` and `sample_planes.rs`); a draw
    /// `ordered` after the current instance's draws (`render/tile_memory.rs`) records into a new
    /// one. Refused (`NotImplemented(Render)`) where a fetched texture's scratch cannot be
    /// copied, and (`Unsupported`) where the device cannot run the sample planes' copies.
    /// `by`: the state of the draw that fetches them, which a scratch's notice names.
    fn next_instance(
        &self,
        (fetched, by): (&[u32], Option<&DrawingState>),
        (shadow, shader_tests): (bool, bool),
        views: ViewSetup,
        ordered: bool,
        hidden: Option<Target>,
        locations: &[Option<u32>],
        reformats: &[Reformat],
    ) -> Result<(Next, Option<Box<PassLayout>>), Error> {
        let device = self.core.device();
        let maker = self.core.command_buffer();
        // The usage a fetched attachment's image needs: input attachment usage where the fetch
        // reads input attachments; in the interlock, whose draws' subpass leaves the attachment
        // out, `STORAGE`, or in a multisampled pass on a device that reads sample planes the
        // planes' copies' (sampled, viewed in its copy format).
        let (needs, storage) = if self.layout.fetch_inputs {
            (INPUT_ATTACHMENT, &[][..])
        } else {
            (STORAGE, fetched)
        };
        let profile = device.compile_device().profile();
        let planes = !self.layout.fetch_inputs
            && framebuffer_fetch::through_planes(profile, self.layout.samples);
        // What the draw's fetched attachments render into: a memoryless one's backing, the
        // scratch of a texture whose image lacks the usage, or themselves.
        let mut standing = Vec::new();
        let mut left_out = Vec::with_capacity(storage.len());
        for &index in fetched {
            let color = self
                .layout
                .colors
                .get(index as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| internal("a fetched attachment the pass does not have"))?;
            let own = &color.target;
            let plan = own.texture.image_plan()?;
            // In the interlock a fetching draw reads and writes the attachment through a storage
            // view in `view` (an 8-bit sRGB format's in the unsigned-integer format of its texel
            // size, a BGRA-order one's in its RGBA-order twin on ANV), which the device must store: the pipeline's link names that usage, so this
            // refuses only a pass its pipelines' device would not have made them for.
            let view = framebuffer_fetch::storage_view_format(profile, own.format);
            if needs == STORAGE && !profile.supports_format(view, FormatUsage::StorageImage) {
                return Err(Error::Unsupported(metal2vulkan::error::Unsupported::needs(
                    metal2vulkan::error::Feature::FramebufferFetch,
                    metal2vulkan::error::Capability::Format {
                        format: view,
                        usage: FormatUsage::StorageImage,
                    },
                )));
            }
            // The attachment's image takes that view where it has `STORAGE` and is its format,
            // or the attachment a view of it in a format the device stores, or the image is
            // mutable to the view (an sRGB format's, which a texture is not made; an RGBA-order
            // twin's, which a render target's free `STORAGE` and a writable texture list).
            let viewed_in_place = framebuffer_fetch::stores_in_place(profile, own.format, plan);
            let usable = if planes {
                // A memoryless attachment whose samples nothing reads after the pass needs no
                // backing: the planes' copies never run on it unless the pass splits, which backs
                // it then (below).
                let unread =
                    memoryless(own) && color.store == StoreAction::DontCare && !color.late_store;
                unread || framebuffer_fetch::planes_copies(plan)
            } else {
                // A view of another format is read and written in its own format, which must
                // be stored too (the image's usage is its own format's).
                plan.usage.contains(needs)
                    && (needs != STORAGE || viewed_in_place)
                    // A storage view of a 3D attachment's planes, or of a 1D one, is not the 2D
                    // image a fetching draw declares (`render_pass::not_2d`).
                    && (needs != STORAGE || !super::render_pass::not_2d(own))
            };
            let copies = if color.stand_in.is_some() || usable {
                None
            } else if memoryless(own) {
                Some(false)
            } else {
                attachment_stand_in::check_copies(own)?;
                Some(true)
            };
            // A stand-in for the whole pass would hide what an instance rendered through a view
            // of the attachment's image: refused (not built).
            let viewed = self
                .earlier
                .iter()
                .chain([&self.current])
                .any(|i| i.reformats.iter().any(|r| r.index == index as usize));
            if copies.is_some() && viewed {
                return Err(Error::NotImplemented(RuntimeGap::Render));
            }
            let target = match copies {
                Some(copies) => {
                    // A memoryless attachment's backing is one its queue keeps; a scratch is the
                    // pass's own.
                    let target = match copies {
                        false => attachment_stand_in::pooled_backing(
                            maker,
                            own,
                            &standing_images(&self.layout, &standing),
                        )?,
                        true => attachment_stand_in::stand_in(maker, own, copies)?,
                    };
                    // A texture fetched in place once its image has `STORAGE` takes its storage
                    // form at a later commit of the device's only uncommitted command buffer
                    // (`texture/storage_form.rs`).
                    let in_place = needs == STORAGE
                        && !planes
                        && viewed_in_place
                        && !super::render_pass::not_2d(own);
                    if copies && in_place {
                        own.texture.want_storage_form();
                    }
                    // A scratch names the draw that fetched it first in its notice.
                    let cause = match copies {
                        true => Cause::Fetch {
                            by: by.map_or_else(
                                || "a draw".to_owned(),
                                |state| state.descriptor().draw_at(index),
                            ),
                        },
                        false => Cause::Backing,
                    };
                    let stand_in = StandIn {
                        own: own.clone(),
                        cause,
                        area: None,
                    };
                    standing.push(Standing::Color(index as usize, target.clone(), stand_in));
                    target
                }
                None => own.clone(),
            };
            // The attachment's sample planes, over the render area, made at its first fetch.
            let clearable = match &color.planes {
                Some(planes) => Some(planes.clearable),
                None if planes => {
                    let made =
                        Planes::new(maker, &target, self.layout.samples, self.layout.extent)?;
                    let clearable = made.clearable;
                    standing.push(Standing::Planes(index as usize, made));
                    Some(clearable)
                }
                None => None,
            };
            if storage.contains(&index) {
                left_out.push((color.load, target, clearable));
            }
        }
        // The pass's hidden depth and stencil, arriving with the draw: its attachment from the
        // draw's instance on.
        if let Some(target) = hidden {
            standing.push(Standing::Hidden(target));
        }
        let images = standing.iter().any(|s| !matches!(s, Standing::Planes(..)));
        let takes = (storage, locations, reformats);
        if self.takes_current(takes, (shadow, shader_tests), views, images) && !ordered {
            let stood = (!standing.is_empty()).then(|| Box::new(stood(&self.layout, standing)));
            return Ok((Next::Current, stood));
        }
        // The first instance runs the pass's load actions. An attachment it leaves out that
        // clears is cleared before the pass where the clear covers its level and layer (the
        // render area is the whole of it), or its sample planes are where they take transfer
        // writes; otherwise an instance that clears it goes first.
        let first = self.earlier.is_empty();
        let mut ahead = true;
        for (load, target, clearable) in &left_out {
            if *load == LoadAction::Clear {
                ahead &= match clearable {
                    Some(clearable) => *clearable,
                    None => cleared_ahead(target, self.layout.extent)?,
                };
            }
        }
        // Nor can an instance that renders some of the pass's layers run a load action that clears:
        // it would clear those alone.
        let loads = covers(views, self.layout.layers) || !clears(&self.layout);
        // Nor can an instance that renders a color the pass clears through a view of another
        // format (its clear value is the attachment's format's): an instance of the pass's own
        // formats clears it first.
        let viewed_clears = reformats.iter().any(|r| {
            self.layout
                .colors
                .get(r.index)
                .and_then(Option::as_ref)
                .is_some_and(|color| color.load == LoadAction::Clear)
        });
        // An instance a tile dispatch follows runs before it, recorded or not.
        // An instance that records the current one's draws again takes its place as an empty
        // one's would (`render/reroute.rs`).
        let replace = !self.dispatched
            && (!self.current.recorded || self.rerouting)
            && (!first || (ahead && loads && !viewed_clears));
        let split = !replace;
        if split {
            // A split pass keeps its memoryless attachments in memory between its instances.
            for (index, color) in self.layout.colors.iter().enumerate() {
                let Some(color) = color.as_ref() else {
                    continue;
                };
                let own = &color.target;
                let stood = standing
                    .iter()
                    .any(|s| matches!(s, Standing::Color(i, ..) if *i == index));
                if color.stand_in.is_none() && !stood && memoryless(own) {
                    let taken = standing_images(&self.layout, &standing);
                    let target = attachment_stand_in::pooled_backing(maker, own, &taken)?;
                    let stand_in = StandIn {
                        own: own.clone(),
                        cause: Cause::Backing,
                        area: None,
                    };
                    standing.push(Standing::Color(index, target, stand_in));
                }
            }
        }
        // So does a pass whose stencil is copied into a shader-side stencil's shadow and back, and
        // one whose depth is copied into its stand-in and back.
        if split || shadow || shader_tests {
            if let Some(ds) = &self.layout.depth_stencil {
                if ds.stand_in.is_none() && memoryless(&ds.target) {
                    let target = attachment_stand_in::stand_in(maker, &ds.target, false)?;
                    let stand_in = StandIn {
                        own: ds.target.clone(),
                        cause: Cause::Backing,
                        area: None,
                    };
                    standing.push(Standing::DepthStencil(target, stand_in));
                }
            }
        }
        let stood = (!standing.is_empty()).then(|| Box::new(stood(&self.layout, standing)));
        let layout = stood.as_deref().unwrap_or(&self.layout);
        let mut shape = PassShape::of(layout)?
            .with_storage(storage)?
            .with_view_mask(views.view_mask)
            .with_locations(locations)?;
        for reformat in reformats {
            match shape.colors.get_mut(reformat.index) {
                Some(Some(format)) => *format = reformat.target.format,
                _ => return Err(internal("a view of a color the pass lacks")),
            }
        }
        let compatible = match self.passes.iter().find(|(s, _)| *s == shape) {
            Some((_, pass)) => Arc::clone(pass),
            None => render_target::shared_compatible_pass(device, &shape)?,
        };
        // Its secondary begins where its caller installs it, but where it records inline
        // after a split (`begin_new_draws`).
        let instance = Instance {
            draws: vk::CommandBuffer::null(),
            shape: Versioned::new(shape),
            compatible,
            recorded: false,
            shadow,
            shader_tests,
            imageblock: false,
            interlocked_writes: InterlockedStorage::NONE,
            indirect_reads: IndirectReads::new(),
            geometry_writes: GeometryWrites::new(),
            tile_memory: false,
            barriers: false,
            views,
            pieces: Vec::new(),
            shadows: InstanceShadows::default(),
            before: Vec::new(),
            reformats: reformats.to_vec(),
            positioned: None,
            // And in a pass that may hold an extended-range color's shadow, on every device.
            rerecord: self.extended.keeps_draws().then(Vec::new),
            bound: Bound::default(),
            drawn: clear_reach::DrawnArea::default(),
            inline: false,
            continues: false,
            kept: None,
            planned: false,
            feedback: Feedback::NONE,
            storage_colors: 0,
        };
        let instance = Box::new(instance);
        Ok((Next::New { instance, split }, stood))
    }

    /// Whether the current instance takes a draw, unordered after the draws before it, that
    /// renders `storage` attachments as storage images, at `locations`, through `reformats`,
    /// testing the stencil (`shadow`) or the depth (`shader_tests`) in the shader, with `views`,
    /// and that brings no image of its own (a stand-in or hidden depth, `images`): the same
    /// shape, after no tile dispatch or memory barrier ([`next_instance`](Self::next_instance)).
    fn takes_current(
        &self,
        (storage, locations, reformats): (&[u32], &[Option<u32>], &[Reformat]),
        (shadow, shader_tests): (bool, bool),
        views: ViewSetup,
        images: bool,
    ) -> bool {
        !images
            && self.current.shape.storage == storage
            && self.current.shape.locations == locations
            && Reformat::same(&self.current.reformats, reformats)
            && self.current.shadow == shadow
            && self.current.shader_tests == shader_tests
            && self.current.views == views
            && !self.dispatched
            && !self.barrier
    }

    /// The shadowed textures and uses the current instance's draws gather into: its own, or
    /// where it continues an instance recorded inline (`Instance::continues`), the first of the
    /// instances it continues, whose copies in go before them all.
    fn gathering(&mut self) -> &mut InstanceShadows {
        if !self.current.continues {
            return &mut self.current.shadows;
        }
        match self
            .earlier
            .iter()
            .rposition(|instance| !instance.continues)
        {
            Some(first) => &mut self.earlier[first].shadows,
            None => &mut self.current.shadows,
        }
    }

    /// The steps the current instance's draws take before it (`Before`): its own, or where it
    /// continues an instance recorded inline (`Instance::continues`), the first of the instances
    /// it continues, before which they record, as they would where that one took the draws.
    fn steps_before(&mut self) -> &mut Vec<Before> {
        if !self.current.continues {
            return &mut self.current.before;
        }
        match self
            .earlier
            .iter()
            .rposition(|instance| !instance.continues)
        {
            Some(first) => &mut self.earlier[first].before,
            None => &mut self.current.before,
        }
    }

    /// What [`gathering`](Self::gathering) gathers into.
    fn gathered(&self) -> &InstanceShadows {
        if !self.current.continues {
            return &self.current.shadows;
        }
        match self
            .earlier
            .iter()
            .rposition(|instance| !instance.continues)
        {
            Some(first) => &self.earlier[first].shadows,
            None => &self.current.shadows,
        }
    }

    /// The next run of an execution's commands (`render/executions.rs`): a draw that reaches the
    /// tile threadgroup memory after another of the run did records into a new secondary of the
    /// same instance, which the execution's pass runs in a render pass instance of its own after
    /// a barrier from everything before (`render/tile_memory.rs`). The pass renders a memoryless
    /// attachment into its backing from the execution's first run on (`render/executions.rs`), as
    /// the encoder's own split keeps it in a stand-in (`next_instance`).
    fn next_run(&self) -> Result<Next, Error> {
        let compatible = Arc::clone(&self.current.compatible);
        let draws = self.core.command_buffer().begin_draws(&compatible)?;
        let instance = Instance {
            draws,
            shape: Versioned::new(self.current.shape.clone()),
            compatible,
            recorded: false,
            shadow: false,
            shader_tests: false,
            imageblock: false,
            interlocked_writes: InterlockedStorage::NONE,
            indirect_reads: IndirectReads::new(),
            geometry_writes: GeometryWrites::new(),
            tile_memory: false,
            barriers: false,
            views: self.current.views,
            pieces: Vec::new(),
            shadows: InstanceShadows::default(),
            before: Vec::new(),
            reformats: self.current.reformats.clone(),
            positioned: None,
            rerecord: None,
            bound: Bound::default(),
            drawn: clear_reach::DrawnArea::default(),
            inline: false,
            continues: false,
            kept: None,
            planned: false,
            feedback: Feedback::NONE,
            storage_colors: 0,
        };
        Ok(Next::New {
            instance: Box::new(instance),
            split: true,
        })
    }

    /// The run of an execution's commands that `instance` begins (`render/executions.rs`): a
    /// command whose draw needs another instance than the run's (attachments it fetches as
    /// storage images, which its subpass leaves out, or views of its own) records into a new
    /// secondary of that shape, which the execution's pass runs in a render pass instance of
    /// its own after a barrier from everything before, as a draw that reaches the tile
    /// threadgroup memory begins a run of its instance's shape; the next command of the
    /// instance's shape begins another; a memoryless attachment is rendered as
    /// [`next_run`](Self::next_run) says. A run whose draws test the stencil in the shader takes
    /// the shadow's copies around it (`render/executions.rs`), as an encoder's instance of such
    /// draws does.
    fn run_of(&self, instance: Instance) -> Result<Next, Error> {
        Ok(Next::New {
            instance: Box::new(instance),
            split: true,
        })
    }

    /// Make `next` the instance the encoder records into (the module docs), with `stood`, the
    /// pass's layout from it on, and `continuation`, the query that continues the active one in
    /// a new instance past a split; a split counted for `cause`, or for the program's barrier or
    /// a tile dispatch since the current instance recorded where one came.
    fn install(
        &mut self,
        next: Next,
        stood: Option<Box<PassLayout>>,
        continuation: Option<Continuation>,
        cause: Split,
    ) {
        if let Some(layout) = stood {
            *self.layout = *layout;
            self.kept_attachments = None;
        }
        if let Next::New {
            mut instance,
            split,
        } = next
        {
            if !self.passes.iter().any(|(s, _)| *s == *instance.shape) {
                let pass = (instance.shape.clone(), Arc::clone(&instance.compatible));
                self.passes.push(pass);
            }
            if let Some(continuation) = continuation {
                let (from, into) = (self.current.draws, instance.draws);
                let raw = self.core.device().shared().vk();
                self.visibility.continue_in(raw, from, into, continuation);
                instance.recorded = true;
                instance.rerecord = None;
            }
            if split {
                let cause = match (self.barrier, self.dispatched) {
                    (true, _) => Split::Barrier,
                    (false, true) => Split::TileDispatch,
                    (false, false) => cause,
                };
                self.core.device().shared().counts.instance_split(cause);
                let previous = std::mem::replace(&mut self.current, *instance);
                self.earlier.push(previous);
                self.barrier_splits += usize::from(self.barrier);
                if self.barrier {
                    self.reload_at.insert(self.earlier.len());
                }
            } else {
                // It takes the place of the current one, whose draws it records again.
                let mut instance = *instance;
                instance.drawn.join(&self.current.drawn);
                let replaced = std::mem::replace(&mut self.current, instance);
                self.end_unrun(replaced);
            }
            self.dispatched = false;
            self.barrier = false;
        }
    }

    /// Give up the draws recording of `instance`, one the pass replaced or gave up before running
    /// it ([`install`](Self::install), `render/reroute.rs`): never submitted,
    /// a deferred one is never replayed, and one of the pool's ends so the pool goes back to the
    /// device's spares unreset (`CommandBuffer::give_up_raw`). One with pieces (the secondaries
    /// of draws whose pipelines were being made) is left to the pool's reset; one whose secondary
    /// never began has none to give up.
    fn end_unrun(&self, instance: Instance) {
        // Nor one recorded inline, in the pass's command buffer or kept (`render/inline_first.rs`,
        // `render/inline_splits.rs`).
        if !instance.pieces.is_empty()
            || instance.draws == vk::CommandBuffer::null()
            || instance.inline
        {
            return;
        }
        // SAFETY: the secondary command buffer is recording, only this encoder records it, and
        // nothing executes it; no query is active in it: a query begins or continues only in an
        // instance none of whose draws record again (`rerecord` `None`, which `recorded` marks
        // too), never one a pass replaces or a reroute leaves.
        let ended = unsafe { self.core.command_buffer().give_up_raw(instance.draws) };
        if ended.is_ok() {
            self.core.device().shared().counts.unrun_instance_ended();
        }
    }

    /// The encoder's core: its command buffer, uses and fences (`fence.rs`).
    pub(super) fn core(&mut self) -> &mut EncoderCore {
        &mut self.core
    }

    /// Order the draws after a memory barrier behind those before it (`fence.rs`): when a draw
    /// or query recorded into the current instance, the next draw records into a new one, after
    /// a barrier from everything before. Earlier instances are behind such a barrier already.
    pub(super) fn barrier_before_next_draw(&mut self) {
        self.barrier_inside = true;
        self.barrier |= self.current.recorded;
        self.unordered.clear();
        self.barriered = true;
    }

    /// Order the fragment stages of the draws after a memory barrier over the render targets
    /// behind those of the draws before it, pixel by pixel (`fence.rs`): when a draw or query
    /// recorded into the current instance, a barrier of its subpass's dependency on itself
    /// (`render_target::between_draws`). Earlier instances are behind a split's barrier already.
    pub(super) fn barrier_between_draws(&mut self) {
        self.barrier_inside = true;
        // A pass that renders a color of a buffer's bytes through a copied stand-in splits there,
        // so the stand-in is merged back and copied in again before the draws after the barrier,
        // which may read the bytes (`reload_at`).
        if self.reloads_at_barriers() {
            self.barrier_before_next_draw();
            return;
        }
        if !self.current.recorded {
            return;
        }
        // The barrier records in the instance: its draws record again no more.
        self.current.rerecord = None;
        self.current.barriers = true;
        self.driver_restarts = true;
        let view_mask = self.current.shape.view_mask;
        // An instance recorded inline after a split keeps it with its draws
        // (`render/inline_splits.rs`).
        if let Some(kept) = &mut self.current.kept {
            kept.barrier(view_mask);
            return;
        }
        let raw = self.core.device().shared().vk();
        // SAFETY: the secondary command buffer (or the pass's, inline) is recording inside the
        // compatible pass of its shape, which `render_target` made, and only this encoder records
        // it.
        unsafe { render_target::between_draws(raw, self.current.draws, view_mask) };
    }

    /// Bound the copy in of each shadow among `shadows` of a texture that presents a color
    /// attachment of the pass and that a draw of `plan` reads at its fragments' own pixels alone
    /// (`texture/attached_reads.rs`): every binding of the texture in its tables is a sampled one
    /// of a fragment slot `compiled`'s fragment module reads so (`StageModule::own_pixel_textures`),
    /// its shadow is cut to a rectangle (`ShadowCopies::cuts`), the draw renders no views, and the
    /// pass has no rasterization-rate map and a render area below `2^24` pixels a side. Its use
    /// reads (`ShadowUse::loads`) the texels inside the rectangle bounding
    /// the draw's Vulkan scissors (`selects`: those of every viewport) widened by a pixel each
    /// way, for its helper invocations, and clamped as the read's coordinate is: on each axis
    /// `[min(max(x0 - 1, 0), w - 1), min(x1 + 1, w))`, `w` the texture's. Whether one was.
    fn bound_own_pixel_reads(
        &self,
        plan: &BindingPlan,
        compiled: &metal2vulkan::pipeline::RenderPipeline,
        selects: bool,
        shadows: &mut [shadow::ShadowUse],
    ) -> bool {
        const BOUND: u32 = 1 << 24;
        let [width, height] = self.layout.extent;
        if self.rate_map.is_some() || plan.views.view_mask != 0 || width >= BOUND || height >= BOUND
        {
            return false;
        }
        let Some(fragment) =
            (compiled.stages.iter()).find(|m| m.stage == metal2vulkan::vk::ShaderStage::FRAGMENT)
        else {
            return false;
        };
        let mut scissors = [vk::Rect2D::default(); MAX_VIEWPORTS];
        let rects = self.state.vk_scissors_in(selects, &mut scissors);
        let mut drawn: Option<[u32; 4]> = None;
        for r in rects
            .iter()
            .filter(|r| r.extent.width > 0 && r.extent.height > 0)
        {
            let (x, y) = (r.offset.x.max(0) as u32, r.offset.y.max(0) as u32);
            let rect = [x, y, x + r.extent.width, y + r.extent.height];
            drawn = Some(match drawn {
                Some([a, b, c, d]) => [
                    a.min(rect[0]),
                    b.min(rect[1]),
                    c.max(rect[2]),
                    d.max(rect[3]),
                ],
                None => rect,
            });
        }
        // A draw of no pixel reads nothing: its copy is left as it was.
        let Some([x0, y0, x1, y1]) = drawn else {
            return false;
        };
        let mut bounded = false;
        for used in shadows.iter_mut() {
            if used.writes || !used.copies.cuts() {
                continue;
            }
            let key = used.copies.shadow_key;
            // The texture whose shadow it is, and whether each of its bindings reads so.
            let mut texture = None;
            let mut own = true;
            for write in &plan.descriptors {
                let slot = match &write.source {
                    WriteSource::Texture { slot, .. }
                    | WriteSource::CombinedImageSampler {
                        image: ImageSource::Texture { slot, .. },
                        ..
                    } => slot,
                    _ => continue,
                };
                let table = match slot.stage {
                    Stage::Vertex => &self.vertex,
                    Stage::Fragment => &self.fragment,
                    Stage::Object => &self.object,
                    Stage::Mesh => &self.mesh,
                    _ => continue,
                };
                let Some(bound) = table.textures.get(&slot.index) else {
                    continue;
                };
                if bound.usage_shadow_copies().map(|c| c.shadow_key) != Some(key) {
                    continue;
                }
                own &= slot.stage == Stage::Fragment
                    && write.ty != metal2vulkan::vk::DescriptorType::STORAGE_IMAGE
                    && fragment.own_pixel_textures.contains(&slot.index);
                texture = Some(bound);
            }
            let Some(texture) = texture.filter(|t| own && t.attached_in(&self.layout)) else {
                continue;
            };
            let (w, h) = (texture.width() as u32, texture.height() as u32);
            if w == 0 || h == 0 {
                continue;
            }
            let lo = |p: u32, side: u32| p.saturating_sub(1).min(side - 1);
            let hi = |p: u32, side: u32| (p + 1).min(side);
            used.loads = Some([lo(x0, w), lo(y0, h), hi(x1, w), hi(y1, h)]);
            bounded = true;
        }
        bounded
    }

    /// Whether a draw of `plan` samples, at a slot of its tables, a texture that presents a color
    /// attachment of the pass (`texture/attached_reads.rs`): a sampled image or combined image
    /// sampler binding, not a storage one.
    fn samples_attachment(&self, plan: &BindingPlan) -> bool {
        plan.descriptors.iter().any(|write| {
            let slot = match &write.source {
                WriteSource::Texture { slot, .. } => slot,
                WriteSource::CombinedImageSampler {
                    image: ImageSource::Texture { slot, .. },
                    ..
                } => slot,
                _ => return false,
            };
            let table = match slot.stage {
                Stage::Vertex => &self.vertex,
                Stage::Fragment => &self.fragment,
                Stage::Object => &self.object,
                Stage::Mesh => &self.mesh,
                _ => return false,
            };
            write.ty != metal2vulkan::vk::DescriptorType::STORAGE_IMAGE
                && table
                    .textures
                    .get(&slot.index)
                    .is_some_and(|texture| texture.attached_in(&self.layout))
        })
    }

    /// Whether the pass renders a color over a buffer's bytes through a copied stand-in, or will
    /// as it ends: one stands in already (rendered apart, without render target usage, or
    /// fetched through a scratch), or the pass has a bound linear color and an instance renders
    /// a color through a stand-in of another format (`linear_attachment::settle` puts the bound
    /// colors apart as the pass ends). A render-target barrier before such a draw stays inside its
    /// instance: a later draw's read of a color `settle` puts apart misses the draws before the
    /// barrier.
    fn reloads_at_barriers(&self) -> bool {
        let device = self.core.device();
        let over_bytes = |target: &render_pass::Target| target.texture.aliases_buffer();
        let colors = self.layout.colors.iter().flatten();
        let mut bound = false;
        for color in colors {
            if let Some(stand_in) = &color.stand_in {
                if stand_in.copies() && over_bytes(&stand_in.own) {
                    return true;
                }
            } else {
                bound |= linear_attachment::bound(device, &color.target).unwrap_or(false);
            }
        }
        bound
            && self
                .earlier
                .iter()
                .chain([&self.current])
                .any(|instance| instance.reformats.iter().any(|r| r.copies))
    }

    /// End a pass whose last command is a tile dispatch with one more instance, of no draws,
    /// which runs the pass's store actions after the dispatch (`tile_dispatch.rs`).
    fn instance_after_dispatch(&mut self) -> Result<(), Error> {
        let (mut next, stood) = self.next_instance(
            (&[], None),
            (false, false),
            ViewSetup::default(),
            false,
            None,
            &[],
            &[],
        )?;
        let continuation = match &next {
            Next::New { split: true, .. } => {
                let device = self.core.device().clone();
                self.visibility.continuation(&device)?
            }
            Next::New { .. } | Next::Current => None,
        };
        self.begin_new_draws(&mut next)?;
        self.install(next, stood, continuation, Split::TileDispatch);
        Ok(())
    }

    /// Begin the secondary of the instance `next` makes, where it has none: every new instance
    /// records into one but one recorded inline after a split (`render/inline_splits.rs`), for
    /// which the draw that installs it begins none (`Instance::draws`). Before its install, where
    /// the caller may still be refused.
    fn begin_new_draws(&self, next: &mut Next) -> Result<(), Error> {
        if let Next::New { instance, .. } = next {
            if instance.draws == vk::CommandBuffer::null() && !instance.inline {
                instance.draws = (self.core.command_buffer()).begin_draws(&instance.compatible)?;
            }
        }
        Ok(())
    }

    /// The pass's hidden depth and stencil where a draw under the encoder's state now uses depth
    /// or the stencil the pass lacks (`EncoderState::needs_hidden`): the image
    /// `hidden_depth_stencil` makes, or what refuses it; `None` where the state needs none.
    fn needed_hidden(&self) -> Result<Option<Target>, Error> {
        let (depth, stencil) = self.layout.aspects();
        match self.state.needs_hidden(depth, stencil) {
            true => self.hidden_depth_stencil().map(Some),
            false => Ok(None),
        }
    }

    /// The image of the pass's hidden depth and stencil, for a draw whose state uses depth or
    /// the stencil the pass lacks (`attachment_stand_in::hidden`). Refused
    /// `NotImplemented(HiddenDepthStencil)` where the encoder gives the pass none: one with a
    /// depth or stencil attachment of its own, none at all, or a rasterization-rate map, and an
    /// execution's draw (whose instance its pass holds) where the execution's call gave the pass
    /// none (The call gives them where its state needs them).
    fn hidden_depth_stencil(&self) -> Result<Target, Error> {
        let colors = self.layout.colors.iter().flatten().next().is_some();
        if self.layout.depth_stencil.is_some()
            || !colors
            || self.rate_map.is_some()
            || self.replaying
        {
            return Err(Error::NotImplemented(RuntimeGap::HiddenDepthStencil));
        }
        attachment_stand_in::hidden(self.core.command_buffer(), &self.layout)
    }

    /// Announce `uncovered-color-kept` where a draw of `pipeline` keeps a pass color at no place
    /// of its record whose contents plain Metal zeroes when its compiler folds the fragment's
    /// output to a constant (`color_reformat::Uncovered::KeptUnlessFolded`).
    fn degrade_kept_colors(&self, pipeline: &DrawingState, routing: &Routing) {
        let Some(&index) = routing.uncovered.values().next() else {
            return;
        };
        let colors = pipeline.descriptor().color_attachments();
        let rule = color_reformat::uncovered(colors.iter().flatten());
        if !draw_shape::writes_colors(pipeline)
            || rule != color_reformat::Uncovered::KeptUnlessFolded
        {
            return;
        }
        self.core
            .device()
            .degrade(Degradation::UncoveredColorKept, || {
                (
                    format!(
                        "a render pipeline state drawn in a pass of color {index}, which it lacks"
                    ),
                    "a color of the state blends through a destination factor of the fragment's \
                     output, so the pass's color keeps its contents where Metal's compiler may \
                     fold that output to a constant and store zero"
                        .to_owned(),
                )
            });
    }

    /// Announce the kept-attachment notices at a draw, or an execution of render commands, of
    /// each kind whose list of the pass's colors is not empty (`tile_store::KeptAttachments`):
    /// decided once for each layout the encoder records under, as its calls change it
    /// ([`kept_attachments`](Self::kept_attachments)).
    fn degrade_kept_attachments(&mut self) {
        let device = self.core.device();
        let layout = &self.layout;
        let kept = (self.kept_attachments)
            .get_or_insert_with(|| tile_store::KeptAttachments::of(layout, device));
        debug_assert_eq!(
            *kept,
            tile_store::KeptAttachments::of(layout, device),
            "the kept colors decided for a layout the encoder changed since"
        );
        Self::degrade_unused_bits(device, &kept.unused_bits);
        Self::degrade_snorm_minimum(device, &kept.snorm_minimum);
        Self::degrade_shared_exponent(device, &kept.shared_exponent);
    }

    /// Announce `extended-range-bits-kept` at a draw in a pass that loads a color attachment of
    /// an extended-range, padded 10-bit or bfloat format (and does not discard it at the store),
    /// whatever the draw's state, naming every such attachment, `kept`
    /// (`tile_store::bits_kept`): Metal's pass, once it draws, rewrites every texel of each tile a
    /// draw reaches, the texels the draws do not write too, where the device's pass keeps those.
    fn degrade_unused_bits(device: &Device, kept: &[String]) {
        if kept.is_empty() {
            return;
        }
        device.degrade(Degradation::ExtendedRangeBitsKept, || {
            (
                format!("a draw in a render pass that loads {}", kept.join(" and ")),
                "the texels the pass's draws do not write keep their bytes: the bits their \
                 codes leave unused, a BGRA10_XR alpha code past [384, 894], a padded word's \
                 low six bits, a padded sRGB twin's code off Metal's table and a bfloat code \
                 off half's grid (a NaN other than 0x7fc0), each of which Metal's pass rewrites \
                 in each tile a draw reaches"
                    .to_owned(),
            )
        });
    }

    /// Announce `snorm-minimum-kept` at a draw in a pass that loads a single-sampled color
    /// attachment of a signed normalized format (and does not discard it at the store), whatever
    /// the draw's state, naming every such attachment, `kept`
    /// (`tile_store::snorm_minimum_kept`): Metal's pass, once it draws, writes each tile a
    /// draw reaches back through the channels' values, so a minimum code (-128, or -32768) the
    /// draws do not write is stored as the code above it, where such an attachment keeps it. Not
    /// on a driver whose pass stores it so too (`DriverFact::TileStoreIsMetals`: MoltenVK's pass
    /// is a Metal pass).
    fn degrade_snorm_minimum(device: &Device, kept: &[String]) {
        if kept.is_empty() {
            return;
        }
        device.degrade(Degradation::SnormMinimumKept, || {
            (
                format!("a draw in a render pass that loads {}", kept.join(" and ")),
                "the texels the pass's draws do not write keep a channel's minimum code (-128, \
                 or -32768), which Metal's pass stores as the code above it (-127, -32767) in \
                 each tile a draw reaches"
                    .to_owned(),
            )
        });
    }

    /// Announce `shared-exponent-kept` at a draw in a pass that loads a single-sampled
    /// `RGB9E5Float` color attachment (and does not discard it at the store), whatever the draw's
    /// state, naming every such attachment, `kept` (`tile_store::shared_exponent_kept`):
    /// Metal's pass, once it draws, writes each tile a draw reaches back through its texels'
    /// values, so a word the draws do not write whose largest mantissa is below 256 under an
    /// exponent above 0 is stored normalized, where such an attachment keeps it. Not on a driver
    /// whose pass stores it so too (`DriverFact::TileStoreIsMetals`).
    fn degrade_shared_exponent(device: &Device, kept: &[String]) {
        if kept.is_empty() {
            return;
        }
        device.degrade(Degradation::SharedExponentKept, || {
            (
                format!("a draw in a render pass that loads {}", kept.join(" and ")),
                "the texels the pass's draws do not write keep their words, where Metal's \
                 pass normalizes a word whose largest mantissa is below 256 (its exponent \
                 above 0) in each tile a draw reaches"
                    .to_owned(),
            )
        });
    }

    /// The pass's color attachments of an extended-range format (`BGR10_XR`, `BGRA10_XR`,
    /// `RGB10A8_2P_XR10` and their sRGB twins), by index, with their formats. Not the padded and
    /// bfloat forms, whose tile memory is unmeasured.
    fn extended_colors(&self) -> BTreeMap<u32, PixelFormat> {
        (0u32..)
            .zip(&self.layout.colors)
            .filter_map(|(index, c)| {
                let format = c.as_ref()?.target.texture.pixel_format();
                ExtendedRange::of(format)
                    .filter(|form| !form.padded() && !form.bfloat())
                    .map(|_| (index, format))
            })
            .collect()
    }

    /// Announce `extended-range-blends-stored-codes` at a draw whose state blends into, or
    /// reads (a fetch, or the interlocked storage form an extended-range blend takes), an
    /// extended-range color of the pass an earlier draw or execution of the pass may have
    /// written, or whose clear Metal's tile memory holds between two codes (`cleared_off_codes`),
    /// naming those colors; then note the extended-range colors the draw may write. Metal's tile
    /// memory holds such a color as `half` through the pass and rounds it to its codes once, at the
    /// store (on an Apple M4, and the same probes on an Apple M3 Max), where the
    /// runtime stores the codes at each draw. Not a color whose `half` shadow the draw held
    /// (`render/extended_shadow.rs`), which reads what Metal's tile memory holds.
    fn degrade_extended_range_blends(&mut self) {
        let extended = self.extended_colors();
        if extended.is_empty() {
            return;
        }
        let (seen, written, draw) = {
            let Some(RenderStateKind::Graphics(state)) =
                self.pipeline.as_ref().map(RenderPipelineState::kind)
            else {
                return;
            };
            let graphics = &state.base().graphics;
            let reads = |index: u32| extended_shadow::reads_color(graphics, index);
            let writes = |index: u32| extended_shadow::writes_color(graphics, index);
            let seen: Vec<String> = extended
                .iter()
                .filter(|&(&index, _)| {
                    reads(index)
                        && self.extended_written.contains(&index)
                        && !self.extended.held(index)
                })
                .map(|(index, format)| format!("color {index}, {format:?}"))
                .collect();
            let written: Vec<u32> = extended
                .keys()
                .copied()
                .filter(|&index| writes(index))
                .collect();
            // The notice names the draw: its fragment function and its first such color's blend.
            let first = extended.keys().copied().find(|&index| {
                reads(index) && self.extended_written.contains(&index) && !self.extended.held(index)
            });
            let draw = first.map(|index| state.descriptor().draw_at(index));
            (seen, written, draw)
        };
        self.extended_written.extend(written);
        if seen.is_empty() {
            return;
        }
        self.core
            .device()
            .degrade(Degradation::ExtendedRangeBlendsStoredCodes, || {
                (
                    format!(
                        "{} that blends into or reads {}, which an earlier draw of its render \
                         pass wrote or its clear left between two codes",
                        draw.unwrap_or_else(|| "a draw".to_owned()),
                        seen.join(" and ")
                    ),
                    "it sees the codes stored at that draw or clear, where Metal's tile memory \
                     holds the color as half until the pass stores it"
                        .to_owned(),
                )
            });
    }

    /// Announce `ycbcr-blend-rounding` at a draw whose state blends into a color of the pass
    /// that is a two-plane YCbCr texture's render shadow (`metal::format::ycbcr_renders`): the
    /// device blends its floats, where Metal's tile blends halves (`ycbcr.rs`).
    fn degrade_ycbcr_blends(&mut self) {
        let Some(RenderStateKind::Graphics(state)) =
            self.pipeline.as_ref().map(RenderPipelineState::kind)
        else {
            return;
        };
        let descriptor = state.descriptor();
        let blended = (0u32..)
            .zip(&self.layout.colors)
            .find_map(|(index, color)| {
                let format = color.as_ref()?.target.texture.pixel_format();
                let attachment = descriptor
                    .color_attachments()
                    .get(index as usize)
                    .copied()
                    .flatten()?;
                (ycbcr_renders(format) && attachment.blending_enabled).then_some((index, format))
            });
        let Some((index, format)) = blended else {
            return;
        };
        let draw = descriptor.draw_at(index);
        self.core.device().degrade(Degradation::YCbCrBlendRounding, || {
            (
                format!("{draw}, blending into color {index} of its render pass, a {format:?} texture"),
                "Metal's tile blends the attachment's halves, an 8-bit format's in half \
                 arithmetic and a 10-bit one's in float rounded toward zero, where the device \
                 blends floats: a blended texel's codes can miss Metal's by one"
                    .to_owned(),
            )
        });
    }

    /// Announce `extended-range-blends-stored-codes` at an execution of render commands in a
    /// pass with an extended-range color, whose commands may blend onto each other or onto
    /// the pass's earlier draws, and note every such color as written: the commands' states are
    /// not known until the execution records.
    fn degrade_extended_range_execution(&mut self) {
        let extended = self.extended_colors();
        if extended.is_empty() {
            return;
        }
        let named: Vec<String> = extended
            .iter()
            .map(|(index, format)| format!("color {index}, {format:?}"))
            .collect();
        self.extended_written.extend(extended.keys().copied());
        self.core
            .device()
            .degrade(Degradation::ExtendedRangeBlendsStoredCodes, || {
                (
                    format!(
                        "an execution of render commands in a render pass of {}",
                        named.join(" and ")
                    ),
                    "its commands' blends and reads of those colors see the codes stored at each \
                     earlier command or draw, where Metal's tile memory holds the colors as half \
                     until the pass stores them"
                        .to_owned(),
                )
            });
    }

    /// The pass colors a draw of `pipeline`, routed by `routing`, renders in another format
    /// (`Reformat`, decided by `viewing`), by pass color index: each a view of the attachment's
    /// image, or a stand-in image, the current instance's where it renders the color so already.
    fn reformats(
        &self,
        pipeline: &DrawingState,
        routing: &Routing,
    ) -> Result<Vec<Reformat>, Error> {
        let wanted = pipeline.descriptor().color_formats();
        let mut reformats = Vec::new();
        for (&index, &other) in &routing.placed {
            let Some(wanted) = wanted.get(index as usize).copied().flatten() else {
                continue;
            };
            let color = self
                .layout
                .colors
                .get(other as usize)
                .and_then(Option::as_ref);
            let color = color.ok_or_else(|| internal("a placed color the pass lacks"))?;
            if color.target.texture.pixel_format() == wanted {
                continue;
            }
            let format = self
                .shape_inputs()
                .color_view_format(pipeline, index, wanted, color)
                .ok_or_else(|| internal("a draw of a color its pass refuses"))?;
            if format == color.target.format {
                continue;
            }
            let viewing = self
                .shape_inputs()
                .viewing(pipeline, index, color, format)?
                .ok_or_else(|| internal("a draw in a format its attachment refuses"))?;
            let index = other as usize;
            let copies = viewing == Viewing::StandIn;
            let current = self
                .current
                .reformats
                .iter()
                .find(|r| r.index == index && r.target.format == format && r.copies == copies);
            let target = match (current, viewing) {
                (Some(current), _) => current.target.clone(),
                (None, Viewing::View) => Target {
                    format,
                    ..color.target.clone()
                },
                (None, Viewing::StandIn) => attachment_stand_in::stand_in_as(
                    self.core.command_buffer(),
                    &color.target,
                    true,
                    format,
                )?,
            };
            reformats.push(Reformat {
                index,
                target,
                copies,
            });
        }
        Ok(reformats)
    }

    /// The bytes of each cell of the tile function's explicit imageblock standing alone that this
    /// pass keeps, where they are fewer than its data (`BindState::kept_imageblock_bytes`);
    /// `None` for every other tile state. Plain Metal keeps the imageblock's bytes that lie below
    /// the pass's imageblock bytes per sample ([`pass_sample_length`](Self::pass_sample_length))
    /// rounded up to 8, the imageblock lying after the colors as the state's
    /// `imageblock_sample_length` places it, and reads every other word as zero, dropping its
    /// writes, a dispatch's own included (Metal probes on two M4s: a 16-byte imageblock after an
    /// `R32Uint` color, at byte 16, kept bytes 0 to 7 in a pass of 20 bytes and none in a pass of
    /// the color's 4).
    fn kept_imageblock_bytes(&self, state: &TileState) -> Result<Option<u32>, Error> {
        let Some((data_size, _)) = state.kernel().tile_imageblock() else {
            return Ok(None);
        };
        let pipeline = state
            .imageblock_sample_length()
            .ok_or_else(|| internal("a tile state's attachment without a texel size"))?;
        let at = pipeline
            .checked_sub(u64::from(data_size))
            .ok_or_else(|| internal("a tile state's imageblock before its colors"))?;
        let pass = self
            .shape_inputs()
            .pass_sample_length()?
            .next_multiple_of(8);
        let kept = pass.saturating_sub(at).min(u64::from(data_size));
        // Below `data_size`, so a `u32`.
        Ok(u32::try_from(kept).ok().filter(|&kept| kept < data_size))
    }

    /// A draw of primitive type 9 (`DrawPrimitiveType::Nine`), by `call`, `empty` where it has
    /// no vertices, indices or instances: it records nothing, and fails the command buffer as
    /// `failure` says, as Metal's GPU does: a hang or a fault fails it from the encoder on and
    /// holds it (`command_buffer/hang.rs`). Checked as a draw of a public
    /// type is (a pipeline state set, a draw kind it takes); under a tessellation pipeline it
    /// draws nothing, as a public type's draw does there.
    fn draw_nine(
        &mut self,
        call: &'static str,
        empty: bool,
        failure: Option<NineFailure>,
    ) -> Result<(), Error> {
        let stand_in = Assembly::Primitives(PrimitiveType::Point);
        let drawn = drawn_by(call, self.pipeline.as_ref(), stand_in, empty)?;
        match (drawn, failure) {
            (Some(_), Some(NineFailure::Hang)) => self.core.hang(),
            (Some(_), Some(NineFailure::Fault(error))) => self.core.fault(error),
            _ => {}
        }
        self.draw_calls += 1;
        self.called = true;
        Ok(())
    }

    /// Plan, bind, and record `draw` of `assembly` (the module's steps), with its index and
    /// indirect buffers and offsets; `empty` when it draws no vertex, index, patch, instance or
    /// mesh threadgroup, which records nothing once the checks plain Metal keeps for an empty
    /// draw passed (the module docs). Every texture the encoder's heaps hold is resident.
    fn draw(
        &mut self,
        call: &'static str,
        assembly: Assembly,
        draw: DrawCall,
        buffers: DrawBuffers<'_>,
        empty: bool,
    ) -> Result<(), Error> {
        self.sync_heaps();
        // And whether it rasterizes, without which its primitives reach no tile (`no_raster_t11`).
        // And whether it chooses its viewport.
        let (reads, rasterizes, selects) =
            match self.pipeline.as_ref().map(RenderPipelineState::kind) {
                Some(RenderStateKind::Graphics(state)) => (
                    state.texture_handle_stages(),
                    !state.base().graphics.rasterizer_discard,
                    state.base().graphics.writes_viewport_index,
                ),
                _ => (RenderStages::empty(), true, true),
            };
        // A pass drawn in color groups draws it in each later group the first group drew it in,
        // where it writes there (`render/color_groups.rs`); a skipped draw is counted as drawn.
        let turn = self.group_turn();
        match turn {
            color_groups::Turn::Later(false) => return Err(color_groups::gap()),
            color_groups::Turn::Later(true) if self.group_skips() => self.group_skipped(),
            _ => {
                let drawn = self.draw_resident(call, assembly, draw, buffers, empty);
                self.group_outcome(turn, drawn.is_ok());
                drawn?;
            }
        }
        // Every draw, one of nothing too; an execution's commands announce at the
        // execution (`executions.rs`).
        if !self.replaying {
            self.degrade_kept_attachments();
            self.degrade_extended_range_blends();
            self.degrade_ycbcr_blends();
        }
        self.draw_calls += 1;
        self.command_encoded(reads);
        self.called = true;
        self.drew |= !empty && rasterizes;
        // A draw that records a command hands the device's pass its depth/stencil state.
        self.drew_depth_writes |= !empty && self.state.depth_writes();
        // The pixels it may write: its viewports' within its scissors, every viewport where it
        // chooses one or an amplification offsets it, and anywhere through a rate map.
        if !empty && rasterizes {
            match self.rate_map.is_some() {
                true => {
                    self.drawn.everywhere();
                    self.coverage.everywhere();
                }
                false => {
                    let selects = selects || self.amplification.offsets_viewports();
                    let mut scissors = [vk::Rect2D::default(); MAX_VIEWPORTS];
                    let rects = self.state.vk_scissors_in(selects, &mut scissors);
                    self.drawn.add(rects);
                    // And its instance's, the one it recorded into.
                    self.current.drawn.add(rects);
                    // What it writes of each color, and blends, inside its Vulkan scissors
                    // (`linear_attachment::Coverage`).
                    // Of each pass color its draws store into: a state color where its routing
                    // places it, and the pass colors its variant zeroes.
                    match self.pipeline.as_ref().map(RenderPipelineState::kind) {
                        Some(RenderStateKind::Graphics(state)) => {
                            let colors = &state.base().graphics.color_attachments;
                            let routing = self
                                .shape_inputs()
                                .routing(state)
                                .map_err(|_| internal("a draw of a state its pass refuses"))?;
                            let zeroed = !self
                                .shape_inputs()
                                .routed_colors(state, &routing)
                                .zeroed
                                .is_empty();
                            let writes = linear_attachment::writes(colors);
                            let writes = linear_attachment::pass_writes(&writes, &routing, zeroed);
                            self.coverage.add(&writes, rects)
                        }
                        _ => self.coverage.everywhere(),
                    }
                }
            }
        }
        Ok(())
    }

    /// A draw the encoder records nothing for (`EncoderCore::skips`): what its tables bind is
    /// held and listed as a recorded draw's is.
    fn skipped_draw(&mut self) {
        self.vertex.retain_exclusive(&mut self.core);
        self.fragment.retain_exclusive(&mut self.core);
        self.object.retain_exclusive(&mut self.core);
        self.mesh.retain_exclusive(&mut self.core);
    }

    /// Whether a function of `interface` statically uses a texture slot of its stage's table
    /// that holds a memoryless texture (`ArgumentTable::uses_memoryless`).
    fn uses_memoryless(&self, interface: &PipelineInterface) -> bool {
        [
            (Stage::Vertex, &self.vertex),
            (Stage::Fragment, &self.fragment),
            (Stage::Object, &self.object),
            (Stage::Mesh, &self.mesh),
        ]
        .into_iter()
        .any(|(stage, table)| table.uses_memoryless(interface, stage))
    }

    /// [`draw`](Self::draw) with the heaps' textures resident.
    fn draw_resident(
        &mut self,
        call: &'static str,
        assembly: Assembly,
        mut draw: DrawCall,
        buffers: DrawBuffers<'_>,
        empty: bool,
    ) -> Result<(), Error> {
        let Some(pipeline) = drawn_by(call, self.pipeline.as_ref(), assembly, empty)? else {
            return Ok(());
        };
        let tessellation = crate::tessellation::patches(&pipeline.base().graphics);
        let tessellated = tessellation.is_some();
        let meshes = pipeline.descriptor().vertex().is_none();
        if let (
            Some(t),
            DrawCall::Patches { control_points, .. }
            | DrawCall::PatchesIndirect { control_points, .. },
        ) = (tessellation, &mut draw)
        {
            // Metal draws with the control points the function states, whatever the draw names
            // (Metal probe cases `control_points_4` and `_0` on an M4; Metal's validation stops
            // it).
            if t.control_points != 0 {
                *control_points = t.control_points;
            }
            // An indexed patch draw takes a control-point index type, a plain one none: every
            // patch draw gives a pipeline with one zero indices where it has none, and one
            // without none.
            let indexed = buffers.control_point_indices.is_some();
            let typed = t.control_point_index_type
                != metal2vulkan::metal::TessellationControlPointIndexType::None;
            if indexed != typed {
                return Err(internal(
                    "a patch draw whose control-point indices its pipeline does not take",
                ));
            }
            if self.factors.buffer.is_none() {
                return Err(draw_problem(call, DrawProblem::NoTessellationFactorBuffer));
            }
        }
        // A draw whose functions statically use a texture slot that holds a memoryless texture,
        // an empty one too, fails the command buffer as plain Metal's does, from its pass on
        // (`EncoderCore::fail_memoryless_use`), and records nothing. A draw recorded again into
        // another instance was checked at its call. An execution's command is the runtime's gap:
        // Metal's answer there is not measured.
        if !self.rerouting && self.uses_memoryless(&pipeline.base().interface) {
            if self.replaying {
                return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
            }
            self.core.fail_memoryless_use();
            return Ok(());
        }
        // An empty draw draws nothing once the checks plain Metal keeps for one passed (a
        // pipeline state, not a tile one, of the draw's kind, and a patch draw's factor buffer:
        // without each, plain Metal stops the process at an empty draw too).
        // Plain Metal draws nothing whatever the rest of the encoder's state (unbound buffers,
        // vertex buffers and textures, an amplification count above the state's most), which
        // its validation stops, so nothing past here is checked.
        if empty {
            return Ok(());
        }
        // A pipeline that scales its factors culls every patch until a scale is set, and while
        // the scale is 0 or negative as a `half` (`tessellation.rs`): the draw records nothing.
        let scaled = pipeline
            .descriptor()
            .vertex()
            .and_then(|d| d.tessellation)
            .is_some_and(|t| t.factor_scale_enabled);
        let culled = tessellated && scaled && self.factors.culls();
        // An amplification count of 0 draws nothing, as plain Metal draws it (Metal probe case
        // `amp0` on an M4; Metal's validation stops it).
        if self.amplification.draws_nothing() {
            return Ok(());
        }
        // A draw that blends into or reads an extended-range color an earlier draw of the pass
        // wrote, or whose clear its codes cannot hold, starts the color's `half` shadow where it
        // can (`render/extended_shadow.rs`).
        self.start_extended_shadows(&pipeline)?;
        // What the draw settles from the encoder's state, as an execution's precompile at commit
        // settles it (`draw_settings`).
        // An indexed strip the encoder's call restarts at another index than Vulkan's, and a draw
        // of one of Metal's private types, is drawn as a list, of its class's list topology
        // (`render/restart.rs`).
        // A draw under the last provoking vertex is drawn as a list on a device without it
        // (`render/provoking.rs`).
        let provoking = self.provoking(call, &pipeline, assembly, &draw);
        let provoking_forms = provoking.forms(assembly, &pipeline);
        let restarted = self.restarted(assembly, &draw, provoking);
        let private = match (assembly, restarted) {
            (Assembly::Private(program), _) => PrivateForm::of(&pipeline, program, &draw),
            (_, Some(_)) => PrivateForm::listed(&draw),
            (_, None) => PrivateForm::default(),
        };
        let assembly = restarted.map_or(assembly, |r| r.assembly());
        // Its settings, and the memo entry its draw state and pipeline are found and kept in
        // (`render/draw_shape.rs`).
        let (settings, hidden, shape_token) = self.shaped_settings(call, &pipeline, assembly)?;
        // A direct indexed draw of a pipeline that writes the primitive ID runs its de-indexed
        // variant; a restarted strip's list keeps the strip's indices, which it does not run.
        let primitive_id_indices = match restarted {
            None => primitive_id_indices(&pipeline, &draw),
            Some(_) => None,
        };
        let (topology, primitive_restart) = (settings.topology, settings.primitive_restart);
        let unsized_points = settings.unsized_points;
        let (native, compensation) = (settings.native, settings.compensation);
        let hidden_depth = settings.hidden;
        let (_, stencil) = self.layout.aspects();
        let max = pipeline.base().graphics.max_vertex_amplification_count;
        let function_tables = pipeline.function_tables();
        let mut unlisted = None;
        let reads_tile_memory = pipeline.reads_tile_memory();
        // Each buffer of the vertex table whose bytes a fragment stage of the pass wrote, read
        // from a copy taken before the pass (`render/pass_start_reads.rs`).
        // The object and mesh tables' too, and an execution's commands', whose copies record
        // into the recording that replays them, before the pass (`render/executions.rs`).
        // The device's sampler states, less those at IDs no set-1 sampler array of the variant
        // holds (`heap_sets::fit_samplers`): taken where the draw needs them
        // (`sampler::ArgumentSamplers`).
        let mut argument_samplers = ArgumentSamplers::untaken();
        let pass_start = self.pass_start.tables(
            &mut self.core,
            [&self.vertex, &self.object, &self.mesh],
            &pipeline.base().interface,
        )?;
        self.degrade_kept_colors(&pipeline, &settings.routing);
        // The pass colors the draw renders through views of other formats, in an instance of
        // its own (`next_instance`). Refused (`NotImplemented(IndirectRenderCommands)`) for an
        // execution's command, whose runs' framebuffers are the pass's.
        let reformats = self.reformats(&pipeline, &settings.routing)?;
        if self.replaying && !reformats.is_empty() {
            return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
        }
        // The encoder's kept bind state, brought to the draw's (`live_bind.rs`), held for the
        // draw: no draw of the encoder runs inside another past here.
        let live = Arc::clone(&self.live);
        let mut held = lock(&live);
        let live = &mut *held;
        // A draw that reads its pass's texture after a barrier inside its instance reads it
        // through a snapshot where it can (`render/snapshot.rs`): asked of the plan, kept where
        // the plan reads so, else `None` and the split.
        let mut snapshot = self.snapshot_wanted(&pipeline, &reformats);
        // Its runtime samplers' address modes as the textures they sample map them: what it
        // settles under, and binds with.
        let sampler_maps = self.sampler_maps(&pipeline, &pass_start)?;
        let Settled {
            shape,
            variant,
            plan,
        } = self.settle_draw(
            live,
            (
                &pipeline,
                &settings,
                primitive_id_indices,
                private,
                provoking_forms,
            ),
            (&pass_start, &sampler_maps),
            (&mut argument_samplers, &mut unlisted),
            (&function_tables, &mut snapshot),
        )?;
        let planes = |form: &ExtendedRange| matches!(form, ExtendedRange::Planes { .. });
        if plan.variant.extended_range.values().any(planes) {
            self.core.device().shared().counts.two_plane_draw();
        }
        match provoking {
            provoking::Provoking::AsIs => {}
            provoking::Provoking::Native => self
                .core
                .device()
                .shared()
                .counts
                .provoking_last_pipeline_draw(),
            provoking::Provoking::Listed(_) | provoking::Provoking::LastCorners => {
                self.core.device().shared().counts.provoking_listed_draw()
            }
        }
        self.give_up_extended_shadows(&pipeline, &live.state);
        // The draw's primitive type: a vertex function that writes the primitive ID divides by its
        // vertices per primitive (`PushField::VerticesPerPrimitive`), and an indexed strip of it
        // does not run de-indexed.
        if let Assembly::Primitives(primitive) = assembly {
            live.state.primitive_type = Some(primitive);
        }
        let device = self.core.device();
        let stages = &variant.pipeline().stages;
        if device.argument_pixel_samplers_noted(stages).is_some() {
            device.degrade_argument_pixel_samplers(stages, argument_samplers.take(device));
        }
        let zeroes_colors = !settings.routed_colors.zeroed.is_empty();
        let DrawSettings {
            failing_state,
            amplification,
            routing,
            ..
        } = &settings;
        let graphics = &variant.pipeline().graphics;
        // A fragment function that runs the draw's depth test itself computes and clamps its
        // depth by the draw's own Vulkan depth range, read as push data.
        if graphics.shader_tests.is_some() {
            let compensated = compensation.is_some_and(|c| c.depth.is_some());
            live.state.depth_range = self.state.shader_tests_range(
                graphics.writes_viewport_index,
                topology,
                self.core.device().compile_device().profile(),
                (native, compensated),
                graphics.writes_depth,
            )?;
        }
        let interface = &variant.pipeline().interface;
        // A pulled attribute's 32-bit displacement past 4 GiB of its buffer
        // (`render/vertex_reach.rs`).
        self.check_vertex_reach(&variant.pipeline().vertex_input.pulled, &draw)?;
        self.vertex
            .check_function_tables(call, Stage::Vertex, interface)?;
        self.fragment
            .check_function_tables(call, Stage::Fragment, interface)?;
        // The layers the draw renders (`render_target::viewed`): views of an amplified draw, or
        // one layer its view mapping moves it to.
        let views = plan.views;
        if views.view_mask.count_ones() > 1 && self.visibility.active() {
            return Err(internal(
                "a draw amplified as views under a visibility query",
            ));
        }
        // A draw counted where the driver counts each sample once per color its fragment shader
        // writes runs, announced (`visibility.rs`).
        let colors = graphics.color_outputs.count_ones();
        if colors > 1 && self.visibility.counting() {
            let defect = RuntimeDriverDefect::OcclusionCountPerColorOutput;
            let device = self.core.device();
            if device.properties().driver_defects.contains(&defect) {
                device.degrade(Degradation::VisibilityCountPerColor, || {
                    (
                        format!(
                            "{call}: a draw whose fragment function writes {colors} color \
                             attachments, under a Counting visibility result"
                        ),
                        "the driver's precise occlusion query counts each sample once per color \
                         output its fragment shader writes"
                            .to_owned(),
                    )
                });
            }
        }
        // The pass's fragment imageblock storage: one cell's and alignment's, and no tile
        // dispatch's aliased layout past the attachments beside it (`fragment_imageblock.rs`).
        let imageblock_use = graphics.fragment_imageblock;
        if let Some(used) = &imageblock_use {
            if self.tile_imageblock {
                return Err(Error::NotImplemented(
                    RuntimeGap::TileAndFragmentImageblocks,
                ));
            }
            if self
                .imageblock
                .as_ref()
                .is_some_and(|held| !held.takes(used))
            {
                return Err(Error::NotImplemented(RuntimeGap::ImageblockMasters));
            }
        }
        // Over the last draw's commands, in their storage (`LiveBind::commands`).
        // A private type's list in the private list form is drawn without indices (`listed_call`).
        let bound = match (private.private_list, restarted) {
            (true, Some(restarted)) => listed_call(restarted.program, &draw),
            _ => draw,
        };
        plan.instantiate_draw_into(&live.state, &bound, &mut live.commands)
            .map_err(Error::Bind)?;
        // A mesh draw's own key, its residual form where the plan's is (`RenderVariant::plan`).
        if let Some(own) = live.commands.specialization.as_mut() {
            if !super::pipeline::reads_views_at_bind(self.core.device()) {
                if let Some(residual) = variant.residual_form(own) {
                    *own = residual;
                }
            }
        }
        let commands = &live.commands;
        // The phase of a draw read through snapshots, which its two draws set
        // (`render/snapshot.rs`).
        let phase_hole = match &snapshot {
            Some(_) => Some(
                snapshot::phase_hole(&commands.push_holes)
                    .ok_or_else(|| internal("a snapshot variant without its phase"))?,
            ),
            None => None,
        };
        self.reaches_unnamed |= !commands.root_tables.is_empty() || reads_handles(&plan);
        // A mesh draw the binder launches no threadgroup of (as plain Metal draws nothing).
        let empty = empty
            || matches!(
                commands.draw,
                metal2vulkan::vk::DrawCommand::DrawMeshTasks {
                    group_count_x,
                    group_count_y,
                    group_count_z,
                } if group_count_x == 0 || group_count_y == 0 || group_count_z == 0
            )
            || commands.mesh_dispatch.as_ref().is_some_and(|dispatch| {
                matches!(dispatch.mesh, MeshGroups::Direct(groups) if groups.contains(&0))
            });
        // A direct draw of an object grid Metal's GPU discards draws nothing and fails the command
        // buffer from its encoder on, as Metal's (`render/mesh_arguments.rs`); a folded object
        // function's direct draw of a grid past the device's pieces fails it too.
        if commands.discards_command_buffer && !culled {
            self.core.discard();
        }
        if commands.object_fold.is_some_and(|fold| fold.past) && !culled {
            self.core.command_buffer().state().mesh_grid_past_limits();
        }
        // Set 1, the heap: its writes at the elements of their resource IDs, and the device's
        // texture and sampler tables at the handle tables.
        let compiled = variant.pipeline();
        let heap = heap_sets::heap_elements(&compiled.layout, &plan, &live.state)?;
        let texture_table = heap_sets::texture_table(self.core.device(), &compiled.layout)?;
        let sampler_table =
            heap_sets::sampler_table(self.core.device(), &compiled.layout, &compiled.dependencies)?;
        // The arguments are GPU memory an earlier command may write; the exact paths are GPU
        // passes over them. The index bounds pass is one (`index_bounds.rs`). An indexed indirect
        // draw takes it where the binder asks for its arguments on the CPU only for its index
        // bound, and where the binder leaves it to the device's range but the device's driver is
        // not measured to read zero past it. A restarted strip's index rewrite pass reads its
        // indices and arguments itself, contained, and its draw fetches only the list it writes: it
        // takes the index bounds pass's place, and keeps an indirect draw on the GPU where the
        // binder asks for its arguments on the CPU only for the index bound. Its reads are ordered
        // as the index bounds pass's are, below. A draw whose index, indirect, patch index or
        // control-point index buffer is detached records nothing, its steps before the pass
        // included: its submission fails.
        let drawn = [
            buffers.index.as_ref().map(|source| source.metal().0),
            buffers.indirect.as_ref().map(|source| source.metal().0),
            buffers.patch_indices.map(|(buffer, _)| buffer),
            buffers.control_point_indices.map(|(buffer, _)| buffer),
        ];
        if self.core.skips(
            drawn
                .into_iter()
                .flatten()
                .map(|buffer| buffer.identity().key),
        ) {
            self.skipped_draw();
            return Ok(());
        }
        // A direct draw of a private type without indices draws the list the CPU writes; a direct
        // indexed draw of a program that never restarts, reading its indices in place, the list
        // a run of the index gather pass writes; every other draw drawn as a list, the list the
        // index rewrite pass writes.
        let (rewrite, listed) = match restarted {
            None => (None, None),
            Some(restarted) => match (&draw, &buffers.index, &buffers.indirect) {
                (
                    &DrawCall::Primitives {
                        vertex_start,
                        vertex_count,
                        ..
                    },
                    None,
                    None,
                ) => {
                    let list = WrittenList::new(
                        &mut self.core,
                        ((restarted.program, restarted.order), restarted.gap.gap()),
                        (vertex_start, vertex_count),
                        private.private_list,
                    )?;
                    (None, Some(Listed::Written(list)))
                }
                (
                    &DrawCall::Indexed {
                        index_count,
                        instance_count,
                        base_vertex,
                        base_instance,
                        ..
                    },
                    Some(index),
                    None,
                ) if restarted.program.gathers()
                    && index.in_place()
                    && !self.replaying
                    && !private.private_list =>
                {
                    let size = restarted.index_size.unwrap_or(4);
                    let arguments = index_rewrite::Arguments::Direct {
                        count: index_count,
                        instances: instance_count,
                        base_vertex,
                        base_instance,
                    };
                    let reads = index_rewrite::reads(Some(index), &arguments, size);
                    if self.unordered.written(&reads) {
                        return Err(Error::NotImplemented(restarted.gap.gap()));
                    }
                    // Indices an earlier draw or tile dispatch of the encoder wrote behind a
                    // memory barrier are read before the draw's render pass instance: the index
                    // rewrite pass's step there writes its list.
                    let gathered = match self.core.uses().written(&reads) {
                        true => None,
                        false => self.core.gather(
                            index_rewrite::GatherRequest {
                                program: restarted.program,
                                order: restarted.order,
                                index,
                                count: index_count,
                                wide: size == 4,
                            },
                            &reads,
                        )?,
                    };
                    match gathered {
                        Some(list) => (None, Some(Listed::Gathered(list))),
                        None => {
                            let pass = RewritePass::new(
                                &mut self.core,
                                (
                                    (restarted.program, restarted.restart, restarted.order),
                                    restarted.gap.gap(),
                                    false,
                                ),
                                Some(index),
                                &arguments,
                                size,
                                reads,
                            )?;
                            (Some(pass), None)
                        }
                    }
                }
                (draw, index, indirect) => {
                    if commands.indirect_needs_arguments && !commands.index_bounds_pass {
                        return Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments));
                    }
                    let arguments = match (draw, index, indirect) {
                        (
                            &DrawCall::Indexed {
                                index_count,
                                instance_count,
                                base_vertex,
                                base_instance,
                                ..
                            },
                            Some(_),
                            None,
                        ) => index_rewrite::Arguments::Direct {
                            count: index_count,
                            instances: instance_count,
                            base_vertex,
                            base_instance,
                        },
                        (
                            DrawCall::Indirect {
                                indexed: Some(_), ..
                            },
                            Some(_),
                            Some(indirect),
                        ) => index_rewrite::Arguments::Indirect(indirect),
                        (DrawCall::Indirect { indexed: None, .. }, None, Some(indirect)) => {
                            index_rewrite::Arguments::IndirectVertices(indirect)
                        }
                        _ => return Err(internal("a list of another draw")),
                    };
                    let index = index.as_ref();
                    let size = restarted.index_size.unwrap_or(4);
                    let reads = index_rewrite::reads(index, &arguments, size);
                    let in_place = index_rewrite::in_place(index, &arguments, size);
                    if self.unordered.written(&in_place) {
                        return Err(Error::NotImplemented(restarted.gap.gap()));
                    }
                    // A direct indexed draw (of a program the gather pass does not take) joins a
                    // run of the index rewrite pass, one dispatch for the run's draws, each in a
                    // workgroup of its own; an indirect draw of a program whose list follows from
                    // its positions joins an indirect run of the index gather pass, which plans
                    // its list on the GPU. Either reads its arguments and indices in place; where
                    // an earlier draw or tile dispatch of the encoder wrote them behind a memory
                    // barrier, the index rewrite pass's step before the draw's render pass
                    // instance writes the list.
                    let gathers = match arguments {
                        index_rewrite::Arguments::Indirect(_) => restarted.program.gathers(),
                        index_rewrite::Arguments::IndirectVertices(_) => {
                            restarted.program.gathers_without_indices()
                        }
                        index_rewrite::Arguments::Direct { .. } => false,
                    };
                    let joins = !self.replaying
                        && !private.private_list
                        && !self.core.uses().written(&reads);
                    let commanded = match (arguments, index, indirect) {
                        (
                            index_rewrite::Arguments::Direct {
                                count,
                                instances,
                                base_vertex,
                                base_instance,
                            },
                            Some(index),
                            None,
                        ) if joins && index.in_place() => self.core.rewrite_run(
                            index_rewrite::RewriteRunRequest {
                                program: restarted.program,
                                restart: restarted.restart,
                                order: restarted.order,
                                index,
                                arguments: (count, instances, base_vertex, base_instance),
                                wide: size == 4,
                            },
                            &reads,
                        )?,
                        (_, index, Some(source))
                            if gathers
                                && joins
                                && source.in_place()
                                && index.is_none_or(DrawSource::in_place) =>
                        {
                            let list = self.core.gather_indirect(
                                index_rewrite::IndirectGatherRequest {
                                    program: restarted.program,
                                    order: restarted.order,
                                    arguments: source,
                                    index,
                                    wide: size == 4,
                                },
                                &reads,
                            )?;
                            if let (Some(_), Some(index)) = (&list, index) {
                                index_rewrite::announce_past_length(self.core.device(), index);
                            }
                            list
                        }
                        _ => None,
                    };
                    match commanded {
                        Some(list) => (None, Some(Listed::Commanded(list))),
                        None => {
                            let pass = RewritePass::new(
                                &mut self.core,
                                (
                                    (restarted.program, restarted.restart, restarted.order),
                                    restarted.gap.gap(),
                                    private.private_list,
                                ),
                                index,
                                &arguments,
                                size,
                                reads,
                            )?;
                            (Some(pass), None)
                        }
                    }
                }
            },
        };
        // A rewritten draw's list, and an indirect gathered draw's, holds its indices bounded.
        let pass = match (commands.indirect_needs_arguments, &buffers) {
            _ if rewrite.is_some() || matches!(listed, Some(Listed::Commanded(_))) => None,
            (
                needs,
                DrawBuffers {
                    index: Some(index),
                    indirect: Some(indirect),
                    ..
                },
            ) if (needs && commands.index_bounds_pass)
                || (!needs && !self.robust_index_fetch()) =>
            {
                Some((index, indirect))
            }
            (false, _) => None,
            (true, _) => return Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments)),
        };
        let bounds = match pass {
            None => None,
            Some((index, indirect)) => {
                let DrawCall::Indirect {
                    indexed: Some((index_type, ..)),
                    ..
                } = draw
                else {
                    return Err(internal(
                        "an index bounds pass of a draw that is not indexed",
                    ));
                };
                let size = match index_type {
                    t if t == metal2vulkan::vk::IndexType::UINT16 => 2,
                    t if t == metal2vulkan::vk::IndexType::UINT32 => 4,
                    _ => return Err(internal("an index bounds pass of another index type")),
                };
                // The pass reads the arguments and indices before the draw's render pass
                // instance at the latest. A write of them since the last memory barrier, which
                // Metal does not order before the draw's fetch, is refused where the draw reads
                // them in place (Before any barrier, a draw's or tile dispatch's write is read
                // through a copy, `DrawSource::racing`, and after one a tile dispatch's is refused
                // before); so is any earlier write where an execution's command draws
                // (`render/executions.rs`), whose instance takes no steps before it.
                let reads = index_bounds::reads(index, indirect, size);
                let in_place = index_bounds::in_place(index, indirect, size);
                if self.unordered.written(&in_place)
                    || (self.replaying && self.core.uses().written(&reads))
                {
                    return Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments));
                }
                // A draw that reads both in place, which nothing of the encoder wrote behind a
                // memory barrier, joins a run of the pass, one plan and one fill for the run's
                // draws; any other takes its own pass, which records before the draw's render
                // pass instance where it must.
                let joined = match !self.replaying
                    && index.in_place()
                    && indirect.in_place()
                    && !self.core.uses().written(&reads)
                {
                    true => self.core.bounds_run(
                        index_bounds::BoundsRunRequest {
                            index,
                            indirect,
                            index_size: size,
                        },
                        &reads,
                    )?,
                    false => None,
                };
                Some(match joined {
                    Some(slot) => Bounded::Run(slot),
                    None => Bounded::Pass(Box::new(BoundsPass::new(
                        &mut self.core,
                        index,
                        indirect,
                        size,
                        reads,
                    )?)),
                })
            }
        };
        // The dynamic states the pipeline holds, which the draw records and its secondary's record
        // keeps (`render/bound.rs`): a pipeline that discards rasterization has no fragment
        // shader or fragment output state, so those states are none of its own (`fragment_state`).
        let device = self.core.device().clone();
        let extended = device.shared().extended_dynamic_state;
        let masks = device.shared().color_write_masks.is_some();
        let draw_dynamic =
            variant.draw_dynamic(|compiled| DrawDynamic::of(&compiled.graphics, extended, masks));
        if !draw_dynamic.recorded {
            return Err(internal("a dynamic state the encoder does not record"));
        }
        let dynamic = &draw_dynamic.states[..];
        let dynamic_strides = dynamic.contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE);
        // The instance the draw records into, and what stands in for the pass's attachments from
        // the draw on: the encoder's layout once the draw is recorded.
        let storage = self.storage_attachments(graphics)?;
        let colors_ordered =
            (device.properties().driver_facts).contains(&DriverFact::InterlockOrdersDraws);
        let (interlocked, interlocked_writes) =
            InterlockedStorage::of(graphics, &storage, colors_ordered);
        let failing = match &failing_state {
            None => None,
            Some(state) => {
                let body_writes = storage.iter().any(|s| s.writes) || imageblock_use.is_some();
                let writes = (body_writes, graphics.fragment_writes_memory);
                Some(state.failing_draw(graphics.early_fragment_tests, stencil, writes)?)
            }
        };
        let mut fetched: Vec<u32> = if self.layout.fetch_inputs {
            graphics
                .input_attachments
                .iter()
                .map(|input| input.color_index)
                .collect()
        } else {
            storage.iter().map(|s| s.color_index).collect()
        };
        fetched.sort_unstable();
        fetched.dedup();
        if self.layout.fetch_inputs {
            self.inputs_read.extend(fetched.iter().copied());
        }
        // A draw that reads and writes some of the instance's interlocked colors and writes none
        // of the others records in it, as one that reads them all would.
        if !self.layout.fetch_inputs && self.fits_storage(&fetched, graphics.color_outputs) {
            fetched.clear();
            fetched.extend_from_slice(&self.current.shape.storage);
        }
        let shadow_use = graphics.shadow_stencil;
        // A draw whose fragment function tests the depth itself, against the pass's stand-in of
        // it (`depth_stand_in.rs`); not among indirect render commands, whose runs take no
        // stand-in's copies (not built).
        let tests_use = graphics.shader_tests;
        if tests_use.is_some() && self.replaying {
            return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
        }
        // The stencil the draw tests in the shader, held in the pass's shadow of it: a returned
        // `[[stencil]]`'s (the shader-side stencil), or, for a function that runs the draw's
        // stencil test itself, the stand-in its tests use, its format and samples, with the feature
        // a refusal of its copies names.
        let stencil_use = match (&shadow_use, tests_use.and_then(|t| t.stencil)) {
            (Some(used), _) => Some((used.format, used.samples, ErrorFeature::StencilExport)),
            (None, Some(used)) => Some((used.format, used.samples, ErrorFeature::FramebufferFetch)),
            (None, None) => None,
        };
        // Vulkan orders one draw's storage writes before another's reads in one render pass
        // instance pixel by pixel alone: a draw that reaches the tile memory after another that
        // did records into a new instance (`render/tile_memory.rs`).
        let ordered = reads_tile_memory && self.current.tile_memory;
        // A draw that samples a color attachment's texture reads the texture's copy, taken before
        // its instance; after a memory barrier inside the current instance, which orders one
        // pixel alone, it records into a new instance, whose copy holds what the draws before the
        // barrier wrote (`texture/attached_reads.rs`). An execution's command reads the
        // attachment in place: it takes no copy.
        // A draw that reads through snapshots reads no attachment's texture otherwise
        // (`RenderCommandEncoder::snapshot_holds`), and stays in its instance: one recorded inline
        // has begun its render pass instance with no color in the feedback loop layout, so the
        // draw records into a new one (`render/snapshot.rs`). So does a draw that binds a texture
        // the current instance holds in that layout as a storage image.
        let attached = !self.replaying && snapshot.is_none() && self.samples_attachment(&plan);
        let after_barrier = attached && (self.current.barriers || self.barrier);
        let snapshot_split = snapshot.as_ref().is_some_and(|w| self.snapshot_apart(w));
        let storage_colors = self.storage_colors(&plan);
        let apart = snapshot_split || storage_colors & self.current.feedback.0 != 0;
        // A draw that writes a memoryless color the pass renders in place, which no instance
        // recorded inline writes: the pass's first draw that records keeps its first instance in
        // a secondary, and a later one records into a new instance after the inline one
        // (`render/inline_first.rs`).
        let deciding = matches!(self.inline, inline_first::Inline::Open { .. });
        let unheld_writes = (self.current.inline || deciding)
            && self.writes_memoryless(&pipeline, routing, zeroes_colors);
        let first_write = self.current.inline && unheld_writes;
        let (next, stood) = self.next_instance(
            (&fetched, Some(&pipeline)),
            (stencil_use.is_some(), tests_use.is_some()),
            views,
            (ordered || after_barrier || apart || first_write) && !self.replaying,
            hidden,
            &routing.locations,
            &reformats,
        )?;
        if first_write && matches!(next, Next::New { split: true, .. }) {
            device.shared().counts.inline_memoryless_write_split();
        }
        if attached {
            device.shared().counts.attachment_read_through_copy();
        }
        if after_barrier && matches!(next, Next::New { split: true, .. }) {
            device.shared().counts.attachment_read_split();
        }
        if snapshot.is_some() {
            device.shared().counts.reading_draw_snapshotted();
        }
        let inline = self.current.inline;
        if snapshot_split && inline && matches!(next, Next::New { split: true, .. }) {
            device.shared().counts.snapshot_inline_split();
        }
        // An execution's commands draw into its instance, of no fetched attachments, shader-side
        // stencil or views (`render/executions.rs`). One ordered
        // after another that reached the tile memory records into the execution's next run
        // instead, which its pass runs in a render pass instance of its own.
        // So does one whose draw needs another instance than the run's, with attachments it
        // fetches as storage or views of its own: its run is of its instance's shape.
        // One whose draw reads and writes attachments in the pixel interlock elsewhere than the
        // pass's instances render them (a stand-in, or sample planes) draws in a run of its own,
        // around which the pass brings them there and back (`render/executions.rs`), as around
        // the encoder's instances of such draws. A command that reaches the fragment imageblock
        // storage draws in the run it falls in, as the encoder's draw does in its instance
        // (`fragment_imageblock.rs`).
        // One that fetches through input attachments into a stand-in draws in a run of its own
        // too, whose framebuffer holds the stand-in in place of the attachment the pass's
        // framebuffers hold, copied in before the run and back after it (`render/executions.rs`).
        // None brings the pass's hidden depth and stencil: the execution's call brought them
        // where its state needs them, and `hidden_depth_stencil` refuses an execution's draw
        // where the call could not.
        let next = match (self.replaying, next) {
            (true, Next::New { instance, .. }) => self.run_of(*instance)?,
            (_, next) => next,
        };
        // A draw recorded again records into the rerouted instance, or fails the reroute.
        if self.rerouting && !matches!(next, Next::Current) {
            return Err(Error::NotImplemented(RuntimeGap::Render));
        }
        // So does one whose command's barrier stands (`run_barrier`).
        let next = match (self.replaying && (ordered || self.run_barrier), next) {
            (true, Next::Current) => self.next_run()?,
            (_, next) => next,
        };
        let split = matches!(next, Next::New { split: true, .. });
        let tile_memory_split = ordered && split;
        let run_barrier_split = self.run_barrier && !ordered && split;
        // The query that continues the active one past a split.
        let continuation = match &next {
            Next::New { split: true, .. } => self.visibility.continuation(&device)?,
            Next::New { .. } | Next::Current => None,
        };
        let layout = stood.as_deref().unwrap_or(&self.layout);
        let render_area = layout.extent;
        let inputs = input_attachments(layout, graphics)?;
        let storage_uses = storage_uses(layout, &storage)?;
        // The pass's shader-side stencil's shadow, made at its first such draw (Or its first draw
        // that runs the stencil test itself).
        let made = match (&self.shadow, &stencil_use) {
            (None, Some((format, samples, feature))) => {
                let ds = layout
                    .depth_stencil
                    .as_ref()
                    .ok_or_else(|| internal("a shader-side stencil without a stencil"))?;
                Some(ShadowStencil::new(
                    self.core.command_buffer(),
                    (*format, *samples),
                    feature.clone(),
                    &ds.target,
                    layout.extent,
                )?)
            }
            _ => None,
        };
        let shadow = match stencil_use {
            Some(_) => made.as_ref().or(self.shadow.as_ref()),
            None => None,
        };
        // The pass's depth stand-in, made at its first such draw; a draw whose new instance needs
        // the depth copied in where the device cannot is refused.
        let made_stand_in = match (&self.depth_stand_in, &tests_use) {
            (None, Some(used)) => {
                let ds = layout
                    .depth_stencil
                    .as_ref()
                    .ok_or_else(|| internal("a shader-side depth test without a depth"))?;
                Some(DepthStandIn::new(
                    self.core.command_buffer(),
                    &used.depth,
                    &ds.target,
                    layout.extent,
                )?)
            }
            _ => None,
        };
        let stand_in = match tests_use {
            Some(_) => made_stand_in.as_ref().or(self.depth_stand_in.as_ref()),
            None => None,
        };
        if let (Some(stand_in), Next::New { split, .. }) = (stand_in, &next) {
            let first = !split && self.earlier.is_empty();
            let load = layout
                .depth_stencil
                .as_ref()
                .and_then(|ds| ds.depth.as_ref())
                .map(|depth| depth.load);
            stand_in.check_filled(first, load)?;
        }
        // The pass's imageblock storage, made at its first such draw or tile dispatch.
        let made_imageblock = match (&self.imageblock, &imageblock_use) {
            (None, Some(used)) => Some(PassImageblock::new(
                self.core.command_buffer(),
                used,
                layout,
            )?),
            _ => None,
        };
        let imageblock = match imageblock_use {
            Some(_) => made_imageblock.as_ref().or(self.imageblock.as_ref()),
            None => None,
        };
        // The pass's tile memory, made at its first draw or dispatch that reads it.
        let tile_buffer = plan
            .descriptors
            .iter()
            .find_map(|write| match write.source {
                WriteSource::TileMemory { bytes } => Some(bytes),
                _ => None,
            })
            .map(|bytes| self.tile_memory.buffer(&device, bytes))
            .transpose()?;
        // A draw that reads through snapshots binds each slot's snapshot in the form its plan
        // names, made at the first draw that takes it so and kept with the texture
        // (`render/snapshot.rs`).
        let mut snapshot_texels = Vec::new();
        let snapshot_views = match &snapshot {
            None => None,
            Some(wanted) => {
                let mut views = Vec::with_capacity(wanted.slots.len());
                for &slot in &wanted.slots {
                    let texture = self
                        .fragment
                        .textures
                        .get(&slot)
                        .ok_or_else(|| internal("a snapshot of an unbound slot"))?;
                    let form = snapshot::form(&plan, slot)
                        .ok_or_else(|| internal("a snapshot of a slot its plan has none of"))?;
                    let texels = texture.snapshot_texels(self.core.command_buffer(), form)?;
                    views.push((slot, texels.snapshot_view(form)?));
                    snapshot_texels.push(texels);
                }
                Some(binding::SnapshotViews {
                    slots: wanted.slots.clone(),
                    views,
                })
            }
        };
        // An indirect mesh draw's arguments pass, whose scratch its draw reads, and a folded
        // object function's direct draw's uploaded fold (`render/mesh_arguments.rs`).
        let arguments = match (
            commands.mesh_arguments,
            buffers.indirect.as_ref(),
            empty || culled,
        ) {
            (Some(form), Some(source), false) => {
                let threads = match draw {
                    DrawCall::MeshThreadgroupsIndirect {
                        threads_per_object_group,
                        ..
                    } => threads_per_object_group,
                    _ => [1; 3],
                };
                Some(mesh_arguments::ArgumentsPass::new(
                    &mut self.core,
                    source,
                    form,
                    threads,
                )?)
            }
            (Some(_), None, false) => {
                return Err(internal("a mesh arguments pass without its arguments"))
            }
            _ => None,
        };
        // Uploaded for a draw that launches nothing too: its object stage's hole takes the
        // record's address all the same.
        let fold = match &commands.object_fold {
            Some(fold) => Some(mesh_arguments::FoldUpload::new(&mut self.core, fold)?),
            None => None,
        };
        // A folded object function reads its region record at its scratch's address.
        let region_record = match (&arguments, &fold) {
            (Some(pass), _) => pass.record()?,
            (None, Some(fold)) => Some(fold.record()?),
            (None, None) => None,
        };
        let stages = [
            (Stage::Vertex, pass_start.or(Stage::Vertex, &self.vertex)),
            (Stage::Fragment, &self.fragment),
            (Stage::Object, pass_start.or(Stage::Object, &self.object)),
            (Stage::Mesh, pass_start.or(Stage::Mesh, &self.mesh)),
        ];
        let mut tables = binding::Tables::new(&stages)
            .with_pass(layout, views)
            .with_residency(&self.residency)
            .with_texture_table(texture_table.as_ref())
            .with_sampler_table(sampler_table.as_ref())
            .with_sampler_maps(sampler_maps);
        // The listed function tables: set 1's elements, or the snapshot a variant without the
        // function-table array reads in set 0, which has no heap of its own.
        tables = tables.with_function_tables(&function_tables);
        if attached {
            tables = tables.with_attachments(layout);
        }
        if let Some(views) = &snapshot_views {
            tables = tables.with_snapshots(views);
        }
        if let Some(heap) = &heap {
            // Taken as the draw settled: its variant has set-1 arrays.
            let states = argument_samplers
                .states()
                .ok_or_else(|| internal("a heap's draw without its argument samplers"))?;
            tables = tables.with_heap(states, heap);
        }
        if let Some(header) = self.replayed.as_ref().and_then(|r| r.merged_header()) {
            tables = tables.with_merged_header(header);
        }
        // A list in the private list form is read at its address, which the device gives the
        // list's memory where it has device addresses, as the form needs.
        if variant.key().private_list {
            let address = match (&listed, &rewrite) {
                (Some(list), _) => list.address(),
                (_, Some(pass)) => pass.address(),
                _ => None,
            };
            let gap = restarted.map_or(RuntimeGap::PrivatePrimitives, |r| r.gap.gap());
            let address = address.ok_or(Error::NotImplemented(gap))?;
            tables = tables.with_indices(address);
        }
        // A de-indexed draw's vertex function reads its indices in place, at their address.
        if variant.key().primitive_id_indices.is_some() {
            let address = buffers
                .index
                .as_ref()
                .and_then(DrawSource::kept)
                .and_then(|kept| kept.address())
                .ok_or(Error::NotImplemented(RuntimeGap::PrimitiveIdIndices))?;
            tables = tables.with_indices(address);
        }
        if let Some(shadow) = shadow {
            tables = tables.with_shadow(shadow.view()?);
        }
        if let Some(stand_in) = stand_in {
            tables = tables.with_depth_stand_in(stand_in.view()?);
        }
        if !live.state.extended_shadows.is_empty() {
            tables = tables.with_extended_shadows(self.extended.views()?);
        }
        if let Some(imageblock) = imageblock {
            tables = tables.with_imageblock(imageblock.view()?);
        }
        if let Some(buffer) = &tile_buffer {
            tables = tables.with_tile_memory(buffer);
        }
        // The factor buffer a tessellation pipeline's control stage reads (a handle of its own:
        // installing the draw's instance below borrows the whole encoder).
        let factors = self.factors.buffer.clone().filter(|_| tessellated);
        if let Some(factors) = &factors {
            tables = tables.with_factors(&factors.buffer, factors.offset);
        }
        if let Some((buffer, offset)) = buffers.patch_indices {
            tables = tables.with_patch_indices(buffer, offset);
        }
        if let Some((buffer, offset)) = buffers.control_point_indices {
            tables = tables.with_control_point_indices(buffer, offset);
        }
        tables.region_record = region_record;
        // An emulated mesh draw's work buffer and records (`render/mesh.rs`), and an emulated
        // tessellation draw's (`render/tessellation_scan.rs`).
        let mesh_work = commands
            .mesh_dispatch
            .as_ref()
            .map(|dispatch| mesh::MeshWork::new(&self.core, dispatch))
            .transpose()?;
        let tessellation_work = commands
            .tessellation_dispatch
            .as_ref()
            .map(|dispatch| tessellation_scan::TessellationWork::new(&self.core, dispatch))
            .transpose()?;
        let pipeline_objects = variant.pipeline();
        let mut blobs = Vec::new();
        for (index, module) in pipeline_objects.stages.iter().enumerate() {
            // A tessellation pipeline's synthesized stages are its vertex function's.
            let stage = match module.stage {
                metal2vulkan::vk::ShaderStage::FRAGMENT => Stage::Fragment,
                metal2vulkan::vk::ShaderStage::TASK_EXT => Stage::Object,
                metal2vulkan::vk::ShaderStage::MESH_EXT => Stage::Mesh,
                // An emulated object or mesh function's compute stage: the object function's first,
                // where the pipeline has one.
                metal2vulkan::vk::ShaderStage::COMPUTE
                    if pipeline_objects.stages.get(index + 1).is_some_and(|next| {
                        next.stage == metal2vulkan::vk::ShaderStage::COMPUTE
                    }) =>
                {
                    Stage::Object
                }
                metal2vulkan::vk::ShaderStage::COMPUTE => Stage::Mesh,
                _ => Stage::Vertex,
            };
            if let Some(blob) = &module.constant_blob {
                blobs.push((stage, blob));
            }
        }
        let command = binding::Command {
            descriptor_bytes: &commands.descriptor_bytes,
            descriptor_ranges: &commands.descriptor_ranges,
            root_tables: commands
                .root_tables
                .iter()
                .map(|(stages, table)| (*stages, table))
                .collect(),
            push_holes: &commands.push_holes,
            uses: &commands.uses,
            blobs,
            mesh: mesh_work.as_ref().map(mesh::MeshWork::regions),
            tessellation: tessellation_work
                .as_ref()
                .map(tessellation_scan::TessellationWork::regions),
        };
        let (index, indirect) = (buffers.index.as_ref(), buffers.indirect.as_ref());
        // Over the last draw's uses, in their storage (`LiveBind::uses`).
        binding::command_uses_into(
            call,
            (&plan, &command, &tables),
            (
                index.map(DrawSource::metal),
                indirect.map(DrawSource::metal),
            ),
            &mut live.uses,
        )?;
        let uses = &mut live.uses;
        // A use of an attachment's memory other than as the attachment, inside its instance,
        // which no copy serves (`texture/attached_reads.rs`): a draw that reads through snapshots
        // reads no attachment otherwise (`snapshot_holds`).
        if snapshot.is_none() {
            device.degrade_attachment_reads(call, layout, uses)?;
        }
        // In a pass of several color groups, what the draw reads of what another group writes
        // (`render/color_groups.rs`): checked at its first recording.
        if !self.rerouting {
            let roots = !commands.root_tables.is_empty();
            self.check_group_reads(uses, roots, &plan)?;
        }
        uses.extend(inputs);
        uses.extend(storage_uses);
        if let Some(shadow) = shadow {
            uses.extend(shadow.draw_uses()?);
        }
        if let Some(stand_in) = stand_in {
            uses.extend(stand_in.draw_uses()?);
        }
        uses.extend(self.extended.draw_uses(&plan)?);
        if let Some(imageblock) = imageblock {
            uses.extend(imageblock.draw_uses()?);
        }
        uses.extend(snapshot_texels.iter().map(Texture::snapshot_use));
        // A draw that names a detached buffer records nothing: its submission fails.
        let named = uses
            .iter()
            .map(|used| used.key)
            .filter(|key| !uses.resident().contains(key));
        if self.core.skips(named) {
            self.skipped_draw();
            return Ok(());
        }
        // The shadowed textures the draw uses, copied around its instance (`render/shadows.rs`).
        let textures = self
            .vertex
            .textures
            .values()
            .chain(self.fragment.textures.values())
            .chain(self.object.textures.values())
            .chain(self.mesh.textures.values())
            .chain(self.residency.textures());
        let mut draw_shadows = shadow::shadow_uses(textures, uses);
        // A draw that reads an attachment's texture at its fragments' own pixels alone copies in
        // only the texels around its scissors (`texture/attached_reads.rs`).
        if attached {
            let selects = graphics.writes_viewport_index || self.amplification.offsets_viewports();
            let compiled = variant.pipeline();
            if self.bound_own_pixel_reads(&plan, compiled, selects, &mut draw_shadows) {
                device.shared().counts.own_pixel_read();
            }
        }
        if self.replaying && !self.gathered().covers(&draw_shadows) {
            return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
        }
        // Over the last draw's, in their storage (`LiveBind::vertex_buffers`).
        live.vertex_buffers.clear();
        for bind in &plan.vertex_buffers {
            let table = pass_start.or(Stage::Vertex, &self.vertex);
            let offset = commands
                .buffer_offsets
                .iter()
                .find(|(binding, _)| *binding == bind.binding)
                .map(|&(_, offset)| offset)
                .ok_or_else(|| internal("a vertex buffer without its offset"))?;
            let stride = commands
                .vertex_strides
                .iter()
                .find(|(binding, _)| *binding == bind.binding)
                .map(|&(_, stride)| u64::from(stride));
            if dynamic_strides && stride.is_none() {
                return Err(internal("a dynamic vertex stride without its value"));
            }
            // The plan's offset is in the buffer's space (`ArgumentTable::stage_table`), which
            // starts at a heap buffer's offset in the heap's `VkBuffer`; inline bytes' counts
            // from their region's start.
            let (raw, start) = match (
                table.buffers.get(&bind.buffer_index),
                table.bytes.get(&bind.buffer_index),
            ) {
                (Some((buffer, _)), _) => buffer.space(),
                (None, Some(bytes)) => (bytes.region.raw(), bytes.region.offset()),
                (None, None) => return Err(internal("the plan binds an unbound vertex buffer")),
            };
            (live.vertex_buffers).push((bind.binding, raw, start + offset, stride.unwrap_or(0)));
        }
        let vertex_buffers = &live.vertex_buffers[..];
        // Its draw state: the one an earlier draw of the variant derived under the same settings'
        // inputs, else derived, and kept (`render/draw_shape.rs`).
        let failed = (failing.as_ref(), shadow_use.is_some(), tests_use.is_some());
        let kept_state = shape_token.and_then(|token| self.shape_memo.state(token, &variant));
        let reused_state = kept_state.is_some();
        let ShapedState {
            state: draw_state,
            viewports,
            depth,
        } = match kept_state {
            Some(kept) => {
                debug_assert!(
                    (self.shape_inputs())
                        .draw_state(graphics, &settings, failed)
                        .is_ok_and(|fresh| fresh == kept),
                    "a draw's state derived again differs from the one kept for its inputs"
                );
                kept
            }
            None => {
                let watch = EachWatch::start();
                let inputs = self.shape_inputs();
                let shaped = inputs.draw_state(graphics, &settings, failed)?;
                // Kept where its derivation announced nothing at each occurrence.
                if let (Some(token), false) = (shape_token, watch.met()) {
                    self.shape_memo.keep_state(token, &variant, &shaped);
                }
                shaped
            }
        };
        let fixed = draw_state.fixed;
        // A rate-mapped pass records the draw once per region of the map's layer of the slice
        // it renders (`rate_mapped_pass.rs`): each amplification drawn one at a time its own
        // slice's, else the draw's one slice, or none known where its primitives choose.
        // `draw_states` holds the recordings of each amplification drawn alone, in order, or of
        // the whole draw.
        let mut draw_states = match &self.rate_map {
            None => DrawStates::Whole([draw_state]),
            Some(pass) => DrawStates::Regions({
                let length = self.render_target_array_length;
                let slices: Vec<Option<u32>> =
                    match pipeline.vertex_chooses_slice() || views.view_mask != 0 {
                        true => vec![None],
                        false if commands.amplifications.is_empty() => {
                            vec![Some(views.base_layer)]
                        }
                        false => (0..commands.amplifications.len() as u32)
                            .map(|a| Some(amplification.layer(a, length, max)))
                            .collect(),
                    };
                slices
                    .into_iter()
                    .map(|slice| {
                        let regions = pass.draws(
                            slice,
                            &viewports[0],
                            self.state.first_scissor(),
                            graphics,
                            &device.shared().region_limits,
                            render_area,
                        )?;
                        Ok(regions
                            .into_iter()
                            .map(|(viewport, scissor)| {
                                let [min_depth, max_depth] =
                                    depth.unwrap_or([viewport.min_depth, viewport.max_depth]);
                                DrawState {
                                    viewports: ViewportList::One(vk::Viewport {
                                        min_depth,
                                        max_depth,
                                        ..viewport
                                    }),
                                    scissors: ViewportList::One(scissor),
                                    ..draw_state.clone()
                                }
                            })
                            .collect())
                    })
                    .collect::<Result<_, Error>>()?
            }),
        };
        // A draw that reaches no fragment runs its vertex function alone, through empty scissors,
        // and so do unsized points.
        if failing == Some(FailingDraw::Untouched) || unsized_points {
            for state in draw_states.iter_mut() {
                for scissor in state.scissors.iter_mut() {
                    *scissor = vk::Rect2D::default();
                }
            }
        }
        let instance = match &next {
            Next::Current => &self.current,
            Next::New { instance, .. } => &**instance,
        };
        // The class's list topology where the pipeline makes the topology dynamic.
        let (keyed_topology, keyed_restart) =
            super::pipeline::keyed_assembly(topology, primitive_restart, &graphics.dynamic_states);
        // A mesh draw's own complete key (its threadgroup size), else the plan's.
        let specialization = commands
            .specialization
            .as_ref()
            .unwrap_or(&plan.specialization);
        let strides: &[(u32, u32)] = match dynamic_strides {
            true => &[],
            false => &commands.vertex_strides,
        };
        // The draw's graphics key, borrowed: made only where the draw looks it up.
        let draw_key = || DrawKey {
            specialization,
            topology: keyed_topology,
            primitive_restart: keyed_restart,
            pass: &instance.shape,
            strides,
            fixed: FixedState::without_dynamic(fixed, &graphics.dynamic_states),
            sample_positions: &self.layout.sample_positions,
        };
        // What each of the draw's recordings takes of its two draws where it reads through
        // snapshots (`render/snapshot.rs`): the drawing draw's masks are the ones the pipeline
        // would have made static.
        let pair_base = match (phase_hole, device.shared().color_write_masks) {
            (Some(hole), Some(masks)) => Some(snapshot::Pair {
                phase: snapshot::Phase::Snapshot,
                first: true,
                hole,
                masks,
                own: {
                    let key = draw_key();
                    super::pipeline::color_write_masks(graphics, key.pass, key.fixed.masks_color)
                },
            }),
            (Some(_), None) => return Err(internal("a snapshot without its write mask command")),
            (None, _) => None,
        };
        // The formats the draw binds, which a specialized notice names.
        let formats = |slot| tables.bound_format(slot);
        (variant.objects()).degrade_bound_formats(&variant.pipeline().stages, Some(&formats));
        // A variant whose notes test no bound value is quiet at every draw, and kept as none
        // (`Device::tests_bound_values`).
        let quiet = !super::Device::tests_bound_values(&variant.pipeline().stages)
            || self
                .quiet_bound_values
                .as_ref()
                .is_some_and(|q| q.holds(&variant, &plan, specialization));
        if !quiet
            && device.degrade_bound_values(
                &variant.pipeline().stages,
                specialization,
                &|k| plan.specialized_value(k),
                Some(&formats),
            )
        {
            self.quiet_bound_values = Some(QuietBoundValues {
                variant: Arc::clone(&variant),
                plan: Arc::clone(&plan),
                specialization: specialization.clone(),
            });
        }
        device.degrade_cross_class(&plan.diagnostics, &formats);
        // Those its notices name, where the device announces any of its variant's kinds at every
        // occurrence (announced as the draw records, below), taken while its tables live.
        let bound_each = device
            .binds_each(variant.objects().kinds())
            .then(|| variant.objects().tested_formats(&formats));
        // The draw's `VkPipeline`: made, or being made on the device's pool while the draw goes
        // on, where its recording can wait for it (`render/pending.rs`): not in an execution's
        // replay, whose host step records now, nor for a draw that records nothing, nor in a pass
        // of custom sample positions, whose instance binds its first draw's pipeline as it
        // begins (`positioned`).
        let compatible = Arc::clone(&instance.compatible);
        // Nor while the encoder records an instance's draws again, whose instance a failed
        // reroute drops (`render/reroute.rs`).
        let deferrable = !self.replaying
            && !self.rerouting
            && !empty
            && !culled
            && self.layout.sample_positions.is_empty();
        // The one an earlier draw of the same settings' inputs found made by the same key (each
        // of its fields compared, or what it is made of), else found, and kept where made
        // (`render/draw_shape.rs`); not a mesh draw's, of its own specialization.
        let looked = PipelineInputs {
            variant: &variant,
            plan: &plan,
            shape: instance.shape.version(),
            compatible: &instance.compatible,
            strides,
            fixed,
            topology: (keyed_topology, keyed_restart),
        };
        let token = shape_token.filter(|_| commands.specialization.is_none());
        let found = token.and_then(|token| self.shape_memo.pipeline(token, &looked));
        let reused_pipeline = found.is_some();
        let draw_pipeline = match found {
            Some(made) => DrawPipeline::Made(made),
            None => {
                let watch = EachWatch::start();
                let key = draw_key();
                // An execution's replay records now: under its command buffer's bound its draw
                // waits for its making on the device's pool (most begun by the settle at commit,
                // `RenderExecution::settle`) until the bound at most, past which it records
                // nothing. Not a device replay's draw, of one recording for every command.
                let bounded = (self.replaying && self.replayed.is_none() && !self.rerouting)
                    && !(empty || culled)
                    && self.layout.sample_positions.is_empty();
                let deadline = match bounded {
                    true => self.core.command_buffer().making_deadline(),
                    false => None,
                };
                let (draw_pipeline, own) = match (deferrable, deadline) {
                    (true, _) => variant.begin_graphics_pipeline(
                        &device,
                        &key,
                        &compatible,
                        (&pipeline, &formats),
                    ),
                    (false, Some(deadline)) => {
                        let begun = variant.begin_graphics_pipeline(
                            &device,
                            &key,
                            &compatible,
                            (&pipeline, &formats),
                        );
                        match begun {
                            (DrawPipeline::Making(making), _) => {
                                match making.wait_until(deadline) {
                                    Some(Ok(made)) => (DrawPipeline::Made(made), true),
                                    // A failed making is made again here, as a caller after one
                                    // does (`making.rs`).
                                    Some(Err(_)) => {
                                        let (made, own) = variant.graphics_pipeline_now(
                                            &device,
                                            &key,
                                            &compatible,
                                            (&pipeline, &formats),
                                        )?;
                                        (DrawPipeline::Made(made), own)
                                    }
                                    // Still being made at the bound: the draw records nothing,
                                    // before anything of it is tracked, as a draw whose pipeline
                                    // a capture misses does; later command buffers draw with it.
                                    None => {
                                        self.drop_late_replayed_draw(&variant, specialization)?;
                                        return Ok(());
                                    }
                                }
                            }
                            made => made,
                        }
                    }
                    (false, None) => {
                        let (made, own) = variant.graphics_pipeline_now(
                            &device,
                            &key,
                            &compatible,
                            (&pipeline, &formats),
                        )?;
                        (DrawPipeline::Made(made), own)
                    }
                };
                // Kept where it is the key's own, made: not its general form; and where finding it
                // announced nothing at each occurrence.
                if let (Some(token), DrawPipeline::Made(made), true, false) =
                    (token, &draw_pipeline, own, watch.met())
                {
                    self.shape_memo.keep_pipeline(token, &looked, made);
                }
                draw_pipeline
            }
        };
        if reused_state && reused_pipeline {
            device.shared().counts.draw_shape_reused();
        }
        // A pipeline made at the state's creation announces its notices at the first draw that
        // finds it made; a draw that began a making announced them as it began it.
        if let DrawPipeline::Made(made) = &draw_pipeline {
            variant.first_use(made, specialization, &formats);
        }
        // Set 0's writes and the heap set's, in the last draw's storage (`LiveBind::writes`),
        // emptied as the draw returns: their keepers hold nothing past it.
        let mut writes = Emptied::new(&mut live.writes);
        // The samplers they name, in the last draw's storage (`LiveBind::samplers`), emptied as
        // the draw returns likewise: its encoder retains them as it records.
        let mut samplers = Emptied::new(&mut live.samplers);
        let objects = Arc::clone(variant.objects());
        // Set 1 the device's (written below, once nothing refuses).
        let bound = {
            let mut state = self.core.command_buffer().state();
            let memo = Some(&mut self.bind_memo);
            binding::bind_into(
                call,
                &device,
                &mut state,
                (&plan, &command, &tables),
                (memo, &mut writes, &mut samplers),
            )?
        };
        // The draw's push bytes, filled in place, over the last draw's (`LiveBind::pushes`).
        binding::push_writes_into(&commands.push_constants, &bound.holes, &mut live.pushes)?;
        // A device replay's draws read each command's slot tables (`render/executions/gpu.rs`):
        // the first command's, in place of the root tables the binder made.
        let replay_holes = match &self.replayed {
            Some(replayed) => replayed.first_tables(&commands.push_holes, &mut live.pushes)?,
            None => Vec::new(),
        };
        let push = &live.pushes;
        // A plan that draws one amplification at a time records the draw once per amplification,
        // in order, each after that amplification's push writes.
        let amplified = commands
            .amplifications
            .iter()
            .map(|writes| {
                let mut all = push.clone();
                all.extend(binding::push_writes(writes.clone(), &[])?);
                Ok(all)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let pushes = match amplified.is_empty() {
            true => std::slice::from_ref(push),
            false => &amplified[..],
        };
        // An emulated tessellation draw records each chunk's draw from its work buffer, after the
        // push write that names the chunk, in order (`render/tessellation_scan.rs`).
        let chunked = match (&commands.tessellation_dispatch, &tessellation_work) {
            (Some(dispatch), Some(work)) => work
                .draws(dispatch, push)?
                .into_iter()
                .map(|(writes, raw, command)| (writes, [(None, Some(raw), command)]))
                .collect(),
            _ => Vec::new(),
        };
        let heap_writes: Vec<_> = writes.extract_if(.., |write| write.set > 0).collect();
        // Set 0: pushed, or the encoder's last set where it holds the same writes, else one from
        // the command buffer's arena, written (`EncoderCore::set_zero`), with its dynamic
        // offsets.
        let ((set, written), pushed): (_, &[binding::SetWrite]) = match objects.pushes_set_zero() {
            true => {
                // Over the last draw's, in their storage (`LiveBind::pushed`).
                binding::SetWrite::push_into(&writes, objects.immutable_pushes(), &mut live.pushed);
                live.offsets.clear();
                ((None, None), &live.pushed)
            }
            false => {
                let offsets = &mut live.offsets;
                (self.core.set_zero(&objects, (&mut writes, offsets))?, &[])
            }
        };
        // The Vulkan draws: the binder's, the index bounds pass's two, or the index rewrite
        // pass's one.
        let sized = device.shared().sized_index_bind;
        let one = |draw| ([draw; 2], 1);
        // A list the CPU or a gather run wrote draws as the indexed draw of it the binder's draw
        // becomes; in the private list form, as the binder's draw.
        let listed_draw = match listed.as_ref().filter(|_| !private.private_list) {
            Some(list) => Some(
                list.draw(commands.draw, sized)
                    .ok_or_else(|| internal("a written list of another draw"))?,
            ),
            None => None,
        };
        let (vk_draws, count) = match (&rewrite, &bounds, index, &mesh_work, commands.draw) {
            _ if listed_draw.is_some() => {
                let (bind, arguments, draw) =
                    listed_draw.ok_or_else(|| internal("a written list's draw"))?;
                one((Some(bind), arguments, draw))
            }
            (Some(pass), ..) => one(pass.draw(sized)),
            (_, Some(pass), Some(source), _, _) => (pass.draws(source, commands.draw, sized)?, 2),
            // A folded object function's direct draw's pieces are its uploaded fold's.
            (
                _,
                _,
                _,
                _,
                metal2vulkan::vk::DrawCommand::DrawMeshTasksIndirect {
                    offset,
                    draw_count,
                    stride,
                },
            ) if fold.is_some() => {
                let (raw, at) = fold
                    .as_ref()
                    .map(mesh_arguments::FoldUpload::commands)
                    .ok_or_else(|| internal("a folded object grid's upload gone"))?;
                one((
                    None,
                    Some(raw),
                    metal2vulkan::vk::DrawCommand::DrawMeshTasksIndirect {
                        offset: at + offset,
                        draw_count,
                        stride,
                    },
                ))
            }
            // A folded object function's indirect draw's pieces are its arguments pass's, both
            // offsets into its scratch.
            (
                _,
                _,
                _,
                _,
                metal2vulkan::vk::DrawCommand::DrawMeshTasksIndirectCount {
                    offset,
                    count_offset,
                    max_draw_count,
                    stride,
                },
            ) if arguments.is_some() => {
                let (raw, at) = arguments
                    .as_ref()
                    .map(mesh_arguments::ArgumentsPass::scratch)
                    .ok_or_else(|| internal("a mesh arguments pass gone"))?;
                one((
                    None,
                    Some(raw),
                    metal2vulkan::vk::DrawCommand::DrawMeshTasksIndirectCount {
                        offset: at + offset,
                        count_offset: at + count_offset,
                        max_draw_count,
                        stride,
                    },
                ))
            }
            // An indirect mesh draw's counts are its arguments pass's.
            (
                _,
                _,
                _,
                _,
                metal2vulkan::vk::DrawCommand::DrawMeshTasksIndirect {
                    draw_count, stride, ..
                },
            ) if arguments.is_some() => {
                let (raw, at) = arguments
                    .as_ref()
                    .map(mesh_arguments::ArgumentsPass::scratch)
                    .ok_or_else(|| internal("a mesh arguments pass gone"))?;
                one((
                    None,
                    Some(raw),
                    metal2vulkan::vk::DrawCommand::DrawMeshTasksIndirect {
                        offset: at,
                        draw_count,
                        stride,
                    },
                ))
            }
            // A scanned emulated mesh draw's arguments are its work buffer's.
            (
                _,
                _,
                _,
                Some(work),
                metal2vulkan::vk::DrawCommand::DrawIndirect {
                    offset,
                    draw_count,
                    stride,
                },
            ) => {
                let (raw, at) = work.at(offset);
                one((
                    None,
                    Some(raw),
                    metal2vulkan::vk::DrawCommand::DrawIndirect {
                        offset: at,
                        draw_count,
                        stride,
                    },
                ))
            }
            _ => one((
                index.and_then(|source| IndexBind::of(source, commands.draw, sized)),
                indirect.map(|source| source.raw),
                commands.draw,
            )),
        };
        let vk_draws = &vk_draws[..count];
        // The draw's steps before the pass (its copies, its index bounds or rewrite pass) read
        // what an earlier draw or tile dispatch of the encoder wrote behind a memory barrier: they
        // record before the draw's render pass instance instead, which follows the writer's.
        let before_instance = !self.replaying && {
            let reads: Vec<Use<ResourceKey>> = index
                .into_iter()
                .chain(indirect)
                .filter_map(DrawSource::copy_reads)
                .chain(
                    bounds
                        .iter()
                        .flat_map(|bounded| bounded.uses().iter().cloned()),
                )
                .chain(rewrite.iter().flat_map(|pass| pass.uses().iter().cloned()))
                .collect();
            self.core.uses().written(&reads) && !self.unordered.written(&reads)
        };
        // An emulated mesh draw's dispatch, or an emulated tessellation draw's scan, which reads
        // what an earlier draw or tile dispatch of the encoder wrote behind a memory barrier, or
        // writes what one read or wrote, records before the draw's render pass instance too.
        let scanned = commands.mesh_dispatch.is_some()
            || commands.tessellation_dispatch.is_some()
            || arguments.is_some();
        let mesh_uses: Vec<Use<ResourceKey>> = match scanned {
            true => uses
                .iter()
                .filter(|used| {
                    (used.read.stages | used.write.stages)
                        .contains(vk::PipelineStageFlags2::COMPUTE_SHADER)
                })
                .cloned()
                .collect(),
            false => Vec::new(),
        };
        // An execution's emulated mesh command records its dispatches before the whole pass
        // (`render/executions.rs`), but where a draw of the pass Metal runs before it (the
        // encoder's before the execution, or an earlier command's) wrote what they read or read
        // or wrote what they write.
        // As Metal orders them: an earlier draw's fragments run after them, so what those wrote
        // is not read, and the dispatches stay before the pass; behind the command's barrier an
        // earlier draw's stages before its fragments run before them, so the dispatches record
        // before the run the barrier begins (`run_barrier`, `Run::before`), after those draws.
        // Refused still: writing what an earlier draw's fragments touched (unmeasured), what
        // touches an earlier draw's other stages without the command's barrier (unordered on
        // Metal, or ordered by a barrier of the encoder the execution does not see), and both.
        let mut earlier = |uses: &[Use<ResourceKey>], stages| {
            self.core.uses().touched_in(uses, stages)
                || uses.iter().any(|used| {
                    self.drawn_before
                        .iter()
                        .any(|known| tracker::touches(known, used, stages))
                })
        };
        // The dispatches' reads alone (`read`), or their writes alone.
        let only = |read: bool| -> Vec<Use<ResourceKey>> {
            mesh_uses
                .iter()
                .filter(|used| match read {
                    true => !used.read.is_empty(),
                    false => !used.write.is_empty(),
                })
                .map(|used| match read {
                    true => Use {
                        write: Access::NONE,
                        ..used.clone()
                    },
                    false => Use {
                        read: Access::NONE,
                        ..used.clone()
                    },
                })
                .collect()
        };
        let work = self.replaying && !(empty || culled);
        let after_geometry = work && earlier(&mesh_uses, GEOMETRY_DRAW_STAGES);
        let reads_fragments = work && earlier(&only(true), FRAGMENT_DRAW_STAGES);
        let writes_fragments = work && earlier(&only(false), FRAGMENT_DRAW_STAGES);
        if writes_fragments || (after_geometry && (reads_fragments || !self.command_barrier)) {
            return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
        }
        // A scan reads Metal's arguments where the draw reads them, after their copy. An arguments
        // pass, which writes no Metal resource, waits only for what wrote them.
        let mesh_before = match self.replaying {
            true => after_geometry,
            false => {
                (self.core.uses().written(&mesh_uses) && !self.unordered.written(&mesh_uses))
                    || (arguments.is_none() && self.core.uses().accessed(&mesh_uses))
                    || before_instance
            }
        };
        let stages = uses
            .iter()
            .fold(vk::PipelineStageFlags2::empty(), |all, used| {
                all | used.read.stages | used.write.stages
            });
        let reaching = (&**uses, uses.resident());
        self.core
            .note_reaching(stages, reaching, Some(&mut live.noted))?;
        self.unordered.add(uses);
        self.pass_start.announce(&device, uses, mesh_before);
        self.pass_start.add(uses);
        let heap_set = match heap {
            Some(_) => Some(heap_sets::bind(
                &device,
                &objects,
                &mut self.core.command_buffer().state().heaps,
                heap_writes,
                uses,
            )?),
            None => None,
        };
        // The query that resumes the active one after a draw whose fragments Vulkan would count
        // and Metal does not.
        let counts_none = matches!(failing, Some(FailingDraw::Runs(_)));
        let resumed = match counts_none && !empty && !culled {
            true => self.visibility.continuation(&device)?,
            false => None,
        };
        // An emulated tessellation draw's scan, over its work buffer, the factors and Metal's
        // arguments.
        let tessellation_steps = match (
            &commands.tessellation_dispatch,
            &tessellation_work,
            empty || culled,
        ) {
            (Some(dispatch), Some(work), false) => {
                let factors = match (&dispatch.factors, &factors) {
                    (Some(range), Some(bound)) => {
                        let (raw, start) = bound.buffer.space();
                        Some((raw, start + range.offset, range.range))
                    }
                    (None, _) => None,
                    (Some(_), None) => {
                        return Err(internal("a tessellation scan without the factor buffer"))
                    }
                };
                let arguments = match (dispatch.arguments, indirect) {
                    (Some(range), Some(source)) => Some((source.raw, range)),
                    (None, _) => None,
                    (Some(_), None) => {
                        return Err(internal("a tessellation scan without its arguments"))
                    }
                };
                Some(tessellation_scan::TessellationSteps::new(
                    &mut self.core,
                    dispatch,
                    work,
                    (factors, arguments),
                    mesh_uses.clone(),
                )?)
            }
            _ => None,
        };
        // An emulated mesh draw's compute work: its object and mesh functions' pipelines under
        // the draw's specialization, its scan, and the draw's sets and compute-stage push writes.
        let mesh_steps = match (&commands.mesh_dispatch, &mesh_work, empty || culled) {
            (Some(dispatch), Some(work), false) => {
                let specialization = commands
                    .specialization
                    .as_ref()
                    .ok_or_else(|| internal("an emulated mesh draw without its specialization"))?;
                let push = pushes
                    .first()
                    .map(|writes| {
                        writes
                            .iter()
                            .filter(|(stages, ..)| stages.contains(vk::ShaderStageFlags::COMPUTE))
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default();
                let pipelines = variant.mesh_pipelines(specialization, &formats)?;
                // An indirect draw with an object function: its object arguments pass, and its
                // object dispatch from the work buffer, where the pass writes it.
                let arguments = match (dispatch.object_arguments, indirect) {
                    (Some(range), Some(source)) => Some(mesh::Scan::new(
                        &mut self.core,
                        OwnedModule::MeshObjectArguments,
                        work,
                        Some((source.raw, range)),
                    )?),
                    (None, _) => None,
                    (Some(_), None) => {
                        return Err(internal("an object arguments pass without its arguments"))
                    }
                };
                let launch = match (dispatch.object_groups, &arguments) {
                    (Some(groups), None) => Some(mesh::Launch::Direct(groups)),
                    (None, Some(_)) => {
                        let (raw, at) = work.at(MeshDispatch::OBJECT_DISPATCH_OFFSET);
                        Some(mesh::Launch::Indirect(raw, at))
                    }
                    (None, None) => None,
                    (Some(_), Some(_)) => {
                        return Err(internal("an object dispatch of both launches"));
                    }
                };
                let object = match (launch, pipelines.object) {
                    (Some(launch), Some(pipeline)) => Some((pipeline, launch)),
                    (None, _) => None,
                    (Some(_), None) => {
                        return Err(internal("an object dispatch without the object's pipeline"))
                    }
                };
                let (scan, launch) = match dispatch.mesh {
                    MeshGroups::Direct(groups) => (None, mesh::Launch::Direct(groups)),
                    MeshGroups::Scanned { arguments } => {
                        let arguments = match (arguments, indirect) {
                            (Some(range), Some(source)) => Some((source.raw, range)),
                            (None, _) => None,
                            (Some(_), None) => {
                                return Err(internal("a mesh scan without its arguments"))
                            }
                        };
                        let scan = mesh::Scan::new(
                            &mut self.core,
                            OwnedModule::MeshScan,
                            work,
                            arguments,
                        )?;
                        let (raw, at) = work.at(MeshDispatch::DISPATCH_OFFSET);
                        (Some(scan), mesh::Launch::Indirect(raw, at))
                    }
                    _ => return Err(internal("mesh threadgroups of another kind")),
                };
                Some(mesh::MeshSteps {
                    layout: objects.layout(),
                    sets: set.as_slice().to_vec(),
                    offsets: live.offsets.clone(),
                    heap: heap_set,
                    push,
                    arguments,
                    object,
                    scan,
                    pipeline: pipelines.mesh,
                    launch,
                    uses: mesh_uses,
                })
            }
            _ => None,
        };
        let records = self.core.records_executions();
        let mut created = match &draw_pipeline {
            DrawPipeline::Made(made) if records => vec![made.created_in],
            _ => Vec::new(),
        };
        if let Some(steps) = mesh_steps.as_ref().filter(|_| records) {
            created.extend(
                steps
                    .object
                    .as_ref()
                    .map(|(pipeline, _)| pipeline.created_in),
            );
            created.push(steps.pipeline.created_in);
        }
        // A draw whose pipeline is being made records into a secondary of its own, between the
        // instance's and a new one, counting in a query lent to it where one is active and it
        // counts samples (`render/pending.rs`): taken before anything records.
        let pending = match &draw_pipeline {
            DrawPipeline::Made(_) => None,
            DrawPipeline::Making(making) => {
                let state = self
                    .pipeline
                    .clone()
                    .ok_or_else(|| internal("a draw without its pipeline state"))?;
                let command_buffer = self.core.command_buffer();
                let draws = command_buffer.begin_draws(&compatible)?;
                let after = command_buffer.begin_draws(&compatible)?;
                let loan = match resumed.is_none() && self.visibility.active() {
                    true => self.visibility.lend(&device, 0)?,
                    false => None,
                };
                Some((making.clone(), state, (draws, after), loan))
            }
        };
        // Its execution record's creation times, which its making sets once made.
        let mut later = None;
        // Whether the draw can record again into an instance that leaves more colors out of its
        // subpass (`render/reroute.rs`). A stand-in arriving with it does not stop it: the pass's
        // layout keeps the stand-in from the draw on, so the draw records again into it, as a later
        // draw would.
        let again = !self.replaying
            && !self.layout.fetch_inputs
            && self.layout.samples == 1
            && self.rate_map.is_none()
            && !self.counts_visibility()
            && !tessellated
            && !meshes
            && imageblock_use.is_none()
            && shadow_use.is_none()
            && tests_use.is_none()
            && !reads_tile_memory
            && !graphics.fragment_writes_memory
            && !graphics.vertex_writes_memory
            && views == ViewSetup::default()
            && pushes.len() == 1
            && !hidden_depth
            && reformats.is_empty()
            && buffers.indirect.is_none()
            && buffers.patch_indices.is_none()
            && buffers.control_point_indices.is_none()
            && !before_instance
            && bounds.is_none()
            && rewrite.is_none()
            && listed.is_none()
            && mesh_steps.is_none()
            && tessellation_steps.is_none()
            && draw_shadows.is_empty()
            && snapshot.is_none();
        // A draw whose pipeline is being made records between its instance's own draws, which
        // an instance recorded inline cannot run: it records into a new instance after it
        // (`render/inline_first.rs`).
        let pending_split =
            pending.is_some() && self.current.inline && matches!(next, Next::Current);
        let (mut next, stood, continuation) = match (pending.is_some() && self.current.inline, next)
        {
            (true, Next::Current) => {
                let (mut next, stood) = self.next_instance(
                    (&fetched, Some(&pipeline)),
                    (stencil_use.is_some(), tests_use.is_some()),
                    views,
                    true,
                    None,
                    &routing.locations,
                    &reformats,
                )?;
                // The current instance would have taken it: the new one continues it.
                if let Next::New { instance, .. } = &mut next {
                    instance.continues = true;
                }
                (next, stood, self.visibility.continuation(&device)?)
            }
            (_, next) => (next, stood, continuation),
        };
        // The colors the draw reads through snapshots are in the feedback loop layout for its
        // instance's render pass instance (`render/snapshot.rs`): a new one's from its begin,
        // which therefore records in a secondary.
        let feedback = snapshot.as_ref().map_or(Feedback::NONE, |w| w.feedback);
        if let Next::New { instance, .. } = &mut next {
            instance.feedback.0 |= feedback.0;
        }
        // The pass's first instance begins inline at its first draw that records, where it can
        // (`render/inline_first.rs`).
        let made_ahead = matches!(draw_pipeline, DrawPipeline::Made(_));
        let inline = self.inline_begin(
            &next,
            stood.as_deref(),
            (!empty && !culled, made_ahead, unheld_writes),
            feedback,
        )?;
        // And a later instance of such a pass after a split, where it can, the instance before's
        // copies planned (`render/inline_splits.rs`).
        let split_inline = match inline {
            Some(_) => None,
            None => self.inline_split(&next, stood.as_deref(), (!empty && !culled, made_ahead)),
        };
        // A new instance that records inline takes no secondary; any other begins its own.
        if inline.is_none() && split_inline.is_none() {
            self.begin_new_draws(&mut next)?;
        }
        // A device replay's draws record as one recording of the draw's, each command's after it
        // (`ReplayedDraws::record`), which its decision at commit holds to.
        if self.replayed.is_some()
            && (pushes.len() != 1
                || vk_draws.len() != 1
                || !chunked.is_empty()
                || pair_base.is_some()
                || !matches!(draw_states, DrawStates::Whole(_))
                || !matches!(draw_pipeline, DrawPipeline::Made(_))
                || !matches!(self.inline, inline_first::Inline::Never))
        {
            return Err(internal(
                "a draw replay's draw of other recordings than one",
            ));
        }
        // Recording starts here: nothing below can be refused.
        if let Some(written) = written {
            self.core.keep_set_zero(written, &writes);
        }
        self.driver_restarts |= tessellation.is_some_and(|t| !t.emulated);
        if !pushed.is_empty() {
            device.shared().counts.draw_pushing_set_zero();
        }
        if let Some(made) = made {
            self.shadow = Some(made);
        }
        self.returned_stencil |= shadow_use.is_some();
        if let Some(made) = made_stand_in {
            self.depth_stand_in = Some(made);
        }
        if let Some(made) = made_imageblock {
            self.imageblock = Some(made);
        }
        // The first draw's alignment is the pass's, where only tile dispatches made the storage.
        if let (Some(held), Some(used)) = (self.imageblock.as_mut(), &imageblock_use) {
            held.adopt(used.alignment);
        }
        // Why the draw splits the pass where it does: the first that holds.
        let cause = if ordered {
            Split::TileMemory
        } else if after_barrier {
            Split::AttachedRead
        } else if apart {
            Split::Feedback
        } else if first_write {
            Split::MemorylessWrite
        } else if pending_split {
            Split::PendingPipeline
        } else if self.replaying {
            Split::Execution
        } else {
            Split::Shape
        };
        self.install(next, stood, continuation, cause);
        self.current.feedback.0 |= feedback.0;
        self.current.storage_colors |= storage_colors;
        if let Some(begin) = inline {
            self.begin_inline(begin);
        }
        if let Some(begin) = split_inline {
            self.begin_inline_split(begin);
        }
        self.installed_extended_draw(&pipeline);
        self.tile_memory_splits += usize::from(tile_memory_split);
        self.barrier_splits += usize::from(run_barrier_split);
        self.run_barrier = false;
        self.gathering().add(&draw_shadows, uses);
        if !empty && !culled {
            for source in index.into_iter().chain(indirect) {
                match (source.copy_steps(&mut self.core), before_instance) {
                    (Some(copy), true) => self.steps_before().push(Before::Copy(copy)),
                    (Some(copy), false) => copy.record(&mut self.core),
                    (None, _) => {}
                }
            }
            self.current.tile_memory |= reads_tile_memory;
            match bounds {
                Some(Bounded::Pass(pass)) => {
                    let pass = pass.into_steps(&mut self.core);
                    match before_instance {
                        true => self.steps_before().push(Before::Bounds(pass)),
                        false => pass.record(&mut self.core),
                    }
                }
                Some(Bounded::Run(slot)) => slot.announce(self.core.device()),
                None => {}
            }
            if let Some(pass) = rewrite {
                let pass = pass.into_steps(&mut self.core);
                match before_instance {
                    true => self.steps_before().push(Before::Rewrite(pass)),
                    false => pass.record(&mut self.core),
                }
            }
            if let Some(mut steps) = mesh_steps {
                steps.retain(&mut self.core);
                match mesh_before {
                    true => self.steps_before().push(Before::Mesh(steps)),
                    false => steps.record(&mut self.core),
                }
            }
            if let Some(pass) = arguments {
                let steps = pass.into_steps(&mut self.core);
                match mesh_before {
                    true => self.steps_before().push(Before::MeshArguments(steps)),
                    false => steps.record(&mut self.core),
                }
            }
            if let Some(steps) = tessellation_steps {
                steps.retain(&mut self.core);
                match mesh_before {
                    true => self.steps_before().push(Before::Tessellation(steps)),
                    false => steps.record(&mut self.core),
                }
            }
            // A draw whose pipeline is being made records the barrier first in its own
            // secondary (`render/pending.rs`).
            // A draw that reaches interlocked storage an earlier draw of the instance wrote, the
            // imageblock's, a color's or the depth's stand-in, records after the barrier
            // (`InterlockedStorage`).
            let after_writes = interlocked.meets(self.current.interlocked_writes);
            let mut storage_barrier = None;
            if after_writes {
                let view_mask = self.current.shape.view_mask;
                match (pending.is_some(), &mut self.current.kept) {
                    (true, _) => storage_barrier = Some(view_mask),
                    // An instance recorded inline after a split keeps it with its draws
                    // (`render/inline_splits.rs`).
                    (false, Some(kept)) => {
                        kept.storage_barrier(view_mask);
                    }
                    // SAFETY: the secondary command buffer (or the pass's, inline) is recording
                    // inside the compatible pass, which `render_target` made, and only this
                    // encoder records it.
                    (false, None) => unsafe {
                        render_target::between_storage_draws(
                            device.shared().vk(),
                            self.current.draws,
                            view_mask,
                        )
                    },
                }
            }
            // A draw that writes what an earlier draw of the instance read as its indirect
            // arguments records after an execution dependency from those reads to its writes'
            // stages (`render_target::after_indirect_reads`), as Metal's draw read them first.
            // Instances follow one another behind a barrier from everything before.
            let written = self.current.indirect_reads.written_in(uses) & PASS_DRAW_STAGES;
            let mut indirect_barrier = None;
            if !written.is_empty() {
                let view_mask = self.current.shape.view_mask;
                let stages = tracker::legacy_stages(written, vk::PipelineStageFlags::ALL_GRAPHICS);
                match (pending.is_some(), &mut self.current.kept) {
                    (true, _) => indirect_barrier = Some((view_mask, stages)),
                    (false, Some(kept)) => kept.indirect_barrier(view_mask, stages),
                    // SAFETY: as for the storage barrier; the stages are those of the draw's
                    // writes, which its pipeline has.
                    (false, None) => unsafe {
                        render_target::after_indirect_reads(
                            device.shared().vk(),
                            self.current.draws,
                            view_mask,
                            stages,
                        )
                    },
                }
            }
            self.current.indirect_reads.add(uses);
            // A draw whose fragment shader reads or writes what an earlier draw of the instance
            // wrote before its fragments records after a memory dependency from those writes
            // (`render_target::after_geometry_writes`): Metal runs every draw's vertex, object
            // and mesh functions before any draw's fragments.
            let mut geometry_barrier = None;
            if let Some(barrier) = self.current.geometry_writes.barrier_before(uses) {
                let view_mask = self.current.shape.view_mask;
                let scopes = render_target::GEOMETRY_SELF_DEPENDENCY;
                let mut legacy = barrier.legacy();
                legacy.src_access &= scopes.src_access;
                legacy.dst_access &= scopes.dst_access;
                match (pending.is_some(), &mut self.current.kept) {
                    (true, _) => geometry_barrier = Some((view_mask, legacy)),
                    (false, Some(kept)) => kept.geometry_barrier(view_mask, legacy),
                    // SAFETY: as for the storage barrier; the source stages are those of the
                    // earlier draws' writes, which their pipelines have, and the access types
                    // are within the dependency's.
                    (false, None) => unsafe {
                        render_target::after_geometry_writes(
                            device.shared().vk(),
                            self.current.draws,
                            view_mask,
                            legacy,
                        )
                    },
                }
            }
            self.current.geometry_writes.add(uses);
            if imageblock_use.is_some() {
                self.current.imageblock = true;
            }
            // The barrier orders every earlier storage write of the instance before the draw.
            self.current.interlocked_writes = match after_writes {
                true => interlocked_writes,
                false => interlocked_writes.or(self.current.interlocked_writes),
            };
            if resumed.is_some() {
                self.visibility
                    .end(device.shared().vk(), self.current.draws);
            }
            // The instance's secondary ends before the draw's own, and a new one goes on after
            // it, the active query continued there and lent to the draw.
            let split = pending.map(|(making, state, (draws, after), loan)| {
                let before = self.current.draws;
                self.current.draws = after;
                let raw = device.shared().vk();
                let query = loan.map(|loan| self.visibility.lend_to(raw, before, after, loan));
                self.current.pieces.push(Piece::Draws(before));
                self.current.pieces.push(Piece::Pending(self.pending.len()));
                (making, state, draws, query)
            });
            let made_now = match &draw_pipeline {
                DrawPipeline::Made(made) => Some(made.raw),
                DrawPipeline::Making(_) => None,
            };
            let mut owned = Vec::new();
            let mut states_recorded = Recorded::default();
            // A draw read through snapshots records its recordings twice: every snapshot draw,
            // then every drawing draw (`render/snapshot.rs`).
            let phases: &[Option<snapshot::Phase>] = match pair_base {
                Some(_) => &[
                    Some(snapshot::Phase::Snapshot),
                    Some(snapshot::Phase::Drawing),
                ],
                None => &[None],
            };
            // An emulated tessellation draw records its chunks in place of the amplifications:
            // each chunk's push writes, then its draw.
            let entries = match chunked.is_empty() {
                true => pushes.len(),
                false => chunked.len(),
            };
            let entry = |a: usize| -> (&[binding::PushWrite], &[_]) {
                match chunked.get(a) {
                    Some((push, draw)) => (&push[..], &draw[..]),
                    None => (&pushes[a][..], vk_draws),
                }
            };
            for &phase in phases {
                let mut first = true;
                // Amplification by amplification, each through its recordings' regions: Metal
                // renders every primitive of one before the next's, and a pixel's regions are one.
                // An emulated tessellation draw's chunks, likewise, in order.
                for a in 0..entries {
                    let (push, vk_draws) = entry(a);
                    for state in draw_states.of(a) {
                        for &(index, indirect, draw) in vk_draws {
                            if indirect.is_some()
                                && phase != Some(snapshot::Phase::Snapshot)
                                && matches!(
                                    draw,
                                    metal2vulkan::vk::DrawCommand::DrawIndirect { .. }
                                        | metal2vulkan::vk::DrawCommand::DrawIndexedIndirect { .. }
                                )
                            {
                                device.shared().counts.pool_compiled(
                                    crate::work_counts::PoolCompiled::IndirectDraw,
                                    1,
                                );
                            }
                            let pair = match (&pair_base, phase) {
                                (Some(base), Some(phase)) => {
                                    let at_first = first;
                                    first = false;
                                    Some(snapshot::Pair {
                                        phase,
                                        first: at_first,
                                        ..base.clone()
                                    })
                                }
                                _ => None,
                            };
                            let recording = Recording {
                                raw: device.shared().vk(),
                                command_buffer: self.current.draws,
                                layout: objects.layout(),
                                shape: objects.set_zero_shape(),
                                pipeline: made_now.unwrap_or_default(),
                                sets: set.as_slice(),
                                offsets: &live.offsets,
                                pushed,
                                push_descriptor: device.shared().push_descriptor,
                                pushes_after_pipeline_binds: (device.shared())
                                    .pushes_after_pipeline_binds,
                                heap: heap_set,
                                push,
                                vertex_buffers,
                                index,
                                indirect,
                                dynamic,
                                topology,
                                primitive_restart,
                                state,
                                extended,
                                mesh: device.shared().mesh_draws.as_ref(),
                                draw,
                                pair,
                            };
                            match (made_now, &mut self.current.kept) {
                                // An instance recorded inline after a split keeps its draws until
                                // its copies are known (`render/inline_splits.rs`).
                                (Some(made), Some(kept)) => kept.draw(&recording, made),
                                // SAFETY: the secondary command buffer (or the pass's, inline) is
                                // recording inside the compatible pass, and only this encoder records
                                // it; every object recorded is live until the command buffer
                                // completes (retained below, and the index bounds pass's uploads with
                                // the command buffer); the record is of what the draws recorded into
                                // that command buffer bound there: the instance's, or the pass's
                                // command buffer's for an inline one.
                                (Some(_), None) => {
                                    let bound = match self.current.inline {
                                        true => &mut self.held_bound,
                                        false => &mut self.current.bound,
                                    };
                                    // A device replay's: each command's draw after it
                                    // (`render/executions/gpu.rs`).
                                    states_recorded += match &self.replayed {
                                        Some(replayed) => unsafe {
                                            replayed.record(&recording, bound, &replay_holes)
                                        },
                                        None => unsafe { recording.record(bound) },
                                    }
                                }
                                (None, _) => owned.push(recording.owned()),
                            }
                        }
                    }
                }
            }
            states_recorded.count(&device.shared().counts);
            if let Some((making, state, draws, query)) = split {
                // The record's times: the pipeline's, then those of the other `VkPipeline`s the
                // draw records, made at the draw, set once the making made it. Every copy of the
                // record shares them, those taken before the command buffer's commit too.
                // Only for a record the device keeps.
                let creation = Arc::new(OnceLock::new());
                if records {
                    let setting = Arc::clone(&creation);
                    let others = created.clone();
                    making.when_finished(Box::new(move |made| {
                        if let Ok(made) = made {
                            let times: Vec<_> =
                                [made.created_in].into_iter().chain(others).collect();
                            // Set once, by its one making.
                            drop(setting.set(times));
                        }
                    }));
                }
                // What a failed making names: the draw's place, and the variant it drew.
                let encoder = self.core.command_buffer().state().open_encoder_place();
                self.pending.push(PendingDraw {
                    draws,
                    pipeline: making,
                    state,
                    encoder,
                    draw: self.draw_calls,
                    variant: variant.key().clone(),
                    specialization: variant.constants(specialization),
                    storage_barrier,
                    indirect_barrier,
                    geometry_barrier,
                    query,
                    recordings: owned,
                    recorded: Mutex::new(None),
                });
                later = Some(creation);
            }
            if let Some(next) = resumed {
                let raw = device.shared().vk();
                self.visibility.resume(raw, self.current.draws, next);
            }
            if let Some(raw) = made_now {
                if !self.layout.sample_positions.is_empty() && self.current.positioned.is_none() {
                    self.current.positioned = Some(raw);
                }
            }
            self.current.recorded = true;
            self.keep_draw(again, (call, assembly, draw), index);
            if pushes.len() > 1 {
                let count = pushes.len();
                device.degrade(Degradation::AmplificationDraws, || {
                    let object = format!("a draw of {count} vertex amplifications");
                    let reason = "Vulkan multiview does not draw them as Metal does: their view \
                                  mappings, their count past the device's views, a pipeline \
                                  that chooses or reads its slice, a visibility result mode, \
                                  whose query a multiview instance would take one per view, or \
                                  a rasterization-rate map whose layers differ, each slice \
                                  drawn by its own layer's regions"
                        .to_owned();
                    (object, reason)
                });
            }
        }
        let padding = index.and_then(DrawSource::padding);
        if let (Some(source), Some(positions)) = (index, padding) {
            let (buffer, offset) = source.metal();
            if offset >= buffer.length() {
                device.degrade(Degradation::ZeroedIndexFetch, || {
                    let object = format!(
                        "an indexed draw from offset {offset} of an index buffer of {} bytes",
                        buffer.length()
                    );
                    let reason = format!(
                        "its offset lies at or past the buffer's length, so each of its \
                         {positions} positions draws index 0 plus the base vertex"
                    );
                    (object, reason)
                });
            }
        }
        // The draw binds the variant: the kinds its making announces at every occurrence, but
        // where the encoder records a draw again (`render/reroute.rs`), which announced them.
        if let (Some(tested), false) = (&bound_each, self.rerouting) {
            let formats = |slot| tested.iter().find(|(s, _)| *s == slot)?.1;
            let compiled = variant.pipeline();
            objects.announce_bound(
                (&compiled.stages, CompiledPipeline::Render(compiled)),
                [(specialization, Some(&formats as BoundFormats<'_>))],
            );
        }
        // A pipeline being made: its creation times once made, and the pass holds it
        // (`render/pending.rs`).
        // Built only where the device keeps records.
        self.core.executed(|| {
            let executed = ExecutedPipeline::Render(Arc::clone(variant.shared_pipeline()));
            let execution = match later {
                None => Execution::new(executed, shape, created),
                Some(creation) => Execution::made_later(executed, shape, creation),
            };
            execution.with_index_padding(padding)
        });
        let made = match draw_pipeline {
            DrawPipeline::Made(made) => Some(made),
            DrawPipeline::Making(_) => None,
        };
        self.last_retained
            .retain_draw(&mut self.core, (pipeline, objects, made));
        self.vertex.retain_exclusive(&mut self.core);
        self.fragment.retain_exclusive(&mut self.core);
        self.object.retain_exclusive(&mut self.core);
        self.mesh.retain_exclusive(&mut self.core);
        for source in index.into_iter().chain(indirect) {
            self.core.retain_shared(source.buffer.held());
        }
        if let Some(factors) = factors {
            self.core.retain_shared(factors.buffer.held());
        }
        // The pass's tile memory, untracked: live until the command buffer completes.
        if let Some(buffer) = tile_buffer {
            self.core.retain(buffer);
        }
        for (buffer, _) in buffers
            .patch_indices
            .into_iter()
            .chain(buffers.control_point_indices)
        {
            self.core.retain_shared(buffer.held());
        }
        self.last_retained
            .retain_samplers(&mut self.core, &mut samplers);
        if let Some(unlisted) = unlisted {
            heap_sets::degrade_unlisted(&device, unlisted, "a render pipeline state's draws");
            // It runs with the states its sampler arrays hold.
            let mut held = self.core.command_buffer().state();
            held.unlisted_argument_sampler(&device);
        }
        // The tables the heap's function-table array may name: their IDs stay theirs until the
        // command buffer completes.
        if heap.is_some() && !function_tables.is_empty() {
            self.core.retain(function_tables);
        }
        Ok(())
    }

    /// The encoder's part of what a draw's shape is derived from (`render/draw_shape.rs`).
    fn shape_inputs(&self) -> ShapeInputs<'_> {
        ShapeInputs {
            device: self.core.device(),
            layout: &self.layout,
            state: &self.state,
            amplification: &self.amplification,
            counts_visibility: self.counts_visibility(),
            rate_map: self.rate_map.as_ref(),
            routed: self.routed.as_ref(),
            versions: [
                self.layout.version(),
                self.state.version(),
                self.amplification.version(),
            ],
        }
    }

    /// [`ShapeInputs::settings`] of a draw of `pipeline` as `assembly`, which gives the pass its
    /// hidden depth and stencil where its state needs them ([`needed_hidden`](Self::needed_hidden)).
    fn draw_settings(
        &self,
        call: &'static str,
        pipeline: &DrawingState,
        assembly: Assembly,
        announce: bool,
    ) -> Result<DrawSettings, Error> {
        let hidden = self.needed_hidden()?.is_some();
        self.shape_inputs()
            .settings(call, (pipeline, assembly), hidden, announce)
    }

    /// A draw's settings ([`draw_settings`](Self::draw_settings)), with the hidden depth and
    /// stencil it gives the pass: those an earlier draw of the encoder derived from the same
    /// inputs, else derived and kept (`render/draw_shape.rs`); with the token of their memo
    /// entry, `None` where their derivation met a notice the device announces at every
    /// occurrence ([`EachWatch`]), which keeps nothing.
    fn shaped_settings(
        &mut self,
        call: &'static str,
        pipeline: &DrawingState,
        assembly: Assembly,
    ) -> Result<(DrawSettings, Option<Target>, Option<ShapeToken>), Error> {
        let hidden = self.needed_hidden()?;
        let given = hidden.is_some();
        let inputs = self.shape_inputs();
        let inputs = (inputs.versions, inputs.counts_visibility);
        if let Some((kept, token)) = self.shape_memo.settings((pipeline, assembly), inputs) {
            debug_assert!(
                (self.shape_inputs())
                    .settings(call, (pipeline, assembly), given, false)
                    .is_ok_and(|fresh| fresh == kept),
                "a draw's settings derived again differ from those kept for its inputs"
            );
            return Ok((kept, hidden, Some(token)));
        }
        let watch = EachWatch::start();
        let settings = self
            .shape_inputs()
            .settings(call, (pipeline, assembly), given, true)?;
        // Derived again at each draw, which announces it again.
        if watch.met() {
            return Ok((settings, hidden, None));
        }
        let token = (self.shape_memo).keep_settings((pipeline, assembly), inputs, &settings);
        Ok((settings, hidden, Some(token)))
    }

    /// The address-mode maps a draw of `pipeline` binds its runtime samplers with
    /// (`address_map.rs`): of its stage tables, `pass_start`'s or the encoder's, and of what the
    /// encoder made resident.
    fn sampler_maps(
        &self,
        pipeline: &DrawingState,
        pass_start: &PassStartTables,
    ) -> Result<SamplerMaps, Error> {
        address_map::sampler_maps(
            &pipeline.base().layout,
            &[
                (Stage::Vertex, pass_start.or(Stage::Vertex, &self.vertex)),
                (Stage::Fragment, &self.fragment),
                (Stage::Object, pass_start.or(Stage::Object, &self.object)),
                (Stage::Mesh, pass_start.or(Stage::Mesh, &self.mesh)),
            ],
            Some(&self.residency),
        )
        .map_err(Error::NotImplemented)
    }

    /// Bring `live`, the encoder's kept bind state, to the bind state a draw of `pipeline` under
    /// `settings` binds in, reading its indices as `primitive_id_indices` says (`BindState::
    /// primitive_id_indices`, [`primitive_id_indices`]), a private type's list as `private` says
    /// (`BindState::quad_primitive_ids` and `private_list`, [`PrivateForm`]),
    /// its triangles' barycentric weights and an emulated mesh's corners as `provoking` says
    /// (`BindState::rotated_weights` and `mesh_last_first`, `render/provoking.rs`),
    /// its vertex, object and mesh tables
    /// `pass_start`'s or the encoder's
    /// ([`refresh_bind_state`](Self::refresh_bind_state)), its runtime samplers' address modes
    /// `sampler_maps` ([`sampler_maps`](Self::sampler_maps)), and answer the variant it runs on in
    /// the encoder's buffer forms ([`BufferForms`]) with its plan and the state's shape: of the
    /// device's sampler states in `argument_samplers`, those at IDs no set-1 sampler array of the
    /// variant holds left out (`heap_sets::fit_samplers`, `unlisted` saying so), and
    /// `function_tables` the pipeline's tables. The shape is the kept one where nothing brought
    /// in may have changed it (`live_bind.rs`).
    fn settle_draw(
        &self,
        live: &mut LiveBind,
        (pipeline, settings, primitive_id_indices, private, provoking): (
            &DrawingState,
            &DrawSettings,
            Option<IndexType>,
            PrivateForm,
            ProvokingForms,
        ),
        (pass_start, sampler_maps): (&PassStartTables, &SamplerMaps),
        (argument_samplers, unlisted): (&mut ArgumentSamplers, &mut Option<(u64, u64)>),
        (function_tables, snapshot): (&[VisibleFunctionTable], &mut Option<snapshot::Wanted>),
    ) -> Result<Settled, Error> {
        // The pass's extended-range colors' shadows, which a variant that writes or reads them
        // reads and writes beside the codes (`render/extended_shadow.rs`).
        let mut shadows = self.extended_shadows_bound();
        // From the state the pipeline state's last draw left, where the encoder set it aside.
        live.select(&pipeline.base().interface);
        loop {
            let mut reshape = self.refresh_bind_state(
                live,
                pass_start,
                (argument_samplers, function_tables),
                (pipeline, sampler_maps),
            );
            let state = &mut live.state;
            // Each field the draw's settings or the loop hold on to compared in place, and copied
            // only where it changed.
            reshape.field_from(&mut state.vertex_amplification, &settings.amplification);
            // The shape reads the compensation's kind alone, and whether a clamp range is set.
            reshape.shaped(
                &mut state.viewport_compensation,
                settings.compensation.map(|c| c.viewport),
                |c| c.map(|c| c.compensation),
            );
            reshape.shaped(
                &mut state.depth_clamp_range,
                settings.written_clamp.map(|w| w.range),
                Option::is_some,
            );
            let colors = &settings.routed_colors;
            reshape.field_from(&mut state.zeroed_colors, &colors.zeroed);
            reshape.field(&mut state.primitive_id_indices, primitive_id_indices);
            reshape.field(&mut state.quad_primitive_ids, private.quad_primitive_ids);
            reshape.field(&mut state.private_list, private.private_list);
            reshape.field(&mut state.rotated_weights, provoking.rotated_weights);
            reshape.field(&mut state.mesh_last_first, provoking.mesh_last_first);
            reshape.field_from(&mut state.dropped_colors, &colors.dropped);
            reshape.field_from(&mut state.extended_shadows, shadows);
            let snapshots = snapshot.as_ref().map_or(&NO_INDICES, |w| &w.slots);
            reshape.field_from(&mut state.snapshot_textures, snapshots);
            let depth_stencil = match &settings.failing_state {
                Some(failing) => failing.failing_depth(),
                None => self.state.depth_stencil_descriptor(),
            };
            // No shape reads the depth/stencil state but through the fragment tests (below).
            state.depth_stencil = depth_stencil;
            let stores = !shadows.is_empty();
            let tests = self.fragment_tests(pipeline, &state.depth_stencil, stores);
            reshape.field(&mut state.fragment_tests, tests);
            let tiles = settings.tiles.map(|(tile, area)| {
                let memory = self
                    .tile_memory
                    .bind_state(self.layout.tile.threadgroup_memory_length);
                (ImageblockSize::new(tile[0], tile[1]), area, memory)
            });
            let (imageblock_size, render_area, tile_threadgroup) = tiles.unwrap_or_default();
            reshape.field(&mut state.imageblock_size, imageblock_size);
            reshape.field(&mut state.render_area, render_area);
            reshape.field(&mut state.tile_threadgroup, tile_threadgroup);
            live.reshaped(reshape);
            let counts = &self.core.device().shared().counts;
            let shape = live.shape(counts, |state| pipeline.kept_shape(state));
            // Through what the pipeline state settled lately (`GraphicsState::settle_shape`).
            let settled = match pipeline.settle_shape(&shape, self.forms) {
                // A variant that reads through snapshots the compile or bind layer refuses: the
                // draw takes the split (`render/snapshot.rs`).
                Err(Error::Unsupported(_) | Error::Bind(BindError::Unsupported(_)))
                    if snapshot.is_some() =>
                {
                    let reason = "the variant that reads through snapshots is refused";
                    snapshot::snapshot_fallback(self.core.device(), reason);
                    *snapshot = None;
                    continue;
                }
                // A variant whose shadows the storage form or the bind layer refuses (a mesh
                // pipeline's, an amplified draw's): the draw binds none, and its caller gives
                // them up (`give_up_extended_shadows`); a draw recorded again fails instead.
                Err(refused @ (Error::Unsupported(_) | Error::Bind(BindError::Unsupported(_))))
                    if !shadows.is_empty() =>
                {
                    if self.rerouting {
                        return Err(refused);
                    }
                    shadows = &NO_INDICES;
                    continue;
                }
                settled => settled?,
            };
            // A plan that does not read through every snapshot the draw asked for: the draw
            // takes the split, settled again without them.
            if let Some(wanted) = snapshot.as_ref() {
                if let Err(reason) = self.snapshot_holds(&settled, wanted) {
                    snapshot::snapshot_fallback(self.core.device(), reason);
                    *snapshot = None;
                    continue;
                }
            }
            let set = settled.variant.pipeline().layout.sets.get(1);
            if argument_samplers.fit(set, unlisted) {
                return Ok(settled);
            }
            // A variant of set-1 arrays writes the listed states' elements: it takes the states,
            // listed again (alike, where the kept listing is current) against them.
            argument_samplers.take(self.core.device());
        }
    }
}

impl RenderCommandEncoder {
    /// `setRenderPipelineState:`. Refused (`Invalid`, measured: Metal's validation stops the
    /// call): a color pixel format other than the pass's, as `DrawProblem::AttachmentFormat`
    /// says (plain Metal stores each place of the state's per-sample record into the pass's
    /// attachment at that place, writes zero bytes at every sample a draw covers to a pass color
    /// at no place where its code reads no attachment and keeps it otherwise, and drops the
    /// output of a state color at no place, as the runtime does; it draws a color whose format is
    /// the pass's in another channel order, such as `RGBA8Unorm` into `BGRA8Unorm`, in the pass's
    /// channel order; it takes the pass's depth and stencil attachments whatever depth and stencil
    /// formats the state names), the raster sample count other than the pass's
    /// (`DrawProblem::SampleCount`; plain Metal draws a render pipeline's state of more samples in
    /// a pass of one as a state of one sample, as the runtime does, but for alpha to coverage and
    /// `get_num_samples()`, which read the state's count, and stay refused), and a tile pipeline's
    /// state whose function aliases an explicit imageblock onto its implicit one, or a drawing
    /// state whose fragment function declares an explicit imageblock, of more imageblock bytes per
    /// sample than the pass gives (`DrawProblem::ImageblockSampleLength`; a tile state whose
    /// explicit imageblock stands alone runs there, as plain Metal runs it). Refused
    /// `NotImplemented` where the formats agree in Metal's terms but not in Vulkan's, and for a
    /// state of another device; `Unsupported` (`IndependentBlending`) where the pass has a color
    /// the state keeps unwritten and the device lacks `independentBlend`. Where the device draws
    /// the pass's colors in groups (`render/color_groups.rs`), refused `NotImplemented`
    /// (`ColorAttachmentGroups`) for a drawing state whose colors with a pixel format are not the
    /// pass's colors with a texture, and for a tile pipeline's state; and for a drawing state the
    /// device draws in groups in a pass it draws in one. Metal's validation stops a state whose
    /// color formats are not the pass's attachments' where both name one, as above; where one side
    /// names none it draws (a color without an attachment nowhere, a pass color without a format
    /// as zeroes), which the groups' variants do not state. A refused call keeps the state set
    /// before.
    pub fn set_render_pipeline_state(&mut self, state: &RenderPipelineState) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_render_pipeline_state(state)),
            false => self.defer(|log| RenderCall::PipelineState(log.hold_render_state(state))),
        }
    }

    /// `setVertexBuffer:offset:atIndex:` (`None` unbinds the index). Refused: an index past the
    /// table, an offset past the buffer's length, a buffer of another device.
    pub fn set_vertex_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_vertex_buffer(buffer, offset, index)),
            false => self.defer(|log| {
                let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
                RenderCall::Buffer(Table::Vertex, buffer, offset, index)
            }),
        }
    }

    /// `setVertexBuffers:offsets:withRange:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_buffer`] is, and when the slices' lengths differ; a
    /// refusal binds none of them.
    pub fn set_vertex_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_vertex_buffers(buffers, offsets, first)),
            false => self.defer(|log| {
                RenderCall::Buffers(
                    Table::Vertex,
                    log.buffer_span(buffers),
                    log.word_span(offsets),
                    first,
                )
            }),
        }
    }

    /// `setVertexBufferOffset:atIndex:`, which clears the index's attribute stride
    /// (`render/vertex_strides.rs`). At an index holding a visible-function table, the stage then
    /// reads the null table there (size 0, every entry null) until a table or buffer is bound at
    /// the index again, as plain Metal reads it. At an index holding inline bytes
    /// ([`set_vertex_bytes`](Self::set_vertex_bytes)), they then read as zeros at any offset until
    /// the index is bound again, as plain Metal reads them (an M4; its debug layer asserts there is
    /// no buffer). At an index with nothing bound it keeps no offset, as plain Metal keeps none
    /// there: the index stays unbound and the draw reads zeros, byte for byte as it reads an index
    /// never bound (`ArgumentTable::take_buffer_offset`). Refused: an index past the table, an
    /// offset past the length of a buffer bound at the index.
    pub fn set_vertex_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_vertex_buffer_offset(offset, index)),
            false => self.defer(|_| RenderCall::BufferOffset(Table::Vertex, offset, index)),
        }
    }

    /// `setVertexBytes:length:atIndex:`: `bytes`, copied now, at vertex buffer index `index` in
    /// place of anything bound there, so a later write of the caller's memory reaches no draw, as
    /// Metal copies them at the call. Every later draw reads the copy until the index is bound
    /// again, as a buffer argument or as a vertex descriptor's buffer; a layout of
    /// `MTLBufferLayoutStrideDynamic` reads it at stride 0, as a buffer bound without a stride
    /// (`render/vertex_strides.rs`). A slot the stage reaches through a pointer loaded from
    /// memory (one of a device-address form) cannot read inline bytes:
    /// `Bind(MissingDeviceAddress)`, at the draw. Refused: an index past the table, and more than
    /// 4096 bytes, Metal's limit for inline bytes (`Invalid(InlineBytesLength)`).
    pub fn set_vertex_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_vertex_bytes(bytes, index)),
            false => {
                self.defer(|log| RenderCall::Bytes(Table::Vertex, log.byte_span(bytes), index))
            }
        }
    }

    /// `setVertexTexture:atIndex:` (`None` unbinds the index). Refused: an index past the table,
    /// a texture of another device.
    pub fn set_vertex_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_vertex_texture(texture, index)),
            false => self.defer(|log| {
                let texture = texture.map(|texture| log.hold_texture(texture));
                RenderCall::Texture(Table::Vertex, texture, index)
            }),
        }
    }

    /// `setVertexTextures:withRange:`. Refused as [`RenderCommandEncoder::set_vertex_texture`]
    /// is; a refusal binds none of them.
    pub fn set_vertex_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_vertex_textures(textures, first)),
            false => self.defer(|log| {
                RenderCall::Textures(Table::Vertex, log.texture_span(textures), first)
            }),
        }
    }

    /// `setVertexSamplerState:atIndex:` (`None` unbinds the index). Refused: an index past the
    /// table, a state of another device.
    pub fn set_vertex_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_vertex_sampler_state(sampler, index)),
            false => self.defer(|log| {
                let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
                RenderCall::Sampler(Table::Vertex, sampler, index)
            }),
        }
    }

    /// `setVertexSamplerStates:withRange:`, or with `clamps` (minimums and maximums)
    /// `setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_sampler_state`] is, and when the slices' lengths differ;
    /// a refusal binds none of them.
    pub fn set_vertex_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_vertex_sampler_states(samplers, clamps, first))
            }
            false => self.defer(|_| {
                RenderCall::Samplers(Table::Vertex, Samplers::of(samplers, clamps, first))
            }),
        }
    }

    /// `setFragmentBuffer:offset:atIndex:` (`None` unbinds the index). Refused as
    /// [`RenderCommandEncoder::set_vertex_buffer`] is.
    pub fn set_fragment_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_fragment_buffer(buffer, offset, index)),
            false => self.defer(|log| {
                let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
                RenderCall::Buffer(Table::Fragment, buffer, offset, index)
            }),
        }
    }

    /// `setFragmentBuffers:offsets:withRange:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_buffers`] is.
    pub fn set_fragment_buffers(
        &mut self,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_fragment_buffers(buffers, offsets, first)),
            false => self.defer(|log| {
                RenderCall::Buffers(
                    Table::Fragment,
                    log.buffer_span(buffers),
                    log.word_span(offsets),
                    first,
                )
            }),
        }
    }

    /// `setFragmentBufferOffset:atIndex:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_buffer_offset`] is.
    pub fn set_fragment_buffer_offset(&mut self, offset: u64, index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_fragment_buffer_offset(offset, index)),
            false => self.defer(|_| RenderCall::BufferOffset(Table::Fragment, offset, index)),
        }
    }

    /// `setFragmentBytes:length:atIndex:`: as
    /// [`set_vertex_bytes`](Self::set_vertex_bytes), in the fragment stage's table. Refused as
    /// that is.
    pub fn set_fragment_bytes(&mut self, bytes: &[u8], index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_fragment_bytes(bytes, index)),
            false => {
                self.defer(|log| RenderCall::Bytes(Table::Fragment, log.byte_span(bytes), index))
            }
        }
    }

    /// `setFragmentTexture:atIndex:` (`None` unbinds the index). Refused as
    /// [`RenderCommandEncoder::set_vertex_texture`] is.
    pub fn set_fragment_texture(
        &mut self,
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_fragment_texture(texture, index)),
            false => self.defer(|log| {
                let texture = texture.map(|texture| log.hold_texture(texture));
                RenderCall::Texture(Table::Fragment, texture, index)
            }),
        }
    }

    /// `setFragmentTextures:withRange:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_textures`] is.
    pub fn set_fragment_textures(
        &mut self,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_fragment_textures(textures, first)),
            false => self.defer(|log| {
                RenderCall::Textures(Table::Fragment, log.texture_span(textures), first)
            }),
        }
    }

    /// `setFragmentSamplerState:atIndex:` (`None` unbinds the index). Refused as
    /// [`RenderCommandEncoder::set_vertex_sampler_state`] is.
    pub fn set_fragment_sampler_state(
        &mut self,
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_fragment_sampler_state(sampler, index)),
            false => self.defer(|log| {
                let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
                RenderCall::Sampler(Table::Fragment, sampler, index)
            }),
        }
    }

    /// `setFragmentSamplerStates:withRange:`, or with `clamps`
    /// `setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:`. Refused as
    /// [`RenderCommandEncoder::set_vertex_sampler_states`] is.
    pub fn set_fragment_sampler_states(
        &mut self,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self
                .at_call(|lowering| lowering.set_fragment_sampler_states(samplers, clamps, first)),
            false => self.defer(|_| {
                RenderCall::Samplers(Table::Fragment, Samplers::of(samplers, clamps, first))
            }),
        }
    }

    /// `setColorStoreAction:atIndex:`: the store action of color attachment `index`, which the
    /// descriptor left `Unknown`. As plain Metal does where its validation stops the call (measured
    /// on an M4): for an attachment without a texture, or over an action the descriptor knew,
    /// nothing changes; a resolving action where nothing resolves resolves nothing
    /// (`MultisampleResolve` keeps no sample, `StoreAndMultisampleResolve` stores them); and one
    /// that does not resolve where a resolve texture is named leaves that texture untouched. A
    /// memoryless attachment holds no memory, so `Store` stores nowhere and
    /// `StoreAndMultisampleResolve` is its resolve alone, the actions the descriptor's own take
    /// (measured; Metal's End Encoding Validation alone objects to either).
    /// Refused (`Invalid`, measured): an index past the attachments and `Unknown` itself.
    pub fn set_color_store_action(&mut self, action: StoreAction, index: u32) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_color_store_action(action, index)),
            false => self.defer(|_| RenderCall::ColorStoreAction(action, index)),
        }
    }

    /// `setDepthStoreAction:`: the depth attachment's store action, which the descriptor left
    /// `Unknown`; without a depth attachment nothing changes, as plain Metal does where its
    /// validation stops the call (measured on an M4); one that does not resolve where a resolve
    /// texture is named leaves that texture untouched and stores the depth by the action, as plain
    /// Metal does where its validation stops it; a memoryless attachment's `Store` stores nowhere,
    /// as the descriptor's own does (measured). Refused (`Invalid`, measured):
    /// an action over one the descriptor knew, `Unknown` itself, a resolving action without a
    /// resolve texture, and `Store` where the stencil attachment is the same texture resolved by
    /// `MultisampleResolve`.
    pub fn set_depth_store_action(&mut self, action: StoreAction) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_depth_store_action(action)),
            false => self.defer(|_| RenderCall::DepthStoreAction(action)),
        }
    }

    /// `setStencilStoreAction:`: the stencil attachment's store action, which the descriptor
    /// left `Unknown`. Refused as [`RenderCommandEncoder::set_depth_store_action`] is, but for
    /// the last: here `MultisampleResolve` where the depth attachment is the same texture,
    /// stored; and without a stencil attachment (unmeasured).
    pub fn set_stencil_store_action(&mut self, action: StoreAction) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_stencil_store_action(action)),
            false => self.defer(|_| RenderCall::StencilStoreAction(action)),
        }
    }

    /// `setColorStoreActionOptions:atIndex:`: the color attachment's store action options, which
    /// change nothing, as plain Metal's: Apple's GPUs ignore them, and plain Metal takes them over
    /// a store action the descriptor knew and under one that keeps no samples, where its validation
    /// stops them (measured on an M4, `sample_positions.rs`). Refused (`Invalid`): an index past
    /// the attachments and an attachment without a texture (unmeasured).
    pub fn set_color_store_action_options(
        &mut self,
        _options: StoreActionOptions,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_color_store_action_options(_options, index))
            }
            false => self.defer(|_| RenderCall::ColorStoreActionOptions(_options, index)),
        }
    }

    /// `setDepthStoreActionOptions:`: the depth attachment's store action options, where the
    /// descriptor left its store action `Unknown`; they change nothing, as plain Metal's (Metal
    /// probe case `opt_depth_late` on an M4). Refused (`Invalid`, measured): an attachment without
    /// a texture, and one whose descriptor knew its store action.
    pub fn set_depth_store_action_options(
        &mut self,
        options: StoreActionOptions,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_depth_store_action_options(options)),
            false => self.defer(|_| RenderCall::DepthStoreActionOptions(options)),
        }
    }

    /// `setStencilStoreActionOptions:`: the stencil attachment's store action options. Refused
    /// as [`RenderCommandEncoder::set_depth_store_action_options`] is; `end_encoding` refuses
    /// `CustomSamplePositions` under a store action that keeps no samples (unmeasured for a
    /// stencil attachment, `sample_positions.rs`).
    pub fn set_stencil_store_action_options(
        &mut self,
        options: StoreActionOptions,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_stencil_store_action_options(options)),
            false => self.defer(|_| RenderCall::StencilStoreActionOptions(options)),
        }
    }

    /// `setViewport:`. A NaN origin x draws nothing, as plain Metal's where its validation stops it
    /// (measured on an M4). Refused (`Invalid`, measured): any other NaN or infinite field. A depth
    /// range past `[0, 1]` is refused on a device that takes it neither natively nor by the vertex
    /// stage's compensation: `NotImplemented` without the features, `Unsupported` on a driver that
    /// draws the compensated depths inexactly (`render_state.rs`). A refused call keeps the
    /// viewports set before.
    pub fn set_viewport(&mut self, viewport: Viewport) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_viewport(viewport)),
            false => self.defer(|_| RenderCall::Viewport(Box::new(viewport))),
        }
    }

    /// `setViewports:count:`. Refused as [`RenderCommandEncoder::set_viewport`] is, and for a
    /// count outside 1 to 16 (measured). On a device without `multiViewport`, a draw that selects
    /// among more than one viewport (its vertex function writes `[[viewport_array_index]]`, or an
    /// amplification's mapping offsets the viewport) is refused at the draw,
    /// `Unsupported(LayeredRendering)` needing the feature; with one it draws through it.
    pub fn set_viewports(&mut self, viewports: &[Viewport]) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_viewports(viewports)),
            false => self.defer(|_| RenderCall::Viewports(viewports.into())),
        }
    }

    /// `setScissorRect:`. A rectangle past the render area is clipped to it, as plain Metal clips
    /// it where its validation stops the call (measured on an M4).
    pub fn set_scissor_rect(&mut self, rect: ScissorRect) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_scissor_rect(rect)),
            false => self.defer(|_| RenderCall::ScissorRect(rect)),
        }
    }

    /// `setScissorRects:count:`. Each rectangle is clipped to the render area as
    /// [`RenderCommandEncoder::set_scissor_rect`] clips it, and more than 16 are taken, as plain
    /// Metal takes them where its validation stops them (Metal probe case `scissors_17` on an M4).
    /// Refused (`Invalid`): no rectangle (unmeasured). A draw whose pipeline selects no viewport
    /// draws through the first; one that selects draws each viewport through the rectangle of its
    /// index, one rectangle serving every viewport and a viewport past two or more drawing nothing,
    /// as plain Metal draws them where its validation stops a count other than 1 and the viewport
    /// count (probes).
    pub fn set_scissor_rects(&mut self, rects: &[ScissorRect]) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_scissor_rects(rects)),
            false => self.defer(|_| RenderCall::ScissorRects(rects.into())),
        }
    }

    /// `setCullMode:`.
    pub fn set_cull_mode(&mut self, mode: CullMode) {
        match self.lowers_at_call() {
            true => self.at_call_unit(|lowering| lowering.set_cull_mode(mode)),
            false => self.defer_unit(|_| RenderCall::CullMode(mode)),
        }
    }

    /// `setFrontFacingWinding:`.
    pub fn set_front_facing_winding(&mut self, winding: Winding) {
        match self.lowers_at_call() {
            true => self.at_call_unit(|lowering| lowering.set_front_facing_winding(winding)),
            false => self.defer_unit(|_| RenderCall::FrontFacingWinding(winding)),
        }
    }

    /// `setDepthStencilState:`: `state` is a `&DepthStencilState`, or `None` for nil, which
    /// makes the encoder's state Metal's default again (a depth compare of `Always` without
    /// writes, and no stencil test), as plain Metal draws after nil where its validation stops
    /// it. Refused: a state of another device. A draw refuses a state that
    /// writes depth in a pass without a depth attachment, or uses the stencil in one without a
    /// stencil attachment (measured).
    pub fn set_depth_stencil_state<'a>(
        &mut self,
        state: impl Into<Option<&'a DepthStencilState>>,
    ) -> Result<(), Error> {
        let state = state.into();
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_depth_stencil_state(state)),
            false => self.defer(|log| {
                RenderCall::DepthStencilState(state.map(|state| log.hold_depth_stencil(state)))
            }),
        }
    }

    /// `setStencilReferenceValue:`: both faces' reference, saturated at 255 as Metal's is
    /// (measured).
    pub fn set_stencil_reference_value(&mut self, value: u32) {
        match self.lowers_at_call() {
            true => self.at_call_unit(|lowering| lowering.set_stencil_reference_value(value)),
            false => self.defer_unit(|_| RenderCall::StencilReferenceValue(value)),
        }
    }

    /// `setStencilFrontReferenceValue:backReferenceValue:`, saturated at 255 as Metal's are.
    pub fn set_stencil_reference_values(&mut self, front: u32, back: u32) {
        match self.lowers_at_call() {
            true => {
                self.at_call_unit(|lowering| lowering.set_stencil_reference_values(front, back))
            }
            false => self.defer_unit(|_| RenderCall::StencilReferenceValues(front, back)),
        }
    }

    /// `setDepthBias:slopeScale:clamp:`. NaN and infinite factors give Metal's measured depths: a
    /// NaN constant biases as +inf; an unbounded infinite or NaN bias saturates the depth (at 1 or
    /// 0, in depth clamp mode at the viewport's larger or smaller depth); a NaN or infinite clamp
    /// clamps nothing; a clamp that bounds an infinite or NaN bias adds itself to the depth.
    /// Refused `NotImplemented(NonFiniteDepthBias)`: factors whose effect differs between flat and
    /// sloped triangles (an infinite slope factor against an unbounded or opposite clamp, or
    /// against the opposite infinite constant); and at a draw of triangles, a clamp's addition that
    /// moves a viewport's depth range past `[0, 1]` or that depth clamp mode bounds. A draw refuses
    /// a clamp other than 0 on a device without `depthBiasClamp` (`Unsupported`).
    pub fn set_depth_bias(
        &mut self,
        depth_bias: f32,
        slope_scale: f32,
        clamp: f32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_depth_bias(depth_bias, slope_scale, clamp))
            }
            false => self.defer(|_| RenderCall::DepthBias(depth_bias, slope_scale, clamp)),
        }
    }

    /// `setDepthClipMode:`. `Clamp` needs `depthClamp` (`Unsupported`).
    pub fn set_depth_clip_mode(&mut self, mode: DepthClipMode) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_depth_clip_mode(mode)),
            false => self.defer(|_| RenderCall::DepthClipMode(mode)),
        }
    }

    /// `setTriangleFillMode:`. `Lines` needs `fillModeNonSolid` (`Unsupported`).
    pub fn set_triangle_fill_mode(&mut self, mode: TriangleFillMode) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_triangle_fill_mode(mode)),
            false => self.defer(|_| RenderCall::TriangleFillMode(mode)),
        }
    }

    /// `setBlendColorRed:green:blue:alpha:`.
    pub fn set_blend_color(&mut self, red: f32, green: f32, blue: f32, alpha: f32) {
        match self.lowers_at_call() {
            true => self.at_call_unit(|lowering| lowering.set_blend_color(red, green, blue, alpha)),
            false => self.defer_unit(|_| RenderCall::BlendColor([red, green, blue, alpha])),
        }
    }

    /// `setTessellationFactorBuffer:offset:instanceStride:` (`tessellation.rs`): the factors a
    /// patch draw reads, `None` for `nil`, from the offset rounded down to a multiple of 2, each
    /// instance's at the instance stride rounded down to a multiple of 4: Metal's validation stops
    /// an offset or a stride that is not a multiple of 4, but plain Metal reads an offset of 2 at
    /// that byte (Metal probe case `factor_offset_2` on an M4), an odd one from the byte below (1
    /// read from 0, 3 from 2, 7 from 6), and steps the instances by a stride rounded down to a
    /// multiple of 4 (Metal probes: strides 2 and 14 drew what 0 and 12 draw, 6 and 10 what 4 and 8
    /// draw: 13 and 15 what 12 draws, 17 and 19 what 16 draws). Refused (`Invalid`) for a buffer of
    /// another device, keeping the buffer set before.
    pub fn set_tessellation_factor_buffer(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        instance_stride: u64,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.set_tessellation_factor_buffer(buffer, offset, instance_stride)
            }),
            false => self.defer(|_| {
                RenderCall::TessellationFactorBuffer(buffer.cloned(), offset, instance_stride)
            }),
        }
    }

    /// `setTessellationFactorScale:` (`tessellation.rs`), which a pipeline that enables
    /// `tessellationFactorScaleEnabled` multiplies its factors by, rounded to `half` as Metal
    /// rounds it (one infinite as a `half` takes the factors to the maximum, and one that is 0,
    /// negative or NaN as a `half` culls every patch; and a Metal probe
    /// on an M4). Every scale is taken, as plain Metal takes those its validation stops (0, a
    /// negative one, NaN, an infinity and a subnormal `float`; Metal probes on an M4): each culls
    /// or draws as its `half` does under a pipeline that enables the scale, and a pipeline that
    /// does not draws as before.
    pub fn set_tessellation_factor_scale(&mut self, scale: f32) {
        match self.lowers_at_call() {
            true => self.at_call_unit(|lowering| lowering.set_tessellation_factor_scale(scale)),
            false => self.defer_unit(|_| RenderCall::TessellationFactorScale(scale)),
        }
    }

    /// `setVertexAmplificationCount:viewMappings:` (`amplification.rs`): each later draw
    /// processes its vertices `count` times, amplification `a` offset by `view_mappings[a]`
    /// (`None` for `nil`, offsets of 0). Refused (`Invalid`): fewer mappings than `count`. A
    /// draw draws a count above its pipeline's `maxVertexAmplificationCount` as plain Metal does
    /// where its validation stops it, and refuses one above a maximum of 1 or above 8
    /// (`DrawProblem::AmplificationCount`); a refused call keeps the amplification set before.
    pub fn set_vertex_amplification_count(
        &mut self,
        count: u64,
        view_mappings: Option<&[VertexAmplificationViewMapping]>,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self
                .at_call(|lowering| lowering.set_vertex_amplification_count(count, view_mappings)),
            false => self.defer(|_| {
                RenderCall::VertexAmplificationCount(count, view_mappings.map(Into::into))
            }),
        }
    }

    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:` (Metal's shorter
    /// forms take one instance and a base instance of 0), of a public primitive type or one of
    /// Metal's private ones ([`DrawPrimitiveType`]), which draws as the list its variant states,
    /// written on the CPU. A draw of no vertices or no instances records nothing (the module docs).
    /// Refused as the module docs say.
    pub fn draw_primitives(
        &mut self,
        primitive_type: impl Into<DrawPrimitiveType>,
        vertex_start: u32,
        vertex_count: u32,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<(), Error> {
        let primitive_type = primitive_type.into();
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_primitives(
                    primitive_type,
                    vertex_start,
                    vertex_count,
                    instance_count,
                    base_instance,
                )
            }),
            false => self.defer(|_| {
                RenderCall::DrawPrimitives(
                    primitive_type,
                    vertex_start,
                    vertex_count,
                    instance_count,
                    base_instance,
                )
            }),
        }
    }

    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:
    /// baseVertex:baseInstance:` (Metal's shorter forms take one instance and a base vertex and
    /// instance of 0). Index positions past the buffer's length draw as the module docs say, and
    /// the indices are read from the offset's low 32 bits rounded down to a multiple of the index
    /// size, as plain Metal reads them (an M4 read an offset of 2^32 + 16 into a buffer past 4 GiB
    /// at 16). A draw of a private primitive type ([`DrawPrimitiveType`]) draws as the list its
    /// variant states over the indices, which a GPU pass writes before the render pass. A draw of
    /// no indices or no instances records nothing. Refused as the module docs say, and for an
    /// index buffer of another device.
    pub fn draw_indexed_primitives(
        &mut self,
        primitive_type: impl Into<DrawPrimitiveType>,
        index_count: u32,
        index_type: IndexType,
        index_buffer: &Buffer,
        index_buffer_offset: u64,
        instance_count: u32,
        base_vertex: i32,
        base_instance: u32,
    ) -> Result<(), Error> {
        let primitive_type = primitive_type.into();
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_indexed_primitives(
                    primitive_type,
                    index_count,
                    index_type,
                    index_buffer,
                    index_buffer_offset,
                    instance_count,
                    base_vertex,
                    base_instance,
                )
            }),
            false => self.defer_indexed(|log| DrawIndexed {
                primitive_type,
                index_count,
                index_type,
                index_buffer: log.hold_buffer(index_buffer),
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            }),
        }
    }

    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:`: Metal's
    /// `MTLDrawPrimitivesIndirectArguments` at `indirect_buffer_offset`. A draw of a private
    /// primitive type ([`DrawPrimitiveType`]) draws as the list its variant states, which a GPU
    /// pass writes before the render pass from the arguments. Refused as the module docs say, and
    /// for an indirect buffer of another device.
    pub fn draw_primitives_indirect(
        &mut self,
        primitive_type: impl Into<DrawPrimitiveType>,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        let primitive_type = primitive_type.into();
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_primitives_indirect(
                    primitive_type,
                    indirect_buffer,
                    indirect_buffer_offset,
                )
            }),
            false => self.defer(|_| {
                RenderCall::DrawPrimitivesIndirect(
                    primitive_type,
                    indirect_buffer.clone(),
                    indirect_buffer_offset,
                )
            }),
        }
    }

    /// `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:
    /// indirectBufferOffset:`: Metal's `MTLDrawIndexedPrimitivesIndirectArguments` at
    /// `indirect_buffer_offset`, drawing `index_type` indices of `index_buffer` from
    /// `index_buffer_offset`, those past the buffer's length as the module docs say, each offset
    /// rounded down as the direct draws' are. A draw of a private primitive type
    /// ([`DrawPrimitiveType`]) draws as the list its variant states over the indices, which a GPU
    /// pass writes before the render pass. Refused as the module docs say, and for buffers of
    /// another device.
    pub fn draw_indexed_primitives_indirect(
        &mut self,
        primitive_type: impl Into<DrawPrimitiveType>,
        index_type: IndexType,
        index_buffer: &Buffer,
        index_buffer_offset: u64,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        let primitive_type = primitive_type.into();
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_indexed_primitives_indirect(
                    primitive_type,
                    index_type,
                    index_buffer,
                    index_buffer_offset,
                    indirect_buffer,
                    indirect_buffer_offset,
                )
            }),
            false => self.defer(|_| {
                RenderCall::DrawIndexedPrimitivesIndirect(Box::new(DrawIndexedIndirect {
                    primitive_type,
                    index_type,
                    index_buffer: index_buffer.clone(),
                    index_buffer_offset,
                    indirect_buffer: indirect_buffer.clone(),
                    indirect_buffer_offset,
                }))
            }),
        }
    }

    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:
    /// baseInstance:` of a tessellation pipeline (`tessellation.rs`): `patch_count` patches of
    /// `number_of_patch_control_points` control points from `patch_start`, `instance_count`
    /// instances from `base_instance`, patch `i` the one the patch index buffer's `uint` at
    /// `patch_start + i` names where one is given (its factors stay at `patch_start + i`), read
    /// from its offset rounded down to a multiple of 4 as plain Metal reads it. A draw of no
    /// patches or no instances records nothing; the draw's control points are its pipeline's where
    /// its function states them, and a pipeline with a control-point index type reads every control
    /// point's index as 0, as plain Metal draws them. Refused as the module docs say, for a patch
    /// index buffer of another device, and as Metal's validation refuses it (`Invalid`, measured):
    /// indices `patch_start + patch_count` past the buffer from the offset.
    pub fn draw_patches(
        &mut self,
        number_of_patch_control_points: u32,
        patch_start: u32,
        patch_count: u32,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_patches(
                    number_of_patch_control_points,
                    patch_start,
                    patch_count,
                    patch_index_buffer,
                    patch_index_buffer_offset,
                    instance_count,
                    base_instance,
                )
            }),
            false => self.defer(|_| {
                RenderCall::DrawPatches(Box::new(DrawPatches {
                    control_points: number_of_patch_control_points,
                    patch_start,
                    patch_count,
                    patch_index_buffer: patch_index_buffer.cloned(),
                    patch_index_buffer_offset,
                    instance_count,
                    base_instance,
                }))
            }),
        }
    }

    /// `drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:
    /// indirectBufferOffset:` of a tessellation pipeline: Metal's `MTLDrawPatchIndirectArguments`
    /// at `indirect_buffer_offset` (a `VkDrawIndirectCommand`'s layout). Refused as
    /// [`RenderCommandEncoder::draw_patches`] and [`RenderCommandEncoder::draw_primitives_indirect`]
    /// are, but for the patch indices past the buffer, whose count the arguments hold: a patch
    /// whose index lies past the buffer's length reads patch index 0 (Metal reads memory past the
    /// buffer there, which its validation does not check). A pipeline with a control-point index
    /// type, which Metal's validation stops, reads every index as 0, as the direct draw does (a
    /// Metal probe `cpi_plain_typed_indirect`).
    pub fn draw_patches_indirect(
        &mut self,
        number_of_patch_control_points: u32,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_patches_indirect(
                    number_of_patch_control_points,
                    patch_index_buffer,
                    patch_index_buffer_offset,
                    indirect_buffer,
                    indirect_buffer_offset,
                )
            }),
            false => self.defer(|_| {
                RenderCall::DrawPatchesIndirect(Box::new(DrawPatchesIndirect {
                    control_points: number_of_patch_control_points,
                    patch_index_buffer: patch_index_buffer.cloned(),
                    patch_index_buffer_offset,
                    indirect_buffer: indirect_buffer.clone(),
                    indirect_buffer_offset,
                }))
            }),
        }
    }

    /// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:
    /// controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:` of a
    /// tessellation pipeline whose `tessellationControlPointIndexType` is not `None`: as
    /// [`RenderCommandEncoder::draw_patches`], but control point `k` of the patch whose index is
    /// `p` is the index (of the pipeline's type) at position `p · N + k` of the control-point index
    /// buffer from its offset (measured), `N` the pipeline's control points where its function
    /// states them, an offset that is not a multiple of the index size rounded down to one, as
    /// plain Metal reads it, where its validation stops the draw. A pipeline whose type is `None`,
    /// which Metal's validation stops, draws the patches' own control points, the buffer unread, as
    /// plain Metal draws them (Metal probe case `indexed_without_type` on an M4). Refused as
    /// `draw_patches` is, for a control-point index buffer of another device, and as Metal's
    /// validation refuses it (`Invalid`, measured): `(patch_start + patch_count) · N` indices past
    /// the buffer from the offset.
    pub fn draw_indexed_patches(
        &mut self,
        number_of_patch_control_points: u32,
        patch_start: u32,
        patch_count: u32,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        control_point_index_buffer: &Buffer,
        control_point_index_buffer_offset: u64,
        instance_count: u32,
        base_instance: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_indexed_patches(
                    number_of_patch_control_points,
                    patch_start,
                    patch_count,
                    patch_index_buffer,
                    patch_index_buffer_offset,
                    control_point_index_buffer,
                    control_point_index_buffer_offset,
                    instance_count,
                    base_instance,
                )
            }),
            false => self.defer(|_| {
                RenderCall::DrawIndexedPatches(Box::new(DrawIndexedPatches {
                    control_points: number_of_patch_control_points,
                    patch_start,
                    patch_count,
                    patch_index_buffer: patch_index_buffer.cloned(),
                    patch_index_buffer_offset,
                    control_point_index_buffer: control_point_index_buffer.clone(),
                    control_point_index_buffer_offset,
                    instance_count,
                    base_instance,
                }))
            }),
        }
    }

    /// `drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:
    /// controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:` of a tessellation
    /// pipeline whose `tessellationControlPointIndexType` is not `None`: the arguments of
    /// [`RenderCommandEncoder::draw_patches_indirect`], the control points of
    /// [`RenderCommandEncoder::draw_indexed_patches`]. Refused as both are, but for the indices
    /// past the buffers, whose counts the arguments hold: an index that lies past the buffer's
    /// length reads 0 (Metal reads memory past the buffer there, which its validation does not
    /// check). A pipeline without a control-point index type, which Metal's validation stops, draws
    /// the patches' own control points, the buffer unread, as the direct draw does (a Metal probe
    /// `cpi_indexed_untyped_indirect`).
    pub fn draw_indexed_patches_indirect(
        &mut self,
        number_of_patch_control_points: u32,
        patch_index_buffer: Option<&Buffer>,
        patch_index_buffer_offset: u64,
        control_point_index_buffer: &Buffer,
        control_point_index_buffer_offset: u64,
        indirect_buffer: &Buffer,
        indirect_buffer_offset: u64,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.draw_indexed_patches_indirect(
                    number_of_patch_control_points,
                    patch_index_buffer,
                    patch_index_buffer_offset,
                    control_point_index_buffer,
                    control_point_index_buffer_offset,
                    indirect_buffer,
                    indirect_buffer_offset,
                )
            }),
            false => self.defer(|_| {
                RenderCall::DrawIndexedPatchesIndirect(Box::new(DrawIndexedPatchesIndirect {
                    control_points: number_of_patch_control_points,
                    patch_index_buffer: patch_index_buffer.cloned(),
                    patch_index_buffer_offset,
                    control_point_index_buffer: control_point_index_buffer.clone(),
                    control_point_index_buffer_offset,
                    indirect_buffer: indirect_buffer.clone(),
                    indirect_buffer_offset,
                }))
            }),
        }
    }

    /// `setVisibilityResultMode:offset:` (`visibility.rs`). An offset that is not a multiple of 8
    /// is rounded down to one, where plain Metal writes the result (measured on an M4). Refused
    /// (`Invalid`, measured) for `Boolean` and `Counting`: a pass without a visibility result
    /// buffer, an offset whose 8 bytes run past the buffer, an offset the encoder took before.
    /// `Counting` is `Unsupported` where the device lacks `occlusionQueryPrecise`. A refused call
    /// keeps the mode set before. While a mode other than `Disabled` is set, amplified draws draw
    /// one amplification at a time, whose samples the one query counts as Metal's does
    /// (`VertexAmplification::one_at_a_time`); a mode set after a draw amplified as views begins
    /// its query in a new instance of the pass's layers, since a query in a multiview instance
    /// takes one query per view. Where the device draws the pass's colors in groups, the first
    /// group's pass counts the results (`render/color_groups.rs`).
    pub fn set_visibility_result_mode(
        &mut self,
        mode: VisibilityResultMode,
        offset: u64,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_visibility_result_mode(mode, offset)),
            false => self.defer(|_| RenderCall::VisibilityResultMode(mode, offset)),
        }
    }

    /// `endEncoding`: begin the pass with its load and store operations, run the draws, and end it.
    /// A store action still `Unknown` is `DontCare`, as plain Metal stores it where its validation
    /// stops it: a color attachment's (measured on an M4), and a depth or stencil attachment's,
    /// resolving or not; a resolve texture is left untouched, as plain Metal leaves it. The encoder
    /// is consumed either way, so a refusal leaves its command buffer abandoned, as Metal stops
    /// there.
    pub fn end_encoding(self) -> Result<(), Error> {
        self.end()
    }
}

/// Color attachment indices, a bit each: the sets of colors a pass's end works out, which would
/// otherwise allocate at every end. Every index is below [`MAX_COLOR_ATTACHMENTS`], as a pass's
/// colors are.
#[derive(Clone, Copy)]
struct ColorIndices(u8);

const _: () = assert!(MAX_COLOR_ATTACHMENTS <= u8::BITS as usize);

impl ColorIndices {
    /// The indices `indices` yields.
    fn of(indices: impl IntoIterator<Item = usize>) -> ColorIndices {
        ColorIndices((indices.into_iter()).fold(0, |bits, index| bits | 1 << index))
    }

    /// Whether it holds `index`.
    fn contains(self, index: usize) -> bool {
        index < MAX_COLOR_ATTACHMENTS && self.0 & 1 << index != 0
    }
}

/// Everything one draw records into the secondary command buffer.
struct Recording<'a> {
    raw: &'a ash::Device,
    command_buffer: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    /// The shape of the layout's set 0 and push-constant ranges
    /// (`VariantObjects::set_zero_shape`).
    shape: u64,
    pipeline: vk::Pipeline,
    sets: &'a [vk::DescriptorSet],
    /// The dynamic offsets set 0 is bound with, one per binding the variant makes dynamic
    /// (`binding::dynamic_writes`).
    offsets: &'a [u32],
    /// Set 0's descriptors, where the variant pushes set 0 (then `sets` is empty), and the push
    /// command.
    pushed: &'a [binding::SetWrite],
    push_descriptor: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
    /// After a pipeline bind the draw's push of set 0 carries at least one write
    /// (`DeviceShared::pushes_after_pipeline_binds`).
    pushes_after_pipeline_binds: bool,
    /// Set 1, the command buffer's heap set of the variant, when the variant has one.
    heap: Option<vk::DescriptorSet>,
    push: &'a [binding::PushWrite],
    /// `(binding, buffer, offset, stride)` of each vertex buffer; the stride is used where the
    /// pipeline makes it dynamic.
    vertex_buffers: &'a [(u32, vk::Buffer, u64, u64)],
    /// An indexed draw's index buffer.
    index: Option<IndexBind>,
    /// An indirect draw's argument buffer.
    indirect: Option<vk::Buffer>,
    /// The dynamic states the pipeline holds: its own, but for those of a part it lacks (the
    /// fragment states of a pipeline that discards rasterization, `fragment_state`).
    dynamic: &'a [vk::DynamicState],
    /// The draw's input assembly, which the pipeline takes as dynamic state where it makes the
    /// topology dynamic.
    topology: metal2vulkan::vk::PrimitiveTopology,
    primitive_restart: bool,
    state: &'a DrawState,
    extended: Option<super::dynamic_state::ExtendedDynamicState>,
    /// The device's mesh draw commands, where it enabled `meshShader`.
    mesh: Option<&'a crate::recorder::MeshDraws>,
    draw: metal2vulkan::vk::DrawCommand,
    /// Its part in a draw read through snapshots (`render/snapshot.rs`): the barrier, write
    /// masks and phase it records before its draw.
    pair: Option<snapshot::Pair>,
}

impl Recording<'_> {
    /// The recording, owned, to record once its pipeline is made (`render/pending.rs`).
    fn owned(&self) -> OwnedRecording {
        OwnedRecording {
            layout: self.layout,
            shape: self.shape,
            sets: self.sets.to_vec(),
            offsets: self.offsets.to_vec(),
            pushed: self.pushed.to_vec(),
            heap: self.heap,
            push: self.push.to_vec(),
            vertex_buffers: self.vertex_buffers.to_vec(),
            index: self.index,
            indirect: self.indirect,
            dynamic: self.dynamic.to_vec(),
            topology: self.topology,
            primitive_restart: self.primitive_restart,
            state: self.state.clone(),
            extended: self.extended,
            draw: self.draw,
            pair: self.pair.clone(),
        }
    }

    /// [`owned`](Self::owned), over `into`'s storage: where an earlier draw's recording left room
    /// enough, it allocates nothing (`inline_splits::SpareKept`).
    fn own_into(&self, into: &mut OwnedRecording) {
        fn refill<T: Copy>(into: &mut Vec<T>, from: &[T]) {
            into.clear();
            into.extend_from_slice(from);
        }
        into.layout = self.layout;
        into.shape = self.shape;
        refill(&mut into.sets, self.sets);
        refill(&mut into.offsets, self.offsets);
        refill(&mut into.pushed, self.pushed);
        into.heap = self.heap;
        into.push.truncate(self.push.len());
        for (at, (stages, offset, bytes)) in self.push.iter().enumerate() {
            match into.push.get_mut(at) {
                Some(write) => {
                    (write.0, write.1) = (*stages, *offset);
                    refill(&mut write.2, bytes);
                }
                None => into.push.push((*stages, *offset, bytes.clone())),
            }
        }
        refill(&mut into.vertex_buffers, self.vertex_buffers);
        into.index = self.index;
        into.indirect = self.indirect;
        refill(&mut into.dynamic, self.dynamic);
        into.topology = self.topology;
        into.primitive_restart = self.primitive_restart;
        into.state = self.state.clone();
        into.extended = self.extended;
        into.draw = self.draw;
        into.pair = self.pair.clone();
    }

    /// Record the draw, leaving out each state command that sets what `bound`, the record of its
    /// command buffer, holds (`render/bound.rs`); the state commands it recorded.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording inside a pass compatible with the pipeline's, every handle
    /// is live until the command buffer completes, and `bound` is the record of what the draws
    /// recorded into it bound (each recorded through it). With a pair, the pass is one
    /// `render_target` made, of no views (`snapshot::Pair::record`).
    unsafe fn record(&self, bound: &mut Bound) -> Recorded {
        let (raw, cb) = (self.raw, self.command_buffer);
        let dynamic = |state| self.dynamic.contains(&state);
        bound.follow(cb);
        let mut recorded = 0;
        let mut writes_pushed = 0;
        let mut equal_shapes = EqualShapes::default();
        // SAFETY: the caller's contract; the extended commands exist wherever the pipeline makes
        // their states dynamic (checked by the draw).
        unsafe {
            if bound.pipeline(
                self.pipeline,
                self.dynamic,
                self.pushes_after_pipeline_binds,
            ) {
                raw.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                recorded += 1;
            }
            if let Some(push) = self.push_descriptor.filter(|_| !self.pushed.is_empty()) {
                let (writes, if_shared) = bound.pushed_set(self.layout, self.shape, self.pushed);
                writes_pushed = writes.len() as u64;
                if let Some(if_shared) = if_shared {
                    equal_shapes = EqualShapes {
                        switches: 1,
                        carried: writes_pushed,
                        if_shared,
                    };
                }
                if !writes.is_empty() {
                    binding::push_set(
                        push,
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.layout,
                        writes,
                    );
                    recorded += 1;
                }
            } else if !self.sets.is_empty() && bound.sets(self.layout, 0, (self.sets, self.offsets))
            {
                raw.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    self.sets,
                    self.offsets,
                );
                recorded += 1;
            }
            if let Some(heap) = self.heap {
                if bound.sets(self.layout, 1, (&[heap], &[])) {
                    raw.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.layout,
                        1,
                        &[heap],
                        &[],
                    );
                    recorded += 1;
                }
            }
            if !self.push.is_empty() && bound.pushes(self.layout, self.push) {
                for (stages, offset, bytes) in self.push {
                    raw.cmd_push_constants(cb, self.layout, *stages, *offset, bytes);
                }
                recorded += 1;
            }
            let strides = dynamic(vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE);
            for &(binding, buffer, offset, stride) in self.vertex_buffers {
                match (strides, &self.extended) {
                    (true, Some(extended)) => {
                        if bound.vertex_buffer(binding, buffer, offset, Some(stride)) {
                            (extended.bind_vertex_buffers2)(
                                cb,
                                binding,
                                1,
                                &buffer,
                                &offset,
                                std::ptr::null(),
                                &stride,
                            );
                            recorded += 1;
                        }
                    }
                    _ => {
                        if bound.vertex_buffer(binding, buffer, offset, None) {
                            raw.cmd_bind_vertex_buffers(cb, binding, &[buffer], &[offset]);
                            recorded += 1;
                        }
                    }
                }
            }
            if let Some(index) = self.index.filter(|index| bound.index_buffer(index)) {
                match index.sized {
                    Some((bind, size)) => {
                        bind(cb, index.buffer, index.offset, size, index.index_type);
                    }
                    None => {
                        raw.cmd_bind_index_buffer(cb, index.buffer, index.offset, index.index_type)
                    }
                }
                recorded += 1;
            }
            let s = self.state;
            let now = Now {
                state: s,
                topology: self.topology,
                primitive_restart: self.primitive_restart,
            };
            for &state in self.dynamic {
                // The write masks, which only a draw read through snapshots makes dynamic, are
                // its pair's to set at every draw (below).
                if state == vk::DynamicState::COLOR_WRITE_MASK_EXT || !bound.dynamic(state, now) {
                    continue;
                }
                let (front, back) = (s.fixed.depth_stencil.front, s.fixed.depth_stencil.back);
                recorded += 1;
                match state {
                    vk::DynamicState::VIEWPORT => raw.cmd_set_viewport(cb, 0, &s.viewports),
                    vk::DynamicState::SCISSOR => raw.cmd_set_scissor(cb, 0, &s.scissors),
                    vk::DynamicState::DEPTH_BIAS => s.depth_bias.record(raw, cb),
                    vk::DynamicState::BLEND_CONSTANTS => {
                        raw.cmd_set_blend_constants(cb, &s.blend_color)
                    }
                    vk::DynamicState::STENCIL_COMPARE_MASK => {
                        for (face, mask) in faces(front.compare_mask, back.compare_mask) {
                            raw.cmd_set_stencil_compare_mask(cb, face, mask);
                        }
                    }
                    vk::DynamicState::STENCIL_WRITE_MASK => {
                        for (face, mask) in faces(front.write_mask, back.write_mask) {
                            raw.cmd_set_stencil_write_mask(cb, face, mask);
                        }
                    }
                    vk::DynamicState::STENCIL_REFERENCE => {
                        let [front, back] = s.stencil_reference;
                        for (face, reference) in faces(front, back) {
                            raw.cmd_set_stencil_reference(cb, face, reference);
                        }
                    }
                    state => {
                        let assembly = (self.topology, self.primitive_restart);
                        let extended = self.extended.as_ref();
                        if !extended.is_some_and(|e| record_extended(e, cb, state, s, assembly)) {
                            recorded -= 1;
                        }
                    }
                }
            }
            if let Some(pair) = &self.pair {
                pair.record(raw, cb, self.layout);
            }
            match self.draw {
                metal2vulkan::vk::DrawCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => raw.cmd_draw(
                    cb,
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                ),
                metal2vulkan::vk::DrawCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                    ..
                } => raw.cmd_draw_indexed(
                    cb,
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                ),
                metal2vulkan::vk::DrawCommand::DrawIndirect {
                    offset,
                    draw_count,
                    stride,
                } => {
                    if let Some(buffer) = self.indirect {
                        raw.cmd_draw_indirect(cb, buffer, offset, draw_count, stride);
                    }
                }
                metal2vulkan::vk::DrawCommand::DrawIndexedIndirect {
                    offset,
                    draw_count,
                    stride,
                    ..
                } => {
                    if let Some(buffer) = self.indirect {
                        raw.cmd_draw_indexed_indirect(cb, buffer, offset, draw_count, stride);
                    }
                }
                // A mesh render pipeline's state compiles only where the device enabled
                // `meshShader`, which loads the commands.
                metal2vulkan::vk::DrawCommand::DrawMeshTasks {
                    group_count_x,
                    group_count_y,
                    group_count_z,
                } => {
                    if let Some(mesh) = self.mesh {
                        mesh.cmd_draw_mesh_tasks(cb, group_count_x, group_count_y, group_count_z);
                    }
                }
                metal2vulkan::vk::DrawCommand::DrawMeshTasksIndirect {
                    offset,
                    draw_count,
                    stride,
                } => {
                    if let (Some(mesh), Some(buffer)) = (self.mesh, self.indirect) {
                        mesh.cmd_draw_mesh_tasks_indirect(cb, buffer, offset, draw_count, stride);
                    }
                }
                metal2vulkan::vk::DrawCommand::DrawMeshTasksIndirectCount {
                    offset,
                    count_offset,
                    max_draw_count,
                    stride,
                } => {
                    if let (Some(mesh), Some(buffer)) = (self.mesh, self.indirect) {
                        mesh.cmd_draw_mesh_tasks_indirect_count(
                            cb,
                            buffer,
                            (offset, count_offset),
                            max_draw_count,
                            stride,
                        );
                    }
                }
            }
        }
        Recorded {
            states: recorded,
            writes_pushed,
            equal_shapes,
        }
    }
}

/// Record extended dynamic `state` at the encoder's value `s` (the draw's input `assembly`,
/// its topology and primitive restart, for those): whether it is a state recorded here.
///
/// # Safety
///
/// `cb` is recording on the device `extended`'s commands were loaded for, after the pipeline was
/// bound.
unsafe fn record_extended(
    extended: &super::dynamic_state::ExtendedDynamicState,
    cb: vk::CommandBuffer,
    state: vk::DynamicState,
    s: &DrawState,
    assembly: (metal2vulkan::vk::PrimitiveTopology, bool),
) -> bool {
    let (topology, primitive_restart) = assembly;
    let ds = &s.fixed.depth_stencil;
    let bool32 = |b: bool| vk::Bool32::from(b);
    // SAFETY: the caller's contract; each command sets one state of the bound pipeline.
    unsafe {
        match state {
            vk::DynamicState::CULL_MODE => (extended.cull_mode)(cb, s.fixed.cull_mode.into()),
            vk::DynamicState::FRONT_FACE => (extended.front_face)(cb, s.fixed.front_face.into()),
            vk::DynamicState::PRIMITIVE_TOPOLOGY => {
                (extended.primitive_topology)(cb, topology.into())
            }
            // Present wherever the state is dynamic (the draw checks it before it records).
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE => {
                if let Some(restart) = extended.primitive_restart_enable {
                    restart(cb, bool32(primitive_restart));
                }
            }
            vk::DynamicState::DEPTH_TEST_ENABLE => {
                (extended.depth_test_enable)(cb, bool32(ds.depth_test_enable))
            }
            vk::DynamicState::DEPTH_WRITE_ENABLE => {
                (extended.depth_write_enable)(cb, bool32(ds.depth_write_enable))
            }
            vk::DynamicState::DEPTH_COMPARE_OP => {
                (extended.depth_compare_op)(cb, ds.depth_compare_op.into())
            }
            vk::DynamicState::STENCIL_TEST_ENABLE => {
                (extended.stencil_test_enable)(cb, bool32(ds.stencil_test_enable))
            }
            vk::DynamicState::STENCIL_OP => {
                let op = |ops: metal2vulkan::metal::state::StencilOpState| {
                    (ops.fail_op, ops.pass_op, ops.depth_fail_op, ops.compare_op)
                };
                for (face, (fail_op, pass_op, depth_fail_op, compare_op)) in
                    faces(op(ds.front), op(ds.back))
                {
                    (extended.stencil_op)(
                        cb,
                        face,
                        fail_op.into(),
                        pass_op.into(),
                        depth_fail_op.into(),
                        compare_op.into(),
                    );
                }
            }
            // The vertex strides are recorded with the vertex buffers.
            _ => return false,
        }
    }
    true
}

/// The faces a stencil state command sets, each with its value: both at once where the front and
/// back values are the same, one command where there were two. The Vulkan specification,
/// `VkStencilFaceFlagBits`: "`VK_STENCIL_FACE_FRONT_AND_BACK` is the combination of
/// `VK_STENCIL_FACE_FRONT_BIT` and `VK_STENCIL_FACE_BACK_BIT`, and specifies that both sets of
/// stencil state are updated."
pub(super) fn faces<T: PartialEq + Copy>(
    front: T,
    back: T,
) -> impl Iterator<Item = (vk::StencilFaceFlags, T)> {
    let both = front == back;
    let first = match both {
        true => (vk::StencilFaceFlags::FRONT_AND_BACK, front),
        false => (vk::StencilFaceFlags::FRONT, front),
    };
    std::iter::once(first).chain((!both).then_some((vk::StencilFaceFlags::BACK, back)))
}

/// `STORAGE` usage, which a storage attachment's image has.
const STORAGE: metal2vulkan::vk::ImageUsageFlags = metal2vulkan::vk::ImageUsageFlags::STORAGE;
const INPUT_ATTACHMENT: metal2vulkan::vk::ImageUsageFlags =
    metal2vulkan::vk::ImageUsageFlags::INPUT_ATTACHMENT;

/// Whether `target`'s texture is memoryless.
fn memoryless(target: &Target) -> bool {
    target.texture.storage_mode() == super::values::StorageMode::Memoryless
}

/// The clears of color attachments a pass's first instance leaves out, recorded before the pass,
/// or of memoryless colors' backings (`render/inline_first.rs`): one allocation, the uses'.
struct Clears {
    /// `(image, clear value, level and layer)` of each clear, at most one per color attachment.
    clears: ColorList<(vk::Image, vk::ClearColorValue, vk::ImageSubresourceRange)>,
    uses: Vec<Use<ResourceKey>>,
}

impl Clears {
    /// The clears of `targets`' levels and layers to their values, color attachments of a pass;
    /// `None` when there is none.
    fn plan<'a>(
        targets: impl IntoIterator<Item = (&'a Target, vk::ClearColorValue)>,
    ) -> Result<Option<Clears>, Error> {
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let mut clears = ColorList::new();
        let mut uses = Vec::new();
        for (target, value) in targets {
            let levels = target.level..target.level + 1;
            let layers = target.layer..target.layer + 1;
            uses.push(target.texture.image_use(levels, layers, write, true)?);
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: target.level,
                level_count: 1,
                base_array_layer: target.layer,
                layer_count: 1,
            };
            clears.push((target.texture.raw_image()?, value, range))?;
        }
        Ok((!clears.is_empty()).then_some(Clears { clears, uses }))
    }

    /// Record the clears into `command_buffer`, between barriers from everything before and to
    /// everything after, as [`Clears::record`] records them in an encoder's command buffer: a
    /// step of a pass recorded later, or after its instances recorded inline, whose uses the
    /// pass took.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it; the
    /// images are live until it completes, rest in `GENERAL` and take transfer writes.
    unsafe fn record_in(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract; each range is one level and layer of its image.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            for (image, value, range) in &self.clears {
                raw.cmd_clear_color_image(
                    command_buffer,
                    *image,
                    vk::ImageLayout::GENERAL,
                    value,
                    &[*range],
                );
            }
            Barrier::everything().record(raw, command_buffer);
        }
    }

    /// Record the clears in `core`'s command buffer, between barriers from everything before
    /// and to everything after.
    fn record(self, core: &mut EncoderCore) {
        let Clears { clears, uses } = self;
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording outside a render pass; the images
            // are live until it completes (the encoder retains the textures), rest in `GENERAL`,
            // and take transfer writes (`cleared_ahead`); each range is one level and layer of
            // its image.
            unsafe {
                for (image, value, range) in &clears {
                    device.cmd_clear_color_image(
                        raw,
                        *image,
                        vk::ImageLayout::GENERAL,
                        value,
                        &[*range],
                    );
                }
            }
        });
        core.barrier(Barrier::everything());
    }
}

/// Whether a color attachment `target` that the pass's first instance leaves out and Metal
/// clears can be cleared before the pass instead (`vkCmdClearColorImage`, with the clear value a
/// load operation takes): the clear covers its whole level and layer, so the render area
/// (`extent`, from the origin) must be all of it, and its image takes transfer writes in the
/// attachment's format (a view of another format's clear value is that format's, which
/// `vkCmdClearColorImage` would take as the image's); an attachment of an image that is not 2D
/// is cleared by its load operation (`vkCmdClearColorImage` clears a 3D image's whole level).
fn cleared_ahead(target: &Target, extent: [u32; 2]) -> Result<bool, Error> {
    let plan = target.texture.image_plan()?;
    Ok(target.extent == extent
        && plan.format == target.format
        && !super::render_pass::not_2d(target)
        && plan
            .usage
            .contains(metal2vulkan::vk::ImageUsageFlags::TRANSFER_DST))
}

/// Whether a load action of `layout` clears an attachment.
fn clears(layout: &PassLayout) -> bool {
    let color = layout.colors.iter().flatten().map(|c| c.load);
    let depth_stencil = layout
        .depth_stencil
        .iter()
        .flat_map(|ds| [ds.depth.as_ref(), ds.stencil.as_ref()])
        .flatten()
        .map(|aspect| aspect.load);
    color
        .chain(depth_stencil)
        .any(|load| load == LoadAction::Clear)
}

/// The extended-range colors of the pass `descriptor` describes, by index, whose `Clear` load
/// leaves Metal's tile memory holding values the clear's stored codes do not read back
/// (`ExtendedRange::clear_read_back`): a draw that reads or blends into one sees the codes, where
/// Metal's sees the clear's halves (`RenderCommandEncoder::degrade_extended_range_blends`).
fn cleared_off_codes(descriptor: &RenderPassDescriptor) -> BTreeSet<u32> {
    (0u32..)
        .zip(&descriptor.color_attachments)
        .filter(|(_, color)| color.load_action == LoadAction::Clear)
        .filter(|(_, color)| {
            color
                .texture
                .as_ref()
                .and_then(|texture| ExtendedRange::of(texture.pixel_format()))
                .and_then(|form| form.clear_read_back(color.clear_color.channels()))
                == Some(false)
        })
        .map(|(index, _)| index)
        .collect()
}

/// The layout of an instance of `shape` in a pass of `layout`: `bare`, the layout without the
/// pass's hidden depth and stencil, for an instance before they arrived (its shape has no
/// depth/stencil attachment), else `layout`.
fn instance_layout<'a>(
    layout: &'a PassLayout,
    bare: Option<&'a PassLayout>,
    shape: &PassShape,
) -> &'a PassLayout {
    match (bare, shape.depth_stencil) {
        (Some(bare), None) => bare,
        _ => layout,
    }
}

/// The images a pass renders attachments into in place of their textures: those of `layout`'s
/// stand-ins and of the ones `standing` brings (`attachment_stand_in::pooled_backing` takes none
/// of them for another attachment of the pass).
fn standing_images(layout: &PassLayout, standing: &[Standing]) -> Vec<Texture> {
    let colors = layout
        .colors
        .iter()
        .flatten()
        .filter(|color| color.stand_in.is_some())
        .map(|color| &color.target);
    let depth_stencil = layout
        .depth_stencil
        .iter()
        .filter(|ds| ds.stand_in.is_some())
        .map(|ds| &ds.target);
    let brought = standing.iter().filter_map(|stand| match stand {
        Standing::Color(_, target, _) | Standing::DepthStencil(target, _) => Some(target),
        Standing::Planes(..) | Standing::Hidden(_) => None,
    });
    colors
        .chain(depth_stencil)
        .chain(brought)
        .map(|target| target.texture.clone())
        .collect()
}

/// `layout` with `standing`'s stand-ins in place of their attachments (`attachment_stand_in.rs`)
/// and its sample planes (`sample_planes.rs`).
fn stood(layout: &PassLayout, standing: Vec<Standing>) -> PassLayout {
    let mut layout = layout.clone();
    for stand in standing {
        match stand {
            Standing::Color(index, target, stand_in) => {
                if let Some(Some(color)) = layout.colors.get_mut(index) {
                    color.target = target;
                    color.stand_in = Some(stand_in);
                }
            }
            Standing::DepthStencil(target, stand_in) => {
                if let Some(ds) = &mut layout.depth_stencil {
                    ds.target = target;
                    ds.stand_in = Some(stand_in);
                }
            }
            Standing::Planes(index, planes) => {
                if let Some(Some(color)) = layout.colors.get_mut(index) {
                    color.planes = Some(planes);
                }
            }
            Standing::Hidden(target) => {
                layout.depth_stencil = Some(DepthStencilTarget::hidden(target));
            }
        }
    }
    layout
}

/// The uses of the color attachments of `layout` that `graphics` reads as input attachments
/// (framebuffer fetch, `framebuffer_fetch.rs`): each read by the fragment shader. The pass reads
/// color attachment `i` as input attachment `i` (`render_target.rs`), which the compile layer's
/// numbering is, where its image (or the stand-in that took its place) takes that usage.
fn input_attachments(
    layout: &PassLayout,
    graphics: &GraphicsState,
) -> Result<Vec<Use<ResourceKey>>, Error> {
    let mut uses = Vec::with_capacity(graphics.input_attachments.len());
    for input in &graphics.input_attachments {
        let color = layout
            .colors
            .get(input.color_index as usize)
            .and_then(Option::as_ref)
            .filter(|_| layout.fetch_inputs)
            .filter(|_| input.input_attachment_index == input.color_index)
            .ok_or_else(|| internal("an input attachment the pass does not read"))?;
        let target = &color.target;
        if !render_target::takes_input(target)? {
            return Err(internal("an input attachment without its usage"));
        }
        uses.push(target.texture.image_use(
            target.level..target.level + 1,
            target.layer..target.layer + 1,
            Access::new(
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            ),
            false,
        )?);
    }
    Ok(uses)
}

/// The uses of a draw's `storage` attachments of `layout`: each read by the fragment shader, and
/// written where the function writes it, in its storage view's level and layer, or in every
/// layer of its sample planes (`sample_planes.rs`).
fn storage_uses(
    layout: &PassLayout,
    storage: &[StorageAttachmentUse],
) -> Result<Vec<Use<ResourceKey>>, Error> {
    let mut uses = Vec::with_capacity(2 * storage.len());
    let stage = vk::PipelineStageFlags2::FRAGMENT_SHADER;
    let read = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ);
    let write = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    for s in storage {
        let color = layout
            .colors
            .get(s.color_index as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| internal("a storage attachment the pass does not have"))?;
        if let StorageAttachmentView::SamplePlanes { .. } = s.view {
            let planes = color
                .planes
                .as_ref()
                .ok_or_else(|| internal("sample planes the pass does not have"))?;
            uses.push(planes.image_use(read, false)?);
            if s.writes {
                uses.push(planes.image_use(write, true)?);
            }
            continue;
        }
        let target = &color.target;
        if !target.texture.image_plan()?.usage.contains(STORAGE) {
            return Err(internal("a storage attachment without storage usage"));
        }
        let levels = target.level..target.level + 1;
        let layers = target.layer..target.layer + 1;
        uses.push(
            target
                .texture
                .image_use(levels.clone(), layers.clone(), read, false)?,
        );
        if s.writes {
            uses.push(target.texture.image_use(levels, layers, write, true)?);
        }
    }
    Ok(uses)
}

/// The offset an indirect draw in `call` reads its `bytes` of arguments from, given `offset` of a
/// buffer of `length` bytes: the offset rounded down to a multiple of 4, where plain Metal reads
/// them and its validation stops the draw (Metal probe cases `ind_offset_misaligned` and
/// `ind_indexed_offset_misaligned` on an M4: arguments at offset 2 drew what those at 0 draw).
/// Refused (`Invalid`), as Metal's validation refuses it: arguments past the buffer's length from
/// there (plain Metal reads past the buffer: undefined).
fn check_arguments(call: &'static str, length: u64, offset: u64, bytes: u64) -> Result<u64, Error> {
    let offset = offset - offset % 4;
    if offset.checked_add(bytes).is_none_or(|end| end > length) {
        return Err(draw_problem(call, DrawProblem::IndirectBufferRange));
    }
    Ok(offset)
}

/// The state a pass of `samples` draws `state` as: in a pass of one sample, a render pipeline's state of
/// more samples draws as its single-sample twin, as plain Metal draws it (Metal probes, 2- and
/// 4-sample states: coverage at the pixel centre, one run per pixel even reading
/// `[[sample_id]]`, which reads 0, `[[position]]` and sample-interpolated inputs at the centre,
/// a `[[sample_mask]]` input of 1 and an output keeping the pixel by its bit 0, a framebuffer
/// fetch and alpha to one as at one sample). Where Metal answers otherwise the state stays
/// itself, which `check_pipeline` refuses (`SampleCount`): alpha to coverage (alphas of 0.3 and
/// 0.5 that drop the texel at one sample kept it), a fragment function that reads the sample
/// count (`get_num_samples()` read the state's), and, unmeasured, a fragment imageblock. Any
/// other state is itself.
pub(super) fn drawn_as(
    samples: u32,
    state: &RenderPipelineState,
) -> Result<RenderPipelineState, Error> {
    let RenderStateKind::Graphics(graphics) = state.kind() else {
        return Ok(state.clone());
    };
    let multisampled = graphics.descriptor().raster_sample_count() > 1;
    let coverage = graphics
        .descriptor()
        .vertex()
        .is_some_and(|d| d.alpha_to_coverage);
    if samples != 1 || !multisampled || coverage {
        return Ok(state.clone());
    }
    let Some(twin) = state.single_sample()? else {
        return Ok(state.clone());
    };
    let RenderStateKind::Graphics(drawn) = twin.kind() else {
        return Err(internal("a drawing state's twin that does not draw"));
    };
    if drawn.reads_sample_count() || drawn.base().graphics.fragment_imageblock.is_some() {
        return Ok(state.clone());
    }
    Ok(twin)
}

/// The drawing state a draw of `assembly` in `call` draws with, of the encoder's `pipeline`, by
/// the checks plain Metal keeps for every draw: refused without a state, with a tile state, and
/// with one of another kind of draw (`DrawKind`). A tessellation pipeline draws patches, and only
/// it does: a draw of primitives with it draws nothing (`None`), as plain Metal draws it (Metal
/// probe case `primitives_with_tessellation` on an M4; Metal's validation stops it), and one of
/// patches with another pipeline stops Metal. A mesh render pipeline draws meshes, and only it
/// does: a mesh draw of no threadgroups (`empty`) with another drawing state draws nothing, as
/// plain Metal draws it (its validation stops it). The encoder's lowering and its checks at the
/// call (`render/call_checks.rs`) both take it.
fn drawn_by(
    call: &'static str,
    pipeline: Option<&RenderPipelineState>,
    assembly: Assembly,
    empty: bool,
) -> Result<Option<DrawingState>, Error> {
    let pipeline = match pipeline.map(RenderPipelineState::kind) {
        None => return Err(invalid(call, Problem::NoPipelineState)),
        Some(RenderStateKind::Tile(_)) => return Err(draw_problem(call, DrawProblem::TileState)),
        Some(RenderStateKind::Graphics(state)) => state,
    };
    let tessellated = crate::tessellation::patches(&pipeline.base().graphics).is_some();
    if tessellated != matches!(assembly, Assembly::Patches) {
        return match assembly {
            Assembly::Primitives(_) | Assembly::Private(_) if tessellated => Ok(None),
            _ => Err(draw_problem(call, DrawProblem::DrawKind)),
        };
    }
    let meshes = pipeline.descriptor().vertex().is_none();
    if meshes != matches!(assembly, Assembly::Meshes) {
        if empty && matches!(assembly, Assembly::Meshes) {
            return Ok(None);
        }
        return Err(draw_problem(call, DrawProblem::DrawKind));
    }
    Ok(Some(pipeline.clone()))
}

/// A patch draw's patch index or control-point index buffer and its offset, as the binder takes
/// them: in the buffer's space (`Buffer::space`), which starts `Buffer::base` bytes before its
/// byte 0, as the factor buffer's (`tessellation.rs`).
fn index_ref(buffer: &Buffer, offset: u64) -> (IndexBufferRef, u64) {
    let base = buffer.base();
    (
        IndexBufferRef::new(0, base + buffer.length()),
        base + offset,
    )
}

/// Whether a command of `plan` reads textures through a handle table: handles its functions load
/// from memory, which reach every texture of the device's heap set, named or not.
pub(super) fn reads_handles(plan: &metal2vulkan::bind::BindingPlan) -> bool {
    plan.descriptors
        .iter()
        .any(|write| matches!(write.source, WriteSource::HandleTable { .. }))
}

/// An `Invalid` draw problem of `call`.
/// The texture a pass of `layout` loads and stores its depth into, where it does: the separate
/// depth texture, the texture of a stand-in its copies serve, or the attachment's own; `None` for
/// a pass that does not load and store a depth (`Texture::degrade_kept_signs`).
fn kept_signs(layout: &PassLayout) -> Option<&Target> {
    let ds = layout.depth_stencil.as_ref()?;
    let depth = ds.depth.as_ref()?;
    let stores = matches!(
        depth.store,
        StoreAction::Store
            | StoreAction::StoreAndMultisampleResolve
            | StoreAction::CustomSampleDepthStore
    );
    if depth.load != LoadAction::Load || !stores {
        return None;
    }
    match (&ds.separate, &ds.stand_in) {
        (Some(separate), _) => separate.depth.as_ref(),
        (None, Some(stand_in)) => stand_in.copies().then_some(&stand_in.own),
        (None, None) => Some(&ds.target),
    }
}

fn draw_problem(call: &'static str, problem: DrawProblem) -> Error {
    invalid(call, Problem::Draw(problem))
}

/// An `Invalid` render pass problem of `call`.
fn store_problem(call: &'static str, problem: RenderPassProblem) -> Error {
    invalid(call, Problem::RenderPass(problem))
}

/// The store action `action` an encoder sets in `call` for `which`, whose descriptor's action was
/// `Unknown` when `late`, which `resolves` into a resolve texture, and whose texture is
/// `memoryless`.
fn late_store(
    call: &'static str,
    which: Attachment,
    late: bool,
    resolves: bool,
    memoryless: bool,
    action: StoreAction,
) -> Result<StoreAction, Error> {
    if !late || action == StoreAction::Unknown {
        return Err(store_problem(
            call,
            RenderPassProblem::StoreActionKnown(which),
        ));
    }
    // `CustomSampleDepthStore` stores a depth attachment as `Store` does, and is no store action
    // of a color or stencil attachment (`render_pass.rs`, a probe). A late one there plain Metal
    // leaves unstored (`csds_late_color`, `csds_late_stencil`: the texture keeps its texels, as
    // Apple's GPU keeps them under `DontCare`), which its validation stops: stored as `DontCare`.
    // One that resolves, unmeasured, is refused.
    let action = match (action, which) {
        (StoreAction::CustomSampleDepthStore, Attachment::Depth) => StoreAction::Store,
        (StoreAction::CustomSampleDepthStore, _) if !resolves => StoreAction::DontCare,
        (StoreAction::CustomSampleDepthStore, _) => {
            return Err(store_problem(
                call,
                RenderPassProblem::CustomSampleDepthStore(which),
            ))
        }
        _ => action,
    };
    // A memoryless attachment holds no memory, so `Store` stores nowhere and
    // `StoreAndMultisampleResolve` is its resolve alone, set late by the encoder exactly as set in
    // the descriptor: one mapping, `render_pass::memoryless_actions`, which the load action goes
    // into as `DontCare` because this call has none and that mapping leaves `DontCare` alone.
    // Plain Metal takes the late setter as it takes the descriptor's, for a color, depth and
    // stencil attachment alike (`probes/memoryless_late_store_action.m` on an Apple M3 Max under
    // macOS 27.0.1, one shape a process, graded at the commit and never at the call): the setter
    // returns, `endEncoding` returns, the command buffer completes with no error, the stages'
    // own counters report 3 vertices and every fragment, a guard buffer placed after the readable
    // destination is untouched and three canary command buffers complete. A memoryless 4-sample
    // attachment set late to `StoreAndMultisampleResolve` lands the same resolve texel as the
    // same pass set to `MultisampleResolve`, and as a private attachment's
    // `StoreAndMultisampleResolve` does, which is what says the resolve half still runs.
    // Metal's validation layer alone objects, at the setter ("Deferred Store Action Validation:
    // storeAction unsupported for Memoryless attachment.") and again at the end of the encoder
    // ("endEncoding called with store action unsupported for Memoryless attachment."); those are
    // layer messages, and `MTL_DEBUG_LAYER_ERROR_MODE=assert` ends the program at them.
    // Nothing is claimed about what the attachment's texels are: Metal specifies no contents for
    // tile memory, and a memoryless attachment has none to read.
    let action = match memoryless {
        true => render_pass::memoryless_actions(LoadAction::DontCare, action).1,
        false => action,
    };
    // One that does not resolve where the attachment resolves leaves the resolve texture
    // untouched and stores the attachment by the action, as plain Metal does where its
    // validation stops it (color, depth and stencil alike): the caller drops
    // the attachment's resolve.
    if !resolves && action.resolves() {
        return Err(store_problem(
            call,
            RenderPassProblem::ResolveWithoutTexture(which),
        ));
    }
    Ok(action)
}

#[cfg(test)]
mod tests {
    use super::{
        check_arguments, draw_problem, late_store, store_problem, DrawProblem, RenderPassProblem,
    };
    use crate::{Attachment, StoreAction};

    /// The encoder sets a store action only over the descriptor's `Unknown`, never to `Unknown`; a
    /// resolving one exactly where the attachment resolves (measured on an M4); and a memoryless
    /// attachment's store as the actions it can hold, storing nowhere.
    #[test]
    fn a_late_store_action_replaces_only_unknown() {
        let call = "set_color_store_action";
        let which = Attachment::Color(2);
        for action in [StoreAction::Store, StoreAction::DontCare] {
            assert_eq!(
                late_store(call, which, true, false, false, action),
                Ok(action)
            );
        }
        for action in [
            StoreAction::MultisampleResolve,
            StoreAction::StoreAndMultisampleResolve,
        ] {
            assert_eq!(
                late_store(call, which, true, true, false, action),
                Ok(action)
            );
            assert_eq!(
                late_store(call, which, true, false, false, action),
                Err(store_problem(
                    call,
                    RenderPassProblem::ResolveWithoutTexture(which)
                ))
            );
        }
        // One that does not resolve where the attachment resolves is taken: the resolve goes.
        for action in [StoreAction::Store, StoreAction::DontCare] {
            assert_eq!(
                late_store(call, which, true, true, false, action),
                Ok(action)
            );
        }
        let known = Err(store_problem(
            call,
            RenderPassProblem::StoreActionKnown(which),
        ));
        assert_eq!(
            late_store(call, which, false, false, false, StoreAction::Store),
            known
        );
        assert_eq!(
            late_store(call, which, true, false, false, StoreAction::Unknown),
            known
        );
        // `CustomSampleDepthStore` stores a depth attachment as `Store`, and a color or stencil
        // one, late, as `DontCare` where it does not resolve (a probe).
        let custom = StoreAction::CustomSampleDepthStore;
        assert_eq!(
            late_store(call, which, true, false, false, custom),
            Ok(StoreAction::DontCare)
        );
        assert_eq!(
            late_store(call, which, true, true, false, custom),
            Err(store_problem(
                call,
                RenderPassProblem::CustomSampleDepthStore(which)
            ))
        );
        let stencil = Attachment::Stencil;
        assert_eq!(
            late_store(call, stencil, true, false, false, custom),
            Ok(StoreAction::DontCare)
        );
        let depth = Attachment::Depth;
        assert_eq!(
            late_store(call, depth, true, false, false, custom),
            Ok(StoreAction::Store)
        );
        assert_eq!(
            late_store(call, depth, false, false, false, custom),
            Err(store_problem(
                call,
                RenderPassProblem::StoreActionKnown(depth)
            ))
        );
        // A memoryless attachment stores nowhere, so a late `Store` of one -- and a depth
        // `CustomSampleDepthStore`, which stores as `Store` does -- is its `DontCare`, and a late
        // `StoreAndMultisampleResolve` is its resolve alone, the same actions the descriptor's own
        // take (`render_pass::memoryless_actions`). Plain Metal takes every one of these late
        // (measured on an Apple M3 Max under macOS 27.0.1, graded at the commit).
        assert_eq!(
            late_store(call, depth, true, false, true, custom),
            Ok(StoreAction::DontCare)
        );
        assert_eq!(
            late_store(call, which, true, false, true, StoreAction::DontCare),
            Ok(StoreAction::DontCare)
        );
        assert_eq!(
            late_store(
                call,
                which,
                true,
                true,
                true,
                StoreAction::MultisampleResolve
            ),
            Ok(StoreAction::MultisampleResolve)
        );
        assert_eq!(
            late_store(call, which, true, false, true, StoreAction::Store),
            Ok(StoreAction::DontCare)
        );
        assert_eq!(
            late_store(
                call,
                which,
                true,
                true,
                true,
                StoreAction::StoreAndMultisampleResolve
            ),
            Ok(StoreAction::MultisampleResolve)
        );
    }

    /// Indirect arguments are read from the offset rounded down to a multiple of 4, as plain
    /// Metal reads them (probe 5: offset 2 drew the arguments at 0), and lie in the buffer, as
    /// Metal's validation requires (probe 5: 20 or 24 bytes of a 16- or 20-byte buffer stop).
    #[test]
    fn indirect_arguments_are_read_as_metal_reads_them() {
        let call = "draw_primitives_indirect";
        assert_eq!(check_arguments(call, 16, 0, 16), Ok(0));
        assert_eq!(check_arguments(call, 36, 16, 20), Ok(16));
        assert_eq!(check_arguments(call, 24, 2, 16), Ok(0));
        assert_eq!(check_arguments(call, 16, 3, 16), Ok(0));
        assert_eq!(check_arguments(call, 24, 7, 20), Ok(4));
        for (length, offset, bytes) in [(16, 4, 16), (20, 4, 20), (16, u64::MAX - 3, 16)] {
            assert_eq!(
                check_arguments(call, length, offset, bytes),
                Err(draw_problem(call, DrawProblem::IndirectBufferRange)),
                "{length} {offset} {bytes}"
            );
        }
    }
}
