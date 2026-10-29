//! Executing an indirect command buffer's render commands in a render encoder
//! (`executeCommandsInBuffer:withRange:`).
//!
//! - **When.** As a compute encoder's (`indirect_execute.rs`), an execution runs the records as
//!   they stand when its batch is submitted: Metal's reads them as it runs, so it sees CPU writes
//!   made after the call, and the GPU's writes before it where an encoder made the buffer
//!   resident for writing. `commit` records the CPU's records as they stand then; where the CPU
//!   writes them again before the pass's batch is ready to submit (an event wait holds it), the
//!   host step records the pass again from the records as they stand then, waiting for no work
//!   on the GPU and announcing nothing. A CPU write after the batch was submitted races the GPU,
//!   as on Metal. So the call records nothing for the commands:
//!   the instance it is encoded in ends its secondary command buffer there and goes on in a new
//!   one, the execution's own secondary standing between the two ([`Piece`](super::Piece)).
//!   A secondary command buffer must be recorded before a primary executes it, so a pass with an
//!   execution leaves its whole pass to record later ([`DeferredPass`]), as a segment of its
//!   encoder (`encoder.rs`, `deferred.rs`): its begin, every instance's secondaries and the
//!   executions', and its end, behind a barrier from everything before and with one to
//!   everything after. The copies and clears before the pass, the reset of its visibility
//!   results, and the copies and results after it, stay the encoder's own; the steps around its
//!   instances are the segment's, recorded as the encoder's own pass records them
//!   (`render/pass_steps.rs`: the shadowed textures' copies, the sample planes', the shader-side
//!   stencil's, the draws' steps before their instances, the tile dispatches, and the instances
//!   of no draws that resolve after the last). `commit` records the segment from the CPU's
//!   records; where an
//!   encoder made a buffer the pass executes resident for writing
//!   (`IndirectCommandBuffer::gpu_written`), `commit` records it for good where every such
//!   execution of the pass runs on the GPU alone (`executions/gpu.rs`, which states which do and
//!   when the decision is taken again), and otherwise the scheduler's host step records it once
//!   the work before it completed, which the device announces once
//!   (`Degradation::HostReplayedIndirectCommands`). The step draws in the large-buffer form,
//!   whose variant no command's buffer lengths change (`BufferForms::Large`), and the commit
//!   settles what it will draw (the step compiled each draw's variant and `VkPipeline` as it met
//!   them, after the commit returned): for each state a record may name (the inherited one, else
//!   every live state made with `supportIndirectCommandBuffers`), the variant and the `VkPipeline`
//!   of a draw of each primitive type of its topology class, indexed or not, in the execution's
//!   instance under the encoder's state at the call ([`DeferredPass::settle`]). The variant is the
//!   one the state's creation compiled in the large-buffer form, which the commit looks up (the
//!   coordinator's ruling of 2026-10-07: where Metal compiles); the commit makes the `VkPipeline`s,
//!   which the pass's instance and the encoder's state key. The step still makes what that leaves:
//!   a tessellation or mesh pipeline's draws; the `VkPipeline` of a draw that fetches attachments,
//!   writes a shadowed stencil, draws views, routes colors to other locations than its instance's,
//!   or binds per-bind strides that are not dynamic state; a draw in a run of another shape; a
//!   command whose buffers ask another variant all the same (the commit compiles that variant where
//!   the encoder's state asks it, an inherited buffer slot bound with bytes among it); and a state
//!   made after the commit. The commit makes the neutrals of the plans it settles, a state made
//!   with `supportIndirectCommandBuffers` made, at its creation, those its variants' plans may bind
//!   with nothing bound (`GraphicsState::make_neutrals`), and the commit makes the runs' passes
//!   and memoryless backings ahead (`ahead.rs`), so the step, a serial point of the command
//!   buffers' stream, seldom makes one; what nothing made before it, the step makes itself
//!   (`neutral.rs`, `creation.rs`).
//! - **Each command** draws as the encoder's `draw_primitives`, `draw_indexed_primitives` or
//!   `draw_patches` would, through the same steps (`RenderCommandEncoder::draw`; part 5c for the
//!   indexed and patch draws), into the instance the execution was encoded in, with the
//!   encoder's state as the call found it: its depth-stencil and raster state, viewports and
//!   scissors and amplification, and its pipeline state and bound tables where the
//!   buffer inherits them (Metal's inherited state is the encoder's at the call: a pipeline
//!   set after it does not reach the commands). Otherwise a command runs with its own pipeline
//!   state, named by its record's ID among the states made with `supportIndirectCommandBuffers`
//!   (the one the CPU set, or one alive at the step), and its own vertex and fragment buffers,
//!   whose addresses resolve against the resources the command buffer's encoders made resident,
//!   every encoder's, before the call or after it (`residency.rs`), then against the device's live
//!   buffers Metal reaches unnamed (`live_buffers.rs`), as a compute command's do; its functions
//!   reach those resources too. Where the pass has a shadowed texture, an execution whose command
//!   buffer made resident, past what its encoder had at the call, a shadowed texture or a
//!   resource whose bytes or image meet the truth of a shadowed texture the pass copies is refused
//!   (`NotImplemented(IndirectRenderCommands)`): the copies around the pass's instances follow what
//!   the call gathered (`render/shadows.rs`). Others (the records of an indirect command buffer
//!   a kernel wrote, a buffer of other bytes) change no copy. An address in none of them (a
//!   no-copy buffer or a heap's buffer no command buffer's commands used yet) misses (the slot
//!   unbound, reading zero, counted in the execution report), and an unset buffer reads zero, as
//!   Metal's did. A command with no draw, no pipeline state, no vertices, indices, patches or
//!   instances runs nothing, and its neighbours run.
//! - **Indexed and patch draws**. Their index, patch index and tessellation factor buffers are
//!   addresses too, resolved as the buffers are, so one of its own that no encoder made resident
//!   draws, as Metal drew from an index buffer it did not hold resident; a miss runs nothing and
//!   counts. Indices past the index buffer draw as the encoder's. A patch draw takes its own factor
//!   buffer, never the encoder's, and its pipeline's control points whatever the record holds, as
//!   Metal's did.
//! - **Control-point indices** (probes, an M4). An indexed patch draw reads them as the encoder's
//!   `draw_indexed_patches` does, of its pipeline's index type, from its control-point index
//!   buffer, an address resolved as the others are; as Metal's did, an index past that buffer's end
//!   reads 0, and so does every index without a buffer (a null one, or a plain patch draw on a
//!   pipeline with an index type); an indexed patch draw on a pipeline without one draws the
//!   patches' own control points. An offset that is not a multiple of the index size is read from
//!   the offset rounded down to one, as Metal reads it (probe 9b's offsets read out of the control
//!   points' range so).
//! - **A patch draw's factor scale** (an M4). Metal multiplies the factors by a kernel's scale
//!   whatever the pipeline, though a state made with `supportIndirectCommandBuffers` never enables
//!   the scale: 2 drew what factors of 2 draw, 0.5 what factors of 0.5 draw (integer partitioning
//!   rounds it up to 1), and 0, -1 and NaN drew nothing. Metal rounds the scale to `half` first
//!   (the M3 Max and the M4 alike): 1.0004 drew what 1 draws, 1.3 and 2.0005 what 2 draws, a scale
//!   subnormal as a `half` what its product draws, one infinite as a `half` (70000, `+∞`) what the
//!   maximum factor draws, and one that is 0 as a `half` (2^-30) nothing. So a scale that is not
//!   positive, or is 0 as a `half`, runs nothing, and one that is another `half` than 1 draws on
//!   the state's factor-scaling twin (`GraphicsState::factor_scaling`) with the encoder's scale the
//!   command's, as the encoder's scale multiplies the factors (the binder rounds it to `half`);
//!   the twin is made the first time a command asks for it.
//! - **Visibility results**. Metal counts the commands' draws in the mode set at the call, as the
//!   encoder's draws (`Boolean` 1, `Counting` their samples, nothing while `Disabled`). The call
//!   ends the active query in the secondary before the execution's and
//!   begins a continuation of its offset in the one after it, and lends the execution a third query
//!   of the same offset (`visibility::Lent`), which its recording begins in its secondary before
//!   the first command and ends after the last; the three combine as any continuations do. A
//!   command's draw that ends the query and begins another, one that begins the execution's next
//!   run (below) or that counts no sample, continues it in a spare the call lent too, of the same
//!   offset: one for each command, and another for each under a state whose compare fails every
//!   fragment without Metal's depth attachment, every command of the buffer for a range read from
//!   one. The recording begins and ends each spare it takes as the encoder's own continuations, and
//!   every other one empty after the last command, counting nothing.
//! - **One start, two lengths** (on an Apple M3 Max and an Apple M4). Where an encoder executes one
//!   buffer's commands from a command it executed them from before with another length, ranges
//!   given or read from a buffer alike, Metal's executions do not each run their own range: one may
//!   run the other's length, and the commands that run change from run to run of the same program.
//!   No stable answer exists, so each execution runs its own range, as Metal documents, and the
//!   device announces it once (`Degradation::RepeatedExecutionStart`), naming the buffer, the
//!   command and both lengths: at the call for two ranges given, at the host step for a range read
//!   from a buffer. A compute encoder's executions run their own ranges on Metal too.
//! - **Tile threadgroup memory** (`render/tile_memory.rs`). A command whose fragment function reads
//!   its threadgroup arguments reaches the pass's buffer of every tile's memory, with the arguments
//!   the encoder set at the call, as the encoder's draws do; the pass's buffer moves into the
//!   [`DeferredPass`] as the encoder ends, or the first such command makes it. Such a command after
//!   another that reached the memory in the same run (the encoder's draws before the call included)
//!   records into the execution's next run, a secondary its pass runs in a render pass instance of
//!   its own after a barrier from everything before, so it sees the earlier command's writes as
//!   Metal's draws do; an instance whose execution may do so (the encoder set an argument at the
//!   call) is given the passes of such runs as it ends (`RunPasses`), and the encoder's draws that
//!   reach the memory after the call record into a new instance. A memoryless attachment keeps its
//!   texels across the runs (below).
//! - **A command's barrier** (an M4). Plain Metal orders a command whose barrier is set after the
//!   draws before it, their writes visible to it, where its validation stops `setBarrier` on a
//!   buffer without mesh draws. Such a command's draw records into the execution's next run, as one
//!   ordered after a draw that reached the tile memory does, in passes the encoder made where a
//!   record may have held a barrier as it ended (`RunPasses`), else ones the commit or the host
//!   step makes, for a barrier the CPU set later. The barrier does not order the
//!   earlier draws' fragments before the command's vertex, object and mesh functions (the M3 Max
//!   and an M4): those read the buffers of their tables from before the pass where a fragment stage
//!   of it wrote them, from a copy the recording takes before the pass
//!   (`render/pass_start_reads.rs`), starting from what the encoder's draws before the call noted.
//! - **An emulated mesh command after a draw it depends on** (an M4). On a device
//!   that emulates mesh draws (`render/mesh.rs`), a command's object and mesh functions run as
//!   compute dispatches, recorded before the pass, as Metal runs them before the fragments of every
//!   earlier draw of the pass: they read none of those fragments' writes, behind the command's
//!   barrier or not. Behind the command's barrier, Metal runs them after the earlier draws' other
//!   stages (their vertex functions), so where those touched what the dispatches use, the
//!   dispatches record before the run the barrier begins, between barriers, after every run before
//!   it (`Run::before`). Refused (`NotImplemented(IndirectRenderCommands)`): dispatches that write
//!   what an earlier draw's fragments touched (unmeasured), and that touch what an earlier draw's
//!   other stages touched without the command's barrier (unordered on Metal, or ordered by a
//!   barrier of the encoder the execution does not see) or beside reading an earlier fragment's
//!   write.
//! - **A command of another instance's shape** (Metal draws a command that
//!   fetches its color attachment, writes the stencil in the shader, reaches a fragment imageblock,
//!   or is amplified, as the encoder's own draw of it). The execution's instance is one of no
//!   fetched attachments, shader-side stencil or views. A command whose draw leaves attachments it
//!   fetches out of its subpass (the pixel interlock) or renders views of its own records into the
//!   execution's next run, begun against a pass of its shape, which its pass runs in a render pass
//!   instance of that shape after a barrier from everything before; the next command of the
//!   instance's shape begins another run, and so does what follows the execution in the instance.
//!   The pass and framebuffer of such a run, and the passes of an instance split so whose execution
//!   was not given its runs' passes as the encoder ended, are made as the pass records and held
//!   until the command buffer completes.
//! - **A memoryless attachment across runs**. Metal's pass holds a memoryless attachment's texels
//!   from its first draw to its last, the commands' too, however they run. The runs' render pass
//!   instances load and store it between them (`RunPasses`), which keeps an image's texels where
//!   its memory is not lazily allocated. Where it is (MoltenVK runs such memory as Metal's
//!   memoryless storage, which keeps nothing past its render pass), and where a command's draw
//!   rendered the attachment into a stand-in (a split of the commands' instance, or a fetch), the
//!   pass renders it into a backing from its first instance on, as the encoder's own instances keep
//!   it in a stand-in from a split on: an image of the pass held until the command buffer
//!   completes, announced as the encoder's (`memoryless-backed`), each instance that runs as
//!   several taking a framebuffer of it. Where the pass reads such an attachment past its instances
//!   (its sample planes, a color resolve drawn after the pass, a depth/stencil resolve), the commit
//!   is refused `NotImplemented(IndirectRenderCommands)`.
//! - **A fetched attachment's stand-in or sample planes**. A run whose
//!   draws read and write attachments in the pixel interlock elsewhere than the pass's instances
//!   render them has its steps around it (`RunSteps`), as the encoder's instances of such draws do
//!   (`render/pass_steps.rs`): the copies from the image the instances render such an attachment
//!   into (the texture, or the encoder's stand-in for it) into the scratch a command's draw made
//!   where that image lacks the storage usage or costs its compression (`attachment_stand_in.rs`),
//!   and back after; and the split of the attachment's samples into its sample planes before the
//!   run and their merge after it, on a device without multisampled storage images
//!   (`sample_planes.rs`), the planes the encoder's or a command's draw made. Each step runs
//!   between barriers, so the run reads what the draws before it rendered and the draws after it
//!   read what it wrote. The scratch and planes a command's draw made are held until the command
//!   buffer completes and announced as the encoder's (`fetch-through-scratch`, `sample-planes`).
//! - **A fetched attachment's stand-in through input attachments** (for
//!   Metal's answer). Where the device reads fetched attachments as input attachments (lavapipe), a
//!   command whose draw reads one whose image takes no input attachment usage (a caller's image,
//!   `texture/wrapped.rs`) makes a stand-in that has it, as the encoder's draw does, which makes
//!   the attachment an input attachment of its subpass: its run is of another shape, and its
//!   framebuffer holds the stand-in in place of the image the pass's instances render
//!   (`Run::framed`), copied from that image before the run and back after it between barriers
//!   (`RunSteps`), as around a run of the pixel interlock's scratch.
//! - **Metal's hidden depth and stencil** (an M4, and an M3 Max, commands alone). In a pass without
//!   a depth or a stencil attachment, a command's draw tests and writes the hidden depth and
//!   stencil Metal gives such a pass under the encoder's state at the call, as the encoder's own
//!   draw then would: shared with the encoder's draws before and after it, each command's write
//!   seen by the next. Plain Metal draws every order so, and its validation stops the encoder's
//!   draws among them, not the commands. Where the state at the call needs them, the call gives the
//!   pass them, so the execution's instance holds them; where the pass can take none (an attachment
//!   of the other aspect, no color attachment, a rate map), a command's draw that needs them is
//!   refused `NotImplemented(HiddenDepthStencil)`, as the encoder's would be.
//! - **A shader-side stencil** (Metal draws a command that writes `[[stencil]]` as the encoder's
//!   own draw). On a device without `VK_EXT_shader_stencil_export` such a draw tests and writes a
//!   shadow of the stencil in the shader (`shadow_stencil.rs`), in an instance of its own. Such a
//!   command records into the execution's next run, as one of another shape does, and the next of
//!   the instance's kind begins another; the pass copies the stencil into the shadow before the run
//!   and back after it, as around an encoder's instance of such draws. The shadow is the encoder's,
//!   or one a command's draw made, whose copies' uses the pass then gains, held until the command
//!   buffer completes and announced (`shader-side-stencil`). Such a run is never its instance's
//!   first or last: the execution's first run and what follows it in the instance bound it.
//! - **A fragment imageblock** (Metal draws a command that reaches it as
//!   the encoder's own draw). A command whose fragment function reaches the fragment imageblock
//!   draws in the run it falls in, over the pass's storage (`fragment_imageblock.rs`): the
//!   encoder's, or one an earlier command's draw made, which the pass clears before its first
//!   instance as the encoder clears its own. Each such draw records after a barrier from the draws
//!   before it in its instance, the encoder's included, and the encoder's such draw after the
//!   execution after one from the commands.
//! - **A range read from a buffer** (`execute_commands_in_buffer_indirect`, a probe on an M3 Max
//!   and an M4: a render encoder's execution takes its range as a compute encoder's does,
//!   `indirect_execute::ExecutionRange::commands`). The encoder copies the range's 8 bytes where
//!   the execution is encoded, behind a barrier from everything before and before the pass, and a
//!   host step records the pass from the copy once the work before it completed. A draw of the same
//!   pass that writes the range before the execution is read by Metal's; the copy would not see it,
//!   so a range a draw of the encoder before the call may write is refused.
//! - **Metal's private primitive types** (`DrawPrimitiveType`, which the CPU's draws take; a
//!   kernel writes points for them, as Metal's encoding does). A command's draw of types 5 to 8
//!   draws as the encoder's draw of the type does, the list its program writes. One of type 9
//!   among a range given at the call, as the records stand then, fails the command buffer there
//!   as the encoder's draw does, the encoder running nothing (`IndirectCommandBuffer::nine_among`;
//!   the M3 Max, macOS 27.0.1: a direct one of 3 vertices or more hung, an indexed one of any
//!   count faulted, as the encoder's); one the step meets, written after the call or among a
//!   range read from a buffer, runs nothing (`RECORD`), where Metal's GPU fails the command buffer
//!   (`docs/FAST-PATHS.md`, known gaps).
//! - **Faults** (`bind::ReplayFaults`, the execution report's). A header holding a producer's
//!   sticky encoding fault runs no command (`ENCODING`); a record that does not decode, sets what
//!   the descriptor does not allow, names a primitive or index type Metal has no value for (or
//!   type 9, above), sets
//!   a barrier, which Metal enables only for mesh draws, has no index or factor buffer (which
//!   ends Metal's process), a buffer offset Metal's validation stops and plain Metal's answer to
//!   is not measured (an odd factor buffer offset or factor instance stride; one 2 past a
//!   multiple of 4 is read as plain Metal reads it), patch indices past its patch index buffer,
//!   which the encoder's `draw_patches` refuses (Metal's GPU-written command reads past it:
//!   unmeasured), runs nothing (`RECORD`). An index buffer, a patch index buffer and a
//!   control-point index buffer at an offset that is not a multiple of its index size (4 for patch
//!   indices) are read from the offset rounded down to one, as Metal reads them. A range read from
//!   a buffer whose location lies past the size, or that passes `2^32 - 1`, runs nothing (`RANGE`):
//!   Metal's GPU faults the command buffer.
//! - **What is refused.** At the call (`Invalid`): a buffer of another device, a range past the
//!   size, and an inherited pipeline state that is unset or made without
//!   `supportIndirectCommandBuffers` (Metal's validation stops each). A range read from a buffer
//!   at an offset that is not a multiple of 4 bytes, and one whose 8 bytes run past the buffer,
//!   are the runtime's own gaps, as a compute encoder's are
//!   (`NotImplemented(ExecutionRangeOffFour)` and `NotImplemented(ExecutionRangePastBuffer)`).
//!   A buffer of compute commands, and a range read
//!   from a buffer that a draw of the encoder before the call may write, are
//!   `NotImplemented(IndirectRenderCommands)`. What a command's draw refuses (a pipeline of
//!   other formats than the pass's, among others) is refused so: it either refuses the commit,
//!   or fails the command buffer where the step records it, `CommandBufferError::Refused` with
//!   the same refusal typed (`CommandBuffer::refusal`). The call checks no record but for type 9
//!   (above): Metal lets a program write the records until it commits. A command's vertex buffer
//!   set without an attribute stride, where its pipeline state's layout of
//!   `MTLBufferLayoutStrideDynamic` reads the stride, on a buffer whose descriptor supports
//!   dynamic attribute strides, is the runtime's own gap, as a compute encoder's kernel buffer is
//!   (`NotImplemented(RuntimeGap::IndirectCommandUnstridedBuffer)`): plain Metal takes the command
//!   and completes it, and its fetch reads the buffer at stride 2^32 - 1, which the 32-bit fetch
//!   cannot place (the gap's doc). On a buffer without that support the records hold no stride
//!   word and Metal reads stride 0, which the plan path gives exactly, so the command is taken, as
//!   the encoder's own draw reads such a buffer at stride 0 (`render/vertex_strides.rs`).

use super::bound::Bound;
use super::mesh::MeshGrid;
use super::pass_start_reads::{PassStartReads, PassStartTables};
use super::pass_steps::PassSteps;
use super::pending_tile::TilePipelines;
use super::provoking::ProvokingForms;
use super::replay::SharedCheck;
use super::shadows::InstanceShadows;
use super::tile_memory::TileMemory;
use super::RenderCall;
use super::RenderCommandEncoder;
use super::{
    index_size, Assembly, Before, DrawBuffers, DrawingState, IndexedDraw, Instance,
    InterlockedStorage, Next, PatchDraw, PendingDraw, Piece, PrivateForm, RenderLowering,
    RenderStateKind, PASS_DRAW_STAGES,
};
use crate::amplification::Amplification;
use crate::argument_table::ArgumentTable;
use crate::attachment_stand_in::{self, Cause, StandIn, StandInCopies};
use crate::blit::reads;
use crate::buffer::Buffer;
use crate::command_buffer::CommandBuffer;
use crate::degradation::{Degradation, SplitCauses};
use crate::device::functions::Vk;
use crate::device::lock;
use crate::encoder::{EncoderCore, RecordedSegment};
use crate::error::{internal, invalid, vulkan, CommandBufferError, Error, Problem, RuntimeGap};
use crate::executed_commands::{ExecutedCommand, RecordsRead};
use crate::fragment_imageblock::PassImageblock;
use crate::indirect_command_buffer::{
    IndirectCommandBuffer, IndirectCommandCheck, IndirectCommandCheckNeeds, NineDraw, Reader,
};
use crate::indirect_dispatch;
use crate::indirect_execute::{self, unstrided_buffer, ExecutionRange};
use crate::pipeline::budget::Urgency;
use crate::pipeline::RenderPipelineState;
use crate::pipeline::Settled;
use crate::pipeline::{
    BufferForms, DrawPipeline, FixedState, GraphicsKey, Pending, RawPipeline, RenderVariant,
};
use crate::rate_mapped_pass::RateMappedPass;
use crate::render_pass::{ColorTarget, PassLayout, Target};
use crate::render_state::EncoderState;
use crate::render_target::{self, Feedback, PassShape, Position, RenderPassObject};
use crate::render_values::DrawPrimitiveType;
use crate::residency::{every_render_stage, Residency};
use crate::resource::ResourceKey;
use crate::sample_planes::{PlanesCommands, PlanesStep};
use crate::sampler::ArgumentSamplers;
use crate::shadow::{self, ShadowCopies, ShadowUse};
use crate::shadow_stencil::ShadowStencil;
use crate::tessellation::{FactorBuffer, Factors};
use crate::tracker::{
    Barrier, EncoderUses, GeometryWrites, IndirectReads, Summary, UnorderedWrites, Use,
};
use crate::versioned::Versioned;
use crate::visibility::{Lent, Visibility};
use crate::written_commands::{Writers, WrittenCommands};
use crate::Device;
use ahead::Ahead;
use ash::vk;
use cause::HostStepCause;
use gpu::GpuForm;
pub(super) use gpu::ReplayedDraws;
use metal2vulkan::bind::indirect_commands::{Action, BufferStage, CommandRecord};
use metal2vulkan::bind::{DrawCall, ReplayFaults, ViewSetup};
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PrimitiveTopologyClass, PrimitiveType, VertexStepFunction,
};
use metal2vulkan::pipeline::{CommandWrites, VertexStride};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

mod ahead;
mod cause;
mod gpu;

/// The call an execution's refusals name.
const CALL: &str = "execute_commands_in_buffer";

/// The pipeline state an execution runs with (`RenderCommandEncoder::inherited_pipeline`).
enum Inherited {
    /// Each command's own: the buffer does not inherit it.
    Commands,
    /// The encoder's, which the buffer inherits.
    Encoder(RenderPipelineState),
    /// None: the encoder's, which the buffer inherits, was made without
    /// `supportIndirectCommandBuffers`, and Metal's GPU faults the encoder.
    Faults,
}

impl RenderLowering {
    /// The lowering of [`RenderCommandEncoder::execute_commands_in_buffer`].
    pub(crate) fn execute_commands_in_buffer(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
    ) -> Result<(), Error> {
        self.execute_range(icb, range, None)
    }

    /// The lowering of [`RenderCommandEncoder::execute_commands_in_buffer_checked`].
    pub(crate) fn execute_commands_in_buffer_checked(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
        check: IndirectCommandCheck,
        needs: IndirectCommandCheckNeeds,
    ) -> Result<(), Error> {
        self.execute_range(icb, range, Some((check, needs)))
    }

    /// [`execute_commands_in_buffer`](Self::execute_commands_in_buffer), with `check` for its
    /// host step where one is given, and what it looks for.
    fn execute_range(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
        check: Option<(IndirectCommandCheck, IndirectCommandCheckNeeds)>,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::Execute(icb.clone(), range.clone()));
        self.core.check_device(CALL, icb.device())?;
        if !icb.is_render() {
            if !icb.runs_nothing_in_a_pass() {
                return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
            }
            icb.command_range(CALL, range)?;
            return Ok(());
        }
        let range = icb.command_range(CALL, range)?;
        // An empty range runs nothing before the inherited pipeline state is looked at, as plain
        // Metal's: an unset state, or one without the support, faults only a range that runs.
        if range.is_empty() {
            return Ok(());
        }
        // Not in a pass of several color groups (`render/color_groups.rs`).
        self.check_group_execution()?;
        let pipeline = match self.inherited_pipeline(CALL, icb)? {
            Inherited::Faults => return Ok(()),
            Inherited::Commands => None,
            Inherited::Encoder(state) => Some(state),
        };
        // A draw of primitive type 9 among the commands as their records stand fails the command
        // buffer as Metal's GPU fails it there, the encoder running nothing.
        if let Some(nine) = icb.nine_among(range.clone())? {
            match nine {
                NineDraw::Hangs => self.core.hang(),
                NineDraw::Faults => self.core.fault(CommandBufferError::PageFault),
            }
            return Ok(());
        }
        self.degrade_repeated_start(icb, &range);
        self.defer_execution(icb, ExecutionRange::Fixed(range), pipeline, check)
    }

    /// The lowering of [`RenderCommandEncoder::execute_commands_in_buffer_indirect`].
    pub(crate) fn execute_commands_in_buffer_indirect(
        &mut self,
        icb: &IndirectCommandBuffer,
        range_buffer: &Buffer,
        offset: u64,
    ) -> Result<(), Error> {
        self.execute_indirect_range(icb, range_buffer, offset, None)
    }

    /// The lowering of [`RenderCommandEncoder::execute_commands_in_buffer_indirect_checked`].
    pub(crate) fn execute_commands_in_buffer_indirect_checked(
        &mut self,
        icb: &IndirectCommandBuffer,
        range_buffer: &Buffer,
        offset: u64,
        check: IndirectCommandCheck,
        needs: IndirectCommandCheckNeeds,
    ) -> Result<(), Error> {
        self.execute_indirect_range(icb, range_buffer, offset, Some((check, needs)))
    }

    /// [`execute_commands_in_buffer_indirect`](Self::execute_commands_in_buffer_indirect), with
    /// `check` for its host step where one is given, and what it looks for.
    fn execute_indirect_range(
        &mut self,
        icb: &IndirectCommandBuffer,
        range_buffer: &Buffer,
        offset: u64,
        check: Option<(IndirectCommandCheck, IndirectCommandCheckNeeds)>,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::ExecuteIndirect(icb.clone(), range_buffer.clone(), offset));
        const CALL_INDIRECT: &str = indirect_execute::CALL_INDIRECT;
        self.core.check_device(CALL_INDIRECT, icb.device())?;
        self.core
            .check_device(CALL_INDIRECT, range_buffer.device())?;
        if !icb.is_render() {
            return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
        }
        indirect_execute::check_range_buffer(range_buffer.length(), offset)?;
        // Not in a pass of several color groups (`render/color_groups.rs`).
        self.check_group_execution()?;
        let pipeline = match self.inherited_pipeline(CALL_INDIRECT, icb)? {
            Inherited::Faults => return Ok(()),
            Inherited::Commands => None,
            Inherited::Encoder(state) => Some(state),
        };
        let read = [reads(
            range_buffer,
            offset..offset + indirect_execute::RANGE_BYTES,
        )];
        if self.core.uses().written(&read) {
            return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
        }
        let copy = indirect_dispatch::copy_arguments(
            &mut self.core,
            range_buffer,
            offset,
            indirect_execute::RANGE_BYTES,
        )?;
        self.defer_execution(icb, ExecutionRange::Indirect(copy), pipeline, check)
    }

    /// The pipeline state an execution of `icb` here runs with (`Inherited`). Where the
    /// encoder's, which `icb` inherits, was made without `supportIndirectCommandBuffers`,
    /// Metal's GPU faults the encoder, which runs nothing, and the command buffer completes with
    /// `CommandBufferError::PageFault`: the encoder is marked so (`EncoderCore::fault`). Refused
    /// (`Invalid`) for `call`: an inherited state that is unset (`NoPipelineState`), at which
    /// Metal's process ends.
    fn inherited_pipeline(
        &mut self,
        call: &'static str,
        icb: &IndirectCommandBuffer,
    ) -> Result<Inherited, Error> {
        match (&self.pipeline, icb.descriptor().inherit_pipeline_state) {
            (_, false) => Ok(Inherited::Commands),
            (None, true) => Err(invalid(call, Problem::NoPipelineState)),
            (Some(state), true) if !state.support_indirect_command_buffers() => {
                self.core.fault(CommandBufferError::PageFault);
                Ok(Inherited::Faults)
            }
            (Some(state), true) => Ok(Inherited::Encoder(state.clone())),
        }
    }

    /// Announce, once per device, an execution of commands `range` of `icb` from a command an
    /// earlier execution of `icb` in this encoder started at with another length, both ranges
    /// given (the host step reads the others, [`DeferredPass::record`]).
    fn degrade_repeated_start(&self, icb: &IndirectCommandBuffer, range: &Range<u32>) {
        let earlier = self
            .executions
            .iter()
            .find_map(|execution| match &execution.range {
                ExecutionRange::Fixed(earlier)
                    if execution.icb == *icb
                        && earlier.start == range.start
                        && earlier.end != range.end =>
                {
                    Some(earlier.len())
                }
                ExecutionRange::Fixed(_) | ExecutionRange::Indirect(_) => None,
            });
        if let Some(earlier) = earlier {
            degrade_repeated_start(self.core.device(), icb, range, earlier);
        }
    }

    /// Leave the execution of `range` of `icb` to record with its pass (the module docs), with
    /// the encoder's pipeline state `pipeline` where `icb` inherits it.
    fn defer_execution(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: ExecutionRange,
        pipeline: Option<RenderPipelineState>,
        check: Option<(IndirectCommandCheck, IndirectCommandCheckNeeds)>,
    ) -> Result<(), Error> {
        self.sync_heaps();
        // Its commands run the states their records name, where it inherits none.
        if !icb.descriptor().inherit_pipeline_state {
            self.writers.runs(icb.written().named().writes);
        }
        // An execution keeps an extended-range, padded or bfloat attachment's texels as a draw
        // does, of reset commands too, and a signed normalized one's minimum codes, and an
        // `RGB9E5Float` one's words.
        self.degrade_kept_attachments();
        self.degrade_extended_range_execution();
        // Its commands write the extended-range colors' codes alone: the shadows end.
        self.end_extended_shadows();
        // Commands that set their own depth/stencil states may write the depth.
        self.depth_writes |= !icb.descriptor().inherit_depth_stencil_state;
        // And its commands hand the device's pass the states they draw under: their own, which
        // may write the depth, or the encoder's.
        self.drew_depth_writes |=
            !icb.descriptor().inherit_depth_stencil_state || self.state.depth_writes();
        // Where a visibility result mode is set, the commands' draws count in a query of their
        // own that continues the active offset, and the encoder's draws after them in another.
        let device = self.core.device().clone();
        // With a spare continuation for each time a command's draw may end the query and begin
        // another: each command may begin another run of the execution, and, under a state whose
        // compare fails every fragment without Metal's depth attachment, count no sample. A range
        // read from a buffer runs at most every command of the buffer.
        let commands = match &range {
            ExecutionRange::Fixed(range) => u64::from(range.end - range.start),
            ExecutionRange::Indirect(_) => icb.size(),
        };
        let (depth, _) = self.layout.aspects();
        let metal_depth = depth
            && !self
                .layout
                .depth_stencil
                .as_ref()
                .is_some_and(|ds| ds.hidden);
        let resumes = self.state.failing_depth(metal_depth).is_some();
        let spares = commands.saturating_mul(1 + u64::from(resumes));
        let queries = self.visibility.lend(&device, spares)?;
        // The commands' draws test and write Metal's hidden depth and stencil under the state at
        // the call, as the encoder's own draw now would, shared with the encoder's draws before
        // and after them: where the state needs them, the call gives the pass
        // them, so the execution's instance holds them. Where the pass can take none, the call
        // gives none, and a command's draw that needs them is refused as before
        // (`hidden_depth_stencil`); an execution of no such draw runs.
        let hidden = self.needed_hidden().ok().flatten();
        // The draws before it are a piece of the instance it joins, or of the one it follows:
        // the current one's secondary begins where the pass left it unbegun (`Instance::draws`).
        self.begin_current_draws()?;
        // The commands draw into an instance of no fetched attachments, shader-side stencil or
        // views: the current one where it is such an instance and no tile dispatch or memory
        // barrier follows it, else a new one, as a draw of such a pipeline records.
        // Not an instance recorded inline, which runs no secondary (`render/inline_first.rs`):
        // a new one that continues it, where it would have taken them. Nor one whose colors are
        // in the feedback loop layout, which runs as one render pass instance and whose
        // descriptors of those colors name that layout (`render/snapshot.rs`): the commands read
        // the attachments in place, in `GENERAL`.
        let continues = self.current.inline
            && self.takes_current(
                (&[], &[], &[]),
                (false, false),
                ViewSetup::default(),
                hidden.is_some(),
            );
        let apart = self.current.inline || self.current.feedback != Feedback::NONE;
        let (mut next, stood) = self.next_instance(
            (&[], None),
            (false, false),
            ViewSetup::default(),
            apart,
            hidden,
            &[],
            &[],
        )?;
        if let Next::New { instance, .. } = &mut next {
            instance.continues = continues;
        }
        let continuation = match &next {
            Next::New { split: true, .. } => self.visibility.continuation(&device)?,
            Next::New { .. } | Next::Current => None,
        };
        self.begin_new_draws(&mut next)?;
        self.install(
            next,
            stood,
            continuation,
            crate::work_counts::Split::Execution,
        );
        let draws = self
            .core
            .command_buffer()
            .begin_draws(&self.current.compatible)?;
        let before = self.current.draws;
        self.current.draws = draws;
        let raw = device.shared().vk();
        let lent = queries.map(|queries| self.visibility.lend_to(raw, before, draws, queries));
        self.current.pieces.push(Piece::Draws(before));
        self.current
            .pieces
            .push(Piece::Execution(self.executions.len()));
        self.current.recorded = true;
        // The execution records in the instance: its draws record again no more.
        self.current.rerecord = None;
        // The commands may reach the pass's fragment imageblock storage: the encoder's draw after
        // them that reaches it records after a barrier from them, as from its own draws.
        self.current.imageblock = true;
        // So may they write the instance's interlocked colors (`InterlockedStorage`).
        self.current.interlocked_writes = InterlockedStorage::ALL;
        self.called = true;
        self.drew = true;
        // Its pixels are not bounded by a viewport the host knows of.
        self.drawn.everywhere();
        self.coverage.everywhere();
        let inherited = icb.descriptor().inherit_buffers;
        // An execution's commands take their flat inputs from the first vertex whatever the
        // encoder's provoking vertex mode (Metal probes on the M3 Max, macOS 27.0.1).
        let mut state = (*self.state).clone();
        state.set_provoking_last(false);
        state.set_provoking_third(false);
        let snapshot = Snapshot {
            layout: self.layout.clone(),
            state,
            vertex: if inherited {
                self.vertex.clone()
            } else {
                ArgumentTable::default()
            },
            fragment: if inherited {
                self.fragment.clone()
            } else {
                ArgumentTable::default()
            },
            object: if inherited {
                self.object.clone()
            } else {
                ArgumentTable::default()
            },
            mesh: if inherited {
                self.mesh.clone()
            } else {
                ArgumentTable::default()
            },
            render_target_array_length: self.render_target_array_length,
            amplification: self.amplification.clone(),
            factors: self.factors.clone(),
            rate_map: self.rate_map.clone(),
            residency: self.residency.clone(),
            tile_imageblock: self.tile_imageblock,
            shape: self.current.shape.clone(),
            compatible: Arc::clone(&self.current.compatible),
            tile_memory: self.tile_memory.arguments(),
            tile_memory_reached: self.current.tile_memory,
            indirect_reads: self.current.indirect_reads.clone(),
            geometry_writes: self.current.geometry_writes.clone(),
            drawn: self.core.uses().tracked_in(PASS_DRAW_STAGES),
            pass_start: self.pass_start.notes(),
        };
        // The commands may reach the tile threadgroup memory wherever the encoder set an argument
        // of it: a draw after them that reaches it records into a new instance, so it sees their
        // writes (`render/tile_memory.rs`).
        self.current.tile_memory |= snapshot.tile_memory.set();
        // The shadowed textures the commands may use, copied around the pass as a draw's are
        // (`render/shadows.rs`): the tables' they take, whose use may write, and the resident
        // ones, writing where their usage may.
        let tables = [
            &snapshot.vertex,
            &snapshot.fragment,
            &snapshot.object,
            &snapshot.mesh,
        ]
        .into_iter()
        .flat_map(|table| table.textures.values())
        .map(|texture| (texture, true));
        let mut shadows = Vec::new();
        for (texture, writes) in tables.chain(snapshot.residency.textures_written()) {
            if let Some(copies) = texture.shadow() {
                let used = ShadowUse {
                    copies: Arc::clone(copies),
                    writes,
                    overwrites: false,
                    kept: false,
                    area: None,
                    loads: None,
                    tile_stores: false,
                };
                shadow::join(&mut shadows, &used);
            }
        }
        self.gathering().add(&shadows, &[]);
        // The bytes the commands may reach, so the copies between the pass's instances and as
        // the encoder ends keep what they read and write (`render/shadows.rs`): the resident
        // buffers and heaps, writing where their usage may, and the encoder's buffers where the
        // commands inherit them, writing.
        let bound = [
            &snapshot.vertex,
            &snapshot.fragment,
            &snapshot.object,
            &snapshot.mesh,
        ]
        .into_iter()
        .flat_map(|table| table.buffers.values())
        .map(|(buffer, _)| (buffer, true));
        for (buffer, write) in snapshot.residency.buffers_written().chain(bound) {
            self.gathering().add_bytes(buffer, write);
        }
        self.core.retain(icb.clone());
        self.executions.push(RenderExecution {
            icb: icb.clone(),
            range,
            pipeline,
            snapshot,
            shadows,
            lent,
            needs: check
                .as_ref()
                .map_or(IndirectCommandCheckNeeds::ALL, |(_, needs)| *needs),
            check: check.map(|(check, _)| Mutex::new(check)),
            read_at: AtomicU64::new(indirect_execute::NOT_READ),
            encoder: self.core.command_buffer().state().open_encoder_place(),
        });
        // Its commands may name any pipeline state, reading handles in any stage.
        self.command_encoded(every_render_stage());
        Ok(())
    }
}

impl RenderCommandEncoder {
    /// `executeCommandsInBuffer:withRange:`: run render commands `range` of `icb` here, as their
    /// records stand when the command buffer's batch that executes them is submitted (the commit
    /// reads them, and the batch's submission again where the CPU wrote them since), or, where an
    /// encoder made `icb` resident for writing, once the work before them completed (a host step
    /// the device announces once, `Degradation::HostReplayedIndirectCommands`). Each draws as this
    /// encoder's `draw_primitives`, `draw_indexed_primitives` or `draw_patches` would, with the
    /// encoder's state as it is now, and its own pipeline state and buffers, or the encoder's
    /// where `icb` inherits them (the module docs).
    /// The encoder's draws before run before them and its draws after run after them.
    ///
    /// Refused, changing nothing (`Invalid`): a buffer of another device, a range that ends past
    /// its size (`IndirectCommandBufferProblem::Range`), and, for a range that is not empty, an
    /// inherited pipeline state that is unset (`NoPipelineState`), at which Metal's process ends.
    /// For a range that is not empty, an inherited state made without
    /// `supportIndirectCommandBuffers` faults Metal's GPU at the encoder: the call is taken, the
    /// encoder runs nothing, not even its load, clear or store actions, and the command buffer
    /// completes with [`CommandBufferError::PageFault`](crate::CommandBufferError::PageFault), the
    /// other encoders run and every event signal fires. A buffer of no command types, or of
    /// dispatches beside draws of primitives, runs nothing, as plain Metal's; one of dispatches
    /// alone, or beside other draws (unmeasured in a pass), is
    /// `NotImplemented(IndirectRenderCommands)`. An empty range runs nothing, whatever the
    /// encoder's pipeline state, as plain Metal's. What a command's draw would be
    /// refused refuses the commit ([`CommandBuffer::commit`]). Where a visibility result mode is
    /// set, the commands' draws count in it (the module docs).
    pub fn execute_commands_in_buffer(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.execute_commands_in_buffer(icb, range)),
            false => self.defer(|_| RenderCall::Execute(icb.clone(), range.clone())),
        }
    }

    /// [`execute_commands_in_buffer`](Self::execute_commands_in_buffer) with a check of the
    /// commands it runs (a runtime extension; Metal has none), which looks for `needs`: where the
    /// commit records the pass from the CPU's records, or a host step from the records as the GPU
    /// and the CPU left them (an encoder made `icb` resident for writing,
    /// `IndirectCommandBuffer::gpu_written`), it first calls `check` with each command of the range
    /// it will record, in order, before it records any of them ([`IndirectCommandCheck`]). Neither
    /// calls it where what `icb`'s records may hold rules out every kind `needs` names
    /// ([`IndirectCommandCheckNeeds`], [`IndirectCommandCheckNeeds::ALL`] included); nor does an
    /// execution that runs on the GPU alone (`render/executions/gpu.rs`), which it may only where
    /// they rule them out at the commit and again as the pass's batch is ready to submit. Refused
    /// as [`execute_commands_in_buffer`](Self::execute_commands_in_buffer) refuses.
    pub fn execute_commands_in_buffer_checked(
        &mut self,
        icb: &IndirectCommandBuffer,
        range: Range<u64>,
        check: IndirectCommandCheck,
        needs: IndirectCommandCheckNeeds,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.execute_commands_in_buffer_checked(icb, range, check, needs)
            }),
            false => self.defer(|_| {
                let check = SharedCheck::new(check);
                RenderCall::ExecuteChecked(Box::new((icb.clone(), range.clone(), check, needs)))
            }),
        }
    }

    /// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`: run the render commands
    /// of `icb` that the `MTLIndirectCommandBufferExecutionRange` at `offset` of `range_buffer`
    /// (`location`, then `length`, each a `u32`) names as the execution runs, as
    /// [`execute_commands_in_buffer`](Self::execute_commands_in_buffer) runs a range given. The
    /// range is taken as Metal's execution takes it (a probe, an M3 Max and an M4): from the
    /// location, clamped at the buffer's size, so a length of 0 or a location at the size runs
    /// nothing and a length past it runs to its end; a location past the size, or a range past
    /// `2^32 - 1`, at which Metal's GPU faults the command buffer, runs nothing, and the command
    /// buffer's execution report states `ReplayFaults::RANGE`. The encoder copies the range where
    /// it is encoded, behind a barrier from everything before, so the range a kernel of an earlier
    /// encoder wrote is read; a host step records the pass once the work before it completed, which
    /// the device announces once (`Degradation::HostReplayedIndirectCommands`).
    ///
    /// A range plain Metal takes and the runtime has no exact answer for is its own gap, not
    /// misuse: 8 bytes that run past `range_buffer`, which Metal reads from whatever follows it
    /// (`NotImplemented(ExecutionRangePastBuffer)`), and an `offset` that is not a multiple of 4
    /// bytes, which Metal's GPU faults the command buffer over where the range read there runs a
    /// command (`NotImplemented(ExecutionRangeOffFour)`; both carry the measurement).
    ///
    /// Refused, changing nothing: what
    /// [`execute_commands_in_buffer`](Self::execute_commands_in_buffer) refuses but a range (an
    /// unset inherited state whatever the range), and a range buffer of another device. A range a draw of the
    /// encoder before the call may write is `NotImplemented(IndirectRenderCommands)`: Metal's
    /// execution reads that draw's write, where the runtime's copy, made before the pass, would
    /// not; as for the index bounds pass, an untracked buffer's writes are not seen. Under an
    /// inherited state made without `supportIndirectCommandBuffers`, whatever the range, `{0, 0}`
    /// included, the encoder faults as `execute_commands_in_buffer`'s does.
    pub fn execute_commands_in_buffer_indirect(
        &mut self,
        icb: &IndirectCommandBuffer,
        range_buffer: &Buffer,
        offset: u64,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.execute_commands_in_buffer_indirect(icb, range_buffer, offset)
            }),
            false => self
                .defer(|_| RenderCall::ExecuteIndirect(icb.clone(), range_buffer.clone(), offset)),
        }
    }

    /// [`execute_commands_in_buffer_indirect`](Self::execute_commands_in_buffer_indirect) with a
    /// check of the commands its host step runs, which always records the pass, looking for
    /// `needs`: the step first calls `check` with each command of the range it read that it will
    /// record, in order, before it records any of them, as
    /// [`execute_commands_in_buffer_checked`](Self::execute_commands_in_buffer_checked)'s does
    /// ([`IndirectCommandCheck`]), unless what `icb`'s records may hold then rules out every kind
    /// `needs` names ([`IndirectCommandCheckNeeds`]). Refused as
    /// [`execute_commands_in_buffer_indirect`](Self::execute_commands_in_buffer_indirect) refuses.
    pub fn execute_commands_in_buffer_indirect_checked(
        &mut self,
        icb: &IndirectCommandBuffer,
        range_buffer: &Buffer,
        offset: u64,
        check: IndirectCommandCheck,
        needs: IndirectCommandCheckNeeds,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| {
                lowering.execute_commands_in_buffer_indirect_checked(
                    icb,
                    range_buffer,
                    offset,
                    check,
                    needs,
                )
            }),
            false => self.defer(|_| {
                let check = SharedCheck::new(check);
                let call = (icb.clone(), range_buffer.clone(), offset, check, needs);
                RenderCall::ExecuteIndirectChecked(Box::new(call))
            }),
        }
    }
}

/// Announce, once per device, an execution of commands `range` of `icb` after one in the same
/// encoder from the same command with length `earlier`, another: Metal's executions then do not
/// each run their own range, and not the same way from run to run (the module docs); the
/// runtime's run their own.
fn degrade_repeated_start(
    device: &Device,
    icb: &IndirectCommandBuffer,
    range: &Range<u32>,
    earlier: usize,
) {
    device.degrade(Degradation::RepeatedExecutionStart, || {
        (
            format!(
                "an indirect command buffer of {} render commands at resource ID {:#x}",
                icb.size(),
                icb.gpu_resource_id().0
            ),
            format!(
                "a render encoder executes it from command {} with length {} after executing it \
                 from there with length {}",
                range.start,
                range.len(),
                earlier
            ),
        )
    });
}

/// An execution of render commands, left to record with its pass ([`DeferredPass`]).
pub(crate) struct RenderExecution {
    icb: IndirectCommandBuffer,
    /// The commands it runs: a range inside the buffer's size, or one read from a buffer.
    range: ExecutionRange,
    /// The encoder's pipeline state where the buffer inherits it.
    pipeline: Option<RenderPipelineState>,
    /// The encoder as the call found it.
    snapshot: Snapshot,
    /// The shadowed textures its commands may use, gathered at the call: a draw that uses
    /// another is refused.
    shadows: Vec<ShadowUse>,
    /// The query its commands' draws count in, where a visibility result mode was set at the
    /// call.
    lent: Option<Lent>,
    /// The caller's check of the commands the host step records
    /// (`execute_commands_in_buffer_checked`), which the step calls before it records them.
    check: Option<Mutex<IndirectCommandCheck>>,
    /// What the check looks for, which decides whether the execution may run on the GPU alone
    /// (`gpu.rs`): every kind where it has no check.
    needs: IndirectCommandCheckNeeds,
    /// Where `commit` recorded it from the CPU's records, the count of the CPU's writes of them it
    /// read them after (`IndirectCommandBuffer::cpu_writes`); `indirect_execute::NOT_READ` until
    /// then ([`rewritten`](Self::rewritten)).
    read_at: AtomicU64,
    /// Its encoder's place among the command buffer's ended encoders, which names a draw of its
    /// commands that its replay drops (`CommandBuffer::drop_late_draw`).
    encoder: usize,
}

/// The encoder's state an execution's draws take: as the call found it, its tables only where
/// the buffer inherits them, and the instance it was encoded in.
struct Snapshot {
    layout: PassLayout,
    state: EncoderState,
    vertex: ArgumentTable,
    fragment: ArgumentTable,
    object: ArgumentTable,
    mesh: ArgumentTable,
    render_target_array_length: u32,
    amplification: Amplification,
    factors: Factors,
    rate_map: Option<RateMappedPass>,
    residency: Residency,
    tile_imageblock: bool,
    shape: PassShape,
    compatible: Arc<RenderPassObject>,
    /// The encoder's tile threadgroup arguments (`render/tile_memory.rs`), over no buffer.
    tile_memory: TileMemory,
    /// A draw of the instance before the call reached the tile threadgroup memory.
    tile_memory_reached: bool,
    /// What the draws of the instance before the call read as their indirect arguments: a
    /// command's draw that writes any of it records after a barrier from those reads.
    indirect_reads: IndirectReads<ResourceKey>,
    /// What the draws of the instance before the call wrote before their fragments: a command's
    /// draw whose fragment shader touches any of it records after a barrier from those writes.
    geometry_writes: GeometryWrites<ResourceKey>,
    /// The tracked uses of the encoder's draws before the call, in the stages a draw runs in
    /// inside its pass: an emulated mesh command's dispatches, recorded before the pass, must not
    /// touch them but as Metal orders them (`RenderCommandEncoder::drawn_before`).
    drawn: Vec<Use<ResourceKey>>,
    /// What the encoder's draws before the call noted of the bytes the pass's fragment stages
    /// read and wrote (`render/pass_start_reads.rs`), which the commands' draws start from.
    pass_start: PassStartReads,
}

/// A command to run: its record, and the render pipeline state it names (`None`: none).
type Command = (CommandRecord, Option<RenderPipelineState>);

/// What an execution's recording found: resident-set misses and faults, and the runs its draws
/// that reach the tile threadgroup memory began, and those its commands' barriers began.
#[derive(Default)]
struct Found {
    misses: u32,
    faults: ReplayFaults,
    tile_memory_splits: usize,
    barrier_splits: usize,
}

/// An execution's recording: the encoder core back, with what the draws noted, retained and
/// executed; its runs, secondaries begun against its instance and ended, each after the first
/// run in a render pass instance of its own after a barrier from everything before (a draw that
/// reaches the tile threadgroup memory after another of the run did begins the next,
/// `render/tile_memory.rs`, and so does a command's barrier); and the pass's storage, with what
/// its draws made of it.
struct Recorded {
    core: EncoderCore,
    runs: Vec<Run>,
    storage: PassStorage,
    /// The pass's layout as the last command's draw left it: with the stand-ins and sample
    /// planes its runs' draws read and write their fetched attachments in.
    layout: PassLayout,
}

/// The steps around a run of an execution whose draws read and write fetched attachments in the
/// pixel interlock elsewhere than the pass's instances hold them: the copies from the image an
/// instance renders such an attachment into (the texture, or the pass's scratch for it) into the
/// scratch the run's draws reach it in (`attachment_stand_in.rs`) and back, and the sample
/// planes' split from that image before the run and merge into it after (`sample_planes.rs`).
/// Each runs between barriers from everything before and to everything after, so the run sees
/// what the draws before it rendered, and the draws after it what it wrote, as the encoder's
/// own instances of such draws do.
struct RunSteps {
    into: Option<StandInCopies>,
    back: Option<StandInCopies>,
    planes: Option<PlanesCommands>,
}

/// Hold until the command buffer completes the images `layout`, an execution's layout as its
/// commands left it, renders and reads its colors in: the stand-ins and sample planes their draws
/// made among them.
fn hold_colors(core: &mut EncoderCore, layout: &PassLayout) {
    let colors = layout.colors.iter().flatten();
    let images: Vec<_> = colors
        .flat_map(|color| {
            let planes = color.planes.as_ref().map(|planes| planes.texture.clone());
            std::iter::once(color.target.texture.clone()).chain(planes)
        })
        .collect();
    core.retain(images);
}

/// Whether `a` and `b` are the same level and layer of the same image.
fn same_subresource(a: &Target, b: &Target) -> Result<bool, Error> {
    Ok(
        a.texture.raw_image()? == b.texture.raw_image()?
            && a.level == b.level
            && a.layer == b.layer,
    )
}

/// `own`, a pass's layout for one of its instances, with `framed`'s stand-ins in place of its
/// color attachments of their indices (the pass's own load and store actions kept).
fn framed_layout<'a>(own: &'a PassLayout, framed: &[(usize, Target)]) -> Cow<'a, PassLayout> {
    if framed.is_empty() {
        return Cow::Borrowed(own);
    }
    let mut layout = own.clone();
    for (index, target) in framed {
        if let Some(Some(color)) = layout.colors.get_mut(*index) {
            color.target = target.clone();
        }
    }
    Cow::Owned(layout)
}

/// A [`DeferredPass`]'s layout as its instances render it, and that layout without the hidden
/// depth and stencil (`bare`): as the encoder ended it, and from the first execution whose
/// commands ran as several on, with the backings of its memoryless attachments in their place
/// (`backed`).
struct Rendered {
    layout: PassLayout,
    bare: Option<PassLayout>,
    backed: Vec<Backing>,
    /// The queue of the pass's command buffer, whose backings a memoryless texture keeps
    /// (`attachment_stand_in::kept_backing`).
    queue: u64,
}

/// The backing of a pass's memoryless attachment, `own`, color `index` or (`None`) the
/// depth/stencil: the image the pass renders it into (`attachment_stand_in.rs`).
#[derive(Clone)]
struct Backing {
    index: Option<usize>,
    target: Target,
    own: Target,
}

impl Rendered {
    /// `layout`, an execution's at its call, with the pass's backings in place of the memoryless
    /// attachments they back.
    fn backing(&self, layout: &PassLayout) -> Result<PassLayout, Error> {
        let mut layout = layout.clone();
        back(&mut layout, &self.backed)?;
        Ok(layout)
    }

    /// Back the pass's memoryless attachments that nothing backs yet, for an execution whose
    /// commands ran as several, `run` its layout as its last command left it. Metal's
    /// attachment holds its texels across the pass; the runs' render pass instances load and
    /// store it between them, which keeps the texels of an image in memory that is not lazily
    /// allocated, but not of one in lazily allocated memory, which MoltenVK runs as Metal's
    /// memoryless storage (`Texture::lazily_allocated`). So the pass renders such an attachment
    /// into a backing from its first instance on, its own load and store actions kept, as the
    /// encoder's own instances keep a memoryless attachment in a stand-in from a split on
    /// (`render.rs`); and so it renders one the commands' draws rendered into a stand-in (a
    /// split of theirs, or a fetch), into that stand-in. Refused
    /// (`NotImplemented(IndirectRenderCommands)`) where the pass reads such an attachment past
    /// its instances: a color one's sample planes (`planes`) or its resolve drawn after the
    /// pass (`color_resolve.rs`), and a depth/stencil one's resolve.
    fn back_memoryless(
        &mut self,
        maker: &CommandBuffer,
        run: &mut PassLayout,
    ) -> Result<(), Error> {
        let refused = || Error::NotImplemented(RuntimeGap::IndirectRenderCommands);
        let mut backed = Vec::new();
        for (index, color) in self.layout.colors.iter().enumerate() {
            let Some(color) = color.as_ref().filter(|color| color.stand_in.is_none()) else {
                continue;
            };
            let ran = run.colors.get(index).and_then(Option::as_ref);
            let ran = ran.map(|ran| (&ran.target, ran.stand_in.as_ref()));
            let Some(backing) = backing(maker, Some(index), &color.target, ran, self.queue)? else {
                continue;
            };
            let drawn_after = color.resolve.as_ref().is_some_and(|r| r.shader.is_some());
            if color.planes.is_some() || drawn_after {
                return Err(refused());
            }
            backed.push(backing);
        }
        if let Some(ds) = self.layout.depth_stencil.as_ref() {
            let ran = run.depth_stencil.as_ref();
            let ran = ran.map(|ran| (&ran.target, ran.stand_in.as_ref()));
            if ds.stand_in.is_none() {
                if let Some(backing) = backing(maker, None, &ds.target, ran, self.queue)? {
                    if ds.resolve.is_some() {
                        return Err(refused());
                    }
                    backed.push(backing);
                }
            }
        }
        back(&mut self.layout, &backed)?;
        if let Some(bare) = &mut self.bare {
            back(bare, &backed)?;
        }
        back(run, &backed)?;
        self.backed.extend(backed);
        Ok(())
    }
}

/// The backing of `own`, color `index` or (`None`) the depth/stencil, where it is a memoryless
/// attachment that needs one (`Rendered::back_memoryless`): the stand-in a command's draw made
/// for it, where `ran` (the attachment in the execution's layout, and what its target stands in
/// for) holds one, else, where its image is lazily allocated, the one its texture keeps for the
/// passes of `queue` (A new one each pass before; `attachment_stand_in::kept_backing`); `None`
/// otherwise.
fn backing(
    maker: &CommandBuffer,
    index: Option<usize>,
    own: &Target,
    ran: Option<(&Target, Option<&StandIn>)>,
    queue: u64,
) -> Result<Option<Backing>, Error> {
    if !super::memoryless(own) {
        return Ok(None);
    }
    let made = match ran {
        Some((target, Some(stand_in))) if same_subresource(&stand_in.own, own)? => {
            Some(target.clone())
        }
        Some(_) | None => None,
    };
    let target = match made {
        Some(target) => target,
        None if own.texture.lazily_allocated() => {
            attachment_stand_in::kept_backing(maker, own, queue, false)?
        }
        None => return Ok(None),
    };
    Ok(Some(Backing {
        index,
        target,
        own: own.clone(),
    }))
}

/// `layout` with `backed`'s images in place of the memoryless attachments they back, where it
/// renders such an attachment itself.
fn back(layout: &mut PassLayout, backed: &[Backing]) -> Result<(), Error> {
    for backing in backed {
        let stand_in = || StandIn {
            own: backing.own.clone(),
            cause: Cause::Backing,
            area: None,
        };
        match backing.index {
            Some(index) => {
                if let Some(Some(color)) = layout.colors.get_mut(index) {
                    if color.stand_in.is_none() && same_subresource(&color.target, &backing.own)? {
                        color.target = backing.target.clone();
                        color.stand_in = Some(stand_in());
                    }
                }
            }
            None => {
                if let Some(ds) = &mut layout.depth_stencil {
                    if ds.stand_in.is_none() && same_subresource(&ds.target, &backing.own)? {
                        ds.target = backing.target.clone();
                        ds.stand_in = Some(stand_in());
                    }
                }
            }
        }
    }
    Ok(())
}

/// The color indices, ascending, that a run of `shape` in a pass reading its fetched attachments
/// as input attachments reads so, where `run`, the execution's layout, renders them into another
/// image than `pass`, the pass's layout as its instances render it: a stand-in a command's draw
/// made for an attachment whose image takes no input attachment usage (`attachment_stand_in.rs`).
fn framed(run: &PassLayout, pass: &PassLayout, shape: &PassShape) -> Result<Vec<u32>, Error> {
    let mut framed = Vec::new();
    for (index, (ran, rendered)) in run.colors.iter().zip(&pass.colors).enumerate() {
        let (Some(ran), Some(rendered)) = (ran, rendered) else {
            continue;
        };
        let index = u32::try_from(index).map_err(|_| internal("a color index past u32"))?;
        if !shape.no_inputs.contains(&index) && !same_subresource(&ran.target, &rendered.target)? {
            framed.push(index);
        }
    }
    Ok(framed)
}

impl RunSteps {
    /// The steps around a run that leaves `storage` (ascending color indices) out of its subpass,
    /// whose draws read them in `run`, the execution's layout, in a pass whose instances render
    /// them as `pass` lays them out; `None` where it needs none (every attachment rendered where
    /// the instances render it, and none read as sample planes). Through input attachments,
    /// `framed` (ascending color indices) are the attachments its framebuffer holds as `run` lays
    /// them out, in place of the pass's: each is copied in and back too.
    fn plan(
        device: &Device,
        run: &PassLayout,
        pass: &PassLayout,
        (storage, framed): (&[u32], &[u32]),
    ) -> Result<Option<RunSteps>, Error> {
        fn color(layout: &PassLayout, index: u32) -> Result<&ColorTarget, Error> {
            layout
                .colors
                .get(index as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| internal("a run's storage attachment the pass does not have"))
        }
        let mut pairs = Vec::new();
        let (mut split, mut merge) = (Vec::new(), Vec::new());
        for &index in storage {
            let (ran, rendered) = (color(run, index)?, color(pass, index)?);
            if !same_subresource(&ran.target, &rendered.target)? {
                pairs.push((&ran.target, &rendered.target, pass.extent));
            }
            if ran.planes.is_some() {
                split.push(PlanesStep::Split(index));
                merge.push(PlanesStep::Merge(index));
            }
        }
        for &index in framed {
            let (ran, rendered) = (color(run, index)?, color(pass, index)?);
            pairs.push((&ran.target, &rendered.target, pass.extent));
        }
        if pairs.is_empty() && split.is_empty() {
            return Ok(None);
        }
        Ok(Some(RunSteps {
            into: StandInCopies::plan(&pairs, true)?,
            back: StandInCopies::plan(&pairs, false)?,
            planes: PlanesCommands::plan(device, run, &[split, merge])?,
        }))
    }

    /// The steps' uses.
    fn uses(&self) -> impl Iterator<Item = &Use<ResourceKey>> {
        let copies = [&self.into, &self.back].into_iter().flatten();
        copies
            .flat_map(|c| c.uses().iter())
            .chain(self.planes.iter().flat_map(|p| p.uses.iter()))
    }

    /// Record the steps before the run.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it; the
    /// images and the steps' objects are live until it completes, and every image rests in
    /// `GENERAL`.
    unsafe fn before(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract.
        unsafe {
            if let Some(into) = &self.into {
                into.record_in(raw, command_buffer);
            }
            if let Some(planes) = &self.planes {
                planes.record(raw, command_buffer, 0);
            }
        }
    }

    /// Record the steps after the run.
    ///
    /// # Safety
    ///
    /// As for [`RunSteps::before`].
    unsafe fn after(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract.
        unsafe {
            if let Some(planes) = &self.planes {
                planes.record(raw, command_buffer, 1);
            }
            if let Some(back) = &self.back {
                back.record_in(raw, command_buffer);
            }
        }
    }
}

/// What the draws of a pass share across its instances and its executions, each made at the
/// first draw that reaches it: the buffer of every tile's threadgroup memory
/// (`render/tile_memory.rs`), the fragment imageblock storage (`fragment_imageblock.rs`), and the
/// shader-side stencil's shadow (`shadow_stencil.rs`).
#[derive(Clone, Default)]
pub(super) struct PassStorage {
    pub(super) tile_memory: Option<Buffer>,
    pub(super) imageblock: Option<PassImageblock>,
    pub(super) shadow: Option<ShadowStencil>,
}

/// One run of an execution's commands: the secondary command buffer its draws recorded into,
/// and the shape and views of the render pass instance it runs in where a command's draw needed
/// another than its instance's (attachments it fetches as storage images, views of its own).
struct Run {
    draws: vk::CommandBuffer,
    other: Option<(PassShape, ViewSetup)>,
    /// Its draws test and write the stencil in the shader: the shadow is filled before it and
    /// copied back after it, as around an encoder's instance of such draws.
    shadow: bool,
    /// The steps around it where its draws reach fetched attachments elsewhere than the pass's
    /// instances hold them.
    steps: Option<RunSteps>,
    /// The stand-ins its framebuffer holds in place of the pass's attachments, by color index:
    /// through input attachments, those its draws read and render where a command's draw made
    /// one (`RunSteps` copies each in before it and back after).
    framed: Vec<(usize, Target)>,
    /// What records before it, after the runs before it, in order: the dispatches of its first
    /// draw, an emulated mesh command's behind its barrier whose compute work uses what an
    /// earlier draw's stages before its fragments touched (`RenderCommandEncoder::draw_resident`).
    /// A command's barrier begins every run that has them.
    before: Vec<Before>,
}

impl Run {
    /// Whether the next piece of its instance may record into it: it is of the instance's shape,
    /// and nothing runs around it.
    fn takes_more(&self) -> bool {
        self.other.is_none() && !self.shadow && self.steps.is_none()
    }
}

/// What a command's record asks for: a draw, nothing (no draw, or one of no vertices, indices or
/// patches), or nothing for a fault (`RECORD`): a value Metal has no meaning for, or a barrier.
enum Asked {
    Draw(Draw<u64>),
    Nothing,
    Fault,
}

/// A command's draw, its buffers as `B`: the addresses the record holds, or the resident buffers
/// and offsets they resolve to.
enum Draw<B> {
    /// A draw without indices of a type other than 9 (`Assembly::of`).
    Primitives(Assembly, DrawCall),
    /// A mesh draw's grid, and its object and mesh threadgroups' threads.
    Mesh(MeshGrid, [u32; 3], [u32; 3]),
    Indexed(IndexedDraw, B),
    Patches {
        draw: PatchDraw,
        /// `None`: the patches' own indices.
        patch_indices: Option<B>,
        control_point_indices: PointIndices<B>,
        factors: FactorSource<B>,
        /// The scale the factors are multiplied by (`None`: 1, the factors as they stand).
        scale: Option<f32>,
    },
}

/// The control-point indices a patch draw reads (probes), its buffer as `B`.
#[derive(Clone, Copy)]
enum PointIndices<B> {
    /// None: the patches' own control points (a plain patch draw, or any on a pipeline without
    /// a control-point index type).
    Own,
    /// Every index 0: no buffer (a null one, or a plain patch draw on a pipeline with an index
    /// type).
    Zero,
    /// The indices of the buffer from its offset, 0 past its end.
    Buffer(B),
}

/// A patch draw's tessellation factor buffer, as `B`, and its instance stride.
struct FactorSource<B> {
    buffer: B,
    instance_stride: u32,
}

/// A command's draw once its addresses were resolved: to run, or nothing for an address in no
/// resident resource (a miss) or a buffer offset Metal's validation stops (`RECORD`).
enum Resolved {
    Draw(Draw<(Buffer, u64)>),
    Missed,
    Fault,
}

impl Asked {
    /// What `action` asks for on a command of `icb`.
    fn of(action: &Action, icb: &IndirectCommandBuffer) -> Asked {
        // A draw whose record the buffer's records cannot hold, which plain Metal takes and runs
        // nothing for: nothing here too (`IndirectCommandBuffer::takes_as_nothing`). The kinds
        // Metal ends its process or faults its GPU for never reach a record: their setter refuses
        // them, and a kernel's write of one is dropped.
        if icb.takes_as_nothing(action) {
            return Asked::Nothing;
        }
        match *action {
            Action::Empty
            | Action::DispatchThreadgroups { .. }
            | Action::DispatchThreads { .. } => Asked::Nothing,
            Action::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => {
                // Type 9 the call did not find (`RenderLowering::execute_range`) runs nothing.
                let Some(assembly) = DrawPrimitiveType::from_raw(primitive_type.into())
                    .ok()
                    .and_then(Assembly::of)
                else {
                    return Asked::Fault;
                };
                if vertex_count == 0 || instance_count == 0 {
                    return Asked::Nothing;
                }
                let draw = DrawCall::Primitives {
                    vertex_start,
                    vertex_count,
                    instance_count,
                    base_instance,
                };
                Asked::Draw(Draw::Primitives(assembly, draw))
            }
            Action::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                instance_count,
                base_vertex,
                base_instance,
            } => {
                // Type 9 the call did not find (`RenderLowering::execute_range`) runs nothing.
                let (Ok(primitive_type), Ok(index_type)) = (
                    DrawPrimitiveType::from_raw(primitive_type.into()),
                    IndexType::try_from(index_type),
                ) else {
                    return Asked::Fault;
                };
                if primitive_type == DrawPrimitiveType::Nine {
                    return Asked::Fault;
                }
                // Metal ends the process for no index buffer.
                if index_buffer == 0 {
                    return Asked::Fault;
                }
                if index_count == 0 || instance_count == 0 {
                    return Asked::Nothing;
                }
                let draw = IndexedDraw {
                    primitive_type,
                    index_count,
                    index_type,
                    instance_count,
                    base_vertex,
                    base_instance,
                };
                Asked::Draw(Draw::Indexed(draw, index_buffer))
            }
            Action::DrawPatches {
                control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                instance_count,
                base_instance,
                factor_buffer,
                factor_instance_stride,
                factor_scale,
            } => Asked::patches(
                PatchDraw {
                    control_points,
                    patch_start,
                    patch_count,
                    instance_count,
                    base_instance,
                },
                patch_index_buffer,
                PointIndices::Own,
                FactorSource {
                    buffer: factor_buffer,
                    instance_stride: factor_instance_stride,
                },
                factor_scale,
            ),
            Action::DrawIndexedPatches {
                control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                control_point_index_buffer,
                instance_count,
                base_instance,
                factor_buffer,
                factor_instance_stride,
                factor_scale,
            } => Asked::patches(
                PatchDraw {
                    control_points,
                    patch_start,
                    patch_count,
                    instance_count,
                    base_instance,
                },
                patch_index_buffer,
                match control_point_index_buffer {
                    0 => PointIndices::Zero,
                    address => PointIndices::Buffer(address),
                },
                FactorSource {
                    buffer: factor_buffer,
                    instance_stride: factor_instance_stride,
                },
                factor_scale,
            ),
            Action::DrawMeshThreadgroups {
                threadgroups,
                object_threads,
                mesh_threads,
            } => Asked::Draw(Draw::Mesh(
                MeshGrid::Threadgroups(threadgroups),
                object_threads,
                mesh_threads,
            )),
            Action::DrawMeshThreads {
                threads,
                object_threads,
                mesh_threads,
            } => Asked::Draw(Draw::Mesh(
                MeshGrid::Threads(threads),
                object_threads,
                mesh_threads,
            )),
        }
    }

    /// What a patch draw or an indexed patch draw (`control_point_indices`) asks for, its patch
    /// index buffer's address `patch_index_buffer` (0: none) and its factor scale's bits
    /// `factor_scale`.
    fn patches(
        draw: PatchDraw,
        patch_index_buffer: u64,
        control_point_indices: PointIndices<u64>,
        factors: FactorSource<u64>,
        factor_scale: u32,
    ) -> Asked {
        // Metal ends the process for no tessellation factor buffer.
        if factors.buffer == 0 {
            return Asked::Fault;
        }
        if draw.patch_count == 0 || draw.instance_count == 0 {
            return Asked::Nothing;
        }
        let scale = match patch_scale(factor_scale) {
            Scale::One => None,
            Scale::By(scale) => Some(scale),
            Scale::Culls => return Asked::Nothing,
        };
        Asked::Draw(Draw::Patches {
            draw,
            patch_indices: (patch_index_buffer != 0).then_some(patch_index_buffer),
            control_point_indices,
            factors,
            scale,
        })
    }
}

/// What a patch command's factor scale does (an M4: Metal multiplies the factors by a kernel's
/// scale whatever the pipeline, which `supportIndirectCommandBuffers` keeps from enabling the scale
/// itself; the M3 Max and the M4 alike: it rounds the scale to `half` first).
#[derive(Debug, PartialEq)]
enum Scale {
    /// 1 as a `half`: the factors as they stand (1.0001 and 1.0004 drew what 1 draws).
    One,
    /// Another positive `half`: the factors multiplied by the scale so rounded, as the encoder's
    /// scale multiplies them (the binder rounds it). 2 drew what factors of 2 draw, 0.5 what
    /// factors of 0.5 draw, 1.3 and 2.0005 what factors of 2 draw, a scale that is subnormal as
    /// a `half` (2^-15 to 2^-24, 10^-6) what factors of 1 draw, and one that is infinite as a
    /// `half` (70000, `+∞`) what the maximum factor draws.
    By(f32),
    /// Not positive (zero, negative, or NaN), or 0 as a `half` (2^-30): every patch culls (0, -1,
    /// NaN and 2^-30 drew nothing).
    Culls,
}

/// What the scale of `bits` (a `float`'s) does to a patch command's factors.
fn patch_scale(bits: u32) -> Scale {
    let scale = f32::from_bits(bits);
    if scale.is_nan() || scale <= 0.0 {
        return Scale::Culls;
    }
    match crate::clear_value::f32_to_f16(scale) {
        0 => Scale::Culls,
        0x3c00 => Scale::One,
        _ => Scale::By(scale),
    }
}

impl Draw<u64> {
    /// The draw, each address resolved against `residency`'s resources: an index buffer's and a
    /// patch index buffer's offset rounded down to a multiple of the index size or of 4, as Metal
    /// reads a command's indices where its validation stops the call,
    /// and a factor buffer's at any byte, whose factors the binder reads from the offset rounded
    /// down to a multiple of 2 and each instance's at the stride rounded down to a multiple of 4,
    /// as plain Metal reads them; a patch draw's control-point indices as its pipeline's index size
    /// `point_index_size` (`None` for a pipeline without a control-point index type) has Metal read
    /// them (probes), from the offset rounded down to a multiple of it, as Metal reads a command's
    /// control-point indices, the host's and a kernel's alike.
    fn resolve(self, residency: &Residency, point_index_size: Option<u64>) -> Resolved {
        let resolve = |address: u64, multiple: u64| match residency.resolve(address) {
            None => Err(Resolved::Missed),
            Some((_, offset)) if !offset.is_multiple_of(multiple) => Err(Resolved::Fault),
            Some(found) => Ok(found),
        };
        let rounded = |address: u64, multiple: u64| {
            resolve(address, 1).map(|(b, offset)| (b, offset - offset % multiple))
        };
        let resolved = match self {
            Draw::Primitives(assembly, draw) => Ok(Draw::Primitives(assembly, draw)),
            Draw::Mesh(grid, object, mesh) => Ok(Draw::Mesh(grid, object, mesh)),
            Draw::Indexed(draw, address) => rounded(address, index_size(draw.index_type))
                .map(|index| Draw::Indexed(draw, index)),
            Draw::Patches {
                draw,
                patch_indices,
                control_point_indices,
                factors,
                scale,
            } => {
                let indices = patch_indices.map(|address| rounded(address, 4)).transpose();
                let points = match (control_point_indices, point_index_size) {
                    (_, None) => Ok(PointIndices::Own),
                    (PointIndices::Own | PointIndices::Zero, Some(_)) => Ok(PointIndices::Zero),
                    // Metal reads the indices from the offset rounded down to a multiple of
                    // the index size (32-bit indices at offsets 1, 2, 3, 6 and 7, 16-bit at 1, 3
                    // and 5; its validation does not stop the offset).
                    (PointIndices::Buffer(address), Some(size)) => {
                        rounded(address, size).map(PointIndices::Buffer)
                    }
                };
                let buffer = resolve(factors.buffer, 1);
                let stride = factors.instance_stride;
                // The words the draw reads, from its patch start, inside the buffer.
                let words = u64::from(draw.patch_start) + u64::from(draw.patch_count);
                let past = |found: &Option<(Buffer, u64)>| {
                    found
                        .as_ref()
                        .is_some_and(|(b, offset)| offset + 4 * words > b.length())
                };
                match (indices, points, buffer) {
                    (Err(missed), _, _) | (_, Err(missed), _) | (_, _, Err(missed)) => Err(missed),
                    (Ok(found), _, Ok(_)) if past(&found) => Err(Resolved::Fault),
                    (Ok(patch_indices), Ok(control_point_indices), Ok(buffer)) => {
                        Ok(Draw::Patches {
                            draw,
                            patch_indices,
                            control_point_indices,
                            factors: FactorSource {
                                buffer,
                                instance_stride: stride,
                            },
                            scale,
                        })
                    }
                }
            }
        };
        match resolved {
            Ok(draw) => Resolved::Draw(draw),
            Err(outcome) => outcome,
        }
    }
}

/// The control points a patch draw of `pipeline` takes where its tessellation function states
/// them (`None` elsewhere).
fn pipeline_control_points(pipeline: &RenderPipelineState) -> Option<u32> {
    match pipeline.kind() {
        RenderStateKind::Graphics(state) => crate::tessellation::patches(&state.base().graphics)
            .map(|p| p.control_points)
            .filter(|&points| points != 0),
        RenderStateKind::Tile(_) => None,
    }
}

/// The size of a control-point index of `pipeline`'s `tessellationControlPointIndexType`, `None`
/// for a type of `None` or a pipeline without tessellation.
fn pipeline_point_index_size(pipeline: &RenderPipelineState) -> Option<u64> {
    use metal2vulkan::metal::TessellationControlPointIndexType as T;
    let RenderStateKind::Graphics(state) = pipeline.kind() else {
        return None;
    };
    match crate::tessellation::patches(&state.base().graphics)?.control_point_index_type {
        T::UInt16 => Some(2),
        T::UInt32 => Some(4),
        _ => None,
    }
}

/// Whether `pipeline` reads vertex buffer `index` at the stride it is bound with: a layout of
/// `MTLBufferLayoutStrideDynamic` (`VertexStride::PerBind`) whose step reads an element (a
/// fixed-function binding of a nonzero divisor, or a pulled attribute of a step other than
/// constant).
fn reads_vertex_stride(pipeline: &RenderPipelineState, index: u32) -> bool {
    let RenderStateKind::Graphics(state) = pipeline.kind() else {
        return false;
    };
    let input = &state.base().vertex_input;
    input
        .bindings
        .iter()
        .any(|b| b.buffer_index == index && b.stride == VertexStride::PerBind && b.divisor != 0)
        || input.pulled.iter().any(|a| {
            a.buffer_index == index
                && a.stride == VertexStride::PerBind
                && a.step_function != VertexStepFunction::Constant
        })
}

/// The object threadgroup memory lengths `record`'s mesh draw runs with: each it set, by index
/// (`IndirectRenderCommand::set_object_threadgroup_memory_length`); one it set none of is 0.
fn object_threadgroup_lengths(record: &CommandRecord) -> BTreeMap<u32, u32> {
    (0u32..)
        .zip(record.threadgroup_memory_lengths)
        .filter(|&(_, length)| length != 0)
        .collect()
}

/// What an execution's settle at commit made for its host step's draw of one primitive type: the
/// variant, and the draw's `VkPipeline` in the execution's instance where the settle could make
/// it, made or being made on the device's pool (`RenderVariant::settle_graphics_pipeline`). The
/// command buffer holds them until it completes.
pub(crate) type SettledDraw = (Arc<RenderVariant>, Option<DrawPipeline>);

impl RenderLowering {
    /// Make, into `made`, what a command's draw of `pipeline` under the encoder's state (an
    /// execution's at its call, `RenderExecution::settle`) compiles at the host step: for each
    /// primitive type its input primitive topology class holds (every type for the unspecified
    /// class), the variant in the large-buffer form and the `VkPipeline` of the draw in the
    /// execution's instance, an indexed draw's alike ([`settle_draw_of`](Self::settle_draw_of)).
    /// A tessellation or mesh pipeline's draws, and a state the pass refuses, are left to the
    /// step, which meets them as they are then. Announces nothing: it draws nothing.
    fn settle_draws(&mut self, pipeline: &RenderPipelineState, made: &mut Vec<SettledDraw>) {
        let RenderStateKind::Graphics(state) = pipeline.kind() else {
            return;
        };
        let vertex = state.descriptor().vertex();
        if crate::tessellation::patches(&state.base().graphics).is_some()
            || vertex.is_none()
            || self.shape_inputs().check_pipeline(CALL, pipeline).is_err()
        {
            return;
        }
        let types: &[PrimitiveType] = match vertex.map(|d| d.input_primitive_topology) {
            Some(PrimitiveTopologyClass::Point) => &[PrimitiveType::Point],
            Some(PrimitiveTopologyClass::Line) => &[PrimitiveType::Line, PrimitiveType::LineStrip],
            Some(PrimitiveTopologyClass::Triangle) => {
                &[PrimitiveType::Triangle, PrimitiveType::TriangleStrip]
            }
            _ => &[
                PrimitiveType::Point,
                PrimitiveType::Line,
                PrimitiveType::LineStrip,
                PrimitiveType::Triangle,
                PrimitiveType::TriangleStrip,
            ],
        };
        // The encoder's pipeline state, which its bind state reads, as a command's draw sets it.
        self.pipeline = Some(pipeline.clone());
        for &primitive_type in types {
            if let Ok(settled) = self.settle_draw_of(state, primitive_type) {
                made.push(settled);
            }
        }
    }

    /// The variant a command's draw of `state` as `primitive_type` runs on at the host step, and
    /// its `VkPipeline` in the execution's instance, each made where it was not: the draw's
    /// settings, bind state, fixed state and key as `draw` settles them
    /// ([`draw_settings`](Self::draw_settings), [`settle_draw`](Self::settle_draw),
    /// `ShapeInputs::draw_fixed`), with the encoder's tables, which are the command's own
    /// but for its buffers. The step makes another only where a command's own buffers ask for
    /// another variant (a vertex buffer at an offset or stride fixed-function input cannot
    /// fetch, a buffer whose length a slot's form depends on outside the large-buffer form's
    /// slots); here none is made where the draw records into a run of another shape than the
    /// instance's (it fetches attachments, tests the stencil in the shader, renders views of its
    /// own, or places its colors elsewhere), or a command's own vertex strides key the pipeline
    /// (a layout of per-bind strides that are not dynamic state).
    fn settle_draw_of(
        &self,
        state: &DrawingState,
        primitive_type: PrimitiveType,
    ) -> Result<SettledDraw, Error> {
        let assembly = Assembly::Primitives(primitive_type);
        let settings = self.draw_settings(CALL, state, assembly, false)?;
        let mut argument_samplers = ArgumentSamplers::taken(self.core.device());
        let function_tables = state.function_tables();
        let live = Arc::clone(&self.live);
        let pass_start = PassStartTables::default();
        let sampler_maps = self.sampler_maps(state, &pass_start)?;
        let settled = self.settle_draw(
            &mut lock(&live),
            (
                state,
                &settings,
                None,
                PrivateForm::default(),
                ProvokingForms::default(),
            ),
            (&pass_start, &sampler_maps),
            (&mut argument_samplers, &mut None),
            (&function_tables, &mut None),
        )?;
        let Settled { variant, plan, .. } = settled;
        let device = self.core.device();
        let stages = &variant.pipeline().stages;
        device.degrade_argument_pixel_samplers(stages, argument_samplers.take(device));
        // The neutrals the plan binds, made here so the step need not (`neutral.rs`). One not made
        // here (the device out of memory) leaves the variant settled: the step makes it where it
        // binds it.
        let _ = crate::binding::make_neutrals(self.core.device(), &plan);
        let graphics = &variant.pipeline().graphics;
        let storage = self.storage_attachments(graphics)?;
        let fetches = match self.layout.fetch_inputs {
            true => !graphics.input_attachments.is_empty(),
            false => !storage.is_empty(),
        };
        let dynamic_strides = graphics
            .dynamic_states
            .contains(&metal2vulkan::vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE);
        let own_strides = !dynamic_strides && plan.vertex_buffers.iter().any(|b| b.per_bind_stride);
        if fetches
            || graphics.shadow_stencil.is_some()
            || graphics.shader_tests.is_some()
            || plan.views != ViewSetup::default()
            || *self.current.shape.locations != *settings.routing.locations
            || own_strides
        {
            return Ok((variant, None));
        }
        let (_, stencil) = self.layout.aspects();
        let failing = match &settings.failing_state {
            None => None,
            Some(failing) => {
                let body_writes =
                    storage.iter().any(|s| s.writes) || graphics.fragment_imageblock.is_some();
                let writes = (body_writes, graphics.fragment_writes_memory);
                Some(failing.failing_draw(graphics.early_fragment_tests, stencil, writes)?)
            }
        };
        let mut fixed =
            self.shape_inputs()
                .draw_fixed(graphics, failing.as_ref(), (false, false))?;
        fixed.provoking_last = settings.provoking_last;
        let clamp = (
            settings.native,
            settings.written_clamp,
            settings.compensation,
            settings.bias,
        );
        self.shape_inputs()
            .clamp_draw_depth(&mut fixed, graphics, clamp)?;
        // The class's list topology where the pipeline makes the topology dynamic.
        let (topology, primitive_restart) = crate::pipeline::keyed_assembly(
            settings.topology,
            settings.primitive_restart,
            &graphics.dynamic_states,
        );
        let key = GraphicsKey {
            specialization: plan.specialization.clone(),
            topology,
            primitive_restart,
            pass: self.current.shape.clone(),
            strides: Vec::new(),
            fixed: FixedState::without_dynamic(fixed, &graphics.dynamic_states),
            sample_positions: self.layout.sample_positions.clone(),
        };
        // The formats the encoder's tables bind, which are the command's own.
        let stages = [
            (metal2vulkan::pipeline::Stage::Vertex, &self.vertex),
            (metal2vulkan::pipeline::Stage::Fragment, &self.fragment),
        ];
        let tables = crate::binding::Tables::new(&stages);
        let formats = |slot| tables.bound_format(slot);
        self.core
            .device()
            .degrade_bound_formats(&variant.pipeline().stages, Some(&formats));
        self.core.device().degrade_bound_values(
            &variant.pipeline().stages,
            &key.specialization,
            &|k| plan.specialized_value(k),
            Some(&formats),
        );
        let device = self.core.device();
        let compatible = &self.current.compatible;
        let drawn =
            variant.settle_graphics_pipeline(device, &key, compatible, (state, &formats))?;
        if let DrawPipeline::Made(raw) = &drawn {
            variant.first_use(raw, &key.specialization, &formats);
        }
        Ok((variant, Some(drawn)))
    }
}

impl RenderExecution {
    /// Whether the commands may reach the pass's tile threadgroup memory: the encoder set an
    /// argument of it at the call (`render/tile_memory.rs`).
    pub(super) fn reaches_tile_memory(&self) -> bool {
        self.snapshot.tile_memory.set()
    }

    /// Whether the execution may split its instance into runs: its commands may reach the tile
    /// threadgroup memory, or a record of the buffer's may hold a barrier, the CPU's or the GPU's.
    pub(super) fn may_split(&self) -> bool {
        self.reaches_tile_memory() || self.icb.may_hold_barriers()
    }

    /// Record the commands into secondary command buffers of `command_buffer` begun against the
    /// execution's instance, through an encoder of `core` (`replaying`, so a draw that needs
    /// another instance begins a run of its shape or is refused, and one ordered after another
    /// that reached the tile threadgroup memory begins the next run, as does a command whose
    /// barrier is set), over the pass's `storage` (each
    /// part `None` where no draw made it yet), in a pass whose instances `rendered` lays out,
    /// with `residency`, the command buffer's (`residency.rs`): from the CPU's records as they
    /// stand now, or, for the host step (`step`), from the records as the GPU and the CPU left
    /// them; or in its GPU form (`gpu`), whose records the host never reads (`gpu.rs`). Commands
    /// that run as several back the pass's memoryless attachments (`Rendered::back_memoryless`).
    fn record(
        &self,
        core: EncoderCore,
        (command_buffer, residency): (&CommandBuffer, &Residency),
        (step, gpu): (bool, Option<&GpuForm>),
        found: &mut Found,
        (storage, rendered): (PassStorage, &mut Rendered),
    ) -> Result<Recorded, Error> {
        let snapshot = &self.snapshot;
        let device = command_buffer.device().clone();
        if let Some(form) = gpu {
            let encoder =
                self.record_gpu(core, (command_buffer, residency), form, (storage, rendered))?;
            return self.finish(encoder, found, rendered, &device);
        }
        let (commands, faults) = self.commands(&device, step)?;
        found.faults = ReplayFaults(found.faults.0 | faults.0);
        // What the commands run, for the command buffer's handlers (`executed_commands.rs`).
        let read = match step {
            true => RecordsRead::HostStep,
            false => RecordsRead::Commit,
        };
        command_buffer.report_executed(&self.icb, read, residency, || {
            let first = self
                .range
                .commands(self.icb.size())
                .map_or(0, |(range, _)| u64::from(range.start));
            let commands = commands
                .iter()
                .enumerate()
                .filter_map(|(at, command)| {
                    let (record, own) = command.as_ref()?;
                    let Asked::Draw(_) = Asked::of(&record.action, &self.icb) else {
                        return None;
                    };
                    let pipeline = self.pipeline.as_ref().or(own.as_ref())?;
                    Some(ExecutedCommand {
                        index: first + at as u64,
                        can_encode_indirect_commands: pipeline.can_encode_indirect_commands(),
                    })
                })
                .collect();
            (commands, false)
        });
        let draws = command_buffer.begin_draws(&snapshot.compatible)?;
        let raw = device.shared().vk();
        // The query the commands' draws count in, begun here and ended after the last of them.
        let visibility = Visibility::borrowing(
            snapshot.layout.visibility.clone(),
            self.lent.clone(),
            raw,
            draws,
        );
        // The host step's draws take the large-buffer form, whose variants `commit` settled
        // (`RenderExecution::settle`).
        let forms = match step {
            true => BufferForms::Large,
            false => BufferForms::Bound,
        };
        // The call's layout, its memoryless attachments in the backings an earlier execution's
        // runs gave the pass.
        let layout = rendered.backing(&snapshot.layout)?;
        let mut encoder = self.replay_encoder(
            core,
            (draws, visibility),
            (storage, layout, residency),
            forms,
        );
        let inherited = self.icb.descriptor().inherit_buffers;
        for (record, own) in commands.into_iter().flatten() {
            // A command's barrier: its draw records into the execution's next run, which its
            // pass runs in a render pass instance of its own after a barrier from everything
            // before, so it sees the earlier draws' writes as plain Metal's does; but where the
            // run so far is one the execution began with nothing in it. The runs take the
            // instance's run passes where the encoder made them (a record may have held a barrier
            // before it ended: the CPU set one, or the GPU may have written the records), else
            // those the commit made ahead or the recording makes (`record_forms`), for a barrier
            // the CPU set later. A barrier neither may have set is a record the runtime does not
            // write, a fault, as before.
            if record.barrier {
                if !self.icb.may_hold_barriers() {
                    found.faults = ReplayFaults(found.faults.0 | ReplayFaults::RECORD.0);
                    continue;
                }
                encoder.run_barrier |= encoder.current.recorded || encoder.earlier.is_empty();
            }
            encoder.command_barrier = record.barrier;
            let asked = Asked::of(&record.action, &self.icb);
            let draw = match asked {
                Asked::Draw(draw) => draw,
                Asked::Nothing => continue,
                Asked::Fault => {
                    found.faults = ReplayFaults(found.faults.0 | ReplayFaults::RECORD.0);
                    continue;
                }
            };
            let Some(pipeline) = self.pipeline.clone().or(own) else {
                continue;
            };
            let draw = match draw.resolve(residency, pipeline_point_index_size(&pipeline)) {
                Resolved::Draw(draw) => draw,
                Resolved::Missed => {
                    found.misses = found.misses.wrapping_add(1);
                    continue;
                }
                Resolved::Fault => {
                    found.faults = ReplayFaults(found.faults.0 | ReplayFaults::RECORD.0);
                    continue;
                }
            };
            // A patch draw that scales its factors runs on its state's factor-scaling twin, the
            // encoder's scale the command's.
            let scale = match &draw {
                Draw::Patches { scale, .. } => *scale,
                Draw::Primitives(..) | Draw::Indexed(..) | Draw::Mesh(..) => None,
            };
            let pipeline = match scale {
                Some(_) => pipeline.factor_scaling()?.unwrap_or(pipeline),
                None => pipeline,
            };
            encoder.factors.scale = scale;
            encoder.shape_inputs().check_pipeline(CALL, &pipeline)?;
            // A mesh draw's object function takes the command's threadgroup memory lengths, 0
            // where it set none, never the encoder's (the M3 Max); a patch draw's take none.
            encoder.object_threadgroup_lengths = match &draw {
                Draw::Mesh(..) => object_threadgroup_lengths(&record),
                Draw::Primitives(..) | Draw::Indexed(..) | Draw::Patches { .. } => BTreeMap::new(),
            };
            let control_points = pipeline_control_points(&pipeline);
            encoder.pipeline = Some(pipeline);
            if !inherited {
                let descriptor = self.icb.descriptor();
                let misses = &mut found.misses;
                let mut table =
                    |stage, count| self.buffers(&record, (stage, count), residency, misses);
                encoder.vertex =
                    table(BufferStage::Vertex, descriptor.max_vertex_buffer_bind_count);
                encoder.fragment = table(
                    BufferStage::Fragment,
                    descriptor.max_fragment_buffer_bind_count,
                );
                encoder.object =
                    table(BufferStage::Object, descriptor.max_object_buffer_bind_count);
                encoder.mesh = table(BufferStage::Mesh, descriptor.max_mesh_buffer_bind_count);
                // A vertex buffer set without a stride that a dynamic layout reads, on a buffer
                // that supports dynamic attribute strides: Metal takes the command and its fetch
                // reads at stride 2^32 - 1, which the 32-bit fetch cannot place
                // (`unstrided_buffer`). Without that support Metal reads stride 0, which the plan
                // path gives exactly.
                let table = &encoder.vertex;
                let unstrided = encoder.pipeline.as_ref().and_then(|pipeline| {
                    let bound = table
                        .buffers
                        .keys()
                        .copied()
                        .filter(|index| !table.attribute_strides.contains_key(index));
                    unstrided_buffer(
                        descriptor.support_dynamic_attribute_stride,
                        bound,
                        |index| reads_vertex_stride(pipeline, index),
                    )
                });
                if let Some(refusal) = unstrided {
                    return Err(refusal);
                }
            }
            match draw {
                Draw::Primitives(assembly, draw) => {
                    encoder.draw(CALL, assembly, draw, DrawBuffers::default(), false)?
                }
                Draw::Indexed(draw, (buffer, offset)) => {
                    encoder.draw_indexed(CALL, draw, &buffer, offset)?;
                }
                Draw::Mesh(grid, object, mesh) => encoder.draw_mesh(CALL, grid, object, mesh)?,
                Draw::Patches {
                    mut draw,
                    patch_indices,
                    control_point_indices,
                    factors,
                    scale: _,
                } => {
                    // Metal draws a command's patches with its pipeline's control points, whatever
                    // the record holds (measured).
                    draw.control_points = control_points.unwrap_or(draw.control_points);
                    let (buffer, offset) = factors.buffer;
                    encoder.factors.buffer = Some(FactorBuffer {
                        buffer,
                        offset,
                        instance_stride: u64::from(factors.instance_stride),
                    });
                    let patch_indices = patch_indices.as_ref().map(|(b, offset)| (b, *offset));
                    // Every index 0: the encoder's buffer of zero indices, past whose end an
                    // index reads 0 too.
                    let zeros = match control_point_indices {
                        PointIndices::Zero => Some(encoder.zero_indices()?),
                        PointIndices::Own | PointIndices::Buffer(_) => None,
                    };
                    let points = match &control_point_indices {
                        PointIndices::Own => None,
                        PointIndices::Zero => zeros.as_ref().map(|zero| (zero, 0)),
                        PointIndices::Buffer((b, offset)) => Some((b, *offset)),
                    };
                    encoder.draw_patch_range(CALL, draw, patch_indices, points)?;
                }
            }
        }
        self.finish(encoder, found, rendered, &device)
    }

    /// The recording `encoder` made of the commands, its runs ended (`record`'s): what its draws
    /// noted into `found`, the pass's memoryless attachments backed into `rendered` where its
    /// commands ran as several, and each run's steps.
    fn finish(
        &self,
        mut encoder: RenderLowering,
        found: &mut Found,
        rendered: &mut Rendered,
        device: &Device,
    ) -> Result<Recorded, Error> {
        let snapshot = &self.snapshot;
        let raw = device.shared().vk();
        // The lent query, or the last spare the replay took, ends in the last run; every spare
        // left begins and ends there, empty.
        encoder.visibility.end(raw, encoder.current.draws);
        encoder.visibility.close_spares(raw, encoder.current.draws);
        let RenderLowering {
            core,
            earlier,
            current,
            tile_memory,
            tile_memory_splits,
            barrier_splits,
            imageblock,
            shadow,
            mut layout,
            ..
        } = encoder;
        found.tile_memory_splits += tile_memory_splits;
        found.barrier_splits += barrier_splits;
        // Commands that ran as several keep the pass's memoryless attachments across their runs,
        // in the backings of the instances that run them.
        if !earlier.is_empty() {
            rendered.back_memoryless(core.command_buffer(), &mut layout)?;
        }
        let pass = &rendered.layout;
        // A run that leaves fetched attachments out of its subpass takes the steps around it
        // that bring them where its draws read them (`RunSteps`), over the layout the commands
        // left: each stand-in and sample planes a draw made stays the layout's for the later
        // draws, so every run's are there.
        // A run of another shape resolves as the pass ends, as each of the encoder's instances
        // does (`render/late_resolve.rs`): its pass takes the resolves of the pass's layout as it
        // ends, which the call's layout may lack (the exact resolve from the sample planes
        // decided as the encoder ends, a resolve target set after the call). Its secondary was
        // begun against a pass compatible with it: a pass of one subpass ignores its resolves.
        let ending = PassShape::of(pass)?;
        let mut runs = Vec::with_capacity(earlier.len() + 1);
        for run in earlier.into_iter().chain([current]) {
            let other =
                (*run.shape != snapshot.shape || run.views != ViewSetup::default()).then(|| {
                    let mut shape = run.shape.clone();
                    super::late_resolve::settle(std::iter::once(&mut shape), &ending);
                    (shape, run.views)
                });
            // Through input attachments, a run whose subpass reads an attachment as an input
            // attachment that the pass's instances render into another image (one that cannot be
            // read so, for which a command's draw made a stand-in) renders the stand-in in its
            // place, copied in before the run and back after it.
            let framed = match &other {
                Some((shape, _)) if layout.fetch_inputs => framed(&layout, pass, shape)?,
                Some(_) | None => Vec::new(),
            };
            let steps = match &other {
                Some((shape, _)) if !shape.storage.is_empty() || !framed.is_empty() => {
                    RunSteps::plan(device, &layout, pass, (&shape.storage, &framed))?
                }
                Some(_) | None => None,
            };
            let framed = framed
                .iter()
                .map(|&index| {
                    let color = layout.colors.get(index as usize).and_then(Option::as_ref);
                    let color = color.ok_or_else(|| internal("a framed color the run lacks"))?;
                    Ok((index as usize, color.target.clone()))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            runs.push(Run {
                draws: run.draws,
                other,
                shadow: run.shadow,
                steps,
                framed,
                before: run.before,
            });
        }
        for run in &runs {
            // SAFETY: only this recording records the secondary command buffers.
            unsafe { core.command_buffer().end_raw(run.draws) }.map_err(vulkan)?;
        }
        Ok(Recorded {
            core,
            runs,
            storage: PassStorage {
                tile_memory: tile_memory.into_buffer(),
                imageblock,
                shadow,
            },
            layout: layout.into_inner(),
        })
    }

    /// An encoder of `core` that replays the commands (`replaying`) into `draws`, a secondary
    /// begun against the execution's instance, with `visibility`, over the pass's `storage`, in
    /// `layout` (the call's), with `residency`, the command buffer's, its draws' variants in
    /// `forms`: the encoder's state as the call found it. `RenderExecution::record`'s; and
    /// `RenderExecution::settle`'s, which records nothing (`draws` null).
    fn replay_encoder(
        &self,
        core: EncoderCore,
        (draws, visibility): (vk::CommandBuffer, Visibility),
        (storage, layout, residency): (PassStorage, PassLayout, &Residency),
        forms: BufferForms,
    ) -> RenderLowering {
        let snapshot = &self.snapshot;
        // The draws use only the shadowed textures the call gathered.
        let mut shadows = InstanceShadows::default();
        shadows.add(&self.shadows, &[]);
        RenderLowering {
            core,
            earlier: Vec::new(),
            current: Instance {
                draws,
                shape: Versioned::new(snapshot.shape.clone()),
                compatible: Arc::clone(&snapshot.compatible),
                recorded: false,
                shadow: false,
                shader_tests: false,
                // The encoder's draws before the call may have reached the pass's fragment
                // imageblock storage: a command's draw that reaches it records after a barrier.
                imageblock: true,
                // So may they have written the instance's interlocked colors.
                interlocked_writes: InterlockedStorage::ALL,
                // And read as their indirect arguments what the commands' draws may write.
                indirect_reads: snapshot.indirect_reads.clone(),
                // And wrote before their fragments what the commands' fragments may touch.
                geometry_writes: snapshot.geometry_writes.clone(),
                tile_memory: snapshot.tile_memory_reached,
                // Its commands read an attachment in place (`texture/attached_reads.rs`).
                barriers: false,
                views: ViewSetup::default(),
                pieces: Vec::new(),
                shadows,
                before: Vec::new(),
                reformats: Vec::new(),
                positioned: None,
                rerecord: None,
                bound: Bound::default(),
                drawn: crate::clear_reach::DrawnArea::default(),
                inline: false,
                continues: false,
                kept: None,
                planned: false,
                feedback: Feedback::NONE,
                storage_colors: 0,
            },
            layout: Versioned::new(layout),
            passes: vec![(snapshot.shape.clone(), Arc::clone(&snapshot.compatible))],
            render_target_array_length: snapshot.render_target_array_length,
            state: Versioned::new(snapshot.state.clone()),
            pipeline: None,
            routed: None,
            vertex: snapshot.vertex.clone(),
            fragment: snapshot.fragment.clone(),
            object: snapshot.object.clone(),
            mesh: snapshot.mesh.clone(),
            // Each command's mesh draw takes its own (`object_threadgroup_lengths`).
            object_threadgroup_lengths: BTreeMap::new(),
            visibility,
            tile: ArgumentTable::default(),
            dispatches: Vec::new(),
            dispatched: false,
            scratch: Vec::new(),
            // The pass's shader-side stencil's shadow, and its fragment imageblock storage.
            shadow: storage.shadow,
            // A replayed draw that returns its stencil announces it as the pass's runs end.
            returned_stencil: false,
            // A replayed draw that tests the depth in the shader is refused (not built).
            depth_stand_in: None,
            imageblock: storage.imageblock,
            tile_imageblock: snapshot.tile_imageblock,
            factors: snapshot.factors.clone(),
            rate_map: snapshot.rate_map.clone(),
            amplification: Versioned::new(snapshot.amplification.clone()),
            barrier: false,
            barrier_splits: 0,
            driver_restarts: false,
            tile_memory_splits: 0,
            // The pass's buffer of every tile's memory, with the arguments the encoder set at the
            // call (`render/tile_memory.rs`).
            tile_memory: snapshot
                .tile_memory
                .arguments()
                .with_buffer(storage.tile_memory),
            residency: residency.clone(),
            writers: Writers::default(),
            executions: Vec::new(),
            pending: Vec::new(),
            draw_calls: 0,
            tile_dispatch_calls: 0,
            replaying: true,
            executing_encoder: Some(self.encoder),
            rerouting: false,
            forms,
            replayed: None,
            drawn_before: snapshot.drawn.clone(),
            run_barrier: false,
            command_barrier: false,
            unordered: UnorderedWrites::new(),
            zero_indices: None,
            unordered_dispatches: UnorderedWrites::new(),
            barriered: false,
            barrier_inside: false,
            reaches_unnamed: true,
            reload_at: std::collections::BTreeSet::new(),
            called: true,
            depth_writes: false,
            drew_depth_writes: false,
            drew: true,
            drawn: crate::clear_reach::DrawnArea::unbounded(),
            coverage: crate::linear_attachment::Coverage::unbounded(),
            // A pass that executes render commands keeps its attachments over buffers in place
            // (`kept_bytes.rs`), whatever its draws read.
            inputs_read: BTreeSet::new(),
            kept_attachments: None,
            // The executing encoder announces for the execution (`defer_execution`).
            extended_written: BTreeSet::new(),
            // The commands' draws hold no shadow: the execution ended the pass's.
            extended: super::ExtendedShadows::none(),
            pass_start: snapshot.pass_start.notes(),
            // An execution's draws restart at the all-ones index whatever the encoder's call.
            restart: None,
            live: Arc::default(),
            log: None,
            quiet_bound_values: None,
            bind_memo: crate::binding::BindMemo::default(),
            shape_memo: Default::default(),
            last_retained: Default::default(),
            heaps_noted: None,
            inline: super::inline_first::Inline::Never,
            held_bound: Default::default(),
            // No pass of several color groups executes render commands
            // (`render/color_groups.rs`).
            grouped: None,
        }
    }

    /// Settle at commit, through an encoder of `core` that records nothing, what the host step's
    /// draws of the commands compile ([`RenderCommandEncoder::settle_draws`]) for each state a
    /// record may name, into `made`: the encoder's where the buffer inherits it, else each of
    /// `states`, those made with `supportIndirectCommandBuffers` alive at the commit; over the
    /// pass's `storage`, with `residency`, the command buffer's. Nothing where the buffer holds
    /// no draw of primitives. Returns the core.
    fn settle(
        &self,
        core: EncoderCore,
        (storage, residency): (PassStorage, &Residency),
        states: &[RenderPipelineState],
        made: &mut Vec<SettledDraw>,
    ) -> EncoderCore {
        let draws = IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED);
        let named = match &self.pipeline {
            Some(pipeline) => std::slice::from_ref(pipeline),
            None => states,
        };
        if !self.icb.descriptor().command_types.intersects(draws) || named.is_empty() {
            return core;
        }
        let snapshot = &self.snapshot;
        let visibility =
            Visibility::settling(snapshot.layout.visibility.clone(), self.lent.as_ref());
        let mut encoder = self.replay_encoder(
            core,
            (vk::CommandBuffer::null(), visibility),
            (storage, snapshot.layout.clone(), residency),
            BufferForms::Large,
        );
        for pipeline in named {
            encoder.settle_draws(pipeline, made);
        }
        let RenderLowering { core, .. } = encoder;
        core
    }

    /// What its check, if any, may see among commands whose records may hold `written`
    /// (`IndirectCommandCheckNeeds::sees`): draws of points, and commands under states that may
    /// encode indirect commands (the encoder's state where the buffer inherits it, else those the
    /// records may name, `IndirectCommandBuffer::may_name_encoding_states`), each where it looks
    /// for them; neither without a check.
    fn check_sees(&self, written: CommandWrites) -> (bool, bool) {
        if self.check.is_none() {
            return (false, false);
        }
        let encodes = || match &self.pipeline {
            Some(pipeline) => pipeline.can_encode_indirect_commands(),
            None => self.icb.may_name_encoding_states(),
        };
        self.needs.sees(written, encodes)
    }

    /// Whether `commit` recorded it from the CPU's records and the CPU wrote the buffer's records
    /// since ([`read_at`](Self::read_at)): the pass's host step records it again, from the
    /// records as they stand then, before its batch is submitted.
    fn rewritten(&self) -> bool {
        let read_at = self.read_at.load(Ordering::Acquire);
        read_at != indirect_execute::NOT_READ && self.icb.cpu_writes() != read_at
    }

    /// Whether its check may see something among its commands, as its buffer's records may hold
    /// them now ([`check_sees`](Self::check_sees)).
    fn check_sees_now(&self) -> bool {
        let (points, encodes) = self.check_sees(self.icb.command_writes());
        points || encodes
    }

    /// The commands of the range, in order, `None` where a record runs nothing for a fault, and
    /// the faults: the CPU's records, or for the host step the records as the GPU and the CPU
    /// left them, each naming its state among the live ones made with
    /// `supportIndirectCommandBuffers`.
    fn commands(
        &self,
        device: &Device,
        step: bool,
    ) -> Result<(Vec<Option<Command>>, ReplayFaults), Error> {
        let (range, faults) = self.range.commands(self.icb.size())?;
        if !step {
            if self.range.indirect() {
                return Err(internal(
                    "a range read from a buffer recorded before its host step",
                ));
            }
            // The execution announces the commands it runs otherwise than Metal's may, as it
            // records them.
            let start = range.start;
            // Counted before the records are read: a write meanwhile records it again.
            self.read_at.store(self.icb.cpu_writes(), Ordering::Release);
            let (commands, released) = self.icb.render_records(range, Reader::Commit)?;
            self.icb.degrade_released(&released);
            // The caller's check sees each command `commit` will record, before it records any,
            // where it may see something.
            if let Some(check) = self.check.as_ref().filter(|_| self.check_sees_now()) {
                let read = commands.iter().map(|(record, _)| record);
                self.icb.check_read(check, start, read);
            }
            return Ok((
                commands.into_iter().map(Some).collect(),
                ReplayFaults::default(),
            ));
        }
        let written = self.icb.written_records(range.clone())?;
        // The caller's check sees each command the step will record, before it records any,
        // where it may see something.
        if let Some(check) = self.check.as_ref().filter(|_| self.check_sees_now()) {
            self.icb
                .check_recorded(check, range.start, &written.records);
        }
        let states = RenderPipelineState::indirect_states(device);
        // A record naming the ID of the state the CPU set runs its program, else the state of its
        // ID among `states` (`IndirectCommandBuffer::step_render_states`); the commands it runs
        // otherwise than Metal's may are announced, as the CPU's records' are.
        let (named, released) = self
            .icb
            .step_render_states(range.start, &written.records, &states);
        self.icb.degrade_released(&released);
        let commands = written
            .records
            .into_iter()
            .zip(named)
            .map(|(record, state)| record.map(|record| (record, state)))
            .collect();
        Ok((commands, ReplayFaults(written.faults.0 | faults.0)))
    }

    /// The command's own buffers of `stage`, below the descriptor's `count` of it, resolved
    /// against `residency`, the command buffer's, each miss counted in `misses`.
    fn buffers(
        &self,
        record: &CommandRecord,
        (stage, count): (BufferStage, u32),
        residency: &Residency,
        misses: &mut u32,
    ) -> ArgumentTable {
        let mut table = ArgumentTable::default();
        for (index, binding) in (0..count).zip(record.bindings.buffers(stage)) {
            let Some(binding) = binding else {
                continue;
            };
            if !binding.names_buffer() {
                indirect_execute::degrade_nil_offset(self.icb.device(), binding, index);
                continue;
            }
            let Some(bound) = residency.resolve(binding.address) else {
                *misses = misses.wrapping_add(1);
                continue;
            };
            table.buffers.insert(index, bound);
            if let Some(stride) = binding.attribute_stride {
                table
                    .attribute_strides
                    .insert(index, u32::try_from(stride).unwrap_or(u32::MAX));
            }
        }
        table
    }
}

/// A pass that executes render commands, left to record as a segment of its encoder (the module
/// docs): each instance's begin, the secondaries its subpass runs, and its end.
pub(crate) struct DeferredPass {
    instances: Vec<PassInstance>,
    /// The index among the pass's instances of the first of `instances`: the count of those the
    /// encoder recorded inline (`held`), else 0. The steps name instances by that index.
    first: usize,
    /// The command buffer the encoder recorded the pass's leading instances inline in, and the
    /// steps after the last of them but its tile dispatches (which the segment records, once their
    /// pipelines are made, `render/pending_tile.rs`), ended and handed over
    /// (`render/inline_first.rs`, `render/inline_splits.rs`, `EncoderCore::hand_over_pass`): the
    /// segment places it after what it records before the pass (the executions' copies and
    /// dispatches before it, a fragment imageblock's clear) and before the later instances, where
    /// the first instance runs in a segment that records it (`EncoderCore::cut_before`).
    held: Option<vk::CommandBuffer>,
    /// The steps around its instances, as the encoder's own pass records them
    /// (`render/pass_steps.rs`): the shadowed textures' copies, the sample planes', the
    /// shader-side stencil's copies, the draws' steps before their instances, the tile
    /// dispatches, and the resolving instances after the last. They held the shadowed textures'
    /// copies alone before, and the rest refused the pass.
    steps: PassSteps,
    /// The attachments' and the steps' uses.
    uses: Vec<Use<ResourceKey>>,
    executions: Vec<RenderExecution>,
    /// Its draws whose pipelines were being made as they were encoded, each recorded once made
    /// (`render/pending.rs`).
    pending: Vec<PendingDraw>,
    /// Its encoder's place among the command buffer's ended encoders: whether a rebind since the
    /// commit moved a buffer one of its commands named, which its recording must not then record
    /// ([`record_forms`](Self::record_forms), as `pending_dispatch.rs` has it).
    encoder: usize,
    /// The pass's storage where the encoder's draws or tile dispatches made it: its buffer of
    /// every tile's threadgroup memory (`render/tile_memory.rs`), its fragment imageblock storage
    /// (`fragment_imageblock.rs`) and its shader-side stencil's shadow (`shadow_stencil.rs`);
    /// else an execution's draw that reaches one makes it, and the pass clears a fragment
    /// imageblock storage so made before its first instance.
    storage: PassStorage,
    /// The pass's layout as the encoder ended it, and without its hidden depth and stencil
    /// (`bare`, for the instances before they arrived), of which a run the instance has no
    /// passes for makes its pass and framebuffer.
    layout: PassLayout,
    bare: Option<PassLayout>,
    /// The shadows of the shadowed textures that are attachments of the pass, or that its draws
    /// or tile dispatches use (`render/shadows.rs`, `shadows::pass_shadows`): the copies around
    /// its instances follow the bytes each execution's call gathered.
    shadows: Vec<Arc<ShadowCopies>>,
    /// What the commit made ahead for the host step that records it (`ahead.rs`), once made.
    ahead: OnceLock<Arc<Ahead>>,
    /// The makings its executions' settle at commit handed to the device's pool
    /// ([`settle`](Self::settle)), which its host step waits for, until its command buffer's bound
    /// at most, as for its draws' pipelines.
    settled: Mutex<Vec<Pending<RawPipeline>>>,
    /// Where `commit` recorded the pass with executions in their GPU form (`gpu.rs`), what each
    /// such execution's buffer's records may hold as the decision found it: a route encoded
    /// since that adds to it sends the pass to the host step ([`outgrown`](Self::outgrown)),
    /// which empties it.
    on_gpu: Mutex<Vec<(Arc<WrittenCommands>, CommandWrites)>>,
    /// The rule that sent the pass to the host step last ([`HostStepCause`]): set where `commit`
    /// does not record it on the GPU alone, and where a route outgrew that decision. Its notice
    /// names it.
    cause: Mutex<Option<HostStepCause>>,
}

/// One render pass instance of a [`DeferredPass`].
pub(super) struct PassInstance {
    pub(super) pass: vk::RenderPass,
    pub(super) framebuffer: vk::Framebuffer,
    /// Its shape, and where it falls among the pass's instances.
    pub(super) shape: PassShape,
    pub(super) position: Position,
    /// What its subpass runs, in order: the secondaries of the encoder's draws, and the
    /// executions'.
    pub(super) pieces: Vec<Piece>,
    /// The passes it runs as, where an execution of it may split it into runs
    /// ([`RenderExecution::reaches_tile_memory`]).
    pub(super) runs: Option<RunPasses>,
}

/// The render passes of an instance an execution splits into runs (`render/tile_memory.rs`),
/// each compatible with the instance's own and its framebuffer: the first run's (the instance's
/// load actions, everything stored), a run's between two others (everything loaded and stored),
/// and the last run's (everything loaded, the instance's store actions and resolves).
pub(super) struct RunPasses {
    pub(super) first: vk::RenderPass,
    pub(super) between: vk::RenderPass,
    pub(super) last: vk::RenderPass,
}

impl RunPasses {
    /// The pass of run `at` of an instance's `count` runs.
    fn at(&self, at: usize, count: usize) -> vk::RenderPass {
        match at {
            0 => self.first,
            at if at + 1 == count => self.last,
            _ => self.between,
        }
    }
}

impl DeferredPass {
    /// The pass of `instances`, whose pieces name `executions` and `pending` draws by index,
    /// with `steps` around them, with the attachments' and the steps' `uses`, its `storage`
    /// where the encoder made it, and its `layout` as the encoder ended it with that layout
    /// without its hidden depth and stencil (`bare`); `held` the command buffer of its leading
    /// instances where the encoder recorded those inline, and their count, and `instances` the
    /// later ones (the field's docs); `encoder` its encoder's place among the command buffer's
    /// ended encoders.
    pub(super) fn new(
        (instances, held): (Vec<PassInstance>, Option<(vk::CommandBuffer, usize)>),
        steps: PassSteps,
        uses: Vec<Use<ResourceKey>>,
        (executions, pending): (Vec<RenderExecution>, Vec<PendingDraw>),
        storage: PassStorage,
        (layout, bare): (PassLayout, Option<PassLayout>),
        shadows: Vec<Arc<ShadowCopies>>,
        encoder: usize,
    ) -> DeferredPass {
        DeferredPass {
            instances,
            first: held.map_or(0, |(_, inlined)| inlined),
            held: held.map(|(held, _)| held),
            steps,
            uses,
            executions,
            pending,
            encoder,
            storage,
            layout,
            bare,
            shadows,
            ahead: OnceLock::new(),
            settled: Mutex::new(Vec::new()),
            on_gpu: Mutex::new(Vec::new()),
            cause: Mutex::new(None),
        }
    }

    /// Make, at the commit of `command_buffer` and on its thread, what the host step that
    /// records the pass needs and would otherwise make on the scheduler's thread, a serial point
    /// of the command buffers' stream: `ahead.rs`. Once; a part it fails to make the step makes
    /// as it meets it.
    pub(crate) fn make_ahead(&self, command_buffer: &CommandBuffer) {
        if self.ahead.get().is_some() {
            return;
        }
        let ahead = Ahead::make(self, command_buffer);
        let _ = self.ahead.set(Arc::new(ahead));
    }

    /// Whether the GPU may write what an execution of the pass reads: its records, where an
    /// encoder made its buffer resident for writing (`IndirectCommandBuffer::gpu_written`, for
    /// good), or the range it reads from a buffer.
    pub(crate) fn gpu_written(&self) -> bool {
        self.executions
            .iter()
            .any(|execution| execution.icb.gpu_written() || execution.range.indirect())
    }

    /// Whether a pipeline one of its draws or tile dispatches records is still being made, or one
    /// its executions' settle handed to the device's pool ([`settle`](Self::settle)).
    pub(crate) fn making(&self) -> bool {
        self.pending.iter().any(|draw| !draw.made())
            || self.steps.tile_pipelines().any(TilePipelines::making)
            || lock(&self.settled)
                .iter()
                .any(|pending| pending.outcome().is_none())
    }

    /// Whether the making of a pipeline one of its draws records failed: its recording, which a
    /// host step does, drops the draw (`render/pending.rs`); or one of its tile dispatches',
    /// which its recording makes again (`render/pending_tile.rs`).
    pub(crate) fn failed(&self) -> bool {
        self.pending.iter().any(PendingDraw::failed)
            || self.steps.tile_pipelines().any(TilePipelines::failed)
    }

    /// Whether it executes render commands, whose records its recording reads: everything else it
    /// records its calls resolved (`Deferred::resolved_at_its_calls`).
    pub(crate) fn executes(&self) -> bool {
        !self.executions.is_empty()
    }

    /// Whether it executes render commands whose records the CPU writes, which its recording
    /// reads as they stand at commit (the module docs).
    pub(crate) fn reads_at_commit(&self) -> bool {
        !self.executions.is_empty() && !self.gpu_written()
    }

    /// What its recording ([`record`](Self::record)) uses, where that is known before it
    /// records: a pass that executes no render commands records its instances alone, one command
    /// of its attachments' and steps' uses (its draws' are its encoder's), as `record` notes
    /// them, whatever pipeline its draws wait for. `None` for one that executes render commands,
    /// whose uses its recording finds.
    pub(crate) fn known_summary(&self, device: &Device) -> Option<Summary<ResourceKey>> {
        if !self.executions.is_empty() {
            return None;
        }
        let aliased = device.shared().aliases.augment(&self.uses);
        let mut uses = EncoderUses::new(false);
        uses.record(
            render_target::ATTACHMENT_STAGES,
            aliased.as_deref().unwrap_or(&self.uses),
            None,
        );
        Some(uses.finish())
    }

    /// Wait until every pipeline its draws and tile dispatches record is made, or failed, each
    /// making hastened first (`pipeline/pool.rs`), until `deadline` at most, its command buffer's
    /// bound, past which its recording drops the draw or tile dispatch (`render/pending.rs`,
    /// `render/pending_tile.rs`).
    pub(crate) fn wait_made(&self, deadline: Option<Instant>) {
        self.hasten(Urgency::Committed);
        for draw in &self.pending {
            // Its outcome is the recording's to read.
            match deadline {
                Some(deadline) => drop(draw.pipeline.wait_until(deadline)),
                None => drop(draw.pipeline.wait()),
            }
        }
        for tile in self.steps.tile_pipelines() {
            tile.wait(deadline);
        }
        let settled = lock(&self.settled).clone();
        for pending in &settled {
            match deadline {
                Some(deadline) => drop(pending.wait_until(deadline)),
                None => drop(pending.wait()),
            }
        }
    }

    /// Run the makings of the pipelines its draws and tile dispatches record before the
    /// device's pool's pieces of a lower rank than `urgency`: a submission waits for them.
    pub(crate) fn hasten(&self, urgency: Urgency) {
        for draw in &self.pending {
            draw.pipeline.hasten(urgency);
        }
        for tile in self.steps.tile_pipelines() {
            tile.hasten(urgency);
        }
        for pending in lock(&self.settled).iter() {
            pending.hasten(urgency);
        }
    }

    /// Run `waker` each time the making of a pipeline its draws or tile dispatches record
    /// finishes: at once for one finished already (none is skipped, so no finish goes
    /// unannounced).
    pub(crate) fn when_made(&self, waker: &(impl Fn() + Clone + Send + 'static)) {
        for draw in &self.pending {
            draw.when_made(waker.clone());
        }
        for tile in self.steps.tile_pipelines() {
            tile.when_made(waker);
        }
        for pending in lock(&self.settled).iter() {
            let waker = waker.clone();
            pending.when_finished(Box::new(move |_| waker()));
        }
    }

    /// Announce, once per device, an execution of a buffer from a command an earlier execution
    /// of it in the pass started at with another length, the ranges read from buffers as the
    /// host step reads them (the encoder announced two given ranges at the call). An empty range
    /// runs nothing and counts for neither.
    fn degrade_repeated_start(&self, device: &Device) -> Result<(), Error> {
        if !self.executions.iter().any(|e| e.range.indirect()) {
            return Ok(());
        }
        // The first length each buffer (by its resource ID) was executed with from each command.
        let mut first: BTreeMap<(u64, u32), usize> = BTreeMap::new();
        for execution in &self.executions {
            let (range, _) = execution.range.commands(execution.icb.size())?;
            if range.is_empty() {
                continue;
            }
            let key = (execution.icb.gpu_resource_id().0, range.start);
            let earlier = *first.entry(key).or_insert(range.len());
            if earlier != range.len() {
                degrade_repeated_start(device, &execution.icb, &range, earlier);
                break;
            }
        }
        Ok(())
    }

    /// Record the pass into a segment of `command_buffer` (the module docs): the executions'
    /// secondaries from the CPU's records as they stand now, or for the host step (`step`) from
    /// the records as the GPU and the CPU left them; then the pass.
    pub(crate) fn record(
        &self,
        command_buffer: &CommandBuffer,
        step: bool,
    ) -> Result<RecordedSegment, Error> {
        self.record_forms(command_buffer, step, &[])
    }

    /// Record the pass at commit, for good, where every execution whose records the GPU may write
    /// takes its GPU form (`gpu.rs`, [`RenderExecution::gpu_form`]) and the others read the CPU's
    /// records as they stand now; `None` where one does not, and a host step records the pass.
    /// The pipelines its draws record are waited for first.
    pub(crate) fn record_on_gpu(
        &self,
        command_buffer: &CommandBuffer,
    ) -> Result<Option<RecordedSegment>, Error> {
        let declined = |cause: HostStepCause| {
            *lock(&self.cause) = Some(cause);
            Ok(None)
        };
        if self.executions.iter().any(|e| e.range.indirect()) {
            let rule = "an execution of the pass reads its range from a buffer";
            return declined(HostStepCause::Rule(rule));
        }
        // A pass with a shadowed texture runs as any other, its copies around its instances
        // (`render/shadows.rs`), unless its commands may reach a shadow's truth past what their
        // calls gathered, which the host step refuses too (`record_forms`).
        let residency = command_buffer.residency()?;
        if self.reaches_past_its_copies(command_buffer.device(), &residency) {
            let rule = "the command buffer made resident, past an execution's call, what meets \
                        the truth of a shadowed texture the pass copies";
            return declined(HostStepCause::Rule(rule));
        }
        let mut core = EncoderCore::segment(command_buffer, false)?;
        let mut forms = Vec::with_capacity(self.executions.len());
        for execution in &self.executions {
            if !execution.icb.gpu_written() {
                forms.push(None);
                continue;
            }
            let (back, form) = execution.gpu_form(core, (self.storage.clone(), &residency));
            core = back;
            match form {
                Ok(form) => forms.push(Some(form)),
                Err(cause) => return declined(cause),
            }
        }
        drop(core);
        self.wait_made(command_buffer.making_deadline());
        if self.making() || self.failed() {
            let rule = "a pipeline state the pass draws with is being made or failed to make";
            return declined(HostStepCause::Rule(rule));
        }
        let recorded = self.record_forms(command_buffer, false, &forms)?;
        let decided = (self.executions.iter().zip(&forms)).filter_map(|(execution, form)| {
            Some((Arc::clone(execution.icb.written()), form.as_ref()?.decided))
        });
        *lock(&self.on_gpu) = decided.collect();
        // What a host step needs where a route outgrows the decision (`outgrown`), made now on
        // the committing thread, as for any pass a host step records.
        self.make_ahead(command_buffer);
        Ok(Some(recorded))
    }

    /// Whether a route encoded since `commit` recorded the pass with executions in their GPU form
    /// may write more into one of their buffers than the decision found (`gpu.rs`): a route that
    /// runs before the pass, which a command buffer enqueued before its own and committed after it
    /// may hold, may have written commands the GPU form does not run; or the CPU wrote the records
    /// of one of its other executions, which `commit` read ([`rewritten`](Self::rewritten)). Until
    /// its batch is submitted, the host step records the pass instead (`Deferred::outgrown`).
    pub(crate) fn outgrown(&self) -> bool {
        let on_gpu = lock(&self.on_gpu);
        !on_gpu.is_empty()
            && (on_gpu
                .iter()
                .any(|(written, decided)| written.writes() != *decided)
                || self.rewritten())
    }

    /// Whether `commit` recorded one of its executions from the CPU's records, which the CPU
    /// wrote since (`RenderExecution::rewritten`): its host step records the pass again, from the
    /// records as they stand then, before its batch is submitted (`Deferred::rewritten`; where
    /// `commit` recorded the pass on the GPU for good, [`outgrown`](Self::outgrown)).
    pub(crate) fn rewritten(&self) -> bool {
        self.executions.iter().any(RenderExecution::rewritten)
    }

    /// [`record`](Self::record), each execution with a GPU form among `forms` (by index) in it.
    fn record_forms(
        &self,
        command_buffer: &CommandBuffer,
        step: bool,
        forms: &[Option<GpuForm>],
    ) -> Result<RecordedSegment, Error> {
        // A rebind since the commit moved a buffer one of its commands named, and no encoding
        // again reached this encoder (`CommandBuffer::encoder_rebound`): its instances'
        // secondaries and its pending draws' and tile dispatches' recordings resolved that buffer
        // where it lay as they were encoded, so recording them now would bind a `VkBuffer` the
        // rebind freed. The Vulkan specification has a descriptor's contents consumed as its
        // command is recorded, and the pages stay mapped, so a driver that reads one writes into
        // pages the program handed back with nothing said. Its submission then runs nothing and
        // the command buffer fails `CommandBufferError::EncodedUseRebound`
        // (`first_submission.rs`), as for a dispatch whose pipelines were being made
        // (`pending_dispatch.rs`).
        if command_buffer.encoder_rebound(self.encoder) {
            return Ok(EncoderCore::unrecorded_segment());
        }
        if step {
            // The host step records what a GPU form left to it, for good: a route outgrew the
            // decision where one was made.
            let mut on_gpu = lock(&self.on_gpu);
            if !on_gpu.is_empty() {
                *lock(&self.cause) = Some(HostStepCause::Outgrown);
            }
            on_gpu.clear();
            drop(on_gpu);
            self.degrade_repeated_start(command_buffer.device())?;
        }
        // The executions resolve and bind against the command buffer's resident resources,
        // every encoder's (`residency.rs`). Where the pass has a shadowed texture, the copies
        // around its instances were planned over the bytes each call gathered, from its
        // encoder's resources then: one made resident beyond them that may touch a shadowed
        // texture's truth is refused.
        let residency = command_buffer.residency()?;
        if self.reaches_past_its_copies(command_buffer.device(), &residency) {
            return Err(Error::NotImplemented(RuntimeGap::IndirectRenderCommands));
        }
        // Its tile dispatches' pipelines, waiting for those being made; a failed one fails the
        // command buffer (`render/pending_tile.rs`).
        let tiles = self.steps.tiles_made(command_buffer)?;
        let mut core = EncoderCore::segment(command_buffer, false)?;
        let mut found = Found::default();
        let mut recorded = Vec::with_capacity(self.executions.len());
        let mut storage = self.storage.clone();
        let mut stood = Vec::new();
        // The pass's layout as its instances render it: its memoryless attachments backed from
        // the first execution whose commands run as several on.
        let mut rendered = Rendered {
            layout: self.layout.clone(),
            bare: self.bare.clone(),
            backed: Vec::new(),
            queue: command_buffer.command_queue().id(),
        };
        for (at, execution) in self.executions.iter().enumerate() {
            let gpu = forms.get(at).and_then(Option::as_ref);
            let back = execution.record(
                core,
                (command_buffer, &residency),
                (step, gpu),
                &mut found,
                (storage, &mut rendered),
            )?;
            core = back.core;
            storage = back.storage;
            // The stand-ins and sample planes the commands' draws made: held until the command
            // buffer completes, and announced below as the encoder's own.
            if back.runs.iter().any(|run| run.steps.is_some()) {
                hold_colors(&mut core, &back.layout);
                stood.push(back.layout);
            }
            recorded.push(back.runs);
        }
        // The fragment imageblock storage a command's draw made, cleared before the pass as the
        // encoder clears its own, held until the command buffer completes, and announced.
        if let (None, Some(made)) = (&self.storage.imageblock, &storage.imageblock) {
            made.clear()?.record(&mut core);
            core.retain(made.held());
            command_buffer.device().degrade_fragment_imageblock(made);
        }
        // The shader-side stencil's copies around each run of draws that test it in the shader,
        // as around an encoder's instance of such draws (`shadow_stencil.rs`): the encoder's
        // shadow, or one a command's draw made, which the pass's uses then gain, held until the
        // command buffer completes and announced. No such run is an instance's first, which runs
        // the pass's load actions, nor its last: the shadow is filled by a copy of the stencil.
        let mut uses = self.uses.clone();
        // The memoryless attachments' backings, attachments of the pass, held until the command
        // buffer completes.
        for backing in &rendered.backed {
            uses.extend(render_target::target_uses(
                &backing.target,
                backing.index.is_none(),
            )?);
        }
        let backings = rendered.backed.iter().map(|b| b.target.texture.clone());
        core.retain(backings.collect::<Vec<_>>());
        // The runs' steps' uses (`RunSteps`).
        uses.extend(
            recorded
                .iter()
                .flatten()
                .filter_map(|run| run.steps.as_ref())
                .flat_map(RunSteps::uses)
                .cloned(),
        );
        // The steps before the runs (`Run::before`).
        uses.extend(
            recorded
                .iter()
                .flatten()
                .flat_map(|run| &run.before)
                .flat_map(|step| step.uses().iter().cloned()),
        );
        let shadow_runs = recorded.iter().flatten().filter(|run| run.shadow).count();
        let shadow_copies = match (&storage.shadow, &rendered.layout.depth_stencil) {
            (Some(shadow), Some(ds)) if shadow_runs > 0 => {
                if self.storage.shadow.is_none() {
                    uses.extend(shadow.copy_uses(&ds.target)?);
                    core.retain(shadow.held());
                    let device = command_buffer.device();
                    device.degrade_shadow_stencil(&rendered.layout, true);
                }
                Some(shadow.copies(&ds.target, None)?)
            }
            (_, _) if shadow_runs == 0 => None,
            (_, _) => {
                return Err(internal(
                    "a run of a shader-side stencil without its shadow or stencil",
                ))
            }
        };
        // Each instance's runs: its pieces in order, a new run at each execution's run past its
        // first, and after a run of another shape than the instance's, since each piece's first
        // secondary was begun against the instance's pass. A run of no passes of the instance's has
        // its pass, and a run of another shape its framebuffer too, made here and held until the
        // command buffer completes.
        let device = command_buffer.device();
        let mut made = (Vec::new(), Vec::new());
        let mut instances = Vec::with_capacity(self.instances.len());
        let mut shapes = Vec::new();
        let mut viewed = false;
        // The instances' own objects hold the pass's memoryless attachments: an instance that
        // runs as several renders their backings instead.
        let backed = !rendered.backed.is_empty();
        // What the commit made ahead for the step (`ahead.rs`): the runs' passes of an instance
        // the encoder gave none, and, where the pass renders the backings the commit kept, each
        // instance's passes and framebuffer over them.
        let ahead = self.ahead.get();
        let ahead_backed = match ahead.and_then(|ahead| ahead.backed.as_ref()) {
            Some(made) if backed && made.backs(&rendered.backed)? => Some(made),
            Some(_) | None => None,
        };
        for (index, instance) in self.instances.iter().enumerate() {
            // Each run: an execution's past its first (its shape and views where they are
            // another's, whether its draws test the stencil in the shader, its steps, and the
            // steps before it), or none for the instance's own, and its secondaries.
            let mut runs: Vec<(Option<&Run>, Vec<vk::CommandBuffer>)> = vec![(None, Vec::new())];
            for piece in &instance.pieces {
                let (first, later) = match *piece {
                    Piece::Draws(draws) => (draws, &[][..]),
                    // A draw's own secondary, recorded once its pipeline is made.
                    Piece::Pending(index) => {
                        let draw = self
                            .pending
                            .get(index)
                            .ok_or_else(|| internal("a pending draw the pass does not hold"))?;
                        (draw.record(command_buffer)?, &[][..])
                    }
                    Piece::Execution(index) => match recorded[index].split_first() {
                        // An execution's first run shares its instance's render pass instance
                        // with what runs before it: nothing records before it alone.
                        Some((first, _)) if !first.before.is_empty() => {
                            return Err(internal("an execution's first run with steps before it"))
                        }
                        Some((first, later)) => (first.draws, later),
                        None => return Err(internal("an execution of no runs")),
                    },
                };
                match runs.last_mut() {
                    Some((last, run)) if last.is_none_or(Run::takes_more) => run.push(first),
                    _ => runs.push((None, vec![first])),
                }
                runs.extend(later.iter().map(|run| (Some(run), vec![run.draws])));
            }
            let count = runs.len();
            let run_passes = instance
                .runs
                .as_ref()
                .or_else(|| ahead?.runs.get(index)?.as_ref());
            let backed_objects = ahead_backed.and_then(|made| made.instances.get(index)?.as_ref());
            let mut objects = Vec::with_capacity(count);
            for (at, (of, run)) in runs.into_iter().enumerate() {
                let other = of.and_then(|of| of.other.as_ref());
                let shadowed = of.is_some_and(|of| of.shadow);
                let steps = of.and_then(|of| of.steps.as_ref());
                let before = of.map_or(&[][..], |of| &of.before[..]);
                let position = Position {
                    first: instance.position.first && at == 0,
                    last: instance.position.last && at + 1 == count,
                };
                let (shape, views) = match other {
                    Some((shape, views)) => (shape, *views),
                    None => (&instance.shape, ViewSetup::default()),
                };
                shapes.push(shape);
                viewed |= views != ViewSetup::default();
                let own = super::instance_layout(&rendered.layout, rendered.bare.as_ref(), shape);
                let (pass, framebuffer) = match (other, run_passes, backed_objects) {
                    (None, _, _) if count == 1 => (instance.pass, instance.framebuffer),
                    (None, Some(passes), _) if !backed => {
                        (passes.at(at, count), instance.framebuffer)
                    }
                    (None, _, Some((passes, framebuffer))) => (passes.at(at, count), *framebuffer),
                    // An instance of several runs holds an execution, so it holds every attachment
                    // in `GENERAL` (`render_target::depth_layout`), as its runs' passes do.
                    (None, None, _) if !backed => {
                        let pass = render_target::final_pass(
                            device,
                            own,
                            shape,
                            position,
                            Feedback::NONE,
                            vk::ImageLayout::GENERAL,
                        )?;
                        let raw = pass.raw;
                        made.0.push(pass);
                        (raw, instance.framebuffer)
                    }
                    (Some(_), _, _) | (None, _, _) => {
                        // Its framebuffer holds the stand-ins its draws render in place of the
                        // pass's attachments, and the backings of its memoryless ones, which its
                        // instance's own framebuffer lacks.
                        let framed = of.map_or(&[][..], |of| &of.framed[..]);
                        let framed = framed_layout(own, framed);
                        let own = &*framed;
                        let (none, general) = (Feedback::NONE, vk::ImageLayout::GENERAL);
                        let pass =
                            render_target::final_pass(device, own, shape, position, none, general)?;
                        let framebuffer =
                            render_target::framebuffer(device, &pass, own, shape, views, none)?;
                        let raw = (pass.raw, framebuffer.raw);
                        made.0.push(pass);
                        made.1.push(framebuffer);
                        raw
                    }
                };
                if (shadowed || steps.is_some()) && (at == 0 || at + 1 == count) {
                    return Err(internal(
                        "a run with steps around it first or last in its instance",
                    ));
                }
                objects.push((pass, framebuffer, run, (shadowed, steps, before)));
            }
            instances.push(objects);
        }
        core.retain(made);
        if let Some(ahead) = ahead {
            core.retain(Arc::clone(ahead));
        }
        // The first instance the encoder recorded inline: after what the segment recorded before
        // the pass, as it runs where the segment records it.
        if let Some(held) = self.held {
            core.cut_before(held)?;
        }
        let steps = &self.steps;
        let first = self.first;
        let shadow_copies = shadow_copies.as_ref();
        core.command(render_target::ATTACHMENT_STAGES, &uses, |raw, cb| {
            // SAFETY: the segment's command buffer is recording outside a render pass; the
            // passes, framebuffers and attachments are live until the command buffer completes
            // (its encoder retained them), as are the steps' objects (held with the pass, which
            // the encoder retained) and the shadowed textures the copies name (retained with the
            // draws, the executions and the attachments that use them); each secondary was
            // recorded against a pass compatible with its instance's and ended; every image rests
            // in `GENERAL`, and each step is recorded between two instances, as the encoder's own
            // pass records it; the shadow's copies, around a run that is neither its instance's
            // first nor its last, name the shadow and the stencil, held so too; the steps before a
            // run past an execution's first are held with the runs, each recorded between barriers
            // outside a render pass.
            unsafe {
                // The tile dispatches after the instances the encoder recorded inline, which the
                // command buffer it handed over, placed before these, leaves out
                // (`render/pending_tile.rs`).
                if let Some(last) = first.checked_sub(1) {
                    steps.dispatches_after(raw, cb, last, &tiles);
                }
                let runs = instances.iter().enumerate().flat_map(|(at, runs)| {
                    let last = runs.len() - 1;
                    runs.iter()
                        .enumerate()
                        .map(move |(run, (pass, framebuffer, cbs, around))| {
                            (at, run, run == last, (*framebuffer, *pass), cbs, *around)
                        })
                });
                for (index, (at, run_index, last_run, (framebuffer, pass), run, around)) in
                    runs.enumerate()
                {
                    let (shadowed, run_steps, before) = around;
                    // Each instance after the first sees everything the one before wrote.
                    if index > 0 {
                        Barrier::everything().record(raw, cb);
                    }
                    // Its steps before its first run, and after its last (`render/pass_steps.rs`).
                    if run_index == 0 {
                        steps.before(raw, cb, at + first);
                    }
                    let copies = shadow_copies.filter(|_| shadowed);
                    if let Some(copies) = copies {
                        copies.record_in(raw, cb, false);
                    }
                    for step in before {
                        step.record_in(raw, cb);
                    }
                    if let Some(run_steps) = run_steps {
                        run_steps.before(raw, cb);
                    }
                    steps.begin(raw, cb, at + first, pass, framebuffer);
                    raw.cmd_execute_commands(cb, run);
                    raw.cmd_end_render_pass(cb);
                    steps.ended(raw, cb, at + first);
                    if let Some(run_steps) = run_steps {
                        run_steps.after(raw, cb);
                    }
                    if let Some(copies) = copies {
                        copies.record_out(raw, cb);
                    }
                    if last_run {
                        steps.after(raw, cb, at + first, &tiles);
                    }
                }
                steps.end(raw, cb);
            }
        });
        if shapes.len() > self.instances.len() {
            let causes = SplitCauses {
                shadow: shadow_runs > 0,
                tile_memory: found.tile_memory_splits,
                barriers: found.barrier_splits,
                viewed,
                ..SplitCauses::default()
            };
            let device = command_buffer.device();
            device.degrade_instances(&rendered.layout, &shapes, causes);
            for layout in &stood {
                device.degrade_instances(layout, &shapes, causes);
                device.degrade_planes(layout);
            }
        }
        // The runs' steps' objects (the sample planes' descriptor sets and framebuffers), and the
        // tile dispatches' pipelines, held until the command buffer completes.
        core.retain(recorded);
        core.retain(tiles);
        let mut segment = core.end_segment()?;
        segment.resident_misses = found.misses;
        segment.faults = found.faults;
        Ok(segment)
    }

    /// Settle at commit what the host step that records the pass would compile
    /// ([`RenderExecution::settle`]): for each execution, the variants and `VkPipeline`s of the
    /// draws of each state its records may name, the encoder's where its buffer inherits it,
    /// else those made with `supportIndirectCommandBuffers` alive now. Each such state's creation
    /// compiled its variant in the large-buffer form, which the commit looks up, and the commit
    /// begins the `VkPipeline`s: under `PipelineWaitPolicy::Bounded` (the default) each making is
    /// handed to the device's pool (`RenderVariant::settle_graphics_pipeline`), kept here for the
    /// host step to wait for until its command buffer's bound at most, so the commit compiles none
    /// on the program's thread; under `Exact` made now. It compiles a variant only where the
    /// encoder's state asks for another, and the step only where a command's own buffers or
    /// strides ask for another, or the draw records into a run of another shape. Returns what it
    /// made, which the command buffer holds until it completes; what it cannot settle, the step
    /// meets as it is then.
    pub(crate) fn settle(&self, command_buffer: &CommandBuffer) -> Vec<SettledDraw> {
        let mut made = Vec::new();
        let Ok(residency) = command_buffer.residency() else {
            return made;
        };
        // The step refuses such a pass (`record`).
        if self.reaches_past_its_copies(command_buffer.device(), &residency) {
            return made;
        }
        let Ok(mut core) = EncoderCore::segment(command_buffer, false) else {
            return made;
        };
        let states = match self.executions.iter().any(|e| e.pipeline.is_none()) {
            true => RenderPipelineState::indirect_states(command_buffer.device()),
            false => Vec::new(),
        };
        for execution in &self.executions {
            let inputs = (self.storage.clone(), &residency);
            core = execution.settle(core, inputs, &states, &mut made);
        }
        // The makings handed to the pool, which the host step waits for.
        lock(&self.settled).extend(made.iter().filter_map(|(_, drawn)| match drawn {
            Some(DrawPipeline::Making(pending)) => Some(pending.clone()),
            _ => None,
        }));
        made
    }

    /// Whether an execution's commands may reach, through `residency` (the command buffer's), a
    /// resource its call did not gather that may touch the truth of a shadowed texture the pass
    /// copies around its instances (`shadows::meets_truths`): those copies follow what each call
    /// gathered (`render/shadows.rs`), so the pass is refused. A resource no shadow's truth shares
    /// (the records of an indirect command buffer a kernel wrote, a buffer of other bytes) changes
    /// none of them.
    fn reaches_past_its_copies(&self, device: &Device, residency: &Residency) -> bool {
        !self.shadows.is_empty()
            && self.executions.iter().any(|execution| {
                (residency.past(&execution.snapshot.residency))
                    .any(|used| super::shadows::meets_truths(device, &self.shadows, used))
            })
    }

    /// Announce, once per device, that a host step records the pass, naming the rule that kept
    /// it from the GPU alone where one did ([`HostStepCause`]).
    pub(crate) fn degrade(&self, device: &Device) {
        let Some(execution) = self
            .executions
            .iter()
            .find(|execution| execution.icb.gpu_written() || execution.range.indirect())
        else {
            return;
        };
        let why = if execution.icb.gpu_written() {
            "an encoder made it resident for writing, so the GPU may write the render commands an \
             execution runs: its render pass is recorded once the work before it completed"
        } else {
            "an execution reads its range from a buffer \
             (executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:), which the GPU may \
             write: its render pass is recorded once the work before it completed"
        };
        let cause = lock(&self.cause).clone();
        device.degrade(Degradation::HostReplayedIndirectCommands, || {
            let why = match cause {
                Some(cause) => format!("{why}, not on the GPU alone, as {cause}"),
                None => why.to_owned(),
            };
            (
                format!(
                    "an indirect command buffer of {} render commands",
                    execution.icb.size()
                ),
                why,
            )
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{patch_scale, Scale};

    /// What each scale does, as Metal rounds it to `half`: 1 nothing to the factors, another
    /// positive `half` multiplies them, and what is not positive or is 0 as a `half` culls (the
    /// probe's 0, -1 and NaN among them).
    #[test]
    fn patch_scales_are_classified_as_metal_measured_them() {
        let cases = [
            (1.0, Scale::One),
            (2.0, Scale::By(2.0)),
            (0.5, Scale::By(0.5)),
            (65504.0, Scale::By(65504.0)),
            (
                f32::from_bits(0x3880_0000),
                Scale::By(f32::from_bits(0x3880_0000)),
            ),
            (0.0, Scale::Culls),
            (-0.0, Scale::Culls),
            (-1.0, Scale::Culls),
            (f32::NEG_INFINITY, Scale::Culls),
            (f32::NAN, Scale::Culls),
            (1.3, Scale::By(1.3)),
            // 1 as a `half`: the measured 1.0001 and 1.0004, and the ties either side of 1, which
            // round to its even significand.
            (1.0001, Scale::One),
            (1.0004, Scale::One),
            (1.0 - 2f32.powi(-12), Scale::One),
            (1.0 + 2f32.powi(-11), Scale::One),
            // The largest `float` below 2 and the smallest above it: past `half`'s significand.
            (
                f32::from_bits(0x3fff_ffff),
                Scale::By(f32::from_bits(0x3fff_ffff)),
            ),
            (
                f32::from_bits(0x4000_0001),
                Scale::By(f32::from_bits(0x4000_0001)),
            ),
            // `half`'s next value above 1, its largest and least subnormals, and 10^-6.
            (
                f32::from_bits(0x3f80_2000),
                Scale::By(f32::from_bits(0x3f80_2000)),
            ),
            (
                f32::from_bits(0x387f_c000),
                Scale::By(f32::from_bits(0x387f_c000)),
            ),
            (2f32.powi(-24), Scale::By(2f32.powi(-24))),
            (1.0e-6, Scale::By(1.0e-6)),
            // Below the least subnormal: 2^-25 is a tie, which rounds to 0, its even neighbour;
            // 1.5 · 2^-25 rounds up to 2^-24; 2^-30 is 0.
            (2f32.powi(-25), Scale::Culls),
            (1.5 * 2f32.powi(-25), Scale::By(1.5 * 2f32.powi(-25))),
            (2f32.powi(-30), Scale::Culls),
            // Infinite as a `half`.
            (65520.0, Scale::By(65520.0)),
            (70000.0, Scale::By(70000.0)),
            (f32::INFINITY, Scale::By(f32::INFINITY)),
        ];
        for (scale, expected) in cases {
            assert_eq!(patch_scale(scale.to_bits()), expected, "{scale}");
        }
    }
}
