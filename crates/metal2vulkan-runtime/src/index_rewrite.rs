//! The index rewrite pass of an indexed strip that restarts where Vulkan does not, of a draw of
//! one of Metal's private primitive types over indices or from indirect arguments, and of a draw
//! under Apple's last provoking vertex that a device without it draws as a list led by each
//! primitive's last vertex (`render/provoking.rs`) (the modules and their contract are
//! `bind::index_rewrite`); the list the CPU writes for such a direct draw without indices
//! ([`WrittenList`]); and the lists of the index gather pass ([`GatherRuns`]). Each list is
//! written in the draw's `RewriteOrder`.
//!
//! A private type's list drawn in the private list form (`VariantKey::private_list`, chosen in
//! `render.rs`) is not drawn as indices: Metal's base vertex lies in the word before it, the
//! draw is a draw of its positions from its first index as first vertex, and the vertex function
//! reads its index and the base at the list's device address ([`RewritePass::address`],
//! [`WrittenList::address`]), which the index pool's block has where the device has device
//! addresses (`gpu_pools.rs`).
//!
//! **The gather pass** (`bind::index_gather_module`) writes the list of a draw whose list follows
//! from its positions, not in the private list form, that reads its indices and arguments in
//! place and that no earlier draw's write orders before its render pass instance: the list the
//! rewrite pass would write, index for index.
//!
//! - A direct indexed draw of a program that never restarts (`RewriteProgram::gathers`: line and
//!   triangle lists, Apple's triangle fan and quads) joins an open run of its index buffer,
//!   program and order (up to [`RUN_DRAWS`] draws, into an output buffer of the run's own of at
//!   least [`RUN_OUTPUT_WORDS`] words), and draws its part of the run's output with
//!   `vkCmdDrawIndexed`.
//! - An indirect draw, indexed of such a program or without indices of any program but a line
//!   loop (`RewriteProgram::gathers_without_indices`), joins an open indirect run of its
//!   arguments buffer, index buffer, program and order (up to [`RUN_DRAWS`] draws), whose plan
//!   (`bind::index_gather_plan_module`) reads Metal's arguments on the GPU, places the lists in
//!   the command buffer's index pool and writes each draw's command; the draw draws its command
//!   with `vkCmdDrawIndexedIndirect` over the pool. A list the pool cannot hold draws nothing and
//!   fails the command buffer with `CommandBufferError::IndexPoolOverflow`, as the rewrite pass's.
//!
//! **The rewrite runs** (`bind::index_rewrite_run_module`): a direct indexed draw of a program the
//! gather pass does not take (a strip, which restarts; Apple's line loop and quad strip), not in
//! the private list form, that reads its indices in place, whose walk takes one group
//! (`bind::IndexRewriteRun`: a few thousand positions) and that no earlier draw's write orders
//! before its render pass instance, joins an open rewrite run of its index buffer, program, `k'`
//! and order (up to [`REWRITE_RUN_DRAWS`] draws, into an output buffer of the run's own of at
//! least [`RUN_OUTPUT_WORDS`] words): each draw's three phases in a workgroup of its own of the
//! run's one dispatch. The draw draws its command with `vkCmdDrawIndexedIndirect` over the run's
//! output: the list its own pass would write, index for index.
//!
//! Its reads are tracked as it joins; the open runs (up to [`OPEN_RUNS`] of either pass) record
//! before the encoder's next command and before its `VkCommandBuffer` ends (`encoder.rs`): each
//! run's table or words written, then between one barrier from everything before and one to
//! everything after, the rewrite runs' dispatches and the indirect runs' plans, one after another,
//! and after the plans one dispatch per gather run. Every other draw's list is the rewrite pass's,
//! of three dispatches between barriers each: on MoltenVK on the M3 Max, 77 us of frame time per
//! draw of two triangles, where a run's draw takes 0.2 to 0.7 us more than at provoking vertex
//! mode 0 (host_ab's `provoking`).
//!
//! After `RenderCommandEncoder::set_primitive_restart_enabled` (Apple's
//! `setPrimitiveRestartEnabled:index:`), Metal restarts an indexed line or triangle strip at the
//! call's index `k'` alone, and draws the index type's all-ones index as a vertex; Vulkan restarts
//! at the all-ones index alone. Where `k'` is not the all-ones index, the render encoder draws the
//! strip as the list this pass writes:
//!
//! - **Before the pass** (hoisted, as the index bounds pass is, `index_bounds.rs`): three
//!   dispatches of the program's modules, one per `RewritePhase` (count, place, write), read
//!   Metal's indices (and an indirect draw's arguments) and write the list and the draw's
//!   `VkDrawIndexedIndirectCommand` into the draw's rewrite words (an upload of
//!   [`IndexRewritePass::BYTES`]), between a barrier from everything before and one to
//!   everything after. The count and write dispatches spread the draw's positions over up to
//!   `IndexRewritePass::MAX_GROUPS` workgroups, as many as its index buffer's whole indices need.
//! - **The list** of a direct draw lies in a buffer of its own, of the program's capacity of the
//!   draw's count, made for the draw; an indirect draw's, whose count only the GPU knows, in the
//!   command buffer's index pool, which the index bounds pass shares (its counter in the arena,
//!   its indices in its queue's block, which the command buffer leases, `gpu_pools.rs`): the count
//!   and place dispatches bind the counter as the output, the write dispatch the block. A list the
//!   pool cannot hold draws nothing and fails the command buffer with
//!   `CommandBufferError::IndexPoolOverflow`.
//! - **In the pass**, one indexed indirect draw of 32-bit indices over the list, of the class's
//!   list topology with restart off (`render.rs`): every primitive keeps its winding, its
//!   provoking vertex and its place in `primitive_id`'s count.
//! - **The hazard**, as the index bounds pass's: where an earlier draw or tile dispatch of the
//!   encoder writes the indices or arguments behind a memory barrier, the pass records before the
//!   draw's own render pass instance, after the writer's; a write since the last barrier that the
//!   pass would read in place is refused (`NotImplemented(RestartedStrip)`), as is any earlier
//!   write where an execution's command draws. A blit's or another encoder's write is ordered
//!   before the pass by the command buffer's tracking, as any compute read is.

use super::buffer::Buffer;
use super::degradation::Degradation;
use super::draw_source::{DrawSource, IndexBind, SizedIndexBind};
use super::encoder::EncoderCore;
use super::error::{internal, Error, RuntimeGap};
use super::gpu_pools::IndexPool;
use super::index_bounds::{self, BoundsRun, BoundsRunRequest, BoundsSlot};
use super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::resource::ResourceKey;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::upload::Upload;
use super::upload::Uploads;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{
    ExecutionStatus, GatherDraw, IndexBufferRef, IndexGather, IndexGatherModule,
    IndexGatherPlanModule, IndexRewriteModule, IndexRewritePass, IndexRewriteRun,
    IndirectGatherDraw, IndirectRef, RewriteOrder, RewriteOutput, RewritePhase, RewriteProgram,
    RewriteRunDraw, RewriteSource,
};
use metal2vulkan::device::VkFeature;

/// A rewritten draw's Metal arguments: a direct draw's values, or an indirect draw's buffer.
#[derive(Clone, Copy)]
pub(super) enum Arguments<'a> {
    /// A direct draw's index count, instances, base vertex and base instance.
    Direct {
        count: u32,
        instances: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// An indirect draw's `MTLDrawIndexedPrimitivesIndirectArguments`.
    Indirect(&'a DrawSource<'a>),
    /// An indirect draw's `MTLDrawPrimitivesIndirectArguments`, of a draw without indices.
    IndirectVertices(&'a DrawSource<'a>),
}

/// Where a pass writes its list.
enum Output {
    /// A direct draw's own buffer.
    Owned(Buffer),
    /// The command buffer's index pool.
    Pooled(IndexPool),
}

/// One draw's pass, planned: what records before the render pass instance, and the draw in it.
pub(super) struct RewritePass {
    words: Upload,
    output: Output,
    /// The kernels of its three dispatches, in order (`RewritePhase::ALL`).
    kernels: [OwnedKernel; 3],
    /// The pass the kernels run: its workgroups.
    pass: IndexRewritePass,
    /// The set of the count and place dispatches, and the write dispatch's where it differs (a
    /// pooled list's: the block where the others bind the counter).
    sets: (OwnedSets, Option<OwnedSets>),
    /// The pass's reads of the Metal arguments and indices.
    uses: Vec<Use<ResourceKey>>,
    /// An indirect draw's Metal index offset and the buffer's length, where the offset lies at or
    /// past the length: every position draws index 0 there, where Metal reads memory.
    past_length: Option<(u64, u64)>,
    /// The list is drawn in the private list form ([`IndexRewritePass::listed`]).
    listed: bool,
}

impl RewritePass {
    /// The pass of `program` restarting at `restart` (`k'`) in `order` over the `index_size`-byte
    /// indices of `index` (`None` for a draw without indices), with `arguments`, in `core`'s
    /// command buffer, whose reads of the Metal buffers are `uses` ([`reads`]), in the private
    /// list form where `listed` (the module docs). Refused
    /// (`NotImplemented(gap)`: `RestartedStrip`, `PrivatePrimitives` for a private type's list,
    /// `ProvokingVertex` for a list of the last provoking vertex) where a direct draw's list is
    /// past what one storage range binds or a `u32` counts, or carries a base instance the
    /// device's indirect draws cannot (`drawIndirectFirstInstance`), and where the binding layer
    /// cannot bind the pass's ranges; and as buffers, uploads, kernels and descriptor sets are
    /// refused.
    pub(super) fn new(
        core: &mut EncoderCore,
        ((program, restart, order), refused, listed): (
            (RewriteProgram, u32, RewriteOrder),
            RuntimeGap,
            bool,
        ),
        index: Option<&DrawSource<'_>>,
        arguments: &Arguments<'_>,
        index_size: u64,
        uses: Vec<Use<ResourceKey>>,
    ) -> Result<RewritePass, Error> {
        let device = core.device().clone();
        let limits = &device.properties().description.limits;
        let gap = || Error::NotImplemented(refused.clone());
        let (source, owned) = match *arguments {
            Arguments::Direct {
                count,
                instances,
                base_vertex,
                base_instance,
            } => {
                if base_instance != 0
                    && !device
                        .enabled()
                        .features
                        .contains(&VkFeature::DrawIndirectFirstInstance)
                {
                    return Err(gap());
                }
                // In the private list form, the base vertex's word before the list.
                let capacity = program.capacity(u64::from(count)) + u64::from(listed);
                let bytes = capacity * 4;
                if bytes > u64::from(limits.max_storage_buffer_range) {
                    return Err(gap());
                }
                let capacity = u32::try_from(capacity).map_err(|_| gap())?;
                let source = RewriteSource::Direct {
                    count,
                    instances,
                    base_vertex,
                    base_instance,
                };
                (source, Some((capacity, bytes.max(16))))
            }
            Arguments::Indirect(indirect) => {
                let source = RewriteSource::Indirect {
                    arguments: IndirectRef::new(0, indirect.at),
                };
                (source, None)
            }
            Arguments::IndirectVertices(indirect) => {
                let source = RewriteSource::IndirectVertices {
                    arguments: IndirectRef::new(0, indirect.at),
                };
                (source, None)
            }
        };
        let output = match owned {
            Some((capacity, _)) => RewriteOutput::Owned { capacity },
            None => RewriteOutput::Pooled {
                pool_bytes: u32::try_from(device.shared().gpu_pools.index_bytes)
                    .map_err(|_| internal("an index pool past a u32's bytes"))?,
            },
        };
        // The indices the draw reads before where Metal's fetch stops (the source's bytes); a draw
        // without indices reads none.
        let terms = match index {
            Some(index) => (index_size, index.at, index.at + index.bound(index.at)),
            None => (4, 0, 0),
        };
        let pass = IndexRewritePass::new(
            source,
            restart,
            terms,
            output,
            limits.min_storage_buffer_offset_alignment,
            limits.max_storage_buffer_range,
        )
        .ok_or_else(gap)?
        .with_order(order);
        let pass = if listed { pass.listed() } else { pass };
        let kernel = |phase| device.owned_kernel(OwnedModule::IndexRewrite(program, phase));
        let kernels = [
            kernel(RewritePhase::Count)?,
            kernel(RewritePhase::Place)?,
            kernel(RewritePhase::Write)?,
        ];
        // A direct draw's list holds no value before the pass writes it: a buffer of the
        // device's, made without a submission.
        let output = match owned {
            Some((_, bytes)) => Output::Owned(device.scratch_buffer(bytes)?),
            None => {
                let queue = core.command_buffer().queue_place().id();
                Output::Pooled(core.command_buffer().state().index_pool(&device, queue)?)
            }
        };
        let (words, status) = {
            let mut state = core.command_buffer().state();
            let words =
                state
                    .uploads
                    .upload(&device, &pass.parameter_bytes(), IndexRewritePass::BYTES)?;
            (words, state.status_words(&device)?)
        };
        let region = |upload: &Upload, len: u64| (upload.raw(), upload.offset(), len);
        let words_range = region(&words, IndexRewritePass::BYTES);
        let mut set = [(vk::Buffer::null(), 0, 0); 5];
        set[IndexRewriteModule::WORDS_BINDING as usize] = words_range;
        // A direct draw reads no arguments, and a draw with no index inside the buffer no index:
        // nothing reads the binding, which takes the words.
        set[IndexRewriteModule::ARGUMENTS_BINDING as usize] =
            match (pass.arguments_range, arguments) {
                (
                    Some(range),
                    Arguments::Indirect(indirect) | Arguments::IndirectVertices(indirect),
                ) => (indirect.raw, range.offset, range.range),
                _ => words_range,
            };
        set[IndexRewriteModule::INDICES_BINDING as usize] = match (pass.indices_range, index) {
            (Some(range), Some(index)) => (index.raw, range.offset, range.range),
            _ => words_range,
        };
        set[IndexRewriteModule::STATUS_BINDING as usize] =
            region(&status, ExecutionStatus::BYTES as u64);
        // A pooled list's count and place dispatches bind the pool's counter, its write dispatch
        // the pool's block, whose indices start where the place dispatch's first word says.
        let (output_range, write_range) = match &output {
            Output::Owned(buffer) => {
                let (raw, at) = buffer.at(0);
                ((raw, at, buffer.length()), None)
            }
            Output::Pooled(pool) => (
                region(&pool.counter, IndexRewritePass::POOL_HEADER_BYTES),
                Some((pool.block, 0, pool.bytes)),
            ),
        };
        set[IndexRewriteModule::OUTPUT_BINDING as usize] = output_range;
        if set
            .iter()
            .chain(&write_range)
            .any(|&(buffer, _, _)| buffer == vk::Buffer::null())
        {
            return Err(internal("an index rewrite pass binding without a buffer"));
        }
        // The three's set layouts are defined alike: one set for all, or one for the write.
        let sets = (
            OwnedSets::write(&device, &kernels[0], &[set.to_vec()])?,
            match write_range {
                Some(range) => {
                    set[IndexRewriteModule::OUTPUT_BINDING as usize] = range;
                    Some(OwnedSets::write(&device, &kernels[2], &[set.to_vec()])?)
                }
                None => None,
            },
        );
        let past_length = match (arguments, index) {
            (Arguments::Indirect(_), Some(index)) => {
                let (buffer, offset) = index.metal();
                (offset >= buffer.length()).then(|| (offset, buffer.length()))
            }
            // A direct draw's positions past the buffer read a padded copy, which the draw
            // announces (`render.rs`); a draw without indices reads none.
            _ => None,
        };
        Ok(RewritePass {
            words,
            output,
            kernels,
            pass,
            sets,
            uses,
            past_length,
            listed,
        })
    }

    /// The device address of the list's region, where the device has device addresses: an owned
    /// list's buffer's byte 0, a pooled one's first index's (past the pool's header), from which
    /// the place dispatch's first index counts. The private list form reads its list there.
    pub(super) fn address(&self) -> Option<u64> {
        match &self.output {
            Output::Owned(buffer) => buffer.device_address(),
            Output::Pooled(pool) => pool
                .address
                .map(|address| address + IndexRewritePass::POOL_HEADER_BYTES),
        }
    }

    /// The pass's reads of the Metal arguments and indices ([`reads`]).
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// The draw of the pass: its index binding, the `VkBuffer` of its arguments, and its command,
    /// with `sized` the device's index range binding. In the private list form, the draw of the
    /// list's positions without indices (`VkDrawIndirectCommand`).
    pub(super) fn draw(
        &self,
        sized: Option<SizedIndexBind>,
    ) -> (
        Option<IndexBind>,
        Option<vk::Buffer>,
        metal2vulkan::vk::DrawCommand,
    ) {
        let offset = self.words.offset() + IndexRewritePass::DRAW_OFFSET;
        if self.listed {
            let draw = metal2vulkan::vk::DrawCommand::DrawIndirect {
                offset,
                draw_count: 1,
                stride: 16,
            };
            return (None, Some(self.words.raw()), draw);
        }
        let (buffer, index_offset, bytes) = match &self.output {
            Output::Owned(buffer) => {
                let (raw, at) = buffer.at(0);
                (raw, at, buffer.length())
            }
            Output::Pooled(pool) => (
                pool.block,
                IndexRewritePass::POOL_HEADER_BYTES,
                pool.index_bytes(),
            ),
        };
        let index_type = metal2vulkan::vk::IndexType::UINT32;
        let draw = metal2vulkan::vk::DrawCommand::DrawIndexedIndirect {
            index_type,
            index_buffer_offset: index_offset,
            offset,
            draw_count: 1,
            stride: 20,
        };
        let bind = IndexBind {
            buffer,
            offset: index_offset,
            index_type: index_type.into(),
            sized: sized.map(|bind| (bind, bytes)),
        };
        (Some(bind), Some(self.words.raw()), draw)
    }

    /// Hold the pass's set and list in `core`'s command buffer until it completes, and announce
    /// an indirect draw from past its index buffer once; what is left records the dispatch, before
    /// the render pass or before the render pass instance the draw records into (`render.rs`).
    /// Call it once the draw records.
    pub(super) fn into_steps(self, core: &mut EncoderCore) -> RewriteSteps {
        let RewritePass {
            output,
            kernels,
            pass,
            sets,
            uses,
            past_length,
            ..
        } = self;
        let (sets, write) = sets;
        let steps = RewriteSteps {
            kernels,
            pass,
            sets: [
                sets.sets.clone(),
                sets.sets.clone(),
                write.as_ref().unwrap_or(&sets).sets.clone(),
            ],
            uses,
        };
        core.retain(sets);
        if let Some(write) = write {
            core.retain(write);
        }
        if let Output::Owned(buffer) = output {
            core.retain(buffer);
        }
        if let Some((offset, length)) = past_length {
            zeroed_fetch(core.device(), offset, length);
        }
        steps
    }
}

/// Announce an indexed indirect draw from `offset` of an index buffer of `length` bytes, at or
/// past its length: each position draws index 0 plus the base vertex.
fn zeroed_fetch(device: &Device, offset: u64, length: u64) {
    device.degrade(Degradation::ZeroedIndexFetch, || {
        (
            format!(
                "an indexed indirect draw from offset {offset} of an index buffer of {length} \
                 bytes"
            ),
            "its offset lies at or past the buffer's length, so each position draws index 0 \
             plus the base vertex"
                .to_owned(),
        )
    });
}

/// Announce, once, an indexed indirect draw of `index` whose offset lies at or past its buffer's
/// length ([`zeroed_fetch`]), which a gather run's list draws as the rewrite pass's does.
pub(super) fn announce_past_length(device: &Device, index: &DrawSource<'_>) {
    let (buffer, offset) = index.metal();
    if offset >= buffer.length() {
        zeroed_fetch(device, offset, buffer.length());
    }
}

/// The list of a direct draw without indices of one of Metal's private primitive types, or under
/// the last provoking vertex on a device without it, which the CPU writes into the command
/// buffer's upload arena while the draw is encoded: its positions are its vertices, so the list
/// follows from its count alone ([`RewriteProgram::sequential_list`] in the draw's order, relative
/// to its first vertex), and the draw draws it from its vertex start as base vertex; in the private
/// list form, the vertex start lies in the word before it, and the draw is the binder's (the
/// module docs). The arena holds it until the command buffer completes, and the host writes it
/// before the draw's batch is submitted (`upload.rs`).
pub(super) struct WrittenList {
    upload: Upload,
    indices: u32,
}

impl WrittenList {
    /// The list of `program` in `order` for a draw of `count` vertices from `vertex_start`, in
    /// `core`'s command buffer, in the private list form where `listed`. Refused
    /// (`NotImplemented(gap)`: `PrivatePrimitives`, or `ProvokingVertex`) outside that form from
    /// a vertex start past `i32::MAX`, which a list's base vertex cannot carry, and for a list of
    /// more indices than a `u32` counts; and as uploads are refused.
    pub(super) fn new(
        core: &mut EncoderCore,
        ((program, order), refused): ((RewriteProgram, RewriteOrder), RuntimeGap),
        (vertex_start, count): (u32, u32),
        listed: bool,
    ) -> Result<WrittenList, Error> {
        let gap = || Error::NotImplemented(refused.clone());
        if !listed {
            i32::try_from(vertex_start).map_err(|_| gap())?;
        }
        let indices = u32::try_from(program.capacity(u64::from(count))).map_err(|_| gap())?;
        let device = core.device().clone();
        let bytes: Vec<u8> = listed
            .then_some(vertex_start)
            .into_iter()
            .chain(program.sequential_list(count, order))
            .flat_map(u32::to_le_bytes)
            .collect();
        let len = (bytes.len() as u64).max(4);
        let upload = core
            .command_buffer()
            .state()
            .uploads
            .upload(&device, &bytes, len)?;
        Ok(WrittenList { upload, indices })
    }

    /// The device address of the list's first word (the vertex start's in the private list
    /// form), where the device has device addresses.
    pub(super) fn address(&self) -> Option<u64> {
        self.upload.device_address()
    }

    /// The draw of the list: its index binding, with `sized` the device's index range binding,
    /// and the binder's draw of the vertices `draw` (`vkCmdDraw`'s), drawn as the indexed draw of
    /// the list from its first vertex as base vertex, with its instances. `None` for another
    /// command.
    pub(super) fn draw(
        &self,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Option<(IndexBind, metal2vulkan::vk::DrawCommand)> {
        let metal2vulkan::vk::DrawCommand::Draw {
            instance_count,
            first_vertex,
            first_instance,
            ..
        } = draw
        else {
            return None;
        };
        let index_type = metal2vulkan::vk::IndexType::UINT32;
        let offset = self.upload.offset();
        let draw = metal2vulkan::vk::DrawCommand::DrawIndexed {
            index_type,
            index_buffer_offset: offset,
            index_count: self.indices,
            instance_count,
            first_index: 0,
            vertex_offset: i32::try_from(first_vertex).ok()?,
            first_instance,
        };
        let bind = IndexBind {
            buffer: self.upload.raw(),
            offset,
            index_type: index_type.into(),
            sized: sized.map(|bind| (bind, (u64::from(self.indices) * 4).max(4))),
        };
        Some((bind, draw))
    }
}

/// The pass's reads of the Metal buffers: an indirect draw's, as the index bounds pass's
/// (`index_bounds::reads`); a direct draw's indices from its offset through its count, or to
/// where Metal's fetch stops; an indirect draw's arguments alone where it has no indices; in the
/// compute stage.
pub(super) fn reads(
    index: Option<&DrawSource<'_>>,
    arguments: &Arguments<'_>,
    index_size: u64,
) -> Vec<Use<ResourceKey>> {
    let read = Access::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    let (index, count) = match (*arguments, index) {
        (Arguments::Indirect(indirect), Some(index)) => {
            return index_bounds::reads(index, indirect, index_size)
        }
        (Arguments::IndirectVertices(indirect), _) => {
            let (buffer, offset) = indirect.metal();
            let (identity, range) = buffer.tracked(offset..offset + 16);
            return vec![Use::reads(
                identity.key,
                identity.tracked,
                UseRange::Bytes(range),
                read,
            )];
        }
        (Arguments::Direct { count, .. }, Some(index)) => (index, u64::from(count)),
        _ => return Vec::new(),
    };
    let (indices, offset) = index.metal();
    let end = IndexBufferRef::new(0, indices.length()).fetch_end(offset);
    let whole = end.saturating_sub(offset) / index_size * index_size;
    let bytes = whole.min(count * index_size);
    if bytes == 0 {
        return Vec::new();
    }
    let (identity, range) = indices.tracked(offset..offset + bytes);
    vec![Use::reads(
        identity.key,
        identity.tracked,
        UseRange::Bytes(range),
        read,
    )]
}

/// Of [`reads`], those of the bytes the pass reads in place, which a write since the encoder's
/// last memory barrier races: an indirect draw's as the index bounds pass's
/// (`index_bounds::in_place`), its arguments alone where it has no indices unless they were
/// copied for a race; a direct draw's indices all (it copies none for a race).
pub(super) fn in_place(
    index: Option<&DrawSource<'_>>,
    arguments: &Arguments<'_>,
    index_size: u64,
) -> Vec<Use<ResourceKey>> {
    match (*arguments, index) {
        (Arguments::Indirect(indirect), Some(index)) => {
            index_bounds::in_place(index, indirect, index_size)
        }
        (Arguments::IndirectVertices(indirect), _) if indirect.racing() => Vec::new(),
        _ => reads(index, arguments, index_size),
    }
}

/// A planned pass's dispatch, its set held by the command buffer ([`RewritePass::into_steps`]).
pub(super) struct RewriteSteps {
    /// The kernels of its three dispatches, in order.
    kernels: [OwnedKernel; 3],
    /// The pass the kernels run: its workgroups.
    pass: IndexRewritePass,
    /// Each dispatch's sets, in order.
    sets: [Vec<vk::DescriptorSet>; 3],
    /// The pass's reads of the Metal arguments and indices.
    uses: Vec<Use<ResourceKey>>,
}

impl RewriteSteps {
    /// The pass's reads of the Metal arguments and indices.
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Record the dispatches into `core`'s command buffer before the pass, between a barrier from
    /// everything before and one to everything after.
    pub(super) fn record(&self, core: &mut EncoderCore) {
        core.barrier(Barrier::everything());
        core.command(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.uses,
            |raw, command_buffer| {
                // SAFETY: the encoder's command buffer is recording outside a render pass.
                unsafe { self.dispatch(raw, command_buffer) }
            },
        );
        core.barrier(Barrier::everything());
    }

    /// Record the dispatches into `command_buffer` between a barrier from everything before and one
    /// to everything after: before a render pass instance of the encoder's pass, after the
    /// instances before it (`render.rs`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is the encoder's, recording outside a render pass, and only the encoder
    /// records it.
    pub(super) unsafe fn record_in(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states; barriers need nothing more.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            self.dispatch(raw, command_buffer);
            Barrier::everything().record(raw, command_buffer);
        }
    }

    /// The pass's three dispatches, in order, each after the one before's writes
    /// (`Barrier::between_dispatches`): the count and write dispatches over the pass's groups,
    /// the place dispatch over one workgroup.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only one encoder records it.
    unsafe fn dispatch(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let phases = self.kernels.iter().zip(RewritePhase::ALL).zip(&self.sets);
        for (at, ((kernel, phase), sets)) in phases.enumerate() {
            // SAFETY: as the caller states; each kernel is the device's, its sets (whose layout
            // each kernel's is defined as) are live until the command buffer completes (retained
            // by `into_steps`), and so are the uploads, the list and the Metal buffers they name.
            unsafe {
                if at > 0 {
                    Barrier::between_dispatches().record(raw, command_buffer);
                }
                raw.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.pipeline,
                );
                raw.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.layout,
                    0,
                    sets,
                    &[],
                );
                raw.cmd_dispatch(command_buffer, phase.workgroups(&self.pass), 1, 1);
            }
        }
    }
}

/// A list a draw draws in place of its Metal primitives, which the CPU or a run of the index gather
/// pass wrote (the index rewrite pass's draws apart, `RewritePass::draw`).
pub(super) enum Listed {
    /// A direct draw's without indices, the CPU's ([`WrittenList`]).
    Written(WrittenList),
    /// A direct indexed draw's, a gather run's ([`GatheredList`]).
    Gathered(GatheredList),
    /// An indirect draw's, an indirect gather run's in the index pool, or a direct indexed draw's,
    /// a rewrite run's in its output, each drawn from the command the GPU wrote
    /// ([`CommandedList`]).
    Commanded(CommandedList),
}

impl Listed {
    /// The draw of the list: its index binding, with `sized` the device's index range binding,
    /// the `VkBuffer` of its arguments where they are the GPU's, and the indexed draw of it the
    /// binder's draw `draw` becomes. `None` for a command of another kind than the list's draw
    /// takes.
    pub(super) fn draw(
        &self,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Option<(IndexBind, Option<vk::Buffer>, metal2vulkan::vk::DrawCommand)> {
        match self {
            Listed::Written(list) => list.draw(draw, sized).map(|(bind, d)| (bind, None, d)),
            Listed::Gathered(list) => list.draw(draw, sized).map(|(bind, d)| (bind, None, d)),
            Listed::Commanded(list) => list.draw(draw, sized),
        }
    }

    /// The device address of a list the CPU wrote, where the device has device addresses: the
    /// private list form's (a gather run's list takes no part in it, `render.rs`).
    pub(super) fn address(&self) -> Option<u64> {
        match self {
            Listed::Written(list) => list.address(),
            Listed::Gathered(_) | Listed::Commanded(_) => None,
        }
    }
}

/// The output words of a run's own buffer, where its first draw's list is shorter.
const RUN_OUTPUT_WORDS: u32 = 1 << 14;

/// The most draws of a run: its table's records, which fit an ordinary chunk of the upload arena
/// (`upload.rs`), and so do an indirect run's draws' commands.
const RUN_DRAWS: u32 = 480;

/// The most draws of a rewrite run: its words, of [`IndexRewriteRun::SLOT_BYTES`] each, fit half
/// an ordinary chunk of the upload arena (`upload.rs`).
const REWRITE_RUN_DRAWS: u32 = 224;

/// The most runs an encoder keeps open before it records them.
const OPEN_RUNS: usize = 8;

/// A direct indexed draw whose list the index gather pass writes (`bind::index_gather_module`):
/// its program and order, its Metal index source, Metal's count, and its index size.
pub(super) struct GatherRequest<'a> {
    pub(super) program: RewriteProgram,
    pub(super) order: RewriteOrder,
    pub(super) index: &'a DrawSource<'a>,
    pub(super) count: u32,
    pub(super) wide: bool,
}

/// An indirect draw whose list an indirect run of the index gather pass writes, planned from
/// Metal's arguments on the GPU (`bind::index_gather_plan_module`): its program and order, its
/// Metal arguments, its Metal index source (`None` without indices), and its index size.
pub(super) struct IndirectGatherRequest<'a> {
    pub(super) program: RewriteProgram,
    pub(super) order: RewriteOrder,
    pub(super) arguments: &'a DrawSource<'a>,
    pub(super) index: Option<&'a DrawSource<'a>>,
    pub(super) wide: bool,
}

/// A direct indexed draw whose list a run of the index rewrite pass writes
/// (`bind::index_rewrite_run_module`): its program, `k'` and order, its Metal index source, Metal's
/// count, instances, base vertex and base instance, and its index size.
pub(super) struct RewriteRunRequest<'a> {
    pub(super) program: RewriteProgram,
    pub(super) restart: u32,
    pub(super) order: RewriteOrder,
    pub(super) index: &'a DrawSource<'a>,
    pub(super) arguments: (u32, u32, i32, u32),
    pub(super) wide: bool,
}

/// The runs of an encoder's gathered draws (the module docs, `index_rewrite.rs`'s gather) and of
/// its bounded ones (`index_bounds.rs`): each one dispatch (an indirect run's and a bounds run's
/// after its plan), recorded before the encoder's next command, between a barrier from
/// everything before and one to everything after for all of them.
#[derive(Default)]
pub(super) struct GatherRuns {
    runs: Vec<GatherRun>,
    rewrites: Vec<RewriteRun>,
    bounds: Vec<BoundsRun>,
}

/// The open runs an encoder records now ([`record_runs`]).
pub(super) struct TakenRuns {
    gathers: Vec<GatherRun>,
    rewrites: Vec<RewriteRun>,
    bounds: Vec<BoundsRun>,
}

/// What a rewrite run's draws share: the index `VkBuffer` and the bytes it binds of it, the
/// program, `k'` and the order.
type RewriteKey = ((vk::Buffer, u64, u64), RewriteProgram, u32, RewriteOrder);

/// One run of the index rewrite pass: its draws, what they share, its words' region, its kernel
/// and set, and its output's `VkBuffer`, offset and bytes.
pub(super) struct RewriteRun {
    run: IndexRewriteRun,
    key: RewriteKey,
    words: Upload,
    kernel: OwnedKernel,
    sets: Vec<vk::DescriptorSet>,
    output: (vk::Buffer, u64, u64),
}

/// What a run's draws share: a direct run's index `VkBuffer` and the bytes it binds of it; an
/// indirect run's arguments `VkBuffer` and bytes, and index `VkBuffer` and bytes where it has
/// indices; and the program and order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RunKey {
    Direct((vk::Buffer, u64, u64), RewriteProgram, RewriteOrder),
    Indirect(
        (vk::Buffer, u64, u64),
        Option<(vk::Buffer, u64, u64)>,
        RewriteProgram,
        RewriteOrder,
    ),
}

/// One run: its draws, what they share, its table's region, its kernel and set, its output, and
/// an indirect run's plan.
pub(super) struct GatherRun {
    gather: IndexGather,
    key: RunKey,
    table: Upload,
    kernel: OwnedKernel,
    sets: Vec<vk::DescriptorSet>,
    output: (vk::Buffer, u64),
    plan: Option<RunPlan>,
}

/// An indirect run's plan: its kernel and set, and the region of its draws' commands.
struct RunPlan {
    kernel: OwnedKernel,
    sets: Vec<vk::DescriptorSet>,
    commands: Upload,
}

/// Where a gathered draw's list lies: its `VkBuffer`, its offset there, and its indices.
pub(super) struct GatheredList {
    buffer: vk::Buffer,
    offset: u64,
    indices: u32,
}

/// Where a list the GPU placed and its command lie: the `VkBuffer` of its indices, the offset
/// there the command's first index counts from and the bytes from it (an indirect gathered
/// draw's: the index pool's block past its header; a rewrite run's draw's: the run's output), and
/// the command's `VkBuffer` and offset there.
pub(super) struct CommandedList {
    indices: (vk::Buffer, u64, u64),
    command: (vk::Buffer, u64),
}

/// What a new run's command buffer holds until it completes: a direct run's output, and its
/// sets.
pub(super) struct GatherHeld {
    pub(super) output: Option<Buffer>,
    pub(super) sets: Vec<OwnedSets>,
}

impl GatherRuns {
    /// Whether no run is open.
    pub(super) fn is_empty(&self) -> bool {
        self.runs.is_empty() && self.rewrites.is_empty() && self.bounds.is_empty()
    }

    /// Whether the runs are as many as an encoder keeps open: they record before another opens.
    pub(super) fn full(&self) -> bool {
        self.runs.len() + self.rewrites.len() + self.bounds.len() >= OPEN_RUNS
    }

    /// The open runs, which record now ([`record_runs`]).
    pub(super) fn take(&mut self) -> TakenRuns {
        TakenRuns {
            gathers: std::mem::take(&mut self.runs),
            rewrites: std::mem::take(&mut self.rewrites),
            bounds: std::mem::take(&mut self.bounds),
        }
    }

    /// `request`'s draw joins an open run of its index buffer, program and order that takes it,
    /// or a new one, with its table's region in `uploads`: the list's place, and what a new run's
    /// command buffer holds. `None` where no run takes the draw (its list is past what a run's
    /// dispatch or bindings take), which the index rewrite pass then writes. Refused as uploads,
    /// buffers, kernels and descriptor sets are refused.
    pub(super) fn join(
        &mut self,
        device: &Device,
        uploads: &mut Uploads,
        request: GatherRequest<'_>,
    ) -> Result<Option<(GatheredList, Option<GatherHeld>)>, Error> {
        let GatherRequest {
            program,
            order,
            index,
            count,
            wide,
        } = request;
        // The run binds the Metal buffer's bytes, from its start, through where Metal's fetch
        // stops (its length rounded up to 4, `IndexBufferRef::fetch_end`).
        let start = index.at - index.metal().1;
        let draw = GatherDraw {
            count,
            index_offset: index.at,
            index_end: index.at + index.bound(index.at),
            wide,
        };
        let key = RunKey::Direct((index.raw, start, index.length), program, order);
        let joined = self
            .runs
            .iter_mut()
            .filter(|run| run.key == key && run.gather.len() < RUN_DRAWS as usize)
            .find_map(|run| {
                let first = run.gather.push(draw)?;
                Some(run.list(first, &draw))
            });
        if let Some(list) = joined {
            return Ok(Some((list, None)));
        }
        // A run's output holds its first draw's list, at least the default.
        let Ok(capacity) = u32::try_from(program.capacity(u64::from(count))) else {
            return Ok(None);
        };
        let words = capacity.max(RUN_OUTPUT_WORDS);
        let limits = &device.properties().description.limits;
        let Some(mut gather) = IndexGather::new(
            (program, order),
            (start, index.length),
            (
                limits.min_storage_buffer_offset_alignment,
                limits.max_storage_buffer_range,
            ),
            words,
        ) else {
            return Ok(None);
        };
        let Some(first) = gather.push(draw) else {
            return Ok(None);
        };
        let output_bytes = u64::from(words) * 4;
        if output_bytes > u64::from(limits.max_storage_buffer_range) {
            return Ok(None);
        }
        let (table, pattern) = run_uploads(device, uploads, &gather)?;
        let output = device.scratch_buffer(output_bytes)?;
        let kernel = device.owned_kernel(OwnedModule::IndexGather)?;
        let (output_raw, output_at) = output.at(0);
        let sets = gather_sets(
            device,
            &kernel,
            (&table, &pattern),
            gather.indices_range().map(|range| (index.raw, range)),
            (output_raw, output_at, output_bytes),
        )?;
        let run = GatherRun {
            gather,
            key,
            table,
            kernel,
            sets: sets.sets.clone(),
            output: (output_raw, output_at),
            plan: None,
        };
        let list = run.list(first, &draw);
        self.runs.push(run);
        Ok(Some((
            list,
            Some(GatherHeld {
                output: Some(output),
                sets: vec![sets],
            }),
        )))
    }

    /// `request`'s indirect draw joins an open indirect run of its arguments and index buffers,
    /// program and order that takes it, or a new one into `pool`, with its regions in `uploads`
    /// and `status` the command buffer's status words: where its list and command lie, and what
    /// a new run's command buffer holds. `None` where no run takes the draw (its arguments or
    /// indices past what a run's bindings take), which the index rewrite pass then writes.
    /// Refused as uploads, kernels and descriptor sets are refused.
    pub(super) fn join_indirect(
        &mut self,
        device: &Device,
        (uploads, status, pool): (&mut Uploads, &Upload, &IndexPool),
        request: IndirectGatherRequest<'_>,
    ) -> Result<Option<(CommandedList, Option<GatherHeld>)>, Error> {
        let IndirectGatherRequest {
            program,
            order,
            arguments,
            index,
            wide,
        } = request;
        // The run binds each Metal buffer's bytes, from its start, through its length (an index
        // buffer's rounded up to 4, `IndexBufferRef::fetch_end`).
        let bytes =
            |source: &DrawSource<'_>| (source.raw, source.at - source.metal().1, source.length);
        let key = RunKey::Indirect(bytes(arguments), index.map(bytes), program, order);
        let draw = IndirectGatherDraw {
            arguments_offset: arguments.at,
            index_offset: index.map_or(0, |index| index.at),
            index_end: index.map_or(0, |index| index.at + index.bound(index.at)),
            wide,
        };
        let joined = self
            .runs
            .iter_mut()
            .filter(|run| run.key == key && run.gather.len() < RUN_DRAWS as usize)
            .find_map(|run| {
                let place = run.gather.push_indirect(draw)?;
                run.pooled(place, pool)
            });
        if let Some(list) = joined {
            return Ok(Some((list, None)));
        }
        let limits = &device.properties().description.limits;
        let Ok(pool_bytes) = u32::try_from(pool.index_bytes()) else {
            return Ok(None);
        };
        let (_, arguments_start, arguments_end) = bytes(arguments);
        let Some(mut gather) = IndexGather::indirect(
            (program, order),
            (arguments_start, arguments_end),
            index.map(|index| (bytes(index).1, index.length)),
            (
                limits.min_storage_buffer_offset_alignment,
                limits.max_storage_buffer_range,
            ),
            pool_bytes,
        ) else {
            return Ok(None);
        };
        let Some(place) = gather.push_indirect(draw) else {
            return Ok(None);
        };
        let Some(arguments_range) = gather.arguments_range() else {
            return Err(internal("an indirect gather run without its arguments"));
        };
        let (table, pattern) = run_uploads(device, uploads, &gather)?;
        let commands = uploads.upload(
            device,
            &[],
            u64::from(RUN_DRAWS) * IndexGatherPlanModule::DRAW_BYTES,
        )?;
        let kernel = device.owned_kernel(OwnedModule::IndexGather)?;
        let sets = gather_sets(
            device,
            &kernel,
            (&table, &pattern),
            match (index, gather.indices_range()) {
                (Some(index), Some(range)) => Some((index.raw, range)),
                _ => None,
            },
            (pool.block, 0, pool.bytes),
        )?;
        let plan_kernel = device.owned_kernel(OwnedModule::IndexGatherPlan)?;
        let region = |upload: &Upload| (upload.raw(), upload.offset(), upload.len());
        let mut set = [(vk::Buffer::null(), 0, 0); 5];
        set[IndexGatherPlanModule::TABLE_BINDING as usize] = region(&table);
        set[IndexGatherPlanModule::ARGUMENTS_BINDING as usize] =
            (arguments.raw, arguments_range.offset, arguments_range.range);
        set[IndexGatherPlanModule::POOL_BINDING as usize] = (
            pool.counter.raw(),
            pool.counter.offset(),
            IndexRewritePass::POOL_HEADER_BYTES,
        );
        set[IndexGatherPlanModule::STATUS_BINDING as usize] =
            (status.raw(), status.offset(), ExecutionStatus::BYTES as u64);
        set[IndexGatherPlanModule::DRAWS_BINDING as usize] = region(&commands);
        let plan_sets = OwnedSets::write(device, &plan_kernel, &[set.to_vec()])?;
        let run = GatherRun {
            gather,
            key,
            table,
            kernel,
            sets: sets.sets.clone(),
            output: (pool.block, 0),
            plan: Some(RunPlan {
                kernel: plan_kernel,
                sets: plan_sets.sets.clone(),
                commands,
            }),
        };
        let list = run
            .pooled(place, pool)
            .ok_or_else(|| internal("an indirect gather run without its plan"))?;
        self.runs.push(run);
        Ok(Some((
            list,
            Some(GatherHeld {
                output: None,
                sets: vec![sets, plan_sets],
            }),
        )))
    }
}

impl GatherRuns {
    /// `request`'s direct indexed draw joins an open rewrite run of its index buffer, program,
    /// `k'` and order that takes it, or a new one, with its words' region in `uploads` and
    /// `status` the command buffer's status words: where its list and command lie, and what a new
    /// run's command buffer holds. `None` where no run takes the draw (a walk of more than one
    /// group, a list past what a run's output binds or a `u32` counts, or a base instance the
    /// device's indirect draws cannot carry), which the draw's own pass then writes. Refused as
    /// uploads, buffers, kernels and descriptor sets are refused.
    pub(super) fn join_rewrite(
        &mut self,
        device: &Device,
        (uploads, status): (&mut Uploads, &Upload),
        request: RewriteRunRequest<'_>,
    ) -> Result<Option<(CommandedList, Option<GatherHeld>)>, Error> {
        let RewriteRunRequest {
            program,
            restart,
            order,
            index,
            arguments: (count, instances, base_vertex, base_instance),
            wide,
        } = request;
        if base_instance != 0
            && !device
                .enabled()
                .features
                .contains(&VkFeature::DrawIndirectFirstInstance)
        {
            return Ok(None);
        }
        // The run binds the Metal buffer's bytes, from its start, through where Metal's fetch
        // stops (its length rounded up to 4, `IndexBufferRef::fetch_end`).
        let start = index.at - index.metal().1;
        let draw = RewriteRunDraw {
            count,
            instances,
            base_vertex,
            base_instance,
            index_offset: index.at,
            index_end: index.at + index.bound(index.at),
            wide,
        };
        let key = ((index.raw, start, index.length), program, restart, order);
        let joined = self
            .rewrites
            .iter_mut()
            .filter(|run| run.key == key && run.run.len() < REWRITE_RUN_DRAWS as usize)
            .find_map(|run| {
                let slot = run.run.push(draw)?;
                Some(run.list(slot))
            });
        if let Some(list) = joined {
            return Ok(Some((list, None)));
        }
        // A run's output holds its first draw's list, at least the default.
        let Ok(capacity) = u32::try_from(program.capacity(u64::from(count))) else {
            return Ok(None);
        };
        let words = capacity.max(RUN_OUTPUT_WORDS);
        let output_bytes = u64::from(words) * 4;
        let limits = &device.properties().description.limits;
        if output_bytes > u64::from(limits.max_storage_buffer_range) {
            return Ok(None);
        }
        let Some(mut run) = IndexRewriteRun::new(
            (program, order, restart),
            (start, index.length),
            (
                limits.min_storage_buffer_offset_alignment,
                limits.max_storage_buffer_range,
            ),
            words,
        ) else {
            return Ok(None);
        };
        let Some(slot) = run.push(draw) else {
            return Ok(None);
        };
        let words_upload = uploads.upload(
            device,
            &[],
            u64::from(REWRITE_RUN_DRAWS) * IndexRewriteRun::SLOT_BYTES,
        )?;
        let output = device.scratch_buffer(output_bytes)?;
        let kernel = device.owned_kernel(OwnedModule::IndexRewriteRun(program))?;
        let region = |upload: &Upload, len: u64| (upload.raw(), upload.offset(), len);
        let words_range = region(&words_upload, words_upload.len());
        let (output_raw, output_at) = output.at(0);
        let mut set = [(vk::Buffer::null(), 0, 0); 5];
        set[IndexRewriteModule::WORDS_BINDING as usize] = words_range;
        // Every draw is direct, and a run whose draws read no index reads none: nothing reads
        // the binding there, which takes the words.
        set[IndexRewriteModule::ARGUMENTS_BINDING as usize] = words_range;
        set[IndexRewriteModule::INDICES_BINDING as usize] = match run.indices_range() {
            Some(range) => (index.raw, range.offset, range.range),
            None => words_range,
        };
        set[IndexRewriteModule::OUTPUT_BINDING as usize] = (output_raw, output_at, output_bytes);
        set[IndexRewriteModule::STATUS_BINDING as usize] =
            region(status, ExecutionStatus::BYTES as u64);
        let sets = OwnedSets::write(device, &kernel, &[set.to_vec()])?;
        let run = RewriteRun {
            run,
            key,
            words: words_upload,
            kernel,
            sets: sets.sets.clone(),
            output: (output_raw, output_at, output_bytes),
        };
        let list = run.list(slot);
        self.rewrites.push(run);
        Ok(Some((
            list,
            Some(GatherHeld {
                output: Some(output),
                sets: vec![sets],
            }),
        )))
    }
}

impl GatherRuns {
    /// `request`'s indexed indirect draw joins an open run of the index bounds pass over its
    /// arguments and index buffers that takes it, or a new one (`index_bounds::join`): its slot,
    /// and what a new run's command buffer holds; `None` where no run takes it.
    pub(super) fn join_bounds(
        &mut self,
        device: &Device,
        regions: (&mut Uploads, &Upload, &IndexPool),
        request: BoundsRunRequest<'_>,
    ) -> Result<Option<(BoundsSlot, Option<GatherHeld>)>, Error> {
        index_bounds::join(&mut self.bounds, device, regions, request)
    }
}

impl RewriteRun {
    /// Where the run's draw of slot `slot` finds its list (the run's output, from its start, the
    /// command's first index its region's) and its command.
    fn list(&self, slot: u32) -> CommandedList {
        let (raw, at, bytes) = self.output;
        CommandedList {
            indices: (raw, at, bytes),
            command: (
                self.words.raw(),
                self.words.offset() + IndexRewriteRun::draw_offset(slot),
            ),
        }
    }
}

/// A new run's table region (of [`RUN_DRAWS`] records) and pattern, in `uploads`.
fn run_uploads(
    device: &Device,
    uploads: &mut Uploads,
    gather: &IndexGather,
) -> Result<(Upload, Upload), Error> {
    let pattern: Vec<u8> = gather
        .pattern()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let table_bytes =
        u64::from(IndexGatherModule::HEADER_WORDS + IndexGatherModule::RECORD_WORDS * RUN_DRAWS)
            * 4;
    let table = uploads.upload(device, &[], table_bytes)?;
    let pattern = uploads.upload(device, &pattern, pattern.len() as u64)?;
    Ok((table, pattern))
}

/// A run's gather set: its table and pattern, its indices (a `VkBuffer` and its bound range; the
/// table where it reads none), and its output.
fn gather_sets(
    device: &Device,
    kernel: &OwnedKernel,
    (table, pattern): (&Upload, &Upload),
    indices: Option<(vk::Buffer, metal2vulkan::bind::BufferRange)>,
    output: (vk::Buffer, u64, u64),
) -> Result<OwnedSets, Error> {
    let region = |upload: &Upload| (upload.raw(), upload.offset(), upload.len());
    let mut set = [(vk::Buffer::null(), 0, 0); 4];
    set[IndexGatherModule::TABLE_BINDING as usize] = region(table);
    set[IndexGatherModule::PATTERN_BINDING as usize] = region(pattern);
    // A run whose draws read no index binds any 4 bytes there.
    set[IndexGatherModule::INDICES_BINDING as usize] = match indices {
        Some((raw, range)) => (raw, range.offset, range.range),
        None => region(table),
    };
    set[IndexGatherModule::OUTPUT_BINDING as usize] = output;
    OwnedSets::write(device, kernel, &[set.to_vec()])
}

impl GatherRun {
    /// The place of the run's list whose first word is `first`, of `draw`.
    fn list(&self, first: u32, draw: &GatherDraw) -> GatheredList {
        let (buffer, at) = self.output;
        GatheredList {
            buffer,
            offset: at + u64::from(first) * 4,
            indices: self.gather.capacity(*draw).unwrap_or(0),
        }
    }

    /// Where an indirect run's draw at `place` finds its list in `pool` and its command; `None`
    /// for a direct run.
    fn pooled(&self, place: u32, pool: &IndexPool) -> Option<CommandedList> {
        let plan = self.plan.as_ref()?;
        Some(CommandedList {
            indices: (
                pool.block,
                IndexRewritePass::POOL_HEADER_BYTES,
                pool.index_bytes(),
            ),
            command: (
                plan.commands.raw(),
                plan.commands.offset() + u64::from(place) * IndexGatherPlanModule::DRAW_BYTES,
            ),
        })
    }
}

impl GatheredList {
    /// The draw of the list: its index binding, with `sized` the device's index range binding,
    /// and the binder's draw of the indices `draw` (`vkCmdDrawIndexed`'s), drawn as the indexed
    /// draw of the list from its first index, with its instances and bases. `None` for another
    /// command.
    pub(super) fn draw(
        &self,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Option<(IndexBind, metal2vulkan::vk::DrawCommand)> {
        let metal2vulkan::vk::DrawCommand::DrawIndexed {
            instance_count,
            vertex_offset,
            first_instance,
            ..
        } = draw
        else {
            return None;
        };
        let index_type = metal2vulkan::vk::IndexType::UINT32;
        let draw = metal2vulkan::vk::DrawCommand::DrawIndexed {
            index_type,
            index_buffer_offset: self.offset,
            index_count: self.indices,
            instance_count,
            first_index: 0,
            vertex_offset,
            first_instance,
        };
        let bind = IndexBind {
            buffer: self.buffer,
            offset: self.offset,
            index_type: index_type.into(),
            sized: sized.map(|bind| (bind, (u64::from(self.indices) * 4).max(4))),
        };
        Some((bind, draw))
    }
}

impl CommandedList {
    /// The draw of the list: its indices bound from their offset, with `sized` the device's index
    /// range binding, the `VkBuffer` of its command, and the binder's draw `draw` (an indirect
    /// one, `vkCmdDrawIndirect`'s or `vkCmdDrawIndexedIndirect`'s, or a rewrite run's direct
    /// `vkCmdDrawIndexed`'s), drawn as the one indexed indirect draw of the command, which
    /// carries Metal's arguments. `None` for another command.
    pub(super) fn draw(
        &self,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Option<(IndexBind, Option<vk::Buffer>, metal2vulkan::vk::DrawCommand)> {
        if !matches!(
            draw,
            metal2vulkan::vk::DrawCommand::DrawIndirect { .. }
                | metal2vulkan::vk::DrawCommand::DrawIndexedIndirect { .. }
                | metal2vulkan::vk::DrawCommand::DrawIndexed { .. }
        ) {
            return None;
        }
        let index_type = metal2vulkan::vk::IndexType::UINT32;
        let (block, offset, index_bytes) = self.indices;
        let (raw, at) = self.command;
        let draw = metal2vulkan::vk::DrawCommand::DrawIndexedIndirect {
            index_type,
            index_buffer_offset: offset,
            offset: at,
            draw_count: 1,
            stride: IndexGatherPlanModule::DRAW_BYTES as u32,
        };
        let bind = IndexBind {
            buffer: block,
            offset,
            index_type: index_type.into(),
            sized: sized.map(|bind| (bind, index_bytes)),
        };
        Some((bind, Some(raw), draw))
    }
}

/// Record `runs` into `command_buffer`: each run's table or words written into its region, then
/// between a barrier from everything before and one to everything after, each rewrite run's
/// dispatch, each bounds run's and indirect run's plan (one after another: each reads and writes
/// the index pool's counter), and after the plans' writes each bounds run's fill and each gather
/// run's dispatch (an indirect run's and a bounds run's from its plan's).
///
/// # Safety
///
/// `command_buffer` is recording outside a render pass, and only one encoder records it; each
/// run's kernels are the device's and its sets and output are held until the command buffer
/// completes ([`GatherHeld`]), and its table's, words' and commands' regions are the command
/// buffer's.
pub(super) unsafe fn record_runs(raw: &Vk, command_buffer: vk::CommandBuffer, taken: &TakenRuns) {
    let runs = &taken.gathers;
    for run in runs {
        run.table.write(&run.gather.table_bytes());
    }
    for run in &taken.rewrites {
        run.words.write(&run.run.words());
    }
    for run in &taken.bounds {
        run.write();
    }
    let bind = |kernel: &OwnedKernel, sets: &[vk::DescriptorSet]| {
        // SAFETY: as the caller states.
        unsafe {
            raw.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline,
            );
            raw.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                kernel.layout,
                0,
                sets,
                &[],
            );
        }
    };
    // SAFETY: as the caller states.
    unsafe {
        Barrier::everything().record(raw, command_buffer);
        for run in &taken.rewrites {
            bind(&run.kernel, &run.sets);
            raw.cmd_dispatch(command_buffer, run.run.workgroups(), 1, 1);
        }
        let plans = runs.iter().filter_map(|run| run.plan.as_ref());
        let bounds = taken.bounds.len();
        for (at, run) in taken.bounds.iter().enumerate() {
            if at > 0 {
                Barrier::between_dispatches().record(raw, command_buffer);
            }
            run.record_plan(raw, command_buffer);
        }
        for (at, plan) in plans.enumerate() {
            if bounds + at > 0 {
                Barrier::between_dispatches().record(raw, command_buffer);
            }
            bind(&plan.kernel, &plan.sets);
            raw.cmd_dispatch(command_buffer, 1, 1, 1);
        }
        if bounds > 0 || runs.iter().any(|run| run.plan.is_some()) {
            Barrier::between_dispatches().record(raw, command_buffer);
        }
        for run in &taken.bounds {
            run.record_fill(raw, command_buffer);
        }
        for run in runs {
            bind(&run.kernel, &run.sets);
            match run.plan {
                Some(_) => raw.cmd_dispatch_indirect(
                    command_buffer,
                    run.table.raw(),
                    run.table.offset() + IndexGather::DISPATCH_OFFSET,
                ),
                None => raw.cmd_dispatch(command_buffer, run.gather.groups(), 1, 1),
            }
        }
        Barrier::everything().record(raw, command_buffer);
    }
}
