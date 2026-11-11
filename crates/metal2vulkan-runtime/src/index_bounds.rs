//! The index bounds pass of an indexed indirect draw whose index fetch nothing bounds (the modules
//! and their contract are `bind::index_bounds`).
//!
//! Without `robustBufferAccess2` every index a Vulkan draw fetches must lie inside the bound
//! range, and an index offset at or past the buffer's length cannot be bound, so the binding plan
//! asks for such a draw's arguments on the CPU (`DrawCommands::indirect_needs_arguments`). Under
//! `robustBufferAccess2` the plan leaves the fetch to the device's bound range, past which Vulkan
//! states no index. Where that bound is the only reason for the CPU read
//! (`DrawCommands::index_bounds_pass`), and where the plan leaves the fetch to a device whose
//! driver is not measured to read zero past the range (`DriverFact::RobustIndexFetchReadsZero`,
//! `device/driver_facts.rs`), the render encoder keeps the draw on the GPU through this pass:
//!
//! - **Before the pass** (hoisted, as a draw's copies are, `draw_source.rs`): one invocation of
//!   the plan module reads Metal's arguments and writes two draws' arguments into the draw's
//!   bounds words (an upload of [`IndexBoundsPass::BYTES`]), reserving the second's indices in
//!   the command buffer's index pool; then the fill module, dispatched indirectly with the
//!   plan's group count, writes the reservation: each position's index inside the buffer, else
//!   0. A barrier from everything before, one between the two, and one to everything after.
//! - **In the pass**, two indirect draws where Metal has one: the first over the Metal index
//!   buffer as the draw binds it, the second over the pool. Exactly one draws the primitives.
//! - **The pool** (decision (a) made it 1 MiB on every device): its counter is a region of the
//!   command buffer's upload arena, zero, so every plan of the command buffer reserves after the
//!   ones before; its indices lie in its queue's block, which the command buffer leases from the
//!   device, a sixty-fourth of the device's largest device-local heap within one binding
//!   (`gpu_pools.rs`).
//!   The plan binds the counter as the pool, the fill the block (whose header bytes nothing reads),
//!   and the second draw the block's indices. A draw it cannot hold draws nothing and sets
//!   `INDEX_POOL_OVERFLOW`, and the command buffer fails with
//!   `CommandBufferError::IndexPoolOverflow` (decisions (b) and (c)): the GPU learns the draw's
//!   count after the draw was recorded against the block.
//! - **The hazard** (which replaced decision (e)): the pass runs before the pass's instances, so
//!   where an earlier draw or tile dispatch of the encoder writes the draw's arguments or indices
//!   (a tracked resource's overlapping range) behind a memory barrier, the render encoder records
//!   it before the draw's own render pass instance instead, after the writer's: Metal's draw reads
//!   the write there (probes, M3 and M4). A write since the last barrier, which Metal does not
//!   order before the draw's fetch, is refused, `NotImplemented(IndirectDrawArguments)` as before;
//!   so is any earlier write where an execution's command draws.
//! - **Runs** (`bind::IndexBoundsRun`): a draw that reads its arguments and indices in place,
//!   where nothing of the encoder wrote them behind a memory barrier and no execution replays it,
//!   joins an open run of draws over its arguments and index buffers in place of its own pass,
//!   which records before the encoder's next command with the encoder's other runs
//!   (`index_rewrite.rs`'s `record_runs`): one plan dispatch of one invocation for the run's
//!   draws, then one fill dispatch for all of them, where each draw's own pass takes two
//!   dispatches between three barriers. Each draw's two draws, its reservation and the status are
//!   the ones its own pass leaves (the module's tests), and its two draws read their commands from
//!   its slot of the run's words.
//! - **Announced** once per device as `indirect-index-bounds-pass` (decision (d)); a draw from
//!   an index offset at or past the buffer's length, whose positions draw index 0 where Metal
//!   reads the memory past the buffer (the probe-ix), also as `zeroed-index-fetch`.

use super::degradation::Degradation;
use super::draw_source::{DrawSource, IndexBind, SizedIndexBind};
use super::encoder::EncoderCore;
use super::error::{internal, Error, RuntimeGap};
use super::gpu_pools::IndexPool;
use super::index_rewrite::GatherHeld;
use super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::resource::ResourceKey;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::upload::{Upload, Uploads};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{
    BoundsRunDraw, ExecutionStatus, IndexBoundsFillModule, IndexBoundsPass, IndexBoundsPlanModule,
    IndexBoundsRun, IndexBufferRef, IndirectRef,
};

/// One draw's pass, planned: what records before the render pass instance, and the two draws in
/// it.
pub(super) struct BoundsPass {
    words: Upload,
    pool: IndexPool,
    plan: OwnedKernel,
    fill: OwnedKernel,
    /// The plan's set, then the fill's.
    sets: [OwnedSets; 2],
    /// The pass's reads of the Metal arguments and indices.
    uses: Vec<Use<ResourceKey>>,
    /// The draw's Metal index offset and the buffer's length, where the offset lies at or past
    /// the length: every position draws index 0 there, where Metal reads memory.
    past_length: Option<(u64, u64)>,
}

impl BoundsPass {
    /// The pass of the indexed indirect draw that reads `index_size`-byte indices from `index`
    /// and its arguments from `indirect`, in `core`'s command buffer, whose reads of the Metal
    /// buffers are `uses` ([`reads`]; the render encoder decides where the pass runs from them).
    /// Refused (`NotImplemented(IndirectDrawArguments)`) where the binding layer cannot bind its
    /// ranges; and as uploads, kernels and descriptor sets are refused.
    pub(super) fn new(
        core: &mut EncoderCore,
        index: &DrawSource<'_>,
        indirect: &DrawSource<'_>,
        index_size: u64,
        uses: Vec<Use<ResourceKey>>,
    ) -> Result<BoundsPass, Error> {
        let device = core.device().clone();
        let limits = &device.properties().description.limits;
        // The indices the draw reads before where Metal's fetch stops (the source's bytes).
        let index_end = index.at + index.bound(index.at);
        let refused = || Error::NotImplemented(RuntimeGap::IndirectDrawArguments);
        let pool_bytes = u32::try_from(device.shared().gpu_pools.index_bytes)
            .map_err(|_| internal("an index pool past a u32's bytes"))?;
        let pass = IndexBoundsPass::new(
            IndirectRef::new(0, indirect.at),
            index_size,
            index.at,
            index_end,
            limits.min_storage_buffer_offset_alignment,
            limits.max_storage_buffer_range,
            pool_bytes,
        )
        .ok_or_else(refused)?;
        let plan = device.owned_kernel(OwnedModule::IndexBoundsPlan)?;
        let fill = device.owned_kernel(OwnedModule::IndexBoundsFill)?;
        let queue = core.command_buffer().queue_place().id();
        let (words, pool, status) = {
            let mut state = core.command_buffer().state();
            let words =
                state
                    .uploads
                    .upload(&device, &pass.parameter_bytes(), IndexBoundsPass::BYTES)?;
            let pool = state.index_pool(&device, queue)?;
            (words, pool, state.status_words(&device)?)
        };
        let region = |upload: &Upload, len: u64| (upload.raw(), upload.offset(), len);
        let words_range = region(&words, IndexBoundsPass::BYTES);
        let counter_range = region(&pool.counter, IndexBoundsPass::POOL_HEADER_BYTES);
        let block_range = (pool.block, 0, pool.bytes);
        let mut plan_set = [(vk::Buffer::null(), 0, 0); 4];
        plan_set[IndexBoundsPlanModule::WORDS_BINDING as usize] = words_range;
        plan_set[IndexBoundsPlanModule::ARGUMENTS_BINDING as usize] = (
            indirect.raw,
            pass.arguments_range.offset,
            pass.arguments_range.range,
        );
        plan_set[IndexBoundsPlanModule::POOL_BINDING as usize] = counter_range;
        plan_set[IndexBoundsPlanModule::STATUS_BINDING as usize] =
            region(&status, ExecutionStatus::BYTES as u64);
        let mut fill_set = [(vk::Buffer::null(), 0, 0); 3];
        fill_set[IndexBoundsFillModule::WORDS_BINDING as usize] = words_range;
        // No index inside the buffer: nothing reads the binding, which takes the words.
        fill_set[IndexBoundsFillModule::INDICES_BINDING as usize] = match pass.indices_range {
            Some(range) => (index.raw, range.offset, range.range),
            None => words_range,
        };
        fill_set[IndexBoundsFillModule::POOL_BINDING as usize] = block_range;
        if plan_set
            .iter()
            .chain(&fill_set)
            .any(|&(buffer, _, _)| buffer == vk::Buffer::null())
        {
            return Err(internal("an index bounds pass binding without a buffer"));
        }
        let sets = [
            OwnedSets::write(&device, &plan, &[plan_set])?,
            OwnedSets::write(&device, &fill, &[fill_set])?,
        ];
        let past_length = past_length(index);
        Ok(BoundsPass {
            words,
            pool,
            plan,
            fill,
            sets,
            uses,
            past_length,
        })
    }

    /// The pass's reads of the Metal arguments and indices ([`reads`]).
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Hold the pass's sets in `core`'s command buffer until it completes and announce the pass
    /// once; what is left records its dispatches, before the render pass or before the render
    /// pass instance the draw records into (`render.rs`). Call it once the draw records.
    pub(super) fn into_steps(self, core: &mut EncoderCore) -> BoundsSteps {
        let BoundsPass {
            words,
            plan,
            fill,
            sets,
            uses,
            past_length,
            ..
        } = self;
        let [plan_sets, fill_sets] = sets;
        let steps = BoundsSteps {
            plan,
            fill,
            sets: [plan_sets.sets.clone(), fill_sets.sets.clone()],
            words: (words.raw(), words.offset()),
            uses,
        };
        core.retain(plan_sets);
        core.retain(fill_sets);
        announce(core.device(), past_length);
        steps
    }

    /// The two draws of the pass ([`two_draws`]).
    pub(super) fn draws(
        &self,
        index: &DrawSource<'_>,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Result<[BoundDraw; 2], Error> {
        let at = |offset: u64| self.words.offset() + offset;
        two_draws(
            (index, draw, sized),
            (
                self.words.raw(),
                at(IndexBoundsPass::FIRST_DRAW_OFFSET),
                at(IndexBoundsPass::SECOND_DRAW_OFFSET),
            ),
            &self.pool,
        )
    }
}

/// One Vulkan draw of a bounded draw: its index binding, the `VkBuffer` of its arguments, and its
/// command.
pub(super) type BoundDraw = (
    Option<IndexBind>,
    Option<vk::Buffer>,
    metal2vulkan::vk::DrawCommand,
);

/// The two draws of a pass or a run's slot, in order, their commands in `words` at `first` and
/// `second`: the first over the Metal index buffer as `index` binds it, the second over `pool`.
/// `draw` is the binder's indexed indirect draw of `index`, and `sized` the device's index range
/// binding.
fn two_draws(
    (index, draw, sized): (
        &DrawSource<'_>,
        metal2vulkan::vk::DrawCommand,
        Option<SizedIndexBind>,
    ),
    (words, first, second): (vk::Buffer, u64, u64),
    pool: &IndexPool,
) -> Result<[BoundDraw; 2], Error> {
    let metal2vulkan::vk::DrawCommand::DrawIndexedIndirect {
        index_type,
        index_buffer_offset,
        ..
    } = draw
    else {
        return Err(internal(
            "an index bounds pass of a draw that is not indexed indirect",
        ));
    };
    // The first draws only where every position lies inside the buffer's whole indices; with
    // none inside, its offset may lie past the `VkBuffer`, so it binds offset 0 and draws
    // nothing.
    let first_offset = if index.bound(index.at) == 0 {
        0
    } else {
        index_buffer_offset
    };
    let first_draw = metal2vulkan::vk::DrawCommand::DrawIndexedIndirect {
        index_type,
        index_buffer_offset: first_offset,
        offset: first,
        draw_count: 1,
        stride: 20,
    };
    let pool_indices = IndexBoundsPass::POOL_HEADER_BYTES;
    let second_draw = metal2vulkan::vk::DrawCommand::DrawIndexedIndirect {
        index_type,
        index_buffer_offset: pool_indices,
        offset: second,
        draw_count: 1,
        stride: 20,
    };
    let pooled = IndexBind {
        buffer: pool.block,
        offset: pool_indices,
        index_type: index_type.into(),
        sized: sized.map(|bind| (bind, pool.index_bytes())),
    };
    Ok([
        (
            IndexBind::of(index, first_draw, sized),
            Some(words),
            first_draw,
        ),
        (Some(pooled), Some(words), second_draw),
    ])
}

/// The index offset and the buffer's length of a draw from `index`, where the offset lies at or
/// past the length: every position draws index 0 there, where Metal reads memory.
fn past_length(index: &DrawSource<'_>) -> Option<(u64, u64)> {
    let (buffer, offset) = index.metal();
    (offset >= buffer.length()).then(|| (offset, buffer.length()))
}

/// Announce the pass once per device, and where the draw's index offset lies at or past the
/// buffer's length (`past_length`), the zeroed fetch.
fn announce(device: &Device, past_length: Option<(u64, u64)>) {
    device.degrade(Degradation::IndirectIndexBoundsPass, || {
        (
            "an indexed indirect draw whose index fetch nothing bounds".to_owned(),
            "the device does not enable robustBufferAccess2, or its driver is not measured to read \
             index 0 past the bound range, or the draw's index offset lies at or past the buffer's \
             length, so the draw's indices are bounded on the GPU before the pass"
                .to_owned(),
        )
    });
    if let Some((offset, length)) = past_length {
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
}

/// The pass's reads of the Metal buffers: the 20 bytes of arguments, and the indices from the
/// draw's offset to where Metal's fetch stops (`IndexBufferRef::fetch_end`), in the compute stage.
pub(super) fn reads(
    index: &DrawSource<'_>,
    indirect: &DrawSource<'_>,
    index_size: u64,
) -> Vec<Use<ResourceKey>> {
    let read = Access::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    let (arguments, offset) = indirect.metal();
    let (indices, index_offset) = index.metal();
    let end = IndexBufferRef::new(0, indices.length()).fetch_end(index_offset);
    let whole = end.saturating_sub(index_offset) / index_size * index_size;
    [
        (arguments, offset..offset + 20),
        (indices, index_offset..index_offset + whole),
    ]
    .into_iter()
    .map(|(buffer, range)| {
        let (identity, range) = buffer.tracked(range);
        Use::reads(identity.key, identity.tracked, UseRange::Bytes(range), read)
    })
    .collect()
}

/// Of [`reads`], those of the bytes the draw reads in place, which a write since the encoder's
/// last memory barrier races: not those of a source copied because a draw wrote them before any
/// barrier (`DrawSource::racing`), whose copy holds the bytes from before the write.
pub(super) fn in_place(
    index: &DrawSource<'_>,
    indirect: &DrawSource<'_>,
    index_size: u64,
) -> Vec<Use<ResourceKey>> {
    // In [`reads`]' order: the arguments, then the indices.
    let copied = [indirect.racing(), index.racing()];
    reads(index, indirect, index_size)
        .into_iter()
        .zip(copied)
        .filter_map(|(read, copied)| (!copied).then_some(read))
        .collect()
}

/// A planned pass's dispatches, its sets held by the command buffer
/// ([`BoundsPass::into_steps`]).
pub(super) struct BoundsSteps {
    plan: OwnedKernel,
    fill: OwnedKernel,
    /// The plan's sets, then the fill's.
    sets: [Vec<vk::DescriptorSet>; 2],
    /// The words' `VkBuffer` and offset: the fill's dispatch arguments lie in them.
    words: (vk::Buffer, u64),
    /// The pass's reads of the Metal arguments and indices.
    uses: Vec<Use<ResourceKey>>,
}

impl BoundsSteps {
    /// The pass's reads of the Metal arguments and indices.
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Record the plan and the fill into `core`'s command buffer before the pass, between a
    /// barrier from everything before and one to everything after (and one between them).
    pub(super) fn record(&self, core: &mut EncoderCore) {
        core.barrier(Barrier::everything());
        core.command(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.uses,
            |raw, command_buffer| {
                // SAFETY: the encoder's command buffer is recording outside a render pass.
                unsafe { self.dispatches(raw, command_buffer) }
            },
        );
        core.barrier(Barrier::everything());
    }

    /// Record the plan and the fill into `command_buffer` between a barrier from everything
    /// before and one to everything after (and one between them): before a render pass instance
    /// of the encoder's pass, after the instances before it (`render.rs`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is the encoder's, recording outside a render pass, and only the encoder
    /// records it.
    pub(super) unsafe fn record_in(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states; barriers need nothing more.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            self.dispatches(raw, command_buffer);
            Barrier::everything().record(raw, command_buffer);
        }
    }

    /// The plan's dispatch, a barrier, and the fill's indirect dispatch.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only one encoder records it.
    unsafe fn dispatches(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let (words, offset) = self.words;
        // SAFETY: as the caller states; the kernels are the device's, and their sets are live
        // until the command buffer completes (retained by `into_steps`), as are the uploads and
        // the Metal buffers the sets name; the fill's dispatch arguments lie in the words, which
        // the plan writes before the barrier between the two.
        unsafe {
            bind_kernel(raw, command_buffer, &self.plan, &self.sets[0]);
            raw.cmd_dispatch(command_buffer, 1, 1, 1);
            Barrier::everything().record(raw, command_buffer);
            bind_kernel(raw, command_buffer, &self.fill, &self.sets[1]);
            raw.cmd_dispatch_indirect(
                command_buffer,
                words,
                offset + IndexBoundsPass::FILL_DISPATCH_OFFSET,
            );
        }
    }
}

/// An indexed indirect draw's index bounds, which records before the pass: its own pass, or its
/// slot of a run (the module docs).
pub(super) enum Bounded {
    Pass(Box<BoundsPass>),
    Run(BoundsSlot),
}

impl Bounded {
    /// The reads of the Metal arguments and indices that record with the draw: its own pass's; a
    /// run's draw's were tracked when it joined.
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        match self {
            Bounded::Pass(pass) => pass.uses(),
            Bounded::Run(_) => &[],
        }
    }

    /// The two draws ([`two_draws`]).
    pub(super) fn draws(
        &self,
        index: &DrawSource<'_>,
        draw: metal2vulkan::vk::DrawCommand,
        sized: Option<SizedIndexBind>,
    ) -> Result<[BoundDraw; 2], Error> {
        match self {
            Bounded::Pass(pass) => pass.draws(index, draw, sized),
            Bounded::Run(slot) => {
                let (words, at) = slot.words;
                two_draws(
                    (index, draw, sized),
                    (
                        words,
                        at + IndexBoundsRun::first_draw_offset(slot.slot),
                        at + IndexBoundsRun::second_draw_offset(slot.slot),
                    ),
                    &slot.pool,
                )
            }
        }
    }
}

/// The most draws of a run: its words fit half an ordinary chunk of the upload arena
/// (`upload.rs`).
const RUN_DRAWS: u32 = 288;

/// An indexed indirect draw whose index fetch a run of the index bounds pass bounds: its Metal
/// index and arguments sources, both read in place, and its index size.
pub(super) struct BoundsRunRequest<'a> {
    pub(super) index: &'a DrawSource<'a>,
    pub(super) indirect: &'a DrawSource<'a>,
    pub(super) index_size: u64,
}

/// What a run's draws share: the arguments' `VkBuffer` and the bytes it binds of it, then the
/// indices'.
type RunKey = ((vk::Buffer, u64, u64), (vk::Buffer, u64, u64));

/// One run of the index bounds pass: its draws, what they share, its words' region, its kernels
/// and their sets (the plan's, then the fill's).
pub(super) struct BoundsRun {
    run: IndexBoundsRun,
    key: RunKey,
    words: Upload,
    plan: OwnedKernel,
    fill: OwnedKernel,
    sets: [Vec<vk::DescriptorSet>; 2],
}

/// A draw's slot of a run: the run's words' `VkBuffer` and offset, the slot, the command
/// buffer's index pool, and the draw's index offset and buffer length where the offset lies at
/// or past the length.
pub(super) struct BoundsSlot {
    words: (vk::Buffer, u64),
    slot: u32,
    pool: IndexPool,
    past_length: Option<(u64, u64)>,
}

impl BoundsSlot {
    /// Announce the bounds as the draw's own pass announces them. Call it once the draw records.
    pub(super) fn announce(&self, device: &Device) {
        announce(device, self.past_length);
    }
}

/// `request`'s draw joins an open run of `runs` over its arguments and index buffers that takes
/// it, or a new one into `pool`, with its words' region in `uploads` and `status` the command
/// buffer's status words: its slot, and what a new run's command buffer holds. `None` where no
/// run takes the draw (its own pass would not be made, or its bytes are past what a run's
/// bindings take), which its own pass then bounds. Refused as uploads, kernels and descriptor
/// sets are refused.
pub(super) fn join(
    runs: &mut Vec<BoundsRun>,
    device: &Device,
    (uploads, status, pool): (&mut Uploads, &Upload, &IndexPool),
    request: BoundsRunRequest<'_>,
) -> Result<Option<(BoundsSlot, Option<GatherHeld>)>, Error> {
    let BoundsRunRequest {
        index,
        indirect,
        index_size,
    } = request;
    // The run binds each Metal buffer's bytes, from its start, through its length (an index
    // buffer's rounded up to 4, `IndexBufferRef::fetch_end`).
    let bytes = |source: &DrawSource<'_>| (source.raw, source.at - source.metal().1, source.length);
    let key = (bytes(indirect), bytes(index));
    let draw = BoundsRunDraw {
        arguments_offset: indirect.at,
        index_size,
        index_offset: index.at,
        index_end: index.at + index.bound(index.at),
    };
    let slot_of = |run: &BoundsRun, slot| BoundsSlot {
        words: (run.words.raw(), run.words.offset()),
        slot,
        pool: pool.clone(),
        past_length: past_length(index),
    };
    let joined = runs
        .iter_mut()
        .filter(|run| run.key == key && run.run.len() < RUN_DRAWS as usize)
        .find_map(|run| {
            let slot = run.run.push(draw)?;
            Some(slot_of(run, slot))
        });
    if let Some(slot) = joined {
        return Ok(Some((slot, None)));
    }
    let limits = &device.properties().description.limits;
    let Ok(pool_bytes) = u32::try_from(pool.index_bytes()) else {
        return Ok(None);
    };
    let ((_, arguments_start, arguments_end), (_, index_start, index_end)) = key;
    let Some(mut run) = IndexBoundsRun::new(
        (arguments_start, arguments_end),
        Some((index_start, index_end)),
        (
            limits.min_storage_buffer_offset_alignment,
            limits.max_storage_buffer_range,
            limits.max_compute_work_group_count[0],
        ),
        pool_bytes,
    ) else {
        return Ok(None);
    };
    let Some(slot) = run.push(draw) else {
        return Ok(None);
    };
    let words = uploads.upload(device, &[], IndexBoundsRun::words_bytes(RUN_DRAWS))?;
    let plan = device.owned_kernel(OwnedModule::IndexBoundsPlanRun)?;
    let fill = device.owned_kernel(OwnedModule::IndexBoundsFillRun)?;
    let region = |upload: &Upload, len: u64| (upload.raw(), upload.offset(), len);
    let words_range = region(&words, words.len());
    let arguments = run.arguments_range();
    let mut plan_set = [(vk::Buffer::null(), 0, 0); 4];
    plan_set[IndexBoundsPlanModule::WORDS_BINDING as usize] = words_range;
    plan_set[IndexBoundsPlanModule::ARGUMENTS_BINDING as usize] =
        (indirect.raw, arguments.offset, arguments.range);
    plan_set[IndexBoundsPlanModule::POOL_BINDING as usize] =
        region(&pool.counter, IndexBoundsPass::POOL_HEADER_BYTES);
    plan_set[IndexBoundsPlanModule::STATUS_BINDING as usize] =
        region(status, ExecutionStatus::BYTES as u64);
    let mut fill_set = [(vk::Buffer::null(), 0, 0); 3];
    fill_set[IndexBoundsFillModule::WORDS_BINDING as usize] = words_range;
    // No index inside the buffer: nothing reads the binding, which takes the words.
    fill_set[IndexBoundsFillModule::INDICES_BINDING as usize] = match run.indices_range() {
        Some(range) => (index.raw, range.offset, range.range),
        None => words_range,
    };
    fill_set[IndexBoundsFillModule::POOL_BINDING as usize] = (pool.block, 0, pool.bytes);
    let plan_sets = OwnedSets::write(device, &plan, &[plan_set])?;
    let fill_sets = OwnedSets::write(device, &fill, &[fill_set])?;
    let run = BoundsRun {
        run,
        key,
        words,
        plan,
        fill,
        sets: [plan_sets.sets.clone(), fill_sets.sets.clone()],
    };
    let slot = slot_of(&run, slot);
    runs.push(run);
    Ok(Some((
        slot,
        Some(GatherHeld {
            output: None,
            sets: vec![plan_sets, fill_sets],
        }),
    )))
}

impl BoundsRun {
    /// Write the run's words into its region: before its plan records.
    pub(super) fn write(&self) {
        self.words.write(&self.run.words());
    }

    /// Record the run's plan, one invocation, into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only one encoder records it; the
    /// run's kernels are the device's, its sets are held until the command buffer completes, and
    /// its words were written ([`write`](Self::write)).
    pub(super) unsafe fn record_plan(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states.
        unsafe {
            bind_kernel(raw, command_buffer, &self.plan, &self.sets[0]);
            raw.cmd_dispatch(command_buffer, 1, 1, 1);
        }
    }

    /// Record the run's fill into `command_buffer`, dispatched indirectly from what its plan
    /// wrote into its words.
    ///
    /// # Safety
    ///
    /// As [`record_plan`](Self::record_plan)'s, after the plan and a barrier from its writes to
    /// the dispatch's reads.
    pub(super) unsafe fn record_fill(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states.
        unsafe {
            bind_kernel(raw, command_buffer, &self.fill, &self.sets[1]);
            raw.cmd_dispatch_indirect(
                command_buffer,
                self.words.raw(),
                self.words.offset() + IndexBoundsRun::DISPATCH_OFFSET,
            );
        }
    }
}

/// Bind `kernel` and its `sets` for the compute dispatches that follow in `command_buffer`.
///
/// # Safety
///
/// `command_buffer` is recording, and only one encoder records it; the kernel is the device's,
/// and its sets are live until the command buffer completes.
unsafe fn bind_kernel(
    raw: &Vk,
    command_buffer: vk::CommandBuffer,
    kernel: &OwnedKernel,
    sets: &[vk::DescriptorSet],
) {
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
}
