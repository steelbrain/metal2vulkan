//! The device's counters: cumulative metrics and live gauges, as relaxed atomics that are always
//! on, each timed part's longest call under a lock taken only when a call passes it, and the last
//! variant makes of compute kernels under a lock taken once a make. Each part of the runtime adds
//! to its own; nothing turns them on or off.

use crate::pipeline::DispatchVariant;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, PoisonError};

/// The most threads a device runs at once: its reaper, block-maker, submit and completion
/// threads, and its cache directory's removal thread while one runs.
const THREADS: usize = 5;

/// The upper bounds, in milliseconds, of [`Metrics::idle_frees_by_age`]'s buckets but the last.
const FREE_AGES_MS: [u64; 5] = [250, 275, 350, 500, 1000];

/// The upper bounds, in milliseconds, of [`Metrics::idle_reuses_by_age`]'s buckets but the last.
const REUSE_AGES_MS: [u64; 6] = [50, 100, 250, 500, 1000, 2000];

/// The bucket of `nanos` under `bounds`, upper bounds in milliseconds: the first bound past it,
/// else the last bucket.
fn bucket(bounds: &[u64], nanos: u64) -> usize {
    (bounds.iter())
        .position(|&ms| nanos < ms * 1_000_000)
        .unwrap_or(bounds.len())
}

/// The bounds of each [`Timed`]'s buckets, in nanoseconds: 1, 5 and 20 ms.
const OVER_NANOS: [u64; 3] = [1_000_000, 5_000_000, 20_000_000];

/// A `vkQueueSubmit2` call longer than this, in nanoseconds (100 ms), is counted in
/// [`Metrics::submit_calls_over_100_ms`].
const LONG_SUBMIT_CALL: u64 = 100_000_000;

/// The buckets of each stage histogram of [`Metrics`]
/// ([`commit_to_scheduled`](Metrics::commit_to_scheduled) and the three after it, and
/// [`event_holds`](Metrics::event_holds)).
pub(crate) const STAGE_BUCKETS: usize = 23;

/// The stage histogram bucket of a span of `nanos` nanoseconds: 0 under 2 us, `i` from 2^i us
/// to under 2^(i+1) us, and the last from 2^22 us (about 4.19 s) on.
pub(crate) fn stage_bucket(nanos: u64) -> usize {
    match nanos / 1_000 {
        0 | 1 => 0,
        micros => (micros.ilog2() as usize).min(STAGE_BUCKETS - 1),
    }
}

/// A stage of a command buffer's life that [`Metrics`] keeps a histogram of.
#[derive(Clone, Copy)]
pub(crate) enum Stage {
    /// [`Metrics::commit_to_scheduled`].
    CommitToScheduled,
    /// [`Metrics::scheduled_to_gpu_start`].
    ScheduledToGpuStart,
    /// [`Metrics::gpu_run`].
    GpuRun,
    /// [`Metrics::gpu_end_to_handler`].
    GpuEndToHandler,
}

/// A part of the consumer's takes, buffer makes and exported block makes the runtime times,
/// counted on its own in [`Metrics`].
#[derive(Clone, Copy)]
pub(crate) enum Part {
    /// [`Metrics::takes`].
    Take,
    /// [`Metrics::take_lock_waits`].
    TakeLock,
    /// [`Metrics::take_makes`].
    TakeMake,
    /// [`Metrics::take_maps`].
    TakeMap,
    /// [`Metrics::maker_makes`].
    MakerMake,
    /// [`Metrics::zero_writes`].
    ZeroWrite,
    /// [`Metrics::long_take_zero_writes`].
    LongTakeZeroWrite,
    /// [`Metrics::export_makes`].
    ExportMake,
    /// [`Metrics::export_allocations`].
    ExportAllocation,
    /// [`Metrics::export_handles`].
    ExportHandle,
    /// [`Metrics::export_claims`].
    ExportClaim,
}

/// The number of [`Part`]s.
const PARTS: usize = Part::ExportClaim as usize + 1;

/// Where a first-use wait made the `VkPipeline` it waited for ([`Metrics::first_use_waits`]).
#[derive(Clone, Copy)]
pub(crate) enum FirstUse {
    /// A compute kernel's pipeline for a new workgroup size, threadgroup memory lengths or tiles
    /// ([`Metrics::first_use_variant_waits`]).
    Variant,
    /// A render content's line-class pipeline, at its first draw of lines
    /// ([`Metrics::first_use_line_waits`]).
    Lines,
    /// A render content's point-class pipeline, at its first draw of points
    /// ([`Metrics::first_use_point_waits`]).
    Points,
}

/// The variant makes [`Counters::variant_makes`] keeps: the last this many to end.
pub(crate) const VARIANT_MAKES: usize = 256;

/// No Metal counterpart: who started a make of a compute kernel's `VkPipeline` for one
/// [`DispatchVariant`] ([`VariantMake::start`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VariantStart {
    /// The state's creation, at a size it knows: a tiling kernel's tiles, an image-only kernel's
    /// square workgroups, or an observing kernel's `requiredThreadsPerThreadgroup`.
    State,
    /// A [`prepare_dispatch`](crate::ComputePipelineState::prepare_dispatch) call, ahead of the
    /// dispatches that run it.
    Ahead,
    /// A dispatch's encoding that found it neither made nor being made, which waited for it: a
    /// first-use wait ([`Metrics::first_use_variant_waits`]).
    FirstUse,
}

/// No Metal counterpart: one ended make of a compute kernel's `VkPipeline`
/// ([`Counters::variant_makes`]): the key it was made for, who started it, when it began and
/// ended, and the first-use waits that waited for it. Times are [`host_time`](crate::host_time)'s.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VariantMake {
    /// Its number among the device's variant makes, from 0, in the order they ended.
    pub number: u64,
    /// The key of the dispatches that run it, as
    /// [`dispatch_variant`](crate::ComputePipelineState::dispatch_variant) answers them; the
    /// states of one content share it.
    pub variant: DispatchVariant,
    /// Who started it.
    pub start: VariantStart,
    /// When it began: when it was entered as being made, before its `vkCreateComputePipelines`
    /// call.
    pub began: u64,
    /// When it ended, made or failed.
    pub ended: u64,
    /// Whether it made the `VkPipeline`. A make that failed is made again by the next call or
    /// dispatch that needs it, a make of its own.
    pub made: bool,
    /// The first-use waits that waited for it: the dispatch that started it where
    /// [`start`](Self::start) is [`VariantStart::FirstUse`], and each dispatch that found it being
    /// made, a make already in flight.
    pub waits: u32,
    /// When the earliest of those waits began; 0 where none waited.
    pub first_wait: u64,
}

/// The last [`VARIANT_MAKES`] variant makes to end, oldest first, and the number of the next.
#[derive(Default)]
struct VariantMakes {
    next: u64,
    kept: VecDeque<VariantMake>,
}

/// A [`Part`]'s longest call: its nanoseconds, its bytes, and whether it made an allocation of
/// its own ([`Timed::longest_made`]); a [`Site`]'s longest allocation, its nanoseconds and bytes
/// ([`SiteAllocations::longest_bytes`]).
#[derive(Clone, Copy, Default)]
struct Longest {
    nanos: u64,
    bytes: u64,
    made: bool,
}

/// A metric the runtime adds to.
#[derive(Clone, Copy)]
pub(crate) enum Metric {
    /// [`Metrics::submissions`].
    Submissions,
    /// [`Metrics::passes`].
    Passes,
    /// [`Metrics::barriers`].
    Barriers,
    /// [`Metrics::pipelines_created`].
    PipelinesCreated,
    /// [`Metrics::blocks_made`].
    BlocksMade,
    /// [`Metrics::dispatches_without_effects`].
    DispatchesWithoutEffects,
    /// [`Metrics::translations`].
    Translations,
    /// [`Metrics::translations_loaded`].
    TranslationsLoaded,
    /// [`Metrics::render_records_read`].
    RenderRecordsRead,
    /// [`Metrics::render_records_stale`].
    RenderRecordsStale,
    /// [`Metrics::creations_before_seed`].
    CreationsBeforeSeed,
    /// [`Metrics::cache_writes_dropped`].
    CacheWritesDropped,
    /// [`Metrics::ring_writes`].
    RingWrites,
    /// [`Metrics::pushes`].
    Pushes,
    /// [`Metrics::pushes_held`].
    PushesHeld,
    /// [`Metrics::upload_waits`].
    UploadWaits,
    /// [`Metrics::staging_sleeps`].
    StagingSleeps,
    /// [`Metrics::synchronizations`].
    Synchronizations,
    /// [`Metrics::allocations_freed`].
    AllocationsFreed,
    /// [`Metrics::submissions_during_frees`].
    SubmissionsDuringFrees,
    /// [`Metrics::untimed_command_buffers`].
    UntimedCommandBuffers,
    /// [`Metrics::sampler_element_writes`].
    SamplerElementWrites,
    /// [`Metrics::stamps_not_available`].
    StampsNotAvailable,
    /// [`Metrics::waits_behind_counter`].
    WaitsBehindCounter,
    /// [`Metrics::external_acquires`].
    ExternalAcquires,
    /// [`Metrics::external_releases`].
    ExternalReleases,
    /// [`Metrics::surfaces_in_place`].
    SurfacesInPlace,
    /// [`Metrics::surfaces_copied`].
    SurfacesCopied,
    /// [`Metrics::texture_clear_barriers`].
    TextureClearBarriers,
    /// [`Metrics::attachment_loop_reads`].
    AttachmentLoopReads,
    /// [`Metrics::sparse_binds`].
    SparseBinds,
    /// [`Metrics::resolves`].
    Resolves,
}

/// The number of [`Metric`]s.
const METRICS: usize = Metric::Resolves as usize + 1;

/// Every [`Metric`], in its order: each at its own index.
const METRIC_ORDER: [Metric; METRICS] = [
    Metric::Submissions,
    Metric::Passes,
    Metric::Barriers,
    Metric::PipelinesCreated,
    Metric::BlocksMade,
    Metric::DispatchesWithoutEffects,
    Metric::Translations,
    Metric::TranslationsLoaded,
    Metric::RenderRecordsRead,
    Metric::RenderRecordsStale,
    Metric::CreationsBeforeSeed,
    Metric::CacheWritesDropped,
    Metric::RingWrites,
    Metric::Pushes,
    Metric::PushesHeld,
    Metric::UploadWaits,
    Metric::StagingSleeps,
    Metric::Synchronizations,
    Metric::AllocationsFreed,
    Metric::SubmissionsDuringFrees,
    Metric::UntimedCommandBuffers,
    Metric::SamplerElementWrites,
    Metric::StampsNotAvailable,
    Metric::WaitsBehindCounter,
    Metric::ExternalAcquires,
    Metric::ExternalReleases,
    Metric::SurfacesInPlace,
    Metric::SurfacesCopied,
    Metric::TextureClearBarriers,
    Metric::AttachmentLoopReads,
    Metric::SparseBinds,
    Metric::Resolves,
];

// Each metric of the order is at its own index.
const _: () = {
    let mut at = 0;
    while at < METRICS {
        assert!(METRIC_ORDER[at] as usize == at);
        at += 1;
    }
};

/// What a command buffer's recording counts as its encoders record, added to the device's
/// counters once an encoder ends ([`Counters::add`]): no command an encoder records writes a
/// counter the device's threads share.
#[derive(Default)]
pub(crate) struct Tally {
    counts: [u64; METRICS],
    /// The metrics counted since the tally was last added, a bit each at its index.
    counted: u32,
}

// Each metric has its bit in a tally's `counted`.
const _: () = assert!(METRICS <= u32::BITS as usize);

impl Tally {
    /// Counts one more of `metric`.
    pub(crate) fn count(&mut self, metric: Metric) {
        self.counts[metric as usize] += 1;
        self.counted |= 1 << metric as usize;
    }
}

/// A gauge the runtime moves.
#[derive(Clone, Copy)]
pub(crate) enum Gauge {
    /// [`Gauges::handles_out`].
    HandlesOut,
    /// [`Gauges::buffers`].
    Buffers,
    /// [`Gauges::textures`].
    Textures,
    /// [`Gauges::samplers`].
    Samplers,
    /// [`Gauges::descriptor_slots`].
    DescriptorSlots,
    /// [`Gauges::pipelines`].
    Pipelines,
    /// [`Gauges::kernels`].
    Kernels,
    /// [`Gauges::kernel_entries`].
    KernelEntries,
    /// [`Gauges::render_contents`].
    RenderContents,
    /// [`Gauges::attachment_views`].
    AttachmentViews,
    /// [`Gauges::blocks`].
    Blocks,
    /// [`Gauges::command_buffers`].
    CommandBuffers,
    /// [`Gauges::events`].
    Events,
    /// [`Gauges::device_samplers`].
    DeviceSamplers,
    /// [`Gauges::spilled_bytes`].
    SpilledBytes,
    /// [`Gauges::managed_spilled_bytes`].
    ManagedSpilledBytes,
    /// [`Gauges::shared_allocation_bytes`].
    SharedAllocationBytes,
    /// [`Gauges::private_allocation_bytes`].
    PrivateAllocationBytes,
    /// [`Gauges::shared_idle_bytes`].
    SharedIdleBytes,
    /// [`Gauges::private_idle_bytes`].
    PrivateIdleBytes,
    /// [`Gauges::staging_bytes`].
    StagingBytes,
    /// [`Gauges::stage_entries`].
    StageEntries,
    /// [`Gauges::stage_entry_bytes`].
    StageEntryBytes,
    /// [`Gauges::pipeline_cache_bytes`].
    PipelineCacheBytes,
    /// [`Gauges::shadows`].
    Shadows,
    /// [`Gauges::shadow_bytes`].
    ShadowBytes,
    /// [`Gauges::textures_copied_over`].
    TexturesCopiedOver,
    /// [`Gauges::timestamp_pools`].
    TimestampPools,
    /// [`Gauges::imports`].
    Imports,
    /// [`Gauges::exports`].
    Exports,
    /// [`Gauges::surface_imports`].
    SurfaceImports,
    /// [`Gauges::surface_buffers`].
    SurfaceBuffers,
    /// [`Gauges::texture_block_bytes`].
    TextureBlockBytes,
    /// [`Gauges::texture_block_held_bytes`].
    TextureBlockHeldBytes,
    /// [`Gauges::texture_block_sparse_bytes`].
    TextureBlockSparseBytes,
    /// [`Gauges::texture_block_sparse_blocks`].
    TextureBlockSparseBlocks,
    /// [`Gauges::private_take_held_bytes`].
    PrivateTakeHeldBytes,
    /// [`Gauges::private_take_bytes`].
    PrivateTakeBytes,
    /// [`Gauges::private_take_sparse_bytes`].
    PrivateTakeSparseBytes,
    /// [`Gauges::private_take_sparse_allocations`].
    PrivateTakeSparseAllocations,
    /// [`Gauges::private_units`].
    PrivateUnits,
    /// [`Gauges::private_unit_used_bytes`].
    PrivateUnitUsedBytes,
    /// [`Gauges::texture_own_bytes`].
    TextureOwnBytes,
    /// [`Gauges::shared_buffer_bytes`].
    SharedBufferBytes,
    /// [`Gauges::private_buffer_bytes`].
    PrivateBufferBytes,
    /// [`Gauges::shared_placed_bytes`].
    SharedPlacedBytes,
    /// [`Gauges::private_placed_bytes`].
    PrivatePlacedBytes,
    /// [`Gauges::import_bytes`].
    ImportBytes,
    /// [`Gauges::surface_import_bytes`].
    SurfaceImportBytes,
}

/// A site that allocates on the consumer's threads, inside its calls, counted on its own in
/// [`Metrics`].
#[derive(Clone, Copy)]
pub(crate) enum Site {
    /// [`Metrics::texture_allocations`].
    Texture,
    /// [`Metrics::texture_block_allocations`].
    TextureBlock,
    /// [`Metrics::ring_block_allocations`].
    RingBlock,
    /// [`Metrics::staging_region_allocations`].
    StagingRegion,
    /// [`Metrics::staging_buffer_allocations`].
    StagingBuffer,
    /// [`Metrics::zero_source_allocations`].
    ZeroSource,
    /// [`Metrics::shared_take_allocations`].
    SharedTake,
    /// [`Metrics::private_take_allocations`].
    PrivateTake,
}

impl Site {
    /// The site of the consumer's takes of `class` ([`Device::take_block`](crate::Device::take_block)).
    pub(crate) fn take(class: crate::MemoryClass) -> Site {
        match class {
            crate::MemoryClass::Shared => Site::SharedTake,
            crate::MemoryClass::Private => Site::PrivateTake,
        }
    }
}

/// The number of [`Site`]s.
const SITES: usize = Site::PrivateTake as usize + 1;

/// Why a texture over a buffer lives in an image of its own that the device copies its bytes into
/// and back (`crate::over`), as the texture's making decides it (`crate::texture::check_over`,
/// then the in-place image's make): the [`Metrics`] pair its copies count in. No Metal
/// counterpart: Metal makes no such copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OverCause {
    /// The buffer's bytes are in host memory's import ([`Metrics::over_copies_host_memory`]).
    HostMemory,
    /// A dma-buf's one import that takes no image here ([`Metrics::over_copies_dma_buf`]).
    DmaBuf,
    /// An import over several dma-bufs ([`Metrics::over_copies_dma_bufs`]).
    DmaBufs,
    /// Memory dedicated to one object ([`Metrics::over_copies_dedicated`]).
    Dedicated,
    /// An exported surface's buffer ([`Metrics::over_copies_surface`]).
    Surface,
    /// A block whose memory spilled ([`Metrics::over_copies_spilled`]).
    Spilled,
    /// No linear image of the format, usage and size ([`Metrics::over_copies_linear`]).
    Linear,
    /// A format feature the linear tiling lacks ([`Metrics::over_copies_linear_features`]).
    LinearFeatures,
    /// The linear image's rows at another pitch ([`Metrics::over_copies_pitch`]).
    Pitch,
    /// A render target's subresource past its reach ([`Metrics::over_copies_render_reach`]).
    RenderReach,
    /// A place off the image's alignment ([`Metrics::over_copies_alignment`]).
    Alignment,
    /// The linear image's other memory requirements ([`Metrics::over_copies_image_memory`]).
    ImageMemory,
}

/// The number of [`OverCause`]s.
const OVER_CAUSES: usize = OverCause::ImageMemory as usize + 1;

/// No Metal counterpart: a device's counters ([`Device::counters`](crate::Device::counters)).
///
/// A cheap handle that stays readable after the device is torn down, so a test can read the
/// gauges at zero once it is gone. Reads are relaxed: each value is exact once the work that
/// moves it has stopped, and close while it runs. What an encoder's commands count (its passes,
/// barriers, pushes, ring writes, dispatches without effects and synchronizes) arrives when the
/// encoder ends, all at once.
#[derive(Clone)]
pub struct Counters {
    atomics: Arc<Atomics>,
}

#[derive(Default)]
struct Atomics {
    /// [`Gauges::threads`]: each slot a running thread's id, 0 where none holds it.
    threads: [AtomicU64; THREADS],
    submissions: AtomicU64,
    submit_infos: AtomicU64,
    submit_call_nanos: AtomicU64,
    submit_call_max_nanos: AtomicU64,
    pipeline_creation_nanos: AtomicU64,
    translation_nanos: AtomicU64,
    submit_calls_over_100_ms: AtomicU64,
    queue_lock_wait_nanos: AtomicU64,
    queue_lock_wait_max_nanos: AtomicU64,
    /// Each [`Stage`]'s histogram, at its index.
    stages: [[AtomicU64; STAGE_BUCKETS]; 4],
    event_holds: [AtomicU64; STAGE_BUCKETS],
    held_starts: AtomicU64,
    held_start_gap_nanos: AtomicU64,
    passes: AtomicU64,
    barriers: AtomicU64,
    bytes_copied: AtomicU64,
    bytes_zeroed: AtomicU64,
    takes_from_free_ranges: AtomicU64,
    block_zero_fills: AtomicU64,
    block_zero_fill_bytes: AtomicU64,
    texture_clears: AtomicU64,
    texture_clear_bytes: AtomicU64,
    texture_clear_barriers: AtomicU64,
    cache_bytes_written: AtomicU64,
    pipelines_created: AtomicU64,
    first_use_waits: AtomicU64,
    first_use_wait_nanos: AtomicU64,
    first_use_wait_max_nanos: AtomicU64,
    /// Each [`FirstUse`] site's waits, at its index: their count, total and longest nanoseconds.
    first_use_sites: [[AtomicU64; 3]; 3],
    /// The makes ahead of a dispatch: their count and total nanoseconds.
    made_ahead: [AtomicU64; 2],
    /// [`Counters::variant_makes`].
    variant_makes: Mutex<VariantMakes>,
    blocks_made: AtomicU64,
    block_bytes_made: AtomicU64,
    consumer_allocations: AtomicU64,
    consumer_allocation_nanos: AtomicU64,
    consumer_allocation_max_nanos: AtomicU64,
    maker_allocation_nanos: AtomicU64,
    /// Each [`Site`]'s allocations, at its index: their count, their bytes and the longest's
    /// nanoseconds; then its takes that landed in an allocation made ahead, their count and bytes.
    sites: [[AtomicU64; 5]; SITES],
    /// Each [`Site`]'s longest allocation, at its index, set whole by the call that passes it.
    site_longest: [Mutex<Longest>; SITES],
    maker_waits: AtomicU64,
    maker_wait_max_nanos: AtomicU64,
    maker_memory_calls: AtomicU64,
    maker_yields: AtomicU64,
    maker_yield_nanos: AtomicU64,
    dispatches_without_effects: AtomicU64,
    shared_long_reuses: AtomicU64,
    private_long_reuses: AtomicU64,
    allocations_mapped: AtomicU64,
    allocations_mapped_on_takes: AtomicU64,
    allocation_map_refusals: AtomicU64,
    translations: AtomicU64,
    translations_loaded: AtomicU64,
    render_records_read: AtomicU64,
    render_records_stale: AtomicU64,
    creations_before_seed: AtomicU64,
    pipeline_cache_looks_skipped: AtomicU64,
    pipeline_cache_lock_waits: AtomicU64,
    pipeline_cache_lock_wait_nanos: AtomicU64,
    pipeline_cache_lock_wait_max_nanos: AtomicU64,
    cache_writes_dropped: AtomicU64,
    ring_writes: AtomicU64,
    pushes: AtomicU64,
    pushes_held: AtomicU64,
    upload_waits: AtomicU64,
    staging_sleeps: AtomicU64,
    synchronizations: AtomicU64,
    commit_lock_wait_nanos: AtomicU64,
    commit_lock_wait_max_nanos: AtomicU64,
    over_copies: AtomicU64,
    over_copy_bytes: AtomicU64,
    over_copies_imported: AtomicU64,
    over_copy_bytes_imported: AtomicU64,
    /// Each [`OverCause`]'s over-copies, at its index: their count and bytes.
    over_by_cause: [[AtomicU64; 2]; OVER_CAUSES],
    attachment_read_copies: AtomicU64,
    attachment_read_copy_bytes: AtomicU64,
    attachment_loop_reads: AtomicU64,
    allocations_freed: AtomicU64,
    submissions_during_frees: AtomicU64,
    submit_call_nanos_during_frees: AtomicU64,
    shared_remade_after_frees: AtomicU64,
    private_remade_after_frees: AtomicU64,
    idle_frees_by_age: [AtomicU64; FREE_AGES_MS.len() + 1],
    idle_reuses_by_age: [AtomicU64; REUSE_AGES_MS.len() + 1],
    managed_bytes_captured: AtomicU64,
    held_releases: AtomicU64,
    held_release_bytes: AtomicU64,
    held_release_starts: AtomicU64,
    untimed_command_buffers: AtomicU64,
    sampler_element_writes: AtomicU64,
    stamps_not_available: AtomicU64,
    runtime_command_buffers: AtomicU64,
    runtime_command_buffer_gpu_nanos: AtomicU64,
    waits_behind_counter: AtomicU64,
    external_acquires: AtomicU64,
    external_releases: AtomicU64,
    surfaces_in_place: AtomicU64,
    surfaces_copied: AtomicU64,
    surface_waits: AtomicU64,
    surface_wait_nanos: AtomicU64,
    surface_wait_max_nanos: AtomicU64,
    surface_wait_submissions: AtomicU64,
    sparse_binds: AtomicU64,
    resolves: AtomicU64,
    /// Each [`Part`]'s calls, at its index: their count, bytes, total and longest nanoseconds,
    /// and those over each of [`OVER_NANOS`].
    parts: [[AtomicU64; 7]; PARTS],
    /// Each [`Part`]'s longest call, at its index, set whole by the call that passes it.
    longest: [Mutex<Longest>; PARTS],
    handles_out: AtomicU64,
    buffers: AtomicU64,
    textures: AtomicU64,
    samplers: AtomicU64,
    pipelines: AtomicU64,
    kernels: AtomicU64,
    kernel_entries: AtomicU64,
    render_contents: AtomicU64,
    attachment_views: AtomicU64,
    command_buffers: AtomicU64,
    descriptor_slots: AtomicU64,
    blocks: AtomicU64,
    events: AtomicU64,
    device_samplers: AtomicU64,
    spilled_bytes: AtomicU64,
    managed_spilled_bytes: AtomicU64,
    shared_allocation_bytes: AtomicU64,
    private_allocation_bytes: AtomicU64,
    shared_idle_bytes: AtomicU64,
    private_idle_bytes: AtomicU64,
    /// When the oldest idle Shared allocation the idle rule weighs became idle, in nanoseconds of
    /// [`host_time`](crate::host_time); 0 where none is.
    shared_idle_since: AtomicU64,
    /// As [`shared_idle_since`](Self::shared_idle_since), of Private allocations.
    private_idle_since: AtomicU64,
    /// The bytes of the Shared reserve's allocations ([`Gauges::shared_reserve_bytes`]).
    shared_reserve_bytes: AtomicU64,
    /// As [`shared_reserve_bytes`](Self::shared_reserve_bytes), of Private allocations.
    private_reserve_bytes: AtomicU64,
    /// The bytes of the Shared long allocations a later take holds
    /// ([`Gauges::shared_long_reused_bytes`]).
    shared_long_reused_bytes: AtomicU64,
    /// As [`shared_long_reused_bytes`](Self::shared_long_reused_bytes), of Private allocations.
    private_long_reused_bytes: AtomicU64,
    staging_bytes: AtomicU64,
    stage_entries: AtomicU64,
    stage_entry_bytes: AtomicU64,
    pipeline_cache_bytes: AtomicU64,
    shadows: AtomicU64,
    shadow_bytes: AtomicU64,
    textures_copied_over: AtomicU64,
    timestamp_pools: AtomicU64,
    imports: AtomicU64,
    exports: AtomicU64,
    surface_imports: AtomicU64,
    surface_buffers: AtomicU64,
    texture_block_bytes: AtomicU64,
    texture_block_held_bytes: AtomicU64,
    texture_block_sparse_bytes: AtomicU64,
    texture_block_sparse_blocks: AtomicU64,
    private_take_held_bytes: AtomicU64,
    private_take_bytes: AtomicU64,
    private_take_sparse_bytes: AtomicU64,
    private_take_sparse_allocations: AtomicU64,
    private_units: AtomicU64,
    private_unit_used_bytes: AtomicU64,
    texture_own_bytes: AtomicU64,
    shared_buffer_bytes: AtomicU64,
    private_buffer_bytes: AtomicU64,
    shared_placed_bytes: AtomicU64,
    private_placed_bytes: AtomicU64,
    import_bytes: AtomicU64,
    surface_import_bytes: AtomicU64,
    /// Segment ends whose barrier reaches the host, for the tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    host_reaches: AtomicU64,
    /// Calibrations of the GPU's clock against the host's, for the tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    calibrations: AtomicU64,
    /// Barriers that ended a blit of an image, and the global memory barriers they held, for the
    /// tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    closings: [AtomicU64; 2],
}

impl Counters {
    /// Counters at zero, for a new device.
    pub(crate) fn new() -> Counters {
        Counters {
            atomics: Arc::default(),
        }
    }

    /// Counts one more of `metric`.
    pub(crate) fn count(&self, metric: Metric) {
        self.metric(metric).fetch_add(1, Relaxed);
    }

    /// Adds what `tally` counted, and leaves it at zero: one atomic add for each metric it
    /// counted, and none where it counted nothing.
    pub(crate) fn add(&self, tally: &mut Tally) {
        while tally.counted != 0 {
            let at = tally.counted.trailing_zeros() as usize;
            tally.counted &= tally.counted - 1;
            let count = tally.counts[at];
            tally.counts[at] = 0;
            self.metric(METRIC_ORDER[at]).fetch_add(count, Relaxed);
        }
    }

    /// The atomic of `metric`.
    fn metric(&self, metric: Metric) -> &AtomicU64 {
        let a = &self.atomics;
        match metric {
            Metric::Submissions => &a.submissions,
            Metric::Passes => &a.passes,
            Metric::Barriers => &a.barriers,
            Metric::PipelinesCreated => &a.pipelines_created,
            Metric::BlocksMade => &a.blocks_made,
            Metric::DispatchesWithoutEffects => &a.dispatches_without_effects,
            Metric::Translations => &a.translations,
            Metric::TranslationsLoaded => &a.translations_loaded,
            Metric::RenderRecordsRead => &a.render_records_read,
            Metric::RenderRecordsStale => &a.render_records_stale,
            Metric::CreationsBeforeSeed => &a.creations_before_seed,
            Metric::CacheWritesDropped => &a.cache_writes_dropped,
            Metric::RingWrites => &a.ring_writes,
            Metric::Pushes => &a.pushes,
            Metric::PushesHeld => &a.pushes_held,
            Metric::UploadWaits => &a.upload_waits,
            Metric::StagingSleeps => &a.staging_sleeps,
            Metric::Synchronizations => &a.synchronizations,
            Metric::AllocationsFreed => &a.allocations_freed,
            Metric::SubmissionsDuringFrees => &a.submissions_during_frees,
            Metric::UntimedCommandBuffers => &a.untimed_command_buffers,
            Metric::SamplerElementWrites => &a.sampler_element_writes,
            Metric::StampsNotAvailable => &a.stamps_not_available,
            Metric::WaitsBehindCounter => &a.waits_behind_counter,
            Metric::ExternalAcquires => &a.external_acquires,
            Metric::ExternalReleases => &a.external_releases,
            Metric::SurfacesInPlace => &a.surfaces_in_place,
            Metric::SurfacesCopied => &a.surfaces_copied,
            Metric::TextureClearBarriers => &a.texture_clear_barriers,
            Metric::AttachmentLoopReads => &a.attachment_loop_reads,
            Metric::SparseBinds => &a.sparse_binds,
            Metric::Resolves => &a.resolves,
        }
    }

    /// One more first-use wait at `site`: an encoding call that made the `VkPipeline` it needed,
    /// for `nanos` nanoseconds, in the total and in its site's.
    pub(crate) fn first_use_wait(&self, site: FirstUse, nanos: u64) {
        let a = &self.atomics;
        a.first_use_waits.fetch_add(1, Relaxed);
        a.first_use_wait_nanos.fetch_add(nanos, Relaxed);
        a.first_use_wait_max_nanos.fetch_max(nanos, Relaxed);
        let [count, total, longest] = &a.first_use_sites[site as usize];
        count.fetch_add(1, Relaxed);
        total.fetch_add(nanos, Relaxed);
        longest.fetch_max(nanos, Relaxed);
    }

    /// One more make of a kernel's pipeline on a
    /// [`prepare_dispatch`](crate::ComputePipelineState::prepare_dispatch) call's thread, for
    /// `nanos` nanoseconds ([`Metrics::made_ahead_variants`]).
    pub(crate) fn made_ahead(&self, nanos: u64) {
        let [count, total] = &self.atomics.made_ahead;
        count.fetch_add(1, Relaxed);
        total.fetch_add(nanos, Relaxed);
    }

    /// One more ended variant make, `make` of its number: kept, and the oldest kept dropped past
    /// [`VARIANT_MAKES`].
    pub(crate) fn variant_made(&self, make: impl FnOnce(u64) -> VariantMake) {
        let mut makes = (self.atomics.variant_makes.lock()).unwrap_or_else(PoisonError::into_inner);
        if makes.kept.len() == VARIANT_MAKES {
            makes.kept.pop_front();
        }
        let number = makes.next;
        makes.next += 1;
        makes.kept.push_back(make(number));
    }

    /// No Metal counterpart: the kept variant makes numbered `from` or later, oldest first. The
    /// device keeps the last 256 to end, so a reader that passes one past the last number it read
    /// gets each make once, and a number missing at the front was dropped before the read.
    pub fn variant_makes(&self, from: u64) -> Vec<VariantMake> {
        let makes = (self.atomics.variant_makes.lock()).unwrap_or_else(PoisonError::into_inner);
        (makes.kept.iter())
            .filter(|make| make.number >= from)
            .copied()
            .collect()
    }

    /// One more surface wait ([`Metrics::surface_waits`]): `nanos` nanoseconds, for `submissions`
    /// of the runtime's submissions.
    pub(crate) fn surface_wait(&self, nanos: u64, submissions: u64) {
        let a = &self.atomics;
        a.surface_waits.fetch_add(1, Relaxed);
        a.surface_wait_nanos.fetch_add(nanos, Relaxed);
        a.surface_wait_max_nanos.fetch_max(nanos, Relaxed);
        a.surface_wait_submissions.fetch_add(submissions, Relaxed);
    }

    /// A commit that waited `nanos` nanoseconds to take the submission lock
    /// ([`Metrics::commit_lock_wait_nanos`]).
    pub(crate) fn commit_lock_wait(&self, nanos: u64) {
        let a = &self.atomics;
        a.commit_lock_wait_nanos.fetch_add(nanos, Relaxed);
        a.commit_lock_wait_max_nanos.fetch_max(nanos, Relaxed);
    }

    /// A look of the reaper's at the pipeline cache skipped
    /// ([`Metrics::pipeline_cache_looks_skipped`]).
    pub(crate) fn pipeline_cache_look_skipped(&self) {
        (self.atomics.pipeline_cache_looks_skipped).fetch_add(1, Relaxed);
    }

    /// A creation through the pipeline cache that waited `nanos` nanoseconds for the reaper's look
    /// at it ([`Metrics::pipeline_cache_lock_waits`]).
    pub(crate) fn pipeline_cache_lock_wait(&self, nanos: u64) {
        let a = &self.atomics;
        a.pipeline_cache_lock_waits.fetch_add(1, Relaxed);
        a.pipeline_cache_lock_wait_nanos.fetch_add(nanos, Relaxed);
        a.pipeline_cache_lock_wait_max_nanos
            .fetch_max(nanos, Relaxed);
    }

    /// A submission of `infos` submit infos ([`Metrics::submit_infos`]).
    pub(crate) fn submitted_infos(&self, infos: u64) {
        self.atomics.submit_infos.fetch_add(infos, Relaxed);
    }

    /// A `vkQueueSubmit2` call that took `nanos` nanoseconds ([`Metrics::submit_call_nanos`]).
    pub(crate) fn submit_call(&self, nanos: u64) {
        let a = &self.atomics;
        a.submit_call_nanos.fetch_add(nanos, Relaxed);
        a.submit_call_max_nanos.fetch_max(nanos, Relaxed);
        if nanos > LONG_SUBMIT_CALL {
            a.submit_calls_over_100_ms.fetch_add(1, Relaxed);
        }
    }

    /// A `vkCreateComputePipelines` or `vkCreateGraphicsPipelines` call that took `nanos`
    /// nanoseconds ([`Metrics::pipeline_creation_nanos`]).
    pub(crate) fn pipeline_creation(&self, nanos: u64) {
        self.atomics
            .pipeline_creation_nanos
            .fetch_add(nanos, Relaxed);
    }

    /// A call into the translator that took `nanos` nanoseconds ([`Metrics::translation_nanos`]).
    pub(crate) fn translation(&self, nanos: u64) {
        self.atomics.translation_nanos.fetch_add(nanos, Relaxed);
    }

    /// A submission that waited `nanos` nanoseconds for the runtime's queue lock before its
    /// calls ([`Metrics::queue_lock_wait_nanos`]).
    pub(crate) fn queue_lock_wait(&self, nanos: u64) {
        let a = &self.atomics;
        a.queue_lock_wait_nanos.fetch_add(nanos, Relaxed);
        a.queue_lock_wait_max_nanos.fetch_max(nanos, Relaxed);
    }

    /// A command buffer's span of `nanos` nanoseconds at `stage`, in its histogram
    /// ([`Metrics::commit_to_scheduled`] and the three after it).
    pub(crate) fn stage(&self, stage: Stage, nanos: u64) {
        self.atomics.stages[stage as usize][stage_bucket(nanos)].fetch_add(1, Relaxed);
    }

    /// A segment the submit thread held `nanos` nanoseconds for an event's value
    /// ([`Metrics::event_holds`]).
    pub(crate) fn event_hold(&self, nanos: u64) {
        self.atomics.event_holds[stage_bucket(nanos)].fetch_add(1, Relaxed);
    }

    /// A group held behind its queue's submissions in flight goes, a completion having left fewer
    /// of them ([`Metrics::held_starts`]).
    pub(crate) fn held_start(&self) {
        self.atomics.held_starts.fetch_add(1, Relaxed);
    }

    /// Such a group's first timed command buffer started `nanos` nanoseconds after the latest end
    /// of those submitted before it ([`Metrics::held_start_gap_nanos`]).
    pub(crate) fn held_start_gap(&self, nanos: u64) {
        self.atomics.held_start_gap_nanos.fetch_add(nanos, Relaxed);
    }

    /// `count` more of the runtime's own command buffers passed to `vkQueueSubmit2`
    /// ([`Metrics::runtime_command_buffers`]).
    pub(crate) fn runtime_command_buffers(&self, count: u64) {
        self.atomics
            .runtime_command_buffers
            .fetch_add(count, Relaxed);
    }

    /// One of the runtime's own command buffers ran `nanos` nanoseconds of the GPU's between its
    /// two timestamps ([`Metrics::runtime_command_buffer_gpu_nanos`]).
    pub(crate) fn runtime_command_buffer_ran(&self, nanos: u64) {
        self.atomics
            .runtime_command_buffer_gpu_nanos
            .fetch_add(nanos, Relaxed);
    }

    /// An allocation of `bytes` made for blocks ([`Metrics::block_bytes_made`]).
    pub(crate) fn allocated(&self, bytes: u64) {
        self.atomics.block_bytes_made.fetch_add(bytes, Relaxed);
    }

    /// An idle allocation whose free began after `nanos` nanoseconds idle
    /// ([`Metrics::idle_frees_by_age`]).
    pub(crate) fn freed_idle(&self, nanos: u64) {
        self.atomics.idle_frees_by_age[bucket(&FREE_AGES_MS, nanos)].fetch_add(1, Relaxed);
    }

    /// An idle allocation taken again after `nanos` nanoseconds idle
    /// ([`Metrics::idle_reuses_by_age`]).
    pub(crate) fn reused_idle(&self, nanos: u64) {
        self.atomics.idle_reuses_by_age[bucket(&REUSE_AGES_MS, nanos)].fetch_add(1, Relaxed);
    }

    /// When the oldest idle allocation of a class (Shared where `shared`) that the idle rule weighs
    /// became idle, in nanoseconds of [`host_time`](crate::host_time); 0 where none is
    /// ([`Gauges::shared_idle_age_nanos`]).
    pub(crate) fn idle_since(&self, shared: bool, since: u64) {
        let a = &self.atomics;
        let atomic = if shared {
            &a.shared_idle_since
        } else {
            &a.private_idle_since
        };
        atomic.store(since, Relaxed);
    }

    /// The bytes of the reserve of a class (Shared where `shared`): its idle allocations the idle
    /// rule does not weigh ([`Gauges::shared_reserve_bytes`]).
    pub(crate) fn reserve_bytes(&self, shared: bool, bytes: u64) {
        let a = &self.atomics;
        let atomic = if shared {
            &a.shared_reserve_bytes
        } else {
            &a.private_reserve_bytes
        };
        atomic.store(bytes, Relaxed);
    }

    /// The bytes of a class's long allocations (Shared where `shared`) that hold a block of a take
    /// after the one each was made for ([`Gauges::shared_long_reused_bytes`]).
    pub(crate) fn long_reused_bytes(&self, shared: bool, bytes: u64) {
        let a = &self.atomics;
        let atomic = if shared {
            &a.shared_long_reused_bytes
        } else {
            &a.private_long_reused_bytes
        };
        atomic.store(bytes, Relaxed);
    }

    /// A `vkAllocateMemory` that took `nanos` nanoseconds, on the block-maker thread where
    /// `maker`, else on a thread of the consumer's ([`Metrics::consumer_allocation_nanos`]).
    pub(crate) fn allocation_time(&self, nanos: u64, maker: bool) {
        let a = &self.atomics;
        if maker {
            a.maker_allocation_nanos.fetch_add(nanos, Relaxed);
        } else {
            a.consumer_allocations.fetch_add(1, Relaxed);
            a.consumer_allocation_nanos.fetch_add(nanos, Relaxed);
            a.consumer_allocation_max_nanos.fetch_max(nanos, Relaxed);
        }
    }

    /// One more import of `bytes` ([`Gauges::imports`] and [`Gauges::import_bytes`]).
    pub(crate) fn imported(&self, bytes: u64) {
        self.rise(Gauge::Imports);
        self.rise_by(Gauge::ImportBytes, bytes);
    }

    /// One import of `bytes` freed.
    pub(crate) fn import_freed(&self, bytes: u64) {
        self.fall(Gauge::Imports);
        self.fall_by(Gauge::ImportBytes, bytes);
    }

    /// One more surface plane import of `bytes` ([`Gauges::surface_imports`] and
    /// [`Gauges::surface_import_bytes`]).
    pub(crate) fn surface_imported(&self, bytes: u64) {
        self.rise(Gauge::SurfaceImports);
        self.rise_by(Gauge::SurfaceImportBytes, bytes);
    }

    /// One surface plane import of `bytes` freed.
    pub(crate) fn surface_import_freed(&self, bytes: u64) {
        self.fall(Gauge::SurfaceImports);
        self.fall_by(Gauge::SurfaceImportBytes, bytes);
    }

    /// One call of `part`, of `bytes` bytes, that took `nanos` nanoseconds ([`Metrics::takes`]
    /// and the other parts').
    pub(crate) fn timed(&self, part: Part, bytes: u64, nanos: u64, made: bool) {
        let [count, all, total, longest, over @ ..] = &self.atomics.parts[part as usize];
        count.fetch_add(1, Relaxed);
        all.fetch_add(bytes, Relaxed);
        total.fetch_add(nanos, Relaxed);
        // Only a call past the longest so far takes the lock, which nothing else holds while it
        // takes another; of two that pass it at once, the longer is kept.
        if nanos > longest.fetch_max(nanos, Relaxed) {
            let record = &self.atomics.longest[part as usize];
            let mut record = record.lock().unwrap_or_else(PoisonError::into_inner);
            if nanos > record.nanos {
                *record = Longest { nanos, bytes, made };
            }
        }
        for (bound, over) in OVER_NANOS.iter().zip(over) {
            if nanos > *bound {
                over.fetch_add(1, Relaxed);
            }
        }
    }

    /// A call that waited `nanos` nanoseconds for the block-maker thread's make of an internal it
    /// needs ([`Metrics::maker_waits`]).
    pub(crate) fn maker_wait(&self, nanos: u64) {
        let a = &self.atomics;
        a.maker_waits.fetch_add(1, Relaxed);
        a.maker_wait_max_nanos.fetch_max(nanos, Relaxed);
    }

    /// A `vkAllocateMemory` or `vkFreeMemory` made ahead of need, before which the thread waited
    /// `waited` nanoseconds for the submit calls its previous one met, `None` where none was left
    /// inside the driver ([`Metrics::maker_memory_calls`]).
    pub(crate) fn maker_memory_call(&self, waited: Option<u64>) {
        let a = &self.atomics;
        a.maker_memory_calls.fetch_add(1, Relaxed);
        if let Some(nanos) = waited {
            a.maker_yields.fetch_add(1, Relaxed);
            a.maker_yield_nanos.fetch_add(nanos, Relaxed);
        }
    }

    /// A `vkAllocateMemory` of `bytes` that took `nanos` nanoseconds at `site`, on a thread of the
    /// consumer's ([`Metrics::texture_allocations`] and the other sites').
    pub(crate) fn site_allocation(&self, site: Site, bytes: u64, nanos: u64) {
        let [count, total, longest, ..] = &self.atomics.sites[site as usize];
        count.fetch_add(1, Relaxed);
        total.fetch_add(bytes, Relaxed);
        // As a part's longest call ([`Counters::timed`]): only a call past the longest so far
        // takes the lock.
        if nanos > longest.fetch_max(nanos, Relaxed) {
            let record = &self.atomics.site_longest[site as usize];
            let mut record = record.lock().unwrap_or_else(PoisonError::into_inner);
            if nanos > record.nanos {
                *record = Longest {
                    nanos,
                    bytes,
                    made: true,
                };
            }
        }
    }

    /// A take of `bytes` at `site` that landed in an allocation the block-maker thread made ahead
    /// of need, the first take to land in it ([`SiteAllocations::takes_ahead`]).
    pub(crate) fn site_take_ahead(&self, site: Site, bytes: u64) {
        let [.., ahead, ahead_bytes] = &self.atomics.sites[site as usize];
        ahead.fetch_add(1, Relaxed);
        ahead_bytes.fetch_add(bytes, Relaxed);
    }

    /// `bytes` more copied for a call ([`Metrics::bytes_copied`]).
    pub(crate) fn copied(&self, bytes: u64) {
        self.atomics.bytes_copied.fetch_add(bytes, Relaxed);
    }

    /// One more copy between a texture over a buffer and its image of its own, of `bytes`
    /// ([`Metrics::over_copies`], [`Metrics::over_copy_bytes`]), which count in
    /// [`Metrics::bytes_copied`] too, and where the bytes shaders reach of the buffer are in an
    /// imported block (`imported`), in [`Metrics::over_copies_imported`] and
    /// [`Metrics::over_copy_bytes_imported`]; and in the pair of the texture's `cause`.
    pub(crate) fn over_copied(&self, bytes: u64, imported: bool, cause: OverCause) {
        let a = &self.atomics;
        a.over_copies.fetch_add(1, Relaxed);
        a.over_copy_bytes.fetch_add(bytes, Relaxed);
        a.bytes_copied.fetch_add(bytes, Relaxed);
        if imported {
            a.over_copies_imported.fetch_add(1, Relaxed);
            a.over_copy_bytes_imported.fetch_add(bytes, Relaxed);
        }
        let [copies, copy_bytes] = &a.over_by_cause[cause as usize];
        copies.fetch_add(1, Relaxed);
        copy_bytes.fetch_add(bytes, Relaxed);
    }

    /// `copies` more copies of attachments' images into their shadows, of `bytes`
    /// ([`Metrics::attachment_read_copies`], [`Metrics::attachment_read_copy_bytes`]), which
    /// count in [`Metrics::bytes_copied`] too.
    pub(crate) fn attachment_read_copied(&self, copies: u64, bytes: u64) {
        let a = &self.atomics;
        a.attachment_read_copies.fetch_add(copies, Relaxed);
        a.attachment_read_copy_bytes.fetch_add(bytes, Relaxed);
        a.bytes_copied.fetch_add(bytes, Relaxed);
    }

    /// `bytes` more of a new buffer zeroed ([`Metrics::bytes_zeroed`]).
    pub(crate) fn zeroed(&self, bytes: u64) {
        self.atomics.bytes_zeroed.fetch_add(bytes, Relaxed);
    }

    /// A take carved from a long allocation made for another take, of a Shared one where `shared`
    /// ([`Metrics::shared_long_reuses`]).
    pub(crate) fn long_reuse(&self, shared: bool) {
        let a = &self.atomics;
        let atomic = if shared {
            &a.shared_long_reuses
        } else {
            &a.private_long_reuses
        };
        atomic.fetch_add(1, Relaxed);
    }

    /// A `vkQueueSubmit2` call of `nanos` nanoseconds that met one of the block-maker thread's
    /// frees ([`Metrics::submit_call_nanos_during_frees`]).
    pub(crate) fn submit_call_during_free(&self, nanos: u64) {
        (self.atomics.submit_call_nanos_during_frees).fetch_add(nanos, Relaxed);
    }

    /// An allocation of the pool, of the Shared class where `shared`, made within a second after
    /// the block-maker thread freed one of its class ([`Metrics::shared_remade_after_frees`]).
    pub(crate) fn remade_after_free(&self, shared: bool) {
        let a = &self.atomics;
        let atomic = if shared {
            &a.shared_remade_after_frees
        } else {
            &a.private_remade_after_frees
        };
        atomic.fetch_add(1, Relaxed);
    }

    /// An allocation the consumer's hook mapped, inside a take where `on_take`
    /// ([`Metrics::allocations_mapped`]).
    pub(crate) fn mapped(&self, on_take: bool) {
        let a = &self.atomics;
        let atomic = if on_take {
            &a.allocations_mapped_on_takes
        } else {
            &a.allocations_mapped
        };
        atomic.fetch_add(1, Relaxed);
    }

    /// An allocation the consumer's hook did not map ([`Metrics::allocation_map_refusals`]).
    pub(crate) fn map_refused(&self) {
        self.atomics.allocation_map_refusals.fetch_add(1, Relaxed);
    }

    /// A take carved from a free range ([`Metrics::takes_from_free_ranges`]).
    pub(crate) fn carved(&self) {
        self.atomics.takes_from_free_ranges.fetch_add(1, Relaxed);
    }

    /// A zero fill of `bytes` the consumer asked of a Private block
    /// ([`Metrics::block_zero_fills`]).
    pub(crate) fn zero_filled(&self, bytes: u64) {
        let a = &self.atomics;
        a.block_zero_fills.fetch_add(1, Relaxed);
        a.block_zero_fill_bytes.fetch_add(bytes, Relaxed);
    }

    /// A new texture's image of `bytes` of memory zeroed in its move home
    /// ([`Metrics::texture_clears`]).
    pub(crate) fn texture_cleared(&self, bytes: u64) {
        let a = &self.atomics;
        a.texture_clears.fetch_add(1, Relaxed);
        a.texture_clear_bytes.fetch_add(bytes, Relaxed);
    }

    /// `bytes` more of Managed buffers' bytes a commit's copy-in copies
    /// ([`Metrics::managed_bytes_captured`]).
    pub(crate) fn captured(&self, bytes: u64) {
        self.atomics
            .managed_bytes_captured
            .fetch_add(bytes, Relaxed);
    }

    /// One more release of `bytes` while an upload not yet submitted held it
    /// ([`Metrics::held_releases`], [`Metrics::held_release_bytes`]), and one that would start a
    /// send where `starts` ([`Metrics::held_release_starts`]).
    pub(crate) fn held_release(&self, bytes: u64, starts: bool) {
        let a = &self.atomics;
        a.held_releases.fetch_add(1, Relaxed);
        a.held_release_bytes.fetch_add(bytes, Relaxed);
        if starts {
            a.held_release_starts.fetch_add(1, Relaxed);
        }
    }

    /// `bytes` more written to the cache directory ([`Metrics::cache_bytes_written`]).
    pub(crate) fn cache_written(&self, bytes: u64) {
        self.atomics.cache_bytes_written.fetch_add(bytes, Relaxed);
    }

    /// One more segment end whose barrier reaches the host.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn reach_host(&self) {
        self.atomics.host_reaches.fetch_add(1, Relaxed);
    }

    /// The segment ends whose barrier reached the host.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn host_reaches(&self) -> u64 {
        self.atomics.host_reaches.load(Relaxed)
    }

    /// One more calibration of the GPU's clock against the host's.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn calibrated(&self) {
        self.atomics.calibrations.fetch_add(1, Relaxed);
    }

    /// The calibrations made.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn calibrations(&self) -> u64 {
        self.atomics.calibrations.load(Relaxed)
    }

    /// One more barrier that ended a blit of an image, holding `globals` global memory barriers.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn closed(&self, globals: u32) {
        let closings = &self.atomics.closings;
        closings[0].fetch_add(1, Relaxed);
        closings[1].fetch_add(u64::from(globals), Relaxed);
    }

    /// The barriers that ended a blit of an image, and the global memory barriers they held.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn closings(&self) -> (u64, u64) {
        let closings = &self.atomics.closings;
        (closings[0].load(Relaxed), closings[1].load(Relaxed))
    }

    /// One more live resource of `gauge`'s kind.
    pub(crate) fn rise(&self, gauge: Gauge) {
        self.gauge(gauge).fetch_add(1, Relaxed);
    }

    /// One more live resource of `gauge`'s kind where fewer than `max` live; false, and no
    /// change, where `max` do.
    pub(crate) fn rise_below(&self, gauge: Gauge, max: u64) -> bool {
        (self.gauge(gauge))
            .try_update(Relaxed, Relaxed, |live| (live < max).then_some(live + 1))
            .is_ok()
    }

    /// One fewer live resource of `gauge`'s kind.
    pub(crate) fn fall(&self, gauge: Gauge) {
        self.gauge(gauge).fetch_sub(1, Relaxed);
    }

    /// `amount` more of `gauge`'s kind: bytes, for [`Gauge::SpilledBytes`] and the idle bytes.
    pub(crate) fn rise_by(&self, gauge: Gauge, amount: u64) {
        self.gauge(gauge).fetch_add(amount, Relaxed);
    }

    /// `amount` fewer of `gauge`'s kind.
    pub(crate) fn fall_by(&self, gauge: Gauge, amount: u64) {
        self.gauge(gauge).fetch_sub(amount, Relaxed);
    }

    /// `gauge` at `value`: what an object the device holds counts itself, read after it moved.
    pub(crate) fn set(&self, gauge: Gauge, value: u64) {
        self.gauge(gauge).store(value, Relaxed);
    }

    fn gauge(&self, gauge: Gauge) -> &AtomicU64 {
        let a = &self.atomics;
        match gauge {
            Gauge::HandlesOut => &a.handles_out,
            Gauge::Buffers => &a.buffers,
            Gauge::Textures => &a.textures,
            Gauge::Samplers => &a.samplers,
            Gauge::DescriptorSlots => &a.descriptor_slots,
            Gauge::Pipelines => &a.pipelines,
            Gauge::Kernels => &a.kernels,
            Gauge::KernelEntries => &a.kernel_entries,
            Gauge::RenderContents => &a.render_contents,
            Gauge::AttachmentViews => &a.attachment_views,
            Gauge::Blocks => &a.blocks,
            Gauge::CommandBuffers => &a.command_buffers,
            Gauge::Events => &a.events,
            Gauge::DeviceSamplers => &a.device_samplers,
            Gauge::SpilledBytes => &a.spilled_bytes,
            Gauge::ManagedSpilledBytes => &a.managed_spilled_bytes,
            Gauge::SharedAllocationBytes => &a.shared_allocation_bytes,
            Gauge::PrivateAllocationBytes => &a.private_allocation_bytes,
            Gauge::SharedIdleBytes => &a.shared_idle_bytes,
            Gauge::PrivateIdleBytes => &a.private_idle_bytes,
            Gauge::StagingBytes => &a.staging_bytes,
            Gauge::StageEntries => &a.stage_entries,
            Gauge::StageEntryBytes => &a.stage_entry_bytes,
            Gauge::PipelineCacheBytes => &a.pipeline_cache_bytes,
            Gauge::Shadows => &a.shadows,
            Gauge::ShadowBytes => &a.shadow_bytes,
            Gauge::TexturesCopiedOver => &a.textures_copied_over,
            Gauge::TimestampPools => &a.timestamp_pools,
            Gauge::Imports => &a.imports,
            Gauge::Exports => &a.exports,
            Gauge::SurfaceImports => &a.surface_imports,
            Gauge::SurfaceBuffers => &a.surface_buffers,
            Gauge::TextureBlockBytes => &a.texture_block_bytes,
            Gauge::TextureBlockHeldBytes => &a.texture_block_held_bytes,
            Gauge::TextureBlockSparseBytes => &a.texture_block_sparse_bytes,
            Gauge::TextureBlockSparseBlocks => &a.texture_block_sparse_blocks,
            Gauge::PrivateTakeHeldBytes => &a.private_take_held_bytes,
            Gauge::PrivateTakeBytes => &a.private_take_bytes,
            Gauge::PrivateTakeSparseBytes => &a.private_take_sparse_bytes,
            Gauge::PrivateTakeSparseAllocations => &a.private_take_sparse_allocations,
            Gauge::PrivateUnits => &a.private_units,
            Gauge::PrivateUnitUsedBytes => &a.private_unit_used_bytes,
            Gauge::TextureOwnBytes => &a.texture_own_bytes,
            Gauge::SharedBufferBytes => &a.shared_buffer_bytes,
            Gauge::PrivateBufferBytes => &a.private_buffer_bytes,
            Gauge::SharedPlacedBytes => &a.shared_placed_bytes,
            Gauge::PrivatePlacedBytes => &a.private_placed_bytes,
            Gauge::ImportBytes => &a.import_bytes,
            Gauge::SurfaceImportBytes => &a.surface_import_bytes,
        }
    }

    /// No Metal counterpart: the cumulative counts since the device was made.
    pub fn metrics(&self) -> Metrics {
        let a = &self.atomics;
        let site = |site: Site| {
            let [count, bytes, longest, ahead, ahead_bytes] = &a.sites[site as usize];
            let record =
                *(a.site_longest[site as usize].lock()).unwrap_or_else(PoisonError::into_inner);
            SiteAllocations {
                count: count.load(Relaxed),
                bytes: bytes.load(Relaxed),
                longest_nanos: longest.load(Relaxed),
                longest_bytes: record.bytes,
                takes_ahead: ahead.load(Relaxed),
                takes_ahead_bytes: ahead_bytes.load(Relaxed),
            }
        };
        let first_use =
            |site: FirstUse| (a.first_use_sites[site as usize].each_ref()).map(|n| n.load(Relaxed));
        let [variant, lines, points] =
            [FirstUse::Variant, FirstUse::Lines, FirstUse::Points].map(first_use);
        let made_ahead = (a.made_ahead.each_ref()).map(|n| n.load(Relaxed));
        let part = |part: Part| {
            let [count, bytes, total, _, over_1, over_5, over_20] = &a.parts[part as usize];
            let longest =
                *(a.longest[part as usize].lock()).unwrap_or_else(PoisonError::into_inner);
            Timed {
                count: count.load(Relaxed),
                bytes: bytes.load(Relaxed),
                total_nanos: total.load(Relaxed),
                longest_nanos: longest.nanos,
                longest_bytes: longest.bytes,
                longest_made: longest.made,
                over_1_ms: over_1.load(Relaxed),
                over_5_ms: over_5.load(Relaxed),
                over_20_ms: over_20.load(Relaxed),
            }
        };
        let over = |cause: OverCause| {
            (a.over_by_cause[cause as usize].each_ref()).map(|n| n.load(Relaxed))
        };
        let stage = |stage: Stage| (a.stages[stage as usize].each_ref()).map(|n| n.load(Relaxed));
        Metrics {
            submissions: a.submissions.load(Relaxed),
            submit_infos: a.submit_infos.load(Relaxed),
            submit_call_nanos: a.submit_call_nanos.load(Relaxed),
            submit_call_max_nanos: a.submit_call_max_nanos.load(Relaxed),
            pipeline_creation_nanos: a.pipeline_creation_nanos.load(Relaxed),
            translation_nanos: a.translation_nanos.load(Relaxed),
            submit_calls_over_100_ms: a.submit_calls_over_100_ms.load(Relaxed),
            queue_lock_wait_nanos: a.queue_lock_wait_nanos.load(Relaxed),
            queue_lock_wait_max_nanos: a.queue_lock_wait_max_nanos.load(Relaxed),
            commit_to_scheduled: stage(Stage::CommitToScheduled),
            scheduled_to_gpu_start: stage(Stage::ScheduledToGpuStart),
            gpu_run: stage(Stage::GpuRun),
            gpu_end_to_handler: stage(Stage::GpuEndToHandler),
            event_holds: (a.event_holds.each_ref()).map(|n| n.load(Relaxed)),
            held_starts: a.held_starts.load(Relaxed),
            held_start_gap_nanos: a.held_start_gap_nanos.load(Relaxed),
            passes: a.passes.load(Relaxed),
            barriers: a.barriers.load(Relaxed),
            bytes_copied: a.bytes_copied.load(Relaxed),
            bytes_zeroed: a.bytes_zeroed.load(Relaxed),
            takes_from_free_ranges: a.takes_from_free_ranges.load(Relaxed),
            block_zero_fills: a.block_zero_fills.load(Relaxed),
            block_zero_fill_bytes: a.block_zero_fill_bytes.load(Relaxed),
            texture_clears: a.texture_clears.load(Relaxed),
            texture_clear_bytes: a.texture_clear_bytes.load(Relaxed),
            texture_clear_barriers: a.texture_clear_barriers.load(Relaxed),
            pipelines_created: a.pipelines_created.load(Relaxed),
            first_use_waits: a.first_use_waits.load(Relaxed),
            first_use_wait_nanos: a.first_use_wait_nanos.load(Relaxed),
            first_use_wait_max_nanos: a.first_use_wait_max_nanos.load(Relaxed),
            first_use_variant_waits: variant[0],
            first_use_variant_wait_nanos: variant[1],
            first_use_variant_wait_max_nanos: variant[2],
            first_use_line_waits: lines[0],
            first_use_line_wait_nanos: lines[1],
            first_use_line_wait_max_nanos: lines[2],
            first_use_point_waits: points[0],
            first_use_point_wait_nanos: points[1],
            first_use_point_wait_max_nanos: points[2],
            made_ahead_variants: made_ahead[0],
            made_ahead_variant_nanos: made_ahead[1],
            blocks_made: a.blocks_made.load(Relaxed),
            block_bytes_made: a.block_bytes_made.load(Relaxed),
            consumer_allocations: a.consumer_allocations.load(Relaxed),
            consumer_allocation_nanos: a.consumer_allocation_nanos.load(Relaxed),
            consumer_allocation_max_nanos: a.consumer_allocation_max_nanos.load(Relaxed),
            maker_allocation_nanos: a.maker_allocation_nanos.load(Relaxed),
            texture_allocations: site(Site::Texture),
            texture_block_allocations: site(Site::TextureBlock),
            ring_block_allocations: site(Site::RingBlock),
            staging_region_allocations: site(Site::StagingRegion),
            staging_buffer_allocations: site(Site::StagingBuffer),
            zero_source_allocations: site(Site::ZeroSource),
            shared_take_allocations: site(Site::SharedTake),
            private_take_allocations: site(Site::PrivateTake),
            maker_waits: a.maker_waits.load(Relaxed),
            maker_wait_max_nanos: a.maker_wait_max_nanos.load(Relaxed),
            maker_memory_calls: a.maker_memory_calls.load(Relaxed),
            maker_yields: a.maker_yields.load(Relaxed),
            maker_yield_nanos: a.maker_yield_nanos.load(Relaxed),
            shared_long_reuses: a.shared_long_reuses.load(Relaxed),
            private_long_reuses: a.private_long_reuses.load(Relaxed),
            allocations_mapped: a.allocations_mapped.load(Relaxed),
            allocations_mapped_on_takes: a.allocations_mapped_on_takes.load(Relaxed),
            allocation_map_refusals: a.allocation_map_refusals.load(Relaxed),
            dispatches_without_effects: a.dispatches_without_effects.load(Relaxed),
            translations: a.translations.load(Relaxed),
            translations_loaded: a.translations_loaded.load(Relaxed),
            render_records_read: a.render_records_read.load(Relaxed),
            render_records_stale: a.render_records_stale.load(Relaxed),
            creations_before_seed: a.creations_before_seed.load(Relaxed),
            pipeline_cache_looks_skipped: a.pipeline_cache_looks_skipped.load(Relaxed),
            pipeline_cache_lock_waits: a.pipeline_cache_lock_waits.load(Relaxed),
            pipeline_cache_lock_wait_nanos: a.pipeline_cache_lock_wait_nanos.load(Relaxed),
            pipeline_cache_lock_wait_max_nanos: a.pipeline_cache_lock_wait_max_nanos.load(Relaxed),
            cache_writes_dropped: a.cache_writes_dropped.load(Relaxed),
            cache_bytes_written: a.cache_bytes_written.load(Relaxed),
            ring_writes: a.ring_writes.load(Relaxed),
            pushes: a.pushes.load(Relaxed),
            pushes_held: a.pushes_held.load(Relaxed),
            upload_waits: a.upload_waits.load(Relaxed),
            staging_sleeps: a.staging_sleeps.load(Relaxed),
            synchronizations: a.synchronizations.load(Relaxed),
            commit_lock_wait_nanos: a.commit_lock_wait_nanos.load(Relaxed),
            commit_lock_wait_max_nanos: a.commit_lock_wait_max_nanos.load(Relaxed),
            over_copies: a.over_copies.load(Relaxed),
            over_copy_bytes: a.over_copy_bytes.load(Relaxed),
            over_copies_imported: a.over_copies_imported.load(Relaxed),
            over_copy_bytes_imported: a.over_copy_bytes_imported.load(Relaxed),
            over_copies_host_memory: over(OverCause::HostMemory)[0],
            over_copy_bytes_host_memory: over(OverCause::HostMemory)[1],
            over_copies_dma_buf: over(OverCause::DmaBuf)[0],
            over_copy_bytes_dma_buf: over(OverCause::DmaBuf)[1],
            over_copies_dma_bufs: over(OverCause::DmaBufs)[0],
            over_copy_bytes_dma_bufs: over(OverCause::DmaBufs)[1],
            over_copies_dedicated: over(OverCause::Dedicated)[0],
            over_copy_bytes_dedicated: over(OverCause::Dedicated)[1],
            over_copies_surface: over(OverCause::Surface)[0],
            over_copy_bytes_surface: over(OverCause::Surface)[1],
            over_copies_spilled: over(OverCause::Spilled)[0],
            over_copy_bytes_spilled: over(OverCause::Spilled)[1],
            over_copies_linear: over(OverCause::Linear)[0],
            over_copy_bytes_linear: over(OverCause::Linear)[1],
            over_copies_linear_features: over(OverCause::LinearFeatures)[0],
            over_copy_bytes_linear_features: over(OverCause::LinearFeatures)[1],
            over_copies_pitch: over(OverCause::Pitch)[0],
            over_copy_bytes_pitch: over(OverCause::Pitch)[1],
            over_copies_render_reach: over(OverCause::RenderReach)[0],
            over_copy_bytes_render_reach: over(OverCause::RenderReach)[1],
            over_copies_alignment: over(OverCause::Alignment)[0],
            over_copy_bytes_alignment: over(OverCause::Alignment)[1],
            over_copies_image_memory: over(OverCause::ImageMemory)[0],
            over_copy_bytes_image_memory: over(OverCause::ImageMemory)[1],
            attachment_read_copies: a.attachment_read_copies.load(Relaxed),
            attachment_read_copy_bytes: a.attachment_read_copy_bytes.load(Relaxed),
            attachment_loop_reads: a.attachment_loop_reads.load(Relaxed),
            allocations_freed: a.allocations_freed.load(Relaxed),
            submissions_during_frees: a.submissions_during_frees.load(Relaxed),
            submit_call_nanos_during_frees: a.submit_call_nanos_during_frees.load(Relaxed),
            shared_remade_after_frees: a.shared_remade_after_frees.load(Relaxed),
            private_remade_after_frees: a.private_remade_after_frees.load(Relaxed),
            idle_frees_by_age: std::array::from_fn(|at| a.idle_frees_by_age[at].load(Relaxed)),
            idle_reuses_by_age: std::array::from_fn(|at| a.idle_reuses_by_age[at].load(Relaxed)),
            managed_bytes_captured: a.managed_bytes_captured.load(Relaxed),
            held_releases: a.held_releases.load(Relaxed),
            held_release_bytes: a.held_release_bytes.load(Relaxed),
            held_release_starts: a.held_release_starts.load(Relaxed),
            untimed_command_buffers: a.untimed_command_buffers.load(Relaxed),
            sampler_element_writes: a.sampler_element_writes.load(Relaxed),
            stamps_not_available: a.stamps_not_available.load(Relaxed),
            runtime_command_buffers: a.runtime_command_buffers.load(Relaxed),
            runtime_command_buffer_gpu_nanos: a.runtime_command_buffer_gpu_nanos.load(Relaxed),
            waits_behind_counter: a.waits_behind_counter.load(Relaxed),
            external_acquires: a.external_acquires.load(Relaxed),
            external_releases: a.external_releases.load(Relaxed),
            surfaces_in_place: a.surfaces_in_place.load(Relaxed),
            surfaces_copied: a.surfaces_copied.load(Relaxed),
            surface_waits: a.surface_waits.load(Relaxed),
            surface_wait_nanos: a.surface_wait_nanos.load(Relaxed),
            surface_wait_max_nanos: a.surface_wait_max_nanos.load(Relaxed),
            surface_wait_submissions: a.surface_wait_submissions.load(Relaxed),
            sparse_binds: a.sparse_binds.load(Relaxed),
            resolves: a.resolves.load(Relaxed),
            takes: part(Part::Take),
            take_lock_waits: part(Part::TakeLock),
            take_makes: part(Part::TakeMake),
            take_maps: part(Part::TakeMap),
            maker_makes: part(Part::MakerMake),
            zero_writes: part(Part::ZeroWrite),
            long_take_zero_writes: part(Part::LongTakeZeroWrite),
            export_makes: part(Part::ExportMake),
            export_allocations: part(Part::ExportAllocation),
            export_handles: part(Part::ExportHandle),
            export_claims: part(Part::ExportClaim),
        }
    }

    /// Holds a slot of [`Gauges::threads`] for the calling thread, a thread the runtime started,
    /// until the guard drops as it ends.
    pub(crate) fn thread_started(&self) -> ThreadSlot {
        let id = os_thread_id();
        let slot = (self.atomics.threads.iter())
            .position(|slot| slot.compare_exchange(0, id, Relaxed, Relaxed).is_ok());
        ThreadSlot {
            atomics: Arc::clone(&self.atomics),
            slot,
        }
    }

    /// No Metal counterpart: the live count of each kind of resource.
    pub fn gauges(&self) -> Gauges {
        let a = &self.atomics;
        let now = crate::host_time();
        let age = |since: &AtomicU64| match since.load(Relaxed) {
            0 => 0,
            since => now.saturating_sub(since),
        };
        Gauges {
            threads: std::array::from_fn(|slot| a.threads[slot].load(Relaxed)),
            handles_out: a.handles_out.load(Relaxed),
            buffers: a.buffers.load(Relaxed),
            textures: a.textures.load(Relaxed),
            samplers: a.samplers.load(Relaxed),
            pipelines: a.pipelines.load(Relaxed),
            kernels: a.kernels.load(Relaxed),
            kernel_entries: a.kernel_entries.load(Relaxed),
            render_contents: a.render_contents.load(Relaxed),
            attachment_views: a.attachment_views.load(Relaxed),
            command_buffers: a.command_buffers.load(Relaxed),
            descriptor_slots: a.descriptor_slots.load(Relaxed),
            blocks: a.blocks.load(Relaxed),
            events: a.events.load(Relaxed),
            device_samplers: a.device_samplers.load(Relaxed),
            spilled_bytes: a.spilled_bytes.load(Relaxed),
            managed_spilled_bytes: a.managed_spilled_bytes.load(Relaxed),
            shared_allocation_bytes: a.shared_allocation_bytes.load(Relaxed),
            private_allocation_bytes: a.private_allocation_bytes.load(Relaxed),
            shared_idle_bytes: a.shared_idle_bytes.load(Relaxed),
            private_idle_bytes: a.private_idle_bytes.load(Relaxed),
            shared_idle_age_nanos: age(&a.shared_idle_since),
            private_idle_age_nanos: age(&a.private_idle_since),
            shared_reserve_bytes: a.shared_reserve_bytes.load(Relaxed),
            private_reserve_bytes: a.private_reserve_bytes.load(Relaxed),
            shared_long_reused_bytes: a.shared_long_reused_bytes.load(Relaxed),
            private_long_reused_bytes: a.private_long_reused_bytes.load(Relaxed),
            staging_bytes: a.staging_bytes.load(Relaxed),
            stage_entries: a.stage_entries.load(Relaxed),
            stage_entry_bytes: a.stage_entry_bytes.load(Relaxed),
            pipeline_cache_bytes: a.pipeline_cache_bytes.load(Relaxed),
            shadows: a.shadows.load(Relaxed),
            shadow_bytes: a.shadow_bytes.load(Relaxed),
            textures_copied_over: a.textures_copied_over.load(Relaxed),
            timestamp_pools: a.timestamp_pools.load(Relaxed),
            imports: a.imports.load(Relaxed),
            exports: a.exports.load(Relaxed),
            surface_imports: a.surface_imports.load(Relaxed),
            surface_buffers: a.surface_buffers.load(Relaxed),
            texture_block_bytes: a.texture_block_bytes.load(Relaxed),
            texture_block_held_bytes: a.texture_block_held_bytes.load(Relaxed),
            texture_block_sparse_bytes: a.texture_block_sparse_bytes.load(Relaxed),
            texture_block_sparse_blocks: a.texture_block_sparse_blocks.load(Relaxed),
            private_take_held_bytes: a.private_take_held_bytes.load(Relaxed),
            private_take_bytes: a.private_take_bytes.load(Relaxed),
            private_take_sparse_bytes: a.private_take_sparse_bytes.load(Relaxed),
            private_take_sparse_allocations: a.private_take_sparse_allocations.load(Relaxed),
            private_units: a.private_units.load(Relaxed),
            private_unit_used_bytes: a.private_unit_used_bytes.load(Relaxed),
            texture_own_bytes: a.texture_own_bytes.load(Relaxed),
            shared_buffer_bytes: a.shared_buffer_bytes.load(Relaxed),
            private_buffer_bytes: a.private_buffer_bytes.load(Relaxed),
            shared_placed_bytes: a.shared_placed_bytes.load(Relaxed),
            private_placed_bytes: a.private_placed_bytes.load(Relaxed),
            import_bytes: a.import_bytes.load(Relaxed),
            surface_import_bytes: a.surface_import_bytes.load(Relaxed),
        }
    }
}

/// No Metal counterpart: cumulative counts since the device was made. The difference of two reads
/// is what happened between them, a frame's or a command buffer's.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Metrics {
    /// `vkQueueSubmit2` calls.
    pub submissions: u64,
    /// The `VkSubmitInfo2`s those calls passed: the command buffers of a submission that no event
    /// signal separates share one, and the upload shares the first.
    pub submit_infos: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) those calls took, from each call to
    /// its return, in total: the time inside the driver, which a driver may spend holding the call
    /// while another thread is inside a memory call
    /// ([`submissions_during_frees`](Self::submissions_during_frees)). The wait for the queue's
    /// lock before them is [`queue_lock_wait_nanos`](Self::queue_lock_wait_nanos).
    pub submit_call_nanos: u64,
    /// The longest such call, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub submit_call_max_nanos: u64,
    /// Those calls longer than 100 ms.
    pub submit_calls_over_100_ms: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) the submit thread waited for the
    /// runtime's queue lock before a submission's `vkQueueSubmit2` calls, in total: one wait a
    /// submission, whose calls then hold the lock. A sparse bind of an import takes the lock too.
    /// Where the queue family has one queue, the lock is also
    /// [`Device::with_present_queue`](crate::Device::with_present_queue)'s, so a wait here is
    /// also the submit thread's wait behind the consumer's presents.
    pub queue_lock_wait_nanos: u64,
    /// The longest such wait, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub queue_lock_wait_max_nanos: u64,
    /// Command buffers by the span from their commit to their scheduling, in nanoseconds of
    /// [`host_time`](crate::host_time), in 23 buckets of powers of two: bucket 0 counts spans
    /// under 2 us, bucket `i` from 1 to 21 spans from 2^i us to under 2^(i+1) us, and bucket 22
    /// spans of 2^22 us (about 4.19 s) or more. The span runs from the host time at the commit to
    /// the [`kernel_end_time`](crate::CommandBuffer::kernel_end_time): its last segment joining a
    /// submission, passed to `vkQueueSubmit2` at once or held in its queue's group, or for one
    /// that presents a drawable, that call's return. Counted as each is scheduled; one that ends
    /// unscheduled counts none.
    pub commit_to_scheduled: [u64; STAGE_BUCKETS],
    /// Command buffers that completed with GPU times by the span from their
    /// [`kernel_end_time`](crate::CommandBuffer::kernel_end_time) to their
    /// [`gpu_start_time`](crate::CommandBuffer::gpu_start_time), in the buckets of
    /// [`commit_to_scheduled`](Self::commit_to_scheduled); 0 where the GPU start is placed before
    /// the kernel end, as for one that presents, scheduled once its call returned. Counted on the
    /// completion thread as each retires, before its handlers run; one that ends with an error,
    /// or completes without GPU times, counts none here or in the two below.
    pub scheduled_to_gpu_start: [u64; STAGE_BUCKETS],
    /// The same command buffers by their GPU run: the span from their
    /// [`gpu_start_time`](crate::CommandBuffer::gpu_start_time) to their
    /// [`gpu_end_time`](crate::CommandBuffer::gpu_end_time), their two timestamps' own span, in
    /// the buckets of [`commit_to_scheduled`](Self::commit_to_scheduled); 0 for one with nothing
    /// for the GPU.
    pub gpu_run: [u64; STAGE_BUCKETS],
    /// The same command buffers by the span from their
    /// [`gpu_end_time`](crate::CommandBuffer::gpu_end_time) to the host time their handlers
    /// begin, whether they have any or not, in the buckets of
    /// [`commit_to_scheduled`](Self::commit_to_scheduled). The completion thread retires a
    /// round's command buffers in order, so the span holds the earlier ones' retirements and
    /// handlers in the round.
    pub gpu_end_to_handler: [u64; STAGE_BUCKETS],
    /// Segments the submit thread held for an event's value that neither the host had set nor a
    /// signal submitted to the driver promised: a value the host sets later, or one a command
    /// buffer not yet submitted signals. A wait whose signal is submitted is not held: it goes to
    /// the driver with its segment's submission. By the span from when the submit thread, with
    /// nothing left to stage, found the segment waiting, to its staging once the value was set or
    /// its signal submitted, in the buckets of
    /// [`commit_to_scheduled`](Self::commit_to_scheduled). Counted as each is staged; one that
    /// ends unstaged counts none.
    pub event_holds: [u64; STAGE_BUCKETS],
    /// Submissions of a group that a queue held, its three submissions before it not yet seen
    /// complete as the submit thread finished a pass, and that went once a completion left fewer
    /// than three of them running: counted as each goes, before its `vkQueueSubmit2` calls. A
    /// group that goes while three are running (at its sixteenth command buffer, an event's
    /// signal, a host wait or [`CommandQueue::send_now`](crate::CommandQueue::send_now)) counts
    /// none, and so does a command buffer sent at once.
    pub held_starts: u64,
    /// The GPU's nanoseconds, in total, from the latest end of the command buffers submitted
    /// before each of those submissions to the start of its own first, where the start is the
    /// later: on their timestamps as the GPU wrote them, the end of every command buffer that
    /// completed with both and the start of the first of the submission's that did, their span
    /// in `timestampPeriod`s. It holds the run of the uploads the submission carries ahead of its
    /// command buffers. Counted on the completion thread as that command buffer retires, before
    /// its handlers run; a submission with no such command buffer, or with none before it,
    /// counts none.
    pub held_start_gap_nanos: u64,
    /// Encoders recorded: compute, blit and render.
    pub passes: u64,
    /// Pipeline-barrier commands recorded into command buffers' work: one at the end of each
    /// segment with commands and of each upload, one before a dispatch at a hazard, one owed by
    /// an encoder that wrote or a memory barrier call, recorded before the next command, two
    /// around each blit copy, fill and synchronize copy, and two around each render pass, the
    /// first of each pair paying what is owed, with one inside a render pass at a memory barrier
    /// after a draw that wrote memory, one at a draw that moves an attachment's image to the
    /// feedback layout, one at a draw that reads an attachment after a memory barrier owed to it,
    /// and two around the copies a draw that reads an attachment makes into its image's shadow
    /// ([`attachment_read_copies`](Self::attachment_read_copies)), or the clear of the shadow
    /// that stands for a copy of what its pass cleared, one for each acquire of an exported
    /// block ([`external_acquires`](Self::external_acquires)), and one for each release of one
    /// ([`external_releases`](Self::external_releases)). Any other draw takes none. An upload
    /// adds one where it moves new textures' images to their home layouts, one more after the
    /// zeroing of those it zeroes ([`texture_clear_barriers`](Self::texture_clear_barriers)), two
    /// around each copy between a texture's image and the staging region or a named buffer
    /// (`replaceRegion` and `getBytes`), and one before a zero fill of bytes an earlier fill of the
    /// upload wrote (a new buffer's zeroed bytes, or a range
    /// [`Device::zero_block_range`](crate::Device::zero_block_range) zeroes).
    pub barriers: u64,
    /// Bytes the runtime copied for a call: `replace_region` and `get_bytes` staging, the copies
    /// between a texture over a buffer and its image of its own
    /// ([`over_copy_bytes`](Self::over_copy_bytes)), and the copies of attachments' images into their
    /// shadows ([`attachment_read_copy_bytes`](Self::attachment_read_copy_bytes)). A blit's own
    /// copy, and a synchronize's, is the program's and is not counted.
    pub bytes_copied: u64,
    /// Bytes of new buffers zeroed: [`Device::new_buffer`](crate::Device::new_buffer)'s from its
    /// `zero_from` to the buffer's end, and a Managed buffer's device copy's past its copied bytes
    /// ([`Device::new_managed_buffer`](crate::Device::new_managed_buffer)).
    pub bytes_zeroed: u64,
    /// Takes carved from a free range of an allocation already made: under the pool's lock, with
    /// no Vulkan call and no call to the consumer's hook. A take that no free range fits makes an
    /// allocation instead ([`consumer_allocations`](Self::consumer_allocations),
    /// [`allocations_mapped_on_takes`](Self::allocations_mapped_on_takes)).
    pub takes_from_free_ranges: u64,
    /// Zero fills the consumer asked of Private blocks
    /// ([`Device::zero_block_range`](crate::Device::zero_block_range)).
    pub block_zero_fills: u64,
    /// The bytes of those fills.
    pub block_zero_fill_bytes: u64,
    /// New textures' images zeroed in their moves home, as a texture the runtime places reads
    /// zero until written ([`Device::new_texture`](crate::Device::new_texture),
    /// [`Device::new_texture_for_views`](crate::Device::new_texture_for_views) and
    /// [`Device::new_managed_texture`](crate::Device::new_managed_texture)): each image's one
    /// clear of every subresource, or its copies from the zero source where its format takes no
    /// clear, on every device.
    pub texture_clears: u64,
    /// The bytes of memory of those images.
    pub texture_clear_bytes: u64,
    /// The barriers that move new textures' images home after their zeroing: one for each upload
    /// that zeroed any ([`texture_clears`](Self::texture_clears)), however many it zeroed, also
    /// counted in [`barriers`](Self::barriers).
    pub texture_clear_barriers: u64,
    /// `VkPipeline`s created, each counted as its `vkCreateComputePipelines` or
    /// `vkCreateGraphicsPipelines` call returns it: one a state's creation makes for a content the
    /// device does not hold, or one a first use makes
    /// ([`first_use_waits`](Self::first_use_waits)). A state of a content the device holds makes
    /// none, a call still inside the driver is not counted yet, and one that failed is not counted.
    pub pipelines_created: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) the `vkCreateComputePipelines` and
    /// `vkCreateGraphicsPipelines` calls took, from each call to its return, in total, those that
    /// failed among them: the driver's side of a pipeline's make. Calls on several threads at once
    /// each count their own time, so where they overlap the total passes the time that went by.
    pub pipeline_creation_nanos: u64,
    /// First-use waits: encoding calls that waited while the `VkPipeline` they needed was made,
    /// for a workgroup size (a threadgroup size, or the tile of an image-only kernel's grid more
    /// than one deep) or threadgroup memory lengths no state of the kernel's content had run with
    /// before, or for a render content's first draw of lines or of points.
    pub first_use_waits: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) the first-use waits took, in total.
    pub first_use_wait_nanos: u64,
    /// The longest first-use wait, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub first_use_wait_max_nanos: u64,
    /// Of the [`first_use_waits`](Self::first_use_waits), those for a compute kernel's pipeline
    /// made on the encoding thread for a workgroup size, threadgroup memory lengths or tiles none
    /// of its states had run with (`vkCreateComputePipelines`).
    pub first_use_variant_waits: u64,
    /// The nanoseconds of the [`first_use_variant_waits`](Self::first_use_variant_waits), in
    /// total.
    pub first_use_variant_wait_nanos: u64,
    /// The longest of the [`first_use_variant_waits`](Self::first_use_variant_waits), in
    /// nanoseconds; a maximum since the device was made.
    pub first_use_variant_wait_max_nanos: u64,
    /// Of the [`first_use_waits`](Self::first_use_waits), those for a render content's
    /// line-class pipeline, made by its first draw of lines.
    pub first_use_line_waits: u64,
    /// The nanoseconds of the [`first_use_line_waits`](Self::first_use_line_waits), in total.
    pub first_use_line_wait_nanos: u64,
    /// The longest of the [`first_use_line_waits`](Self::first_use_line_waits), in nanoseconds;
    /// a maximum since the device was made.
    pub first_use_line_wait_max_nanos: u64,
    /// Of the [`first_use_waits`](Self::first_use_waits), those for a render content's
    /// point-class pipeline, made by its first draw of points.
    pub first_use_point_waits: u64,
    /// The nanoseconds of the [`first_use_point_waits`](Self::first_use_point_waits), in total.
    pub first_use_point_wait_nanos: u64,
    /// The longest of the [`first_use_point_waits`](Self::first_use_point_waits), in
    /// nanoseconds; a maximum since the device was made.
    pub first_use_point_wait_max_nanos: u64,
    /// The compute kernels' pipelines that
    /// [`prepare_dispatch`](crate::ComputePipelineState::prepare_dispatch) made on its caller's
    /// thread, ahead of the dispatches that run them (`vkCreateComputePipelines`).
    pub made_ahead_variants: u64,
    /// The nanoseconds of the [`made_ahead_variants`](Self::made_ahead_variants), in total.
    pub made_ahead_variant_nanos: u64,
    /// `VkDeviceMemory` allocations made: the allocations blocks are ranges of (the texture
    /// blocks among them), the zero source and the parameter ring's blocks.
    pub blocks_made: u64,
    /// Bytes of the allocations blocks are ranges of, made: what the device allocated for
    /// blocks, by the block-maker thread or in a take. The difference of two reads is what a
    /// frame allocated.
    pub block_bytes_made: u64,
    /// `vkAllocateMemory` calls made on the consumer's threads, inside its calls: takes that no
    /// free range fits, textures' allocations of their own and copies' staging buffers of their
    /// own; and the runtime's fixed-size internals (the zero source, the
    /// parameter ring's blocks and the staging region) only where the block-maker thread's make
    /// of one failed. A driver may hold the device's other calls for an allocation's whole length.
    pub consumer_allocations: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) those calls took, in total.
    pub consumer_allocation_nanos: u64,
    /// The longest of those calls, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub consumer_allocation_max_nanos: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) the allocations made ahead of need
    /// took, in total: the block-maker thread's, off the consumer's threads (the runtime's
    /// fixed-size internals among them), and each class's first allocation in the device's
    /// making.
    pub maker_allocation_nanos: u64,
    /// Of [`consumer_allocations`](Self::consumer_allocations), those of textures' images that
    /// took an allocation of their own, not a place in a texture block: an image the device
    /// requires an allocation of its own for or whose memory types leave out the Private type,
    /// and one made where device-local memory had no texture block left. A texture's shadow and
    /// a counted copy's image are among them.
    pub texture_allocations: SiteAllocations,
    /// Those of texture blocks: a texture's place that no texture block had room for took a
    /// Private block, and no free range of the pool, which the block-maker thread keeps a wholly
    /// free 16 MiB unit of ahead per class, fit it, so the take allocated. Its
    /// [`takes_ahead`](SiteAllocations::takes_ahead) are the texture blocks that landed in such a
    /// unit instead.
    pub texture_block_allocations: SiteAllocations,
    /// Those of the parameter ring's 2 MiB blocks: a write into the ring that found no chunk free
    /// where the block-maker thread, which makes each block ahead of need, made none (its make
    /// failed, or it had stopped).
    pub ring_block_allocations: SiteAllocations,
    /// Those of the staging region, which the block-maker thread makes as its first work: made
    /// by the device's first `replace_region` or `get_bytes` copy of 16 MiB or less only where
    /// that make failed.
    pub staging_region_allocations: SiteAllocations,
    /// Those of staging buffers of their own, one for each `replace_region` or `get_bytes` copy
    /// larger than the 16 MiB staging region.
    pub staging_buffer_allocations: SiteAllocations,
    /// Those of the zero source, the buffer of zeros that Private fills and new block-compressed
    /// textures copy from, which the block-maker thread makes as its first work: made by the
    /// first such fill or texture only where that make failed.
    pub zero_source_allocations: SiteAllocations,
    /// Those of the consumer's takes of Shared blocks
    /// ([`Device::take_block`](crate::Device::take_block)) that no free range fit; its
    /// [`takes_ahead`](SiteAllocations::takes_ahead) are those that landed in a unit the
    /// block-maker thread made ahead instead.
    pub shared_take_allocations: SiteAllocations,
    /// As [`shared_take_allocations`](Self::shared_take_allocations), of the consumer's takes of
    /// Private blocks. Texture blocks' are
    /// [`texture_block_allocations`](Self::texture_block_allocations).
    pub private_take_allocations: SiteAllocations,
    /// Calls that waited for the block-maker thread's make of one of the runtime's fixed-size
    /// internals they need, which it makes off every caller's thread: the zero source, a
    /// parameter ring block or the staging region (the sites above). Such a
    /// wait is a stall of the call as an allocation on its thread would be, made only while the
    /// make runs: in the first moments of the device's life, or where the ring's writes outrun
    /// its next block.
    pub maker_waits: u64,
    /// The longest of those waits, in nanoseconds of [`host_time`](crate::host_time); a maximum
    /// since the device was made, so the difference of two reads does not apply to it.
    pub maker_wait_max_nanos: u64,
    /// `vkAllocateMemory` and `vkFreeMemory` calls made ahead of need, off the consumer's calls:
    /// the block-maker thread's (the runtime's fixed-size internals, the pool's units, and the
    /// frees of idle allocations and of the imported and exported blocks given back), and each
    /// class's first allocation in the device's making. A driver may hold a submit made during
    /// such a call until it returns, and through calls made back to back, so each one starts once
    /// every one of the runtime's `vkQueueSubmit2` calls that was inside the driver as the
    /// previous one returned has returned: a submit waits out at most one of these calls.
    pub maker_memory_calls: u64,
    /// Of [`maker_memory_calls`](Self::maker_memory_calls), those that waited before they started
    /// for a `vkQueueSubmit2` call the previous one met; the others started at once.
    pub maker_yields: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) those waits took, in total.
    pub maker_yield_nanos: u64,
    /// Shared takes carved from a long allocation, one a take longer than the 16 MiB unit made for
    /// itself, after that take: a take of the same length, which a long allocation alone serves.
    pub shared_long_reuses: u64,
    /// As [`shared_long_reuses`](Self::shared_long_reuses), of Private takes.
    pub private_long_reuses: u64,
    /// Allocations of the pool the consumer's
    /// [`BlockMappings::map`](crate::BlockMappings::map) mapped ahead of need: on the block-maker
    /// thread, and each class's first in the device's making, off every take's path.
    pub allocations_mapped: u64,
    /// Allocations of the pool the consumer's hook mapped inside a take, on the consumer's thread:
    /// a take that no free range fits.
    pub allocations_mapped_on_takes: u64,
    /// Allocations the consumer's hook did not map, which the runtime freed with none of their
    /// blocks handed out.
    pub allocation_map_refusals: u64,
    /// Dispatches of a kernel that writes no memory (`PipelineInterface::writes_memory`),
    /// direct and indirect: each was checked as Metal checks it, and nothing was recorded, bound
    /// or read for it, as it has no effect a program can observe.
    pub dispatches_without_effects: u64,
    /// Functions the translator lowered, each a module of a pipeline content the device did not
    /// hold compiled: of a compute content, one for a kernel that does not observe its threadgroup
    /// partition and two for one that does; of a render content, one for each of its functions
    /// that the cache directory did not hold. A state of a content the device holds lowers
    /// nothing.
    pub translations: u64,
    /// Lowered kernels a pipeline state took from the cache directory in place of a lowering.
    pub translations_loaded: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) the runtime's calls into the translator
    /// took, in total, those refused among them: the translation's side of a pipeline's make.
    /// They are the reads of a kernel's inputs and of a render descriptor's variant, which decode
    /// their functions, the lowerings of functions from AIR to SPIR-V
    /// ([`translations`](Self::translations)), and the links and plans of a pipeline's lowered
    /// functions. The digests that key the device's caches count none, and so does a lowering
    /// taken from the cache directory. Calls on several threads at once each count their own
    /// time.
    pub translation_nanos: u64,
    /// Render contents a state took by their record in the cache directory
    /// ([`DeviceOptions::pipeline_cache_dir`]): the keys of the content's lowerings, kept under the
    /// translator's key of its descriptor, which reads no function, so the state read the
    /// lowerings with no decode or specialization of either function. Each lowering it read is in
    /// [`translations_loaded`](Self::translations_loaded) too.
    ///
    /// [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
    pub render_records_read: u64,
    /// Render contents whose record in the cache directory named a lowering the directory no
    /// longer held (a removed generation took it, or it was not written): each took the path of
    /// a content with no record, decoding and specializing its functions, and kept the record
    /// again.
    pub render_records_stale: u64,
    /// `VkPipeline` creations that went through the device's empty pipeline cache while the
    /// reaper read and took in the device's blob ([`DeviceOptions::pipeline_cache_dir`]): each
    /// compiled without the blob's data, and those pipelines are not in this run's blob, so a
    /// later run compiles them again, once. Zero with no directory, or where the device's first
    /// creation came after the seed.
    ///
    /// [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
    pub creations_before_seed: u64,
    /// The reaper's looks at the pipeline cache skipped because a creation began after creation
    /// was quiet: a look reads the cache's size and data only while no creation runs, and never
    /// waits for one, so it looks again once creation is quiet again.
    pub pipeline_cache_looks_skipped: u64,
    /// Creations through the pipeline cache that waited for a look of the reaper's at it: a
    /// creation that begins during a look waits for that look's reads alone. Their time is in
    /// [`pipeline_cache_lock_wait_nanos`](Self::pipeline_cache_lock_wait_nanos), the longest in
    /// [`pipeline_cache_lock_wait_max_nanos`](Self::pipeline_cache_lock_wait_max_nanos).
    pub pipeline_cache_lock_waits: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) creations through the pipeline cache
    /// waited for the reaper's looks at it, in total
    /// ([`pipeline_cache_lock_waits`](Self::pipeline_cache_lock_waits) counts the waits).
    pub pipeline_cache_lock_wait_nanos: u64,
    /// The longest such wait, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub pipeline_cache_lock_wait_max_nanos: u64,
    /// Lowerings, refusals and render contents' records not kept in the cache directory: each was
    /// handed to the reaper, which writes them off every guest thread, while the ones waiting for
    /// it held 8 MiB, and was dropped rather than waited for. A later device lowers it again.
    pub cache_writes_dropped: u64,
    /// Bytes written to the cache directory ([`DeviceOptions::pipeline_cache_dir`]): lowered
    /// kernels, and the device's pipeline cache blob each time it is written: once creation is
    /// quiet where the cache grew by more than an eighth since the last write, and at the
    /// device's drop where it changed.
    ///
    /// [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
    pub cache_bytes_written: u64,
    /// Writes into the parameter ring: a dispatch's parameter block, its fields past the push
    /// window (a kernel with more than 28 buffers), and each `set_bytes` copy, a render
    /// encoder's too.
    pub ring_writes: u64,
    /// `vkCmdPushConstants` recorded: a draw's or a dispatch's push bytes, where the command
    /// buffer did not already hold them.
    pub pushes: u64,
    /// Draws and dispatches whose push bytes the command buffer already held, which recorded no
    /// push: every pipeline shares the device's one layout and push range, so a push stays for the
    /// draws and dispatches after it.
    pub pushes_held: u64,
    /// Calls that waited for an upload to complete before they recorded: the upload at its
    /// bound of calls or bytes, or the staging region with no room for a copy.
    pub upload_waits: u64,
    /// Sleeps of `replace_region` and `get_bytes` calls whose copy found the staging region with
    /// no room while its oldest span was another call's, still writing its bytes or holding them
    /// for a `get_bytes` to read: each a sleep, with no spin, until that call released the span.
    pub staging_sleeps: u64,
    /// `synchronizeResource:` calls: each a copy into the host's bytes of a Managed buffer with a
    /// device copy or of a Managed texture with named bytes, and nothing for any other resource;
    /// none a submission of its own.
    pub synchronizations: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) commits waited to take the submission
    /// lock, which every queue's commits and the submit thread take, in total.
    pub commit_lock_wait_nanos: u64,
    /// The longest such wait, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub commit_lock_wait_max_nanos: u64,
    /// Copies a draw that reads an attachment of its own pass through a texture makes into the
    /// image's shadow first, on a device without the attachment feedback loop, which Metal does
    /// not make: one for each area of a level, a range of its layers and a rectangle, at the
    /// pass's first draw that reads the image and at a draw after a memory barrier after draws,
    /// of what the draw reads, and at another draw, of what it reads that the shadow does not
    /// hold yet. What the pass's first draw reads of the level and layer the pass cleared,
    /// inside Metal's render area, is no copy: the shadow's is cleared to the pass's clear colour.
    pub attachment_read_copies: u64,
    /// The bytes of those copies, which count in [`bytes_copied`](Self::bytes_copied) too.
    pub attachment_read_copy_bytes: u64,
    /// Draws that read an attachment of their own pass through the attachment feedback loop, a
    /// fragment function's colour read (`[[color(n)]]`) or a texture argument over the
    /// attachment: each with the image in the feedback layout, after the barrier that moved it
    /// there or the barrier an earlier draw owed it, counted in [`barriers`](Self::barriers). A
    /// count, no bytes. Metal records no barrier for such a draw. On a device without the loop
    /// such a draw reads the image's shadow instead, its copies in
    /// [`attachment_read_copies`](Self::attachment_read_copies).
    pub attachment_loop_reads: u64,
    /// Copies between a texture over a buffer the device's linear image does not hold in place
    /// and its image of its own ([`Buffer::new_texture`](crate::Buffer::new_texture)): the
    /// buffer's bytes into the image, in the upload that leads a submission or after a command
    /// wrote the buffer, and the image back into the buffer after a command wrote the texture.
    /// The textures held so are [`Gauges::textures_copied_over`].
    pub over_copies: u64,
    /// The bytes of those copies, which count in [`bytes_copied`](Self::bytes_copied) too.
    pub over_copy_bytes: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture whose buffer's bytes are in an
    /// imported block ([`Device::import_host_memory`](crate::Device::import_host_memory),
    /// `Device::import_dma_buf` on Unix); the rest are over a buffer in a
    /// block of the runtime's own, where the device's linear image does not take the texture.
    pub over_copies_imported: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_imported: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer in host memory's import
    /// ([`Device::import_host_memory`](crate::Device::import_host_memory)): Vulkan shares that memory's
    /// ownership with the host, and an image made for it starts undefined, a layout whose move
    /// keeps no bytes.
    pub over_copies_host_memory: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_host_memory: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer in a dma-buf's one import (`Device::import_dma_buf`, on
    /// Unix) on a device without `VK_EXT_queue_family_foreign`, or whose external image query
    /// (`VkPhysicalDeviceExternalImageFormatInfo`) takes no linear image of the texture's format,
    /// usage and size for the handle, or only one dedicated to its memory.
    pub over_copies_dma_buf: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_dma_buf: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer in an import over several dma-bufs
    /// (`Device::import_dma_bufs`, on Unix), whose buffer is sparse.
    pub over_copies_dma_bufs: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_dma_bufs: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer in memory dedicated to one object, which no image of the
    /// runtime's is bound in (`VUID-vkBindImageMemory-memory-10926`): an exported block's
    /// ([`Device::new_exported_block`](crate::Device::new_exported_block)), dedicated to its
    /// buffer; a surface's planes in its copy form lie in one.
    pub over_copies_dedicated: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_dedicated: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over an exported surface's buffer
    /// ([`Surface::buffer`](crate::Surface::buffer)).
    pub over_copies_surface: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_surface: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer in a Private block that spilled: its allocation went to
    /// another memory type than the Private one, the device-local type being out of memory
    /// ([`Gauges::spilled_bytes`](crate::Gauges::spilled_bytes)).
    pub over_copies_spilled: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_spilled: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer the device makes no linear image of, for its format, usage and
    /// size (`vkGetPhysicalDeviceImageFormatProperties2` with linear tiling).
    pub over_copies_linear: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_linear: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer whose format's linear tiling lacks a feature its usage uses
    /// that the format's optimal tiling has (copies, sampling and its linear filter, a shader's
    /// reads, writes and atomics without a format, rendering and blending).
    pub over_copies_linear_features: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_linear_features: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer whose linear image the device lays at another row pitch than
    /// the texture's `bytes_per_row`, or with its subresource past its memory's start
    /// (`vkGetImageSubresourceLayout`).
    pub over_copies_pitch: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_pitch: u64,
    /// Of [`over_copies`](Self::over_copies), those of a render target over a buffer whose linear image's subresource, as the device
    /// advertises it, ends past the texture's own rows (its height times its pitch) or past the
    /// buffer's end: a store, a clear or a resolve may write back any byte of it.
    pub over_copies_render_reach: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_render_reach: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer whose place in its memory is not a multiple of its linear
    /// image's memory alignment (`VkMemoryRequirements::alignment`).
    pub over_copies_alignment: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_alignment: u64,
    /// Of [`over_copies`](Self::over_copies), those of a texture over a buffer whose linear image's other memory requirements its memory does
    /// not meet: the device requires an allocation of its own for it, its memory types exclude
    /// the memory's, or it ends past the memory's end.
    pub over_copies_image_memory: u64,
    /// The bytes of those copies, which count in [`over_copy_bytes`](Self::over_copy_bytes) too.
    pub over_copy_bytes_image_memory: u64,
    /// Idle allocations the block-maker thread freed: one at a time, each once it had been idle,
    /// holding no live block, for 250 ms, whatever the device's queues were doing.
    pub allocations_freed: u64,
    /// `vkQueueSubmit2` calls that met one of the block-maker thread's `vkFreeMemory` calls: made
    /// while it was inside one, or inside the driver as one began. Its frees are an idle
    /// allocation's, an imported or exported block's given back, a texture's allocation of its
    /// own, a surface plane's import, a copy's staging buffer of its own, and one it made and
    /// could not use. A driver may hold such a call until the free returns: on the NVIDIA cards
    /// measured in Thunderbolt enclosures, about 1.1 ms per MiB of host-visible memory. A call
    /// made during a free is counted as it enters the driver, one that a free began beside as it
    /// returns.
    pub submissions_during_frees: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) those calls took, from each call to its
    /// return, in total: the time the driver may have held them behind the frees, with their own
    /// time in the driver. Each call's time is added as it returns.
    pub submit_call_nanos_during_frees: u64,
    /// Allocations of the pool's Shared class, unit or long, made within a second after the
    /// block-maker thread freed an idle allocation of the class: memory given back and taken
    /// again, the pool looping. Each such allocation costs its own `vkAllocateMemory` and, a
    /// second or more later, its free.
    pub shared_remade_after_frees: u64,
    /// As [`shared_remade_after_frees`](Self::shared_remade_after_frees), of the Private class.
    pub private_remade_after_frees: u64,
    /// The frees of [`allocations_freed`](Self::allocations_freed) by how long each allocation had
    /// been idle when its free began, in buckets of under 250 ms (a spilled allocation that
    /// emptied, freed at once), 250 to 275 ms, 275 to 350 ms, 350 to 500 ms, 500 ms to 1 s, and
    /// 1 s or more.
    pub idle_frees_by_age: [u64; 6],
    /// Idle allocations taken again, by how long each had been idle: in buckets of under 50 ms,
    /// 50 to 100 ms, 100 to 250 ms, 250 to 500 ms, 500 ms to 1 s, 1 to 2 s, and 2 s or more. Only
    /// a class's reserve, which the idle rule does not free, is taken again past 250 ms.
    pub idle_reuses_by_age: [u64; 7],
    /// Bytes the commits' copy-ins copy on the GPU from Managed buffers' bytes into their device
    /// copies: each range `didModifyRange:` named since the last capture, once, on a device whose
    /// Managed buffers have device copies
    /// ([`Buffer::did_modify_range`](crate::Buffer::did_modify_range)).
    pub managed_bytes_captured: u64,
    /// Releases of a texture or a block while an upload not yet submitted held it: a texture
    /// whose last handle and last command buffer were gone, or a block given back, while a call's
    /// upload (a copy, a zero fill, a move home, an acquire of another API's writes) still held
    /// its image or its memory. Each such release has the uploads go at once, with no wait, so
    /// the memory goes back once they complete, with no later call.
    pub held_releases: u64,
    /// The bytes of the images and blocks [`held_releases`](Self::held_releases) counts.
    pub held_release_bytes: u64,
    /// Of [`held_releases`](Self::held_releases), each the first since the uploads not yet
    /// submitted came to hold the object: the sends those releases started, at most one for each
    /// upload holding a released object.
    pub held_release_starts: u64,
    /// Command buffers that ran untimed, their GPU start and end times 0
    /// ([`CommandBuffer::gpu_start_time`](crate::CommandBuffer::gpu_start_time)): at their first
    /// record, every pair of the device's timestamp pools was taken and the device made no new
    /// pool ([`Gauges::timestamp_pools`]). The runtime's own command buffers that found no pair
    /// as they began count here too, and their time in
    /// [`runtime_command_buffer_gpu_nanos`](Self::runtime_command_buffer_gpu_nanos) is lost.
    pub untimed_command_buffers: u64,
    /// Elements of set 1's sampler arrays written (`vkUpdateDescriptorSets`): a sampler state's or
    /// a constexpr sampler's placement and its going write only the elements whose sampler
    /// differs from what the index's element holds, so an equal state made at the index of one
    /// released, while work that used it is pending, writes none.
    pub sampler_element_writes: u64,
    /// Command buffers whose submission the completion thread saw signal its progress value while
    /// their timestamp pair read as not available, so their GPU start and end times stay 0
    /// ([`CommandBuffer::gpu_start_time`](crate::CommandBuffer::gpu_start_time)). A driver whose
    /// timestamps become available only after the signal it orders them before counts every such
    /// command buffer here; a submission the device cancelled at a reset, which signals having run
    /// nothing, counts too. The runtime's own command buffers whose pair reads so count here too,
    /// and add no time to
    /// [`runtime_command_buffer_gpu_nanos`](Self::runtime_command_buffer_gpu_nanos).
    pub stamps_not_available: u64,
    /// The runtime's own command buffers passed to `vkQueueSubmit2`, none of them a Metal command
    /// buffer: the uploads, each leading a submission or going alone (zero fills, moves home and
    /// the zeroing of new textures, staging copies, acquires of another API's writes, and copies
    /// between a texture over a buffer and its image of its own), and the Managed buffers'
    /// copy-ins ([`managed_bytes_captured`](Self::managed_bytes_captured)). Counted before the
    /// call, as [`submissions`](Self::submissions) is.
    pub runtime_command_buffers: u64,
    /// The GPU's nanoseconds those command buffers ran, in total: each one's span from a
    /// timestamp written as its commands begin to one written after its last, in
    /// `timestampPeriod`s, counted on the completion thread as it retires. It is no Metal command
    /// buffer's [`gpu_run`](Self::gpu_run). One whose pair reads as not available
    /// ([`stamps_not_available`](Self::stamps_not_available)), or that took no pair
    /// ([`untimed_command_buffers`](Self::untimed_command_buffers)), adds nothing.
    pub runtime_command_buffer_gpu_nanos: u64,
    /// Rounds of the completion thread in which the progress semaphore's counter read a value
    /// that a host wait on it with a timeout of 0 then reported not reached, so nothing retired in
    /// that round. A command buffer retires only once a host wait has seen its submission's
    /// value, the order Vulkan defines for what the guest then reads; this counts a driver whose
    /// counter runs ahead of its wait.
    pub waits_behind_counter: u64,
    /// Acquires of another API's writes into exported blocks recorded
    /// ([`Block::external_writes_done`](crate::Block::external_writes_done)) and into exported
    /// surfaces ([`Surface::external_writes_done`](crate::Surface::external_writes_done), and an
    /// in-place surface's buffer at the [`Surface::buffer`](crate::Surface::buffer) call that
    /// makes it): one barrier each, in the upload, also counted in [`barriers`](Self::barriers).
    /// Also one for each texture in place over a buffer in a dma-buf's import
    /// (`Device::import_dma_buf`, on Unix), whose move home acquires its image and the buffer's
    /// range under it from the foreign queue family, in the barrier of the upload's moves home.
    pub external_acquires: u64,
    /// Releases of exported blocks and surfaces to another API recorded
    /// ([`BlitCommandEncoder::release_to_external`](crate::BlitCommandEncoder::release_to_external),
    /// [`BlitCommandEncoder::release_surface_to_external`](crate::BlitCommandEncoder::release_surface_to_external)):
    /// one barrier each, in the command buffer's work, also counted in
    /// [`barriers`](Self::barriers).
    pub external_releases: u64,
    /// Surfaces made in place
    /// ([`Device::new_exported_surface`](crate::Device::new_exported_surface),
    /// [`SurfaceForm::InPlace`](crate::SurfaceForm::InPlace)).
    pub surfaces_in_place: u64,
    /// Surfaces made in the copy form
    /// ([`Device::new_exported_surface`](crate::Device::new_exported_surface),
    /// [`SurfaceForm::Copy`](crate::SurfaceForm::Copy)).
    pub surfaces_copied: u64,
    /// Waits of surfaces made in place for the submission that moves their planes to the general
    /// layout ([`Surface::wait_until_ready`](crate::Surface::wait_until_ready)), on the
    /// consumer's thread: one per call. The make
    /// ([`Device::new_exported_surface`](crate::Device::new_exported_surface)) waits for none.
    pub surface_waits: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) the surface waits took, in total.
    pub surface_wait_nanos: u64,
    /// The longest surface wait, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub surface_wait_max_nanos: u64,
    /// The runtime's submissions the surface waits waited for, in total: at each wait's start,
    /// those of the device not yet complete, through the one that moves the planes, which is
    /// among them unless it had completed before the wait began (a submission made after the
    /// make, for any command buffer, takes the moves). Another API's work is not counted: a wait
    /// much longer than these submissions' GPU times waited behind work the runtime did not
    /// submit.
    pub surface_wait_submissions: u64,
    /// `vkQueueBindSparse` calls: one for each import over several dma-bufs
    /// (`Device::import_dma_bufs`, on Unix), binding its imports into its buffer. Each signals a
    /// value the next `vkQueueSubmit2` waits for.
    pub sparse_binds: u64,
    /// Resolves of render passes' multisample attachments, each by Vulkan's resolve in the pass's
    /// rendering as it ends (`MTLStoreActionMultisampleResolve` and
    /// `MTLStoreActionStoreAndMultisampleResolve`): one per attachment a pass resolves, counted
    /// as its encoder ends.
    pub resolves: u64,
    /// Takes from the pool on the consumer's threads, each whole: a block of
    /// [`Device::take_block`](crate::Device::take_block) and a texture block, carved from a free
    /// range or from an allocation the take made ([`take_makes`](Self::take_makes)); their bytes
    /// the blocks'.
    pub takes: Timed,
    /// The takes' waits for the pool's lock, which the block-maker thread and the other takes
    /// hold too; no bytes.
    pub take_lock_waits: Timed,
    /// Allocations a take made for itself on the consumer's thread, where no free range fit it:
    /// `vkCreateBuffer`, `vkAllocateMemory`, `vkBindBufferMemory`, the buffer's address and, for a
    /// Shared one, `vkMapMemory`, together; their bytes the allocations'. The `vkAllocateMemory`
    /// calls alone are [`shared_take_allocations`](Self::shared_take_allocations),
    /// [`private_take_allocations`](Self::private_take_allocations) and
    /// [`texture_block_allocations`](Self::texture_block_allocations).
    pub take_makes: Timed,
    /// The consumer's [`BlockMappings::map`](crate::BlockMappings::map) of those allocations, on
    /// the take's thread; their bytes the allocations'.
    pub take_maps: Timed,
    /// Allocations of the pool made ahead of need, off every take: the block-maker thread's, and
    /// each class's first in the device's making; timed as [`take_makes`](Self::take_makes)
    /// times a take's, their bytes the allocations'.
    pub maker_makes: Timed,
    /// The host's writes of zeros into a mapped block by
    /// [`Device::new_buffer`](crate::Device::new_buffer)'s `zero_from`, on the consumer's thread:
    /// `ptr::write_bytes` (the C library's `memset`) through the block's mapping; their bytes the
    /// zeros. A block the host does not map is zeroed by a GPU fill instead, which is not timed
    /// here ([`block_zero_fills`](Self::block_zero_fills)).
    pub zero_writes: Timed,
    /// Of the [`zero_writes`](Self::zero_writes), those into a block longer than
    /// [`MemoryClass::block_len`](crate::MemoryClass::block_len): a take past the pool's unit,
    /// which made its allocation on the consumer's thread
    /// ([`take_makes`](Self::take_makes)), zeroed through its mapping; their bytes the zeros.
    pub long_take_zero_writes: Timed,
    /// [`Device::new_exported_block`](crate::Device::new_exported_block) calls that made their
    /// block, each whole, on the consumer's thread; their bytes the allocations'. Its steps are
    /// timed apart: [`export_allocations`](Self::export_allocations),
    /// [`export_handles`](Self::export_handles) and [`export_claims`](Self::export_claims); the
    /// rest of a make is its buffer's creation, its memory's binding and its address.
    pub export_makes: Timed,
    /// Of those makes, each `vkAllocateMemory` of the block's allocation, dedicated to its buffer
    /// and exportable as its handle (`VkExportMemoryAllocateInfo`); their bytes the allocations'.
    pub export_allocations: Timed,
    /// Of those makes, each export of the allocation's handle (`vkGetMemoryFdKHR`,
    /// `vkGetMemoryWin32HandleKHR`); their bytes the allocations'.
    pub export_handles: Timed,
    /// Of those makes, each claim of the made block, which enters it in the device's map of
    /// claims; their bytes the allocations'.
    pub export_claims: Timed,
}

/// No Metal counterpart: the calls of one part of the consumer's takes, buffer makes and exported
/// block makes ([`Metrics::takes`] and the others'), each timed alone with
/// [`host_time`](crate::host_time).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timed {
    /// The calls.
    pub count: u64,
    /// The bytes they took, made, mapped or wrote, as the part says.
    pub bytes: u64,
    /// Their nanoseconds, in total.
    pub total_nanos: u64,
    /// The longest, in nanoseconds; a maximum since the device was made, so the difference of two
    /// reads does not apply to it.
    pub longest_nanos: u64,
    /// The longest call's bytes, as the part counts them; with
    /// [`longest_nanos`](Self::longest_nanos), from the same call.
    pub longest_bytes: u64,
    /// Whether the longest call made an allocation of its own: for
    /// [`Metrics::takes`], a take no free range fit; every take make, maker make, export make and
    /// export allocation is one ([`Metrics::take_makes`], [`Metrics::maker_makes`],
    /// [`Metrics::export_makes`], [`Metrics::export_allocations`]); false for the other parts.
    /// From the same call as [`longest_nanos`](Self::longest_nanos).
    pub longest_made: bool,
    /// Those longer than 1 ms.
    pub over_1_ms: u64,
    /// Those longer than 5 ms.
    pub over_5_ms: u64,
    /// Those longer than 20 ms.
    pub over_20_ms: u64,
}

/// No Metal counterpart: the `vkAllocateMemory` calls one site of the runtime made on the
/// consumer's threads, inside its calls ([`Metrics::texture_allocations`] and the other
/// sites'): each call, whether it succeeded or not, timed alone with
/// [`host_time`](crate::host_time). At a site of takes from the pool, also the takes that memory
/// made ahead of need served instead ([`takes_ahead`](Self::takes_ahead)).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SiteAllocations {
    /// The calls.
    pub count: u64,
    /// The bytes they asked for.
    pub bytes: u64,
    /// The longest call, in nanoseconds; a maximum since the device was made, so the difference
    /// of two reads does not apply to it.
    pub longest_nanos: u64,
    /// The bytes the longest call asked for; with [`longest_nanos`](Self::longest_nanos), from
    /// the same call.
    pub longest_bytes: u64,
    /// At a site of takes from the pool ([`Metrics::shared_take_allocations`],
    /// [`Metrics::private_take_allocations`], [`Metrics::texture_block_allocations`]): the takes
    /// that landed in an allocation the block-maker thread made ahead of need while the class
    /// held no wholly free unit, each the first take to land in it, so that no allocation made
    /// before it had room for the take. Each is a take that would otherwise have allocated at
    /// the site, on the consumer's thread. A take landing in a class's first allocation, made in
    /// the device's making, is not among them. 0 at the other sites.
    pub takes_ahead: u64,
    /// Those takes' bytes, each the block's length.
    pub takes_ahead_bytes: u64,
}

/// A thread's slot in [`Gauges::threads`], given back as it drops.
pub(crate) struct ThreadSlot {
    atomics: Arc<Atomics>,
    slot: Option<usize>,
}

impl Drop for ThreadSlot {
    fn drop(&mut self) {
        if let Some(slot) = self.slot {
            self.atomics.threads[slot].store(0, Relaxed);
        }
    }
}

/// The calling thread's id as the OS numbers it: Linux's `gettid`, the Apple platforms'
/// `pthread_threadid_np`, Windows' `GetCurrentThreadId`; 0 on any other.
fn os_thread_id() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `gettid` takes no argument and cannot fail.
        unsafe { libc::syscall(libc::SYS_gettid) as u64 }
    }
    #[cfg(target_vendor = "apple")]
    {
        let mut id = 0;
        // SAFETY: a null thread names the calling one, and `id` outlives the call.
        unsafe { libc::pthread_threadid_np(0 as libc::pthread_t, &mut id) };
        id
    }
    #[cfg(windows)]
    {
        // SAFETY: it takes no argument and cannot fail.
        u64::from(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() })
    }
    #[cfg(not(any(target_os = "linux", target_vendor = "apple", windows)))]
    {
        0
    }
}

/// No Metal counterpart: the live count of each kind of resource. Every one is zero once the
/// device is torn down.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gauges {
    /// The OS id of each thread the device runs (its reaper, block-maker, submit and completion
    /// threads, and its cache directory's removal thread while one runs), in no order, 0 in a
    /// slot no thread holds; all 0 on an OS whose thread ids the runtime does not read. The
    /// harness tells the runtime's own threads from the driver's by them.
    pub threads: [u64; THREADS],
    /// 1 while a handle of the device is out, the consumer's or one a handler or a listener
    /// holds; 0 from the moment the last one drops, which starts the teardown. After
    /// [`Instance::wait_until_devices_torn_down`](crate::Instance::wait_until_devices_torn_down)
    /// answers false, 1 says a handle is still out; 0 with a thread in
    /// [`threads`](Self::threads) says the teardown still runs on that thread.
    pub handles_out: u64,
    /// Buffers.
    pub buffers: u64,
    /// Textures: each one's image and its views, until the reaper frees them and gives the
    /// block-maker thread their memory of their own to free.
    pub textures: u64,
    /// Sampler states.
    pub samplers: u64,
    /// `VkPipeline`s.
    pub pipelines: u64,
    /// Compiled kernels: each the translation of one compute pipeline content (its descriptor's
    /// kernel, constant values, linked functions and compile fields), with its shader modules and
    /// their `VkPipeline`s, shared by every state of that content. One lives while a state or
    /// the device's map of kernels holds it.
    pub kernels: u64,
    /// Entries of the device's map of compiled contents, compute and render, which holds at most
    /// 1,024 and forgets the least recently used past them.
    pub kernel_entries: u64,
    /// Compiled render pipelines: each the translation of one render pipeline content (its
    /// descriptor's functions, their constant values and the fields that change the compile),
    /// with its shader modules and their `VkPipeline`s, shared by every state of that content.
    /// One lives while a state or the device's map of contents holds it.
    pub render_contents: u64,
    /// The image views textures keep for render passes: one for each level and slice a pass
    /// rendered into, freed with the texture.
    pub attachment_views: u64,
    /// Command buffers not yet retired.
    pub command_buffers: u64,
    /// Held indices of set 1's arrays: one for each index a texture, a sampler state or a
    /// constexpr sampler owns. A placement that takes an index from another object adds none.
    pub descriptor_slots: u64,
    /// Live `VkDeviceMemory` allocations: those blocks are ranges of, the idle ones among them,
    /// each import's (one for each dma-buf of a block over several), and the runtime's own. Each counts against the device's
    /// `maxMemoryAllocationCount` ([`Limit::MemoryAllocations`](crate::Limit::MemoryAllocations)).
    pub blocks: u64,
    /// Events and shared events (timeline semaphores).
    pub events: u64,
    /// `VkSampler`s the runtime holds: the default samplers of the sampler arrays, and the
    /// sampler states' device samplers, one for each create info. They count against the
    /// device's `maxSamplerAllocationCount`.
    pub device_samplers: u64,
    /// Bytes of Private memory that went to another memory type because the device-local one
    /// was out of memory: blocks whose [`BlockInfo::device_local`](crate::BlockInfo) is false and
    /// textures' allocations of their own.
    pub spilled_bytes: u64,
    /// Bytes of Managed buffers whose device copy the consumer placed in a Private block that
    /// spilled to another memory type than the Private one (device-local memory had no room when
    /// the block's allocation was made), so their kernels read host memory; counted per buffer
    /// while it lives. Metal has no counterpart.
    pub managed_spilled_bytes: u64,
    /// Bytes of the pool's live Shared allocations: the blocks handed out, the free ranges
    /// between them and the idle allocations, as `VkDeviceMemory` the device holds for the
    /// consumer's blocks. Metal's nearest counterpart is `MTLDevice.currentAllocatedSize`, which
    /// counts every class together.
    pub shared_allocation_bytes: u64,
    /// As [`shared_allocation_bytes`](Self::shared_allocation_bytes), of Private allocations.
    pub private_allocation_bytes: u64,
    /// Bytes of idle Shared allocations, which hold no block anyone holds: those idle for less
    /// than 250 ms, which the idle rule frees at that age, and the reserve made ahead of need,
    /// the class's wholly free unit (one allocation of 16 MiB, at most).
    pub shared_idle_bytes: u64,
    /// Bytes of idle Private allocations, as [`shared_idle_bytes`](Self::shared_idle_bytes) for
    /// Shared ones.
    pub private_idle_bytes: u64,
    /// How long the oldest idle Shared allocation the idle rule weighs (not the reserve) has been
    /// idle, in nanoseconds; 0 where none is. The rule frees it at 250 ms, so this stays under
    /// that plus the frees ahead of it.
    pub shared_idle_age_nanos: u64,
    /// As [`shared_idle_age_nanos`](Self::shared_idle_age_nanos), of Private allocations.
    pub private_idle_age_nanos: u64,
    /// Bytes of the Shared reserve: the idle allocation made ahead of need that the idle rule does
    /// not free, the class's wholly free unit, the last idle one of 16 MiB in the order made.
    /// Counted in [`shared_idle_bytes`](Self::shared_idle_bytes) too.
    pub shared_reserve_bytes: u64,
    /// As [`shared_reserve_bytes`](Self::shared_reserve_bytes), of Private allocations.
    pub private_reserve_bytes: u64,
    /// Bytes of the Shared long allocations, each one a take longer than the 16 MiB unit made for
    /// itself, that a later take of its length holds: the whole allocation, until its block comes
    /// back.
    pub shared_long_reused_bytes: u64,
    /// As [`shared_long_reused_bytes`](Self::shared_long_reused_bytes), of Private allocations.
    pub private_long_reused_bytes: u64,
    /// Bytes of staging for copies between textures and the CPU (`replace_region`, `get_bytes`):
    /// the staging region, once made, and the staging buffers of their own that copies
    /// larger than the region hold until the upload that copies through them is gone.
    pub staging_bytes: u64,
    /// The translator's specialized entries and module decodes the device keeps for its pipeline
    /// creations: at most 32 entries and 4 decodes, the least recently used going past them.
    pub stage_entries: u64,
    /// The bytes those entries and decodes hold, each counted as the ledger bytes making it held:
    /// at most 64 MiB of entries and 32 MiB of decodes.
    pub stage_entry_bytes: u64,
    /// The size of the data in the device's `VkPipelineCache`, as `vkGetPipelineCacheData` gave
    /// it when the reaper last looked, once creation was quiet: at most 32 MiB at a look, past
    /// which the reaper replaces the cache with a fresh one, and this falls to the fresh one's.
    pub pipeline_cache_bytes: u64,
    /// Shadows, on a device without the attachment feedback loop: one image for each image a
    /// draw read as an attachment of its own pass through a texture, made at the first such draw
    /// and freed with the image.
    pub shadows: u64,
    /// The bytes of the shadows' memory.
    pub shadow_bytes: u64,
    /// Textures over a buffer the runtime holds in an image of their own, where the device's
    /// linear image does not take the texture or the buffer's bytes are imported
    /// ([`Buffer::new_texture`](crate::Buffer::new_texture)): the copies between them and their
    /// buffers are [`Metrics::over_copies`]. A view is its texture's. The rest of the textures over
    /// buffers are in place.
    pub textures_copied_over: u64,
    /// The device's timestamp query pools, each of 4096 queries, a pair for each command buffer
    /// from its first record until its times are read at its completion: made as command buffers
    /// need them, and kept until the device drops.
    pub timestamp_pools: u64,
    /// Imports whose memory is not yet freed: one for each imported block
    /// (`Device::import_dma_buf` on Unix,
    /// [`Device::import_host_memory`](crate::Device::import_host_memory)), and one for each
    /// dma-buf of a block over several (`Device::import_dma_bufs` on Unix). Each is also one
    /// of [`blocks`](Self::blocks).
    pub imports: u64,
    /// Exported blocks ([`Device::new_exported_block`](crate::Device::new_exported_block)) whose
    /// memory is not yet freed, an opaque handle's surface in place among them
    /// ([`Device::new_exported_surface`](crate::Device::new_exported_surface)): each is also one
    /// of [`blocks`](Self::blocks).
    pub exports: u64,
    /// Surface plane imports whose memory is not yet freed, each also one of
    /// [`blocks`](Self::blocks): a dma-buf surface's in place
    /// ([`Device::new_exported_surface`](crate::Device::new_exported_surface)), each its own
    /// dedicated import of the surface's dma-buf, a dma-buf surface's buffer among them, on Unix;
    /// and a Direct3D 11 texture's (`Device::import_d3d11_surface`), each its own dedicated
    /// import of the plane's handle, on Windows.
    pub surface_imports: u64,
    /// Surfaces' buffers in place ([`Surface::buffer`](crate::Surface::buffer)) not yet freed:
    /// an opaque handle's, a buffer on the surface's one allocation, and a dma-buf's, on one more
    /// import of the dma-buf, which is also one of [`surface_imports`](Self::surface_imports) and
    /// of [`blocks`](Self::blocks). A copy form's buffer is its block's, and not counted here.
    pub surface_buffers: u64,
    /// Bytes of textures' places in texture blocks: each placed texture's image size. Those
    /// blocks are Private blocks, counted in [`private_placed_bytes`](Self::private_placed_bytes)
    /// whole.
    pub texture_block_bytes: u64,
    /// Bytes of the texture blocks the device holds, each whole: the 16 MiB ones, the one kept
    /// empty for the next place among them, and each texture's of its own length where it is
    /// longer. Against [`texture_block_bytes`](Self::texture_block_bytes), the rest is the free
    /// bytes between places, which no place ever moves out of.
    pub texture_block_held_bytes: u64,
    /// Bytes of the sparse texture blocks: those holding a place whose places' bytes are less
    /// than a sixteenth of their length, each whole. Counted in
    /// [`texture_block_held_bytes`](Self::texture_block_held_bytes) too.
    pub texture_block_sparse_bytes: u64,
    /// The sparse texture blocks
    /// [`texture_block_sparse_bytes`](Self::texture_block_sparse_bytes) counts.
    pub texture_block_sparse_blocks: u64,
    /// Bytes of the pool's Private allocations that hold a block the consumer took
    /// ([`Device::take_block`](crate::Device::take_block)), each whole: 16 MiB units, and a long
    /// allocation that a take longer than one made for itself. No texture block lies in one: a
    /// texture block fills its allocation. The allocations are freed only once every block in
    /// them is given back.
    pub private_take_held_bytes: u64,
    /// Bytes of the Private blocks the consumer took and has not given back, which lie in those
    /// allocations: [`private_placed_bytes`](Self::private_placed_bytes) less the texture blocks.
    /// The live buffers placed in them are [`private_buffer_bytes`](Self::private_buffer_bytes).
    pub private_take_bytes: u64,
    /// Bytes of the sparse allocations among
    /// [`private_take_held_bytes`](Self::private_take_held_bytes): those whose blocks out are
    /// less than a sixteenth of their length, each whole.
    pub private_take_sparse_bytes: u64,
    /// The sparse allocations [`private_take_sparse_bytes`](Self::private_take_sparse_bytes)
    /// counts.
    pub private_take_sparse_allocations: u64,
    /// The pool's Private units, its allocations of 16 MiB
    /// ([`MemoryClass::block_len`](crate::MemoryClass::block_len)), that hold a block: blocks the
    /// consumer took, or a texture block, which fills its unit (the one kept empty for the next
    /// place among them). An idle unit is not one, nor a long allocation. A unit goes back only
    /// once it holds nothing, and nothing moves out of one, so against
    /// [`private_unit_used_bytes`](Self::private_unit_used_bytes) the rest of their 16 MiB each
    /// is free bytes that only takes and places that fit them use.
    pub private_units: u64,
    /// Bytes in use inside the [`private_units`](Self::private_units): the lengths of the blocks
    /// the consumer took there, and the image sizes of the textures placed in their texture
    /// blocks.
    pub private_unit_used_bytes: u64,
    /// Bytes of textures' allocations of their own (shadows and counted copies' images among
    /// them), not in the pool.
    pub texture_own_bytes: u64,
    /// Bytes of live buffers in Shared blocks, each its length, Managed buffers' bytes among them:
    /// a buffer the consumer placed over another's bytes counts again.
    pub shared_buffer_bytes: u64,
    /// As [`shared_buffer_bytes`](Self::shared_buffer_bytes), of Private blocks, Managed buffers'
    /// device copies not among them.
    pub private_buffer_bytes: u64,
    /// Bytes of Shared blocks taken from the pool and not given back: against
    /// [`shared_allocation_bytes`](Self::shared_allocation_bytes), the rest of the pool's Shared
    /// allocations is [`shared_idle_bytes`](Self::shared_idle_bytes) and the free ranges between
    /// blocks.
    pub shared_placed_bytes: u64,
    /// As [`shared_placed_bytes`](Self::shared_placed_bytes), of Private blocks, texture blocks
    /// among them.
    pub private_placed_bytes: u64,
    /// Bytes of the imports [`imports`](Self::imports) counts, as each was allocated.
    pub import_bytes: u64,
    /// Bytes of the surface plane imports [`surface_imports`](Self::surface_imports) counts, as
    /// each was allocated.
    pub surface_import_bytes: u64,
}
