//! The device's counters: cumulative metrics and live gauges, as relaxed atomics that are always
//! on. Each part of the runtime adds to its own; nothing turns them on or off.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

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
    /// [`Metrics::blocks_reused`].
    BlocksReused,
    /// [`Metrics::dispatches_without_effects`].
    DispatchesWithoutEffects,
    /// [`Metrics::translations`].
    Translations,
    /// [`Metrics::translations_loaded`].
    TranslationsLoaded,
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
    /// [`Metrics::synchronizations`].
    Synchronizations,
    /// [`Metrics::groups_beside_another_queue`].
    GroupsBesideAnotherQueue,
    /// [`Metrics::lone_submissions_across_queues`].
    LoneSubmissionsAcrossQueues,
    /// [`Metrics::forced_flushes_across_queues`].
    ForcedFlushesAcrossQueues,
    /// [`Metrics::over_copies`].
    OverCopies,
    /// [`Metrics::allocations_freed`].
    AllocationsFreed,
    /// [`Metrics::submissions_during_frees`].
    SubmissionsDuringFrees,
    /// [`Metrics::capture_waits`].
    CaptureWaits,
    /// [`Metrics::capture_streams`].
    CaptureStreams,
    /// [`Metrics::capture_overflows`].
    CaptureOverflows,
}

/// A gauge the runtime moves.
#[derive(Clone, Copy)]
pub(crate) enum Gauge {
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
    /// [`Gauges::shared_idle_bytes`].
    SharedIdleBytes,
    /// [`Gauges::private_idle_bytes`].
    PrivateIdleBytes,
    /// [`Gauges::capture_bytes`].
    CaptureBytes,
    /// [`Gauges::staging_bytes`].
    StagingBytes,
    /// [`Gauges::stage_entries`].
    StageEntries,
    /// [`Gauges::stage_entry_bytes`].
    StageEntryBytes,
    /// [`Gauges::pipeline_cache_bytes`].
    PipelineCacheBytes,
}

/// No Metal counterpart: a device's counters ([`Device::counters`](crate::Device::counters)).
///
/// A cheap handle that stays readable after the device is torn down, so a test can read the
/// gauges at zero once it is gone. Reads are relaxed: each value is exact once the work that
/// moves it has stopped, and close while it runs.
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
    passes: AtomicU64,
    barriers: AtomicU64,
    bytes_copied: AtomicU64,
    bytes_zeroed: AtomicU64,
    cache_bytes_written: AtomicU64,
    pipelines_created: AtomicU64,
    first_use_waits: AtomicU64,
    first_use_wait_nanos: AtomicU64,
    first_use_wait_max_nanos: AtomicU64,
    blocks_made: AtomicU64,
    block_bytes_made: AtomicU64,
    consumer_allocations: AtomicU64,
    consumer_allocation_nanos: AtomicU64,
    consumer_allocation_max_nanos: AtomicU64,
    maker_allocation_nanos: AtomicU64,
    dispatches_without_effects: AtomicU64,
    blocks_reused: AtomicU64,
    translations: AtomicU64,
    translations_loaded: AtomicU64,
    creations_before_seed: AtomicU64,
    cache_writes_dropped: AtomicU64,
    ring_writes: AtomicU64,
    pushes: AtomicU64,
    pushes_held: AtomicU64,
    upload_waits: AtomicU64,
    synchronizations: AtomicU64,
    commit_lock_wait_nanos: AtomicU64,
    commit_lock_wait_max_nanos: AtomicU64,
    groups_beside_another_queue: AtomicU64,
    lone_submissions_across_queues: AtomicU64,
    forced_flushes_across_queues: AtomicU64,
    over_copies: AtomicU64,
    allocations_freed: AtomicU64,
    submissions_during_frees: AtomicU64,
    idle_frees_by_age: [AtomicU64; FREE_AGES_MS.len() + 1],
    idle_reuses_by_age: [AtomicU64; REUSE_AGES_MS.len() + 1],
    managed_bytes_captured: AtomicU64,
    capture_waits: AtomicU64,
    capture_streams: AtomicU64,
    capture_overflows: AtomicU64,
    capture_overflow_bytes: AtomicU64,
    capture_copy_nanos: AtomicU64,
    capture_wait_nanos: AtomicU64,
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
    shared_idle_bytes: AtomicU64,
    private_idle_bytes: AtomicU64,
    /// When the oldest idle Shared allocation the idle rule weighs became idle, in nanoseconds of
    /// [`host_time`](crate::host_time); 0 where none is.
    shared_idle_since: AtomicU64,
    /// As [`shared_idle_since`](Self::shared_idle_since), of Private allocations.
    private_idle_since: AtomicU64,
    capture_bytes: AtomicU64,
    staging_bytes: AtomicU64,
    stage_entries: AtomicU64,
    stage_entry_bytes: AtomicU64,
    pipeline_cache_bytes: AtomicU64,
    /// Segment ends whose barrier reaches the host, for the tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    host_reaches: AtomicU64,
    /// Calibrations of the GPU's clock against the host's, for the tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    calibrations: AtomicU64,
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
        let a = &self.atomics;
        let atomic = match metric {
            Metric::Submissions => &a.submissions,
            Metric::Passes => &a.passes,
            Metric::Barriers => &a.barriers,
            Metric::PipelinesCreated => &a.pipelines_created,
            Metric::BlocksMade => &a.blocks_made,
            Metric::BlocksReused => &a.blocks_reused,
            Metric::DispatchesWithoutEffects => &a.dispatches_without_effects,
            Metric::Translations => &a.translations,
            Metric::TranslationsLoaded => &a.translations_loaded,
            Metric::CreationsBeforeSeed => &a.creations_before_seed,
            Metric::CacheWritesDropped => &a.cache_writes_dropped,
            Metric::RingWrites => &a.ring_writes,
            Metric::Pushes => &a.pushes,
            Metric::PushesHeld => &a.pushes_held,
            Metric::UploadWaits => &a.upload_waits,
            Metric::Synchronizations => &a.synchronizations,
            Metric::GroupsBesideAnotherQueue => &a.groups_beside_another_queue,
            Metric::LoneSubmissionsAcrossQueues => &a.lone_submissions_across_queues,
            Metric::ForcedFlushesAcrossQueues => &a.forced_flushes_across_queues,
            Metric::OverCopies => &a.over_copies,
            Metric::AllocationsFreed => &a.allocations_freed,
            Metric::SubmissionsDuringFrees => &a.submissions_during_frees,
            Metric::CaptureWaits => &a.capture_waits,
            Metric::CaptureStreams => &a.capture_streams,
            Metric::CaptureOverflows => &a.capture_overflows,
        };
        atomic.fetch_add(1, Relaxed);
    }

    /// One more first-use wait: an encoding call that made the `VkPipeline` it needed, for
    /// `nanos` nanoseconds.
    pub(crate) fn first_use_wait(&self, nanos: u64) {
        let a = &self.atomics;
        a.first_use_waits.fetch_add(1, Relaxed);
        a.first_use_wait_nanos.fetch_add(nanos, Relaxed);
        a.first_use_wait_max_nanos.fetch_max(nanos, Relaxed);
    }

    /// A commit that waited `nanos` nanoseconds to take the submission lock
    /// ([`Metrics::commit_lock_wait_nanos`]).
    pub(crate) fn commit_lock_wait(&self, nanos: u64) {
        let a = &self.atomics;
        a.commit_lock_wait_nanos.fetch_add(nanos, Relaxed);
        a.commit_lock_wait_max_nanos.fetch_max(nanos, Relaxed);
    }

    /// A submission of `infos` submit infos ([`Metrics::submit_infos`]).
    pub(crate) fn submitted_infos(&self, infos: u64) {
        self.atomics.submit_infos.fetch_add(infos, Relaxed);
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

    /// `bytes` more copied for a call ([`Metrics::bytes_copied`]).
    pub(crate) fn copied(&self, bytes: u64) {
        self.atomics.bytes_copied.fetch_add(bytes, Relaxed);
    }

    /// `bytes` more of a new buffer zeroed ([`Metrics::bytes_zeroed`]).
    pub(crate) fn zeroed(&self, bytes: u64) {
        self.atomics.bytes_zeroed.fetch_add(bytes, Relaxed);
    }

    /// `bytes` more of Managed buffers' bytes a commit copied ([`Metrics::managed_bytes_captured`]).
    pub(crate) fn captured(&self, bytes: u64) {
        self.atomics
            .managed_bytes_captured
            .fetch_add(bytes, Relaxed);
    }

    /// `bytes` more of them copied into staging blocks of an overflow's own
    /// ([`Metrics::capture_overflow_bytes`]).
    pub(crate) fn overflowed(&self, bytes: u64) {
        self.atomics
            .capture_overflow_bytes
            .fetch_add(bytes, Relaxed);
    }

    /// `nanos` more a capture spent copying bytes on the host ([`Metrics::capture_copy_nanos`]).
    pub(crate) fn capture_copied(&self, nanos: u64) {
        self.atomics.capture_copy_nanos.fetch_add(nanos, Relaxed);
    }

    /// `nanos` more a capture spent waiting ([`Metrics::capture_wait_nanos`]).
    pub(crate) fn capture_waited(&self, nanos: u64) {
        self.atomics.capture_wait_nanos.fetch_add(nanos, Relaxed);
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
            Gauge::SharedIdleBytes => &a.shared_idle_bytes,
            Gauge::PrivateIdleBytes => &a.private_idle_bytes,
            Gauge::CaptureBytes => &a.capture_bytes,
            Gauge::StagingBytes => &a.staging_bytes,
            Gauge::StageEntries => &a.stage_entries,
            Gauge::StageEntryBytes => &a.stage_entry_bytes,
            Gauge::PipelineCacheBytes => &a.pipeline_cache_bytes,
        }
    }

    /// No Metal counterpart: the cumulative counts since the device was made.
    pub fn metrics(&self) -> Metrics {
        let a = &self.atomics;
        Metrics {
            submissions: a.submissions.load(Relaxed),
            submit_infos: a.submit_infos.load(Relaxed),
            passes: a.passes.load(Relaxed),
            barriers: a.barriers.load(Relaxed),
            bytes_copied: a.bytes_copied.load(Relaxed),
            bytes_zeroed: a.bytes_zeroed.load(Relaxed),
            pipelines_created: a.pipelines_created.load(Relaxed),
            first_use_waits: a.first_use_waits.load(Relaxed),
            first_use_wait_nanos: a.first_use_wait_nanos.load(Relaxed),
            first_use_wait_max_nanos: a.first_use_wait_max_nanos.load(Relaxed),
            blocks_made: a.blocks_made.load(Relaxed),
            block_bytes_made: a.block_bytes_made.load(Relaxed),
            consumer_allocations: a.consumer_allocations.load(Relaxed),
            consumer_allocation_nanos: a.consumer_allocation_nanos.load(Relaxed),
            consumer_allocation_max_nanos: a.consumer_allocation_max_nanos.load(Relaxed),
            maker_allocation_nanos: a.maker_allocation_nanos.load(Relaxed),
            blocks_reused: a.blocks_reused.load(Relaxed),
            dispatches_without_effects: a.dispatches_without_effects.load(Relaxed),
            translations: a.translations.load(Relaxed),
            translations_loaded: a.translations_loaded.load(Relaxed),
            creations_before_seed: a.creations_before_seed.load(Relaxed),
            cache_writes_dropped: a.cache_writes_dropped.load(Relaxed),
            cache_bytes_written: a.cache_bytes_written.load(Relaxed),
            ring_writes: a.ring_writes.load(Relaxed),
            pushes: a.pushes.load(Relaxed),
            pushes_held: a.pushes_held.load(Relaxed),
            upload_waits: a.upload_waits.load(Relaxed),
            synchronizations: a.synchronizations.load(Relaxed),
            commit_lock_wait_nanos: a.commit_lock_wait_nanos.load(Relaxed),
            commit_lock_wait_max_nanos: a.commit_lock_wait_max_nanos.load(Relaxed),
            groups_beside_another_queue: a.groups_beside_another_queue.load(Relaxed),
            lone_submissions_across_queues: a.lone_submissions_across_queues.load(Relaxed),
            forced_flushes_across_queues: a.forced_flushes_across_queues.load(Relaxed),
            over_copies: a.over_copies.load(Relaxed),
            allocations_freed: a.allocations_freed.load(Relaxed),
            submissions_during_frees: a.submissions_during_frees.load(Relaxed),
            idle_frees_by_age: std::array::from_fn(|at| a.idle_frees_by_age[at].load(Relaxed)),
            idle_reuses_by_age: std::array::from_fn(|at| a.idle_reuses_by_age[at].load(Relaxed)),
            managed_bytes_captured: a.managed_bytes_captured.load(Relaxed),
            capture_waits: a.capture_waits.load(Relaxed),
            capture_streams: a.capture_streams.load(Relaxed),
            capture_overflows: a.capture_overflows.load(Relaxed),
            capture_overflow_bytes: a.capture_overflow_bytes.load(Relaxed),
            capture_copy_nanos: a.capture_copy_nanos.load(Relaxed),
            capture_wait_nanos: a.capture_wait_nanos.load(Relaxed),
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
            shared_idle_bytes: a.shared_idle_bytes.load(Relaxed),
            private_idle_bytes: a.private_idle_bytes.load(Relaxed),
            shared_idle_age_nanos: age(&a.shared_idle_since),
            private_idle_age_nanos: age(&a.private_idle_since),
            capture_bytes: a.capture_bytes.load(Relaxed),
            staging_bytes: a.staging_bytes.load(Relaxed),
            stage_entries: a.stage_entries.load(Relaxed),
            stage_entry_bytes: a.stage_entry_bytes.load(Relaxed),
            pipeline_cache_bytes: a.pipeline_cache_bytes.load(Relaxed),
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
    /// wait or signal separates share one, and the upload shares the first.
    pub submit_infos: u64,
    /// Encoders recorded: compute, blit and render.
    pub passes: u64,
    /// Pipeline-barrier commands recorded into command buffers' work: one at the end of each
    /// segment with commands and of each upload, one before a dispatch at a hazard, one owed by
    /// an encoder that wrote or a memory barrier call, recorded before the next command, two
    /// around each blit copy, fill and synchronize copy, and two around each render pass, the
    /// first of each pair paying what is owed, with one inside a render pass at a memory barrier
    /// after a draw that wrote memory, one at a draw that moves an attachment's image to the
    /// feedback layout, and one at a draw that reads an attachment after a memory barrier owed to
    /// it. Any other draw takes none.
    pub barriers: u64,
    /// Bytes the runtime copied for a call: `replace_region` and `get_bytes` staging, and the
    /// copies between a texture over a buffer and its image of its own
    /// ([`over_copies`](Self::over_copies)). A blit's own copy, and a synchronize's, is the
    /// program's and is not counted.
    pub bytes_copied: u64,
    /// Bytes of new buffers zeroed: [`Device::new_buffer`](crate::Device::new_buffer)'s from its
    /// `zero_from` to the buffer's end, and a Managed buffer's device copy's past its copied bytes
    /// ([`Device::new_managed_buffer`](crate::Device::new_managed_buffer)). Nothing else the
    /// runtime makes is zeroed.
    pub bytes_zeroed: u64,
    /// `VkPipeline`s created.
    pub pipelines_created: u64,
    /// First-use waits: encoding calls that waited while the `VkPipeline` they needed was made,
    /// for a workgroup size (a threadgroup size, or the tile of an image-only kernel's grid more
    /// than one deep) or threadgroup memory lengths no state of the kernel's content had run with
    /// before, or for a render content's first draw of lines.
    pub first_use_waits: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) the first-use waits took, in total.
    pub first_use_wait_nanos: u64,
    /// The longest first-use wait, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub first_use_wait_max_nanos: u64,
    /// `VkDeviceMemory` allocations made: the allocations blocks are ranges of (the texture
    /// blocks among them), the zero source and the parameter ring's blocks.
    pub blocks_made: u64,
    /// Bytes of the allocations blocks are ranges of, made: what the device allocated for
    /// blocks, by the block-maker thread or in a take. The difference of two reads is what a
    /// frame allocated.
    pub block_bytes_made: u64,
    /// `vkAllocateMemory` calls made on the consumer's threads, inside its calls: takes that no
    /// free range fits, textures' allocations of their own, the staging region and the parameter
    /// ring's blocks, and the zero source. A driver may hold the device's other calls for an
    /// allocation's whole length.
    pub consumer_allocations: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) those calls took, in total.
    pub consumer_allocation_nanos: u64,
    /// The longest of those calls, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub consumer_allocation_max_nanos: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) the allocations made ahead of need
    /// took, in total: the block-maker thread's, off the consumer's threads, and each class's
    /// first allocation in the device's making.
    pub maker_allocation_nanos: u64,
    /// Blocks taken again: a take of the length of a block given back and kept, which took that
    /// block with its number, address and mapping, and no new range or allocation.
    pub blocks_reused: u64,
    /// Dispatches of a kernel that writes no memory (`PipelineInterface::writes_memory`),
    /// direct and indirect: each was checked as Metal checks it, and nothing was recorded, bound
    /// or read for it, as it has no effect a program can observe.
    pub dispatches_without_effects: u64,
    /// Kernels the translator lowered, each a module of a compute pipeline content the device did
    /// not hold compiled: one for a kernel that does not observe its threadgroup partition, two
    /// for one that does. A state of a content the device holds lowers nothing.
    pub translations: u64,
    /// Lowered kernels a pipeline state took from the cache directory in place of a lowering.
    pub translations_loaded: u64,
    /// `VkPipeline` creations that went through the device's empty pipeline cache while the
    /// reaper read and took in the device's blob ([`DeviceOptions::pipeline_cache_dir`]): each
    /// compiled without the blob's data, and is in the blob only where the merge took it. Zero
    /// with no directory, or where the device's first creation came after the seed.
    ///
    /// [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
    pub creations_before_seed: u64,
    /// Lowerings and refusals not kept in the cache directory: each was handed to the reaper,
    /// which writes them off every guest thread, while the ones waiting for it held 8 MiB, and was
    /// dropped rather than waited for. A later device lowers it again.
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
    /// Groups opened while another queue held one. Each queue groups its own command buffers,
    /// so these are the groups one queue of the same command buffers would not have opened.
    pub groups_beside_another_queue: u64,
    /// Submissions sent with fewer than three of their own queue's submissions in flight while the
    /// device had three or more in flight across its queues: the ones one queue of the same
    /// command buffers would have held in a group.
    pub lone_submissions_across_queues: u64,
    /// Groups sent with no reason of their own queue's: neither full, nor ended by an event's
    /// signal or a semaphore wait, nor below three submissions in flight, nor asked for by a host
    /// wait or the end. No path of the submit thread sends one, so this stays 0; it is counted to
    /// show it.
    pub forced_flushes_across_queues: u64,
    /// Copies between a texture over a buffer the device's linear image does not hold in place
    /// and its image of its own ([`Buffer::new_texture`](crate::Buffer::new_texture)): the
    /// buffer's bytes into the image, in the upload that leads a submission or after a command
    /// wrote the buffer, and the image back into the buffer after a command wrote the texture.
    pub over_copies: u64,
    /// Idle allocations the block-maker thread freed: one at a time, each once it had been idle,
    /// holding no live block, for 250 ms, whatever the device's queues were doing.
    pub allocations_freed: u64,
    /// Submissions made while the block-maker thread was inside an idle allocation's
    /// `vkFreeMemory`. A driver may hold such a submission until the free returns: on the NVIDIA
    /// cards measured in Thunderbolt enclosures, about 1.1 ms per MiB of host-visible memory.
    pub submissions_during_frees: u64,
    /// The frees of [`allocations_freed`](Self::allocations_freed) by how long each allocation had
    /// been idle when its free began, in buckets of under 250 ms (a spilled allocation that
    /// emptied, freed at once), 250 to 275 ms, 275 to 350 ms, 350 to 500 ms, 500 ms to 1 s, and
    /// 1 s or more.
    pub idle_frees_by_age: [u64; 6],
    /// Idle allocations taken again, by how long each had been idle: in buckets of under 50 ms,
    /// 50 to 100 ms, 100 to 250 ms, 250 to 500 ms, 500 ms to 1 s, 1 to 2 s, and 2 s or more. Only
    /// a class's reserve, which the idle rule does not free, is taken again past 250 ms.
    pub idle_reuses_by_age: [u64; 7],
    /// Bytes commits copied on the host from Managed buffers' bytes into staging memory: each range
    /// `didModifyRange:` named since the last capture, once, on a device whose Managed buffers have
    /// device copies ([`Buffer::did_modify_range`](crate::Buffer::did_modify_range)).
    pub managed_bytes_captured: u64,
    /// Waits of commits for a submission to complete for room in the capture ring: an earlier
    /// command buffer's copy, or the commit's own ring-full sent ahead.
    pub capture_waits: u64,
    /// Ring-fulls of a commit's own bytes sent as copy submissions of their own, which the commit
    /// waited for before it used the ring again.
    pub capture_streams: u64,
    /// Commits whose capture took staging blocks of its own, as a command buffer committed before
    /// was held on the host.
    pub capture_overflows: u64,
    /// The bytes those commits copied into such blocks.
    pub capture_overflow_bytes: u64,
    /// Host time commits spent copying Managed buffers' bytes into the capture ring and into
    /// staging blocks of their own, in nanoseconds.
    pub capture_copy_nanos: u64,
    /// Host time commits spent waiting in their captures, in nanoseconds: for a submission to
    /// complete for room in the ring (an earlier command buffer's copy, or the commit's own
    /// ring-full sent ahead, from its flush to its completion), and for command buffers committed
    /// before to be submitted.
    pub capture_wait_nanos: u64,
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
    /// Buffers.
    pub buffers: u64,
    /// Textures: each one's image, its views and its memory, until the reaper frees them.
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
    /// and the runtime's own.
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
    /// Bytes of idle Shared allocations, which hold no block anyone holds: those idle for less
    /// than 250 ms, which the idle rule frees at that age, and the reserve made ahead of need
    /// that keeps the free bytes at 8 MiB or more (one allocation of 16 MiB, at most).
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
    /// Bytes of staging for Managed buffers' captures: the capture ring, once a commit made it, and
    /// the staging blocks of overflows that command buffers not yet retired hold.
    pub capture_bytes: u64,
    /// Bytes of staging for copies between textures and the CPU (`replace_region`, `get_bytes`):
    /// the staging region, once a copy made it, and the staging buffers of their own that copies
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
}
