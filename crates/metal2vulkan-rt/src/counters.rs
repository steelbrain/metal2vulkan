//! The device's counters: cumulative metrics and live gauges, as relaxed atomics that are always
//! on. Each part of the runtime adds to its own; nothing turns them on or off.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

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
    /// [`Metrics::dedicated_blocks_reused`].
    DedicatedBlocksReused,
    /// [`Metrics::dispatches_without_effects`].
    DispatchesWithoutEffects,
    /// [`Metrics::translations`].
    Translations,
    /// [`Metrics::translations_loaded`].
    TranslationsLoaded,
    /// [`Metrics::ring_writes`].
    RingWrites,
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
    /// [`Gauges::shared_idle_bytes`].
    SharedIdleBytes,
    /// [`Gauges::private_idle_bytes`].
    PrivateIdleBytes,
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
    submissions: AtomicU64,
    passes: AtomicU64,
    barriers: AtomicU64,
    bytes_copied: AtomicU64,
    cache_bytes_written: AtomicU64,
    pipelines_created: AtomicU64,
    first_use_waits: AtomicU64,
    first_use_wait_nanos: AtomicU64,
    first_use_wait_max_nanos: AtomicU64,
    blocks_made: AtomicU64,
    dispatches_without_effects: AtomicU64,
    dedicated_blocks_reused: AtomicU64,
    translations: AtomicU64,
    translations_loaded: AtomicU64,
    ring_writes: AtomicU64,
    upload_waits: AtomicU64,
    synchronizations: AtomicU64,
    commit_lock_wait_nanos: AtomicU64,
    commit_lock_wait_max_nanos: AtomicU64,
    groups_beside_another_queue: AtomicU64,
    lone_submissions_across_queues: AtomicU64,
    forced_flushes_across_queues: AtomicU64,
    over_copies: AtomicU64,
    buffers: AtomicU64,
    textures: AtomicU64,
    samplers: AtomicU64,
    pipelines: AtomicU64,
    kernels: AtomicU64,
    kernel_entries: AtomicU64,
    render_contents: AtomicU64,
    command_buffers: AtomicU64,
    descriptor_slots: AtomicU64,
    blocks: AtomicU64,
    events: AtomicU64,
    device_samplers: AtomicU64,
    spilled_bytes: AtomicU64,
    shared_idle_bytes: AtomicU64,
    private_idle_bytes: AtomicU64,
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
            Metric::DedicatedBlocksReused => &a.dedicated_blocks_reused,
            Metric::DispatchesWithoutEffects => &a.dispatches_without_effects,
            Metric::Translations => &a.translations,
            Metric::TranslationsLoaded => &a.translations_loaded,
            Metric::RingWrites => &a.ring_writes,
            Metric::UploadWaits => &a.upload_waits,
            Metric::Synchronizations => &a.synchronizations,
            Metric::GroupsBesideAnotherQueue => &a.groups_beside_another_queue,
            Metric::LoneSubmissionsAcrossQueues => &a.lone_submissions_across_queues,
            Metric::ForcedFlushesAcrossQueues => &a.forced_flushes_across_queues,
            Metric::OverCopies => &a.over_copies,
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

    /// `bytes` more copied for a call ([`Metrics::bytes_copied`]).
    pub(crate) fn copied(&self, bytes: u64) {
        self.atomics.bytes_copied.fetch_add(bytes, Relaxed);
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
            Gauge::Blocks => &a.blocks,
            Gauge::CommandBuffers => &a.command_buffers,
            Gauge::Events => &a.events,
            Gauge::DeviceSamplers => &a.device_samplers,
            Gauge::SpilledBytes => &a.spilled_bytes,
            Gauge::SharedIdleBytes => &a.shared_idle_bytes,
            Gauge::PrivateIdleBytes => &a.private_idle_bytes,
        }
    }

    /// No Metal counterpart: the cumulative counts since the device was made.
    pub fn metrics(&self) -> Metrics {
        let a = &self.atomics;
        Metrics {
            submissions: a.submissions.load(Relaxed),
            passes: a.passes.load(Relaxed),
            barriers: a.barriers.load(Relaxed),
            bytes_copied: a.bytes_copied.load(Relaxed),
            pipelines_created: a.pipelines_created.load(Relaxed),
            first_use_waits: a.first_use_waits.load(Relaxed),
            first_use_wait_nanos: a.first_use_wait_nanos.load(Relaxed),
            first_use_wait_max_nanos: a.first_use_wait_max_nanos.load(Relaxed),
            blocks_made: a.blocks_made.load(Relaxed),
            dedicated_blocks_reused: a.dedicated_blocks_reused.load(Relaxed),
            dispatches_without_effects: a.dispatches_without_effects.load(Relaxed),
            translations: a.translations.load(Relaxed),
            translations_loaded: a.translations_loaded.load(Relaxed),
            cache_bytes_written: a.cache_bytes_written.load(Relaxed),
            ring_writes: a.ring_writes.load(Relaxed),
            upload_waits: a.upload_waits.load(Relaxed),
            synchronizations: a.synchronizations.load(Relaxed),
            commit_lock_wait_nanos: a.commit_lock_wait_nanos.load(Relaxed),
            commit_lock_wait_max_nanos: a.commit_lock_wait_max_nanos.load(Relaxed),
            groups_beside_another_queue: a.groups_beside_another_queue.load(Relaxed),
            lone_submissions_across_queues: a.lone_submissions_across_queues.load(Relaxed),
            forced_flushes_across_queues: a.forced_flushes_across_queues.load(Relaxed),
            over_copies: a.over_copies.load(Relaxed),
        }
    }

    /// No Metal counterpart: the live count of each kind of resource.
    pub fn gauges(&self) -> Gauges {
        let a = &self.atomics;
        Gauges {
            buffers: a.buffers.load(Relaxed),
            textures: a.textures.load(Relaxed),
            samplers: a.samplers.load(Relaxed),
            pipelines: a.pipelines.load(Relaxed),
            kernels: a.kernels.load(Relaxed),
            kernel_entries: a.kernel_entries.load(Relaxed),
            render_contents: a.render_contents.load(Relaxed),
            command_buffers: a.command_buffers.load(Relaxed),
            descriptor_slots: a.descriptor_slots.load(Relaxed),
            blocks: a.blocks.load(Relaxed),
            events: a.events.load(Relaxed),
            device_samplers: a.device_samplers.load(Relaxed),
            spilled_bytes: a.spilled_bytes.load(Relaxed),
            shared_idle_bytes: a.shared_idle_bytes.load(Relaxed),
            private_idle_bytes: a.private_idle_bytes.load(Relaxed),
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
    /// Encoders recorded: compute and blit, and later render.
    pub passes: u64,
    /// Pipeline-barrier commands recorded into command buffers' work: one at the end of each
    /// segment with commands and of each upload, one before a dispatch at a hazard, one owed by
    /// an encoder that wrote or a memory barrier call, recorded before the next command, and two
    /// around each blit copy, fill and synchronize copy, the first of which pays what is owed. A block's zero fill, made
    /// apart from any command buffer, is not counted.
    pub barriers: u64,
    /// Bytes the runtime copied for a call: `replace_region` and `get_bytes` staging, and the
    /// copies between a texture over a buffer and its image of its own
    /// ([`over_copies`](Self::over_copies)). A blit's own copy, and a synchronize's, is the
    /// program's and is not counted.
    pub bytes_copied: u64,
    /// `VkPipeline`s created.
    pub pipelines_created: u64,
    /// First-use waits: encoding calls that waited while the `VkPipeline` they needed was made,
    /// for a threadgroup size or threadgroup memory lengths no state of the kernel's content had
    /// run with before.
    pub first_use_waits: u64,
    /// The nanoseconds of [`host_time`](crate::host_time) the first-use waits took, in total.
    pub first_use_wait_nanos: u64,
    /// The longest first-use wait, in nanoseconds; a maximum since the device was made, so the
    /// difference of two reads does not apply to it.
    pub first_use_wait_max_nanos: u64,
    /// `VkDeviceMemory` allocations made: pooled blocks (the texture blocks among them) and
    /// dedicated ones, the zero source and the parameter ring's blocks.
    pub blocks_made: u64,
    /// Dedicated blocks taken again: a request past its class's pooled length that took a
    /// dedicated block given back and kept idle, which it fits and fills at least half of,
    /// instead of a new allocation.
    pub dedicated_blocks_reused: u64,
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
    /// Bytes written to the cache directory ([`DeviceOptions::pipeline_cache_dir`]): lowered
    /// kernels, and each `VkPipeline`'s data where it did not come from its file.
    ///
    /// [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
    pub cache_bytes_written: u64,
    /// Writes into the parameter ring: a dispatch's parameter block, its fields past the push
    /// window (a kernel with more than 28 buffers), and each `set_bytes` copy.
    pub ring_writes: u64,
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
    /// Submissions sent with fewer than two of their own queue's submissions in flight while the
    /// device had two or more in flight across its queues: the ones one queue of the same
    /// command buffers would have held in a group.
    pub lone_submissions_across_queues: u64,
    /// Groups sent with no reason of their own queue's: neither full, nor ended by an event's
    /// signal or a semaphore wait, nor below two submissions in flight, nor asked for by a host
    /// wait or the end. No path of the submit thread sends one, so this stays 0; it is counted to
    /// show it.
    pub forced_flushes_across_queues: u64,
    /// Copies between a texture over a buffer the device's linear image does not hold in place
    /// and its image of its own ([`Buffer::new_texture`](crate::Buffer::new_texture)): the
    /// buffer's bytes into the image, in the upload that leads a submission or after a command
    /// wrote the buffer, and the image back into the buffer after a command wrote the texture.
    pub over_copies: u64,
}

/// No Metal counterpart: the live count of each kind of resource. Every one is zero once the
/// device is torn down.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gauges {
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
    /// Command buffers not yet retired.
    pub command_buffers: u64,
    /// Occupied indices of set 1's arrays: one for each texture made at an index and each sampler
    /// state.
    pub descriptor_slots: u64,
    /// Live `VkDeviceMemory` allocations: blocks the consumer holds, blocks ready or idle in the
    /// pool, and the runtime's own.
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
    /// Bytes of idle Shared blocks the pool keeps, pooled and dedicated, zeroed, that no one
    /// holds: the ready block and at most the class's bound of 256 MiB beside it.
    pub shared_idle_bytes: u64,
    /// Bytes of idle Private blocks the pool keeps, pooled and dedicated, zeroed, that no one
    /// holds: the ready block and at most the class's bound of 128 MiB beside it.
    pub private_idle_bytes: u64,
}
