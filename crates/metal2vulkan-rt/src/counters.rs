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
}

/// A gauge the runtime moves.
#[derive(Clone, Copy)]
pub(crate) enum Gauge {
    /// [`Gauges::buffers`].
    Buffers,
    /// [`Gauges::pipelines`].
    Pipelines,
    /// [`Gauges::blocks`].
    Blocks,
    /// [`Gauges::command_buffers`].
    CommandBuffers,
    /// [`Gauges::events`].
    Events,
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
    pipelines_created: AtomicU64,
    blocks_made: AtomicU64,
    buffers: AtomicU64,
    textures: AtomicU64,
    samplers: AtomicU64,
    pipelines: AtomicU64,
    command_buffers: AtomicU64,
    descriptor_slots: AtomicU64,
    blocks: AtomicU64,
    events: AtomicU64,
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
        };
        atomic.fetch_add(1, Relaxed);
    }

    /// One more live resource of `gauge`'s kind.
    pub(crate) fn rise(&self, gauge: Gauge) {
        self.gauge(gauge).fetch_add(1, Relaxed);
    }

    /// One fewer live resource of `gauge`'s kind.
    pub(crate) fn fall(&self, gauge: Gauge) {
        self.gauge(gauge).fetch_sub(1, Relaxed);
    }

    fn gauge(&self, gauge: Gauge) -> &AtomicU64 {
        let a = &self.atomics;
        match gauge {
            Gauge::Buffers => &a.buffers,
            Gauge::Pipelines => &a.pipelines,
            Gauge::Blocks => &a.blocks,
            Gauge::CommandBuffers => &a.command_buffers,
            Gauge::Events => &a.events,
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
            blocks_made: a.blocks_made.load(Relaxed),
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
            command_buffers: a.command_buffers.load(Relaxed),
            descriptor_slots: a.descriptor_slots.load(Relaxed),
            blocks: a.blocks.load(Relaxed),
            events: a.events.load(Relaxed),
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
    /// Encoders recorded: compute, and later blit and render.
    pub passes: u64,
    /// Pipeline-barrier commands recorded into command buffers' work: one at the end of each
    /// segment with commands and of each upload, one before a dispatch at a hazard, and one owed
    /// by an encoder that wrote or a memory barrier call, recorded before the next command. A
    /// block's zero fill, made apart from any command buffer, is not counted.
    pub barriers: u64,
    /// Bytes the runtime copied for a call: `replace_region` and `get_bytes` staging, and the
    /// counted copy at commit where a device offers no linear image.
    pub bytes_copied: u64,
    /// `VkPipeline`s created.
    pub pipelines_created: u64,
    /// `VkDeviceMemory` allocations made: pooled and dedicated blocks, the zero source and the
    /// parameter ring's blocks, and later image blocks.
    pub blocks_made: u64,
}

/// No Metal counterpart: the live count of each kind of resource. Every one is zero once the
/// device is torn down.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gauges {
    /// Buffers.
    pub buffers: u64,
    /// Textures, views included.
    pub textures: u64,
    /// Sampler states.
    pub samplers: u64,
    /// `VkPipeline`s.
    pub pipelines: u64,
    /// Command buffers not yet retired.
    pub command_buffers: u64,
    /// Occupied entries of the descriptor arrays.
    pub descriptor_slots: u64,
    /// Live `VkDeviceMemory` allocations: blocks the consumer holds, blocks ready or idle in the
    /// pool, and the runtime's own.
    pub blocks: u64,
    /// Events and shared events (timeline semaphores).
    pub events: u64,
}
