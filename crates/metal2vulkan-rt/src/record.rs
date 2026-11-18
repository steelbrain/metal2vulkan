//! Recording: a command pool and the command buffers allocated from it, recycled from one Metal
//! command buffer (or upload) to the next once its work has completed.

use crate::counters::{Counters, Metric};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::hazard::Hazards;
use crate::reaper::{Free, Job};
use ash::vk;
use std::sync::Mutex;

/// The most idle pools the device keeps; one given back past them is destroyed.
const IDLE_MAX: usize = 64;

/// A command pool and the command buffers allocated from it so far, the first `used` of them
/// recording or recorded, and the hazard sets of the Metal command buffer recording with it,
/// recycled with it so their table is not made again for each command buffer.
pub(crate) struct Recorder {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    used: usize,
    pub(crate) hazards: Hazards,
}

impl Recorder {
    /// The next command buffer, begun for one submission.
    pub(crate) fn next(&mut self, device: &ash::Device) -> Result<vk::CommandBuffer, Error> {
        if self.used == self.buffers.len() {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the pool is this recorder's, used by one thread at a time.
            let buffer = unsafe { device.allocate_command_buffers(&info) }
                .map_err(Error::vulkan("vkAllocateCommandBuffers"))?[0];
            self.buffers.push(buffer);
        }
        let buffer = self.buffers[self.used];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the buffer is in its initial state: allocated now or reset with its pool.
        unsafe { device.begin_command_buffer(buffer, &begin) }
            .map_err(Error::vulkan("vkBeginCommandBuffer"))?;
        self.used += 1;
        Ok(buffer)
    }
}

/// What a barrier orders, and what it makes earlier writes visible to.
#[derive(Clone, Copy)]
pub(crate) enum Reach {
    /// Within a segment: the dispatches after it after the dispatches before it, their shader
    /// reads and writes seeing the earlier shader writes, and where `indirect` (the dispatch that
    /// follows it reads its arguments from a buffer), their indirect argument reads too. The
    /// other commands a segment records that read or write memory, blit copies and render
    /// passes, order themselves against every command before and after them with barriers of
    /// their own, a pass's after it owed where it moves no attachment and recorded as
    /// [`Reach::Commands`] whatever reach pays it (`Building::owe_pass`), so it orders the
    /// segment's commands as [`Reach::Commands`] would.
    Dispatches { indirect: bool },
    /// Every later command, in this submission and every later one on the queue.
    Commands,
    /// Every later command, and the host once it has waited for the work.
    Host,
}

impl Reach {
    /// Whether the barrier orders indirect argument reads after the writes before it.
    pub(crate) fn reaches_indirect(self) -> bool {
        match self {
            Reach::Dispatches { indirect } => indirect,
            Reach::Commands | Reach::Host => true,
        }
    }
}

/// Records one global memory barrier: the writes earlier in submission order are made visible
/// to what `reach` names, and the commands it names wait for the earlier ones. It is the only
/// barrier the runtime records for buffers: none names a buffer or a range. The caller counts it
/// where it is part of the queues' work.
pub(crate) fn barrier(device: &ash::Device, buffer: vk::CommandBuffer, reach: Reach) {
    let barrier = [memory_barrier(reach)];
    let dependency = vk::DependencyInfo::default().memory_barriers(&barrier);
    // SAFETY: the buffer is recording.
    unsafe { device.cmd_pipeline_barrier2(buffer, &dependency) };
}

/// The global memory barrier [`barrier`] records for `reach`, for a caller that records it with
/// image barriers of its own in one command.
pub(crate) fn memory_barrier(reach: Reach) -> vk::MemoryBarrier2<'static> {
    let (source, written, stages, accesses) = match reach {
        Reach::Dispatches { indirect: false } => (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        ),
        Reach::Dispatches { indirect: true } => (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT,
            vk::AccessFlags2::SHADER_READ
                | vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::INDIRECT_COMMAND_READ,
        ),
        Reach::Commands => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        ),
        Reach::Host => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::MEMORY_READ
                | vk::AccessFlags2::MEMORY_WRITE
                | vk::AccessFlags2::HOST_READ
                | vk::AccessFlags2::HOST_WRITE,
        ),
    };
    vk::MemoryBarrier2::default()
        .src_stage_mask(source)
        .src_access_mask(written)
        .dst_stage_mask(stages)
        .dst_access_mask(accesses)
}

/// Ends a segment's command buffer after its one barrier: a later command buffer's commands run
/// after this one's, as Metal orders the command buffers of a queue, and where `host` (the segment
/// wrote Shared memory, or signals an event) the barrier also reaches the host, so Shared memory
/// reads the GPU's writes once the command buffer completes or the host sees the event. A segment
/// begins with no barrier: the one ending each earlier segment, upload and fill covers it.
pub(crate) fn end(
    device: &ash::Device,
    buffer: vk::CommandBuffer,
    host: bool,
    counters: &Counters,
) -> Result<(), Error> {
    barrier(
        device,
        buffer,
        if host { Reach::Host } else { Reach::Commands },
    );
    counters.count(Metric::Barriers);
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if host {
        counters.reach_host();
    }
    // SAFETY: the buffer is recording.
    unsafe { device.end_command_buffer(buffer) }.map_err(Error::vulkan("vkEndCommandBuffer"))
}

/// The device's idle recorders.
pub(crate) struct Recorders {
    family: u32,
    idle: Mutex<Vec<Recorder>>,
}

impl Recorders {
    pub(crate) fn new(family: u32) -> Recorders {
        Recorders {
            family,
            idle: Mutex::new(Vec::new()),
        }
    }

    /// An idle recorder, or a new one.
    pub(crate) fn take(&self, device: &ash::Device) -> Result<Recorder, Error> {
        if let Some(recorder) = lock(&self.idle).pop() {
            return Ok(recorder);
        }
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(self.family);
        // SAFETY: the device is live.
        let pool = unsafe { device.create_command_pool(&info, None) }
            .map_err(Error::vulkan("vkCreateCommandPool"))?;
        Ok(Recorder {
            pool,
            buffers: Vec::new(),
            used: 0,
            hazards: Hazards::default(),
        })
    }

    /// Takes back a recorder whose work has completed, or that never submitted: its pool is reset
    /// for the next one, or, past [`IDLE_MAX`], given to the reaper to destroy.
    ///
    /// The reset keeps the pool's memory for its next command buffer. Releasing it at every reset
    /// made the UHD Graphics 630's Windows driver take about 170 us more per command buffer (a
    /// burst of 2,000 command buffers 19.6 ms against 360 ms, job m3-max-46545).
    pub(crate) fn give_back(&self, core: &Core, mut recorder: Recorder) {
        // SAFETY: no command buffer of the pool is pending. A failed reset (a lost device) leaves
        // the pool to be destroyed.
        let reset = unsafe {
            core.device
                .reset_command_pool(recorder.pool, vk::CommandPoolResetFlags::empty())
        };
        let mut idle = lock(&self.idle);
        if reset.is_ok() && idle.len() < IDLE_MAX {
            recorder.used = 0;
            recorder.hazards.clear();
            idle.push(recorder);
        } else {
            drop(idle);
            core.reaper
                .give(Job::Free(Free::CommandPool(recorder.pool)));
        }
    }

    /// At teardown: destroys every idle recorder.
    pub(crate) fn destroy_all(&mut self, device: &ash::Device) {
        let idle = self
            .idle
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for recorder in idle.drain(..) {
            // SAFETY: the device is idle; destroying the pool frees its command buffers.
            unsafe { device.destroy_command_pool(recorder.pool, None) };
        }
    }
}
