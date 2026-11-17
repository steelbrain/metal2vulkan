//! Recording: a command pool and the command buffers allocated from it, recycled from one Metal
//! command buffer (or upload) to the next once its work has completed.

use crate::counters::{Counters, Metric};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::reaper::{Free, Job};
use ash::vk;
use std::sync::Mutex;

/// The most idle pools the device keeps; one given back past them is destroyed.
const IDLE_MAX: usize = 64;

/// A command pool and the command buffers allocated from it so far, the first `used` of them
/// recording or recorded.
pub(crate) struct Recorder {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    used: usize,
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

/// What a barrier makes earlier writes visible to.
#[derive(Clone, Copy)]
pub(crate) enum Reach {
    /// Every later command, in this submission and every later one on the queue.
    Commands,
    /// Every later command, and the host once it has waited for the work.
    Host,
}

/// Records one global memory barrier: every write earlier in submission order is made visible
/// to what `reach` names, and every later command waits for every earlier one. It is the only
/// barrier the runtime records for buffers: none names a buffer or a range. The caller counts it
/// where it is part of the queues' work.
pub(crate) fn barrier(device: &ash::Device, buffer: vk::CommandBuffer, reach: Reach) {
    let (stages, accesses) = match reach {
        Reach::Commands => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        ),
        Reach::Host => (
            vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::MEMORY_READ
                | vk::AccessFlags2::MEMORY_WRITE
                | vk::AccessFlags2::HOST_READ
                | vk::AccessFlags2::HOST_WRITE,
        ),
    };
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(stages)
        .dst_access_mask(accesses)];
    let dependency = vk::DependencyInfo::default().memory_barriers(&barrier);
    // SAFETY: the buffer is recording.
    unsafe { device.cmd_pipeline_barrier2(buffer, &dependency) };
}

/// Ends a segment's command buffer after its one barrier, which reaches the host: Shared memory
/// reads the GPU's writes once the command buffer completes or the host sees an event it signals,
/// and a later command buffer's commands run after this one's, as Metal orders the command
/// buffers of a queue. A segment begins with no barrier: the one ending each earlier segment,
/// upload and fill covers it.
pub(crate) fn end(
    device: &ash::Device,
    buffer: vk::CommandBuffer,
    counters: &Counters,
) -> Result<(), Error> {
    barrier(device, buffer, Reach::Host);
    counters.count(Metric::Barriers);
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
        })
    }

    /// Takes back a recorder whose work has completed, or that never submitted: its pool is reset
    /// for the next one, or, past [`IDLE_MAX`], given to the reaper to destroy.
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
