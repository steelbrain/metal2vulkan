//! Zero fills of memory the host does not map: fill and copy commands, recorded into the
//! upload that goes ahead of later commits, or into a command buffer of the fill's own, submitted
//! on the runtime's queue and waited for.

use crate::counters::{Counters, Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::record::{barrier, Reach};
use ash::vk;
use std::sync::{Mutex, PoisonError};

/// Bytes in the zero source: at least the 6 of a range with no whole 4-byte word inside it.
const ZEROS: u64 = 16;

/// The commands that zero `offset..end`: a fill of the whole 4-byte words inside the range, and
/// copies from the zero source for the bytes outside them (a fill writes whole words only).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    /// The words the fill zeroes.
    pub(crate) fill: Option<(u64, u64)>,
    /// The ranges copied from the zero source, each at most 6 bytes.
    pub(crate) copies: Vec<(u64, u64)>,
}

pub(crate) fn plan(offset: u64, end: u64) -> Plan {
    let first = offset.next_multiple_of(4);
    let last = end - end % 4;
    if first < last {
        Plan {
            fill: Some((first, last)),
            copies: [(offset, first), (last, end)]
                .into_iter()
                .filter(|(start, end)| start < end)
                .collect(),
        }
    } else {
        Plan {
            fill: None,
            copies: (offset < end)
                .then_some((offset, end))
                .into_iter()
                .collect(),
        }
    }
}

/// Records the commands that zero `offset..end` of `buffer` into `command`.
///
/// # Safety
///
/// `command` is recording; `zeros` is the zero source; `buffer` has transfer-destination usage and
/// `offset..end` lies in it.
pub(crate) unsafe fn record(
    device: &ash::Device,
    command: vk::CommandBuffer,
    zeros: vk::Buffer,
    buffer: vk::Buffer,
    offset: u64,
    end: u64,
) {
    let plan = plan(offset, end);
    let copies: Vec<vk::BufferCopy> = plan
        .copies
        .iter()
        .map(|&(start, end)| vk::BufferCopy {
            src_offset: 0,
            dst_offset: start,
            size: end - start,
        })
        .collect();
    // SAFETY: the caller's contract.
    unsafe {
        if let Some((start, end)) = plan.fill {
            device.cmd_fill_buffer(command, buffer, start, end - start, 0);
        }
        if !copies.is_empty() {
            device.cmd_copy_buffer(command, zeros, buffer, &copies);
        }
    }
}

/// The zero source, and the objects of a fill submitted and waited for, behind a lock: such fills
/// run one at a time.
pub(crate) struct Filler {
    /// The zero source, [`ZEROS`] bytes that read zero; `state` owns it.
    zeros: vk::Buffer,
    state: Mutex<Objects>,
}

#[derive(Default)]
struct Objects {
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    /// [`ZEROS`] bytes of zero, the source of the copies.
    zeros: vk::Buffer,
    zeros_memory: vk::DeviceMemory,
}

impl Filler {
    /// Makes the command pool, command buffer, fence and zero source.
    ///
    /// # Safety
    ///
    /// `device` is live; `family` is the family of its runtime queue; `host_type` is a
    /// host-visible, host-coherent memory type that buffers of every usage can use.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        family: u32,
        host_type: u32,
        counters: &Counters,
    ) -> Result<Filler, Error> {
        let mut objects = Objects::default();
        // SAFETY: the caller's contract.
        match unsafe { objects.make(device, family, host_type, counters) } {
            Ok(()) => Ok(Filler {
                zeros: objects.zeros,
                state: Mutex::new(objects),
            }),
            Err(error) => {
                // SAFETY: nothing made is in use.
                if unsafe { objects.destroy(device) } {
                    counters.fall(Gauge::Blocks);
                }
                Err(error)
            }
        }
    }

    /// The zero source, for [`record`].
    pub(crate) fn zeros(&self) -> vk::Buffer {
        self.zeros
    }

    /// Zeroes `offset..end` of `buffer` and waits for it.
    pub(crate) fn zero(
        &self,
        core: &Core,
        buffer: vk::Buffer,
        offset: u64,
        end: u64,
    ) -> Result<(), Error> {
        let objects = lock(&self.state);
        let device = &core.device;
        let command = objects.command;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
        let submit = vk::SubmitInfo2::default().command_buffer_infos(&commands);
        // SAFETY: the lock gives this call the pool, command buffer and fence, and the previous
        // fill completed before its call returned; `buffer` has transfer-destination usage and
        // `offset..end` lies in it; the runtime queue is held while it is used.
        unsafe {
            device
                .reset_command_pool(objects.pool, vk::CommandPoolResetFlags::empty())
                .map_err(Error::vulkan("vkResetCommandPool"))?;
            device
                .begin_command_buffer(command, &begin)
                .map_err(Error::vulkan("vkBeginCommandBuffer"))?;
            record(device, command, objects.zeros, buffer, offset, end);
            barrier(device, command, Reach::Commands);
            device
                .end_command_buffer(command)
                .map_err(Error::vulkan("vkEndCommandBuffer"))?;
            {
                let queue = lock(&core.queues.runtime);
                device
                    .queue_submit2(*queue, &[submit], objects.fence)
                    .map_err(Error::vulkan("vkQueueSubmit2"))?;
            }
            core.counters.count(Metric::Submissions);
            device
                .wait_for_fences(&[objects.fence], true, u64::MAX)
                .map_err(Error::vulkan("vkWaitForFences"))?;
            device
                .reset_fences(&[objects.fence])
                .map_err(Error::vulkan("vkResetFences"))?;
        }
        Ok(())
    }

    /// Destroys the fill's objects. True where it freed the zero source's allocation, whose
    /// gauge the caller lowers.
    ///
    /// # Safety
    ///
    /// `device` is the one they were made on, and no fill is pending.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) -> bool {
        let objects = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: the caller's contract.
        unsafe { objects.destroy(device) }
    }
}

impl Objects {
    /// # Safety
    ///
    /// As [`Filler::new`]. What it made before an error stays in `self`, for [`Objects::destroy`].
    unsafe fn make(
        &mut self,
        device: &ash::Device,
        family: u32,
        host_type: u32,
        counters: &Counters,
    ) -> Result<(), Error> {
        let pool = vk::CommandPoolCreateInfo::default().queue_family_index(family);
        let fence = vk::FenceCreateInfo::default();
        let zeros = vk::BufferCreateInfo::default()
            .size(ZEROS)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the caller's contract; each object is made before it is used, and the zero
        // source's memory is mapped only here.
        unsafe {
            self.pool = device
                .create_command_pool(&pool, None)
                .map_err(Error::vulkan("vkCreateCommandPool"))?;
            let command = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.command = device
                .allocate_command_buffers(&command)
                .map_err(Error::vulkan("vkAllocateCommandBuffers"))?[0];
            self.fence = device
                .create_fence(&fence, None)
                .map_err(Error::vulkan("vkCreateFence"))?;
            self.zeros = device
                .create_buffer(&zeros, None)
                .map_err(Error::vulkan("vkCreateBuffer"))?;
            let requirements = device.get_buffer_memory_requirements(self.zeros);
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(host_type);
            self.zeros_memory = device
                .allocate_memory(&allocate, None)
                .map_err(Error::vulkan("vkAllocateMemory"))?;
            counters.rise(Gauge::Blocks);
            counters.count(Metric::BlocksMade);
            device
                .bind_buffer_memory(self.zeros, self.zeros_memory, 0)
                .map_err(Error::vulkan("vkBindBufferMemory"))?;
            let pointer = device
                .map_memory(self.zeros_memory, 0, ZEROS, vk::MemoryMapFlags::empty())
                .map_err(Error::vulkan("vkMapMemory"))?;
            // A new allocation's contents are undefined; the memory is host-coherent, so the
            // zeros need no flush.
            pointer.cast::<u8>().write_bytes(0, ZEROS as usize);
            device.unmap_memory(self.zeros_memory);
        }
        Ok(())
    }

    /// # Safety
    ///
    /// As [`Filler::destroy`]. A null handle is skipped.
    unsafe fn destroy(&mut self, device: &ash::Device) -> bool {
        let allocated = self.zeros_memory != vk::DeviceMemory::null();
        // SAFETY: the caller's contract; destroying a null handle does nothing, and freeing a
        // command pool frees its command buffer.
        unsafe {
            device.destroy_buffer(self.zeros, None);
            if allocated {
                device.free_memory(self.zeros_memory, None);
            }
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
        }
        allocated
    }
}
