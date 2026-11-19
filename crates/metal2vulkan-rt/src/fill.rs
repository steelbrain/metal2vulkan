//! Zero fills of memory the host does not map: fill and copy commands, recorded into the
//! upload that goes ahead of later commits.

use crate::counters::{Counters, Gauge, Metric, Site};
use crate::error::Error;
use crate::memory;
use ash::vk;

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

/// The zero source of the copies [`record`] makes: [`ZEROS`] bytes that read zero.
#[derive(Default)]
pub(crate) struct Filler {
    zeros: vk::Buffer,
    zeros_memory: vk::DeviceMemory,
}

impl Filler {
    /// Makes the zero source, whose allocation counts against `max_allocations`, the device's
    /// `maxMemoryAllocationCount` ([`memory::allocate`]).
    ///
    /// # Safety
    ///
    /// `device` is live; `host_type` is a host-visible, host-coherent memory type that buffers of
    /// every usage can use.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        host_type: u32,
        counters: &Counters,
        max_allocations: u32,
    ) -> Result<Filler, Error> {
        let mut filler = Filler::default();
        // SAFETY: the caller's contract.
        match unsafe { filler.make(device, host_type, counters, max_allocations) } {
            Ok(()) => Ok(filler),
            Err(error) => {
                // SAFETY: nothing made is in use.
                if unsafe { filler.destroy(device) } {
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

    /// Destroys the zero source. True where it freed its allocation, whose gauge the caller
    /// lowers.
    ///
    /// # Safety
    ///
    /// `device` is the one it was made on, and no command that reads it is pending. A null handle
    /// is skipped.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) -> bool {
        let allocated = self.zeros_memory != vk::DeviceMemory::null();
        // SAFETY: the caller's contract; destroying a null handle does nothing.
        unsafe {
            device.destroy_buffer(self.zeros, None);
            if allocated {
                device.free_memory(self.zeros_memory, None);
            }
        }
        allocated
    }

    /// # Safety
    ///
    /// As [`Filler::new`]. What it made before an error stays in `self`, for
    /// [`Filler::destroy`].
    unsafe fn make(
        &mut self,
        device: &ash::Device,
        host_type: u32,
        counters: &Counters,
        max_allocations: u32,
    ) -> Result<(), Error> {
        let zeros = vk::BufferCreateInfo::default()
            .size(ZEROS)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the caller's contract; the buffer is made before it is bound, and the memory is
        // mapped only here.
        unsafe {
            self.zeros = device
                .create_buffer(&zeros, None)
                .map_err(Error::vulkan("vkCreateBuffer"))?;
            let requirements = device.get_buffer_memory_requirements(self.zeros);
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(host_type);
            self.zeros_memory = memory::allocate(
                device,
                counters,
                max_allocations,
                &allocate,
                Some(Site::ZeroSource),
            )?;
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
}
