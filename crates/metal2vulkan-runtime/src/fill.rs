//! Zero fills of memory the host does not map, and the zeros of new textures no clear writes:
//! fill and copy commands, recorded into the upload that goes ahead of later commits.

use crate::counters::{Counters, Gauge, Metric, Site};
use crate::device::Core;
use crate::error::Error;
use crate::memory::{self, Internal};
use crate::texture::CopyUnit;
use ash::vk;

/// Bytes in the zero source: at least the 6 of a range with no whole 4-byte word inside it, and
/// a row of blocks of a block-compressed texture 16,384 texels wide (4,096 blocks of 16 bytes),
/// so that [`zero_regions`] copies a texture of Metal's widest in bands of whole rows.
pub(crate) const ZEROS: u64 = 64 << 10;

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

/// The zero source the commands that zero `offset..end` copy from ([`plan`]): made where they copy
/// bytes outside whole words ([`zeros`]), so a call before the block-maker thread made it waits
/// for that make; `None` where they fill whole words alone, which waits for nothing.
///
/// Err: what [`zeros`] returns.
pub(crate) fn zeros_for(core: &Core, offset: u64, end: u64) -> Result<Option<vk::Buffer>, Error> {
    match plan(offset, end).copies.is_empty() {
        true => Ok(None),
        false => zeros(core).map(Some),
    }
}

/// Records the commands that zero `offset..end` of `buffer` into `command`.
///
/// # Safety
///
/// `command` is recording; `zeros` is what [`zeros_for`] answered for `offset..end`; `buffer` has
/// transfer-destination usage and `offset..end` lies in it.
pub(crate) unsafe fn record(
    device: &ash::Device,
    command: vk::CommandBuffer,
    zeros: Option<vk::Buffer>,
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
        if let Some(zeros) = zeros.filter(|_| !copies.is_empty()) {
            device.cmd_copy_buffer(command, zeros, buffer, &copies);
        }
    }
}

/// The copies from the start of the zero source that write every element of an image of
/// `unit`'s elements, of level-0 `extent`, `levels` and `layers` (`volume`: a 3D image, whose
/// depth slices are copied one by one): for each level, layer and slice, bands of whole rows of
/// elements of at most [`ZEROS`] bytes, and where one row is larger, runs along each row of at
/// most [`ZEROS`] bytes. Each copy's extent is whole elements, or reaches the subresource's edge.
pub(crate) fn zero_regions(
    unit: CopyUnit,
    extent: [u32; 3],
    levels: u32,
    layers: u32,
    volume: bool,
) -> Vec<vk::BufferImageCopy> {
    let [bw, bh] = unit.block;
    let bytes = u64::from(unit.bytes);
    let mut regions = Vec::new();
    for level in 0..levels {
        let [w, h, d] = extent.map(|axis| (axis >> level.min(31)).max(1));
        let (across, down) = (w.div_ceil(bw), h.div_ceil(bh));
        let row = u64::from(across) * bytes;
        // Elements along a row and rows of them in one copy, each at most `ZEROS` bytes in all.
        let (run, band) = match row <= ZEROS {
            true => (across, (ZEROS / row).min(u64::from(down)) as u32),
            false => ((ZEROS / bytes) as u32, 1),
        };
        let slices = if volume { d } else { 1 };
        for layer in 0..layers {
            for z in 0..slices {
                for y in (0..down).step_by(band as usize) {
                    for x in (0..across).step_by(run as usize) {
                        let [left, top] = [x * bw, y * bh];
                        regions.push(vk::BufferImageCopy {
                            buffer_offset: 0,
                            buffer_row_length: 0,
                            buffer_image_height: 0,
                            image_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: unit.aspect,
                                mip_level: level,
                                base_array_layer: layer,
                                layer_count: 1,
                            },
                            image_offset: vk::Offset3D {
                                x: left as i32,
                                y: top as i32,
                                z: z as i32,
                            },
                            image_extent: vk::Extent3D {
                                width: (run * bw).min(w - left),
                                height: (band * bh).min(h - top),
                                depth: 1,
                            },
                        });
                    }
                }
            }
        }
    }
    regions
}

/// The zero source, made: the block-maker thread makes it as its first work once the device's
/// making starts it (`crate::memory::Internal::ZeroSource`), off every caller's thread, so a call
/// before then waits for that make
/// ([`Metrics::maker_waits`](crate::Metrics::maker_waits)). Where that make failed, the call makes
/// it on its own thread, counted in
/// [`Metrics::zero_source_allocations`](crate::Metrics::zero_source_allocations).
///
/// Err: what [`Filler::new`] returns, where the block-maker thread made none.
pub(crate) fn zeros(core: &Core) -> Result<vk::Buffer, Error> {
    if core.filler.get().is_none() {
        core.pool.await_made(Internal::ZeroSource);
        make(core)?;
    }
    Ok(made(core))
}

/// The zero source where it is made, for commands recorded under the work lock: a call that
/// records a copy from it made it first ([`zeros`]); a null handle, which no command reads, before
/// then.
pub(crate) fn made(core: &Core) -> vk::Buffer {
    core.filler.get().map_or(vk::Buffer::null(), Filler::zeros)
}

/// Makes the device's zero source where none is made yet: on the block-maker thread, or on a
/// call's own thread where that thread made none. One made while another call's was kept first is
/// destroyed.
///
/// Err: what [`Filler::new`] returns.
pub(crate) fn make(core: &Core) -> Result<(), Error> {
    if core.filler.get().is_some() {
        return Ok(());
    }
    // SAFETY: the device is live; the Shared type is host-visible, and the floor chose it for
    // buffers of every usage.
    let filler = unsafe {
        Filler::new(
            &core.device,
            core.memory_types.shared,
            core.coherence.is_some(),
            &core.counters,
            core.max_allocations,
        )
    }?;
    if let Err(mut second) = core.filler.set(filler) {
        // SAFETY: nothing used it.
        if unsafe { second.destroy(&core.device) } {
            core.counters.fall(Gauge::Blocks);
        }
    }
    Ok(())
}

/// The zero source of the copies [`record`] and [`zero_regions`] make: [`ZEROS`] bytes that read
/// zero.
#[derive(Default)]
pub(crate) struct Filler {
    zeros: vk::Buffer,
    zeros_memory: vk::DeviceMemory,
}

impl Filler {
    /// Makes the zero source, whose allocation counts against `max_allocations`, the device's
    /// `maxMemoryAllocationCount` ([`memory::allocate`]); `flush` where `host_type` is not
    /// host-coherent, so its zeros are flushed (`crate::coherence`).
    ///
    /// # Safety
    ///
    /// `device` is live; `host_type` is a host-visible memory type that buffers of every usage
    /// can use.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        host_type: u32,
        flush: bool,
        counters: &Counters,
        max_allocations: u32,
    ) -> Result<Filler, Error> {
        let mut filler = Filler::default();
        // SAFETY: the caller's contract.
        match unsafe { filler.make(device, host_type, flush, counters, max_allocations) } {
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

    /// The zero source, for [`record`] and [`zero_regions`].
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
                memory::free(device, self.zeros_memory);
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
        flush: bool,
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
            // A new allocation's contents are undefined. Where the Shared type is not
            // host-coherent, the zeros are flushed before the memory is unmapped
            // (`crate::coherence`): the mapping's end is a multiple of every atom, so the whole
            // of it is named.
            pointer.cast::<u8>().write_bytes(0, ZEROS as usize);
            if flush {
                let zeros = [vk::MappedMemoryRange::default()
                    .memory(self.zeros_memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                let flushed = device.flush_mapped_memory_ranges(&zeros);
                counters.coherence(crate::coherence::Call::Flush, ZEROS, flushed.is_err());
            }
            device.unmap_memory(self.zeros_memory);
        }
        Ok(())
    }
}
