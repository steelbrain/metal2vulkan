//! The new runtime (`metal2vulkan-rt`) behind the harness's traits.
//!
//! The new runtime places a buffer where its consumer says, in a block the consumer took
//! (`Device::take_block`, `Device::new_buffer`); the harness is that consumer here. It places each
//! buffer at the lowest free range of a block of its storage mode that holds it, at a multiple of
//! 256 bytes, taking a pooled block when none does and a dedicated block for a buffer longer than
//! a pooled one. A dropped buffer frees its range at once, or releases its dedicated block: the
//! harness drops a buffer only once the work that used it has completed, as Metal frees a
//! buffer's memory only then. So a buffer made after one of the same length was dropped takes
//! the same range, which the runtime zeroes as `newBufferWithLength:options:` does.
//!
//! The runtime makes buffers only so far, so the command runs its creation workloads alone
//! ([`run::Buffers`](crate::run::Buffers)).

use crate::{Result, Storage};
use ash::vk;
use metal2vulkan_rt as rt;
use std::cell::RefCell;
use std::ops::Range;
use std::ptr::NonNull;
use std::rc::Rc;

/// The alignment of each buffer's offset in its block: the largest
/// `minStorageBufferOffsetAlignment` Vulkan allows.
const ALIGN: u64 = 256;

/// The new runtime's `MTLCreateSystemDefaultDevice`: the first physical device of the most
/// preferred type (discrete, integrated, virtual, CPU, other) that opens, in the loader's order
/// within a type, as the old runtime chooses.
pub fn system_default() -> Result<Device> {
    let instance = rt::Instance::new(&[])?;
    let (_, vk_instance) = instance.raw();
    let mut candidates: Vec<(u8, String, rt::PhysicalDevice)> = (instance.physical_devices()?)
        .into_iter()
        .map(|physical| {
            // SAFETY: `physical` is one of the instance's physical devices, and the instance is
            // live.
            let properties = unsafe { vk_instance.get_physical_device_properties(physical.raw()) };
            let rank = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                _ => 4,
            };
            let name = (properties.device_name_as_c_str()).map_or_else(
                |_| "?".to_owned(),
                |name| name.to_string_lossy().into_owned(),
            );
            (rank, name, physical)
        })
        .collect();
    candidates.sort_by_key(|(rank, ..)| *rank);
    let mut refused = Vec::new();
    for (_, name, physical) in candidates {
        match rt::Device::new(&physical, rt::DeviceOptions::default()) {
            Ok(device) => {
                return Ok(Device {
                    heap: Rc::new(RefCell::new(Heap {
                        device: device.clone(),
                        blocks: Vec::new(),
                    })),
                    device,
                    name,
                })
            }
            Err(why) => refused.push(format!("{name}: {why}")),
        }
    }
    Err(format!("no physical device opens: {}", refused.join("; ")).into())
}

/// The new runtime's device, with the harness's placements in its blocks.
pub struct Device {
    device: rt::Device,
    name: String,
    heap: Rc<RefCell<Heap>>,
}

/// `MTLBuffer`: the runtime's buffer, and its place, freed when it is dropped.
pub struct Buffer {
    buffer: rt::Buffer,
    placed: Placed,
    heap: Rc<RefCell<Heap>>,
}

/// The blocks the harness took, each with its free ranges.
struct Heap {
    device: rt::Device,
    blocks: Vec<Block>,
}

/// A pooled block the harness took.
struct Block {
    id: rt::BlockId,
    class: rt::MemoryClass,
    free: Ranges,
}

/// Where a buffer lies: a range of a pooled block, or a dedicated block of its own (`None`).
struct Placed {
    block: rt::BlockId,
    range: Option<Range<u64>>,
}

impl Heap {
    /// A place for a buffer of `length` bytes in `class` memory.
    fn place(&mut self, class: rt::MemoryClass, length: u64) -> Result<Placed> {
        if length > rt::BLOCK_LEN {
            let block = self.device.take_block(class, length)?.id;
            return Ok(Placed { block, range: None });
        }
        let span = length.next_multiple_of(ALIGN);
        let found = (self.blocks.iter_mut())
            .filter(|block| block.class == class)
            .find_map(|block| Some((block.id, block.free.take(span)?)));
        let (block, start) = match found {
            Some(found) => found,
            None => {
                let info = self.device.take_block(class, span)?;
                let mut free = Ranges::new(info.len);
                let start =
                    (free.take(span)).ok_or("a new pooled block is shorter than a buffer")?;
                self.blocks.push(Block {
                    id: info.id,
                    class,
                    free,
                });
                (info.id, start)
            }
        };
        Ok(Placed {
            block,
            range: Some(start..start + span),
        })
    }

    /// Frees `placed`: its range, or its dedicated block, which the runtime frees once its last
    /// buffer is gone.
    fn free(&mut self, placed: &Placed) {
        match &placed.range {
            Some(range) => {
                let block = (self.blocks.iter_mut())
                    .find(|block| block.id == placed.block)
                    .expect("the harness keeps every pooled block it placed a buffer in");
                block.free.give(range.clone());
            }
            None => (self.device.release_block(placed.block))
                .expect("the harness holds each dedicated block until its buffer is dropped"),
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.heap.borrow_mut().free(&self.placed);
    }
}

/// A block's free ranges, in order, none touching another.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Ranges(pub(crate) Vec<Range<u64>>);

impl Ranges {
    /// A block of `len` bytes, all free.
    pub(crate) fn new(len: u64) -> Ranges {
        Ranges(vec![0..len])
    }

    /// The start of the lowest free range at least `len` bytes long, its first `len` bytes taken;
    /// `None` where no free range is that long.
    pub(crate) fn take(&mut self, len: u64) -> Option<u64> {
        let at = self
            .0
            .iter()
            .position(|free| free.end - free.start >= len)?;
        let start = self.0[at].start;
        if self.0[at].end - start == len {
            self.0.remove(at);
        } else {
            self.0[at].start += len;
        }
        Some(start)
    }

    /// Gives `range`, a taken one, back, merged with the free ranges it touches.
    pub(crate) fn give(&mut self, range: Range<u64>) {
        let at = self.0.partition_point(|free| free.start < range.start);
        let mut range = range;
        if self.0.get(at).is_some_and(|next| next.start == range.end) {
            range.end = self.0.remove(at).end;
        }
        match at.checked_sub(1).map(|before| &mut self.0[before]) {
            Some(before) if before.end == range.start => before.end = range.end,
            _ => self.0.insert(at, range),
        }
    }
}

impl crate::Device for Device {
    type Buffer = Buffer;

    fn name(&self) -> String {
        self.name.clone()
    }

    /// Placed as the module's docs say, and zeroed by the runtime.
    fn new_buffer(&self, length: u64, storage: Storage) -> Result<Buffer> {
        let class = match storage {
            Storage::Shared => rt::MemoryClass::Shared,
            Storage::Private => rt::MemoryClass::Private,
        };
        let placed = self.heap.borrow_mut().place(class, length)?;
        let start = placed.range.as_ref().map_or(0, |range| range.start);
        match self.device.new_buffer(placed.block, start, length, true) {
            Ok(buffer) => Ok(Buffer {
                buffer,
                placed,
                heap: Rc::clone(&self.heap),
            }),
            Err(why) => {
                self.heap.borrow_mut().free(&placed);
                Err(why.into())
            }
        }
    }

    /// Its live buffers, and its live `VkDeviceMemory` blocks: those the harness holds, those
    /// its pool keeps ready or idle, and its own.
    fn gauges(&self) -> Vec<(&'static str, u64)> {
        let gauges = self.device.counters().gauges();
        vec![("buffers", gauges.buffers), ("blocks", gauges.blocks)]
    }
}

impl crate::Buffer for Buffer {
    fn contents(&self) -> Option<NonNull<u8>> {
        self.buffer.contents()
    }
}
