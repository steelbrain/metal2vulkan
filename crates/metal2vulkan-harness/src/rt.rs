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
//! Its known gaps and the device's limits (`Error::Unsupported`, `Error::Limit`) are typed
//! refusals ([`Refused`]); its other errors are failures. It makes no texture or sampler state
//! yet: a plan with either is refused when it is made, so its texture and sampler types have no
//! value.

use crate::plan::{self, Constant, Filter};
use crate::{Code, Refused, Result, Size, Storage, Waits};
use ash::vk;
use metal2vulkan_rt as rt;
use std::cell::RefCell;
use std::convert::Infallible;
use std::ops::Range;
use std::ptr::NonNull;
use std::rc::Rc;

/// The alignment of each buffer's offset in its block: the largest
/// `minStorageBufferOffsetAlignment` Vulkan allows.
const ALIGN: u64 = 256;

/// The command buffers a queue holds at once: `newCommandQueue`'s.
const QUEUE_COMMAND_BUFFERS: u32 = 64;

/// The runtime's `error` as the harness's: a known gap or a device limit is a typed refusal.
pub(crate) fn error(error: rt::Error) -> crate::Error {
    match error {
        rt::Error::Unsupported(_) | rt::Error::Limit(_) => Box::new(Refused(error.to_string())),
        error => Box::new(error),
    }
}

/// A size as the runtime takes it; past 32 bits on an axis, a typed refusal.
pub(crate) fn size(size: Size) -> Result<[u32; 3]> {
    let [x, y, z] = size.map(u32::try_from);
    match (x, y, z) {
        (Ok(x), Ok(y), Ok(z)) => Ok([x, y, z]),
        _ => Err(Box::new(Refused(format!(
            "{size:?} is past the 32-bit sizes the runtime takes"
        )))),
    }
}

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
            let block = self.device.take_block(class, length).map_err(error)?.id;
            return Ok(Placed { block, range: None });
        }
        let span = length.next_multiple_of(ALIGN);
        let found = (self.blocks.iter_mut())
            .filter(|block| block.class == class)
            .find_map(|block| Some((block.id, block.free.take(span)?)));
        let (block, start) = match found {
            Some(found) => found,
            None => {
                let info = self.device.take_block(class, span).map_err(error)?;
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
        Ranges(std::iter::once(0..len).collect())
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
                Err(error(why))
            }
        }
    }

    /// Its live buffers, and its live `VkDeviceMemory` blocks: those the harness holds, those
    /// its pool keeps ready or idle, and its own.
    fn gauges(&self) -> Vec<(&'static str, u64)> {
        named(self.device.counters().gauges())
    }

    /// The same gauges through the device's counters, which stay readable once it is torn down.
    fn gauges_after_drop(&self) -> Option<crate::GaugeReader> {
        let counters = self.device.counters();
        Some(Box::new(move || named(counters.gauges())))
    }

    /// `Metrics::pipelines_created`.
    fn pipelines_created(&self) -> u64 {
        self.device.counters().metrics().pipelines_created
    }

    /// `Metrics`' passes (encoders), barriers and bytes copied.
    fn counts(&self) -> Vec<(&'static str, u64)> {
        let metrics = self.device.counters().metrics();
        vec![
            ("passes", metrics.passes),
            ("barriers", metrics.barriers),
            ("bytes_copied", metrics.bytes_copied),
        ]
    }

    /// `Metrics::submissions` (`vkQueueSubmit2` calls).
    fn submissions(&self) -> u64 {
        self.device.counters().metrics().submissions
    }

    /// `Metrics::first_use_waits`, with their time and the longest.
    fn first_use_waits(&self) -> Option<Waits> {
        let metrics = self.device.counters().metrics();
        Some(Waits {
            count: metrics.first_use_waits,
            nanos: metrics.first_use_wait_nanos,
            longest_nanos: metrics.first_use_wait_max_nanos,
        })
    }
}

/// A library is the translator's, as the runtime takes it; a function is its descriptor, with
/// its constant values.
impl crate::ComputeDevice for Device {
    type Library = rt::Library;
    type Function = rt::FunctionDescriptor;
    type ComputePipelineState = rt::ComputePipelineState;
    type CommandQueue = rt::CommandQueue;
    type CommandBuffer = rt::CommandBuffer;
    type ComputeCommandEncoder = rt::ComputeCommandEncoder;
    type Texture = Infallible;
    type SamplerState = Infallible;

    fn new_library(&self, code: Code<'_>) -> Result<rt::Library> {
        Ok(match code {
            Code::Metallib(bytes) => rt::Library::from_metallib(bytes)?,
            Code::Air(bytes) => rt::Library::from_air(bytes)?,
        })
    }

    fn new_compute_pipeline_state(
        &self,
        function: &rt::FunctionDescriptor,
    ) -> Result<rt::ComputePipelineState> {
        let descriptor =
            rt::ComputePipelineDescriptor::new(rt::StageSource::Function(function.clone()));
        (self.device)
            .new_compute_pipeline_state(&descriptor)
            .map_err(error)
    }

    fn new_command_queue(&self) -> Result<rt::CommandQueue> {
        (self.device)
            .new_command_queue(QUEUE_COMMAND_BUFFERS)
            .map_err(error)
    }

    /// A typed refusal: the runtime makes no texture yet.
    fn new_texture(&self, _width: u32, _height: u32) -> Result<Infallible> {
        Err(Box::new(Refused(
            "the new runtime makes no texture yet".to_owned(),
        )))
    }

    /// A typed refusal: the runtime makes no sampler state yet.
    fn new_sampler_state(&self, _filter: Filter) -> Result<Infallible> {
        Err(Box::new(Refused(
            "the new runtime makes no sampler state yet".to_owned(),
        )))
    }
}

impl crate::Library<Device> for rt::Library {
    fn new_function(&self, name: &str, constants: &[Constant]) -> Result<rt::FunctionDescriptor> {
        let mut descriptor = rt::FunctionDescriptor::new(self.function(name)?);
        if !constants.is_empty() {
            descriptor.constants = plan::values(constants)?;
            descriptor.specialized = true;
        }
        Ok(descriptor)
    }
}

impl crate::CommandQueue<Device> for rt::CommandQueue {
    fn command_buffer(&self) -> Result<rt::CommandBuffer> {
        rt::CommandQueue::command_buffer(self).map_err(error)
    }
}

/// The runtime gives times in nanoseconds of its host clock; the harness takes seconds.
impl crate::CommandBuffer<Device> for rt::CommandBuffer {
    fn compute_command_encoder(&self) -> Result<rt::ComputeCommandEncoder> {
        rt::CommandBuffer::compute_command_encoder(self, rt::DispatchType::Serial).map_err(error)
    }

    fn commit(&self) -> Result<()> {
        rt::CommandBuffer::commit(self).map_err(error)
    }

    fn wait_until_completed(&self) -> Result<()> {
        rt::CommandBuffer::wait_until_completed(self);
        match self.status() {
            rt::CommandBufferStatus::Completed => Ok(()),
            status => {
                Err(format!("the command buffer ended {status:?}: {:?}", self.error()).into())
            }
        }
    }

    fn gpu_start_time(&self) -> Result<f64> {
        Ok(rt::CommandBuffer::gpu_start_time(self) as f64 / 1e9)
    }

    fn gpu_end_time(&self) -> Result<f64> {
        Ok(rt::CommandBuffer::gpu_end_time(self) as f64 / 1e9)
    }
}

impl crate::ComputeCommandEncoder<Device> for rt::ComputeCommandEncoder {
    fn set_compute_pipeline_state(&mut self, state: &rt::ComputePipelineState) -> Result<()> {
        rt::ComputeCommandEncoder::set_compute_pipeline_state(self, state).map_err(error)
    }

    fn set_buffer(&mut self, buffer: &Buffer, offset: u64, index: u32) -> Result<()> {
        rt::ComputeCommandEncoder::set_buffer(self, Some(&buffer.buffer), offset, index)
            .map_err(error)
    }

    fn set_bytes(&mut self, bytes: &[u8], index: u32) -> Result<()> {
        rt::ComputeCommandEncoder::set_bytes(self, bytes, index).map_err(error)
    }

    fn set_texture(&mut self, texture: &Infallible, _index: u32) -> Result<()> {
        match *texture {}
    }

    fn set_sampler_state(&mut self, sampler: &Infallible, _index: u32) -> Result<()> {
        match *sampler {}
    }

    fn dispatch_threadgroups(&mut self, groups: Size, threads: Size) -> Result<()> {
        rt::ComputeCommandEncoder::dispatch_threadgroups(self, size(groups)?, size(threads)?)
            .map_err(error)
    }

    fn dispatch_threads(&mut self, threads: Size, per_group: Size) -> Result<()> {
        rt::ComputeCommandEncoder::dispatch_threads(self, size(threads)?, size(per_group)?)
            .map_err(error)
    }

    fn end_encoding(self) -> Result<()> {
        rt::ComputeCommandEncoder::end_encoding(self);
        Ok(())
    }
}

/// The gauges the harness reports, by name: all of them, so the matrix's `@drop` check covers
/// every kind of resource.
fn named(gauges: rt::Gauges) -> Vec<(&'static str, u64)> {
    vec![
        ("buffers", gauges.buffers),
        ("textures", gauges.textures),
        ("samplers", gauges.samplers),
        ("pipelines", gauges.pipelines),
        ("command_buffers", gauges.command_buffers),
        ("descriptor_slots", gauges.descriptor_slots),
        ("blocks", gauges.blocks),
        ("events", gauges.events),
    ]
}

impl crate::Buffer for Buffer {
    fn contents(&self) -> Option<NonNull<u8>> {
        self.buffer.contents()
    }
}
