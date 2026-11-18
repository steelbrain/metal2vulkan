//! The new runtime (`metal2vulkan-rt`) behind the harness's traits.
//!
//! The new runtime places a buffer where its consumer says, in a block the consumer took
//! (`Device::take_block`, `Device::new_buffer`); the harness is that consumer here. It places each
//! buffer at the lowest free range of a block of its storage mode that holds it, at a multiple of
//! 256 bytes, taking a block of `MemoryClass::block_len` (16 MiB) when none does, as a guest's
//! allocator takes chunks of that length, and a block of its own for a buffer longer than that. A dropped buffer frees its range at once, or releases its dedicated block: the
//! harness drops a buffer only once the work that used it has completed, as Metal frees a
//! buffer's memory only then. So a buffer made after one of the same length was dropped takes
//! the same range, which the runtime zeroes as `newBufferWithLength:options:` does.
//!
//! A `Managed` buffer is the runtime's (`Device::new_managed_buffer`) over a range of a Shared
//! block: its `contents` are the CPU's bytes, which the harness writes and names with
//! `did_modify_range`, and where the device is neither integrated nor a CPU the runtime keeps a
//! device copy, which the next commit brings up to date.
//!
//! Its known gaps and the device's limits (`Error::Unsupported`, `Error::Limit`) are typed
//! refusals ([`Refused`]); its other errors are failures.
//!
//! The runtime places a texture or a sampler state at the index its consumer gives, from the
//! consumer's grants, and asks it for the index of each sampler a pipeline state numbers. The
//! harness grants itself every texture index of both classes and every sampler index once the
//! device is made (`Indices`): each texture a shader reaches takes the next index of its class
//! (a storage texture's with `ShaderWrite` usage, a sampled texture's without), each sampler
//! state the next sampler index, passing over one an earlier object still holds, and a constexpr
//! sampler the index the first state that numbered its state words took, as Metal gives an equal
//! one the same ID. A texture read's or write's numbered sampler takes nothing at its index: 0.
//!
//! The new runtime makes no heap and no texture over a buffer yet: a plan with such a texture is
//! refused ([`Refused`]).

use crate::plan::{self, Constant, TextureSource};
use crate::{Code, Refused, Result, Size, Storage, Waits};
use ash::vk;
use metal2vulkan_rt as rt;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
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
/// within a type, as the old runtime chooses, with its cache directory `store` where one is given
/// (else none).
pub fn system_default(store: Option<&Path>) -> Result<Device> {
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
        let mut options = rt::DeviceOptions::default();
        options.pipeline_cache_dir = store.map(Path::to_path_buf);
        match rt::Device::new(&physical, options) {
            Ok(device) => {
                for class in [
                    rt::IndexClass::Texture,
                    rt::IndexClass::StorageTexture,
                    rt::IndexClass::Sampler,
                ] {
                    let range = class.range();
                    device.grant_indices(class, range.start, range.end - range.start)?;
                }
                return Ok(Device {
                    indices: RefCell::new(Indices::default()),
                    heap: Rc::new(RefCell::new(Heap {
                        device: device.clone(),
                        blocks: Vec::new(),
                    })),
                    device,
                    name,
                });
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
    indices: RefCell<Indices>,
}

/// The texture and sampler indices the harness hands out (the module's docs): the last index of
/// each class taken, and each constexpr sampler's index by its state words.
#[derive(Default)]
struct Indices {
    sampled: u32,
    texture: u32,
    sampler: u32,
    constexprs: HashMap<[u64; 2], u32>,
}

impl Indices {
    /// The last index of `class` taken.
    fn last(&mut self, class: rt::IndexClass) -> &mut u32 {
        match class {
            rt::IndexClass::Sampler => &mut self.sampler,
            rt::IndexClass::StorageTexture => &mut self.texture,
            _ => &mut self.sampled,
        }
    }

    /// The next index of `class` after `last`, past the end back to its first; a sampler index a
    /// constexpr sampler took is passed over.
    fn next(&self, class: rt::IndexClass, last: u32) -> u32 {
        let range = class.range();
        let mut index = last;
        loop {
            index = match index + 1 {
                next if range.contains(&next) => next,
                _ => range.start,
            };
            if class != rt::IndexClass::Sampler || !self.constexprs.values().any(|&i| i == index) {
                return index;
            }
        }
    }
}

impl Device {
    /// A `Managed` buffer over `length` bytes at `start` of Shared block `block`: its bytes zeroed
    /// as a zeroed Shared buffer over them zeroes them, and its device copy, where it has one,
    /// zeroed with nothing copied.
    fn new_managed_buffer(
        &self,
        block: rt::BlockId,
        start: u64,
        length: u64,
    ) -> std::result::Result<rt::Buffer, rt::Error> {
        drop(self.device.new_buffer(block, start, length, Some(0))?);
        self.device.new_managed_buffer(block, start, length, 0)
    }

    /// `make` at the next index of `class` that no earlier object still holds.
    fn place<T>(
        &self,
        class: rt::IndexClass,
        make: impl Fn(u32) -> std::result::Result<T, rt::Error>,
    ) -> Result<T> {
        let range = class.range();
        for _ in range.clone() {
            let index = {
                let mut indices = self.indices.borrow_mut();
                let last = *indices.last(class);
                let index = indices.next(class, last);
                *indices.last(class) = index;
                index
            };
            match make(index) {
                Err(rt::Error::Refused(rt::Refusal::IndexInUse { .. })) => continue,
                made => return made.map_err(error),
            }
        }
        Err(Box::new(Refused(format!(
            "every {class:?} index is in use"
        ))))
    }

    /// The indices of a pipeline state's numbered samplers: a constexpr sampler takes the index
    /// its state words took first; a read's or a write's sampler takes nothing at its index.
    fn sampler_ids(
        &self,
        samplers: &[rt::NumberedSampler],
    ) -> std::result::Result<Vec<u32>, rt::Error> {
        let mut indices = self.indices.borrow_mut();
        let ids = samplers.iter().map(|sampler| match sampler {
            rt::NumberedSampler::Constexpr(words) => match indices.constexprs.get(words) {
                Some(&index) => index,
                None => {
                    let index = indices.next(rt::IndexClass::Sampler, indices.sampler);
                    indices.sampler = index;
                    indices.constexprs.insert(*words, index);
                    index
                }
            },
            _ => 0,
        });
        Ok(ids.collect())
    }
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
    block: rt::Block,
    class: rt::MemoryClass,
    free: Ranges,
}

/// Where a buffer lies.
enum Placed {
    /// A range of a pooled block.
    Pooled {
        block: rt::BlockId,
        range: Range<u64>,
    },
    /// A dedicated block of its own, whose handle the buffer holds: the runtime frees it once the
    /// handle, the buffer and the work that used it are gone.
    Dedicated(rt::Block),
}

impl Placed {
    /// The block's number and the buffer's offset in it.
    fn at(&self) -> (rt::BlockId, u64) {
        match self {
            Placed::Pooled { block, range } => (*block, range.start),
            Placed::Dedicated(block) => (block.id, 0),
        }
    }
}

impl Heap {
    /// A place for a buffer of `length` bytes in `class` memory.
    fn place(&mut self, class: rt::MemoryClass, length: u64) -> Result<Placed> {
        if length > class.block_len() {
            let block = self.device.take_block(class, length).map_err(error)?;
            return Ok(Placed::Dedicated(block));
        }
        let span = length.next_multiple_of(ALIGN);
        let found = (self.blocks.iter_mut())
            .filter(|block| block.class == class)
            .find_map(|block| Some((block.block.id, block.free.take(span)?)));
        let (block, start) = match found {
            Some(found) => found,
            None => {
                let block = (self.device.take_block(class, class.block_len())).map_err(error)?;
                let mut free = Ranges::new(block.len);
                let start =
                    (free.take(span)).ok_or("a new pooled block is shorter than a buffer")?;
                let id = block.id;
                self.blocks.push(Block { block, class, free });
                (id, start)
            }
        };
        Ok(Placed::Pooled {
            block,
            range: start..start + span,
        })
    }

    /// Frees `placed`'s range of a pooled block; a dedicated block goes with its handle, which
    /// `placed` holds.
    fn free(&mut self, placed: &Placed) {
        if let Placed::Pooled { block, range } = placed {
            let pooled = (self.blocks.iter_mut())
                .find(|kept| kept.block.id == *block)
                .expect("the harness keeps every pooled block it placed a buffer in");
            pooled.free.give(range.clone());
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

    /// Placed as the module's docs say, and zeroed by the runtime; a `Managed` buffer's bytes
    /// zeroed by the harness, and its device copy, where it has one, by the runtime.
    fn new_buffer(&self, length: u64, storage: Storage) -> Result<Buffer> {
        let class = match storage {
            Storage::Shared | Storage::Managed => rt::MemoryClass::Shared,
            Storage::Private => rt::MemoryClass::Private,
        };
        let placed = self.heap.borrow_mut().place(class, length)?;
        let (block, start) = placed.at();
        let made = match storage {
            Storage::Managed => self.new_managed_buffer(block, start, length),
            Storage::Shared | Storage::Private => {
                self.device.new_buffer(block, start, length, Some(0))
            }
        };
        match made {
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

    /// `Metrics::submit_infos`.
    fn submit_infos(&self) -> Option<u64> {
        Some(self.device.counters().metrics().submit_infos)
    }

    /// `Metrics::consumer_allocations`, with their time and the longest.
    fn allocations(&self) -> Option<Waits> {
        let metrics = self.device.counters().metrics();
        Some(Waits {
            count: metrics.consumer_allocations,
            nanos: metrics.consumer_allocation_nanos,
            longest_nanos: metrics.consumer_allocation_max_nanos,
        })
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
    type BlitCommandEncoder = rt::BlitCommandEncoder;
    type Texture = rt::Texture;
    type SamplerState = rt::SamplerState;
    type RenderPipelineState = rt::RenderPipelineState;
    type RenderCommandEncoder = rt::RenderCommandEncoder;

    fn new_library(&self, code: Code<'_>) -> Result<rt::Library> {
        Ok(match code {
            Code::Metallib(bytes) => rt::Library::from_metallib(bytes)?,
            Code::Air(bytes) => rt::Library::from_air(bytes)?,
        })
    }

    fn new_compute_pipeline_state(
        &self,
        function: &rt::FunctionDescriptor,
        private: &[rt::FunctionDescriptor],
    ) -> Result<rt::ComputePipelineState> {
        let function = plan::with_private(function.clone(), private);
        let descriptor = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(function));
        (self.device)
            .new_compute_pipeline_state(&descriptor, |samplers| self.sampler_ids(samplers))
            .map_err(error)
    }

    fn new_render_pipeline_state(
        &self,
        vertex: &rt::FunctionDescriptor,
        fragment: &rt::FunctionDescriptor,
        color: rt::PixelFormat,
    ) -> Result<rt::RenderPipelineState> {
        let mut descriptor =
            rt::RenderPipelineDescriptor::new(rt::StageSource::Function(vertex.clone()));
        descriptor.fragment = Some(rt::StageSource::Function(fragment.clone()));
        descriptor.color_attachments[0] = Some(rt::ColorAttachmentDescriptor::new(color));
        (self.device)
            .new_render_pipeline_state(&descriptor, |samplers| self.sampler_ids(samplers))
            .map_err(error)
    }

    fn new_command_queue(&self) -> Result<rt::CommandQueue> {
        (self.device)
            .new_command_queue(QUEUE_COMMAND_BUFFERS)
            .map_err(error)
    }

    fn new_texture(&self, texture: &plan::TexturePlan) -> Result<rt::Texture> {
        let refused = match texture.source {
            TextureSource::Plain => None,
            TextureSource::Heap => {
                Some("newTextureWithDescriptor: of a heap: the new runtime makes no heap yet")
            }
            TextureSource::OverBuffer => Some(
                "newTextureWithDescriptor:offset:bytesPerRow: of a buffer: the new runtime makes \
                 no texture over a buffer yet",
            ),
            TextureSource::View { .. } => return Err("a view is made from its parent".into()),
        };
        if let Some(refused) = refused {
            return Err(Box::new(Refused(refused.to_owned())));
        }
        let mut descriptor = rt::TextureDescriptor::new(
            texture.texture_type,
            texture.pixel_format,
            texture.width,
            texture.height,
        );
        descriptor.array_length = texture.array_length;
        descriptor.usage = texture.usage;
        descriptor.storage_mode = match texture.storage {
            Storage::Shared => rt::StorageMode::Shared,
            Storage::Managed => rt::StorageMode::Managed,
            Storage::Private => rt::StorageMode::Private,
        };
        match class(texture.usage) {
            Some(class) => self.place(class, |index| {
                self.device.new_texture(&descriptor, Some(index))
            }),
            None => self.device.new_texture(&descriptor, None).map_err(error),
        }
    }

    /// At an index as [`new_texture`](crate::ComputeDevice::new_texture) places a texture.
    fn new_texture_view(
        &self,
        parent: &rt::Texture,
        texture: &plan::TexturePlan,
        slices: Range<u32>,
    ) -> Result<rt::Texture> {
        let (pixel_format, texture_type) = (texture.pixel_format, texture.texture_type);
        let view = |index| {
            parent.new_texture_view(pixel_format, texture_type, 0..1, slices.clone(), index)
        };
        match class(texture.usage) {
            Some(class) => self.place(class, |index| view(Some(index))),
            None => view(None).map_err(error),
        }
    }

    fn new_sampler_state(&self, sampler: plan::SamplerPlan) -> Result<rt::SamplerState> {
        let descriptor = sampler.descriptor();
        self.place(rt::IndexClass::Sampler, |index| {
            self.device.new_sampler_state(&descriptor, index)
        })
    }
}

/// The index class of a texture of `usage`, where shaders reach it: unknown usage is every
/// usage, as the runtime takes it; a texture no shader reaches takes no index.
fn class(usage: rt::TextureUsage) -> Option<rt::IndexClass> {
    let usage = match usage.is_empty() {
        true => rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE,
        false => usage,
    };
    match usage.contains(rt::TextureUsage::SHADER_WRITE) {
        true => Some(rt::IndexClass::StorageTexture),
        false => (usage.contains(rt::TextureUsage::SHADER_READ)).then_some(rt::IndexClass::Texture),
    }
}

impl crate::Texture for rt::Texture {
    fn replace_region(
        &self,
        slice: u32,
        [width, height]: [u32; 2],
        bytes: &[u8],
        bytes_per_row: u64,
    ) -> Result<()> {
        let region = rt::Region {
            origin: [0, 0, 0],
            size: [width, height, 1],
        };
        rt::Texture::replace_region(self, region, 0, slice, bytes, bytes_per_row, 0).map_err(error)
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

    fn blit_command_encoder(&self) -> Result<rt::BlitCommandEncoder> {
        rt::CommandBuffer::blit_command_encoder(self).map_err(error)
    }

    fn render_command_encoder(&self, target: &rt::Texture) -> Result<rt::RenderCommandEncoder> {
        let mut d = rt::RenderPassDescriptor::default();
        let color = &mut d.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = rt::LoadAction::Clear;
        color.store_action = rt::StoreAction::Store;
        color.clear_color = rt::ClearColor::new(0.0, 0.0, 0.0, 0.0);
        rt::CommandBuffer::render_command_encoder(self, &d).map_err(error)
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

    fn add_completed_handler(&self, handler: Box<dyn FnOnce(f64) + Send>) -> Result<()> {
        rt::CommandBuffer::add_completed_handler(self, move |done| {
            let late = rt::host_time().saturating_sub(done.gpu_end_time());
            handler(late as f64 / 1e3)
        })
        .map_err(error)
    }
}

/// A copy between a buffer and a texture as the runtime takes it: the buffer's layout, the
/// size and the texture's place.
fn buffer_copy(copy: &plan::BufferCopy) -> (rt::BufferLayout, [u32; 3], rt::TexturePlace) {
    let [width, height] = copy.size;
    let [x, y] = copy.origin;
    (
        rt::BufferLayout {
            offset: copy.offset,
            bytes_per_row: copy.bytes_per_row,
            bytes_per_image: copy.bytes_per_image,
        },
        [width, height, 1],
        rt::TexturePlace {
            slice: copy.slice,
            level: 0,
            origin: [x, y, 0],
        },
    )
}

impl crate::BlitCommandEncoder<Device> for rt::BlitCommandEncoder {
    fn copy_from_texture(
        &mut self,
        copy: &plan::TextureCopy,
        source: &rt::Texture,
        destination: &rt::Texture,
    ) -> Result<()> {
        let place = |slice, [x, y]: [u32; 2]| rt::TexturePlace {
            slice,
            level: 0,
            origin: [x, y, 0],
        };
        let [width, height] = copy.size;
        rt::BlitCommandEncoder::copy_from_texture(
            self,
            source,
            place(copy.source_slice, copy.source_origin),
            [width, height, 1],
            destination,
            place(copy.destination_slice, copy.destination_origin),
        )
        .map_err(error)
    }

    fn copy_from_buffer(
        &mut self,
        copy: &plan::BufferCopy,
        buffer: &Buffer,
        texture: &rt::Texture,
    ) -> Result<()> {
        let (layout, size, place) = buffer_copy(copy);
        rt::BlitCommandEncoder::copy_from_buffer(self, &buffer.buffer, layout, size, texture, place)
            .map_err(error)
    }

    fn copy_from_texture_to_buffer(
        &mut self,
        copy: &plan::BufferCopy,
        texture: &rt::Texture,
        buffer: &Buffer,
    ) -> Result<()> {
        let (layout, size, place) = buffer_copy(copy);
        rt::BlitCommandEncoder::copy_from_texture_to_buffer(
            self,
            texture,
            place,
            size,
            &buffer.buffer,
            layout,
        )
        .map_err(error)
    }

    fn fill_buffer(&mut self, buffer: &Buffer, range: Range<u64>, value: u8) -> Result<()> {
        rt::BlitCommandEncoder::fill_buffer(self, &buffer.buffer, range, value).map_err(error)
    }

    /// `synchronize_resource`: of a `Managed` buffer with a device copy, the copy into its CPU
    /// bytes; of any other buffer, nothing.
    fn synchronize_resource(&mut self, buffer: &Buffer) -> Result<()> {
        let resource = rt::Resource::Buffer(&buffer.buffer);
        rt::BlitCommandEncoder::synchronize_resource(self, resource).map_err(error)
    }

    fn end_encoding(self) -> Result<()> {
        rt::BlitCommandEncoder::end_encoding(self);
        Ok(())
    }
}

impl crate::RenderCommandEncoder<Device> for rt::RenderCommandEncoder {
    fn set_render_pipeline_state(&mut self, state: &rt::RenderPipelineState) -> Result<()> {
        rt::RenderCommandEncoder::set_render_pipeline_state(self, state).map_err(error)
    }

    fn set_vertex_buffer(&mut self, buffer: &Buffer, offset: u64, index: u32) -> Result<()> {
        rt::RenderCommandEncoder::set_vertex_buffer(self, Some(&buffer.buffer), offset, index)
            .map_err(error)
    }

    fn draw_triangles(&mut self, vertex_start: u32, vertex_count: u32) -> Result<()> {
        let triangle = rt::PrimitiveType::Triangle;
        rt::RenderCommandEncoder::draw_primitives(self, triangle, vertex_start, vertex_count, 1, 0)
            .map_err(error)
    }

    fn end_encoding(self) -> Result<()> {
        rt::RenderCommandEncoder::end_encoding(self);
        Ok(())
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

    fn set_texture(&mut self, texture: &rt::Texture, index: u32) -> Result<()> {
        rt::ComputeCommandEncoder::set_texture(self, Some(texture), index).map_err(error)
    }

    fn set_sampler_state(&mut self, sampler: &rt::SamplerState, index: u32) -> Result<()> {
        rt::ComputeCommandEncoder::set_sampler_state(self, Some(sampler), index).map_err(error)
    }

    fn set_threadgroup_memory_length(&mut self, length: u32, index: u32) -> Result<()> {
        rt::ComputeCommandEncoder::set_threadgroup_memory_length(self, length, index).map_err(error)
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
        ("kernels", gauges.kernels),
        ("kernel_entries", gauges.kernel_entries),
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

    /// `Buffer::did_modify_range`: the next commit takes the range's bytes into the device copy,
    /// where the buffer has one.
    fn did_modify_range(&self, offset: u64, length: u64) -> Result<()> {
        self.buffer.did_modify_range(offset, length).map_err(error)
    }
}
