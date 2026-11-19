//! The old runtime (`metal2vulkan-runtime`) behind the harness's traits, each method its call of
//! the same name. The cutover deletes this file, its `mod` line, the `--runtime old` arm of the
//! command and the crate's dependency on the old runtime.

use crate::plan::{self, Constant, TextureSource};
use crate::{Code, Result, Size, Storage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime as runtime;
use runtime::{
    BlitOption, CommandBufferStatus, HeapDescriptor, HeapType, Origin, PipelineOptions,
    ResourceOptions, TextureDescriptor,
};
use std::path::Path;
use std::ptr::NonNull;

/// The old runtime's `MTLCreateSystemDefaultDevice`: the preferred Vulkan physical device
/// (discrete, then integrated, virtual, CPU) it can open, with its default options and its
/// shader cache in `store` where one is given (else its per-user directory).
pub fn system_default(store: Option<&Path>) -> Result<runtime::Device> {
    let Some(store) = store else {
        return Ok(runtime::Device::system_default()?);
    };
    let mut options = runtime::DeviceOptions::default();
    options.shader_cache = runtime::ShaderCacheOptions::in_directory(store);
    Ok(runtime::Device::system_default_with(&options)?)
}

fn size(size: Size) -> runtime::Size {
    runtime::Size::new(size[0], size[1], size[2])
}

impl crate::Device for runtime::Device {
    type Buffer = runtime::Buffer;

    fn name(&self) -> String {
        runtime::Device::name(self).to_owned()
    }

    fn new_buffer(&self, length: u64, storage: Storage) -> Result<runtime::Buffer> {
        Ok(runtime::Device::new_buffer(self, length, options(storage))?)
    }

    /// Its pooled and its dedicated `VkDeviceMemory` blocks with a placement in them
    /// (`WorkCounts`). Its empty blocks are left out: a bounded set it keeps for the next
    /// placements, which fills as other blocks empty.
    fn gauges(&self) -> Vec<(&'static str, u64)> {
        let counts = self.work_counts();
        vec![
            ("memory_blocks", counts.memory_blocks_live),
            ("dedicated_blocks", counts.dedicated_memory_blocks_live),
        ]
    }

    /// `None`: its counts end with the device.
    fn gauges_after_drop(&self) -> Option<crate::GaugeReader> {
        None
    }

    /// None: the old runtime keeps no such count.
    fn totals(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }

    /// None: the old runtime gives no account of its threads.
    fn threads(&self) -> Vec<u64> {
        Vec::new()
    }

    /// The `VkPipeline`s its pipeline states' variants made (`WorkCounts::variant_pipelines_made`):
    /// compute regions, replays and draws, those made at a state's creation among them.
    fn pipelines_created(&self) -> u64 {
        self.work_counts().variant_pipelines_made
    }

    /// None: it counts its encoders and barriers by kind, in no one total.
    fn counts(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }

    /// `WorkCounts::queue_submissions`.
    fn submissions(&self) -> u64 {
        self.work_counts().queue_submissions
    }

    /// None: it counts no submit infos.
    fn submit_infos(&self) -> Option<u64> {
        None
    }

    /// `None`: it does not count them.
    fn first_use_waits(&self) -> Option<crate::Waits> {
        None
    }

    /// `None`: it does not count them.
    fn allocations(&self) -> Option<crate::Waits> {
        None
    }
}

impl crate::ComputeDevice for runtime::Device {
    type Library = runtime::Library;
    type Function = runtime::Function;
    type ComputePipelineState = runtime::ComputePipelineState;
    type CommandQueue = runtime::CommandQueue;
    type CommandBuffer = runtime::CommandBuffer;
    type ComputeCommandEncoder = runtime::ComputeCommandEncoder;
    type BlitCommandEncoder = runtime::BlitCommandEncoder;
    type Texture = Texture;
    type SamplerState = runtime::SamplerState;
    type RenderPipelineState = runtime::RenderPipelineState;
    type RenderCommandEncoder = runtime::RenderCommandEncoder;

    fn new_library(&self, code: Code<'_>) -> Result<runtime::Library> {
        match code {
            Code::Metallib(bytes) => Ok(self.new_library_with_data(bytes)?),
            Code::Air(bytes) => {
                let library = metal2vulkan::library::Library::from_air(bytes)?;
                Ok(self.new_library_with_compile_library(library))
            }
        }
    }

    fn new_compute_pipeline_state(
        &self,
        function: &runtime::Function,
        private: &[runtime::Function],
    ) -> Result<runtime::ComputePipelineState> {
        let options = PipelineOptions::default();
        if private.is_empty() {
            return Ok(self.new_compute_pipeline_state_with_function(function, &options)?);
        }
        let private: Vec<_> = private
            .iter()
            .map(runtime::Function::function_descriptor)
            .collect();
        let function = plan::with_private(function.function_descriptor(), &private);
        let descriptor = ComputePipelineDescriptor::new(StageSource::Function(function));
        Ok(runtime::Device::new_compute_pipeline_state(
            self,
            &descriptor,
            &options,
        )?)
    }

    fn new_render_pipeline_state(
        &self,
        vertex: &runtime::Function,
        fragment: &runtime::Function,
        color: metal2vulkan::metal::PixelFormat,
    ) -> Result<runtime::RenderPipelineState> {
        let vertex = StageSource::Function(vertex.function_descriptor());
        let mut descriptor = RenderPipelineDescriptor::new(vertex);
        descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(color));
        Ok(runtime::Device::new_render_pipeline_state(
            self,
            &descriptor,
            &PipelineOptions::default(),
        )?)
    }

    fn new_command_queue(&self) -> Result<runtime::CommandQueue> {
        Ok(runtime::Device::new_command_queue(self)?)
    }

    fn new_texture(&self, texture: &plan::TexturePlan) -> Result<Texture> {
        let options = options(texture.storage);
        let mut descriptor = TextureDescriptor::default();
        descriptor.texture_type = texture.texture_type;
        descriptor.pixel_format = texture.pixel_format;
        descriptor.width = texture.width.into();
        descriptor.height = texture.height.into();
        descriptor.array_length = texture.array_length.into();
        descriptor.usage = texture.usage;
        descriptor.resource_options = options;
        let made = |texture, heap, buffer| Texture {
            texture,
            _heap: heap,
            _buffer: buffer,
        };
        Ok(match texture.source {
            TextureSource::Plain => {
                made(runtime::Device::new_texture(self, &descriptor)?, None, None)
            }
            TextureSource::Heap => {
                let size = self.heap_texture_size_and_align(&descriptor)?.size;
                let heap =
                    self.new_heap(&HeapDescriptor::new(HeapType::Automatic, size, options))?;
                made(heap.new_texture(&descriptor)?, Some(heap), None)
            }
            TextureSource::OverBuffer => {
                let align =
                    self.minimum_linear_texture_alignment_for_pixel_format(texture.pixel_format)?;
                let row = (texture.row_bytes())
                    .ok_or("a texture's format has no known size")?
                    .next_multiple_of(align);
                let length = row * u64::from(texture.height);
                let buffer = runtime::Device::new_buffer(self, length, options)?;
                made(buffer.new_texture(&descriptor, 0, row)?, None, Some(buffer))
            }
            TextureSource::View { .. } => return Err("a view is made from its parent".into()),
        })
    }

    fn new_texture_view(
        &self,
        parent: &Texture,
        texture: &plan::TexturePlan,
        slices: std::ops::Range<u32>,
    ) -> Result<Texture> {
        let view = parent.texture.new_texture_view_with_levels_slices(
            texture.pixel_format,
            texture.texture_type,
            0..1,
            u64::from(slices.start)..u64::from(slices.end),
        )?;
        Ok(Texture {
            texture: view,
            _heap: None,
            _buffer: None,
        })
    }

    fn new_sampler_state(&self, sampler: plan::SamplerPlan) -> Result<runtime::SamplerState> {
        Ok(runtime::Device::new_sampler_state(
            self,
            &sampler.descriptor(),
        )?)
    }
}

/// The resource options of `storage`, the other modes Metal's defaults.
fn options(storage: Storage) -> ResourceOptions {
    match storage {
        Storage::Shared => ResourceOptions::shared(),
        Storage::Private => ResourceOptions::private(),
        Storage::Managed => ResourceOptions::new(
            runtime::StorageMode::Managed,
            runtime::CpuCacheMode::DefaultCache,
            runtime::HazardTrackingMode::Default,
        ),
    }
}

/// A texture of the old runtime with the heap or the buffer it was made of, which it holds as
/// long as it lives: the old runtime's texture holds only their memory, and reaches a heap's
/// bytes for a CPU write through the heap.
pub struct Texture {
    texture: runtime::Texture,
    _heap: Option<runtime::Heap>,
    _buffer: Option<runtime::Buffer>,
}

impl crate::Texture for Texture {
    fn replace_region(
        &self,
        slice: u32,
        [width, height]: [u32; 2],
        bytes: &[u8],
        bytes_per_row: u64,
    ) -> Result<()> {
        let region = runtime::Region {
            origin: runtime::Origin { x: 0, y: 0, z: 0 },
            size: runtime::Size {
                width: width.into(),
                height: height.into(),
                depth: 1,
            },
        };
        let texture = &self.texture;
        Ok(texture.replace_region(region, 0, slice.into(), bytes, bytes_per_row, 0)?)
    }
}

impl crate::Buffer for runtime::Buffer {
    fn contents(&self) -> Option<NonNull<u8>> {
        runtime::Buffer::contents(self)
    }

    fn did_modify_range(&self, offset: u64, length: u64) -> Result<()> {
        Ok(runtime::Buffer::did_modify_range(
            self,
            offset..offset + length,
        )?)
    }
}

impl crate::Library<runtime::Device> for runtime::Library {
    fn new_function(&self, name: &str, constants: &[Constant]) -> Result<runtime::Function> {
        if constants.is_empty() {
            return Ok(self.new_function_with_name(name)?);
        }
        Ok(self.new_function_with_name_constant_values(name, &plan::values(constants)?)?)
    }
}

impl crate::CommandQueue<runtime::Device> for runtime::CommandQueue {
    fn command_buffer(&self) -> Result<runtime::CommandBuffer> {
        Ok(runtime::CommandQueue::command_buffer(self)?)
    }
}

impl crate::CommandBuffer<runtime::Device> for runtime::CommandBuffer {
    fn compute_command_encoder(&self) -> Result<runtime::ComputeCommandEncoder> {
        Ok(runtime::CommandBuffer::compute_command_encoder(self)?)
    }

    fn blit_command_encoder(&self) -> Result<runtime::BlitCommandEncoder> {
        Ok(runtime::CommandBuffer::blit_command_encoder(self)?)
    }

    fn render_command_encoder(&self, target: &Texture) -> Result<runtime::RenderCommandEncoder> {
        let mut d = runtime::RenderPassDescriptor::new();
        let color = &mut d.color_attachments[0];
        color.texture = Some(target.texture.clone());
        color.load_action = runtime::LoadAction::Clear;
        color.store_action = runtime::StoreAction::Store;
        color.clear_color = runtime::ClearColor::new(0.0, 0.0, 0.0, 0.0);
        Ok(runtime::CommandBuffer::render_command_encoder(self, &d)?)
    }

    fn commit(&self) -> Result<()> {
        Ok(runtime::CommandBuffer::commit(self)?)
    }

    fn wait_until_completed(&self) -> Result<()> {
        runtime::CommandBuffer::wait_until_completed(self)?;
        match self.status() {
            CommandBufferStatus::Completed => Ok(()),
            status => {
                Err(format!("the command buffer ended {status:?}: {:?}", self.error()).into())
            }
        }
    }

    fn gpu_start_time(&self) -> Result<f64> {
        Ok(runtime::CommandBuffer::gpu_start_time(self)?)
    }

    fn gpu_end_time(&self) -> Result<f64> {
        Ok(runtime::CommandBuffer::gpu_end_time(self)?)
    }

    fn add_completed_handler(&self, handler: Box<dyn FnOnce(f64) + Send>) -> Result<()> {
        Ok(runtime::CommandBuffer::add_completed_handler(
            self,
            Box::new(move |done| {
                let end = runtime::CommandBuffer::gpu_end_time(done).unwrap_or(0.0);
                handler((runtime::host_time() - end) * 1e6)
            }),
        )?)
    }

    fn add_scheduled_handler(&self, handler: Box<dyn FnOnce() + Send>) -> Result<()> {
        Ok(runtime::CommandBuffer::add_submission_handler(
            self,
            Box::new(move |_| handler()),
        )?)
    }
}

impl crate::BlitCommandEncoder<runtime::Device> for runtime::BlitCommandEncoder {
    fn copy_from_texture(
        &mut self,
        copy: &plan::TextureCopy,
        source: &Texture,
        destination: &Texture,
    ) -> Result<()> {
        let origin = |[x, y]: [u32; 2]| Origin::new(x.into(), y.into(), 0);
        let [width, height] = copy.size;
        Ok(self.copy_from_texture_to_texture(
            &source.texture,
            copy.source_slice.into(),
            0,
            origin(copy.source_origin),
            runtime::Size::new(width.into(), height.into(), 1),
            &destination.texture,
            copy.destination_slice.into(),
            0,
            origin(copy.destination_origin),
        )?)
    }

    fn copy_from_buffer(
        &mut self,
        copy: &plan::BufferCopy,
        buffer: &runtime::Buffer,
        texture: &Texture,
    ) -> Result<()> {
        let [x, y] = copy.origin;
        let [width, height] = copy.size;
        Ok(self.copy_from_buffer_to_texture(
            buffer,
            copy.offset,
            copy.bytes_per_row,
            copy.bytes_per_image,
            runtime::Size::new(width.into(), height.into(), 1),
            &texture.texture,
            copy.slice.into(),
            0,
            Origin::new(x.into(), y.into(), 0),
            BlitOption::None,
        )?)
    }

    fn copy_from_texture_to_buffer(
        &mut self,
        copy: &plan::BufferCopy,
        texture: &Texture,
        buffer: &runtime::Buffer,
    ) -> Result<()> {
        let [x, y] = copy.origin;
        let [width, height] = copy.size;
        Ok(runtime::BlitCommandEncoder::copy_from_texture_to_buffer(
            self,
            &texture.texture,
            copy.slice.into(),
            0,
            Origin::new(x.into(), y.into(), 0),
            runtime::Size::new(width.into(), height.into(), 1),
            buffer,
            copy.offset,
            copy.bytes_per_row,
            copy.bytes_per_image,
            BlitOption::None,
        )?)
    }

    fn fill_buffer(
        &mut self,
        buffer: &runtime::Buffer,
        range: std::ops::Range<u64>,
        value: u8,
    ) -> Result<()> {
        Ok(runtime::BlitCommandEncoder::fill_buffer(
            self, buffer, range, value,
        )?)
    }

    fn synchronize_resource(&mut self, buffer: &runtime::Buffer) -> Result<()> {
        let resource = runtime::ResourceRef::Buffer(buffer);
        Ok(runtime::BlitCommandEncoder::synchronize_resource(
            self, resource,
        )?)
    }

    fn end_encoding(self) -> Result<()> {
        Ok(runtime::BlitCommandEncoder::end_encoding(self)?)
    }
}

impl crate::RenderCommandEncoder<runtime::Device> for runtime::RenderCommandEncoder {
    fn set_render_pipeline_state(&mut self, state: &runtime::RenderPipelineState) -> Result<()> {
        Ok(runtime::RenderCommandEncoder::set_render_pipeline_state(
            self, state,
        )?)
    }

    fn set_vertex_buffer(
        &mut self,
        buffer: &runtime::Buffer,
        offset: u64,
        index: u32,
    ) -> Result<()> {
        Ok(runtime::RenderCommandEncoder::set_vertex_buffer(
            self,
            Some(buffer),
            offset,
            index,
        )?)
    }

    fn set_fragment_texture(&mut self, texture: &Texture, index: u32) -> Result<()> {
        Ok(runtime::RenderCommandEncoder::set_fragment_texture(
            self,
            Some(&texture.texture),
            index,
        )?)
    }

    fn draw_triangles(&mut self, vertex_start: u32, vertex_count: u32) -> Result<()> {
        let triangle = metal2vulkan::metal::PrimitiveType::Triangle;
        Ok(runtime::RenderCommandEncoder::draw_primitives(
            self,
            triangle,
            vertex_start,
            vertex_count,
            1,
            0,
        )?)
    }

    fn end_encoding(self) -> Result<()> {
        Ok(runtime::RenderCommandEncoder::end_encoding(self)?)
    }
}

impl crate::ComputeCommandEncoder<runtime::Device> for runtime::ComputeCommandEncoder {
    fn set_compute_pipeline_state(&mut self, state: &runtime::ComputePipelineState) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::set_compute_pipeline_state(
            self, state,
        )?)
    }

    fn set_buffer(&mut self, buffer: &runtime::Buffer, offset: u64, index: u32) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::set_buffer(
            self,
            Some(buffer),
            offset,
            index,
        )?)
    }

    fn set_bytes(&mut self, bytes: &[u8], index: u32) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::set_bytes(
            self, bytes, index,
        )?)
    }

    fn set_texture(&mut self, texture: &Texture, index: u32) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::set_texture(
            self,
            Some(&texture.texture),
            index,
        )?)
    }

    fn set_sampler_state(&mut self, sampler: &runtime::SamplerState, index: u32) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::set_sampler_state(
            self,
            Some(sampler),
            index,
        )?)
    }

    fn set_threadgroup_memory_length(&mut self, length: u32, index: u32) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::set_threadgroup_memory_length(self, length, index)?)
    }

    fn dispatch_threadgroups(&mut self, groups: Size, threads: Size) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::dispatch_threadgroups(
            self,
            size(groups),
            size(threads),
        )?)
    }

    fn dispatch_threads(&mut self, threads: Size, per_group: Size) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::dispatch_threads(
            self,
            size(threads),
            size(per_group),
        )?)
    }

    fn end_encoding(self) -> Result<()> {
        Ok(runtime::ComputeCommandEncoder::end_encoding(self)?)
    }
}
