//! The runtime's side: Metal's calls on `metal2vulkan_runtime`, on the Vulkan device its loader
//! finds (MoltenVK on a Mac). The device is opened as the consumer opens it (reims-vgpu's
//! `device_options`): GPU times from the host's clock, the exact completion and uncompressed heap
//! textures, the rest of the policy at its defaults, the shader cache in a directory of the
//! bench's own.

use crate::gpu::{Format, Gpu};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::{CompletionPolicy, HeapCompressionPolicy};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, CullMode, DispatchType, IndexType, IndirectCommandType,
    PixelFormat, PrimitiveType, ResourceUsage, SamplerMinMagFilter, TextureUsage, Winding,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime as rt;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

/// The degradation notices the runtime's device announced.
pub static NOTICES: AtomicU64 = AtomicU64::new(0);

/// The other messages the runtime's device's sink heard.
pub static MESSAGES: AtomicU64 = AtomicU64::new(0);

pub struct Runtime {
    device: rt::Device,
    queue: rt::CommandQueue,
}

impl Runtime {
    /// Open the device with its shader cache in `cache`, announcing the kinds of `announce_each`
    /// at every occurrence, its notices counted ([`NOTICES`]) as a consumer's sink hears them,
    /// and its other messages ([`MESSAGES`]), the sink hearing those of `least` severity and above.
    pub fn open(
        cache: &Path,
        announce_each: &[rt::Degradation],
        least: rt::MessageSeverity,
    ) -> Runtime {
        let mut options = rt::DeviceOptions::default();
        options.announce_each = announce_each.iter().copied().collect();
        let sink = rt::MessageSink::new(|message| {
            let count = match message.degradation {
                Some(_) => &NOTICES,
                None => &MESSAGES,
            };
            count.fetch_add(1, Ordering::Relaxed);
        });
        options.messages = Some(sink.with_least_severity(least));
        options.gpu_times_from_host = true;
        options.policy.completion = CompletionPolicy::Exact;
        options.policy.heap_compression = HeapCompressionPolicy::Uncompressed;
        options.shader_cache = rt::ShaderCacheOptions::in_directory(cache);
        let device = rt::Device::system_default_with(&options).expect("open the runtime's device");
        let queue = device.new_command_queue().expect("new_command_queue");
        Runtime { device, queue }
    }

    pub fn device(&self) -> &rt::Device {
        &self.device
    }
}

/// Command `index` of `commands`' setters (`indirectComputeCommandAtIndex:`).
fn compute_command(
    commands: &rt::IndirectCommandBuffer,
    index: usize,
) -> rt::IndirectComputeCommand {
    commands
        .indirect_compute_command_at(index as u64)
        .expect("indirect_compute_command_at")
}

/// Command `index` of `commands`' setters (`indirectRenderCommandAtIndex:`).
fn render_command(commands: &rt::IndirectCommandBuffer, index: usize) -> rt::IndirectRenderCommand {
    commands
        .indirect_render_command_at(index as u64)
        .expect("indirect_render_command_at")
}

fn pixel_format(format: Format) -> PixelFormat {
    match format {
        Format::Bgra8 => PixelFormat::BGRA8Unorm,
        Format::Rgba8 => PixelFormat::RGBA8Unorm,
        Format::Rgba16Float => PixelFormat::RGBA16Float,
        Format::Depth32Float => PixelFormat::Depth32Float,
    }
}

impl Gpu for Runtime {
    type Library = rt::Library;
    type RenderPipeline = rt::RenderPipelineState;
    type ComputePipeline = rt::ComputePipelineState;
    type Buffer = rt::Buffer;
    type Texture = rt::Texture;
    type Sampler = rt::SamplerState;
    type CommandBuffer = rt::CommandBuffer;
    type RenderEncoder = rt::RenderCommandEncoder;
    type ComputeEncoder = rt::ComputeCommandEncoder;
    type Heap = rt::Heap;
    type DepthStencil = rt::DepthStencilState;
    type IndirectCommands = rt::IndirectCommandBuffer;
    type SharedEvent = rt::SharedEvent;
    type Listener = rt::SharedEventListener;
    type Pass = rt::RenderPassDescriptor;
    type BlitEncoder = rt::BlitCommandEncoder;
    type Fence = rt::Fence;

    fn label(&self) -> String {
        format!("runtime on {}", self.device.name())
    }

    fn library(&self, metallib: &Path) -> rt::Library {
        let bytes = std::fs::read(metallib).expect("read the metallib");
        self.device
            .new_library_with_data(&bytes)
            .expect("new_library_with_data")
    }

    fn render_pipeline(
        &self,
        library: &rt::Library,
        vertex: &str,
        fragment: &str,
        format: Format,
        blend: bool,
    ) -> rt::RenderPipelineState {
        let function = |name: &str| {
            let function = library
                .new_function_with_name(name)
                .expect("new_function_with_name");
            StageSource::Function(function.function_descriptor())
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
        descriptor.fragment = Some(function(fragment));
        let mut color = ColorAttachmentDescriptor::new(pixel_format(format));
        color.blending_enabled = blend;
        descriptor.color_attachments[0] = Some(color);
        self.device
            .new_render_pipeline_state(&descriptor, &rt::PipelineOptions::default())
            .expect("new_render_pipeline_state")
    }

    fn depth_pipeline(
        &self,
        library: &rt::Library,
        vertex: &str,
        fragment: &str,
        blend: bool,
    ) -> rt::RenderPipelineState {
        let function = |name: &str| {
            let function = library
                .new_function_with_name(name)
                .expect("new_function_with_name");
            StageSource::Function(function.function_descriptor())
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
        descriptor.fragment = Some(function(fragment));
        let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm);
        if blend {
            color.blending_enabled = true;
            color.source_rgb_blend_factor = BlendFactor::SourceAlpha;
            color.source_alpha_blend_factor = BlendFactor::One;
            color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
            color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
        }
        descriptor.color_attachments[0] = Some(color);
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
        self.device
            .new_render_pipeline_state(&descriptor, &rt::PipelineOptions::default())
            .expect("new_render_pipeline_state")
    }

    fn depth_stencil(&self, writes: bool) -> rt::DepthStencilState {
        let mut descriptor = DepthStencilDescriptor::default();
        descriptor.depth_compare_function = if writes {
            CompareFunction::Less
        } else {
            CompareFunction::LessEqual
        };
        descriptor.depth_write_enabled = writes;
        self.device
            .new_depth_stencil_state(&descriptor)
            .expect("new_depth_stencil_state")
    }

    fn compute_pipeline(&self, library: &rt::Library, kernel: &str) -> rt::ComputePipelineState {
        let function = library
            .new_function_with_name(kernel)
            .expect("new_function_with_name");
        self.device
            .new_compute_pipeline_state_with_function(&function, &rt::PipelineOptions::default())
            .expect("new_compute_pipeline_state_with_function")
    }

    fn indirect_compute_pipeline(
        &self,
        library: &rt::Library,
        kernel: &str,
    ) -> rt::ComputePipelineState {
        let function = library
            .new_function_with_name(kernel)
            .expect("new_function_with_name");
        let mut descriptor =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        descriptor.support_indirect_command_buffers = true;
        self.device
            .new_compute_pipeline_state(&descriptor, &rt::PipelineOptions::default())
            .expect("new_compute_pipeline_state")
    }

    fn indirect_render_pipeline(
        &self,
        library: &rt::Library,
        vertex: &str,
        fragment: &str,
    ) -> rt::RenderPipelineState {
        let function = |name: &str| {
            let function = library
                .new_function_with_name(name)
                .expect("new_function_with_name");
            StageSource::Function(function.function_descriptor())
        };
        let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
        descriptor.fragment = Some(function(fragment));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        descriptor.support_indirect_command_buffers = true;
        self.device
            .new_render_pipeline_state(&descriptor, &rt::PipelineOptions::default())
            .expect("new_render_pipeline_state")
    }

    fn indirect_draws(&self, count: usize, vertex_buffers: usize) -> rt::IndirectCommandBuffer {
        let mut descriptor = rt::IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.inherit_pipeline_state = true;
        descriptor.inherit_buffers = false;
        descriptor.max_vertex_buffer_bind_count = vertex_buffers as u32;
        descriptor.max_fragment_buffer_bind_count = 0;
        self.device
            .new_indirect_command_buffer(&descriptor, count as u64, rt::ResourceOptions::shared())
            .expect("new_indirect_command_buffer")
    }

    fn indirect_commands(&self, count: usize) -> rt::IndirectCommandBuffer {
        let mut descriptor = rt::IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        descriptor.inherit_pipeline_state = false;
        descriptor.inherit_buffers = false;
        descriptor.max_kernel_buffer_bind_count = 1;
        self.device
            .new_indirect_command_buffer(&descriptor, count as u64, rt::ResourceOptions::private())
            .expect("new_indirect_command_buffer")
    }

    fn commands_id(&self, commands: &rt::IndirectCommandBuffer) -> u64 {
        commands.gpu_resource_id().0
    }

    fn pipeline_id(&self, state: &rt::ComputePipelineState) -> u64 {
        state.gpu_resource_id().0
    }

    fn cpu_indirect_commands(&self, count: usize) -> rt::IndirectCommandBuffer {
        let mut descriptor = rt::IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        descriptor.inherit_pipeline_state = false;
        descriptor.inherit_buffers = false;
        descriptor.max_kernel_buffer_bind_count = 1;
        self.device
            .new_indirect_command_buffer(&descriptor, count as u64, rt::ResourceOptions::shared())
            .expect("new_indirect_command_buffer")
    }

    fn cpu_indirect_draws(&self, count: usize) -> rt::IndirectCommandBuffer {
        let mut descriptor = rt::IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.inherit_pipeline_state = false;
        descriptor.inherit_buffers = false;
        descriptor.max_vertex_buffer_bind_count = 1;
        descriptor.max_fragment_buffer_bind_count = 1;
        self.device
            .new_indirect_command_buffer(&descriptor, count as u64, rt::ResourceOptions::shared())
            .expect("new_indirect_command_buffer")
    }

    fn command_pipeline(
        &self,
        commands: &rt::IndirectCommandBuffer,
        index: usize,
        state: &rt::ComputePipelineState,
    ) {
        compute_command(commands, index)
            .set_compute_pipeline_state(state)
            .expect("set_compute_pipeline_state");
    }

    fn command_kernel_buffer(
        &self,
        commands: &rt::IndirectCommandBuffer,
        index: usize,
        buffer: &rt::Buffer,
        offset: usize,
    ) {
        compute_command(commands, index)
            .set_kernel_buffer(Some(buffer), offset as u64, 0)
            .expect("set_kernel_buffer");
    }

    fn command_dispatch(&self, commands: &rt::IndirectCommandBuffer, index: usize, groups: usize) {
        compute_command(commands, index)
            .concurrent_dispatch_threadgroups(
                rt::Size::new(groups as u64, 1, 1),
                rt::Size::new(64, 1, 1),
            )
            .expect("concurrent_dispatch_threadgroups");
    }

    fn command_barrier(&self, commands: &rt::IndirectCommandBuffer, index: usize) {
        compute_command(commands, index)
            .set_barrier()
            .expect("set_barrier");
    }

    fn command_kernel_buffer_address(
        &self,
        commands: &rt::IndirectCommandBuffer,
        index: usize,
    ) -> u64 {
        compute_command(commands, index)
            .kernel_buffer(0)
            .expect("kernel_buffer")
            .expect("a kernel buffer")
            .address
            .0
    }

    fn command_has_barrier(&self, commands: &rt::IndirectCommandBuffer, index: usize) -> bool {
        compute_command(commands, index)
            .has_barrier()
            .expect("has_barrier")
    }

    fn draw_command_pipeline(
        &self,
        commands: &rt::IndirectCommandBuffer,
        index: usize,
        state: &rt::RenderPipelineState,
    ) {
        render_command(commands, index)
            .set_render_pipeline_state(state)
            .expect("set_render_pipeline_state");
    }

    fn draw_command_vertex_buffer(
        &self,
        commands: &rt::IndirectCommandBuffer,
        index: usize,
        buffer: &rt::Buffer,
        offset: usize,
    ) {
        render_command(commands, index)
            .set_vertex_buffer(Some(buffer), offset as u64, 0)
            .expect("set_vertex_buffer");
    }

    fn draw_command_draw(&self, commands: &rt::IndirectCommandBuffer, index: usize) {
        render_command(commands, index)
            .draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)
            .expect("draw_primitives");
    }

    fn draw_command_vertex_buffer_address(
        &self,
        commands: &rt::IndirectCommandBuffer,
        index: usize,
    ) -> u64 {
        render_command(commands, index)
            .vertex_buffer(0)
            .expect("vertex_buffer")
            .expect("a vertex buffer")
            .address
            .0
    }

    fn buffer(&self, length: usize) -> rt::Buffer {
        self.device
            .new_buffer(length as u64, rt::ResourceOptions::shared())
            .expect("new_buffer")
    }

    fn buffer_with(&self, bytes: &[u8]) -> rt::Buffer {
        self.device
            .new_buffer_with_bytes(bytes, rt::ResourceOptions::shared())
            .expect("new_buffer_with_bytes")
    }

    unsafe fn buffer_no_copy(&self, pointer: *mut u8, length: usize) -> rt::Buffer {
        let pointer = NonNull::new(pointer).expect("a pointer");
        // SAFETY: the caller's contract: page-aligned read-write memory that outlives the buffer.
        unsafe {
            self.device.new_buffer_with_bytes_no_copy(
                rt::HostMemory::new(pointer),
                length as u64,
                rt::ResourceOptions::shared(),
                None,
            )
        }
        .expect("new_buffer_with_bytes_no_copy")
    }

    fn heap(&self, size: usize) -> rt::Heap {
        let descriptor = rt::HeapDescriptor::new(
            rt::HeapType::Placement,
            size as u64,
            rt::ResourceOptions::shared(),
        );
        self.device.new_heap(&descriptor).expect("new_heap")
    }

    unsafe fn heap_no_copy(&self, pointer: *mut u8, size: usize) -> Option<rt::Heap> {
        let pointer = NonNull::new(pointer).expect("a pointer");
        let descriptor = rt::HeapDescriptor::new(
            rt::HeapType::Placement,
            size as u64,
            rt::ResourceOptions::shared(),
        );
        // SAFETY: the caller's contract: page-aligned read-write memory that outlives the heap.
        let heap = unsafe {
            self.device.new_heap_with_bytes_no_copy(
                rt::HostMemory::new(pointer),
                &descriptor,
                None,
                None,
            )
        };
        Some(heap.expect("new_heap_with_bytes_no_copy"))
    }

    fn heap_texture(
        &self,
        heap: &rt::Heap,
        width: usize,
        height: usize,
        offset: usize,
    ) -> rt::Texture {
        let mut descriptor = rt::TextureDescriptor::texture_2d(
            PixelFormat::BGRA8Unorm,
            width as u64,
            height as u64,
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = rt::ResourceOptions::shared();
        descriptor.allow_gpu_optimized_contents = false;
        heap.new_texture_at_offset(&descriptor, offset as u64)
            .expect("new_texture_at_offset")
    }

    fn buffer_texture(
        &self,
        buffer: &rt::Buffer,
        (width, height): (usize, usize),
        offset: usize,
        bytes_per_row: usize,
    ) -> rt::Texture {
        let mut descriptor = rt::TextureDescriptor::texture_2d(
            PixelFormat::BGRA8Unorm,
            width as u64,
            height as u64,
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = rt::ResourceOptions::shared();
        buffer
            .new_texture(&descriptor, offset as u64, bytes_per_row as u64)
            .expect("new_texture over a buffer")
    }

    fn texture(&self, format: Format, width: usize, height: usize, target: bool) -> rt::Texture {
        let mut descriptor = rt::TextureDescriptor::texture_2d(
            pixel_format(format),
            width as u64,
            height as u64,
            false,
        );
        if target {
            descriptor.usage = TextureUsage::RENDER_TARGET;
            descriptor.resource_options = rt::ResourceOptions::private();
        } else {
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = rt::ResourceOptions::shared();
        }
        self.device.new_texture(&descriptor).expect("new_texture")
    }

    fn tracked_heap(&self, size: usize) -> rt::Heap {
        let options = rt::ResourceOptions::new(
            rt::StorageMode::Shared,
            rt::CpuCacheMode::DefaultCache,
            rt::HazardTrackingMode::Tracked,
        );
        let descriptor = rt::HeapDescriptor::new(rt::HeapType::Placement, size as u64, options);
        self.device.new_heap(&descriptor).expect("new_heap")
    }

    fn heap_render_texture(
        &self,
        heap: &rt::Heap,
        format: Format,
        side: usize,
        offset: usize,
    ) -> rt::Texture {
        let mut descriptor = rt::TextureDescriptor::texture_2d(
            pixel_format(format),
            side as u64,
            side as u64,
            false,
        );
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = rt::ResourceOptions::shared();
        heap.new_texture_at_offset(&descriptor, offset as u64)
            .expect("new_texture_at_offset")
    }

    fn replace_region(
        &self,
        texture: &rt::Texture,
        size: (usize, usize),
        bytes: &[u8],
        row: usize,
    ) {
        let region = rt::Region::new_2d(0, 0, size.0 as u64, size.1 as u64);
        texture
            .replace_region(region, 0, 0, bytes, row as u64, 0)
            .expect("replace_region");
    }

    fn get_bytes(&self, texture: &rt::Texture, size: (usize, usize), bytes: &mut [u8], row: usize) {
        let region = rt::Region::new_2d(0, 0, size.0 as u64, size.1 as u64);
        texture
            .get_bytes(bytes, row as u64, 0, region, 0, 0)
            .expect("get_bytes");
    }

    fn read_buffer(&self, buffer: &rt::Buffer, bytes: &mut [u8]) {
        assert!(
            bytes.len() as u64 <= buffer.length(),
            "a read within the buffer"
        );
        let contents = buffer.contents().expect("a Shared buffer's contents");
        // SAFETY: `contents` holds the buffer's `length` bytes, which no GPU work writes now (the
        // caller's contract), and `bytes` is as long as the copy.
        unsafe {
            std::ptr::copy_nonoverlapping(contents.as_ptr(), bytes.as_mut_ptr(), bytes.len())
        };
    }

    fn write_buffer(&self, buffer: &rt::Buffer, value: u8) {
        let contents = buffer.contents().expect("a Shared buffer's contents");
        // SAFETY: `contents` holds the buffer's `length` bytes, which no GPU work uses now (the
        // caller's contract).
        unsafe { std::ptr::write_bytes(contents.as_ptr(), value, buffer.length() as usize) };
    }

    fn sampler(&self, linear: bool) -> rt::SamplerState {
        let mut descriptor = SamplerDescriptor::new();
        if linear {
            descriptor.min_filter = SamplerMinMagFilter::Linear;
            descriptor.mag_filter = SamplerMinMagFilter::Linear;
        }
        self.device
            .new_sampler_state(&descriptor)
            .expect("new_sampler_state")
    }

    fn command_buffer(&self) -> rt::CommandBuffer {
        self.queue.command_buffer().expect("command_buffer")
    }

    fn render_encoder(
        &self,
        command_buffer: &rt::CommandBuffer,
        target: &rt::Texture,
        clear: bool,
    ) -> rt::RenderCommandEncoder {
        let mut pass = rt::RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = if clear {
            rt::LoadAction::Clear
        } else {
            rt::LoadAction::Load
        };
        color.store_action = rt::StoreAction::Store;
        command_buffer
            .render_command_encoder(&pass)
            .expect("render_command_encoder")
    }

    fn pass(&self, target: &rt::Texture) -> rt::RenderPassDescriptor {
        let mut pass = rt::RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = rt::LoadAction::Clear;
        color.store_action = rt::StoreAction::Store;
        pass
    }

    fn pass_encoder(
        &self,
        command_buffer: &rt::CommandBuffer,
        pass: &rt::RenderPassDescriptor,
    ) -> rt::RenderCommandEncoder {
        command_buffer
            .render_command_encoder(pass)
            .expect("render_command_encoder")
    }

    fn depth_render_encoder(
        &self,
        command_buffer: &rt::CommandBuffer,
        target: &rt::Texture,
        depth: &rt::Texture,
    ) -> rt::RenderCommandEncoder {
        let mut pass = rt::RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = rt::LoadAction::Clear;
        color.store_action = rt::StoreAction::Store;
        color.clear_color = rt::ClearColor {
            red: 0.1,
            green: 0.1,
            blue: 0.12,
            alpha: 1.0,
        };
        let attachment = &mut pass.depth_attachment;
        attachment.texture = Some(depth.clone());
        attachment.load_action = rt::LoadAction::Clear;
        attachment.store_action = rt::StoreAction::DontCare;
        attachment.clear_depth = 1.0;
        command_buffer
            .render_command_encoder(&pass)
            .expect("render_command_encoder")
    }

    fn set_render_pipeline(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        state: &rt::RenderPipelineState,
    ) {
        encoder
            .set_render_pipeline_state(state)
            .expect("set_render_pipeline_state");
    }

    fn set_depth_stencil(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        state: &rt::DepthStencilState,
    ) {
        encoder
            .set_depth_stencil_state(state)
            .expect("set_depth_stencil_state");
    }

    fn set_cull_back(&self, encoder: &mut rt::RenderCommandEncoder, back: bool) {
        encoder.set_cull_mode(if back { CullMode::Back } else { CullMode::None });
    }

    fn set_counter_clockwise(&self, encoder: &mut rt::RenderCommandEncoder) {
        encoder.set_front_facing_winding(Winding::CounterClockwise);
    }

    fn set_vertex_bytes(&self, encoder: &mut rt::RenderCommandEncoder, bytes: &[u8], index: usize) {
        encoder
            .set_vertex_bytes(bytes, index as u32)
            .expect("set_vertex_bytes");
    }

    fn set_vertex_buffer(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        buffer: &rt::Buffer,
        offset: usize,
        index: usize,
    ) {
        encoder
            .set_vertex_buffer(Some(buffer), offset as u64, index as u32)
            .expect("set_vertex_buffer");
    }

    fn set_vertex_buffers(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        buffers: &[&rt::Buffer],
        offsets: &[usize],
        first: usize,
    ) {
        let buffers: [Option<&rt::Buffer>; 2] = [Some(buffers[0]), Some(buffers[1])];
        let offsets = [offsets[0] as u64, offsets[1] as u64];
        encoder
            .set_vertex_buffers(&buffers, &offsets, first as u32)
            .expect("set_vertex_buffers");
    }

    fn set_fragment_buffer(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        buffer: &rt::Buffer,
        offset: usize,
        index: usize,
    ) {
        encoder
            .set_fragment_buffer(Some(buffer), offset as u64, index as u32)
            .expect("set_fragment_buffer");
    }

    fn set_fragment_texture(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        texture: &rt::Texture,
        index: usize,
    ) {
        encoder
            .set_fragment_texture(Some(texture), index as u32)
            .expect("set_fragment_texture");
    }

    fn set_fragment_sampler(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        sampler: &rt::SamplerState,
        index: usize,
    ) {
        encoder
            .set_fragment_sampler_state(Some(sampler), index as u32)
            .expect("set_fragment_sampler_state");
    }

    fn draw_strip(&self, encoder: &mut rt::RenderCommandEncoder, vertices: usize) {
        encoder
            .draw_primitives(PrimitiveType::TriangleStrip, 0, vertices as u32, 1, 0)
            .expect("draw_primitives");
    }

    fn draw_indexed(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        count: usize,
        indices: &rt::Buffer,
    ) {
        encoder
            .draw_indexed_primitives(
                PrimitiveType::Triangle,
                count as u32,
                IndexType::UInt16,
                indices,
                0,
                1,
                0,
                0,
            )
            .expect("draw_indexed_primitives");
    }

    fn draw_triangles(&self, encoder: &mut rt::RenderCommandEncoder, start: usize, count: usize) {
        encoder
            .draw_primitives(PrimitiveType::Triangle, start as u32, count as u32, 1, 0)
            .expect("draw_primitives");
    }

    fn draw_indexed_at(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        count: usize,
        indices: &rt::Buffer,
        offset: usize,
    ) {
        encoder
            .draw_indexed_primitives(
                PrimitiveType::Triangle,
                count as u32,
                IndexType::UInt16,
                indices,
                offset as u64,
                1,
                0,
                0,
            )
            .expect("draw_indexed_primitives");
    }

    fn draw_indexed_strip_at(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        count: usize,
        indices: &rt::Buffer,
        offset: usize,
    ) {
        encoder
            .draw_indexed_primitives(
                PrimitiveType::TriangleStrip,
                count as u32,
                IndexType::UInt16,
                indices,
                offset as u64,
                1,
                0,
                0,
            )
            .expect("draw_indexed_primitives");
    }

    fn draw_triangles_indirect(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        arguments: &rt::Buffer,
        offset: usize,
    ) {
        encoder
            .draw_primitives_indirect(PrimitiveType::Triangle, arguments, offset as u64)
            .expect("draw_primitives_indirect");
    }

    fn draw_indexed_indirect(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        indices: &rt::Buffer,
        arguments: &rt::Buffer,
        offset: usize,
    ) {
        encoder
            .draw_indexed_primitives_indirect(
                PrimitiveType::Triangle,
                IndexType::UInt16,
                indices,
                0,
                arguments,
                offset as u64,
            )
            .expect("draw_indexed_primitives_indirect");
    }

    fn set_provoking_vertex_mode(&self, encoder: &mut rt::RenderCommandEncoder, mode: u64) {
        encoder
            .set_provoking_vertex_mode(mode)
            .expect("set_provoking_vertex_mode");
    }

    fn use_in_vertex(&self, encoder: &mut rt::RenderCommandEncoder, buffer: &rt::Buffer) {
        encoder
            .use_resource(
                buffer.as_resource(),
                ResourceUsage::READ,
                metal2vulkan::metal::RenderStages::VERTEX,
            )
            .expect("use_resource");
    }

    fn execute_draws(
        &self,
        encoder: &mut rt::RenderCommandEncoder,
        commands: &rt::IndirectCommandBuffer,
        count: usize,
    ) {
        encoder
            .execute_commands_in_buffer(commands, 0..count as u64)
            .expect("execute_commands_in_buffer");
    }

    fn end_render(&self, encoder: rt::RenderCommandEncoder) {
        encoder.end_encoding().expect("end_encoding");
    }

    fn compute_encoder(&self, command_buffer: &rt::CommandBuffer) -> rt::ComputeCommandEncoder {
        command_buffer
            .compute_command_encoder()
            .expect("compute_command_encoder")
    }

    fn concurrent_compute_encoder(
        &self,
        command_buffer: &rt::CommandBuffer,
    ) -> rt::ComputeCommandEncoder {
        command_buffer
            .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)
            .expect("compute_command_encoder_with_dispatch_type")
    }

    fn set_compute_pipeline(
        &self,
        encoder: &mut rt::ComputeCommandEncoder,
        state: &rt::ComputePipelineState,
    ) {
        encoder
            .set_compute_pipeline_state(state)
            .expect("set_compute_pipeline_state");
    }

    fn set_buffer(
        &self,
        encoder: &mut rt::ComputeCommandEncoder,
        buffer: &rt::Buffer,
        offset: usize,
        index: usize,
    ) {
        encoder
            .set_buffer(Some(buffer), offset as u64, index as u32)
            .expect("set_buffer");
    }

    fn set_texture(
        &self,
        encoder: &mut rt::ComputeCommandEncoder,
        texture: &rt::Texture,
        index: usize,
    ) {
        encoder
            .set_texture(Some(texture), index as u32)
            .expect("set_texture");
    }

    fn dispatch(&self, encoder: &mut rt::ComputeCommandEncoder, groups: usize, threads: usize) {
        encoder
            .dispatch_threadgroups(
                rt::Size::new(groups as u64, 1, 1),
                rt::Size::new(threads as u64, 1, 1),
            )
            .expect("dispatch_threadgroups");
    }

    fn use_commands(
        &self,
        encoder: &mut rt::ComputeCommandEncoder,
        commands: &rt::IndirectCommandBuffer,
    ) {
        encoder
            .use_resource(commands.as_resource(), ResourceUsage::WRITE)
            .expect("use_resource");
    }

    fn use_heap(&self, encoder: &mut rt::ComputeCommandEncoder, heap: &rt::Heap) {
        encoder.use_heap(heap).expect("use_heap");
    }

    fn update_fence(&self, encoder: &mut rt::ComputeCommandEncoder, fence: &rt::Fence) {
        encoder.update_fence(fence).expect("update_fence");
    }

    fn wait_for_fence(&self, encoder: &mut rt::ComputeCommandEncoder, fence: &rt::Fence) {
        encoder.wait_for_fence(fence).expect("wait_for_fence");
    }

    fn use_buffers_read(&self, encoder: &mut rt::ComputeCommandEncoder, buffers: &[&rt::Buffer]) {
        let resources: Vec<rt::ResourceRef> = buffers.iter().map(|b| b.as_resource()).collect();
        encoder
            .use_resources(&resources, ResourceUsage::READ)
            .expect("use_resources");
    }

    fn use_buffers_drawn(&self, encoder: &mut rt::RenderCommandEncoder, buffers: &[&rt::Buffer]) {
        let resources: Vec<rt::ResourceRef> = buffers.iter().map(|b| b.as_resource()).collect();
        let stages =
            metal2vulkan::metal::RenderStages::VERTEX | metal2vulkan::metal::RenderStages::FRAGMENT;
        encoder
            .use_resources(&resources, ResourceUsage::READ, stages)
            .expect("use_resources");
    }

    fn use_buffer(&self, encoder: &mut rt::ComputeCommandEncoder, buffer: &rt::Buffer) {
        encoder
            .use_resource(
                buffer.as_resource(),
                ResourceUsage::READ.union(ResourceUsage::WRITE),
            )
            .expect("use_resource");
    }

    fn use_buffers(&self, encoder: &mut rt::ComputeCommandEncoder, buffers: &[&rt::Buffer]) {
        let resources: Vec<_> = buffers.iter().map(|buffer| buffer.as_resource()).collect();
        encoder
            .use_resources(&resources, ResourceUsage::READ)
            .expect("use_resources");
    }

    fn execute_commands(
        &self,
        encoder: &mut rt::ComputeCommandEncoder,
        commands: &rt::IndirectCommandBuffer,
        count: usize,
    ) {
        encoder
            .execute_commands_in_buffer(commands, 0..count as u64)
            .expect("execute_commands_in_buffer");
    }

    fn end_compute(&self, encoder: rt::ComputeCommandEncoder) {
        encoder.end_encoding().expect("end_encoding");
    }

    fn reset_commands(
        &self,
        command_buffer: &rt::CommandBuffer,
        commands: &rt::IndirectCommandBuffer,
        count: usize,
    ) {
        let mut blit = command_buffer
            .blit_command_encoder()
            .expect("blit_command_encoder");
        blit.reset_commands_in_buffer(commands, 0..count as u64)
            .expect("reset_commands_in_buffer");
        blit.end_encoding().expect("end_encoding");
    }

    fn blit_encoder(&self, command_buffer: &rt::CommandBuffer) -> rt::BlitCommandEncoder {
        command_buffer
            .blit_command_encoder()
            .expect("blit_command_encoder")
    }

    fn copy_buffer(
        &self,
        encoder: &mut rt::BlitCommandEncoder,
        (from, from_offset): (&rt::Buffer, usize),
        (to, to_offset): (&rt::Buffer, usize),
        size: usize,
    ) {
        encoder
            .copy_from_buffer_to_buffer(from, from_offset as u64, to, to_offset as u64, size as u64)
            .expect("copy_from_buffer_to_buffer");
    }

    fn fill_buffer_in(
        &self,
        encoder: &mut rt::BlitCommandEncoder,
        buffer: &rt::Buffer,
        offset: usize,
        length: usize,
    ) {
        encoder
            .fill_buffer(buffer, offset as u64..(offset + length) as u64, 0x5a)
            .expect("fill_buffer");
    }

    fn copy_to_texture(
        &self,
        encoder: &mut rt::BlitCommandEncoder,
        source: &rt::Buffer,
        texture: &rt::Texture,
        (width, height): (usize, usize),
        bytes_per_row: usize,
    ) {
        encoder
            .copy_from_buffer_to_texture(
                source,
                0,
                bytes_per_row as u64,
                (bytes_per_row * height) as u64,
                rt::Size::new(width as u64, height as u64, 1),
                texture,
                0,
                0,
                rt::Origin::new(0, 0, 0),
                rt::BlitOption::None,
            )
            .expect("copy_from_buffer_to_texture");
    }

    fn end_blit(&self, encoder: rt::BlitCommandEncoder) {
        encoder.end_encoding().expect("end_encoding");
    }

    fn fill(&self, command_buffer: &rt::CommandBuffer, buffer: &rt::Buffer, length: usize) {
        let mut blit = command_buffer
            .blit_command_encoder()
            .expect("blit_command_encoder");
        blit.fill_buffer(buffer, 0..length as u64, 0x5a)
            .expect("fill_buffer");
        blit.end_encoding().expect("end_encoding");
    }

    fn upload(
        &self,
        command_buffer: &rt::CommandBuffer,
        source: &rt::Buffer,
        textures: &[&rt::Texture],
        (width, height): (usize, usize),
        bytes_per_row: usize,
    ) {
        let mut blit = command_buffer
            .blit_command_encoder()
            .expect("blit_command_encoder");
        for texture in textures {
            self.copy_to_texture(&mut blit, source, texture, (width, height), bytes_per_row);
        }
        blit.end_encoding().expect("end_encoding");
    }

    fn commit(&self, command_buffer: &rt::CommandBuffer) {
        command_buffer.commit().expect("commit");
    }

    fn wait(&self, command_buffer: &rt::CommandBuffer) {
        command_buffer
            .wait_until_completed()
            .expect("wait_until_completed");
        let status = command_buffer.status();
        assert_eq!(
            status,
            rt::CommandBufferStatus::Completed,
            "the command buffer ended {status:?}: {:?}",
            command_buffer.error()
        );
    }

    fn wait_only(&self, command_buffer: &rt::CommandBuffer) {
        command_buffer
            .wait_until_completed()
            .expect("wait_until_completed");
    }

    fn status(&self, command_buffer: &rt::CommandBuffer) -> u64 {
        command_buffer.status().raw()
    }

    fn fence(&self) -> rt::Fence {
        self.device.new_fence()
    }

    fn shared_event(&self) -> rt::SharedEvent {
        self.device.new_shared_event().expect("new_shared_event")
    }

    fn event_value(&self, event: &rt::SharedEvent) -> u64 {
        event.signaled_value()
    }

    fn set_event_value(&self, event: &rt::SharedEvent, value: u64) {
        event.set_signaled_value(value);
    }

    fn set_when_scheduled(
        &self,
        command_buffer: &rt::CommandBuffer,
        event: &rt::SharedEvent,
        value: u64,
    ) {
        let event = event.clone();
        command_buffer
            .add_scheduled_handler(Box::new(move |_| event.set_signaled_value(value)))
            .expect("add_scheduled_handler");
    }

    fn signal_when_scheduled(
        &self,
        command_buffer: &rt::CommandBuffer,
        event: &rt::SharedEvent,
        value: u64,
    ) {
        command_buffer
            .encode_signal_event_scheduled(event, value)
            .expect("encode_signal_event_scheduled");
    }

    fn listener(&self) -> rt::SharedEventListener {
        rt::SharedEventListener::new()
    }

    fn notify(
        &self,
        event: &rt::SharedEvent,
        listener: &rt::SharedEventListener,
        value: u64,
        block: Box<dyn FnOnce() + Send>,
    ) {
        event.notify_listener(listener, value, Box::new(move |_, _| block()));
    }

    fn check_submissions(&self, command_buffer: &rt::CommandBuffer) -> bool {
        command_buffer
            .set_submission_check(std::sync::Arc::new(|_: &rt::Submission<'_>| None))
            .expect("set_submission_check");
        true
    }

    fn on_completed(&self, command_buffer: &rt::CommandBuffer, handler: Box<dyn FnOnce() + Send>) {
        command_buffer
            .add_completed_handler(Box::new(move |_| handler()))
            .expect("add_completed_handler");
    }

    fn gpu_seconds(&self, _command_buffer: &rt::CommandBuffer) -> Option<f64> {
        // On MoltenVK the runtime answers GPU times from the host's clock (their docs), which
        // is no measure of the GPU.
        None
    }

    fn times(&self, command_buffer: &rt::CommandBuffer) -> [f64; 4] {
        [
            command_buffer.kernel_start_time(),
            command_buffer.kernel_end_time(),
            command_buffer.gpu_start_time().unwrap_or(0.0),
            command_buffer.gpu_end_time().unwrap_or(0.0),
        ]
    }

    fn pool<R>(&self, f: impl FnOnce() -> R) -> R {
        f()
    }

    unsafe fn driver_floor(&self, windows: usize, pages: *mut u8) {
        // SAFETY: the caller's contract, passed on.
        unsafe { crate::vkfloor::measure(self, &self.device, windows, pages) };
    }

    fn work_counts(&self) -> Vec<(&'static str, f64)> {
        let c = self.device.work_counts();
        let replayed = &c.commands_replayed;
        [
            ("queue_submissions", c.queue_submissions),
            ("submit_infos", c.submit_infos),
            ("vk_command_buffers_submitted", c.command_buffers_submitted),
            ("first_work_buffers", c.command_buffers_begun_batch_work),
            ("barrier_buffers", c.command_buffers_begun_batch_barriers),
            ("finales_apart", c.finales_apart),
            ("render_passes_begun", replayed.cmd_begin_render_pass),
            ("render_encoders_ended", c.render_encoders_ended),
            ("passes_begun_instances", c.render_passes_begun_instances),
            ("passes_begun_inline", c.render_passes_begun_inline),
            (
                "passes_begun_inline_splits",
                c.render_passes_begun_inline_splits,
            ),
            ("passes_begun_resolves", c.render_passes_begun_resolves),
            ("splits_barrier", c.instance_splits_barrier),
            ("splits_tile_dispatch", c.instance_splits_tile_dispatch),
            ("splits_tile_memory", c.instance_splits_tile_memory),
            ("splits_attached_read", c.instance_splits_attached_read),
            ("splits_feedback", c.instance_splits_feedback),
            (
                "splits_memoryless_write",
                c.instance_splits_memoryless_write,
            ),
            (
                "splits_pending_pipeline",
                c.instance_splits_pending_pipeline,
            ),
            ("splits_execution", c.instance_splits_execution),
            ("splits_visibility", c.instance_splits_visibility),
            ("splits_shape", c.instance_splits_shape),
            ("pipeline_barriers", replayed.cmd_pipeline_barrier),
            ("vk_images_destroyed", c.object_churn.vk_images_destroyed),
            ("images_own_memory", c.object_churn.images_own_memory),
            ("images_parked", c.object_churn.images_parked),
            ("images_recycled", c.object_churn.images_recycled),
            ("images_evicted", c.object_churn.images_evicted),
            ("requirement_images_made", c.requirement_images_made),
            ("arena_pools_made", c.arena_pools_made),
            ("command_pools_made", c.command_pools_made),
            ("dispatches", replayed.cmd_dispatch),
            ("draws", replayed.cmd_draw),
            ("gpu_form_commands", c.gpu_form_commands),
            ("merged_draw_commands", c.merged_draw_commands),
            ("indirect_records_read", c.indirect_records_read),
            ("dispatches_indirect", replayed.cmd_dispatch_indirect),
            ("wakes_handler_threads", c.wakes_handler_threads),
            ("scheduled_where_submitted", c.scheduled_where_submitted),
            ("scheduler_passes", c.scheduler_passes),
            ("scheduler_sleeps", c.scheduler_sleeps),
            ("scheduler_wakes_commit", c.scheduler_wakes_commit),
            ("scheduler_wakes_host", c.scheduler_wakes_host),
            (
                "dispatches_settled_as_before",
                c.dispatches_settled_as_before,
            ),
            (
                "dispatches_planned_as_before",
                c.dispatches_planned_as_before,
            ),
            (
                "provoking_last_pipeline_draws",
                c.provoking_last_pipeline_draws,
            ),
            ("provoking_listed_draws", c.provoking_listed_draws),
            ("gathered_draws", c.gathered_draws),
            ("bounded_run_draws", c.bounded_run_draws),
            ("variant_pipelines_made", c.variant_pipelines_made),
            ("host_pointer_imports_made", c.host_pointer_imports_made),
            ("completed_waits", c.completed_waits),
            ("device_memory_bytes", c.device_memory_allocated),
            ("cleared_bytes", c.new_buffer_bytes_cleared),
            ("new_buffer_zero_fills", c.new_buffer_zero_fills),
            (
                "replayed_begin_command_buffer",
                replayed.begin_command_buffer,
            ),
            ("replayed_end_command_buffer", replayed.end_command_buffer),
            ("replayed_cmd_bind_pipeline", replayed.cmd_bind_pipeline),
            ("replayed_cmd_set_viewport", replayed.cmd_set_viewport),
            ("replayed_cmd_set_scissor", replayed.cmd_set_scissor),
            ("replayed_cmd_set_depth_bias", replayed.cmd_set_depth_bias),
            (
                "replayed_cmd_set_blend_constants",
                replayed.cmd_set_blend_constants,
            ),
            (
                "replayed_cmd_set_stencil_compare_mask",
                replayed.cmd_set_stencil_compare_mask,
            ),
            (
                "replayed_cmd_set_stencil_write_mask",
                replayed.cmd_set_stencil_write_mask,
            ),
            (
                "replayed_cmd_set_stencil_reference",
                replayed.cmd_set_stencil_reference,
            ),
            (
                "replayed_cmd_bind_descriptor_sets",
                replayed.cmd_bind_descriptor_sets,
            ),
            (
                "replayed_cmd_bind_index_buffer",
                replayed.cmd_bind_index_buffer,
            ),
            (
                "replayed_cmd_bind_vertex_buffers",
                replayed.cmd_bind_vertex_buffers,
            ),
            ("replayed_cmd_draw_indexed", replayed.cmd_draw_indexed),
            ("replayed_cmd_draw_indirect", replayed.cmd_draw_indirect),
            (
                "replayed_cmd_draw_indexed_indirect",
                replayed.cmd_draw_indexed_indirect,
            ),
            ("replayed_cmd_copy_buffer", replayed.cmd_copy_buffer),
            ("replayed_cmd_copy_image", replayed.cmd_copy_image),
            ("replayed_cmd_blit_image", replayed.cmd_blit_image),
            (
                "replayed_cmd_copy_buffer_to_image",
                replayed.cmd_copy_buffer_to_image,
            ),
            (
                "replayed_cmd_copy_image_to_buffer",
                replayed.cmd_copy_image_to_buffer,
            ),
            ("replayed_cmd_update_buffer", replayed.cmd_update_buffer),
            ("replayed_cmd_fill_buffer", replayed.cmd_fill_buffer),
            (
                "replayed_cmd_clear_color_image",
                replayed.cmd_clear_color_image,
            ),
            (
                "replayed_cmd_clear_depth_stencil_image",
                replayed.cmd_clear_depth_stencil_image,
            ),
            (
                "replayed_cmd_clear_attachments",
                replayed.cmd_clear_attachments,
            ),
            ("replayed_cmd_begin_query", replayed.cmd_begin_query),
            ("replayed_cmd_end_query", replayed.cmd_end_query),
            (
                "replayed_cmd_reset_query_pool",
                replayed.cmd_reset_query_pool,
            ),
            ("replayed_cmd_write_timestamp", replayed.cmd_write_timestamp),
            (
                "replayed_cmd_copy_query_pool_results",
                replayed.cmd_copy_query_pool_results,
            ),
            ("replayed_cmd_push_constants", replayed.cmd_push_constants),
            ("replayed_cmd_end_render_pass", replayed.cmd_end_render_pass),
            (
                "replayed_cmd_execute_commands",
                replayed.cmd_execute_commands,
            ),
            ("replayed_cmd_set_cull_mode", replayed.cmd_set_cull_mode),
            ("replayed_cmd_set_front_face", replayed.cmd_set_front_face),
            (
                "replayed_cmd_set_primitive_topology",
                replayed.cmd_set_primitive_topology,
            ),
            (
                "replayed_cmd_bind_vertex_buffers2",
                replayed.cmd_bind_vertex_buffers2,
            ),
            (
                "replayed_cmd_set_depth_test_enable",
                replayed.cmd_set_depth_test_enable,
            ),
            (
                "replayed_cmd_set_depth_write_enable",
                replayed.cmd_set_depth_write_enable,
            ),
            (
                "replayed_cmd_set_depth_compare_op",
                replayed.cmd_set_depth_compare_op,
            ),
            (
                "replayed_cmd_set_stencil_test_enable",
                replayed.cmd_set_stencil_test_enable,
            ),
            ("replayed_cmd_set_stencil_op", replayed.cmd_set_stencil_op),
            (
                "replayed_cmd_push_descriptor_set",
                replayed.cmd_push_descriptor_set,
            ),
            (
                "replayed_cmd_set_color_write_mask",
                replayed.cmd_set_color_write_mask,
            ),
            (
                "replayed_cmd_bind_index_buffer2",
                replayed.cmd_bind_index_buffer2,
            ),
            ("replayed_cmd_draw_mesh_tasks", replayed.cmd_draw_mesh_tasks),
            (
                "replayed_cmd_draw_mesh_tasks_indirect",
                replayed.cmd_draw_mesh_tasks_indirect,
            ),
            ("copy_regions_split", c.copy_regions_split),
            ("spare_command_pools_taken", c.spare_command_pools_taken),
            ("spare_command_pools_reset", c.spare_command_pools_reset),
            ("live_tables_uploaded", c.live_tables_uploaded),
            ("timeline_waits", c.timeline_waits),
            ("missed_completion_wakes", c.missed_completion_wakes),
            ("completed_where_seen", c.completed_where_seen),
            ("timestamp_slots_taken", c.timestamp_slots_taken),
            (
                "timestamp_slots_reset_on_host",
                c.timestamp_slots_reset_on_host,
            ),
            ("calibrations", c.calibrations),
            ("calibration_readings", c.calibration_readings),
            ("calibration_deviation_most", c.calibration_deviation_most),
            ("calibration_disagreements", c.calibration_disagreements),
            ("storage_forms_taken", c.storage_forms_taken),
            ("addressable_buffers_made", c.addressable_buffers_made),
            ("mipmaps_filtered", c.mipmaps_filtered),
            ("mipmap_first_levels_fetched", c.mipmap_first_levels_fetched),
            ("pipelines_pooled", c.pipelines_pooled),
            ("residual_form_draws", c.residual_form_draws),
            ("residual_form_makings", c.residual_form_makings),
            ("pipeline_pool_busy", c.pipeline_pool_busy),
            ("shader_modules_made", c.shader_modules_made),
            ("shader_modules_shared", c.shader_modules_shared),
            ("pipeline_layouts_made", c.pipeline_layouts_made),
            ("pipeline_layouts_shared", c.pipeline_layouts_shared),
            ("flights_followed", c.flights_followed),
            ("predicated_loop_runs_again", c.predicated_loop_runs_again),
            ("variants_taken_back", c.variants_taken_back),
            (
                "unlisted_argument_sampler_buffers",
                c.unlisted_argument_sampler_buffers,
            ),
            ("host_step_sampler_sets_made", c.host_step_sampler_sets_made),
            ("creation_pipelines_made", c.creation_pipelines_made),
            ("creation_pipelines_drawn", c.creation_pipelines_drawn),
            ("sampler_sets_made", c.sampler_sets_made),
            ("sampler_sets_shared", c.sampler_sets_shared),
            ("sampler_sets_destroyed", c.sampler_sets_destroyed),
            ("ahead_render_passes_made", c.ahead_render_passes_made),
            ("render_passes_made", c.render_passes_made),
            ("ahead_framebuffers_made", c.ahead_framebuffers_made),
            ("kept_backings_made", c.kept_backings_made),
            ("pooled_backings_made", c.pooled_backings_made),
            ("storage_placeholders_made", c.storage_placeholders_made),
            (
                "storage_placeholder_attachments",
                c.storage_placeholder_attachments,
            ),
            ("color_resolve_passes_made", c.color_resolve_passes_made),
            ("color_resolve_passes_kept", c.color_resolve_passes_kept),
            ("ahead_nanos", c.ahead_nanos),
            ("driver_diagnostics_withheld", c.driver_diagnostics_withheld),
            ("memory_blocks_live", c.memory_blocks_live),
            ("empty_memory_blocks_live", c.empty_memory_blocks_live),
            ("pooled_memory_blocks_made", c.pooled_memory_blocks_made),
            ("pooled_memory_blocks_freed", c.pooled_memory_blocks_freed),
            (
                "dedicated_memory_blocks_live",
                c.dedicated_memory_blocks_live,
            ),
            ("host_imports_live", c.host_imports_live),
            ("freed_memory_parked", c.freed_memory_parked),
            ("freed_memory_waiting", c.freed_memory_waiting),
            ("stand_ins_made", c.stand_ins_made),
            ("stand_in_bytes", c.stand_in_bytes),
            ("stale_detached_reaches", c.stale_detached_reaches),
            ("detached_storage_reaches", c.detached_storage_reaches),
            ("heap_slots_rewritten", c.heap_slots_rewritten),
            ("heap_slot_writes_deferred", c.heap_slot_writes_deferred),
            ("host_range_lookups", c.host_range_lookups),
            ("heap_textures_tiled", c.heap_textures_tiled),
            (
                "heap_sample_move_images_made",
                c.heap_sample_move_images_made,
            ),
            (
                "heap_sample_move_images_lent",
                c.heap_sample_move_images_lent,
            ),
            ("requirement_limits_asked", c.requirement_limits_asked),
            (
                "heap_texture_lossless_layouts",
                c.heap_texture_lossless_layouts,
            ),
            (
                "heap_texture_block_compressed_layouts",
                c.heap_texture_block_compressed_layouts,
            ),
            (
                "heap_texture_untiled_layouts",
                c.heap_texture_untiled_layouts,
            ),
            ("fills_recorded", c.fills_recorded),
            ("fills_by_transfers", c.fills_by_transfers),
            ("ycbcr_refreshes", c.ycbcr_refreshes),
            ("ycbcr_whole_conversions", c.ycbcr_whole_conversions),
            ("attachment_clears_recorded", c.attachment_clears_recorded),
            ("image_blits_recorded", c.image_blits_recorded),
            (
                "query_result_copies_recorded",
                c.query_result_copies_recorded,
            ),
            ("visibility_queries_begun", c.visibility_queries_begun),
            ("indirect_draws_recorded", c.indirect_draws_recorded),
            ("gpu_pool_blocks_made", c.gpu_pool_blocks_made),
            ("gpu_pool_blocks_released", c.gpu_pool_blocks_released),
            ("gpu_pool_leases_peak", c.gpu_pool_leases_peak),
            ("internal_submissions", c.internal_submissions),
            ("batch_submit_nanos", c.batch_submit_nanos),
            ("command_buffers_begun", c.command_buffers_begun),
            (
                "encoder_command_buffers_begun",
                c.encoder_command_buffers_begun,
            ),
            (
                "command_buffers_begun_encoders",
                c.command_buffers_begun_encoders,
            ),
            (
                "command_buffers_begun_segments",
                c.command_buffers_begun_segments,
            ),
            (
                "command_buffers_begun_continuations",
                c.command_buffers_begun_continuations,
            ),
            (
                "command_buffers_begun_secondaries",
                c.command_buffers_begun_secondaries,
            ),
            (
                "command_buffers_begun_kept_barriers",
                c.command_buffers_begun_kept_barriers,
            ),
            (
                "command_buffers_begun_internal",
                c.command_buffers_begun_internal,
            ),
            (
                "encoder_primaries_begun_ahead",
                c.encoder_primaries_begun_ahead,
            ),
            ("primaries_begun_ahead", c.primaries_begun_ahead),
            ("barrier_buffers_kept", c.barrier_buffers_kept),
            ("stamped_buffers_kept", c.stamped_buffers_kept),
            ("stamped_buffers_reused", c.stamped_buffers_reused),
            ("finales_folded", c.finales_folded),
            ("grouped_submissions", c.grouped_submissions),
            ("batches_grouped", c.batches_grouped),
            ("group_ends_full", c.group_ends_full),
            ("group_ends_wait", c.group_ends_wait),
            ("group_ends_apart", c.group_ends_apart),
            ("group_ends_first_work", c.group_ends_first_work),
            ("group_ends_drained", c.group_ends_drained),
            ("group_ends_checks", c.group_ends_checks),
            ("group_ends_held", c.group_ends_held),
            ("batches_with_none_pending", c.batches_with_none_pending),
            ("batches_with_one_pending", c.batches_with_one_pending),
            (
                "batches_with_two_or_three_pending",
                c.batches_with_two_or_three_pending,
            ),
            (
                "batches_with_four_or_more_pending",
                c.batches_with_four_or_more_pending,
            ),
            ("groups_kept_pending", c.groups_kept_pending),
            ("groups_sent_for_waits", c.groups_sent_for_waits),
            ("registrations_waited", c.registrations_waited),
            ("scheduler_wakes_completion", c.scheduler_wakes_completion),
            ("scheduler_sleep_slices", c.scheduler_sleep_slices),
            ("scheduler_gpu_waits", c.scheduler_gpu_waits),
            ("wakes_completion_waiters", c.wakes_completion_waiters),
            (
                "wakes_command_buffer_waiters",
                c.wakes_command_buffer_waiters,
            ),
            ("wakes_event_waiters", c.wakes_event_waiters),
            ("wakes_spared", c.wakes_spared),
            ("slot_returns", c.slot_returns),
            ("slot_return_nanos", c.slot_return_nanos),
            ("slot_waits", c.slot_waits),
            ("slot_wait_nanos", c.slot_wait_nanos),
            ("event_waits_met", c.event_waits_met),
            ("shared_event_waits_held", c.shared_event_waits_held),
            ("bare_starts", c.bare_starts),
            ("prefixed_starts", c.prefixed_starts),
            ("command_barriers", c.command_barriers),
            ("unrun_instances_ended", c.unrun_instances_ended),
            ("untouched_passes_skipped", c.untouched_passes_skipped),
            ("inline_passes", c.inline_passes),
            ("inline_passes_split", c.inline_passes_split),
            ("inline_split_instances", c.inline_split_instances),
            (
                "inline_memoryless_write_splits",
                c.inline_memoryless_write_splits,
            ),
            (
                "render_log_batches_by_thread",
                c.render_log_batches_by_thread,
            ),
            (
                "render_log_batches_by_program",
                c.render_log_batches_by_program,
            ),
            ("render_log_waits", c.render_log_waits),
            ("render_log_ends_by_thread", c.render_log_ends_by_thread),
            ("render_log_ends_by_program", c.render_log_ends_by_program),
            (
                "render_log_commits_by_thread",
                c.render_log_commits_by_thread,
            ),
            (
                "render_log_commits_by_program",
                c.render_log_commits_by_program,
            ),
            ("passes_not_inline_encoders", c.passes_not_inline_encoders),
            ("passes_not_inline_layouts", c.passes_not_inline_layouts),
            (
                "passes_not_inline_memoryless",
                c.passes_not_inline_memoryless,
            ),
            ("passes_not_inline_actions", c.passes_not_inline_actions),
            (
                "passes_not_inline_actions_dont_care_load",
                c.passes_not_inline_actions_dont_care_load,
            ),
            (
                "passes_not_inline_actions_unequal_reach",
                c.passes_not_inline_actions_unequal_reach,
            ),
            (
                "passes_not_inline_actions_resolve",
                c.passes_not_inline_actions_resolve,
            ),
            (
                "passes_not_inline_actions_store",
                c.passes_not_inline_actions_store,
            ),
            (
                "passes_not_inline_buffer_textures",
                c.passes_not_inline_buffer_textures,
            ),
            (
                "passes_not_inline_attachments",
                c.passes_not_inline_attachments,
            ),
            ("passes_not_inline_storage", c.passes_not_inline_storage),
            ("passes_not_inline_stand_ins", c.passes_not_inline_stand_ins),
            (
                "passes_not_inline_memoryless_writes",
                c.passes_not_inline_memoryless_writes,
            ),
            ("passes_not_inline_draws", c.passes_not_inline_draws),
            ("commits_handed_off", c.commits_handed_off),
            ("images_started_on_the_host", c.images_started_on_the_host),
            ("host_texel_copies", c.host_texel_copies),
            ("staging_submissions", c.staging_submissions),
            ("uncached_host_images", c.uncached_host_images),
            ("cpu_block_decodes", c.cpu_block_decodes),
            ("block_decode_passes", c.block_decode_passes),
            ("second_twins", c.second_twins),
            ("second_twin_checks", c.second_twin_checks),
            ("second_twin_decodes", c.second_twin_decodes),
            ("queue_drains", c.queue_drains),
            ("stand_in_copies_made", c.stand_in_copies_made),
            (
                "stand_in_copies_made_at_draws",
                c.stand_in_copies_made_at_draws,
            ),
            ("draw_shapes_made", c.draw_shapes_made),
            ("draw_shapes_found", c.draw_shapes_found),
            ("argument_samplers_taken", c.argument_samplers_taken),
            ("bind_views_worked_out", c.bind_views_worked_out),
            ("stage_tables_read", c.stage_tables_read),
            ("stage_tables_read_in_part", c.stage_tables_read_in_part),
            ("no_copy_lies_read", c.no_copy_lies_read),
            ("commands_noted_as_before", c.commands_noted_as_before),
            ("draws_settled", c.draws_settled),
            ("two_plane_draws", c.two_plane_draws),
            ("bound_notices_made", c.bound_notices_made),
            ("argument_tables_retained", c.argument_tables_retained),
            ("argument_objects_retained", c.argument_objects_retained),
            (
                "graphics_pipelines_looked_up",
                c.graphics_pipelines_looked_up,
            ),
            ("linear_passes_in_place", c.linear_passes_in_place),
            (
                "linear_passes_through_copies",
                c.linear_passes_through_copies,
            ),
            ("kept_bytes_passes", c.kept_bytes_passes),
            ("color_group_passes", c.color_group_passes),
            ("color_group_twins", c.color_group_twins),
            ("color_group_draws_skipped", c.color_group_draws_skipped),
            ("draw_states_recorded", c.draw_states_recorded),
            ("draws_pushing_set_zero", c.draws_pushing_set_zero),
            ("set_zero_writes_pushed", c.set_zero_writes_pushed),
            (
                "set_zero_equal_shape_switches",
                c.set_zero_equal_shape_switches,
            ),
            (
                "set_zero_writes_on_equal_shape_switches",
                c.set_zero_writes_on_equal_shape_switches,
            ),
            (
                "set_zero_writes_if_shapes_shared",
                c.set_zero_writes_if_shapes_shared,
            ),
            ("dispatches_pushing_set_zero", c.dispatches_pushing_set_zero),
            ("set_zero_sets_reused", c.set_zero_sets_reused),
            ("set_zero_sets_found", c.set_zero_sets_found),
            ("draw_shapes_reused", c.draw_shapes_reused),
            (
                "baked_sampler_dispatches_apart",
                c.baked_sampler_dispatches_apart,
            ),
            ("host_deallocators_delayed", c.host_deallocators_delayed),
            ("host_import_releases_posted", c.host_import_releases_posted),
            (
                "host_import_releases_pending",
                c.host_import_releases_pending,
            ),
            ("host_keep_alives_kept", c.host_keep_alives_kept),
            ("host_keep_alives_covered", c.host_keep_alives_covered),
            ("host_mappings_queried", c.host_mappings_queried),
            ("dma_buf_imports", c.dma_buf_imports),
            ("dma_buf_memories_live", c.dma_buf_memories_live),
            ("dma_buf_image_memories_live", c.dma_buf_image_memories_live),
            ("host_imports_released", c.host_imports_released),
            ("host_imports_reimported", c.host_imports_reimported),
            ("host_imports_resident", c.host_imports_resident),
            ("host_imports_lost", c.host_imports_lost),
            ("host_import_fds_kept", c.host_import_fds_kept),
            ("host_imports_kept_for_fds", c.host_imports_kept_for_fds),
            ("host_imports_used", c.host_imports_used),
            ("host_imports_read_apart", c.host_imports_read_apart),
            (
                "host_imports_read_apart_address",
                c.host_imports_read_apart_address,
            ),
            (
                "host_imports_read_apart_vk_buffer",
                c.host_imports_read_apart_vk_buffer,
            ),
            (
                "host_imports_read_apart_binding",
                c.host_imports_read_apart_binding,
            ),
            (
                "host_imports_read_apart_texture",
                c.host_imports_read_apart_texture,
            ),
            (
                "host_imports_read_apart_second_buffer",
                c.host_imports_read_apart_second_buffer,
            ),
            (
                "host_imports_read_apart_unreleased",
                c.host_imports_read_apart_unreleased,
            ),
            ("host_remaps_reported", c.host_remaps_reported),
            ("host_remaps_over_imports", c.host_remaps_over_imports),
            ("host_imports_remapped", c.host_imports_remapped),
            (
                "host_pointer_import_bytes_made",
                c.host_pointer_import_bytes_made,
            ),
            ("host_imports_kept_of_heap", c.host_imports_kept_of_heap),
            ("host_imports_kept_rebound", c.host_imports_kept_rebound),
            (
                "host_imports_rebound_unlisted",
                c.host_imports_rebound_unlisted,
            ),
            ("host_rebind_imports_made", c.host_rebind_imports_made),
            ("host_rebind_imports_covered", c.host_rebind_imports_covered),
            ("host_imports_kept_pointer", c.host_imports_kept_pointer),
            (
                "host_imports_kept_no_stand_in",
                c.host_imports_kept_no_stand_in,
            ),
            ("dma_buf_sparse_buffers", c.dma_buf_sparse_buffers),
            ("dma_buf_images", c.dma_buf_images),
            ("dma_buf_image_fallbacks", c.dma_buf_image_fallbacks),
            ("modifier_feature_queries", c.modifier_feature_queries),
            ("shadowed_textures", c.shadowed_textures),
            ("shadows_zeroed", c.shadows_zeroed),
            ("shadows_over_fresh_pages", c.shadows_over_fresh_pages),
            ("fresh_page_buffers_made", c.fresh_page_buffers_made),
            ("fresh_page_bytes", c.fresh_page_bytes),
            ("shadow_copies", c.shadow_copies),
            ("shadow_texels_copied_in", c.shadow_texels_copied_in),
            (
                "shadow_texels_copied_in_images",
                c.shadow_texels_copied_in_images,
            ),
            (
                "shadow_texels_first_copied_in",
                c.shadow_texels_first_copied_in,
            ),
            (
                "shadow_texels_first_copied_in_images",
                c.shadow_texels_first_copied_in_images,
            ),
            (
                "shadow_texels_first_copied_in_private_buffers",
                c.shadow_texels_first_copied_in_private_buffers,
            ),
            (
                "shadow_texels_first_copied_in_mapped_buffers",
                c.shadow_texels_first_copied_in_mapped_buffers,
            ),
            (
                "shadow_texels_first_copied_in_host_pages",
                c.shadow_texels_first_copied_in_host_pages,
            ),
            (
                "shadow_texels_first_copied_in_aliased_host_pages",
                c.shadow_texels_first_copied_in_aliased_host_pages,
            ),
            (
                "shadow_texels_first_copied_in_surfaces",
                c.shadow_texels_first_copied_in_surfaces,
            ),
            (
                "shadow_texels_first_copied_in_large_surfaces",
                c.shadow_texels_first_copied_in_large_surfaces,
            ),
            (
                "shadow_texels_first_copied_in_again",
                c.shadow_texels_first_copied_in_again,
            ),
            (
                "shadow_texels_recopied_beyond_held",
                c.shadow_texels_recopied_beyond_held,
            ),
            (
                "shadow_texels_recopied_after_changes",
                c.shadow_texels_recopied_after_changes,
            ),
            (
                "shadow_texels_recopied_after_other_shadows",
                c.shadow_texels_recopied_after_other_shadows,
            ),
            (
                "shadow_texels_recopied_after_two_ways_merges",
                c.shadow_texels_recopied_after_two_ways_merges,
            ),
            (
                "shadow_texels_recopied_after_writes",
                c.shadow_texels_recopied_after_writes,
            ),
            (
                "shadow_texels_recopied_after_writes_within",
                c.shadow_texels_recopied_after_writes_within,
            ),
            (
                "shadow_texels_recopied_after_dropped_writes",
                c.shadow_texels_recopied_after_dropped_writes,
            ),
            (
                "shadow_texels_bounded_copied_in",
                c.shadow_texels_bounded_copied_in,
            ),
            (
                "shadow_texels_recopied_in_images",
                c.shadow_texels_recopied_in_images,
            ),
            (
                "shadow_texels_recopied_in_private_buffers",
                c.shadow_texels_recopied_in_private_buffers,
            ),
            (
                "shadow_texels_recopied_in_mapped_buffers",
                c.shadow_texels_recopied_in_mapped_buffers,
            ),
            (
                "shadow_texels_recopied_in_host_pages",
                c.shadow_texels_recopied_in_host_pages,
            ),
            (
                "shadow_texels_recopied_in_aliased_host_pages",
                c.shadow_texels_recopied_in_aliased_host_pages,
            ),
            (
                "shadow_texels_recopied_in_surfaces",
                c.shadow_texels_recopied_in_surfaces,
            ),
            (
                "shadow_texels_recopied_in_large_surfaces",
                c.shadow_texels_recopied_in_large_surfaces,
            ),
            ("shadow_texels_copied_back", c.shadow_texels_copied_back),
            (
                "shadow_copies_back_dispatched",
                c.shadow_copies_back_dispatched,
            ),
            ("shadow_copies_in_dispatched", c.shadow_copies_in_dispatched),
            ("shadow_keeps_from_staging", c.shadow_keeps_from_staging),
            (
                "shadow_merge_copies_dispatched",
                c.shadow_merge_copies_dispatched,
            ),
            (
                "shadow_merge_scratches_reused",
                c.shadow_merge_scratches_reused,
            ),
            (
                "shadow_merge_scratches_zeroed",
                c.shadow_merge_scratches_zeroed,
            ),
            ("row_copies", c.row_copies),
            ("row_copy_regions", c.row_copy_regions),
            ("row_copy_regions_most", c.row_copy_regions_most),
            ("row_copies_dispatched", c.row_copies_dispatched),
            ("shadow_merges_fetched", c.shadow_merges_fetched),
            (
                "shadow_merges_kept_in_images",
                c.shadow_merges_kept_in_images,
            ),
            (
                "shadow_merge_kept_images_made",
                c.shadow_merge_kept_images_made,
            ),
            (
                "shadow_merge_kept_images_followed",
                c.shadow_merge_kept_images_followed,
            ),
            (
                "shadow_merge_kept_images_live",
                c.shadow_merge_kept_images_live,
            ),
            ("shadow_attachment_passes", c.shadow_attachment_passes),
            ("shadow_attachment_textures", c.shadow_attachment_textures),
            (
                "shadow_attachment_loads_within",
                c.shadow_attachment_loads_within,
            ),
            (
                "shadow_attachment_loads_whole",
                c.shadow_attachment_loads_whole,
            ),
            ("shadow_attachment_clears", c.shadow_attachment_clears),
            (
                "shadow_attachment_clears_in_part",
                c.shadow_attachment_clears_in_part,
            ),
            (
                "shadow_attachment_loads_untouched",
                c.shadow_attachment_loads_untouched,
            ),
            (
                "shadow_attachment_tiles_covered",
                c.shadow_attachment_tiles_covered,
            ),
            (
                "shadow_attachment_tiles_bounded",
                c.shadow_attachment_tiles_bounded,
            ),
            ("shadows_sharing_truth", c.shadows_sharing_truth),
            ("owned_descriptor_pools", c.owned_descriptor_pools),
            ("shadow_texels_kept", c.shadow_texels_kept),
            ("bounded_shadow_copies", c.bounded_shadow_copies),
            ("views_through_swizzle", c.views_through_swizzle),
            ("view_shadows", c.view_shadows),
            (
                "attachment_reads_through_copies",
                c.attachment_reads_through_copies,
            ),
            ("attachment_read_splits", c.attachment_read_splits),
            ("reading_draws_snapshotted", c.reading_draws_snapshotted),
            ("snapshot_inline_splits", c.snapshot_inline_splits),
            ("feedback_loop_alias_images", c.feedback_loop_alias_images),
            ("depth_attachments_moved", c.depth_attachments_moved),
            ("depth_instances_held", c.depth_instances_held),
            ("held_depths_released", c.held_depths_released),
            ("own_pixel_reads", c.own_pixel_reads),
            ("replay_logs_kept", c.replay_logs_kept),
            ("replay_log_bytes", c.replay_log_bytes),
            ("encoders_reencoded", c.encoders_reencoded),
            ("segments_recorded_again", c.segments_recorded_again),
            ("replay_tables_pointed", c.replay_tables_pointed),
            ("reencodes_refused", c.reencodes_refused),
            ("fences_made", c.fences_made),
            ("pipeline_libraries_made", c.pipeline_libraries_made),
            (
                "fragment_shader_libraries_made",
                c.fragment_shader_libraries_made,
            ),
            ("pipelines_linked", c.pipelines_linked),
            ("dispatch_pipelines_pooled", c.dispatch_pipelines_pooled),
            ("tile_pipelines_pooled", c.tile_pipelines_pooled),
            (
                "upload_chunks_in_device_memory",
                c.upload_chunks_in_device_memory,
            ),
            ("tile_dispatches_recorded", c.tile_dispatches_recorded),
            (
                "churn_vk_buffers_destroyed",
                c.object_churn.vk_buffers_destroyed,
            ),
            (
                "churn_vk_buffer_views_destroyed",
                c.object_churn.vk_buffer_views_destroyed,
            ),
            ("churn_vk_memory_freed", c.object_churn.vk_memory_freed),
            ("vk_images_made", c.object_churn.vk_images_made),
            ("vk_buffers_made", c.object_churn.vk_buffers_made),
            ("vk_image_views_made", c.object_churn.vk_image_views_made),
            (
                "vk_memory_allocations",
                c.object_churn.vk_memory_allocations,
            ),
            ("vk_image_bindings", c.object_churn.vk_image_bindings),
            ("vk_buffer_bindings", c.object_churn.vk_buffer_bindings),
            ("churn_images_in_buffers", c.object_churn.images_in_buffers),
            ("churn_images_in_heaps", c.object_churn.images_in_heaps),
            ("churn_images_other", c.object_churn.images_other),
            ("churn_textures_image", c.object_churn.textures_image),
            ("churn_textures_in_heaps", c.object_churn.textures_in_heaps),
            ("churn_textures_shadow", c.object_churn.textures_shadow),
            ("churn_textures_alias", c.object_churn.textures_alias),
            ("churn_textures_texels", c.object_churn.textures_texels),
            ("churn_buffers_recycled", c.object_churn.buffers_recycled),
            ("churn_buffers_parked", c.object_churn.buffers_parked),
            ("churn_buffers_evicted", c.object_churn.buffers_evicted),
            (
                "churn_imports_inside_live",
                c.object_churn.imports_inside_live,
            ),
            ("keeps_render_cut", c.shadow_keeps.render_cut),
            ("keeps_render_cut_texels", c.shadow_keeps.render_cut_texels),
            ("keeps_render_whole", c.shadow_keeps.render_whole),
            (
                "keeps_render_whole_texels",
                c.shadow_keeps.render_whole_texels,
            ),
            ("keeps_compute_cut", c.shadow_keeps.compute_cut),
            (
                "keeps_compute_cut_texels",
                c.shadow_keeps.compute_cut_texels,
            ),
            ("keeps_compute_whole", c.shadow_keeps.compute_whole),
            (
                "keeps_compute_whole_texels",
                c.shadow_keeps.compute_whole_texels,
            ),
            ("keeps_blit_cut", c.shadow_keeps.blit_cut),
            ("keeps_blit_cut_texels", c.shadow_keeps.blit_cut_texels),
            ("keeps_blit_whole", c.shadow_keeps.blit_whole),
            ("keeps_blit_whole_texels", c.shadow_keeps.blit_whole_texels),
        ]
        .into_iter()
        .map(|(name, count)| (name, count as f64))
        .collect()
    }
}
