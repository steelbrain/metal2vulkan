//! The page of `replayed_frames.rs` as a page's bind groups draw it: each command the GPU writes
//! sets its own fragment buffer, an argument buffer whose entry names a texture of its own, and
//! the executing pass makes many buffers and every command's texture resident.
//!
//! The owned fixture `icb_vertex_encode_textured`: a pass with rasterization off draws one point
//! per instance [`DRAWS`] times, setting `ivt_encode`'s state before each; the draw of instance
//! `i` encodes command `i`, a draw of 6 16-bit indices of primitive type `i % 5` whose fragment
//! buffer 0 is the argument buffer slot `i` of the slots buffer points at, its entry naming
//! texture `i`. A second pass makes [`BUFFERS`] buffers resident (the commands' argument buffers,
//! the rest buffers no command binds), and the [`DRAWS`] textures, then executes the commands,
//! which inherit its state (`ivt_vertex`, `ivt_fragment`): the GPU may write the records, so a
//! host step records the execution from them, finding each command's fragment buffer by its
//! address among the resident ones. [`replayed_textured_frames`] prints the frames as
//! `replayed_frames` does (`REPLAYED-FRAMES` lines); the step's time should grow with the
//! records, not with them times the resident resources. [`DRAWS_VAR`] and [`BUFFERS_VAR`] set
//! other counts. It is ignored by default: run it by name with `--run-ignored all`.

use super::replayed_frames::{function, indices, measured_device, time_frames, SIDE};
use super::{buffer_with, read_buffer, Messages, ScratchCache};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandQueue, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, Origin, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The variable that sets the commands the pass encodes, each with a texture of its own.
const DRAWS_VAR: &str = "REPLAYED_TEXTURED_DRAWS";

/// The variable that sets the buffers the execution's pass makes resident, the commands'
/// argument buffers among them (at least one a command).
const BUFFERS_VAR: &str = "REPLAYED_TEXTURED_BUFFERS";

/// The commands the pass encodes by default.
const DRAWS: usize = 5001;

/// The buffers the execution's pass makes resident by default: three a command and a few more.
const BUFFERS: usize = 15018;

/// The fixture's states and resources on a device.
struct Page {
    draws: usize,
    encode: RenderPipelineState,
    draw: RenderPipelineState,
    icb: IndirectCommandBuffer,
    table: Buffer,
    indices: Buffer,
    /// The address of each command's argument buffer.
    slots: Buffer,
    /// The buffers the execution's pass makes resident: the commands' argument buffers, one a
    /// command, then buffers no command binds.
    buffers: Vec<Buffer>,
    /// Each command's texture, which its argument buffer's entry names.
    textures: Vec<Texture>,
    target: Texture,
    surface: Texture,
    pixels: Buffer,
}

/// A count [`DRAWS_VAR`] or [`BUFFERS_VAR`] sets, `default` where it is unset.
fn count(var: &str, default: usize) -> usize {
    std::env::var(var).map_or(default, |count| count.trim().parse().expect(var))
}

impl Page {
    fn new(device: &Device, draws: usize, buffers: usize) -> Page {
        let bitcode = llvm_assemble(&engine_dir().join("icb_vertex_encode_textured.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let rgba = PixelFormat::RGBA8Unorm;
        let options = PipelineOptions::default();
        let mut descriptor = RenderPipelineDescriptor::new(function(&library, "ivt_encode"));
        descriptor.rasterization_enabled = false;
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(rgba));
        let encode = device
            .new_render_pipeline_state(&descriptor, &options)
            .unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(function(&library, "ivt_vertex"));
        descriptor.fragment = Some(function(&library, "ivt_fragment"));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(rgba));
        descriptor.support_indirect_command_buffers = true;
        let draw = device
            .new_render_pipeline_state(&descriptor, &options)
            .unwrap();
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW_INDEXED;
        commands.inherit_pipeline_state = true;
        commands.inherit_buffers = false;
        commands.max_fragment_buffer_bind_count = 1;
        let icb = (device.new_indirect_command_buffer(
            &commands,
            draws as u64,
            ResourceOptions::shared(),
        ))
        .unwrap();
        let table = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let textures: Vec<_> = (0..draws)
            .map(|at| {
                let mut descriptor = TextureDescriptor::texture_2d(rgba, 4, 4, false);
                descriptor.usage = TextureUsage::SHADER_READ;
                descriptor.resource_options = ResourceOptions::shared();
                let texture = device.new_texture(&descriptor).unwrap();
                let texel = [(at % 251) as u8, (at / 251 % 251) as u8, 0x80, 0xff];
                let texels = texel.repeat(16);
                (texture.replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &texels, 16, 0)).unwrap();
                texture
            })
            .collect();
        let entries = textures.iter().map(|texture| {
            buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()).unwrap()
        });
        let unbound = (draws..buffers.max(draws))
            .map(|_| device.new_buffer(256, ResourceOptions::shared()).unwrap());
        let buffers: Vec<_> = entries.chain(unbound).collect();
        let slots = (buffers[..draws].iter()).flat_map(|entry| entry.gpu_address().0.to_le_bytes());
        let slots = buffer_with(device, &slots.collect::<Vec<_>>()).unwrap();
        let texture = |usage: TextureUsage| {
            let mut descriptor = TextureDescriptor::texture_2d(rgba, SIDE, SIDE, false);
            descriptor.usage = usage;
            descriptor.resource_options = ResourceOptions::shared();
            device.new_texture(&descriptor).unwrap()
        };
        Page {
            draws,
            encode,
            draw,
            icb,
            table,
            indices: buffer_with(device, &indices(37)).unwrap(),
            slots,
            buffers,
            textures,
            target: texture(TextureUsage::RENDER_TARGET),
            surface: texture(TextureUsage::SHADER_READ),
            pixels: device
                .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
                .unwrap(),
        }
    }

    fn pass(&self) -> RenderPassDescriptor {
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        pass
    }

    /// One frame on `queue`, committed: the encoding pass, then the execution's, then a copy of
    /// the target's pixels.
    fn frame(&self, queue: &CommandQueue) -> CommandBuffer {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&self.pass()).unwrap();
        let vertex = RenderStages::VERTEX;
        let uses = [
            (self.icb.as_resource(), ResourceUsage::WRITE),
            (self.table.as_resource(), ResourceUsage::READ),
            (self.indices.as_resource(), ResourceUsage::READ),
            (self.slots.as_resource(), ResourceUsage::READ),
            (self.surface.as_resource(), ResourceUsage::READ),
        ];
        for (resource, usage) in uses {
            encoder.use_resource(resource, usage, vertex).unwrap();
        }
        encoder.set_vertex_buffer(Some(&self.table), 0, 0).unwrap();
        encoder
            .set_vertex_buffer(Some(&self.indices), 0, 1)
            .unwrap();
        encoder.set_vertex_buffer(Some(&self.slots), 0, 2).unwrap();
        for draw in 0..self.draws as u64 {
            encoder.set_render_pipeline_state(&self.encode).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Point, 0, 1, 1, draw)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&self.pass()).unwrap();
        encoder.set_render_pipeline_state(&self.draw).unwrap();
        let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        let buffers = self.buffers.iter().map(Buffer::as_resource);
        for resource in buffers.chain(self.textures.iter().map(Texture::as_resource)) {
            encoder
                .use_resource(resource, ResourceUsage::READ, stages)
                .unwrap();
        }
        encoder
            .execute_commands_in_buffer(&self.icb, 0..self.draws as u64)
            .unwrap();
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let size = Size::new(SIDE, SIDE, 1);
        let (row, image) = (SIDE * 4, SIDE * SIDE * 4);
        let origin = Origin::new(0, 0, 0);
        let none = BlitOption::None;
        (blit.copy_from_texture_to_buffer(
            &self.target,
            0,
            0,
            origin,
            size,
            &self.pixels,
            0,
            row,
            image,
            none,
        ))
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn replayed_textured_frames() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = measured_device(&messages, &cache);
    let page = Page::new(
        &device,
        count(DRAWS_VAR, DRAWS),
        count(BUFFERS_VAR, BUFFERS),
    );
    let pixels = || read_buffer(&page.pixels, 0, SIDE * SIDE * 4).unwrap();
    let frame = |queue: &CommandQueue, _| page.frame(queue);
    time_frames(&device, &page.surface, None, frame, pixels);
}
