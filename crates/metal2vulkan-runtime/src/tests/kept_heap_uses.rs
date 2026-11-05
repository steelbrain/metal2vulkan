//! A host step's run of kept chunks of draws that read textures through the heap reports the uses
//! their recording noted (`EncoderCore::run_kept`, `render/executions/kept.rs`), and a step that
//! records the chunks apart those its draws noted, the uses of what the encoder made resident
//! among them, which the pass's core notes once for the chunks (`encoder::ResidentPart`): the
//! batches the scheduler plans for either step hold the same tracked uses, by resource, range,
//! reads and writes (their stages and accesses), as those of the step that recorded the commands
//! inline, so the tracker orders a kept run after earlier writers as it orders a fresh recording
//! (`CommandBuffer::watch_planned`, a device test's).
//!
//! The page is the validation lane `kept_textured`'s at two chunks' commands (the owned fixture
//! `icb_vertex_encode_textured`): a pass with rasterization off encodes one indexed draw a
//! command, whose fragment buffer is an argument buffer naming a texture of its own; a second pass
//! makes the argument buffers and the textures resident and executes the commands, which the GPU
//! wrote, so a host step records them. Frame 0 records inline, frame 1 records apart and keeps
//! both chunks, frame 2 runs them kept. On a device without
//! `descriptorBindingUpdateUnusedWhilePending` no frame keeps a chunk, and the frames' uses are
//! compared all the same.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::heap_sets::unused_while_pending;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

/// The commands the page encodes and executes: two chunks' (`kept_commands::CHUNK` is 256).
const COMMANDS: usize = 512;

/// The page's states and resources.
struct Page {
    encode: RenderPipelineState,
    draw: RenderPipelineState,
    icb: IndirectCommandBuffer,
    table: Buffer,
    indices: Buffer,
    slots: Buffer,
    entries: Vec<Buffer>,
    textures: Vec<Texture>,
    target: Texture,
}

/// A Shared buffer holding `bytes`.
fn buffer_with(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds `bytes.len()` bytes at `contents`, and no command uses it yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

impl Page {
    fn new(device: &Device) -> Page {
        let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../metal2vulkan-validation/fixtures/public/engine/icb_vertex_encode_textured.ll",
        );
        let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
        let library = device.new_library_with_compile_library(
            metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
        );
        let function = |name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            StageSource::Function(function.function_descriptor())
        };
        let rgba = PixelFormat::RGBA8Unorm;
        let options = PipelineOptions::default();
        let mut descriptor = RenderPipelineDescriptor::new(function("ivt_encode"));
        descriptor.rasterization_enabled = false;
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(rgba));
        let encode = device
            .new_render_pipeline_state(&descriptor, &options)
            .unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(function("ivt_vertex"));
        descriptor.fragment = Some(function("ivt_fragment"));
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
        let icb = device
            .new_indirect_command_buffer(&commands, COMMANDS as u64, ResourceOptions::shared())
            .unwrap();
        let table = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes());
        let textures: Vec<_> = (0..COMMANDS)
            .map(|at| {
                let mut descriptor = TextureDescriptor::texture_2d(rgba, 4, 4, false);
                descriptor.usage = TextureUsage::SHADER_READ;
                descriptor.resource_options = ResourceOptions::shared();
                let texture = device.new_texture(&descriptor).unwrap();
                let texels = [(at % 251) as u8, (at / 251) as u8, 0x80, 0xff].repeat(16);
                (texture.replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &texels, 16, 0)).unwrap();
                texture
            })
            .collect();
        let entries: Vec<_> = (textures.iter())
            .map(|texture| buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()))
            .collect();
        let slots: Vec<u8> = (entries.iter())
            .flat_map(|entry| entry.gpu_address().0.to_le_bytes())
            .collect();
        // Six indices a command, 64 commands' apart, then the zero half the fixture reads past.
        let indices: Vec<u8> = ((0..384u16).map(|at| (at * 37 + 11) % 64))
            .chain([0; 384])
            .flat_map(u16::to_le_bytes)
            .collect();
        let mut descriptor = TextureDescriptor::texture_2d(rgba, 64, 64, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        Page {
            encode,
            draw,
            icb,
            table,
            indices: buffer_with(device, &indices),
            slots: buffer_with(device, &slots),
            entries,
            textures,
            target: device.new_texture(&descriptor).unwrap(),
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

    /// One frame on `queue`, its planned batches watched, committed: the encoding pass, then the
    /// execution's.
    fn frame(&self, queue: &CommandQueue) -> CommandBuffer {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&self.pass()).unwrap();
        let uses = [
            (self.icb.as_resource(), ResourceUsage::WRITE),
            (self.table.as_resource(), ResourceUsage::READ),
            (self.indices.as_resource(), ResourceUsage::READ),
            (self.slots.as_resource(), ResourceUsage::READ),
        ];
        for (resource, usage) in uses {
            (encoder.use_resource(resource, usage, RenderStages::VERTEX)).unwrap();
        }
        encoder.set_vertex_buffer(Some(&self.table), 0, 0).unwrap();
        encoder
            .set_vertex_buffer(Some(&self.indices), 0, 1)
            .unwrap();
        encoder.set_vertex_buffer(Some(&self.slots), 0, 2).unwrap();
        for draw in 0..COMMANDS as u32 {
            encoder.set_render_pipeline_state(&self.encode).unwrap();
            (encoder.draw_primitives(PrimitiveType::Point, 0, 1, 1, draw)).unwrap();
        }
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&self.pass()).unwrap();
        encoder.set_render_pipeline_state(&self.draw).unwrap();
        let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        let entries = self.entries.iter().map(Buffer::as_resource);
        for resource in entries.chain(self.textures.iter().map(Texture::as_resource)) {
            (encoder.use_resource(resource, ResourceUsage::READ, stages)).unwrap();
        }
        (encoder.execute_commands_in_buffer(&self.icb, 0..COMMANDS as u64)).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.watch_planned();
        command_buffer.commit().unwrap();
        command_buffer
    }
}

/// The tracked uses of a frame's planned batches, each once, as text: resource, range, reads
/// and writes.
fn planned_uses(command_buffer: &CommandBuffer) -> BTreeSet<String> {
    let planned = command_buffer.planned();
    (planned.iter())
        .flat_map(|footprint| footprint.uses().iter().map(|used| format!("{used:?}")))
        .collect()
}

#[test]
fn a_kept_heap_chunk_s_run_reports_the_uses_an_inline_recording_reports() {
    let scratch = Scratch::new("kept-heap-uses");
    let device = device(&scratch.0);
    let shared = unused_while_pending(&device);
    let queue = device.new_command_queue().unwrap();
    let page = Page::new(&device);
    let mut frames = Vec::new();
    for frame in 0..3 {
        let before = device.work_counts();
        let command_buffer = page.frame(&queue);
        command_buffer
            .wait_until_completed_for(Duration::from_secs(60))
            .unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let after = device.work_counts();
        let kept = after.host_step_chunks_kept - before.host_step_chunks_kept;
        let apart = after.kept_contexts_made - before.kept_contexts_made;
        let expected = match (shared, frame) {
            (true, 1) => (0, 2),
            (true, 2) => (2, 0),
            _ => (0, 0),
        };
        assert_eq!(
            (kept, apart),
            expected,
            "frame {frame}: chunks kept, chunks kept apart"
        );
        frames.push(planned_uses(&command_buffer));
    }
    // The resident textures among them, which only the execution's draws reach.
    let textures = (page.textures.iter())
        .filter(|texture| {
            let key = format!("key: {:?},", texture.use_key());
            frames[0].iter().any(|used| used.contains(&key))
        })
        .count();
    assert_eq!(
        textures, COMMANDS,
        "the inline frame's uses name every texture"
    );
    for (at, frame) in frames.iter().enumerate().skip(1) {
        assert!(
            *frame == frames[0],
            "frame {at}'s planned uses differ from the inline frame's: {} only in it, {} only inline",
            frame.difference(&frames[0]).count(),
            frames[0].difference(frame).count()
        );
    }
}
