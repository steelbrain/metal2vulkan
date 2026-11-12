//! What a device's spare command pools keep of rate-mapped passes' recordings: a GPU hazard
//! (`m2v_gpu_hazards`), as a driver whose command buffers take their state from a fixed
//! device-wide pool fails once the recordings it holds fill it.
//!
//! A rate-mapped pass records each draw once per region of the map (`rate_mapped_pass.rs`), so a
//! pass of an indirect command buffer's draws a host step replays over a map of many regions
//! records hundreds of thousands of draws, each with its own viewport and scissor. ANV (Mesa
//! 26.2.6, an Iris Xe) takes each such draw's viewport and scissor state from its device's
//! dynamic state pool of 1 GiB, and a command buffer holds what its recording took until it is
//! begun again, reset or freed. Here [`LIGHT_POOLS`] light command buffers at once leave as many
//! spare command pools; then each round commits [`HEAVY_AT_ONCE`] heavy command buffers at once,
//! each a pass of [`EXECUTIONS`] executions of the commands over a map of [`ZONES`] zones a side
//! alternating between full and half quality, and waits for them. A heavy command buffer takes
//! the spare holding the fewest command buffers, so each round's go to pools no heavy one used.
//! Every round must complete without an error, after [`ROUNDS`] rounds.

use super::{buffer_with, open_device, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{RateMapDescriptor, RateMapLayerDescriptor};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceOptions,
    EventRef, Heap, HeapDescriptor, HeapType, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, Origin, PipelineOptions, RasterizationRateMap,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, SharedEvent, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use std::time::Instant;

/// The textures the draws read, and the draws: draw `j` is command `5 * j` (`ivt_encode`).
const TEXTURES: u64 = 8;

/// The commands each execution runs.
const COMMANDS: u64 = 5 * TEXTURES;

/// The command buffers committed at once before the rounds, each leaving a spare pool.
const LIGHT_POOLS: usize = 48;

/// The heavy command buffers each round commits at once.
const HEAVY_AT_ONCE: usize = 2;

/// The rounds.
const ROUNDS: usize = 16;

/// The executions of the commands in a heavy command buffer's rate-mapped pass.
const EXECUTIONS: usize = 96;

/// The map's zones on each axis, alternating between quality 1 and 0.5.
const ZONES: usize = 32;

/// The map's screen side, in pixels: 64 a zone.
const SCREEN: u32 = 64 * ZONES as u32;

/// What the program made.
struct Page {
    encode: RenderPipelineState,
    draw: RenderPipelineState,
    icb: IndirectCommandBuffer,
    table: Buffer,
    indices: Buffer,
    slots: Buffer,
    entries: Vec<Buffer>,
    heap: Heap,
    _textures: Vec<Texture>,
    queue: CommandQueue,
    small: Buffer,
}

impl Page {
    fn new(device: &Device) -> Page {
        let bitcode = llvm_assemble(&engine_dir().join("icb_vertex_encode_textured.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
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
            .new_indirect_command_buffer(&commands, COMMANDS, ResourceOptions::shared())
            .unwrap();
        let table = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let mut texture = TextureDescriptor::texture_2d(rgba, 1, 1, false);
        texture.usage = TextureUsage::SHADER_READ;
        texture.resource_options = ResourceOptions::private();
        let answer = device.heap_texture_size_and_align(&texture).unwrap();
        let bytes = answer.size.next_multiple_of(answer.align) * (TEXTURES + 1);
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                bytes,
                ResourceOptions::private(),
            ))
            .unwrap();
        let textures: Vec<Texture> = (0..TEXTURES)
            .map(|_| heap.new_texture(&texture).unwrap())
            .collect();
        let colors = vec![200u8; 4 * TEXTURES as usize];
        let source = buffer_with(device, &colors).unwrap();
        let filling = queue.command_buffer().unwrap();
        let mut blit = filling.blit_command_encoder().unwrap();
        for (j, texture) in (0..).zip(&textures) {
            blit.copy_from_buffer_to_texture(
                &source,
                4 * j,
                4,
                0,
                Size::new(1, 1, 1),
                texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
        }
        blit.end_encoding().unwrap();
        filling.commit().unwrap();
        completed(&filling);
        let entries: Vec<Buffer> = (textures.iter())
            .map(|texture| buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()).unwrap())
            .collect();
        let mut slots = vec![0u8; 8 * COMMANDS as usize];
        let mut indices = vec![0u16; 6 * 64];
        for (j, entry) in (0u64..).zip(&entries) {
            let command = 5 * j as usize;
            slots[8 * command..8 * command + 8]
                .copy_from_slice(&entry.gpu_address().0.to_le_bytes());
            for point in 0..6u64 {
                indices[6 * command + point as usize] = (6 * j + point) as u16;
            }
        }
        let indices: Vec<u8> = indices.into_iter().flat_map(u16::to_le_bytes).collect();
        Page {
            encode,
            draw,
            icb,
            table,
            indices: buffer_with(device, &indices).unwrap(),
            slots: buffer_with(device, &slots).unwrap(),
            entries,
            heap,
            _textures: textures,
            queue,
            small: buffer_with(device, &[0u8; 64]).unwrap(),
        }
    }

    /// A light command buffer, committed: it waits for `event` to reach 1, then fills a few bytes.
    fn light(&self, event: &SharedEvent) -> CommandBuffer {
        let command_buffer = self.queue.command_buffer().unwrap();
        (command_buffer.encode_wait_for_event(EventRef::Shared(event), 1)).unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&self.small, 0..64, 1).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    }

    /// A heavy command buffer, committed: a pass writes the commands, then a pass over `rate`
    /// executes them [`EXECUTIONS`] times into `target`.
    fn heavy(&self, target: &Texture, rate: &RasterizationRateMap) -> CommandBuffer {
        let command_buffer = self.queue.command_buffer().unwrap();
        let pass = |rate: Option<&RasterizationRateMap>| {
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.store_action = StoreAction::Store;
            pass.rasterization_rate_map = rate.cloned();
            pass
        };
        let mut encoder = command_buffer.render_command_encoder(&pass(None)).unwrap();
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
        (encoder.set_vertex_buffer(Some(&self.indices), 0, 1)).unwrap();
        encoder.set_vertex_buffer(Some(&self.slots), 0, 2).unwrap();
        encoder.set_render_pipeline_state(&self.encode).unwrap();
        for j in 0..TEXTURES as u32 {
            (encoder.draw_primitives(PrimitiveType::Point, 0, 1, 1, 5 * j)).unwrap();
        }
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer
            .render_command_encoder(&pass(Some(rate)))
            .unwrap();
        encoder.set_render_pipeline_state(&self.draw).unwrap();
        let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        encoder.use_heap(&self.heap, stages).unwrap();
        for entry in &self.entries {
            (encoder.use_resource(entry.as_resource(), ResourceUsage::READ, stages)).unwrap();
        }
        for _ in 0..EXECUTIONS {
            (encoder.execute_commands_in_buffer(&self.icb, 0..COMMANDS)).unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    }
}

/// Wait for `command_buffer`, which must complete without an error.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The map: [`ZONES`] zones a side, alternating between quality 1 and 0.5.
fn map(device: &Device) -> RasterizationRateMap {
    let qualities: Vec<f32> = (0..ZONES)
        .map(|zone| if zone % 2 == 0 { 1.0 } else { 0.5 })
        .collect();
    let mut descriptor = RateMapDescriptor::new([SCREEN, SCREEN]);
    descriptor.layers = vec![RateMapLayerDescriptor::new(qualities.clone(), qualities)];
    device.new_rasterization_rate_map(&descriptor).unwrap()
}

/// The spares' and draws' counts a round reports.
fn line(counts: &WorkCounts) -> String {
    format!(
        "draw_indexed {} set_viewport {} taken {} freed past recent uses {} past held {} shed {} \
         host steps {}",
        counts.commands_recorded.cmd_draw_indexed,
        counts.commands_recorded.cmd_set_viewport,
        counts.spare_command_pools_taken,
        counts.spare_command_buffers_freed_past_recent_uses,
        counts.spare_command_buffers_freed_past_held,
        counts.spare_command_buffers_shed,
        counts.host_steps_recorded,
    )
}

#[test]
fn rate_mapped_recordings_in_spare_pools_leave_the_drivers_pools_room() {
    let device = open_device(TEST_DEVICE, DeviceOptions::default()).unwrap();
    let page = Page::new(&device);
    let map = map(&device);
    let size = map.physical_size_for_layer(0);
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let event = device.new_shared_event().unwrap();
    let lights: Vec<CommandBuffer> = (0..LIGHT_POOLS).map(|_| page.light(&event)).collect();
    event.set_signaled_value(1);
    lights.iter().for_each(completed);
    drop(lights);
    let start = Instant::now();
    eprintln!(
        "RM-VOLUME physical {}x{}, before the rounds: {}",
        size.width,
        size.height,
        line(&device.work_counts())
    );
    for round in 0..ROUNDS {
        let heavy: Vec<CommandBuffer> = (0..HEAVY_AT_ONCE)
            .map(|_| page.heavy(&target, &map))
            .collect();
        heavy.iter().for_each(completed);
        drop(heavy);
        eprintln!(
            "RM-VOLUME round {round} at {:.1} s: {}",
            start.elapsed().as_secs_f64(),
            line(&device.work_counts())
        );
    }
}
