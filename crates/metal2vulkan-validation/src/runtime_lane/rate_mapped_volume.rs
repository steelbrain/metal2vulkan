//! What a device's spare command pools keep of rate-mapped passes' recordings: a GPU hazard
//! (`m2v_gpu_hazards`), as a driver whose command buffers take their state from a fixed
//! device-wide pool fails once the recordings it holds fill it.
//!
//! A rate-mapped pass records each draw once per region of the map (`rate_mapped_pass.rs`), so a
//! pass of an indirect command buffer's draws a host step replays over a map of many regions
//! records hundreds of thousands of draws, each with its own viewport and scissor. ANV (Mesa
//! 26.2.6, an Iris Xe) takes each such draw's viewport and scissor state from its device's
//! dynamic state pool of 1 GiB, and a command buffer holds what its recording took until it is
//! begun again, reset or freed. Here [`LIGHT_POOLS`] light command buffers, polled to
//! completion, leave as many spare command pools, each holding a primary begun ahead; then each
//! round commits [`HEAVY_AT_ONCE`] heavy command buffers at once, each a pass of executions of the
//! commands over a map of [`ZONES`] zones a side alternating between full and half quality, as
//! many as record about [`REGION_DRAWS`] draws, and waits for them. Every round
//! must complete without an error, after [`ROUNDS`] rounds.
//!
//! Printed (`RM-VOLUME`), per round: the spares' and draws' counts, and the process's DRM memory
//! where the kernel reports it ([`drm_total`]), which holds the buffer objects of the driver's
//! state pools and batches as they grew: ANV's state pools never shrink. On an Iris Xe (Mesa
//! 26.2.6) it stayed within 534 MiB, each heavy command buffer's pools freeing their command
//! buffers as it gave them back (the spares' `RECORDED`). Where the spares kept them, each
//! round's heavy command buffers took the pools of the round before, which held those
//! recordings until then, and it reached 751 MiB.

use super::{buffer_with, open_device, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{
    RasterizationRateMap as CompiledMap, RateMapDescriptor, RateMapLayerDescriptor, RegionLimits,
};
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
use std::time::{Duration, Instant};

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

/// The draws a heavy command buffer's rate-mapped pass records, about: its executions' draws,
/// each once per region of the map.
const REGION_DRAWS: usize = 800_000;

/// The map's zones on each axis, alternating between quality 1 and 0.5: Metal's most samples an
/// axis.
const ZONES: usize = 9;

/// The map's screen side, in pixels.
const SCREEN: u32 = 2048;

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
    /// executes them `executions` times into `target`.
    fn heavy(
        &self,
        target: &Texture,
        rate: &RasterizationRateMap,
        executions: usize,
    ) -> CommandBuffer {
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
        for _ in 0..executions {
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

/// The map's descriptor: [`ZONES`] zones a side, alternating between quality 1 and 0.5.
fn descriptor() -> RateMapDescriptor {
    let qualities: Vec<f32> = (0..ZONES)
        .map(|zone| if zone % 2 == 0 { 1.0 } else { 0.5 })
        .collect();
    let mut descriptor = RateMapDescriptor::new([SCREEN, SCREEN]);
    descriptor.layers = vec![RateMapLayerDescriptor::new(qualities.clone(), qualities)];
    descriptor
}

/// The regions a draw over the whole screen of the map records, by the compile layer's plan.
fn regions() -> usize {
    let limits = RegionLimits::new([16384, 16384], [-32768.0, 32767.0]);
    let whole = [0.0, 0.0, f64::from(SCREEN), f64::from(SCREEN)];
    let plan = CompiledMap::new(&descriptor()).unwrap();
    plan.regions(0, whole, None, &limits)
        .unwrap()
        .regions()
        .len()
}

/// The process's DRM memory, in MiB: `drm-total-*` of each DRM client its device files name
/// (`/proc/self/fdinfo`), summed; `None` where none is reported.
fn drm_total() -> Option<u64> {
    let mut clients = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir("/proc/self/fdinfo").ok()?.flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Some(client) = (text.lines()).find_map(|line| line.strip_prefix("drm-client-id:"))
        else {
            continue;
        };
        let mut kib = 0;
        for (key, value) in text.lines().filter_map(|line| line.split_once(':')) {
            let mut words = value.split_whitespace();
            let (Some(number), unit) = (words.next(), words.next()) else {
                continue;
            };
            let Ok(number) = number.parse::<u64>() else {
                continue;
            };
            if key.starts_with("drm-total-") {
                kib += match unit {
                    Some("GiB") => number << 20,
                    Some("MiB") => number << 10,
                    Some("KiB") => number,
                    _ => number >> 10,
                };
            }
        }
        clients.insert(client.trim().to_owned(), kib);
    }
    (!clients.is_empty()).then(|| clients.values().sum::<u64>() >> 10)
}

/// The spares' and draws' counts a round reports, and the process's DRM memory.
fn line(counts: &WorkCounts) -> String {
    let drm = drm_total().map_or("not reported".to_owned(), |mib| format!("{mib} MiB"));
    format!(
        "draw_indexed {} set_viewport {} taken {} freed past recent uses {} past held {} shed {} \
         host steps {} freed past recorded {} drm total {drm}",
        counts.commands_recorded.cmd_draw_indexed,
        counts.commands_recorded.cmd_set_viewport,
        counts.spare_command_pools_taken,
        counts.spare_command_buffers_freed_past_recent_uses,
        counts.spare_command_buffers_freed_past_held,
        counts.spare_command_buffers_shed,
        counts.host_steps_recorded,
        counts.spare_command_buffers_freed_past_recorded,
    )
}

#[test]
fn rate_mapped_recordings_in_spare_pools_leave_the_drivers_pools_room() {
    let device = open_device(TEST_DEVICE, DeviceOptions::default()).unwrap();
    let page = Page::new(&device);
    let map = device.new_rasterization_rate_map(&descriptor()).unwrap();
    let regions = regions();
    let executions = REGION_DRAWS.div_ceil(TEXTURES as usize * regions);
    let size = map.physical_size_for_layer(0);
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, size.width, size.height, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    let event = device.new_shared_event().unwrap();
    let lights: Vec<CommandBuffer> = (0..LIGHT_POOLS).map(|_| page.light(&event)).collect();
    event.set_signaled_value(1);
    let deadline = Instant::now() + WAIT;
    while (lights.iter()).any(|light| light.status() != CommandBufferStatus::Completed) {
        assert!(
            Instant::now() < deadline,
            "the light command buffers completed"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    // Their pools' primaries are begun ahead once their completed handlers ran.
    std::thread::sleep(Duration::from_millis(200));
    lights.iter().for_each(completed);
    drop(lights);
    let start = Instant::now();
    eprintln!(
        "RM-VOLUME physical {}x{}, {regions} regions, {executions} executions a heavy pass, \
         before the rounds: {}",
        size.width,
        size.height,
        line(&device.work_counts())
    );
    for round in 0..ROUNDS {
        let heavy: Vec<CommandBuffer> = (0..HEAVY_AT_ONCE)
            .map(|_| page.heavy(&target, &map, executions))
            .collect();
        std::thread::scope(|scope| {
            for command_buffer in &heavy {
                scope.spawn(move || completed(command_buffer));
            }
        });
        drop(heavy);
        eprintln!(
            "RM-VOLUME round {round} at {:.1} s: {}",
            start.elapsed().as_secs_f64(),
            line(&device.work_counts())
        );
    }
}
