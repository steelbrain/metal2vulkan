//! A page's frame whose indirect command buffer its own vertex function writes, on the device the
//! cfg selects, without the validation layer, while two other queues commit work: what the host
//! step of the execution costs, and how long the other queues' command buffers wait behind it.
//!
//! The owned fixture `icb_vertex_encode_indexed`: a pass with rasterization off draws one point
//! per instance [`DRAWS`] times, setting `ive_encode`'s state before each, as a page's validation
//! pass does; the draw of instance `i` encodes command `i`, a draw of 6 16-bit indices of
//! primitive type `i % 5` (`ive_encode` may call `draw_indexed_primitives` alone). The pass makes
//! the buffer resident for writing, the indices for reading, and a texture the presenter samples
//! for reading. A second pass binds a buffer for its vertex stage and executes the buffer's
//! [`DRAWS`] commands, which inherit its state (`ive_vertex`, `ive_fragment`) and buffers: the GPU
//! may write the records, so a host step records the execution from them.
//!
//! Meanwhile a presenter queue commits a blit that reads the texture, and a bystander queue one
//! that fills a buffer of its own, each one at a time, again as each completes, until the frame
//! completes. [`replayed_frames`] prints, per frame (the first cold, the rest warm), as
//! `REPLAYED-FRAMES` lines: the frame's time from commit to completion, the counters' moves that
//! split it (host steps and their time, pipelines made and their time, records read), and the
//! presenter's and the bystander's command buffers' times from commit to completion, largest and
//! count, and a hash of the target's pixels after the frame. The step keeps the chunks it records
//! from the third frame on (the runtime's `render/executions/kept.rs`): each warm frame from then
//! runs every chunk again, its records compared with those it was recorded from
//! (`WorkCounts::host_step_chunks_kept`, `host_step_records_compared`, `host_step_compare_nanos`).
//! [`CHANGE_VAR`] sets a frame from which the GPU writes other records (the encoding pass binds the
//! indices' zero half), which that frame's step records inline, as one that keeps nothing, and
//! the next one's records apart and keeps; [`EACH_VAR`] makes them change on each frame from it
//! (the two halves in turn), so each frame's step records inline. [`FRAMES_VAR`] sets the frames (4 by default), and
//! [`RESIDENT_VAR`] how many buffers
//! and textures, one of each, the execution's pass makes resident besides, as a page's bind
//! groups do (none by default): the step's time should grow with the records alone, not with
//! them times the resident resources. Each warm frame makes or finds few of its draws' bind
//! shapes, at most [`WARM_SHAPES`] (`WorkCounts::draw_shapes_found` and `draw_shapes_made`): the
//! replayed draws alternate primitive types, and each type's shape is one the state took lately
//! (`live_bind::RecentShapes`), found without listing the resident resources again. It is
//! ignored by default: run it by name with `--run-ignored all`.
//!
//! [`a_replayed_execution_holds_only_what_shares_its_resources`] checks what the frame holds of
//! other queues' work while its host step is not recorded: a later commit on another queue waits
//! behind it only where Metal orders the two, as one writes what the other uses.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device,
    validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    ColorWriteMask, IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, Library, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// The variable that sets the frames timed.
const FRAMES_VAR: &str = "REPLAYED_FRAMES_FRAMES";

/// The variable that sets how many buffers and textures the execution's pass makes resident
/// besides (none by default).
const RESIDENT_VAR: &str = "REPLAYED_FRAMES_RESIDENT";

/// The variable that sets the frame from which the encoding pass binds the indices' zero half
/// ([`ZEROS_AT`]), so the GPU writes other records from then on (none by default).
const CHANGE_VAR: &str = "REPLAYED_FRAMES_CHANGE_AT";

/// The variable that, set, makes the records change on each frame from [`CHANGE_VAR`]'s: the
/// encoding pass binds the indices' zero half and their first half in turn.
const EACH_VAR: &str = "REPLAYED_FRAMES_CHANGE_EACH";

/// The commands the pass encodes, one a draw, in [`replayed_frames`].
const DRAWS: u32 = 5000;

/// The commands the pass encodes in [`a_replayed_execution_holds_only_what_shares_its_resources`]:
/// enough that the host step takes longer than another queue's commit.
const CHECKED_DRAWS: u32 = 256;

/// The target's width and height.
pub(super) const SIDE: u64 = 64;

/// The most bind shapes a warm frame makes or finds among those its drawing states settled: a
/// few for each pass (the encoding pass's one state, the execution's five primitive types), not
/// one per replayed draw.
const WARM_SHAPES: u64 = 32;

/// The fixture's states and buffers on a device.
pub(super) struct Page {
    /// The commands the pass encodes and the second executes.
    draws: u32,
    encode: RenderPipelineState,
    draw: RenderPipelineState,
    icb: IndirectCommandBuffer,
    table: Buffer,
    indices: Buffer,
    /// The buffer the execution's pass binds for its vertex stage, which the commands inherit.
    inherited: Buffer,
    target: Texture,
    surface: Texture,
    /// The buffers and textures the execution's pass makes resident besides, as a page's bind
    /// groups, one of each a draw, do ([`RESIDENT_VAR`]).
    resident: Vec<(Buffer, Texture)>,
    /// What each frame copies the target's pixels into.
    pixels: Buffer,
}

/// What a frame takes in place of the page's own ([`Page::frame_with`]): the inputs a kept host
/// step's recording is keyed on, one at a time (`kept_steps.rs`). The page's own by default.
#[derive(Default)]
pub(super) struct Inputs<'a> {
    /// The buffer the encoding pass binds the indices from, and its offset there, which the
    /// records' index addresses follow.
    pub(super) indices: Option<(&'a Buffer, u64)>,
    /// The encoding pass leaves the indices' buffer unresident: an execution finds it among the
    /// device's live buffers.
    pub(super) indices_unresident: bool,
    /// The buffer and offset the execution's pass binds for its vertex stage, which the commands
    /// inherit.
    pub(super) inherited: Option<(&'a Buffer, u64)>,
    /// The state the execution's pass sets, which the commands inherit.
    pub(super) draw: Option<&'a RenderPipelineState>,
    /// A scissor rectangle the execution's pass sets before the call.
    pub(super) scissor: Option<ScissorRect>,
    /// A buffer the execution's pass makes resident besides.
    pub(super) resident: Option<&'a Buffer>,
}

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let bitcode = llvm_assemble(&engine_dir().join("icb_vertex_encode_indexed.ll")).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// `name` of `library`, as a stage's source.
pub(super) fn function(library: &Library, name: &str) -> StageSource {
    let function = library.new_function_with_name(name).unwrap();
    StageSource::Function(function.function_descriptor())
}

/// The state the commands draw with on `device` (`ive_vertex`, `ive_fragment`), its color written
/// under `mask`.
pub(super) fn draw_state(device: &Device, mask: ColorWriteMask) -> RenderPipelineState {
    let library = library(device);
    let mut descriptor = RenderPipelineDescriptor::new(function(&library, "ive_vertex"));
    descriptor.fragment = Some(function(&library, "ive_fragment"));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    color.write_mask = mask;
    descriptor.color_attachments[0] = Some(color);
    descriptor.support_indirect_command_buffers = true;
    (device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())).unwrap()
}

/// The indices of [`indices`] each half holds: 6 for each of 64 commands.
const HALF: usize = 64 * 6;

/// Where the zero half of [`indices`] starts, in bytes: a frame that binds the indices there
/// draws every command's primitives at vertex 0, so the GPU writes other records and the target
/// shows other pixels.
pub(super) const ZEROS_AT: u64 = (HALF * 2) as u64;

/// The fixture's indices: `scramble`'s, 6 for each of 64 commands, then as many zeros.
pub(super) fn indices(scramble: u16) -> Vec<u8> {
    let scrambled = (0..HALF as u16).map(|at| (at * scramble + 11) % 64);
    (scrambled.chain([0; HALF]))
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// A hash of `pixels` (FNV-1a), as the lanes print it.
pub(super) fn pixels_hash(pixels: &[u8]) -> u64 {
    pixels.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

impl Page {
    pub(super) fn new(device: &Device, draws: u32) -> Page {
        let library = library(device);
        let rgba = PixelFormat::RGBA8Unorm;
        let mut descriptor = RenderPipelineDescriptor::new(function(&library, "ive_encode"));
        descriptor.rasterization_enabled = false;
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(rgba));
        let options = PipelineOptions::default();
        let encode = device
            .new_render_pipeline_state(&descriptor, &options)
            .unwrap();
        let draw = draw_state(device, ColorWriteMask::ALL);
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW_INDEXED;
        commands.inherit_pipeline_state = true;
        commands.inherit_buffers = true;
        let icb = device
            .new_indirect_command_buffer(&commands, u64::from(draws), ResourceOptions::shared())
            .unwrap();
        let table = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let indices = buffer_with(device, &indices(37)).unwrap();
        let texture = |usage: TextureUsage| {
            let mut descriptor = TextureDescriptor::texture_2d(rgba, SIDE, SIDE, false);
            descriptor.usage = usage;
            descriptor.resource_options = ResourceOptions::shared();
            device.new_texture(&descriptor).unwrap()
        };
        let resident = std::env::var(RESIDENT_VAR).map_or(0, |count| count.parse().unwrap());
        let resident = (0..resident)
            .map(|_| {
                let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
                let mut descriptor = TextureDescriptor::texture_2d(rgba, 4, 4, false);
                descriptor.usage = TextureUsage::SHADER_READ;
                (buffer, device.new_texture(&descriptor).unwrap())
            })
            .collect();
        Page {
            draws,
            encode,
            draw,
            icb,
            table,
            indices,
            inherited: device.new_buffer(256, ResourceOptions::shared()).unwrap(),
            target: texture(TextureUsage::RENDER_TARGET),
            surface: texture(TextureUsage::SHADER_READ),
            resident,
            pixels: device
                .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
                .unwrap(),
        }
    }

    /// The buffer the execution's pass binds for its vertex stage, which the commands inherit.
    pub(super) fn inherited(&self) -> &Buffer {
        &self.inherited
    }

    /// The fixture's indices, which the encoding pass binds.
    pub(super) fn indices(&self) -> &Buffer {
        &self.indices
    }

    /// The target's pixels as the last frame left them.
    pub(super) fn pixels(&self) -> Vec<u8> {
        read_buffer(&self.pixels, 0, SIDE * SIDE * 4).unwrap()
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
        self.frame_with(queue, &Inputs::default())
    }

    /// [`frame`](Self::frame) with `inputs` in place of the page's own.
    pub(super) fn frame_with(&self, queue: &CommandQueue, inputs: &Inputs) -> CommandBuffer {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&self.pass()).unwrap();
        let vertex = RenderStages::VERTEX;
        let (indices, at) = inputs.indices.unwrap_or((&self.indices, 0));
        let uses = [
            (self.icb.as_resource(), ResourceUsage::WRITE),
            (self.table.as_resource(), ResourceUsage::READ),
            (self.surface.as_resource(), ResourceUsage::READ),
        ];
        let indices_use =
            (!inputs.indices_unresident).then(|| (indices.as_resource(), ResourceUsage::READ));
        for (resource, usage) in uses.into_iter().chain(indices_use) {
            encoder.use_resource(resource, usage, vertex).unwrap();
        }
        encoder.set_vertex_buffer(Some(&self.table), 0, 0).unwrap();
        encoder.set_vertex_buffer(Some(indices), at, 1).unwrap();
        for draw in 0..self.draws {
            encoder.set_render_pipeline_state(&self.encode).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Point, 0, 1, 1, draw)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&self.pass()).unwrap();
        let draw = inputs.draw.unwrap_or(&self.draw);
        encoder.set_render_pipeline_state(draw).unwrap();
        let (inherited, at) = inputs.inherited.unwrap_or((&self.inherited, 0));
        encoder.set_vertex_buffer(Some(inherited), at, 0).unwrap();
        if let Some(rect) = inputs.scissor {
            encoder.set_scissor_rect(rect).unwrap();
        }
        let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        let resident = self.resident.iter();
        let resident =
            resident.flat_map(|(buffer, texture)| [buffer.as_resource(), texture.as_resource()]);
        for resource in resident.chain(inputs.resident.map(Buffer::as_resource)) {
            encoder
                .use_resource(resource, ResourceUsage::READ, stages)
                .unwrap();
        }
        encoder
            .execute_commands_in_buffer(&self.icb, 0..u64::from(self.draws))
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

/// What another queue's command buffers took, one at a time from commit to completion, while
/// `done` is unset.
#[derive(Debug, Default)]
struct Waits {
    count: usize,
    largest: Duration,
    total: Duration,
}

/// Commit `encode`'s command buffers on `queue`, each after the last completed, until `done`.
fn commit_until(queue: &CommandQueue, done: &AtomicBool, encode: impl Fn(&CommandBuffer)) -> Waits {
    let mut waits = Waits::default();
    while !done.load(Ordering::Acquire) {
        let command_buffer = queue.command_buffer().unwrap();
        encode(&command_buffer);
        let start = Instant::now();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let took = start.elapsed();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        waits.count += 1;
        waits.total += took;
        waits.largest = waits.largest.max(took);
    }
    waits
}

/// The frames to time: 4 unless [`FRAMES_VAR`] sets another count.
fn frames() -> usize {
    std::env::var(FRAMES_VAR).map_or(4, |count| count.trim().parse().expect(FRAMES_VAR))
}

/// The frame from which the GPU writes other records, where [`CHANGE_VAR`] sets one.
fn change_at() -> Option<usize> {
    let set = std::env::var(CHANGE_VAR)
        .ok()
        .filter(|at| !at.trim().is_empty());
    set.map(|at| at.trim().parse().expect(CHANGE_VAR))
}

/// What `after` moved past `before` of the counters that split a frame.
fn split(before: &WorkCounts, after: &WorkCounts) -> String {
    let ms = |nanos: u64| nanos as f64 / 1e6;
    format!(
        "host_steps {} host_step_ms {:.1} render_pipelines_made_ms {:.1} \
         compute_pipelines_made_ms {:.1} variant_pipelines_made {} records_read {} \
         heads_held_for_host_steps {} heads_held_for_earlier_commits {} shapes_found_or_made {} \
         host_step_chunks_kept {} host_step_chunks_recorded_again {} \
         host_step_chunk_copies_put_off {} host_step_records_compared {} \
         host_step_compare_ms {:.2} indirect_record_bytes_copied {} scheduler_passes {} footprints_built {} footprint_lock_waits {} \
         root_table_ranges_noted {} resident_uses_made {} resident_uses_gathered {} \
         resident_address_indexes_made {} stage_tables_read {} draws_rebound {} draws_as_before {} \
         heap_tables_worked_out {} host_range_lookups {} argument_samplers_taken {} \
         draw_heap_parts_made {} resident_uses_noted {} resident_uses_against_fragments {} \
         resident_uses_unordered_writes {} resident_uses_indirect_reads {} \
         resident_uses_geometry_writes {} resident_uses_geometry_checked {} \
         resident_uses_again_checked {} draw_resident_textures_looked_at {} \
         replayed_heap_draws_unshared {} replayed_heap_draws_writing {} \
         replayed_heap_draws_noted {} replayed_heap_draws_over_sampler_tables {} \
         replayed_heap_draws_table_textures {}",
        after.host_steps_recorded - before.host_steps_recorded,
        ms(after.host_step_nanos - before.host_step_nanos),
        ms(after.render_pipeline_create_nanos - before.render_pipeline_create_nanos),
        ms(after.compute_pipeline_create_nanos - before.compute_pipeline_create_nanos),
        after.variant_pipelines_made - before.variant_pipelines_made,
        after.indirect_records_read - before.indirect_records_read,
        after.heads_held_for_host_steps - before.heads_held_for_host_steps,
        after.heads_held_for_earlier_commits - before.heads_held_for_earlier_commits,
        shapes(before, after),
        after.host_step_chunks_kept - before.host_step_chunks_kept,
        after.host_step_chunks_recorded_again - before.host_step_chunks_recorded_again,
        after.host_step_chunk_copies_put_off - before.host_step_chunk_copies_put_off,
        after.host_step_records_compared - before.host_step_records_compared,
        ms(after.host_step_compare_nanos - before.host_step_compare_nanos),
        after.indirect_record_bytes_copied - before.indirect_record_bytes_copied,
        after.scheduler_passes - before.scheduler_passes,
        after.footprints_built - before.footprints_built,
        after.footprint_lock_waits - before.footprint_lock_waits,
        after.root_table_ranges_noted - before.root_table_ranges_noted,
        after.resident_uses_made - before.resident_uses_made,
        after.resident_uses_gathered - before.resident_uses_gathered,
        after.resident_address_indexes_made - before.resident_address_indexes_made,
        after.stage_tables_read - before.stage_tables_read,
        after.draws_rebound - before.draws_rebound,
        after.draws_as_before - before.draws_as_before,
        after.heap_tables_worked_out - before.heap_tables_worked_out,
        after.host_range_lookups - before.host_range_lookups,
        after.argument_samplers_taken - before.argument_samplers_taken,
        after.draw_heap_parts_made - before.draw_heap_parts_made,
        after.resident_uses_noted - before.resident_uses_noted,
        after.resident_uses_against_fragments - before.resident_uses_against_fragments,
        after.resident_uses_unordered_writes - before.resident_uses_unordered_writes,
        after.resident_uses_indirect_reads - before.resident_uses_indirect_reads,
        after.resident_uses_geometry_writes - before.resident_uses_geometry_writes,
        after.resident_uses_geometry_checked - before.resident_uses_geometry_checked,
        after.resident_uses_again_checked - before.resident_uses_again_checked,
        after.draw_resident_textures_looked_at - before.draw_resident_textures_looked_at,
        after.replayed_heap_draws_unshared - before.replayed_heap_draws_unshared,
        after.replayed_heap_draws_writing - before.replayed_heap_draws_writing,
        after.replayed_heap_draws_noted - before.replayed_heap_draws_noted,
        after.replayed_heap_draws_over_sampler_tables
            - before.replayed_heap_draws_over_sampler_tables,
        after.replayed_heap_draws_table_textures - before.replayed_heap_draws_table_textures,
    )
}

/// The bind shapes draws made or found among those their drawing states settled between `before`
/// and `after`.
fn shapes(before: &WorkCounts, after: &WorkCounts) -> u64 {
    (after.draw_shapes_found + after.draw_shapes_made)
        - (before.draw_shapes_found + before.draw_shapes_made)
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn replayed_frames() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = measured_device(&messages, &cache);
    let page = Page::new(&device, DRAWS);
    let changed = Inputs {
        indices: Some((&page.indices, ZEROS_AT)),
        ..Inputs::default()
    };
    let change_at = change_at();
    let each = std::env::var_os(EACH_VAR).is_some();
    let frame = |queue: &CommandQueue, frame: usize| {
        let changes = change_at.filter(|&at| frame >= at);
        let inputs = match changes.is_some_and(|at| !each || (frame - at) % 2 == 0) {
            true => &changed,
            false => &Inputs::default(),
        };
        page.frame_with(queue, inputs)
    };
    let pixels = || page.pixels();
    time_frames(&device, &page.surface, Some(WARM_SHAPES), frame, pixels);
}

/// A device as a consumer's: without the validation layer, keeping no execution record.
pub(super) fn measured_device(messages: &Messages, cache: &ScratchCache) -> Device {
    validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap()
}

/// Times [`frames`] frames on `device`, each the command buffer `frame` commits on a queue of its
/// own (given the frame's index), waited for, while a presenter queue commits a blit that copies
/// out `surface` and a bystander queue one that fills a buffer of its own, each one at a time,
/// again as each completes, until the frame completes. Prints a `REPLAYED-FRAMES` line for each
/// (the module docs), `pixels`' hash after it last; each warm frame makes or finds at most
/// `warm_shapes` bind shapes, where set.
pub(super) fn time_frames(
    device: &Device,
    surface: &Texture,
    warm_shapes: Option<u64>,
    frame: impl Fn(&CommandQueue, usize) -> CommandBuffer,
    pixels: impl Fn() -> Vec<u8>,
) {
    let name = device.name().to_owned();
    let (page_queue, presenter, bystander) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let read = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let own = device
        .new_buffer(1 << 16, ResourceOptions::shared())
        .unwrap();
    for index in 0..frames() {
        let done = AtomicBool::new(false);
        let before = device.work_counts();
        let (took, presented, stood) = thread::scope(|scope| {
            let presented = scope.spawn(|| {
                commit_until(&presenter, &done, |command_buffer| {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    let size = Size::new(SIDE, SIDE, 1);
                    blit.copy_from_texture_to_buffer(
                        surface,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        size,
                        &read,
                        0,
                        SIDE * 4,
                        SIDE * SIDE * 4,
                        BlitOption::None,
                    )
                    .unwrap();
                    blit.end_encoding().unwrap();
                })
            });
            let stood = scope.spawn(|| {
                commit_until(&bystander, &done, |command_buffer| {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(&own, 0..1 << 16, 7).unwrap();
                    blit.end_encoding().unwrap();
                })
            });
            let start = Instant::now();
            let command_buffer = frame(&page_queue, index);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let took = start.elapsed();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            done.store(true, Ordering::Release);
            (took, presented.join().unwrap(), stood.join().unwrap())
        });
        let after = device.work_counts();
        println!(
            "REPLAYED-FRAMES {name} frame {index}: {:.1} ms (encoding to completion), {}; \
             presenter {} command buffers, largest {:.1} ms, mean {:.1} ms; bystander {}, \
             largest {:.1} ms, mean {:.1} ms; pixels {:016x}",
            took.as_secs_f64() * 1e3,
            split(&before, &after),
            presented.count,
            presented.largest.as_secs_f64() * 1e3,
            presented.total.as_secs_f64() * 1e3 / presented.count.max(1) as f64,
            stood.count,
            stood.largest.as_secs_f64() * 1e3,
            stood.total.as_secs_f64() * 1e3 / stood.count.max(1) as f64,
            pixels_hash(&pixels()),
        );
        let shapes = shapes(&before, &after);
        assert!(
            index == 0 || warm_shapes.is_none_or(|warm| shapes <= warm),
            "warm frame {index} on {name} made or found {shapes} bind shapes"
        );
    }
}

/// The encoders of a page's frame ([`Page::frame_with`]): the encoding pass, the executing
/// pass and the blit that copies out its pixels.
const FRAME_ENCODERS: u64 = 3;

/// While the page's frame is held for its host step (`WorkCounts::heads_held_for_host_steps`), a
/// bystander on another queue commits a command buffer at a time, again as each completes, until
/// the frame completes: each commit and completion makes a pass of the scheduler's, which compares
/// the bystander's head with the held frame by their footprints (`scheduler/order.rs`). Each
/// batch's footprint is built once, however many passes look at it, and again only where what it
/// is built from changed, as where the frame's host step records its batch
/// (`WorkCounts::footprints_built`): at most one a bystander's command buffer and two for each of
/// the frame's batches, which its [`FRAME_ENCODERS`] bound; a footprint built at each look builds
/// one a pass. The validation layer stays silent.
#[test]
fn a_batch_s_footprint_is_built_once_however_many_passes_look() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let page = Page::new(&device, CHECKED_DRAWS);
        let (page_queue, other) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let own = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        // A first frame makes the pipelines the next one records with.
        let first = page.frame(&page_queue);
        first.wait_until_completed_for(WAIT).unwrap();
        let done = AtomicBool::new(false);
        let before = device.work_counts();
        let stood = thread::scope(|scope| {
            let stood = scope.spawn(|| {
                commit_until(&other, &done, |command_buffer| {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(&own, 0..256, 7).unwrap();
                    blit.end_encoding().unwrap();
                })
            });
            let frame = page.frame(&page_queue);
            frame.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(frame.status(), CommandBufferStatus::Completed);
            done.store(true, Ordering::Release);
            stood.join().unwrap()
        });
        let after = device.work_counts();
        let held = after.heads_held_for_host_steps - before.heads_held_for_host_steps;
        let built = after.footprints_built - before.footprints_built;
        let passes = after.scheduler_passes - before.scheduler_passes;
        let committed = stood.count as u64;
        eprintln!(
            "FOOTPRINTS {name}: {built} built in {passes} passes, the frame held {held} times, \
             {committed} bystander command buffers"
        );
        assert!(
            held > 0,
            "on {name}: the frame was not held for its host step"
        );
        assert!(
            built <= committed + 2 * FRAME_ENCODERS,
            "on {name}: {built} footprints built for {committed} bystander command buffers and \
             the frame, in {passes} passes"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// What another queue's command buffer encodes in a case of
/// [`a_replayed_execution_holds_only_what_shares_its_resources`].
type Encode<'a> = dyn Fn(&CommandBuffer) + 'a;

/// While the page's frame is held for its host step (`WorkCounts::heads_held_for_host_steps`
/// moves), another queue's command buffer committed after it is held behind it
/// (`WorkCounts::heads_held_for_earlier_commits`) only where Metal orders the two: a bystander
/// that fills a buffer of its own, and a presenter that copies out the texture the frame only
/// reads, go at once; a presenter that fills the buffer the execution's commands inherit waits.
/// The validation layer stays silent.
#[test]
fn a_replayed_execution_holds_only_what_shares_its_resources() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let page = Page::new(&device, CHECKED_DRAWS);
        let (page_queue, other) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let read = device
            .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
            .unwrap();
        let own = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let fill = |buffer: &Buffer, command_buffer: &CommandBuffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(buffer, 0..256, 7).unwrap();
            blit.end_encoding().unwrap();
        };
        let copy_out = |command_buffer: &CommandBuffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let size = Size::new(SIDE, SIDE, 1);
            let (row, image) = (SIDE * 4, SIDE * SIDE * 4);
            let origin = Origin::new(0, 0, 0);
            let none = BlitOption::None;
            blit.copy_from_texture_to_buffer(
                &page.surface,
                0,
                0,
                origin,
                size,
                &read,
                0,
                row,
                image,
                none,
            )
            .unwrap();
            blit.end_encoding().unwrap();
        };
        let cases: [(&str, bool, &Encode); 3] = [
            (
                "a bystander that fills a buffer of its own",
                false,
                &|command_buffer| fill(&own, command_buffer),
            ),
            (
                "a presenter that copies out the texture the frame reads",
                false,
                &copy_out,
            ),
            (
                "a presenter that fills the buffer the commands inherit",
                true,
                &|command_buffer| fill(&page.inherited, command_buffer),
            ),
        ];
        for (case, shares, encode) in cases {
            let before = device.work_counts();
            let frame = page.frame(&page_queue);
            let command_buffer = other.command_buffer().unwrap();
            encode(&command_buffer);
            command_buffer.commit().unwrap();
            for command_buffer in [&frame, &command_buffer] {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(
                    command_buffer.status(),
                    CommandBufferStatus::Completed,
                    "{case}"
                );
            }
            let after = device.work_counts();
            let step = after.heads_held_for_host_steps - before.heads_held_for_host_steps;
            assert!(
                step > 0,
                "{case} on {name}: the frame was not held for its host step"
            );
            let held = after.heads_held_for_earlier_commits - before.heads_held_for_earlier_commits;
            assert_eq!(
                held > 0,
                shares,
                "{case} on {name}: {held} looks held it behind the frame"
            );
        }
    }
    layer_reported_nothing_at_all(&messages);
}
