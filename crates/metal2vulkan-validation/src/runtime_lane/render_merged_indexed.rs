//! Indexed executions of render commands a kernel writes, merged into one draw (the runtime's
//! `render/executions/gpu.rs`, `bind::DrawReplay::indexed`), on the device the cfg selects, under
//! the validation layer. `icb_merged_indexed`'s kernels encode command `tid` from `draws[tid]`
//! (first index, index count, instance count, base vertex, base instance) as an indexed triangle
//! draw of 16-bit or 32-bit indices, with its own mark as vertex buffer 0 and its own tint as
//! fragment buffer 0; its vertex function places vertex `vid` of instance `iid` at `xy[vid]` moved
//! down `iid` rows and carries the mark, `vid` and `iid` flat to its fragment function, which
//! returns them with its command's tint into a `RGBA32Uint` target. They draw on the GPU alone,
//! in one merged draw, each vertex its command's index plus its base vertex and each fragment its
//! own command's tint: the same pixels as the host step's, which a barrier the CPU set and
//! cleared sends them to. Executed again and again through the host step under a heap's textures
//! made resident, their recordings' keys list what the command buffer made resident once a
//! frame, however many executions take it.

use super::degrade_notices::{assert_made_once, counts, made_compute_kinds, on_device_with};
use super::indirect_host_steps::{complete, HOST_STEPS};
use super::render_executions::{forced_with, icb};
use super::{at_plan, buffer_with};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor,
    RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, ComputePipelineState, Device, HeapDescriptor, HeapType,
    IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};
use std::collections::BTreeSet;

/// The target's width and height, and the rows an instance moves its vertices down.
const W: u64 = 64;
const H: u64 = 48;
const ROW: f32 = 8.0;
/// The buffer's commands.
const COMMANDS: u64 = 6;

/// A command's draw: first index, index count, instance count, base vertex, base instance.
type Draw = [u32; 5];

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let bitcode = llvm_assemble(&engine_dir().join("icb_merged_indexed.ll")).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

fn function(library: &Library, name: &str) -> FunctionDescriptor {
    library
        .new_function_with_name(name)
        .unwrap()
        .function_descriptor()
}

/// The writer of `size`-bit indices (16 or 32).
fn writer(device: &Device, library: &Library, size: u32) -> ComputePipelineState {
    let kernel = function(library, &format!("icb_indexed_{size}"));
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The descriptor of the commands' state, with `supportIndirectCommandBuffers`.
fn state_descriptor(library: &Library) -> RenderPipelineDescriptor {
    let stage = |name: &str| StageSource::Function(function(library, name));
    let mut descriptor = RenderPipelineDescriptor::new(stage("icb_indexed_vertex"));
    descriptor.fragment = Some(stage("icb_indexed_fragment"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    descriptor.support_indirect_command_buffers = true;
    descriptor
}

/// Indexed draws of three vertex buffers and one fragment buffer that inherit the encoder's state.
fn inheriting() -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW_INDEXED;
    descriptor.inherit_pipeline_state = true;
    descriptor.max_vertex_buffer_bind_count = 3;
    descriptor.max_fragment_buffer_bind_count = 1;
    descriptor
}

fn words(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// Sixteen three-pixel triangles, triangle `k` of vertices `3k`, `3k + 1`, `3k + 2`, each at its
/// own place.
fn triangles() -> Vec<[f32; 2]> {
    (0..48u32)
        .map(|v| {
            let (k, corner) = (v / 3, v % 3);
            let x = 1.0 + 4.0 * (k % 15) as f32;
            let y = 1.0 + 4.0 * (k / 15) as f32;
            match corner {
                0 => [x, y],
                1 => [x + 3.0, y],
                _ => [x, y + 3.0],
            }
        })
        .collect()
}

/// The indices: position `p` names corner `p mod 3` of triangle `15 - p / 3`, the triangles in
/// reverse order.
fn indices() -> Vec<u32> {
    (0..48u32).map(|p| 3 * (15 - p / 3) + p % 3).collect()
}

/// The target: `RGBA32Uint`, `W` by `H`, `Shared`, a render target.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Uint, W, H, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// The target's pixels, row by row.
fn pixels(target: &Texture) -> Vec<[u32; 4]> {
    let mut bytes = vec![0u8; (16 * W * H) as usize];
    target
        .get_bytes(&mut bytes, 16 * W, 0, Region::new_2d(0, 0, W, H), 0, 0)
        .unwrap();
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|p| {
            std::array::from_fn(|i| u32::from_le_bytes(p[4 * i..4 * i + 4].try_into().unwrap()))
        })
        .collect()
}

/// Run `writer` writing `draws` with indices of `size` bits into a new buffer of [`COMMANDS`]
/// commands (a barrier the CPU set and cleared in its last command where `host_step`), then a
/// pass that executes all of them: the target's pixels, and the merged draws' commands, the GPU
/// form's and the records the host read across the run.
fn run(
    device: &Device,
    (writer, state): (&ComputePipelineState, &RenderPipelineState),
    (draws, size, host_step): (&[Draw], u32, bool),
) -> (Vec<[u32; 4]>, (u64, u64, u64)) {
    let icb = icb(device, &inheriting(), COMMANDS);
    if host_step {
        let command = icb.indirect_render_command_at(COMMANDS - 1).unwrap();
        command.set_barrier().unwrap();
        command.clear_barrier().unwrap();
    }
    let index_bytes: Vec<u8> = match size {
        16 => indices()
            .into_iter()
            .flat_map(|i| (i as u16).to_le_bytes())
            .collect(),
        _ => words(indices()),
    };
    let buffers: [Buffer; 7] = [
        buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap(),
        buffer_with(device, &words(draws.iter().flatten().copied())).unwrap(),
        buffer_with(device, &words((0..COMMANDS as u32).map(|c| 100 + c))).unwrap(),
        buffer_with(device, &floats([W as f32, H as f32, ROW, 0.0])).unwrap(),
        buffer_with(device, &floats(triangles().into_iter().flatten())).unwrap(),
        buffer_with(device, &words((0..COMMANDS as u32).map(|c| 200 + c))).unwrap(),
        buffer_with(device, &index_bytes).unwrap(),
    ];
    let target = target(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let before = device.work_counts();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .use_resource(icb.as_resource(), ResourceUsage::WRITE)
        .unwrap();
    encoder.set_compute_pipeline_state(writer).unwrap();
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(Some(buffer), 0, index as u32).unwrap();
    }
    let grid = Size::new(draws.len() as u64, 1, 1);
    encoder.dispatch_threads(grid, grid).unwrap();
    encoder.end_encoding().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    // The marks, frame, vertices and indices the vertex stage reads, the tints the fragment
    // stage reads.
    for (buffer, stages) in [
        (&buffers[2], RenderStages::VERTEX),
        (&buffers[3], RenderStages::VERTEX),
        (&buffers[4], RenderStages::VERTEX),
        (&buffers[6], RenderStages::VERTEX),
        (&buffers[5], RenderStages::FRAGMENT),
    ] {
        encoder
            .use_resource(buffer.as_resource(), ResourceUsage::READ, stages)
            .unwrap();
    }
    encoder
        .execute_commands_in_buffer(&icb, 0..COMMANDS)
        .unwrap();
    encoder.end_encoding().unwrap();
    complete(&command_buffer);
    let after = device.work_counts();
    let counted = (
        after.merged_draw_commands - before.merged_draw_commands,
        after.gpu_form_commands - before.gpu_form_commands,
        after.indirect_records_read - before.indirect_records_read,
    );
    (pixels(&target), counted)
}

/// Indexed triangles of commands a kernel writes, 16-bit and 32-bit indices alike, draw in one
/// merged draw, the host reading no record, what the host step draws of them: each vertex its
/// command's index plus its base vertex in each of its instances, a trailing index drawing
/// nothing, a command of no instance (or past what the kernel wrote) drawing nothing, a later
/// command over an earlier one's pixel winning, and each fragment its own command's tint.
#[test]
fn merged_indexed_draws_draw_what_the_host_step_draws() {
    on_device_with(at_plan, |device, messages| {
        let library = library(device);
        let state = device
            .new_render_pipeline_state(&state_descriptor(&library), &PipelineOptions::default())
            .unwrap();
        // Triangles 15; 14 and 13 in two instances; 12 and a trailing index; none; and 10 moved
        // to 11 by its base vertex: no two overlap.
        let draws: Vec<Draw> = vec![
            [0, 3, 1, 0, 0],
            [3, 6, 2, 0, 1],
            [9, 4, 1, 0, 0],
            [12, 3, 0, 0, 0],
            [15, 3, 1, 3, 0],
        ];
        let mut outputs = Vec::new();
        for host_step in [false, true] {
            let mut made = match host_step {
                false => forced_with(device, &[state_descriptor(&library)]),
                true => BTreeSet::new(),
            };
            for size in [16, 32] {
                let writer = writer(device, &library, size);
                let kernel = function(&library, &format!("icb_indexed_{size}"));
                made.extend(made_compute_kinds(device, kernel));
                let (drawn, (merged, gpu, read)) =
                    run(device, (&writer, &state), (&draws, size, host_step));
                match host_step {
                    false => assert_eq!(
                        (merged, gpu, read),
                        (COMMANDS, COMMANDS, 0),
                        "{size}: merged"
                    ),
                    true => assert!(
                        merged == 0 && gpu == 0 && read >= COMMANDS,
                        "{size}: {read}"
                    ),
                }
                outputs.push(drawn);
            }
            if !host_step {
                assert_made_once(messages, &[], &made);
            }
        }
        let host_notices = counts(&messages.take_notices());
        assert!(
            host_notices.contains_key(&HOST_STEPS[0]),
            "{host_notices:?}"
        );
        let (merged, host) = outputs.split_at(2);
        for (size, (merged, host)) in [16, 32].iter().zip(merged.iter().zip(host)) {
            let tints: BTreeSet<u32> = merged.iter().map(|p| p[3]).filter(|&t| t != 0).collect();
            assert_eq!(
                tints,
                BTreeSet::from([200, 201, 202, 204]),
                "{size}: each drawing command's own tint"
            );
            assert_eq!(
                merged, host,
                "{size}: the merged draw against the host step"
            );
        }
    });
}

/// The textures a heap of the device's holds, which the frames of
/// [`a_host_step_lists_its_residency_once_for_every_execution`] make resident.
const HEAP_TEXTURES: u64 = 64;

/// The executions of the buffer's commands in each frame's pass.
const EXECUTIONS: u64 = 4;

/// A frame's command buffer whose pass executes the commands a kernel writes, a barrier the CPU
/// set and cleared sending them to the host step, [`EXECUTIONS`] times, both encoders using a
/// heap of [`HEAP_TEXTURES`] textures: from its second frame on, its executions keep their
/// commands' recordings (`render/executions/kept.rs`), each keyed by what the command buffer made
/// resident, which its residency lists once for all of them (`Residency::kept_key`), not once an
/// execution: the listing costs the resources a frame, not the resources times the executions.
#[test]
fn a_host_step_lists_its_residency_once_for_every_execution() {
    on_device_with(at_plan, |device, _messages| {
        let library = library(device);
        let state = device
            .new_render_pipeline_state(&state_descriptor(&library), &PipelineOptions::default())
            .unwrap();
        let writer = writer(device, &library, 32);
        let draws: Vec<Draw> = vec![[0, 3, 1, 0, 0], [3, 6, 2, 0, 1], [9, 4, 1, 0, 0]];
        let icb = icb(device, &inheriting(), COMMANDS);
        let command = icb.indirect_render_command_at(COMMANDS - 1).unwrap();
        command.set_barrier().unwrap();
        command.clear_barrier().unwrap();
        let buffers: [Buffer; 7] = [
            buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap(),
            buffer_with(device, &words(draws.iter().flatten().copied())).unwrap(),
            buffer_with(device, &words((0..COMMANDS as u32).map(|c| 100 + c))).unwrap(),
            buffer_with(device, &floats([W as f32, H as f32, ROW, 0.0])).unwrap(),
            buffer_with(device, &floats(triangles().into_iter().flatten())).unwrap(),
            buffer_with(device, &words((0..COMMANDS as u32).map(|c| 200 + c))).unwrap(),
            buffer_with(device, &words(indices())).unwrap(),
        ];
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Placement,
                4 << 20,
                ResourceOptions::private(),
            ))
            .unwrap();
        let mut placed = Vec::new();
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::private();
        let fit = device.heap_texture_size_and_align(&descriptor).unwrap();
        let stride = fit.size.next_multiple_of(fit.align);
        for at in 0..HEAP_TEXTURES {
            placed.push(
                heap.new_texture_at_offset(&descriptor, at * stride)
                    .unwrap(),
            );
        }
        let target = target(device);
        let queue = device.new_command_queue().unwrap();
        let mut listed = Vec::new();
        for _ in 0..4 {
            let command_buffer = queue.command_buffer().unwrap();
            let before = device.work_counts();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder
                .use_resource(icb.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            encoder.use_heap(&heap).unwrap();
            encoder.set_compute_pipeline_state(&writer).unwrap();
            for (index, buffer) in buffers.iter().enumerate() {
                encoder.set_buffer(Some(buffer), 0, index as u32).unwrap();
            }
            let grid = Size::new(draws.len() as u64, 1, 1);
            encoder.dispatch_threads(grid, grid).unwrap();
            encoder.end_encoding().unwrap();
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.store_action = StoreAction::Store;
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder
                .use_heap(&heap, RenderStages::VERTEX | RenderStages::FRAGMENT)
                .unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            for (buffer, stages) in [
                (&buffers[2], RenderStages::VERTEX),
                (&buffers[3], RenderStages::VERTEX),
                (&buffers[4], RenderStages::VERTEX),
                (&buffers[6], RenderStages::VERTEX),
                (&buffers[5], RenderStages::FRAGMENT),
            ] {
                encoder
                    .use_resource(buffer.as_resource(), ResourceUsage::READ, stages)
                    .unwrap();
            }
            for _ in 0..EXECUTIONS {
                encoder
                    .execute_commands_in_buffer(&icb, 0..COMMANDS)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let after = device.work_counts();
            assert!(
                after.host_steps_recorded > before.host_steps_recorded,
                "the frame's executions take the host step"
            );
            listed.push(after.resident_kept_keys_listed - before.resident_kept_keys_listed);
        }
        // What the command buffer made resident: the heap and its textures, the indirect command
        // buffer and the five buffers.
        let resident = 1 + HEAP_TEXTURES + 1 + 5;
        for (frame, &listed) in listed.iter().enumerate().skip(1) {
            assert!(
                listed > 0 && listed <= 2 * resident,
                "frame {frame}: {listed} listed of {resident} resident, {EXECUTIONS} executions \
                 ({listed:?})"
            );
        }
        drop(placed);
    });
}
