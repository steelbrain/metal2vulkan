//! What a draw's functions reach of the command buffer's resident resources, by the stages that
//! write memory, on the device the cfg selects, under the validation layer. An indirect command
//! buffer an earlier compute encoder made resident for writing is in the command buffer's
//! residency for writing, in every render stage (`runtime/residency.rs`); the host step draws its
//! commands, whose vertex and fragment functions (the owned fixture `tob_renders`' `tob_triangle`
//! and `tob_solid`: the triangle of the points at vertex buffer 0, an unsized buffer the host
//! step's large-buffer form reaches through the resident root table, and the color at fragment
//! buffer 0) write no memory. Their draws read the resident resources and write none, as their functions
//! can write none (`bind::plan::slots::used_uses`): no draw waits on another's (the pass records
//! as many barriers whatever its commands' count), and no draw's vertex stage is taken to write
//! what its fragments touch (nothing announced past the host step).

use super::degrade_notices::{assert_made_once, on_device};
use super::indirect_host_steps::HOST_STEPS;
use super::render_executions::{forced_with, icb};
use super::{buffer_with, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{IndirectCommandType, PixelFormat, PrimitiveType, RenderStages};
use metal2vulkan::metal::{ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBufferStatus, Device, IndirectCommandBufferDescriptor, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    StoreAction, Texture, TextureDescriptor,
};

/// The target's side, in pixels.
const SIDE: u64 = 8;

/// Green, which every command draws over the whole target, and the clear, blue.
const GREEN: [u8; 4] = [0, 255, 0, 255];

/// The descriptor of `tob_triangle` and `tob_solid` into an `RGBA8Unorm` color attachment, made
/// with `supportIndirectCommandBuffers`.
fn descriptor(device: &Device) -> RenderPipelineDescriptor {
    let ll = engine_dir().join("tob_renders.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_triangle"));
    descriptor.fragment = Some(function("tob_solid"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    descriptor
}

/// A pass that clears `target` to blue and stores it.
fn pass(target: &Texture) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    pass
}

/// The pipeline barriers the device recorded, and the target's pixels, of one command buffer:
/// a compute encoder that makes a buffer of `commands` draws resident for writing, then a pass
/// that executes them, each a draw of `state` of the whole target in green, its corners and color
/// made resident for the vertex and fragment stages.
fn run(device: &Device, state: &RenderPipelineState, commands: u64) -> (u64, Vec<[u8; 4]>) {
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&target).unwrap();
    // A triangle over the whole target.
    let corners: Vec<u8> = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let corners = buffer_with(device, &corners).unwrap();
    let color: Vec<u8> = [0.0f32, 1.0, 0.0, 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let color = buffer_with(device, &color).unwrap();
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 1;
    descriptor.max_fragment_buffer_bind_count = 1;
    let buffer = icb(device, &descriptor, commands);
    for index in 0..commands {
        let command = buffer.indirect_render_command_at(index).unwrap();
        command.set_render_pipeline_state(state).unwrap();
        command.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        command.set_fragment_buffer(Some(&color), 0, 0).unwrap();
        command
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }
    // From before the command buffer's first encoder: each command is recorded at its call, so a
    // window opened at the commit would see only what the commit path records.
    let before = device.work_counts().commands_recorded.cmd_pipeline_barrier;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .use_resource(buffer.as_resource(), ResourceUsage::WRITE)
        .unwrap();
    encoder.end_encoding().unwrap();
    assert!(buffer.gpu_written());
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    let stages = RenderStages::VERTEX.union(RenderStages::FRAGMENT);
    for used in [&corners, &color] {
        encoder
            .use_resource(used.as_resource(), ResourceUsage::READ, stages)
            .unwrap();
    }
    encoder
        .execute_commands_in_buffer(&buffer, 0..commands)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let barriers = device.work_counts().commands_recorded.cmd_pipeline_barrier - before;
    let mut bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
    target
        .get_bytes(
            &mut bytes,
            4 * SIDE,
            0,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
    let pixels = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    (barriers, pixels)
}

/// The host step's draws of commands whose functions write no memory, in a command buffer whose
/// earlier compute encoder made their buffer resident for writing, write no resident resource:
/// a pass of 16 commands records as many barriers as one of a single command (none waits on
/// another's), each draws, and nothing past the host step is announced (no draw's vertex stage
/// writes bytes its fragments touch).
#[test]
fn replayed_draws_of_functions_that_write_nothing_write_no_resident_buffer() {
    on_device(|device, messages| {
        let descriptor = descriptor(device);
        let state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let mut barriers = Vec::new();
        // The first run on the device also records the device's own first-use work, which the
        // count takes from every command buffer: measured twice, it is the later reading.
        for commands in [1, 1, 16] {
            let (recorded, pixels) = run(device, &state, commands);
            assert!(
                pixels.iter().all(|&p| p == GREEN),
                "{commands} commands drew"
            );
            barriers.push(recorded);
        }
        assert_eq!(
            barriers[1], barriers[2],
            "no draw waits on another's: {barriers:?}"
        );
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &[descriptor]));
    });
}
