//! Executions of render commands a kernel writes whose inherited state's vertex function fetches
//! `[[stage_in]]` attributes (the runtime's `render/executions/gpu.rs`, `bind::DrawReplay`), on
//! the device the cfg selects, under the validation layer. `icb_merged_attributes`' kernels encode
//! command `tid` from `draws[tid]` with its own vertex buffers at its own offsets: layout 0 its
//! vertices' pixel positions (attribute 0), layout 1 its marks (attribute 1, per instance); its
//! vertex function places each vertex at its position moved down a row per instance and carries
//! the mark, `vid` and `iid` flat into a `RGBA32Uint` target, and writes no memory, and its
//! fragment function reads no buffer.
//!
//! - Where every attribute is fetched in the shader (each layout's stride one fixed-function input
//!   cannot align), draws and indexed draws of triangles draw what the host step draws of them,
//!   which a barrier the CPU set and cleared sends them to: each vertex its command's own vertex
//!   and instance, from its command's own buffers. The replay variant refuses a vertex function
//!   that fetches attributes (`LoweringInputs::replay_bindings`), so both take the host step.

use super::degrade_notices::{counts, on_device_with};
use super::indirect_host_steps::{complete, HOST_STEPS};
use super::render_executions::icb;
use super::{at_plan, buffer_with};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, RenderStages, ResourceUsage, TextureUsage, VertexFormat,
    VertexStepFunction,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor,
    RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, ComputePipelineState, Device, IndirectCommandBufferDescriptor, Library,
    LoadAction, PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The target's width and height, and the rows an instance moves its vertices down.
const W: u64 = 64;
const H: u64 = 48;
const ROW: f32 = 8.0;
/// The buffer's commands.
const COMMANDS: u64 = 4;
/// A stride of layout 0 (a `float2` position) and of layout 1 (a `uint` mark) that
/// fixed-function input cannot align: the shader fetches both attributes.
const PULLED_STRIDES: (u32, u32) = (10, 6);

/// A command's draw: first vertex (or first index), vertex (or index) count, instance count, base
/// vertex, base instance, the byte offsets of its positions and its marks, and the stride its
/// positions are bound at (`ima_strided`).
type Draw = [u32; 8];

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let bitcode = llvm_assemble(&engine_dir().join("icb_merged_attributes.ll")).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

fn function(library: &Library, name: &str) -> FunctionDescriptor {
    library
        .new_function_with_name(name)
        .unwrap()
        .function_descriptor()
}

/// The kernel `name` that writes the commands.
fn writer(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
    let descriptor = ComputePipelineDescriptor::new(StageSource::Function(function(library, name)));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The vertex descriptor of positions at stride `strides.0` and per-instance marks at stride
/// `strides.1`, each at offset 0 of its layout.
fn vertex_descriptor(strides: (u32, u32)) -> VertexDescriptor {
    let mut v = VertexDescriptor::new();
    v.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(strides.0)));
    let mut marks = VertexBufferLayoutDescriptor::new(Some(strides.1));
    marks.step_function = VertexStepFunction::PerInstance;
    v.layouts.insert(1, marks);
    v.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float2, 0, 0),
    );
    v.attributes
        .insert(1, VertexAttributeDescriptor::new(VertexFormat::UInt, 0, 1));
    v
}

/// The descriptor of the commands' state, with `supportIndirectCommandBuffers`.
fn state_descriptor(library: &Library, strides: (u32, u32)) -> RenderPipelineDescriptor {
    let stage = |name: &str| StageSource::Function(function(library, name));
    let mut descriptor = RenderPipelineDescriptor::new(stage("ima_vertex"));
    descriptor.fragment = Some(stage("ima_fragment"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    descriptor.vertex_descriptor = Some(vertex_descriptor(strides));
    descriptor.support_indirect_command_buffers = true;
    descriptor
}

/// Draws of `command_types` of three vertex buffers that inherit the encoder's state.
fn inheriting(command_types: IndirectCommandType) -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = command_types;
    descriptor.inherit_pipeline_state = true;
    descriptor.max_vertex_buffer_bind_count = 3;
    descriptor.max_fragment_buffer_bind_count = 0;
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

/// `elements` at `stride` bytes apart, each followed by filler bytes to the stride, and a stride
/// of filler past the last.
fn strided(elements: impl IntoIterator<Item = Vec<u8>>, stride: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for element in elements {
        let at = out.len();
        out.extend_from_slice(&element);
        out.resize(at + stride as usize, 0xEE);
    }
    out.resize(out.len() + stride as usize, 0xEE);
    out
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

/// Run `writer` writing `draws` (of `command_types`) into a new buffer of [`COMMANDS`] commands (a
/// barrier the CPU set and cleared in its last command where `host_step`), positions and marks
/// laid out at `strides`, then a pass that executes all of them: the target's pixels, and the
/// merged draws' commands, the GPU form's and the records the host read across the run.
fn run(
    device: &Device,
    (writer, state): (&ComputePipelineState, &RenderPipelineState),
    (command_types, draws, strides, host_step): (IndirectCommandType, &[Draw], (u32, u32), bool),
) -> (Vec<[u32; 4]>, (u64, u64, u64)) {
    let icb = icb(device, &inheriting(command_types), COMMANDS);
    if host_step {
        let command = icb.indirect_render_command_at(COMMANDS - 1).unwrap();
        command.set_barrier().unwrap();
        command.clear_barrier().unwrap();
    }
    let positions = triangles().into_iter().map(floats);
    let marks = (0..16u32).map(|j| words([100 + j]));
    let buffers: [Buffer; 6] = [
        buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap(),
        buffer_with(device, &words(draws.iter().flatten().copied())).unwrap(),
        buffer_with(device, &strided(positions, strides.0)).unwrap(),
        buffer_with(device, &strided(marks, strides.1)).unwrap(),
        buffer_with(device, &floats([W as f32, H as f32, ROW, 0.0])).unwrap(),
        buffer_with(device, &words(0..48)).unwrap(),
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
    // The positions, marks, frame and indices the vertex stage reads.
    for buffer in &buffers[2..] {
        encoder
            .use_resource(
                buffer.as_resource(),
                ResourceUsage::READ,
                RenderStages::VERTEX,
            )
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

/// The draws of [`run`]: triangle 0; triangle 3 in two instances from 1; triangles 5 and 6 at
/// instance 3; triangle 7 in two instances. Each command's positions and marks start at its own
/// offset, so its first vertex, base vertex and base instance name elements past them; no two
/// commands' pixels meet. Indexed (`indexed`), the indices count from 0 and a command's first
/// index and base vertex add to the vertex start of its draw without indices.
fn draws(indexed: bool, strides: (u32, u32)) -> Vec<Draw> {
    let (xy, mark) = strides;
    // First vertex, count, instances, base vertex (indexed), base instance, the positions'
    // element offset, the marks' element offset.
    let rows: [[u32; 7]; COMMANDS as usize] = [
        [0, 3, 1, 0, 0, 0, 0],
        [7, 3, 2, 4, 1, 2, 0],
        [9, 6, 1, 0, 3, 6, 2],
        [21, 3, 2, 6, 0, 0, 6],
    ];
    rows.iter()
        .map(
            |&[first, count, instances, base_vertex, base_instance, xy_at, mark_at]| {
                let (first, base_vertex) = match indexed {
                    true => (first - base_vertex, base_vertex),
                    false => (first, 0),
                };
                [
                    first,
                    count,
                    instances,
                    base_vertex,
                    base_instance,
                    xy_at * xy,
                    mark_at * mark,
                    xy,
                ]
            },
        )
        .collect()
}

/// The marks of [`draws`]' instances: element `base instance + instance` of each command's marks.
const MARKS: [u32; 6] = [100, 101, 102, 105, 106, 107];

/// Draws and indexed draws of triangles whose vertex function fetches both its attributes in the
/// shader, positions per vertex and marks per instance (each layout's stride one fixed-function
/// input cannot align, so no layout is bound in fixed function), draw what the host step draws of
/// them: each vertex fetches its command's own vertex (its vertex start or its index plus its base
/// vertex) and instance (its base instance plus the instance), from its command's own buffers.
/// Both take the host step: the replay variant refuses a vertex function that fetches attributes.
#[test]
fn pulled_attributes_of_executed_draws_draw_what_the_host_step_draws() {
    on_device_with(at_plan, |device, messages| {
        let library = library(device);
        let state = device
            .new_render_pipeline_state(
                &state_descriptor(&library, PULLED_STRIDES),
                &PipelineOptions::default(),
            )
            .unwrap();
        for (kernel, command_types, indexed) in [
            ("ima_draws", IndirectCommandType::DRAW, false),
            ("ima_indexed", IndirectCommandType::DRAW_INDEXED, true),
        ] {
            let writer = writer(device, &library, kernel);
            let draws = draws(indexed, PULLED_STRIDES);
            let mut outputs = Vec::new();
            let mut forms = Vec::new();
            for host_step in [false, true] {
                let (drawn, counted) = run(
                    device,
                    (&writer, &state),
                    (command_types, &draws, PULLED_STRIDES, host_step),
                );
                outputs.push(drawn);
                forms.push(counted);
            }
            let marks: BTreeSet<u32> = outputs[1]
                .iter()
                .map(|p| p[0])
                .filter(|&m| m != 0)
                .collect();
            assert_eq!(
                marks,
                BTreeSet::from(MARKS),
                "{kernel}: each instance's own mark in the host step"
            );
            assert!(
                outputs[0] == outputs[1],
                "{kernel}: the execution ({:?}) against the host step: {:?}",
                forms[0],
                outputs[0]
                    .iter()
                    .zip(&outputs[1])
                    .enumerate()
                    .filter(|(_, (m, h))| m != h)
                    .take(8)
                    .collect::<Vec<_>>()
            );
            for (merged, gpu, read) in forms {
                assert!(
                    merged == 0 && gpu == 0 && read >= COMMANDS,
                    "{kernel}: the host step {forms:?}"
                );
            }
        }
        let notices = counts(&messages.take_notices());
        assert!(notices.contains_key(&HOST_STEPS[0]), "{notices:?}");
    });
}
