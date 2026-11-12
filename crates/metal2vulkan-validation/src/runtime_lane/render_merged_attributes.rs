//! Executions of render commands a kernel writes whose inherited state's vertex function fetches
//! `[[stage_in]]` attributes (the runtime's `render/executions/gpu.rs`, `bind::DrawReplay`), on
//! the device the cfg selects, under the validation layer. `icb_merged_attributes`' kernels encode
//! command `tid` from `draws[tid]` with its own vertex buffers at its own offsets: layout 0 its
//! vertices' pixel positions (attribute 0), layout 1 its marks (attribute 1, per instance); its
//! vertex function places each vertex at its position moved down a row per instance and carries
//! the mark, `vid` and `iid` flat into a `RGBA32Uint` target, and writes no memory, and its
//! fragment function reads no buffer.
//!
//! A draw replay pulls every layout fixed-function input would fetch
//! (`bind::draw_replay_vertex_layouts`), and a pulled attribute reads its command's buffer, so
//! draws and indexed draws of triangles run as one merged draw and triangle strips as one draw
//! per command, on the GPU alone, and draw what the host step draws of them (a barrier the CPU
//! set and cleared sends them there): each vertex its command's own vertex and instance, from its
//! command's own buffers. So do layouts whose strides fixed-function input cannot align, which
//! the shader fetches anyway, and a per-instance layout of a step rate of 2. A layout whose stride
//! is set per bind, which each command binds at its own, keeps the host step.

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
    Buffer, ClearColor, ComputePipelineState, Degradation, Device, IndirectCommandBufferDescriptor,
    Library, LoadAction, PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;

/// The target's width and height, and the rows an instance moves its vertices down.
const W: u64 = 64;
const H: u64 = 48;
const ROW: f32 = 8.0;
/// The buffer's commands.
const COMMANDS: u64 = 4;

/// The state's vertex layouts and how the buffers lay their elements out.
#[derive(Clone, Copy, Debug)]
struct Layouts {
    /// Layout 0's stride, a `float2` position (`None`: set per bind, which `ima_strided` binds at
    /// `xy`).
    positions: Option<u32>,
    /// The bytes from one position to the next.
    xy: u32,
    /// Layout 1's stride, a `uint` mark per instance, and its step rate.
    marks: u32,
    rate: u32,
}

/// Strides fixed-function input cannot align: the shader fetches both attributes.
const PULLED: Layouts = Layouts {
    positions: Some(10),
    xy: 10,
    marks: 6,
    rate: 1,
};

/// Strides fixed-function input fetches, marks at step rate `rate`.
const fn fixed(rate: u32) -> Layouts {
    Layouts {
        positions: Some(8),
        xy: 8,
        marks: 4,
        rate,
    }
}

/// Positions at a stride each command binds at its own.
const PER_BIND: Layouts = Layouts {
    positions: None,
    xy: 8,
    marks: 4,
    rate: 1,
};

/// A writer kernel: its name, the commands' type, whether it draws indexed, and whether its
/// commands merge into one draw (a list type) or draw one draw each (a strip).
type Kernel = (&'static str, IndirectCommandType, bool, bool);

const KERNELS: [Kernel; 3] = [
    ("ima_draws", IndirectCommandType::DRAW, false, true),
    ("ima_strips", IndirectCommandType::DRAW, false, false),
    ("ima_indexed", IndirectCommandType::DRAW_INDEXED, true, true),
];

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

/// The vertex descriptor of `layouts`, each attribute at offset 0 of its layout.
fn vertex_descriptor(layouts: Layouts) -> VertexDescriptor {
    let mut v = VertexDescriptor::new();
    v.layouts
        .insert(0, VertexBufferLayoutDescriptor::new(layouts.positions));
    let mut marks = VertexBufferLayoutDescriptor::new(Some(layouts.marks));
    marks.step_function = VertexStepFunction::PerInstance;
    marks.step_rate = layouts.rate;
    v.layouts.insert(1, marks);
    v.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float2, 0, 0),
    );
    v.attributes
        .insert(1, VertexAttributeDescriptor::new(VertexFormat::UInt, 0, 1));
    v
}

/// The commands' state of `layouts`, with `supportIndirectCommandBuffers`.
fn state(device: &Device, library: &Library, layouts: Layouts) -> RenderPipelineState {
    let stage = |name: &str| StageSource::Function(function(library, name));
    let mut descriptor = RenderPipelineDescriptor::new(stage("ima_vertex"));
    descriptor.fragment = Some(stage("ima_fragment"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    descriptor.vertex_descriptor = Some(vertex_descriptor(layouts));
    descriptor.support_indirect_command_buffers = true;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Draws of `command_types` of three vertex buffers that inherit the encoder's state, which set
/// their own attribute strides where `strided` (`supportDynamicAttributeStride`).
fn inheriting(
    command_types: IndirectCommandType,
    strided: bool,
) -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = command_types;
    descriptor.support_dynamic_attribute_stride = strided;
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

/// The commands' rows: first vertex, count, instances, base vertex (indexed), base instance, the
/// positions' element offset, the marks' element offset. Triangle 0; triangle 3 in two instances
/// from 1; triangles 5 and 6 at instance 3; triangle 7 in two instances. Each command's positions
/// and marks start at its own offset, so its first vertex, base vertex and base instance name
/// elements past them; no two commands' pixels meet.
const ROWS: [[u32; 7]; COMMANDS as usize] = [
    [0, 3, 1, 0, 0, 0, 0],
    [7, 3, 2, 4, 1, 2, 0],
    [9, 6, 1, 0, 3, 6, 2],
    [21, 3, 2, 6, 0, 0, 6],
];

/// The draws of [`ROWS`] laid out as `layouts` lays the buffers out. Indexed (`indexed`), the
/// indices count from 0 and a command's first index and base vertex add to the vertex start of
/// its draw without indices.
fn draws(indexed: bool, layouts: Layouts) -> Vec<Draw> {
    ROWS.iter()
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
                    xy_at * layouts.xy,
                    mark_at * layouts.marks,
                    layouts.xy,
                ]
            },
        )
        .collect()
}

/// The marks [`ROWS`]' instances fetch at step rate `rate`: Metal's element `base + (instance -
/// base) / rate` of each command's marks, mark `100 + j` at element `j` of the buffer.
fn marks(rate: u32) -> BTreeSet<u32> {
    ROWS.iter()
        .flat_map(|&[_, _, instances, _, base, _, mark_at]| {
            (0..instances).map(move |i| 100 + mark_at + base + i / rate)
        })
        .collect()
}

/// Run `writer` writing `draws` (of `command_types`) into a new buffer of [`COMMANDS`] commands (a
/// barrier the CPU set and cleared in its last command where `host_step`), positions and marks
/// laid out as `layouts` lays them, then a pass of `state` that executes all of them: the
/// target's pixels, and the merged draws' commands, the GPU form's and the records the host read
/// across the run.
fn run(
    device: &Device,
    (writer, state): (&ComputePipelineState, &RenderPipelineState),
    (command_types, draws, layouts, host_step): (IndirectCommandType, &[Draw], Layouts, bool),
) -> (Vec<[u32; 4]>, (u64, u64, u64)) {
    let strided = layouts.positions.is_none();
    let icb = icb(device, &inheriting(command_types, strided), COMMANDS);
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
        buffer_with(device, &strided(positions, layouts.xy)).unwrap(),
        buffer_with(device, &strided(marks, layouts.marks)).unwrap(),
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

/// The marks the covered pixels of `drawn` carry.
fn drawn_marks(drawn: &[[u32; 4]]) -> BTreeSet<u32> {
    drawn.iter().map(|p| p[0]).filter(|&m| m != 0).collect()
}

/// Each of [`KERNELS`] under the state of `layouts` on `device`: its execution runs on the GPU
/// alone (merged, or one draw per command), the host reading no record, and draws what the host
/// step draws of it, each instance its own mark.
fn replayed_as_the_host_step_draws(device: &Device, layouts: Layouts) {
    let library = library(device);
    let state = state(device, &library, layouts);
    for (kernel, command_types, indexed, merges) in KERNELS {
        let writer = writer(device, &library, kernel);
        let draws = draws(indexed, layouts);
        let mut outputs = Vec::new();
        let mut forms = Vec::new();
        for host_step in [false, true] {
            let (drawn, counted) = run(
                device,
                (&writer, &state),
                (command_types, &draws, layouts, host_step),
            );
            outputs.push(drawn);
            forms.push(counted);
        }
        let what = format!("{kernel} {layouts:?}");
        assert_eq!(
            drawn_marks(&outputs[1]),
            marks(layouts.rate),
            "{what}: each instance's own mark in the host step"
        );
        assert!(
            outputs[0] == outputs[1],
            "{what}: the execution ({:?}) against the host step: {:?}",
            forms[0],
            outputs[0]
                .iter()
                .zip(&outputs[1])
                .enumerate()
                .filter(|(_, (m, h))| m != h)
                .take(8)
                .collect::<Vec<_>>()
        );
        let merged = if merges { COMMANDS } else { 0 };
        assert_eq!(forms[0], (merged, COMMANDS, 0), "{what}: on the GPU alone");
        let (merged, gpu, read) = forms[1];
        assert!(
            merged == 0 && gpu == 0 && read >= COMMANDS,
            "{what}: the host step {:?}",
            forms[1]
        );
    }
}

/// Draws, strips and indexed draws whose vertex function fetches both its attributes in the
/// shader, positions per vertex and marks per instance (each layout's stride one fixed-function
/// input cannot align), run on the GPU alone and draw what the host step draws of them: each
/// vertex fetches its command's own vertex (its vertex start or its index plus its base vertex)
/// and instance (its base instance plus the instance), from its command's own buffers, merged
/// draws included, whose own vertex and instance indices count the merged draw's corners.
#[test]
fn pulled_attributes_of_executed_draws_fetch_their_commands_elements() {
    on_device_with(at_plan, |device, messages| {
        replayed_as_the_host_step_draws(device, PULLED);
        let notices = counts(&messages.take_notices());
        assert!(notices.contains_key(&HOST_STEPS[0]), "{notices:?}");
    });
}

/// Layouts fixed-function input would fetch, marks at step rates 1 and 2, are pulled in a draw
/// replay (`bind::draw_replay_vertex_layouts`) with no vertex buffer bound at the encoder: the
/// executions run on the GPU alone and draw what the host step draws of them, a rate of 2
/// dividing each command's instances from its base instance.
#[test]
fn fixed_function_layouts_of_executed_draws_are_pulled() {
    on_device_with(at_plan, |device, messages| {
        for rate in [1, 2] {
            replayed_as_the_host_step_draws(device, fixed(rate));
        }
        let notices = counts(&messages.take_notices());
        assert!(notices.contains_key(&HOST_STEPS[0]), "{notices:?}");
    });
}

/// A layout whose stride is set per bind, which each command binds at its own (`ima_strided`),
/// keeps its execution from the GPU: the host step draws it, each instance its own mark, and the
/// device's notice names the rule (`DrawReplayRule::VertexAttributes`).
#[test]
fn a_layout_of_a_stride_set_per_bind_keeps_the_host_step() {
    on_device_with(at_plan, |device, messages| {
        let library = library(device);
        let state = state(device, &library, PER_BIND);
        let writer = writer(device, &library, "ima_strided");
        let draws = draws(false, PER_BIND);
        let (drawn, (merged, gpu, read)) = run(
            device,
            (&writer, &state),
            (IndirectCommandType::DRAW, &draws, PER_BIND, false),
        );
        assert!(
            merged == 0 && gpu == 0 && read >= COMMANDS,
            "the host step: {merged} {gpu} {read}"
        );
        assert_eq!(drawn_marks(&drawn), marks(1));
        let notices = messages.take_notices();
        let host_step = notices
            .iter()
            .find(|m| {
                m.degradation
                    .as_ref()
                    .is_some_and(|d| d.degradation == Degradation::HostReplayedIndirectCommands)
            })
            .expect("the host step's notice");
        assert!(
            host_step
                .text
                .contains("vertex function fetches attributes at a stride set per bind"),
            "{}",
            host_step.text
        );
    });
}
