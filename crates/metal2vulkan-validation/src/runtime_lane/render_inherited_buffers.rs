//! Executions of render commands a kernel writes into a buffer that inherits the encoder's
//! buffers, and into one whose commands bind their own (the runtime's `render/executions/gpu.rs`,
//! **Inherited buffers**; `bind::DrawReplay`), on the device the cfg selects, under the validation
//! layer. `icb_merged_arguments`' kernels encode command `tid` from `draws[tid]` (first index,
//! index count, instance count, base vertex, base instance) as an indexed triangle draw of 32-bit
//! indices, resetting the command where its index count is 0: `ica_own` binds the command's own
//! buffers (positions, marks and the frame at vertex buffers 0 to 2, the argument buffer and its
//! own tint at fragment buffers 0 and 1), `ica_inherited` none. The state's vertex function
//! fetches `[[stage_in]]` positions (layout 0) and marks (layout 1, per vertex) and reads the
//! frame; its fragment function reads an argument buffer of two textures and a constant pointer
//! and a tint, into a `RGBA32Uint` target.
//!
//! Both run on the GPU alone, in one merged draw, the host recording no step and reading no
//! record, and draw what the host step draws of them (a barrier the CPU set and cleared sends them
//! there). The inherited buffers are the encoder's as they stand at the call: the positions in a
//! heap's buffer the encoder alone names (no `useResource:`, no `useHeap:`), the marks bound at a
//! non-zero offset, which a blit of the same command buffer writes before the pass, the frame and
//! the argument buffer bound at one offset and moved by `setVertexBufferOffset:` and
//! `setFragmentBufferOffset:` before the call, and the tint inline bytes (`setFragmentBytes:`).

use super::commit_costs::process_cpu_us;
use super::degrade_notices::{counts, on_device_with};
use super::indirect_host_steps::{complete, HOST_STEPS};
use super::render_executions::icb;
use super::{at_plan, buffer_with, validated_device_with, Messages, ScratchCache};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, RenderStages, ResourceUsage, TextureUsage, VertexFormat,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, CommandWrites, ComputePipelineDescriptor, FunctionDescriptor,
    RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, ComputePipelineState, DebugMessage, Device, HeapDescriptor,
    HeapType, IndirectCommandBuffer, IndirectCommandBufferDescriptor, Library, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeSet;
use std::time::Instant;

/// The target's width and height, and the rows an instance moves its vertices down.
const W: u64 = 64;
const H: u64 = 48;
const ROW: f32 = 8.0;
/// The buffer's commands.
const COMMANDS: u64 = 8;

/// Where the encoder binds what the commands inherit: the marks, and the frame and the argument
/// buffer once their offsets moved (each first bound at 0).
const MARKS_AT: u64 = 64;
const FRAME_AT: u64 = 32;
const ARGUMENTS_AT: u64 = 256;
/// The inline bytes' tint, every command's where they inherit the encoder's buffers.
const INHERITED_TINT: u32 = 300;

/// A command's draw: first index, index count (0 resets the command), instance count, base
/// vertex, base instance.
type Draw = [u32; 5];

/// Triangle 15; 14 and 13 in instances 1 and 2; a reset; 12 and a trailing index; a reset; 10
/// moved to 11 by its base vertex; 9 in instance 2; 8. No two overlap.
const DRAWS: [Draw; COMMANDS as usize] = [
    [0, 3, 1, 0, 0],
    [3, 6, 2, 0, 1],
    [9, 0, 1, 0, 0],
    [9, 4, 1, 0, 0],
    [13, 0, 1, 0, 0],
    [15, 3, 1, 3, 0],
    [18, 3, 1, 0, 2],
    [21, 3, 1, 0, 0],
];

/// The words the argument buffer's pointer reaches, word `mark % 4` read.
const WORDS: [u32; 4] = [7, 11, 13, 17];

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let bitcode = llvm_assemble(&engine_dir().join("icb_merged_arguments.ll")).unwrap();
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

/// The commands' state, with `supportIndirectCommandBuffers`: attribute 0 (a `float2` position)
/// in layout 0, attribute 1 (a `uint` mark) in layout 1, both per vertex.
fn state(device: &Device, library: &Library) -> RenderPipelineState {
    let stage = |name: &str| StageSource::Function(function(library, name));
    let mut descriptor = RenderPipelineDescriptor::new(stage("ica_vertex"));
    descriptor.fragment = Some(stage("ica_fragment"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    let mut input = VertexDescriptor::new();
    input
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(8)));
    input
        .layouts
        .insert(1, VertexBufferLayoutDescriptor::new(Some(4)));
    input.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float2, 0, 0),
    );
    input
        .attributes
        .insert(1, VertexAttributeDescriptor::new(VertexFormat::UInt, 0, 1));
    descriptor.vertex_descriptor = Some(input);
    descriptor.support_indirect_command_buffers = true;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Indexed draws under the encoder's pipeline state, which inherit its buffers where `inherited`
/// and else bind three vertex buffers and two fragment buffers of their own.
fn commands(inherited: bool) -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW_INDEXED;
    descriptor.inherit_pipeline_state = true;
    descriptor.inherit_buffers = inherited;
    if !inherited {
        descriptor.max_vertex_buffer_bind_count = 3;
        descriptor.max_fragment_buffer_bind_count = 2;
    }
    descriptor
}

fn words(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// `bytes` after `at` filler bytes.
fn at(at: u64, bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0xEE; at as usize];
    out.extend_from_slice(bytes);
    out
}

/// Sixteen three-pixel triangles, triangle `k` of vertices `3k`, `3k + 1`, `3k + 2`, each at its
/// own place.
fn triangles() -> Vec<f32> {
    (0..48u32)
        .flat_map(|v| {
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

/// A 4 x 1 `RGBA8Unorm` texture whose texel `x` is red `red + x`.
fn texture(device: &Device, red: u8) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let texels: Vec<u8> = (0..4).flat_map(|x| [red + x, 0, 0, 255]).collect();
    texture
        .replace_region(Region::new_2d(0, 0, 4, 1), 0, 0, &texels, 16, 0)
        .unwrap();
    texture
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

/// What one run of the commands counted: merged draw commands, GPU-form commands, records the
/// host read, host steps recorded.
type Counted = (u64, u64, u64, u64);

/// The resources the commands reach: the positions in a heap's buffer, the marks (filler until a
/// run's blit copies `staged_marks` in), the frame, the argument buffer of the two textures and
/// the words it points at (`pointed`), the tints and the indices.
struct Inputs {
    /// The heap the positions lie in, held for the runs.
    _heap: metal2vulkan_runtime::Heap,
    positions: Buffer,
    marks: Buffer,
    staged_marks: Buffer,
    frame: Buffer,
    arguments: Buffer,
    tints: Buffer,
    indices: Buffer,
    pointed: Buffer,
    textures: [Texture; 2],
}

impl Inputs {
    fn new(device: &Device) -> Inputs {
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 16,
                ResourceOptions::shared(),
            ))
            .unwrap();
        let xy = floats(triangles());
        let positions = heap
            .new_buffer(xy.len() as u64, ResourceOptions::shared())
            .unwrap();
        let contents = positions.contents().unwrap();
        // SAFETY: the buffer is `Shared`, its `xy.len()` bytes are the CPU's, and no command reads
        // them yet.
        unsafe { std::ptr::copy_nonoverlapping(xy.as_ptr(), contents.as_ptr(), xy.len()) };
        let textures = [texture(device, 10), texture(device, 20)];
        let pointed = buffer_with(device, &words(WORDS)).unwrap();
        let mut table = Vec::new();
        for texture in &textures {
            table.extend_from_slice(&texture.gpu_resource_id().0.to_le_bytes());
        }
        table.extend_from_slice(&pointed.gpu_address().0.to_le_bytes());
        let marks = at(MARKS_AT, &words((0..48).map(|v| 100 + v / 3)));
        Inputs {
            _heap: heap,
            positions,
            marks: buffer_with(device, &vec![0xEE; marks.len()]).unwrap(),
            staged_marks: buffer_with(device, &marks).unwrap(),
            frame: buffer_with(
                device,
                &at(FRAME_AT, &floats([W as f32, H as f32, ROW, 0.0])),
            )
            .unwrap(),
            arguments: buffer_with(device, &at(ARGUMENTS_AT, &table)).unwrap(),
            tints: buffer_with(device, &words((0..COMMANDS as u32).map(|c| 200 + c))).unwrap(),
            indices: buffer_with(device, &words(indices())).unwrap(),
            pointed,
            textures,
        }
    }
}

/// Run `writer` writing [`DRAWS`] into a new buffer of [`COMMANDS`] commands that inherits the
/// encoder's buffers where `inherited` (a barrier the CPU set and cleared in its last command
/// where `host_step`), then a pass of `state` that executes all of them: the target's pixels, and
/// what the run counted.
fn run(
    device: &Device,
    (writer, state, inputs): (&ComputePipelineState, &RenderPipelineState, &Inputs),
    (inherited, host_step): (bool, bool),
) -> (Vec<[u32; 4]>, Counted) {
    let icb = icb(device, &commands(inherited), COMMANDS);
    if host_step {
        let command = icb.indirect_render_command_at(COMMANDS - 1).unwrap();
        command.set_barrier().unwrap();
        command.clear_barrier().unwrap();
    }
    let draws = buffer_with(device, &words(DRAWS.iter().flatten().copied())).unwrap();
    let target = target(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let before = device.work_counts();
    encode(
        &command_buffer,
        (writer, state, inputs),
        (&icb, &draws, &target),
        inherited,
    );
    complete(&command_buffer);
    let after = device.work_counts();
    let counted = (
        after.merged_draw_commands - before.merged_draw_commands,
        after.gpu_form_commands - before.gpu_form_commands,
        after.indirect_records_read - before.indirect_records_read,
        after.host_steps_recorded - before.host_steps_recorded,
    );
    (pixels(&target), counted)
}

/// Encode into `command_buffer` a blit of the marks, then `writer` writing `draws` into every
/// command of `icb`, then a pass of `state` into `target` that executes all of them, with the
/// encoder's buffers bound as the module docs say where `inherited`, else made resident for the
/// commands' own.
fn encode(
    command_buffer: &CommandBuffer,
    (writer, state, inputs): (&ComputePipelineState, &RenderPipelineState, &Inputs),
    (icb, draws, target): (&IndirectCommandBuffer, &Buffer, &Texture),
    inherited: bool,
) {
    let count = icb.size();
    let device = command_buffer.device();
    let encoded = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
    // The marks, written on the GPU before the pass reads them.
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let marks = inputs.marks.length() - MARKS_AT;
    blit.copy_from_buffer_to_buffer(
        &inputs.staged_marks,
        MARKS_AT,
        &inputs.marks,
        MARKS_AT,
        marks,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .use_resource(icb.as_resource(), ResourceUsage::WRITE)
        .unwrap();
    encoder.set_compute_pipeline_state(writer).unwrap();
    for (index, buffer, offset) in [
        (0, &encoded, 0),
        (1, draws, 0),
        (2, &inputs.positions, 0),
        (3, &inputs.marks, MARKS_AT),
        (4, &inputs.frame, FRAME_AT),
        (5, &inputs.arguments, ARGUMENTS_AT),
        (6, &inputs.tints, 0),
        (7, &inputs.indices, 0),
    ] {
        encoder.set_buffer(Some(buffer), offset, index).unwrap();
    }
    let grid = Size::new(count, 1, 1);
    encoder
        .dispatch_threads(grid, Size::new(count.min(64), 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    // What the commands reach through addresses and resource IDs: the indices, the argument
    // buffer's words and textures, and the buffers commands of their own bind.
    let mut used = vec![
        (&inputs.indices, RenderStages::VERTEX),
        (&inputs.pointed, RenderStages::FRAGMENT),
    ];
    if inherited {
        // The encoder's buffers, as they stand at the call.
        encoder
            .set_vertex_buffer(Some(&inputs.positions), 0, 0)
            .unwrap();
        encoder
            .set_vertex_buffer(Some(&inputs.marks), MARKS_AT, 1)
            .unwrap();
        encoder
            .set_vertex_buffer(Some(&inputs.frame), 0, 2)
            .unwrap();
        encoder.set_vertex_buffer_offset(FRAME_AT, 2).unwrap();
        encoder
            .set_fragment_buffer(Some(&inputs.arguments), 0, 0)
            .unwrap();
        encoder.set_fragment_buffer_offset(ARGUMENTS_AT, 0).unwrap();
        encoder
            .set_fragment_bytes(&INHERITED_TINT.to_le_bytes(), 1)
            .unwrap();
    } else {
        used.extend([
            (&inputs.positions, RenderStages::VERTEX),
            (&inputs.marks, RenderStages::VERTEX),
            (&inputs.frame, RenderStages::VERTEX),
            (&inputs.arguments, RenderStages::FRAGMENT),
            (&inputs.tints, RenderStages::FRAGMENT),
        ]);
    }
    for (buffer, stages) in used {
        encoder
            .use_resource(buffer.as_resource(), ResourceUsage::READ, stages)
            .unwrap();
    }
    for texture in &inputs.textures {
        encoder
            .use_resource(
                texture.as_resource(),
                ResourceUsage::READ,
                RenderStages::FRAGMENT,
            )
            .unwrap();
    }
    encoder.execute_commands_in_buffer(icb, 0..count).unwrap();
    encoder.end_encoding().unwrap();
}

/// The tints the drawing commands' pixels carry: the inline bytes' plus the instance where the
/// commands inherit the encoder's buffers, else each command's own tint (`200 + tid`) plus it.
fn tints(inherited: bool) -> BTreeSet<u32> {
    DRAWS
        .iter()
        .zip(0u32..)
        .filter(|([_, count, ..], _)| *count >= 3)
        .flat_map(|(&[_, _, instances, _, base], tid)| {
            let tint = match inherited {
                true => INHERITED_TINT,
                false => 200 + tid,
            };
            (base..base + instances).map(move |iid| tint + iid)
        })
        .collect()
}

/// Commands that inherit the encoder's buffers, and commands that bind their own, run on the GPU
/// alone in one merged draw, the host reading no record, and once the states' pipelines are made
/// (a second run: the first may wait at a step for what the states' creations left making,
/// `WorkCounts::creations_left_making`) recording no step, and draw what the host step draws of
/// them: each vertex its index plus its base vertex, its position and mark from
/// the layouts and the frame from the encoder's (or the command's) buffers at their offsets as
/// they stood at the call, a reset command or a trailing index drawing nothing, and each fragment
/// the textures and the word its argument buffer reaches at its mark, and its tint.
#[test]
fn inherited_buffers_draw_on_the_gpu_what_the_host_step_draws() {
    on_device_with(at_plan, |device, messages| {
        let library = library(device);
        let state = state(device, &library);
        let inputs = Inputs::new(device);
        let mut notices = Vec::new();
        for (inherited, kernel) in [(true, "ica_inherited"), (false, "ica_own")] {
            let writer = writer(device, &library, kernel);
            let (cold, cold_counted) = run(device, (&writer, &state, &inputs), (inherited, false));
            let (gpu, gpu_counted) = run(device, (&writer, &state, &inputs), (inherited, false));
            let stepped = messages.take_notices();
            let (host, host_counted) = run(device, (&writer, &state, &inputs), (inherited, true));
            let what = kernel;
            let (merged, form, read, _) = cold_counted;
            assert_eq!(
                (merged, form, read),
                (COMMANDS, COMMANDS, 0),
                "{what}: the first run on the GPU alone, merged; announced {:?}",
                reasons(&stepped)
            );
            assert_eq!(
                gpu_counted,
                (COMMANDS, COMMANDS, 0, 0),
                "{what}: on the GPU alone, merged; announced {:?}",
                reasons(&stepped)
            );
            assert!(
                cold == gpu,
                "{what}: the first run draws what the second does"
            );
            notices.extend(stepped);
            notices.extend(messages.take_notices());
            let (merged, form, read, steps) = host_counted;
            assert!(
                merged == 0 && form == 0 && read >= COMMANDS && steps > 0,
                "{what}: the host step {host_counted:?}"
            );
            let covered: Vec<&[u32; 4]> = host.iter().filter(|p| p[0] != 0).collect();
            assert!(!covered.is_empty(), "{what}: the host step drew");
            for p in &covered {
                let x = p[0] % 4;
                assert_eq!(
                    [p[1], p[2]],
                    [(10 + x) | (20 + x) << 8, WORDS[x as usize]],
                    "{what}: the argument buffer's textures and word at mark {}",
                    p[0]
                );
            }
            let drawn: BTreeSet<u32> = covered.iter().map(|p| p[3]).collect();
            assert_eq!(drawn, tints(inherited), "{what}: the tints");
            assert!(
                gpu == host,
                "{what}: the GPU form against the host step: {:?}",
                gpu.iter()
                    .zip(&host)
                    .enumerate()
                    .filter(|(_, (g, h))| g != h)
                    .take(8)
                    .collect::<Vec<_>>()
            );
        }
        let notices = counts(&notices);
        assert!(notices.contains_key(&HOST_STEPS[0]), "{notices:?}");
    });
}

/// Each degrade notice of `notices`: what it names, and why.
pub(super) fn reasons(notices: &[DebugMessage]) -> Vec<String> {
    (notices.iter())
        .filter_map(|message| message.degradation.as_ref())
        .map(|notice| format!("{}: {}", notice.object, notice.reason))
        .collect()
}

/// A kernel that writes compute commands into another buffer, made resident for writing in the
/// same command buffer, writes no command into these (the runtime's `write_routes.rs`, **Of their
/// kind alone**): draws that inherit the encoder's buffers, and draws that bind their own, keep
/// the GPU form, merged, the host reading no record and, once the states' pipelines are made (a
/// second command buffer), recording no step, their records holding no dispatch, and draw what
/// they draw alone.
#[test]
fn a_compute_command_writer_beside_them_leaves_the_draws_on_the_gpu() {
    on_device_with(at_plan, |device, messages| {
        let library = library(device);
        let state = state(device, &library);
        let inputs = Inputs::new(device);
        let bitcode = llvm_assemble(&engine_dir().join("icb_inherited_dispatches.ll")).unwrap();
        let dispatches = device.new_library_with_air(&bitcode).unwrap();
        let dispatcher = self::writer(device, &dispatches, "cid_encode");
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        descriptor.inherit_pipeline_state = true;
        descriptor.inherit_buffers = true;
        let beside = icb(device, &descriptor, 4);
        let encoded = buffer_with(device, &beside.gpu_resource_id().0.to_le_bytes()).unwrap();
        let rows = buffer_with(device, &words(DRAWS.iter().flatten().copied())).unwrap();
        let target = target(device);
        let queue = device.new_command_queue().unwrap();
        for (inherited, kernel) in [(true, "ica_inherited"), (false, "ica_own")] {
            let writer = writer(device, &library, kernel);
            let (alone, _) = run(device, (&writer, &state, &inputs), (inherited, false));
            let draws = icb(device, &commands(inherited), COMMANDS);
            // One command buffer: the dispatcher's writes, then the draws' writer and their pass.
            let frame = || {
                let command_buffer = queue.command_buffer().unwrap();
                let before = device.work_counts();
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                encoder
                    .use_resource(beside.as_resource(), ResourceUsage::WRITE)
                    .unwrap();
                encoder.set_compute_pipeline_state(&dispatcher).unwrap();
                encoder.set_buffer(Some(&encoded), 0, 0).unwrap();
                let grid = Size::new(4, 1, 1);
                encoder.dispatch_threads(grid, grid).unwrap();
                encoder.end_encoding().unwrap();
                encode(
                    &command_buffer,
                    (&writer, &state, &inputs),
                    (&draws, &rows, &target),
                    inherited,
                );
                complete(&command_buffer);
                let after = device.work_counts();
                let counted = (
                    after.merged_draw_commands - before.merged_draw_commands,
                    after.gpu_form_commands - before.gpu_form_commands,
                    after.indirect_records_read - before.indirect_records_read,
                    after.host_steps_recorded - before.host_steps_recorded,
                );
                (pixels(&target), counted)
            };
            let (cold, (merged, form, read, _)) = frame();
            let (drawn, counted) = frame();
            let notices = reasons(&messages.take_notices());
            assert_eq!(
                (merged, form, read),
                (COMMANDS, COMMANDS, 0),
                "{kernel}: the first command buffer on the GPU alone, merged; announced {notices:?}"
            );
            assert_eq!(
                counted,
                (COMMANDS, COMMANDS, 0, 0),
                "{kernel}: on the GPU alone, merged; announced {notices:?}"
            );
            assert!(
                !draws.command_writes().may(CommandWrites::DISPATCH),
                "{kernel}: {:?}",
                draws.command_writes()
            );
            assert!(
                cold == alone,
                "{kernel}: the first draws beside a compute writer"
            );
            assert!(
                drawn == alone,
                "{kernel}: the draws beside a compute writer"
            );
        }
        assert!(beside.command_writes().may(CommandWrites::DISPATCH));
    });
}

/// [`inherited_buffers_frames`]' commands, its cold frames and the frames it measures after them.
const FRAME_COMMANDS: u64 = 8192;
const COLD: usize = 3;
const FRAMES: usize = 30;
/// Set, [`inherited_buffers_frames`]' executions take the host step (a barrier the CPU set and
/// cleared in the last command): the form such an execution took before it ran on the GPU.
const HOST_STEP_VAR: &str = "M2V_INHERITED_FRAMES_HOST_STEP";

/// `values` as "median / 90th percentile / largest".
fn spread(mut values: Vec<f64>) -> String {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    format!(
        "{:.2} / {:.2} / {:.2}",
        at(50),
        at(90),
        values[values.len() - 1]
    )
}

/// What frames of a game's GPU-driven pass cost on the device the cfg selects, without the
/// validation layer: report only, in release builds, to weigh the form before and after a change
/// ([`HOST_STEP_VAR`]). Each frame, on one queue, `ica_inherited` writes [`FRAME_COMMANDS`]
/// commands of a buffer that inherits the encoder's state and buffers, every other one an indexed
/// draw of one triangle and the rest resets, and a pass executes all of them, with the encoder's
/// buffers bound as the lane test binds them; the frame is committed and waited for. Printed as
/// `INHERITED-FRAMES` lines: the median, the 90th percentile and the largest of the [`FRAMES`]
/// after the first [`COLD`], of the frame's time (encoding to completion), the process's CPU time
/// over it (every thread's, the host step's among them) and its GPU time (`GPUStartTime` to
/// `GPUEndTime`); and the device's counters over them, a frame each.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn inherited_buffers_frames() {
    let host_step = std::env::var_os(HOST_STEP_VAR).is_some();
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let library = library(&device);
    let state = state(&device, &library);
    let writer = writer(&device, &library, "ica_inherited");
    let inputs = Inputs::new(&device);
    let icb = icb(&device, &commands(true), FRAME_COMMANDS);
    if host_step {
        let command = icb.indirect_render_command_at(FRAME_COMMANDS - 1).unwrap();
        command.set_barrier().unwrap();
        command.clear_barrier().unwrap();
    }
    // Every other command a draw of one of the triangles, the rest resets.
    let draws = (0..FRAME_COMMANDS as u32).flat_map(|c| match c % 2 {
        0 => [3 * (c / 2 % 16), 3, 1, 0, 0],
        _ => [0; 5],
    });
    let draws = buffer_with(&device, &words(draws)).unwrap();
    let target = target(&device);
    let queue = device.new_command_queue().unwrap();
    let (mut wall, mut cpu, mut gpu) = (Vec::new(), Vec::new(), Vec::new());
    let mut before = device.work_counts();
    for frame in 0..COLD + FRAMES {
        if frame == COLD {
            before = device.work_counts();
        }
        let cpu_at = process_cpu_us();
        let start = Instant::now();
        let command_buffer = queue.command_buffer().unwrap();
        encode(
            &command_buffer,
            (&writer, &state, &inputs),
            (&icb, &draws, &target),
            true,
        );
        complete(&command_buffer);
        if frame >= COLD {
            wall.push(start.elapsed().as_secs_f64() * 1e3);
            cpu.push((process_cpu_us() - cpu_at) / 1e3);
            let span =
                command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
            gpu.push(span * 1e3);
        }
    }
    let after = device.work_counts();
    let frames = FRAMES as f64;
    let per = |f: fn(&WorkCounts) -> u64| (f(&after) - f(&before)) as f64 / frames;
    let form = if host_step { "host step" } else { "GPU form" };
    eprintln!(
        "INHERITED-FRAMES {} {form}: {FRAME_COMMANDS} commands, half reset; frame ms {}; host CPU \
         ms {}; GPU ms {}",
        device.name(),
        spread(wall),
        spread(cpu),
        spread(gpu)
    );
    eprintln!(
        "INHERITED-FRAMES {} {form}, a frame: host steps {:.2} taking {:.2} ms, records read {:.0}, \
         record bytes copied {:.0}, GPU-form commands {:.0}, merged {:.0}",
        device.name(),
        per(|c| c.host_steps_recorded),
        per(|c| c.host_step_nanos) / 1e6,
        per(|c| c.indirect_records_read),
        per(|c| c.indirect_record_bytes_copied),
        per(|c| c.gpu_form_commands),
        per(|c| c.merged_draw_commands),
    );
}
