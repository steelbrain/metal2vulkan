//! The encoder trail's GPU forms (`encoder_trail.rs`, **GPU forms**): an execution of render
//! commands the GPU wrote (a blit's copy of commands the CPU encoded) through a range read from a
//! buffer, which `commit` records in its merged GPU form (`render/executions/gpu.rs`), is named by
//! its shape, and its scratch, which lies where the device places `Private` memory (device-local
//! without unified memory: `WorkCounts::gpu_form_device_local_bytes`), reads through the GPU's copy of it as the GPU
//! left it once the command buffer completed: the range the GPU copied, the merged draw's
//! corners, the arguments of the commands inside the range and the most of them, and status
//! words of no fault; of draws of primitives, and of indexed draws over many blocks of the merge
//! passes (`bind/replay/merged.rs`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). Its functions are `tob_rect` and `tob_solid` of
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::{bytes_of, library, pass};
use crate::{
    Buffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions,
    IndirectCommandBufferDescriptor, IndirectRenderCommand, PipelineOptions, RenderPipelineState,
    ResourceOptions, ShaderCacheOptions, TextureDescriptor,
};
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};

/// Each command's vertices and instances: the range read from the buffer runs commands 1 and 2.
const DRAWS: [(u32, u32); 4] = [(6, 1), (7, 2), (12, 3), (6, 4)];
const RANGE: [u32; 2] = [1, 2];

/// A device of the kind the cfg selects with its encoder trail on, its shader cache in `cache`.
fn trail_device(cache: &std::path::Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Integrated,
                DeviceKind::Discrete,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        encoder_trail: true,
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The state the commands inherit: `tob_rect`, then `tob_solid`, into `RGBA8Unorm`, for
/// indirect command buffers.
fn inherited_state(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/tob_renders.ll");
    let vertex = library.new_function_with_name("tob_rect").unwrap();
    let fragment = library.new_function_with_name("tob_solid").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The commands of `types`, `capacity` of them, each `encode`d by the CPU (given its index) with `tob_rect`'s corners
/// and `tob_solid`'s color bound, copied by a blit into a second buffer, whose records the GPU
/// then wrote, executed over an 8x8 target through `range`, which a buffer holds, with
/// `resident` resident to the vertex stage too: the command buffer completed, and the texts of
/// its trail's GPU forms, whose scratches lay in device-local memory; empty, printed, where the
/// device records no execution in its GPU form.
fn executed_through_range(
    device: &Device,
    (types, capacity): (IndirectCommandType, u64),
    encode: impl Fn(u64, &IndirectRenderCommand, (&Buffer, &Buffer)),
    (range, resident): ([u32; 2], &[&Buffer]),
) -> Vec<String> {
    let state = inherited_state(device);
    let shared = ResourceOptions::shared();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    let corners = device.new_buffer_with_bytes(&corners, shared).unwrap();
    let color = bytes_of(&[0.2f32, 0.4, 0.6, 1.0], f32::to_le_bytes);
    let color = device.new_buffer_with_bytes(&color, shared).unwrap();
    let range = bytes_of(&range, u32::to_le_bytes);
    let range = device.new_buffer_with_bytes(&range, shared).unwrap();
    let mut commands = IndirectCommandBufferDescriptor::new();
    commands.command_types = types;
    commands.inherit_pipeline_state = true;
    commands.max_vertex_buffer_bind_count = 1;
    commands.max_fragment_buffer_bind_count = 1;
    let icb = device
        .new_indirect_command_buffer(&commands, capacity, shared)
        .unwrap();
    for at in 0..capacity {
        let command = icb.indirect_render_command_at(at).unwrap();
        encode(at, &command, (&corners, &color));
    }
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let written = device
        .new_indirect_command_buffer(&commands, capacity, shared)
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_indirect_command_buffer(&icb, 0..capacity, &written, 0)
        .unwrap();
    blit.end_encoding().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let read = ResourceUsage::READ;
    for buffer in [&corners].into_iter().chain(resident.iter().copied()) {
        encoder
            .use_resource(buffer.as_resource(), read, RenderStages::VERTEX)
            .unwrap();
    }
    encoder
        .use_resource(color.as_resource(), read, RenderStages::FRAGMENT)
        .unwrap();
    encoder
        .execute_commands_in_buffer_indirect(&written, &range, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    let forms = command_buffer.trail_form_texts();
    let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    let counts = device.work_counts();
    let local = counts.gpu_form_device_local_bytes;
    eprintln!(
        "TRAIL-GPU-FORMS {}: GPU {:.3} ms, {local} scratch bytes device-local; {forms:?}",
        device.name(),
        gpu * 1e3
    );
    if forms.is_empty() && counts.gpu_form_commands == 0 {
        eprintln!("TRAIL-GPU-FORMS: the device records no execution in its GPU form");
    } else {
        // The scratch is `Private` memory, which lies where the device places it (`memory.rs`'s
        // `candidates`): device-local without unified memory; with it, a type that is not
        // device-local first where the device has one (a Radeon 780M's system memory beside its
        // carve-out), which counts nothing.
        let private = device.scratch_buffer(16).unwrap();
        let device_local = private.own_pages_device_local() == Some(true);
        assert!(
            device_local || device.properties().has_unified_memory,
            "a `Private` buffer outside device-local memory without unified memory"
        );
        assert_eq!(
            local > 0,
            device_local,
            "a GPU form's scratch lies where the device places `Private` memory"
        );
    }
    forms
}

/// Four commands of triangles the CPU wrote, copied by a blit into a second buffer, whose records
/// the GPU then wrote, executed through the range `RANGE` a buffer holds: `commit` records the
/// execution in its merged GPU form, and its trail names it with its range read from the buffer
/// at offset 0. Once the command buffer completed, its scratch holds the
/// range the GPU copied, the merged draw of commands 1 and 2's whole triangles (7 vertices make
/// 2 triangles, times 2 instances; 12 make 4, times 3: 48 corners) in its second draw, from
/// vertex `2^31`, the arguments of those two commands alone (7 times 2 and 12 times 3, the most
/// command 2's), and status words of no fault.
#[test]
fn a_merged_gpu_form_names_its_range_and_draws_as_its_scratch_holds_them() {
    let scratch = Scratch::new("trail-gpu-forms");
    let device = trail_device(&scratch.0);
    let encode =
        |at: u64, command: &IndirectRenderCommand, (corners, color): (&Buffer, &Buffer)| {
            let (vertices, instances) = DRAWS[at as usize];
            command.set_vertex_buffer(Some(corners), 0, 0).unwrap();
            command.set_fragment_buffer(Some(color), 0, 0).unwrap();
            command
                .draw_primitives(
                    PrimitiveType::Triangle,
                    0,
                    vertices.into(),
                    instances.into(),
                    0,
                )
                .unwrap();
        };
    let shape = (IndirectCommandType::DRAW, DRAWS.len() as u64);
    let forms = executed_through_range(&device, shape, encode, (RANGE, &[]));
    if forms.is_empty() {
        return;
    }
    let [form] = &forms[..] else {
        panic!("one GPU form, not {forms:?}");
    };
    let expected = [
        "an execution's GPU form, recorded at commit: merged into one draw, of primitive type \
         Triangle, over 4 command(s) from 0, its range read from a buffer at offset 0; read while \
         the work is pending: the range copied 1+2,",
        " the merged draws [0 vertices x 0 from vertex 0, instance 0; 48 vertices x 1 \
         from vertex 2147483648, instance 0],",
        " 2 argument(s) that draw, 50 vertices or indices times instances, the most command 2's: \
         12 x 3 from 0, instance 0",
        "; status bits 0x0, resident misses 0, replay faults 0x0",
    ];
    assert_eq!(form, &expected.concat());
}

/// The indexed execution's commands, the range read from the buffer that runs the first
/// [`INDEXED_RANGE`]`[1]` of them, and the 4-byte indices of the buffer they draw from.
const INDEXED_CAPACITY: u64 = 8162;
const INDEXED_RANGE: [u32; 2] = [0, 86];
const INDICES: u32 = 600;

/// [`INDEXED_CAPACITY`] commands of indexed triangles, of which command `c` of the first
/// [`INDEXED_RANGE`]`[1]` draws `6 (c + 1)` of [`INDICES`] 4-byte indices from the buffer's start
/// in one instance, and the rest nothing, executed through the range [`INDEXED_RANGE`] a buffer
/// holds (as a compositor's GPU-written draws run, `render/executions/gpu.rs`): `commit` records
/// the execution in its merged GPU form, over 90 blocks of 91 commands, and the scratch holds its
/// first draw of no instance (a total of 22446 corners holds no whole chunk) and its second of
/// every corner, the 86 commands' arguments, the most command 85's of 516 indices with its index
/// entry's 600, and status words of no fault.
#[test]
fn an_indexed_merged_gpu_form_through_a_range_draws_no_whole_chunk_below_one() {
    let scratch = Scratch::new("trail-gpu-forms-indexed");
    let device = trail_device(&scratch.0);
    let indices: Vec<u32> = (0..INDICES).map(|i| i % 6).collect();
    let indices = bytes_of(&indices, u32::to_le_bytes);
    let indices = (device.new_buffer_with_bytes(&indices, ResourceOptions::shared())).unwrap();
    let encode =
        |at: u64, command: &IndirectRenderCommand, (corners, color): (&Buffer, &Buffer)| {
            if at >= u64::from(INDEXED_RANGE[1]) {
                return;
            }
            command.set_vertex_buffer(Some(corners), 0, 0).unwrap();
            command.set_fragment_buffer(Some(color), 0, 0).unwrap();
            command
                .draw_indexed_primitives(
                    PrimitiveType::Triangle,
                    6 * (at + 1),
                    IndexType::UInt32,
                    &indices,
                    0,
                    1,
                    0,
                    0,
                )
                .unwrap();
        };
    let shape = (IndirectCommandType::DRAW_INDEXED, INDEXED_CAPACITY);
    let forms = executed_through_range(&device, shape, encode, (INDEXED_RANGE, &[&indices]));
    if forms.is_empty() {
        return;
    }
    let [form] = &forms[..] else {
        panic!("one GPU form, not {forms:?}");
    };
    let expected = [
        "an execution's GPU form, recorded at commit: merged into one draw, indexed, of primitive \
         type Triangle, over 8162 command(s) from 0, its range read from a buffer at offset 0; \
         read while the work is pending: the range copied 0+86,",
        " the merged draws [0 vertices x 0 from vertex 0, instance 0; 22446 vertices x 1 \
         from vertex 2147483648, instance 0],",
        " 86 argument(s) that draw, 22446 vertices or indices times instances, the most command \
         85's: 516 x 1 from 0, instance 0 (600 whole indices of 4 byte(s))",
        "; status bits 0x0, resident misses 0, replay faults 0x0",
    ];
    assert_eq!(form, &expected.concat());
}
