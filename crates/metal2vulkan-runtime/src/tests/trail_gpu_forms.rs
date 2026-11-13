//! The encoder trail's GPU forms (`encoder_trail.rs`, **GPU forms**): an execution of render
//! commands through a range read from a buffer, which `commit` records in its merged GPU form
//! (`render/executions/gpu.rs`), is named by its shape, and its scratch reads as the GPU left it
//! once the command buffer completed: the range the GPU copied, the merged draw's corners, the
//! arguments of the commands inside the range and the most of them, and status words of no fault.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). Its functions are `tob_rect` and `tob_solid` of
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::{bytes_of, library, pass};
use crate::{
    CommandBufferStatus, Device, DeviceKind, DeviceOptions, IndirectCommandBufferDescriptor,
    PipelineOptions, ResourceOptions, ShaderCacheOptions, TextureDescriptor,
};
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
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

/// Four commands of triangles the CPU wrote, executed through the range `RANGE` a buffer holds:
/// `commit` records the execution in its merged GPU form, and its trail names it with its range
/// read from the buffer at offset 0. Once the command buffer completed, its scratch holds the
/// range the GPU copied, the merged draw of commands 1 and 2's whole triangles (7 vertices make
/// 2 triangles, times 2 instances; 12 make 4, times 3: 48 corners) in its second draw, from
/// vertex `2^31`, the arguments of those two commands alone (7 times 2 and 12 times 3, the most
/// command 2's), and status words of no fault.
#[test]
fn a_merged_gpu_form_names_its_range_and_draws_as_its_scratch_holds_them() {
    let scratch = Scratch::new("trail-gpu-forms");
    let device = trail_device(&scratch.0);
    let library = library(&device, "engine/tob_renders.ll");
    let vertex = library.new_function_with_name("tob_rect").unwrap();
    let fragment = library.new_function_with_name("tob_solid").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.support_indirect_command_buffers = true;
    let state = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let shared = ResourceOptions::shared();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    let corners = device.new_buffer_with_bytes(&corners, shared).unwrap();
    let color = bytes_of(&[0.2f32, 0.4, 0.6, 1.0], f32::to_le_bytes);
    let color = device.new_buffer_with_bytes(&color, shared).unwrap();
    let range = bytes_of(&RANGE, u32::to_le_bytes);
    let range = device.new_buffer_with_bytes(&range, shared).unwrap();
    let mut commands = IndirectCommandBufferDescriptor::new();
    commands.command_types = IndirectCommandType::DRAW;
    commands.inherit_pipeline_state = true;
    commands.max_vertex_buffer_bind_count = 1;
    commands.max_fragment_buffer_bind_count = 1;
    let icb = device
        .new_indirect_command_buffer(&commands, DRAWS.len() as u64, shared)
        .unwrap();
    for (at, (vertices, instances)) in DRAWS.into_iter().enumerate() {
        let command = icb.indirect_render_command_at(at as u64).unwrap();
        command.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        command.set_fragment_buffer(Some(&color), 0, 0).unwrap();
        command
            .draw_primitives(
                PrimitiveType::Triangle,
                0,
                vertices.into(),
                instances.into(),
                0,
            )
            .unwrap();
    }
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let read = ResourceUsage::READ;
    encoder
        .use_resource(corners.as_resource(), read, RenderStages::VERTEX)
        .unwrap();
    encoder
        .use_resource(color.as_resource(), read, RenderStages::FRAGMENT)
        .unwrap();
    encoder
        .execute_commands_in_buffer_indirect(&icb, &range, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    let forms = command_buffer.trail_form_texts();
    eprintln!("TRAIL-GPU-FORMS {}: {forms:?}", device.name());
    if forms.is_empty() && device.work_counts().gpu_form_commands == 0 {
        eprintln!("TRAIL-GPU-FORMS: the device records no execution in its GPU form");
        return;
    }
    let [form] = &forms[..] else {
        panic!("one GPU form, not {forms:?}");
    };
    let expected = [
        "an execution's GPU form, recorded at commit: merged into one draw, of primitive type \
         Triangle, over 4 command(s) from 0, its range read from a buffer at offset 0; read while \
         the work is pending: the range copied 1+2,",
        " the merged draws [2147483646 vertices x 0 from vertex 0, instance 0; 48 vertices x 1 \
         from vertex 2147483648, instance 0],",
        " 2 argument(s) that draw, 50 vertices or indices times instances, the most command 2's: \
         12 x 3 from 0, instance 0",
        "; status bits 0x0, resident misses 0, replay faults 0x0",
    ];
    assert_eq!(form, &expected.concat());
}
