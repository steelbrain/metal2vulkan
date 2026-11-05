//! A render encoder's vertex buffers at a stride of their own (`runtime/render/vertex_strides.rs`)
//! on the device the cfg selects, under the validation layer. The owned fixture
//! `ob3_vertex_input` draws a point per vertex with rasterization disabled, each vertex writing
//! the `uint` attribute it fetched to `out[v]`; its pipeline here takes the attribute from buffer
//! 0 at offset 0 under a layout of `MTLBufferLayoutStrideDynamic`, so the bind's stride chooses
//! the words it reads. Buffer 0 holds `100 + i` at word `i`.
//!
//! - The stride form reads vertex `v` at `offset + v * stride`; so do the range form and the
//!   offset form, which sets both.
//! - A buffer bound without a stride (the plain setters, the plain offset form after a stride
//!   form, or `MTLAttributeStrideStatic`) under a dynamic layout reads at stride 0, as plain
//!   Metal does (an M4; an M3 Max and an M4), where Metal's debug layer stops the draw; a stride
//!   past 32 bits is refused at the setter.
//! - An executed indirect render command's vertex buffer set without a stride under a dynamic
//!   layout, on a command buffer whose descriptor supports dynamic attribute strides, refuses the
//!   commit the runtime's own gap (`IndirectCommandUnstridedBuffer`): plain Metal takes such a
//!   command and completes it, and its fetch reads the buffer at stride 2^32 - 1, which the
//!   runtime's 32-bit fetch cannot place.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    IndirectCommandType, PrimitiveType, RenderStages, ResourceUsage, VertexFormat,
};
use metal2vulkan::pipeline::{
    RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    VertexDescriptor,
};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, Device, Error, IndirectCommandBufferDescriptor,
    IndirectRenderCommand, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, RuntimeGap,
};
use std::path::Path;

/// The vertices each draw has.
const VERTICES: u32 = 8;

/// The fixture's pipeline: its vertex function, rasterization disabled, its attribute a `uint`
/// at offset 0 of buffer 0, whose layout's stride is `stride` (`None`: dynamic).
pub(super) fn pipeline(device: &Device, stride: Option<u32>) -> RenderPipelineState {
    pipeline_of(device, stride, false)
}

/// [`pipeline`], made with `supportIndirectCommandBuffers` where `indirect`.
fn pipeline_of(device: &Device, stride: Option<u32>, indirect: bool) -> RenderPipelineState {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/ob3_vertex_input.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = library.new_function_with_name("ob3_vertex_input").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    descriptor.support_indirect_command_buffers = indirect;
    let mut vertex = VertexDescriptor::new();
    vertex
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(stride));
    vertex
        .attributes
        .insert(0, VertexAttributeDescriptor::new(VertexFormat::UInt, 0, 0));
    descriptor.vertex_descriptor = Some(vertex);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The words each vertex wrote after a draw of `pipeline` in a 1x1 pass without attachments,
/// buffer 0 bound by `bind` first; or the refusal of the binding or the draw, which records
/// nothing.
pub(super) fn drawn(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    data: &Buffer,
    bind: impl FnOnce(&mut RenderCommandEncoder, &Buffer) -> Result<(), Error>,
) -> Result<Vec<u32>, Error> {
    let device = queue.device();
    let out = buffer_with(device, &[0u8; VERTICES as usize * 4]).unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&out), 0, 1).unwrap();
    let draw = bind(&mut encoder, data)
        .and_then(|()| encoder.draw_primitives(PrimitiveType::Point, 0, VERTICES, 1, 0));
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    draw?;
    Ok(read_buffer(&out, 0, u64::from(VERTICES) * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

/// The words vertex `v` reads at `first + v * step` words.
pub(super) fn read_at(first: u32, step: u32) -> Vec<u32> {
    (0..VERTICES).map(|v| 100 + first + v * step).collect()
}

/// Each stride form binds the stride a dynamic layout reads; a buffer bound without one reads at
/// stride 0, as plain Metal's does; a stride past 32 bits binds its low 32 bits.
#[test]
fn vertex_buffers_bind_the_strides_a_dynamic_layout_reads() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let pipeline = pipeline(&device, None);
        let queue = device.new_command_queue().unwrap();
        let words: Vec<u8> = (0..32u32).flat_map(|i| (100 + i).to_le_bytes()).collect();
        let data = buffer_with(&device, &words).unwrap();
        let draw = |bind: &dyn Fn(&mut RenderCommandEncoder, &Buffer) -> Result<(), Error>| {
            drawn(&queue, &pipeline, &data, bind)
        };
        assert_eq!(
            draw(&|e, b| e.set_vertex_buffer_with_stride(Some(b), 0, 8, 0)),
            Ok(read_at(0, 2)),
            "the stride form"
        );
        assert_eq!(
            draw(&|e, b| {
                e.set_vertex_buffer_with_stride(Some(b), 0, 8, 0)?;
                e.set_vertex_buffer_offset_with_stride(4, 12, 0)
            }),
            Ok(read_at(1, 3)),
            "the offset form with a stride"
        );
        assert_eq!(
            draw(&|e, b| e.set_vertex_buffers_with_strides(&[Some(b)], &[0], &[16], 0)),
            Ok(read_at(0, 4)),
            "the range form"
        );
        assert_eq!(
            draw(&|e, b| e.set_vertex_buffer(Some(b), 0, 0)),
            Ok(read_at(0, 0)),
            "the plain buffer form"
        );
        assert_eq!(
            draw(&|e, b| {
                e.set_vertex_buffer_with_stride(Some(b), 0, 8, 0)?;
                e.set_vertex_buffer(Some(b), 4, 0)
            }),
            Ok(read_at(1, 0)),
            "the plain buffer form after the stride form"
        );
        assert_eq!(
            draw(&|e, b| {
                e.set_vertex_buffer_with_stride(Some(b), 0, 8, 0)?;
                e.set_vertex_buffer_offset(4, 0)
            }),
            Ok(read_at(1, 0)),
            "the plain offset form after the stride form"
        );
        assert_eq!(
            draw(&|e, b| e.set_vertex_buffer_with_stride(Some(b), 0, u64::MAX, 0)),
            Ok(read_at(0, 0)),
            "MTLAttributeStrideStatic"
        );
        // A stride past 32 bits reads at its low 32 bits, as plain Metal reads it (an M4; Metal's
        // validation takes it).
        assert_eq!(
            draw(&|e, b| e.set_vertex_buffer_with_stride(Some(b), 0, (1 << 32) + 8, 0)),
            Ok(read_at(0, 2)),
            "a stride of 2^32 + 8"
        );
        assert_eq!(
            draw(&|e, b| e.set_vertex_buffer_with_stride(Some(b), 0, (1 << 33) + 4, 0)),
            Ok(read_at(0, 1)),
            "a stride of 2^33 + 4"
        );
        assert_eq!(
            draw(&|e, b| e.set_vertex_buffer_with_stride(Some(b), 0, 1 << 32, 0)),
            Ok(read_at(0, 0)),
            "a stride of 2^32"
        );
        assert_eq!(
            draw(&|e, b| {
                e.set_vertex_buffer_with_stride(Some(b), 0, 8, 0)?;
                e.set_vertex_buffer_offset_with_stride(4, (1 << 63) + 12, 0)
            }),
            Ok(read_at(1, 3)),
            "the offset form with a stride of 2^63 + 12"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// The words each vertex wrote after an indirect render command drew `pipeline` (made with
/// `supportIndirectCommandBuffers`) in a 1x1 pass without attachments, its buffer 0 bound by
/// `bind`; or the commit's refusal.
fn executed(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    data: &Buffer,
    bind: impl FnOnce(&IndirectRenderCommand, &Buffer) -> Result<(), Error>,
) -> Result<Vec<u32>, Error> {
    let device = queue.device();
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.support_dynamic_attribute_stride = true;
    let icb = device
        .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
        .unwrap();
    let out = buffer_with(device, &[0u8; VERTICES as usize * 4]).unwrap();
    let command = icb.indirect_render_command_at(0).unwrap();
    command.set_render_pipeline_state(pipeline).unwrap();
    command.set_vertex_buffer(Some(&out), 0, 1).unwrap();
    bind(&command, data).unwrap();
    command
        .draw_primitives(PrimitiveType::Point, 0, VERTICES.into(), 1, 0)
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    for (buffer, usage) in [(data, ResourceUsage::READ), (&out, ResourceUsage::WRITE)] {
        encoder
            .use_resource(buffer.as_resource(), usage, RenderStages::VERTEX)
            .unwrap();
    }
    encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit()?;
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    Ok(read_buffer(&out, 0, u64::from(VERTICES) * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

/// An executed indirect render command's vertex buffer set without a stride under a dynamic
/// layout, on a command buffer that supports dynamic attribute strides, refuses the commit the
/// runtime's own gap (`IndirectCommandUnstridedBuffer`), where plain Metal takes the command and
/// its fetch reads the buffer at stride 2^32 - 1; the same command with the stride form draws as
/// the encoder's draw at that stride.
#[test]
fn an_executed_command_s_unstrided_vertex_buffer_is_refused() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let pipeline = pipeline_of(&device, None, true);
        let queue = device.new_command_queue().unwrap();
        let words: Vec<u8> = (0..32u32).flat_map(|i| (100 + i).to_le_bytes()).collect();
        let data = buffer_with(&device, &words).unwrap();
        assert_eq!(
            executed(&queue, &pipeline, &data, |c, b| c.set_vertex_buffer(
                Some(b),
                0,
                0
            )),
            Err(Error::NotImplemented(
                RuntimeGap::IndirectCommandUnstridedBuffer { index: 0 }
            )),
            "the plain form"
        );
        assert_eq!(
            executed(&queue, &pipeline, &data, |c, b| {
                c.set_vertex_buffer_with_stride(Some(b), 4, 12, 0)
            }),
            Ok(read_at(1, 3)),
            "the stride form"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// A buffer of `length` bytes, more than `2^32`, for a fetch past 4 GiB: on macOS a no-copy
/// buffer over fresh anonymous memory, with the mapping it holds; elsewhere device memory. Only
/// the GPU hazards make one (`m2v_gpu_hazards`).
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
pub(super) struct Large {
    pub(super) buffer: Buffer,
    #[cfg(target_os = "macos")]
    _mapping: super::host_memory::Mapping,
}

#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
impl Large {
    /// The buffer on `device`; `None`, printed, where its `max_buffer_length` holds no such
    /// buffer (RADV, ANV and lavapipe hold at most `2^32` bytes).
    pub(super) fn new(device: &Device, length: u64) -> Option<Large> {
        let max = device.properties().max_buffer_length;
        if length > max {
            eprintln!(
                "skipped: a buffer of {length} bytes on {}: max_buffer_length {max}",
                device.name()
            );
            return None;
        }
        #[cfg(target_os = "macos")]
        {
            let mapping = super::host_memory::Mapping::anonymous(length as usize);
            // SAFETY: the mapping is fresh private memory the value owns, unmapped after the
            // buffer is dropped (the fields' order), once every command buffer that reads it
            // completed (each test waits for its own).
            let buffer = unsafe {
                super::host_memory::no_copy(
                    device,
                    mapping.pointer(),
                    length as usize,
                    ResourceOptions::shared(),
                    None,
                )
            }
            .unwrap();
            Some(Large {
                buffer,
                _mapping: mapping,
            })
        }
        #[cfg(not(target_os = "macos"))]
        Some(Large {
            buffer: device
                .new_buffer(length, ResourceOptions::private())
                .unwrap(),
        })
    }
}

/// A draw that may read a pulled attribute past `2^32` bytes of its buffer's bind offset, in a
/// buffer that runs past them, is refused `NotImplemented(VertexFetchPast4GiB)`, naming the
/// buffer index and the furthest byte: the compile layer's 32-bit displacement would wrap where
/// Metal reads at 64 bits (`runtime/render/vertex_reach.rs`). The fixture's `uint` at a static
/// stride of 6 is pulled (fixed-function input takes no stride its alignment does not divide).
/// In a buffer of `2^32 + 4096` bytes: the encoder's draw of vertex 715827883 (bytes `2^32 + 2`
/// to `2^32 + 6`) is refused, and so is an indexed draw of 32-bit indices, which may name any
/// vertex; a draw of vertices 0 to 7 and an indexed draw of 16-bit indices are encoded; an
/// indirect command buffer's command drawing vertex 715827883 refuses the commit. A draw of
/// vertices 0 to 7, whose furthest byte lies below `2^32`, draws from the buffer as from any
/// other: each vertex `v` reads the word `100 + v` written at byte `6 v`. Skipped, printed, where
/// the device's `max_buffer_length` holds no such buffer (RADV, ANV and lavapipe hold at most
/// `2^32` bytes). A GPU hazard (`m2v_gpu_hazards`): the buffer is 4 GiB and 4 KiB, device memory or
/// imported host memory.
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
#[test]
fn vertex_fetches_past_4_gib_of_a_buffer_are_refused() {
    use metal2vulkan::metal::IndexType;
    const LENGTH: u64 = (1 << 32) + 4096;
    const FAR: u32 = 715_827_883;
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let Some(large) = Large::new(&device, LENGTH) else {
            return;
        };
        let big = &large.buffer;
        let pipeline = pipeline(&device, Some(6));
        let queue = device.new_command_queue().unwrap();
        let out = buffer_with(&device, &[0u8; VERTICES as usize * 4]).unwrap();
        let short: Vec<u8> = (0..8u16).flat_map(u16::to_le_bytes).collect();
        let long: Vec<u8> = (0..8u32).flat_map(u32::to_le_bytes).collect();
        let (short, long) = (
            buffer_with(&device, &short).unwrap(),
            buffer_with(&device, &long).unwrap(),
        );
        let past = |reach: u64| {
            Err(Error::NotImplemented(RuntimeGap::VertexFetchPast4GiB {
                buffer_index: 0,
                reach,
            }))
        };
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&out), 0, 1).unwrap();
        encoder.set_vertex_buffer(Some(big), 0, 0).unwrap();
        assert_eq!(
            encoder.draw_primitives(PrimitiveType::Point, FAR, 1, 1, 0),
            past(u64::from(FAR) * 6 + 4),
            "a draw of vertex {FAR}"
        );
        assert_eq!(
            encoder.draw_primitives(PrimitiveType::Point, 0, VERTICES, 1, 0),
            Ok(()),
            "a draw of the first vertices"
        );
        assert_eq!(
            encoder.draw_indexed_primitives(
                PrimitiveType::Point,
                8,
                IndexType::UInt16,
                &short,
                0,
                1,
                0,
                0
            ),
            Ok(()),
            "an indexed draw of 16-bit indices"
        );
        assert_eq!(
            encoder.draw_indexed_primitives(
                PrimitiveType::Point,
                8,
                IndexType::UInt32,
                &long,
                0,
                1,
                0,
                0
            ),
            past(u64::from(u32::MAX) * 6 + 4),
            "an indexed draw of 32-bit indices"
        );
        encoder.end_encoding().unwrap();
        drop(command_buffer);
        // A draw below `2^32` draws: the words `100 + v` at bytes `6 v`, blitted in.
        let words: Vec<u8> = (0..VERTICES)
            .flat_map(|v| {
                let mut word = (100 + v).to_le_bytes().to_vec();
                word.extend([0, 0]);
                word
            })
            .collect();
        let words = buffer_with(&device, &words).unwrap();
        let fill = queue.command_buffer().unwrap();
        let mut blit = fill.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&words, 0, big, 0, words.length())
            .unwrap();
        blit.end_encoding().unwrap();
        fill.commit().unwrap();
        fill.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(fill.status(), CommandBufferStatus::Completed);
        assert_eq!(
            drawn(&queue, &pipeline, big, |e, b| e.set_vertex_buffer(
                Some(b),
                0,
                0
            )),
            Ok(read_at(0, 1)),
            "a draw of the first vertices of a buffer past 4 GiB"
        );
        // An indirect command buffer's command, which inherits the encoder's buffers.
        let pipeline = pipeline_of(&device, Some(6), true);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.inherit_buffers = true;
        let icb = device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
            .unwrap();
        let command = icb.indirect_render_command_at(0).unwrap();
        command.set_render_pipeline_state(&pipeline).unwrap();
        command
            .draw_primitives(PrimitiveType::Point, FAR.into(), 1, 1, 0)
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_vertex_buffer(Some(&out), 0, 1).unwrap();
        encoder.set_vertex_buffer(Some(big), 0, 0).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        assert_eq!(
            command_buffer.commit(),
            past(u64::from(FAR) * 6 + 4),
            "an executed command's draw of vertex {FAR}"
        );
        eprintln!(
            "vertex fetches past 4 GiB of a buffer refused, and below it drawn, on {}",
            device.name()
        );
        drop((command_buffer, icb, large));
    }
    layer_reported_nothing_at_all(&messages);
}
