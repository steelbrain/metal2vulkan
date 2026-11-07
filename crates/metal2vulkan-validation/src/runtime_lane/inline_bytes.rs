//! Inline bytes (`setBytes:length:atIndex:` and its stage forms; `argument_table.rs`) on the device
//! the cfg selects, under the validation layer. Each call's bytes are copied at the call into the
//! command buffer's upload arena and bound where the function reads the index:
//!
//! - A kernel reads them through a `constant` pointer and through a `const device` one (the
//!   owned synthetic kernels below, which copy word `gid` of buffer 0 to word `gid` of buffer
//!   1): each dispatch reads the bytes of the call before it, so two calls in one encoder keep
//!   their own bytes; a buffer bound at the index replaces the bytes, and bytes set there replace
//!   the buffer; 4096 bytes, Metal's limit, are read whole, and 4097 are refused at the call;
//!   after `setBufferOffset:atIndex:` at the index the bytes read as zeros, as plain Metal reads
//!   them (an M4).
//! - A vertex function's attribute reads them as its vertex buffer (the owned fixture
//!   `ob3_vertex_input`, `vertex_strides.rs`): at the layout's own stride, and under
//!   `MTLBufferLayoutStrideDynamic` at stride 0 after the plain setter and at the stride form's
//!   stride, as plain Metal reads them; at `MTLAttributeStrideStatic`, `2^32 - 1`, vertex 0 reads
//!   the first element and every vertex past it zero, as an M4 read them.
//! - A fragment function reads them through a `const device` pointer and a `constant`
//!   reference (`cfb_ts_read_bytes` of the owned fixture `cfb_tile_shadows`).
//!
//! The object stage's form runs `mesh_draws`' object cases (`mesh_draws.rs`).

use super::vertex_strides::{drawn, pipeline, read_at, read_static};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, ComputeCommandEncoder, ComputePipelineState, Device, Error,
    Invalid, LoadAction, PipelineOptions, Problem, RenderCommandEncoder, RenderPassDescriptor,
    ResourceOptions, Size, StoreAction, TextureDescriptor,
};

/// The kernels: `inline_bytes_constant` reads buffer 0 through a `constant` pointer,
/// `inline_bytes_device` through a `const device` one; each thread copies its word to buffer 1.
const KERNELS: &str = r#"; Owned synthetic kernels: copy word gid of buffer 0 to buffer 1.
source_filename = "inline_bytes.metal"

define void @inline_bytes_constant(ptr addrspace(2) %in, ptr addrspace(1) %out, i32 %gid) {
entry:
  %g = zext i32 %gid to i64
  %p = getelementptr inbounds i32, ptr addrspace(2) %in, i64 %g
  %v = load i32, ptr addrspace(2) %p, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

define void @inline_bytes_device(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {
entry:
  %g = zext i32 %gid to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %g
  %v = load i32, ptr addrspace(1) %p, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

!air.kernel = !{!0, !6}
!0 = !{ptr @inline_bytes_constant, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!6 = !{ptr @inline_bytes_device, !1, !7}
!7 = !{!8, !4, !5}
!8 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
"#;

/// Metal's limit for inline bytes, in 32-bit words.
const LIMIT_WORDS: u32 = 1024;

/// `count` words from `first`, each `first + i` scrambled, little-endian.
fn words(first: u32, count: u32) -> Vec<u8> {
    (first..first + count)
        .flat_map(|i| i.wrapping_mul(0x9E37_79B9).to_le_bytes())
        .collect()
}

/// The state of kernel `name` on `device`.
fn kernel(device: &Device, name: &str) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(KERNELS).unwrap())
        .unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A buffer of `words` words, each `0xEEEEEEEE`.
fn out(device: &Device, words: u32) -> Buffer {
    buffer_with(device, &vec![0xEE; words as usize * 4]).unwrap()
}

/// A dispatch of `threads` threads on `e` writing buffer `out`.
fn dispatch(e: &mut ComputeCommandEncoder, out: &Buffer, threads: u32) {
    e.set_buffer(Some(out), 0, 1).unwrap();
    e.dispatch_threads(
        Size::new(threads.into(), 1, 1),
        Size::new(threads.min(16).into(), 1, 1),
    )
    .unwrap();
}

/// Each kernel reads the bytes of the call before its dispatch, and a buffer bound at the index
/// in place of bytes, or bytes in place of a buffer; 4096 bytes are read whole and 4097 refused;
/// an offset set at the index then reads zeros.
#[test]
fn kernels_read_each_calls_inline_bytes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        for name in ["inline_bytes_constant", "inline_bytes_device"] {
            let state = kernel(&device, name);
            let (a, b, c, d) = (
                words(0, LIMIT_WORDS),
                words(5000, 16),
                words(6000, 16),
                words(7000, 5),
            );
            let buffer = buffer_with(&device, &c).unwrap();
            let zeros = vec![0; 64];
            let outs = [LIMIT_WORDS, 16, 16, 5, 16].map(|n| out(&device, n));
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_bytes(&a, 0).unwrap();
            dispatch(&mut encoder, &outs[0], LIMIT_WORDS);
            let mut changed = b.clone();
            encoder.set_bytes(&changed, 0).unwrap();
            // The caller's memory after the call reaches no dispatch.
            changed.fill(0x55);
            dispatch(&mut encoder, &outs[1], 16);
            encoder.set_buffer(Some(&buffer), 0, 0).unwrap();
            dispatch(&mut encoder, &outs[2], 16);
            encoder.set_bytes(&d, 0).unwrap();
            dispatch(&mut encoder, &outs[3], 5);
            encoder.set_bytes(&words(8000, 16), 0).unwrap();
            encoder.set_buffer_offset(16, 0).unwrap();
            dispatch(&mut encoder, &outs[4], 16);
            assert_eq!(
                encoder.set_bytes(&vec![0; 4097], 0),
                Err(Error::Invalid(Invalid::new(
                    "set_bytes",
                    Problem::InlineBytesLength {
                        length: 4097,
                        limit: 4096,
                    },
                ))),
                "{name}: 4097 bytes"
            );
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            for (i, (out, want)) in outs.iter().zip([&a, &b, &c, &d, &zeros]).enumerate() {
                assert_eq!(
                    &read_buffer(out, 0, want.len() as u64).unwrap(),
                    want,
                    "{name}: dispatch {i}"
                );
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}

/// A vertex attribute reads inline bytes as its vertex buffer: at the layout's stride, and under
/// a dynamic layout at stride 0 after the plain setter and at the stride form's stride; at
/// `MTLAttributeStrideStatic` the draw is refused.
#[test]
fn vertex_attributes_read_inline_bytes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let bytes: Vec<u8> = (0..32u32).flat_map(|i| (100 + i).to_le_bytes()).collect();
        let unused = buffer_with(&device, &[0; 16]).unwrap();
        let fixed = pipeline(&device, Some(4));
        assert_eq!(
            drawn(&queue, &fixed, &unused, |e, _| e
                .set_vertex_bytes(&bytes, 0)),
            Ok(read_at(0, 1)),
            "the layout's stride"
        );
        let dynamic = pipeline(&device, None);
        let draw = |bind: &dyn Fn(&mut RenderCommandEncoder) -> Result<(), Error>| {
            drawn(&queue, &dynamic, &unused, |e, _| bind(e))
        };
        assert_eq!(
            draw(&|e| e.set_vertex_bytes(&bytes, 0)),
            Ok(read_at(0, 0)),
            "the plain form"
        );
        assert_eq!(
            draw(&|e| e.set_vertex_bytes_with_stride(&bytes, 8, 0)),
            Ok(read_at(0, 2)),
            "the stride form"
        );
        assert_eq!(
            draw(&|e| e.set_vertex_bytes_with_stride(&bytes, (1 << 32) + 12, 0)),
            Ok(read_at(0, 3)),
            "a stride of 2^32 + 12"
        );
        assert_eq!(
            draw(&|e| e.set_vertex_bytes_with_stride(&bytes, u64::MAX, 0)),
            Ok(read_static()),
            "MTLAttributeStrideStatic"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// A fragment function reads inline bytes through a `const device` pointer and a `constant`
/// reference: `cfb_ts_read_bytes` writes word `at.x + c.y * at.y + c.x` of buffer 0 to word
/// `c.y * 4 + c.x` of buffer 2 for each of the 4x4 pixels whose coordinates are multiples of 8.
#[test]
fn fragments_read_inline_bytes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("cfb_tile_shadows.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("cfb_ts_full"));
        descriptor.fragment = Some(function("cfb_ts_read_bytes"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 32, 32, false);
        target.usage = TextureUsage::RENDER_TARGET;
        let target = device.new_texture(&target).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target);
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::DontCare;
        // Words 3.. of the bytes, from `at.x = 3`, rows 5 words apart.
        let b = words(0, 3 + 5 * 4);
        let at: Vec<u8> = [3u32, 5].iter().flat_map(|w| w.to_le_bytes()).collect();
        let o = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_fragment_bytes(&b, 0).unwrap();
        encoder.set_fragment_bytes(&at, 1).unwrap();
        encoder.set_fragment_buffer(Some(&o), 0, 2).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let want: Vec<u8> = (0..4u32)
            .flat_map(|y| (0..4u32).map(move |x| 3 + y * 5 + x))
            .flat_map(|w| b[w as usize * 4..w as usize * 4 + 4].to_vec())
            .collect();
        assert_eq!(read_buffer(&o, 0, 64).unwrap(), want);
    }
    layer_reported_nothing_at_all(&messages);
}
