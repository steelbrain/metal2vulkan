//! A kernel's `[[stage_in]]` layouts of `MTLBufferLayoutStrideDynamic` on the device the cfg
//! selects, under the validation layer. The owned fixture `kernel_stage_in`'s `stage_in_fetch`
//! reports, per thread of a 5 by 3 grid, the attributes it fetched: a `float3` at offset 4 and a
//! `half2` at offset 16 of layout 1 (stepping by the thread's x), a `ushort4` of layout 2 (by its
//! y), and a constant `char` of layout 3. The compute encoder binds the source buffers with
//! `set_buffer_with_stride` and its forms. Metal's answer, as an M4 reads it:
//!
//! - a dynamic layout reads element `e` at `e * stride` from the bind offset, byte-exact at any
//!   alignment, for the stride the buffer was last bound with by the stride form, the range form
//!   or the offset form;
//! - a buffer bound without a stride (a plain setter, or the plain offset form after a stride
//!   form) reads at stride 0, as Metal without its validation layer does (with it, every element
//!   past the first reads as zeros);
//! - a layout of a fixed stride reads at its own, whatever stride the buffer is bound with.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
use metal2vulkan::metal::IndirectCommandType;
use metal2vulkan::metal::{AttributeFormat, StepFunction};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, ComputeCommandEncoder, ComputePipelineState, Device,
    PipelineOptions, Region, Size,
};
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
use metal2vulkan_runtime::{Error, IndirectCommandBufferDescriptor, ResourceOptions, RuntimeGap};
use std::path::Path;

/// The grid: 5 by 3 threads.
const W: u32 = 5;
const H: u32 = 3;

/// The elements each source buffer holds.
const ELEMENTS: u32 = 16;

/// The fixture's pipeline: its layouts 1 and 2 of `MTLBufferLayoutStrideDynamic` where
/// `dynamic`, else of 24 and 8 bytes; layout 3 constant.
fn pipeline(device: &Device, dynamic: bool) -> ComputePipelineState {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/kernel_stage_in.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = library.new_function_with_name("stage_in_fetch").unwrap();
    let mut input = StageInputDescriptor::new();
    for (index, format, offset, buffer) in [
        (0, AttributeFormat::Float3, 4, 1),
        (1, AttributeFormat::UShort4, 0, 2),
        (2, AttributeFormat::Char, 5, 3),
        (3, AttributeFormat::Half2, 16, 1),
    ] {
        input.attributes.insert(
            index,
            StageInputAttributeDescriptor::new(format, offset, buffer),
        );
    }
    for (buffer, stride, step) in [
        (1, 24, StepFunction::ThreadPositionInGridX),
        (2, 8, StepFunction::ThreadPositionInGridY),
        (3, 16, StepFunction::Constant),
    ] {
        let mut layout = StageInputLayoutDescriptor::new((!dynamic).then_some(stride));
        layout.step_function = step;
        // Metal requires a constant step's rate to be 0.
        layout.step_rate = u32::from(step != StepFunction::Constant);
        input.layouts.insert(buffer, layout);
    }
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.stage_input = Some(input);
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Layout 1's element `e` at stride 24: `(10e + 1, 10e + 2, 10e + 3)` as `float` at offset 4 and
/// the `half` bits `(0x3c00 + e, 0x4000 + e)` at offset 16; the rest `0xaa`.
fn layout1() -> Vec<u8> {
    let mut b = vec![0xaa; (24 * ELEMENTS) as usize];
    for e in 0..ELEMENTS {
        let at = (24 * e) as usize;
        for k in 0..3u32 {
            let v = (10 * e + k + 1) as f32;
            b[at + 4 + 4 * k as usize..][..4].copy_from_slice(&v.to_le_bytes());
        }
        b[at + 16..][..2].copy_from_slice(&(0x3c00 + e as u16).to_le_bytes());
        b[at + 18..][..2].copy_from_slice(&(0x4000 + e as u16).to_le_bytes());
    }
    b
}

/// Layout 2's element `e` at stride 8: `(e, e + 100, e + 200, 0xffff - e)` as `ushort4`.
fn layout2() -> Vec<u8> {
    let mut b = Vec::new();
    for e in 0..ELEMENTS as u16 {
        for v in [e, e + 100, e + 200, 0xffff - e] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    b
}

/// Layout 3: the constant `char` -7 at offset 5.
fn layout3() -> Vec<u8> {
    let mut b = vec![0x11; 16];
    b[5] = (-7i8) as u8;
    b
}

/// What thread `t` reports where it reads layout 1 at byte `a1` and layout 2 at byte `a2`: the
/// bytes there, whatever their alignment.
fn expected_at(a1: usize, a2: usize, t: u32) -> [u32; 12] {
    let (l1, l2) = (layout1(), layout2());
    let word = |b: &[u8], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    let half = |b: &[u8], at: usize| u32::from(u16::from_le_bytes([b[at], b[at + 1]]));
    [
        word(&l1, a1 + 4),
        word(&l1, a1 + 8),
        word(&l1, a1 + 12),
        1.0f32.to_bits(),
        half(&l2, a2),
        half(&l2, a2 + 2),
        half(&l2, a2 + 4),
        half(&l2, a2 + 6),
        (-7i32) as u32,
        half(&l1, a1 + 16),
        half(&l1, a1 + 18),
        t,
    ]
}

/// The source buffers (layouts 1 to 3) and the output.
struct Sources {
    layouts: [Buffer; 3],
    out: Buffer,
}

impl Sources {
    fn new(device: &Device) -> Sources {
        Sources {
            layouts: [layout1(), layout2(), layout3()].map(|b| buffer_with(device, &b).unwrap()),
            out: buffer_with(device, &vec![0u8; (48 * W * H) as usize]).unwrap(),
        }
    }
}

/// The words `pipeline` writes in one dispatch of the grid, its source buffers bound by `bind`
/// after the output.
fn dispatched(
    device: &Device,
    pipeline: &ComputePipelineState,
    bind: impl FnOnce(&mut ComputeCommandEncoder, &Sources),
) -> Vec<u32> {
    let sources = Sources::new(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline).unwrap();
    encoder.set_buffer(Some(&sources.out), 0, 0).unwrap();
    bind(&mut encoder, &sources);
    encoder
        .set_stage_in_region(Region::new_3d(0, 0, 0, W.into(), H.into(), 1))
        .unwrap();
    encoder
        .dispatch_threads(Size::new(W.into(), H.into(), 1), Size::new(2, 2, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&sources.out, 0, u64::from(48 * W * H))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Require that every thread of `words` read layout 1 at stride `s1` and layout 2 at `s2`.
fn read_at(what: &str, words: &[u32], s1: u32, s2: u32) {
    for y in 0..H {
        for x in 0..W {
            let t = y * W + x;
            let at = (12 * t) as usize;
            assert_eq!(
                words[at..at + 12],
                expected_at((s1 * x) as usize, (s2 * y) as usize, t),
                "{what}: thread ({x}, {y})"
            );
        }
    }
}

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A dynamic layout reads the stride its buffer was bound with, by each of the three forms, the
/// layouts' own strides and others (none, unaligned, a halfword's); a plain setter, and the plain
/// offset form after a stride form, read at stride 0.
#[test]
fn a_dynamic_layout_reads_the_bound_stride_as_metal() {
    on_device(|device| {
        let pipeline = pipeline(device, true);
        for (s1, s2) in [(24, 8), (0, 16), (12, 2), (6, 0), (25, 3)] {
            let words = dispatched(device, &pipeline, |e, s| {
                for (index, (buffer, stride)) in s.layouts.iter().zip([s1, s2, 7]).enumerate() {
                    e.set_buffer_with_stride(Some(buffer), 0, stride.into(), index as u32 + 1)
                        .unwrap();
                }
            });
            read_at(&format!("stride form {s1}, {s2}"), &words, s1, s2);
            let words = dispatched(device, &pipeline, |e, s| {
                let buffers: Vec<Option<&Buffer>> = s.layouts.iter().map(Some).collect();
                e.set_buffers_with_strides(&buffers, &[0; 3], &[s1.into(), s2.into(), 7], 1)
                    .unwrap();
            });
            read_at(&format!("range form {s1}, {s2}"), &words, s1, s2);
            let words = dispatched(device, &pipeline, |e, s| {
                for (index, (buffer, stride)) in s.layouts.iter().zip([s1, s2, 7]).enumerate() {
                    let index = index as u32 + 1;
                    e.set_buffer(Some(buffer), 0, index).unwrap();
                    e.set_buffer_offset_with_stride(0, stride.into(), index)
                        .unwrap();
                }
            });
            read_at(&format!("offset form {s1}, {s2}"), &words, s1, s2);
        }
        let words = dispatched(device, &pipeline, |e, s| {
            for (index, buffer) in s.layouts.iter().enumerate() {
                e.set_buffer(Some(buffer), 0, index as u32 + 1).unwrap();
            }
        });
        read_at("plain setter", &words, 0, 0);
        let words = dispatched(device, &pipeline, |e, s| {
            for (index, buffer) in s.layouts.iter().enumerate() {
                let index = index as u32 + 1;
                e.set_buffer_with_stride(Some(buffer), 0, 24, index)
                    .unwrap();
                e.set_buffer_offset(0, index).unwrap();
            }
        });
        read_at("plain offset form after a stride", &words, 0, 0);
    });
}

/// A stride past 32 bits reads at its low 32 bits, by each of the three forms, as plain Metal
/// reads it (an M4 and an M3 Max: `2^32 + 8` at 8, `2^33 + 4` at 4, `2^32` at 0; Metal's validation
/// takes them).
#[test]
fn a_stride_past_32_bits_reads_at_its_low_32_bits() {
    on_device(|device| {
        let pipeline = pipeline(device, true);
        for (s1, s2) in [(24u32, 8u32), (0, 16), (12, 2)] {
            let wide = |s: u32, k: u64| (k << 32) | u64::from(s);
            let words = dispatched(device, &pipeline, |e, s| {
                for (index, (buffer, stride)) in s.layouts.iter().zip([s1, s2, 7]).enumerate() {
                    let stride = wide(stride, index as u64 + 1);
                    e.set_buffer_with_stride(Some(buffer), 0, stride, index as u32 + 1)
                        .unwrap();
                }
            });
            read_at(
                &format!("stride form {s1}, {s2} past 32 bits"),
                &words,
                s1,
                s2,
            );
            let words = dispatched(device, &pipeline, |e, s| {
                let buffers: Vec<Option<&Buffer>> = s.layouts.iter().map(Some).collect();
                let strides = [wide(s1, 1 << 31), wide(s2, 2), wide(7, 3)];
                e.set_buffers_with_strides(&buffers, &[0; 3], &strides, 1)
                    .unwrap();
            });
            read_at(
                &format!("range form {s1}, {s2} past 32 bits"),
                &words,
                s1,
                s2,
            );
            let words = dispatched(device, &pipeline, |e, s| {
                for (index, (buffer, stride)) in s.layouts.iter().zip([s1, s2, 7]).enumerate() {
                    let index = index as u32 + 1;
                    e.set_buffer(Some(buffer), 0, index).unwrap();
                    e.set_buffer_offset_with_stride(0, wide(stride, 5), index)
                        .unwrap();
                }
            });
            read_at(
                &format!("offset form {s1}, {s2} past 32 bits"),
                &words,
                s1,
                s2,
            );
        }
    });
}

/// A layout of a fixed stride reads at its own whatever stride its buffer is bound with.
#[test]
fn a_fixed_layout_reads_its_own_stride_whatever_the_bind() {
    on_device(|device| {
        let pipeline = pipeline(device, false);
        let words = dispatched(device, &pipeline, |e, s| {
            for (index, (buffer, stride)) in s.layouts.iter().zip([32, 4, 12]).enumerate() {
                e.set_buffer_with_stride(Some(buffer), 0, stride, index as u32 + 1)
                    .unwrap();
            }
        });
        read_at("a fixed layout under bound strides", &words, 24, 8);
    });
}

/// The owned fixture `ob3_stage_input` (each thread writes the `uint` attribute it fetched to
/// `out[t]`, buffer 0), its attribute at offset 0 of layout 1, stepping by the thread's x at a
/// fixed stride of 6, made with `supportIndirectCommandBuffers`.
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
fn strided_pipeline(device: &Device) -> ComputePipelineState {
    stage_in_pipeline(device, StepFunction::ThreadPositionInGridX)
}

/// `strided_pipeline` stepping by `step` (a constant step at rate 0, which reads the
/// attribute's offset alone).
pub(super) fn stage_in_pipeline(device: &Device, step: StepFunction) -> ComputePipelineState {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/ob3_stage_input.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = library.new_function_with_name("ob3_stage_input").unwrap();
    let mut input = StageInputDescriptor::new();
    input.attributes.insert(
        0,
        StageInputAttributeDescriptor::new(AttributeFormat::UInt, 0, 1),
    );
    let mut layout = StageInputLayoutDescriptor::new(Some(6));
    layout.step_function = step;
    layout.step_rate = u32::from(step != StepFunction::Constant);
    input.layouts.insert(1, layout);
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.stage_input = Some(input);
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A dispatch whose kernel may fetch a `[[stage_in]]` attribute past `2^32` bytes of its
/// buffer's bind offset, in a buffer that runs past them, is refused
/// `NotImplemented(StageInFetchPast4GiB)`, naming the buffer index and the furthest byte: the
/// compile layer's 32-bit displacement would wrap where Metal reads at 64 bits
/// (`runtime/fetch_reach.rs`). `strided_pipeline` in a buffer of `2^32 + 4096` bytes, whose
/// bytes `6 t` hold the word `100 + t`: a dispatch of 8 threads over a region of the first 8
/// elements, and one over a region of size 0 there (no clamp), read those words as from any
/// buffer; a dispatch of one thread over a region from element 715827883 (bytes `2^32 + 2` to
/// `2^32 + 6`) is refused, and an indirect command buffer's command of it refuses the commit.
/// Skipped, printed, where the device's `max_buffer_length` holds no such buffer (RADV, ANV and
/// lavapipe hold at most `2^32` bytes). A GPU hazard (`m2v_gpu_hazards`): the buffer is 4 GiB and
/// 4 KiB, device memory or imported host memory.
#[cfg(any(m2v_cpu_icd, m2v_gpu_hazards))]
#[test]
fn stage_in_fetches_past_4_gib_of_a_buffer_are_refused() {
    const LENGTH: u64 = (1 << 32) + 4096;
    const FAR: u64 = 715_827_883;
    const THREADS: u32 = 8;
    on_device(|device| {
        let Some(large) = super::vertex_strides::Large::new(device, LENGTH) else {
            return;
        };
        let big = &large.buffer;
        let pipeline = strided_pipeline(device);
        let queue = device.new_command_queue().unwrap();
        let words: Vec<u8> = (0..THREADS)
            .flat_map(|t| {
                let mut word = (100 + t).to_le_bytes().to_vec();
                word.extend([0, 0]);
                word
            })
            .collect();
        let words = buffer_with(device, &words).unwrap();
        let fill = queue.command_buffer().unwrap();
        let mut blit = fill.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&words, 0, big, 0, words.length())
            .unwrap();
        blit.end_encoding().unwrap();
        fill.commit().unwrap();
        fill.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(fill.status(), CommandBufferStatus::Completed);
        let out = buffer_with(device, &[0u8; THREADS as usize * 4]).unwrap();
        let past = |reach: u64| {
            Error::NotImplemented(RuntimeGap::StageInFetchPast4GiB {
                buffer_index: 1,
                reach,
            })
        };
        // A dispatch of `threads` over the region of `size` from `origin`: the words it wrote,
        // or its refusal, which records nothing.
        let run = |origin: u64, size: u64, threads: u64| -> Result<Vec<u32>, Error> {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&pipeline).unwrap();
            encoder.set_buffer(Some(&out), 0, 0).unwrap();
            encoder.set_buffer(Some(big), 0, 1).unwrap();
            encoder
                .set_stage_in_region(Region::new_3d(origin, 0, 0, size, 1, 1))
                .unwrap();
            let dispatched = encoder.dispatch_threads(Size::new(threads, 1, 1), Size::new(1, 1, 1));
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            dispatched?;
            Ok(read_buffer(&out, 0, u64::from(THREADS) * 4)
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect())
        };
        let first: Vec<u32> = (0..THREADS).map(|t| 100 + t).collect();
        let threads = u64::from(THREADS);
        assert_eq!(
            run(0, threads, threads),
            Ok(first.clone()),
            "a region of the first elements"
        );
        assert_eq!(
            run(0, 0, threads),
            Ok(first),
            "a region of size 0 at the first elements"
        );
        assert_eq!(
            run(FAR, 1, 1),
            Err(past(FAR * 6 + 4)),
            "a region from element {FAR}"
        );
        // An indirect command buffer's command, which inherits the encoder's buffers.
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
        descriptor.inherit_buffers = true;
        let icb = device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
            .unwrap();
        let command = icb.indirect_compute_command_at(0).unwrap();
        command.set_compute_pipeline_state(&pipeline).unwrap();
        command
            .set_stage_in_region(Region::new_3d(FAR, 0, 0, 1, 1, 1))
            .unwrap();
        command
            .concurrent_dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_buffer(Some(&out), 0, 0).unwrap();
        encoder.set_buffer(Some(big), 0, 1).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        assert_eq!(
            command_buffer.commit(),
            Err(past(FAR * 6 + 4)),
            "an executed command's dispatch from element {FAR}"
        );
        eprintln!(
            "stage-in fetches past 4 GiB of a buffer refused, and below it run, on {}",
            device.name()
        );
        drop((command_buffer, icb, large));
    });
}
