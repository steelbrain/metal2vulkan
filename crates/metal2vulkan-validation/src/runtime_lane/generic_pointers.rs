//! MSL 4.1's generic pointers, on the device the cfg selects, under the validation layer: the owned
//! fixture `generic_pointers`'s `generic_spaces` (generic pointers into the device, thread and
//! threadgroup spaces, read, written and asked their space) and `generic_null` (generic pointers
//! the run keeps or makes null, asked their space) answer what the M3 Max answers
//! (`generic_pointers.m`).
//!
//! Each kernel runs eight threads in one threadgroup over `in[i] = i` (16 words); the fixture's
//! header says what each word is. Every answer is a small integer or a half, exact in `float`.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Size,
};
use std::path::Path;

/// The threads, all in one threadgroup.
const THREADS: usize = 8;

/// What the M3 Max answers (macOS 27.0.1) for `generic_spaces`, eight words a thread.
#[rustfmt::skip]
const SPACES: [[f32; 8]; THREADS] = [
    [100.0, 200.5, 302.0, 100.0, 3.5, 3.0, 2.0, 1.0],
    [101.0, 202.5, 304.0, 109.0, 3.5, 3.0, 2.0, 1.0],
    [102.0, 204.5, 306.0, 102.0, 3.5, 3.0, 2.0, 1.0],
    [103.0, 206.5, 308.0, 111.0, 3.5, 3.0, 2.0, 1.0],
    [104.0, 204.5, 310.0, 104.0, 3.5, 3.0, 2.0, 1.0],
    [105.0, 206.5, 312.0, 113.0, 3.5, 3.0, 2.0, 1.0],
    [106.0, 208.5, 314.0, 106.0, 3.5, 3.0, 2.0, 1.0],
    [107.0, 210.5, 300.0, 115.0, 3.5, 3.0, 2.0, 1.0],
];

/// What the M3 Max answers for `generic_null`, four words a thread: the spaces of the device,
/// thread and threadgroup pointers it kept (1, 2, 3) or made null (4), then of a null pointer.
#[rustfmt::skip]
const NULL: [[f32; 4]; THREADS] = [
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 4.0, 4.0],
];

/// The fixture's `entry`, made into a pipeline.
fn pipeline(device: &Device, entry: &str) -> ComputePipelineState {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/generic_pointers.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = library.new_function_with_name(entry).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// The first `words` floats `entry` writes to buffer 0, over `in[i] = i` at buffer 1.
fn dispatched(device: &Device, entry: &str, words: usize) -> Vec<f32> {
    let pipeline = pipeline(device, entry);
    let input: Vec<u8> = (0..16u16)
        .flat_map(|i| f32::from(i).to_le_bytes())
        .collect();
    let out = buffer_with(device, &vec![0; 4 * words]).unwrap();
    let input = buffer_with(device, &input).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&input), 0, 1).unwrap();
    encoder
        .dispatch_threads(
            Size::new(THREADS as u64, 1, 1),
            Size::new(THREADS as u64, 1, 1),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, 4 * words as u64)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// Generic pointers of one space each, and generic pointers the run makes null, read, write and
/// answer their spaces as Metal does.
#[test]
fn generic_pointers_answer_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let spaces = dispatched(&device, "generic_spaces", 8 * THREADS);
        for (t, (got, want)) in spaces.chunks(8).zip(SPACES.iter()).enumerate() {
            assert_eq!(got, want, "generic_spaces: thread {t}");
        }
        let null = dispatched(&device, "generic_null", 4 * THREADS);
        for (t, (got, want)) in null.chunks(4).zip(NULL.iter()).enumerate() {
            assert_eq!(got, want, "generic_null: thread {t}");
        }
    }
    layer_reported_nothing_at_all(&messages);
}
