//! A thread pointer that may be null, on the device the cfg selects, under the validation layer:
//! the owned fixture `nullable_private`'s kernel, whose `closure_alloc` hands out the next element
//! of a thread's array at an index only the run knows, or null once the array's room is used up,
//! and whose `noinline` callees ask whether that pointer is null and write through it where it is
//! not, answers what the M3 Max answers (`nullable_private.m`).
//!
//! Eight threads in one threadgroup over `in[i] = i + 0.5` (16 words); the fixture's header says
//! what each word is. Every answer is a small integer or a half, exact in `float`.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{CommandBufferStatus, PipelineOptions, Size};

/// The threads, all in one threadgroup.
const THREADS: usize = 8;

/// The words each thread writes.
const WORDS: usize = 8;

/// What the M3 Max answers (macOS 27.0.1), eight words a thread: four weights (-1 where the
/// pointer is null), the closures handed out, the sum of their types, the sum of their data, and 1
/// where the fifth request is null.
#[rustfmt::skip]
const METAL: [[f32; WORDS]; THREADS] = [
    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    [3.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.5, 1.0],
    [5.0, 7.0, -1.0, -1.0, 2.0, 3.0, 6.0, 1.0],
    [7.0, 9.0, 11.0, -1.0, 3.0, 6.0, 13.5, 1.0],
    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    [11.0, -1.0, -1.0, -1.0, 1.0, 1.0, 5.5, 1.0],
    [13.0, 15.0, -1.0, -1.0, 2.0, 3.0, 14.0, 1.0],
    [15.0, 17.0, 19.0, -1.0, 3.0, 6.0, 25.5, 1.0],
];

/// A thread pointer that is an array element or null compares with null, and reads and writes the
/// element where it is not null, as Metal's does.
#[test]
fn a_thread_pointer_that_may_be_null_answers_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let ll = metal2vulkan_devtools::engine_dir().join("nullable_private.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = library.new_function_with_name("nullable_private").unwrap();
        let pipeline = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let input: Vec<u8> = (0..16u16)
            .flat_map(|i| (f32::from(i) + 0.5).to_le_bytes())
            .collect();
        let out = buffer_with(&device, &[0; 4 * WORDS * THREADS]).unwrap();
        let input = buffer_with(&device, &input).unwrap();
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
        let got: Vec<f32> = read_buffer(&out, 0, (4 * WORDS * THREADS) as u64)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        for (t, (got, want)) in got.chunks(WORDS).zip(METAL.iter()).enumerate() {
            assert_eq!(got, want, "nullable_private: thread {t}");
        }
    }
    layer_reported_nothing_at_all(&messages);
}
