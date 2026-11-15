//! Tensors whose private tensor intrinsics are AIR 2.9's, on the device the cfg selects, under the
//! validation layer: the owned fixture `tns_air29` (built as MSL 4.1 for macOS 27, so its
//! `air.init_strided_private_tensor` calls, device and threadgroup, carry AIR 2.9's three added
//! operands and its `air.get_extent_private_tensor` calls the added one) answers what the M3 Max
//! answers (`tns_air29.m`): each of the 32 threads' tensors answers the extents it was initialized
//! with, read from a buffer (32, 16, 16 and 32; macOS 27.0.1: every thread).

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{CommandBufferStatus, PipelineOptions, Size};
use std::path::PathBuf;

/// The threads: one simdgroup in one threadgroup.
const THREADS: u64 = 32;

/// The extents the buffer holds, each thread's four words.
const EXTENTS: [i32; 4] = [32, 16, 16, 32];

fn companion() -> PathBuf {
    metal2vulkan_devtools::engine_dir().join("tns_air29.ll")
}

/// The operands of `name`'s declaration in `text`, counted at its parentheses' top level.
fn operands(text: &str, name: &str) -> usize {
    let line = text
        .lines()
        .find(|l| l.starts_with("declare") && l.contains(&format!("@{name}(")))
        .unwrap_or_else(|| panic!("{name} is not declared"));
    let open = line.find(&format!("@{name}(")).unwrap() + name.len() + 2;
    let mut depth = 0;
    let mut commas = 0;
    for c in line[open..].chars() {
        match c {
            '(' => depth += 1,
            ')' if depth == 0 => break,
            ')' => depth -= 1,
            ',' if depth == 0 => commas += 1,
            _ => {}
        }
    }
    commas + 1
}

/// The AIR 2.9 forms of the private tensor intrinsics run and answer as Metal does.
#[test]
fn air_2_9_tensors_answer_as_metal() {
    let text = std::fs::read_to_string(companion()).unwrap();
    for (name, n) in [
        ("air.init_strided_private_tensor.i32.global", 9),
        ("air.init_strided_private_tensor.i32.local", 9),
        ("air.get_extent_private_tensor.i32", 4),
    ] {
        assert_eq!(operands(&text, name), n, "{name}: the AIR 2.9 form");
    }
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let library = device
            .new_library_with_air(&llvm_assemble(&companion()).unwrap())
            .unwrap();
        let function = library.new_function_with_name("tns_air29").unwrap();
        let pipeline = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let a = buffer_with(&device, &[1; 512]).unwrap();
        let ext: Vec<u8> = EXTENTS.iter().flat_map(|e| e.to_le_bytes()).collect();
        let ext = buffer_with(&device, &ext).unwrap();
        let out = buffer_with(&device, &vec![0; 16 * THREADS as usize]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_buffer(Some(&a), 0, 0).unwrap();
        encoder.set_buffer(Some(&ext), 0, 1).unwrap();
        encoder.set_buffer(Some(&out), 0, 2).unwrap();
        encoder
            .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(THREADS, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let got: Vec<i32> = read_buffer(&out, 0, 16 * THREADS)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| i32::from_le_bytes(*w))
            .collect();
        for (t, words) in got.chunks(4).enumerate() {
            assert_eq!(words, EXTENTS, "thread {t}");
        }
    }
    layer_reported_nothing_at_all(&messages);
}
