//! Compute threadgroups against Metal's limit and the device's, through the runtime, on the device
//! the cfg selects, under the validation layer.
//!
//! The owned `ttg_compute_simd_scratch` holds a 24 KiB threadgroup array and shuffles a float4
//! across its simdgroup, so a device that emulates 32-lane simdgroups in 32 KiB of shared memory
//! (lavapipe) holds 512 of its threads a threadgroup, where Metal holds 1024. Its state answers
//! the device's number; a threadgroup past Metal's 1024 runs nothing, as plain Metal's does; one
//! within Metal's limit but past the device's is the device's gap, refused typed in every dispatch
//! form, and never run as nothing. A state whose descriptor requires a threadgroup size answers
//! that size's threads, within the device's number.

use super::indirect_host_steps::state;
use super::tile_states::on_device;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::error::{BindError, Feature};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputeCommandEncoder, ComputePipelineState, Device, Error,
    PipelineOptions, Size,
};

/// Metal's threads a threadgroup.
const METAL: u64 = 1024;

/// The kernel's answer for threadgroup `g` of `size` threads at thread `t`: the staged value of
/// thread `size - 1 - t` shuffled down a lane, lane 31 filled from lane 0's `float4(0)`.
fn expected(size: u64, g: u64, t: u64) -> [f32; 4] {
    if t % 32 == 31 {
        [0.0; 4]
    } else {
        [(size - 2 - t) as f32, g as f32, 0.0, 1.0]
    }
}

/// Whether `error` is the device's typed refusal of a threadgroup it cannot hold.
fn device_gap(error: &Error) -> bool {
    matches!(
        error,
        Error::Bind(BindError::Unsupported(u))
            if matches!(u.feature, Feature::LocalSize { .. } | Feature::ThreadgroupMemorySize)
    )
}

/// Run `encode` in a compute encoder of a fresh command buffer, commit it and wait.
fn run(device: &Device, encode: impl FnOnce(&mut ComputeCommandEncoder)) -> CommandBufferStatus {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let status = command_buffer.status();
    if status == CommandBufferStatus::Error {
        let refusal = command_buffer.refusal();
        assert!(
            refusal.as_ref().is_some_and(device_gap),
            "a failed command buffer's refusal: {refusal:?}"
        );
    }
    status
}

/// The state answers the device's number; a threadgroup of it runs Metal's answer, one past
/// Metal's 1024 runs nothing, and one within Metal's limit but past the device's is refused
/// typed: where encoded for a direct dispatch, and where encoded or as the command buffer's
/// typed refusal for an indirect one.
#[test]
fn a_compute_threadgroup_past_the_devices_limit_within_metals_is_refused_typed() {
    on_device(|device| {
        let kernel = state(device, "ttg_compute_simd_scratch", false);
        let max = u64::from(kernel.max_total_threads_per_threadgroup());
        assert!((32..=METAL).contains(&max), "the state answers {max}");
        eprintln!("ttg_compute_simd_scratch: the state answers {max} threads");
        // Whole simdgroups, at the device's number.
        let size = max - max % 32;
        let out = buffer_with(device, &vec![0xAB; (2 * size * 16) as usize]).unwrap();
        let past = buffer_with(device, &vec![0xAB; (2 * METAL * 16) as usize]).unwrap();
        let status = run(device, |encoder| {
            encoder.set_compute_pipeline_state(&kernel).unwrap();
            encoder.set_buffer(Some(&past), 0, 0).unwrap();
            // Past Metal's 1024: nothing runs.
            encoder
                .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(2 * METAL, 1, 1))
                .unwrap();
            if max < METAL {
                // Within Metal's 1024, past the device's: refused typed, recording nothing.
                let band = Size::new(METAL, 1, 1);
                let refused = encoder.dispatch_threadgroups(Size::new(1, 1, 1), band);
                assert!(
                    refused.as_ref().is_err_and(device_gap),
                    "dispatch_threadgroups past {max}: {refused:?}"
                );
                let refused = encoder.dispatch_threads(Size::new(METAL, 1, 1), band);
                assert!(
                    refused.as_ref().is_err_and(device_gap),
                    "dispatch_threads past {max}: {refused:?}"
                );
            }
            encoder.set_buffer(Some(&out), 0, 0).unwrap();
            encoder
                .dispatch_threadgroups(Size::new(2, 1, 1), Size::new(size, 1, 1))
                .unwrap();
        });
        assert_eq!(status, CommandBufferStatus::Completed);
        assert!(
            read_buffer(&past, 0, 2 * METAL * 16)
                .unwrap()
                .iter()
                .all(|&b| b == 0xAB),
            "a threadgroup past Metal's limit ran"
        );
        let words: Vec<f32> = read_buffer(&out, 0, 2 * size * 16)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        for g in 0..2 {
            for t in 0..size {
                let at = ((g * size + t) * 4) as usize;
                assert_eq!(
                    words[at..at + 4],
                    expected(size, g, t),
                    "threadgroup {g} thread {t}"
                );
            }
        }
        if max < METAL {
            // The indirect form within Metal's limit, past the device's: refused typed where it
            // is encoded, or as the command buffer's typed refusal; never a completed nothing.
            let arguments: Vec<u8> = [1u32, 1, 1].iter().flat_map(|w| w.to_le_bytes()).collect();
            let arguments = buffer_with(device, &arguments).unwrap();
            let mut encoded = None;
            let status = run(device, |encoder| {
                encoder.set_compute_pipeline_state(&kernel).unwrap();
                encoder.set_buffer(Some(&out), 0, 0).unwrap();
                encoded = Some(encoder.dispatch_threadgroups_indirect(
                    &arguments,
                    0,
                    Size::new(METAL, 1, 1),
                ));
            });
            match encoded {
                Some(Err(error)) => assert!(device_gap(&error), "encoded: {error:?}"),
                Some(Ok(())) => assert_eq!(
                    status,
                    CommandBufferStatus::Error,
                    "an indirect threadgroup past {max} completed"
                ),
                None => unreachable!("the encoder ran"),
            }
        }
    });
}

/// The state of the owned fixture `stem`'s kernel, its descriptor requiring `required` threads per
/// threadgroup.
fn required_state(device: &Device, stem: &str, required: [u32; 3]) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(stem).unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.required_threads_per_threadgroup = Some(required);
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A compute state whose descriptor requires a threadgroup size answers that size's threads as
/// its `maxTotalThreadsPerThreadgroup`, as Metal's compute state does where Metal compiles it for
/// the first time (an M3 Max, macOS 27.0.1: 1 for 1x1x1, 21 for 7x3x1, 128 for 8x8x2). Where the
/// device holds fewer threads than the size, the answer is the device's limit, which the state
/// with no size answers.
#[test]
fn a_compute_state_with_a_required_size_answers_its_threads() {
    on_device(|device| {
        let stem = "ttg_compute_simd_scratch";
        let unrequired = state(device, stem, false).max_total_threads_per_threadgroup();
        assert!((32..=1024).contains(&unrequired), "{unrequired}");
        for (required, threads) in [([1, 1, 1], 1), ([7, 3, 1], 21), ([8, 8, 2], 128)] {
            let made = required_state(device, stem, required);
            assert_eq!(
                made.max_total_threads_per_threadgroup(),
                threads.min(unrequired),
                "{required:?}"
            );
            let [width, height, depth] = required.map(u64::from);
            assert_eq!(
                made.required_threads_per_threadgroup(),
                Size::new(width, height, depth)
            );
        }
    });
}
