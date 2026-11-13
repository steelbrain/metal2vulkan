//! What a device's recovery carries of the program's pipeline states (`pipeline/carry.rs`): a
//! compute state made before it is the same state on the new `VkDevice`, its base variant made
//! again there from what its creation compiled, and its dispatches there run it; and how long a
//! recovery takes with many live states.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch};
use super::device_recovery_carry::live_vk_devices;
use super::encoder_holds::library;
use crate::{
    CommandBufferStatus, CommandQueue, ComputePipelineState, Device, Error, Function,
    PipelineOptions, ResourceOptions, Size,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::time::{Duration, Instant};

/// The threads each dispatch runs: one float each.
const COUNT: u64 = 256;

/// The live compute states the timed recovery carries.
const STATES: u32 = 200;

/// `state`, `scale_and_offset`, dispatched over [`COUNT`] floats `i` with a scale of 2, as
/// `queue`'s GPU work runs it, completed: what it wrote, `3 i` each. Refused as the encoding
/// refuses the state.
fn run(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
) -> Result<Vec<f32>, Error> {
    let input: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device.new_buffer_with_bytes(&input, ResourceOptions::shared())?;
    let output = device.new_buffer(COUNT * 4, ResourceOptions::shared())?;
    let scale = device.new_buffer_with_bytes(&2.0f32.to_le_bytes(), ResourceOptions::shared())?;
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder()?;
    encoder.set_compute_pipeline_state(state)?;
    encoder.set_buffer(Some(&input), 0, 0)?;
    encoder.set_buffer(Some(&output), 0, 1)?;
    encoder.set_buffer(Some(&scale), 0, 2)?;
    encoder.dispatch_threads(Size::new(COUNT, 1, 1), Size::new(32, 1, 1))?;
    encoder.end_encoding()?;
    command_buffer.commit()?;
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    let pointer = output.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed work wrote and nothing else uses.
    let bytes = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), (COUNT * 4) as usize) };
    Ok((bytes.as_chunks::<4>().0.iter())
        .map(|word| f32::from_le_bytes(*word))
        .collect())
}

/// What [`run`] reads back.
fn expected() -> Vec<f32> {
    (0..COUNT).map(|i| 3.0 * i as f32).collect()
}

/// `scale_and_offset` of the runtime's example fixture, on `device`.
fn kernel(device: &Device) -> Function {
    library(device, "examples/runtime_examples.ll")
        .new_function_with_name("scale_and_offset")
        .unwrap()
}

/// A state of `function` whose dispatches take `threads` threads a threadgroup, which its creation
/// predicts, labelled `label`.
fn state(device: &Device, function: &Function, threads: u64, label: &str) -> ComputePipelineState {
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.required_threads_per_threadgroup = Some([threads, 1, 1]);
    let options = PipelineOptions {
        label: Some(label.to_owned()),
        ..PipelineOptions::default()
    };
    device
        .new_compute_pipeline_state(&descriptor, &options)
        .unwrap()
}

/// A compute state made before a recovery is the same state after it (its `uniqueIdentifier` and
/// label), on the next generation, as the recovery's notice counts; its dispatches there run it
/// and write what they wrote before; and the lost generation goes while the state lives.
#[test]
fn a_recovery_carries_a_compute_state() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device, &kernel(&device), 32, "scaled");
    assert_eq!(run(&device, &queue, &state).unwrap(), expected());
    let identity = (state.unique_identifier(), state.label().map(str::to_owned));
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains(
            ", and 1 pipeline states made again (their predicted pipelines queued), 0 more whose \
             making there failed"
        ),
        "{said:?}"
    );
    assert!(state.device().generation_serves());
    assert_eq!(
        (state.unique_identifier(), state.label().map(str::to_owned)),
        identity
    );
    assert_eq!(run(&device, &queue, &state).unwrap(), expected());
    assert_eq!(live_vk_devices(&device, 1), 1);
}

/// How long `device`'s recovery takes, `recover_for_test` returning once the new `VkDevice`
/// serves with what it carried.
fn recovery_time(device: &Device) -> Duration {
    let began = Instant::now();
    device.recover_for_test().unwrap();
    began.elapsed()
}

/// A recovery's wall time with no live state, and with [`STATES`] live compute states of as many
/// pipelines, each made again with its predicted `VkPipeline` queued (logged as
/// `RECOVERY-WALL`); and every state made again runs there.
#[test]
fn a_recovery_with_many_live_compute_states_is_timed() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    let function = kernel(&device);
    let bare = recovery_time(&device);
    let states: Vec<_> = (1..=u64::from(STATES))
        .map(|threads| state(&device, &function, threads, "timed"))
        .collect();
    let from = heard.lock().unwrap().len();
    let carried = recovery_time(&device);
    eprintln!(
        "RECOVERY-WALL: {}: no live state {bare:?}, {STATES} live compute states {carried:?}",
        device.name()
    );
    let said = recovery_notices(&heard, from, "the device");
    assert!(
        said[0].contains(&format!(
            ", and {STATES} pipeline states made again (their predicted pipelines queued), 0 more \
             whose making there failed"
        )),
        "{said:?}"
    );
    // The state whose dispatches take the 32 threads a threadgroup `run` dispatches.
    assert_eq!(run(&device, &queue, &states[31]).unwrap(), expected());
}
