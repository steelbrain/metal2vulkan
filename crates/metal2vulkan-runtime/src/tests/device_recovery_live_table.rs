//! A lost generation's table of the live buffers (`live_buffers.rs`) goes at the generation's
//! rest (`LiveBuffers::close`), so the lost `VkDevice` goes while the buffers the table listed
//! live on, carried: the table's buffer, of the runtime's own there, held the generation while a
//! listed buffer lived, and the recovery carried the listed buffers on.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).
//! The function is the owned fixture `examples/runtime_examples.ll` (`llvm-as`).

use super::device_recovery::{device, outcome, Scratch};
use super::device_recovery_carry::{fill, live_vk_devices, read, BYTES};
use super::encoder_holds::library;
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, PipelineOptions, ResourceOptions, ResourceRef, Size,
};
use metal2vulkan::metal::{IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The floats an indirect command's dispatch writes, `i` each.
const WORDS: u64 = 64;

/// `scale_and_offset` of `device`'s example library, for indirect command buffers.
pub(super) fn indirect_state(device: &Device) -> ComputePipelineState {
    let function = library(device, "examples/runtime_examples.ll")
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// An indirect command buffer of `device` of one concurrent dispatch of 3 kernel buffers, in
/// `Shared` storage.
pub(super) fn dispatches(device: &Device) -> IndirectCommandBuffer {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    descriptor.max_kernel_buffer_bind_count = 3;
    device
        .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
        .unwrap()
}

/// Encode `icb`'s command 0 as `scale_and_offset` of `state` from `input` into `output`, scaled
/// by `scale`, over [`WORDS`] threads.
pub(super) fn encode_command(
    icb: &IndirectCommandBuffer,
    state: &ComputePipelineState,
    [input, output, scale]: [&Buffer; 3],
) {
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_compute_pipeline_state(state).unwrap();
    command.set_kernel_buffer(Some(input), 0, 0).unwrap();
    command.set_kernel_buffer(Some(output), 0, 1).unwrap();
    command.set_kernel_buffer(Some(scale), 0, 2).unwrap();
    command
        .concurrent_dispatch_threads(Size::new(WORDS, 1, 1), Size::new(32, 1, 1))
        .unwrap();
}

/// `icb`'s command 0 executed by `queue`'s GPU work, its buffers resident, after `output` is set
/// to all-ones bytes and `input` and `scale` to zero, completed: `output`'s floats.
pub(super) fn execute(
    device: &Device,
    queue: &CommandQueue,
    icb: &IndirectCommandBuffer,
    [input, output, scale]: [&Buffer; 3],
) -> Vec<f32> {
    fill(queue, input, 0);
    fill(queue, scale, 0);
    fill(queue, output, 0xff);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    for (buffer, usage) in [
        (input, ResourceUsage::READ),
        (output, ResourceUsage::WRITE),
        (scale, ResourceUsage::READ),
    ] {
        (encoder.use_resource(ResourceRef::Buffer(buffer), usage)).unwrap();
    }
    encoder.execute_commands_in_buffer(icb, 0..1).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    let bytes = read(device, queue, output).unwrap();
    (bytes.as_chunks::<4>().0.iter())
        .take(WORDS as usize)
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// What [`execute`] reads back from a dispatch that ran: `i` each.
pub(super) fn dispatched() -> Vec<f32> {
    (0..WORDS).map(|i| i as f32).collect()
}

/// An indirect command buffer's execution makes the generation's table of the live buffers, to
/// resolve its records' addresses (`CommandBuffer::residency`); the buffer and its state go
/// before the recovery, its three buffers live past it. The lost `VkDevice` goes while they
/// live, and stays gone once the program lets go of them; without the table's close at the
/// generation's rest, two holds of the lost generation stayed for the life of the process.
#[test]
fn a_lost_generation_s_live_buffers_table_goes_at_its_rest() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let compute = indirect_state(&device);
    let icb = dispatches(&device);
    let private = || {
        device
            .new_buffer(BYTES, ResourceOptions::private())
            .unwrap()
    };
    let (input, output, scale) = (private(), private(), private());
    let buffers = [&input, &output, &scale];
    encode_command(&icb, &compute, buffers);
    let tables = device.work_counts().live_tables_uploaded;
    assert_eq!(execute(&device, &queue, &icb, buffers), dispatched());
    // The execution made the generation's table: the path under test ran.
    assert!(device.work_counts().live_tables_uploaded > tables);
    drop(icb);
    drop(compute);
    device.recover_for_test().unwrap();
    assert_eq!(live_vk_devices(&device, 1), 1, "the buffers live");
    drop((input, output, scale, queue));
    assert_eq!(live_vk_devices(&device, 1), 1, "all let go of");
}
