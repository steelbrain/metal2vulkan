//! A tiled texture's copy taken at use (`shadow.rs`'s **Copies taken at use**) by work its commit
//! recorded and its host step records again: the execution of an indirect command buffer the CPU
//! encoded and wrote again after the commit (`Execute::rewritten`), whose command reads the
//! texture at level 1 through an argument buffer naming its resource ID, the texture made resident
//! (`useResource`). Metal makes no state for indirect command buffers whose function takes a
//! texture directly. The commit's recording takes the copy and is never submitted; the step's
//! recording, which runs, takes it where that one took it, so the read finds level 1 made from
//! the bytes, and the next use takes no copy and reads every level as taken.
//!
//! The command buffer waits for a shared event, which holds its batch, and so its host step, until
//! the host signals; the CPU writes the buffer's command again before the signal. The step is
//! counted (`WorkCounts::host_steps_recorded`).
//!
//! The kernel is `rt_heap_zero` of the owned fixture
//! `validation/fixtures/public/engine/kernel_read_through.ll`: `out[i]` is the argument buffer's
//! texture read at `p[i].xy` and level `p[i].w`.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`TILED-STEP`): the steps, the
//! reads and the next use's copies.

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::library;
use super::tiled_texture_device::{near_each, rig, texel_of, Rig, NONE};
use super::tiled_texture_failed::grey_texture;
use super::tiled_texture_reencode::{behind_a_wait, GREY, SHAPE};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, IndirectCommandBufferDescriptor,
    PipelineOptions, ResourceOptions, Size,
};
use metal2vulkan::metal::{IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The reads: four threads, each of texel (1, 1) of level 1.
const READS: u64 = 4;

/// A state of the fixture's `rt_heap_zero` that an indirect command buffer's commands may set.
fn state(rig: &Rig) -> ComputePipelineState {
    let function = library(&rig.device, "engine/kernel_read_through.ll")
        .new_function_with_name("rt_heap_zero")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    (rig.device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default()))
    .unwrap()
}

/// A `Shared` buffer of `bytes`.
fn holding(rig: &Rig, bytes: &[u8]) -> Buffer {
    (rig.device
        .new_buffer_with_bytes(bytes, ResourceOptions::shared()))
    .unwrap()
}

#[test]
fn a_take_its_host_step_records_again_is_taken_there() {
    let scratch = Scratch::new("tiled-step");
    let rig = rig(&scratch);
    let state = state(&rig);
    let (_target, texture) = grey_texture(&rig);
    let argument = holding(&rig, &texture.gpu_resource_id().0.to_le_bytes());
    let positions: Vec<u8> = ([1i32, 1, 0, 1].repeat(READS as usize).iter())
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let positions = holding(&rig, &positions);
    let out = holding(&rig, &[0; 16 * READS as usize]);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    descriptor.max_kernel_buffer_bind_count = 3;
    let icb = (rig.device)
        .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
        .unwrap();
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_compute_pipeline_state(&state).unwrap();
    for (index, buffer) in (0..).zip([&positions, &out, &argument]) {
        command.set_kernel_buffer(Some(buffer), 0, index).unwrap();
    }
    let threads = Size::new(READS, 1, 1);
    command
        .concurrent_dispatch_threads(threads, threads)
        .unwrap();
    let steps = rig.device.work_counts().host_steps_recorded;
    let (command_buffer, gate) = behind_a_wait(&rig, |command_buffer| {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let buffers = [
            positions.as_resource(),
            out.as_resource(),
            argument.as_resource(),
        ];
        let usage = ResourceUsage::READ | ResourceUsage::WRITE;
        encoder.use_resources(&buffers, usage).unwrap();
        // The texture's first use: the command reads it through the argument buffer.
        (encoder.use_resource(texture.as_resource(), ResourceUsage::READ)).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
    });
    // The CPU writes the command again after the commit: the host step records the execution
    // again from the records as they stand then.
    command
        .concurrent_dispatch_threads(threads, threads)
        .unwrap();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let (status, error) = (command_buffer.status(), command_buffer.error());
    let stepped = rig.device.work_counts().host_steps_recorded - steps;
    let samples = Rig::samples(&out);
    eprintln!("TILED-STEP: {status:?} {error:?}, {stepped} host steps, level 1 read {samples:?}");
    assert_eq!(status, CommandBufferStatus::Completed, "{error:?}");
    assert!(stepped >= 1, "the host step recorded the execution again");
    // The step's recording took the copy where the commit's took it, before its read.
    let texels: Vec<u8> = samples.into_iter().flat_map(texel_of).collect();
    assert!(near_each(&texels, GREY), "level 1 read {texels:?}");
    let (levels, copies) = rig.levels(&texture, SHAPE);
    eprintln!("TILED-STEP: the next use {copies:?}");
    assert_eq!(copies, NONE, "the next use after the step's take");
    for (level, bytes) in levels.iter().enumerate() {
        let first = bytes.get(..4);
        assert!(near_each(bytes, GREY), "level {level} as taken: {first:?}");
    }
}
