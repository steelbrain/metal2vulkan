//! A tiled texture's copy taken at use (`shadow.rs`'s **Copies taken at use**) by work its commit
//! recorded and its host step records again: the execution of an indirect command buffer the CPU
//! encoded and wrote again after the commit (`Execute::rewritten`), whose command inherits the
//! encoder's bindings and samples the texture at LOD 1. The commit's recording takes the copy and
//! is never submitted; the step's recording, which runs, takes it where that one took it, so the
//! sample reads level 1 made from the bytes, and the next use takes no copy and reads every level
//! as taken.
//!
//! The command buffer waits for a shared event, which holds its batch, and so its host step, until
//! the host signals; the CPU writes the buffer's command again before the signal. The step is
//! counted (`WorkCounts::host_steps_recorded`).
//!
//! The kernel is `sample_level` of the owned fixture `validation/fixtures/public/engine/sampling.ll`:
//! `t.sample(s, float2(gid) * 0.25, level(1.0))` into `out[gid]`.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Printed (`TILED-STEP`): the steps, the
//! samples and the next use's copies.

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::library;
use super::tiled_texture_device::{near_each, rig, texel_of, Rig, NONE};
use super::tiled_texture_failed::grey_texture;
use super::tiled_texture_reencode::{behind_a_wait, GREY, SHAPE};
use crate::{
    CommandBufferStatus, ComputePipelineState, IndirectCommandBufferDescriptor, PipelineOptions,
    ResourceOptions, Size,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{IndirectCommandType, SamplerMipFilter};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// A state of the fixture's `sample_level` that an indirect command buffer's commands may set.
fn state(rig: &Rig) -> ComputePipelineState {
    let function = library(&rig.device, "engine/sampling.ll")
        .new_function_with_name("sample_level")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    (rig.device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default()))
    .unwrap()
}

#[test]
fn a_take_its_host_step_records_again_is_taken_there() {
    let scratch = Scratch::new("tiled-step");
    let rig = rig(&scratch);
    let state = state(&rig);
    let (_target, texture) = grey_texture(&rig);
    let out = (rig.device.new_buffer(64, ResourceOptions::shared())).unwrap();
    let mut sampling = SamplerDescriptor::new();
    sampling.mip_filter = SamplerMipFilter::Nearest;
    let sampler = rig.device.new_sampler_state(&sampling).unwrap();
    // One command that inherits the encoder's texture, sampler and buffer.
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    descriptor.inherit_buffers = true;
    let icb = (rig.device)
        .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
        .unwrap();
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_compute_pipeline_state(&state).unwrap();
    let four = Size::new(4, 1, 1);
    command.concurrent_dispatch_threads(four, four).unwrap();
    let steps = rig.device.work_counts().host_steps_recorded;
    let (command_buffer, gate) = behind_a_wait(&rig, |command_buffer| {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        // The texture's first use.
        encoder.set_texture(Some(&texture), 0).unwrap();
        encoder.set_sampler_state(Some(&sampler), 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 0).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
    });
    // The CPU writes the command again after the commit: the host step records the execution
    // again from the records as they stand then.
    command.concurrent_dispatch_threads(four, four).unwrap();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let (status, error) = (command_buffer.status(), command_buffer.error());
    let stepped = rig.device.work_counts().host_steps_recorded - steps;
    let samples = Rig::samples(&out);
    eprintln!("TILED-STEP: {status:?} {error:?}, {stepped} host steps, level 1 {samples:?}");
    assert_eq!(status, CommandBufferStatus::Completed, "{error:?}");
    assert!(stepped >= 1, "the host step recorded the execution again");
    // The step's recording took the copy where the commit's took it, before its sample.
    let sampled: Vec<u8> = samples.into_iter().flat_map(texel_of).collect();
    assert!(near_each(&sampled, GREY), "level 1 sampled {sampled:?}");
    let (levels, copies) = rig.levels(&texture, SHAPE);
    eprintln!("TILED-STEP: the next use {copies:?}");
    assert_eq!(copies, NONE, "the next use after the step's take");
    for (level, bytes) in levels.iter().enumerate() {
        let first = bytes.get(..4);
        assert!(near_each(bytes, GREY), "level {level} as taken: {first:?}");
    }
}
