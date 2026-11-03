//! What executing an indirect command buffer the CPU encoded costs on the device the cfg selects,
//! without the validation layer: report only, in release builds, to weigh the runtime's
//! execution path before and after a change to it.
//!
//! The workload is the consumer's (macOS guests over the runtime): a buffer of `n` commands, each
//! a render pipeline state set and a draw of one triangle (`rt6_resolve_half`, `rt6_resolve_white`
//! into a 64x64 `BGRA8Unorm` target), or a compute pipeline state set, two kernel buffers (the
//! second at a 16-byte window of its own, so no two commands write the same bytes), a threadgroup
//! length and a dispatch of one threadgroup of 4 threads (`icb_replay_template`), encoded once
//! and then executed whole, each execution in a command buffer of its own, committed and waited
//! for:
//!
//! - **`unchanged`**: no command written since the last execution;
//! - **`one re-encoded`**: command 0 encoded again before the execution, as it was;
//! - **`all re-encoded`**: every command encoded again, as it was.
//!
//! Each case runs [`WARM`] untimed executions first, then [`TIMED`], and is printed as an
//! `EXECUTE-COST` line: the median and the range of the encoder's time (its creation through
//! `end_encoding`), of `commit` and of the time from `commit` until `wait_until_completed`
//! returned, and of the wall time over all three, per execution; and the device's counters over
//! the timed executions: the records read and decoded (`WorkCounts::indirect_records_read`), and
//! the chunks of commands run from a kept recording and the commands those held
//! (`WorkCounts::indirect_chunks_kept`, `indirect_commands_kept`).
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, whose lib test binary keeps the system allocator as a
//! consumer's process does.

use super::draw_costs::percentiles;
use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, LoadAction, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, ResourceRef, Size, StoreAction, Texture,
    TextureDescriptor, WorkCounts,
};
use std::time::{Duration, Instant};

/// The untimed and the timed executions of each case.
const WARM: usize = 2;
const TIMED: usize = 5;
/// The command counts measured.
const COUNTS: [u64; 2] = [1024, 16384];
/// The target's side.
const TARGET_SIDE: u64 = 64;

/// What is written before an execution (the module docs).
#[derive(Clone, Copy, Debug)]
enum Case {
    Unchanged,
    OneReencoded,
    AllReencoded,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Case::Unchanged => "unchanged",
            Case::OneReencoded => "one re-encoded",
            Case::AllReencoded => "all re-encoded",
        }
    }
}

/// The commands' kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Render,
    Compute,
}

/// One execution's times.
struct Times {
    encoded: Duration,
    commit: Duration,
    wait: Duration,
}

/// The device, its queue and the states the commands set.
struct Workload {
    device: Device,
    queue: CommandQueue,
    render: RenderPipelineState,
    compute: ComputePipelineState,
    target: Texture,
    params: Buffer,
    out: Buffer,
}

impl Workload {
    fn new(device: Device) -> Workload {
        let library = |ll: &str| {
            device
                .new_library_with_air(&llvm_assemble(&engine_dir().join(ll)).unwrap())
                .unwrap()
        };
        let resolves = library("rt6_resolves.ll");
        let function = |name| {
            StageSource::Function(
                resolves
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut render = RenderPipelineDescriptor::new(function("rt6_resolve_half"));
        render.fragment = Some(function("rt6_resolve_white"));
        render.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        render.support_indirect_command_buffers = true;
        let render = device
            .new_render_pipeline_state(&render, &PipelineOptions::default())
            .unwrap();
        let kernel = library("icb_replay_template.ll")
            .new_function_with_name("icb_replay_template")
            .unwrap();
        let mut compute =
            ComputePipelineDescriptor::new(StageSource::Function(kernel.function_descriptor()));
        compute.support_indirect_command_buffers = true;
        let compute = device
            .new_compute_pipeline_state(&compute, &PipelineOptions::default())
            .unwrap();
        let mut target =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, TARGET_SIDE, TARGET_SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::private();
        let params = [3u32, 7].map(u32::to_le_bytes).concat();
        Workload {
            queue: device.new_command_queue().unwrap(),
            target: device.new_texture(&target).unwrap(),
            params: buffer_with(&device, &params).unwrap(),
            out: buffer_with(&device, &vec![0; 16 * COUNTS[1] as usize]).unwrap(),
            render,
            compute,
            device,
        }
    }

    /// A buffer of `n` commands of `kind`.
    fn icb(&self, kind: Kind, n: u64) -> IndirectCommandBuffer {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        match kind {
            Kind::Render => descriptor.command_types = IndirectCommandType::DRAW,
            Kind::Compute => {
                descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
                descriptor.max_kernel_buffer_bind_count = 2;
                descriptor.max_kernel_threadgroup_memory_bind_count = 1;
            }
        }
        self.device
            .new_indirect_command_buffer(&descriptor, n, ResourceOptions::shared())
            .unwrap()
    }

    /// Encode command `index` of `icb`, of `kind`.
    fn encode(&self, icb: &IndirectCommandBuffer, kind: Kind, index: u64) {
        match kind {
            Kind::Render => {
                let command = icb.indirect_render_command_at(index).unwrap();
                command.set_render_pipeline_state(&self.render).unwrap();
                command
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            }
            Kind::Compute => {
                let command = icb.indirect_compute_command_at(index).unwrap();
                command.set_compute_pipeline_state(&self.compute).unwrap();
                command.set_kernel_buffer(Some(&self.params), 0, 0).unwrap();
                command
                    .set_kernel_buffer(Some(&self.out), 16 * index, 1)
                    .unwrap();
                command.set_threadgroup_memory_length(16, 0).unwrap();
                command
                    .concurrent_dispatch_threadgroups(Size::new(1, 1, 1), Size::new(4, 1, 1))
                    .unwrap();
            }
        }
    }

    /// Execute every command of `icb`, of `kind`, in a command buffer of its own, committed and
    /// waited for.
    fn execute(&self, icb: &IndirectCommandBuffer, kind: Kind) -> Times {
        let start = Instant::now();
        let command_buffer = self.queue.command_buffer().unwrap();
        match kind {
            Kind::Render => {
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(self.target.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                encoder
                    .execute_commands_in_buffer(icb, 0..icb.size())
                    .unwrap();
                encoder.end_encoding().unwrap();
            }
            Kind::Compute => {
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                let used = [
                    ResourceRef::Buffer(&self.params),
                    ResourceRef::Buffer(&self.out),
                ];
                encoder
                    .use_resources(&used, ResourceUsage::READ.union(ResourceUsage::WRITE))
                    .unwrap();
                encoder
                    .execute_commands_in_buffer(icb, 0..icb.size())
                    .unwrap();
                encoder.end_encoding().unwrap();
            }
        }
        let encoded = start.elapsed();
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        let commit = committed.elapsed();
        let waited = Instant::now();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let wait = waited.elapsed();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Times {
            encoded,
            commit,
            wait,
        }
    }
}

/// `values` in milliseconds as "median (low to high)".
fn spread(values: Vec<f64>) -> String {
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(0.0, f64::max);
    let [_, median, _] = percentiles(values);
    format!("{median:.2} ({low:.2} to {high:.2})")
}

/// The execution path's costs, printed (the module docs), for `kinds`.
fn measure(kinds: &[Kind]) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let name = device.name().to_owned();
    let work = Workload::new(device);
    eprintln!(
        "EXECUTE-COST {name}: {WARM} untimed and {TIMED} timed executions a case, each in a \
         command buffer of its own"
    );
    for &kind in kinds {
        for n in COUNTS {
            let icb = work.icb(kind, n);
            for index in 0..n {
                work.encode(&icb, kind, index);
            }
            for case in [Case::Unchanged, Case::OneReencoded, Case::AllReencoded] {
                let run = || {
                    match case {
                        Case::Unchanged => {}
                        Case::OneReencoded => work.encode(&icb, kind, 0),
                        Case::AllReencoded => {
                            for index in 0..n {
                                work.encode(&icb, kind, index);
                            }
                        }
                    }
                    work.execute(&icb, kind)
                };
                for _ in 0..WARM {
                    run();
                }
                let before = work.device.work_counts();
                let times: Vec<Times> = (0..TIMED).map(|_| run()).collect();
                let after = work.device.work_counts();
                let delta = |f: fn(&WorkCounts) -> u64| f(&after) - f(&before);
                let ms = |f: fn(&Times) -> Duration| {
                    spread(times.iter().map(|t| f(t).as_secs_f64() * 1e3).collect())
                };
                eprintln!(
                    "EXECUTE-COST {name}: {kind:?} n={n} {}: wall {} ms; encode {} ms, commit {} \
                     ms, commit to completion {} ms; over {TIMED}: {} records read, {} \
                     chunks kept holding {} commands",
                    case.name(),
                    ms(|t| t.encoded + t.commit + t.wait),
                    ms(|t| t.encoded),
                    ms(|t| t.commit),
                    ms(|t| t.wait),
                    delta(|c| c.indirect_records_read),
                    delta(|c| c.indirect_chunks_kept),
                    delta(|c| c.indirect_commands_kept),
                );
            }
        }
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn execution_costs() {
    measure(&[Kind::Render, Kind::Compute]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn execution_costs_render() {
    measure(&[Kind::Render]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn execution_costs_compute() {
    measure(&[Kind::Compute]);
}
