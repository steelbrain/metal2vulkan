//! What the CPU spends encoding the commands of a render indirect command buffer through
//! `IndirectRenderCommand`'s setters, in nanoseconds a command, the median and least of [`REPS`]
//! loops of [`COMMANDS`] commands after one untimed, on a buffer of [`COMMANDS`] (whose records
//! outgrow a CPU's last-level cache) and on one of [`CACHED`] (whose records fit in its L2), its
//! commands encoded in turn over and over:
//!
//! - **full slot**: `indirect_render_command_at(i)`, a render pipeline state, two vertex buffers
//!   (one at `(i % 16) * 16`), a fragment buffer and a draw of primitives, the consumer's
//!   `icb_slots_cost` loop through the runtime alone;
//! - **compute slot**: `indirect_compute_command_at(i)` of a buffer of dispatches, a compute
//!   pipeline state, two kernel buffers (one at `(i % 16) * 16`) and a dispatch of threadgroups;
//! - **set_each**: each slot's setters in one `set_each` call;
//! - **set_render_command**, **set_compute_command**: each slot's setters in one call of the
//!   buffer's, with no command's handle;
//! - **three fields**: `indirect_render_command_at(i)`, a vertex buffer at `(i % 8) * 256`, a
//!   fragment buffer at `(i % 4) * 64` and a draw of primitives, a consumer's loop;
//! - **draw only**: the command and its draw;
//! - **command only**: `indirect_render_command_at(i)` alone.
//!
//! Printed as `ICB-ENCODE` lines, beside the sizes of `Result<(), Error>` and `Error`. Nothing is
//! submitted. Ignored: run it in release, under the GPU safety cfg.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use super::indirect_command_buffer_device::{compute_state, render_state, EXAMPLES};
use crate::{
    DrawPrimitiveType, Error, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    IndirectComputeSet, IndirectRenderSet, ResourceOptions, Size,
};
use metal2vulkan::bind::indirect_commands::COMMAND_BYTES;
use metal2vulkan::metal::IndirectCommandType;
use std::hint::black_box;
use std::time::Instant;

/// The commands of the large buffer, and the commands encoded a loop.
const COMMANDS: u64 = 16384;

/// The commands of the small buffer.
const CACHED: u64 = 256;

/// The timed loops of each case, after one untimed.
const REPS: usize = 20;

/// The median and least of [`REPS`] runs of `encode` over [`COMMANDS`] commands, the `i`th of
/// them at index `i % size`, after one untimed, in nanoseconds a command.
fn per_command(size: u64, encode: impl Fn(u64)) -> (f64, f64) {
    let run = || {
        let start = Instant::now();
        for i in 0..COMMANDS {
            encode(i % size);
        }
        start.elapsed().as_nanos() as f64 / COMMANDS as f64
    };
    run();
    let mut times: Vec<f64> = (0..REPS).map(|_| run()).collect();
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times[0])
}

fn print(case: &str, (median, least): (f64, f64)) {
    eprintln!("ICB-ENCODE {case}: median {median:.1} ns a command, least {least:.1}");
}

/// The setters' cost a command (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn icb_encode_costs() {
    let scratch = Scratch::new("icb-encode");
    let device = device(&scratch.0);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW
        | IndirectCommandType::DRAW_INDEXED
        | IndirectCommandType::DRAW_PATCHES
        | IndirectCommandType::DRAW_INDEXED_PATCHES;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = 1;
    let shared = ResourceOptions::shared();
    let (vertex, fragment) = (
        device.new_buffer(65536, shared).unwrap(),
        device.new_buffer(65536, shared).unwrap(),
    );
    let strip = DrawPrimitiveType::from_raw(4).unwrap();
    let triangle = DrawPrimitiveType::from_raw(3).unwrap();
    let render = render_state(&device, true);
    let compute = compute_state(&device, EXAMPLES, "scale_and_offset", true);
    let mut compute_descriptor = IndirectCommandBufferDescriptor::new();
    compute_descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
    compute_descriptor.max_kernel_buffer_bind_count = 2;
    let group = Size::new(1, 1, 1);
    eprintln!(
        "ICB-ENCODE {}: Result<(), Error> {} bytes, Error {} bytes, a record {COMMAND_BYTES} \
         bytes, {COMMANDS} commands a loop",
        device.name(),
        size_of::<Result<(), Error>>(),
        size_of::<Error>(),
    );
    for size in [COMMANDS, CACHED] {
        let icb: IndirectCommandBuffer = device
            .new_indirect_command_buffer(&descriptor, size, shared)
            .unwrap();
        let records = size * COMMAND_BYTES as u64 / 1024;
        print(
            &format!("{size} commands ({records} KiB of records), full slot"),
            per_command(size, |i| {
                let command = icb.indirect_render_command_at(i).unwrap();
                command.set_render_pipeline_state(&render).unwrap();
                command
                    .set_vertex_buffer(Some(&vertex), (i % 16) * 16, 0)
                    .unwrap();
                command.set_vertex_buffer(Some(&fragment), 0, 1).unwrap();
                command.set_fragment_buffer(Some(&vertex), 256, 0).unwrap();
                command.draw_primitives(triangle, 0, 3, 1, 0).unwrap();
            }),
        );
        let render_sets = |i: u64| {
            [
                IndirectRenderSet::SetRenderPipelineState(&render),
                IndirectRenderSet::SetVertexBuffer {
                    buffer: Some(&vertex),
                    offset: (i % 16) * 16,
                    index: 0,
                },
                IndirectRenderSet::SetVertexBuffer {
                    buffer: Some(&fragment),
                    offset: 0,
                    index: 1,
                },
                IndirectRenderSet::SetFragmentBuffer {
                    buffer: Some(&vertex),
                    offset: 256,
                    index: 0,
                },
                IndirectRenderSet::DrawPrimitives {
                    primitive_type: triangle,
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: 0,
                },
            ]
        };
        print(
            &format!("{size} commands, full slot, set_each"),
            per_command(size, |i| {
                let command = icb.indirect_render_command_at(i).unwrap();
                command.set_each(render_sets(i)).unwrap();
            }),
        );
        print(
            &format!("{size} commands, full slot, set_render_command"),
            per_command(size, |i| {
                icb.set_render_command(i, render_sets(i)).unwrap();
            }),
        );
        let dispatches: IndirectCommandBuffer = device
            .new_indirect_command_buffer(&compute_descriptor, size, shared)
            .unwrap();
        print(
            &format!("{size} commands, compute slot"),
            per_command(size, |i| {
                let command = dispatches.indirect_compute_command_at(i).unwrap();
                command.set_compute_pipeline_state(&compute).unwrap();
                command
                    .set_kernel_buffer(Some(&vertex), (i % 16) * 16, 0)
                    .unwrap();
                command.set_kernel_buffer(Some(&fragment), 0, 1).unwrap();
                command
                    .concurrent_dispatch_threadgroups(group, group)
                    .unwrap();
            }),
        );
        let compute_sets = |i: u64| {
            [
                IndirectComputeSet::SetComputePipelineState(&compute),
                IndirectComputeSet::SetKernelBuffer {
                    buffer: Some(&vertex),
                    offset: (i % 16) * 16,
                    index: 0,
                },
                IndirectComputeSet::SetKernelBuffer {
                    buffer: Some(&fragment),
                    offset: 0,
                    index: 1,
                },
                IndirectComputeSet::ConcurrentDispatchThreadgroups {
                    threadgroups: group,
                    threads_per_threadgroup: group,
                },
            ]
        };
        print(
            &format!("{size} commands, compute slot, set_each"),
            per_command(size, |i| {
                let command = dispatches.indirect_compute_command_at(i).unwrap();
                command.set_each(compute_sets(i)).unwrap();
            }),
        );
        print(
            &format!("{size} commands, compute slot, set_compute_command"),
            per_command(size, |i| {
                dispatches.set_compute_command(i, compute_sets(i)).unwrap();
            }),
        );
        print(
            &format!("{size} commands, three fields"),
            per_command(size, |i| {
                let command = icb.indirect_render_command_at(i).unwrap();
                command
                    .set_vertex_buffer(Some(&vertex), (i % 8) * 256, 0)
                    .unwrap();
                command
                    .set_fragment_buffer(Some(&fragment), (i % 4) * 64, 0)
                    .unwrap();
                command.draw_primitives(strip, 1, 3, 2, 7).unwrap();
            }),
        );
        print(
            &format!("{size} commands, draw only"),
            per_command(size, |i| {
                let command = icb.indirect_render_command_at(i).unwrap();
                command.draw_primitives(strip, 1, 3, 2, 7).unwrap();
            }),
        );
        print(
            &format!("{size} commands, command only"),
            per_command(size, |i| {
                black_box(icb.indirect_render_command_at(i).unwrap());
            }),
        );
    }
}
