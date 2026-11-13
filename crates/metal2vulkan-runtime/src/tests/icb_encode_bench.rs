//! What the CPU spends encoding the commands of a render indirect command buffer through
//! `IndirectRenderCommand`'s setters, in nanoseconds a command, the median and least of [`REPS`]
//! loops of [`COMMANDS`] commands after one untimed, on a buffer of [`COMMANDS`] (whose records
//! outgrow a CPU's last-level cache) and on one of [`CACHED`] (whose records fit in its L2), its
//! commands encoded in turn over and over:
//!
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
use crate::{
    DrawPrimitiveType, Error, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    ResourceOptions,
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
            &format!("{size} commands ({records} KiB of records), three fields"),
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
