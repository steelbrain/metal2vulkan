//! What command buffers of many compute encoders leave of the process's memory, as the device's
//! spares keep, free and trim their command buffers (`spares.rs`): a guest's simulation encodes a
//! compute pass for each of its steps, so a command buffer takes a primary for each. Each round of
//! a [`Shape`] commits its command buffers in turn, a few in flight, each of a count of compute
//! encoders, each encoder one dispatch of `scale_and_offset` over 64 floats; then waits for them
//! and for the frees their pools' give-backs posted.
//!
//! Printed (`SPARES-COMPUTE`), after each round: how long it took, glibc's heap in use (`mallinfo2`'s `uordblks`,
//! every arena's), the process's anonymous resident memory (`RssAnon`), the command buffers the
//! spares hold, those the device allocated so far, those the spares freed so far (past recent
//! uses, past the bound, shed), the pools' trims so far, the command pools made so far, and the
//! trim rounds so far with the command buffers freed untrimmed they destroyed (`spares.rs`).
//! Ignored: run each alone, by name, under `--cfg m2v_gpu` (or `m2v_cpu_icd`):
//! `cargo nextest run -p metal2vulkan-runtime --lib --run-ignored all spares_compute --no-capture`.

use crate::pipeline::compile_memory::heap_bytes;
use crate::tests::encoder_holds::library;
use crate::tests::freed_memory::device;
use crate::{PipelineOptions, ResourceOptions, Size};
use std::collections::VecDeque;
use std::time::Duration;

/// The longest a command buffer is waited for.
const WAIT: Duration = Duration::from_secs(60);

/// A load: first `pools` command buffers of one compute encoder each alive at once, each taking
/// a command pool of its own, committed and waited for; then `rounds` rounds of `each` command
/// buffers, committed in turn with at most `in_flight` of them in flight (none: each waited for
/// as it is committed), command buffer `index` of round `round` of `encoders(round, index)`
/// compute encoders.
struct Shape {
    pools: usize,
    rounds: usize,
    each: usize,
    in_flight: usize,
    encoders: fn(usize, usize) -> usize,
}

/// The compute encoders of [`spares_compute`]'s rounds' command buffers, a count a round, in turn.
const COUNTS: [usize; 6] = [2000, 40, 600, 3000, 150, 1200];

/// The compute encoders of [`spares_compute_in_flight`]'s command buffers, in turn, each round
/// starting five further on.
const MIXED: [usize; 16] = [
    3000, 7, 120, 900, 15, 2200, 45, 400, 1, 1600, 60, 250, 3, 1200, 30, 700,
];

/// The process's anonymous resident memory in KiB (`RssAnon` of `/proc/self/status`); none
/// where the system does not report it.
fn anonymous_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status
        .lines()
        .find_map(|line| line.strip_prefix("RssAnon:"))?;
    line.split_whitespace().next()?.parse().ok()
}

/// Four command buffers a round, two in flight, of one count of [`COUNTS`] a round, in turn, each
/// count three times.
#[test]
#[ignore = "a measurement: run it alone, by name"]
fn spares_compute() {
    run(&Shape {
        pools: 0,
        rounds: 18,
        each: 4,
        in_flight: 2,
        encoders: |round, _| COUNTS[round % COUNTS.len()],
    });
}

/// 32 command buffers a round, 16 in flight, of the counts of [`MIXED`] in turn: more pools out of
/// the spares at once, each command buffer of a count unlike the one before.
#[test]
#[ignore = "a measurement: run it alone, by name"]
fn spares_compute_in_flight() {
    run(&Shape {
        pools: 0,
        rounds: 18,
        each: 32,
        in_flight: 16,
        encoders: |round, index| MIXED[(round * 5 + index) % MIXED.len()],
    });
}

/// A guest's fluid simulation (Geekbench's, under a guest on the Iris Xe): command buffers of 62
/// compute encoders, each waited for as it is committed, after 214 command pools were made at
/// once (as a guest's boot made them): the spares hold past [`HELD`](crate::spares::HELD), and
/// shed.
#[test]
#[ignore = "a measurement: run it alone, by name"]
fn spares_compute_fluid() {
    run(&Shape {
        pools: 214,
        rounds: 18,
        each: 100,
        in_flight: 0,
        encoders: |_, _| 62,
    });
}

/// Run `shape` on the device, printing after each round.
fn run(shape: &Shape) {
    let scratch = std::env::temp_dir().join(format!("m2v-spares-compute-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let device = device(&scratch);
    // Room for every command buffer alive at once: past a queue's count, making one waits for
    // one to complete, as Metal's does (64 for `new_command_queue`).
    let queue = device
        .new_command_queue_with_max_command_buffer_count(shape.pools.max(64))
        .unwrap();
    let examples = library(&device, "examples/runtime_examples.ll");
    let kernel = examples.new_function_with_name("scale_and_offset").unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&kernel, &PipelineOptions::default())
        .unwrap();
    let options = ResourceOptions::private();
    let input = device.new_buffer(256, options).unwrap();
    let output = device.new_buffer(256, options).unwrap();
    let scale = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let encode = |command_buffer: &crate::CommandBuffer, encoders: usize| {
        for _ in 0..encoders {
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_buffer(Some(&input), 0, 0).unwrap();
            encoder.set_buffer(Some(&output), 0, 1).unwrap();
            encoder.set_buffer(Some(&scale), 0, 2).unwrap();
            (encoder.dispatch_threads(Size::new(64, 1, 1), Size::new(64, 1, 1))).unwrap();
            encoder.end_encoding().unwrap();
        }
    };
    let alive: Vec<_> = (0..shape.pools)
        .map(|_| {
            let command_buffer = queue.command_buffer().unwrap();
            encode(&command_buffer, 1);
            command_buffer
        })
        .collect();
    for command_buffer in &alive {
        command_buffer.commit().unwrap();
    }
    for command_buffer in alive {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
    for round in 0..shape.rounds {
        let start = std::time::Instant::now();
        let mut flight = VecDeque::new();
        let mut recorded = 0;
        for index in 0..shape.each {
            let command_buffer = queue.command_buffer().unwrap();
            let encoders = (shape.encoders)(round, index);
            encode(&command_buffer, encoders);
            recorded += encoders;
            command_buffer.commit().unwrap();
            flight.push_back(command_buffer);
            if flight.len() > shape.in_flight {
                let done = flight.pop_front().unwrap();
                done.wait_until_completed_for(WAIT).unwrap();
            }
        }
        for done in flight {
            done.wait_until_completed_for(WAIT).unwrap();
        }
        let took = start.elapsed().as_millis();
        device.shared().frees.drain();
        let churn = device.shared().spares.churn();
        let counts = device.work_counts();
        let allocated: u64 = churn.allocated.values().sum();
        let heap = heap_bytes().map_or("-".to_owned(), |(used, _)| format!("{}", used >> 20));
        let anonymous = anonymous_kib().map_or("-".to_owned(), |kib| format!("{}", kib >> 10));
        println!(
            "SPARES-COMPUTE {}: round {round}, {} command buffers of {recorded} compute encoders \
             in all, {} in flight, in {took} ms: heap in use {heap} MiB, anonymous {anonymous} MiB; the spares \
             hold {}; allocated {allocated}; freed past recent uses {}, past the bound {}, shed \
             {}; trims {}; command pools made {}; trim rounds {}, trimmed {}",
            device.name(),
            shape.each,
            shape.in_flight,
            device.shared().spares.command_buffers_held(),
            churn.past_recent_uses,
            churn.past_held,
            churn.shed,
            counts.spare_command_pool_trims,
            counts.command_pools_made,
            counts.spare_command_pool_trim_rounds,
            counts.spare_command_buffers_trimmed_in_rounds,
        );
    }
    drop((input, output, scale, state, kernel, examples, queue, device));
    let _ = std::fs::remove_dir_all(&scratch);
}
