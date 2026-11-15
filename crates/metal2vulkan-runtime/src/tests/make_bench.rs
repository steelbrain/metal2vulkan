//! What making a command buffer costs the thread that makes it (`CommandQueue::command_buffer`,
//! `queue.rs`), while the runtime's other threads finish the ones made before, as a guest's are
//! made and finished: one queue makes [`WARM`] command buffers untimed, then [`TIMED`], each
//! committed as it is made and its handle dropped at once, so the runtime's threads let go of it
//! as it completes; every [`WAIT_EVERY`]th is waited for, so the queue never holds its 64
//! uncompleted ones and no make waits for a slot (`WorkCounts::slot_waits`, checked none). Two
//! shapes: `empty`, committed with no encoder, and `one encoder`, a blit encoder's fill of 256
//! bytes of a buffer.
//!
//! Printed (`MAKE-BENCH`) per shape: the make's time in nanoseconds (its 10th, 50th, 90th and 99th
//! percentiles and its mean), the median of the whole command buffer's, from its make through its
//! commit, and how many of the timed ones were made on a state built for them rather than one a
//! released command buffer left (`WorkCounts::command_buffer_states_made`). `make_bench_profiled` makes [`PROFILED`] of `one encoder`, for a profiler. Ignored:
//! run it alone, by name, under `--cfg m2v_cpu_icd` (or `m2v_gpu`), in release:
//! `cargo nextest run -p metal2vulkan-runtime --lib --release --run-ignored all make_bench
//! --no-capture`.

use crate::tests::freed_memory::device;
use crate::{Buffer, CommandQueue, ResourceOptions};
use std::time::{Duration, Instant};

/// The command buffers of each shape made untimed, then timed.
const WARM: usize = 2_000;
const TIMED: usize = 20_000;

/// The command buffers of the profiled run.
const PROFILED: usize = 200_000;

/// Every this many, a command buffer is waited for.
const WAIT_EVERY: usize = 16;

/// The longest a command buffer is waited for.
const WAIT: Duration = Duration::from_secs(30);

/// What each command buffer holds.
#[derive(Clone, Copy, Debug)]
enum Shape {
    Empty,
    OneEncoder,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::Empty => "empty",
            Shape::OneEncoder => "one encoder",
        }
    }
}

/// Make `count` command buffers of `shape` on `queue`, `buffer` the one encoder's, each committed
/// as it is made: each one's make time and its time from its make through its commit, in
/// nanoseconds.
fn make(queue: &CommandQueue, buffer: &Buffer, shape: Shape, count: usize) -> (Vec<u64>, Vec<u64>) {
    let (mut makes, mut wholes) = (Vec::with_capacity(count), Vec::with_capacity(count));
    let nanos = |since: Instant| u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX);
    for n in 0..count {
        let start = Instant::now();
        let command_buffer = queue.command_buffer().unwrap();
        makes.push(nanos(start));
        if let Shape::OneEncoder = shape {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(buffer, 0..256, n as u8).unwrap();
            blit.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        wholes.push(nanos(start));
        if n % WAIT_EVERY == WAIT_EVERY - 1 {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
    }
    (makes, wholes)
}

/// The 10th, 50th, 90th and 99th percentiles of `values`, and their mean.
fn spread(mut values: Vec<u64>) -> [u64; 5] {
    values.sort_unstable();
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    let mean = values.iter().sum::<u64>() / values.len().max(1) as u64;
    [at(10), at(50), at(90), at(99), mean]
}

/// Run `shapes`, `count` timed of each after [`WARM`] untimed, printed.
fn run(shapes: &[Shape], count: usize) {
    let scratch = std::env::temp_dir().join(format!("m2v-make-bench-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let device = device(&scratch);
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::default()).unwrap();
    for &shape in shapes {
        make(&queue, &buffer, shape, WARM);
        let before = device.work_counts();
        let (makes, wholes) = make(&queue, &buffer, shape, count);
        let after = device.work_counts();
        let built = after.command_buffer_states_made - before.command_buffer_states_made;
        let [p10, p50, p90, p99, mean] = spread(makes);
        let [_, whole, ..] = spread(wholes);
        println!(
            "MAKE-BENCH {}: {}: {count} command buffers: command_buffer() p10 {p10} ns, p50 {p50}, \
             p90 {p90}, p99 {p99}, mean {mean}; made through committed p50 {whole} ns; states \
             built {built}",
            device.name(),
            shape.name(),
        );
        assert_eq!(
            after.slot_waits, before.slot_waits,
            "a make waited for a slot"
        );
    }
    drop((buffer, queue, device));
    let _ = std::fs::remove_dir_all(&scratch);
}

#[test]
#[ignore = "a measurement: run alone on its device by name"]
fn make_bench() {
    run(&[Shape::Empty, Shape::OneEncoder], TIMED);
}

#[test]
#[ignore = "a measurement for a profiler: run alone on its device by name"]
fn make_bench_profiled() {
    run(&[Shape::OneEncoder], PROFILED);
}
