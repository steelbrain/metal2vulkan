//! What a load shaped like a guest's walk makes and frees of the spares' command buffers
//! (`spares.rs`), and what its threads spend. A compositor's queue commits a command buffer of
//! [`FRAME`] render passes every [`FRAME_EVERY`], as a guest's window server committed about 94
//! passes a frame at 41 frames a second; [`SMALL_QUEUES`] queues of their own each commit a
//! command buffer of one pass, one blit encoder or two, in turn, every [`SMALL_EVERY`], as the
//! walk's other command buffers, about 1,600 a second together. Each queue keeps [`IN_FLIGHT`]
//! command buffers in flight. A pass, without attachments, draws a point of `ob3_index_buffer`
//! (rasterization disabled) one to four times; a blit encoder fills 256 bytes of a buffer.
//!
//! Printed (`SPARES-WALK`), over [`MEASURE`] after [`WARM`]: the command buffers the device
//! allocated a second, by the place that asked; those the spares freed a second, by why, and the
//! pools' trims; the command buffers the spares held at its start and end; each kind of command
//! buffer's thread time, encoded and committed (its wait excluded); and the process's CPU time a
//! second. Ignored: run it alone, by name, under `--cfg m2v_gpu` (or `m2v_cpu_icd`), in release:
//! `cargo nextest run -p metal2vulkan-runtime --lib --release --run-ignored all spares_walk
//! --no-capture`.

use crate::tests::encoder_holds::library;
use crate::tests::freed_memory::device;
use crate::{
    Buffer, CommandBuffer, CommandQueue, Device, PipelineOptions, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions,
};
use metal2vulkan::metal::PrimitiveType;
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

/// The render passes of the compositor's command buffers.
const FRAME: usize = 94;

/// The compositor's command buffers' period: about 41 a second.
const FRAME_EVERY: Duration = Duration::from_micros(24_400);

/// The queues of small command buffers.
const SMALL_QUEUES: usize = 3;

/// Each small queue's period: about 1,600 a second together.
const SMALL_EVERY: Duration = Duration::from_micros(1_875);

/// The command buffers each queue keeps in flight.
const IN_FLIGHT: usize = 2;

/// The load before the measurement, and the measurement.
const WARM: Duration = Duration::from_secs(3);
const MEASURE: Duration = Duration::from_secs(5);

/// The longest a command buffer is waited for.
const WAIT: Duration = Duration::from_secs(30);

/// The calling thread's CPU time in nanoseconds.
fn thread_ns() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a live `timespec` the call writes.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
    now.tv_sec as u64 * 1_000_000_000 + now.tv_nsec as u64
}

/// The process's CPU time in nanoseconds, its user and system time.
fn process_ns() -> u64 {
    // SAFETY: an all-zero `rusage` is valid, and `getrusage` writes the struct it is given.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: as above.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let ns = |time: libc::timeval| time.tv_sec as u64 * 1_000_000_000 + time.tv_usec as u64 * 1_000;
    ns(usage.ru_utime) + ns(usage.ru_stime)
}

/// What a queue encodes into each command buffer.
#[derive(Clone, Copy)]
enum Shape {
    /// The compositor's: [`FRAME`] passes.
    Frame,
    /// A small queue's: one pass, one blit encoder or two, in turn.
    Small,
}

/// The objects the command buffers' encoders use.
struct Scene<'a> {
    state: &'a RenderPipelineState,
    seen: &'a Buffer,
}

impl Scene<'_> {
    /// Encode a pass of `draws` draws into `command_buffer`.
    fn pass(&self, command_buffer: &CommandBuffer, draws: usize) {
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(self.state).unwrap();
        encoder.set_vertex_buffer(Some(self.seen), 0, 0).unwrap();
        for _ in 0..draws {
            (encoder.draw_primitives(PrimitiveType::Point, 0, 1, 1, 0)).unwrap();
        }
        encoder.end_encoding().unwrap();
    }

    /// Encode `count` blit encoders into `command_buffer`, each filling 256 bytes of `target`.
    fn fills(&self, command_buffer: &CommandBuffer, target: &Buffer, count: usize) {
        for at in 0..count {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(target, 0..256, at as u8).unwrap();
            blit.end_encoding().unwrap();
        }
    }
}

/// Commit command buffers of `shape` on `queue` every `every` until `end`, [`IN_FLIGHT`] in
/// flight: the thread time spent encoding and committing those begun at or after `from`, and
/// their count.
fn commit(
    scene: &Scene<'_>,
    queue: &CommandQueue,
    target: &Buffer,
    shape: Shape,
    every: Duration,
    (from, end): (Instant, Instant),
) -> (u64, u64) {
    let (mut spent, mut counted) = (0, 0);
    let mut flight = VecDeque::new();
    let start = Instant::now();
    for n in 0u32.. {
        let next = start + every * n;
        let now = Instant::now();
        if now >= end {
            break;
        }
        if next > now {
            std::thread::sleep(next - now);
        }
        let begun = (Instant::now(), thread_ns());
        let command_buffer = queue.command_buffer().unwrap();
        match shape {
            Shape::Frame => {
                for pass in 0..FRAME {
                    scene.pass(&command_buffer, 1 + pass % 4);
                }
            }
            Shape::Small => match n % 3 {
                0 => scene.pass(&command_buffer, 1),
                at => scene.fills(&command_buffer, target, at as usize),
            },
        }
        command_buffer.commit().unwrap();
        if begun.0 >= from {
            spent += thread_ns() - begun.1;
            counted += 1;
        }
        flight.push_back(command_buffer);
        if flight.len() > IN_FLIGHT {
            let done = flight.pop_front().unwrap();
            done.wait_until_completed_for(WAIT).unwrap();
        }
    }
    for done in flight {
        done.wait_until_completed_for(WAIT).unwrap();
    }
    (spent, counted)
}

/// The device's command buffers allocated so far, by place, and the spares' frees and the pools'
/// trims so far: past recent uses, past the bound, shed; trims.
fn churn(device: &Device) -> (BTreeMap<String, u64>, [u64; 4]) {
    let churn = device.shared().spares.churn();
    let trims = device.work_counts().spare_command_pool_trims;
    let freed = [churn.past_recent_uses, churn.past_held, churn.shed, trims];
    (churn.allocated, freed)
}

#[test]
#[ignore = "a measurement: run it alone, by name, in release"]
fn spares_walk() {
    let scratch = std::env::temp_dir().join(format!("m2v-spares-walk-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let device = device(&scratch);
    let function = library(&device, "engine/ob3_index_buffer.ll")
        .new_function_with_name("ob3_index_buffer")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    let state = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let seen = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let scene = Scene {
        state: &state,
        seen: &seen,
    };
    let queues: Vec<CommandQueue> = (0..=SMALL_QUEUES)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    let targets: Vec<Buffer> = (0..=SMALL_QUEUES)
        .map(|_| device.new_buffer(4096, ResourceOptions::private()).unwrap())
        .collect();
    let from = Instant::now() + WARM;
    let end = from + MEASURE;
    let (spent, start, finish) = std::thread::scope(|scope| {
        let threads: Vec<_> = (queues.iter().zip(&targets).enumerate())
            .map(|(at, (queue, target))| {
                let (shape, every) = match at {
                    0 => (Shape::Frame, FRAME_EVERY),
                    _ => (Shape::Small, SMALL_EVERY),
                };
                let scene = &scene;
                scope.spawn(move || commit(scene, queue, target, shape, every, (from, end)))
            })
            .collect();
        std::thread::sleep(from.saturating_duration_since(Instant::now()));
        let start = (
            churn(&device),
            device.shared().spares.command_buffers_held(),
            process_ns(),
        );
        std::thread::sleep(end.saturating_duration_since(Instant::now()));
        let finish = (
            churn(&device),
            device.shared().spares.command_buffers_held(),
            process_ns(),
        );
        let spent: Vec<(u64, u64)> = (threads.into_iter())
            .map(|thread| thread.join().unwrap())
            .collect();
        (spent, start, finish)
    });
    let seconds = MEASURE.as_secs_f64();
    let rate = |count: u64| count as f64 / seconds;
    let (frame, small) = (
        spent[0],
        spent[1..]
            .iter()
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1)),
    );
    let each = |(ns, count): (u64, u64)| ns as f64 / count.max(1) as f64 / 1000.0;
    println!(
        "SPARES-WALK {}: over {seconds} s after {} s: {} compositor command buffers of {FRAME} \
         passes, {:.1} us of thread time each; {} small ones, {:.1} us each; process CPU {:.1} \
         ms a second",
        device.name(),
        WARM.as_secs(),
        frame.1,
        each(frame),
        small.1,
        each(small),
        (finish.2 - start.2) as f64 / seconds / 1e6,
    );
    let ((made_before, freed_before), (made_after, freed_after)) = (&start.0, &finish.0);
    let mut made: Vec<(String, u64)> = (made_after.iter())
        .map(|(at, count)| {
            (
                at.clone(),
                count - made_before.get(at).copied().unwrap_or(0),
            )
        })
        .filter(|(_, count)| *count > 0)
        .collect();
    made.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    let total: u64 = made.iter().map(|(_, count)| count).sum();
    let places: Vec<String> = (made.iter())
        .map(|(at, count)| format!("{at} {:.1}", rate(*count)))
        .collect();
    println!(
        "SPARES-WALK allocated {:.1} a second: {}",
        rate(total),
        places.join(", ")
    );
    let freed: Vec<f64> = (0..4)
        .map(|at| rate(freed_after[at] - freed_before[at]))
        .collect();
    println!(
        "SPARES-WALK freed a second: past recent uses {:.1}, past the bound {:.1}, shed {:.1}; \
         trims {:.1} a second; the spares held {} then {}",
        freed[0], freed[1], freed[2], freed[3], start.1, finish.1,
    );
    drop((targets, queues, seen, state, function, device));
    let _ = std::fs::remove_dir_all(&scratch);
}
