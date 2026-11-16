//! How soon a command buffer that waits for a shared event's value completes once the host sets
//! the value: a command buffer that only waits for one event's value and signals another's,
//! committed and left waiting [`SETTLE`], then the value set (`set_signaled_value`) at `t0`; the
//! times from `t0` to its completion seen by a waiting thread and to its completed handler. Its
//! batch is held on the host until the value comes (the device takes no wait for a value no
//! submitted signal reaches), so the time is the scheduler's wake, its submission, and the
//! completion's turn.
//!
//! The shapes, [`REPEATS`] each, interleaved:
//! - **Alone**: nothing else on the device.
//! - **Behind 1** and **behind 4**: that many command buffers of another queue on the same
//!   `VkQueue`, each a fill of [`FILL_BYTES`], committed just before it and still running at
//!   `t0`. Its submission goes behind theirs, so each time is also given from the later of `t0`
//!   and their last GPU end.
//! - **Unshared**: as behind 4, with a command queue made unshared committing a command buffer just
//!   before, so that the other queues' batches past about 2 ms of expected work are held for it
//!   (`unshared_hold.rs`), the waiting command buffer's among them.
//!
//! Printed (`EVENT-WAKE`), per shape: each time's median, 90th percentile and slowest, and how
//! many of the scheduler's sleeps ended at its longest wait with a batch let go
//! (`WorkCounts::scheduler_slice_rescues`).
//!
//! Ignored: run it in release under `--cfg m2v_gpu`,
//! `cargo test --release -p metal2vulkan-runtime --lib event_wake_bench -- --ignored --nocapture`.

use super::blit_holds::{plan_device, Scratch, WAIT};
use crate::{host_time, Buffer, CommandQueue, Device, ResourceOptions};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The repeats of each shape.
const REPEATS: usize = 50;
/// How long the waiting command buffer waits before the value is set.
const SETTLE: Duration = Duration::from_millis(2);
/// The bytes of each fill the other queue's command buffers make.
const FILL_BYTES: u64 = 128 << 20;

/// A shape: the other queue's command buffers ahead of it, and whether an unshared queue
/// committed just before.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    ahead: usize,
    unshared: bool,
}

const SHAPES: [Shape; 4] = [
    Shape {
        name: "alone",
        ahead: 0,
        unshared: false,
    },
    Shape {
        name: "behind 1",
        ahead: 1,
        unshared: false,
    },
    Shape {
        name: "behind 4",
        ahead: 4,
        unshared: false,
    },
    Shape {
        name: "unshared, behind 4",
        ahead: 4,
        unshared: true,
    },
];

/// Per repeat, in seconds: `t0` to the completion seen, `t0` to the handler, and each from the
/// later of `t0` and the other queue's last GPU end.
#[derive(Default)]
struct Times {
    seen: Vec<f64>,
    handler: Vec<f64>,
    seen_after: Vec<f64>,
    handler_after: Vec<f64>,
    rescues: u64,
}

fn spread(values: &[f64]) -> String {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |q: f64| sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)] * 1e6;
    format!(
        "median {:.0} us, p90 {:.0} us, slowest {:.0} us",
        at(0.5),
        at(0.9),
        sorted[sorted.len() - 1] * 1e6
    )
}

/// The queues and objects one repeat uses.
struct Rig {
    device: Device,
    waiting: CommandQueue,
    other: CommandQueue,
    unshared: CommandQueue,
    fill: Buffer,
}

/// One repeat of `shape` with event value `value`, into `times`.
fn repeat(rig: &Rig, shape: Shape, value: u64, times: &mut Times) {
    let device = &rig.device;
    let waited = device.new_shared_event().unwrap();
    let signalled = device.new_shared_event().unwrap();
    if shape.unshared {
        let command_buffer = rig.unshared.command_buffer().unwrap();
        command_buffer.commit().unwrap();
    }
    let ahead: Vec<_> = (0..shape.ahead)
        .map(|_| {
            let command_buffer = rig.other.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&rig.fill, 0..FILL_BYTES, value as u8)
                .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
        })
        .collect();
    let command_buffer = rig.waiting.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event((&waited).into(), value)
        .unwrap();
    command_buffer
        .encode_signal_event((&signalled).into(), value)
        .unwrap();
    let handled = Arc::new(Mutex::new(None::<f64>));
    let noted = Arc::clone(&handled);
    command_buffer
        .add_completed_handler(Box::new(move |_| {
            *noted.lock().unwrap() = Some(host_time());
        }))
        .unwrap();
    command_buffer.commit().unwrap();
    std::thread::sleep(SETTLE);
    let before = device.work_counts().scheduler_slice_rescues;
    let t0 = host_time();
    waited.set_signaled_value(value);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let seen = host_time();
    let start = Instant::now();
    let handler = loop {
        if let Some(at) = *handled.lock().unwrap() {
            break at;
        }
        assert!(start.elapsed() < WAIT, "the completed handler never ran");
        std::thread::sleep(Duration::from_micros(50));
    };
    super::scheduler_wakes::asleep(device, "the waiting command buffer's pass");
    times.rescues += device.work_counts().scheduler_slice_rescues - before;
    let mut ahead_end = t0;
    for other in &ahead {
        other.wait_until_completed_for(WAIT).unwrap();
        if let Ok(end) = other.gpu_end_time() {
            ahead_end = ahead_end.max(end);
        }
    }
    times.seen.push(seen - t0);
    times.handler.push(handler - t0);
    times.seen_after.push(seen - ahead_end);
    times.handler_after.push(handler - ahead_end);
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn event_wake_bench() {
    let scratch = Scratch::new("event-wake");
    let device = plan_device(&scratch.0);
    println!("EVENT-WAKE on {}", device.name());
    let rig = Rig {
        waiting: device.new_command_queue().unwrap(),
        other: device.new_command_queue().unwrap(),
        unshared: device.new_command_queue_unshared(16).unwrap(),
        fill: device
            .new_buffer(FILL_BYTES, ResourceOptions::private())
            .unwrap(),
        device,
    };
    let mut times: Vec<Times> = SHAPES.iter().map(|_| Times::default()).collect();
    let mut value = 0;
    for _ in 0..REPEATS {
        for (shape, times) in SHAPES.iter().zip(&mut times) {
            value += 1;
            repeat(&rig, *shape, value, times);
        }
    }
    for (shape, times) in SHAPES.iter().zip(&times) {
        println!(
            "EVENT-WAKE {}: t0 -> completion seen {}; t0 -> handler {}; from the later of t0 and \
             the other queue's last GPU end: seen {}, handler {}; slice rescues {}",
            shape.name,
            spread(&times.seen),
            spread(&times.handler),
            spread(&times.seen_after),
            spread(&times.handler_after),
            times.rescues
        );
    }
}
