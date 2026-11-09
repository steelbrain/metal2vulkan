//! How long a command buffer's completion takes to reach its completed handlers and its waiter
//! where its command pool's give-back frees command buffers (`spares.rs`): a command buffer of
//! [`BURST`] blit encoders fills its pool, then [`RECENT_USES`] command buffers of one encoder
//! each take the pool again, the last of them the one whose give-back frees the burst's command
//! buffers. Each command buffer's status is read on a thread of its own from its commit until it
//! reads final; the time from then to its first completed handler, and to its waiter's return, is
//! taken for that last command buffer and for those before it, whose give-backs free none. Each
//! round waits [`SETTLE`] after its burst and after its last command buffer, so that what a
//! give-back left to the device's free thread ran before the next command buffer takes the pool.
//!
//! Printed (`SPARES-LATENCY`): the medians and slowest of both, and the rounds whose burst's
//! command buffers went from the spares. Ignored: run it alone, by name, under `--cfg m2v_gpu`
//! (or `m2v_cpu_icd`), in release:
//! `cargo nextest run -p metal2vulkan-runtime --lib --release --run-ignored all spares_latency
//! --no-capture`.

use crate::spares::RECENT_USES;
use crate::tests::freed_memory::device;
use crate::{Buffer, CommandBufferStatus, CommandQueue, ResourceOptions};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The blit encoders of each round's burst.
const BURST: usize = 64;

/// The rounds.
const ROUNDS: usize = 32;

/// The longest a command buffer is waited for.
const WAIT: Duration = Duration::from_secs(30);

/// How long a round waits after its burst and after its last command buffer.
const SETTLE: Duration = Duration::from_millis(50);

/// Commit a command buffer of `queue` of `encoders` blit encoders, each filling `target`, and
/// wait for it: the time from its status reading final, on a thread that reads it from the
/// commit on, to its first completed handler, and to the wait's return.
fn measured(queue: &CommandQueue, target: &Buffer, encoders: usize) -> (Duration, Duration) {
    let command_buffer = queue.command_buffer().unwrap();
    for at in 0..encoders {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(target, 0..4096, at as u8).unwrap();
        blit.end_encoding().unwrap();
    }
    let handled = Arc::new(Mutex::new(None));
    let handler = Arc::clone(&handled);
    command_buffer
        .add_completed_handler(Box::new(move |_| {
            *handler.lock().unwrap() = Some(Instant::now());
        }))
        .unwrap();
    let watched = command_buffer.clone();
    let watch = std::thread::spawn(move || loop {
        let status = watched.status();
        if matches!(
            status,
            CommandBufferStatus::Completed | CommandBufferStatus::Error
        ) {
            return Instant::now();
        }
        std::hint::spin_loop();
    });
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let returned = Instant::now();
    let final_at = watch.join().unwrap();
    let handled =
        (handled.lock().unwrap()).expect("its completed handlers ran before its waiter returned");
    (
        handled.saturating_duration_since(final_at),
        returned.saturating_duration_since(final_at),
    )
}

/// The median and slowest of `times`, in microseconds.
fn spread(times: &mut [Duration]) -> String {
    times.sort_unstable();
    let us = |time: Duration| time.as_secs_f64() * 1e6;
    format!(
        "median {:.1} us, slowest {:.1} us",
        us(times[times.len() / 2]),
        us(times[times.len() - 1])
    )
}

#[test]
#[ignore = "a measurement: run alone on its device by name"]
fn spares_latency() {
    let scratch = std::env::temp_dir().join(format!("m2v-spares-latency-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let device = device(&scratch);
    let queue = device.new_command_queue().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let (mut none_handler, mut none_wait) = (Vec::new(), Vec::new());
    let (mut freeing_handler, mut freeing_wait) = (Vec::new(), Vec::new());
    let mut freed = 0;
    for _ in 0..ROUNDS {
        measured(&queue, &target, BURST);
        std::thread::sleep(SETTLE);
        let held = device.shared().spares.command_buffers_held();
        for use_ in 1..=RECENT_USES {
            let (handler, wait) = measured(&queue, &target, 1);
            match use_ == RECENT_USES {
                true => {
                    freeing_handler.push(handler);
                    freeing_wait.push(wait);
                }
                false => {
                    none_handler.push(handler);
                    none_wait.push(wait);
                }
            }
        }
        std::thread::sleep(SETTLE);
        if device.shared().spares.command_buffers_held() + BURST / 2 <= held {
            freed += 1;
        }
    }
    println!(
        "SPARES-LATENCY {}: {ROUNDS} rounds, the burst's command buffers gone from the spares \
         after {freed}; from the status reading final to the first completed handler: give-backs \
         that free none {}, the give-back that frees the burst's {}; to the waiter's return: {} \
         and {}",
        device.name(),
        spread(&mut none_handler),
        spread(&mut freeing_handler),
        spread(&mut none_wait),
        spread(&mut freeing_wait),
    );
    drop((target, queue, device));
    let _ = std::fs::remove_dir_all(&scratch);
}
