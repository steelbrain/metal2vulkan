//! Command buffers' GPU times across queues, under the GPU cfg on every GPU: a short command
//! buffer reports its completion, with its GPU times, while a long one of another queue committed
//! after it still runs.
//!
//! Not under the CPU device cfg: lavapipe's read of a query pool's results waits for the whole
//! device to idle (`lvp_GetQueryPoolResults`, with or without the wait flag), so there the short
//! one's completion waits for the long one.

use crate::tests::command::on_device::{each_device, finished, within, Arena, Busy};
use crate::tests::compute::on_device::step;
use crate::{
    host_time, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState,
    Device, DispatchType, MemoryClass,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The words of the long command buffer's buffer, each dispatch's grid: 64 MiB.
const WORDS: u32 = 1 << 24;

/// The GPU time the long command buffer is sized to, in nanoseconds: far past the time a
/// completion takes to reach its handler, so on a device whose timestamps are read apart the
/// long one is still running when the short one's handler runs.
const LONG_NS: u64 = 250_000_000;

/// The most dispatches the long command buffer holds, on the fastest device.
const MOST: u64 = 20_000;

/// The dispatches whose time from commit to completion sizes the long command buffer: enough that
/// the commit's and the completion's own latency are a small part of it.
const PROBE: u64 = 64;

/// A Private buffer of [`WORDS`] words.
fn words(device: &Device) -> Buffer {
    let length = u64::from(WORDS) * 4;
    let block = (device.take_block(MemoryClass::Private, length)).expect("a Private block");
    (device.new_buffer(block.id, 0, length, None)).expect("a buffer")
}

/// A command buffer of `queue` of `count` dispatches of `state` over `buffer`, in one serial
/// encoder, so each waits for the one before it.
fn long(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    buffer: &Buffer,
    count: u64,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().expect("a command buffer");
    let mut encoder =
        (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
    encoder.set_compute_pipeline_state(state).expect("a state");
    encoder.set_buffer(Some(buffer), 0, 0).expect("a buffer");
    for _ in 0..count {
        (encoder.dispatch_threads([WORDS, 1, 1], [256, 1, 1])).expect("a dispatch");
    }
    encoder.end_encoding();
    command_buffer
}

/// When a command buffer's completed handler ran, on the host clock, with the GPU times it
/// read there, and the handlers' order.
#[derive(Default)]
struct Reports {
    order: Vec<&'static str>,
    short: Option<(u64, u64, u64)>,
}

/// A command buffer's GPU times are its own, not every queue's: a short command buffer
/// committed on one queue, then a long one (about [`LONG_NS`] of GPU time) on another, both
/// submitted before either runs (behind a batch that waits for the host, [`Busy`]), so the
/// short one runs first and completes while the long one runs. The short one's completed
/// handler, with its GPU times, runs before the long one's GPU end, and before the long one's
/// handler. A driver whose timestamps become readable only once every queue's work, or all
/// the work submitted after them, completes holds the short one's completion until the long
/// one ends, which a program sees as a stall of one queue behind another's frame.
#[test]
fn every_device_reports_a_short_command_buffer_while_a_long_one_after_it_runs() {
    each_device("stamps_across_queues", |name, _, device| {
        let state = step(&device);
        let buffer = words(&device);
        let (short_queue, long_queue) = (
            device.new_command_queue(4).expect("a queue"),
            device.new_command_queue(4).expect("a queue"),
        );
        // The long command buffer's size: as many dispatches as take LONG_NS, from the host
        // time [`PROBE`] take from their commit to their completion. Its GPU times cannot size
        // it: Vulkan lets a timestamp be written at any stage logically later than the one named,
        // so a command buffer's start and end may read as one instant (KosmicKrisp's do for one
        // compute encoder's work). A first command buffer of one dispatch runs before it, so
        // nothing a first use waits for is timed.
        let first = long(&long_queue, &state, &buffer, 1);
        first.commit().expect("a commit");
        finished(name, &first);
        let probe = long(&long_queue, &state, &buffer, PROBE);
        let began = Instant::now();
        probe.commit().expect("a commit");
        finished(name, &probe);
        let took = u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
        assert_eq!(
            probe.status(),
            CommandBufferStatus::Completed,
            "{name}: the probe's status, its error {:?}",
            probe.error()
        );
        let count = (LONG_NS * PROBE).div_ceil(took.max(1)).clamp(8, MOST);

        // Both command buffers are made before the batch that holds the queue's work, so
        // nothing they make waits behind it.
        let reports = Arc::new(Mutex::new(Reports::default()));
        let mut arena = Arena::new(&device);
        let small = arena.buffer();
        let short = short_queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (short.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(&state).expect("a state");
        encoder.set_buffer(Some(&small), 0, 0).expect("a buffer");
        encoder
            .dispatch_threads([64, 1, 1], [64, 1, 1])
            .expect("a dispatch");
        encoder.end_encoding();
        let reported = Arc::clone(&reports);
        (short.add_completed_handler(move |done| {
            let at = host_time();
            let mut reports = reported.lock().expect("the reports");
            reports.order.push("short");
            reports.short = Some((at, done.gpu_start_time(), done.gpu_end_time()));
        }))
        .expect("a handler");
        let long = long(&long_queue, &state, &buffer, count);
        let reported = Arc::clone(&reports);
        (long.add_completed_handler(move |_| {
            reported.lock().expect("the reports").order.push("long")
        }))
        .expect("a handler");
        let busy = Busy::new(&device);
        for command_buffer in [&short, &long] {
            command_buffer.commit().expect("a commit");
            within(name, || {
                command_buffer.status() == CommandBufferStatus::Scheduled
            });
        }
        busy.open();
        finished(name, &short);
        finished(name, &long);
        within(name, || {
            reports.lock().expect("the reports").order.len() == 2
        });
        let reports = reports.lock().expect("the reports");
        let (at, start, end) = reports.short.expect("the short one's report");
        let long_end = long.gpu_end_time();
        println!(
            "{name}: short handler at {at}, its GPU {start}..{end}; long GPU {}..{long_end} ({count} dispatches)",
            long.gpu_start_time()
        );
        assert!(
            start > 0 && end >= start,
            "{name}: the short one's GPU times in its handler"
        );
        assert!(
            at < long_end,
            "{name}: the short one's handler ran {} us after the long one's GPU end",
            (at - long_end) / 1000
        );
        assert_eq!(
            reports.order,
            ["short", "long"],
            "{name}: the handlers' order"
        );
        drop(busy);
    });
}
