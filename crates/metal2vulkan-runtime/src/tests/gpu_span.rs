//! A command buffer's GPU span holds its own work: what goes ahead of it in the queue, the Managed
//! buffers' copy-ins in its own submission and the command buffers committed before it, is outside
//! `gpu_start_time..gpu_end_time`, as a Mac reports it (an upload sits between a command buffer's
//! scheduling and its GPU start there).

use crate::times::{Stamped, Ticks};

/// Whether `command_buffer`'s pair starts at or after `copy_in`'s ends, on the GPU's counter and
/// across its wrap ([`Ticks::nanos`]): equal stamps count as after.
pub(crate) fn starts_after(ticks: &Ticks, copy_in: Stamped, command_buffer: Stamped) -> bool {
    ticks.nanos(copy_in.end, command_buffer.start) >= 0.0
}

/// [`starts_after`] holds for a start after the copy-in's end, at it, and past the counter's wrap,
/// and fails for a start before it, the case the device test exists to catch.
#[test]
fn a_start_before_the_copy_in_s_end_fails_the_ordering() {
    let ticks = Ticks::new(52.083_332, 36);
    let top = (1u64 << 36) - 1;
    let pair = |own, start, end| Stamped { own, start, end };
    let copy_in = pair(true, 900, 1000);
    assert!(starts_after(&ticks, copy_in, pair(false, 1053, 1100)));
    assert!(starts_after(&ticks, copy_in, pair(false, 1000, 1100)));
    assert!(!starts_after(&ticks, copy_in, pair(false, 999, 1100)));
    assert!(starts_after(
        &ticks,
        pair(true, top - 20, top - 10),
        pair(false, 5, 9)
    ));
    assert!(!starts_after(
        &ticks,
        pair(true, 0, 5),
        pair(false, top - 10, top - 5)
    ));
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod on_device {
    use crate::device::lock;
    use crate::tests::command::on_device::{
        each_device, each_device_in_both_forms, finished, Arena,
    };
    use crate::tests::compute::on_device::step;
    use crate::tests::managed::on_device::new_managed;
    use crate::{
        Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
        DispatchType, MemoryClass,
    };
    use std::time::Instant;

    /// The words of a long command buffer's buffer, each dispatch's grid: 64 MiB.
    pub(crate) const WORDS: u32 = 1 << 24;

    /// The most dispatches a long command buffer holds, on the fastest device.
    const MOST: u64 = 20_000;

    /// The dispatches whose time from commit to completion sizes a long command buffer: enough
    /// that the commit's and the completion's own latency are a small part of it.
    const PROBE: u64 = 64;

    /// The bytes of the Managed range named ahead of the short command buffer: 256 MiB, which a
    /// Thunderbolt link copies in about 130 ms.
    const NAMED: u64 = 256 << 20;

    /// A Private buffer of [`WORDS`] words.
    pub(crate) fn words(device: &Device) -> Buffer {
        let length = u64::from(WORDS) * 4;
        let block = (device.take_block(MemoryClass::Private, length)).expect("a Private block");
        (device.new_buffer(block.id, 0, length, None)).expect("a buffer")
    }

    /// A command buffer of `queue` of `count` dispatches of `state` over `buffer`, `threads` each
    /// ([`WORDS`] for a long one), in one serial encoder, so each waits for the one before it.
    pub(crate) fn long(
        queue: &CommandQueue,
        state: &ComputePipelineState,
        buffer: &Buffer,
        count: u64,
        threads: u32,
    ) -> CommandBuffer {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        encoder.set_compute_pipeline_state(state).expect("a state");
        encoder.set_buffer(Some(buffer), 0, 0).expect("a buffer");
        for _ in 0..count {
            (encoder.dispatch_threads([threads, 1, 1], [threads.min(256), 1, 1]))
                .expect("a dispatch");
        }
        encoder.end_encoding();
        command_buffer
    }

    /// The dispatches of [`long`] over [`WORDS`] that take about `nanos` on `queue`'s device, from
    /// the host time [`PROBE`] take from their commit to their completion (a command buffer's GPU
    /// times cannot size it: Vulkan lets a timestamp be written at any stage logically later than
    /// the one named). A first command buffer of one dispatch runs before it, so nothing a first
    /// use waits for is timed.
    pub(crate) fn dispatches_for(
        name: &str,
        queue: &CommandQueue,
        state: &ComputePipelineState,
        buffer: &Buffer,
        nanos: u64,
    ) -> u64 {
        let first = long(queue, state, buffer, 1, WORDS);
        first.commit().expect("a commit");
        finished(name, &first);
        let probe = long(queue, state, buffer, PROBE, WORDS);
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
        (nanos * PROBE).div_ceil(took.max(1)).clamp(8, MOST)
    }

    /// A Managed buffer's copy-in is outside the GPU span of the command buffer it goes ahead of:
    /// a command buffer of one small dispatch over a Managed buffer whose [`NAMED`] bytes were
    /// named before its commit starts, on the GPU's own counter, at or after its copy-in ends
    /// ([`starts_after`](super::starts_after), over the pairs the completion thread read,
    /// `Core::stamped`). A start written before the copy-in ends puts the upload inside it. The
    /// spans' lengths are not compared: beside other work on the GPU a short command buffer's span
    /// stretches past its copy-in's while its start stays after the copy-in's end. Equal stamps
    /// pass, as a driver that samples at encoder boundaries gives them (the runtime README's known
    /// gaps, "Mesa's Apple-GPU driver"). Only where the Managed buffer has a device copy.
    #[test]
    fn every_device_leaves_a_copy_in_out_of_the_gpu_span_it_goes_ahead_of() {
        each_device_in_both_forms("copy-in outside the span", |name, _, device| {
            if device.has_unified_memory() {
                println!("copy-in outside the span: {name}: unified, no copy-in");
                return;
            }
            let core = device.core();
            let cpu = (device.take_block(MemoryClass::Shared, NAMED))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let managed = (new_managed(&device, cpu.id, 0, NAMED, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let state = step(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            // A first command buffer, so the timed one meets no first use.
            let warm = long(&queue, &state, &managed, 1, 64);
            warm.commit().expect("a commit");
            finished(name, &warm);
            lock(&core.stamped).clear();
            (managed.did_modify_range(0, NAMED)).unwrap_or_else(|error| panic!("{name}: {error}"));
            let short = long(&queue, &state, &managed, 1, 64);
            short.commit().expect("a commit");
            finished(name, &short);
            assert_eq!(short.error(), None, "{name}");
            let pairs = lock(&core.stamped).clone();
            assert_eq!(
                pairs.iter().map(|pair| pair.own).collect::<Vec<_>>(),
                [true, false],
                "{name}: the copy-in's pair, then the command buffer's, both read: {pairs:?}"
            );
            let (copy_in, command_buffer) = (pairs[0], pairs[1]);
            let after = core.ticks.nanos(copy_in.end, command_buffer.start);
            println!(
                "copy-in outside the span: {name}: copy-in {} us, command buffer {} us, its start \
                 {after} ns after the copy-in's end ({pairs:?})",
                core.ticks.span(copy_in.start, copy_in.end) / 1000,
                core.ticks.span(command_buffer.start, command_buffer.end) / 1000,
            );
            assert!(
                super::starts_after(&core.ticks, copy_in, command_buffer),
                "{name}: the command buffer starts {} ns before its copy-in ends",
                -after
            );
            drop((warm, short, managed, queue));
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// A command buffer's GPU start is not before the end of the one committed before it on its
    /// queue: a long command buffer (about 60 ms of dispatches) and a short one committed at once
    /// after it.
    #[test]
    fn every_device_starts_a_command_buffer_after_the_one_before_it_ends() {
        each_device("start after the one before", |name, _, device| {
            let state = step(&device);
            let buffer = words(&device);
            let queue = device.new_command_queue(4).expect("a queue");
            let count = dispatches_for(name, &queue, &state, &buffer, 60_000_000);
            let mut arena = Arena::new(&device);
            let small = arena.buffer();
            let first = long(&queue, &state, &buffer, count, WORDS);
            let second = long(&queue, &state, &small, 1, 64);
            first.commit().expect("a commit");
            second.commit().expect("a commit");
            finished(name, &first);
            finished(name, &second);
            let (end, start) = (first.gpu_end_time(), second.gpu_start_time());
            println!(
                "start after the one before: {name}: first {}..{end}, second {start}..{} ({count} dispatches)",
                first.gpu_start_time(),
                second.gpu_end_time()
            );
            assert!(
                start >= end,
                "{name}: the second's GPU start is {} us before the first's GPU end",
                (end - start) / 1000
            );
            drop((first, second, small, arena, buffer, queue));
        });
    }
}
