//! Why batches went in kernel submissions of their own under `CompletionPolicy::Grouped` (the
//! runtime's `command_buffer/group.rs`, `WorkCounts::batches_alone_unshared` and kin), on the
//! device the cfg selects.
//!
//! [`a_batch_that_goes_alone_is_counted_once_by_why`], under the validation layer and the
//! default `CompletionPolicy::Grouped`: on a device whose batches group, each command buffer below is counted once by why it went alone, and on
//! any other device none is, while a batch of an unshared queue counts as one everywhere
//! (`WorkCounts::batches_of_unshared_queues`):
//! - a fresh queue's first command buffer: its queue's expected run is unknown;
//! - one committed with `commit_and_submit` while nothing is in flight: the committing thread
//!   submits it alone, or the scheduler's thread, which took it in first, holds it and sends it
//!   alone;
//! - one committed with `commit`, every commit taken in before the wait: held, and sent alone as
//!   no earlier submission of its `VkQueue` is in flight;
//! - one with a completion sync file, where the device exports one: a binary signal;
//! - one of an unshared queue: never in a group;
//! - then one committed with `commit` on the first queue again: held, and sent alone, as an
//!   unshared queue holds its `VkQueue` (a device of one graphics queue: RADV, ANV, lavapipe),
//!   where no group is kept, or else as nothing is in flight.
//!
//! `steady_presents` (a measurement, not a check; ignored, run by name in a release build) runs a
//! steady stream of presents on a device without the validation layer: each frame (16.7 ms apart),
//! three application queues commit [`APP_BUFFERS`] command buffers each from threads of their own,
//! of fills of about 150 to 300 us, 100 to 200 us and 20 to 60 us of the GPU's time, and a
//! compositor queue commits [`COMPOSITOR_BUFFERS`] of about 100 to 200 us 2 ms later, the last
//! with a completion sync file it exports, every one with `commit_and_submit`, as the consumer's
//! guest commits. Four legs: the compositor's queue unshared or not, under each
//! `CompletionPolicy`. Each prints a `STEADY-PRESENT` line: a frame's kernel submissions, batches
//! grouped and each reason a batch went alone, and how late the host saw each command buffer
//! complete (its completed handler against its `GPUEndTime`), with the share seen complete only
//! after the next command buffer the GPU ran ended.

use super::grouped_submissions::{groups, host_seconds, taken_in, warm};
use super::{layer_reported_nothing_at_all, loader, open_device, validated_device_with};
use super::{Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::CompletionPolicy;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandQueue, Device, DeviceOptions, ResourceOptions,
    ShaderCacheOptions, WorkCounts,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The reasons, in the order of the runtime's `Alone`: the ten of batches found with no group
/// held, then the nine of batches held in a group that went holding them alone.
const REASONS: [&str; 19] = [
    "unshared",
    "binary-wait",
    "binary-signal",
    "sparse",
    "pieced",
    "fence",
    "run-unknown",
    "run-long",
    "first-work",
    "at-commit",
    "held:one-pending",
    "held:unshared-hold",
    "held:gpu-wait",
    "held:handlers",
    "held:waited-for",
    "held:kept",
    "held:sent-by-next",
    "held:other",
    "held:unkept",
];

/// The places in [`REASONS`] the test reads.
const UNSHARED: usize = 0;
const BINARY_SIGNAL: usize = 2;
const RUN_UNKNOWN: usize = 6;
const AT_COMMIT: usize = 9;
const ONE_PENDING: usize = 10;
const UNSHARED_HOLD: usize = 11;
const WAITED_FOR: usize = 14;

/// The counts of [`REASONS`], in their order.
fn alone(counts: &WorkCounts) -> [u64; 19] {
    [
        counts.batches_alone_unshared,
        counts.batches_alone_binary_waits,
        counts.batches_alone_binary_signals,
        counts.batches_alone_sparse,
        counts.batches_alone_pieced,
        counts.batches_alone_fences,
        counts.batches_alone_run_unknown,
        counts.batches_alone_run_long,
        counts.batches_alone_first_work,
        counts.batches_alone_at_commit,
        counts.batches_held_alone_one_pending,
        counts.batches_held_alone_unshared_hold,
        counts.batches_held_alone_gpu_waits,
        counts.batches_held_alone_handlers,
        counts.batches_held_alone_waited_for,
        counts.batches_held_alone_kept,
        counts.batches_held_alone_sent_by_next,
        counts.batches_held_alone_other,
        counts.batches_held_alone_unkept,
    ]
}

/// What the device's counts moved by since `before`: the alone counts, the unshared queues'
/// batches, and, for a failure's report, the batches submitted or held (every batch with GPU
/// work, by how far its queue ran behind), those grouped, those at commit, the command buffers
/// taken with no submission, and the kernel submissions.
fn moved(device: &Device, before: &WorkCounts) -> ([u64; 19], u64, String) {
    let after = device.work_counts();
    let (was, now) = (alone(before), alone(&after));
    let since = |count: fn(&WorkCounts) -> u64| count(&after) - count(before);
    let others = format!(
        "batches {}, grouped {}, at commit {}, unsubmitted {}, kernel submissions {}",
        since(|c| {
            c.batches_with_none_pending
                + c.batches_with_one_pending
                + c.batches_with_two_or_three_pending
                + c.batches_with_four_or_more_pending
        }),
        since(|c| c.batches_grouped),
        since(|c| c.batches_submitted_at_commit),
        since(|c| c.command_buffers_unsubmitted),
        since(|c| c.queue_submissions),
    );
    (
        std::array::from_fn(|at| now[at] - was[at]),
        after.batches_of_unshared_queues - before.batches_of_unshared_queues,
        others,
    )
}

/// The one reason `counts` holds, its place in [`REASONS`]: none where it holds none, and a
/// failure where it holds more than one batch.
fn only(counts: &[u64; 19], what: &str) -> Option<usize> {
    assert!(
        counts.iter().sum::<u64>() <= 1,
        "{what}: counted more than once: {}",
        named(counts)
    );
    counts.iter().position(|&count| count == 1)
}

/// `counts` by name, the reasons that moved.
fn named(counts: &[u64; 19]) -> String {
    let moved: Vec<String> = (REASONS.iter().zip(counts))
        .filter(|(_, &count)| count > 0)
        .map(|(name, count)| format!("{name} {count}"))
        .collect();
    match moved.is_empty() {
        true => "none".to_owned(),
        false => moved.join(", "),
    }
}

/// How a command buffer is committed.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Commit {
    /// `commit_and_submit`, waited for at once.
    AtCommit,
    /// `commit`, waited for once every commit was taken in ([`taken_in`]).
    Plain,
}

/// Fill `buffer` on `queue` in one command buffer, committed as `commit` says, with a completion
/// sync file where `sync_file` (exported after the commit), then wait for it.
fn fill(device: &Device, queue: &CommandQueue, buffer: &Buffer, commit: Commit, sync_file: bool) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..buffer.length(), 7).unwrap();
    blit.end_encoding().unwrap();
    if sync_file {
        command_buffer.add_completion_sync_file().unwrap();
    }
    match commit {
        Commit::AtCommit => command_buffer.commit_and_submit().unwrap(),
        Commit::Plain => {
            command_buffer.commit().unwrap();
            taken_in(device);
        }
    }
    if sync_file {
        drop(command_buffer.export_completion_sync_file().unwrap());
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The module docs' check.
#[test]
fn a_batch_that_goes_alone_is_counted_once_by_why() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        // The lane's devices take `DevicePolicy::exact_answers()`, `CompletionPolicy::Exact`
        // among them, under which nothing groups and no reason is counted: this one the default.
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.completion = CompletionPolicy::Grouped;
        })
        .unwrap();
        let grouping = groups(&device);
        let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let unshared_buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        // Each buffer filled once first, on a queue of its own: no step below takes a buffer's
        // first work.
        let first = device.new_command_queue().unwrap();
        for target in [&buffer, &unshared_buffer] {
            fill(&device, &first, target, Commit::Plain, false);
        }
        let queue = device.new_command_queue().unwrap();
        let step = |what: &str, run: &dyn Fn()| {
            let before = device.work_counts();
            run();
            let (counts, unshared, others) = moved(&device, &before);
            eprintln!(
                "ALONE {what}: {}; unshared queues' batches {unshared}; {others}; the device \
                 groups: {grouping}",
                named(&counts)
            );
            (only(&counts, what), unshared)
        };
        // The expected reason where the device groups; none elsewhere.
        let expect = |reason: usize| grouping.then_some(reason);

        let (why, _) = step("a fresh queue's first", &|| {
            fill(&device, &queue, &buffer, Commit::AtCommit, false)
        });
        assert_eq!(why, expect(RUN_UNKNOWN));

        warm(&device, &queue);
        let (why, _) = step("at commit, nothing in flight", &|| {
            fill(&device, &queue, &buffer, Commit::AtCommit, false)
        });
        // The committing thread's, or the scheduler's thread's where it took the commit in first:
        // held, sent alone as nothing is in flight, or for the wait that followed.
        match grouping {
            true => assert!(
                matches!(why, Some(AT_COMMIT | ONE_PENDING | WAITED_FOR)),
                "{why:?}"
            ),
            false => assert_eq!(why, None),
        }

        let (why, _) = step("committed, nothing in flight", &|| {
            fill(&device, &queue, &buffer, Commit::Plain, false)
        });
        assert_eq!(why, expect(ONE_PENDING));

        let sync_files = {
            let probe = queue.command_buffer().unwrap();
            probe.add_completion_sync_file().is_ok()
        };
        if sync_files {
            let (why, _) = step("with a completion sync file", &|| {
                fill(&device, &queue, &buffer, Commit::AtCommit, true)
            });
            assert_eq!(why, expect(BINARY_SIGNAL));
        } else {
            eprintln!("ALONE the device exports no sync file: that step is left out");
        }

        let sharing_before = device.work_counts().unshared_queues_sharing_a_vk_queue;
        let unshared_queue = device.new_command_queue_unshared(64).unwrap();
        let shares = device.work_counts().unshared_queues_sharing_a_vk_queue > sharing_before;
        let (why, unshared) = step("an unshared queue's", &|| {
            fill(
                &device,
                &unshared_queue,
                &unshared_buffer,
                Commit::AtCommit,
                false,
            )
        });
        assert_eq!(why, expect(UNSHARED));
        assert_eq!(unshared, 1, "the unshared queue's batches");

        let (why, unshared) = step("committed beside an unshared queue", &|| {
            fill(&device, &queue, &buffer, Commit::Plain, false)
        });
        assert_eq!(unshared, 0, "the unshared queues' batches");
        assert_eq!(
            why,
            expect(match shares {
                true => UNSHARED_HOLD,
                false => ONE_PENDING,
            }),
            "the unshared queue shares a VkQueue: {shares}"
        );
    }
    layer_reported_nothing_at_all(&messages);
}

/// The command buffers each application queue of [`steady_presents`] commits a frame.
const APP_BUFFERS: usize = 4;
/// The command buffers its compositor queue commits a frame, the last with a sync file.
const COMPOSITOR_BUFFERS: usize = 3;
/// The frames each leg runs unmeasured, then measured.
const WARM_FRAMES: usize = 30;
const FRAMES: usize = 240;
/// A frame's period: 60 presents a second.
const PERIOD: Duration = Duration::from_nanos(16_666_667);
/// How long after a frame's start the compositor commits.
const COMPOSITOR_AFTER: Duration = Duration::from_millis(2);

/// Each queue's fills' GPU times in microseconds, from a frame's first command buffer on: three
/// application queues', then the compositor's.
const RUNS: [[u64; APP_BUFFERS]; 3] =
    [[300, 150, 250, 200], [100, 200, 150, 120], [20, 60, 40, 30]];
const COMPOSITOR_RUNS: [u64; COMPOSITOR_BUFFERS] = [200, 150, 100];

/// The bytes a fill writes in a microsecond of the device's GPU: one command buffer of a 64 MiB
/// fill on a warmed queue, by its GPU time.
fn fill_rate(device: &Device, queue: &CommandQueue) -> f64 {
    const BYTES: u64 = 64 << 20;
    let buffer = device
        .new_buffer(BYTES, ResourceOptions::private())
        .unwrap();
    let mut rate = 0.0;
    for _ in 0..3 {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..BYTES, 1).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let took =
            command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        rate = BYTES as f64 / (took * 1e6);
    }
    rate
}

/// The bytes of a fill of `micros` of GPU time at `rate`, in whole 4 KiB pages, at least one.
fn bytes_for(micros: u64, rate: f64) -> u64 {
    (((micros as f64 * rate) as u64).div_ceil(4096) * 4096).max(4096)
}

/// One command buffer of the measurement once it completed: its GPU start and end and its
/// completed handler's start, each in seconds on the host's clock.
#[derive(Clone, Copy)]
struct Seen {
    start: f64,
    end: f64,
    handled: f64,
}

/// The command buffers of the measurement seen complete, as their completed handlers note them.
type Noted = Arc<Mutex<Vec<Seen>>>;

/// Commit a command buffer on `queue` that fills `bytes` of `buffer`, with a completion sync file
/// (exported at once) where `sync_file`, with `commit_and_submit`, and let it go: its completed
/// handler counts it in `done`, and, where `seen` is given, first notes its [`Seen`] there.
fn commit_fill(
    queue: &CommandQueue,
    (buffer, bytes): (&Buffer, u64),
    sync_file: bool,
    (done, seen): (&Arc<AtomicUsize>, Option<&Noted>),
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..bytes, bytes as u8).unwrap();
    blit.end_encoding().unwrap();
    let (done, seen) = (Arc::clone(done), seen.map(Arc::clone));
    command_buffer
        .add_completed_handler(Box::new(move |finished: &CommandBuffer| {
            let handled = host_seconds();
            assert_eq!(finished.error(), None);
            let times = (finished.gpu_start_time(), finished.gpu_end_time());
            if let (Some(seen), (Ok(start), Ok(end))) = (seen, times) {
                (seen.lock().unwrap()).push(Seen {
                    start,
                    end,
                    handled,
                });
            }
            done.fetch_add(1, Ordering::SeqCst);
        }))
        .unwrap();
    let sync_file = sync_file && command_buffer.add_completion_sync_file().is_ok();
    command_buffer.commit_and_submit().unwrap();
    if sync_file {
        drop(command_buffer.export_completion_sync_file().unwrap());
    }
}

/// Wait, without blocking on a command buffer (which would send a group holding it at once),
/// until `done` reaches `count`.
fn wait_done(done: &AtomicUsize, count: usize) {
    let deadline = Instant::now() + WAIT;
    while done.load(Ordering::SeqCst) < count {
        assert!(
            Instant::now() < deadline,
            "a command buffer never completed"
        );
        std::thread::sleep(Duration::from_micros(50));
    }
}

/// The value at `at` (0 to 1) of `values`, sorted.
fn percentile(values: &[f64], at: f64) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * at).round() as usize]
}

/// One leg of the measurement (the module docs).
fn steady_leg(completion: CompletionPolicy, unshared: bool) {
    let cache = ScratchCache::new().unwrap();
    let mut options = DeviceOptions::default();
    options.policy.completion = completion;
    options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
    options.loader = loader();
    let device = open_device(TEST_DEVICE, options).unwrap();
    let grouping = groups(&device);
    let apps: Vec<CommandQueue> = (0..RUNS.len())
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    let compositor = match unshared {
        true => device.new_command_queue_unshared(64).unwrap(),
        false => device.new_command_queue().unwrap(),
    };
    for queue in apps.iter().chain([&compositor]) {
        warm(&device, queue);
    }
    let rate = fill_rate(&device, &apps[0]);
    let sizes: Vec<Vec<u64>> = (RUNS.iter())
        .map(|runs| runs.iter().map(|&micros| bytes_for(micros, rate)).collect())
        .chain([COMPOSITOR_RUNS
            .iter()
            .map(|&micros| bytes_for(micros, rate))
            .collect()])
        .collect();
    let buffers: Vec<Buffer> = (sizes.iter())
        .map(|sizes| {
            let most = sizes.iter().copied().max().unwrap();
            device.new_buffer(most, ResourceOptions::private()).unwrap()
        })
        .collect();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let done = Arc::new(AtomicUsize::new(0));
    let a_frame = RUNS.len() * APP_BUFFERS + COMPOSITOR_BUFFERS;
    let mut before = None;
    let mut started = Instant::now();
    for frame in 0..WARM_FRAMES + FRAMES {
        // At most two frames in flight: the one before last completed before this one commits.
        wait_done(&done, frame.saturating_sub(1) * a_frame);
        if frame == WARM_FRAMES {
            wait_done(&done, frame * a_frame);
            before = Some(device.work_counts());
            started = Instant::now();
        }
        let measured = (frame >= WARM_FRAMES).then_some(&seen);
        let start = Instant::now();
        std::thread::scope(|scope| {
            for ((queue, buffer), sizes) in apps.iter().zip(&buffers).zip(&sizes) {
                let done = &done;
                scope.spawn(move || {
                    for &bytes in sizes {
                        commit_fill(queue, (buffer, bytes), false, (done, measured));
                    }
                });
            }
            std::thread::sleep(COMPOSITOR_AFTER.saturating_sub(start.elapsed()));
            let (buffer, sizes) = (&buffers[RUNS.len()], &sizes[RUNS.len()]);
            for (at, &bytes) in sizes.iter().enumerate() {
                let last = at + 1 == sizes.len();
                commit_fill(&compositor, (buffer, bytes), last, (&done, measured));
            }
        });
        std::thread::sleep(PERIOD.saturating_sub(start.elapsed()));
    }
    wait_done(&done, (WARM_FRAMES + FRAMES) * a_frame);
    let wall = started.elapsed();
    let (before, after) = (before.unwrap(), device.work_counts());
    let frames = FRAMES as f64;
    let per_frame =
        |count: fn(&WorkCounts) -> u64| (count(&after) - count(&before)) as f64 / frames;
    let (was, now) = (alone(&before), alone(&after));
    let reasons: Vec<String> = (REASONS.iter().enumerate())
        .filter(|&(at, _)| now[at] > was[at])
        .map(|(at, name)| format!("{name} {:.2}", (now[at] - was[at]) as f64 / frames))
        .collect();
    let mut seen = seen.lock().unwrap().clone();
    seen.sort_by(|a, b| a.start.total_cmp(&b.start));
    let lags: Vec<f64> = seen
        .iter()
        .map(|one| (one.handled - one.end) * 1e6)
        .collect();
    // Seen complete only after the end of the command buffer the GPU ran next.
    let after_next = (seen.windows(2))
        .filter(|pair| pair[0].handled > pair[1].end)
        .count();
    let gpu: Vec<f64> = seen.iter().map(|one| (one.end - one.start) * 1e6).collect();
    eprintln!(
        "STEADY-PRESENT {} {completion:?} compositor unshared {unshared}: the device groups \
         {grouping}; {:.0} bytes/us; a frame: {:.2} kernel submissions (vkQueueSubmit calls), \
         {:.2} grouped calls holding {:.2} batches, kept groups {:.2}, unshared queues' batches \
         {:.2}, batches by pending [{:.2}, {:.2}, {:.2}, {:.2}], went alone: {}; GPU per command \
         buffer p50 {:.0} us; completion seen after GPUEndTime p50 {:.0} us, p99 {:.0}, max \
         {:.0}; {after_next} of {} seen only after the next one's end; {:.2} s",
        device.name(),
        rate,
        per_frame(|c| c.queue_submissions),
        per_frame(|c| c.grouped_submissions),
        per_frame(|c| c.batches_grouped),
        per_frame(|c| c.groups_kept_pending),
        per_frame(|c| c.batches_of_unshared_queues),
        per_frame(|c| c.batches_with_none_pending),
        per_frame(|c| c.batches_with_one_pending),
        per_frame(|c| c.batches_with_two_or_three_pending),
        per_frame(|c| c.batches_with_four_or_more_pending),
        match reasons.is_empty() {
            true => "none".to_owned(),
            false => reasons.join(", "),
        },
        percentile(&gpu, 0.5),
        percentile(&lags, 0.5),
        percentile(&lags, 0.99),
        percentile(&lags, 1.0),
        seen.len(),
        wall.as_secs_f64(),
    );
}

/// The module docs' measurement: four legs, the compositor's queue unshared, then not, each
/// under `Grouped`, then `Exact`.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all, in a release build"]
fn steady_presents() {
    for unshared in [true, false] {
        for completion in [CompletionPolicy::Grouped, CompletionPolicy::Exact] {
            steady_leg(completion, unshared);
        }
    }
}
