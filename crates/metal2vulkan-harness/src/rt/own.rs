//! The runtime's rows of its own ([`OwnRow`]), each on a device of its own on the session's
//! GPU, so that nothing an earlier workload left in the session's device runs beside it.
//!
//! The idle rows read the runtime's count of each idle free by the allocation's idle time when
//! the free began (`Metrics::idle_frees_by_age`), whose second bucket, 250 to 275 ms, holds a
//! free that began within 25 ms of the idle age (250 ms). A small block held to the end keeps the
//! class's first allocation live. The class keeps one wholly free unit, its reserve: the last idle
//! unit allocation in the order made, which the idle rule never frees. So each block of a unit is
//! taken once the block-maker thread has made the class's wholly free unit ahead of need
//! (`Gauges::shared_reserve_bytes`), and the row waits for the one it makes after the last: every
//! unit given back was made before the reserve, and is weighed, and freed, at its own idle age.
//! Each block is given back with no free running: far enough apart without commits, and with them
//! once every free before it ended. A free in any other bucket fails the row: one that began 25 ms
//! or more past its idle age, or one before it.
//!
//! The export row times `vkQueueSubmit2` of one command buffer that records a global memory
//! barrier, on the presenting queue, waited for by a fence before the next, in rounds that take
//! turns with no export and with 64 live; its times are the medians of the rounds' medians, in
//! microseconds. A device that exports no handle reports so and times nothing.
//!
//! The releases row reads the runtime's gauges with its work in flight, never at an idle device:
//! a release that waits for more than the work committed before it is freed only as the GPU idles,
//! so a read at idle would not see it wait.
//!
//! The two-queue row sizes queue A's command buffers by one fill's GPU run, the least of three
//! on a queue of its own, and times queue B's from just before its commit to its completed
//! handler. A's thread commits while its queue has room for a command buffer, so its backlog
//! stays full throughout.

use super::error;
use crate::authored::OwnRow;
use crate::run::OwnReport;
use crate::Result;
use ash::vk;
use metal2vulkan_runtime as rt;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

/// `Metrics::idle_frees_by_age`'s buckets, in milliseconds of idle time when the free began.
const AGES: [&str; 6] = [
    "<250", "250-275", "275-350", "350-500", "500-1000", ">=1000",
];

/// The bucket of a free that began within 25 ms of its idle age.
const ON_TIME: usize = 1;

/// How far apart the idle row without commits gives its blocks back: past a free's length on
/// every device measured (an allocation of 16 MiB on the NVIDIA cards in Thunderbolt enclosures,
/// about 18 ms at 1.1 ms a MiB), so no free waits for another.
const APART: Duration = Duration::from_millis(150);

/// When the idle row with commits gives its first block back, after its commits began.
const FIRST_AFTER: Duration = Duration::from_millis(500);

/// How often the idle row with commits commits a command buffer.
const COMMIT_EVERY: Duration = Duration::from_millis(2);

/// The longest an idle row waits for its frees.
const WAIT: Duration = Duration::from_secs(10);

/// The exports live in the export row's second kind of round: the consumer's cap of surface
/// caches.
const EXPORTS: usize = 64;

/// The bytes of each export of the export row.
const EXPORT_LEN: u64 = 1 << 16;

/// The submissions of each kind in a round of the export row.
const SUBMITS: usize = 200;

/// The rounds of the export row.
const ROUNDS: usize = 3;

/// The iterations of the releases row, and those after which it reads the gauges.
const ITERATIONS: u32 = 200;
const READ_AFTER: [u32; 2] = [100, 200];

/// The command buffers the releases row keeps in flight.
const IN_FLIGHT: usize = 3;

/// The textures each iteration of the releases row makes, each with a view, and their side.
const TEXTURES: u32 = 8;
const SIDE: u32 = 64;

/// The bytes of the buffer the releases row's command buffers write, and the fills of it each
/// makes: 1 GiB written a command buffer.
const FILL_LEN: u64 = 256 << 20;
const FILLS: u32 = 4;

/// The iterations' worth of releases waiting and of live textures the releases row allows at
/// each read: one for each command buffer in flight and one for the one being made.
const HELD_ITERATIONS: u64 = IN_FLIGHT as u64 + 1;

/// The iterations' worth of the texture blocks' bytes the releases row allows at each read: the
/// iterations whose images today's runtime still holds ([`releases_under_work`]).
const PLACED_ITERATIONS: u64 = 2 * IN_FLIGHT as u64 - 1;

/// The command buffers queue A of the two-queue row keeps uncompleted, its queue's maximum; the
/// GPU run it sizes each to, in nanoseconds, and the most fills it gives one for it.
const BACKLOG: u32 = 8;
const LONG_RUN: u64 = 20_000_000;
const MOST_FILLS: u64 = 256;

/// Queue B's maximum in the two-queue row, past the command buffers it keeps uncompleted, and the
/// bytes of the buffer each of its command buffers fills once.
const B_COUNT: u32 = 64;
const SMALL_LEN: u64 = 4096;

/// How often the two-queue row commits queue B's command buffer, and how many it commits in each
/// part.
const FRAME: Duration = Duration::from_micros(16_667);
const FRAMES: u32 = 60;

/// `row` on a device of its own on `physical`.
pub(super) fn run(physical: &rt::PhysicalDevice, row: OwnRow) -> Result<OwnReport> {
    let device = rt::Device::new(physical, rt::DeviceOptions::default()).map_err(error)?;
    match row {
        OwnRow::IdleFrees => idle_frees(&device, false),
        OwnRow::IdleFreesUnderCommits => idle_frees(&device, true),
        OwnRow::ExportSubmitCost => export_submit_cost(&device),
        OwnRow::ReleasesUnderWork => releases_under_work(&device),
        OwnRow::TwoQueues => two_queues(&device),
    }
}

/// The two-queue row ([`OwnRow::TwoQueues`]): its part with unrelated work, then its part in
/// the consumer's shape, each of A's command buffers signalling at its tail and a waiter on B
/// committed right after it.
fn two_queues(device: &rt::Device) -> Result<OwnReport> {
    let block = (device.take_block(rt::MemoryClass::Private, FILL_LEN)).map_err(error)?;
    let long = (device.new_buffer(block.id, 0, FILL_LEN, None)).map_err(error)?;
    let small_block = (device.take_block(rt::MemoryClass::Shared, SMALL_LEN)).map_err(error)?;
    let small = (device.new_buffer(small_block.id, 0, SMALL_LEN, None)).map_err(error)?;
    // One fill's GPU run, the least of five on a queue of its own after a warm-up of
    // [`MOST_FILLS`] fills that brings the GPU's clock up, sizes A's command buffers.
    let calibrate = device.new_command_queue(1).map_err(error)?;
    let mut fill_run = u64::MAX;
    for fills in [MOST_FILLS, 1, 1, 1, 1, 1] {
        let command_buffer = calibrate.command_buffer().map_err(error)?;
        fill(&command_buffer, &long, FILL_LEN, fills)?;
        command_buffer.commit().map_err(error)?;
        command_buffer.wait_until_completed().map_err(error)?;
        let run = (command_buffer.gpu_end_time()).saturating_sub(command_buffer.gpu_start_time());
        if fills == 1 {
            fill_run = fill_run.min(run.max(1));
        }
    }
    let fills = LONG_RUN.div_ceil(fill_run).clamp(1, MOST_FILLS);
    let unrelated = two_queue_part(device, &long, &small, fills, false)?;
    let signalled = two_queue_part(device, &long, &small, fills, true)?;
    drop((long, block, small, small_block));
    let at = |values: &[f64], p| crate::run::percentile(values, p).unwrap_or(0.0);
    let times = vec![
        ("b_p50", at(&unrelated.b, 50)),
        ("b_p90", at(&unrelated.b, 90)),
        ("b_max", at(&unrelated.b, 100)),
        ("a_run", at(&unrelated.a_run, 50)),
        ("a_gap_p50", at(&unrelated.a_gap, 50)),
        ("a_gap_p90", at(&unrelated.a_gap, 90)),
        ("a_per_cb", unrelated.a_per_cb),
        ("wait_p50", at(&signalled.after, 50)),
        ("wait_p90", at(&signalled.after, 90)),
        ("wait_a_gap_p50", at(&signalled.a_gap, 50)),
        ("wait_a_gap_p90", at(&signalled.a_gap, 90)),
        ("wait_a_per_cb", signalled.a_per_cb),
    ];
    let ms = |us: f64| us / 1000.0;
    let lines = vec![
        format!(
            "B every {FRAME:?} beside A's backlog of {BACKLOG}, {fills} fills each, {:.1} ms \
             on the GPU at p50: B committed to completed p50 {:.1} ms, p90 {:.1} ms, max \
             {:.1} ms; A's GPU idle before each of its p50 {:.1} us, p90 {:.1} us, its GPU time \
             a command buffer {:.1} us; {} held for room",
            ms(at(&unrelated.a_run, 50)),
            ms(at(&unrelated.b, 50)),
            ms(at(&unrelated.b, 90)),
            ms(at(&unrelated.b, 100)),
            at(&unrelated.a_gap, 50),
            at(&unrelated.a_gap, 90),
            unrelated.a_per_cb,
            unrelated.held,
        ),
        format!(
            "each of A's signalling at its tail, a waiter on B committed right after it: B's \
             GPU start after the signaller's GPU end p50 {:.1} us, p90 {:.1} us; A's GPU idle \
             before each of its p50 {:.1} us, p90 {:.1} us, its GPU time a command buffer {:.1} \
             us; B committed to completed p50 {:.1} ms; {} held for room",
            at(&signalled.after, 50),
            at(&signalled.after, 90),
            at(&signalled.a_gap, 50),
            at(&signalled.a_gap, 90),
            signalled.a_per_cb,
            ms(at(&signalled.b, 50)),
            signalled.held,
        ),
    ];
    Ok(OwnReport { lines, times })
}

/// What a part of the two-queue row measured, in microseconds: each of B's command buffers'
/// commit to completed, each GPU start of B's after the GPU end of the signaller it waited for,
/// each of A's GPU runs and the GPU idle before each of A's, from the end of A's before it, less
/// B's runs between them; and the command buffers held for room during it.
struct Part {
    b: Vec<f64>,
    after: Vec<f64>,
    a_run: Vec<f64>,
    a_gap: Vec<f64>,
    a_per_cb: f64,
    held: u64,
}

/// One part of the two-queue row: queue A keeps [`BACKLOG`] command buffers of `fills` fills of
/// `long` uncompleted, from a thread of its own. Without `signal`, once it has, queue B commits
/// [`FRAMES`] command buffers, one each [`FRAME`], each one fill of `small`. With it, each of
/// A's signals a shared event with its count after its fills, and A's thread commits on B right
/// after it a command buffer that waits for that count, then fills `small` once, for as long.
/// Every command buffer is to complete, and each of B's after the signaller it waited for.
fn two_queue_part(
    device: &rt::Device,
    long: &rt::Buffer,
    small: &rt::Buffer,
    fills: u64,
    signal: bool,
) -> Result<Part> {
    let event = device.new_shared_event().map_err(error)?;
    let a = device.new_command_queue(BACKLOG).map_err(error)?;
    let b = device.new_command_queue(B_COUNT).map_err(error)?;
    let held = || device.counters().metrics().held_for_room;
    let before = held();
    let committed = Mutex::new(Vec::new());
    let lock = || committed.lock().unwrap_or_else(PoisonError::into_inner);
    let frames = Mutex::new(Vec::new());
    let done: Arc<Mutex<Vec<(usize, Instant)>>> = Arc::default();
    let stop = AtomicBool::new(false);
    // B's command buffer, waiting first for `value` where it is above 0, committed now, its
    // completion timed by its completed handler.
    let commit_b = |value: u64| {
        let fail = |error: rt::Error| error.to_string();
        let command_buffer = b.command_buffer().map_err(fail)?;
        if value > 0 {
            (command_buffer.encode_wait_for_event(event.as_event(), value)).map_err(fail)?;
        }
        fill(&command_buffer, small, SMALL_LEN, 1).map_err(|error| error.to_string())?;
        let mut frames = frames.lock().unwrap_or_else(PoisonError::into_inner);
        let (done, index) = (Arc::clone(&done), frames.len());
        (command_buffer.add_completed_handler(move |_| {
            let mut done = done.lock().unwrap_or_else(PoisonError::into_inner);
            done.push((index, Instant::now()));
        }))
        .map_err(fail)?;
        let at = Instant::now();
        command_buffer.commit().map_err(fail)?;
        frames.push((command_buffer, value, at));
        Ok::<_, String>(())
    };
    let (ran, made) = thread::scope(|scope| {
        // A commits while it has room for a command buffer, as Metal's `commandBuffer` waits.
        let backlog = scope.spawn(|| {
            let fail = |error: rt::Error| error.to_string();
            while !stop.load(Ordering::Relaxed) {
                let command_buffer = a.command_buffer().map_err(fail)?;
                fill(&command_buffer, long, FILL_LEN, fills).map_err(|error| error.to_string())?;
                let mut committed = lock();
                let value = committed.len() as u64 + 1;
                if signal {
                    (command_buffer.encode_signal_event(event.as_event(), value)).map_err(fail)?;
                }
                command_buffer.commit().map_err(fail)?;
                committed.push(command_buffer);
                drop(committed);
                if signal {
                    commit_b(value)?;
                }
            }
            Ok::<_, String>(())
        });
        let ran = (|| -> Result<()> {
            let since = Instant::now();
            while lock().len() < BACKLOG as usize {
                if since.elapsed() > WAIT || backlog.is_finished() {
                    return Err(format!("A committed {} of {BACKLOG}", lock().len()).into());
                }
                thread::sleep(Duration::from_millis(1));
            }
            let started = Instant::now();
            for frame in 0..FRAMES {
                thread::sleep((started + FRAME * frame).saturating_duration_since(Instant::now()));
                if !signal {
                    commit_b(0)?;
                }
            }
            Ok(())
        })();
        stop.store(true, Ordering::Relaxed);
        let made = (backlog.join()).unwrap_or_else(|_| Err("A's thread panicked".into()));
        (ran, made)
    });
    made?;
    ran?;
    let of_a = committed
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    let frames = frames.into_inner().unwrap_or_else(PoisonError::into_inner);
    let all = (of_a.iter()).chain(frames.iter().map(|(command_buffer, ..)| command_buffer));
    for command_buffer in all {
        command_buffer.wait_until_completed().map_err(error)?;
        let status = command_buffer.status();
        if status != rt::CommandBufferStatus::Completed {
            return Err(format!("a command buffer ended {status:?}").into());
        }
    }
    // Completed handlers run after the completion that ends the waits.
    let started = Instant::now();
    while done.lock().unwrap_or_else(PoisonError::into_inner).len() < frames.len() {
        if started.elapsed() > WAIT {
            return Err(format!("B's completed handlers did not run in {WAIT:?}").into());
        }
        thread::sleep(Duration::from_millis(1));
    }
    let done = done.lock().unwrap_or_else(PoisonError::into_inner);
    let us = |nanos: f64| nanos / 1000.0;
    // The GPU time from `earlier`'s end to `later`'s start, in microseconds.
    let gap = |earlier: &rt::CommandBuffer, later: &rt::CommandBuffer| {
        us(later.gpu_start_time() as f64 - earlier.gpu_end_time() as f64)
    };
    let mut after = Vec::new();
    for (command_buffer, value, _) in frames.iter().filter(|(_, value, _)| *value > 0) {
        let signaller = &of_a[*value as usize - 1];
        if command_buffer.gpu_end_time() < signaller.gpu_end_time() {
            return Err(format!("B's command buffer ended before the signal of {value}").into());
        }
        after.push(gap(signaller, command_buffer));
    }
    // A's GPU time a command buffer, its first start to its last end over their count.
    let a_per_cb = match (of_a.first(), of_a.last()) {
        (Some(first), Some(last)) => {
            us(last.gpu_end_time() as f64 - first.gpu_start_time() as f64) / of_a.len() as f64
        }
        _ => 0.0,
    };
    Ok(Part {
        a_per_cb,
        b: (done.iter())
            .map(|&(index, at)| us(at.duration_since(frames[index].2).as_nanos() as f64))
            .collect(),
        after,
        a_run: (of_a.iter())
            .map(|command_buffer| {
                us(command_buffer
                    .gpu_end_time()
                    .saturating_sub(command_buffer.gpu_start_time()) as f64)
            })
            .collect(),
        // The GPU idle between two of A's: their gap less B's runs within it.
        a_gap: (of_a.windows(2))
            .map(|pair| {
                let (from, to) = (pair[0].gpu_end_time(), pair[1].gpu_start_time());
                let within = (frames.iter()).map(|(command_buffer, ..)| {
                    let start = command_buffer.gpu_start_time().max(from);
                    command_buffer.gpu_end_time().min(to).saturating_sub(start)
                });
                gap(&pair[0], &pair[1]) - us(within.sum::<u64>() as f64)
            })
            .collect(),
        held: held() - before,
    })
}

/// Encodes into `command_buffer` `fills` fills of `buffer`'s first `len` bytes.
fn fill(
    command_buffer: &rt::CommandBuffer,
    buffer: &rt::Buffer,
    len: u64,
    fills: u64,
) -> Result<()> {
    let mut encoder = command_buffer.blit_command_encoder().map_err(error)?;
    for fill in 0..fills {
        (encoder.fill_buffer(buffer, 0..len, fill as u8)).map_err(error)?;
    }
    encoder.end_encoding();
    Ok(())
}

/// The releases row ([`OwnRow::ReleasesUnderWork`]).
fn releases_under_work(device: &rt::Device) -> Result<OwnReport> {
    let counters = device.counters();
    let class = rt::IndexClass::Texture;
    let range = class.range();
    (device.grant_indices(class, range.start, range.end - range.start)).map_err(error)?;
    let block = (device.take_block(rt::MemoryClass::Private, FILL_LEN)).map_err(error)?;
    let buffer = (device.new_buffer(block.id, 0, FILL_LEN, None)).map_err(error)?;
    let queue = device.new_command_queue(IN_FLIGHT as u32).map_err(error)?;
    let mut descriptor = rt::TextureDescriptor::new(
        rt::TextureType::Type2D,
        rt::PixelFormat::RGBA8Unorm,
        SIDE,
        SIDE,
    );
    descriptor.usage = rt::TextureUsage::SHADER_READ;
    descriptor.storage_mode = rt::StorageMode::Private;
    let mut indices = range;
    let mut index = || indices.next().ok_or("the texture indices ran out");
    // The command buffer at the front, waited for, is to have completed.
    let completed = |command_buffer: rt::CommandBuffer| -> Result<()> {
        command_buffer.wait_until_completed().map_err(error)?;
        match command_buffer.status() {
            rt::CommandBufferStatus::Completed => Ok(()),
            status => Err(format!("a command buffer ended {status:?}").into()),
        }
    };
    let mut in_flight = VecDeque::with_capacity(IN_FLIGHT);
    // The live textures and the texture blocks' bytes one iteration's textures and views add.
    let mut each = None;
    let mut reads = Vec::new();
    for iteration in 1..=ITERATIONS {
        let before = counters.gauges();
        let mut made = Vec::with_capacity(2 * TEXTURES as usize);
        for _ in 0..TEXTURES {
            let texture = (device.new_texture(&descriptor, Some(index()?))).map_err(error)?;
            let format = rt::PixelFormat::RGBA8Unorm;
            let view = texture.new_texture_view(
                format,
                rt::TextureType::Type2D,
                0..1,
                0..1,
                Some(index()?),
            );
            made.push(view.map_err(error)?);
            made.push(texture);
        }
        // Read at the first iteration, before anything is released.
        each.get_or_insert_with(|| {
            let after = counters.gauges();
            (
                after.textures - before.textures,
                after.texture_block_bytes - before.texture_block_bytes,
            )
        });
        let command_buffer = queue.command_buffer().map_err(error)?;
        let mut encoder = command_buffer.blit_command_encoder().map_err(error)?;
        for _ in 0..FILLS {
            (encoder.fill_buffer(&buffer, 0..FILL_LEN, iteration as u8)).map_err(error)?;
        }
        encoder.end_encoding();
        command_buffer.commit().map_err(error)?;
        in_flight.push_back(command_buffer);
        // Released under the work in flight, the command buffer just committed among it.
        drop(made);
        if in_flight.len() == IN_FLIGHT {
            completed(in_flight.pop_front().ok_or("a command buffer in flight")?)?;
        }
        if READ_AFTER.contains(&iteration) {
            reads.push(counters.gauges());
        }
    }
    for command_buffer in in_flight {
        completed(command_buffer)?;
    }
    drop((buffer, block));
    let (textures, bytes) = each.ok_or("no iteration ran")?;
    let [first, second] = [&reads[0], &reads[1]];
    // The bounds, held at both reads, in what the first iteration added. A read follows the wait
    // for the command buffer committed `IN_FLIGHT - 1` iterations before. Each texture and view
    // leaves its image or views and its index, released at its drop against the command buffers
    // committed by then: `HELD_ITERATIONS` iterations' worth of releases waiting and of live
    // textures, one for each command buffer in flight and one for the one being made. An image
    // stays longer, in today's structure: the upload that zeroed it rides its iteration's command
    // buffer and holds it until that retires, and its release is then queued against every
    // command buffer committed so far, the `IN_FLIGHT - 1` after its own. So an iteration's views
    // stay `IN_FLIGHT` iterations and its images `PLACED_ITERATIONS`, here 3 and 5, and the texture
    // blocks' bytes, the images' places, are allowed `PLACED_ITERATIONS` iterations' worth. A
    // runtime that frees an image sooner only lowers the reads.
    let gauges = [
        (
            "releases_waiting",
            first.releases_waiting,
            second.releases_waiting,
            2 * textures * HELD_ITERATIONS,
        ),
        (
            "textures",
            first.textures,
            second.textures,
            textures * HELD_ITERATIONS,
        ),
        (
            "texture_block_bytes",
            first.texture_block_bytes,
            second.texture_block_bytes,
            bytes * PLACED_ITERATIONS,
        ),
    ];
    let shown = (gauges.iter())
        .map(|(gauge, first, second, bound)| {
            format!("{gauge} {first} -> {second} (at most {bound})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    let [at, then] = READ_AFTER;
    let line = format!(
        "{shown} from iteration {at} to {then}, {IN_FLIGHT} command buffers writing 1 GiB each in flight"
    );
    let over: Vec<&str> = (gauges.iter())
        .filter(|(_, first, second, bound)| first.max(second) > bound)
        .map(|(gauge, ..)| *gauge)
        .collect();
    if !over.is_empty() {
        return Err(format!(
            "{} passed its bound under the work: {line}",
            over.join(", ")
        )
        .into());
    }
    Ok(OwnReport {
        lines: vec![line],
        times: Vec::new(),
    })
}

/// The idle row (the module's docs): four Shared blocks, each taken in a unit made ahead of need,
/// given back [`APART`], or with `commits` two, while a command buffer is committed every
/// [`COMMIT_EVERY`], the first [`FIRST_AFTER`] in and the second once the first's free ended.
fn idle_frees(device: &rt::Device, commits: bool) -> Result<OwnReport> {
    let shared = rt::MemoryClass::Shared;
    let counters = device.counters();
    // Until the class holds a wholly free unit the block-maker thread made ahead of need.
    let made_ahead = || {
        let since = Instant::now();
        while counters.gauges().shared_reserve_bytes == 0 {
            if since.elapsed() > WAIT {
                return Err(format!("no Shared unit made ahead in {WAIT:?}"));
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    };
    let small = device.take_block(shared, 1).map_err(error)?;
    let count = if commits { 2 } else { 4 };
    let mut blocks = Vec::new();
    for _ in 0..count {
        made_ahead()?;
        blocks.push(
            device
                .take_block(shared, shared.block_len())
                .map_err(error)?,
        );
    }
    // The unit made after the last take is the reserve, made after every unit given back.
    made_ahead()?;
    let queue = device.new_command_queue(1).map_err(error)?;
    let before = counters.metrics();
    let freed = || counters.metrics().allocations_freed - before.allocations_freed;
    let started = Instant::now();
    let (mut given, mut last, mut committed) = (0u64, None::<Instant>, 0u64);
    while freed() < count {
        if started.elapsed() > WAIT {
            return Err(
                format!("{} of {count} idle allocations freed in {WAIT:?}", freed()).into(),
            );
        }
        let next = match commits {
            false => last.is_none_or(|at| at.elapsed() >= APART),
            true => match given {
                0 => started.elapsed() >= FIRST_AFTER,
                _ => freed() == given,
            },
        };
        // A block is given back by its release alone: a dropped one would be given back too.
        if next && !blocks.is_empty() {
            device.release_block(blocks.remove(0)).map_err(error)?;
            given += 1;
            last = Some(Instant::now());
        }
        if commits {
            queue
                .command_buffer()
                .and_then(|command_buffer| command_buffer.commit())
                .map_err(error)?;
            committed += 1;
            thread::sleep(COMMIT_EVERY);
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }
    let after = counters.metrics();
    device.release_block(small).map_err(error)?;
    let frees = after.allocations_freed - before.allocations_freed;
    let by_age: Vec<u64> = (after.idle_frees_by_age.iter())
        .zip(&before.idle_frees_by_age)
        .map(|(after, before)| after - before)
        .collect();
    let shown = (AGES.iter().zip(&by_age))
        .map(|(age, count)| format!("{age} {count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let during = match commits {
        true => format!(" while {committed} command buffers were committed"),
        false => String::new(),
    };
    let line = format!("{frees} frees{during}, by ms idle when each began: {shown}");
    let late: u64 = by_age[ON_TIME + 1..].iter().sum();
    if late > 0 {
        return Err(format!(
            "{late} of {frees} frees began 25 ms or more past the idle age: {line}"
        )
        .into());
    }
    if by_age[ON_TIME] != frees {
        return Err(format!("not every free began within 25 ms of the idle age: {line}").into());
    }
    Ok(OwnReport {
        lines: vec![line],
        times: Vec::new(),
    })
}

/// The export row (the module's docs).
fn export_submit_cost(device: &rt::Device) -> Result<OwnReport> {
    let handles = [
        rt::ExportHandle::OpaqueFd,
        rt::ExportHandle::DmaBuf,
        rt::ExportHandle::OpaqueWin32,
    ];
    let mut chosen = None;
    for handle in handles {
        match device.new_exported_block(handle, EXPORT_LEN) {
            Ok(export) => {
                chosen = Some((handle, export));
                break;
            }
            Err(rt::Error::Limit(_)) => {}
            Err(why) => return Err(error(why)),
        }
    }
    let Some((handle, export)) = chosen else {
        return Ok(OwnReport {
            lines: vec!["exports no handle: nothing timed".to_owned()],
            times: Vec::new(),
        });
    };
    drop(export);
    exports_freed(device)?;
    let raw = device.raw();
    let ash = raw.device;
    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(raw.queue_family_index);
    // SAFETY: the family is the device's.
    let pool = unsafe { ash.create_command_pool(&pool_info, None) }?;
    let allocate = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    // SAFETY: the pool is live; the command buffer is new, and records one barrier.
    let command = unsafe {
        let command = ash.allocate_command_buffers(&allocate)?[0];
        ash.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
        let dependency = vk::DependencyInfo::default().memory_barriers(&barrier);
        ash.cmd_pipeline_barrier2(command, &dependency);
        ash.end_command_buffer(command)?;
        command
    };
    // SAFETY: the device is live.
    let fence = unsafe { ash.create_fence(&vk::FenceCreateInfo::default(), None) }?;
    // The median nanoseconds of one submit.
    let submits = || -> Result<u64> {
        let mut times = Vec::with_capacity(SUBMITS);
        for _ in 0..SUBMITS {
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
            let submit = [vk::SubmitInfo2::default().command_buffer_infos(&commands)];
            let start = Instant::now();
            // SAFETY: the command buffer is executable and not pending (the fence waited for its
            // last submission); the queue is held for the call.
            device
                .with_present_queue(|queue| unsafe { ash.queue_submit2(queue, &submit, fence) })?;
            times.push(start.elapsed().as_nanos() as u64);
            // SAFETY: the fence was submitted.
            unsafe {
                ash.wait_for_fences(&[fence], true, u64::MAX)?;
                ash.reset_fences(&[fence])?;
            }
        }
        times.sort_unstable();
        Ok(times[SUBMITS / 2])
    };
    let timed = rounds(device, handle, &submits);
    // SAFETY: nothing is pending: each submission was waited for.
    unsafe {
        ash.destroy_fence(fence, None);
        ash.destroy_command_pool(pool, None);
    }
    let (mut none, mut live) = timed?;
    none.sort_unstable();
    live.sort_unstable();
    let (none, live) = (none[ROUNDS / 2] as f64 / 1e3, live[ROUNDS / 2] as f64 / 1e3);
    Ok(OwnReport {
        lines: vec![format!(
            "{handle:?}: a submit takes {none:.3} us with no export and {live:.3} us with {EXPORTS} live"
        )],
        times: vec![("none", none), ("exports_64", live)],
    })
}

/// The export row's [`ROUNDS`] of `submits` (a submission's median nanoseconds) with no export
/// and with [`EXPORTS`] of `handle` live: each kind's medians.
fn rounds(
    device: &rt::Device,
    handle: rt::ExportHandle,
    submits: &dyn Fn() -> Result<u64>,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let (mut none, mut live) = (Vec::new(), Vec::new());
    for _ in 0..ROUNDS {
        none.push(submits()?);
        let exports = (0..EXPORTS)
            .map(|_| device.new_exported_block(handle, EXPORT_LEN))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(error)?;
        live.push(submits()?);
        drop(exports);
        exports_freed(device)?;
    }
    Ok((none, live))
}

/// Waits, at most [`WAIT`], until `device` holds no export.
fn exports_freed(device: &rt::Device) -> Result<()> {
    let started = Instant::now();
    while device.counters().gauges().exports != 0 {
        if started.elapsed() > WAIT {
            return Err(format!("the exports not freed in {WAIT:?}").into());
        }
        thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}
