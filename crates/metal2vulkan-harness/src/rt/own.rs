//! The new runtime's rows of its own ([`OwnRow`]), each on a device of its own on the session's
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

use super::error;
use crate::authored::OwnRow;
use crate::run::OwnReport;
use crate::Result;
use ash::vk;
use metal2vulkan_runtime as rt;
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

/// `row` on a device of its own on `physical`.
pub(super) fn run(physical: &rt::PhysicalDevice, row: OwnRow) -> Result<OwnReport> {
    let device = rt::Device::new(physical, rt::DeviceOptions::default()).map_err(error)?;
    match row {
        OwnRow::IdleFrees => idle_frees(&device, false),
        OwnRow::IdleFreesUnderCommits => idle_frees(&device, true),
        OwnRow::ExportSubmitCost => export_submit_cost(&device),
    }
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
