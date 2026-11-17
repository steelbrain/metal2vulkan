//! A command buffer's times: Metal's `kernelStartTime`, `kernelEndTime`, `GPUStartTime` and
//! `GPUEndTime`, all in nanoseconds of one host clock, [`host_time`].
//!
//! Each command buffer owns two timestamp queries in its queue's pool. Its first command buffer
//! resets them and writes the first where its commands begin; its last segment writes the second
//! where it ends. The GPU end time is the host time at which the completion thread saw the command
//! buffer complete, and the GPU start time is that less the exact GPU duration between the two
//! timestamps, so ends never decrease in completion order. The kernel times are the host time at
//! commit and when the submit call that passed the last segment returned.

use crate::device::{lock, Core};
use crate::error::Error;
use ash::vk;

/// No Metal counterpart: the host clock every time of a command buffer counts, in nanoseconds:
/// `CLOCK_UPTIME_RAW` on macOS, which `mach_absolute_time` and so Metal's times count,
/// `CLOCK_MONOTONIC_RAW` elsewhere on Unix, and the performance counter (`QueryPerformanceCounter`)
/// on Windows. 0 where the clock cannot be read.
pub fn host_time() -> u64 {
    #[cfg(unix)]
    {
        read_clock().unwrap_or(0)
    }
    #[cfg(windows)]
    {
        read_counter().unwrap_or(0)
    }
}

/// The host's clock now.
#[cfg(unix)]
fn read_clock() -> Option<u64> {
    #[cfg(target_os = "macos")]
    const CLOCK: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
    #[cfg(not(target_os = "macos"))]
    const CLOCK: libc::clockid_t = libc::CLOCK_MONOTONIC_RAW;
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to fill.
    if unsafe { libc::clock_gettime(CLOCK, &mut now) } != 0 {
        return None;
    }
    let seconds = u64::try_from(now.tv_sec).ok()?;
    let nanos = u64::try_from(now.tv_nsec).ok()?;
    Some(seconds * 1_000_000_000 + nanos)
}

/// The performance counter now, in nanoseconds of its frequency, which is fixed at boot.
#[cfg(windows)]
fn read_counter() -> Option<u64> {
    use std::sync::OnceLock;
    use windows_sys::Win32::System::Performance::{
        QueryPerformanceCounter, QueryPerformanceFrequency,
    };
    static FREQUENCY: OnceLock<Option<u64>> = OnceLock::new();
    let frequency = FREQUENCY.get_or_init(|| {
        let mut frequency = 0_i64;
        // SAFETY: `frequency` is a valid value for the call to fill.
        let read = unsafe { QueryPerformanceFrequency(&mut frequency) };
        (read != 0)
            .then(|| u64::try_from(frequency).ok())
            .flatten()
            .filter(|&frequency| frequency > 0)
    });
    let mut ticks = 0_i64;
    // SAFETY: `ticks` is a valid value for the call to fill.
    let read = unsafe { QueryPerformanceCounter(&mut ticks) };
    let ticks = (read != 0).then(|| u64::try_from(ticks).ok()).flatten()?;
    let nanos = u128::from(ticks) * 1_000_000_000 / u128::from((*frequency)?);
    u64::try_from(nanos).ok()
}

/// A command buffer's times in nanoseconds of [`host_time`], each 0 until known.
#[derive(Clone, Copy, Default)]
pub(crate) struct Times {
    pub(crate) kernel_start: u64,
    pub(crate) kernel_end: u64,
    pub(crate) gpu_start: u64,
    pub(crate) gpu_end: u64,
}

/// The device's timestamp facts: what a tick is, and which bits count.
#[derive(Clone, Copy)]
pub(crate) struct Ticks {
    /// `timestampPeriod`: nanoseconds per tick.
    period: f64,
    /// The queue family's `timestampValidBits`, as a mask.
    mask: u64,
}

impl Ticks {
    pub(crate) fn new(period: f32, valid_bits: u32) -> Ticks {
        Ticks {
            period: f64::from(period),
            mask: u64::MAX >> (64 - valid_bits.clamp(1, 64)),
        }
    }

    /// The GPU times of a command buffer seen complete at `seen` with others, from its two
    /// timestamps and `last`, the end timestamp of the one of them that ended last: the end is
    /// `seen` less the ticks from its end to `last`, and the start the end less the ticks between
    /// its timestamps, each within the valid bits, which stays exact across the counter's wrap.
    /// Any values give times no later than `seen`.
    pub(crate) fn place(&self, seen: u64, last: u64, start: u64, end: u64) -> (u64, u64) {
        let nanos = |from: u64, to: u64| {
            let ticks = to.wrapping_sub(from) & self.mask;
            // A float-to-integer cast saturates.
            (ticks as f64 * self.period) as u64
        };
        let end_time = seen.saturating_sub(nanos(end, last));
        (end_time.saturating_sub(nanos(start, end)), end_time)
    }
}

/// A queue's timestamp pool: two queries per command buffer the queue may have made and not yet
/// retired.
pub(crate) struct Stamps {
    pub(crate) pool: vk::QueryPool,
}

impl Stamps {
    /// A pool for `slots` command buffers.
    pub(crate) fn new(core: &Core, slots: u32) -> Result<Stamps, Error> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(slots.saturating_mul(2));
        // SAFETY: the info is complete.
        let pool = unsafe { core.device.create_query_pool(&info, None) }
            .map_err(Error::vulkan("vkCreateQueryPool"))?;
        lock(&core.query_pools).push(pool);
        Ok(Stamps { pool })
    }

    /// The first of `slot`'s two queries.
    pub(crate) fn first(slot: u32) -> u32 {
        slot * 2
    }
}

/// Resets a slot's two queries on the host (`hostQueryReset`, in the floor), as a command buffer
/// takes the slot: before any submission that names them, which a host reset happens before
/// ("anything that happened-before" `vkQueueSubmit2` is in its first synchronization scope), so no
/// command buffer the GPU runs resets a query.
pub(crate) fn reset(device: &ash::Device, pool: vk::QueryPool, first: u32) {
    // SAFETY: the device enables `hostQueryReset`; both queries are in the pool; the slot's
    // previous command buffer retired, so no submitted command that names them is pending, and
    // its readback finished before it gave the slot back.
    unsafe { device.reset_query_pool(pool, first, 2) };
}

/// Records, at the start of a command buffer's first `VkCommandBuffer`, its start timestamp. Its
/// two queries were reset on the host as it took its slot ([`reset`]).
pub(crate) fn write_start(
    device: &ash::Device,
    command: vk::CommandBuffer,
    pool: vk::QueryPool,
    first: u32,
) {
    // SAFETY: the buffer is recording outside any render pass; the query is in the pool and was
    // reset on the host before any submission of it.
    unsafe {
        device.cmd_write_timestamp2(command, vk::PipelineStageFlags2::TOP_OF_PIPE, pool, first);
    }
}

/// Records the end timestamp: written once every earlier command of the queue has completed.
pub(crate) fn write_end(
    device: &ash::Device,
    command: vk::CommandBuffer,
    pool: vk::QueryPool,
    first: u32,
) {
    // SAFETY: the buffer is recording, and the query is in the pool and was reset on the host
    // before any submission of it.
    unsafe {
        device.cmd_write_timestamp2(
            command,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            pool,
            first + 1,
        )
    };
}
