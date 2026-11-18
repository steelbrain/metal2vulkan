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

    /// Nanoseconds from timestamp `from` to the later `to`, within the valid bits, which stays
    /// exact across the counter's wrap.
    fn nanos(&self, from: u64, to: u64) -> f64 {
        (to.wrapping_sub(from) & self.mask) as f64 * self.period
    }

    /// Nanoseconds the valid bits count before they wrap.
    fn range(&self) -> f64 {
        (self.mask as f64 + 1.0) * self.period
    }
}

/// Passes of the completion thread after which the anchor moves later by the least lateness it
/// saw in them, so it follows a GPU clock that runs slow against the host's.
const WINDOW: u32 = 64;

/// The GPU's clock placed on the host's, for the completion thread: the host time of one end
/// timestamp, taken from when the thread saw that command buffer complete, which is late by
/// however long the thread took. It moves earlier at once when a later completion is seen sooner
/// than it predicts, so no end is placed after it was seen; it moves later only by the least
/// lateness of [`WINDOW`] passes, so ends placed against it keep their order.
#[derive(Default)]
pub(crate) struct Anchor {
    /// The host nanoseconds of `tick`, with their fraction; None until the first completion.
    at: Option<(f64, u64)>,
    /// The least lateness of the passes since the anchor last moved later.
    least: f64,
    passes: u32,
}

impl Anchor {
    /// A pass of the thread saw command buffers complete at host time `seen`, the last of which
    /// wrote the end timestamp `last`: the anchor moves to it.
    pub(crate) fn see(&mut self, ticks: &Ticks, seen: u64, last: u64) {
        let seen = seen as f64;
        let predicted = self
            .at
            // Past half the counter's range, its ticks no longer tell the time between.
            .filter(|&(host, _)| seen - host < ticks.range() / 2.0)
            .map(|(host, tick)| host + ticks.nanos(tick, last))
            .filter(|&predicted| predicted <= seen);
        let Some(predicted) = predicted else {
            *self = Anchor {
                at: Some((seen, last)),
                least: f64::INFINITY,
                passes: 0,
            };
            return;
        };
        self.least = self.least.min(seen - predicted);
        self.passes += 1;
        let mut host = predicted;
        if self.passes == WINDOW {
            host += self.least;
            self.least = f64::INFINITY;
            self.passes = 0;
        }
        self.at = Some((host, last));
    }

    /// The GPU times of a command buffer seen in the last pass, from its two timestamps: the end
    /// the ticks from its end to the pass's last before the anchor, the start its own duration
    /// before that. 0 and 0 before any pass.
    pub(crate) fn place(&self, ticks: &Ticks, start: u64, end: u64) -> (u64, u64) {
        let Some((host, tick)) = self.at else {
            return (0, 0);
        };
        // A float-to-integer cast saturates at 0.
        let end_time = (host - ticks.nanos(end, tick)) as u64;
        (
            end_time.saturating_sub(ticks.nanos(start, end) as u64),
            end_time,
        )
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
        reset(&core.device, pool, 0, slots.saturating_mul(2));
        Ok(Stamps { pool })
    }

    /// The first of `slot`'s two queries.
    pub(crate) fn first(slot: u32) -> u32 {
        slot * 2
    }
}

/// Resets `count` queries from `first` on the host (`hostQueryReset`, in the floor): every query
/// as the pool is made, and as a command buffer retires, those of its slot's two that a
/// submission wrote, so a free slot's queries are always reset. A reset happens before any later
/// submission that names them ("anything that happened-before" `vkQueueSubmit2` is in its first
/// synchronization scope), so no command buffer the GPU runs resets a query. A query no
/// submission wrote (its command buffer ended or was dropped before its commands went) is still
/// reset, so it is not reset again: the validation layer (Vulkan SDK 1.4.363) reports a reset of
/// a query not available while other work naming the pool is pending.
pub(crate) fn reset(device: &ash::Device, pool: vk::QueryPool, first: u32, count: u32) {
    if count == 0 {
        return;
    }
    // SAFETY: the device enables `hostQueryReset`; the queries are in the pool; no submitted
    // command that names them is pending (the pool is new, or the command buffer that wrote them
    // retired), and their readback, on the thread that resets them, finished first.
    unsafe { device.reset_query_pool(pool, first, count) };
}

/// Records, at the start of a command buffer's first `VkCommandBuffer`, its start timestamp. Its
/// two queries are reset: a free slot's always are ([`reset`]).
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
