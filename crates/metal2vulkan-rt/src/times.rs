//! A command buffer's times: Metal's `kernelStartTime`, `kernelEndTime`, `GPUStartTime` and
//! `GPUEndTime`, all in nanoseconds of one host clock, [`host_time`].
//!
//! Each command buffer owns two timestamp queries in its queue's pool. Its first command buffer
//! writes the first where its commands begin; its last segment writes the second where it ends.
//! The GPU times are the two timestamps placed on the host clock by the device's last calibration
//! (`VK_EXT_calibrated_timestamps`, one reading of the GPU's clock and the host's together): one
//! as the device starts, then one at a completion once [`CALIBRATION_PERIOD`] has passed since
//! the last. The kernel times are the host time at commit and when the submit call that passed the
//! last segment returned.

use crate::device::{lock, Core};
use crate::error::Error;
use ash::vk;

/// No Metal counterpart: the host clock every time of a command buffer counts, its GPU times
/// included, in nanoseconds: `CLOCK_MONOTONIC` on Linux and other Unix systems,
/// `CLOCK_MONOTONIC_RAW` on macOS, and the performance counter (`QueryPerformanceCounter`) on
/// Windows, each the host time domain the device calibrates its GPU clock against
/// ([`HOST_DOMAIN`]). On macOS it counts while the system sleeps, where Metal's times
/// (`mach_absolute_time`) do not. 0 where the clock cannot be read.
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
    const CLOCK: libc::clockid_t = libc::CLOCK_MONOTONIC_RAW;
    #[cfg(not(target_os = "macos"))]
    const CLOCK: libc::clockid_t = libc::CLOCK_MONOTONIC;
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
    use windows_sys::Win32::System::Performance::QueryPerformanceCounter;
    let mut ticks = 0_i64;
    // SAFETY: `ticks` is a valid value for the call to fill.
    let read = unsafe { QueryPerformanceCounter(&mut ticks) };
    host_nanos((read != 0).then(|| u64::try_from(ticks).ok()).flatten()?)
}

/// A value of [`HOST_DOMAIN`] in nanoseconds: the clock's own on Unix, the performance counter's
/// ticks at its frequency, fixed at boot, on Windows.
fn host_nanos(value: u64) -> Option<u64> {
    #[cfg(windows)]
    {
        use std::sync::OnceLock;
        use windows_sys::Win32::System::Performance::QueryPerformanceFrequency;
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
        let nanos = u128::from(value) * 1_000_000_000 / u128::from((*frequency)?);
        u64::try_from(nanos).ok()
    }
    #[cfg(not(windows))]
    {
        Some(value)
    }
}

/// The host time domain of [`host_time`]'s clock, which the floor requires the device to
/// calibrate with its own (`VK_TIME_DOMAIN_DEVICE_EXT`).
#[cfg(all(unix, not(target_os = "macos")))]
pub const HOST_DOMAIN: vk::TimeDomainKHR = vk::TimeDomainKHR::CLOCK_MONOTONIC;
/// The host time domain of [`host_time`]'s clock, which the floor requires the device to
/// calibrate with its own (`VK_TIME_DOMAIN_DEVICE_EXT`).
#[cfg(target_os = "macos")]
pub const HOST_DOMAIN: vk::TimeDomainKHR = vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW;
/// The host time domain of [`host_time`]'s clock, which the floor requires the device to
/// calibrate with its own (`VK_TIME_DOMAIN_DEVICE_EXT`).
#[cfg(windows)]
pub const HOST_DOMAIN: vk::TimeDomainKHR = vk::TimeDomainKHR::QUERY_PERFORMANCE_COUNTER;

/// How long the completion thread places GPU times by one calibration: once this much has passed
/// since the last, the next completion calibrates again before it places them.
pub(crate) const CALIBRATION_PERIOD: u64 = 100_000_000;

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

    /// Nanoseconds from timestamp `from` to `to`, negative where `to` is the earlier, within the
    /// valid bits: exact across the counter's wrap for two timestamps less than half its range
    /// apart.
    fn nanos(&self, from: u64, to: u64) -> f64 {
        let ahead = to.wrapping_sub(from) & self.mask;
        if ahead <= self.mask / 2 {
            ahead as f64 * self.period
        } else {
            -((from.wrapping_sub(to) & self.mask) as f64 * self.period)
        }
    }
}

/// One reading of the GPU's clock and the host's together (`vkGetCalibratedTimestampsEXT`): the
/// device's tick and the host nanoseconds of the same moment.
#[derive(Clone, Copy)]
pub(crate) struct Calibration {
    tick: u64,
    host: u64,
}

impl Calibration {
    /// Reads the two clocks in one call, or None where the call fails.
    pub(crate) fn read(core: &Core) -> Option<Calibration> {
        let infos = [
            vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE),
            vk::CalibratedTimestampInfoKHR::default().time_domain(HOST_DOMAIN),
        ];
        // SAFETY: the device enables the extension (the floor's), and calibrates both domains.
        let (values, _) = unsafe { core.calibrated.get_calibrated_timestamps(&infos) }.ok()?;
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        core.counters.calibrated();
        Some(Calibration {
            tick: values[0],
            host: host_nanos(values[1])?,
        })
    }

    /// The host time it read.
    pub(crate) fn host(&self) -> u64 {
        self.host
    }

    /// The host times of a command buffer's two timestamps. A float-to-integer cast saturates at
    /// 0.
    pub(crate) fn place(&self, ticks: &Ticks, start: u64, end: u64) -> (u64, u64) {
        let at = |tick: u64| (self.host as f64 + ticks.nanos(self.tick, tick)) as u64;
        (at(start), at(end))
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
