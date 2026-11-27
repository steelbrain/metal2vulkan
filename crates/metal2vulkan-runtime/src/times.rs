//! A command buffer's times: Metal's `kernelStartTime`, `kernelEndTime`, `GPUStartTime` and
//! `GPUEndTime`, all in nanoseconds of one host clock, [`host_time`].
//!
//! Each command buffer takes a pair of timestamp queries from the device's pools ([`StampPools`])
//! as it records its first commands. Its first command buffer writes the first where its commands
//! begin; its last segment writes the second as its last command, after the copies back of the
//! images its commands wrote and its end barrier, so the pair spans all the GPU work done for it.
//! One that finds no pair runs untimed. The runtime's own command buffers, which no Metal command
//! buffer is (the uploads and the Managed buffers' copy-ins), take a pair each too, written first
//! and last ([`begin_own`], [`end_own`]), and their spans are summed as they retire
//! (`Metrics::runtime_command_buffer_gpu_nanos`).
//! The GPU times are the two timestamps placed on the host clock by the device's last calibration
//! (`VK_EXT_calibrated_timestamps`, one reading of the GPU's clock and the host's together): one
//! as the device starts, then one at a completion once [`CALIBRATION_PERIOD`] has passed since
//! the last, the calibration moved earlier where it would place a command buffer's end after the
//! host time the completion thread saw it complete. The kernel times are the host time at commit
//! and when the submit call that passed the last segment returned.

use crate::counters::{Counters, Gauge, Metric};
use crate::device::{lock, Core};
use crate::readback;
use ash::vk;
use std::sync::{Mutex, PoisonError};

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

    /// Whole nanoseconds from timestamp `start` to `end`, 0 where `end` is the earlier.
    pub(crate) fn span(&self, start: u64, end: u64) -> u64 {
        self.nanos(start, end) as u64
    }

    /// Nanoseconds from timestamp `from` to `to`, negative where `to` is the earlier, within the
    /// valid bits: exact across the counter's wrap for two timestamps less than half its range
    /// apart.
    pub(crate) fn nanos(&self, from: u64, to: u64) -> f64 {
        let ahead = to.wrapping_sub(from) & self.mask;
        if ahead <= self.mask / 2 {
            ahead as f64 * self.period
        } else {
            -((from.wrapping_sub(to) & self.mask) as f64 * self.period)
        }
    }
}

/// For the tests: one pair of timestamps the completion thread read as it retired their command
/// buffer, in the GPU's raw ticks ([`Ticks`]).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stamped {
    /// Whether the pair is the runtime's own command buffer's (an upload or a Managed buffer's
    /// copy-in) rather than a Metal command buffer's.
    pub(crate) own: bool,
    pub(crate) start: u64,
    pub(crate) end: u64,
}

/// One reading of the GPU's clock and the host's together (`vkGetCalibratedTimestampsEXT`): the
/// device's tick and the host nanoseconds of the same moment.
#[derive(Clone, Copy)]
pub(crate) struct Calibration {
    pub(crate) tick: u64,
    pub(crate) host: u64,
}

impl Calibration {
    /// Reads the two clocks in one call, or None where the call fails.
    pub(crate) fn read(core: &Core) -> Option<Calibration> {
        let infos = [
            vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE),
            vk::CalibratedTimestampInfoKHR::default().time_domain(HOST_DOMAIN),
        ];
        // The call's own time, counted with it (`Metrics::calibrations`): two clock reads.
        let began = host_time();
        // SAFETY: the device enables the extension (the floor's), and calibrates both domains.
        let read = unsafe { core.calibrated.get_calibrated_timestamps(&infos) };
        core.counters.calibrated(host_time().saturating_sub(began));
        let (values, _) = read.ok()?;
        let host = host_nanos(values[1])?;
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let host = host
            + core
                .calibration_lateness
                .load(std::sync::atomic::Ordering::Relaxed);
        Some(Calibration {
            tick: values[0],
            host,
        })
    }

    /// The host time it read.
    pub(crate) fn host(&self) -> u64 {
        self.host
    }

    /// The host times of a command buffer's two timestamps, apart by the ticks' own span
    /// ([`Ticks::span`]). Where its end would fall after `seen`, the host time the completion
    /// thread saw it complete, the calibration's host time moves earlier by that excess first,
    /// for this placement and every later one, so the two move together and later command
    /// buffers keep their order against this one. A calibration's host time is late where the
    /// driver reads it after the GPU's tick (Mesa's drivers read the domains in turn, in the
    /// order asked, the GPU's first), and each completion bounds it from above, so the move only
    /// takes it toward the moment of the tick. A float-to-integer cast saturates at 0.
    pub(crate) fn place(
        &mut self,
        ticks: &Ticks,
        (start, end): (u64, u64),
        seen: u64,
    ) -> (u64, u64) {
        let span = ticks.span(start, end);
        let start = self.at(ticks, start);
        let excess = (start + span).saturating_sub(seen);
        self.host = self.host.saturating_sub(excess);
        let start = start.saturating_sub(excess);
        (start, start + span)
    }

    /// The host time of timestamp `tick` by this calibration as it stands, which moves nothing:
    /// for the spans of the runtime's own command buffers ([`Spans`]).
    pub(crate) fn at(&self, ticks: &Ticks, tick: u64) -> u64 {
        (self.host as f64 + ticks.nanos(self.tick, tick)) as u64
    }
}

/// How many of the latest GPU spans the completion thread keeps ([`Spans`]): more than the
/// command buffers, uploads and copy-ins of a queue's three submissions in flight at sixteen
/// command buffers each.
pub(crate) const SPANS_KEPT: usize = 128;

/// The completion thread's latest GPU spans on the host clock, of the command buffers it retired
/// with both timestamps, the program's and the runtime's own: [`SPANS_KEPT`] of them, the oldest
/// replaced first, in place, with nothing allocated.
pub(crate) struct Spans {
    spans: [(u64, u64); SPANS_KEPT],
    next: usize,
    kept: usize,
}

impl Spans {
    pub(crate) fn new() -> Spans {
        Spans {
            spans: [(0, 0); SPANS_KEPT],
            next: 0,
            kept: 0,
        }
    }

    /// One more span, from `start` to `end`; one of no length is not kept.
    pub(crate) fn push(&mut self, (start, end): (u64, u64)) {
        if end <= start {
            return;
        }
        self.spans[self.next] = (start, end);
        self.next = (self.next + 1) % SPANS_KEPT;
        self.kept = (self.kept + 1).min(SPANS_KEPT);
    }

    /// The nanoseconds from `from` to `to` that lie under any span kept, the spans merged where
    /// they overlap (`Metrics::scheduled_to_gpu_start_behind_nanos`): one pass over the spans
    /// kept, then a sort of those that meet the range, in place.
    pub(crate) fn covered(&self, from: u64, to: u64) -> u64 {
        let mut met = [(0, 0); SPANS_KEPT];
        let mut count = 0;
        for &(start, end) in &self.spans[..self.kept] {
            let (start, end) = (start.max(from), end.min(to));
            if start < end {
                met[count] = (start, end);
                count += 1;
            }
        }
        let met = &mut met[..count];
        met.sort_unstable();
        let (mut covered, mut reached) = (0, from);
        for &(start, end) in met.iter() {
            let start = start.max(reached);
            if start < end {
                covered += end - start;
                reached = end;
            }
        }
        covered
    }
}

/// The queries of one of the device's timestamp pools: the pairs of 2,048 command buffers. Neither
/// a pool's most queries nor the most pools a device makes is a Vulkan query; the smallest met is
/// this many a pool (Mesa's Apple-GPU driver, whose pools are Metal counter heaps of at most 4,096
/// entries, at most 32 of them live in a process).
pub(crate) const POOL_QUERIES: u32 = 4096;

/// A command buffer's two timestamp queries: its start at `first`, its end at the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pair {
    pub(crate) pool: vk::QueryPool,
    pub(crate) first: u32,
}

impl Pair {
    /// Its two timestamps, read once its command buffer's submission signalled: none where they
    /// read as not available ([`readback::timestamps`]), each such read counted
    /// (`Metrics::stamps_not_available`).
    ///
    /// # Safety
    ///
    /// The pair's pool is `device`'s, and no other thread resets its two queries while it reads.
    pub(crate) unsafe fn times(
        self,
        device: &ash::Device,
        counters: &Counters,
    ) -> Option<(u64, u64)> {
        // SAFETY: by the caller.
        let read = unsafe { readback::timestamps(device, self.pool, self.first) };
        if read.is_none() {
            counters.count(Metric::StampsNotAvailable);
        }
        read
    }
}

/// The device's timestamp pools, which every queue's command buffers take their pairs from: made
/// on demand, [`POOL_QUERIES`] queries each, kept until the device's teardown, so their count is
/// the most command buffers ever timed at once over 2,048. A pair given back is taken again before
/// any of a pool not yet taken, and a new pool is made only once every pair is taken. Once the
/// device refuses a pool, none is asked for again: a command buffer that then finds no pair free
/// runs untimed (`Metrics::untimed_command_buffers`).
pub(crate) struct StampPools {
    state: Mutex<PoolsState>,
}

#[derive(Default)]
struct PoolsState {
    pools: Vec<vk::QueryPool>,
    /// The pairs given back, every query of each reset.
    free: Vec<Pair>,
    /// The pairs of the newest pool below it have been taken.
    next: u32,
    /// The device refused a pool.
    refused: bool,
}

impl StampPools {
    pub(crate) fn new() -> StampPools {
        StampPools {
            state: Mutex::new(PoolsState::default()),
        }
    }

    /// A free pair, both its queries reset: one given back, else the next of the newest pool, else
    /// the first of a new pool; `None` where every pair is taken and the device refuses a pool, or
    /// refused one before. A new pool is counted ([`Gauge::TimestampPools`]).
    pub(crate) fn take(&self, device: &ash::Device, counters: &Counters) -> Option<Pair> {
        let mut state = lock(&self.state);
        if let Some(pair) = state.free.pop() {
            return Some(pair);
        }
        if let Some(&pool) = state.pools.last() {
            if state.next < POOL_QUERIES / 2 {
                let first = state.next * 2;
                state.next += 1;
                return Some(Pair { pool, first });
            }
        }
        if state.refused {
            return None;
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(POOL_QUERIES);
        // SAFETY: the info is complete.
        let Ok(pool) = (unsafe { device.create_query_pool(&info, None) }) else {
            state.refused = true;
            return None;
        };
        reset(device, pool, 0, POOL_QUERIES);
        counters.rise(Gauge::TimestampPools);
        state.pools.push(pool);
        state.next = 1;
        Some(Pair { pool, first: 0 })
    }

    /// Gives back `pair`, both its queries reset.
    pub(crate) fn give_back(&self, pair: Pair) {
        lock(&self.state).free.push(pair);
    }

    /// At the device's teardown, once it is idle: destroys every pool.
    pub(crate) fn destroy(&mut self, device: &ash::Device, counters: &Counters) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        for pool in state.pools.drain(..) {
            // SAFETY: the device is idle, and no call reads the pool.
            unsafe { device.destroy_query_pool(pool, None) };
            counters.fall(Gauge::TimestampPools);
        }
        state.free.clear();
    }
}

/// Resets `count` queries from `first` on the host (`hostQueryReset`, in the floor): every query
/// as the pool is made, and as a command buffer retires, those of its pair that a submission
/// wrote, so a free pair's queries are always reset. A reset happens before any later
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
/// two queries are reset: a free pair's always are ([`reset`]).
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

/// Records the end timestamp: written once every earlier command of the queue has completed
/// (`vkCmdWriteTimestamp2`'s first synchronization scope is every command earlier in submission
/// order, at its stage, and in a first scope the bottom-of-pipe stage is every stage).
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

/// Takes the pair of one of the runtime's own command buffers, which no Metal command buffer is
/// (an upload or a Managed buffer's copy-in), and writes its start into `command`, which has just
/// begun recording: None, counted (`Metrics::untimed_command_buffers`), where every pair is taken
/// and the device makes no new pool.
pub(crate) fn begin_own(core: &Core, command: vk::CommandBuffer) -> Option<Pair> {
    let pair = core.stamps.take(&core.device, &core.counters);
    match pair {
        Some(pair) => write_start(&core.device, command, pair.pool, pair.first),
        None => core.counters.count(Metric::UntimedCommandBuffers),
    }
    pair
}

/// Writes the end of `pair` ([`begin_own`]) into `command`: its last command, after its last
/// barrier, so the pair spans every command of it.
pub(crate) fn end_own(device: &ash::Device, command: vk::CommandBuffer, pair: Option<Pair>) {
    if let Some(pair) = pair {
        write_end(device, command, pair.pool, pair.first);
    }
}

/// How one of the runtime's own command buffers retires.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum OwnEnd {
    /// It never went to `vkQueueSubmit2`: its queries are still reset.
    Unsent,
    /// Its submission completed.
    Completed,
    /// Its submission was in flight at a loss, the device now idle: nothing of it is read.
    Lost,
}

/// Gives back `pair` ([`begin_own`]) as its command buffer retires (`end`). Where it completed,
/// its span is added to `Metrics::runtime_command_buffer_gpu_nanos` where both timestamps read,
/// and one that reads as not available is counted ([`Pair::times`]); where it was submitted, both
/// queries are reset first ([`reset`]). Answers the two timestamps where it completed and both
/// read, for the completion thread's spans ([`Spans`]).
pub(crate) fn retire_own(core: &Core, pair: Option<Pair>, end: OwnEnd) -> Option<(u64, u64)> {
    let pair = pair?;
    let mut read = None;
    if end == OwnEnd::Completed {
        // SAFETY: the pair's pool is the device's, and only this retirement resets its two
        // queries, after this read.
        read = unsafe { pair.times(&core.device, &core.counters) };
        if let Some((start, stop)) = read {
            (core.counters).runtime_command_buffer_ran(core.ticks.span(start, stop));
        }
    }
    if end != OwnEnd::Unsent {
        reset(&core.device, pair.pool, pair.first, 2);
    }
    core.stamps.give_back(pair);
    read
}
