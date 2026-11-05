//! A command buffer's GPU and kernel times: Metal's `GPUStartTime`, `GPUEndTime`, `kernelStartTime`
//! and `kernelEndTime`, in seconds of the host's clock ([`host_time`]), as Metal gives them in the
//! host's time base.
//!
//! - **The host's clock.** `CLOCK_UPTIME_RAW` on macOS, the clock of `mach_absolute_time` and so
//!   of Metal's times and `CACurrentMediaTime`; `CLOCK_MONOTONIC_RAW` elsewhere on Unix, its
//!   counterpart (monotonic, never slewed, stopped while the system sleeps). Other platforms count
//!   the runtime's own monotonic clock, which no device can calibrate against.
//! - **GPU times.** The command buffer's first batch writes a timestamp at the top of its work
//!   (after a barrier its first encoder waits behind for earlier work, once every command before
//!   it completed, where Metal's start follows such work's end), and its last batch one after the
//!   barrier that ends it (`command_buffer/submit.rs`), into a slot of one of the device's
//!   timestamp pools ([`Slot`]), which it takes as its first encoder opens where its slot holds
//!   the start that encoder writes (below), else at its commit.
//! - **The start in the first encoder.** A slot holds three queries ([`PREFIXED`]) where the
//!   pools are not Metal counter sample buffers and the driver's submissions do not stall on
//!   semaphore operations: the start a command buffer of barriers writes, the end, and the start
//!   the first encoder's own command buffer writes as it opens, its first commands (its prefix:
//!   the reset of the end and that start, then the start at the top of the pipe). Where the batch
//!   plans no barrier before that encoder and its command buffer reaches the queue as recorded,
//!   the batch begins with no command buffer of its own, and the prefix's start is the command
//!   buffer's. It is the same place: either start is written at the top of the pipe, after every
//!   command submitted before the batch's work and before any of it; where batches are ordered by
//!   their ends (`batch_order.rs`), at the bottom, once every command submitted before it
//!   completed, when such a driver starts the batch. Elsewhere (a barrier planned
//!   there, a caller's image moved first, the encoder run again or not at all) a command buffer of
//!   barriers writes the first start before the barrier, as without a prefix, and that start is
//!   read; the prefix's is written after it and never read. A Metal counter sample buffer's slot
//!   holds the two alone ([`PLAIN`]), as does a slot where every batch with work starts with a
//!   command buffer of barriers anyway (`DriverFact::SemaphoreOperationsStallSubmits`,
//!   `command_buffer/submit.rs`), and its starts are the barriers'. The completion turn reads
//!   the start the batch took and the end once the command buffer completed, and gives the slot
//!   back. Their difference, in ticks of `timestampPeriod` nanoseconds, is exact on every device
//!   that writes timestamps.
//! - **Reset on the host as a command buffer takes its slot**, on a device the runtime created,
//!   which enables `hostQueryReset` (Vulkan 1.2 requires it: "If Vulkan 1.2 is supported, the
//!   following features must be supported: ... hostQueryReset", Feature Requirements):
//!   `vkResetQueryPool` "sets the status of query indices [firstQuery, firstQuery + queryCount -
//!   1] to unavailable", before any of the command buffer's batches is submitted, and a queue
//!   submission's first synchronization scope "includes execution of vkQueueSubmit on the host and
//!   anything that happened-before it" (Host Write Ordering Guarantees). A slot is free only once
//!   its last holding's work completed, or was never submitted (its command buffer gives it back
//!   as it completes, or as it goes uncommitted), and only its holder resets it: "Submitted
//!   commands that refer to the range ... must have completed execution", and no other thread
//!   reads or resets it meanwhile (`vkResetQueryPool`'s valid usage). So no command buffer the
//!   GPU runs resets a query: neither the starts' command buffers of barriers, which the scheduler
//!   keeps and submits again for every holding of a slot (`command_buffer/kept_barriers.rs`), nor
//!   a first encoder's prefix. An adopted device (`Device::from_raw`), which states nothing of
//!   `hostQueryReset`, resets them in those command buffers: the start's command buffer resets its
//!   slot's start and end, and the first encoder's prefix the end and its own start, each before
//!   its timestamp. Counted: `WorkCounts::timestamp_slots_reset_on_host`.
//! - **Pools made as needed**. Each pool holds [`SLOTS`] slots. The device makes its first at its
//!   first timed command buffer, and another whenever one takes a slot while every slot of the
//!   ones it has is held, until the device refuses one: its own limit, which Vulkan states nowhere
//!   else (a pool is 6144 queries where a slot holds three, else 4096; the runtime bounds none,
//!   as Metal times every command buffer in flight). It releases them at its teardown. A slot is
//!   taken from the pool that gave the last one first, and the others only when that one has none
//!   free: the pools are searched as one fills, not for every slot. On a driver whose pools are Metal counter sample
//!   buffers (`DriverFact::TimestampPoolsAreCounterSampleBuffers`: MoltenVK) it makes one, of
//!   [`PLAIN`] queries a slot, never more: Metal keeps at most 32 of those live per GPU, each of
//!   at most 4096 timestamps (measured through MoltenVK 1.4.1 on an M3 Max and an M4: the 33rd
//!   failed, as did a pool of 8192), so a pool per command buffer ran out with 32 command buffers
//!   in flight, and each pool more is one fewer for another device on the GPU. One pool of the
//!   most timestamps holds a slot for 2048 command buffers in flight. A command buffer committed
//!   while every slot is held and no pool more can be made has no timestamps: its GPU times are
//!   refused as on a device that writes none, and the device announces it once
//!   (`Degradation::UntimedCommandBuffer`).
//!   Waiting for a slot instead could wait for work that needs the program to go on.
//! - **Calibrated where the device can.** With `VK_KHR_calibrated_timestamps` (or its `VK_EXT_`
//!   form) and a host time domain the runtime can read (`CLOCK_MONOTONIC_RAW` or
//!   `CLOCK_MONOTONIC`), a pair of GPU and host readings taken as the command buffer completes (on
//!   the completion turn's thread, before its times are placed) places each timestamp on the host's
//!   clock, within the pair's stated deviation: Metal's answer. Never at the commit: a reading of
//!   the GPU's clock waits for a GPU powered down to wake, which on an RTX 5080 Laptop GPU in
//!   NVIDIA's runtime D3 (driver 615.71.09) held the committing thread 973 to 977 ms after 20 s
//!   idle, where Metal's commit returns at once; the GPU that just completed the work is awake. A
//!   domain that is not the host's clock is moved onto it by reading both clocks at once
//!   (MoltenVK's `CLOCK_MONOTONIC_RAW` is macOS's, which counts sleep; Metal's does not). A pair is
//!   read again while each reading halves the deviation the one before it stated, and while the
//!   smallest so far states more than [`SETTLED`], [`READINGS`] at most, and the one of the
//!   smallest places the times ([`least_deviation`]). The specification leaves that to the
//!   application, which "can choose to re-issue the timestamp calibration call pursuing a lower
//!   deviation value" (`vkGetCalibratedTimestampsKHR`). A reading's two values may stand as far
//!   apart as it states. On an Iris Xe (ANV, Mesa 26.2.4, i915), the first reading after the GPU
//!   idled 100 ms stated 165 us at the median, and the device's timestamp in it was read that much
//!   after its host time: placed by it, every GPU time came 156 us early at the median (105 to 193
//!   us over 200 readings). The reading after it stated 7.6 us and agreed with the next ones within
//!   0.4 us. A thread preempted between a reading's two values states the time it lost: on an RX
//!   7900 XTX (RADV) under a full parallel suite, two readings in turn stated about 70 us each.
//! - **Checked against the command buffer's life.** A time the pair places is the GPU's, and the
//!   GPU ran the work between the first batch's submission (the kernel start) and the moment the
//!   completion turn saw it complete. A pair that places the start before that start, or the
//!   end after that completion, by more than its stated deviation and the clocks' drift since it
//!   ([`DRIFT_PPM`]) allow, disagrees with the device's own timestamps: the command buffer is
//!   placed as below, as on an uncalibrated device, counted
//!   (`WorkCounts::calibration_disagreements`) and announced once
//!   (`Degradation::UncalibratedGpuTimes`). A Radeon 780M (RADV PHOENIX, Mesa 26.0.8, Linux 7.0)
//!   after two MODE2 GPU resets wrote timestamp queries 65.15 s ahead of its own calibrated
//!   `DEVICE` domain, steadily, where an RX 7900 XTX beside it agreed within its deviation:
//!   placed by the pair, every GPU time came 65 s after its command buffer completed.
//! - **Uncalibrated otherwise.** Without them, the end is placed at the host time the command
//!   buffer completed and the start the timestamps' difference before it: the duration exact,
//!   both times late by the completion's latency. The device announces it once
//!   (`Degradation::UncalibratedGpuTimes`).
//! - **None without timestamps.** A queue family whose `timestampValidBits` is 0 writes none: the
//!   times are refused (`Unsupported`, `Feature::GpuTimes`). So are a command buffer's that found
//!   no slot free.
//! - **Kernel times** are the host times at which the scheduler began submitting the command
//!   buffer's first batch and finished submitting its last. They are kept apart from the command
//!   buffer's state ([`KernelTimes`]), which the scheduler holds while it records the batch's
//!   own command buffers and submits it: a read waits for no submission in
//!   progress and for no GPU work, as Metal's reads wait for none.
//! - **No encoder, no timestamps**. A command buffer that encoded nothing and waits for no event or
//!   semaphore takes no slot and writes no timestamps. Its GPU start and end are both where its
//!   queue reached it: its kernel end time, or the GPU end of the queue's command buffer before it
//!   where that is later. Metal answers so for one, a zero duration (measured on an M4 and an M3
//!   Max): on its own, a microsecond after its `kernelEndTime`; committed behind a 2.5-ms fill, 112
//!   us after the fill's `GPUEndTime`, 2.7 ms after its own `kernelEndTime`; waiting for an event
//!   the host signals, 7 us after the signal, its `kernelEndTime` 6 us after it. The completion
//!   turn places each command buffer as it completes, in its queue's order: each completed
//!   command buffer moves its queue's latest GPU start and end. On MoltenVK the two timestamps cost
//!   such a command buffer about 150 us of latency (each is sampled in a blit pass of its own),
//!   against about 15 us without. One that waits keeps its timestamps: when the GPU passes a wait
//!   for another queue's signal is not a host time the runtime knows.
//! - **From the host, where the device is told to** (`DeviceOptions::gpu_times_from_host`,
//!   `RawDevice::gpu_times_from_host`; off by default), **or where its driver's timestamps wait
//!   for every queue's** (`DriverFact::TimestampsWaitForEveryQueue`: MoltenVK 1.4.1 on Apple's
//!   GPUs, whatever the options say). No command buffer writes timestamps. One with work answers as
//!   its GPU start its kernel end time, or its queue's latest GPU end before it where later, and as
//!   its GPU end the host time the runtime saw it complete (never before the start): a window its
//!   GPU work ran in. The device announces it once, at the first such answer
//!   (`Degradation::GpuTimesFromHost`). A device whose queue family writes no timestamps answers so
//!   too, instead of refusing.
//!
//!   Every answer is one Metal could give. Four rules hold by construction:
//!   1. **Metal's clock.** The times are seconds of the host's clock ([`host_time`]): on macOS
//!      `CLOCK_UPTIME_RAW`, the clock of `mach_absolute_time`, which Metal's `GPUStartTime` and
//!      `GPUEndTime` count on Apple's GPUs.
//!   2. **Inside the command buffer's life.** Its commit <= `kernelStartTime` <= `kernelEndTime`
//!      <= `GPUStartTime` <= `GPUEndTime` <= the moment its completion is observable: the end is
//!      read in the completion turn before the command buffer's status becomes `Completed`
//!      and before its completed handlers run, and the start is never later than the end.
//!   3. **A serial queue.** In a queue's commit order, which is the order its command buffers
//!      complete and are placed in (`command_buffer/completion.rs`), each one's GPU start is at
//!      or after the GPU end of the one before it, so starts and ends only move forward and no
//!      two command buffers of a queue overlap: one after another, as Metal's serial queue can
//!      run them. An encoder-less command buffer, which writes no timestamps on any device, is
//!      placed so too (a zero duration at its queue's latest end or its kernel end).
//!   4. **No counter samples.** The runtime samples no counters (`MTLCounterSampleBuffer`, the
//!      encoders' `sampleCountersInBuffer`, `MTLDevice::sampleTimestamps`:
//!      `RuntimeGap::CounterSampleBuffers`), so no sample can fall outside its command buffer's
//!      window or disagree with the host's clock.
//!
//!   How far they sit from Metal's: the start can be early by the time from the end of the
//!   submission to the GPU's start of the work, or late by the time the runtime took to see the
//!   queue's command buffer before it complete; the end is late by the time the runtime took to
//!   see this one complete; so the duration holds both latencies, not the GPU's own time.
//!
//! The pair a command buffer's times are placed by is the device's latest, taken at a completion
//! at most [`REFRESH`] before its own: one calibration per device per 100 ms, not one per command
//! buffer. The GPU's clock and the host's drift apart by parts per million, so a pair that old
//! moves a time by nanoseconds to microseconds, within what the calibration states for itself on
//! the devices measured.

use super::device::lock;
use super::exit_guard::{self, Call};
use super::release;
use super::work_counts::Counters;
use ash::vk;
use std::collections::BTreeSet;
use std::ffi::{c_char, CStr, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest a device's calibration pair serves the completions after it.
pub(crate) const REFRESH: Duration = Duration::from_millis(100);

/// The most readings one calibration pair takes ([`least_deviation`]).
pub(crate) const READINGS: u32 = 4;

/// The deviation, in nanoseconds, above which a pair reads again even where its readings stopped
/// falling ([`least_deviation`]): 20 us, the error validation holds calibrated GPU times to
/// (`runtime_lane::gpu_times`), above the 4 to 7.6 us of a settled reading on the devices
/// measured.
pub(crate) const SETTLED: u64 = 20_000;

/// How far apart, in parts per million of the time since a pair, the GPU's clock and the host's
/// may drift before a placed time counts as disagreeing with the command buffer's life (the module
/// docs): above a crystal's tolerance, and NTP's own slew limit, so that a sound pair never trips
/// the check, and far below a disagreement that places a time seconds away.
pub(crate) const DRIFT_PPM: u64 = 500;

/// The slots of each of a device's timestamp pools: of [`PLAIN`] queries each where its pools are
/// Metal counter sample buffers, 4096 timestamps, the most one holds (the module documentation).
pub(crate) const SLOTS: u32 = 2048;

/// The queries of a slot where the pools are not Metal counter sample buffers: the start a
/// command buffer of barriers writes, the end, and the start the first encoder writes (the module
/// documentation).
pub(crate) const PREFIXED: u32 = 3;

/// The queries of a slot where the pools are Metal counter sample buffers: the start and the end.
pub(crate) const PLAIN: u32 = 2;

/// The most timestamps MoltenVK made one Metal counter sample buffer of, on an M3 Max and an M4.
const SAMPLE_BUFFER_TIMESTAMPS: u32 = 4096;
const _: () = assert!(PLAIN * SLOTS <= SAMPLE_BUFFER_TIMESTAMPS);

/// The host time now, in seconds of the clock the runtime's GPU and kernel times count (the
/// module documentation): `CLOCK_UPTIME_RAW` on macOS, which `mach_absolute_time` and
/// `CACurrentMediaTime` count, and `CLOCK_MONOTONIC_RAW` elsewhere on Unix.
pub fn host_time() -> f64 {
    host_nanos() as f64 / 1e9
}

/// The host time now, in nanoseconds of the host's clock.
pub(crate) fn host_nanos() -> u64 {
    #[cfg(unix)]
    {
        read_clock(BASE).unwrap_or(0)
    }
    #[cfg(not(unix))]
    {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// The host's clock.
#[cfg(target_os = "macos")]
const BASE: libc::clockid_t = libc::CLOCK_UPTIME_RAW;
#[cfg(all(unix, not(target_os = "macos")))]
const BASE: libc::clockid_t = libc::CLOCK_MONOTONIC_RAW;

/// `clock` now, in nanoseconds.
#[cfg(unix)]
fn read_clock(clock: libc::clockid_t) -> Option<u64> {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to fill.
    if unsafe { libc::clock_gettime(clock, &mut now) } != 0 {
        return None;
    }
    let seconds = u64::try_from(now.tv_sec).ok()?;
    let nanos = u64::try_from(now.tv_nsec).ok()?;
    Some(seconds * 1_000_000_000 + nanos)
}

/// The host clock behind a Vulkan host time domain, where the runtime can read it.
#[cfg(unix)]
fn domain_clock(domain: vk::TimeDomainKHR) -> Option<libc::clockid_t> {
    match domain {
        vk::TimeDomainKHR::CLOCK_MONOTONIC => Some(libc::CLOCK_MONOTONIC),
        vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW => Some(libc::CLOCK_MONOTONIC_RAW),
        _ => None,
    }
}

/// The host time domain a device's calibration uses, from those it offers: the host's clock
/// where offered, else `CLOCK_MONOTONIC_RAW`, else `CLOCK_MONOTONIC`. `None` without the
/// device's own domain or a host domain the runtime can read.
pub(crate) fn host_domain(offered: &[vk::TimeDomainKHR]) -> Option<vk::TimeDomainKHR> {
    if !offered.contains(&vk::TimeDomainKHR::DEVICE) {
        return None;
    }
    #[cfg(unix)]
    {
        let preferred = [
            vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW,
            vk::TimeDomainKHR::CLOCK_MONOTONIC,
        ];
        let base = preferred
            .into_iter()
            .find(|&domain| domain_clock(domain) == Some(BASE));
        base.into_iter()
            .chain(preferred)
            .find(|domain| offered.contains(domain))
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// How a device times its command buffers on the GPU.
#[derive(Clone, Copy)]
pub(crate) struct GpuClock {
    /// Nanoseconds per timestamp tick (`timestampPeriod`).
    pub(crate) period: f64,
    /// The bits a timestamp holds (`timestampValidBits` of the runtime's queue family): 1 to 64.
    pub(crate) valid_bits: u32,
    /// The calibration, where the device has one.
    pub(crate) calibration: Option<Calibration>,
}

/// A device's calibrated timestamps: its entry point and the host domain it reads.
#[derive(Clone, Copy)]
pub(crate) struct Calibration {
    pub(crate) get: vk::PFN_vkGetCalibratedTimestampsKHR,
    pub(crate) device: vk::Device,
    pub(crate) domain: vk::TimeDomainKHR,
}

/// A GPU timestamp and the host time, on the host's clock, at one moment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pair {
    pub(crate) ticks: u64,
    pub(crate) host: u64,
}

/// A calibration reading: its pair, and how far apart, in nanoseconds, its two values may stand:
/// the deviation the call stated, and for a host domain that is not the host's clock, the span
/// of the host's readings around the domain's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reading {
    pub(crate) pair: Pair,
    pub(crate) deviation: u64,
}

/// A command buffer's life on the host's clock, in nanoseconds: from the moment the scheduler
/// began submitting its first batch to the moment a completion turn saw its last complete.
/// Its GPU work ran inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Life {
    pub(crate) kernel_start: u64,
    pub(crate) completed: u64,
}

/// How a command buffer's timestamps were placed on the host's clock ([`GpuClock::span`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Placement {
    /// By the device's calibration pair.
    Calibrated,
    /// By its completion: the device had no pair as it completed.
    Uncalibrated,
    /// By its completion: the pair placed a time outside its life by `by` nanoseconds, more than
    /// the pair's stated `deviation` and the clocks' drift allow (the module docs).
    Disagreed { by: u64, deviation: u64 },
}

/// Of the readings `read` gives, the one of the smallest deviation, and how many `read` was
/// called for: one more while the latest halved the deviation the one before it stated, or while
/// the smallest so far states more than [`SETTLED`], [`READINGS`] at most (the module docs).
/// `None` where the first fails; a later failure ends them.
pub(crate) fn least_deviation(mut read: impl FnMut() -> Option<Reading>) -> Option<(Reading, u32)> {
    let mut least = read()?;
    let mut previous = least.deviation;
    let mut calls = 1;
    while calls < READINGS {
        calls += 1;
        let Some(next) = read() else {
            break;
        };
        if next.deviation < least.deviation {
            least = next;
        }
        if next.deviation.saturating_mul(2) > previous && least.deviation <= SETTLED {
            break;
        }
        previous = next.deviation;
    }
    Some((least, calls))
}

impl GpuClock {
    /// The clock of a queue family with `valid_bits` timestamp bits and `period` nanoseconds
    /// per tick; `None` where the family writes no timestamps.
    pub(crate) fn new(
        valid_bits: u32,
        period: f32,
        calibration: Option<Calibration>,
    ) -> Option<GpuClock> {
        (valid_bits > 0).then_some(GpuClock {
            period: f64::from(period),
            valid_bits: valid_bits.min(64),
            calibration,
        })
    }

    /// A calibrated pair now, where the device has calibration and the readings succeed: the
    /// reading of the smallest deviation ([`least_deviation`]), and how many readings it took.
    pub(crate) fn pair(&self) -> Option<(Reading, u32)> {
        let calibration = self.calibration?;
        least_deviation(|| calibration.reading())
    }

    /// The host time, in nanoseconds of the host's clock, of the timestamp `ticks`, by `pair`.
    pub(crate) fn place(&self, pair: Pair, ticks: u64) -> u64 {
        let delta = self.signed_ticks(ticks, pair.ticks);
        let nanos = (delta as f64 * self.period).round() as i128;
        u64::try_from(i128::from(pair.host) + nanos).unwrap_or(0)
    }

    /// The host times, in nanoseconds of the host's clock, of a command buffer's timestamps
    /// `start` and `end`, and how they were placed: by `reading`'s pair where they lie inside the
    /// command buffer's `life` within the reading's deviation and the clocks' drift since it
    /// ([`DRIFT_PPM`]); else, and without a reading, ending where `life` ends, the host time its
    /// completion was seen, and starting their duration before (the module docs).
    pub(crate) fn span(
        &self,
        reading: Option<Reading>,
        start: u64,
        end: u64,
        life: Life,
    ) -> ((u64, u64), Placement) {
        let by_completion = (
            life.completed.saturating_sub(self.duration(start, end)),
            life.completed,
        );
        let Some(Reading { pair, deviation }) = reading else {
            return (by_completion, Placement::Uncalibrated);
        };
        let placed = (self.place(pair, start), self.place(pair, end));
        // How far each may stand from the truth: the deviation the pair stated, and the drift
        // since the pair to the bound it is checked against.
        let allowed = |bound: u64| {
            let drift = bound.abs_diff(pair.host).saturating_mul(DRIFT_PPM) / 1_000_000;
            deviation.saturating_add(drift)
        };
        let early = life.kernel_start.saturating_sub(placed.0);
        let late = placed.1.saturating_sub(life.completed);
        if early > allowed(life.kernel_start) || late > allowed(life.completed) {
            let by = early.max(late);
            return (by_completion, Placement::Disagreed { by, deviation });
        }
        (placed, Placement::Calibrated)
    }

    /// The nanoseconds from timestamp `start` to `end`.
    pub(crate) fn duration(&self, start: u64, end: u64) -> u64 {
        let delta = self.signed_ticks(end, start).max(0);
        (delta as f64 * self.period).round() as u64
    }

    /// `to - from` in ticks, modulo the timestamps' bits, as the nearer of the two directions.
    fn signed_ticks(&self, to: u64, from: u64) -> i64 {
        if self.valid_bits >= 64 {
            return to.wrapping_sub(from) as i64;
        }
        let modulus = 1u64 << self.valid_bits;
        let difference = to.wrapping_sub(from) & (modulus - 1);
        if difference >= modulus / 2 {
            difference as i64 - modulus as i64
        } else {
            difference as i64
        }
    }
}

/// The extension's names, each with its commands' names: `VK_KHR_calibrated_timestamps`, then
/// the `VK_EXT_` form it promotes, whose commands vk.xml aliases to the same signatures.
#[derive(Debug)]
struct Names {
    extension: &'static CStr,
    domains: &'static CStr,
    get: &'static CStr,
}

/// The two forms, preferred first.
const FORMS: [Names; 2] = [
    Names {
        extension: vk::KHR_CALIBRATED_TIMESTAMPS_NAME,
        domains: c"vkGetPhysicalDeviceCalibrateableTimeDomainsKHR",
        get: c"vkGetCalibratedTimestampsKHR",
    },
    Names {
        extension: vk::EXT_CALIBRATED_TIMESTAMPS_NAME,
        domains: c"vkGetPhysicalDeviceCalibrateableTimeDomainsEXT",
        get: c"vkGetCalibratedTimestampsEXT",
    },
];

/// One form of the calibrated timestamps extension, which a device enables.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Extension(&'static Names);

impl Extension {
    /// The preferred form among `names`: the extensions a device lists, or enabled.
    pub(crate) fn among(names: &BTreeSet<CString>) -> Option<Extension> {
        FORMS
            .iter()
            .find(|form| names.contains(form.extension))
            .map(Extension)
    }

    /// Its name.
    pub(crate) fn name(self) -> &'static CStr {
        self.0.extension
    }
}

/// Why a device answers GPU times from the host's clock (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FromHost {
    /// Its options or the consumer's adopted device say so.
    Asked,
    /// Its driver's timestamps wait for every queue's
    /// (`DriverFact::TimestampsWaitForEveryQueue`).
    Driver,
}

impl FromHost {
    /// Why, as the degradation's notice says it.
    pub(crate) fn because(self) -> &'static str {
        match self {
            FromHost::Asked => {
                "the device was made to answer them from the host's clock (gpu_times_from_host)"
            }
            FromHost::Driver => {
                "the driver samples each timestamp in a pass that waits for every queue's \
                 timestamped work (DriverFact::TimestampsWaitForEveryQueue)"
            }
        }
    }
}

/// How a device times its command buffers: its clock, `None` where its queue family writes no
/// timestamps, whether (and why) it answers GPU times from the host instead, its latest
/// calibration pair, and its timestamp pools, as made.
pub(crate) struct Timing {
    clock: Option<GpuClock>,
    from_host: Option<FromHost>,
    latest: Mutex<Option<(Instant, Reading)>>,
    /// Its slots hold the start the first encoder writes ([`PREFIXED`]).
    prefixed: bool,
    /// Its device enables `hostQueryReset`: a slot is reset on the host as it is taken (the module
    /// docs).
    host_reset: bool,
    pools: Mutex<Pools>,
}

/// A device's timestamp pools, made as its command buffers in flight need them, and the most it
/// may make, where it has a most (the module docs).
pub(crate) struct Pools {
    made: Vec<Pool>,
    /// The pool the last slot was taken from, asked first.
    at: usize,
    /// One on a driver whose pools are Metal counter sample buffers; else none until the device
    /// refuses a pool, then the pools it made.
    most: Option<usize>,
    /// The queries of each slot: [`PREFIXED`] or [`PLAIN`].
    per_slot: u32,
}

impl Pools {
    /// No pool yet, and at most `most` to be made where given, each slot of `per_slot` queries.
    pub(crate) fn new(most: Option<usize>, per_slot: u32) -> Pools {
        Pools {
            made: Vec::new(),
            at: 0,
            most,
            per_slot,
        }
    }

    /// A free slot of a pool made, the one that gave the last slot asked first, else of a pool
    /// `make` makes now of the queries it is given where the device has made fewer than its most;
    /// `None` when every slot is held and no pool more can be made. A pool `make` cannot make
    /// (`None`) ends the making: the device makes no more.
    pub(crate) fn take(&mut self, make: impl FnOnce(u32) -> Option<vk::QueryPool>) -> Option<Slot> {
        let count = self.made.len();
        for step in 0..count {
            let at = (self.at + step) % count;
            if let Some(slot) = self.made[at].take() {
                self.at = at;
                return Some(slot);
            }
        }
        if self.most.is_some_and(|most| count >= most) {
            return None;
        }
        let Some(queries) = make(self.per_slot * SLOTS) else {
            self.most = Some(count);
            return None;
        };
        self.made.push(Pool::new(queries, SLOTS, self.per_slot));
        self.at = count;
        self.made.last()?.take()
    }

    /// The slots of the pools made.
    pub(crate) fn slots(&self) -> usize {
        self.made.len() * SLOTS as usize
    }

    /// The pools made.
    pub(crate) fn queries(&self) -> impl Iterator<Item = vk::QueryPool> + '_ {
        self.made.iter().map(|pool| pool.queries)
    }
}

/// One of a device's timestamp query pools, of [`SLOTS`] slots, and the slots no command buffer
/// holds.
pub(crate) struct Pool {
    queries: vk::QueryPool,
    free: Arc<Mutex<Vec<u32>>>,
    /// The slots taken so far: each take's lease is the count after it.
    leases: AtomicU64,
    /// The queries of each slot.
    per_slot: u32,
}

impl Pool {
    /// The `slots` slots of `queries`, a pool of `per_slot * slots` timestamp queries, every one
    /// free.
    pub(crate) fn new(queries: vk::QueryPool, slots: u32, per_slot: u32) -> Pool {
        Pool {
            queries,
            // Popped from the end: slot 0 first.
            free: Arc::new(Mutex::new((0..slots).rev().collect())),
            leases: AtomicU64::new(0),
            per_slot,
        }
    }

    /// A free slot, `None` when every one is held.
    pub(crate) fn take(&self) -> Option<Slot> {
        let index = lock(&self.free).pop()?;
        Some(Slot {
            queries: self.queries,
            index,
            per_slot: self.per_slot,
            lease: self.leases.fetch_add(1, Ordering::Relaxed) + 1,
            free: Arc::clone(&self.free),
        })
    }
}

/// A slot of one of its device's timestamp pools: a command buffer's start and end, and where it
/// has three queries, the start its first encoder writes (the module docs). The command buffer
/// holds it from its first encoder's opening, or its commit, until the completion turn read
/// them; dropping it gives it back.
#[derive(Debug)]
pub(crate) struct Slot {
    queries: vk::QueryPool,
    index: u32,
    per_slot: u32,
    /// This holding of the slot: above every earlier holding's of its pool. A slot is taken again
    /// only once given back, so a holding of a slot with a higher lease began after every lower
    /// one's command buffer had its timestamps read, or failed (`command_buffer/kept_barriers.rs`
    /// takes a kept command buffer that writes the slot's queries again only then).
    lease: u64,
    free: Arc<Mutex<Vec<u32>>>,
}

impl Slot {
    /// The timestamp pool it is of.
    pub(crate) fn queries(&self) -> vk::QueryPool {
        self.queries
    }

    /// Its first query, the start a command buffer of barriers writes; the end is the next.
    pub(crate) fn first(&self) -> u32 {
        self.index * self.per_slot
    }

    /// The query of the start its command buffer's first encoder writes, where it has one (the
    /// module docs): after the end's.
    pub(crate) fn prefix(&self) -> Option<u32> {
        (self.per_slot == PREFIXED).then(|| self.first() + 2)
    }

    /// This holding's lease (the field's docs).
    pub(crate) fn lease(&self) -> u64 {
        self.lease
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        lock(&self.free).push(self.index);
    }
}

impl Timing {
    /// The timing of a device whose runtime queue family has `valid_bits` timestamp bits, ticking
    /// every `period` nanoseconds, calibrated by `calibration` where it has one ([`calibrated`]).
    /// With `from_host`, it answers GPU times from the host and writes no timestamps (the module
    /// docs), for that reason. With `scarce_pools`
    /// (`DriverFact::TimestampPoolsAreCounterSampleBuffers`) it makes one timestamp pool at most,
    /// else as many as its command buffers in flight need until the device refuses one. With
    /// `prefixed`, a slot also holds the start its command buffer's first encoder writes
    /// ([`PREFIXED`] queries; else [`PLAIN`]); never with `scarce_pools`. With `host_reset` (the
    /// device enables `hostQueryReset`), a slot is reset on the host as it is taken.
    pub(crate) fn new(
        valid_bits: u32,
        period: f32,
        calibration: Option<Calibration>,
        from_host: Option<FromHost>,
        scarce_pools: bool,
        prefixed: bool,
        host_reset: bool,
    ) -> Timing {
        Timing {
            clock: GpuClock::new(valid_bits, period, calibration),
            from_host,
            latest: Mutex::new(None),
            prefixed: prefixed && !scarce_pools,
            host_reset,
            pools: Mutex::new(match (scarce_pools, prefixed) {
                (true, _) => Pools::new(Some(1), PLAIN),
                (false, true) => Pools::new(None, PREFIXED),
                (false, false) => Pools::new(None, PLAIN),
            }),
        }
    }

    /// Whether its slots hold the start a command buffer's first encoder writes as it opens (the
    /// module docs): its command buffers take their slots then.
    pub(crate) fn prefixed(&self) -> bool {
        self.prefixed
    }

    /// Whether a slot is reset on the host as it is taken (the module docs): no command buffer
    /// resets its queries.
    pub(crate) fn resets_on_host(&self) -> bool {
        self.host_reset
    }

    /// A slot for a timed command buffer, from the device's pools, a pool made where
    /// every slot of the ones made is held ([`Pools::take`]), its queries reset on the host where
    /// the device resets them there (the module docs); `None` where the device writes no
    /// timestamps, or every slot is held and no pool more can be made.
    ///
    /// # Safety
    ///
    /// `device` is the live device of this timing.
    pub(crate) unsafe fn slot(&self, device: &ash::Device, counts: &Counters) -> Option<Slot> {
        self.clock.as_ref()?;
        let slot = lock(&self.pools).take(|count| {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(count);
            // SAFETY: the caller's; `info` is valid.
            unsafe { device.create_query_pool(&info, None) }.ok()
        })?;
        if self.host_reset {
            // SAFETY: the device enables `hostQueryReset`; the slot's queries are its pool's, and
            // free: its last holding's work completed or was never submitted, and no other
            // thread reads or resets them (the module docs).
            unsafe { device.reset_query_pool(slot.queries, slot.first(), slot.per_slot) };
            exit_guard::called(Call::ResetQueryPool);
            counts.timestamp_slot_reset_on_host();
        }
        Some(slot)
    }

    /// The slots of the timestamp pools the device made.
    pub(crate) fn slots(&self) -> usize {
        lock(&self.pools).slots()
    }

    /// Release the device's timestamp pools, at its teardown, inside its section.
    ///
    /// # Safety
    ///
    /// `device` is the live device of this timing, and none of its command buffers is pending.
    pub(crate) unsafe fn release(&self, device: &ash::Device) {
        for queries in lock(&self.pools).queries() {
            // SAFETY: the caller's.
            unsafe { release::object(device, queries) };
        }
    }

    /// The device's clock, where its queue family writes timestamps.
    pub(crate) fn clock(&self) -> Option<&GpuClock> {
        self.clock.as_ref()
    }

    /// Whether the device answers GPU times from the host, writing no timestamps (the module
    /// docs).
    pub(crate) fn answers_from_host(&self) -> bool {
        self.from_host.is_some()
    }

    /// Why the device answers GPU times from the host; `None` where it does not.
    pub(crate) fn host_clock_reason(&self) -> Option<FromHost> {
        self.from_host
    }

    /// The device's latest calibration reading, its pair and stated deviation, taken now when
    /// the last is [`REFRESH`] old or there is none, and counted in `counts`; `None` where the
    /// device has no calibration or the reading fails. A completion turn's, in its section, once
    /// the command buffer's timestamps were read (the module docs).
    pub(crate) fn pair(&self, counts: &Counters) -> Option<Reading> {
        let clock = self.clock.as_ref()?;
        clock.calibration?;
        let mut latest = lock(&self.latest);
        if let Some((at, reading)) = *latest {
            if at.elapsed() < REFRESH {
                return Some(reading);
            }
        }
        exit_guard::called(Call::GetCalibratedTimestamps);
        let (reading, readings) = clock.pair()?;
        counts.calibrated(readings, reading.deviation);
        *latest = Some((Instant::now(), reading));
        Some(reading)
    }
}

/// The calibration of `device` of `physical`, made from `instance` through `entry`, with
/// `extension` enabled where it is: where it offers its own time domain and one of the host's the
/// runtime reads ([`host_domain`]).
///
/// # Safety
///
/// `device` is a live device of `physical`, made from `instance` with `extension` enabled, and
/// the loader serves `vkGetDeviceProcAddr` for it (its loader word is in place).
pub(crate) unsafe fn calibrated(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    device: &ash::Device,
    extension: Option<Extension>,
) -> Option<Calibration> {
    // SAFETY: the caller's.
    extension.and_then(|extension| unsafe {
        calibration(entry, instance, physical, device, extension.0)
    })
}

impl Calibration {
    /// One reading of the device's time domain and the host's, where it succeeds: a completion
    /// turn's ([`GpuClock::pair`]) and priming's (`device/prime.rs`).
    pub(crate) fn reading(self) -> Option<Reading> {
        let infos = [
            vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE),
            vk::CalibratedTimestampInfoKHR::default().time_domain(self.domain),
        ];
        let mut values = [0u64; 2];
        let mut deviation = 0u64;
        #[cfg(unix)]
        let domain = domain_clock(self.domain)?;
        // SAFETY: the device is live (the caller holds it) and enabled the extension the entry
        // point belongs to; the arrays hold one value per info.
        let result = unsafe {
            (self.get)(
                self.device,
                infos.len() as u32,
                infos.as_ptr(),
                values.as_mut_ptr(),
                &mut deviation,
            )
        };
        if result != vk::Result::SUCCESS {
            return None;
        }
        #[cfg(unix)]
        {
            // The domain's clock onto the host's: both read at once, the host's around it.
            let (host, around) = if domain == BASE {
                (values[1], 0)
            } else {
                let first = read_clock(BASE)?;
                let during = read_clock(domain)?;
                let last = read_clock(BASE)?;
                let middle = first + last.saturating_sub(first) / 2;
                let offset = i128::from(middle) - i128::from(during);
                let host = u64::try_from(i128::from(values[1]) + offset).ok()?;
                (host, last.saturating_sub(first))
            };
            Some(Reading {
                pair: Pair {
                    ticks: values[0],
                    host,
                },
                deviation: deviation.saturating_add(around),
            })
        }
        #[cfg(not(unix))]
        {
            None
        }
    }
}

/// The calibration of `device` through `names`, where it offers its own time domain and one of
/// the host's the runtime reads.
///
/// # Safety
///
/// As [`calibrated`]'s, `names` the form enabled.
unsafe fn calibration(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    device: &ash::Device,
    names: &Names,
) -> Option<Calibration> {
    // SAFETY: the instance is live; a command it does not have is null.
    let domains =
        unsafe { entry.get_instance_proc_addr(instance.handle(), names.domains.as_ptr()) }?;
    // SAFETY: the loader gave the command of this name, of this signature.
    let domains = unsafe {
        std::mem::transmute::<
            unsafe extern "system" fn(),
            vk::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR,
        >(domains)
    };
    let mut count = 0u32;
    // SAFETY: `physical` is the instance's; the first call counts.
    let counted = unsafe { domains(physical, &mut count, std::ptr::null_mut()) };
    if counted != vk::Result::SUCCESS {
        return None;
    }
    let mut offered = vec![vk::TimeDomainKHR::default(); count as usize];
    // SAFETY: as above, into `count` elements.
    let listed = unsafe { domains(physical, &mut count, offered.as_mut_ptr()) };
    if listed != vk::Result::SUCCESS && listed != vk::Result::INCOMPLETE {
        return None;
    }
    offered.truncate(count as usize);
    let domain = host_domain(&offered)?;
    let name: *const c_char = names.get.as_ptr();
    // SAFETY: the caller's: the device is live and the loader serves the call.
    let get = unsafe { (instance.fp_v1_0().get_device_proc_addr)(device.handle(), name) }?;
    Some(Calibration {
        // SAFETY: the device gave the command of this name, of this signature.
        get: unsafe {
            std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetCalibratedTimestampsKHR>(
                get,
            )
        },
        device: device.handle(),
        domain,
    })
}

/// A command buffer's kernel times, in nanoseconds of the host's clock, 0 until known: apart
/// from its state, so a read waits for no submission in progress (the module docs).
#[derive(Debug, Default)]
pub(crate) struct KernelTimes {
    /// The scheduler began submitting its first batch.
    start: AtomicU64,
    /// The scheduler finished submitting its last batch.
    end: AtomicU64,
}

impl KernelTimes {
    /// The scheduler begins submitting the first batch now.
    pub(crate) fn begin(&self) {
        self.start.store(host_nanos(), Ordering::Release);
    }

    /// The scheduler finished submitting the last batch at host time `at`.
    pub(crate) fn end_at(&self, at: u64) {
        self.end.store(at, Ordering::Release);
    }

    /// When the scheduler began submitting the first batch; 0 before.
    pub(crate) fn start(&self) -> u64 {
        self.start.load(Ordering::Acquire)
    }

    /// When the scheduler finished submitting the last batch; 0 before.
    pub(crate) fn end(&self) -> u64 {
        self.end.load(Ordering::Acquire)
    }
}

/// A command buffer's other times, in its state: host times in nanoseconds of the host's clock,
/// 0 until known.
#[derive(Debug, Default)]
pub(crate) struct Times {
    /// Its slot of the device's timestamp pool, from its first encoder's opening (or its commit)
    /// until the completion turn read its timestamps.
    pub(crate) slot: Option<Slot>,
    /// Its first batch wrote the start into the slot's third query alone, the first encoder's
    /// prefix (the module docs): the start read is that one, not the first.
    pub(crate) prefixed: bool,
    /// What its timestamps gave.
    pub(crate) stamped: Stamped,
    /// The device's calibration reading as it completed, where the device has one.
    pub(crate) pair: Option<Reading>,
    /// A completion turn saw its last batch complete.
    pub(crate) completed: u64,
    /// Its GPU start and end, once read.
    pub(crate) gpu: Option<(u64, u64)>,
    /// Its GPU start and end where it wrote no timestamps: placed among its queue's command
    /// buffers by the completion turn as it completed.
    pub(crate) placed: Option<(u64, u64)>,
}

/// What a command buffer's timestamps gave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Stamped {
    /// Nothing yet: it has not completed, or the device writes no timestamps.
    #[default]
    Unread,
    /// It encoded nothing and waits for nothing, and writes none: its GPU start and end are
    /// where its queue reached it, as Metal's (the module docs).
    NoWork,
    /// Its device answers GPU times from the host, and it writes none (the module docs).
    FromHost,
    /// Every slot of the device's pools was held at its commit, and no pool more could be made:
    /// it has none.
    NoSlot,
    /// Read once it completed: its start and end ticks, where the GPU wrote both.
    Read(Option<(u64, u64)>),
    /// The read failed with this result.
    Failed(vk::Result),
}

/// `nanos` of the host's clock in seconds; 0 stays 0.
pub(crate) fn seconds(nanos: u64) -> f64 {
    nanos as f64 / 1e9
}
