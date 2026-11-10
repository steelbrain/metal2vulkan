//! The four-queue storm on the device the cfg selects and, on a Mac's GPU, on Metal in the same
//! test: what light command buffers wait while another queue keeps the GPU busy.
//!
//! One busy queue keeps two command buffers in flight, each one dispatch of one thread spinning
//! (the owned fixture `kernel_spin_on_an_unraised_word`, which loads a word no one raises until
//! its cap of loads runs out), its cap grown until one alone takes about [`BUSY`] of GPU time (its
//! GPU window, [`busy_alone`]). Three light queues each commit bursts of [`BURST`] command
//! buffers, each one blit filling 256 bytes of a `Private` buffer, and wait for a burst's last
//! before the next, as a guest rings its doorbells and then waits on a stamp. Two phases of [`PHASE`] each, run in
//! [`SLICES`] slices: the busy queue alone, then the busy queue and the light queues together.
//! Every input is the lane's own.
//!
//! Metal runs command buffers of different queues side by side: a light burst waits at most for
//! the busy command buffer running when it was committed (on the Apple M4 another queue's work
//! does not start while one long dispatch of one threadgroup runs, on Metal and MoltenVK alike),
//! not behind the busy queue's next one too, as it does where every queue is one `VkQueue`.
//! Where Metal runs (a Mac's GPU), the light bursts' median and 99th percentile, each in units of
//! a busy command buffer, must stay within half a busy command buffer and one busy command buffer
//! of the reference's, the 99th percentile also within twice the host's delivery of a completion
//! (below), and the busy queue must keep its throughput beside the light queues as well as the
//! reference does, within a quarter of it (`metal::judge`). The unit, for both
//! sides, is the median GPU window of the reference's busy command buffers that ran beside the
//! light queues, by Metal's GPU timestamps ([`Storm::unit`], [`interleaved`]): the cap those
//! timestamps chose is the runtime's too, and both spin the same loads, so the same GPU work sets
//! the unit on both sides and the host's load moves it on neither. The fixture calls nothing a
//! translation drops: `rfs_agx_yield`'s `agx_yield_lookback`, the spin before it, yields
//! (`llvm.agx3.yield`) in Metal's compilation alone, and on the M3 Max took 7.4 ms of GPU time at
//! a cap of 8,192 loads in Metal and 2.5 to 3.8 ms in the runtime's windows (the host's view of
//! them, below). This one, alone on the M3 Max, 5 times each: 3.60 to 4.62 ms at a cap of 32,768
//! in Metal and 4.11 to 4.72 ms in the runtime's windows; at the cap the storm chose, 8.62 to 8.77
//! ms in Metal and 8.39 to 8.46 ms in the runtime's windows, in 5 storms of 5. A busy command buffer's time from its commit to
//! its wait's return, alone, stretches with the host's load: on the M3 Max in a full parallel suite
//! it was 19.75 ms against a GPU window of 7.39 ms. The runtime's own windows, where its device
//! writes no timestamps (MoltenVK 1.4.1 on Apple's GPUs, `gpu_time.rs`), run from the later of its
//! submission and its queue's previous completion to its own completion as the runtime saw it,
//! which a loaded host shrinks as well as stretches: beside the runtime lane's other tests on the
//! M3 Max their median read 0.98 ms where the spin alone took 7.93 ms, so the runtime's bursts read
//! 6.97 such windows against the reference's 2.32 busy command buffers. They are printed, and
//! judge nothing. Where no reference runs, a side's unit is the median of its own windows. The
//! reference and the runtime run their slices in turn ([`interleaved`]), so a change in the host's
//! load meets both, and each storm prints the host's load average before and after it. With the
//! reference's unit on the M3 Max, the runtime passed 5 storms of 5 on a quiet host (its bursts'
//! median 1.95 to 2.01 busy command buffers, 99th percentile 2.06 to 2.19, against the reference's
//! 1.66 to 2.03 and 2.12 to 2.18); with 10 ms spun at the start of each of its completion turns (a
//! probe build) it failed 3 of 3, its median 3.53 against 2.03 and its busy queue keeping 0.64 to
//! 0.75 of its throughput against 1.01 to 1.03.
//! Beside the runtime lane's other tests it passed 1 storm of 4, failing at the 99th percentile on
//! MoltenVK's completion path, so on macOS it runs alone (`.config/nextest.toml` keeps the
//! numbers).
//!
//! The host still delays the threads a light burst goes through, and the runtime's burst goes
//! through two more than Metal's: its commit is taken up on the device's scheduler thread, and its
//! completion, which MoltenVK's completed handler signals on Metal's completion thread as Metal's
//! own handler is, reaches the waiting thread through the `VkQueue`'s waiter. So the 99th
//! percentile may also exceed the reference's by twice the reference's own delivery of a
//! completion at its 99th percentile, from a light burst's GPU end by Metal's timestamps to the
//! return of the wait for it (Metal's completion thread, then the waiting thread), measured in the
//! same slices ([`Storm::delivery`]). On the M3 Max under the harness's utility clamp
//! (`taskpolicy -c utility`, which lowers the runtime's user-interactive threads with every other),
//! that delivery took 0.33 to 0.51 ms at the 99th percentile with no load added, 0.04 to 0.06
//! busy command buffers, in 4 storms. Running alone in a full suite at a load average of 23.9, the
//! runtime's 99th percentile was 4.71 busy command buffers against the reference's 3.15, its median
//! within 0.07 of the reference's, plain Metal's own bursts taking 12.31 ms at the 99th percentile
//! against 0.60 to 0.66 ms with no load added. Beside 8 threads spinning at the default class,
//! above the clamp (a load average of 22.5), the runtime's 99th percentile was 7.10 busy command
//! buffers against the reference's 5.13, its median 2.16 against 2.08, and the reference's
//! deliveries took 11.76 ms (2.77 busy command buffers) at their 99th percentile. Beside 16 such
//! threads, one for each of its CPUs, those deliveries took 203 ms, and the runtime's median was
//! 15.55 busy command buffers against 8.41: the storm fails there.
//!
//! In our plain Vulkan programs of the runtime's submissions on an M4, every queue on MoltenVK's
//! one queue of family 0 (as the runtime once did) put the median 1 to 1.4 busy command buffers
//! above Metal's in the same run; a queue per family, 0.4 at most.
//!
//! The reference is Metal's own storm, but on MoltenVK 1.4.1 with Metal argument buffers and a
//! residency set ([`moltenvk_fences`]). There MoltenVK encodes device-wide `MTLFence`s, one set per
//! barrier stage, around every encoder of every queue: each encoder waits at its start on every
//! stage's first fence, updates another of its own stage's at its end, and each submission, as its
//! last command buffer ends, carries that one into its stage's first fence (`MVKCommandEncoder`'s
//! `barrierWait` and `barrierUpdate`, `MVKCommandEncodingContext::syncFences`,
//! `MVKCommandBuffer.mm` and `MVKQueue.mm`). So a light queue's blit waits for the compute fence
//! that the busy queue's latest committed command buffer updates as it ends: about two busy command
//! buffers, where Metal waits for none. Metal's own storm under that fence pattern
//! (`metal::Fences::MoltenVk`) waits as long. Our probe, our own Swift storm on an Apple M3
//! Max under macOS 27.0.1: a light burst's median was 1.91 and 1.95 busy command buffers with the
//! fences, 0.09 and 0.08 without. The runtime's was 1.92 and 1.95 in the same hour, at the default
//! QoS and at utility. Nothing on the runtime's side removes it: the runtime needs argument buffers
//! (`device/argument_buffers.rs`), MoltenVK has no setting for its residency set, and the runtime
//! runs on stock MoltenVK. So there the runtime's storm is judged against Metal's under MoltenVK's
//! fences, with the same margins, and the lane measures the runtime's own share; plain Metal's
//! storm runs too and is printed (docs/FAST-PATHS.md, device gaps).
//!
//! The runtime's device opens without the validation layer, as Metal runs without its own: the
//! layer's cost on every call is no part of what the storm compares (the lanes that check the
//! device's queues under the layer are `handles`, `command_queues` and the rest).

use super::{
    at_plan, nothing_reported_without_the_layer, validated_device_with, Messages, ScratchCache,
    VALIDATION_LAYER,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan::vk::DriverId;
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    PipelineOptions, ResourceOptions, Size,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

/// How long one busy command buffer alone takes, at least, once calibrated.
const BUSY: Duration = Duration::from_millis(6);

/// The light queues.
const LIGHTS: usize = 3;

/// The command buffers of a light burst.
const BURST: usize = 8;

/// Each phase's length, over its slices.
const PHASE: Duration = Duration::from_secs(2);

/// The slices each phase runs in: where two sides are compared, their slices alternate
/// ([`interleaved`]).
const SLICES: u32 = 4;

/// The longest anything waits for a command buffer.
const WAIT: Duration = Duration::from_secs(30);

/// The tries of a storm whose busy work changed speed under it.
const TRIES: usize = 3;

/// The MoltenVK releases (`driverVersion`) whose encoding [`moltenvk_fences`] was read at: 1.4.1,
/// and 1.4.2, whose fence code (`barrierWait`, `barrierUpdate`, `syncFences` and the residency
/// set's making) is 1.4.1's. On the M3 Max under macOS 27.0.1 at 1.4.2, this lane judged without
/// the fences saw a light burst's median of 1.93 busy command buffers, within the fenced probe's
/// 1.91 to 1.95 above, and it passes judged with them.
const MOLTENVK_READ: [u32; 2] = [10401, 10402];

/// `maxPerStageDescriptorUpdateAfterBindSampledImages` where MoltenVK binds through tier-2 Metal
/// argument buffers (`device/argument_buffers.rs` reads it so at [`MOLTENVK_READ`]).
const TIER_2_ARGUMENT_BUFFERS: u32 = 1_000_000;

/// Whether `device` is MoltenVK encoding its device-wide barrier fences around every encoder (the
/// module docs): MoltenVK at [`MOLTENVK_READ`], whose `MVKDevice::initQueues` makes a residency
/// set where argument buffers are in use and the GPU is Apple6 or later under macOS 15 or later
/// (`MVKPhysicalDevice::initMetalFeatures`), and whose encoders then wait on and update the
/// fences. Every Apple GPU a Mac has is Apple6 or later.
fn moltenvk_fences(device: &Device) -> bool {
    let description = &device.properties().description;
    description.driver_id == Some(DriverId::MOLTENVK)
        && MOLTENVK_READ.contains(&description.driver_version)
        && description.vendor_id == 0x106b
        && description
            .limits
            .max_per_stage_descriptor_update_after_bind_sampled_images
            == TIER_2_ARGUMENT_BUFFERS
        && super::host_release().is_some_and(|[major, ..]| major >= 15)
}

/// The busy kernel's typed AIR text and its LLVM IR, the fixture's.
fn fixture(extension: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "fixtures/public/engine/kernel_spin_on_an_unraised_word.{extension}"
    ))
}

/// What the storm needs of a side, the runtime or Metal: a busy command buffer, a light burst,
/// and a wait. Each light thread makes its own target, so a side's buffers stay on the thread
/// that made them.
trait Side: Sync {
    /// A committed command buffer, which the thread that committed it waits for.
    type Committed;
    /// A light queue's buffer.
    type Target;
    /// The side's name, for the report.
    fn name(&self) -> String;
    /// Commit one busy command buffer on the busy queue, its spin capped at `cap` loads.
    fn busy(&self, cap: u32) -> Result<Self::Committed, String>;
    /// A light queue's buffer of 256 bytes, `Private`.
    fn target(&self) -> Result<Self::Target, String>;
    /// Commit [`BURST`] light command buffers on light queue `light`, each filling `target` with
    /// `value`: the last.
    fn burst(
        &self,
        light: usize,
        target: &Self::Target,
        value: u8,
    ) -> Result<Self::Committed, String>;
    /// Wait for `committed`, which must complete.
    fn wait(&self, committed: Self::Committed) -> Result<(), String>;
    /// Wait for `committed`, which must complete, and give its completion's delivery: from its
    /// GPU end by the GPU's own timestamps to the wait's return. `None` where the side has no
    /// timestamps of its own GPU end (the runtime's on MoltenVK, `gpu_time.rs`): only the
    /// reference's is read ([`Storm::delivery`]).
    fn wait_delivered(&self, committed: Self::Committed) -> Result<Option<Duration>, String>;
    /// Wait for `committed`, which must complete, and give its GPU window: from its GPU start
    /// time to its GPU end time.
    fn wait_timed(&self, committed: Self::Committed) -> Result<Duration, String>;
}

/// A storm's measurements.
#[derive(Debug)]
struct Storm {
    /// The GPU windows of the busy command buffers that completed beside the light queues,
    /// sorted.
    windows: Vec<Duration>,
    /// The unit the bursts are counted in: the median of [`Storm::windows`], or another side's
    /// where its GPU windows set it ([`interleaved`]).
    unit: Duration,
    /// Whether another side's GPU windows set [`Storm::unit`].
    unit_given: bool,
    /// The busy work's GPU window alone, the median of three, before the phases and after them.
    gpu: [Duration; 2],
    /// Why the busy work's speed changed across the phases in every try ([`tried`]), so the
    /// storm compares nothing with Metal's; `None` where it held.
    unsteady: Option<String>,
    /// Busy command buffers completed a second, the busy queue alone.
    alone: f64,
    /// Busy command buffers completed a second beside the light queues.
    beside: f64,
    /// Every light burst, from its first commit to its last's completion, sorted.
    bursts: Vec<Duration>,
    /// Each light burst's completion delivery, where the side gives it
    /// ([`Side::wait_delivered`]), sorted.
    deliveries: Vec<Duration>,
    /// The host's load average over a minute, before the storm and after it, where the host
    /// gives one.
    load: [Option<f64>; 2],
}

impl Storm {
    /// The bursts' `p`th percentile in busy command buffers.
    fn percentile(&self, p: usize) -> f64 {
        let at = (self.bursts.len() * p / 100).min(self.bursts.len() - 1);
        self.bursts[at].as_secs_f64() / self.unit.as_secs_f64()
    }

    /// The light bursts' completion delivery at the 99th percentile, in busy command buffers,
    /// where the side gives it ([`Side::wait_delivered`]).
    fn delivery(&self) -> Option<f64> {
        let at = (self.deliveries.len() * 99 / 100).min(self.deliveries.len().max(1) - 1);
        (self.deliveries.get(at)).map(|delivery| delivery.as_secs_f64() / self.unit.as_secs_f64())
    }

    /// The fraction of its throughput alone the busy queue keeps beside the light queues.
    fn kept(&self) -> f64 {
        self.beside / self.alone
    }

    /// The report line.
    fn line(&self, side: &str) -> String {
        let unit = self.unit.as_secs_f64();
        let load =
            |load: Option<f64>| load.map_or("unknown".to_owned(), |load| format!("{load:.1}"));
        format!(
            "STORM {side}: busy {:.2} ms of GPU beside the light queues ({} command buffers; \
             {:.2} ms alone before the phases, {:.2} ms after); unit {:.2} ms{}; bursts {} median \
             {:.2} ms ({:.2} busy), p99 {:.2} ms ({:.2} busy); busy {:.0}/s alone, {:.0}/s beside \
             ({:.2} kept){}; load average {} before, {} after{}",
            self.windows[self.windows.len() / 2].as_secs_f64() * 1e3,
            self.windows.len(),
            self.gpu[0].as_secs_f64() * 1e3,
            self.gpu[1].as_secs_f64() * 1e3,
            unit * 1e3,
            if self.unit_given {
                " (the reference's)"
            } else {
                ""
            },
            self.bursts.len(),
            self.percentile(50) * unit * 1e3,
            self.percentile(50),
            self.percentile(99) * unit * 1e3,
            self.percentile(99),
            self.alone,
            self.beside,
            self.kept(),
            self.delivery().map_or(String::new(), |delivery| format!(
                "; completions delivered {:.2} ms after their GPU end at the 99th percentile",
                delivery * unit * 1e3
            )),
            load(self.load[0]),
            load(self.load[1]),
            self.unsteady
                .as_ref()
                .map_or(String::new(), |why| format!("; unsteady: {why}")),
        )
    }
}

/// The host's load average over the last minute, where the host gives one.
fn load_average() -> Option<f64> {
    let mut loads = [0.0f64; 1];
    // SAFETY: `getloadavg` writes at most the one element it is given.
    let read = unsafe { libc::getloadavg(loads.as_mut_ptr(), 1) };
    (read == 1).then_some(loads[0])
}

/// One busy command buffer alone at `cap`, three times: the median of their GPU windows, the
/// spin's speed. The GPU window is from the device's timestamps (on Metal, and the runtime where
/// its device writes them; elsewhere a window the work ran in, `gpu_time.rs`), without the
/// commit's and the wait's own time, which a host loaded by other processes stretches.
fn busy_alone<S: Side>(side: &S, cap: u32) -> Result<Duration, String> {
    let mut windows = Vec::new();
    for _ in 0..3 {
        windows.push(side.wait_timed(side.busy(cap)?)?);
    }
    windows.sort();
    Ok(windows[1])
}

/// What a slice of a phase measured, or the phase's slices together.
#[derive(Debug, Default)]
struct Measured {
    /// The busy command buffers completed, and in how long.
    done: u32,
    elapsed: Duration,
    /// Each light burst's time.
    bursts: Vec<Duration>,
    /// Each light burst's completion delivery, where the side gives it.
    deliveries: Vec<Duration>,
    /// Each completed busy command buffer's GPU window.
    windows: Vec<Duration>,
}

impl Measured {
    /// Add `slice`'s measurements.
    fn add(&mut self, slice: Measured) {
        self.done += slice.done;
        self.elapsed += slice.elapsed;
        self.bursts.extend(slice.bursts);
        self.deliveries.extend(slice.deliveries);
        self.windows.extend(slice.windows);
    }

    /// The busy command buffers completed a second.
    fn rate(&self) -> f64 {
        f64::from(self.done) / self.elapsed.as_secs_f64()
    }
}

/// A slice of `length` of a phase: the busy queue, with the light queues where `lights`. The busy
/// command buffers completed, with their GPU windows, and every light burst's time.
fn phase<S: Side>(side: &S, cap: u32, lights: bool, length: Duration) -> Result<Measured, String> {
    let stop = AtomicBool::new(false);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let fail = |error: String| {
        failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert(error);
    };
    let bursts: Mutex<(Vec<Duration>, Vec<Duration>)> = Mutex::new((Vec::new(), Vec::new()));
    let (done, elapsed, windows) = std::thread::scope(|scope| {
        let busy = scope.spawn(|| {
            let began = Instant::now();
            let mut flight = std::collections::VecDeque::new();
            let mut windows = Vec::new();
            while !stop.load(Ordering::SeqCst) {
                if flight.len() == 2 {
                    match side.wait_timed(flight.pop_front().expect("two in flight")) {
                        Ok(window) => windows.push(window),
                        Err(error) => {
                            fail(error);
                            return (0, began.elapsed(), windows);
                        }
                    }
                }
                match side.busy(cap) {
                    Ok(committed) => flight.push_back(committed),
                    Err(error) => {
                        fail(error);
                        return (0, began.elapsed(), windows);
                    }
                }
            }
            let elapsed = began.elapsed();
            let done = windows.len() as u32;
            for committed in flight {
                if let Err(error) = side.wait(committed) {
                    fail(error);
                }
            }
            (done, elapsed, windows)
        });
        let light_threads: Vec<_> = (0..if lights { LIGHTS } else { 0 })
            .map(|light| {
                let (stop, bursts, fail) = (&stop, &bursts, &fail);
                scope.spawn(move || {
                    let run = || -> Result<(Vec<Duration>, Vec<Duration>), String> {
                        let target = side.target()?;
                        // The light queue's first burst, not timed.
                        side.wait(side.burst(light, &target, 0)?)?;
                        let (mut times, mut deliveries) = (Vec::new(), Vec::new());
                        let mut value = 0u8;
                        while !stop.load(Ordering::SeqCst) {
                            value = value.wrapping_add(1);
                            let at = Instant::now();
                            let delivered =
                                side.wait_delivered(side.burst(light, &target, value)?)?;
                            times.push(at.elapsed());
                            deliveries.extend(delivered);
                        }
                        Ok((times, deliveries))
                    };
                    match run() {
                        Ok((times, deliveries)) => {
                            let mut all = bursts.lock().unwrap_or_else(PoisonError::into_inner);
                            all.0.extend(times);
                            all.1.extend(deliveries);
                        }
                        Err(error) => fail(error),
                    }
                })
            })
            .collect();
        std::thread::sleep(length);
        stop.store(true, Ordering::SeqCst);
        for thread in light_threads {
            thread.join().expect("a light thread");
        }
        busy.join().expect("the busy thread")
    });
    if let Some(error) = failure.into_inner().unwrap_or_else(PoisonError::into_inner) {
        return Err(error);
    }
    let (bursts, deliveries) = bursts.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ok(Measured {
        done,
        elapsed,
        bursts,
        deliveries,
        windows,
    })
}

/// A side's storm under way: its busy work's cap, grown until one busy command buffer alone takes
/// [`BUSY`] of GPU time (or another side's, [`interleaved`]), its GPU window alone before the
/// phases, and what its slices measured.
struct Run<'a, S: Side> {
    side: &'a S,
    cap: u32,
    before: Duration,
    load: Option<f64>,
    alone: Measured,
    beside: Measured,
}

impl<'a, S: Side> Run<'a, S> {
    /// The busy kernel's first use, then its cap grown until its GPU window takes [`BUSY`].
    fn calibrated(side: &'a S) -> Result<Run<'a, S>, String> {
        let load = load_average();
        side.wait(side.busy(1 << 8)?)?;
        let mut cap: u32 = 1 << 8;
        let before = loop {
            let window = busy_alone(side, cap)?;
            if window >= BUSY || cap >= 1 << 28 {
                break window;
            }
            cap = cap.saturating_mul(2);
        };
        Ok(Run {
            side,
            cap,
            before,
            load,
            alone: Measured::default(),
            beside: Measured::default(),
        })
    }

    /// The busy kernel's first use, then its GPU window alone at `cap`, another side's
    /// ([`interleaved`]).
    #[cfg(all(m2v_gpu, target_os = "macos"))]
    fn at(side: &'a S, cap: u32) -> Result<Run<'a, S>, String> {
        let load = load_average();
        side.wait(side.busy(1 << 8)?)?;
        let before = busy_alone(side, cap)?;
        Ok(Run {
            side,
            cap,
            before,
            load,
            alone: Measured::default(),
            beside: Measured::default(),
        })
    }

    /// A slice of each phase: the busy queue alone, then beside the light queues.
    fn slice(&mut self) -> Result<(), String> {
        let length = PHASE / SLICES;
        let alone = phase(self.side, self.cap, false, length)?;
        self.alone.add(alone);
        let beside = phase(self.side, self.cap, true, length)?;
        self.beside.add(beside);
        Ok(())
    }

    /// The storm its slices measured, with the busy work's GPU window alone after them, its
    /// bursts counted in `unit` where another side's GPU windows give it ([`interleaved`]), else
    /// in the median of its own windows beside the light queues. A side that sets its own unit is
    /// unsteady where its window alone after the slices is not within half and twice its window
    /// before them (a Radeon 780M alone ran the spin at 9.4 ms of GPU time, then 3.1 ms once its
    /// clock rose); another side's windows are printed.
    fn finish(self, unit: Option<Duration>) -> Result<Storm, String> {
        let after = busy_alone(self.side, self.cap)?;
        let (before, name) = (self.before, self.side.name());
        let (alone, beside) = (self.alone.rate(), self.beside.rate());
        let Measured {
            mut bursts,
            mut deliveries,
            mut windows,
            ..
        } = self.beside;
        if bursts.is_empty() {
            return Err(format!("{name}: no light burst completed"));
        }
        if windows.is_empty() {
            return Err(format!(
                "{name}: no busy command buffer completed beside the light queues"
            ));
        }
        bursts.sort();
        deliveries.sort();
        windows.sort();
        let unsteady = (unit.is_none() && (after * 2 < before || after > before * 2)).then(|| {
            format!(
                "{name}: the busy work's GPU window was {before:?} before the phases and {after:?} \
                 after"
            )
        });
        Ok(Storm {
            unit: unit.unwrap_or(windows[windows.len() / 2]),
            unit_given: unit.is_some(),
            windows,
            gpu: [before, after],
            unsteady,
            alone,
            beside,
            bursts,
            deliveries,
            load: [self.load, load_average()],
        })
    }
}

/// `once`'s storms, tried again (at most [`TRIES`] times) while one of them is unsteady
/// ([`Run::finish`]): a storm measured across a change of the spin's speed compares nothing with
/// Metal's, so the last try is given with why, which the comparison with Metal refuses
/// (`metal::judge`).
fn tried<const N: usize>(
    mut once: impl FnMut() -> Result<[Storm; N], String>,
) -> Result<[Storm; N], String> {
    let mut last = None;
    for _ in 0..TRIES {
        let storms = once()?;
        let unsteady: Vec<&String> = storms
            .iter()
            .filter_map(|storm| storm.unsteady.as_ref())
            .collect();
        if unsteady.is_empty() {
            return Ok(storms);
        }
        for why in unsteady {
            eprintln!("STORM retry: {why}");
        }
        last = Some(storms);
    }
    Ok(last.expect("at least one try"))
}

/// The storm on `side`: its busy work calibrated, then the two phases, slice by slice.
fn storm<S: Side>(side: &S) -> Result<Storm, String> {
    let [storm] = tried(|| {
        let mut run = Run::calibrated(side)?;
        for _ in 0..SLICES {
            run.slice()?;
        }
        Ok([run.finish(None)?])
    })?;
    Ok(storm)
}

/// The storms on `a`, the runtime, and on `b`, Metal: `b`'s busy work calibrated by its GPU
/// timestamps, and its cap given to `a`, so both spin the same loads, then their slices in turn: a
/// slice of each phase on `a`, then on `b`, [`SLICES`] times, so a change in the host's load meets
/// both (the module docs). `b`'s GPU windows alone give both sides' unit and judge the spin's
/// speed steady: `a`'s, where its device writes no timestamps, are the host's view of them, which
/// the host's load shrinks as well as stretches, printed.
#[cfg(all(m2v_gpu, target_os = "macos"))]
fn interleaved<A: Side, B: Side>(a: &A, b: &B) -> Result<[Storm; 2], String> {
    tried(|| {
        let mut b = Run::calibrated(b)?;
        let mut a = Run::at(a, b.cap)?;
        for _ in 0..SLICES {
            a.slice()?;
            b.slice()?;
        }
        let b = b.finish(None)?;
        let a = a.finish(Some(b.unit))?;
        Ok([a, b])
    })
}

/// The runtime's side: its device, a busy queue and [`LIGHTS`] light queues, and the busy
/// kernel's state.
struct Runtime {
    device: Device,
    queues: Vec<CommandQueue>,
    state: ComputePipelineState,
}

impl Runtime {
    fn new(device: Device) -> Runtime {
        let library = device
            .new_library_with_air(&llvm_assemble(&fixture("ll")).unwrap())
            .unwrap();
        let function = library
            .new_function_with_name("spin_on_an_unraised_word")
            .unwrap();
        let state = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let queues = (0..=LIGHTS)
            .map(|_| device.new_command_queue().unwrap())
            .collect();
        Runtime {
            device,
            queues,
            state,
        }
    }
}

impl Side for Runtime {
    type Committed = CommandBuffer;
    type Target = Buffer;

    fn name(&self) -> String {
        format!("the runtime on {}", self.device.name())
    }

    fn busy(&self, cap: u32) -> Result<CommandBuffer, String> {
        let failed = |error: metal2vulkan_runtime::Error| format!("a busy command buffer: {error}");
        let with = |bytes: &[u8]| {
            self.device
                .new_buffer_with_bytes(bytes, ResourceOptions::shared())
                .map_err(failed)
        };
        let (word, loads, cap) = (with(&[0; 4])?, with(&[0; 4])?, with(&cap.to_le_bytes())?);
        let command_buffer = self.queues[0].command_buffer().map_err(failed)?;
        let mut encoder = command_buffer.compute_command_encoder().map_err(failed)?;
        encoder
            .set_compute_pipeline_state(&self.state)
            .map_err(failed)?;
        encoder.set_buffer(Some(&word), 0, 0).map_err(failed)?;
        encoder.set_buffer(Some(&loads), 0, 1).map_err(failed)?;
        encoder.set_buffer(Some(&cap), 0, 2).map_err(failed)?;
        let one = Size::new(1, 1, 1);
        encoder.dispatch_threadgroups(one, one).map_err(failed)?;
        encoder.end_encoding().map_err(failed)?;
        command_buffer.commit().map_err(failed)?;
        Ok(command_buffer)
    }

    fn target(&self) -> Result<Buffer, String> {
        self.device
            .new_buffer(256, ResourceOptions::private())
            .map_err(|error| format!("a light buffer: {error}"))
    }

    fn burst(&self, light: usize, target: &Buffer, value: u8) -> Result<CommandBuffer, String> {
        let failed =
            |error: metal2vulkan_runtime::Error| format!("a light command buffer: {error}");
        let mut last = None;
        for _ in 0..BURST {
            let command_buffer = self.queues[1 + light].command_buffer().map_err(failed)?;
            let mut blit = command_buffer.blit_command_encoder().map_err(failed)?;
            blit.fill_buffer(target, 0..256, value).map_err(failed)?;
            blit.end_encoding().map_err(failed)?;
            command_buffer.commit().map_err(failed)?;
            last = Some(command_buffer);
        }
        last.ok_or_else(|| "an empty burst".to_owned())
    }

    fn wait(&self, command_buffer: CommandBuffer) -> Result<(), String> {
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
        match command_buffer.status() {
            CommandBufferStatus::Completed => Ok(()),
            status => Err(format!(
                "a command buffer ended {status:?}: {:?}",
                command_buffer.error()
            )),
        }
    }

    fn wait_delivered(&self, command_buffer: CommandBuffer) -> Result<Option<Duration>, String> {
        self.wait(command_buffer).map(|()| None)
    }

    fn wait_timed(&self, command_buffer: CommandBuffer) -> Result<Duration, String> {
        self.wait(command_buffer.clone())?;
        let start = command_buffer.gpu_start_time();
        let end = command_buffer.gpu_end_time();
        match (start, end) {
            (Ok(start), Ok(end)) => Ok(Duration::from_secs_f64((end - start).max(0.0))),
            times => Err(format!("a busy command buffer's GPU times: {times:?}")),
        }
    }
}

/// Plain Metal's side on this Mac's GPU: a busy queue and [`LIGHTS`] light queues, and the busy
/// kernel's state, from the fixture's typed AIR text.
#[cfg(all(m2v_gpu, target_os = "macos"))]
mod metal {
    use super::{fixture, Side, Storm, BURST};
    use crate::gpu::compile;
    use crate::gpu::wait::metal::{settle_watched, watch, Watch};
    use metal2vulkan_devtools::metal_assemble;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{NSRange, NSString};
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLFence, MTLLibrary,
        MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::time::Duration;

    /// How much longer than Metal's a light burst's median may be, in busy command buffers.
    const MEDIAN_MARGIN: f64 = 0.5;

    /// How much longer than Metal's a light burst's 99th percentile may be, in busy command
    /// buffers.
    const P99_MARGIN: f64 = 1.0;

    /// How much more of its throughput alone than Metal's the busy queue may lose beside the
    /// light queues, as a fraction of its throughput alone.
    const BUSY_MARGIN: f64 = 0.25;

    /// The host's thread hand-offs a light burst makes on the runtime's side beyond Metal's (the
    /// module docs), each allowed the reference's completion delivery at its 99th percentile
    /// ([`Storm::delivery`]) on top of [`P99_MARGIN`].
    const HAND_OFFS: f64 = 2.0;

    /// The runtime's storm `ours` against Metal's `metal` (the module docs), both counted in
    /// Metal's unit and measured at a steady spin ([`super::tried`]).
    pub(super) fn judge(ours: &Storm, metal: &Storm) {
        for storm in [ours, metal] {
            assert!(storm.unsteady.is_none(), "{:?}", storm.unsteady);
        }
        let (median, metal_median) = (ours.percentile(50), metal.percentile(50));
        assert!(
            median <= metal_median + MEDIAN_MARGIN,
            "a light burst's median is {median:.2} busy command buffers, Metal's {metal_median:.2}"
        );
        let (p99, metal_p99) = (ours.percentile(99), metal.percentile(99));
        let delivery = metal.delivery().expect("Metal's completions' delivery");
        assert!(
            p99 <= metal_p99 + P99_MARGIN + HAND_OFFS * delivery,
            "a light burst's 99th percentile is {p99:.2} busy command buffers, Metal's \
             {metal_p99:.2}, its completions delivered in {delivery:.2} at their 99th percentile"
        );
        assert!(
            ours.kept() >= metal.kept() - BUSY_MARGIN,
            "the busy queue keeps {:.2} of its throughput beside the light queues, Metal's {:.2}",
            ours.kept(),
            metal.kept()
        );
    }

    /// The fences Metal's side encodes around each of its encoders.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Fences {
        /// None: Metal's own storm.
        Plain,
        /// MoltenVK 1.4.1's under argument buffers and a residency set (the module docs): each
        /// encoder waits, at its start, on one device-wide fence per barrier stage, and its
        /// stage's fence is updated after it (MoltenVK does so at its submission's end; each
        /// submission here is one command buffer of one encoder). The storm's encoders are a
        /// compute one and blits; MoltenVK's vertex and fragment fences, which nothing here
        /// updates, hold none of them, so the compute and copy fences are the pattern's whole
        /// effect.
        MoltenVk,
    }

    /// The compute and copy stages' fences of [`Fences::MoltenVk`].
    struct StageFences {
        compute: Retained<ProtocolObject<dyn MTLFence>>,
        copy: Retained<ProtocolObject<dyn MTLFence>>,
    }

    pub(super) struct Metal {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queues: Vec<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
        state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        fences: Option<StageFences>,
    }

    impl Metal {
        pub(super) fn new(fences: Fences) -> Metal {
            let device = crate::metal::system_device().unwrap();
            let air = metal_assemble(&fixture("air.ll")).unwrap();
            let library = crate::metal::load_library(&device, &air, None).unwrap();
            let function = library
                .newFunctionWithName(&NSString::from_str("spin_on_an_unraised_word"))
                .expect("the busy kernel");
            // Under the compile deadline, as every Metal compilation of the crate
            // (`gpu/compile.rs`).
            let state = compile::bounded("the storm's busy kernel", || {
                device.newComputePipelineStateWithFunction_error(&function)
            })
            .unwrap();
            let queues = (0..=super::LIGHTS)
                .map(|_| device.newCommandQueue().expect("a Metal queue"))
                .collect();
            let fences = match fences {
                Fences::Plain => None,
                Fences::MoltenVk => Some(StageFences {
                    compute: device.newFence().expect("a Metal fence"),
                    copy: device.newFence().expect("a Metal fence"),
                }),
            };
            Metal {
                device,
                queues,
                state,
                fences,
            }
        }

        /// A `Shared` buffer holding `bytes`.
        fn with(&self, bytes: &[u8]) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
            let pointer =
                NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>()).ok_or("a null slice")?;
            // SAFETY: `pointer` addresses `bytes.len()` live bytes, which Metal copies.
            unsafe {
                self.device.newBufferWithBytes_length_options(
                    pointer,
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| "a Metal buffer".to_owned())
        }
    }

    impl Side for Metal {
        type Committed = (Retained<ProtocolObject<dyn MTLCommandBuffer>>, Watch);
        type Target = Retained<ProtocolObject<dyn MTLBuffer>>;

        fn name(&self) -> String {
            match self.fences {
                None => format!("Metal on {}", self.device.name()),
                Some(_) => format!("Metal under MoltenVK's fences on {}", self.device.name()),
            }
        }

        fn busy(&self, cap: u32) -> Result<Self::Committed, String> {
            let buffers = [
                self.with(&[0; 4])?,
                self.with(&[0; 4])?,
                self.with(&cap.to_le_bytes())?,
            ];
            let command_buffer = self.queues[0].commandBuffer().ok_or("a command buffer")?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or("a compute encoder")?;
            if let Some(fences) = &self.fences {
                encoder.waitForFence(&fences.compute);
                encoder.waitForFence(&fences.copy);
            }
            encoder.setComputePipelineState(&self.state);
            for (index, buffer) in buffers.iter().enumerate() {
                // SAFETY: the command buffer retains the buffer, bound at the index the kernel
                // reads, at offset 0.
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
            }
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
            if let Some(fences) = &self.fences {
                encoder.updateFence(&fences.compute);
            }
            encoder.endEncoding();
            let watched = watch(&command_buffer);
            command_buffer.commit();
            Ok((command_buffer, watched))
        }

        fn target(&self) -> Result<Self::Target, String> {
            self.device
                .newBufferWithLength_options(256, MTLResourceOptions::StorageModePrivate)
                .ok_or_else(|| "a Metal light buffer".to_owned())
        }

        fn burst(
            &self,
            light: usize,
            target: &Self::Target,
            value: u8,
        ) -> Result<Self::Committed, String> {
            let mut last = None;
            for _ in 0..BURST {
                let command_buffer = self.queues[1 + light]
                    .commandBuffer()
                    .ok_or("a command buffer")?;
                let blit = command_buffer
                    .blitCommandEncoder()
                    .ok_or("a blit encoder")?;
                if let Some(fences) = &self.fences {
                    blit.waitForFence(&fences.compute);
                    blit.waitForFence(&fences.copy);
                }
                blit.fillBuffer_range_value(target, NSRange::new(0, 256), value);
                if let Some(fences) = &self.fences {
                    blit.updateFence(&fences.copy);
                }
                blit.endEncoding();
                let watched = watch(&command_buffer);
                command_buffer.commit();
                last = Some((command_buffer, watched));
            }
            last.ok_or_else(|| "an empty burst".to_owned())
        }

        fn wait(&self, (command_buffer, watched): Self::Committed) -> Result<(), String> {
            settle_watched(&command_buffer, watched, "the queue storm")
        }

        /// Its GPU end and the wait's return in the same seconds: Metal's GPU times count the
        /// clock `host_time` reads (`CLOCK_UPTIME_RAW`, which `mach_absolute_time` counts).
        fn wait_delivered(&self, committed: Self::Committed) -> Result<Option<Duration>, String> {
            let command_buffer = committed.0.clone();
            self.wait(committed)?;
            let returned = metal2vulkan_runtime::host_time();
            let delivery = returned - command_buffer.GPUEndTime();
            Ok(Some(Duration::from_secs_f64(delivery.max(0.0))))
        }

        fn wait_timed(&self, committed: Self::Committed) -> Result<Duration, String> {
            let command_buffer = committed.0.clone();
            self.wait(committed)?;
            let window = command_buffer.GPUEndTime() - command_buffer.GPUStartTime();
            Ok(Duration::from_secs_f64(window.max(0.0)))
        }
    }
}

/// The four-queue storm on the runtime's device: every command buffer completes, the busy queue
/// runs in both phases and the light queues beside it. Where Metal runs, plain Metal's storm runs
/// first and is printed; then the runtime's storm and the reference's, their slices in turn
/// ([`interleaved`]), and, each measured at a steady spin, the runtime's light bursts wait no more
/// than the reference's by more than the margins, and its busy queue keeps its throughput as the
/// reference does: the reference is Metal's under MoltenVK's fences where the runtime's device
/// encodes them ([`moltenvk_fences`]), else plain Metal's (the module docs). Elsewhere nothing is
/// compared, and a storm whose spin changed speed in every try is printed as unsteady.
#[test]
fn four_queues_storm_as_metal_s_do() {
    #[cfg(all(m2v_gpu, target_os = "macos"))]
    {
        let plain = storm(&metal::Metal::new(metal::Fences::Plain)).unwrap();
        eprintln!("{}", plain.line("Metal"));
    }
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        // Without the validation layer (the module docs).
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options
                .layers
                .retain(|layer| layer.as_c_str() != VALIDATION_LAYER);
            at_plan(options)
        })
        .unwrap();
        let runtime = Runtime::new(device);
        // SAFETY: read while the device lives.
        let families = unsafe { runtime.device.raw() }.queue_families.to_vec();
        let fences = moltenvk_fences(&runtime.device);
        #[cfg(all(m2v_gpu, target_os = "macos"))]
        let (ours, reference) = {
            let (fenced, name) = match fences {
                true => (metal::Fences::MoltenVk, "Metal under MoltenVK's fences"),
                false => (metal::Fences::Plain, "Metal"),
            };
            let [ours, reference] = interleaved(&runtime, &metal::Metal::new(fenced)).unwrap();
            eprintln!("{}", reference.line(name));
            (ours, reference)
        };
        #[cfg(not(all(m2v_gpu, target_os = "macos")))]
        let ours = storm(&runtime).unwrap();
        eprintln!(
            "{} (queue families {families:?}; MoltenVK's barrier fences: {fences})",
            ours.line(&runtime.name())
        );
        assert!(
            ours.alone > 0.0 && ours.beside > 0.0,
            "{ours:?} (MoltenVK's barrier fences: {fences})"
        );
        #[cfg(all(m2v_gpu, target_os = "macos"))]
        metal::judge(&ours, &reference);
    }
    nothing_reported_without_the_layer(&messages, "nothing the driver said fails");
}
