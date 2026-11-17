//! The device's check of what reading a short `constant` window costs through each view a uniform
//! view can take, on a thread of its own, once per driver, release and runtime: the measured answer
//! translation takes as the description's measured workarounds
//! (`DeviceDescription::measured`), in place of any driver or GPU identity.
//!
//! - **What it measures.** The translator's view check kernels (`bind::view_check_module`): 256
//!   workgroups of 64 invocations, each reading a window 256 times, the window's first 1 KiB or
//!   all 4 KiB (a push-constant mask), through a uniform buffer, a storage buffer and a uniform
//!   texel buffer of `R32G32B32A32_UINT` texels, every invocation the same vector at each step
//!   (**uniform**), each its own (**per-lane**), or two vectors at constant offsets (**fields**).
//!   Each window lies in the device's own memory (a `Private` buffer of the runtime's), and the
//!   uniform and per-lane reads through a uniform buffer and a uniform texel buffer run again over
//!   host pages the device imported by their pointer (the memory a no-copy buffer's view reads),
//!   where the device imports them. Each arm is one dispatch between two timestamps in an
//!   internal submission waited for; a round runs every arm once, the order turned by one each
//!   round; [`ROUNDS`] rounds after one that warms the device, and each arm's answer is the median
//!   of its rounds, its timestamps' ticks times `timestampPeriod`.
//! - **Its verdicts** ([`facts_of`]): `DriverWorkaround::UniformBufferPerLaneReadsSerialize` where
//!   the storage buffer's per-lane reads took less than [`PER_LANE_COSTLY`] of the uniform
//!   buffer's at both windows. `DriverWorkaround::ExternalUniformReadsSkipACache` is judged on
//!   the reads it moves to the texel view, over the imported pages, where it is about: indexed
//!   reads, so the uniform reads, and the per-lane reads too where the per-lane fact does not
//!   hold (where it does, those keep their storage view). It holds where, of those reads at both
//!   windows, the uniform texel buffer took less than [`TEXEL_NO_SLOWER`] of the uniform
//!   buffer's time in every one (no read moved is slower) and less than [`TEXEL_FASTER`] in at
//!   least one (one is clearly faster). Both facts may hold: translation gives a slot with a
//!   per-lane read its storage view, one with indexed reads all uniform its texel view, and one
//!   read at constant offsets alone its uniform buffer (`lower::access::view_kind`). The fields
//!   arm decides nothing; the report states it. [`ViewVerdict::Unchecked`] where the device writes
//!   no timestamps the host reads or answers GPU times from the host, refuses any object or
//!   command, or its loader word was found overwritten. No driver, vendor or device identity
//!   enters it.
//! - **Its thresholds**, from bare Vulkan sweeps of the same reads: NVIDIA's storage buffer read
//!   per-lane words in 0.011 to 0.107 of the uniform buffer's time (RTX 3090, RTX 2080, RTX 5080
//!   Laptop GPU), RADV's in 0.96 to 1.00 (Radeon 780M, RX 7900 XTX): 0.3 lies about midway by
//!   ratio. ANV on the Iris Xe of RPL-P (Mesa 26.2.6; job m3-max-37780, 64 and 1024 reads an
//!   invocation), the uniform texel buffer's time over the uniform buffer's, both over host
//!   memory, worked out from each one's ratios to the uniform buffer over device memory: each
//!   invocation walking the window 0.026 to 0.058 at 1 KiB and 0.16 to 0.17 at 4 KiB, each
//!   reading its own word 0.081 to 0.084 and 0.19 to 0.23, every invocation one word 0.41 to 0.49
//!   and 0.49 to 0.55, and each workgroup walking it 0.23 to 0.32 and 0.83 to 0.86 (the uniform
//!   buffer there 1.15 to 5.56 times its own time over device memory, the texel buffer 0.99 to
//!   1.00 times its own). Where a
//!   uniform buffer keeps its cache over imported memory the two read alike: RADV on the Radeon
//!   780M (Mesa 26.2.6) read this check's uniform walks over imported pages through the texel
//!   buffer in 0.93 and 0.97 of the uniform buffer's time (job m3-max-38330). So 0.7 lies between
//!   the Iris Xe's clear wins and a tie, and 1.0, no slower, is above every Iris Xe read moved,
//!   its closest the workgroup's walk of 4 KiB.
//! - **Its cost**: 4 KiB of the device's memory for the window, 64 KiB for the words the
//!   invocations store, 64 KiB of host pages where the device imports them, eight kernels, their
//!   pipelines, layouts and sets, two texel views and a two-timestamp query pool, while it runs;
//!   the GPU time of [`ROUNDS`] + 1 rounds of its arms; and the host's wait for each arm.
//! - **When translation takes it.** Device creation never waits for it. Until it answers, the
//!   device compiles with the profile of its identity alone (a uniform view stays a uniform
//!   buffer); once it answers with a fact, every compile from then on takes the profile of the
//!   description with those facts measured (`Device::compile_device`), whose fingerprint differs,
//!   so the shader cache and the memo key what it compiles apart from what it compiled before. A
//!   pipeline made before keeps its form. A compile in progress keeps the compile device it began
//!   with (`pipeline::Sources`). An answer read from the record applies from the device's first
//!   pipeline.
//! - **Kept.** A measured answer is written as one record in the shader cache's content directory
//!   (`shader_cache/store.rs`), named by the driver's identity, the check's version and the
//!   runtime's ([`record_name`]): its medians, from which the facts are worked out again as it is
//!   read. A device of the same driver and release reads it as it is made and runs no check. An
//!   unchecked answer is kept nowhere.
//! - **Never beside the device's work.** It runs from the device's making, where a program's
//!   first work begins, and its steps are ones that work could wait on: a kernel's making holds
//!   the driver for as long as it compiles, and a timed dispatch is an internal submission, which
//!   holds the device's internal lock and its first `VkQueue` until the GPU ran it
//!   (`Device::submit_internal`). Beside the runtime lane on the RTX 5080 Laptop GPU, its kernels'
//!   making took 833 ms in all, and another queue's command buffer committed meanwhile read
//!   `Scheduled` 103.2 ms after its commit. So before each kernel's making and each timed
//!   dispatch it stands by ([`stand_by`]) until the device's work is quiet
//!   (`DeviceShared::work_quiet_for`): no batch committed waits for its submission, none
//!   submitted is in flight, and no use of a `VkQueue` has been under way for [`COMPILE_QUIET`]
//!   before a making (one cannot stop once begun) or [`ARM_QUIET`] before a dispatch, counted
//!   from no earlier than the check's start (a program's first work comes right after the
//!   device's making). A batch committed after that read and submitted before the dispatch's
//!   own submission would put the dispatch behind the program's work on the first `VkQueue`,
//!   and the internal lock the dispatch holds while it waits for the GPU with it: on the RX 7900
//!   XTX (RADV) a sparse heap's `new_texture` and a no-copy buffer's `rebind_bytes_no_copy`
//!   returned only once the program's 2 s of busy work ended (job m3-max-39371). So the
//!   dispatch is submitted only
//!   where no batch is in flight as read with its place in the first work's order held
//!   (`Device::submit_internal_if_idle`), which every batch takes before its `vkQueueSubmit`:
//!   it runs behind none of the program's work, and where one went in flight it is not
//!   submitted and stands by again. Its stand-bys take [`STAND_BY_MOST`] in all at most: a
//!   device whose work never pauses that long within it ends the check there, unchecked, with no
//!   further attempt (its next device or process checks again, as an unchecked answer is kept
//!   nowhere); until an answer translation takes no fact. It stands by for no one while a call
//!   waits for its answer (`Device::settle_view_check`), its dispatches then submitted at once.
//!   Counted: `WorkCounts::device_checks` (`view_check_stand_bys`,
//!   `view_check_stand_by_nanos`).
//! - **Out of the device's way, and its thread**, as the tiling-cost check's
//!   (`tiling_check.rs`): it steps aside before each kernel's making and each round for the
//!   device's allocations and frees on other threads (`step_aside.rs`), at once while a call waits
//!   for its answer; each kernel's making and each round's Vulkan work runs inside a section of
//!   the exit guard of its own; it holds its device through the registry of the runtime's holds
//!   (`device/holds.rs`), reading the stop before each round and as it stands by. Its dispatches
//!   are internal submissions, counted as the device's (`WorkCounts::internal_submissions`).
//! - **Over one generation.** Each generation runs a check of its own as it is made (`line.rs`;
//!   a recovery's new `VkDevice` too), whose answer its calls read. Where its generation is lost
//!   (a hang, a recovery), the check ends there, read before each round and as it stands by, and
//!   its answer is [`ViewVerdict::Unchecked`] saying so, for the lost generation's handles; it
//!   lets go of its objects and its hold of the lost `VkDevice`. A recovery carries its buffers
//!   to the next generation as it carries the runtime's own (`device/carry.rs`), so each step
//!   that reaches their Vulkan objects (the sets' making, a timed dispatch) holds their parts
//!   first (`Buffer::held_parts`) and runs only where the generation is not lost then: no handle
//!   the carry made on the next generation meets the lost one's `VkDevice`, and the lost parts
//!   stay until the step ends.

use crate::buffer::{Buffer, HeldParts};
use crate::device::holds::Holder;
use crate::device::{lock, Device, WeakDevice};
use crate::error::{internal, vulkan, Error};
use crate::gpu_time::GpuClock;
use crate::in_place::DriverIdentity;
use crate::shader_cache::store::{hex, Lookup};
use crate::tiling_check::{median, Stamps};
use crate::{exit_guard, release};
use ash::vk;
use metal2vulkan::bind::{view_check_module, CheckReads, CheckView, ViewCheckModule};
use metal2vulkan::device::{DeviceDescription, DevicePolicy, DriverWorkaround};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The measured rounds; one more warms the device first.
pub(crate) const ROUNDS: usize = 7;

/// The storage buffer's per-lane time over the uniform buffer's below which the per-lane fact
/// holds (the module docs).
pub(crate) const PER_LANE_COSTLY: f64 = 0.3;

/// The uniform texel buffer's time over the uniform buffer's, over imported pages, below which a
/// read the texel fact moves is clearly faster through it; the fact needs one (the module docs).
pub(crate) const TEXEL_FASTER: f64 = 0.7;

/// The same ratio below which a read the texel fact moves is no slower through the texel buffer;
/// the fact needs every one (the module docs).
pub(crate) const TEXEL_NO_SLOWER: f64 = 1.0;

/// The workgroups of each dispatch.
const GROUPS: u32 = 256;

/// The reads of each invocation.
const READS: u32 = 256;

/// The window's two lengths, in 16-byte vectors: 1 KiB and 4 KiB.
const WINDOWS: [u32; 2] = [64, ViewCheckModule::WINDOW_VECS];

/// The window's bytes, the longer length's.
const WINDOW_BYTES: u64 = ViewCheckModule::WINDOW_VECS as u64 * 16;

/// The host pages the imported arms read: the window at their start.
const HOST_BYTES: u64 = 1 << 16;

/// The bytes of the words the invocations store.
const OUT_BYTES: u64 = (GROUPS * ViewCheckModule::LOCAL_SIZE) as u64 * 4;

/// The kernels the arms run, by view and pattern.
const KERNELS: [(CheckView, CheckReads); 8] = [
    (CheckView::Uniform, CheckReads::Uniform),
    (CheckView::Uniform, CheckReads::PerLane),
    (CheckView::Uniform, CheckReads::Fields),
    (CheckView::Storage, CheckReads::Uniform),
    (CheckView::Storage, CheckReads::PerLane),
    (CheckView::Texel, CheckReads::Uniform),
    (CheckView::Texel, CheckReads::PerLane),
    (CheckView::Texel, CheckReads::Fields),
];

/// The kernels the imported pages' arms run: the reads the texel fact moves, through either view.
const HOST_KERNELS: [(CheckView, CheckReads); 4] = [
    (CheckView::Uniform, CheckReads::Uniform),
    (CheckView::Texel, CheckReads::Uniform),
    (CheckView::Uniform, CheckReads::PerLane),
    (CheckView::Texel, CheckReads::PerLane),
];

/// One arm: its kernel (an index of [`KERNELS`]), whether its window lies in the imported pages,
/// and its window's vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Arm {
    kernel: usize,
    host: bool,
    vecs: u32,
}

/// The arms in the order a record holds their medians: each window's device arms, then each
/// window's imported ones.
fn arms() -> Vec<Arm> {
    let device = WINDOWS.into_iter().flat_map(|vecs| {
        (0..KERNELS.len()).map(move |kernel| Arm {
            kernel,
            host: false,
            vecs,
        })
    });
    let host = WINDOWS.into_iter().flat_map(|vecs| {
        HOST_KERNELS.into_iter().map(move |pair| Arm {
            kernel: KERNELS.iter().position(|k| *k == pair).unwrap_or(0),
            host: true,
            vecs,
        })
    });
    device.chain(host).collect()
}

/// The arms' count: a record's medians.
const ARMS: usize = 2 * KERNELS.len() + 2 * HOST_KERNELS.len();

/// What the device's view check answered ([`ViewCosts::verdict`]). No Metal counterpart: Metal
/// answers no such question.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewVerdict {
    /// Not known yet: the check runs.
    Pending,
    /// Measured, in this process or read from the shader cache's record: its facts hold.
    Measured,
    /// No measurement: the device writes no timestamps the host reads, answers GPU times from the
    /// host, refused an object or command, or the run did not end. Translation takes no fact.
    /// [`Device::view_check_report`] says why.
    Unchecked,
}

/// The device's view check's answer ([`Device::view_costs`]): its verdict, the facts translation
/// takes from it, and the ratios they come from. No Metal counterpart: Metal answers no such
/// question.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ViewCosts {
    /// The verdict.
    pub verdict: ViewVerdict,
    /// The workarounds translation takes from it: `UniformBufferPerLaneReadsSerialize`,
    /// `ExternalUniformReadsSkipACache`, both or neither. Empty unless measured.
    pub facts: BTreeSet<DriverWorkaround>,
    /// The storage buffer's per-lane time over the uniform buffer's, device memory, the larger of
    /// the two windows'. `None` unless measured.
    pub per_lane: Option<f64>,
    /// The least of the uniform texel buffer's times over the uniform buffer's, imported pages,
    /// of the reads the texel fact moves at the two windows: the uniform reads, and the per-lane
    /// reads where the per-lane fact does not hold. `None` unless measured and the device imports
    /// host pages.
    pub texel_fastest: Option<f64>,
    /// The greatest of the same ratios. `None` where `texel_fastest` is.
    pub texel_slowest: Option<f64>,
}

impl ViewCosts {
    /// An answer of `verdict` with nothing measured.
    fn unmeasured(verdict: ViewVerdict) -> ViewCosts {
        ViewCosts {
            verdict,
            facts: BTreeSet::new(),
            per_lane: None,
            texel_fastest: None,
            texel_slowest: None,
        }
    }
}

/// The medians one run measured, in nanoseconds, in [`arms`]' order; 0 for an arm not run (the
/// imported ones, where the device imports no host pages).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Medians(pub(crate) [u64; ARMS]);

impl Medians {
    /// The median of the arm of `pair` over imported pages or not, at `vecs`; `None` where it was
    /// not run.
    fn of(&self, pair: (CheckView, CheckReads), host: bool, vecs: u32) -> Option<u64> {
        let kernel = KERNELS.iter().position(|k| *k == pair)?;
        let at = arms()
            .iter()
            .position(|arm| *arm == Arm { kernel, host, vecs })?;
        self.0.get(at).copied().filter(|&nanos| nanos > 0)
    }

    /// The two windows' ratios of `over`'s arm to `under`'s, `host` the memory; `None` where an
    /// arm was not run.
    fn ratios(
        &self,
        over: (CheckView, CheckReads),
        under: (CheckView, CheckReads),
        host: bool,
    ) -> Option<[f64; 2]> {
        let at = |vecs| {
            let over = self.of(over, host, vecs)?;
            let under = self.of(under, host, vecs)?;
            Some(over as f64 / under as f64)
        };
        Some([at(WINDOWS[0])?, at(WINDOWS[1])?])
    }

    /// The larger of the two windows' ratios of `over`'s arm to `under`'s, `host` the memory.
    fn ratio(
        &self,
        over: (CheckView, CheckReads),
        under: (CheckView, CheckReads),
        host: bool,
    ) -> Option<f64> {
        let [one, other] = self.ratios(over, under, host)?;
        Some(one.max(other))
    }

    /// The answer they give.
    pub(crate) fn costs(&self) -> ViewCosts {
        let per_lane = self.ratio(
            (CheckView::Storage, CheckReads::PerLane),
            (CheckView::Uniform, CheckReads::PerLane),
            false,
        );
        let lanes = per_lane.is_some_and(|r| r < PER_LANE_COSTLY);
        // The reads the texel fact moves: the uniform ones, and the per-lane ones unless the
        // per-lane fact keeps them in their storage view.
        let moved: &[CheckReads] = if lanes {
            &[CheckReads::Uniform]
        } else {
            &[CheckReads::Uniform, CheckReads::PerLane]
        };
        let texels: Option<Vec<[f64; 2]>> = (moved.iter())
            .map(|&reads| self.ratios((CheckView::Texel, reads), (CheckView::Uniform, reads), true))
            .collect();
        let texels = texels.map(|pairs| pairs.concat());
        let texel_fastest = (texels.as_deref()).map(|r| r.iter().copied().fold(f64::MAX, f64::min));
        let texel_slowest = (texels.as_deref()).map(|r| r.iter().copied().fold(0f64, f64::max));
        ViewCosts {
            verdict: ViewVerdict::Measured,
            facts: facts_of(per_lane, texel_fastest, texel_slowest),
            per_lane,
            texel_fastest,
            texel_slowest,
        }
    }
}

/// The facts of a run whose per-lane ratio is `per_lane` and the least and greatest texel ratios
/// of the reads the texel fact moves `texel_fastest` and `texel_slowest` (the module docs): each
/// where its ratios were measured and lie within their bounds.
pub(crate) fn facts_of(
    per_lane: Option<f64>,
    texel_fastest: Option<f64>,
    texel_slowest: Option<f64>,
) -> BTreeSet<DriverWorkaround> {
    let mut facts = BTreeSet::new();
    if per_lane.is_some_and(|r| r < PER_LANE_COSTLY) {
        facts.insert(DriverWorkaround::UniformBufferPerLaneReadsSerialize);
    }
    if texel_fastest.is_some_and(|r| r < TEXEL_FASTER)
        && texel_slowest.is_some_and(|r| r < TEXEL_NO_SLOWER)
    {
        facts.insert(DriverWorkaround::ExternalUniformReadsSkipACache);
    }
    facts
}

/// A device's view check (`line::Invariant::views`): whether it started, its answer once known
/// with how it came and its medians, the calls waiting for it, and the compile device of the
/// measured facts once they are known.
pub(crate) struct ViewCheck {
    started: AtomicBool,
    settled: Mutex<Option<(ViewCosts, Option<Medians>, String)>>,
    wake: Condvar,
    waiters: AtomicU32,
    /// The description and policy the device's compile device was made of, which the measured
    /// one is made of with its facts.
    base: (DeviceDescription, DevicePolicy),
    /// The compile device of the measured facts, set once where the answer holds any.
    measured: OnceLock<metal2vulkan::device::Device>,
    /// In the runtime's own device tests, the most its stand-bys take in all, in nanoseconds,
    /// where not 0 ([`STAND_BY_MOST`] where 0).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    most: std::sync::atomic::AtomicU64,
    /// In the runtime's own device tests, the kernels' makings begun, each counted as its
    /// stand-by ends ([`kernels_begun`](Self::kernels_begun)).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    begun: AtomicU32,
}

impl ViewCheck {
    /// A device's, not started, its compile device made of `description` under `policy`.
    pub(crate) fn new(description: DeviceDescription, policy: DevicePolicy) -> ViewCheck {
        ViewCheck {
            started: AtomicBool::new(false),
            settled: Mutex::new(None),
            wake: Condvar::new(),
            waiters: AtomicU32::new(0),
            base: (description, policy),
            measured: OnceLock::new(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            most: std::sync::atomic::AtomicU64::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            begun: AtomicU32::new(0),
        }
    }

    /// The most its stand-bys take in all before it ends unchecked: [`STAND_BY_MOST`], or, in the
    /// runtime's own device tests, the bound a test set (`stand_by_at_most`).
    fn stand_by_most(&self) -> Duration {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if let set @ 1.. = self.most.load(Ordering::Acquire) {
            return Duration::from_nanos(set);
        }
        STAND_BY_MOST
    }

    /// In the runtime's own device tests, the kernels' makings begun so far: counted as each one's
    /// stand-by ends, before its translation and its pipeline's making, which may end long after
    /// on a loaded host (`WorkCounts::compute_pipeline_create_nanos` counts a making as it ends).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn kernels_begun(&self) -> u32 {
        self.begun.load(Ordering::Acquire)
    }

    /// In the runtime's own device tests, bound its stand-bys to `most` in all, read at each look.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn stand_by_at_most(&self, most: Duration) {
        let nanos = u64::try_from(most.as_nanos()).unwrap_or(u64::MAX).max(1);
        self.most.store(nanos, Ordering::Release);
    }

    /// The compile device of the measured facts, once the answer holds any.
    pub(crate) fn measured(&self) -> Option<&metal2vulkan::device::Device> {
        self.measured.get()
    }

    /// Note the check started; whether it had not been before.
    fn start(&self) -> bool {
        !self.started.swap(true, Ordering::AcqRel)
    }

    /// The answer now: pending until it is known.
    fn costs(&self) -> ViewCosts {
        (lock(&self.settled).as_ref()).map_or_else(
            || ViewCosts::unmeasured(ViewVerdict::Pending),
            |(costs, ..)| costs.clone(),
        )
    }

    /// Settle the answer, `how` saying how it came: where it holds facts, the measured compile
    /// device is made first, so every compile after the answer takes it.
    pub(crate) fn settle(&self, costs: ViewCosts, medians: Option<Medians>, how: String) {
        let mut how = how;
        if !costs.facts.is_empty() {
            let (description, policy) = &self.base;
            let mut description = description.clone();
            description.measured.extend(costs.facts.iter().copied());
            match metal2vulkan::device::Device::new(description, *policy) {
                Ok(device) => {
                    let _ = self.measured.set(device);
                }
                Err(error) => how = format!("{how}; its compile device was refused: {error}"),
            }
        }
        *lock(&self.settled) = Some((costs, medians, how));
        self.wake.notify_all();
    }

    /// Whether a call waits for the answer.
    fn awaited(&self) -> bool {
        self.waiters.load(Ordering::Acquire) > 0
    }

    /// Wait until the answer is known, or `timeout` passed: whether it is.
    fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        self.waiters.fetch_add(1, Ordering::AcqRel);
        let _waiting = Waiting(&self.waiters);
        let mut settled = lock(&self.settled);
        while settled.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            settled = match self.wake.wait_timeout(settled, left) {
                Ok((settled, _)) => settled,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        true
    }

    /// The answer in words: its verdict and facts, the two ratios, and every arm's median.
    fn report(&self) -> String {
        let settled = lock(&self.settled);
        let Some((costs, medians, how)) = settled.as_ref() else {
            return "verdict Pending".to_owned();
        };
        let mut out = format!("verdict {:?}, facts {:?}", costs.verdict, costs.facts);
        let ratio = |r: Option<f64>| r.map_or("none".to_owned(), |r| format!("{r:.3}"));
        if let Some(medians) = medians {
            out += &format!(
                "; storage/uniform per-lane {} (costly below {PER_LANE_COSTLY}), texel/uniform \
                 over imported pages of the reads it moves fastest {} (faster below \
                 {TEXEL_FASTER}) and slowest {} (no slower below {TEXEL_NO_SLOWER:.1}), \
                 texel/uniform fields {}; medians of {ROUNDS} rounds in us:",
                ratio(costs.per_lane),
                ratio(costs.texel_fastest),
                ratio(costs.texel_slowest),
                ratio(medians.ratio(
                    (CheckView::Texel, CheckReads::Fields),
                    (CheckView::Uniform, CheckReads::Fields),
                    false
                )),
            );
            for (arm, nanos) in arms().iter().zip(medians.0) {
                let (view, reads) = KERNELS[arm.kernel];
                let memory = if arm.host { "imported" } else { "device" };
                out += &format!(
                    " {view:?}/{reads:?}/{memory}/{}KiB {:.1}",
                    arm.vecs * 16 / 1024,
                    nanos as f64 / 1e3
                );
            }
        }
        out + "; " + how
    }
}

/// A call waiting for the answer ([`ViewCheck::wait`]), counted until it returns.
struct Waiting<'a>(&'a AtomicU32);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Device {
    /// No Metal counterpart (Metal answers no such question): wait until the device's view check
    /// answered or `timeout` passed: whether it did. The check starts as the device is made, on
    /// a thread of its own, and no call but this one waits for it; its answer is kept in the
    /// shader cache's directory for the driver, its release and the runtime's, so a later device
    /// of the same reads it at once. It measures what reading a short `constant` window costs
    /// through a uniform buffer, a storage buffer and a uniform texel buffer, and translation
    /// takes its facts from then on ([`Device::view_costs`]).
    pub fn settle_view_check(&self, timeout: Duration) -> bool {
        self.current().view_check().wait(timeout)
    }

    /// No Metal counterpart: the device's view check's answer now ([`Device::settle_view_check`]):
    /// [`ViewVerdict::Pending`] with no fact until it is known.
    pub fn view_costs(&self) -> ViewCosts {
        self.current().view_check().costs()
    }

    /// No Metal counterpart: the device's view check's answer in words: its verdict and facts, the
    /// ratios they come from, every arm's median, and whether it ran in this process (how long)
    /// or was read from the shader cache's record; why, where it is unchecked.
    pub fn view_check_report(&self) -> String {
        self.current().view_check().report()
    }

    /// As the device is made: the check's answer from the record where one holds it, else its
    /// thread. Where the thread cannot start, the answer is unchecked. In the runtime's own
    /// tests a device made with `DeviceOptions::view_facts` takes those facts and runs no check.
    pub(crate) fn recall_view_check(&self, forced: Option<BTreeSet<DriverWorkaround>>) {
        let check = self.view_check();
        if !check.start() {
            return;
        }
        if let Some(facts) = forced {
            let costs = ViewCosts {
                facts,
                ..ViewCosts::unmeasured(ViewVerdict::Measured)
            };
            check.settle(costs, None, "fixed by the test's options".to_owned());
            return;
        }
        let shared = self.shared();
        let identity = self.driver_identity();
        let looked = (shared.shader_cache.content_store())
            .load_content(&record_name(&identity), |bytes| decode(&identity, &bytes));
        if let Lookup::Hit(medians) = looked {
            let how = "read from the shader cache's record".to_owned();
            check.settle(medians.costs(), Some(medians), how);
            return;
        }
        let weak = self.downgrade();
        let builder = std::thread::Builder::new().name("m2v-view-check".to_owned());
        if exit_guard::spawn(builder, move || run(weak)).is_err() {
            let unchecked = ViewCosts::unmeasured(ViewVerdict::Unchecked);
            check.settle(unchecked, None, "its thread did not start".to_owned());
        }
    }
}

/// The check's thread: the run, then its answer settled and, where measured, recorded.
fn run(weak: WeakDevice) {
    crate::in_place::mark_checking();
    crate::step_aside::mark_a_check_s_thread(crate::step_aside::Counted::Apart);
    let began = Instant::now();
    let Some(device) = weak.hold() else {
        return;
    };
    let Some(measured) = measure(&device, began) else {
        // The generation was lost under the check: it answers unchecked, so no call waits for an
        // answer that never comes (the module docs).
        if device.shared().lost() {
            let unchecked = ViewCosts::unmeasured(ViewVerdict::Unchecked);
            let why = "the device was lost during the check".to_owned();
            device.view_check().settle(unchecked, None, why);
        }
        // Else the process is exiting, or the program let the device go: nothing is kept.
        return;
    };
    let check = device.view_check();
    match measured {
        Ok(medians) => {
            let store = device.shared().shader_cache.content_store();
            if store.has_directory() {
                let identity = device.driver_identity();
                let _ =
                    store.write_record_soon(&record_name(&identity), encode(&identity, &medians));
            }
            let took = began.elapsed().as_secs_f64() * 1e3;
            let how = format!("ran {took:.1} ms in this process");
            check.settle(medians.costs(), Some(medians), how);
        }
        Err(why) => {
            let unchecked = ViewCosts::unmeasured(ViewVerdict::Unchecked);
            check.settle(unchecked, None, why);
        }
    }
}

/// The run's medians, the check begun at `began`; why not, where it measured none; `None` where
/// the process is exiting, the program's last drop stopped the device's holds, or the generation
/// was lost.
fn measure(device: &Device, began: Instant) -> Option<Result<Medians, String>> {
    let shared = device.shared();
    let clock = match (shared.timing.clock(), shared.timing.answers_from_host()) {
        (Some(clock), false) => *clock,
        _ => {
            let why = "the device writes no timestamps the host reads, or answers GPU times from \
                       the host";
            return Some(Err(why.to_owned()));
        }
    };
    if word_overwritten(device) {
        let why = "the device's loader word was found overwritten (the driver names the device in \
                   its messages)";
        return Some(Err(why.to_owned()));
    }
    let mut standing = Standing::default();
    if let Err(why) = stand_by(device, (COMPILE_QUIET, began, false), &mut standing)? {
        return Some(Err(why));
    }
    let made = {
        let _busy = exit_guard::enter()?;
        Kernels::new(device)
    };
    let mut kernels = match made {
        Ok(kernels) => kernels,
        Err(why) => return Some(Err(why)),
    };
    for at in 0..KERNELS.len() {
        step_aside(device);
        if let Err(why) = stand_by(device, (COMPILE_QUIET, began, false), &mut standing)? {
            return Some(Err(why));
        }
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        device.view_check().begun.fetch_add(1, Ordering::AcqRel);
        let _busy = exit_guard::enter()?;
        if let Err(error) = kernels.make_kernel(at) {
            return Some(Err(format!("its kernels were not made: {error}")));
        }
    }
    let sets = {
        let _busy = exit_guard::enter()?;
        kernels.make_sets()
    };
    if let Err(error) = sets {
        return Some(Err(format!("its kernels' sets were not made: {error}")));
    }
    let measured = rounds(device, &clock, &kernels, (began, &mut standing));
    let busy = exit_guard::enter();
    drop(kernels);
    drop(busy);
    measured
}

/// The rounds of every arm over `kernels`, the check begun at `began`, its stand-bys so far
/// `standing`, as [`measure`] answers them.
fn rounds(
    device: &Device,
    clock: &GpuClock,
    kernels: &Kernels,
    (began, standing): (Instant, &mut Standing),
) -> Option<Result<Medians, String>> {
    let stamps = {
        let _busy = exit_guard::enter()?;
        Stamps::new(device)
    };
    let stamps = match stamps {
        Ok(stamps) => stamps,
        Err(error) => return Some(Err(format!("its query pool was not made: {error}"))),
    };
    let arms: Vec<(usize, Arm)> = (arms().into_iter().enumerate())
        .filter(|(_, arm)| !arm.host || kernels.host.is_some())
        .collect();
    let mut times: Vec<Vec<u64>> = vec![Vec::new(); ARMS];
    // When the check's own last dispatch ended: no use of a queue since is the device's.
    let mut own_end = None;
    for round in 0..=ROUNDS {
        if !device.holds_go_on(Holder::ViewCheck) {
            return None;
        }
        step_aside(device);
        if word_overwritten(device) {
            let why = "the device's loader word was found overwritten during its rounds";
            return Some(Err(why.to_owned()));
        }
        for turn in 0..arms.len() {
            let (at, arm) = arms[(turn + round) % arms.len()];
            // Its dispatch goes only where no batch is in flight as it is submitted, and stands by
            // again where one went in flight since its stand-by read the work quiet; at once while
            // a call waits for the answer (the module docs).
            let timed = loop {
                let after = (ARM_QUIET, own_end.unwrap_or(began), own_end.is_some());
                if let Err(why) = stand_by(device, after, standing)? {
                    return Some(Err(why));
                }
                let _busy = exit_guard::enter()?;
                let _held = kernels.held()?;
                let record = |raw: &ash::Device, command_buffer: vk::CommandBuffer| {
                    // SAFETY: the internal command buffer is recording outside a render pass and
                    // only this records it; the kernels' objects live until they are dropped,
                    // after the rounds, and every submission is waited for.
                    unsafe { kernels.record(raw, command_buffer, arm) }
                };
                let timed = match device.view_check().awaited() {
                    true => stamps.time(device, clock, record).map(Some),
                    false => stamps.time_if_idle(device, clock, record),
                };
                match timed {
                    Ok(None) => standing.missed += 1,
                    Ok(Some(nanos)) => break Ok(nanos),
                    Err(error) => break Err(error),
                }
            };
            own_end = Some(Instant::now());
            match timed {
                Ok(nanos) if round > 0 => times[at].push(nanos),
                Ok(_) => {}
                Err(error) => return Some(Err(format!("its {arm:?} round failed: {error}"))),
            }
        }
    }
    let mut medians = [0u64; ARMS];
    for (at, times) in times.iter_mut().enumerate() {
        medians[at] = median(times);
    }
    Some(Ok(Medians(medians)))
}

/// How long no use of the device's `VkQueue`s may have been under way before a kernel's making,
/// which cannot stop once begun: as long as the device's allocations and frees must have been
/// quiet before a check's step (`step_aside::QUIET`).
const COMPILE_QUIET: Duration = crate::step_aside::QUIET;

/// The same before a timed dispatch: the gap that ends a burst of submissions
/// (`queue_quiet::SETTLE`).
const ARM_QUIET: Duration = crate::device::queue_quiet::SETTLE;

/// How often a stand-by looks again.
const LOOK_EVERY: Duration = Duration::from_micros(250);

/// The most the check's stand-bys take in all before it ends unchecked (the module docs).
const STAND_BY_MOST: Duration = Duration::from_secs(10);

/// The check's stand-bys over its run ([`stand_by`]): how long they took in all, and the timed
/// dispatches that found a batch in flight as they were submitted, and stood by again.
#[derive(Default)]
struct Standing {
    stood: Duration,
    missed: u32,
}

/// Before a step the device's work could wait on, stand by until that work is quiet (the module
/// docs): no batch committed waits for its submission, none submitted is in flight, and no use of
/// a `VkQueue` has been under way for `quiet` since `since` (the check's start, or the end of its
/// own last dispatch), or, where `own` (`since` ended the check's own dispatch), since then, where
/// that was the last use. A device that has used no `VkQueue` yet is quiet only from the check's
/// start: its program's first work comes right after its making. Not while a call waits for the
/// answer. `None` where the program's last drop stops the device's holds, or where the generation
/// was lost (the module docs); why the check ends unchecked, where its stand-bys, `standing` so
/// far, reached [`STAND_BY_MOST`] in all. Counted where it waited.
fn stand_by(
    device: &Device,
    (quiet, since, own): (Duration, Instant, bool),
    standing: &mut Standing,
) -> Option<Result<(), String>> {
    let shared = device.shared();
    let check = device.view_check();
    let began = Instant::now();
    let mut stood = false;
    loop {
        if shared.holds.stopping() || shared.lost() {
            return None;
        }
        if check.awaited() {
            break;
        }
        let pause = match shared.work_quiet_for() {
            Some(unused) if own && unused + LOOK_EVERY >= since.elapsed() => break,
            Some(unused) => match quiet.checked_sub(unused.min(since.elapsed())) {
                None | Some(Duration::ZERO) => break,
                Some(left) => left,
            },
            None => LOOK_EVERY,
        };
        let most = check.stand_by_most();
        if standing.stood + began.elapsed() >= most {
            let waited = began.elapsed();
            shared.counts.view_check_stood_by(waited);
            standing.stood += waited;
            let why = format!(
                "the device's work never paused for it: it stood by {:.1} s in all, its most \
                 ({} timed dispatches found a batch in flight as they were submitted)",
                standing.stood.as_secs_f64(),
                standing.missed
            );
            return Some(Err(why));
        }
        stood = true;
        std::thread::sleep(pause.min(LOOK_EVERY));
    }
    if stood {
        let waited = began.elapsed();
        shared.counts.view_check_stood_by(waited);
        standing.stood += waited;
    }
    Some(Ok(()))
}

/// Before a kernel's making or a round, step aside for the device's allocations and frees on
/// other threads (`step_aside.rs`): not while the program's last drop stops the check or a call
/// waits for its answer. Counted where it waited.
fn step_aside(device: &Device) {
    let shared = device.shared();
    let activity = shared.allocations.allocations.activity();
    let stop = || shared.holds.stopping() || device.view_check().awaited();
    if let Some(waited) = activity.step_aside(stop) {
        shared.counts.check_stepped_aside(waited);
    }
}

/// Whether the runtime found `device`'s loader word overwritten (`loader_word.rs`).
fn word_overwritten(device: &Device) -> bool {
    // SAFETY: the device is live while `device` is.
    unsafe { device.shared().loader_word.overwritten() }
}

/// The descriptor type of the window through `view`.
fn descriptor_type(view: CheckView) -> Result<vk::DescriptorType, String> {
    match view {
        CheckView::Uniform => Ok(vk::DescriptorType::UNIFORM_BUFFER),
        CheckView::Storage => Ok(vk::DescriptorType::STORAGE_BUFFER),
        CheckView::Texel => Ok(vk::DescriptorType::UNIFORM_TEXEL_BUFFER),
        other => Err(format!("no descriptor for the view {other:?}")),
    }
}

/// The parts of the check's buffers on its generation, held for a step that reaches their Vulkan
/// objects ([`Held::of`]). Its fields go in the order they are declared, the reverse of their
/// taking, as each hold's scope of this thread's reads of parts nests in the one before it
/// (`NoticesFor`).
struct Held<'a> {
    _host: Option<HeldParts<'a>>,
    _out: HeldParts<'a>,
    _window: HeldParts<'a>,
}

impl<'a> Held<'a> {
    /// The parts of `window`, `out` and `host`'s buffer, buffers of `device`'s generation, held:
    /// `None` where that generation was lost, as read once all are held. A recovery carries them
    /// to the next generation only after the loss (`device/carry.rs`), so where it was not lost
    /// they are this generation's, and a recovery that begins meanwhile lets them go only once
    /// the hold ends (`Buffer::held_parts`); where it was, their parts may be the next
    /// generation's, which no call of this generation's `VkDevice` may meet.
    fn of(
        device: &Device,
        window: &'a Buffer,
        out: &'a Buffer,
        host: Option<&'a crate::in_place_check::ClassPages>,
    ) -> Option<Held<'a>> {
        let window = window.held_parts();
        let out = out.held_parts();
        let host = host.map(|pages| pages.buffer.held_parts());
        let held = Held {
            _host: host,
            _out: out,
            _window: window,
        };
        (!device.shared().lost()).then_some(held)
    }
}

/// The check's objects: the window and out buffers, the imported pages where the device imports
/// them, the kernels' pipelines, and one set for each view and memory. Released as they go.
struct Kernels {
    device: Device,
    window: Buffer,
    out: Buffer,
    /// The imported pages, where the device imports them.
    host: Option<crate::in_place_check::ClassPages>,
    modules: Vec<vk::ShaderModule>,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    layouts: Vec<vk::PipelineLayout>,
    pipelines: Vec<vk::Pipeline>,
    pool: vk::DescriptorPool,
    views: Vec<vk::BufferView>,
    /// The set of each kernel's view over the device window and over the imported pages.
    sets: Vec<(vk::DescriptorSet, vk::DescriptorSet)>,
}

impl Kernels {
    /// Its buffers' parts, held for a timed dispatch ([`Held::of`]); `None` where its generation
    /// was lost.
    fn held(&self) -> Option<Held<'_>> {
        Held::of(&self.device, &self.window, &self.out, self.host.as_ref())
    }

    /// The objects on `device`, inside a section of the exit guard; why not, where it refuses one.
    fn new(device: &Device) -> Result<Kernels, String> {
        let window = (device.scratch_buffer(WINDOW_BYTES))
            .map_err(|error| format!("its window was refused: {error}"))?;
        let out = (device.scratch_buffer(OUT_BYTES))
            .map_err(|error| format!("its out buffer was refused: {error}"))?;
        let host = crate::in_place_check::pointer_pages(device, HOST_BYTES).ok();
        Ok(Kernels {
            device: device.clone(),
            window,
            out,
            host,
            modules: Vec::new(),
            set_layouts: Vec::new(),
            layouts: Vec::new(),
            pipelines: Vec::new(),
            pool: vk::DescriptorPool::null(),
            views: Vec::new(),
            sets: Vec::new(),
        })
    }

    /// Make the module, layouts and pipeline of the kernel at `at` of [`KERNELS`], the kernels
    /// before it made; inside a section of the exit guard. Those made before a failure go with the
    /// drop.
    fn make_kernel(&mut self, at: usize) -> Result<(), Error> {
        let raw: &ash::Device = self.device.shared().vk();
        let compile = self.device.compile_device();
        let (view, reads) = KERNELS[at];
        let module =
            view_check_module(compile, view, reads).map_err(crate::device::device_error)?;
        let info = vk::ShaderModuleCreateInfo::default().code(&module.spirv);
        // SAFETY: the code is live SPIR-V words for the call.
        let made = unsafe { raw.create_shader_module(&info, None) }.map_err(vulkan)?;
        self.modules.push(made);
        let ty = descriptor_type(view).map_err(|why| internal(&why))?;
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(ViewCheckModule::WINDOW_BINDING)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(ViewCheckModule::OUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `info` and its array are live for the call.
        let set_layout =
            unsafe { raw.create_descriptor_set_layout(&info, None) }.map_err(vulkan)?;
        self.set_layouts.push(set_layout);
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(ViewCheckModule::PUSH_BYTES)];
        let set_layouts = [set_layout];
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: `info` and its arrays are live for the call.
        let layout = unsafe { raw.create_pipeline_layout(&info, None) }.map_err(vulkan)?;
        self.layouts.push(layout);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(made)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: every pointer in `info` outlives the call; no pipeline cache.
        let pipelines =
            unsafe { raw.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        let pipelines = pipelines.map_err(|(_, error)| vulkan(error))?;
        let pipeline =
            (pipelines.first().copied()).ok_or_else(|| internal("the driver made no pipeline"))?;
        self.pipelines.push(pipeline);
        (self.device.shared()).name_pipeline(pipeline, || {
            format!("metal2vulkan: view check {view:?} {reads:?}")
        });
        Ok(())
    }

    /// Make the descriptor pool, the texel views and every kernel's two sets, every kernel made;
    /// inside a section of the exit guard. Those made before a failure go with the drop. Refused
    /// (`DeviceLost`) where its generation was lost ([`Held::of`]).
    fn make_sets(&mut self) -> Result<(), Error> {
        // The buffers' parts, held while their handles go into the views and sets ([`Held::of`]).
        let Some(_held) = Held::of(&self.device, &self.window, &self.out, self.host.as_ref())
        else {
            return Err(Error::DeviceLost);
        };
        let raw: &ash::Device = self.device.shared().vk();
        let sets = 2 * KERNELS.len() as u32;
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: sets,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2 * sets,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                descriptor_count: sets,
            },
        ];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(sets)
            .pool_sizes(&sizes);
        // SAFETY: `info` and its array are live for the call.
        self.pool = unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
        let device_window = self.window.at(0);
        let host_window = self.host.as_ref().map(|pages| pages.buffer.at(0));
        let out = self.out.at(0);
        // Every descriptor offset Vulkan bounds is at most 256 bytes' alignment
        // (`minUniformBufferOffsetAlignment`, `minStorageBufferOffsetAlignment`,
        // `minTexelBufferOffsetAlignment`).
        let placed = [Some(device_window), host_window, Some(out)];
        if placed
            .into_iter()
            .flatten()
            .any(|(_, at)| !at.is_multiple_of(256))
        {
            return Err(internal("a window lies off 256 bytes in its VkBuffer"));
        }
        let texel_view = |(buffer, offset): (vk::Buffer, u64)| {
            let info = vk::BufferViewCreateInfo::default()
                .buffer(buffer)
                .format(vk::Format::R32G32B32A32_UINT)
                .offset(offset)
                .range(WINDOW_BYTES);
            // SAFETY: the buffer is live, made with the uniform texel buffer usage.
            unsafe { raw.create_buffer_view(&info, None) }.map_err(vulkan)
        };
        let device_view = texel_view(device_window)?;
        self.views.push(device_view);
        let host_view = match host_window {
            Some(window) => {
                let view = texel_view(window)?;
                self.views.push(view);
                Some(view)
            }
            None => None,
        };
        for (at, (view, _)) in KERNELS.iter().enumerate() {
            let layouts = [self.set_layouts[at]; 2];
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&layouts);
            // SAFETY: the pool holds the sets' descriptors (sized above).
            let made = unsafe { raw.allocate_descriptor_sets(&info) }.map_err(vulkan)?;
            let (on_device, on_host) = (made[0], made[1]);
            self.sets.push((on_device, on_host));
            let ty = descriptor_type(*view).map_err(|why| internal(&why))?;
            let windows = [
                (on_device, Some(device_window), Some(device_view)),
                (on_host, host_window, host_view),
            ];
            for (set, window, texels) in windows {
                let Some((buffer, offset)) = window else {
                    continue;
                };
                let infos = [vk::DescriptorBufferInfo {
                    buffer,
                    offset,
                    range: WINDOW_BYTES,
                }];
                let outs = [vk::DescriptorBufferInfo {
                    buffer: out.0,
                    offset: out.1,
                    range: OUT_BYTES,
                }];
                let texel_views: Vec<vk::BufferView> = texels.into_iter().collect();
                let mut window_write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(ViewCheckModule::WINDOW_BINDING)
                    .descriptor_type(ty);
                window_write = match ty {
                    vk::DescriptorType::UNIFORM_TEXEL_BUFFER => {
                        window_write.texel_buffer_view(&texel_views)
                    }
                    _ => window_write.buffer_info(&infos),
                };
                let writes = [
                    window_write,
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(ViewCheckModule::OUT_BINDING)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&outs),
                ];
                // SAFETY: the set, buffers and view are live; no command uses the set yet.
                unsafe { raw.update_descriptor_sets(&writes, &[]) };
            }
        }
        Ok(())
    }

    /// Record `arm`'s dispatch into `command_buffer`.
    ///
    /// # Safety
    ///
    /// The command buffer is recording outside a render pass, on this device.
    unsafe fn record(&self, raw: &ash::Device, command_buffer: vk::CommandBuffer, arm: Arm) {
        let (on_device, on_host) = self.sets[arm.kernel];
        let set = if arm.host { on_host } else { on_device };
        let push = ViewCheckModule::push(arm.vecs, READS);
        let bind = vk::PipelineBindPoint::COMPUTE;
        // SAFETY: the caller's; the pipeline, layout and set are live and match.
        unsafe {
            raw.cmd_bind_pipeline(command_buffer, bind, self.pipelines[arm.kernel]);
            let layout = self.layouts[arm.kernel];
            raw.cmd_bind_descriptor_sets(command_buffer, bind, layout, 0, &[set], &[]);
            let stages = vk::ShaderStageFlags::COMPUTE;
            raw.cmd_push_constants(command_buffer, layout, stages, 0, &push);
            raw.cmd_dispatch(command_buffer, GROUPS, 1, 1);
        }
    }
}

impl Drop for Kernels {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard.
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw: &ash::Device = self.device.shared().vk();
        // SAFETY: every submission that used the objects completed (each was waited for).
        unsafe {
            for &pipeline in &self.pipelines {
                release::object(raw, pipeline);
            }
            for &layout in &self.layouts {
                release::object(raw, layout);
            }
            for &set_layout in &self.set_layouts {
                release::object(raw, set_layout);
            }
            for &module in &self.modules {
                release::object(raw, module);
            }
            for &view in &self.views {
                release::object(raw, view);
            }
            if self.pool != vk::DescriptorPool::null() {
                release::object(raw, self.pool);
            }
        }
    }
}

/// The version of what the check measures and of how its record is named and framed: a change to
/// either bumps it, and a record of another version is never read.
pub(crate) const CHECK_VERSION: u32 = 2;

/// A record's first bytes.
const MAGIC: [u8; 8] = *b"M2VVIEW\0";

/// The domain tag of a record's name.
const NAME_TAG: &[u8] = b"metal2vulkan view check record\0";

/// What the check measures, in a record's name.
const CHECK_KIND: &[u8] = b"constant window views\0";

/// A record's length: magic, version, runtime, identity, the medians, checksum.
const RECORD_BYTES: usize = 8 + 4 + 4 + 32 + ARMS * 8 + 32;

/// The digest of `identity` that names its record and that the record holds again.
fn digest(identity: &DriverIdentity) -> [u8; 32] {
    identity.digest_for(NAME_TAG, CHECK_KIND, CHECK_VERSION)
}

/// The name of the record of `identity`'s answer in the shader cache's content directory.
pub(crate) fn record_name(identity: &DriverIdentity) -> String {
    hex(&digest(identity))
}

/// The bytes of a record of `medians` for `identity`: magic `"M2VVIEW\0"`, `CHECK_VERSION`,
/// `RUNTIME_VERSION` (each `u32`), the identity's digest for this check (32 bytes), each arm's
/// median in nanoseconds in [`arms`]' order (`u64`), and the SHA-256 of every byte before it.
/// Integers are little-endian.
pub(crate) fn encode(identity: &DriverIdentity, medians: &Medians) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_BYTES);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&CHECK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&metal2vulkan::cache::RUNTIME_VERSION.to_le_bytes());
    bytes.extend_from_slice(&digest(identity));
    for nanos in medians.0 {
        bytes.extend_from_slice(&nanos.to_le_bytes());
    }
    let checksum: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// The medians `bytes` record for `identity`: `None` for anything but a whole record of this
/// version and runtime, for this identity, whose checksum holds and whose device arms are not zero.
pub(crate) fn decode(identity: &DriverIdentity, bytes: &[u8]) -> Option<Medians> {
    if bytes.len() != RECORD_BYTES {
        return None;
    }
    let (body, checksum) = bytes.split_at(RECORD_BYTES - 32);
    let computed: [u8; 32] = Sha256::digest(body).into();
    if computed[..] != *checksum {
        return None;
    }
    let (magic, rest) = body.split_at(8);
    let (version, rest) = rest.split_at(4);
    let (runtime, rest) = rest.split_at(4);
    let (identity_digest, rest) = rest.split_at(32);
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap_or([0xff; 4]));
    if magic != MAGIC
        || word(version) != CHECK_VERSION
        || word(runtime) != metal2vulkan::cache::RUNTIME_VERSION
        || identity_digest != digest(identity)
    {
        return None;
    }
    let mut medians = [0u64; ARMS];
    for (at, median) in medians.iter_mut().enumerate() {
        *median = u64::from_le_bytes(rest.get(at * 8..at * 8 + 8)?.try_into().ok()?);
    }
    let device_arms = 2 * KERNELS.len();
    (medians[..device_arms].iter().all(|&nanos| nanos > 0)).then_some(Medians(medians))
}

#[cfg(test)]
mod tests;
