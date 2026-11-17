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
//!   uniform reads through a uniform buffer and a uniform texel buffer again over host pages the
//!   device imported by their pointer (the memory a no-copy buffer's view reads), where the device
//!   imports them. Each arm is one dispatch between two timestamps in an internal submission
//!   waited for; a round runs every arm once, the order turned by one each round; [`ROUNDS`]
//!   rounds after one that warms the device, and each arm's answer is the median of its rounds,
//!   its timestamps' ticks times `timestampPeriod`.
//! - **Its verdicts** ([`facts_of`]): `DriverWorkaround::UniformBufferPerLaneReadsSerialize` where
//!   the storage buffer's per-lane reads took less than [`PER_LANE_COSTLY`] of the uniform
//!   buffer's at both windows; `DriverWorkaround::ExternalUniformReadsSkipACache` where the
//!   uniform texel buffer's uniform reads over the imported pages took less than
//!   [`TEXEL_FASTER`] of the uniform buffer's there at both windows. Both may hold: translation
//!   gives a slot with a per-lane read its storage view, one with indexed reads all uniform its
//!   texel view, and one read at constant offsets alone its uniform buffer
//!   (`lower::access::view_kind`). The fields arm decides nothing; the report states it.
//!   [`ViewVerdict::Unchecked`] where the device writes no timestamps the host reads or answers
//!   GPU times from the host, refuses any object or command, or its loader word was found
//!   overwritten. No driver, vendor or device identity enters it.
//! - **Its thresholds**, from bare Vulkan sweeps of the same reads: NVIDIA's storage buffer read
//!   per-lane words in 0.011 to 0.107 of the uniform buffer's time (RTX 3090, RTX 2080, RTX 5080
//!   Laptop GPU), RADV's in 0.96 to 1.00 (Radeon 780M, RX 7900 XTX): 0.3 lies about midway by
//!   ratio. ANV's uniform texel buffer on the Iris Xe of RPL-P read one word for every invocation
//!   over imported memory in 0.40 to 0.49 of the uniform buffer's time there (the uniform buffer
//!   2.0 to 2.5 times its own over device memory, the texel buffer 0.99 to 1.00), where a driver
//!   whose uniform buffer keeps its cache over imported memory reads both alike: 0.7 lies between.
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
//! - **Out of the device's way, and its thread**, as the tiling-cost check's
//!   (`tiling_check.rs`): it steps aside before each round for the device's allocations and frees
//!   on other threads (`step_aside.rs`), at once while a call waits for its answer; each round's
//!   Vulkan work runs inside a section of the exit guard; it holds its device through the registry
//!   of the runtime's holds (`device/holds.rs`), reading the stop before each round; and its
//!   submissions are counted as the checks' (`WorkCounts::device_checks`).

use crate::buffer::Buffer;
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

/// The uniform texel buffer's uniform time over the uniform buffer's, over imported pages, below
/// which the texel fact holds (the module docs).
pub(crate) const TEXEL_FASTER: f64 = 0.7;

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

/// The kernels the imported pages' arms run.
const HOST_KERNELS: [(CheckView, CheckReads); 2] = [
    (CheckView::Uniform, CheckReads::Uniform),
    (CheckView::Texel, CheckReads::Uniform),
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
/// takes from it, and the two ratios they come from. No Metal counterpart: Metal answers no such
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
    /// The uniform texel buffer's uniform time over the uniform buffer's, imported pages, the
    /// larger of the two windows'. `None` unless measured and the device imports host pages.
    pub texel: Option<f64>,
}

impl ViewCosts {
    /// An answer of `verdict` with nothing measured.
    fn unmeasured(verdict: ViewVerdict) -> ViewCosts {
        ViewCosts {
            verdict,
            facts: BTreeSet::new(),
            per_lane: None,
            texel: None,
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

    /// The larger of the two windows' ratios of `over`'s arm to `under`'s, `host` the memory.
    fn ratio(
        &self,
        over: (CheckView, CheckReads),
        under: (CheckView, CheckReads),
        host: bool,
    ) -> Option<f64> {
        WINDOWS
            .iter()
            .map(|&vecs| {
                let over = self.of(over, host, vecs)?;
                let under = self.of(under, host, vecs)?;
                Some(over as f64 / under as f64)
            })
            .try_fold(0f64, |most, r| r.map(|r| most.max(r)))
    }

    /// The answer they give.
    pub(crate) fn costs(&self) -> ViewCosts {
        let per_lane = self.ratio(
            (CheckView::Storage, CheckReads::PerLane),
            (CheckView::Uniform, CheckReads::PerLane),
            false,
        );
        let texel = self.ratio(
            (CheckView::Texel, CheckReads::Uniform),
            (CheckView::Uniform, CheckReads::Uniform),
            true,
        );
        ViewCosts {
            verdict: ViewVerdict::Measured,
            facts: facts_of(per_lane, texel),
            per_lane,
            texel,
        }
    }
}

/// The facts of a run whose per-lane ratio is `per_lane` and texel ratio `texel` (the module
/// docs): each where its ratio was measured and is below its bound.
pub(crate) fn facts_of(per_lane: Option<f64>, texel: Option<f64>) -> BTreeSet<DriverWorkaround> {
    let mut facts = BTreeSet::new();
    if per_lane.is_some_and(|r| r < PER_LANE_COSTLY) {
        facts.insert(DriverWorkaround::UniformBufferPerLaneReadsSerialize);
    }
    if texel.is_some_and(|r| r < TEXEL_FASTER) {
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
        }
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
                 uniform over imported pages {} (faster below {TEXEL_FASTER}), texel/uniform \
                 fields {}; medians of {ROUNDS} rounds in us:",
                ratio(costs.per_lane),
                ratio(costs.texel),
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
                verdict: ViewVerdict::Measured,
                facts,
                per_lane: None,
                texel: None,
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
    let Some(measured) = measure(&device) else {
        // The process is exiting, or the program let the device go: nothing is kept.
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

/// The run's medians; why not, where it measured none; `None` where the process is exiting or the
/// program's last drop stopped the device's holds.
fn measure(device: &Device) -> Option<Result<Medians, String>> {
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
    let made = {
        let _busy = exit_guard::enter()?;
        Kernels::new(device)
    };
    let kernels = match made {
        Ok(kernels) => kernels,
        Err(why) => return Some(Err(why)),
    };
    let measured = rounds(device, &clock, &kernels);
    let busy = exit_guard::enter();
    drop(kernels);
    drop(busy);
    measured
}

/// The rounds of every arm over `kernels`, as [`measure`] answers them.
fn rounds(device: &Device, clock: &GpuClock, kernels: &Kernels) -> Option<Result<Medians, String>> {
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
            let _busy = exit_guard::enter()?;
            let timed = stamps.time(device, clock, |raw, command_buffer| {
                // SAFETY: the internal command buffer is recording outside a render pass and only
                // this records it; the kernels' objects live until they are dropped, after the
                // rounds, and every submission is waited for.
                unsafe { kernels.record(raw, command_buffer, arm) }
            });
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

/// Before a round, step aside for the device's allocations and frees on other threads
/// (`step_aside.rs`): not while the program's last drop stops the check or a call waits for its
/// answer. Counted where it waited.
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
    /// The objects on `device`, inside a section of the exit guard; why not, where it refuses one.
    fn new(device: &Device) -> Result<Kernels, String> {
        let window = (device.scratch_buffer(WINDOW_BYTES))
            .map_err(|error| format!("its window was refused: {error}"))?;
        let out = (device.scratch_buffer(OUT_BYTES))
            .map_err(|error| format!("its out buffer was refused: {error}"))?;
        let host = crate::in_place_check::pointer_pages(device, HOST_BYTES).ok();
        let mut kernels = Kernels {
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
        };
        kernels
            .make()
            .map_err(|error| format!("its kernels were not made: {error}"))?;
        Ok(kernels)
    }

    /// Make every object; those made before a failure go with the drop.
    fn make(&mut self) -> Result<(), Error> {
        let raw: &ash::Device = self.device.shared().vk();
        let compile = self.device.compile_device();
        for (view, reads) in KERNELS {
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
            let pipeline = (pipelines.first().copied())
                .ok_or_else(|| internal("the driver made no pipeline"))?;
            self.pipelines.push(pipeline);
            (self.device.shared()).name_pipeline(pipeline, || {
                format!("metal2vulkan: view check {view:?} {reads:?}")
            });
        }
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
pub(crate) const CHECK_VERSION: u32 = 1;

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
