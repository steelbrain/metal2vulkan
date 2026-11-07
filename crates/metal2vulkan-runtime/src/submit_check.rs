//! The device's checks of its submissions, on a thread of its own, once per driver, release,
//! kernel and runtime: whether `VkSubmitInfo`s that signal nothing merge into one submission of
//! the driver's ([`Check::Merge`]), whether a host-pointer import the device holds adds to the cost
//! of every submission ([`Check::ImportTax`]), and whether a pointer import reads the pages mapped
//! at its pointer at each submission ([`Check::Follows`]). Each answers a question the runtime
//! once read from the driver's identity, its `driverID`, vendor and `deviceID`, so that a GPU no
//! row named took every slow route: the check measures the device itself, and nothing of a
//! decision reads its identity.
//!
//! - **Merge.** [`ROUNDS`] rounds of each of two forms, alternating, each of [`MERGED`] empty
//!   command buffers on the device's first `VkQueue`: one `vkQueueSubmit` of [`MERGED`]
//!   `VkSubmitInfo`s, the last with a fence, against [`MERGED`] calls of one each, the last with
//!   the fence; the time inside the calls. A driver whose one call costs at most [`MERGE_RATIO`]
//!   of the calls apart merges them (`Holds`): Mesa's common queue does
//!   (`vulkan/runtime/vk_queue.c`, `vk_queue_submits_merge`: "Don't merge if there are signals in
//!   between"), so its batches may wait and go in one call (`command_buffer/group.rs`); elsewhere
//!   each batch takes its own. Measured with a probe of the same rounds (2026-11-07, Mesa 26.2.6,
//!   Linux 7.3.1): RADV on the Radeon 780M 13.2 against 37.3 us (0.35), the RX 7900 XTX 13.4
//!   against 35.0 (0.38), the RX 550 11.6 against 35.1 (0.33); ANV on the Iris Xe of RPL-P 5.1
//!   against 13.1 (0.39); MoltenVK 1.4.2 on the M3 Max 4.5 against 4.6 (0.98).
//! - **Import tax.** One import of [`TAX_BYTES`] of anonymous pages the check maps and writes,
//!   by their pointer, held through every other of [`TAX_PHASES`] phases of [`TAX_SUBMITS`]
//!   submissions of one empty command buffer with a fence, none held through the rest; the time
//!   inside each call. Where the median with the import held passes the median without by more
//!   than [`TAX_FLOOR`] or an eighth of it, whichever is more, a held import costs every
//!   submission (`Holds`), and the runtime keeps its own long buffers out of fresh pages
//!   (`fresh_pages.rs`): each would add that cost, and more for its length, to every submission
//!   for its whole life, where its fresh pages save one clear of its bytes. RADV lists every
//!   memory it imported in each kernel submission, and amdgpu gets every page of a pointer import
//!   again in each (`amdgpu_cs_parser_bos`): the probe measured 10.3 us more a submission for one
//!   import of 16 MiB on the Radeon 780M (24.1 for 64 MiB), 6.8 on the RX 7900 XTX (35.7), 45.8
//!   on the RX 550 (191.3), against 0.2 on the Iris Xe (as much as a plain allocation of 64 MiB
//!   costs there) and none on the M3 Max.
//! - **Follows** (Linux alone). Two pages the check maps, the first written with a pattern, are
//!   imported by their pointer and bound to a buffer; a copy of the first page into the second is
//!   submitted, then new anonymous pages are mapped over the first (`mmap` with `MAP_FIXED`) and
//!   written with another pattern, and the same copy submitted again; and a second pair, its first
//!   page replaced before any submission, copied once. Where every copy read the pages mapped at
//!   the first page's address as it was submitted, the driver's pointer imports follow their
//!   address (`Holds`), and an idle no-copy buffer's import may be released and made again at its
//!   pointer at its next use (`host_residency.rs`), which reads the pages the kept import would.
//!   Where a copy read the pages the import was made over, the driver keeps them (`Fails`), as
//!   NVIDIA's pins them, and as Metal wires a no-copy buffer's pages when it is made: such an
//!   import stays live. Not run on another system: on macOS every Vulkan driver imports a pointer
//!   through Metal, which wires the pages as it makes the buffer, so the import keeps them, and a
//!   remap of pages the kernel wired is not made. Anonymous pages replace anonymous pages, mapped
//!   throughout, so no submission meets a range without pages; only the first page of each pair
//!   is replaced, and the GPU writes only the second.
//! - **Its verdicts** are `Pending` until it answers, and the device takes the form a driver
//!   without each answer takes: every batch its own call, its long buffers placed and cleared as
//!   ever, every pointer import kept. `Unchecked` where a check could not run (no
//!   `VK_EXT_external_memory_host`, a call refused, too few rounds), with the same forms.
//! - **Its cost**: [`ROUNDS`] rounds of each merge form and [`TAX_PHASES`] phases of
//!   [`TAX_SUBMITS`] submissions, each waited for, every command buffer empty; the import of
//!   [`TAX_BYTES`] made and freed [`TAX_PHASES`] times over pages written once; and on Linux
//!   three copies of 256 bytes. The probe's rounds took 3 to 12 ms on those devices.
//! - **Kept.** Once every check answered, the verdicts are written as one record in the shader
//!   cache's content directory (`shader_cache/store.rs`), named by the driver's identity, the
//!   kernel's release (the import tax is the kernel's work on RADV), the check's version and the
//!   runtime's ([`record_name`]): a device of the same driver, release and kernel reads them as
//!   it is made and runs no check. A failed write leaves the verdicts for the device's life.
//! - **Its thread** holds its device through the registry of the runtime's own holds
//!   (`device/holds.rs`) for its whole run, and reads the stop between its steps (each round, each
//!   phase, each copy): the program's last drop stops it there and waits for the step in progress
//!   and the release of its objects, so the device goes on the program's thread, and the check
//!   ends unrecorded. Each step's Vulkan work runs inside a section of the exit guard
//!   (`exit_guard.rs`). Its submissions are counted apart (`WorkCounts::submit_check_calls`), not
//!   among the device's.

use crate::device::holds::Holder;
use crate::device::relay::By;
use crate::device::submit_stalls::Uses;
use crate::device::{lock, Device, WeakDevice};
use crate::exit_guard::{self, Call};
use crate::in_place::DriverIdentity;
use crate::release;
use crate::shader_cache::store::{hex, Lookup};
use ash::vk;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// The command buffers of each merge round, and the `VkSubmitInfo`s of its one call.
pub(crate) const MERGED: usize = 4;

/// The rounds of each merge form, after [`WARM`] of each untimed.
pub(crate) const ROUNDS: usize = 16;

/// The untimed rounds of each form first.
const WARM: usize = 2;

/// The most one call of the merge's infos costs, as a share of its calls apart, where the driver
/// merges them.
pub(crate) const MERGE_RATIO: f64 = 0.6;

/// The bytes of the import the tax check holds: past the import's own cost, a length's
/// (amdgpu's work a page), half the shortest buffer of the runtime's own that takes fresh pages.
pub(crate) const TAX_BYTES: u64 = 16 << 20;

/// The tax check's phases, every other one with the import held (the first without).
pub(crate) const TAX_PHASES: usize = 6;

/// The submissions timed in each phase, after one untimed.
pub(crate) const TAX_SUBMITS: usize = 8;

/// The least a held import adds to a submission, in nanoseconds, that the tax check counts: a
/// microsecond, about 0.25% of a core at the 2500 submissions a second of a guest's frames.
pub(crate) const TAX_FLOOR: i64 = 1_000;

/// How long one wait for a fence lasts before it is repeated.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The longest a making whose path the verdicts choose once for good waits for them (the in-place
/// check's fresh-pages class, `in_place_check.rs`; a guest RAM with a source of dma-bufs,
/// `host_ram.rs`): their run takes milliseconds.
pub(crate) const ANSWER_WAIT: Duration = Duration::from_secs(10);

/// The bytes the follows check copies.
#[cfg(target_os = "linux")]
const COPIED: u64 = 256;

/// A check, in the order they run and the record holds their verdicts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Check {
    /// `VkSubmitInfo`s that signal nothing merge into one submission of the driver's.
    Merge,
    /// A held host-pointer import adds to every submission's cost.
    ImportTax,
    /// A pointer import reads the pages mapped at its pointer at each submission.
    Follows,
}

impl Check {
    /// Every check, in order.
    pub(crate) const ALL: [Check; 3] = [Check::Merge, Check::ImportTax, Check::Follows];

    fn index(self) -> usize {
        self as usize
    }
}

/// A check's verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Not answered yet.
    Pending,
    /// The property holds.
    Holds,
    /// It does not.
    Fails,
    /// The check could not run.
    Unchecked,
}

impl Verdict {
    fn byte(self) -> u8 {
        match self {
            Verdict::Pending => 0,
            Verdict::Holds => 1,
            Verdict::Fails => 2,
            Verdict::Unchecked => 3,
        }
    }

    fn of_byte(byte: u8) -> Verdict {
        match byte {
            1 => Verdict::Holds,
            2 => Verdict::Fails,
            3 => Verdict::Unchecked,
            _ => Verdict::Pending,
        }
    }
}

/// What the checks measured, in nanoseconds, in a record's order: one call of the merge's infos
/// and its calls apart (medians), and one submission with the import held and without.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Measured {
    pub(crate) merged: i64,
    pub(crate) apart: i64,
    pub(crate) held: i64,
    pub(crate) bare: i64,
}

/// A device's verdicts of its submission checks, and their state (the module docs).
pub(crate) struct SubmitChecks {
    verdicts: [AtomicU8; 3],
    measured: [AtomicI64; 4],
    started: AtomicBool,
    settled: Mutex<bool>,
    wake: Condvar,
    /// The checks' run in this process, in nanoseconds; 0 where none ended.
    took: AtomicU64,
}

impl SubmitChecks {
    /// A device's, every verdict pending.
    pub(crate) fn new() -> SubmitChecks {
        SubmitChecks {
            verdicts: Default::default(),
            measured: Default::default(),
            started: AtomicBool::new(false),
            settled: Mutex::new(false),
            wake: Condvar::new(),
            took: AtomicU64::new(0),
        }
    }

    /// `check`'s verdict now.
    pub(crate) fn verdict(&self, check: Check) -> Verdict {
        Verdict::of_byte(self.verdicts[check.index()].load(Ordering::Acquire))
    }

    /// Whether `check` answered that its property holds.
    pub(crate) fn holds(&self, check: Check) -> bool {
        self.verdict(check) == Verdict::Holds
    }

    /// Whether `check` answered that its property does not hold.
    pub(crate) fn fails(&self, check: Check) -> bool {
        self.verdict(check) == Verdict::Fails
    }

    fn set(&self, check: Check, verdict: Verdict) {
        self.verdicts[check.index()].store(verdict.byte(), Ordering::Release);
    }

    fn measured(&self) -> Measured {
        let at = |i: usize| self.measured[i].load(Ordering::Acquire);
        Measured {
            merged: at(0),
            apart: at(1),
            held: at(2),
            bare: at(3),
        }
    }

    /// Give `check` the verdict `verdict` for a test that holds the device to the path it
    /// chooses, whatever the device measured (`tests/host_ram_dma_buf.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd), target_os = "linux"))]
    pub(crate) fn force(&self, check: Check, verdict: Verdict) {
        self.set(check, verdict);
    }

    fn set_measured(&self, measured: Measured) {
        let values = [
            measured.merged,
            measured.apart,
            measured.held,
            measured.bare,
        ];
        for (slot, value) in self.measured.iter().zip(values) {
            slot.store(value, Ordering::Release);
        }
    }

    /// Note the checks started; whether they had not before.
    fn start(&self) -> bool {
        !self.started.swap(true, Ordering::AcqRel)
    }

    /// Every verdict still pending is unchecked, and the checks settle.
    fn settle_unchecked(&self) {
        for check in Check::ALL {
            if self.verdict(check) == Verdict::Pending {
                self.set(check, Verdict::Unchecked);
            }
        }
        self.settle(None);
    }

    /// The verdicts are known, with the run where it ran here.
    fn settle(&self, took: Option<Duration>) {
        if let Some(took) = took {
            let nanos = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX);
            self.took.store(nanos.max(1), Ordering::Release);
        }
        *lock(&self.settled) = true;
        self.wake.notify_all();
    }

    /// Wait until the verdicts are known, or `timeout` passed: whether they are.
    fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut settled = lock(&self.settled);
        while !*settled {
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
}

/// The merge's verdict from the medians of its two forms.
pub(crate) fn merge_verdict(merged: i64, apart: i64) -> Verdict {
    match apart > 0 && merged as f64 <= MERGE_RATIO * apart as f64 {
        true => Verdict::Holds,
        false => Verdict::Fails,
    }
}

/// The import tax's verdict from the medians of a submission with the import held and without.
pub(crate) fn tax_verdict(held: i64, bare: i64) -> Verdict {
    match held - bare > TAX_FLOOR.max(bare / 8) {
        true => Verdict::Holds,
        false => Verdict::Fails,
    }
}

/// The version of what the checks measure and of how their record is named and framed: a change
/// to either bumps it, and a record of another version is never read.
pub(crate) const CHECK_VERSION: u32 = 1;

/// A record's first bytes.
const MAGIC: [u8; 8] = *b"M2VSUBC\0";

/// The domain tag of a record's name.
const NAME_TAG: &[u8] = b"metal2vulkan submission checks record\0";

/// A record's length: magic, version, runtime, identity, verdicts, measured, checksum.
const RECORD_BYTES: usize = 8 + 4 + 4 + 32 + 3 + 4 * 8 + 32;

/// The kernel's release (`uname -r`), which the import tax's cost is the work of on RADV: empty
/// where the system answers none.
fn kernel_release() -> Vec<u8> {
    #[cfg(unix)]
    {
        // SAFETY: an all-zero `utsname` is a valid value for `uname` to fill.
        let mut name: libc::utsname = unsafe { std::mem::zeroed() };
        // SAFETY: `name` is a valid `utsname` to write.
        if unsafe { libc::uname(&mut name) } == 0 {
            return (name.release.iter())
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
        }
    }
    Vec::new()
}

/// The digest of `identity` and the kernel's release that names a record and that the record
/// holds again.
fn digest(identity: &DriverIdentity) -> [u8; 32] {
    let mut kind = b"submissions\0".to_vec();
    kind.extend_from_slice(&kernel_release());
    kind.push(0);
    identity.digest_for(NAME_TAG, &kind, CHECK_VERSION)
}

/// The name of the record of `identity`'s verdicts in the shader cache's content directory.
pub(crate) fn record_name(identity: &DriverIdentity) -> String {
    hex(&digest(identity))
}

/// The bytes of a record of `verdicts` and `measured` for `identity`:
///
/// ```text
/// magic          8 bytes   "M2VSUBC\0"
/// version        u32       CHECK_VERSION
/// runtime        u32       RUNTIME_VERSION
/// identity       32 bytes  the identity's and the kernel's digest for these checks
/// verdicts       3 bytes   one per check, in Check::ALL's order: Holds 1, Fails 2, Unchecked 3
/// measured       4 x i64   merged, apart, held, bare, in nanoseconds
/// checksum       32 bytes  SHA-256 of every byte before it
/// ```
///
/// Integers are little-endian.
pub(crate) fn encode(
    identity: &DriverIdentity,
    verdicts: [Verdict; 3],
    measured: Measured,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_BYTES);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&CHECK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&metal2vulkan::cache::RUNTIME_VERSION.to_le_bytes());
    bytes.extend_from_slice(&digest(identity));
    bytes.extend(verdicts.map(Verdict::byte));
    for value in [
        measured.merged,
        measured.apart,
        measured.held,
        measured.bare,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let checksum: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// The verdicts and measurements `bytes` record for `identity`: `None` for anything but a whole
/// record of this version and runtime, for this identity and kernel, whose checksum holds and
/// whose every verdict is an answer.
pub(crate) fn decode(identity: &DriverIdentity, bytes: &[u8]) -> Option<([Verdict; 3], Measured)> {
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
    let (verdicts, rest) = rest.split_at(3);
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap_or([0xff; 4]));
    if magic != MAGIC
        || word(version) != CHECK_VERSION
        || word(runtime) != metal2vulkan::cache::RUNTIME_VERSION
        || identity_digest != digest(identity)
    {
        return None;
    }
    let verdicts = [0, 1, 2].map(|i| Verdict::of_byte(verdicts[i]));
    if verdicts.contains(&Verdict::Pending) {
        return None;
    }
    let value = |i: usize| i64::from_le_bytes(rest[8 * i..8 * i + 8].try_into().unwrap_or([0; 8]));
    let measured = Measured {
        merged: value(0),
        apart: value(1),
        held: value(2),
        bare: value(3),
    };
    Some((verdicts, measured))
}

impl Device {
    /// As the device is made: its verdicts where a record holds them (settled then), else its
    /// checks start on a thread of their own. A record that does not decode is no record.
    pub(crate) fn recall_submit_checks(&self) {
        let shared = self.shared();
        let identity = self.driver_identity();
        let name = record_name(&identity);
        let looked = (shared.shader_cache.content_store())
            .load_content(&name, |bytes| decode(&identity, &bytes));
        if let Lookup::Hit((verdicts, measured)) = looked {
            let checks = &shared.submit_checks;
            for (check, verdict) in Check::ALL.into_iter().zip(verdicts) {
                checks.set(check, verdict);
            }
            checks.set_measured(measured);
            checks.start();
            shared.counts.submit_checks_recalled();
            checks.settle(None);
            return;
        }
        self.start_submit_checks();
    }

    /// Start the checks on a thread of their own, where they have not started. Where the thread
    /// cannot start, every verdict is unchecked.
    fn start_submit_checks(&self) {
        let checks = &self.shared().submit_checks;
        if !checks.start() {
            return;
        }
        let weak = self.downgrade();
        let builder = std::thread::Builder::new().name("m2v-submit-check".to_owned());
        if exit_guard::spawn(builder, move || run(&weak)).is_err() {
            checks.settle_unchecked();
        }
    }

    /// Whether the device's `VkSubmitInfo`s that signal nothing merge into one submission of the
    /// driver's, as its check found (`submit_check.rs`), so that its batches may wait in a group
    /// (`CompletionPolicy::Grouped`): `false` until it answers. A runtime extension for the lanes
    /// that hold a device to the path it chooses.
    #[doc(hidden)]
    pub fn submissions_merge(&self) -> bool {
        self.shared().submit_checks.holds(Check::Merge)
    }

    /// Whether the device's long buffers of its own take fresh pages (`fresh_pages.rs`): its
    /// check found a held import adds nothing to a submission's cost (`submit_check.rs`); `false`
    /// until it answers. A runtime extension for the lanes that hold a device to the path it
    /// chooses.
    #[doc(hidden)]
    pub fn fresh_pages_pay(&self) -> bool {
        self.shared().submit_checks.fails(Check::ImportTax)
    }

    /// Whether a host-pointer import the device holds adds to the cost of every submission, as
    /// its check found (`submit_check.rs`): where it does, a guest RAM with a source of dma-bufs
    /// imports its chunks as dma-bufs first (`host_ram.rs`). `false` until it answers.
    pub(crate) fn pointer_imports_tax_submissions(&self) -> bool {
        self.shared().submit_checks.holds(Check::ImportTax)
    }

    /// No Metal counterpart: whether the device's imports of host memory by its pointer read the
    /// pages mapped at the pointer at each submission, as its check found: `true` where they do,
    /// and an idle no-copy buffer's import is released and made again at its next use; `false`
    /// where they keep the pages they were made over (as Metal's no-copy buffer does), where the
    /// check could not run or did not run (macOS), and until it answers
    /// ([`settle_submit_checks`](Device::settle_submit_checks) waits for it). A runtime extension:
    /// where it is `true`, pages mapped over a live no-copy buffer's addresses are what its later
    /// command buffers read and write.
    pub fn pointer_imports_follow_their_address(&self) -> bool {
        self.shared().submit_checks.holds(Check::Follows)
    }

    /// Wait until the device's submission checks answered, or `timeout` passed: whether they did.
    /// A runtime extension for the lanes and benchmarks that hold a device to the paths its
    /// verdicts choose.
    #[doc(hidden)]
    pub fn settle_submit_checks(&self, timeout: Duration) -> bool {
        self.shared().submit_checks.wait(timeout)
    }

    /// The device's submission checks' verdicts and measurements, and how long they ran in this
    /// process, in words. A runtime extension for the lanes and benchmarks that report them.
    #[doc(hidden)]
    pub fn submit_check_report(&self) -> String {
        let checks = &self.shared().submit_checks;
        let m = checks.measured();
        let us = |nanos: i64| nanos as f64 / 1e3;
        let took = match checks.took.load(Ordering::Acquire) {
            0 => "not run here".to_owned(),
            nanos => format!("ran {:.2} ms", nanos as f64 / 1e6),
        };
        format!(
            "merge {:?} ({:.1} us in one call of {MERGED} against {:.1} us apart), import tax \
             {:?} ({:.1} us a submission with {} MiB held against {:.1} us), follows {:?}, {took}",
            checks.verdict(Check::Merge),
            us(m.merged),
            us(m.apart),
            checks.verdict(Check::ImportTax),
            us(m.held),
            TAX_BYTES >> 20,
            us(m.bare),
            checks.verdict(Check::Follows),
        )
    }
}

/// The checks' thread: the device held through its run, each check's verdict given as it is
/// known, then the record. Stopped by the program's last drop, it records nothing.
fn run(weak: &WeakDevice) {
    let Some(device) = weak.hold() else {
        return;
    };
    let started = Instant::now();
    let checks = &device.shared().submit_checks;
    let Some(objects) = Objects::make(&device) else {
        checks.settle_unchecked();
        return;
    };
    let ran = measure(&device, &objects);
    let released = objects.release(&device);
    let Some(measured) = ran.filter(|_| released) else {
        // The process is exiting, or the program let the device go: nothing is kept.
        return;
    };
    checks.set_measured(measured);
    device.shared().counts.submit_checks_ran();
    let verdicts = Check::ALL.map(|check| checks.verdict(check));
    if !verdicts.contains(&Verdict::Pending) {
        let identity = device.driver_identity();
        let bytes = encode(&identity, verdicts, measured);
        // A failed write leaves the verdicts to the device's life alone.
        let _ = (device.shared().shader_cache.content_store())
            .write_record_soon(&record_name(&identity), bytes);
    }
    checks.settle(Some(started.elapsed()));
}

/// Each check in turn, its verdict set as it is known: what they measured, `None` where the
/// process is exiting or the program's last drop stopped the device's holds.
fn measure(device: &Device, objects: &Objects) -> Option<Measured> {
    let checks = &device.shared().submit_checks;
    let mut measured = Measured::default();
    match merge(device, objects)? {
        Some((merged, apart)) => {
            measured.merged = merged;
            measured.apart = apart;
            checks.set(Check::Merge, merge_verdict(merged, apart));
        }
        None => checks.set(Check::Merge, Verdict::Unchecked),
    }
    match tax(device, objects)? {
        Some((held, bare)) => {
            measured.held = held;
            measured.bare = bare;
            checks.set(Check::ImportTax, tax_verdict(held, bare));
        }
        None => checks.set(Check::ImportTax, Verdict::Unchecked),
    }
    checks.set(Check::Follows, follows(device, objects)?);
    Some(measured)
}

/// Whether the check goes on to its next step (the program's last drop stops it).
fn goes_on(device: &Device) -> bool {
    device.holds_go_on(Holder::SubmitCheck)
}

/// The merge's medians, in nanoseconds: one call of [`MERGED`] infos, and [`MERGED`] calls.
/// `Some(None)` where a call was refused; `None` where the check stops.
fn merge(device: &Device, objects: &Objects) -> Option<Option<(i64, i64)>> {
    let mut merged = Vec::with_capacity(ROUNDS);
    let mut apart = Vec::with_capacity(ROUNDS);
    for round in 0..WARM + ROUNDS {
        for together in [true, false] {
            if !goes_on(device) {
                return None;
            }
            let Some(nanos) = objects.submit_round(device, together)? else {
                return Some(None);
            };
            if round >= WARM {
                match together {
                    true => merged.push(nanos),
                    false => apart.push(nanos),
                }
            }
        }
    }
    Some(Some((median(&mut merged), median(&mut apart))))
}

/// The import tax's medians, in nanoseconds: a submission with the import of [`TAX_BYTES`]
/// held, and without. `Some(None)` where the device imports no host pointer, the system maps no
/// pages or a call was refused; `None` where the check stops.
fn tax(device: &Device, objects: &Objects) -> Option<Option<(i64, i64)>> {
    let Some(pages) = Pages::map(TAX_BYTES) else {
        return Some(None);
    };
    pages.fill(0x5a);
    let mut held = Vec::with_capacity(TAX_PHASES / 2 * TAX_SUBMITS);
    let mut bare = Vec::with_capacity(TAX_PHASES / 2 * TAX_SUBMITS);
    for phase in 0..TAX_PHASES {
        let holding = phase % 2 == 1;
        if !goes_on(device) {
            return None;
        }
        let import = match holding {
            true => match Import::make(device, &pages, false)? {
                Some(import) => Some(import),
                None => return Some(None),
            },
            false => None,
        };
        let mut times = Vec::with_capacity(TAX_SUBMITS);
        let mut refused = false;
        for submission in 0..=TAX_SUBMITS {
            match objects.submit_one(device)? {
                Some(nanos) if submission > 0 => times.push(nanos),
                Some(_) => {}
                None => {
                    refused = true;
                    break;
                }
            }
        }
        if let Some(import) = import {
            import.release(device)?;
        }
        if refused {
            return Some(None);
        }
        match holding {
            true => held.extend(times),
            false => bare.extend(times),
        }
    }
    Some(Some((median(&mut held), median(&mut bare))))
}

/// The follows check's verdict (the module docs); `None` where the check stops.
#[cfg(target_os = "linux")]
fn follows(device: &Device, objects: &Objects) -> Option<Verdict> {
    let Some(page) = crate::host_memory::host_page() else {
        return Some(Verdict::Unchecked);
    };
    // The kept import, its first page replaced between two submissions.
    let Some(kept) = follows_pair(device, objects, page, false)? else {
        return Some(Verdict::Unchecked);
    };
    if !goes_on(device) {
        return None;
    }
    // An import whose first page is replaced before its first submission.
    let Some(fresh) = follows_pair(device, objects, page, true)? else {
        return Some(Verdict::Unchecked);
    };
    Some(match (kept, fresh) {
        (Read::New, Read::New) => Verdict::Holds,
        (Read::Old, Read::Old) => Verdict::Fails,
        _ => Verdict::Unchecked,
    })
}

/// Not run off Linux (the module docs).
#[cfg(not(target_os = "linux"))]
fn follows(_device: &Device, _objects: &Objects) -> Option<Verdict> {
    Some(Verdict::Unchecked)
}

/// What a copy of the replaced page read.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// The pages the import was made over.
    Old,
    /// The pages mapped over them.
    New,
    /// Neither.
    Other,
}

/// Patterns the follows check writes: the pages an import was made over, and those mapped over
/// them.
#[cfg(target_os = "linux")]
const OLD: u8 = 0xa1;
#[cfg(target_os = "linux")]
const NEW: u8 = 0xb2;

/// One pair of pages imported and copied (the module docs): what the copy after the remap read
/// (`before_first` replaces the first page before any submission). `Some(None)` where the device
/// takes no such import or a copy failed; `None` where the check stops.
#[cfg(target_os = "linux")]
fn follows_pair(
    device: &Device,
    objects: &Objects,
    page: u64,
    before_first: bool,
) -> Option<Option<Read>> {
    let Some(pages) = Pages::map(2 * page) else {
        return Some(None);
    };
    pages.fill(OLD);
    let Some(import) = Import::make(device, &pages, true)? else {
        return Some(None);
    };
    let read = |pages: &Pages| {
        // SAFETY: the second page is the check's, mapped, and no copy into it is pending.
        let bytes = unsafe {
            std::slice::from_raw_parts(pages.start.as_ptr().add(page as usize), COPIED as usize)
        };
        match bytes {
            _ if bytes.iter().all(|&b| b == OLD) => Read::Old,
            _ if bytes.iter().all(|&b| b == NEW) => Read::New,
            _ => Read::Other,
        }
    };
    let answer = (|| {
        if !before_first {
            pages.clear_second(page);
            if !objects.copy(device, &import, page)? {
                return Some(None);
            }
            if read(&pages) != Read::Old {
                return Some(None);
            }
        }
        if !pages.replace_first(page) {
            return Some(None);
        }
        pages.clear_second(page);
        if !objects.copy(device, &import, page)? {
            return Some(None);
        }
        Some(Some(read(&pages)))
    })();
    import.release(device)?;
    answer
}

/// Anonymous private pages the check maps, read-write, unmapped as this goes where every import
/// of them was freed (an import left unreleased, as the process exits, keeps them mapped to the
/// end).
struct Pages {
    start: NonNull<u8>,
    length: u64,
    /// The live imports of these pages.
    imports: Cell<u32>,
}

impl Pages {
    /// `length` bytes of new pages; `None` where the system maps none.
    #[cfg(unix)]
    fn map(length: u64) -> Option<Pages> {
        let bytes = usize::try_from(length).ok()?;
        // SAFETY: a new private anonymous mapping, at an address the kernel picks.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return None;
        }
        Some(Pages {
            start: NonNull::new(pointer.cast())?,
            length,
            imports: Cell::new(0),
        })
    }

    /// None off Unix.
    #[cfg(not(unix))]
    fn map(_length: u64) -> Option<Pages> {
        None
    }

    /// Write `byte` into every byte.
    fn fill(&self, byte: u8) {
        // SAFETY: the pages are the check's, mapped read-write, and nothing reads them meanwhile.
        unsafe { std::ptr::write_bytes(self.start.as_ptr(), byte, self.length as usize) };
    }

    /// Zero the second page's bytes the copy writes.
    #[cfg(target_os = "linux")]
    fn clear_second(&self, page: u64) {
        // SAFETY: as `fill`; no copy into the page is pending.
        unsafe {
            std::ptr::write_bytes(self.start.as_ptr().add(page as usize), 0, COPIED as usize)
        };
    }

    /// Map new anonymous pages over the first page (`MAP_FIXED`), written with [`NEW`]: whether
    /// the system did.
    #[cfg(target_os = "linux")]
    fn replace_first(&self, page: u64) -> bool {
        // SAFETY: the first page is the check's own mapping, replaced in one call, so the range
        // is mapped throughout; no copy of it is pending.
        let pointer = unsafe {
            libc::mmap(
                self.start.as_ptr().cast(),
                page as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if pointer != self.start.as_ptr().cast() {
            return false;
        }
        // SAFETY: the new page is mapped read-write, the check's.
        unsafe { std::ptr::write_bytes(self.start.as_ptr(), NEW, page as usize) };
        true
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        if self.imports.get() != 0 {
            return;
        }
        #[cfg(unix)]
        // SAFETY: the mapping is the check's, and no import of it is live.
        unsafe {
            libc::munmap(self.start.as_ptr().cast(), self.length as usize);
        }
    }
}

/// An import of a [`Pages`]' bytes by their pointer, in its own `VkDeviceMemory`, and, for the
/// follows check, a buffer bound to it.
struct Import<'p> {
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    pages: &'p Pages,
}

impl<'p> Import<'p> {
    /// The import of `pages`, with a buffer bound where `bound`, in a host-coherent type the
    /// pointer takes. `Some(None)` where the device imports no host pointer, the pages take no
    /// such type, or a call was refused; `None` where the process is exiting.
    fn make(device: &Device, pages: &'p Pages, bound: bool) -> Option<Option<Import<'p>>> {
        let shared = device.shared();
        let Ok(importer) = &shared.host_import else {
            return Some(None);
        };
        let _busy = exit_guard::enter()?;
        let raw = shared.vk();
        // SAFETY: the pages are this process's, at a host page, whose size the import alignment
        // divides where the device imports at all (it is checked below).
        let Ok(types) = (unsafe { importer.memory_types(raw, pages.start) }) else {
            return Some(None);
        };
        let unit = importer.page.max(importer.alignment);
        if !pages.length.is_multiple_of(unit) || !(pages.start.as_ptr() as u64).is_multiple_of(unit)
        {
            return Some(None);
        }
        let mut buffer = vk::Buffer::null();
        let mut allowed = types;
        if bound {
            let mut external = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
            let info = shared.sharing().buffer(
                vk::BufferCreateInfo::default()
                    .size(pages.length)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                    .push_next(&mut external),
            );
            // SAFETY: the device is live and `info` and its chain valid, inside a section.
            buffer = match unsafe { crate::creation::buffer(raw, &info) } {
                Ok(buffer) => buffer,
                Err(_) => return Some(None),
            };
            exit_guard::called(Call::CreateBuffer);
            // SAFETY: `buffer` is a live buffer of this device.
            let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
            exit_guard::called(Call::GetBufferMemoryRequirements);
            allowed &= requirements.memory_type_bits;
        }
        let coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory = lock(&shared.memory);
        let type_index = (0..32).find(|&index| {
            allowed & (1 << index) != 0
                && memory
                    .type_flags(index)
                    .is_some_and(|flags| flags.contains(coherent))
        });
        drop(memory);
        let discard = |buffer: vk::Buffer| {
            if buffer != vk::Buffer::null() {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw, buffer) };
            }
        };
        let Some(type_index) = type_index else {
            discard(buffer);
            return Some(None);
        };
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            .host_pointer(pages.start.as_ptr().cast());
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(pages.length)
            .memory_type_index(type_index)
            .push_next(&mut import);
        // SAFETY: the pages stay mapped until after this memory is freed (`Pages` outlives every
        // `Import` and stays mapped while one is live), at a multiple of the import alignment, of
        // a type they take.
        let memory = match unsafe { crate::creation::allocate_memory(raw, &info) } {
            Ok(memory) => memory,
            Err(_) => {
                exit_guard::called(Call::AllocateMemory);
                discard(buffer);
                return Some(None);
            }
        };
        exit_guard::called(Call::AllocateMemory);
        if bound {
            // SAFETY: the memory is the pages', of a type the buffer takes and its whole length;
            // the buffer is unbound.
            if unsafe { crate::creation::bind_buffer(raw, buffer, memory, 0) }.is_err() {
                exit_guard::called(Call::BindBufferMemory);
                discard(buffer);
                // SAFETY: the memory is unused.
                unsafe { release::object(raw, memory) };
                return Some(None);
            }
            exit_guard::called(Call::BindBufferMemory);
        }
        pages.imports.set(pages.imports.get() + 1);
        Some(Some(Import {
            memory,
            buffer,
            pages,
        }))
    }

    /// Free it, inside a section: `None` where the process is exiting, the import then left to
    /// its end with its pages mapped.
    fn release(self, device: &Device) -> Option<()> {
        let _busy = exit_guard::enter()?;
        let raw = device.shared().vk();
        // SAFETY: no submission that names them is pending (each was waited for).
        unsafe {
            if self.buffer != vk::Buffer::null() {
                release::object(raw, self.buffer);
            }
            release::object(raw, self.memory);
        }
        self.pages.imports.set(self.pages.imports.get() - 1);
        Some(())
    }
}

/// The checks' Vulkan objects: a command pool of the first `VkQueue`'s family, [`MERGED`] empty
/// command buffers and, on Linux, one for the follows check's copy, and a fence.
struct Objects {
    pool: vk::CommandPool,
    empty: Vec<vk::CommandBuffer>,
    #[cfg(target_os = "linux")]
    copy: vk::CommandBuffer,
    fence: vk::Fence,
}

impl Objects {
    /// The objects, the empty command buffers recorded; `None` where a call was refused (what was
    /// made released) or the process is exiting.
    fn make(device: &Device) -> Option<Objects> {
        let shared = device.shared();
        let _busy = exit_guard::enter()?;
        let raw = shared.vk();
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(shared.queues[0].family);
        // SAFETY: the device is live, inside a section.
        let pool = unsafe { raw.create_command_pool(&info, None) };
        exit_guard::called(Call::CreateCommandPool);
        let mut objects = Objects {
            pool: pool.ok()?,
            empty: Vec::new(),
            #[cfg(target_os = "linux")]
            copy: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        };
        if objects.fill(device).is_none() {
            drop(_busy);
            objects.release(device);
            return None;
        }
        Some(objects)
    }

    /// Make and record the command buffers and make the fence, inside the caller's section.
    fn fill(&mut self, device: &Device) -> Option<()> {
        let shared = device.shared();
        let raw = shared.vk();
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MERGED as u32 + u32::from(cfg!(target_os = "linux")));
        // SAFETY: the pool is the check's, which this thread alone uses; the device's loader word
        // is put back first (`loader_word.rs`).
        let allocated = unsafe {
            shared.loader_word.restore();
            raw.allocate_command_buffers(&allocate)
        };
        exit_guard::called(Call::AllocateCommandBuffers);
        self.empty = allocated.ok()?;
        #[cfg(target_os = "linux")]
        {
            self.copy = self.empty.pop()?;
        }
        let begin = vk::CommandBufferBeginInfo::default();
        for &buffer in &self.empty {
            // SAFETY: a new command buffer of the pool, recorded once on this thread.
            unsafe {
                raw.begin_command_buffer(buffer, &begin).ok()?;
                exit_guard::called(Call::BeginCommandBuffer);
                raw.end_command_buffer(buffer).ok()?;
                exit_guard::called(Call::EndCommandBuffer);
            }
        }
        // SAFETY: the device is live, inside a section.
        self.fence = unsafe { raw.create_fence(&vk::FenceCreateInfo::default(), None) }.ok()?;
        exit_guard::called(Call::CreateFence);
        Some(())
    }

    /// Submit `infos` to the device's first `VkQueue`, the last signalling the fence, in one call
    /// where `together`, else one call each, and wait for the fence: the nanoseconds inside the
    /// calls. `Some(None)` where a call was refused or the wait failed; `None` where the process
    /// is exiting.
    fn submit(
        &self,
        device: &Device,
        buffers: &[vk::CommandBuffer],
        together: bool,
    ) -> Option<Option<i64>> {
        let shared = device.shared();
        let calls = match together {
            true => 1,
            false => buffers.len(),
        };
        let nanos = {
            let _busy = exit_guard::enter()?;
            let raw = shared.vk();
            let infos: Vec<vk::SubmitInfo<'_>> = (buffers.iter())
                .map(|buffer| {
                    vk::SubmitInfo::default().command_buffers(std::slice::from_ref(buffer))
                })
                .collect();
            // SAFETY: the queue is used under its lock, inside a section; each command buffer is
            // recorded, ended and in no pending submission (each round waits for the last); the
            // fence is unsignaled and in no pending submission.
            let submitted = shared.use_queue(0, By::Runtime, Uses::Unknown, |queue| unsafe {
                let mut spent = Duration::ZERO;
                let last = infos.len() - 1;
                let parts: Vec<(&[vk::SubmitInfo<'_>], vk::Fence)> = match together {
                    true => vec![(&infos[..], self.fence)],
                    false => (infos.iter().enumerate())
                        .map(|(i, info)| {
                            let fence = if i == last {
                                self.fence
                            } else {
                                vk::Fence::null()
                            };
                            (std::slice::from_ref(info), fence)
                        })
                        .collect(),
                };
                for (infos, fence) in parts {
                    let start = Instant::now();
                    let made = raw.queue_submit(queue, infos, fence);
                    spent += start.elapsed();
                    if made.is_err() {
                        return Err(spent);
                    }
                }
                Ok(spent)
            });
            exit_guard::called(Call::QueueSubmit);
            shared.counts.submit_check_called(calls);
            match submitted {
                Ok(spent) => Some(spent),
                Err(_) if calls > 1 => None,
                Err(_) => return Some(None),
            }
        };
        // A refused call among several: those before it, and so the fence, were not all
        // submitted, so the device waits for its queue rather than the fence.
        let Some(nanos) = nanos else {
            return self.idle(device).map(|_| None);
        };
        if !self.wait(device)? {
            return Some(None);
        }
        Some(Some(i64::try_from(nanos.as_nanos()).unwrap_or(i64::MAX)))
    }

    /// One merge round ([`submit`](Self::submit) of every empty command buffer).
    fn submit_round(&self, device: &Device, together: bool) -> Option<Option<i64>> {
        self.submit(device, &self.empty, together)
    }

    /// One submission of one empty command buffer with the fence.
    fn submit_one(&self, device: &Device) -> Option<Option<i64>> {
        self.submit(device, &self.empty[..1], true)
    }

    /// Record and submit a copy of [`COPIED`] bytes from the start of `import`'s buffer to
    /// `page` bytes in, then a barrier to the host's reads, and wait for it: whether it ran.
    /// `None` where the process is exiting.
    #[cfg(target_os = "linux")]
    fn copy(&self, device: &Device, import: &Import, page: u64) -> Option<bool> {
        {
            let _busy = exit_guard::enter()?;
            let raw = device.shared().vk();
            let region = vk::BufferCopy::default().dst_offset(page).size(COPIED);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // SAFETY: the copy command buffer is the check's, in no pending submission (each is
            // waited for), its pool made to reset its buffers; the buffer is live, and its two
            // ranges do not overlap.
            unsafe {
                if raw.begin_command_buffer(self.copy, &begin).is_err() {
                    return Some(false);
                }
                exit_guard::called(Call::BeginCommandBuffer);
                raw.cmd_copy_buffer(self.copy, import.buffer, import.buffer, &[region]);
                exit_guard::called(Call::CmdCopyBuffer);
                raw.cmd_pipeline_barrier(
                    self.copy,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
                exit_guard::called(Call::CmdPipelineBarrier);
                if raw.end_command_buffer(self.copy).is_err() {
                    return Some(false);
                }
                exit_guard::called(Call::EndCommandBuffer);
            }
        }
        Some(self.submit(device, &[self.copy], true)?.is_some())
    }

    /// Wait for the fence, a slice at a time, each a section, then reset it: whether it was
    /// signalled. `None` where the process is exiting.
    fn wait(&self, device: &Device) -> Option<bool> {
        loop {
            let _busy = exit_guard::enter()?;
            let raw = device.shared().vk();
            // SAFETY: the check's fence, signalled by a submission made.
            let waited =
                unsafe { raw.wait_for_fences(&[self.fence], true, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitForFences);
            match waited {
                Ok(()) => {
                    // SAFETY: the fence is signalled, in no pending submission.
                    let reset = unsafe { raw.reset_fences(&[self.fence]) };
                    exit_guard::called(Call::ResetFences);
                    return Some(reset.is_ok());
                }
                Err(vk::Result::TIMEOUT) => continue,
                Err(_) => return Some(false),
            }
        }
    }

    /// Wait until the first `VkQueue` is idle and reset the fence, after a refused call: `None`
    /// where the process is exiting.
    fn idle(&self, device: &Device) -> Option<()> {
        let _busy = exit_guard::enter()?;
        let shared = device.shared();
        let raw = shared.vk();
        // SAFETY: the queue is used under its lock, inside a section.
        let _ = shared.use_queue(0, By::Runtime, Uses::Unknown, |queue| unsafe {
            raw.queue_wait_idle(queue)
        });
        exit_guard::called(Call::QueueWaitIdle);
        // SAFETY: no submission of the fence is pending once the queue is idle.
        let _ = unsafe { raw.reset_fences(&[self.fence]) };
        exit_guard::called(Call::ResetFences);
        Some(())
    }

    /// Let the objects go, inside a section: whether it ran (`false` once the process is
    /// exiting, the objects then left to its end). No submission naming them is pending: each
    /// was waited for.
    fn release(&self, device: &Device) -> bool {
        let Some(_busy) = exit_guard::enter() else {
            return false;
        };
        let raw = device.shared().vk();
        // SAFETY: no pending submission names them; a null handle is skipped.
        unsafe {
            if self.fence != vk::Fence::null() {
                release::object(raw, self.fence);
            }
            release::object(raw, self.pool);
        }
        true
    }
}

/// The median of `values` (sorted here); 0 for none.
fn median(values: &mut [i64]) -> i64 {
    values.sort_unstable();
    values.get(values.len() / 2).copied().unwrap_or(0)
}
