//! The device's checks of its submissions, on a thread of its own, once per driver, release,
//! kernel and runtime: whether `VkSubmitInfo`s that signal nothing merge into one submission of
//! the driver's ([`Check::Merge`]), and whether a host-pointer import the device holds adds to the
//! cost of every submission ([`Check::ImportTax`]). Each answers a question the runtime once read
//! from the driver's identity, its `driverID`, vendor and `deviceID`, so that a GPU no
//! row named took every slow route: the check measures the device itself, and nothing of a
//! decision reads its identity.
//!
//! - **Merge.** [`ROUNDS`] rounds of each of two forms, alternating, each of [`MERGED`] empty
//!   command buffers on the device's first `VkQueue`: one `vkQueueSubmit` of [`MERGED`]
//!   `VkSubmitInfo`s, the last with a fence, against [`MERGED`] calls of one each, the last with
//!   the fence; the time inside each call. The calls apart's floor is the sum of each call's own
//!   ([`apart_floor`]: the least of the first calls of the rounds, plus that of the second...). A
//!   driver whose one call's floor is at most
//!   [`MERGE_RATIO`] of the calls' apart in both halves of the rounds merges them (`Holds`), and
//!   one whose floor passes it in both does not (`Fails`): Mesa's common queue does
//!   (`vulkan/runtime/vk_queue.c`, `vk_queue_submits_merge`: "Don't merge if there are signals in
//!   between"), so its batches may wait and go in one call (`command_buffer/group.rs`); elsewhere
//!   each batch takes its own. Measured with a probe of the same rounds (medians, 2026-11-07,
//!   Mesa 26.2.6, Linux 7.3.1): RADV on the Radeon 780M 13.2 against 37.3 us (0.35), the RX 7900
//!   XTX 13.4 against 35.0 (0.38), the RX 550 11.6 against 35.1 (0.33); ANV on the Iris Xe of
//!   RPL-P 5.1 against 13.1 (0.39); MoltenVK 1.4.2 on the M3 Max 4.5 against 4.6 (0.98).
//! - **Import tax.** [`TAX_IMPORTS`] imports of [`TAX_BYTES`] of anonymous pages in all, each of
//!   pages of its own the check maps and writes, by their pointer, held through the middle half
//!   of [`TAX_PHASES`] phases of [`TAX_SUBMITS`] submissions of one empty command buffer with a
//!   fence ([`tax_holds`]), none held through the quarters before and after it; the time inside
//!   each call. Each half of the phases so holds as many phases with the imports as without, and
//!   one set of imports, made before the first phase that holds them and released after the last,
//!   serves the try: each import and its release holds back every other thread's submission on
//!   some drivers (NVIDIA's), which a set made and released for each phase that holds them did
//!   four times over. Where, in both halves of the
//!   phases, the floor of the submissions with the imports held passes that of those without by
//!   more than [`TAX_FLOOR`] or an eighth of it, whichever is more, a held import costs every
//!   submission (`Holds`), and where it passes in neither, it does not (`Fails`). Several imports,
//!   not one of the same bytes: a driver may pay per listed object, whatever its length, or per
//!   page. ANV on i915, which has no `VM_BIND`, lists every buffer object of the device in every
//!   execbuffer, about 0.35 us each whatever its size (from-scratch Vulkan on the Iris Xe of
//!   RPL-P, Mesa 26.2.6, Linux 7.3.1: 18.6 us a submission with 32 imports of 32 MiB held, 107.5
//!   with 256 of 1 MiB, 8.3 with none), below the floor for one import and past it for
//!   [`TAX_IMPORTS`]. Where it does, the runtime keeps its own long buffers out of
//!   fresh pages (`fresh_pages.rs`): each would add that cost, and more for its length, to every
//!   submission for its whole life, where its fresh pages save one clear of its bytes. And it
//!   releases a no-copy buffer's idle pointer import and imports it again at its next blit
//!   (`host_residency.rs`): under `HostMemory::new`'s contract the pages at the pointer stay the
//!   same while the buffer lives, so the new import reads what the kept one would on any driver,
//!   and every submission meanwhile is spared the import's cost. Where a held import costs
//!   nothing (NVIDIA's on the RTX 5080 Laptop GPU, below), a release would save nothing and cost
//!   the import again, so every pointer import stays imported. RADV lists every memory it
//!   imported in each kernel submission, and amdgpu gets every page of a pointer import again in
//!   each (`amdgpu_cs_parser_bos`): the probe measured 10.3 us more a submission
//!   for one import of 16 MiB on the Radeon 780M (24.1 for 64 MiB), 6.8 on the RX 7900 XTX
//!   (35.7), 45.8 on the RX 550 (191.3), against 0.2 on the Iris Xe for one import (its cost is
//!   per import, above) and none on the M3 Max.
//! - **Load beside it.** Other work on the device or the host's CPU adds to a submission's time
//!   and never takes from it, so each timed check reads floors, a form's least time in each half of
//!   its rounds or phases, and answers only where the two halves agree. Each half holds several
//!   rounds or phases of each form, so one that load lengthened throughout moves no floor. A try
//!   whose halves disagree is made again, at most [`TRIES`] times; where none agrees, the check
//!   answers `Unchecked` for the device's life and the record is not written, so the next device
//!   measures again. Seen on the Radeon 780M while the checks compared medians of single tries: a
//!   device made while a test's thread made and released buffers in a loop
//!   (`tests/in_place_check.rs`) placed its long buffers in fresh pages, which follows only from
//!   an answer that a held import adds nothing, where the quiet devices of the same run answered
//!   that it adds about 6 us; its record would have kept that answer for the kernel's release. And
//!   on the M3 Max, under the load of the device tests beside it, a comparison of each phase alone
//!   split three tries in a row where its floors over all the phases differed by 0.1 us. Load
//!   that lengthens every submission of the phases with the imports held still passes the tax's
//!   bound where a submission costs little: on lavapipe, beside the device tests, one device
//!   answered `Holds` from 1.4 against 0.2 us and the next device of the same test `Fails` from
//!   2.1 against 1.8 (one import of 16 MiB held, the check's form before it held
//!   [`TAX_IMPORTS`]), and the first's record would have kept its answer for the kernel's
//!   release. And on the RTX 5080 Laptop GPU (NVIDIA 615.71.09, Linux), beside a loop of the
//!   device tests, one device of six answered `Holds` from 3.4 against 1.9 us where its quiet
//!   devices answered `Fails` from 2.1 to 2.2 us held and bare alike (2026-11-09), and that
//!   process took the routes of a taxed device the quiet ones never take. So an import tax
//!   `Holds` stands only where its try's floors pass its bound [`TAX_MARGIN`] times over: short
//!   of that it answers `Fails` for the device's life, unrecorded, and the next device measures
//!   again. Past it, it is recorded only where each of [`TAX_REPEATS`] tries more passes so
//!   again; short of that, it answers for the device's life alone. The same load splits a
//!   try's halves: on the M3 Max, in three runs of the runtime's device tests (2026-11-09,
//!   MoltenVK 1.4.2), one of the six devices that printed their verdicts had all three of its
//!   import tax tries split, its last 2.5 against 1.2 us over all its phases, where the other five
//!   answered `Fails` (1.1 to 2.6 us against 1.2 to 3.0), and it answered `Unchecked`, so its long
//!   buffers took no fresh pages where the others' did. Tries that all split so answer `Fails` for
//!   the device's life, unrecorded, where the last one's floors fall short of the margin, and
//!   `Unchecked` only past it. Load moves the merge's floors either way, its halves agreeing: in
//!   one full run of the runtime's and the validation crate's device tests on the Radeon 780M
//!   (RADV, Mesa 26.2.6, Linux 7.3.1), 32 of 2,531 devices answered `Fails`, one call of four at
//!   0.6 to 1.15 of the calls apart where a quiet device's is 0.35; and on the M3 Max beside loops
//!   of the runtime's device tests (MoltenVK 1.4.2), 145 of 15,181 answered `Holds`, at
//!   0.29 and up where its quiet tries are 0.81 to 0.91 (2026-11-10). One try recorded each, for
//!   the kernel's release. So a merge verdict is recorded only where its try's floors are clear of
//!   that load ([`merge_clear`]: a `Holds` at most [`MERGE_HOLDS_CLEAR`] of the calls apart, a
//!   `Fails` at least [`MERGE_FAILS_CLEAR`]) and each of [`MERGE_REPEATS`] tries more answers it
//!   so ([`merge_tried`]); short of that it answers for the device's life, unrecorded, and the
//!   next device measures again. Those `Holds` on the M3 Max came from floors that were each the
//!   least of a form's rounds, inside one call for the one call of the infos and inside
//!   [`MERGED`] for the calls apart: load that lengthened any one call of a round lengthened its
//!   sum, so each round of the calls apart took a hit about [`MERGED`] times as often, and their
//!   floor rose where the one call's stayed, which reads as a merge. On an adopted MoltenVK
//!   device, whose queues are not relayed, a `Holds` for its life groups its batches
//!   (`command_buffer/group.rs`) on a driver that runs each `VkSubmitInfo` as a Metal command
//!   buffer of its own. So each call apart has a floor of its own, over as many rounds as the
//!   one call's, and the form's floor is their sum ([`apart_floor`]): each floor is the least of
//!   as many calls, equally exposed to the load, and a hit to one call of each round, whichever,
//!   moves none.
//! - **Its verdicts** are `Pending` until it answers, and the device takes the form a driver
//!   without each answer takes: every batch its own call, its long buffers placed and cleared as
//!   ever, every pointer import kept. `Unchecked` where a check could not run (no
//!   `VK_EXT_external_memory_host`, a call refused, too few rounds) or its tries disagreed (the
//!   import tax's only past its margin, above), with the same forms.
//! - **Its cost**: [`ROUNDS`] rounds of each merge form and [`TAX_PHASES`] phases of
//!   [`TAX_SUBMITS`] submissions, each waited for, every command buffer empty; the
//!   [`TAX_IMPORTS`] imports of [`TAX_BYTES`] in all made and freed once a try, over pages
//!   written once, none bound to a buffer or read by a command; each timed check again, at most
//!   [`TRIES`] times in all, where its halves disagree, and the import tax's [`TAX_REPEATS`]
//!   tries more where it answers `Holds` past its margin, and the merge's [`MERGE_REPEATS`] where
//!   its answer is clear of the load beside it. The probe's rounds took 3 to 12 ms on those
//!   devices.
//! - **Kept.** Once every check answered and each timed check's answer earned its record (no
//!   check's tries all disagreed; the merge's answer was clear of the load and repeated, and an
//!   import tax `Holds` passed its margin and repeated, above),
//!   the verdicts are written as one record in the shader cache's content directory
//!   (`shader_cache/store.rs`), named by the driver's identity, the kernel's release (the import
//!   tax is the kernel's work on RADV), the check's version and the runtime's ([`record_name`]):
//!   a device of the same driver, release and kernel reads them as it is made and runs no check.
//!   A failed write leaves the verdicts for the device's life. A device made with
//!   `DeviceOptions::assume_submissions_merge` (a test hook) reads and writes no record: its merge
//!   check holds from its making, the merge's floors only reported, and its other checks answer
//!   for its life.
//! - **Out of the device's way.** Before each step that allocates, frees, maps or unmaps pages,
//!   or submits (each round, each import and its release, each submission of a phase), the
//!   thread steps aside for the device's allocations and frees on other threads
//!   (`step_aside.rs`): it goes on once none is in progress and none ended for
//!   `step_aside::QUIET`, at once while a call waits for its verdicts (`ANSWER_WAIT`,
//!   `Device::settle_submit_checks`). Beside a program's burst of makings its verdicts so come
//!   after the burst. Each step first waits until as long as the thread's own allocations and
//!   frees since the last one took has passed since they ended (`step_aside::OwnCall`): NVIDIA's
//!   driver holds back every other thread's `vkQueueSubmit` while it frees a pointer import, so
//!   the import tax's 32 frees in a row held the device's submissions for their whole run, and a
//!   compositor's command buffers beside long work waited up to 21 ms. Paced, the device's
//!   submissions go between them.
//! - **Its thread** holds nothing of the device, as the queue-switch check's does
//!   (`queue_switch_check.rs`): only the parts of it the checks work with ([`Parts`]), never a
//!   handle, so the device's last handle never goes on it. It reads the stop between its steps
//!   (each round, each phase): the device's teardown stops it there and waits for the
//!   step in progress, the release of its objects and of its parts, and the checks end
//!   unrecorded, what had not answered `Unchecked`. So does the device's loss, at the call that
//!   answers it (`VK_ERROR_DEVICE_LOST`, from a submission or a wait): nothing is submitted again,
//!   since a later submission of the same fence and command buffers would find them pending (as
//!   the validation layer reads a loss), and nothing a lost device measured is recorded. Each
//!   step's Vulkan work runs inside a section of the exit guard (`exit_guard.rs`). Its
//!   submissions are counted apart (`WorkCounts::submit_check_calls`), not among the device's,
//!   with the relay's mark after each of its uses of the queue where the device relays its queues'
//!   failures (`device/relay.rs`, `By::Checks`); the commands it records go through the driver's
//!   own entry points, counted among none of the device's (`WorkCounts::commands_recorded`,
//!   `recorder::unrouted`); its allocations and releases are counted as the checks'
//!   (`WorkCounts::device_checks`), not among the process's (`object_churn`). Its
//!   submission calls hold the device's loader word (`loader_word.rs`).
//! - **A stalled submission.** A wait for one of its submissions, empty command buffers, that
//!   passes [`STALL`] takes the device as lost for the checks: nothing is submitted again, the
//!   checks end unrecorded, what had not answered `Unchecked`, and the instance's sink hears it
//!   (id name [`STALLED_ID`], `Info` severity). The submission's fence and command buffers are
//!   still pending, so the checks' thread leaves them to the device's teardown
//!   ([`SubmitChecks::stop`]), which waits for that fence up to [`STALL`] more and releases them
//!   once it signalled. Where it still has not, the teardown leaves the device, its objects and
//!   its instance to the process's end, as where the process began to exit: its wait for the
//!   device's idle would never return. Seen on the RTX 3090 (NVIDIA 620.70.06, Linux), with
//!   processes making devices with extra compute and video decode queues at once (the HOWTO's
//!   host note): with six, one process's checks waited on their fence with no end, its teardown
//!   behind them; with eight, 5 of 48 processes' checks met a submission that had not completed
//!   at [`STALL`], where every other process's longest wait for one (in
//!   [`Device::submit_check_report`]) was 1.3 to 12.0 ms (lavapipe's 0.05 ms).

use crate::allocations::Allocations;
use crate::command_buffer::fences::Fences;
use crate::device::functions::{Functions, Vk};
use crate::device::relay::{By, QueueUses};
use crate::device::submit_stalls::Uses;
use crate::device::{lock, Device, QueueSlot};
use crate::driver_facts::inform_through;
use crate::exit_guard::{self, Call};
use crate::freed_memory::Parking;
use crate::host_memory::{Importer, PointerImport};
use crate::in_place::DriverIdentity;
use crate::loader_word::{DeviceWord, Held};
use crate::recorder::{self, Command};
use crate::release;
use crate::shader_cache::store::{hex, Lookup, Store};
use crate::work_counts::Counters;
use crate::MessageSink;
use ash::vk;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The command buffers of each merge round, and the `VkSubmitInfo`s of its one call.
pub(crate) const MERGED: usize = 4;

/// The rounds of each merge form, after [`WARM`] of each untimed.
pub(crate) const ROUNDS: usize = 16;

/// The untimed rounds of each form first.
const WARM: usize = 2;

/// The tries of a timed check (the merge, the import tax) whose halves disagree, before
/// it answers `Unchecked` for the device's life and nothing is recorded (the module docs).
pub(crate) const TRIES: usize = 3;

/// The most one call of the merge's infos costs, as a share of its calls apart, where the driver
/// merges them.
pub(crate) const MERGE_RATIO: f64 = 0.6;

/// The most one call of the merge's infos costs, as a share of its calls apart, over a whole try,
/// for a `Holds` to be recorded ([`merge_clear`]): the quiet Radeon 780M's is 0.35.
pub(crate) const MERGE_HOLDS_CLEAR: f64 = 0.5;

/// The least one call of the merge's infos costs, as a share of its calls apart, over a whole try,
/// for a `Fails` to be recorded ([`merge_clear`]): the quiet M3 Max's (MoltenVK 1.4.2) are 0.81
/// to 0.91.
pub(crate) const MERGE_FAILS_CLEAR: f64 = 0.75;

/// The tries more a merge verdict past its clear bound answers again, each so, before it is
/// recorded ([`merge_tried`]).
pub(crate) const MERGE_REPEATS: usize = 2;

/// The bytes of the imports the tax check holds, in all: past each import's own cost, a
/// length's (amdgpu's work a page), half the shortest buffer of the runtime's own that takes fresh
/// pages.
pub(crate) const TAX_BYTES: u64 = 16 << 20;

/// The imports the tax check holds, of [`TAX_BYTES`] in all: a driver's cost per listed object,
/// whatever its length, adds up past [`TAX_FLOOR`] where it is a tenth of it or more (a guest's
/// desktop holds a hundred imports and more).
pub(crate) const TAX_IMPORTS: u64 = 32;

/// The tax check's phases, the middle half of them with the imports held ([`tax_holds`]).
pub(crate) const TAX_PHASES: usize = 8;

/// Whether the tax check's phase `at` holds the imports: the middle half of the phases, so that
/// each half of them holds as many phases with the imports as without, the first and the last
/// without, and the phases that hold them run one after another, served by one set of imports.
/// A drift of the load beside the check over a try lengthens one form in one half as much as the
/// other form in the other half: it splits the halves (no verdict) rather than moving both one
/// way.
pub(crate) fn tax_holds(at: usize) -> bool {
    (TAX_PHASES / 4..TAX_PHASES - TAX_PHASES / 4).contains(&at)
}

/// The submissions timed in each phase, after one untimed.
pub(crate) const TAX_SUBMITS: usize = 8;

/// The least a held import adds to a submission, in nanoseconds, that the tax check counts: a
/// microsecond, about 0.25% of a core at the 2500 submissions a second of a guest's frames.
pub(crate) const TAX_FLOOR: i64 = 1_000;

/// How many times over an import tax `Holds` passes its bound ([`tax_verdict`]) for it to stand,
/// and, repeated, to be recorded: load beside the check that lengthened every submission with the
/// imports held passed the bound by a fifth on lavapipe and by half on the RTX 5080 (the module
/// docs), where the taxes the probe measured pass it six times over and more (its medians: RADV on
/// three GPUs, ANV on i915 with 32 imports held).
pub(crate) const TAX_MARGIN: i64 = 2;

/// The tries more an import tax `Holds` past [`TAX_MARGIN`] answers again, each past it, before
/// it is recorded.
pub(crate) const TAX_REPEATS: usize = 2;

/// How long one wait for a fence lasts before it is repeated.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The longest the checks wait for one of their submissions, empty command buffers on the
/// device's first `VkQueue`, before they take it as stalled (the module docs, "A stalled
/// submission"); and the longest the device's teardown waits for it again before it leaves the
/// device to the process's end ([`SubmitChecks::stop`]). Past their longest wait for one that
/// completed, 12.0 ms on the RTX 3090 with eight processes making devices at once, by about a
/// thousand times; as long as a call waits for their verdicts (`ANSWER_WAIT`).
pub(crate) const STALL: Duration = Duration::from_secs(10);

/// The id name of the message that tells of a stalled submission of the checks.
pub(crate) const STALLED_ID: &str = "metal2vulkan-submit-check-stalled";

/// The longest a making whose path the verdicts choose once for good waits for them (a guest RAM
/// with a source of dma-bufs, `host_ram.rs`): their run takes milliseconds.
pub(crate) const ANSWER_WAIT: Duration = Duration::from_secs(10);

/// A check, in the order they run and the record holds their verdicts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Check {
    /// `VkSubmitInfo`s that signal nothing merge into one submission of the driver's.
    Merge,
    /// A held host-pointer import adds to every submission's cost.
    ImportTax,
}

impl Check {
    /// How many checks there are.
    pub(crate) const COUNT: usize = 2;

    /// Every check, in order.
    pub(crate) const ALL: [Check; Check::COUNT] = [Check::Merge, Check::ImportTax];

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
/// and its calls apart, and one submission with the import held and without (floors, each the
/// least of its form in the try that answered).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Measured {
    pub(crate) merged: i64,
    pub(crate) apart: i64,
    pub(crate) held: i64,
    pub(crate) bare: i64,
}

/// A device's verdicts of its submission checks, and their state (the module docs).
pub(crate) struct SubmitChecks {
    verdicts: [AtomicU8; Check::COUNT],
    measured: [AtomicI64; 4],
    started: AtomicBool,
    settled: Mutex<bool>,
    wake: Condvar,
    /// The checks' run in this process, in nanoseconds; 0 where none ended.
    took: AtomicU64,
    /// The tries whose blocks or phases disagreed, in this process (the module docs).
    split: AtomicU64,
    /// The import tax answered `Holds` past its margin but short of its record, in this process
    /// ([`tax_tried`]).
    unearned: AtomicBool,
    /// The import tax answered `Holds` short of its margin, so `Fails`, in this process
    /// ([`tax_tried`]).
    short: AtomicBool,
    /// The merge answered short of its record, in this process ([`merge_tried`]).
    merge_unearned: AtomicBool,
    /// The merge check holds without its answer, and no record is kept
    /// (`DeviceOptions::assume_submissions_merge`).
    merge_assumed: bool,
    /// The import tax's tries, in this process: each makes and releases [`TAX_IMPORTS`] imports
    /// once ([`tax_holds`]).
    pub(crate) tax_tries: AtomicU64,
    /// The device's teardown stopped the checks ([`SubmitChecks::stop`]): read between steps.
    stopping: AtomicBool,
    /// The calls waiting for the verdicts ([`SubmitChecks::wait`]), for which no step steps
    /// aside (`step_aside.rs`).
    waiters: AtomicU32,
    /// Whether the checks' thread runs, and what tells its end.
    running: Mutex<bool>,
    ending: Condvar,
    /// The longest the checks waited for one of their submissions in this process, in
    /// nanoseconds, a stalled one's wait included ([`STALL`]).
    longest_wait: AtomicU64,
    /// A submission of the checks did not complete within [`STALL`] (the module docs, "A stalled
    /// submission").
    stalled: AtomicBool,
    /// The objects a stalled submission names, still pending, left to the device's teardown
    /// ([`SubmitChecks::stop`]).
    left: Mutex<Option<Objects>>,
    /// The crate's device tests' hold of the checks between their steps
    /// (`tests/teardown_threads.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) held: crate::freed_memory::held::Held,
    /// The next wait for the checks' fence answers as a lost device's (`VK_ERROR_DEVICE_LOST`),
    /// whatever the driver answered: a device test's injection (`tests::submit_check_device`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) lost_at_wait: AtomicBool,
    /// The next submission of the checks waits for this timeline semaphore to reach this value,
    /// so its fence signals only once something signals that: a device test's submission that
    /// never completes (`tests::submit_check_device`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) wait_at_submit: Mutex<Option<(vk::Semaphore, u64)>>,
    /// The bound the checks' waits and the teardown's take in place of [`STALL`]: a device
    /// test's.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) stall: Mutex<Option<Duration>>,
}

impl SubmitChecks {
    /// A device's, every verdict pending; the merge check's `Holds` from the start where
    /// `merge_assumed` (`DeviceOptions::assume_submissions_merge`).
    pub(crate) fn new(merge_assumed: bool) -> SubmitChecks {
        let checks = SubmitChecks {
            verdicts: Default::default(),
            measured: Default::default(),
            started: AtomicBool::new(false),
            settled: Mutex::new(false),
            wake: Condvar::new(),
            took: AtomicU64::new(0),
            split: AtomicU64::new(0),
            unearned: AtomicBool::new(false),
            short: AtomicBool::new(false),
            merge_unearned: AtomicBool::new(false),
            merge_assumed,
            tax_tries: AtomicU64::new(0),
            stopping: AtomicBool::new(false),
            waiters: AtomicU32::new(0),
            running: Mutex::new(false),
            ending: Condvar::new(),
            longest_wait: AtomicU64::new(0),
            stalled: AtomicBool::new(false),
            left: Mutex::new(None),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            held: Default::default(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            lost_at_wait: AtomicBool::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            wait_at_submit: Mutex::new(None),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            stall: Mutex::new(None),
        };
        if merge_assumed {
            checks.set(Check::Merge, Verdict::Holds);
        }
        checks
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

    /// Whether the device's teardown stopped the checks: read between their steps.
    fn stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    /// Whether a call waits for the verdicts ([`SubmitChecks::wait`]).
    fn awaited(&self) -> bool {
        self.waiters.load(Ordering::Acquire) > 0
    }

    /// The checks' thread runs, or (`false`) ended: the device's teardown waits for its end.
    fn set_running(&self, running: bool) {
        *lock(&self.running) = running;
        if !running {
            self.ending.notify_all();
        }
    }

    /// The device's teardown, inside its section of the exit guard: the checks stop at their
    /// next step, and this waits until their thread ended, its objects released; whether the
    /// device's work may be waited for. That thread holds no device handle (its [`Parts`]), so it
    /// never runs this teardown. Where a submission of theirs stalled, this waits for it up to
    /// [`STALL`] more and releases the objects it names once it completed. Where it still has
    /// not, or the process began to exit meanwhile, `false`: a wait for the device's idle would
    /// wait for it, so the device and those objects are left to the process's end (the module
    /// docs, "A stalled submission").
    pub(crate) fn stop(&self, raw: &Vk) -> bool {
        self.stopping.store(true, Ordering::SeqCst);
        {
            let mut running = lock(&self.running);
            while *running {
                running =
                    (self.ending.wait(running)).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
        let left = lock(&self.left).take();
        left.is_none_or(|left| left.release_once_complete(raw, self.bound()))
    }

    /// The longest the checks wait for one of their submissions, and the teardown for a stalled
    /// one: [`STALL`], or a device test's bound.
    fn bound(&self) -> Duration {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if let Some(bound) = *lock(&self.stall) {
            return bound;
        }
        STALL
    }

    /// A wait of the checks for one of their submissions took `waited`: the longest is kept.
    fn waited(&self, waited: Duration) {
        let nanos = u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX);
        self.longest_wait.fetch_max(nanos, Ordering::Relaxed);
    }

    /// Whether a submission of the checks stalled ([`STALL`]).
    pub(crate) fn stalled(&self) -> bool {
        self.stalled.load(Ordering::Acquire)
    }

    /// Whether the objects a stalled submission names wait for the device's teardown: a device
    /// test's.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn left_to_teardown(&self) -> bool {
        lock(&self.left).is_some()
    }

    /// Let the tests' hold of the checks go once the device's teardown stopped them, or once it
    /// was let go already.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn release_once_stopping(&self) {
        self.held.release_once(&self.stopping);
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

    /// Wait until the verdicts are known, or `timeout` passed: whether they are. Meanwhile the
    /// checks step aside for nothing (`step_aside.rs`).
    fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        self.waiters.fetch_add(1, Ordering::AcqRel);
        let _waiting = Waiting(&self.waiters);
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

/// The merge's verdict from its two forms' floors in one block.
pub(crate) fn merge_verdict(merged: i64, apart: i64) -> Verdict {
    match apart > 0 && merged as f64 <= MERGE_RATIO * apart as f64 {
        true => Verdict::Holds,
        false => Verdict::Fails,
    }
}

/// Whether a merge try's `floors` over the whole try, `(merged, apart)`, are clear of the load
/// beside it for `verdict` to be recorded: a `Holds` at most [`MERGE_HOLDS_CLEAR`] of the calls
/// apart, a `Fails` at least [`MERGE_FAILS_CLEAR`] of them ([`merge_tried`]).
pub(crate) fn merge_clear(verdict: Verdict, (merged, apart): (i64, i64)) -> bool {
    let share = |bound: f64| bound * apart as f64;
    match verdict {
        Verdict::Holds => apart > 0 && merged as f64 <= share(MERGE_HOLDS_CLEAR),
        Verdict::Fails => apart > 0 && merged as f64 >= share(MERGE_FAILS_CLEAR),
        _ => false,
    }
}

/// The import tax's verdict from the floors of a phase with the import held and of one without.
pub(crate) fn tax_verdict(held: i64, bare: i64) -> Verdict {
    match held - bare > tax_bound(bare) {
        true => Verdict::Holds,
        false => Verdict::Fails,
    }
}

/// The most a held import adds to a submission of `bare` nanoseconds without it that is no tax:
/// [`TAX_FLOOR`], or an eighth of the submission, whichever is more.
fn tax_bound(bare: i64) -> i64 {
    TAX_FLOOR.max(bare / 8)
}

/// Whether floors `(held, bare)` pass the import tax's bound [`TAX_MARGIN`] times over: what
/// each try of a `Holds` that is recorded answers ([`tax_tried`]).
pub(crate) fn tax_clear((held, bare): (i64, i64)) -> bool {
    held - bare > TAX_MARGIN * tax_bound(bare)
}

/// What one try of a timed check answered: the verdict its two halves agree on (`None` where they
/// disagree), and its two forms' floors over the whole try, in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Timed {
    pub(crate) verdict: Option<Verdict>,
    pub(crate) floors: (i64, i64),
}

/// One merge try's answer from its rounds' times, in nanoseconds, of one call of the infos and
/// of each of the calls apart: each half of the rounds compares the two forms' floors
/// ([`merge_verdict`]), the calls apart's the sum of each call's own ([`apart_floor`]).
pub(crate) fn merge_try(merged: &[i64], apart: &[[i64; MERGED]]) -> Timed {
    let (merged_first, merged_second) = merged.split_at(merged.len() / 2);
    let (apart_first, apart_second) = apart.split_at(apart.len() / 2);
    Timed {
        verdict: agree(
            merge_verdict(floor(merged_first), apart_floor(apart_first)),
            merge_verdict(floor(merged_second), apart_floor(apart_second)),
        ),
        floors: (floor(merged), apart_floor(apart)),
    }
}

/// The floor of the calls apart over `rounds`: the sum of each call's floor, the least of the
/// call at its place in every round (the first, the second... the last, with the fence). Each
/// call so has as many tries to come out clear of the load beside the check as the one call of
/// the infos, and load that lengthens one of the calls of every round (a different one in each)
/// moves no floor, where the least of the rounds' sums took that load in every round (the module
/// docs, "Load beside it").
pub(crate) fn apart_floor(rounds: &[[i64; MERGED]]) -> i64 {
    (0..MERGED)
        .map(|at| (rounds.iter()).map(|calls| calls[at]).min().unwrap_or(0))
        .sum()
}

/// One import tax try's answer from its phases' submission times, in nanoseconds, those
/// [`tax_holds`] names with the imports held: in each half of the phases, the floor of those with
/// them held against that of those without ([`tax_verdict`]).
pub(crate) fn tax_try(phases: &[Vec<i64>]) -> Timed {
    // The floor of the phases of `phases` with the import held (`held`) or without, by their
    // place among all the phases.
    let of = |phases: &[(usize, &Vec<i64>)], held: bool| {
        let times = phases.iter().filter(|&&(at, _)| tax_holds(at) == held);
        times.map(|(_, times)| floor(times)).min().unwrap_or(0)
    };
    let placed: Vec<(usize, &Vec<i64>)> = phases.iter().enumerate().collect();
    let (first, second) = placed.split_at(placed.len() / 2);
    let verdict = |phases: &[(usize, &Vec<i64>)]| tax_verdict(of(phases, true), of(phases, false));
    Timed {
        verdict: agree(verdict(first), verdict(second)),
        floors: (of(&placed, true), of(&placed, false)),
    }
}

/// The verdict both halves give; `None` where they differ.
fn agree(first: Verdict, second: Verdict) -> Option<Verdict> {
    (first == second).then_some(first)
}

/// The least of `times`, the time no load beside the check added to (it only adds); 0 for none.
fn floor(times: &[i64]) -> i64 {
    times.iter().min().copied().unwrap_or(0)
}

/// The version of what the checks measure, of what earns a record and of how it is named and
/// framed: a change to any bumps it, and a record of another version is never read (one of 3 may
/// hold an import tax `Holds` short of [`TAX_MARGIN`], one of 5 a merge verdict of one loaded
/// try, one of 6 three verdicts, one of 7 a merge verdict from the least of the calls apart's
/// sums rather than [`apart_floor`]).
pub(crate) const CHECK_VERSION: u32 = 8;

/// A record's first bytes.
const MAGIC: [u8; 8] = *b"M2VSUBC\0";

/// The domain tag of a record's name.
const NAME_TAG: &[u8] = b"metal2vulkan submission checks record\0";

/// A record's length: magic, version, runtime, identity, verdicts, measured, checksum.
const RECORD_BYTES: usize = 8 + 4 + 4 + 32 + Check::COUNT + 4 * 8 + 32;

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
/// verdicts       2 bytes   one per check, in Check::ALL's order: Holds 1, Fails 2, Unchecked 3
/// measured       4 x i64   merged, apart, held, bare, in nanoseconds
/// checksum       32 bytes  SHA-256 of every byte before it
/// ```
///
/// Integers are little-endian.
pub(crate) fn encode(
    identity: &DriverIdentity,
    verdicts: [Verdict; Check::COUNT],
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
pub(crate) fn decode(
    identity: &DriverIdentity,
    bytes: &[u8],
) -> Option<([Verdict; Check::COUNT], Measured)> {
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
    let (verdicts, rest) = rest.split_at(Check::COUNT);
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap_or([0xff; 4]));
    if magic != MAGIC
        || word(version) != CHECK_VERSION
        || word(runtime) != metal2vulkan::cache::RUNTIME_VERSION
        || identity_digest != digest(identity)
    {
        return None;
    }
    let verdicts: [Verdict; Check::COUNT] = std::array::from_fn(|i| Verdict::of_byte(verdicts[i]));
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
    /// checks start on a thread of their own. A record that does not decode is no record. Where
    /// the merge check holds without its answer (`DeviceOptions::assume_submissions_merge`), no
    /// record is read: the checks run, and keep none.
    pub(crate) fn recall_submit_checks(&self) {
        let shared = self.shared();
        if shared.submit_checks.merge_assumed {
            self.start_submit_checks();
            return;
        }
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
        // Their command buffers, made here, as the device is made (`Made`).
        let Some(made) = Made::make(self) else {
            checks.settle_unchecked();
            return;
        };
        let pool = made.pool;
        let parts = Parts::of(self);
        let builder = std::thread::Builder::new().name("m2v-submit-check".to_owned());
        checks.set_running(true);
        let spawned = exit_guard::spawn(builder, move || {
            // Made first, so it goes last: the parts' shares go before the teardown that waits
            // for the thread's end goes on, however the run ends.
            let _ends = Ends(Arc::clone(&parts.checks));
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            crate::loader_word::on_a_check_s_thread();
            crate::step_aside::mark_a_check_s_thread(crate::step_aside::Counted::Apart);
            let parts = parts;
            run(&parts, made);
        });
        if spawned.is_err() {
            // SAFETY: the pool is the checks' and nothing of it was used.
            unsafe { release::object(self.shared().vk(), pool) };
            checks.set_running(false);
            checks.settle_unchecked();
        }
    }

    /// Whether the device's `VkSubmitInfo`s that signal nothing merge into one submission of the
    /// driver's, as its check found (`submit_check.rs`), so that its batches may wait in a group
    /// (`CompletionPolicy::Grouped`): `false` until it answers. A runtime extension for the lanes
    /// that hold a device to the path it chooses.
    #[doc(hidden)]
    pub fn submissions_merge(&self) -> bool {
        self.current().submissions_merge_now()
    }

    /// [`Device::submissions_merge`] on the generation that serves (`device/line.rs`).
    pub(crate) fn submissions_merge_now(&self) -> bool {
        self.shared().submit_checks.holds(Check::Merge)
    }

    /// Whether the device's long buffers of its own take fresh pages (`fresh_pages.rs`): its
    /// check found a held import adds nothing to a submission's cost (`submit_check.rs`); `false`
    /// until it answers. A runtime extension for the lanes that hold a device to the path it
    /// chooses.
    #[doc(hidden)]
    pub fn fresh_pages_pay(&self) -> bool {
        self.current().fresh_pages_pay_now()
    }

    /// [`Device::fresh_pages_pay`] on the generation that serves (`device/line.rs`).
    pub(crate) fn fresh_pages_pay_now(&self) -> bool {
        self.shared().submit_checks.fails(Check::ImportTax)
    }

    /// Whether a host-pointer import the device holds adds to the cost of every submission, as
    /// its check found (`submit_check.rs`): where it does, a guest RAM with a source of dma-bufs
    /// imports its chunks as dma-bufs first (`host_ram.rs`), and a no-copy buffer's idle pointer
    /// import is released and imported again at its next blit (`host_residency.rs`). `false` until
    /// it answers.
    pub(crate) fn pointer_imports_tax_submissions(&self) -> bool {
        self.shared().submit_checks.holds(Check::ImportTax)
    }

    /// Whether the device releases a no-copy buffer's idle import of its pointer and imports it
    /// again at its next blit (`host_residency.rs`): its check found a held import adds to every
    /// submission's cost (`submit_check.rs`); `false` until it answers, and for a buffer made
    /// before then. A runtime extension for the lanes that hold a device to the path it chooses.
    #[doc(hidden)]
    pub fn releases_idle_pointer_imports(&self) -> bool {
        self.current().pointer_imports_tax_submissions()
    }

    /// Wait until the device's submission checks answered, or `timeout` passed: whether they did.
    /// A runtime extension for the lanes and benchmarks that hold a device to the paths its
    /// verdicts choose.
    #[doc(hidden)]
    pub fn settle_submit_checks(&self, timeout: Duration) -> bool {
        self.current().settle_submit_checks_now(timeout)
    }

    /// [`Device::settle_submit_checks`] on the generation that serves (`device/line.rs`).
    pub(crate) fn settle_submit_checks_now(&self, timeout: Duration) -> bool {
        self.shared().submit_checks.wait(timeout)
    }

    /// The device's submission checks' verdicts and measurements, and how long they ran in this
    /// process, in words. A runtime extension for the lanes and benchmarks that report them.
    #[doc(hidden)]
    pub fn submit_check_report(&self) -> String {
        self.current().submit_check_report_now()
    }

    /// [`Device::submit_check_report`] on the generation that serves (`device/line.rs`).
    pub(crate) fn submit_check_report_now(&self) -> String {
        let checks = &self.shared().submit_checks;
        let m = checks.measured();
        let us = |nanos: i64| nanos as f64 / 1e3;
        let took = match checks.took.load(Ordering::Acquire) {
            0 => "not run here".to_owned(),
            nanos => format!("ran {:.2} ms", nanos as f64 / 1e6),
        };
        let split = match checks.split.load(Ordering::Relaxed) {
            0 => String::new(),
            tries => format!(", {tries} tries split"),
        };
        let unearned = match checks.unearned.load(Ordering::Relaxed) {
            true => ", its import tax Holds short of a record",
            false => "",
        };
        let short = match checks.short.load(Ordering::Relaxed) {
            true => ", its import tax Holds short of its margin",
            false => "",
        };
        let merge_unearned = match checks.merge_unearned.load(Ordering::Relaxed) {
            true => ", its merge answer short of a record",
            false => "",
        };
        let tax_tries = match checks.tax_tries.load(Ordering::Relaxed) {
            0 => String::new(),
            tries => format!(", {tries} import tax tries"),
        };
        let assumed = match checks.merge_assumed {
            true => " assumed",
            false => "",
        };
        let longest = match checks.longest_wait.load(Ordering::Relaxed) {
            0 => String::new(),
            nanos => format!(
                ", its longest wait for a submission {:.2} ms",
                nanos as f64 / 1e6
            ),
        };
        let stalled = match checks.stalled() {
            true => ", a submission stalled",
            false => "",
        };
        format!(
            "merge {:?}{assumed} ({:.1} us in one call of {MERGED} against {:.1} us apart), import tax \
             {:?} ({:.1} us a submission with {} imports of {} MiB in all held against {:.1} us), \
             {took}{split}{merge_unearned}{unearned}{short}{tax_tries}{longest}{stalled} (floors)",
            checks.verdict(Check::Merge),
            us(m.merged),
            us(m.apart),
            checks.verdict(Check::ImportTax),
            us(m.held),
            TAX_IMPORTS,
            TAX_BYTES >> 20,
            us(m.bare),
        )
    }
}

/// The parts of a device its checks work with, none of which holds the device (the module
/// docs): shares of its functions, its `VkQueue`s and what a use of one reaches, its loader
/// word, its verdicts and its shader cache's store, and what it reads of the device as the checks
/// start.
struct Parts {
    functions: Arc<Functions>,
    queues: Arc<[QueueSlot]>,
    parking: Option<Arc<Parking>>,
    counts: Arc<Counters>,
    fences: Arc<Fences>,
    loader_word: Arc<DeviceWord>,
    checks: Arc<SubmitChecks>,
    /// The device's allocations, whose work in progress the checks step aside for
    /// (`step_aside.rs`).
    allocations: Arc<Allocations>,
    store: Arc<Store>,
    /// The device's import of host pointers, where it imports them.
    pointers: Option<PointerImport>,
    /// Its memory types' properties, by index.
    type_flags: Vec<vk::MemoryPropertyFlags>,
    identity: DriverIdentity,
    /// The commands the checks record their command buffers with.
    recording: Recording,
    /// The instance's sink, which hears a stalled submission (the module docs).
    sink: Option<MessageSink>,
}

/// The commands the checks record their own command buffers with, as the device's driver gave
/// them (`recorder::unrouted`): counted among none of the device's
/// (`WorkCounts::commands_recorded`), as their submissions are among none of its submissions.
#[derive(Clone, Copy)]
struct Recording {
    begin: vk::PFN_vkBeginCommandBuffer,
    end: vk::PFN_vkEndCommandBuffer,
}

impl Recording {
    /// `shared`'s.
    fn of(shared: &crate::device::DeviceShared) -> Recording {
        let slot = shared.recording();
        let commands = shared.vk().fp_v1_0();
        // SAFETY: each is the device's command of its name, of its signature.
        unsafe {
            Recording {
                begin: recorder::unrouted(
                    slot,
                    Command::BeginCommandBuffer,
                    commands.begin_command_buffer,
                ),
                end: recorder::unrouted(
                    slot,
                    Command::EndCommandBuffer,
                    commands.end_command_buffer,
                ),
            }
        }
    }
}

impl Parts {
    /// `device`'s.
    fn of(device: &Device) -> Parts {
        let shared = device.shared();
        let type_flags = {
            let memory = lock(&shared.memory);
            (0..vk::MAX_MEMORY_TYPES as u32)
                .map_while(|index| memory.type_flags(index))
                .collect()
        };
        Parts {
            functions: shared.functions(),
            queues: Arc::clone(&shared.queues),
            parking: shared.parking.get().cloned(),
            counts: Arc::clone(&shared.counts),
            fences: Arc::clone(&shared.fences),
            loader_word: Arc::clone(&shared.loader_word),
            checks: Arc::clone(&shared.submit_checks),
            allocations: Arc::clone(&shared.allocations.allocations),
            store: shared.shader_cache.content_store_share(),
            pointers: (shared.host_import.as_ref().ok())
                .and_then(Importer::pointers)
                .cloned(),
            type_flags,
            identity: device.driver_identity(),
            recording: Recording::of(shared),
            sink: shared.instance.sink().cloned(),
        }
    }

    /// A submission of the checks did not complete within the bound after `waited` ([`STALL`]):
    /// noted, and the sink told, outside every section of the exit guard.
    fn stall(&self, waited: Duration) {
        let checks = &*self.checks;
        checks.waited(waited);
        checks.stalled.store(true, Ordering::Release);
        let text = format!(
            "the device's submission checks waited {:.1} s for a submission of an empty command \
             buffer on its first VkQueue (family {}), which did not complete: they stop, nothing \
             is submitted again, each check not answered is Unchecked for the device's life, and \
             nothing is recorded; the device's teardown waits for that submission up to {:.1} s \
             more and, where it still has not completed, leaves the device to the process's end",
            waited.as_secs_f64(),
            self.queues[0].family,
            checks.bound().as_secs_f64(),
        );
        inform_through(self.sink.as_ref(), STALLED_ID, text);
    }

    /// The device's functions.
    fn vk(&self) -> &Vk {
        self.functions.get()
    }

    /// What a use of one of the device's `VkQueue`s reaches.
    fn queue_uses(&self) -> QueueUses<'_> {
        QueueUses {
            queues: &self.queues,
            parking: self.parking.as_deref(),
            counts: &self.counts,
            fences: &self.fences,
            functions: &self.functions,
        }
    }

    /// A hold of the device's loader word across a step, where the word was found overwritten
    /// (`loader_word.rs`): none elsewhere.
    fn step_held(&self) -> Option<Held<'_>> {
        // SAFETY: the device is live while its checks run (its teardown stops them first).
        unsafe { self.loader_word.step_hold(None) }
    }

    /// Whether the checks go on to their next step (the device's teardown stops them), once they
    /// stepped aside for the device's other work ([`Parts::step`]).
    fn goes_on(&self) -> bool {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        self.checks.held.pass();
        self.step();
        !self.checks.stopping()
    }

    /// Step aside for the device's allocations and frees on other threads before a step that
    /// allocates, frees, maps or submits (`step_aside.rs`): not while the checks stop or a call
    /// waits for their verdicts. Counted where it waited.
    fn step(&self) {
        let checks = &*self.checks;
        let activity = self.allocations.activity();
        if let Some(waited) = activity.step_aside(|| checks.stopping() || checks.awaited()) {
            self.counts.check_stepped_aside(waited);
        }
    }
}

/// A call waiting for the verdicts ([`SubmitChecks::wait`]), counted until it returns.
struct Waiting<'a>(&'a AtomicU32);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Tells the device's teardown that the checks' thread ended, however it ends.
struct Ends(Arc<SubmitChecks>);

impl Drop for Ends {
    fn drop(&mut self) {
        self.0.set_running(false);
    }
}

/// The checks' thread: each check's verdict given as it is known, then the record. Stopped by
/// the device's teardown, it settles what it had not answered `Unchecked` and records nothing.
fn run(parts: &Parts, made: Made) {
    let checks = &*parts.checks;
    let started = Instant::now();
    parts.step();
    let Some(objects) = Objects::make(parts, made) else {
        checks.settle_unchecked();
        return;
    };
    let ran = measure(parts, &objects);
    if checks.stalled() {
        // The stalled submission's fence and command buffers are still pending: they go with the
        // device's teardown, once it sees it complete ([`SubmitChecks::stop`]).
        *lock(&checks.left) = Some(objects);
        checks.settle_unchecked();
        return;
    }
    parts.step();
    let released = objects.release(parts);
    let Some((measured, kept)) = ran.filter(|_| released) else {
        // The process is exiting, or the device's teardown stopped the checks: nothing is kept.
        checks.settle_unchecked();
        return;
    };
    checks.set_measured(measured);
    parts.counts.submit_checks_ran();
    let verdicts = Check::ALL.map(|check| checks.verdict(check));
    if kept && !checks.merge_assumed && !verdicts.contains(&Verdict::Pending) {
        let identity = &parts.identity;
        let bytes = encode(identity, verdicts, measured);
        // A failed write leaves the verdicts to the device's life alone.
        let _ = parts.store.write_record_soon(&record_name(identity), bytes);
    }
    checks.settle(Some(started.elapsed()));
}

/// Each check in turn, its verdict set as it is known: what they measured, and whether their
/// verdicts may be recorded (no timed check's tries all disagreed, an import tax `Holds` earned
/// its record: [`tax_tried`]); `None` where the process is exiting, or the device's teardown or
/// its loss stopped the checks.
fn measure(parts: &Parts, objects: &Objects) -> Option<(Measured, bool)> {
    let checks = &parts.checks;
    let (merges, (merged, apart), merge_kept) = merge_tried(checks, || merge(parts, objects))?;
    if !checks.merge_assumed {
        checks.set(Check::Merge, merges);
    }
    let (taxed, (held, bare), tax_kept) = tax_tried(checks, || {
        checks.tax_tries.fetch_add(1, Ordering::Relaxed);
        tax(parts, objects)
    })?;
    checks.set(Check::ImportTax, taxed);
    let measured = Measured {
        merged,
        apart,
        held,
        bare,
    };
    Some((measured, merge_kept && tax_kept))
}

/// A timed check's verdict and floors over at most [`TRIES`] tries of `once`, and whether they
/// may be recorded: the first try whose blocks or phases agree answers. Where none does, the
/// check is `Unchecked` for the device's life and not recorded (load beside the check may have
/// split them, and the next device measures again). `Unchecked`, recorded, where a call was
/// refused (`once` answered `Some(None)`); `None` where the check stops.
pub(crate) fn tried(
    checks: &SubmitChecks,
    mut once: impl FnMut() -> Option<Option<Timed>>,
) -> Option<(Verdict, (i64, i64), bool)> {
    let mut floors = (0, 0);
    for _ in 0..TRIES {
        let Some(timed) = once()? else {
            return Some((Verdict::Unchecked, (0, 0), true));
        };
        floors = timed.floors;
        if let Some(verdict) = timed.verdict {
            return Some((verdict, floors, true));
        }
        checks.split.fetch_add(1, Ordering::Relaxed);
    }
    Some((Verdict::Unchecked, floors, false))
}

/// The merge's verdict and floors over its tries of `once` ([`tried`]), and whether they may be
/// recorded (the module docs, "Load beside it"). A `Holds` or a `Fails` is recorded only where its
/// try's floors are clear of the load beside it ([`merge_clear`]) and each of [`MERGE_REPEATS`]
/// tries more answers it so again; short of that it answers for the device's life alone, with the
/// floors of its first try, and the next device measures again; a refused call ends the repeats,
/// unrecorded. Tries that all split, or a refused call, answer as [`tried`] does. `None` where the
/// check stops.
pub(crate) fn merge_tried(
    checks: &SubmitChecks,
    mut once: impl FnMut() -> Option<Option<Timed>>,
) -> Option<(Verdict, (i64, i64), bool)> {
    let (verdict, floors, kept) = tried(checks, &mut once)?;
    if verdict == Verdict::Unchecked {
        return Some((verdict, floors, kept));
    }
    let mut earned = merge_clear(verdict, floors);
    for _ in 0..MERGE_REPEATS {
        if !earned {
            break;
        }
        earned = match once()? {
            Some(again) => again.verdict == Some(verdict) && merge_clear(verdict, again.floors),
            None => false,
        };
    }
    if !earned {
        checks.merge_unearned.store(true, Ordering::Relaxed);
    }
    Some((verdict, floors, earned))
}

/// The import tax's verdict and floors over its tries of `once` ([`tried`]), and whether they may
/// be recorded (the module docs, "Load beside it"). A `Holds` stands only where its floors pass
/// the bound [`TAX_MARGIN`] times over ([`tax_clear`]), and is recorded only where each of
/// [`TAX_REPEATS`] tries more answers `Holds` so again; short of the repeats it answers for the
/// device's life alone, with the floors of its first try, and the next device measures again; a
/// refused call ends the repeats, unrecorded. A `Holds` short of the margin, and tries that all
/// split where the last one's floors over all its phases fall short of it, answer `Fails` for the
/// device's life, unrecorded: a cost no try told from the load beside it. Tries that all split
/// past it answer `Unchecked` so. `None` where the check stops.
pub(crate) fn tax_tried(
    checks: &SubmitChecks,
    mut once: impl FnMut() -> Option<Option<Timed>>,
) -> Option<(Verdict, (i64, i64), bool)> {
    let (verdict, floors, kept) = tried(checks, &mut once)?;
    let split = verdict == Verdict::Unchecked && !kept;
    if (split || verdict == Verdict::Holds) && !tax_clear(floors) {
        if verdict == Verdict::Holds {
            checks.short.store(true, Ordering::Relaxed);
        }
        return Some((Verdict::Fails, floors, false));
    }
    if verdict != Verdict::Holds {
        return Some((verdict, floors, kept));
    }
    let mut earned = true;
    for _ in 0..TAX_REPEATS {
        earned = match once()? {
            Some(again) => again.verdict == Some(Verdict::Holds) && tax_clear(again.floors),
            None => false,
        };
        if !earned {
            break;
        }
    }
    if !earned {
        checks.unearned.store(true, Ordering::Relaxed);
    }
    Some((verdict, floors, earned))
}

/// One merge try ([`merge_try`]) of one call of [`MERGED`] infos against [`MERGED`] calls.
/// `Some(None)` where a call was refused; `None` where the check stops.
fn merge(parts: &Parts, objects: &Objects) -> Option<Option<Timed>> {
    let mut merged = Vec::with_capacity(ROUNDS);
    let mut apart = Vec::with_capacity(ROUNDS);
    for round in 0..WARM + ROUNDS {
        for together in [true, false] {
            if !parts.goes_on() {
                return None;
            }
            let Some(calls) = objects.submit_round(parts, together)? else {
                return Some(None);
            };
            if round >= WARM {
                match together {
                    true => merged.push(calls.iter().sum()),
                    false => apart.push(std::array::from_fn(|at| calls[at])),
                }
            }
        }
    }
    Some(Some(merge_try(&merged, &apart)))
}

/// One import tax try ([`tax_try`]) of a submission with the [`TAX_IMPORTS`] imports of
/// [`TAX_BYTES`] in all held, and without. `Some(None)` where the device imports no host pointer,
/// the system maps no pages or a call was refused; `None` where the check stops.
fn tax(parts: &Parts, objects: &Objects) -> Option<Option<Timed>> {
    let Some(pieces) = (0..TAX_IMPORTS)
        .map(|_| {
            parts.step();
            Pages::map(TAX_BYTES / TAX_IMPORTS)
        })
        .collect::<Option<Vec<Pages>>>()
    else {
        return Some(None);
    };
    for pages in &pieces {
        pages.fill(0x5a);
    }
    let mut phases = Vec::with_capacity(TAX_PHASES);
    // One set of imports, made before the first phase that holds them and released after the
    // last ([`tax_holds`]).
    let mut imports = Vec::new();
    for phase in 0..TAX_PHASES {
        if !parts.goes_on() {
            return release_all(parts, imports.drain(..)).and(None);
        }
        if tax_holds(phase) && imports.is_empty() {
            for pages in &pieces {
                parts.step();
                match Import::make(parts, pages) {
                    Some(Some(import)) => imports.push(import),
                    made => return release_all(parts, imports.drain(..)).and(made.map(|_| None)),
                }
            }
        }
        let mut times = Vec::with_capacity(TAX_SUBMITS);
        let mut refused = false;
        for submission in 0..=TAX_SUBMITS {
            parts.step();
            let Some(submitted) = objects.submit_one(parts) else {
                return release_all(parts, imports.drain(..)).and(None);
            };
            match submitted {
                Some(nanos) if submission > 0 => times.push(nanos),
                Some(_) => {}
                None => {
                    refused = true;
                    break;
                }
            }
        }
        if refused || !tax_holds(phase + 1) {
            release_all(parts, imports.drain(..))?;
        }
        if refused {
            return Some(None);
        }
        phases.push(times);
    }
    Some(Some(tax_try(&phases)))
}

/// Release the tax check's `imports` in order, each step stepping aside first: `None` where the
/// process is exiting, the rest then left to its end.
fn release_all<'p>(parts: &Parts, imports: impl IntoIterator<Item = Import<'p>>) -> Option<()> {
    for import in imports {
        parts.step();
        import.release(parts)?;
    }
    Some(())
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

/// An import of a [`Pages`]' bytes by their pointer, in its own `VkDeviceMemory`, bound to
/// nothing.
struct Import<'p> {
    memory: vk::DeviceMemory,
    pages: &'p Pages,
}

impl<'p> Import<'p> {
    /// The import of `pages`, in a host-coherent type the pointer takes. `Some(None)` where the
    /// device imports no host pointer, the pages take no such type, or a call was refused; `None`
    /// where the process is exiting.
    fn make(parts: &Parts, pages: &'p Pages) -> Option<Option<Import<'p>>> {
        let Some(pointers) = &parts.pointers else {
            return Some(None);
        };
        let _held = parts.step_held();
        let _busy = exit_guard::enter()?;
        let raw = parts.vk();
        // SAFETY: the pages are this process's, at a host page, whose size the import alignment
        // divides where the device imports at all (it is checked below).
        let Ok(types) = (unsafe { pointers.memory_types(raw, pages.start) }) else {
            return Some(None);
        };
        let unit = pointers.unit();
        if !pages.length.is_multiple_of(unit) || !(pages.start.as_ptr() as u64).is_multiple_of(unit)
        {
            return Some(None);
        }
        let coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let type_index = (parts.type_flags.iter().zip(0u32..))
            .find(|&(flags, index)| types & (1 << index) != 0 && flags.contains(coherent))
            .map(|(_, index)| index);
        let Some(type_index) = type_index else {
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
                return Some(None);
            }
        };
        exit_guard::called(Call::AllocateMemory);
        pages.imports.set(pages.imports.get() + 1);
        Some(Some(Import { memory, pages }))
    }

    /// Free it, inside a section: `None` where the process is exiting, the import then left to
    /// its end with its pages mapped.
    fn release(self, parts: &Parts) -> Option<()> {
        let _held = parts.step_held();
        let _busy = exit_guard::enter()?;
        let raw = parts.vk();
        // SAFETY: no submission names it (none binds it).
        unsafe { release::object(raw, self.memory) };
        self.pages.imports.set(self.pages.imports.get() - 1);
        Some(())
    }
}

/// The checks' command pool, of the first `VkQueue`'s family, and its [`MERGED`] empty command
/// buffers, made as the checks start, as the device is made, before the program holds it: their
/// `vkAllocateCommandBuffers` is the one call of the checks' that the loader serves, which reads
/// the device's loader word (`loader_word.rs`), so none of theirs reads it beside the program's
/// work, whose messages MoltenVK writes it with. The checks' thread records them, once
/// ([`Objects::make`]).
struct Made {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
}

impl Made {
    /// `device`'s, on the thread that makes it; `None` where a call was refused (the pool then
    /// released).
    fn make(device: &Device) -> Option<Made> {
        let shared = device.shared();
        let raw = shared.vk();
        let info = vk::CommandPoolCreateInfo::default().queue_family_index(shared.queues[0].family);
        // SAFETY: the device is live.
        let pool = unsafe { raw.create_command_pool(&info, None) }.ok()?;
        exit_guard::called(Call::CreateCommandPool);
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MERGED as u32);
        // The loader serves it; the device's loader word is put back first.
        match device.allocate_command_buffers(&allocate) {
            Ok(buffers) => Some(Made { pool, buffers }),
            Err(_) => {
                // SAFETY: the pool is the checks', none of its command buffers made.
                unsafe { release::object(raw, pool) };
                None
            }
        }
    }
}

/// The checks' Vulkan objects: their command pool and command buffers ([`Made`]), recorded, and a
/// fence.
struct Objects {
    pool: vk::CommandPool,
    empty: Vec<vk::CommandBuffer>,
    fence: vk::Fence,
}

impl Objects {
    /// The objects of `made`, the empty command buffers recorded; `None` where a call was refused
    /// (what was made released) or the process is exiting (`made` then left to its end).
    fn make(parts: &Parts, made: Made) -> Option<Objects> {
        let _held = parts.step_held();
        let _busy = exit_guard::enter()?;
        let mut objects = Objects {
            pool: made.pool,
            empty: made.buffers,
            fence: vk::Fence::null(),
        };
        if objects.fill(parts).is_none() {
            drop(_busy);
            objects.release(parts);
            return None;
        }
        Some(objects)
    }

    /// Record the command buffers and make the fence, inside the caller's section.
    fn fill(&mut self, parts: &Parts) -> Option<()> {
        let raw = parts.vk();
        let begin = vk::CommandBufferBeginInfo::default();
        let recording = parts.recording;
        for &buffer in &self.empty {
            // SAFETY: a new command buffer of the pool, recorded once on this thread.
            unsafe {
                (recording.begin)(buffer, &begin).result().ok()?;
                exit_guard::called(Call::BeginCommandBuffer);
                (recording.end)(buffer).result().ok()?;
                exit_guard::called(Call::EndCommandBuffer);
            }
        }
        // SAFETY: the device is live, inside a section.
        self.fence = unsafe { raw.create_fence(&vk::FenceCreateInfo::default(), None) }.ok()?;
        exit_guard::called(Call::CreateFence);
        Some(())
    }

    /// Submit `infos` to the device's first `VkQueue`, the last signalling the fence, in one call
    /// where `together`, else one call each, and wait for the fence: the nanoseconds inside each
    /// call, in order. `Some(None)` where a call was refused or the wait failed; `None` where the
    /// process is exiting or the device was lost (the module docs, "Its thread").
    fn submit(
        &self,
        parts: &Parts,
        buffers: &[vk::CommandBuffer],
        together: bool,
    ) -> Option<Option<Vec<i64>>> {
        let calls = match together {
            true => 1,
            false => buffers.len(),
        };
        let (submitted, step, mark) = {
            let _busy = exit_guard::enter()?;
            let raw = parts.vk();
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            let injected = lock(&parts.checks.wait_at_submit).take();
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default();
            let infos: Vec<vk::SubmitInfo<'_>> = (buffers.iter())
                .map(|buffer| {
                    vk::SubmitInfo::default().command_buffers(std::slice::from_ref(buffer))
                })
                .collect();
            // A device test's submission that never completes (`SubmitChecks::wait_at_submit`):
            // its first info waits for the test's timeline semaphore.
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            let infos = match &injected {
                Some((semaphore, value)) => {
                    let mut infos = infos;
                    timeline = timeline.wait_semaphore_values(std::slice::from_ref(value));
                    infos[0] = infos[0]
                        .wait_semaphores(std::slice::from_ref(semaphore))
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])
                        .push_next(&mut timeline);
                    infos
                }
                None => infos,
            };
            // SAFETY: the queue is used under its lock, inside a section; each command buffer is
            // recorded, ended and in no pending submission (each round waits for the last); the
            // fence is unsignaled and in no pending submission. The device's word is held across
            // the calls and the relay's mark after them (MoltenVK names the device as it logs
            // each, `loader_word.rs`), taken inside the queue's lock, and let go before the wait
            // unless the word was found overwritten. The mark is kept, for the step to wait for.
            let uses = parts.queue_uses();
            let use_queue = |queue: vk::Queue| unsafe {
                let held = parts.loader_word.held();
                let mut spent = Vec::with_capacity(infos.len());
                let last = infos.len() - 1;
                let calls: Vec<(&[vk::SubmitInfo<'_>], vk::Fence)> = match together {
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
                let mut submitted = Ok(());
                for (infos, fence) in calls {
                    let start = Instant::now();
                    submitted = raw.queue_submit(queue, infos, fence);
                    spent.push(i64::try_from(start.elapsed().as_nanos()).unwrap_or(i64::MAX));
                    if submitted.is_err() {
                        break;
                    }
                }
                (held, submitted.map(|()| spent))
            };
            let ((held, submitted), mark) =
                uses.use_queue_kept(0, By::Checks, Uses::Unknown, true, use_queue);
            // Kept through the wait where the device's word was found overwritten: MoltenVK
            // names the device as each Metal command buffer completes, the relay's mark's last.
            // SAFETY: the device is live while its checks run.
            let step = unsafe { parts.loader_word.step_hold(Some(held)) };
            exit_guard::called(Call::QueueSubmit);
            parts.counts.submit_check_called(calls);
            (submitted, step, mark)
        };
        let nanos = match submitted {
            Ok(spent) => spent,
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                parts.loader_word.end_step(step, mark);
                return None;
            }
            Err(_) => {
                parts.loader_word.end_step(step, mark);
                // A refused call among several: those before it, and so the fence, were not all
                // submitted, so the device waits for its queue rather than the fence.
                return match calls > 1 {
                    true => self.idle(parts).map(|_| None),
                    false => Some(None),
                };
            }
        };
        let waited = self.wait(parts);
        // A stalled submission's relay mark completes no sooner than it: a step held whole lets
        // go of the device's word without waiting for the mark.
        let step = step.filter(|_| !parts.checks.stalled());
        parts.loader_word.end_step(step, mark);
        if !waited? {
            return Some(None);
        }
        Some(Some(nanos))
    }

    /// One merge round ([`submit`](Self::submit) of every empty command buffer): the time inside
    /// each call, one where `together`, else [`MERGED`].
    fn submit_round(&self, parts: &Parts, together: bool) -> Option<Option<Vec<i64>>> {
        self.submit(parts, &self.empty, together)
    }

    /// One submission of one empty command buffer with the fence: the time inside the call.
    fn submit_one(&self, parts: &Parts) -> Option<Option<i64>> {
        let times = self.submit(parts, &self.empty[..1], true)?;
        Some(times.map(|times| times.iter().sum()))
    }

    /// Wait for the fence, a slice at a time, each a section, then reset it: whether it was
    /// signalled. `None` where the process is exiting, the device was lost, or the submission
    /// stalled: it did not complete within the checks' bound ([`STALL`]), and the checks take
    /// the device as lost for them (the module docs, "A stalled submission").
    fn wait(&self, parts: &Parts) -> Option<bool> {
        let checks = &*parts.checks;
        let started = Instant::now();
        loop {
            let busy = exit_guard::enter()?;
            let raw = parts.vk();
            // SAFETY: the check's fence, signalled by a submission made.
            let waited =
                unsafe { raw.wait_for_fences(&[self.fence], true, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitForFences);
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            let waited = match checks.lost_at_wait.swap(false, Ordering::AcqRel) {
                true => Err(vk::Result::ERROR_DEVICE_LOST),
                false => waited,
            };
            match waited {
                Ok(()) => {
                    checks.waited(started.elapsed());
                    // SAFETY: the fence is signalled, in no pending submission.
                    let reset = unsafe { raw.reset_fences(&[self.fence]) };
                    exit_guard::called(Call::ResetFences);
                    return Some(reset.is_ok());
                }
                Err(vk::Result::TIMEOUT) if started.elapsed() < checks.bound() => continue,
                Err(vk::Result::TIMEOUT) => {
                    drop(busy);
                    parts.stall(started.elapsed());
                    return None;
                }
                Err(vk::Result::ERROR_DEVICE_LOST) => return None,
                Err(_) => return Some(false),
            }
        }
    }

    /// Wait until the first `VkQueue` is idle and reset the fence, after a refused call: `None`
    /// where the process is exiting.
    fn idle(&self, parts: &Parts) -> Option<()> {
        let (step, mark) = {
            let _busy = exit_guard::enter()?;
            let raw = parts.vk();
            // SAFETY: the queue is used under its lock, inside a section. The device's word is
            // held from the wait's return across the relay's mark after it, taken inside the
            // queue's lock, and where it was found overwritten through the wait for the mark, as
            // a submission's is (`loader_word.rs`).
            let use_queue = |queue: vk::Queue| unsafe {
                let _ = raw.queue_wait_idle(queue);
                parts.loader_word.held()
            };
            let (held, mark) =
                (parts.queue_uses()).use_queue_kept(0, By::Checks, Uses::Unknown, true, use_queue);
            exit_guard::called(Call::QueueWaitIdle);
            // SAFETY: the device is live while its checks run.
            (unsafe { parts.loader_word.step_hold(Some(held)) }, mark)
        };
        parts.loader_word.end_step(step, mark);
        let _busy = exit_guard::enter()?;
        // SAFETY: no submission of the fence is pending once the queue is idle.
        let _ = unsafe { parts.vk().reset_fences(&[self.fence]) };
        exit_guard::called(Call::ResetFences);
        Some(())
    }

    /// Let the objects go, inside a section: whether it ran (`false` once the process is
    /// exiting, the objects then left to its end). No submission naming them is pending: each
    /// was waited for, or the device was lost, which Vulkan counts as their completion.
    fn release(&self, parts: &Parts) -> bool {
        let _held = parts.step_held();
        let Some(_busy) = exit_guard::enter() else {
            return false;
        };
        let raw = parts.vk();
        // SAFETY: no pending submission names them; a null handle is skipped.
        unsafe {
            if self.fence != vk::Fence::null() {
                release::object(raw, self.fence);
            }
            release::object(raw, self.pool);
        }
        true
    }

    /// The objects a stalled submission names ([`SubmitChecks::stop`]), inside the device's
    /// teardown's section: wait for its fence a slice at a time, up to `bound`, then release
    /// them. Whether it completed: `false` where the bound passed or the process began to exit
    /// meanwhile, the objects then left to the process's end with the device.
    fn release_once_complete(self, raw: &Vk, bound: Duration) -> bool {
        let started = Instant::now();
        loop {
            // SAFETY: the checks' fence, of the stalled submission made.
            let waited =
                unsafe { raw.wait_for_fences(&[self.fence], true, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitForFences);
            match waited {
                // Signalled, or the device lost, which Vulkan counts as the submission's
                // completion.
                Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => break,
                Err(vk::Result::TIMEOUT) if started.elapsed() < bound && !exit_guard::closed() => {}
                Err(_) => return false,
            }
        }
        // SAFETY: the submission that named them completed, and none other is pending.
        unsafe {
            release::object(raw, self.fence);
            release::object(raw, self.pool);
        }
        true
    }
}
