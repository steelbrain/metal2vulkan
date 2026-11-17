//! The warm pass: [`Device::warm`] makes, on a thread of its own and below every other making,
//! what the creations its records describe make, before the program asks for them.
//!
//! - **One thread a device, in order.** A call queues its records and returns; the device's warm
//!   thread (`metal2vulkan-warm`, started by the call that finds none running) takes each call's
//!   records in the order given, one at a time, and ends once none are left. On macOS it runs at
//!   the utility class (`exit_guard::lower_to_utility`); its pipeline makings rank below every
//!   other making of the process ([`Urgency::Warm`](crate::pipeline::budget::Urgency::Warm)),
//!   on its thread and in the pieces it hands the device's pool, which run as part of the pass
//!   ([`within`]). It holds its device only while it warms a record, through the registry of the
//!   runtime's own holds (`device/holds.rs`), and reads the stop before each: the program's last
//!   drop stops the pass there and waits for the record being made, so the device goes on the
//!   program's thread, and the pass with it.
//! - **The creation itself** (`pipeline::warm`). Each record is read back (its libraries from the
//!   shader cache directory, `shader_cache/libraries.rs`, unless it was made in this process) and
//!   created as the program created it, under the default options, inside one exit-guard section;
//!   the state is let go at once. So the translations land in the shader cache and the predicted
//!   `VkPipeline`s' data in the driver's records under the very names the program's creation of the
//!   same descriptor asks for. A mesh render pipeline is not made: its creation keeps nothing past
//!   the state.
//! - **And a compute record's dispatches** (`pipeline/dispatch_record.rs`). Once a compute
//!   record's state is created, each dispatch the record names is planned as a creation's
//!   prediction plans its first one, with the record's threadgroup memory lengths, and each of its
//!   regions' `VkPipeline`s made on the pass's thread, each region's residual form beside it
//!   where that differs ([`WarmStatistics::dispatch_pipelines_warmed`]): their data lands in the
//!   driver's records under the names the program's dispatch of that shape asks for, and a
//!   dispatch binding a storage view off offset 0 asks for.
//! - **The guest sees nothing.** On the pass's threads ([`warming`]) a creation takes no
//!   identifier, no resource ID and no sampler number, makes no neutral and keeps no room, takes
//!   nothing back from and gives nothing to the kept variants, makes no immutable sampler (a
//!   layout that bakes one stops the creation after its translation: [`translation_only`]), takes
//!   from the memo none of the program's live variants and lets it hand out none of its own (the
//!   memo's translations it takes as any creation does), announces nothing and counts nothing
//!   announced, makes no depth stand-in copy, numbers no creation log line, and hands the
//!   pipeline-record hook nothing. A failed creation is counted and dropped.
//! - **No answer of another build.** A record holds a descriptor, never a translation, so what
//!   the pass makes of it is what the program's own creation of that descriptor makes on this
//!   build, through the same lookups: translations from and into this build's directory of the
//!   shader cache, which only this translator reads and writes (`shader_cache::this_build`), and
//!   a render creation's answers under this runtime's `RUNTIME_VERSION`
//!   (`shader_cache/creation.rs`). A record an earlier runtime or translator made is warmed by
//!   this one's.
//! - **Unread records and failed creations are said, never fatal.** A record the pass cannot read
//!   (of another format, damaged, or naming a library the directory no longer holds) is counted;
//!   the device's first is announced through its message sink, else on stderr, as the warning
//!   `metal2vulkan-warm-record-unread`, and the pass goes on with the next record. A record whose
//!   creation fails is counted too, and the first failure of each kind the device meets is
//!   announced the same way as the warning `metal2vulkan-warm-record-failed`, naming the kind
//!   ([`failure_kind`]: the runtime's error and, for a translator refusal, the refused feature and
//!   what it needs), never a name, location or text of the record's shaders; past
//!   [`FAILURE_KINDS`] kinds, later ones are counted alone.
//! - **A failure for good is kept, and answers later passes.** Where a creation's failure is a
//!   fact of the record on this device ([`kept_refusal`]: a typed refusal, such as a kernel past
//!   the stage bound its driver compiles intact or past the private memory a thread holds), its
//!   kind is kept in the shader cache directory under the record, the driver's identity, the
//!   profile, the runtime's version and the build (`shader_cache/warm_refusals.rs`). A later pass
//!   over the same directory looks a record up there before it reads it, and answers one it finds
//!   as failed of that kind at once, its creation not made
//!   ([`WarmStatistics::records_failed_before`]), so a refused translation is not run again at
//!   every boot. A transient failure (out of memory, a device lost, a deadline, a cancellation)
//!   is never kept; nor is one a later boot can see otherwise (a missing input, such as a dynamic
//!   library file), so the next pass makes those creations again. The program's own creation of
//!   such a descriptor still translates, and meets the refusal itself.
//! - **The byte budget.** The pass stops before its next record once the driver's pipeline cache
//!   data its makings added (each record's growth as a making writes it, `shader_cache/driver.rs`)
//!   reaches the call's `byte_budget`; the records left are counted, not made.
//! - **The record in making.** While the pass makes a record, the device answers its place among
//!   the records handed to its calls, in the order handed, and how long its making has worked
//!   ([`WarmStatistics::record_in_making`], [`WarmStatistics::record_in_making_nanos`]): the wall
//!   time while the pass's thread, or a piece of the record's it handed the pool, works for it
//!   ([`within`]), never while each only waits ([`waiting`]: for a budget turn, for the
//!   large-making words, for another making or for its pieces). A piece that outlives its
//!   record's making works for that record's, not the next one's.

use super::codec;
use super::{PipelineRecord, WarmStatistics};
use crate::device::holds::Holder;
use crate::device::{lock, Device, WeakDevice};
use crate::exit_guard;
use metal2vulkan::cache::Hash256;
use metal2vulkan::library::Library;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The most libraries a pass keeps read from the directory at once: records in first-use order
/// name the same few in turn, and each can be large.
const LIBRARIES: usize = 16;

/// One record's making on a warm call, shared by the pass's thread and the pieces it hands the
/// pool while it makes the record.
#[derive(Default)]
pub(crate) struct Pass {
    /// The driver's pipeline cache data the call's makings added, in bytes: one count for all of
    /// the call's records.
    bytes: Arc<AtomicU64>,
    /// The record's creation stopped after its translation ([`translation_only`]).
    translation_only: AtomicBool,
    /// How long the record's making has worked ([`WarmStatistics::record_in_making_nanos`]).
    clock: Clock,
}

impl Pass {
    /// The making of a record of the call whose makings' data `bytes` counts.
    fn of_call(bytes: &Arc<AtomicU64>) -> Pass {
        Pass {
            bytes: Arc::clone(bytes),
            ..Pass::default()
        }
    }

    /// How long the record's making has worked, up to now: the wall time while a thread working
    /// for it (the pass's, or a piece's of the pool) was not waiting ([`waiting`]).
    pub(crate) fn worked(&self) -> Duration {
        self.clock.worked()
    }
}

/// The wall time while one thread or more worked for a making ([`Pass::worked`]).
#[derive(Default)]
struct Clock(Mutex<Worked>);

#[derive(Default)]
struct Worked {
    /// The threads working now.
    working: usize,
    /// Since when one has, while one does.
    since: Option<Instant>,
    /// The time worked before `since`.
    before: Duration,
}

impl Clock {
    /// A thread starts working.
    fn begin(&self) {
        let mut worked = lock(&self.0);
        worked.working += 1;
        if worked.working == 1 {
            worked.since = Some(Instant::now());
        }
    }

    /// A thread stops working.
    fn end(&self) {
        let mut worked = lock(&self.0);
        worked.working = worked.working.saturating_sub(1);
        if worked.working == 0 {
            if let Some(since) = worked.since.take() {
                worked.before += since.elapsed();
            }
        }
    }

    fn worked(&self) -> Duration {
        let worked = lock(&self.0);
        worked.before + worked.since.map_or(Duration::ZERO, |since| since.elapsed())
    }
}

/// A thread's work for a making, ended when dropped (by an unwinding too).
struct Working<'a>(&'a Clock);

impl Drop for Working<'_> {
    fn drop(&mut self) {
        self.0.end();
    }
}

/// A thread's wait inside its work for a making ([`waiting`]), its work begun again when dropped.
struct Waits(Arc<Pass>);

impl Drop for Waits {
    fn drop(&mut self) {
        let _ = WAITING.try_with(|waiting| waiting.set(false));
        self.0.clock.begin();
    }
}

thread_local! {
    /// The pass this thread works for now, if any ([`within`]).
    static PASS: RefCell<Option<Arc<Pass>>> = const { RefCell::new(None) };
    /// This thread waits now inside its work for a pass ([`waiting`]).
    static WAITING: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread works for a warm pass now: the pass's thread, or a pool's piece it handed.
/// Past the thread's thread-locals' end (a program's exit handler, `this_thread.rs`), none.
pub(crate) fn warming() -> bool {
    PASS.try_with(|pass| pass.borrow().is_some())
        .unwrap_or(false)
}

/// The pass this thread works for now, which a piece it hands the pool works for too.
pub(crate) fn pass() -> Option<Arc<Pass>> {
    PASS.try_with(|pass| pass.borrow().clone()).ok().flatten()
}

/// Run `f` on this thread as part of `pass`, as it returns or unwinds restoring what the thread
/// worked for before; `pass`'s making works meanwhile ([`Pass::worked`]), but while `f` waits
/// ([`waiting`]).
pub(crate) fn within<T>(pass: &Arc<Pass>, f: impl FnOnce() -> T) -> T {
    /// Puts the earlier pass back.
    struct Restore(Option<Arc<Pass>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = PASS.try_with(|pass| *pass.borrow_mut() = self.0.clone());
        }
    }
    let earlier = PASS
        .try_with(|current| current.replace(Some(Arc::clone(pass))))
        .ok()
        .flatten();
    // A thread working for `pass` already (a piece the pool ran on the thread that handed it)
    // works for it once.
    let again = (earlier.as_ref()).is_some_and(|earlier| Arc::ptr_eq(earlier, pass));
    let _restore = Restore(earlier);
    let _working = (!again).then(|| {
        pass.clock.begin();
        Working(&pass.clock)
    });
    f()
}

/// Run `wait`, a wait of this thread for a budget turn, the large-making words or another making
/// (`pipeline/budget.rs`, `compile_memory.rs`, `making.rs`, `pool.rs`), with the making it works
/// for, if any, not working meanwhile ([`WarmStatistics::record_in_making_nanos`]); a wait inside
/// another is the outer one's.
pub(crate) fn waiting<T>(wait: impl FnOnce() -> T) -> T {
    let Some(pass) = pass() else {
        return wait();
    };
    if WAITING
        .try_with(|waiting| waiting.replace(true))
        .unwrap_or(true)
    {
        return wait();
    }
    pass.clock.end();
    let _again = Waits(pass);
    wait()
}

/// Count `bytes` of driver data a making added toward this thread's pass, if it works for one
/// (`shader_cache/driver.rs`).
pub(crate) fn added_pipeline_bytes(bytes: u64) {
    let _ = PASS.try_with(|pass| {
        if let Some(pass) = &*pass.borrow() {
            pass.bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    });
}

/// The error a warm creation stops with where its layout bakes an immutable sampler, which a warm
/// pass makes none of (`pipeline/objects.rs`): its translations are kept, its objects and
/// pipelines not made. The pass counts the record as translated, not failed.
pub(crate) fn translation_only() -> crate::Error {
    let _ = PASS.try_with(|pass| {
        if let Some(pass) = &*pass.borrow() {
            pass.translation_only.store(true, Ordering::Relaxed);
        }
    });
    crate::error::internal("a warm pass makes no immutable sampler")
}

/// A device's warm calls: the records queued, whether its thread runs, the record being made,
/// and the counts.
#[derive(Default)]
pub(crate) struct Warmer {
    queue: Mutex<Queue>,
    /// The record a pass makes now, if any: its place among the records handed, and its making
    /// ([`WarmStatistics::record_in_making`]).
    making: Mutex<Option<(u64, Arc<Pass>)>>,
    counts: Counts,
}

#[derive(Default)]
struct Queue {
    calls: VecDeque<Call>,
    running: bool,
    /// The records handed to the device's calls so far.
    handed: u64,
}

/// One warm call: its records in order, the place of its first among the records handed, and its
/// budget.
struct Call {
    records: Vec<PipelineRecord>,
    first: u64,
    byte_budget: u64,
}

/// The record a pass makes, the warmer's while it lives ([`Warmer::making`]).
struct InMaking<'a>(&'a Mutex<Option<(u64, Arc<Pass>)>>);

impl Drop for InMaking<'_> {
    fn drop(&mut self) {
        *lock(self.0) = None;
    }
}

/// What the device's passes did, by record.
#[derive(Default)]
struct Counts {
    warmed: AtomicU64,
    translated: AtomicU64,
    failed: AtomicU64,
    unread: AtomicU64,
    not_warmable: AtomicU64,
    not_run: AtomicU64,
    pipeline_bytes: AtomicU64,
    /// The `VkPipeline`s the passes made for the dispatches compute records name.
    dispatch_pipelines: AtomicU64,
    /// The device's first unread record was announced ([`UNREAD`]).
    said_unread: AtomicBool,
    /// Of `failed`, those answered from a failure an earlier pass kept
    /// ([`WarmStatistics::records_failed_before`]).
    failed_before: AtomicU64,
    /// The kinds of failed creation the device announced ([`FAILED`]), at most
    /// [`FAILURE_KINDS`].
    said_failed: Mutex<Vec<String>>,
}

/// The id name of the warning of a device's first record a warm pass could not read.
pub(crate) const UNREAD: &str = "metal2vulkan-warm-record-unread";

/// The id name of the warning of the first record of each kind of failure
/// ([`failure_kind`]) whose creation failed on a device's warm pass.
pub(crate) const FAILED: &str = "metal2vulkan-warm-record-failed";

/// The most kinds of failure a device announces; a kind past them is counted alone.
const FAILURE_KINDS: usize = 32;

/// What became of one record.
enum Outcome {
    /// Made, with the `VkPipeline`s made for the dispatches it names.
    Warmed(u64),
    Translated,
    /// Its creation failed, of the kind given ([`failure_kind`]); `before` where the pass answered
    /// it from the failure an earlier pass kept, not making it.
    Failed {
        kind: String,
        before: bool,
    },
    /// Not read, for the reason given.
    Unread(&'static str),
    NotWarmable,
}

impl Warmer {
    /// Its counts, into `statistics`.
    pub(crate) fn read(&self, statistics: &mut WarmStatistics) {
        let counts = &self.counts;
        let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        statistics.records_warmed = read(&counts.warmed);
        statistics.records_translated = read(&counts.translated);
        statistics.records_failed = read(&counts.failed);
        statistics.records_failed_before = read(&counts.failed_before);
        statistics.records_unread = read(&counts.unread);
        statistics.records_not_warmable = read(&counts.not_warmable);
        statistics.records_not_run = read(&counts.not_run);
        statistics.pipeline_bytes = read(&counts.pipeline_bytes);
        statistics.dispatch_pipelines_warmed = read(&counts.dispatch_pipelines);
        if let Some((at, pass)) = &*lock(&self.making) {
            statistics.record_in_making = Some(*at);
            statistics.record_in_making_nanos =
                u64::try_from(pass.worked().as_nanos()).unwrap_or(u64::MAX);
        }
    }

    /// Count `outcome`, a record's on `device`; the device's first unread record, and its first
    /// failed creation of each kind, are announced (the module docs).
    fn count(&self, device: &Device, outcome: Outcome) {
        let counts = &self.counts;
        let counter = match outcome {
            Outcome::Warmed(dispatch_pipelines) => {
                (counts.dispatch_pipelines).fetch_add(dispatch_pipelines, Ordering::Relaxed);
                &counts.warmed
            }
            Outcome::Translated => &counts.translated,
            Outcome::Failed { kind, before } => {
                if before {
                    counts.failed_before.fetch_add(1, Ordering::Relaxed);
                }
                let mut said = lock(&counts.said_failed);
                if said.len() < FAILURE_KINDS && !said.contains(&kind) {
                    say_failed(device, &kind);
                    said.push(kind);
                }
                &counts.failed
            }
            Outcome::Unread(why) => {
                if !counts.said_unread.swap(true, Ordering::Relaxed) {
                    say_unread(device, why);
                }
                &counts.unread
            }
            Outcome::NotWarmable => &counts.not_warmable,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn not_run(&self, records: usize) {
        self.counts
            .not_run
            .fetch_add(records as u64, Ordering::Relaxed);
    }
}

impl Device {
    /// No Metal counterpart: it stands in for Metal's own pipeline caches across processes, which
    /// keep the pipelines a program made in the OS's caches and make them again from there at
    /// their creation. Make, in the background, what the creations `records` describe make
    /// (handed earlier by
    /// [`DeviceOptions::pipeline_records`](crate::DeviceOptions::pipeline_records), in a process
    /// over the same shader cache directory), in the order given, until the driver's pipeline
    /// cache data the pass added reaches `byte_budget` bytes: their translations into the shader
    /// cache and their predicted `VkPipeline`s' data into the driver's records, so the program's
    /// creations of the same descriptors find them. Returns at once. Data a making finds in the
    /// directory adds nothing ([`WarmStatistics::pipeline_bytes`]): over the directory whose
    /// creations made the records, the budget binds only where a making writes new data.
    ///
    /// - It makes no object the program sees and changes no answer: no state, identifier,
    ///   resource ID, sampler number, notice or memory the program's calls decide; a state the
    ///   program creates meanwhile is made as without it.
    /// - What it makes is what the program's own creation of each descriptor makes on this build:
    ///   a record holds a descriptor, never a translation, so it reads no translation of another
    ///   translator and no creation record of another runtime version (the shader cache keeps both
    ///   apart). A record an earlier version made is warmed by this one.
    /// - It runs on one thread of the device's, at the utility class on macOS, its pipeline
    ///   makings after every other making of the process; calls queue behind each other.
    /// - Nothing fails and nothing ends the run. A record it cannot read (of another format,
    ///   damaged, or naming a library the directory no longer holds) is counted
    ///   ([`Device::warm_statistics`]) and dropped, the device's first announced as the warning
    ///   `metal2vulkan-warm-record-unread`; a creation that fails is counted and dropped, the
    ///   device's first of each kind announced as the warning `metal2vulkan-warm-record-failed`,
    ///   which names the kind (the runtime's error and, for a translator refusal, the refused
    ///   feature and what it needs) and nothing of the record's shaders. A mesh
    ///   render pipeline's record is counted and not made (its creation keeps nothing past the
    ///   state); a layout that bakes an immutable sampler is translated only.
    /// - A creation refused for good on this device (a typed refusal: a kernel past what its
    ///   driver compiles intact, or past the private memory a thread holds) is kept in the shader
    ///   cache directory, keyed by the record, the driver's identity (vendor, device, driver
    ///   version, `pipelineCacheUUID`), the profile, the runtime's version and the translator's
    ///   build; a later pass over the directory answers that record as failed at once, not making
    ///   it again ([`WarmStatistics::records_failed_before`]). A transient failure is never kept.
    pub fn warm(&self, records: Vec<PipelineRecord>, byte_budget: u64) {
        self.current().warm_now(records, byte_budget)
    }

    /// [`Device::warm`] on the generation that serves (`device/line.rs`).
    pub(crate) fn warm_now(&self, records: Vec<PipelineRecord>, byte_budget: u64) {
        let warmer = &self.shared().warmer;
        let mut queue = lock(&warmer.queue);
        let first = queue.handed;
        queue.handed += records.len() as u64;
        queue.calls.push_back(Call {
            records,
            first,
            byte_budget,
        });
        if queue.running {
            return;
        }
        queue.running = true;
        drop(queue);
        let device = self.downgrade();
        let started = exit_guard::spawn(
            std::thread::Builder::new().name("metal2vulkan-warm".into()),
            move || run(&device),
        );
        if started.is_err() {
            // No thread: the calls queued are not run, and a later call tries again.
            let mut queue = lock(&warmer.queue);
            let left: usize = queue.calls.drain(..).map(|call| call.records.len()).sum();
            queue.running = false;
            drop(queue);
            warmer.not_run(left);
        }
    }
}

/// The warm thread of `device`: its calls in order, until none is left or the device went.
fn run(device: &WeakDevice) {
    exit_guard::lower_to_utility();
    loop {
        let Some(held) = device.hold() else {
            return;
        };
        let call = {
            let mut queue = lock(&held.shared().warmer.queue);
            match queue.calls.pop_front() {
                Some(call) => call,
                None => {
                    queue.running = false;
                    return;
                }
            }
        };
        drop(held);
        warm_call(device, call);
    }
}

/// Warm `call`'s records on `device`, in order, within its budget (the module docs).
fn warm_call(device: &WeakDevice, call: Call) {
    let bytes = Arc::new(AtomicU64::new(0));
    let mut libraries = Libraries::default();
    let Call {
        records,
        first,
        byte_budget,
    } = call;
    let total = records.len();
    for (at, record) in records.iter().enumerate() {
        let Some(held) = device.hold() else {
            return;
        };
        if !held.holds_go_on(Holder::WarmPass) {
            return;
        }
        let warmer = &held.shared().warmer;
        if bytes.load(Ordering::Relaxed) >= byte_budget {
            warmer.not_run(total - at);
            return;
        }
        // Every Vulkan call of the creation runs inside this section, which an exit waits for
        // (`exit_guard`); once the process is exiting, nothing more is made.
        let Some(_busy) = exit_guard::enter() else {
            warmer.not_run(total - at);
            return;
        };
        let before = bytes.load(Ordering::Relaxed);
        let pass = Arc::new(Pass::of_call(&bytes));
        // The record in making from here until its outcome is counted: the six counts' sum is its
        // place meanwhile.
        let at = first + at as u64;
        *lock(&warmer.making) = Some((at, Arc::clone(&pass)));
        let in_making = InMaking(&warmer.making);
        let outcome = within(&pass, || {
            // A test's gate holds the record here, working, as a slow read would (`tests::warm`).
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            gate::pass(warmer, at);
            warm_record(&held, record, &mut libraries, &pass)
        });
        drop(in_making);
        let added = bytes.load(Ordering::Relaxed) - before;
        warmer
            .counts
            .pipeline_bytes
            .fetch_add(added, Ordering::Relaxed);
        warmer.count(&held, outcome);
    }
}

/// A test's gate in front of the makings of a device's records (`tests::warm`): a record whose
/// place it is closed for waits there, working for its making as a slow read or translation
/// would, until it opens.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod gate {
    use super::Warmer;
    use crate::device::lock;
    use std::collections::BTreeSet;
    use std::sync::{Condvar, Mutex, PoisonError};

    /// The places closed, by the warmer they are its device's of.
    static CLOSED: Mutex<BTreeSet<(usize, u64)>> = Mutex::new(BTreeSet::new());
    static OPENED: Condvar = Condvar::new();

    fn key(warmer: &Warmer, at: u64) -> (usize, u64) {
        (std::ptr::from_ref(warmer) as usize, at)
    }

    /// Close the gate for the record at `at` among those handed to `warmer`'s device.
    pub(crate) fn close(warmer: &Warmer, at: u64) {
        lock(&CLOSED).insert(key(warmer, at));
    }

    /// Open it.
    pub(crate) fn open(warmer: &Warmer, at: u64) {
        lock(&CLOSED).remove(&key(warmer, at));
        OPENED.notify_all();
    }

    /// Wait while it is closed.
    pub(super) fn pass(warmer: &Warmer, at: u64) {
        let mut closed = lock(&CLOSED);
        while closed.contains(&key(warmer, at)) {
            closed = OPENED.wait(closed).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// Announce that a record handed to `device`'s warm pass could not be read, for `why`, through
/// the device's message sink, else on stderr: a warning, which ends nothing.
fn say_unread(device: &Device, why: &str) {
    let text = format!(
        "metal2vulkan: a pipeline record handed to Device::warm could not be read ({why}): it is \
         not warmed, the pass goes on with the next record, and the program's creation of its \
         descriptor makes what it needs when asked. This device's later unread records are \
         counted alone (`WarmStatistics::records_unread`)."
    );
    let message = crate::DebugMessage {
        severity: crate::MessageSeverity::Warning,
        kind: crate::MessageKind {
            general: true,
            ..crate::MessageKind::default()
        },
        id_name: UNREAD.to_owned(),
        id: 0,
        text,
        degradation: None,
    };
    match device.shared().instance.sink() {
        Some(sink) => sink.deliver(&message),
        None => eprintln!("{}", message.text),
    }
}

/// Announce that the creation of a record handed to `device`'s warm pass failed, of `kind`
/// ([`failure_kind`]), through the device's message sink, else on stderr: a warning, which ends
/// nothing.
fn say_failed(device: &Device, kind: &str) {
    let text = format!(
        "metal2vulkan: a pipeline record's creation failed on Device::warm ({kind}): it is not \
         warmed, the pass goes on with the next record, and the program's creation of its \
         descriptor meets the same when asked. This device's later failures of this kind are \
         counted alone (`WarmStatistics::records_failed`)."
    );
    let message = crate::DebugMessage {
        severity: crate::MessageSeverity::Warning,
        kind: crate::MessageKind {
            general: true,
            ..crate::MessageKind::default()
        },
        id_name: FAILED.to_owned(),
        id: 0,
        text,
        degradation: None,
    };
    match device.shared().instance.sink() {
        Some(sink) => sink.deliver(&message),
        None => eprintln!("{}", message.text),
    }
}

/// The kind of a warm creation's `error`, as [`FAILED`] names it: the runtime error's variant,
/// and for a translator's error its variant, with a refusal's feature and what it needs and an
/// input error's class. Never a symbol, location or message, which can name the record's
/// shaders.
pub(crate) fn failure_kind(error: &crate::Error) -> String {
    use metal2vulkan::error::Error as Translation;
    let unsupported = |u: &metal2vulkan::error::Unsupported| {
        format!("Unsupported{{{:?}; {:?}}}", u.feature, u.needs)
    };
    let translation = |e: &Translation| match e {
        Translation::Unsupported(u) => unsupported(u),
        Translation::InvalidInput(i) => format!("InvalidInput{{{:?}}}", i.class()),
        other => variant(other),
    };
    match error {
        crate::Error::Compile(e) => format!("Compile({})", translation(e)),
        crate::Error::Unsupported(u) => unsupported(u),
        other => variant(other),
    }
}

/// The name of the variant `value` is: the start of its `Debug`, before any field.
fn variant(value: &impl std::fmt::Debug) -> String {
    let debug = format!("{value:?}");
    let end = debug
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(debug.len());
    debug[..end].to_owned()
}

/// Make on `device` what `record`'s creation makes (the module docs).
fn warm_record(
    device: &Device,
    record: &PipelineRecord,
    libraries: &mut Libraries,
    pass: &Pass,
) -> Outcome {
    let bytes = record.bytes();
    let cache = &device.shared().shader_cache;
    let compile = device.compile_device();
    if let Some(kind) = cache.warm_refusal(compile, bytes) {
        return Outcome::Failed { kind, before: true };
    }
    let read =
        match record.libraries() {
            made if !made.is_empty() => codec::decode(bytes, made),
            _ => match libraries.of(device, bytes) {
                Some(named) => codec::decode(bytes, &named),
                None => return Outcome::Unread(
                    "it names a library the shader cache directory does not hold, or is no record",
                ),
            },
        };
    let Ok(pipeline) = read else {
        return Outcome::Unread("its libraries do not hold what it names, or it is damaged");
    };
    match crate::pipeline::warm(device, &pipeline) {
        Ok(Some(dispatch_pipelines)) => Outcome::Warmed(dispatch_pipelines),
        Ok(None) => Outcome::NotWarmable,
        Err(_) if pass.translation_only.load(Ordering::Relaxed) => Outcome::Translated,
        Err(error) => {
            let kind = failure_kind(&error);
            if kept_refusal(&error) {
                cache.keep_warm_refusal(compile, bytes, &kind);
            }
            Outcome::Failed {
                kind,
                before: false,
            }
        }
    }
}

/// Whether `error`, a warm creation's, is a fact of its record on this device, so kept for later
/// passes (the module docs): a typed refusal ([`crate::Error::Unsupported`]: every translator
/// refusal of the record's stages on this profile reaches the program as one, and the runtime's
/// own refusals of what the device lacks), or a translator outcome the translator states is
/// permanent for its input on this profile (`metal2vulkan::error::Error::is_permanent_for`), but
/// a pipeline input the creation lacks (`NeedsInput`: a dynamic library file a later boot may
/// find). Never a transient failure: out of memory, a lost device, a Vulkan error, a deadline or
/// a cancellation (`Permanence::Never`), or any other error of the runtime's.
fn kept_refusal(error: &crate::Error) -> bool {
    use metal2vulkan::error::{Error as Translation, Permanence};
    match error {
        crate::Error::Unsupported(_) => true,
        crate::Error::Compile(Translation::NeedsInput(_)) => false,
        crate::Error::Compile(outcome) => outcome.is_permanent_for() != Permanence::Never,
        _ => false,
    }
}

/// The libraries a pass read from the directory, by content hash: at most [`LIBRARIES`], all let
/// go once it holds that many.
#[derive(Default)]
struct Libraries(HashMap<Hash256, Library>);

impl Libraries {
    /// The libraries `record`'s bytes name, in their places' order, each from this pass's or the
    /// device's shader cache directory; `None` where one is in neither, or the bytes are no
    /// record.
    fn of(&mut self, device: &Device, record: &[u8]) -> Option<Vec<Library>> {
        let table = codec::libraries(record).ok()?;
        let cache = &device.shared().shader_cache;
        let mut named = Vec::with_capacity(table.len());
        for (hash, length) in table {
            if let Some(library) = self.0.get(&hash) {
                named.push(library.clone());
                continue;
            }
            let library = cache.library(&hash, length)?;
            if self.0.len() >= LIBRARIES {
                self.0.clear();
            }
            self.0.insert(hash, library.clone());
            named.push(library);
        }
        Some(named)
    }
}

#[cfg(test)]
mod tests {
    use super::{failure_kind, kept_refusal};
    use metal2vulkan::error::{
        BudgetExceeded, DiagName, Error, Feature, Location, LowerFamily, RuntimeInput, Unsupported,
    };

    /// A failure's kind names the runtime error's variant and a translator refusal's feature and
    /// needs, and nothing the refusal names of the shader: its symbol and its location (which
    /// names the function) are left out.
    #[test]
    fn a_failure_kind_names_nothing_of_the_shader() {
        let mut refused = Unsupported::not_implemented(Feature::Lowering(LowerFamily::Access)).at(
            Location::function(DiagName::new("a_guest_function"), Some(3), Some(7)),
        );
        refused.symbol = Some(DiagName::new("a_guest_symbol"));
        let compiled = failure_kind(&crate::Error::Compile(Error::Unsupported(refused.clone())));
        assert_eq!(
            compiled,
            "Compile(Unsupported{Lowering(Access); NotImplemented})"
        );
        let unsupported = failure_kind(&crate::Error::Unsupported(refused));
        assert_eq!(unsupported, "Unsupported{Lowering(Access); NotImplemented}");
        assert_eq!(failure_kind(&crate::Error::DeviceLost), "DeviceLost");
        for kind in [compiled, unsupported] {
            assert!(!kind.contains("a_guest"), "{kind}");
        }
    }

    /// A refusal of the record on this device is kept for later passes: a typed refusal (a stage
    /// past the stage bound, or past the private memory a thread holds), and a translator
    /// outcome permanent for its input. A transient failure is not (a deadline, a cancellation, a
    /// host allocation that failed, out of memory, a lost device), nor a missing input that a
    /// later boot may supply, nor a runtime error that is no refusal.
    #[test]
    fn only_a_refusal_for_good_is_kept() {
        use metal2vulkan::device::VkLimit;
        use metal2vulkan::error::{Capability, InternalError, InternalKind, Limit, Needs};
        let mut stage = Unsupported::not_implemented(Feature::StageSize);
        stage.needs = Needs::StageBound;
        let storage = Capability::Limit {
            limit: VkLimit::FunctionStorageBytesPerThread,
            required: 256 << 10,
            available: 32 << 10,
        };
        let private = Unsupported::needs(Feature::PrivateMemory, storage);
        let kept = [
            crate::Error::Unsupported(stage.clone()),
            crate::Error::Unsupported(private),
            crate::Error::Compile(Error::Unsupported(stage)),
            crate::Error::Compile(Error::LimitExceeded(Limit::WorkUnits)),
        ];
        for error in &kept {
            assert!(kept_refusal(error), "{error:?}");
        }
        let library = RuntimeInput::DynamicLibrary {
            install_name: DiagName::new("/a/library.metallib"),
        };
        let allocation = InternalError::new(InternalKind::AllocationFailed, "a host allocation");
        let retried = [
            crate::Error::Compile(Error::Budget(BudgetExceeded::Time)),
            crate::Error::Compile(Error::Cancelled),
            crate::Error::Compile(Error::Internal(allocation)),
            crate::Error::Compile(Error::NeedsInput(library)),
            crate::Error::OutOfMemory(crate::MemoryKind::Device),
            crate::Error::DeviceLost,
        ];
        for error in &retried {
            assert!(!kept_refusal(error), "{error:?}");
        }
    }
}
