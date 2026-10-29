//! The warm pass: [`Device::warm`] makes, on a thread of its own and below every other making,
//! what the creations its records describe make, before the program asks for them.
//!
//! - **One thread a device, in order.** A call queues its records and returns; the device's warm
//!   thread (`metal2vulkan-warm`, started by the call that finds none running) takes each call's
//!   records in the order given, one at a time, and ends once none are left. On macOS it runs at
//!   the utility class (`exit_guard::lower_to_utility`); its pipeline makings rank below every
//!   other making of the process ([`Urgency::Warm`](crate::pipeline::budget::Urgency::Warm)),
//!   on its thread and in the pieces it hands the device's pool, which run as part of the pass
//!   ([`within`]). It holds its device only while it warms a record, so a device the program let
//!   go goes, and the pass with it.
//! - **The creation itself** (`pipeline::warm`). Each record is read back (its libraries from the
//!   shader cache directory, `shader_cache/libraries.rs`, unless it was made in this process) and
//!   created as the program created it, under the default options, inside one exit-guard section;
//!   the state is let go at once. So the translations land in the shader cache and the predicted
//!   `VkPipeline`s' data in the driver's records under the very names the program's creation of the
//!   same descriptor asks for. A mesh render pipeline is not made: its creation keeps nothing past
//!   the state.
//! - **The guest sees nothing.** On the pass's threads ([`warming`]) a creation takes no
//!   identifier, no resource ID and no sampler number, makes no neutral and keeps no room, takes
//!   nothing back from and gives nothing to the kept variants, makes no immutable sampler (a
//!   layout that bakes one stops the creation after its translation: [`translation_only`]), takes
//!   from the memo none of the program's live variants and lets it hand out none of its own (the
//!   memo's translations it takes as any creation does), announces nothing and counts nothing
//!   announced, makes no depth stand-in copy, numbers no creation log line, and hands the
//!   pipeline-record hook nothing. A failed creation is counted and dropped, never said.
//! - **No answer of another build.** A record holds a descriptor, never a translation, so what
//!   the pass makes of it is what the program's own creation of that descriptor makes on this
//!   build, through the same lookups: translations from and into this build's directory of the
//!   shader cache, which only this translator reads and writes (`shader_cache::this_build`), and
//!   a render creation's answers under this runtime's `RUNTIME_VERSION`
//!   (`shader_cache/creation.rs`). A record an earlier runtime or translator made is warmed by
//!   this one's.
//! - **Unread records are said, never fatal.** A record the pass cannot read (of another format,
//!   damaged, or naming a library the directory no longer holds) is counted; the device's first
//!   is announced through its message sink, else on stderr, as the warning
//!   `metal2vulkan-warm-record-unread`, and the pass goes on with the next record.
//! - **The byte budget.** The pass stops before its next record once the driver's pipeline cache
//!   data its makings added (each record's growth as a making writes it, `shader_cache/driver.rs`)
//!   reaches the call's `byte_budget`; the records left are counted, not made.

use super::codec;
use super::{PipelineRecord, WarmStatistics};
use crate::device::{lock, Device, WeakDevice};
use crate::exit_guard;
use metal2vulkan::cache::Hash256;
use metal2vulkan::library::Library;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The most libraries a pass keeps read from the directory at once: records in first-use order
/// name the same few in turn, and each can be large.
const LIBRARIES: usize = 16;

/// One warm call's progress, shared by the pass's thread and the pieces it hands the pool.
#[derive(Default)]
pub(crate) struct Pass {
    /// The driver's pipeline cache data its makings added, in bytes.
    bytes: AtomicU64,
    /// The current record's creation stopped after its translation ([`translation_only`]).
    translation_only: AtomicBool,
}

thread_local! {
    /// The pass this thread works for now, if any ([`within`]).
    static PASS: RefCell<Option<Arc<Pass>>> = const { RefCell::new(None) };
}

/// Whether this thread works for a warm pass now: the pass's thread, or a pool's piece it handed.
pub(crate) fn warming() -> bool {
    PASS.with(|pass| pass.borrow().is_some())
}

/// The pass this thread works for now, which a piece it hands the pool works for too.
pub(crate) fn pass() -> Option<Arc<Pass>> {
    PASS.with(|pass| pass.borrow().clone())
}

/// Run `f` on this thread as part of `pass`, as it returns or unwinds restoring what the thread
/// worked for before.
pub(crate) fn within<T>(pass: &Arc<Pass>, f: impl FnOnce() -> T) -> T {
    /// Puts the earlier pass back.
    struct Restore(Option<Arc<Pass>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PASS.with(|pass| *pass.borrow_mut() = self.0.clone());
        }
    }
    let earlier = PASS.with(|current| current.replace(Some(Arc::clone(pass))));
    let _restore = Restore(earlier);
    f()
}

/// Count `bytes` of driver data a making added toward this thread's pass, if it works for one
/// (`shader_cache/driver.rs`).
pub(crate) fn added_pipeline_bytes(bytes: u64) {
    PASS.with(|pass| {
        if let Some(pass) = &*pass.borrow() {
            pass.bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    });
}

/// The error a warm creation stops with where its layout bakes an immutable sampler, which a warm
/// pass makes none of (`pipeline/objects.rs`): its translations are kept, its objects and
/// pipelines not made. The pass counts the record as translated, not failed.
pub(crate) fn translation_only() -> crate::Error {
    PASS.with(|pass| {
        if let Some(pass) = &*pass.borrow() {
            pass.translation_only.store(true, Ordering::Relaxed);
        }
    });
    crate::error::internal("a warm pass makes no immutable sampler")
}

/// A device's warm calls: the records queued, whether its thread runs, and the counts.
#[derive(Default)]
pub(crate) struct Warmer {
    queue: Mutex<Queue>,
    counts: Counts,
}

#[derive(Default)]
struct Queue {
    calls: VecDeque<Call>,
    running: bool,
}

/// One warm call: its records in order, and its budget.
struct Call {
    records: Vec<PipelineRecord>,
    byte_budget: u64,
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
    /// The device's first unread record was announced ([`UNREAD`]).
    said_unread: AtomicBool,
}

/// The id name of the warning of a device's first record a warm pass could not read.
pub(crate) const UNREAD: &str = "metal2vulkan-warm-record-unread";

/// What became of one record.
enum Outcome {
    Warmed,
    Translated,
    Failed,
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
        statistics.records_unread = read(&counts.unread);
        statistics.records_not_warmable = read(&counts.not_warmable);
        statistics.records_not_run = read(&counts.not_run);
        statistics.pipeline_bytes = read(&counts.pipeline_bytes);
    }

    /// Count `outcome`, a record's on `device`; the device's first unread record is announced
    /// (the module docs).
    fn count(&self, device: &Device, outcome: Outcome) {
        let counts = &self.counts;
        let counter = match outcome {
            Outcome::Warmed => &counts.warmed,
            Outcome::Translated => &counts.translated,
            Outcome::Failed => &counts.failed,
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
    /// creations of the same descriptors find them. Returns at once.
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
    ///   `metal2vulkan-warm-record-unread`; a creation that fails is counted and dropped. A mesh
    ///   render pipeline's record is counted and not made (its creation keeps nothing past the
    ///   state); a layout that bakes an immutable sampler is translated only.
    pub fn warm(&self, records: Vec<PipelineRecord>, byte_budget: u64) {
        let warmer = &self.shared().warmer;
        let mut queue = lock(&warmer.queue);
        queue.calls.push_back(Call {
            records,
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
        let Some(held) = device.upgrade() else {
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
    let pass = Arc::new(Pass::default());
    let mut libraries = Libraries::default();
    let Call {
        records,
        byte_budget,
    } = call;
    let total = records.len();
    for (at, record) in records.iter().enumerate() {
        let Some(held) = device.upgrade() else {
            return;
        };
        let warmer = &held.shared().warmer;
        if pass.bytes.load(Ordering::Relaxed) >= byte_budget {
            warmer.not_run(total - at);
            return;
        }
        // Every Vulkan call of the creation runs inside this section, which an exit waits for
        // (`exit_guard`); once the process is exiting, nothing more is made.
        let Some(_busy) = exit_guard::enter() else {
            warmer.not_run(total - at);
            return;
        };
        let before = pass.bytes.load(Ordering::Relaxed);
        pass.translation_only.store(false, Ordering::Relaxed);
        let outcome = within(&pass, || warm_record(&held, record, &mut libraries, &pass));
        let added = pass.bytes.load(Ordering::Relaxed) - before;
        warmer
            .counts
            .pipeline_bytes
            .fetch_add(added, Ordering::Relaxed);
        warmer.count(&held, outcome);
    }
}

/// Announce that a record handed to `device`'s warm pass could not be read, for `why`, through
/// the device's message sink, else on stderr: a warning, which ends nothing.
fn say_unread(device: &Device, why: &str) {
    let text = format!(
        "metal2vulkan: a pipeline record handed to Device::warm could not be read ({why}): it is          not warmed, the pass goes on with the next record, and the program's creation of its          descriptor makes what it needs when asked. This device's later unread records are          counted alone (`WarmStatistics::records_unread`)."
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

/// Make on `device` what `record`'s creation makes (the module docs).
fn warm_record(
    device: &Device,
    record: &PipelineRecord,
    libraries: &mut Libraries,
    pass: &Pass,
) -> Outcome {
    let bytes = record.bytes();
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
        Ok(true) => Outcome::Warmed,
        Ok(false) => Outcome::NotWarmable,
        Err(_) if pass.translation_only.load(Ordering::Relaxed) => Outcome::Translated,
        Err(_) => Outcome::Failed,
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
