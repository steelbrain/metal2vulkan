//! The render pipeline creation log: a default-off diagnostic, `METAL2VULKAN_LOG_PIPELINES`
//! (`metal2vulkan::env_vars`), that tells which part of a draw's key made it pay for a `VkPipeline`
//! of its own, and where that creation's time went.
//!
//! - **What it prints.** One line per render `VkPipeline` a device makes for a draw (or a host
//!   step's draw), through the device's message sink at `Info` severity (id name
//!   `metal2vulkan-pipeline-creation`), else on stderr: the state (a number the log gives each
//!   render pipeline state in the order they are made, its label and its functions' names), the
//!   pipeline's number among the state's, the creation's wall time, the SPIR-V words of the
//!   largest module the pipeline is made from (`largest module`; the owner's first-boot review: the
//!   driver's compile time follows it), and its key. The key is printed as fields: the variant key
//!   (a variant is a translation of its own), each stage's SPIR-V (a hash and its length), each
//!   specialization constant (its id, its meaning and its value), each stage's precision notes
//!   under the pipeline's specialization (`notes <stage>`, where it has any: a degrade notice names
//!   a note once per device, so a census of a log counts the states each note reaches; a note the
//!   bound formats decide is named as the module alone reads it), `deferred-discard <stage>` where
//!   the stage's discards wait until after its stores against its driver's store-dropping defect
//!   (its census's `fragment.discard.deferred`), `helper-guard <stage>` with the number of the
//!   stage's selections whose helper invocations take another arm against the same defect (its
//!   census's `fragment.helper-guard`), and the graphics key's other parts (topology, pass,
//!   strides, fixed state, sample positions). A state's first pipeline prints every field; a later
//!   one prints the fields that differ from the state's closest earlier pipeline (the one with the
//!   fewest differing fields), each as `old -> new` with the parts both share elided.
//! - **MoltenVK's split.** Where the device's driver is MoltenVK and its performance tracking is
//!   on (`MVK_CONFIG_PERFORMANCE_TRACKING=1`), the line adds what MoltenVK's own statistics
//!   (`vkGetPerformanceStatisticsMVK`) gained during the creation, per activity: SPIR-V to MSL,
//!   MSL compile, a library from the shader module's cache, function retrieval, specialization,
//!   and the pipeline compile, each as its count and its total milliseconds. The statistics are
//!   the device's: a creation that overlapped another on another thread says so, and its split
//!   then holds both.
//! - **Where the making's time went** (the consumer's cold-boot holds). The line adds `parts:`, a
//!   split of the creation's time as its making notes it on its thread ([`note`]): `content`,
//!   the digests of the pipeline's parts that name its record and its libraries; for a pipeline
//!   linked from libraries, each part's library, `made` by this making (a part with shaders
//!   through its record: `record loaded` where its driver cache started from the record, `record
//!   missed` where not) or `another caller's` (found made, or waited for while another caller
//!   made it), then the `link`; else the whole `pipeline`'s driver making, its record loaded or
//!   missed. The translation of the state's functions is not in it: a variant is translated
//!   before any of its pipelines is made, on the program's thread.
//! - **Bounded, and off by default.** Nothing is kept or measured while the variable is unset.
//!   With it set, each state keeps the fields of its last [`HISTORY`] pipelines, and nothing
//!   else changes: the same pipelines are made, from the same keys, at the same points.
//! - **Numbered where the making begins** (the owner's HITCH rule 1). A pipeline's number and the
//!   fields it differs in are decided where its making begins ([`StateLog::begin`]), on the thread
//!   of the program's call (a draw, or a state's creation, in its keys' order), never on a pipeline
//!   pool's thread or in the order makings finish; the line, with the creation's time, is delivered
//!   once it is made, wherever it was.

use super::graphics::GraphicsKey;
use crate::device::lock;
use crate::Device;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::pipeline::{RenderPipeline, SpecKey};
use metal2vulkan::vk::{DriverId, ShaderStage};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The pipelines a state keeps for its diffs.
const HISTORY: usize = 256;

/// The census name of a fragment function's discards that wait until after its stores.
const DEFERRED_DISCARD_STRATEGY: &str = "fragment.discard.deferred";

/// The census name of a fragment function's selections whose helper invocations take another
/// arm.
const HELPER_GUARD_STRATEGY: &str = "fragment.helper-guard";

/// Whether the log is on: read once, at the first state made.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(metal2vulkan::env_vars::log_pipelines)
}

thread_local! {
    /// The split of the making whose line this thread times now ([`Line::made`]), if any: each
    /// part noted, with how long it took.
    static SPLIT: RefCell<Option<Vec<(String, Duration)>>> = const { RefCell::new(None) };
    /// Whether the cache the driver making this thread ran last was handed holds its record
    /// ([`record_loaded`]), while a split is noted.
    static RECORD: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether this thread notes the split of a making's time now: its line is timed
/// ([`Line::made`]). Each part's timing is taken only then.
pub(crate) fn splitting() -> bool {
    SPLIT.with(|split| split.borrow().is_some())
}

/// Note a part of the making this thread's line times, which took `took`; nothing where none is
/// timed.
pub(crate) fn note(part: impl FnOnce() -> String, took: Duration) {
    SPLIT.with(|split| {
        if let Some(parts) = split.borrow_mut().as_mut() {
            parts.push((part(), took));
        }
    });
}

/// The cache the driver making this thread runs was handed was made from its record, or not
/// (`shader_cache/driver.rs`): a making of a record whose cache the device already holds live says
/// what the making that made it said. While a split is noted.
pub(crate) fn record_loaded(loaded: bool) {
    if splitting() {
        RECORD.with(|record| record.set(Some(loaded)));
    }
}

/// What [`record_loaded`] last noted on this thread, read and reset: `None` where the making
/// kept no record.
pub(crate) fn take_record() -> Option<bool> {
    RECORD.take()
}

/// Run `make`, noting the split of its time on this thread ([`note`]): what it made and the
/// parts noted, the split of a making that encloses it (a line timed around this one) put back
/// after it.
fn split_of<T>(make: impl FnOnce() -> T) -> (T, Vec<(String, Duration)>) {
    let enclosing = SPLIT.replace(Some(Vec::new()));
    let made = make();
    let parts = SPLIT.replace(enclosing).unwrap_or_default();
    (made, parts)
}

/// The split's text in a line: `; parts: ` and each part with its milliseconds, in the order
/// noted; empty where none was noted.
fn split_text(parts: &[(String, Duration)]) -> String {
    if parts.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = parts
        .iter()
        .map(|(part, took)| format!("{part} {:.1} ms", took.as_secs_f64() * 1000.0))
        .collect();
    format!("; parts: {}", parts.join(", "))
}

/// How a part made through its record names it in the split.
pub(crate) fn record_text(record: Option<bool>) -> &'static str {
    match record {
        Some(true) => ", record loaded",
        Some(false) => ", record missed",
        None => "",
    }
}

/// The hash a line names a stage's SPIR-V `words` by: `DefaultHasher`'s (SipHash-1-3, its zero
/// keys) of the words as a slice.
fn spirv_hash(words: &[u32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    words.hash(&mut hasher);
    hasher.finish()
}

/// [`spirv_hash`] of each of `pipeline`'s stages, in its stages' order: what a line of one of its
/// variants' `VkPipeline`s names them by, worked out by the variant beside its objects
/// (`RenderVariant::of`), off the creation's path.
pub(crate) fn spirv_hashes(pipeline: &RenderPipeline) -> Box<[u64]> {
    pipeline
        .stages
        .iter()
        .map(|module| spirv_hash(&module.spirv))
        .collect()
}

/// The states numbered so far.
static STATES: AtomicU64 = AtomicU64::new(0);

/// Creations in progress, and those begun, over the process: whether a creation overlapped
/// another.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static BEGUN: AtomicU64 = AtomicU64::new(0);

/// One render pipeline state's part of the log.
pub(crate) struct StateLog {
    /// How the line names the state.
    name: String,
    /// The fields of its pipelines made so far, the latest last, at most [`HISTORY`].
    made: Mutex<VecDeque<Fields>>,
    /// Its pipelines made so far.
    count: AtomicU64,
}

/// A key's fields, by name.
type Fields = BTreeMap<String, String>;

impl StateLog {
    /// The log of a state labelled `label` whose functions are `functions`, numbered now; `None`
    /// while the log is off.
    pub(crate) fn new(label: Option<&str>, functions: &[String]) -> Option<StateLog> {
        if !enabled() {
            return None;
        }
        let number = STATES.fetch_add(1, Ordering::Relaxed) + 1;
        let label = label.map_or_else(String::new, |label| format!(" {label:?}"));
        Some(StateLog {
            name: format!("state {number}{label} ({})", functions.join(", ")),
            made: Mutex::new(VecDeque::new()),
            count: AtomicU64::new(0),
        })
    }

    /// Make the `VkPipeline` of `key` of the variant of `variant` (compiled as `pipeline`, its
    /// stages' SPIR-V hashed as [`spirv_hashes`] hashes it) on `device` with `make`, and log it:
    /// [`begin`](Self::begin) and [`Line::made`] on this thread.
    pub(crate) fn made<T, E>(
        &self,
        device: &Device,
        keys: (&VariantKey, &RenderPipeline, &GraphicsKey, &[u64]),
        make: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        self.begin(keys).made(device, make)
    }

    /// Begin the line of the `VkPipeline` of `key` of the variant of `variant` (compiled as
    /// `pipeline`, its stages' SPIR-V hashed as `spirv`, [`spirv_hashes`]'s): its number among
    /// the state's, and the fields it differs in from the state's closest earlier pipeline,
    /// decided now, on the thread of the program's call.
    pub(crate) fn begin(
        &self,
        (variant, pipeline, key, spirv): (&VariantKey, &RenderPipeline, &GraphicsKey, &[u64]),
    ) -> Line {
        let number = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        let fields = fields(variant, pipeline, key, spirv);
        let mut made = lock(&self.made);
        let described = describe(&fields, &made, number);
        if made.len() == HISTORY {
            made.pop_front();
        }
        made.push_back(fields);
        Line {
            head: format!("{} pipeline {number}", self.name),
            described,
            note: "",
            words: largest_module(pipeline),
        }
    }
}

/// A pipeline's line, begun ([`StateLog::begin`]): what it names and how its key differs, decided;
/// its making's time and MoltenVK's split to come.
pub(crate) struct Line {
    /// The state and the pipeline's number.
    head: String,
    /// The key's part ([`describe`]).
    described: String,
    /// Where it is made, where not at a draw ([`Line::at_creation`]).
    note: &'static str,
    /// The SPIR-V words of the largest module the pipeline is made from ([`largest_module`]).
    words: usize,
}

impl Line {
    /// The line of a pipeline its state's creation makes: its predicted key's.
    pub(crate) fn at_creation(mut self) -> Line {
        self.note = "; made at its state's creation, the predicted key";
        self
    }

    /// The line of a pipeline its state's creation hands the pool after its predicted key's
    /// general form, nothing waiting for it: the predicted key itself (`render/residual.rs`).
    pub(crate) fn ahead_at_creation(mut self) -> Line {
        self.note = "; begun at its state's creation after its general form, the predicted key";
        self
    }

    /// Make the pipeline with `make` on this thread, and deliver the line.
    pub(crate) fn made<T, E>(
        self,
        device: &Device,
        make: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let statistics = MoltenVk::of(device);
        let before = statistics.as_ref().and_then(MoltenVk::read);
        let overlapped = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) > 0;
        let begun = BEGUN.fetch_add(1, Ordering::SeqCst);
        // The wait before the driver's call (in the pool, for a turn) is named apart from it.
        // A pool's piece waited in its pool for its turn; a program's thread waits inside `make`,
        // and what an earlier making of its left unread is not this one's.
        let queued = match (super::budget::take_waited(), super::pool::in_piece()) {
            (queued, true) => queued,
            (_, false) => std::time::Duration::ZERO,
        };
        let start = Instant::now();
        // This making's split, the one of a making that encloses it put back after it.
        let (made, parts) = split_of(make);
        let turn = super::budget::take_waited();
        let elapsed = start.elapsed().saturating_sub(turn);
        let waited = queued + turn;
        let overlapped = overlapped || BEGUN.load(Ordering::SeqCst) != begun + 1;
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
        let after = statistics.as_ref().and_then(MoltenVk::read);
        let mut line = format!(
            "metal2vulkan: pipeline creation: {}: {} in {:.1} ms",
            self.head,
            match made {
                Ok(_) => "made",
                Err(_) => "failed",
            },
            elapsed.as_secs_f64() * 1000.0
        );
        if waited >= std::time::Duration::from_millis(1) {
            line.push_str(&format!(
                " after waiting {:.1} ms for its turn",
                waited.as_secs_f64() * 1000.0
            ));
        }
        line.push_str(&split_text(&parts));
        line.push_str(&format!("; largest module {} words", self.words));
        line.push_str(self.note);
        match (&statistics, before, after) {
            (Some(_), Some(before), Some(after)) => {
                line.push_str(&split(&before, &after, overlapped));
            }
            (Some(_), _, _) => line.push_str("; MoltenVK's statistics unread"),
            (None, _, _) if overlapped => line.push_str("; overlapped another creation"),
            (None, _, _) => {}
        }
        line.push_str(&self.described);
        deliver(device, line);
        made
    }
}

/// The SPIR-V words of the largest of `pipeline`'s modules its render `VkPipeline` is made from:
/// every stage's but an emulated object or mesh function's compute stage, which runs in a
/// pipeline of its own (`mesh_pipelines`).
fn largest_module(pipeline: &RenderPipeline) -> usize {
    pipeline
        .stages
        .iter()
        .filter(|module| module.stage != ShaderStage::COMPUTE)
        .map(|module| module.spirv.len())
        .max()
        .unwrap_or(0)
}

/// The fields of the `VkPipeline` of `key` of the variant of `variant`, compiled as `pipeline`.
fn fields(
    variant: &VariantKey,
    pipeline: &RenderPipeline,
    key: &GraphicsKey,
    spirv: &[u64],
) -> Fields {
    let mut fields = Fields::new();
    fields.insert("variant".into(), format!("{variant:?}"));
    let mut meanings: BTreeMap<u32, SpecKey> = BTreeMap::new();
    for (i, module) in pipeline.stages.iter().enumerate() {
        let hash = spirv
            .get(i)
            .copied()
            .unwrap_or_else(|| spirv_hash(&module.spirv));
        fields.insert(
            format!("spirv {:?}", module.stage),
            format!("{hash:016x} ({} words)", module.spirv.len()),
        );
        for constant in &module.specialization.constants {
            meanings.entry(constant.id).or_insert(constant.key);
        }
        // The precision notes the stage holds under this pipeline's specialization, which a
        // degrade notice names once per device: a census counts the states each note reaches.
        // A note one of whose clauses tests a value the stage reads as data (a bound texture's
        // format class), which no pipeline's key carries, is listed where a value some draw may
        // bind holds it: the pipeline serves every bound format.
        let notes = module.precision_notes_under(&key.specialization);
        if !notes.is_empty() {
            fields.insert(format!("notes {:?}", module.stage), format!("{notes:?}"));
        }
        // The stage's discards wait until after its stores, against its driver's
        // store-dropping defect (`DriverWorkaround::DiscardDropsSimdgroupStores`).
        if module
            .strategies
            .entries
            .contains_key(DEFERRED_DISCARD_STRATEGY)
        {
            fields.insert(
                format!("deferred-discard {:?}", module.stage),
                "true".into(),
            );
        }
        // The stage's selections whose helper invocations take another arm, against the same
        // defect where no discard is involved.
        if let Some(guard) = module.strategies.entries.get(HELPER_GUARD_STRATEGY) {
            fields.insert(
                format!("helper-guard {:?}", module.stage),
                guard.uses.to_string(),
            );
        }
    }
    let specialization = &key.specialization;
    for entry in &specialization.entries {
        let at = entry.offset as usize;
        let bytes = specialization
            .data
            .get(at..at + entry.size as usize)
            .unwrap_or(&[]);
        let value = match <[u8; 4]>::try_from(bytes) {
            Ok(word) => u32::from_le_bytes(word).to_string(),
            Err(_) => format!("{bytes:02x?}"),
        };
        let name = match meanings.get(&entry.constant_id) {
            Some(meaning) => format!("spec {} {meaning:?}", entry.constant_id),
            None => format!("spec {}", entry.constant_id),
        };
        fields.insert(name, value);
    }
    fields.insert("topology".into(), format!("{:?}", key.topology));
    fields.insert(
        "primitive restart".into(),
        key.primitive_restart.to_string(),
    );
    fields.insert("pass".into(), format!("{:?}", key.pass));
    fields.insert("strides".into(), format!("{:?}", key.strides));
    fields.insert("fixed".into(), format!("{:?}", key.fixed));
    fields.insert(
        "sample positions".into(),
        format!("{:?}", key.sample_positions),
    );
    fields
}

/// The line's key part for the state's pipeline `number`: every field of its first, else the
/// fields that differ from the closest of `earlier`, the state's pipelines kept before it.
fn describe(fields: &Fields, earlier: &VecDeque<Fields>, number: u64) -> String {
    let differing = |other: &Fields| -> Vec<String> {
        let names: std::collections::BTreeSet<&String> =
            fields.keys().chain(other.keys()).collect();
        names
            .into_iter()
            .filter_map(|name| {
                let (old, new) = (other.get(name), fields.get(name));
                (old != new).then(|| {
                    let (old, new) = elided(
                        old.map_or("(none)", String::as_str),
                        new.map_or("(none)", String::as_str),
                    );
                    format!("{name}: {old} -> {new}")
                })
            })
            .collect()
    };
    let closest = earlier
        .iter()
        .enumerate()
        .map(|(at, other)| (at, differing(other)))
        .min_by_key(|(at, diff)| (diff.len(), std::cmp::Reverse(*at)));
    match closest {
        None => {
            let all: Vec<String> = fields
                .iter()
                .map(|(name, value)| format!("{name}: {value}"))
                .collect();
            format!("; its first, key: {}", all.join("; "))
        }
        Some((_, diff)) if diff.is_empty() => {
            "; the same key as an earlier pipeline of the state (made again after the memo let \
             it go)"
                .into()
        }
        Some((at, diff)) => format!(
            "; differs from the state's pipeline {} in {} field(s): {}",
            number - earlier.len() as u64 + at as u64,
            diff.len(),
            diff.join("; ")
        ),
    }
}

/// `old` and `new` with the text they share at both ends elided, back to the part of a debug
/// print they differ in (from after the separator before it to the separator after it). Where
/// they differ right inside a field's bracketed value (`name: {…}`, a set, a list or a tuple), the
/// part runs from the field's name to the value's closing bracket, so the line keeps the name.
fn elided(old: &str, new: &str) -> (String, String) {
    let shared = old
        .char_indices()
        .zip(new.chars())
        .take_while(|((_, a), b)| a == b)
        .last()
        .map_or(0, |((at, a), _)| at + a.len_utf8());
    let opening = |c: char| matches!(c, ',' | '{' | '(' | '[');
    let closing = |c: char| matches!(c, ',' | '}' | ')' | ']');
    let separator = old[..shared].rfind(opening);
    // The bracket a field's value opens with, where the two differ right after it.
    let bracket = separator.and_then(|at| {
        let named = old[at + 1..shared].trim().is_empty() && old[..at].trim_end().ends_with(':');
        let close = match &old[at..=at] {
            "{" => '}',
            "(" => ')',
            "[" => ']',
            _ => return None,
        };
        named.then_some((at, close))
    });
    let start = match bracket {
        Some((at, _)) => old[..at].rfind(opening).map_or(0, |at| at + 1),
        None => separator.map_or(0, |at| at + 1),
    };
    let start = shared - old[start..shared].trim_start().len();
    let suffix = old[shared..]
        .chars()
        .rev()
        .zip(new[shared..].chars().rev())
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a.len_utf8())
        .sum::<usize>();
    let tail = &old[old.len() - suffix..];
    let kept = match tail.find(closing) {
        // The field's closing bracket goes with its value.
        Some(at) if bracket.is_some_and(|(_, close)| tail[at..].starts_with(close)) => at + 1,
        Some(at) => at,
        None => suffix,
    };
    let dropped = suffix - kept;
    let cut = |text: &str| {
        format!(
            "{}{}{}",
            if start > 0 { "…" } else { "" },
            &text[start..text.len() - dropped],
            if dropped > 0 { "…" } else { "" }
        )
    };
    (cut(old), cut(new))
}

/// Hand `line` to the device's message sink, else to stderr.
fn deliver(device: &Device, line: String) {
    device.inform("metal2vulkan-pipeline-creation", line);
}

/// MoltenVK's `MVKPerformanceTracker`, as `mvk_private_api.h` (MoltenVK 1.4) lays it out.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Tracker {
    count: u32,
    latest: f64,
    previous: f64,
    average: f64,
    minimum: f64,
    maximum: f64,
}

impl Tracker {
    /// The milliseconds of all its activities.
    fn total(&self) -> f64 {
        self.average * f64::from(self.count)
    }
}

/// MoltenVK's `MVKPerformanceStatistics`: its shader compilation trackers, then the pipeline
/// cache's (3), the queue's (9) and the device's (1), which the log does not read.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Statistics {
    hash_shader_code: Tracker,
    spirv_to_msl: Tracker,
    msl_compile: Tracker,
    msl_load: Tracker,
    msl_compress: Tracker,
    msl_decompress: Tracker,
    shader_library_from_cache: Tracker,
    function_retrieval: Tracker,
    function_specialization: Tracker,
    pipeline_compile: Tracker,
    glsl_to_spirv: Tracker,
    others: [Tracker; 13],
}

/// `vkGetPerformanceStatisticsMVK`.
type GetStatistics =
    unsafe extern "system" fn(ash::vk::Device, *mut Statistics, *mut usize) -> ash::vk::Result;

/// A MoltenVK device's statistics.
struct MoltenVk {
    get: GetStatistics,
    device: ash::vk::Device,
}

impl MoltenVk {
    /// The statistics of `device`, where its driver is MoltenVK and exports them.
    fn of(device: &Device) -> Option<MoltenVk> {
        if device.properties().description.driver_id != Some(DriverId::MOLTENVK) {
            return None;
        }
        let get = statistics_function()?;
        let device = device.shared().vk().handle();
        Some(MoltenVk { get, device })
    }

    /// The statistics now, where MoltenVK reads them in this layout.
    fn read(&self) -> Option<Statistics> {
        let mut statistics = Statistics::default();
        let mut size = std::mem::size_of::<Statistics>();
        // SAFETY: MoltenVK's own function, called on the device it made (the loader and the layers
        // keep a dispatchable handle the driver's), with a structure of the size passed, which it
        // writes at most.
        let result = unsafe { (self.get)(self.device, &mut statistics, &mut size) };
        (result == ash::vk::Result::SUCCESS && size == std::mem::size_of::<Statistics>())
            .then_some(statistics)
    }
}

/// The split of a creation: what each of MoltenVK's activities gained from `before` to `after`.
fn split(before: &Statistics, after: &Statistics, overlapped: bool) -> String {
    let activities = [
        ("SPIR-V to MSL", before.spirv_to_msl, after.spirv_to_msl),
        ("MSL compile", before.msl_compile, after.msl_compile),
        ("MSL load", before.msl_load, after.msl_load),
        (
            "library from cache",
            before.shader_library_from_cache,
            after.shader_library_from_cache,
        ),
        (
            "function retrieval",
            before.function_retrieval,
            after.function_retrieval,
        ),
        (
            "specialization",
            before.function_specialization,
            after.function_specialization,
        ),
        (
            "pipeline compile",
            before.pipeline_compile,
            after.pipeline_compile,
        ),
    ];
    let parts: Vec<String> = activities
        .iter()
        .filter(|(_, old, new)| new.count != old.count)
        .map(|(name, old, new)| {
            format!(
                "{name} {}x {:.1} ms",
                new.count.wrapping_sub(old.count),
                new.total() - old.total()
            )
        })
        .collect();
    let overlap = match overlapped {
        true => ", overlapped another creation (the split holds both)",
        false => "",
    };
    match parts.is_empty() {
        true => format!(
            "; MoltenVK recorded no activity{overlap} (its performance tracking is off unless \
             MVK_CONFIG_PERFORMANCE_TRACKING=1)"
        ),
        false => format!("; MoltenVK{overlap}: {}", parts.join(", ")),
    }
}

/// MoltenVK's `vkGetPerformanceStatisticsMVK`, where the process loaded MoltenVK: found once,
/// among the symbols the process exports, else in the loaded `libMoltenVK.dylib` (the Vulkan
/// loader opens a driver with its symbols local), else none.
fn statistics_function() -> Option<GetStatistics> {
    static FOUND: OnceLock<Option<usize>> = OnceLock::new();
    let found = *FOUND.get_or_init(find_statistics_function);
    // SAFETY: the address is the exported function's, of `GetStatistics`'s signature
    // (`mvk_private_api.h`).
    found.map(|address| unsafe { std::mem::transmute::<usize, GetStatistics>(address) })
}

// dyld's image list (`<mach-o/dyld.h>`), which `libc` declares only as deprecated.
#[cfg(target_vendor = "apple")]
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(image_index: u32) -> *const std::ffi::c_char;
}

#[cfg(target_vendor = "apple")]
fn find_statistics_function() -> Option<usize> {
    use std::ffi::CStr;
    const NAME: &CStr = c"vkGetPerformanceStatisticsMVK";
    // SAFETY: `dlsym` of a NUL-terminated name in the process's global symbols.
    let global = unsafe { libc::dlsym(libc::RTLD_DEFAULT, NAME.as_ptr()) };
    if !global.is_null() {
        return Some(global as usize);
    }
    // SAFETY: dyld's image list, read by index below its count; each name is a NUL-terminated
    // path dyld keeps while the image is loaded.
    let images = unsafe { _dyld_image_count() };
    for index in 0..images {
        // SAFETY: as above.
        let path = unsafe { _dyld_get_image_name(index) };
        if path.is_null() {
            continue;
        }
        // SAFETY: as above.
        let path = unsafe { CStr::from_ptr(path) };
        if !path.to_bytes().ends_with(b"/libMoltenVK.dylib")
            && !path.to_bytes().ends_with(b"/MoltenVK.framework/MoltenVK")
        {
            continue;
        }
        // SAFETY: `RTLD_NOLOAD` opens only an image already loaded, and its handle is kept (the
        // image stays loaded while the driver does anyway).
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOLOAD | libc::RTLD_LAZY) };
        if handle.is_null() {
            continue;
        }
        // SAFETY: `dlsym` of a NUL-terminated name in an open image.
        let symbol = unsafe { libc::dlsym(handle, NAME.as_ptr()) };
        if !symbol.is_null() {
            return Some(symbol as usize);
        }
    }
    None
}

#[cfg(not(target_vendor = "apple"))]
fn find_statistics_function() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::{
        describe, elided, note, record_loaded, record_text, split_of, split_text, splitting,
        take_record, Fields,
    };
    use std::collections::VecDeque;
    use std::time::Duration;

    /// A making's split holds the parts noted while its line is timed, in order, and nothing
    /// outside one; a line timed inside another keeps its own parts, and the enclosing line's
    /// go on after it. A record's load is noted only while a split is.
    #[test]
    fn a_making_s_split_holds_its_own_parts_in_order() {
        assert!(!splitting());
        note(
            || panic!("a part is named where no line is timed"),
            Duration::ZERO,
        );
        record_loaded(true);
        assert_eq!(take_record(), None, "no record noted outside a split");
        let ((inner, record), outer) = split_of(|| {
            assert!(splitting());
            note(|| "content".to_owned(), Duration::from_micros(300));
            let inner = split_of(|| note(|| "link".to_owned(), Duration::from_millis(2)));
            record_loaded(false);
            let record = take_record();
            note(
                || format!("fragment shader library (made{})", record_text(record)),
                Duration::from_millis(6812),
            );
            (inner, record)
        });
        assert!(!splitting());
        assert_eq!(record, Some(false));
        assert_eq!(inner.1, [("link".to_owned(), Duration::from_millis(2))]);
        assert_eq!(
            split_text(&outer),
            "; parts: content 0.3 ms, fragment shader library (made, record missed) 6812.0 ms"
        );
        assert_eq!(split_text(&[]), "");
    }

    fn fields(pairs: &[(&str, &str)]) -> Fields {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    /// A differing value keeps the part of the print it differs in, the rest elided.
    #[test]
    fn values_keep_the_part_they_differ_in() {
        assert_eq!(elided("2", "5"), ("2".into(), "5".into()));
        assert_eq!(
            elided(
                "Pass { colors: [R8G8B8A8_UNORM], samples: 1 }",
                "Pass { colors: [B8G8R8A8_UNORM], samples: 1 }"
            ),
            (
                "…colors: [R8G8B8A8_UNORM]…".into(),
                "…colors: [B8G8R8A8_UNORM]…".into()
            )
        );
        // A value that differs right inside a field's set keeps the field's name.
        assert_eq!(
            elided(
                "Variant { storage: {}, srgb_textures: {}, ycbcr_textures: {} }",
                "Variant { storage: {}, srgb_textures: {Slot { index: 3 }}, ycbcr_textures: {} }"
            ),
            (
                "…srgb_textures: {}…".into(),
                "…srgb_textures: {Slot { index: 3 }}…".into()
            )
        );
        assert_eq!(
            elided("Pair { at: (1, 2) }", "Pair { at: (3, 2) }"),
            ("…at: (1…".into(), "…at: (3…".into())
        );
        assert_eq!(
            elided(
                "Fixed { cull: NONE, depth: Less }",
                "Fixed { cull: BACK, depth: Less }"
            ),
            ("…cull: NONE…".into(), "…cull: BACK…".into())
        );
        // With no separator, the whole values.
        assert_eq!(elided("ab", "abc"), ("ab".into(), "abc".into()));
    }

    /// A state's first pipeline prints its key; a later one the fields that differ from its
    /// closest earlier pipeline, numbered among the state's.
    #[test]
    fn a_pipeline_names_the_fields_that_differ_from_its_closest() {
        let first = fields(&[("spec 1", "2"), ("spec 2", "7"), ("pass", "P")]);
        assert_eq!(
            describe(&first, &VecDeque::new(), 1),
            "; its first, key: pass: P; spec 1: 2; spec 2: 7"
        );
        let second = fields(&[("spec 1", "3"), ("spec 2", "8"), ("pass", "P")]);
        let third = fields(&[("spec 1", "3"), ("spec 2", "7"), ("pass", "P")]);
        let earlier = VecDeque::from([first, second]);
        assert_eq!(
            describe(&third, &earlier, 3),
            "; differs from the state's pipeline 2 in 1 field(s): spec 2: 8 -> 7"
        );
        let again = earlier[0].clone();
        assert!(describe(&again, &earlier, 3).starts_with("; the same key as an earlier"));
    }
}
