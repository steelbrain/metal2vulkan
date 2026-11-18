//! The player (`docs/REPLAY-STREAM.md`, section 10): it reads a stream's calls in order and drives a
//! runtime through a [`Backend`], relocating what it uploads, waiting for the command buffers
//! that can complete (never for one an event wait holds), and reporting per group of command
//! buffers waited for, per frame and in total what the stream fixed and what the runtime
//! executed, with the checksums the stream asks for and the Private buffer ranges it holds around
//! a dispatch compared.

use super::addresses::AddressMap;
use super::stream::{Call, StreamError, StreamReader};
use super::texwrites::TexWrites;
use super::value::{Hash, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::io::Write;
use std::sync::Arc;

/// A runtime's counters, cumulative, by name.
pub type Counters = BTreeMap<&'static str, u64>;

/// The counts the stream fixes, the same on every play of it on any runtime: the calls played,
/// the command buffers committed, the encoders made, the draws and dispatches their calls issued,
/// the words the player relocated, the 8-byte values at offsets 4 past a multiple of 8 that lie in
/// a live range (left as they are), and the bytes it wrote.
const CALLS: &str = "calls";
const COMMITS: &str = "commits";
const ENCODERS: &str = "encoders";
const DRAWS: &str = "draws";
const DISPATCHES: &str = "dispatches";
const RELOCATED_WORDS: &str = "relocated_words";
const MISALIGNED_WORDS: &str = "misaligned_address_words";
const WRITTEN_BYTES: &str = "written_bytes";
/// The address arguments of the calls played, beside which `relocated_words` shows a data word
/// that only looks like an address.
const ADDRESS_ARGUMENTS: &str = "address_arguments";
/// Writes and uploads into an object with no recorded range, whose words are not relocated.
const UNRELOCATED_WRITES: &str = "unrelocated_writes";
const FIXED: [&str; 10] = [
    CALLS,
    COMMITS,
    ENCODERS,
    DRAWS,
    DISPATCHES,
    RELOCATED_WORDS,
    MISALIGNED_WORDS,
    WRITTEN_BYTES,
    ADDRESS_ARGUMENTS,
    UNRELOCATED_WRITES,
];

/// The host's record of a Private buffer range's bytes around one dispatch
/// (`docs/REPLAY-STREAM.md`, section 8).
const PRIVATE_BYTES: &str = "m2v.privateBytes";

/// The ids of the player's own objects start past this one, above every id a stream gives.
const OWN: u64 = 1 << 63;

/// What a backend made for a call that makes an object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Made {
    /// The address the runtime gave the object's range.
    pub address: Option<u64>,
}

/// One runtime behind the player. Objects are named by their stream ids; the player resolves blobs
/// and relocates before a call reaches the backend, and handles `m2v.write`, `m2v.check`,
/// `m2v.present`, `m2v.commitPart`, `m2v.privateBytes`, `gpuResourceID`, `release` of a device,
/// and the program's waits itself. A copy from the guest's RAM reaches it with its data relocated
/// (the player's `upload` says where the data lands), a record of the host's own plumbing as the
/// plain Metal call it stands for (`plain`), and a copied buffer's descriptor without its
/// `contents`, which reach it as a write once the buffer is made (`copied`).
pub trait Backend {
    /// Run `call`: make its result, or run its receiver's method, or drop the object `release`
    /// names.
    fn call(&mut self, call: &Call) -> Result<Made, PlayError>;
    /// Put `bytes` at `offset` of the memory of block, buffer or surface `target`.
    fn write(&mut self, target: u64, offset: u64, bytes: &[u8]) -> Result<(), PlayError>;
    /// Wait until committed command buffer `command_buffer` completed.
    fn wait(&mut self, command_buffer: u64) -> Result<(), PlayError>;
    /// Completed command buffer `command_buffer`'s GPU start and end, in nanoseconds of one
    /// clock, where the runtime gives them: a frame's GPU busy time is their union
    /// (`Times::gpu_busy`).
    fn gpu_interval(&mut self, command_buffer: u64) -> Option<(u64, u64)>;
    /// The bytes of buffer or texture `object`, for a checksum.
    fn read(&mut self, object: u64) -> Result<Vec<u8>, PlayError>;
    /// The `gpuResourceID` the runtime gives `object`; none where its getter raises.
    fn resource_id(&mut self, object: u64) -> Result<Option<u64>, PlayError>;
    /// The runtime's size and alignment, in bytes, of what `call` places in a placement heap: for
    /// `newTextureWithDescriptor:offset:`, its `heapTextureSizeAndAlignWithDescriptor:` of the
    /// texture descriptor that `descriptor`'s calls make and set, in order (the call that made
    /// it, then its setters); for `newBufferWithLength:options:offset:` (`descriptor` empty), its
    /// `heapBufferSizeAndAlignWithLength:options:`.
    fn heap_size_and_align(
        &mut self,
        call: &Call,
        descriptor: &[Call],
    ) -> Result<(u64, u64), PlayError>;
    /// The runtime's counters now.
    fn counters(&mut self) -> Counters;
    /// Lines of the backend's own for the report, after the player's.
    fn lines(&self) -> Vec<String>;
    /// The runtime's gauges now: what it holds live, which a frame's end reports as it stands
    /// (`Report::ends`), never as growth.
    fn live(&mut self) -> Counters;
    /// Whether the player takes the `m2v.privateBytes` snapshots on this backend: they split the
    /// program's compute encoders around a dispatch with blits between buffers and fences of the
    /// player's own. Where it does not, those dispatches play as they are, counted
    /// (`Report::unsnapshotted`).
    fn snapshots(&self) -> bool;

    /// What the backend skipped so far, of every kind, in one count: the player notes the frame
    /// and call where it first grows (`Report::first_skip`). A backend that skips nothing answers
    /// 0.
    fn skips(&self) -> u64 {
        0
    }

    /// The frame (counted from 0) and the stream position of the call the player plays next, for
    /// a backend that says where it skipped something.
    fn locate(&mut self, _frame: u64, _call: u64) {}

    /// The recording device's `hasUnifiedMemory`, from the stream's `m2v.deviceIdentity` record:
    /// how it kept Managed buffers. A backend with one form of Managed buffer ignores it.
    fn recorded_unified_memory(&mut self, _unified: bool) {}

    /// Whether texture `texture` (not a view) is Managed, where the backend knows
    /// (`Report::texture_writes`).
    fn managed_texture(&self, _texture: u64) -> Option<bool> {
        None
    }
}

/// Why a play stopped.
#[derive(Debug)]
pub enum PlayError {
    Stream(StreamError),
    /// A selector, argument or descriptor key the backend does not implement.
    Refused {
        selector: String,
        why: String,
    },
    /// The runtime refused or failed a call.
    Runtime {
        selector: String,
        error: String,
    },
    /// The stream contradicts itself (an object it never made, a wait nothing can meet).
    Invalid(String),
}

impl fmt::Display for PlayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlayError::Stream(error) => write!(f, "stream: {error}"),
            PlayError::Refused { selector, why } => write!(f, "refused {selector}: {why}"),
            PlayError::Runtime { selector, error } => write!(f, "{selector}: {error}"),
            PlayError::Invalid(why) => write!(f, "invalid stream: {why}"),
        }
    }
}

impl From<StreamError> for PlayError {
    fn from(error: StreamError) -> Self {
        PlayError::Stream(error)
    }
}

/// One `m2v.check` or present readback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Check {
    pub object: u64,
    pub sha256: Hash,
    /// The recording's own checksum, where the stream holds one.
    pub recorded: Option<Hash>,
}

/// An object whose `gpuResourceID` in the runtime is not the one the program saw: the bytes that
/// hold the program's ID name no object of the play.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdDiffers {
    pub object: u64,
    pub recorded: u64,
    /// None where the runtime's getter raises.
    pub played: Option<u64>,
}

/// One `m2v.privateBytes` record compared after its dispatch: how many of the range's 4-byte words
/// (the last one shorter where the length is no multiple of 4) the runtime left otherwise than the
/// recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrivateBytes {
    pub buffer: u64,
    /// The dispatch's position in the stream.
    pub dispatch: u64,
    pub location: u64,
    pub length: u64,
    pub differing: u64,
}

/// What one span of a play counted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Fixed by the stream: the same on every play of it, on any runtime.
    pub stream: Counters,
    /// As the runtime executed it: its counters' growth, which differs between runtimes, and
    /// between plays on one runtime where what it records depends on timing.
    pub executed: Counters,
}

impl Counts {
    fn add(&mut self, from: &Counts) {
        add(&mut self.stream, &from.stream);
        add(&mut self.executed, &from.executed);
    }
}

/// What a play did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Each group of command buffers waited for together (normally the one just committed), with
    /// what was counted since the group before. A group's calls are those played before it
    /// completed: the call that waited for it counts in the next.
    pub completed: Vec<(Vec<u64>, Counts)>,
    /// Each frame's counts: the groups completed since the present before.
    pub frames: Vec<Counts>,
    pub checks: Vec<Check>,
    pub ids: Vec<IdDiffers>,
    pub private: Vec<PrivateBytes>,
    /// How many times the player ended a program's compute encoder and made it again around a
    /// dispatch an `m2v.privateBytes` record names (twice a dispatch).
    pub splits: u64,
    /// The executed counts of the groups that held such a dispatch, which no group, frame or total
    /// holds: the runtime's work for the player's splits is not told apart from the program's
    /// there (a runtime may record the barriers a split needs at the program's next command or at
    /// the commit).
    pub split_executed: Counters,
    /// The dispatches an `m2v.privateBytes` record names that played without the snapshot, as the
    /// backend takes none (`Backend::snapshots`).
    pub unsnapshotted: u64,
    /// The live recorded ranges a newer one met in part, which the address map ended
    /// (`AddressMap::insert`).
    pub parted: u64,
    /// The placement heaps the player laid out again for the runtime's sizes and alignments
    /// (`relay`).
    pub relaid: Relaid,
    /// The whole play's counts, the player's own work left out. Its calls are the calls played:
    /// where a play stopped, the stopping call is the next.
    pub total: Counts,
    /// Each frame's end, as `frames` counts them.
    pub ends: Vec<End>,
    /// The CPU's writes into Managed textures, and those a committed user may still be using.
    pub texture_writes: TextureWrites,
    /// The recorded devices other than the one the guest makes its objects on, whose identity
    /// records the play takes nothing from (`Player::guest_device`).
    pub other_devices: Vec<u64>,
    /// The frame and the call (its position in the stream) where the backend first skipped
    /// something (`Backend::skips`): the play is no verdict past it.
    pub first_skip: Option<(u64, u64)>,
}

/// The CPU's writes into Managed textures (`replaceRegion:...` and any other call on a texture
/// that carries bytes), into the textures the backend says are Managed: how many, their bytes and
/// textures. Of them, those made while a command buffer that used the texture was committed and the
/// stream records no completion of it (only the program's CPU waits record one), by the presents
/// between that user's commit and the write (0, 1, 2, 3 or more), and the first few: position,
/// texture, user and presents. And the writes into textures whose storage the backend does not
/// tell.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextureWrites {
    pub writes: u64,
    pub bytes: u64,
    pub textures: u64,
    pub pending: u64,
    pub pending_bytes: u64,
    pub pending_textures: u64,
    pub by_presents: [u64; 4],
    pub first: Vec<(u64, u64, u64, u64)>,
    pub unknown: u64,
}

impl TextureWrites {
    /// The report's `texture_writes` line.
    pub fn line(&self) -> String {
        let [now, one, two, more] = self.by_presents;
        let first: Vec<String> = (self.first.iter())
            .map(|(at, texture, user, presents)| {
                format!("{at} #{texture} used by #{user} {presents} presents before")
            })
            .collect();
        let first = match first.is_empty() {
            true => String::new(),
            false => format!("; first {}", first.join(", ")),
        };
        format!(
            "texture_writes {} into Managed textures ({} bytes, {} textures); {} with a committed \
             user not seen complete ({} bytes, {} textures), by presents since its commit 0: \
             {now} 1: {one} 2: {two} 3+: {more}; {} into textures of untold storage{first}",
            self.writes,
            self.bytes,
            self.textures,
            self.pending,
            self.pending_bytes,
            self.pending_textures,
            self.unknown,
        )
    }
}

/// A frame's end: how long the frame took to play, and what the process and the runtime held.
/// None of it is fixed by the stream; it differs between plays.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct End {
    /// Nanoseconds from the end of the frame before (the first call played, for the first) to
    /// this one's, the wait for its work included.
    pub wall_nanos: u64,
    /// The process's resident memory then, in bytes, where the system tells it (`VmRSS` on
    /// Linux).
    pub resident_bytes: Option<u64>,
    /// The runtime's gauges then (`Backend::live`).
    pub live: Counters,
    /// Where the frame's wall time went.
    pub times: Times,
}

/// Where a frame's wall time went, as the player sees it from outside the runtime: the
/// nanoseconds its thread spent in each kind of work, and beside them the GPU's busy time. The
/// runtime's own threads are not in it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Times {
    /// Reading and decoding the stream: its calls, and the blobs a write or a snapshot takes.
    pub stream: u64,
    /// Putting bytes in the runtime's memory (`Backend::write`): `m2v.write`s and the snapshots'.
    pub write: u64,
    /// Reading bytes back for checks and snapshots (`Backend::read`).
    pub read: u64,
    /// Waiting for command buffers to complete (`Backend::wait`).
    pub wait: u64,
    /// Calls that make or release a buffer, texture, heap or block, and the longest one.
    pub resource: u64,
    pub resource_max: u64,
    /// Calls that make a library, a function or a pipeline state, and the longest one.
    pub pipeline: u64,
    pub pipeline_max: u64,
    /// Every other call the runtime runs: encoders and their commands, commits, other objects.
    pub other: u64,
    /// The host's copies from the guest's RAM, which the backend stages and copies: the
    /// recording's plumbing, the player's share and not the runtime's.
    pub upload: u64,
    /// Reading the runtime's counters (`Backend::counters`).
    pub count: u64,
    /// The union of the GPU intervals of the command buffers the frame waited for
    /// (`Backend::gpu_interval`); 0 where the backend gives none.
    pub gpu_busy: u64,
}

impl Times {
    /// What of `wall` the player's thread spent on none of the kinds above: the player's own
    /// work.
    pub fn player(&self, wall: u64) -> u64 {
        let host = [
            self.stream,
            self.write,
            self.read,
            self.wait,
            self.resource,
            self.pipeline,
            self.other,
            self.upload,
            self.count,
        ];
        wall.saturating_sub(host.iter().sum())
    }

    fn add(&mut self, cost: Cost, nanos: u64) {
        match cost {
            Cost::Resource => {
                self.resource += nanos;
                self.resource_max = self.resource_max.max(nanos);
            }
            Cost::Pipeline => {
                self.pipeline += nanos;
                self.pipeline_max = self.pipeline_max.max(nanos);
            }
            Cost::Other => self.other += nanos,
            Cost::Upload => self.upload += nanos,
        }
    }
}

/// What a backend call's time counts as (`Times`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cost {
    Resource,
    Pipeline,
    Upload,
    Other,
}

impl Cost {
    /// The cost of a call of `selector`, which makes an object or runs a method; a release's is
    /// its object's (`Player::resources`).
    fn of(selector: &str) -> Cost {
        const RESOURCES: [&str; 5] = [
            "newBuffer",
            "newTexture",
            "newHeap",
            "m2v.newBlock",
            "m2v.newBufferInBlock",
        ];
        const PIPELINES: [&str; 3] = ["newLibrary", "newDefaultLibrary", "newFunction"];
        if RESOURCES.iter().any(|p| selector.starts_with(p)) {
            Cost::Resource
        } else if selector.starts_with("m2v.copyFromGuestRam") {
            Cost::Upload
        } else if PIPELINES.iter().any(|p| selector.starts_with(p))
            || (selector.starts_with("new") && selector.contains("PipelineState"))
        {
            Cost::Pipeline
        } else {
            Cost::Other
        }
    }
}

/// The address arguments `value` holds, at any depth.
fn addresses(value: &Value) -> u64 {
    match value {
        Value::Address(_) => 1,
        Value::List(items) => items.iter().map(addresses).sum(),
        Value::Map(entries) => entries.iter().map(|(_, item)| addresses(item)).sum(),
        _ => 0,
    }
}

/// The nanoseconds since `began`.
fn since(began: std::time::Instant) -> u64 {
    u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Run `call` on `backend`, its time counted in `times` as `cost`.
fn timed<B: Backend>(
    backend: &mut B,
    times: &mut Times,
    call: &Call,
    cost: Cost,
) -> Result<Made, PlayError> {
    let began = std::time::Instant::now();
    let made = backend.call(call);
    times.add(cost, since(began));
    made
}

/// The nanoseconds `intervals` cover, where they overlap counted once.
fn covered(intervals: &mut [(u64, u64)]) -> u64 {
    intervals.sort_unstable();
    let (mut total, mut reach) = (0, 0);
    for &(start, end) in intervals.iter() {
        let start = start.max(reach);
        if end > start {
            total += end - start;
            reach = end;
        }
    }
    total
}

/// The process's resident memory in bytes: `VmRSS` of `/proc/self/status`, where there is one.
fn resident_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kib * 1024)
}

/// The placement heaps whose layout the player changed for the runtime's answers: how many, the
/// offsets it moved in them and the bytes it added to their sizes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Relaid {
    /// The heaps laid out again.
    pub heaps: u64,
    /// The offsets the stream placed at that moved.
    pub moved: u64,
    /// The bytes added to the heaps' sizes.
    pub grown: u64,
}

fn counted(counts: &Counts) -> String {
    let pairs = |counters: &Counters| -> String {
        let pairs = counters.iter();
        pairs
            .map(|(key, value)| format!(" {key}={value}"))
            .collect()
    };
    let (stream, executed) = (pairs(&counts.stream), pairs(&counts.executed));
    format!(" stream{stream} executed{executed}")
}

impl Report {
    /// What a verdict line adds where the backend skipped something: past the first skip, the
    /// play is no verdict.
    pub fn verdict_note(&self) -> Option<String> {
        let (frame, call) = self.first_skip?;
        Some(format!("not a verdict past frame {frame}, call {call}"))
    }

    /// The report's lines (`docs/REPLAY-STREAM.md`, section 10).
    pub fn lines(&self) -> Vec<String> {
        let mut lines = self.lines_but_frames();
        // The frames' lines follow the completed groups' (one line each).
        let at = self.completed.len();
        let frames = (0..self.frames.len()).flat_map(|frame| self.frame_lines(frame));
        lines.splice(at..at, frames);
        lines
    }

    /// Frame `at`'s lines: its counts (`frame`) and its end (`end`).
    pub fn frame_lines(&self, at: usize) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(counters) = self.frames.get(at) {
            lines.push(format!("frame {at}{}", counted(counters)));
        }
        if let Some(end) = self.ends.get(at) {
            let resident =
                (end.resident_bytes).map_or(String::new(), |r| format!(" resident_bytes={r}"));
            let live: String = (end.live.iter())
                .map(|(key, value)| format!(" {key}={value}"))
                .collect();
            let t = &end.times;
            lines.push(format!(
                "end {at} wall_nanos={}{resident} times stream={} write={} read={} wait={} \
                 resource={} resource_max={} pipeline={} pipeline_max={} other={} upload={} \
                 count={} player={} gpu_busy={} live{live}",
                end.wall_nanos,
                t.stream,
                t.write,
                t.read,
                t.wait,
                t.resource,
                t.resource_max,
                t.pipeline,
                t.pipeline_max,
                t.other,
                t.upload,
                t.count,
                t.player(end.wall_nanos),
                t.gpu_busy,
            ));
        }
        lines
    }

    /// The lines but the frames' (`frame_lines`): for a report that wrote each frame's as it
    /// ended (`Player::with_frame_lines`).
    pub fn lines_but_frames(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (ids, counters) in &self.completed {
            let ids: String = ids.iter().map(|id| format!(" #{id}")).collect();
            lines.push(format!("completed{ids}{}", counted(counters)));
        }
        for check in &self.checks {
            let verdict = match check.recorded {
                None => "unrecorded",
                Some(recorded) if recorded == check.sha256 => "match",
                Some(_) => "differs",
            };
            let sha = super::value::hex(&check.sha256);
            lines.push(format!("check #{} {sha} {verdict}", check.object));
        }
        for id in &self.ids {
            let played = id
                .played
                .map_or("none".to_owned(), |played| format!("{played:#x}"));
            lines.push(format!("id #{} {:#x} {played}", id.object, id.recorded));
        }
        if !self.other_devices.is_empty() {
            let devices: String = (self.other_devices.iter())
                .map(|d| format!(" #{d}"))
                .collect();
            lines.push(format!("devices apart from the guest's:{devices}"));
        }
        for private in &self.private {
            let verdict = match private.differing {
                0 => "match".to_owned(),
                differing => format!("differs {differing}/{}", private.length.div_ceil(4)),
            };
            let (at, length) = (private.location, private.length);
            let (buffer, dispatch) = (private.buffer, private.dispatch);
            lines.push(format!(
                "private #{buffer} {dispatch} {at}+{length} {verdict}"
            ));
        }
        if self.splits > 0 {
            let executed = self.split_executed.iter();
            let executed: String = executed
                .map(|(key, value)| format!(" {key}={value}"))
                .collect();
            lines.push(format!("split {} executed{executed}", self.splits));
        }
        if self.unsnapshotted > 0 {
            lines.push(format!("unsnapshotted {}", self.unsnapshotted));
        }
        if self.parted > 0 {
            lines.push(format!("parted {}", self.parted));
        }
        let relaid = self.relaid;
        if relaid.heaps > 0 {
            lines.push(format!(
                "relaid {} moved {} grown {}",
                relaid.heaps, relaid.moved, relaid.grown
            ));
        }
        lines.push(self.texture_writes.line());
        lines.push(format!("total{}", counted(&self.total)));
        lines
    }
}

/// What the player knows of an object.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Device,
    CommandBuffer,
    /// An object a command buffer or an encoder made.
    Encoder,
    Event,
    Other,
}

/// A command buffer not yet waited for.
#[derive(Default)]
struct Pending {
    queue: u64,
    waits: Vec<(u64, u64)>,
    signals: Vec<(u64, u64)>,
    committed: bool,
    /// Released by the stream while it could not yet be waited for.
    released: bool,
    /// The Private buffer ranges around its dispatches, compared once it completed.
    snapshots: Vec<Snapshot>,
}

/// A Private buffer range the stream holds the bytes of before and after one dispatch: an
/// `m2v.privateBytes` record's receiver, range and blobs.
#[derive(Clone, Copy)]
struct Around {
    buffer: u64,
    location: u64,
    length: u64,
    before: Hash,
    after: Hash,
}

impl Around {
    /// The dispatch's position and the range of `m2v.privateBytes` record `call`:
    /// `[dispatch, {location, length}, before, after]`.
    fn of(call: &Call) -> Result<(u64, Around), PlayError> {
        let bad = || {
            invalid(format!(
                "{PRIVATE_BYTES} takes [dispatch, {{location, length}}, before, after], not {}",
                call
            ))
        };
        let [dispatch, Value::Map(range), Value::Blob(before), Value::Blob(after)] = &call.args[..]
        else {
            return Err(bad());
        };
        let field = |key: &str| range.iter().find(|(k, _)| &**k == key)?.1.as_uint();
        let (Some(dispatch), Some(location), Some(length)) =
            (dispatch.as_uint(), field("location"), field("length"))
        else {
            return Err(bad());
        };
        let around = Around {
            buffer: call.receiver,
            location,
            length,
            before: *before,
            after: *after,
        };
        Ok((dispatch, around))
    }
}

/// An open compute encoder, kept while a record names a later dispatch: the call that made it and
/// each call that set its state (`set…` and `use…`), which a fresh encoder takes again.
struct Encoder {
    made: Call,
    state: Vec<Call>,
}

/// The ranges around one dispatch, until its command buffer completes: `staging`, a Shared buffer
/// of the player's own, holds their bytes before, then from `half` their bytes after; `fence`, of
/// the player's own too, orders the copies and the dispatch.
struct Snapshot {
    dispatch: u64,
    staging: u64,
    fence: u64,
    half: u64,
    /// Each range and where its bytes lie in each half.
    ranges: Vec<(Around, u64)>,
}

/// What a placement heap holds at each recorded offset: the largest size and alignment the
/// runtime answers for the resources placed there.
type Placements = BTreeMap<u64, (u64, u64)>;

/// A placement heap's layout on the runtime: its size, and the offset the player places each
/// resource at by the offset the stream places it at.
struct Layout {
    size: u64,
    offsets: HashMap<u64, u64>,
}

/// The layout of a placement heap of `size` bytes, as a program asking the runtime's sizes and
/// alignments lays it out, whose resources the stream places at the offsets of `at`, each with the
/// most bytes and the largest alignment the runtime answers for its resources there. Each offset
/// keeps at least its distance from the one below, moves up until the resources below it end, and
/// rounds up to its alignment; the heap keeps at least its distance past the last offset and holds
/// the resources there. Resources at one offset still share it; resources at different offsets no
/// longer meet. Where every resource ends before the next offset and every offset is aligned, the
/// layout is the stream's.
fn relay(size: u64, at: &Placements) -> Layout {
    let mut offsets = HashMap::with_capacity(at.len());
    let mut below: Option<(u64, u64, u64)> = None;
    for (&recorded, &(bytes, align)) in at {
        let floor = match below {
            None => recorded,
            Some((was, placed, held)) => (placed + (recorded - was)).max(placed + held),
        };
        let placed = floor.next_multiple_of(align.max(1));
        offsets.insert(recorded, placed);
        below = Some((recorded, placed, bytes));
    }
    let size = match below {
        None => size,
        Some((was, placed, held)) => (placed + size.saturating_sub(was)).max(placed + held),
    };
    Layout { size, offsets }
}

/// Whether `selector` makes a texture descriptor (`docs/REPLAY-STREAM.md`, section 9).
fn makes_texture_descriptor(selector: &str) -> bool {
    selector == "m2v.textureDescriptor" || selector.starts_with("MTLTextureDescriptor.")
}

/// The recorded size of the heap `call` (`newHeapWithDescriptor:`) makes, where it is a placement
/// heap (`MTLHeapTypePlacement`).
fn placement_heap(call: &Call) -> Option<u64> {
    let entries = call.args.first()?.as_map()?;
    let field = |key: &str| entries.iter().find(|(k, _)| &**k == key)?.1.as_uint();
    (field("heapType")? == 1).then_some(field("size")?)
}

/// The argument of a placement heap's call that is the offset it places at.
fn placement_offset(selector: &str) -> Option<usize> {
    match selector {
        "newTextureWithDescriptor:offset:" => Some(1),
        "newBufferWithLength:options:offset:" => Some(2),
        _ => None,
    }
}

/// Plays one stream on one backend.
pub struct Player<'b, B: Backend> {
    backend: &'b mut B,
    addresses: AddressMap,
    kinds: HashMap<u64, Kind>,
    /// Each live object's device and the number of objects made before it.
    owner: HashMap<u64, (u64, u64)>,
    made_count: u64,
    pending: HashMap<u64, Pending>,
    /// The committed command buffers not yet waited for, in commit order.
    committed: Vec<u64>,
    /// Each event's value as far as the stream has set or completed it.
    events: HashMap<u64, u64>,
    /// The live objects whose `gpuResourceID` the player compared.
    compared: HashSet<u64>,
    /// The live texture descriptors marked `outside` and the textures made of them: objects outside
    /// the program's IDs, whose `gpuResourceID` the program never saw (section 5).
    outside: HashSet<u64>,
    /// The `m2v.privateBytes` records, read ahead, by the position of the dispatch each names.
    ahead: HashMap<u64, Vec<Around>>,
    /// The placement heaps' layouts on the runtime, read ahead, by heap.
    layouts: HashMap<u64, Layout>,
    /// The recorded device the guest makes its objects on, read ahead: the one that most calls
    /// making an object have as their receiver (the lowest id of a tie). Only its identity record
    /// tells the backend the Managed form; none where no call makes an object on a device.
    guest_device: Option<u64>,
    /// The next call's position: the calls stepped so far.
    position: u64,
    /// The compute encoders open while a record names a later dispatch.
    encoders: HashMap<u64, Encoder>,
    /// The snapshots whose command buffers completed, not yet compared.
    done: VecDeque<Snapshot>,
    /// The objects of the player's own made so far.
    own_made: u64,
    baseline: Counters,
    /// The counts the stream fixes, since the last group.
    fixed: Counters,
    frame: Counts,
    /// When the frame being played began: the first call played, or the frame before's end.
    frame_began: Option<std::time::Instant>,
    /// Whether the player takes the `m2v.privateBytes` snapshots where the backend does
    /// (`Player::without_snapshots`).
    snapshots: bool,
    /// Where the frame's wall time went so far, and the GPU intervals of the command buffers it
    /// waited for.
    times: Times,
    intervals: Vec<(u64, u64)>,
    /// The live objects that calls of `Cost::Resource` made, whose release costs as theirs.
    resources: HashSet<u64>,
    report: Report,
    /// The last frame to play (`until_frame`), counted from 0 as `Report::frames` counts them.
    last_frame: Option<usize>,
    /// Where each frame's lines go as it ends (`with_frame_lines`).
    frame_out: Option<&'b mut dyn Write>,
    /// The CPU's writes into Managed textures and their textures' users.
    texwrites: TexWrites,
}

/// `now` past `base`, key by key.
fn grown(now: &Counters, base: &Counters) -> Counters {
    let grown = now.iter();
    let grown = grown.map(|(&k, &v)| (k, v.saturating_sub(base.get(k).copied().unwrap_or(0))));
    grown.collect()
}

fn add(into: &mut Counters, from: &Counters) {
    for (&k, &v) in from {
        *into.entry(k).or_default() += v;
    }
}

/// A call the player makes of its own: `selector` on `receiver` with `args`, making `result`.
fn own_call(selector: &str, receiver: u64, result: u64, args: &[Value]) -> Call {
    Call {
        selector: Arc::from(selector),
        receiver,
        result,
        args: args.to_vec(),
        answers: Vec::new(),
    }
}

fn invalid(why: impl Into<String>) -> PlayError {
    PlayError::Invalid(why.into())
}

/// Where an upload's data lands: in `target`, each piece (the `pieces` argument's entry of that
/// index) from the target's byte paired with it.
pub(super) struct Upload {
    pub target: u64,
    pub pieces: Vec<(u64, usize)>,
}

/// The argument of an upload that lists its pieces, `[[byte, data], ...]`.
const PIECES: usize = 1;

/// The data of piece `entry` of an upload's `args` ([`upload`] checked it is bytes).
pub(super) fn piece(args: &[Value], entry: usize) -> &[u8] {
    match &args[PIECES].as_list().expect("an upload's pieces")[entry] {
        Value::List(pair) => pair[1].as_bytes().expect("a piece's bytes"),
        _ => unreachable!("an upload's piece"),
    }
}

/// The guest RAM runs of a copy back to it, each `[byte, [start, end]]`: each byte of the buffer
/// and its run's length. The runs' addresses are the host's, which nothing reads.
pub(super) fn placed_runs(runs: &Value) -> Option<Vec<(u64, u64)>> {
    let placed = |run: &Value| match run.as_list()? {
        [at, run] => match run.as_uints().as_deref() {
            Some(&[start, end]) if start <= end => Some((at.as_uint()?, end - start)),
            _ => None,
        },
        _ => None,
    };
    runs.as_list()?.iter().map(placed).collect()
}

/// Where the data of `call` lands, where it is a copy from the guest's RAM
/// (`m2v.copyFromGuestRam`, `m2v.copyFromGuestRamAt`; `docs/REPLAY-STREAM.md`, section 7):
/// `[buffer, [[byte, data], ...]]`, the data resolved bytes.
pub(super) fn upload(call: &Call) -> Result<Option<Upload>, PlayError> {
    if !matches!(
        &*call.selector,
        "m2v.copyFromGuestRam" | "m2v.copyFromGuestRamAt"
    ) {
        return Ok(None);
    }
    let bad = |what: &str| invalid(format!("{}: {what}", call.selector));
    let target = call.args.first().and_then(Value::as_object);
    let target = target.ok_or_else(|| bad("argument 0 is no buffer"))?;
    let entries = call.args.get(PIECES).and_then(Value::as_list);
    let entries = entries.ok_or_else(|| bad("argument 1 is no list of pieces"))?;
    let pieces = entries
        .iter()
        .enumerate()
        .map(|(entry, pair)| match pair.as_list() {
            Some([at, Value::Bytes(_)]) => at.as_uint().map(|at| (at, entry)),
            _ => None,
        });
    let pieces: Option<Vec<_>> = pieces.collect();
    let pieces = pieces.ok_or_else(|| bad("a piece is no [byte, data]"))?;
    Ok(Some(Upload { target, pieces }))
}

/// The plain Metal call a record of the host's own plumbing stands for
/// (`docs/REPLAY-STREAM.md`, section 4): a buffer the host made exportable, or whose texture
/// rows may keep bytes between them, is `newBufferWithLength:options:`; a queue it kept apart,
/// `newCommandQueue`. None for an exported buffer the host did not make (result 0): the program
/// then made its buffer by a later call.
fn plain(mut call: Call) -> Option<Call> {
    match &*call.selector {
        "m2v.newBufferOfLengthPaddingUnobserved"
        | "m2v.newExportableBufferOfLength"
        | "m2v.newDmaBufBufferOfLengthPaddingUnobserved" => {
            if call.result == 0 {
                return None;
            }
            call.selector = Arc::from("newBufferWithLength:options:");
            call.args.truncate(2);
        }
        "m2v.newCommandQueueUnshared" => call.selector = Arc::from("newCommandQueue"),
        _ => {}
    }
    Some(call)
}

/// Whether the host committed the command buffer of `m2v.commitPart` `call`: where its answer `cut`
/// is `"Cut"`, or absent; not where it is `"Fails"` or `{Whole: reason}`.
fn cut(call: &Call) -> Result<bool, PlayError> {
    let Some(cut) = call.answer("cut") else {
        return Ok(true);
    };
    match cut {
        Value::Str(cut) if cut == "Cut" => Ok(true),
        Value::Str(cut) if cut == "Fails" => Ok(false),
        Value::Map(whole) if whole.len() == 1 && &*whole[0].0 == "Whole" => Ok(false),
        _ => Err(invalid(format!("m2v.commitPart's cut {cut}"))),
    }
}

/// `descriptor`, a `newBufferWithDescriptor:` map, with its `contents` nil, and the bytes they
/// held: what a buffer made as a copy of host memory starts with, which the player writes once the
/// buffer is made (`docs/REPLAY-STREAM.md`, section 5).
fn copied(descriptor: Value) -> (Value, Option<Vec<u8>>) {
    let Value::Map(fields) = descriptor else {
        return (descriptor, None);
    };
    let mut contents = None;
    let fields = fields.into_iter().map(|(key, value)| match value {
        Value::Bytes(bytes) if &*key == "contents" => {
            contents = Some(bytes);
            (key, Value::Nil)
        }
        value => (key, value),
    });
    let fields = fields.collect();
    (Value::Map(fields), contents)
}

/// Whether a blob of `around` holds as many bytes as its range.
fn sized(around: &Around, blob: &[u8]) -> Result<(), PlayError> {
    match blob.len() as u64 == around.length {
        true => Ok(()),
        false => Err(invalid(format!(
            "{PRIVATE_BYTES} of #{}: a blob of {} bytes for a range of {}",
            around.buffer,
            blob.len(),
            around.length
        ))),
    }
}

/// The program's waits for a command buffer, as a stream records them: names of records, not
/// waits (the player's own waits are its backend's, under `gpu::wait`). Spelled in pieces so the
/// check that every device wait is bounded (`gpu::tests`) finds none here.
pub const WAIT_COMPLETED: &str = concat!("waitUntil", "Completed");
pub const WAIT_SCHEDULED: &str = concat!("waitUntil", "Scheduled");

/// The calls whose first argument is data the GPU reads as constants, which may hold addresses.
fn sets_bytes(selector: &str) -> bool {
    selector.starts_with("set") && selector.ends_with("Bytes:length:atIndex:")
}

impl<'b, B: Backend> Player<'b, B> {
    pub fn new(backend: &'b mut B) -> Self {
        let baseline = backend.counters();
        let fixed = FIXED.into_iter().map(|key| (key, 0)).collect();
        Player {
            backend,
            addresses: AddressMap::default(),
            kinds: HashMap::new(),
            owner: HashMap::new(),
            made_count: 0,
            pending: HashMap::new(),
            committed: Vec::new(),
            events: HashMap::new(),
            compared: HashSet::new(),
            outside: HashSet::new(),
            ahead: HashMap::new(),
            layouts: HashMap::new(),
            guest_device: None,
            position: 0,
            encoders: HashMap::new(),
            done: VecDeque::new(),
            own_made: 0,
            baseline,
            fixed,
            frame: Counts::default(),
            frame_began: None,
            snapshots: true,
            times: Times::default(),
            intervals: Vec::new(),
            resources: HashSet::new(),
            report: Report::default(),
            last_frame: None,
            frame_out: None,
            texwrites: TexWrites::default(),
        }
    }

    /// The player that stops after frame `last`'s `m2v.present`, counted from 0 as
    /// `Report::frames` counts them: it completes the work committed so far and reports as at the
    /// stream's end.
    pub fn until_frame(mut self, last: usize) -> Self {
        self.last_frame = Some(last);
        self
    }

    /// The player that writes each frame's lines (`Report::frame_lines`) to `out` as the frame
    /// ends, flushed, so that a play that never ends leaves the frames it finished.
    pub fn with_frame_lines(mut self, out: &'b mut dyn Write) -> Self {
        self.frame_out = Some(out);
        self
    }

    /// The player that takes no `m2v.privateBytes` snapshots on any backend: every dispatch plays as
    /// the program made it, counted (`Report::unsnapshotted`), so that two runtimes' plays do the
    /// same work and their times and memory compare.
    pub fn without_snapshots(mut self) -> Self {
        self.snapshots = false;
        self
    }

    /// Play every call of `stream`; the report holds what ran, whether or not it all did.
    pub fn play(mut self, stream: &mut StreamReader) -> (Report, Result<(), PlayError>) {
        let start = self.baseline.clone();
        let mut left_out = Counters::new();
        let result = self.read_ahead(stream);
        let result = result.and_then(|()| self.all(stream, &mut left_out));
        let ended = result.and_then(|()| self.finish());
        let ended = ended.and_then(|()| self.compare(stream, &mut left_out));
        let now = self.backend.counters();
        let mut executed = grown(&now, &start);
        let split = &self.report.split_executed;
        for (key, value) in &mut executed {
            let out = [&left_out, split].map(|out| out.get(key).copied().unwrap_or(0));
            *value = value.saturating_sub(out.iter().sum());
        }
        let mut stream = self.fixed.clone();
        for (_, group) in &self.report.completed {
            add(&mut stream, &group.stream);
        }
        self.report.total = Counts { stream, executed };
        self.report.texture_writes = self.texwrites.counts();
        (self.report, ended)
    }

    fn all(&mut self, stream: &mut StreamReader, left_out: &mut Counters) -> Result<(), PlayError> {
        loop {
            self.frame_began.get_or_insert_with(std::time::Instant::now);
            let began = std::time::Instant::now();
            let next = stream.next_call()?;
            self.times.stream += since(began);
            let Some(call) = next else {
                break;
            };
            let issued = self.issued(&call);
            let (frame, skips) = (self.report.frames.len() as u64, self.backend.skips());
            self.backend.locate(frame, self.position);
            match self.ahead.remove(&self.position) {
                Some(ranges) if self.snapshots && self.backend.snapshots() => {
                    self.split(stream, call, ranges, left_out)?
                }
                Some(_) => {
                    self.report.unsnapshotted += 1;
                    self.step(stream, call, left_out)?
                }
                None => self.step(stream, call, left_out)?,
            }
            if self.report.first_skip.is_none() && self.backend.skips() > skips {
                self.report.first_skip = Some((frame, self.position));
            }
            self.position += 1;
            for key in [Some(CALLS), issued].into_iter().flatten() {
                *self.fixed.entry(key).or_default() += 1;
            }
            self.compare(stream, left_out)?;
            if (self.last_frame).is_some_and(|last| self.report.frames.len() > last) {
                break;
            }
        }
        Ok(())
    }

    /// Index the `m2v.privateBytes` records by the dispatch each names, which comes before it, and
    /// lay out each placement heap for the runtime's answers (`relay`). A stream error ends the
    /// reading: the play meets it where it lies, as it meets a placement whose size the runtime
    /// does not answer.
    fn read_ahead(&mut self, stream: &StreamReader) -> Result<(), PlayError> {
        let mut ahead = stream.reopened()?;
        let mut position = 0;
        // Each live texture descriptor's calls: the one that made it, then its setters.
        let mut descriptors: HashMap<u64, Vec<Call>> = HashMap::new();
        // Each placement heap's recorded size and, by offset, the most bytes and the largest
        // alignment the runtime answers for what the stream places there.
        let mut heaps: HashMap<u64, (u64, Placements)> = HashMap::new();
        // The objects made on each recorded device.
        let mut made_on: BTreeMap<u64, u64> = BTreeMap::new();
        while let Ok(Some(call)) = ahead.next_call() {
            if &*call.selector == "MTLCreateSystemDefaultDevice" {
                made_on.entry(call.result).or_default();
            } else if call.result != 0 {
                if let Some(made) = made_on.get_mut(&call.receiver) {
                    *made += 1;
                }
            }
            match &*call.selector {
                PRIVATE_BYTES => {
                    let (dispatch, around) = Around::of(&call)?;
                    if dispatch >= position {
                        return Err(invalid(format!(
                            "{PRIVATE_BYTES} at call {position} names call {dispatch}, no earlier one"
                        )));
                    }
                    self.ahead.entry(dispatch).or_default().push(around);
                }
                "release" => drop(descriptors.remove(&call.receiver)),
                "newHeapWithDescriptor:" => {
                    if let Some(size) = placement_heap(&call) {
                        heaps.insert(call.result, (size, BTreeMap::new()));
                    }
                }
                selector if call.result != 0 && makes_texture_descriptor(selector) => {
                    descriptors.insert(call.result, vec![call]);
                }
                selector => match descriptors.get_mut(&call.receiver) {
                    Some(calls) => calls.push(call),
                    None => {
                        let at = placement_offset(selector);
                        let heap = heaps.get_mut(&call.receiver);
                        let (Some(at), Some((_, placed))) = (at, heap) else {
                            position += 1;
                            continue;
                        };
                        let offset = call.args.get(at).and_then(Value::as_uint);
                        let descriptor = call.args.first().and_then(Value::as_object);
                        let descriptor = descriptor.and_then(|d| descriptors.get(&d));
                        let descriptor = descriptor.map_or(&[][..], Vec::as_slice);
                        let answer = self.backend.heap_size_and_align(&call, descriptor);
                        if let (Some(offset), Ok((bytes, align))) = (offset, answer) {
                            let most = placed.entry(offset).or_default();
                            *most = (most.0.max(bytes), most.1.max(align));
                        }
                    }
                },
            }
            position += 1;
        }
        let most = made_on.iter().filter(|(_, &made)| made > 0);
        let most = most.max_by_key(|&(&device, &made)| (made, std::cmp::Reverse(device)));
        self.guest_device = most.map(|(&device, _)| device);
        self.report.other_devices = (made_on.keys().copied())
            .filter(|&device| self.guest_device.is_some_and(|guest| guest != device))
            .collect();
        for (heap, (size, placed)) in heaps {
            let layout = relay(size, &placed);
            let moved = layout.offsets.iter().filter(|(was, now)| was != now);
            let moved = moved.count() as u64;
            if moved > 0 || layout.size != size {
                self.report.relaid.heaps += 1;
                self.report.relaid.moved += moved;
                self.report.relaid.grown += layout.size.saturating_sub(size);
                self.layouts.insert(heap, layout);
            }
        }
        Ok(())
    }

    /// `call` with the size of the placement heap it makes, or the offset it places at in one, as
    /// the heap's layout on the runtime has it (`relay`).
    fn placed(&self, mut call: Call) -> Call {
        if &*call.selector == "newHeapWithDescriptor:" {
            let Some(layout) = self.layouts.get(&call.result) else {
                return call;
            };
            if let Some(Value::Map(entries)) = call.args.first_mut() {
                for (key, value) in entries {
                    if &**key == "size" {
                        *value = Value::Uint(layout.size);
                    }
                }
            }
            return call;
        }
        let (Some(at), Some(layout)) = (
            placement_offset(&call.selector),
            self.layouts.get(&call.receiver),
        ) else {
            return call;
        };
        if let Some(offset) = call.args.get_mut(at) {
            let placed = offset.as_uint().and_then(|o| layout.offsets.get(&o));
            if let Some(&placed) = placed {
                *offset = Value::Uint(placed);
            }
        }
        call
    }

    fn finish(&mut self) -> Result<(), PlayError> {
        self.drain()?;
        if self.committed.is_empty() {
            return Ok(());
        }
        Err(invalid(format!(
            "command buffers {:?} still wait on events at the stream's end",
            self.committed
        )))
    }

    fn step(
        &mut self,
        stream: &mut StreamReader,
        call: Call,
        left_out: &mut Counters,
    ) -> Result<(), PlayError> {
        // A message to nil does nothing: a call of receiver 0 that makes nothing and names no
        // class (a class's call is `MTLX.selector`, the host's `m2v.selector`).
        if call.receiver == 0 && call.result == 0 && !call.selector.contains('.') {
            return Ok(());
        }
        match &*call.selector {
            "m2v.write" => self.write(stream, &call),
            "m2v.check" => {
                let recorded = call.answer("sha256").and_then(Value::as_list);
                let objects = call.args.iter().enumerate().map(|(at, object)| {
                    let wanted = recorded.and_then(|r| r.get(at)).and_then(Value::as_bytes);
                    (object.as_object(), wanted.and_then(|w| w.try_into().ok()))
                });
                let objects: Vec<_> = objects.collect();
                self.drain()?;
                for (object, recorded) in objects {
                    let object = object.ok_or_else(|| invalid("m2v.check of no object"))?;
                    self.check(object, recorded, left_out)?;
                }
                Ok(())
            }
            "m2v.present" => {
                self.drain()?;
                self.texwrites.presented();
                self.report.frames.push(self.frame.clone());
                self.frame = Counts::default();
                let now = std::time::Instant::now();
                let began = self.frame_began.replace(now).unwrap_or(now);
                let mut times = self.times;
                times.gpu_busy = covered(&mut self.intervals);
                self.times = Times::default();
                self.intervals.clear();
                self.report.ends.push(End {
                    wall_nanos: u64::try_from((now - began).as_nanos()).unwrap_or(u64::MAX),
                    resident_bytes: resident_bytes(),
                    live: self.backend.live(),
                    times,
                });
                if let Some(out) = &mut self.frame_out {
                    // A report line lost to a full disk loses only itself, as in the report file.
                    let at = self.report.frames.len() - 1;
                    for line in self.report.frame_lines(at) {
                        let _ = writeln!(out, "{line}");
                    }
                    let _ = out.flush();
                }
                Ok(())
            }
            "commit" => self.commit(call.receiver),
            "m2v.commitPart" => match cut(&call)? {
                true => self.commit(call.receiver),
                false => Ok(()),
            },
            "gpuResourceID" => self.resource_id(&call),
            // A split or part of result 0 is one the host did not cut.
            "m2v.split" | "m2v.part" if call.result == 0 => Ok(()),
            "m2v.split" => {
                let queue = self.queue_of(call.receiver)?;
                self.commit(call.receiver)?;
                self.part(queue, call.result)
            }
            "m2v.part" => {
                let queue = self.queue_of(call.receiver)?;
                self.part(queue, call.result)
            }
            // The guest's device's identity: another device's (the host's own) does not say how the
            // guest's Managed buffers were kept.
            "m2v.deviceIdentity" => {
                let unified = call.answer("hasUnifiedMemory").and_then(Value::as_bool);
                let guest = self.guest_device.is_none_or(|guest| guest == call.receiver);
                if let (Some(unified), true) = (unified, guest) {
                    self.backend.recorded_unified_memory(unified);
                }
                Ok(())
            }
            "m2v.setCompletionJoinsNext"
            | "m2v.holdsNoPredication"
            | "m2v.rebindNoCopy"
            | "m2v.fold" => Ok(()),
            // Read ahead: it plays around the dispatch it names (`split`).
            PRIVATE_BYTES => Ok(()),
            WAIT_COMPLETED | WAIT_SCHEDULED => {
                if &*call.selector == WAIT_COMPLETED {
                    self.texwrites.waited(call.receiver);
                }
                self.drain()?;
                match self.committed.contains(&call.receiver) {
                    true => Err(invalid(format!(
                        "the CPU waits for #{}, which an event no record signals holds",
                        call.receiver
                    ))),
                    false => Ok(()),
                }
            }
            "encodeWaitForEvent:value:" | "encodeSignalEvent:value:" => {
                let event = call.args.first().and_then(Value::as_object);
                let value = call.args.get(1).and_then(Value::as_uint);
                let (Some(event), Some(value)) = (event, value) else {
                    return Err(invalid(format!("{} of no event and value", call.selector)));
                };
                let pending = self.pending.get_mut(&call.receiver);
                let pending =
                    pending.ok_or_else(|| invalid("an event command of no command buffer"))?;
                match &*call.selector {
                    "encodeWaitForEvent:value:" => pending.waits.push((event, value)),
                    _ => pending.signals.push((event, value)),
                }
                self.run(&call).map(drop)
            }
            "setSignaledValue:" => {
                let value = call.args.first().and_then(Value::as_uint);
                let value = value.ok_or_else(|| invalid("setSignaledValue: of no value"))?;
                self.run(&call)?;
                self.events.insert(call.receiver, value);
                self.drain()
            }
            "waitUntilSignaledValue:timeoutMS:" => {
                let value = call.args.first().and_then(Value::as_uint).unwrap_or(0);
                self.texwrites.event_waited(call.receiver, value);
                self.drain()?;
                match self.events.get(&call.receiver).copied().unwrap_or(0) >= value {
                    true => Ok(()),
                    false => Err(invalid(format!(
                        "the CPU waits for #{} to reach {value}, which no record reaches",
                        call.receiver
                    ))),
                }
            }
            "release" => {
                self.texwrites.released(call.receiver);
                self.release(call.receiver)
            }
            _ => self.make(stream, call),
        }
    }

    /// The count `call` adds once played, where it is an encoder's draw or dispatch.
    fn issued(&self, call: &Call) -> Option<&'static str> {
        if self.kinds.get(&call.receiver) != Some(&Kind::Encoder) {
            return None;
        }
        match &*call.selector {
            selector if selector.starts_with("draw") => Some(DRAWS),
            selector if selector.starts_with("dispatch") => Some(DISPATCHES),
            _ => None,
        }
    }

    /// Commit `command_buffer`, then wait for what can complete.
    fn commit(&mut self, command_buffer: u64) -> Result<(), PlayError> {
        let pending = self.pending.get_mut(&command_buffer);
        let pending = pending.ok_or_else(|| invalid("a commit of no command buffer"))?;
        pending.committed = true;
        let (queue, signals) = (pending.queue, pending.signals.clone());
        self.texwrites.committed(command_buffer, queue, &signals);
        self.committed.push(command_buffer);
        self.run(&own_call("commit", command_buffer, 0, &[]))?;
        *self.fixed.entry(COMMITS).or_default() += 1;
        self.drain()
    }

    /// The queue of `command_buffer`, which is not yet waited for.
    fn queue_of(&self, command_buffer: u64) -> Result<u64, PlayError> {
        let queue = self.pending.get(&command_buffer).map(|p| p.queue);
        queue.ok_or_else(|| invalid("a part of no command buffer"))
    }

    /// Make `part`, a command buffer of `queue`.
    fn part(&mut self, queue: u64, part: u64) -> Result<(), PlayError> {
        let call = own_call("commandBuffer", queue, part, &[]);
        self.run(&call)?;
        self.made(&call, part);
        Ok(())
    }

    /// Compare the ID the program saw of `call`'s receiver, where it saw one, with the runtime's,
    /// once per object, and report the object where they differ.
    fn resource_id(&mut self, call: &Call) -> Result<(), PlayError> {
        let object = call.receiver;
        if !self.kinds.contains_key(&object) {
            return Err(invalid(format!(
                "gpuResourceID of #{object}, made by no call"
            )));
        }
        if self.outside.contains(&object) {
            return Err(invalid(format!(
                "gpuResourceID of #{object}, a texture its descriptor marks outside the \
                 program's IDs"
            )));
        }
        // A getter that raised answers nothing.
        let Some(recorded) = call.answer("gpuResourceID") else {
            return Ok(());
        };
        let recorded = recorded.as_uint();
        let recorded = recorded.ok_or_else(|| invalid("gpuResourceID answers no uint"))?;
        if !self.compared.insert(object) {
            return Ok(());
        }
        let played = self.backend.resource_id(object)?;
        if played != Some(recorded) {
            let differs = IdDiffers {
                object,
                recorded,
                played,
            };
            self.report.ids.push(differs);
        }
        Ok(())
    }

    /// A call the backend runs, its blobs and addresses resolved, and what it made recorded; a
    /// copied buffer's `contents` written once it is made.
    fn make(&mut self, stream: &mut StreamReader, call: Call) -> Result<(), PlayError> {
        let Some(call) = plain(call) else {
            return Ok(());
        };
        let mut relocated = 0;
        let mut args = Vec::with_capacity(call.args.len());
        let mut contents = None;
        for (at, arg) in call.args.iter().enumerate() {
            let arg = self.resolve(stream, arg, &mut relocated)?;
            let arg = match (at, &*call.selector) {
                (0, "newBufferWithDescriptor:") => {
                    let (descriptor, bytes) = copied(arg);
                    contents = bytes;
                    descriptor
                }
                _ => arg,
            };
            args.push(arg);
        }
        *self.fixed.entry(RELOCATED_WORDS).or_default() += relocated;
        *self.fixed.entry(ADDRESS_ARGUMENTS).or_default() +=
            args.iter().map(addresses).sum::<u64>();
        if sets_bytes(&call.selector) {
            if let Some(Value::Bytes(data)) = args.first_mut() {
                self.relocate(data, 0);
            }
        }
        let resolved = self.placed(Call { args, ..call });
        let resolved = self.uploaded(resolved)?;
        let made = self.run(&resolved)?;
        self.track(&resolved);
        let backend = &*self.backend;
        let managed = |texture| backend.managed_texture(texture);
        self.texwrites.saw(&resolved, self.position, managed);
        let object = resolved.result;
        if object == 0 {
            return Ok(());
        }
        self.made(&resolved, object);
        if Cost::of(&resolved.selector) == Cost::Resource {
            self.resources.insert(object);
        }
        let address = resolved.answer("gpuAddress");
        let length = resolved.answer("gpuLength");
        if address.is_some() || length.is_some() {
            let (Some(address), Some(length)) = (
                address.and_then(Value::as_address),
                length.and_then(Value::as_uint),
            ) else {
                return Err(invalid(format!(
                    "#{object}: gpuAddress takes an address and gpuLength a uint, both or neither"
                )));
            };
            let base = made.address.ok_or_else(|| PlayError::Runtime {
                selector: resolved.selector.to_string(),
                error: format!("#{object} has a recorded range and no address in the runtime"),
            })?;
            let ended = self.addresses.insert(object, address, length, base);
            self.report.parted += ended.map_err(PlayError::Invalid)?;
        }
        match contents {
            Some(bytes) => self.put(object, 0, bytes),
            None => Ok(()),
        }
    }

    /// `call` with the data of the upload it is, if any, relocated where its target answered a
    /// `gpuAddress`, and the data counted.
    fn uploaded(&mut self, mut call: Call) -> Result<Call, PlayError> {
        let Some(upload) = upload(&call)? else {
            return Ok(call);
        };
        let relocated = self.addresses.holds(upload.target);
        if !relocated {
            *self.fixed.entry(UNRELOCATED_WRITES).or_default() += 1;
        }
        let Some(Value::List(entries)) = call.args.get_mut(PIECES) else {
            unreachable!("an upload's pieces");
        };
        for (at, entry) in upload.pieces {
            let Value::List(pair) = &mut entries[entry] else {
                unreachable!("an upload's piece");
            };
            let Value::Bytes(data) = &mut pair[1] else {
                unreachable!("a piece's bytes");
            };
            if relocated {
                self.relocate(data, at);
            }
            *self.fixed.entry(WRITTEN_BYTES).or_default() += data.len() as u64;
        }
        Ok(call)
    }

    /// Record what `call` made: its kind and its device.
    fn made(&mut self, call: &Call, object: u64) {
        let selector = &*call.selector;
        let kind = match selector {
            "MTLCreateSystemDefaultDevice" => Kind::Device,
            "newEvent" | "newSharedEvent" | "newSharedEventWithHandle:" => Kind::Event,
            _ if selector.starts_with("commandBuffer") => Kind::CommandBuffer,
            _ => match self.kinds.get(&call.receiver) {
                Some(Kind::CommandBuffer | Kind::Encoder) => Kind::Encoder,
                _ => Kind::Other,
            },
        };
        self.kinds.insert(object, kind);
        let marked = match (selector, call.args.first()) {
            ("m2v.textureDescriptor", Some(Value::Map(fields))) => (fields.iter())
                .any(|(key, value)| &**key == "outside" && *value == Value::Bool(true)),
            ("newTextureWithDescriptor:", Some(Value::Object(descriptor))) => {
                self.outside.contains(descriptor)
            }
            _ => false,
        };
        if marked {
            self.outside.insert(object);
        }
        if kind == Kind::Encoder {
            *self.fixed.entry(ENCODERS).or_default() += 1;
        }
        if kind == Kind::CommandBuffer {
            let pending = Pending {
                queue: call.receiver,
                ..Pending::default()
            };
            self.pending.insert(object, pending);
        }
        let device = match kind {
            Kind::Device => Some(object),
            _ => match self.kinds.get(&call.receiver) {
                Some(Kind::Device) => Some(call.receiver),
                _ => self.owner.get(&call.receiver).map(|&(device, _)| device),
            },
        };
        if let Some(device) = device {
            self.owner.insert(object, (device, self.made_count));
        }
        self.made_count += 1;
    }

    /// `value` with its blobs read and its addresses relocated.
    fn resolve(
        &self,
        stream: &mut StreamReader,
        value: &Value,
        relocated: &mut u64,
    ) -> Result<Value, PlayError> {
        Ok(match value {
            Value::Blob(hash) => Value::Bytes(stream.blob(hash)?),
            Value::Address(at) => match self.addresses.relocated(*at) {
                Some(moved) => {
                    *relocated += 1;
                    Value::Address(moved)
                }
                None => Value::Address(*at),
            },
            Value::List(items) => Value::List(
                items
                    .iter()
                    .map(|item| self.resolve(stream, item, relocated))
                    .collect::<Result<_, _>>()?,
            ),
            Value::Map(entries) => Value::Map(
                entries
                    .iter()
                    .map(|(key, item)| Ok((key.clone(), self.resolve(stream, item, relocated)?)))
                    .collect::<Result<_, PlayError>>()?,
            ),
            other => other.clone(),
        })
    }

    fn write(&mut self, stream: &mut StreamReader, call: &Call) -> Result<(), PlayError> {
        let target = call.receiver;
        if !self.kinds.contains_key(&target) {
            return Err(invalid(format!("m2v.write to #{target}, made by no call")));
        }
        let (pairs, rest) = call.args.as_chunks::<2>();
        if pairs.is_empty() || !rest.is_empty() {
            return Err(invalid("m2v.write takes pairs of an offset and its data"));
        }
        self.drain()?;
        for [offset, data] in pairs {
            let offset = offset.as_uint();
            let offset = offset.ok_or_else(|| invalid("m2v.write of no offset"))?;
            let data = match data {
                Value::Bytes(data) => data.clone(),
                Value::Blob(hash) => self.blob(stream, hash)?,
                _ => return Err(invalid("m2v.write of no data")),
            };
            self.put(target, offset, data)?;
        }
        Ok(())
    }

    /// Put `data` at `offset` of `target`, relocated where the target answered a `gpuAddress`, and
    /// count it.
    fn put(&mut self, target: u64, offset: u64, mut data: Vec<u8>) -> Result<(), PlayError> {
        match self.addresses.holds(target) {
            true => self.relocate(&mut data, offset),
            false => *self.fixed.entry(UNRELOCATED_WRITES).or_default() += 1,
        }
        *self.fixed.entry(WRITTEN_BYTES).or_default() += data.len() as u64;
        self.write_bytes(target, offset, &data)
    }

    /// Relocate `data`, which starts `offset` bytes into its target, and count what it found.
    fn relocate(&mut self, data: &mut [u8], offset: u64) {
        let (moved, misaligned) = self.addresses.relocate(data, offset);
        *self.fixed.entry(RELOCATED_WORDS).or_default() += moved;
        *self.fixed.entry(MISALIGNED_WORDS).or_default() += misaligned;
    }

    /// Read `object` and record its checksum; the readback's own work is no command buffer's.
    fn check(
        &mut self,
        object: u64,
        recorded: Option<Hash>,
        left_out: &mut Counters,
    ) -> Result<(), PlayError> {
        let bytes = self.uncounted(left_out, |player| player.read_back(object))?;
        self.report.checks.push(Check {
            object,
            sha256: Sha256::digest(&bytes).into(),
            recorded,
        });
        Ok(())
    }

    /// Run `work`, the player's own, and leave what the runtime counted while it ran out of the
    /// counts: out of the group it would fall in, and into `left_out`, which the total leaves out.
    fn uncounted<T>(
        &mut self,
        left_out: &mut Counters,
        work: impl FnOnce(&mut Self) -> Result<T, PlayError>,
    ) -> Result<T, PlayError> {
        let before = self.counters();
        let done = work(self);
        let own = grown(&self.counters(), &before);
        add(left_out, &own);
        add(&mut self.baseline, &own);
        done
    }

    /// Keep what a split needs of the compute encoders open while a record names a later dispatch:
    /// the call that made each and each call that set its state.
    fn track(&mut self, call: &Call) {
        let selector = &*call.selector;
        if selector.starts_with("computeCommandEncoder") {
            if !self.ahead.is_empty() {
                let made = call.clone();
                let open = Encoder {
                    made,
                    state: Vec::new(),
                };
                self.encoders.insert(call.result, open);
            }
            return;
        }
        let Some(open) = self.encoders.get_mut(&call.receiver) else {
            return;
        };
        if selector == "endEncoding" {
            self.encoders.remove(&call.receiver);
        } else if selector.starts_with("set") || selector.starts_with("use") {
            open.state.push(call.clone());
        }
    }

    /// A fresh id for an object of the player's own.
    fn own_object(&mut self) -> u64 {
        self.own_made += 1;
        OWN + self.own_made
    }

    /// Run `selector` on `receiver` with `args`, making `result`: a call of the player's own.
    fn own(
        &mut self,
        receiver: u64,
        selector: &str,
        result: u64,
        args: &[Value],
    ) -> Result<(), PlayError> {
        let call = own_call(selector, receiver, result, args);
        self.run(&call).map(drop)
    }

    /// Play dispatch `call` with the Private buffer `ranges` around it: put each range's bytes
    /// before in a Shared buffer of the player's own; then end the dispatch's compute encoder,
    /// copy them into the ranges in a blit encoder and make the compute encoder afresh with the
    /// state the program gave it; dispatch; then the same with a blit that copies the ranges out.
    /// A fence of the player's own orders each encoder after the one before it. The ranges are
    /// compared once the command buffer completed.
    fn split(
        &mut self,
        stream: &mut StreamReader,
        call: Call,
        ranges: Vec<Around>,
        left_out: &mut Counters,
    ) -> Result<(), PlayError> {
        let dispatch = self.position;
        let open = match self.issued(&call) {
            Some(DISPATCHES) => self.encoders.get(&call.receiver),
            _ => None,
        };
        let Some(made) = open.map(|open| open.made.clone()) else {
            return Err(invalid(format!(
                "{PRIVATE_BYTES} names call {dispatch}, {}, no compute encoder's dispatch",
                call.selector
            )));
        };
        let command_buffer = made.receiver;
        let device = self.owner.get(&command_buffer).map(|&(device, _)| device);
        let device = device.ok_or_else(|| invalid(format!("#{command_buffer} has no device")))?;
        let mut placed = Vec::with_capacity(ranges.len());
        let mut half = 0;
        for around in ranges {
            placed.push((around, half));
            half = (half + around.length).next_multiple_of(16);
        }
        let (staging, fence) = (self.own_object(), self.own_object());
        let copy_in = |&(around, at): &(Around, u64)| {
            let (buffer, location) = (around.buffer, around.location);
            [staging, at, buffer, location, around.length]
        };
        let length = [Value::Uint((2 * half).max(16)), Value::Uint(0)];
        self.own(device, "newBufferWithLength:options:", staging, &length)?;
        self.own(device, "newFence", fence, &[])?;
        for &(around, at) in &placed {
            let mut before = self.blob(stream, &around.before)?;
            sized(&around, &before)?;
            if self.addresses.holds(around.buffer) {
                self.relocate(&mut before, around.location);
            }
            *self.fixed.entry(WRITTEN_BYTES).or_default() += around.length;
            self.write_bytes(staging, at, &before)?;
        }
        self.blit_between(&made, fence, placed.iter().map(copy_in))?;
        self.step(stream, call, left_out)?;
        let copy_out = |&(around, at): &(Around, u64)| {
            let (buffer, location) = (around.buffer, around.location);
            [buffer, location, staging, half + at, around.length]
        };
        self.blit_between(&made, fence, placed.iter().map(copy_out))?;
        let pending = self.pending.get_mut(&command_buffer);
        let pending = pending.ok_or_else(|| invalid("a dispatch of no open command buffer"))?;
        pending.snapshots.push(Snapshot {
            dispatch,
            staging,
            fence,
            half,
            ranges: placed,
        });
        Ok(())
    }

    /// End compute encoder `made`'s result, copy `copies` (each `[from, offset, to, offset, size]`)
    /// in a blit encoder of its command buffer, and make the encoder afresh with the state the
    /// program gave it; each encoder waits for `fence`, which the one before it updates.
    fn blit_between(
        &mut self,
        made: &Call,
        fence: u64,
        copies: impl Iterator<Item = [u64; 5]>,
    ) -> Result<(), PlayError> {
        let (encoder, command_buffer) = (made.result, made.receiver);
        let fenced = [Value::Object(fence)];
        self.own(encoder, "updateFence:", 0, &fenced)?;
        self.own(encoder, "endEncoding", 0, &[])?;
        let blit = self.own_object();
        self.own(command_buffer, "blitCommandEncoder", blit, &[])?;
        self.own(blit, "waitForFence:", 0, &fenced)?;
        for [from, offset, to, to_offset, size] in copies {
            let args = [
                Value::Object(from),
                Value::Uint(offset),
                Value::Object(to),
                Value::Uint(to_offset),
                Value::Uint(size),
            ];
            let copy = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
            self.own(blit, copy, 0, &args)?;
        }
        self.own(blit, "updateFence:", 0, &fenced)?;
        self.own(blit, "endEncoding", 0, &[])?;
        self.backend_release(blit)?;
        self.run(made)?;
        self.report.splits += 1;
        self.own(encoder, "waitForFence:", 0, &fenced)?;
        if let Some(open) = self.encoders.get(&encoder) {
            for state in &open.state {
                timed(
                    &mut *self.backend,
                    &mut self.times,
                    state,
                    Cost::of(&state.selector),
                )?;
            }
        }
        Ok(())
    }

    /// Compare each range of the snapshots whose command buffers completed with the recording's
    /// bytes after its dispatch, word by word, relocated as its bytes before were.
    fn compare(
        &mut self,
        stream: &mut StreamReader,
        left_out: &mut Counters,
    ) -> Result<(), PlayError> {
        while let Some(snapshot) = self.done.pop_front() {
            let (staging, fence) = (snapshot.staging, snapshot.fence);
            let bytes = self.uncounted(left_out, |player| {
                let bytes = player.read_back(staging)?;
                player.backend_release(staging)?;
                player.backend_release(fence)?;
                Ok(bytes)
            })?;
            for (around, at) in snapshot.ranges {
                let mut after = stream.blob(&around.after)?;
                sized(&around, &after)?;
                if self.addresses.holds(around.buffer) {
                    self.addresses.relocate(&mut after, around.location);
                }
                let start = (snapshot.half + at) as usize;
                let played = bytes.get(start..start + after.len());
                let played = played.ok_or_else(|| PlayError::Runtime {
                    selector: PRIVATE_BYTES.into(),
                    error: format!("#{} read back {} bytes", snapshot.staging, bytes.len()),
                })?;
                let words = played.chunks(4).zip(after.chunks(4));
                let differing = words.filter(|(played, after)| played != after).count();
                self.report.private.push(PrivateBytes {
                    buffer: around.buffer,
                    dispatch: snapshot.dispatch,
                    location: around.location,
                    length: around.length,
                    differing: differing as u64,
                });
            }
        }
        Ok(())
    }

    /// Drop `object`: a device with everything it made, a command buffer once it completed.
    fn release(&mut self, object: u64) -> Result<(), PlayError> {
        let kind = self.kinds.remove(&object);
        if kind.is_none() {
            return Err(invalid(format!("release of #{object}, made by no call")));
        }
        if kind == Some(Kind::Device) {
            let mut owned: Vec<(u64, u64)> = (self.owner.iter())
                .filter(|&(&id, &(device, _))| device == object && id != object)
                .map(|(&id, &(_, made))| (made, id))
                .collect();
            owned.sort_unstable();
            for (_, id) in owned.into_iter().rev() {
                self.kinds.remove(&id);
                self.drop_object(id)?;
            }
        }
        self.drop_object(object)
    }

    fn drop_object(&mut self, object: u64) -> Result<(), PlayError> {
        self.encoders.remove(&object);
        self.addresses.remove(object);
        self.owner.remove(&object);
        self.compared.remove(&object);
        self.outside.remove(&object);
        if let Some(pending) = self.pending.get_mut(&object) {
            if pending.committed {
                pending.released = true;
                return Ok(());
            }
            self.pending.remove(&object);
        }
        self.backend_release(object)
    }

    fn backend_release(&mut self, object: u64) -> Result<(), PlayError> {
        let cost = match self.resources.remove(&object) {
            true => Cost::Resource,
            false => Cost::Other,
        };
        let call = own_call("release", object, 0, &[]);
        timed(&mut *self.backend, &mut self.times, &call, cost).map(drop)
    }

    /// Run `call` on the backend, its time counted by its selector (`Cost::of`).
    fn run(&mut self, call: &Call) -> Result<Made, PlayError> {
        timed(
            &mut *self.backend,
            &mut self.times,
            call,
            Cost::of(&call.selector),
        )
    }

    /// Put `bytes` at `offset` of `target` on the backend, timed (`Times::write`).
    fn write_bytes(&mut self, target: u64, offset: u64, bytes: &[u8]) -> Result<(), PlayError> {
        let began = std::time::Instant::now();
        let written = self.backend.write(target, offset, bytes);
        self.times.write += since(began);
        written
    }

    /// The bytes of `object` on the backend, timed (`Times::read`).
    fn read_back(&mut self, object: u64) -> Result<Vec<u8>, PlayError> {
        let began = std::time::Instant::now();
        let bytes = self.backend.read(object);
        self.times.read += since(began);
        bytes
    }

    /// The runtime's counters now, timed (`Times::count`).
    fn counters(&mut self) -> Counters {
        let began = std::time::Instant::now();
        let counters = self.backend.counters();
        self.times.count += since(began);
        counters
    }

    /// The bytes of blob `hash` of `stream`, timed (`Times::stream`).
    fn blob(&mut self, stream: &mut StreamReader, hash: &Hash) -> Result<Vec<u8>, PlayError> {
        let began = std::time::Instant::now();
        let bytes = stream.blob(hash);
        self.times.stream += since(began);
        Ok(bytes?)
    }

    /// The committed command buffers that can complete, in commit order: each one whose queue's
    /// earlier ones can, and whose event waits the CPU's values or their signals meet.
    fn runnable(&self) -> Vec<u64> {
        let mut values = self.events.clone();
        let mut ready: HashSet<u64> = HashSet::new();
        loop {
            let mut grew = false;
            let mut held: HashSet<u64> = HashSet::new();
            for id in &self.committed {
                let pending = &self.pending[id];
                if ready.contains(id) {
                    continue;
                }
                let met =
                    |(event, value): &(u64, u64)| values.get(event).copied().unwrap_or(0) >= *value;
                if held.contains(&pending.queue) || !pending.waits.iter().all(met) {
                    held.insert(pending.queue);
                    continue;
                }
                ready.insert(*id);
                for &(event, value) in &pending.signals {
                    let known = values.entry(event).or_default();
                    *known = (*known).max(value);
                }
                grew = true;
            }
            if !grew {
                break;
            }
        }
        self.committed
            .iter()
            .copied()
            .filter(|id| ready.contains(id))
            .collect()
    }

    /// Wait for every command buffer that can complete, and record the counters' growth for them:
    /// for a group that held a split (`split`), in the report's split counts instead.
    fn drain(&mut self) -> Result<(), PlayError> {
        let ready = self.runnable();
        if ready.is_empty() {
            return Ok(());
        }
        let mut split = false;
        for id in &ready {
            let began = std::time::Instant::now();
            self.backend.wait(*id)?;
            self.times.wait += since(began);
            if let Some(interval) = self.backend.gpu_interval(*id) {
                self.intervals.push(interval);
            }
            let pending = self.pending.remove(id).expect("a pending command buffer");
            split |= !pending.snapshots.is_empty();
            self.done.extend(pending.snapshots);
            for (event, value) in pending.signals {
                let known = self.events.entry(event).or_default();
                *known = (*known).max(value);
            }
            if pending.released {
                self.backend_release(*id)?;
            }
        }
        self.committed.retain(|id| !ready.contains(id));
        let now = self.counters();
        let executed = grown(&now, &self.baseline);
        let executed = match split {
            false => executed,
            true => {
                add(&mut self.report.split_executed, &executed);
                Counters::new()
            }
        };
        let group = Counts {
            stream: self.fixed.clone(),
            executed,
        };
        self.frame.add(&group);
        self.report.completed.push((ready, group));
        self.baseline = now;
        self.fixed.values_mut().for_each(|v| *v = 0);
        Ok(())
    }
}
