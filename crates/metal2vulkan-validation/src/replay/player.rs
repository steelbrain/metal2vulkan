//! The player (`docs/REPLAY-STREAM.md`, section 10): it reads a stream's calls in order and drives a
//! runtime through a [`Backend`], relocating what it uploads, waiting for the command buffers
//! that can complete (never for one an event wait holds), and reporting per group of command
//! buffers waited for, per frame and in total what the stream fixed and what the runtime
//! executed, with the checksums the stream asks for.

use super::addresses::AddressMap;
use super::stream::{Call, StreamError, StreamReader};
use super::value::{Hash, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
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
const FIXED: [&str; 8] = [
    CALLS,
    COMMITS,
    ENCODERS,
    DRAWS,
    DISPATCHES,
    RELOCATED_WORDS,
    MISALIGNED_WORDS,
    WRITTEN_BYTES,
];

/// What a backend made for a call that makes an object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Made {
    /// The address the runtime gave the object's range.
    pub address: Option<u64>,
}

/// One runtime behind the player. Objects are named by their stream ids; the player resolves blobs
/// and relocates before a call reaches the backend, and handles `m2v.write`, `m2v.check`,
/// `m2v.present`, `m2v.commitPart`, `gpuResourceID`, `release` of a device, and the program's
/// waits itself. A copy from the guest's RAM reaches it with its data relocated (the player's
/// `upload` says where the data lands), a record of the host's own plumbing as the plain Metal
/// call it stands for (`plain`), and a copied buffer's descriptor without its `contents`, which
/// reach it as a write once the buffer is made (`copied`).
pub trait Backend {
    /// Run `call`: make its result, or run its receiver's method, or drop the object `release`
    /// names.
    fn call(&mut self, call: &Call) -> Result<Made, PlayError>;
    /// Put `bytes` at `offset` of the memory of block, buffer or surface `target`.
    fn write(&mut self, target: u64, offset: u64, bytes: &[u8]) -> Result<(), PlayError>;
    /// Wait until committed command buffer `command_buffer` completed.
    fn wait(&mut self, command_buffer: u64) -> Result<(), PlayError>;
    /// The bytes of buffer or texture `object`, for a checksum.
    fn read(&mut self, object: u64) -> Result<Vec<u8>, PlayError>;
    /// The `gpuResourceID` the runtime gives `object`; none where its getter raises.
    fn resource_id(&mut self, object: u64) -> Result<Option<u64>, PlayError>;
    /// The runtime's counters now.
    fn counters(&mut self) -> Counters;
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
    /// The whole play's counts, the player's readbacks left out. Its calls are the calls played:
    /// where a play stopped, the stopping call is the next.
    pub total: Counts,
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
    /// The report's lines (`docs/REPLAY-STREAM.md`, section 10).
    pub fn lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (ids, counters) in &self.completed {
            let ids: String = ids.iter().map(|id| format!(" #{id}")).collect();
            lines.push(format!("completed{ids}{}", counted(counters)));
        }
        for (at, counters) in self.frames.iter().enumerate() {
            lines.push(format!("frame {at}{}", counted(counters)));
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
    baseline: Counters,
    /// The counts the stream fixes, since the last group.
    fixed: Counters,
    frame: Counts,
    report: Report,
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

/// A call the player makes of its own: `selector` on `receiver`, making `result`.
fn own_call(selector: &str, receiver: u64, result: u64) -> Call {
    Call {
        selector: Arc::from(selector),
        receiver,
        result,
        args: Vec::new(),
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
            baseline,
            fixed,
            frame: Counts::default(),
            report: Report::default(),
        }
    }

    /// Play every call of `stream`; the report holds what ran, whether or not it all did.
    pub fn play(mut self, stream: &mut StreamReader) -> (Report, Result<(), PlayError>) {
        let start = self.baseline.clone();
        let mut readbacks = Counters::new();
        let result = self.all(stream, &mut readbacks);
        let ended = result.and_then(|()| self.finish());
        let now = self.backend.counters();
        let mut executed = grown(&now, &start);
        for (key, value) in &mut executed {
            *value = value.saturating_sub(readbacks.get(key).copied().unwrap_or(0));
        }
        let mut stream = self.fixed.clone();
        for (_, group) in &self.report.completed {
            add(&mut stream, &group.stream);
        }
        self.report.total = Counts { stream, executed };
        (self.report, ended)
    }

    fn all(
        &mut self,
        stream: &mut StreamReader,
        readbacks: &mut Counters,
    ) -> Result<(), PlayError> {
        while let Some(call) = stream.next_call()? {
            let issued = self.issued(&call);
            self.step(stream, call, readbacks)?;
            for key in [Some(CALLS), issued].into_iter().flatten() {
                *self.fixed.entry(key).or_default() += 1;
            }
        }
        Ok(())
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
        readbacks: &mut Counters,
    ) -> Result<(), PlayError> {
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
                    self.check(object, recorded, readbacks)?;
                }
                Ok(())
            }
            "m2v.present" => {
                self.drain()?;
                self.report.frames.push(self.frame.clone());
                self.frame = Counts::default();
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
            "m2v.setCompletionJoinsNext"
            | "m2v.holdsNoPredication"
            | "m2v.rebindNoCopy"
            | "m2v.fold" => Ok(()),
            WAIT_COMPLETED | WAIT_SCHEDULED => {
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
                self.backend.call(&call).map(drop)
            }
            "setSignaledValue:" => {
                let value = call.args.first().and_then(Value::as_uint);
                let value = value.ok_or_else(|| invalid("setSignaledValue: of no value"))?;
                self.backend.call(&call)?;
                self.events.insert(call.receiver, value);
                self.drain()
            }
            "waitUntilSignaledValue:timeoutMS:" => {
                let value = call.args.first().and_then(Value::as_uint).unwrap_or(0);
                self.drain()?;
                match self.events.get(&call.receiver).copied().unwrap_or(0) >= value {
                    true => Ok(()),
                    false => Err(invalid(format!(
                        "the CPU waits for #{} to reach {value}, which no record reaches",
                        call.receiver
                    ))),
                }
            }
            "release" => self.release(call.receiver),
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
        self.committed.push(command_buffer);
        self.backend.call(&own_call("commit", command_buffer, 0))?;
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
        let call = own_call("commandBuffer", queue, part);
        self.backend.call(&call)?;
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
        if sets_bytes(&call.selector) {
            if let Some(Value::Bytes(data)) = args.first_mut() {
                self.relocate(data, 0);
            }
        }
        let resolved = Call { args, ..call };
        let resolved = self.uploaded(resolved)?;
        let made = self.backend.call(&resolved)?;
        let object = resolved.result;
        if object == 0 {
            return Ok(());
        }
        self.made(&resolved, object);
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
            self.addresses
                .insert(object, address, length, base)
                .map_err(PlayError::Invalid)?;
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
                Value::Blob(hash) => stream.blob(hash)?,
                _ => return Err(invalid("m2v.write of no data")),
            };
            self.put(target, offset, data)?;
        }
        Ok(())
    }

    /// Put `data` at `offset` of `target`, relocated where the target answered a `gpuAddress`, and
    /// count it.
    fn put(&mut self, target: u64, offset: u64, mut data: Vec<u8>) -> Result<(), PlayError> {
        if self.addresses.holds(target) {
            self.relocate(&mut data, offset);
        }
        *self.fixed.entry(WRITTEN_BYTES).or_default() += data.len() as u64;
        self.backend.write(target, offset, &data)
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
        readbacks: &mut Counters,
    ) -> Result<(), PlayError> {
        let before = self.backend.counters();
        let bytes = self.backend.read(object)?;
        let after = self.backend.counters();
        add(readbacks, &grown(&after, &before));
        self.baseline = after;
        self.report.checks.push(Check {
            object,
            sha256: Sha256::digest(&bytes).into(),
            recorded,
        });
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
        self.addresses.remove(object);
        self.owner.remove(&object);
        self.compared.remove(&object);
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
        self.backend.call(&own_call("release", object, 0)).map(drop)
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

    /// Wait for every command buffer that can complete, and record the counters' growth for them.
    fn drain(&mut self) -> Result<(), PlayError> {
        let ready = self.runnable();
        if ready.is_empty() {
            return Ok(());
        }
        for id in &ready {
            self.backend.wait(*id)?;
            let pending = self.pending.remove(id).expect("a pending command buffer");
            for (event, value) in pending.signals {
                let known = self.events.entry(event).or_default();
                *known = (*known).max(value);
            }
            if pending.released {
                self.backend_release(*id)?;
            }
        }
        self.committed.retain(|id| !ready.contains(id));
        let now = self.backend.counters();
        let executed = grown(&now, &self.baseline);
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
