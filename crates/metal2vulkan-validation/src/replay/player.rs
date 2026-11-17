//! The player (`docs/REPLAY-STREAM.md`, section 10): it reads a stream's calls in order and drives a
//! runtime through a [`Backend`], relocating what it uploads, reproducing resource IDs, waiting for
//! the command buffers that can complete (never for one an event wait holds), and reporting the
//! runtime's counters per group of command buffers waited for, per frame and in total, with the
//! checksums the stream asks for.

use super::addresses::AddressMap;
use super::stream::{Call, StreamError, StreamReader};
use super::value::{Hash, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

/// A runtime's counters, cumulative, by name.
pub type Counters = BTreeMap<&'static str, u64>;

/// The player's own counters: the words it relocated, the 8-byte values at offsets 4 past a
/// multiple of 8 that lie in a live range (left as they are), and the bytes it wrote.
const RELOCATED_WORDS: &str = "relocated_words";
const MISALIGNED_WORDS: &str = "misaligned_address_words";
const WRITTEN_BYTES: &str = "written_bytes";
const OWN: [&str; 3] = [RELOCATED_WORDS, MISALIGNED_WORDS, WRITTEN_BYTES];

/// What a backend made for a call that makes an object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Made {
    /// The address the runtime gave the object's range.
    pub address: Option<u64>,
    /// The object's resource ID.
    pub resource_id: Option<u64>,
}

/// One runtime behind the player. Objects are named by their stream ids; the player resolves blobs
/// and relocates before a call reaches the backend, and handles `m2v.write`, `m2v.check`,
/// `m2v.present`, `release` of a device, and the program's waits itself.
pub trait Backend {
    /// Run `call`: make its result, or run its receiver's method, or drop the object `release`
    /// names. A call that makes an object at a recorded `gpuResourceID` answer makes it at that ID.
    fn call(&mut self, call: &Call) -> Result<Made, PlayError>;
    /// Put `bytes` at `offset` of the memory of block, buffer or surface `target`.
    fn write(&mut self, target: u64, offset: u64, bytes: &[u8]) -> Result<(), PlayError>;
    /// Wait until committed command buffer `command_buffer` completed.
    fn wait(&mut self, command_buffer: u64) -> Result<(), PlayError>;
    /// The bytes of buffer or texture `object`, for a checksum.
    fn read(&mut self, object: u64) -> Result<Vec<u8>, PlayError>;
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
    /// The runtime made an object at another resource ID than the recorded one.
    ResourceId {
        object: u64,
        recorded: u64,
        made: Option<u64>,
    },
}

impl fmt::Display for PlayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlayError::Stream(error) => write!(f, "stream: {error}"),
            PlayError::Refused { selector, why } => write!(f, "refused {selector}: {why}"),
            PlayError::Runtime { selector, error } => write!(f, "{selector}: {error}"),
            PlayError::Invalid(why) => write!(f, "invalid stream: {why}"),
            PlayError::ResourceId {
                object,
                recorded,
                made,
            } => write!(
                f,
                "#{object}: made at resource ID {made:?}, recorded {recorded}"
            ),
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

/// What a play did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Each group of command buffers waited for together (normally the one just committed), with
    /// the counters' growth since the group before.
    pub completed: Vec<(Vec<u64>, Counters)>,
    /// Each frame's counters: the groups completed since the present before.
    pub frames: Vec<Counters>,
    pub checks: Vec<Check>,
    /// The counters' growth over the whole play, the player's readbacks left out.
    pub total: Counters,
}

fn counted(counters: &Counters) -> String {
    let pairs = counters
        .iter()
        .map(|(key, value)| format!(" {key}={value}"));
    pairs.collect()
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
        lines.push(format!("total{}", counted(&self.total)));
        lines
    }
}

/// What the player knows of an object.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Device,
    CommandBuffer,
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
    baseline: Counters,
    /// The player's own counters since the last group.
    own: Counters,
    frame: Counters,
    report: Report,
}

/// `now` past `base`, key by key, with `own` added.
fn grown(now: &Counters, base: &Counters, own: &Counters) -> Counters {
    let mut out: Counters = now
        .iter()
        .map(|(&k, &v)| (k, v.saturating_sub(base.get(k).copied().unwrap_or(0))))
        .collect();
    for (&k, &v) in own {
        *out.entry(k).or_default() += v;
    }
    out
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

/// The calls whose first argument is data the GPU reads as constants, which may hold addresses.
fn sets_bytes(selector: &str) -> bool {
    selector.starts_with("set") && selector.ends_with("Bytes:length:atIndex:")
}

impl<'b, B: Backend> Player<'b, B> {
    pub fn new(backend: &'b mut B) -> Self {
        let baseline = backend.counters();
        let own = OWN.into_iter().map(|key| (key, 0)).collect();
        Player {
            backend,
            addresses: AddressMap::default(),
            kinds: HashMap::new(),
            owner: HashMap::new(),
            made_count: 0,
            pending: HashMap::new(),
            committed: Vec::new(),
            events: HashMap::new(),
            baseline,
            own,
            frame: Counters::new(),
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
        let mut total = grown(&now, &start, &Counters::new());
        for (key, value) in &mut total {
            *value = value.saturating_sub(readbacks.get(key).copied().unwrap_or(0));
        }
        for key in OWN {
            let grouped = self.report.completed.iter().map(|(_, group)| group[key]);
            *total.entry(key).or_default() += grouped.sum::<u64>() + self.own[key];
        }
        self.report.total = total;
        (self.report, ended)
    }

    fn all(
        &mut self,
        stream: &mut StreamReader,
        readbacks: &mut Counters,
    ) -> Result<(), PlayError> {
        while let Some(call) = stream.next_call()? {
            self.step(stream, call, readbacks)?;
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
                let shown = call.args.first().and_then(Value::as_object);
                let shown = shown.ok_or_else(|| invalid("m2v.present of no texture"))?;
                self.drain()?;
                self.check(shown, None, readbacks)?;
                self.report.frames.push(self.frame.clone());
                self.frame.clear();
                Ok(())
            }
            "commit" | "m2v.commitPart" => self.commit(call.receiver),
            "m2v.split" => {
                let queue = self.queue_of(call.receiver)?;
                self.commit(call.receiver)?;
                self.part(queue, call.result)
            }
            "m2v.part" => {
                let queue = self.queue_of(call.receiver)?;
                self.part(queue, call.result)
            }
            "m2v.setCompletionJoinsNext:" | "m2v.holdsNoPredication" | "m2v.rebind" => Ok(()),
            "waitUntilCompleted" | "waitUntilScheduled" => {
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

    /// Commit `command_buffer`, then wait for what can complete.
    fn commit(&mut self, command_buffer: u64) -> Result<(), PlayError> {
        let pending = self.pending.get_mut(&command_buffer);
        let pending = pending.ok_or_else(|| invalid("a commit of no command buffer"))?;
        pending.committed = true;
        self.committed.push(command_buffer);
        self.backend.call(&own_call("commit", command_buffer, 0))?;
        self.drain()
    }

    /// The queue of `command_buffer`, which is not yet waited for.
    fn queue_of(&self, command_buffer: u64) -> Result<u64, PlayError> {
        let queue = self.pending.get(&command_buffer).map(|p| p.queue);
        queue.ok_or_else(|| invalid("a part of no command buffer"))
    }

    /// Make `part`, a command buffer of `queue`.
    fn part(&mut self, queue: u64, part: u64) -> Result<(), PlayError> {
        if part == 0 {
            return Err(invalid("a part that makes no command buffer"));
        }
        let call = own_call("commandBuffer", queue, part);
        self.backend.call(&call)?;
        self.made(&call, part);
        Ok(())
    }

    /// A call the backend runs, its blobs and addresses resolved, and what it made recorded.
    fn make(&mut self, stream: &mut StreamReader, call: Call) -> Result<(), PlayError> {
        let mut relocated = 0;
        let mut args = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            args.push(self.resolve(stream, arg, &mut relocated)?);
        }
        *self.own.entry(RELOCATED_WORDS).or_default() += relocated;
        if sets_bytes(&call.selector) {
            if let Some(Value::Bytes(data)) = args.first_mut() {
                self.relocate(data, 0);
            }
        }
        let resolved = Call { args, ..call };
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
        if let Some(recorded) = resolved.answer("gpuResourceID").and_then(Value::as_uint) {
            if made.resource_id != Some(recorded) {
                return Err(PlayError::ResourceId {
                    object,
                    recorded,
                    made: made.resource_id,
                });
            }
        }
        Ok(())
    }

    /// Record what `call` made: its kind and its device.
    fn made(&mut self, call: &Call, object: u64) {
        let selector = &*call.selector;
        let kind = match selector {
            "MTLCreateSystemDefaultDevice" => Kind::Device,
            "newEvent" | "newSharedEvent" | "newSharedEventWithHandle:" => Kind::Event,
            _ if selector.starts_with("commandBuffer") => Kind::CommandBuffer,
            _ => Kind::Other,
        };
        self.kinds.insert(object, kind);
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
        if call.args.is_empty() || call.args.len() % 2 != 0 {
            return Err(invalid("m2v.write takes pairs of an offset and its data"));
        }
        self.drain()?;
        for pair in call.args.chunks_exact(2) {
            let offset = pair[0].as_uint();
            let offset = offset.ok_or_else(|| invalid("m2v.write of no offset"))?;
            let mut data = match &pair[1] {
                Value::Bytes(data) => data.clone(),
                Value::Blob(hash) => stream.blob(hash)?,
                _ => return Err(invalid("m2v.write of no data")),
            };
            if self.addresses.holds(target) {
                self.relocate(&mut data, offset);
            }
            *self.own.entry(WRITTEN_BYTES).or_default() += data.len() as u64;
            self.backend.write(target, offset, &data)?;
        }
        Ok(())
    }

    /// Relocate `data`, which starts `offset` bytes into its target, and count what it found.
    fn relocate(&mut self, data: &mut [u8], offset: u64) {
        let (moved, misaligned) = self.addresses.relocate(data, offset);
        *self.own.entry(RELOCATED_WORDS).or_default() += moved;
        *self.own.entry(MISALIGNED_WORDS).or_default() += misaligned;
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
        add(readbacks, &grown(&after, &before, &Counters::new()));
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
        let group = grown(&now, &self.baseline, &self.own);
        add(&mut self.frame, &group);
        self.report.completed.push((ready, group));
        self.baseline = now;
        self.own.values_mut().for_each(|v| *v = 0);
        Ok(())
    }
}
