//! The Shared buffers a dispatch read before anything the stream records put bytes in them: a
//! buffer of the Shared class, made without bytes, that a dispatch of a compute encoder reads,
//! whose command buffer commits before any recorded write, upload or GPU pass reached its bytes,
//! since its making or, in a block, before it. Such a buffer plays with whatever its memory held,
//! which is where a recording that missed the program's writes shows. Counted on the report's
//! `unwritten` line, once each, never refused.
//!
//! What a dispatch reads and writes: the buffer bound at each buffer slot its pipeline state uses,
//! by the state's access there (`ComputePipelineState::buffer_accesses`: a `ReadOnly` slot reads,
//! a `WriteOnly` one writes, a `ReadWrite` one both; a slot the state does not use, neither); a
//! buffer `useResource:usage:` names (each of a `useResources:count:usage:` list), by its usage's
//! read and write bits; and, read and written, a
//! buffer any other call of its encoder named since its last dispatch, and every bound buffer of a
//! dispatch whose state the play skipped. Those last two are counted as read, which they may not
//! be, and the line says how many. In a command buffer, a dispatch that writes a buffer comes
//! before the later dispatches that read it.
//!
//! What reaches a buffer's bytes: an `m2v.write` into it, into its block over them or into another
//! buffer of the block over them; a call of a blit or render encoder that names it or another
//! buffer of the block over them (a copy, a fill, an upload from the guest's RAM, an attachment);
//! and the commit of a command buffer whose dispatch writes it or another buffer of the block over
//! them. A block's bytes reached before a buffer is made over them reach the buffer, as those
//! after do. Every commit the player makes counts here: a `commit`, an `m2v.split` and an
//! `m2v.commitPart` it cuts are each the command buffer's commit to the backend. A texture over a
//! buffer is not followed.

use super::super::stream::Call;
use super::super::value::Value;
use super::objects_named;
use metal2vulkan_runtime::BindingAccess;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;

/// How many of the buffers counted the report names.
const NAMED: usize = 8;

/// `MTLResourceUsageRead`, `MTLResourceUsageWrite` and `MTLResourceUsageSample` (a read).
const USAGE_READ: u64 = 1;
const USAGE_WRITE: u64 = 2;
const USAGE_SAMPLE: u64 = 4;

#[derive(Default)]
pub(in crate::replay) struct Unwritten {
    /// Each live Shared block or heap: the byte ranges of it reached, merged, each its start and
    /// end.
    blocks: HashMap<u64, BTreeMap<u64, u64>>,
    /// Each live buffer made in a Shared block: the block and its byte range there.
    placed: HashMap<u64, (u64, Range<u64>)>,
    /// The live Shared buffers nothing has reached.
    waiting: HashSet<u64>,
    /// Those of them that lie in a block, by the block.
    in_block: HashMap<u64, Vec<u64>>,
    /// Each open compute encoder's bindings and state.
    encoders: HashMap<u64, Encoder>,
    /// What each uncommitted command buffer's dispatches read and wrote.
    dispatched: HashMap<u64, Dispatched>,
    /// The buffers a committed dispatch read while waiting, those of them only counted as read
    /// (`Encoder::named`, a skipped state), and the first of them with the command buffer that
    /// read each.
    count: u64,
    unsure: u64,
    first: Vec<(u64, u64)>,
}

/// A compute encoder's bindings of waiting buffers, and its state's buffer accesses.
#[derive(Default)]
struct Encoder {
    /// The waiting buffer bound at each buffer slot.
    slots: BTreeMap<u32, u64>,
    /// The waiting buffers `useResource:usage:` named, each with whether its usage reads and
    /// whether it writes.
    used: BTreeMap<u64, (bool, bool)>,
    /// The waiting buffers its other calls named since its last dispatch.
    named: Vec<u64>,
    /// Its state's buffer slots and their accesses; None before a state is set, or where the
    /// play skipped the state.
    accesses: Option<Vec<(u32, BindingAccess)>>,
}

/// The waiting buffers a command buffer's dispatches read before any of them wrote them, each
/// with whether the dispatch surely read it, and those they wrote.
#[derive(Default)]
struct Dispatched {
    read: Vec<(u64, bool)>,
    wrote: Vec<u64>,
}

impl Dispatched {
    /// A dispatch's use of waiting buffer `id`: it reads it (`surely`, or only counted so) and
    /// writes it as `reads` and `writes` say.
    fn uses(&mut self, id: u64, reads: bool, writes: bool, surely: bool) {
        if reads && !self.wrote.contains(&id) && !self.read.iter().any(|(read, _)| *read == id) {
            self.read.push((id, surely));
        }
        if writes && !self.wrote.contains(&id) {
            self.wrote.push(id);
        }
    }
}

/// Adds `range` to the merged `ranges`, joining those it meets or touches.
fn add(ranges: &mut BTreeMap<u64, u64>, range: Range<u64>) {
    if range.start >= range.end {
        return;
    }
    let (mut start, mut end) = (range.start, range.end);
    let met: Vec<u64> = (ranges.range(..=end).rev())
        .take_while(|&(_, &at_end)| at_end >= start)
        .map(|(&at, _)| at)
        .collect();
    for at in met {
        let at_end = ranges.remove(&at).expect("a range met");
        (start, end) = (start.min(at), end.max(at_end));
    }
    ranges.insert(start, end);
}

/// Whether the merged `ranges` meet `range`.
fn meets(ranges: &BTreeMap<u64, u64>, range: &Range<u64>) -> bool {
    let before = ranges.range(..range.end).next_back();
    before.is_some_and(|(_, &end)| end > range.start)
}

impl Unwritten {
    /// Shared block or heap `id` made: the bytes of it reached are followed from here, for the
    /// buffers made in it later.
    pub(in crate::replay) fn block(&mut self, id: u64) {
        self.blocks.insert(id, BTreeMap::new());
    }

    /// Shared buffer `id` made without bytes; `placed` is its block and its range there, where it
    /// lies in one. One made over bytes of its block reached before is not waiting.
    pub(in crate::replay) fn made(&mut self, id: u64, placed: Option<(u64, Range<u64>)>) {
        let Some((block, range)) = placed else {
            self.waiting.insert(id);
            return;
        };
        let reached = (self.blocks.get(&block)).is_some_and(|ranges| meets(ranges, &range));
        if !reached {
            self.waiting.insert(id);
            self.in_block.entry(block).or_default().push(id);
        }
        self.placed.insert(id, (block, range));
    }

    /// Bytes `range` of buffer or block `id` written.
    pub(in crate::replay) fn written(&mut self, id: u64, range: Range<u64>) {
        self.waiting.remove(&id);
        match self.placed.get(&id) {
            Some((block, at)) => {
                let block = *block;
                let range = at.start + range.start..at.start + range.end;
                self.reach(block, range);
            }
            None => self.reach(id, range),
        }
    }

    /// Buffer or block `id` named by a GPU pass, which may write any of its bytes.
    fn named(&mut self, id: u64) {
        self.waiting.remove(&id);
        match self.placed.get(&id) {
            Some((block, at)) => {
                let (block, range) = (*block, at.clone());
                self.reach(block, range);
            }
            None => self.reach(id, 0..u64::MAX),
        }
    }

    /// Bytes `range` of block `block` reached: every waiting buffer of the block over them is.
    fn reach(&mut self, block: u64, range: Range<u64>) {
        if let Some(ranges) = self.blocks.get_mut(&block) {
            add(ranges, range.clone());
        }
        let Some(ids) = self.in_block.get_mut(&block) else {
            return;
        };
        let (waiting, placed) = (&mut self.waiting, &self.placed);
        ids.retain(|id| {
            let over = (placed.get(id))
                .is_some_and(|(_, at)| at.start < range.end && range.start < at.end);
            if over {
                waiting.remove(id);
            }
            waiting.contains(id)
        });
    }

    /// Compute encoder `encoder`'s state set: its buffer slots' accesses, or None where the play
    /// skipped the state.
    pub(in crate::replay) fn state(
        &mut self,
        encoder: u64,
        accesses: Option<Vec<(u32, BindingAccess)>>,
    ) {
        self.encoders.entry(encoder).or_default().accesses = accesses;
    }

    /// Call `call` of an encoder of command buffer `command_buffer`, a compute encoder where
    /// `compute`.
    pub(in crate::replay) fn encoded(&mut self, call: &Call, command_buffer: u64, compute: bool) {
        if !compute {
            if self.waiting.is_empty() {
                return;
            }
            let mut named = Vec::new();
            objects_named(&call.args, &mut named);
            for id in named {
                self.named(id);
            }
            return;
        }
        if &*call.selector == "endEncoding" {
            self.encoders.remove(&call.receiver);
            return;
        }
        if self.waiting.is_empty() {
            return;
        }
        let uint = |at: usize| call.args.get(at).and_then(Value::as_uint);
        let slot = |at: usize| uint(at).and_then(|slot| u32::try_from(slot).ok());
        let waiting = &self.waiting;
        let waiting_of = |value: Option<&Value>| {
            (value.and_then(Value::as_object)).filter(|id| waiting.contains(id))
        };
        let encoder = self.encoders.entry(call.receiver).or_default();
        match &*call.selector {
            "setBuffer:offset:atIndex:" | "setBytes:length:atIndex:" => {
                if let Some(at) = slot(2) {
                    match waiting_of(call.args.first()) {
                        Some(id) => encoder.slots.insert(at, id),
                        None => encoder.slots.remove(&at),
                    };
                }
            }
            "setBuffers:offsets:withRange:" => {
                let buffers = call.args.first().and_then(Value::as_list);
                let location = (call.args.get(2).and_then(Value::as_map))
                    .and_then(|range| range.iter().find(|(key, _)| &**key == "location"))
                    .and_then(|(_, location)| location.as_uint())
                    .and_then(|location| u32::try_from(location).ok());
                if let (Some(buffers), Some(location)) = (buffers, location) {
                    for (at, buffer) in (location..).zip(buffers) {
                        match waiting_of(Some(buffer)) {
                            Some(id) => encoder.slots.insert(at, id),
                            None => encoder.slots.remove(&at),
                        };
                    }
                }
            }
            // The resource as the stream records it, `{Buffer: object}`; of the plural, each of
            // the list (the C array and its count are one list).
            "useResource:usage:" | "useResources:count:usage:" => {
                let resources = match &*call.selector {
                    "useResource:usage:" => call.args.first().map(std::slice::from_ref),
                    _ => call.args.first().and_then(Value::as_list),
                };
                let usage = uint(1);
                for resource in resources.into_iter().flatten() {
                    let id = resource.as_resource().filter(|id| waiting.contains(id));
                    if let (Some(id), Some(usage)) = (id, usage) {
                        let reads = usage & (USAGE_READ | USAGE_SAMPLE) != 0 || usage == 0;
                        encoder.used.insert(id, (reads, usage & USAGE_WRITE != 0));
                    }
                }
            }
            selector if selector.starts_with("dispatch") => {
                let dispatched = self.dispatched.entry(command_buffer).or_default();
                for (&at, &id) in &encoder.slots {
                    let access = match &encoder.accesses {
                        None => Some(BindingAccess::ReadWrite),
                        Some(accesses) => (accesses.iter())
                            .find(|(slot, _)| *slot == at)
                            .map(|&(_, access)| access),
                    };
                    let (reads, writes) = match access {
                        None => (false, false),
                        Some(BindingAccess::WriteOnly) => (false, true),
                        Some(BindingAccess::ReadOnly) => (true, false),
                        Some(_) => (true, true),
                    };
                    dispatched.uses(id, reads, writes, encoder.accesses.is_some());
                }
                for (&id, &(reads, writes)) in &encoder.used {
                    dispatched.uses(id, reads, writes, true);
                }
                for id in encoder.named.drain(..) {
                    dispatched.uses(id, true, true, false);
                }
            }
            _ => {
                let mut named = Vec::new();
                objects_named(&call.args, &mut named);
                named.retain(|id| waiting.contains(id));
                encoder.named.extend(named);
            }
        }
    }

    /// Command buffer `command_buffer` committed: the buffers its dispatches read while waiting
    /// are counted, once each, and those they wrote are reached.
    pub(in crate::replay) fn committed(&mut self, command_buffer: u64) {
        let Some(dispatched) = self.dispatched.remove(&command_buffer) else {
            return;
        };
        for (id, surely) in dispatched.read {
            if self.waiting.remove(&id) {
                self.count += 1;
                self.unsure += u64::from(!surely);
                if self.first.len() < NAMED {
                    self.first.push((id, command_buffer));
                }
            }
        }
        for id in dispatched.wrote {
            self.named(id);
        }
    }

    /// Object `id` released.
    pub(in crate::replay) fn released(&mut self, id: u64) {
        self.waiting.remove(&id);
        if let Some((block, _)) = self.placed.remove(&id) {
            if let Some(ids) = self.in_block.get_mut(&block) {
                ids.retain(|other| *other != id);
            }
        }
        self.blocks.remove(&id);
        self.in_block.remove(&id);
        self.encoders.remove(&id);
        self.dispatched.remove(&id);
    }

    /// The report's line.
    pub(in crate::replay) fn line(&self) -> String {
        let first: Vec<String> = (self.first.iter())
            .map(|(buffer, command_buffer)| format!("#{buffer} in #{command_buffer}"))
            .collect();
        let first = match first.is_empty() {
            true => String::new(),
            false => format!("; first {}", first.join(", ")),
        };
        format!(
            "unwritten {}: Shared buffers a dispatch read before any recorded write, upload or GPU \
             pass reached their bytes, {} of them counted as read though they may not be (named \
             by an encoder call that neither binds nor uses them, or bound under a skipped \
             state){first}",
            self.count, self.unsure
        )
    }
}
