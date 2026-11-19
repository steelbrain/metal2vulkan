//! The GPU's reads of a Managed object's bytes that no announcement took to its GPU copy, as a
//! play that keeps two copies would hold them (`docs/REPLAY-STREAM.md`, section 10, the
//! `managed reads` line): counted from the stream's calls alone, so the same on every runtime and
//! device. Two kinds:
//!
//! - **Stale:** bytes an `m2v.write` put into the object that no `didModifyRange:` of it
//!   announced and no GPU write covered since. Where the stream has no identity record, a no-copy
//!   Managed buffer's writes before its first use are not stale: its GPU copy takes them, as the
//!   player plays both runtimes (section 5).
//! - **Made with:** the bytes a Managed buffer was made with (a descriptor's `contents`, to its
//!   `zeroFrom` where it is a no-copy one, a no-copy buffer's pages, `newBufferWithBytes:`'s
//!   bytes; bytes that are all zero left out, as the GPU copy holds zeros there anyway), before a
//!   `didModifyRange:` or a GPU write covered them. Both runtimes serve them: each GPU copy
//!   starts from them.
//!
//! Stale reads play as recorded on both runtimes, as Metal leaves what the GPU reads there
//! undefined: where two plays' digests differ, they are the first suspect.
//!
//! A Managed object is a buffer of any maker whose options say Managed, a Managed block and the
//! buffers in it (the block's bytes), and a texture over such a buffer or a view of it (the
//! buffer's bytes from the texture's offset). A GPU use is any command of an encoder that names the
//! object, judged when its command buffer commits: a blit's source range and a copy to the guest's
//! RAM read; a blit's, a fill's and a guest-RAM copy's destination bytes are a GPU write; every
//! other naming (a binding, `useResource:`, `synchronizeResource:`, an attachment) reads the whole
//! object, as the stream carries no buffer access. An indirect command buffer's command is judged
//! where it is encoded. A command buffer released uncommitted runs nothing.

use super::args::option_sets;
use super::covers::named;
use super::player::{ManagedReads, ReadClass};
use super::stream::Call;
use super::value::Value;
use std::collections::{HashMap, HashSet};

/// `MTLStorageModeManaged`.
const MANAGED: u64 = 1;

/// The storage mode of resource options `options`.
fn mode(options: u64) -> u64 {
    (options >> 4) & 0xf
}

/// Where a Managed object's bytes lie: the object whose bytes they are (a block, or the buffer
/// itself), its range there, and the buffer counted for a read of it.
#[derive(Clone, Copy)]
struct Span {
    base: u64,
    start: u64,
    end: u64,
    counted: u64,
}

/// A use of an object a command buffer's commit judges: a read or a GPU write, of a range of the
/// object's bytes from its start, or of all of them.
#[derive(Clone, Copy)]
enum Use {
    Read(u64, Option<(u64, u64)>),
    Write(u64, Option<(u64, u64)>),
}

/// A kind's count: the buffers read, the uses, and the first read's frame.
#[derive(Default)]
struct Kind {
    objects: HashSet<u64>,
    uses: u64,
    first_frame: Option<u64>,
}

impl Kind {
    fn read(&mut self, counted: u64, frame: u64) {
        self.objects.insert(counted);
        self.uses += 1;
        self.first_frame.get_or_insert(frame);
    }

    fn class(&self) -> ReadClass {
        ReadClass {
            objects: self.objects.len() as u64,
            uses: self.uses,
            first_frame: self.first_frame,
        }
    }
}

#[derive(Default)]
pub(super) struct Stale {
    /// The guest device's `hasUnifiedMemory`, where the stream has an identity record.
    unified: Option<bool>,
    spans: HashMap<u64, Span>,
    /// The no-copy Managed buffers made with no bytes that no committed command buffer named yet,
    /// each with the command buffer that named it first: their writes reach the GPU copy where the
    /// stream has no identity record.
    fresh: HashMap<u64, Option<u64>>,
    /// Each base's bytes the CPU wrote that nothing announced or covered.
    written: HashMap<u64, Vec<(u64, u64)>>,
    /// Each base's bytes it was made with that nothing announced or covered.
    made: HashMap<u64, Vec<(u64, u64)>>,
    /// Each open encoder's command buffer.
    encoders: HashMap<u64, u64>,
    /// The indirect command buffers' commands.
    commands: HashSet<u64>,
    /// Each command buffer's uses, judged at its commit.
    pending: HashMap<u64, Vec<Use>>,
    stale: Kind,
    made_with: Kind,
}

/// `ranges` with `[start, end)` added, merged where they meet.
fn add(ranges: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    if end <= start {
        return;
    }
    let (mut start, mut end) = (start, end);
    ranges.retain(|&(s, e)| {
        let apart = e < start || s > end;
        if !apart {
            start = start.min(s);
            end = end.max(e);
        }
        apart
    });
    ranges.push((start, end));
}

/// `ranges` with `[start, end)` taken out.
fn remove(ranges: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    let mut kept = Vec::with_capacity(ranges.len() + 1);
    for &(s, e) in ranges.iter() {
        if e <= start || s >= end {
            kept.push((s, e));
            continue;
        }
        if s < start {
            kept.push((s, start));
        }
        if e > end {
            kept.push((end, e));
        }
    }
    *ranges = kept;
}

fn meets(ranges: Option<&Vec<(u64, u64)>>, start: u64, end: u64) -> bool {
    (ranges.into_iter().flatten()).any(|&(s, e)| s < end && start < e)
}

/// An `NSRange`'s `{location, length}` as `(start, end)`.
fn range(value: Option<&Value>) -> Option<(u64, u64)> {
    let fields = value?.as_map()?;
    let field = |key: &str| fields.iter().find(|(k, _)| &**k == key)?.1.as_uint();
    let start = field("location")?;
    Some((start, start.saturating_add(field("length")?)))
}

/// The length of a piece of data: bytes (blobs are read before a call reaches the count).
fn length(value: &Value) -> Option<u64> {
    value.as_bytes().map(|bytes| bytes.len() as u64)
}

/// The bytes a buffer was made with, where they count: `None` for none, and for inline bytes that
/// are all zero.
fn made_with(value: Option<&Value>, upto: Option<u64>) -> Option<u64> {
    let bytes = value?.as_bytes()?;
    let length = upto.map_or(bytes.len() as u64, |upto| upto.min(bytes.len() as u64));
    (length > 0 && bytes.iter().any(|&b| b != 0)).then_some(length)
}

impl Stale {
    /// The guest device's identity record: whether its memory is unified.
    pub(super) fn identity(&mut self, unified: bool) {
        self.unified.get_or_insert(unified);
    }

    /// The counts so far.
    pub(super) fn counts(&self) -> ManagedReads {
        ManagedReads {
            one_copy: self.unified == Some(true),
            no_identity: self.unified.is_none(),
            stale: self.stale.class(),
            made_with: self.made_with.class(),
        }
    }

    fn span(&self, object: u64) -> Option<Span> {
        self.spans.get(&object).copied()
    }

    /// A Managed buffer or block `object` made of `length` bytes, its own base, and the bytes it
    /// was made with, where they count.
    fn managed(&mut self, object: u64, length: u64, made: Option<u64>) {
        let span = Span {
            base: object,
            start: 0,
            end: length,
            counted: object,
        };
        self.spans.insert(object, span);
        if let Some(made) = made {
            add(self.made.entry(object).or_default(), 0, made.min(length));
        }
    }

    /// A call played, its blobs read.
    pub(super) fn saw(&mut self, call: &Call, frame: u64) {
        let selector = &*call.selector;
        let uint = |at: usize| call.args.get(at).and_then(Value::as_uint);
        let made = call.result;
        match selector {
            "newBufferWithLength:options:"
            | "newBufferWithLength:options:offset:"
            | "m2v.newBufferOfLengthPaddingUnobserved"
            | "m2v.newExportableBufferOfLength"
            | "m2v.newDmaBufBufferOfLengthPaddingUnobserved" => {
                if let (Some(length), Some(options), true) = (uint(0), uint(1), made != 0) {
                    if mode(options) == MANAGED {
                        self.managed(made, length, None);
                    }
                }
                return;
            }
            "newBufferWithBytes:length:options:"
            | "newBufferWithBytesNoCopy:length:options:deallocator:" => {
                if let (Some(length), Some(options), true) = (uint(1), uint(2), made != 0) {
                    if mode(options) == MANAGED {
                        let bytes = made_with(call.args.first(), None);
                        self.managed(made, length, bytes);
                        if selector.contains("NoCopy") && bytes.is_none() {
                            self.fresh.insert(made, None);
                        }
                    }
                }
                return;
            }
            "newBufferWithDescriptor:" => {
                let Some(fields) = call.args.first().and_then(Value::as_map) else {
                    return;
                };
                let field = |key: &str| fields.iter().find(|(k, _)| &**k == key).map(|(_, v)| v);
                let sets = field("options")
                    .and_then(Value::as_list)
                    .unwrap_or_default();
                let options = option_sets(call, 0, sets).unwrap_or(0);
                let length = field("length").and_then(Value::as_uint);
                if let (Some(length), MANAGED, true) = (length, mode(options), made != 0) {
                    let no_copy = field("noCopy").and_then(Value::as_bool) == Some(true);
                    let upto = no_copy.then(|| field("zeroFrom").and_then(Value::as_uint));
                    let bytes = made_with(field("contents"), upto.flatten());
                    self.managed(made, length, bytes);
                    if no_copy && bytes.is_none() {
                        self.fresh.insert(made, None);
                    }
                }
                return;
            }
            "endEncoding" => {
                self.encoders.remove(&call.receiver);
                return;
            }
            "m2v.newBlockWithLength:storageMode:" => {
                if let (Some(length), Some(MANAGED), true) = (uint(0), uint(1), made != 0) {
                    self.managed(made, length, None);
                }
                return;
            }
            "m2v.newBufferInBlock:offset:length:options:" => {
                let block = call.args.first().and_then(Value::as_object);
                let block = block.and_then(|block| self.span(block));
                if let (Some(block), Some(offset), Some(length), Some(options)) =
                    (block, uint(1), uint(2), uint(3))
                {
                    if mode(options) == MANAGED && made != 0 {
                        let start = block.start.saturating_add(offset);
                        let span = Span {
                            base: block.base,
                            start,
                            end: start.saturating_add(length),
                            counted: made,
                        };
                        self.spans.insert(made, span);
                    }
                }
                return;
            }
            "newTextureWithDescriptor:offset:bytesPerRow:"
            | "newTiledTextureWithDescriptor:offset:bytesPerRow:"
            | "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:" => {
                if let (Some(buffer), Some(offset)) = (self.span(call.receiver), uint(1)) {
                    let start = buffer.start.saturating_add(offset).min(buffer.end);
                    self.spans.insert(made, Span { start, ..buffer });
                }
                return;
            }
            _ if selector.starts_with("newTextureView") => {
                if let Some(texture) = self.span(call.receiver) {
                    self.spans.insert(made, texture);
                }
                return;
            }
            "didModifyRange:" => {
                if let (Some(span), Some((start, end))) =
                    (self.span(call.receiver), range(call.args.first()))
                {
                    let (start, end) = (span.start + start, span.start.saturating_add(end));
                    for ranges in [&mut self.written, &mut self.made] {
                        if let Some(ranges) = ranges.get_mut(&span.base) {
                            remove(ranges, start, end);
                        }
                    }
                }
                return;
            }
            _ => {}
        }
        if selector.contains("CommandEncoder") && !selector.contains("Argument") && made != 0 {
            let command_buffer = self.encoders.get(&call.receiver).copied();
            self.encoders
                .insert(made, command_buffer.unwrap_or(call.receiver));
        }
        if selector.starts_with("indirect") && selector.contains("CommandAtIndex") && made != 0 {
            self.commands.insert(made);
        }
        let command_buffer = match self.encoders.get(&call.receiver) {
            Some(&command_buffer) => Some(command_buffer),
            None if self.commands.contains(&call.receiver) => None,
            None if selector.contains("CommandEncoder") && !selector.contains("Argument") => {
                Some(call.receiver)
            }
            None => return,
        };
        let uses = self.uses(call);
        if uses.is_empty() {
            return;
        }
        match command_buffer {
            Some(command_buffer) => {
                for &one in &uses {
                    let (Use::Read(object, _) | Use::Write(object, _)) = one;
                    if let Some(user) = self.fresh.get_mut(&object) {
                        user.get_or_insert(command_buffer);
                    }
                }
                self.pending.entry(command_buffer).or_default().extend(uses);
            }
            None => {
                for one in uses {
                    self.judge(one, frame);
                }
            }
        }
    }

    /// The uses of the Managed objects `call`, a command, names.
    fn uses(&self, call: &Call) -> Vec<Use> {
        let object = |at: usize| call.args.get(at).and_then(Value::as_object);
        let uint = |at: usize| call.args.get(at).and_then(Value::as_uint);
        let managed = |object: Option<u64>| object.filter(|o| self.spans.contains_key(o));
        let mut uses = Vec::new();
        match &*call.selector {
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:" => {
                let size = uint(4).unwrap_or(0);
                if let (Some(from), Some(at)) = (managed(object(0)), uint(1)) {
                    uses.push(Use::Read(from, Some((at, at.saturating_add(size)))));
                }
                if let (Some(to), Some(at)) = (managed(object(2)), uint(3)) {
                    uses.push(Use::Write(to, Some((at, at.saturating_add(size)))));
                }
            }
            "fillBuffer:range:value:" => {
                if let (Some(to), Some(bytes)) = (managed(object(0)), range(call.args.get(1))) {
                    uses.push(Use::Write(to, Some(bytes)));
                }
            }
            "m2v.copyFromGuestRam" | "m2v.copyFromGuestRamAt" => {
                let pieces = call
                    .args
                    .get(1)
                    .and_then(Value::as_list)
                    .unwrap_or_default();
                if let Some(to) = managed(object(0)) {
                    for piece in pieces {
                        if let Some([at, data]) = piece.as_list() {
                            if let (Some(at), Some(n)) = (at.as_uint(), length(data)) {
                                uses.push(Use::Write(to, Some((at, at.saturating_add(n)))));
                            }
                        }
                    }
                }
            }
            "m2v.copyToGuestRam" => {
                if let Some(from) = managed(object(0)) {
                    uses.push(Use::Read(from, None));
                }
            }
            selector
                if selector.starts_with("copyFromTexture:") && selector.contains("toBuffer:") =>
            {
                let mut objects = Vec::new();
                named(&call.args, &mut objects);
                let to = objects
                    .into_iter()
                    .skip(1)
                    .filter(|o| self.spans.contains_key(o));
                uses.extend(to.map(|to| Use::Write(to, None)));
            }
            _ => {
                let mut objects = Vec::new();
                named(&call.args, &mut objects);
                objects.sort_unstable();
                objects.dedup();
                let read = objects.into_iter().filter(|o| self.spans.contains_key(o));
                uses.extend(read.map(|from| Use::Read(from, None)));
            }
        }
        uses
    }

    /// One use, at frame `frame`.
    fn judge(&mut self, one: Use, frame: u64) {
        let (Use::Read(object, bytes) | Use::Write(object, bytes)) = one;
        let Some(span) = self.span(object) else {
            return;
        };
        let (start, end) = match bytes {
            Some((start, end)) => (
                span.start.saturating_add(start),
                span.start.saturating_add(end),
            ),
            None => (span.start, span.end),
        };
        match one {
            Use::Read(..) => {
                if meets(self.written.get(&span.base), start, end) {
                    self.stale.read(span.counted, frame);
                }
                if meets(self.made.get(&span.base), start, end) {
                    self.made_with.read(span.counted, frame);
                }
            }
            Use::Write(..) => {
                for ranges in [&mut self.written, &mut self.made] {
                    if let Some(ranges) = ranges.get_mut(&span.base) {
                        remove(ranges, start, end);
                    }
                }
            }
        }
    }

    /// `length` bytes the CPU put at `offset` of `target` (`m2v.write`).
    pub(super) fn wrote(&mut self, target: u64, offset: u64, length: u64) {
        let Some(span) = self.span(target) else {
            return;
        };
        if self.unified.is_none() && self.fresh.contains_key(&target) {
            return;
        }
        let start = span.start.saturating_add(offset);
        add(
            self.written.entry(span.base).or_default(),
            start,
            start.saturating_add(length),
        );
    }

    /// `command_buffer` committed at frame `frame`: its uses judged, and the buffers it named
    /// first fresh no more.
    pub(super) fn committed(&mut self, command_buffer: u64, frame: u64) {
        for one in self.pending.remove(&command_buffer).unwrap_or_default() {
            self.judge(one, frame);
        }
        (self.fresh).retain(|_, user| *user != Some(command_buffer));
    }

    /// `object` released.
    pub(super) fn released(&mut self, object: u64) {
        self.spans.remove(&object);
        self.fresh.remove(&object);
        self.written.remove(&object);
        self.made.remove(&object);
        self.encoders.remove(&object);
        self.commands.remove(&object);
        self.pending.remove(&object);
    }
}
