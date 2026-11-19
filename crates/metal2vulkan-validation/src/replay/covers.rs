//! The bytes the player's compares leave out (`docs/REPLAY-STREAM.md`, section 10): those of a
//! buffer or texture placed in a placement heap that a write of another texture placed there
//! covered, from that write until the object is written again. Metal leaves an aliased resource's
//! contents undefined once another resource written over its memory, so the recording's bytes
//! there are its runtime's layout of the texture's texels, and a play's are its runtime's (or, where
//! the backend made the heap's buffers and textures apart, the bytes from before).
//!
//! Decided by structure alone: each object's heap, recorded offset and length (a buffer's own, a
//! texture's as the runtime answers its size, read ahead), and the texture writes: a draw's or a
//! dispatch's to the texture slots its pipeline state's interface marks written
//! ([`Backend::texture_writes`](super::player::Backend::texture_writes); none where the backend
//! answers no access), a render pass's to every texture its descriptor names, a blit's to its
//! destination, `generateMipmapsForTexture:`, a `useResource:usage:` with write usage, and the
//! CPU's `replaceRegion:…`. A write of a texture over a buffer is the buffer's. An object is
//! written again by the player's writes into it (`m2v.write`, its contents at creation, a copy
//! from the guest's RAM, a snapshot's bytes before), a blit into it, a write of a texture over it,
//! and, as the stream carries no buffer access, any binding of it to an encoder, a
//! `useResource:usage:` with write usage, or a `useHeap:` of its heap.

use super::stream::Call;
use super::value::Value;
use std::collections::{HashMap, HashSet};

/// The stage of a pipeline state whose texture slots a draw or dispatch writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlotStage {
    /// A compute pipeline state's kernel.
    Kernel,
    /// A render pipeline state's vertex function.
    Vertex,
    /// A render pipeline state's fragment function.
    Fragment,
}

/// An object placed in a placement heap: its heap and its recorded range there.
#[derive(Clone, Copy)]
struct Placed {
    heap: u64,
    start: u64,
    end: u64,
}

/// An open encoder: its pipeline state and the textures its slots hold, by stage and index.
#[derive(Default)]
struct Encoder {
    render: bool,
    pipeline: Option<u64>,
    slots: HashMap<(SlotStage, u64), u64>,
}

#[derive(Default)]
pub(super) struct Covers {
    placed: HashMap<u64, Placed>,
    /// Each heap's live placed objects.
    heaps: HashMap<u64, HashSet<u64>>,
    /// Each placed object's covered ranges, from its start.
    covered: HashMap<u64, Vec<(u64, u64)>>,
    /// The live textures, views included, each view's texture, and each texture over a buffer's
    /// buffer.
    textures: HashSet<u64>,
    base: HashMap<u64, u64>,
    over: HashMap<u64, u64>,
    /// Each descriptor object's root, and the textures each root names.
    described: HashMap<u64, u64>,
    holds: HashMap<u64, Vec<u64>>,
    encoders: HashMap<u64, Encoder>,
    /// The written slots the backend answered, by pipeline state and stage.
    writes: HashMap<(u64, SlotStage), Vec<u64>>,
    /// The times an object was covered.
    pub(super) covers: u64,
}

/// The objects `values` name, lists and maps included.
fn named(values: &[Value], out: &mut Vec<u64>) {
    for value in values {
        match value {
            Value::Object(id) => out.push(*id),
            Value::List(items) => named(items, out),
            Value::Map(fields) => {
                for (_, item) in fields {
                    named(std::slice::from_ref(item), out);
                }
            }
            _ => {}
        }
    }
}

/// An `NSRange`'s `{location, length}`.
fn range(value: Option<&Value>) -> Option<(u64, u64)> {
    let fields = value?.as_map()?;
    let field = |key: &str| fields.iter().find(|(k, _)| &**k == key)?.1.as_uint();
    Some((field("location")?, field("length")?))
}

impl Covers {
    fn base_of(&self, id: u64) -> u64 {
        self.base.get(&id).copied().unwrap_or(id)
    }

    /// Whether any byte of `object` (a view: its texture) is covered.
    pub(super) fn covered(&self, object: u64) -> bool {
        let object = self.base_of(object);
        self.covered
            .get(&object)
            .is_some_and(|ranges| !ranges.is_empty())
    }

    /// Whether any of the `length` bytes at `offset` of `object` is covered.
    pub(super) fn meets(&self, object: u64, offset: u64, length: u64) -> bool {
        let end = offset.saturating_add(length);
        (self.covered.get(&object).into_iter().flatten()).any(|&(s, e)| s < end && offset < e)
    }

    /// `object` written: the `length` bytes at `offset`, or, where none, all of it.
    pub(super) fn written(&mut self, object: u64, bytes: Option<(u64, u64)>) {
        let Some(ranges) = self.covered.get_mut(&object) else {
            return;
        };
        match bytes {
            None => ranges.clear(),
            Some((offset, length)) => {
                let end = offset.saturating_add(length);
                let mut left = Vec::with_capacity(ranges.len());
                for &(s, e) in ranges.iter() {
                    if e <= offset || end <= s {
                        left.push((s, e));
                        continue;
                    }
                    if s < offset {
                        left.push((s, offset));
                    }
                    if end < e {
                        left.push((end, e));
                    }
                }
                *ranges = left;
            }
        }
        if ranges.is_empty() {
            self.covered.remove(&object);
        }
    }

    /// Texture `texture` (a view: its texture) written: over a buffer, the buffer; placed in a
    /// heap, itself again, and every other object placed there that its range meets covered.
    fn texture_written(&mut self, texture: u64) {
        let texture = self.base_of(texture);
        if let Some(&buffer) = self.over.get(&texture) {
            self.written(buffer, None);
            return;
        }
        self.covered.remove(&texture);
        let Some(&Placed { heap, start, end }) = self.placed.get(&texture) else {
            return;
        };
        let Some(live) = self.heaps.get(&heap) else {
            return;
        };
        for &other in live {
            let Some(at) = self.placed.get(&other).filter(|_| other != texture) else {
                continue;
            };
            let (s, e) = (start.max(at.start), end.min(at.end));
            if s < e {
                (self.covered.entry(other).or_default()).push((s - at.start, e - at.start));
                self.covers += 1;
            }
        }
    }

    /// Whether a write of `texture` changes what a compare reads: it or its buffer is placed.
    fn tracked(&self, texture: u64) -> bool {
        let texture = self.base_of(texture);
        let texture = self.over.get(&texture).copied().unwrap_or(texture);
        self.placed.contains_key(&texture)
    }

    /// Call `call`, played, its blobs resolved: `offset` the offset the stream placed its result
    /// at in a placement heap, and `size` the runtime's size of a texture placed so (read ahead).
    /// `writes` answers the texture slots a pipeline state's stage writes.
    pub(super) fn saw(
        &mut self,
        call: &Call,
        offset: Option<u64>,
        size: Option<u64>,
        writes: impl Fn(u64, SlotStage) -> Option<Vec<u64>>,
    ) {
        let (selector, receiver, result) = (&*call.selector, call.receiver, call.result);
        let mut names = Vec::new();
        named(&call.args, &mut names);
        if receiver == 0 && result != 0 && selector.starts_with("MTL") {
            if selector.contains("Descriptor") {
                self.described.insert(result, result);
            }
            return;
        }
        if let Some(&root) = self.described.get(&receiver) {
            if result != 0 {
                self.described.insert(result, root);
            }
            let held: Vec<u64> = (names.iter())
                .filter(|id| self.textures.contains(id))
                .map(|&id| self.base_of(id))
                .collect();
            if !held.is_empty() {
                self.holds.entry(root).or_default().extend(held);
            }
            return;
        }
        if result != 0 && selector.starts_with("newTexture") {
            if self.textures.contains(&receiver) {
                self.base.insert(result, self.base_of(receiver));
            } else if selector.starts_with("newTextureWithDescriptor:offset:bytesPerRow:")
                || selector.starts_with("newLinearTextureWithDescriptor:")
            {
                self.over.insert(result, receiver);
            }
            self.textures.insert(result);
            if let ("newTextureWithDescriptor:offset:", Some(at), Some(size)) =
                (selector, offset, size)
            {
                self.place(receiver, result, at, size);
            }
            return;
        }
        if let ("newBufferWithLength:options:offset:", Some(at)) = (selector, offset) {
            if let Some(length) = call.args.first().and_then(Value::as_uint) {
                self.place(receiver, result, at, length);
            }
            return;
        }
        if result != 0 && selector.contains("CommandEncoder") {
            let render = selector.contains("ender");
            let attachments: Vec<u64> = (names.iter())
                .filter_map(|id| self.described.get(id))
                .filter_map(|root| self.holds.get(root))
                .flatten()
                .copied()
                .collect();
            for texture in attachments {
                if self.tracked(texture) {
                    self.texture_written(texture);
                }
            }
            let encoder = Encoder {
                render,
                ..Encoder::default()
            };
            self.encoders.insert(result, encoder);
            return;
        }
        if self.encoders.contains_key(&receiver) {
            self.encoded(call, &names, writes);
            return;
        }
        if self.textures.contains(&receiver) && selector.starts_with("replaceRegion:") {
            self.texture_written(receiver);
        }
    }

    /// `object` placed in heap `heap`, `length` bytes at recorded offset `at`.
    fn place(&mut self, heap: u64, object: u64, at: u64, length: u64) {
        let end = at.saturating_add(length);
        self.placed.insert(
            object,
            Placed {
                heap,
                start: at,
                end,
            },
        );
        self.heaps.entry(heap).or_default().insert(object);
    }

    /// A call on an open encoder.
    fn encoded(
        &mut self,
        call: &Call,
        names: &[u64],
        writes: impl Fn(u64, SlotStage) -> Option<Vec<u64>>,
    ) {
        let (selector, receiver) = (&*call.selector, call.receiver);
        let object = |at: usize| call.args.get(at).and_then(Value::as_object);
        let uint = |at: usize| call.args.get(at).and_then(Value::as_uint);
        match selector {
            "setComputePipelineState:" | "setRenderPipelineState:" => {
                if let Some(encoder) = self.encoders.get_mut(&receiver) {
                    encoder.pipeline = object(0);
                }
            }
            "setTexture:atIndex:" | "setVertexTexture:atIndex:" | "setFragmentTexture:atIndex:" => {
                let stage = match selector {
                    "setVertexTexture:atIndex:" => SlotStage::Vertex,
                    "setFragmentTexture:atIndex:" => SlotStage::Fragment,
                    _ => SlotStage::Kernel,
                };
                let (Some(index), Some(encoder)) = (uint(1), self.encoders.get_mut(&receiver))
                else {
                    return;
                };
                match object(0) {
                    Some(texture) => encoder.slots.insert((stage, index), texture),
                    None => encoder.slots.remove(&(stage, index)),
                };
            }
            "endEncoding" => {
                self.encoders.remove(&receiver);
            }
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:" => {
                if let (Some(to), Some(at), Some(size)) = (object(2), uint(3), uint(4)) {
                    self.written(to, Some((at, size)));
                }
            }
            "fillBuffer:range:value:" => {
                if let Some(buffer) = object(0) {
                    self.written(buffer, range(call.args.get(1)));
                }
            }
            "generateMipmapsForTexture:" => {
                if let Some(texture) = object(0).filter(|&t| self.tracked(t)) {
                    self.texture_written(texture);
                }
            }
            "useResource:usage:" | "useResource:usage:stages:" => {
                const WRITE: u64 = 2;
                if let (Some(resource), Some(usage)) = (object(0), uint(1)) {
                    if usage & WRITE != 0 {
                        match self.textures.contains(&resource) {
                            true if self.tracked(resource) => self.texture_written(resource),
                            true => {}
                            false => self.written(resource, None),
                        }
                    }
                }
            }
            "useHeap:" | "useHeap:stages:" => {
                let live = object(0).and_then(|heap| self.heaps.get(&heap)).cloned();
                for placed in live.into_iter().flatten() {
                    if !self.textures.contains(&placed) {
                        self.written(placed, None);
                    }
                }
            }
            _ if selector.contains("toTexture:") => {
                let to = names.iter().rev().find(|id| self.textures.contains(id));
                if let Some(&texture) = to.filter(|&&t| self.tracked(t)) {
                    self.texture_written(texture);
                }
            }
            _ if selector.contains("toBuffer:") => {
                let to = names.iter().rev().find(|id| !self.textures.contains(id));
                if let Some(&buffer) = to {
                    self.written(buffer, None);
                }
            }
            _ if selector.starts_with("set") && selector.contains("Buffer") => {
                for &buffer in names {
                    self.written(buffer, None);
                }
            }
            _ if selector.starts_with("dispatch") || selector.starts_with("draw") => {
                self.issued(receiver, writes);
            }
            _ => {}
        }
    }

    /// A draw or dispatch on encoder `encoder`: the textures its pipeline state's written slots
    /// hold, written.
    fn issued(&mut self, encoder: u64, writes: impl Fn(u64, SlotStage) -> Option<Vec<u64>>) {
        let Some(open) = self.encoders.get(&encoder) else {
            return;
        };
        let held: Vec<(SlotStage, u64, u64)> = (open.slots.iter())
            .filter(|&(_, &texture)| self.tracked(texture))
            .map(|(&(stage, index), &texture)| (stage, index, texture))
            .collect();
        let Some(pipeline) = open.pipeline.filter(|_| !held.is_empty()) else {
            return;
        };
        let stages: &[SlotStage] = match open.render {
            true => &[SlotStage::Vertex, SlotStage::Fragment],
            false => &[SlotStage::Kernel],
        };
        for &stage in stages {
            (self.writes.entry((pipeline, stage)))
                .or_insert_with(|| writes(pipeline, stage).unwrap_or_default());
        }
        let mut written = Vec::new();
        for &(stage, index, texture) in &held {
            let slots = self.writes.get(&(pipeline, stage));
            if slots.is_some_and(|slots| slots.contains(&index)) {
                written.push(texture);
            }
        }
        for texture in written {
            self.texture_written(texture);
        }
    }

    /// Object `id` released.
    pub(super) fn released(&mut self, id: u64) {
        if let Some(placed) = self.placed.remove(&id) {
            if let Some(live) = self.heaps.get_mut(&placed.heap) {
                live.remove(&id);
            }
        }
        self.heaps.remove(&id);
        self.covered.remove(&id);
        self.textures.remove(&id);
        self.base.remove(&id);
        self.over.remove(&id);
        self.described.remove(&id);
        self.holds.remove(&id);
        self.encoders.remove(&id);
        self.writes.retain(|&(pipeline, _), _| pipeline != id);
    }
}
