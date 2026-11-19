//! The stream's heaps on the runtime (`docs/REPLAY-STREAM.md`, sections 5 and 10): a placement
//! heap laid out again for the runtime's sizes and alignments (`relay`), and, on a backend that
//! makes objects at their recorded addresses, a heap's buffers and textures made apart
//! (`HeapSplit`), with the textures whose recorded range met a live buffer's counted
//! (`Overlaps`).

use super::stream::Call;
use super::value::Value;
use std::collections::{BTreeMap, HashMap};

/// How a backend that makes objects at their recorded addresses (`Backend::places_at_recorded`)
/// makes a heap the stream made with a recorded range (`Backend::split_heap`), read ahead, by
/// what the stream places in it: its buffers in a heap at the recorded address, its textures in
/// one outside the recorded addresses. A heap the stream places only buffers in is the first
/// alone, one it places no buffer in the second alone, one it places both in both. Decided by
/// what the stream places in it, whatever its type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeapSplit {
    /// The stream places a buffer in the heap: a heap at its recorded address, of its recorded
    /// size, holds its buffers, each at its recorded offset. It never grows.
    pub buffers: bool,
    /// The size of the heap outside the recorded addresses that holds the heap's textures, laid
    /// out for the runtime's sizes (`relay` of its textures alone); none where the stream places
    /// a buffer and no texture in it.
    pub textures: Option<u64>,
}

/// What a placement heap holds at each recorded offset: the largest size and alignment the
/// runtime answers for the resources placed there.
pub(super) type Placements = BTreeMap<u64, (u64, u64)>;

/// A placement heap's layout on the runtime: its size, and the offset the player places each
/// resource at by the offset the stream places it at.
pub(super) struct Layout {
    pub size: u64,
    pub offsets: HashMap<u64, u64>,
    /// Whether it lays out the heap's textures alone, which a backend that makes objects at their
    /// recorded addresses makes in a heap of their own (`HeapSplit::textures`): the heap's buffers
    /// keep their recorded offsets, and the heap its recorded size.
    pub textures_only: bool,
}

/// The layout of a placement heap of `size` bytes, as a program asking the runtime's sizes and
/// alignments lays it out, whose resources the stream places at the offsets of `at`, each with the
/// most bytes and the largest alignment the runtime answers for its resources there. Each offset
/// keeps at least its distance from the one below, moves up until the resources below it end, and
/// rounds up to its alignment; the heap keeps at least its distance past the last offset and holds
/// the resources there. Resources at one offset still share it; resources at different offsets no
/// longer meet. Where every resource ends before the next offset and every offset is aligned, the
/// layout is the stream's.
pub(super) fn relay(size: u64, at: &Placements, textures_only: bool) -> Layout {
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
    Layout {
        size,
        offsets,
        textures_only,
    }
}

/// The recorded size of the heap `call` (`newHeapWithDescriptor:`) makes.
pub(super) fn heap_size(call: &Call) -> Option<u64> {
    descriptor_field(call, "size")
}

/// The recorded size of the heap `call` (`newHeapWithDescriptor:`) makes, where it is a placement
/// heap (`MTLHeapTypePlacement`).
pub(super) fn placement_heap(call: &Call) -> Option<u64> {
    (descriptor_field(call, "heapType")? == 1).then_some(heap_size(call)?)
}

fn descriptor_field(call: &Call, key: &str) -> Option<u64> {
    let entries = call.args.first()?.as_map()?;
    entries.iter().find(|(k, _)| &**k == key)?.1.as_uint()
}

/// Whether `selector` places a buffer in its receiver where that is a heap: a placement heap's
/// `newBufferWithLength:options:offset:` or an automatic heap's `newBufferWithLength:options:`.
pub(super) fn places_buffer(selector: &str) -> bool {
    matches!(
        selector,
        "newBufferWithLength:options:offset:" | "newBufferWithLength:options:"
    )
}

/// Whether `selector` places a texture in its receiver where that is a heap: a placement heap's
/// `newTextureWithDescriptor:offset:` or an automatic heap's `newTextureWithDescriptor:`.
pub(super) fn places_texture(selector: &str) -> bool {
    matches!(
        selector,
        "newTextureWithDescriptor:offset:" | "newTextureWithDescriptor:"
    )
}

/// The argument of a placement heap's call that is the offset it places at.
pub(super) fn placement_offset(selector: &str) -> Option<usize> {
    match selector {
        "newTextureWithDescriptor:offset:" => Some(1),
        "newBufferWithLength:options:offset:" => Some(2),
        _ => None,
    }
}

/// The live buffers and textures the stream places in each placement heap with a recorded range,
/// by heap, and the textures whose recorded range met a live buffer's, each counted once (`met`),
/// whichever of the two was placed first: a texture's range starts at the heap's recorded address
/// plus its offset and holds the bytes the runtime answers for its descriptor (the stream holds no
/// texture's size), a buffer's is the range it answered. Such a texture and buffer shared memory
/// in the recording; a backend that makes the heap's buffers and textures apart (`HeapSplit`) no
/// longer shares it.
#[derive(Default)]
pub(super) struct Overlaps {
    /// Each heap's live buffers and live textures, by id: each one's range and, for a texture,
    /// whether it met a buffer.
    heaps: HashMap<u64, Live>,
    /// The heap of each live resource placed in one.
    of: HashMap<u64, u64>,
    /// The textures that met a live buffer.
    pub met: u64,
}

#[derive(Default)]
struct Live {
    buffers: HashMap<u64, (u64, u64)>,
    textures: HashMap<u64, (u64, u64, bool)>,
}

fn meet((start, end): (u64, u64), (other, other_end): (u64, u64)) -> bool {
    start < other_end && other < end
}

impl Overlaps {
    /// Buffer `buffer` placed in `heap`, at `range`.
    pub fn buffer(&mut self, heap: u64, buffer: u64, range: (u64, u64)) {
        let live = self.heaps.entry(heap).or_default();
        for (start, end, met) in live.textures.values_mut() {
            if !*met && meet((*start, *end), range) {
                *met = true;
                self.met += 1;
            }
        }
        live.buffers.insert(buffer, range);
        self.of.insert(buffer, heap);
    }

    /// Texture `texture` placed in `heap`, at `range`.
    pub fn texture(&mut self, heap: u64, texture: u64, range: (u64, u64)) {
        let live = self.heaps.entry(heap).or_default();
        let met = live.buffers.values().any(|&buffer| meet(buffer, range));
        self.met += u64::from(met);
        live.textures.insert(texture, (range.0, range.1, met));
        self.of.insert(texture, heap);
    }

    /// `object` released: no longer live, where it was placed in a heap.
    pub fn release(&mut self, object: u64) {
        let Some(heap) = self.of.remove(&object) else {
            return;
        };
        if let Some(live) = self.heaps.get_mut(&heap) {
            live.buffers.remove(&object);
            live.textures.remove(&object);
        }
    }
}

/// A buffer's recorded range, from `call`'s answers.
pub(super) fn recorded_range(call: &Call) -> Option<(u64, u64)> {
    let address = call.answer("gpuAddress")?.as_address()?;
    let length = call.answer("gpuLength").and_then(Value::as_uint)?;
    Some((address, address + length))
}
