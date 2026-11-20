//! The stream's heaps on the runtime (`docs/REPLAY-STREAM.md`, sections 5 and 10): a placement
//! heap laid out again for the runtime's sizes and alignments (`relay`).

use super::stream::Call;
use std::collections::{BTreeMap, HashMap};

/// What a placement heap holds at each recorded offset: the largest size and alignment the
/// runtime answers for the resources placed there.
pub(super) type Placements = BTreeMap<u64, (u64, u64)>;

/// A placement heap's layout on the runtime: its size, and the offset the player places each
/// resource at by the offset the stream places it at.
pub(super) struct Layout {
    pub size: u64,
    pub offsets: HashMap<u64, u64>,
}

/// The layout of a placement heap of `size` bytes, as a program asking the runtime's sizes and
/// alignments lays it out, whose resources the stream places at the offsets of `at`, each with the
/// most bytes and the largest alignment the runtime answers for its resources there. Each offset
/// keeps at least its distance from the one below, moves up until the resources below it end, and
/// rounds up to its alignment; the heap keeps at least its distance past the last offset and holds
/// the resources there. Resources at one offset still share it; resources at different offsets no
/// longer meet. Where every resource ends before the next offset and every offset is aligned, the
/// layout is the stream's.
pub(super) fn relay(size: u64, at: &Placements) -> Layout {
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
