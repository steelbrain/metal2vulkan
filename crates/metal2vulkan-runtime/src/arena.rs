//! The private arena: device memory the runtime owns that holds the private memory of a kernel
//! past the translator's floor (`RuntimeFact::PrivateArena`), one slot of the kernel's slot bytes
//! for each invocation of a dispatch, at the address the dispatch pushes
//! (`PushField::PrivateArenaAddress`). Metal holds such a kernel's private memory itself.
//!
//! - **Budget.** A dispatch's slots take at most one sixteenth of the device's largest
//!   device-local heap, and at most its longest block ([`budget`]), or one workgroup's slots where
//!   those are more. A dispatch whose invocations' slots pass it runs as several dispatches, each
//!   of as many whole workgroups as the budget holds ([`piece`]), each one's region carrying its
//!   first threadgroup in Metal's grid (`crate::encoder`). An indirect dispatch, whose counts the
//!   runtime never reads, cannot be split or sized: it is refused
//!   ([`Gap::IndirectPrivateArena`](crate::Gap::IndirectPrivateArena)).
//! - **Memory.** A Private block of the pool, never zeroed: Metal's private memory is undefined
//!   until the thread writes it. The device keeps a weak reference to the last block it took; each
//!   command buffer that dispatches into a block keeps it until it retires, so the block goes back
//!   to the pool once the last of them does, and a dispatch that needs no more than it holds while
//!   one of them lives takes it again ([`take`]). A dispatch that needs more takes a block
//!   of its need rounded up to a power of two, within the budget.
//! - **Order.** The pieces of a dispatch share their block's slots, and so may the dispatches
//!   before them: each piece is ordered after the commands before it by a barrier, but the first
//!   piece in a block the dispatch took new, which no command reached before.
//!
//! Counted in [`Metrics::private_arena_dispatches`](crate::Metrics::private_arena_dispatches),
//! [`Metrics::private_arena_pieces`](crate::Metrics::private_arena_pieces) and
//! [`Metrics::private_arena_blocks`](crate::Metrics::private_arena_blocks), and the takes that
//! allocated in [`Metrics::private_arena_allocations`](crate::Metrics::private_arena_allocations).

use crate::counters::{Metric, Site};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::memory::{self, MemoryClass};
use ash::vk;
use std::sync::{Arc, Weak};

/// The share of the largest device-local heap a dispatch's slots take at most: one sixteenth.
const HEAP_SHARE: u64 = 16;

/// The most bytes a dispatch's slots take on a device of `memory`'s heaps whose longest block is
/// `max_block_len`: a sixteenth of its largest device-local heap, within that block.
pub(crate) fn budget(memory: &vk::PhysicalDeviceMemoryProperties, max_block_len: u64) -> u64 {
    let largest = (memory.memory_heaps_as_slice().iter())
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
        .unwrap_or(0);
    (largest / HEAP_SHARE).min(max_block_len)
}

/// The workgroups per axis of each piece of a dispatch of `groups` workgroups of `invocations`
/// invocations each, whose slots are `slot_bytes` each, under `budget` bytes and the device's
/// `max` workgroups per axis: as many whole workgroups as the budget's slots hold, at least one,
/// filling `x`, then `y`, then `z`, so every piece but an edge one has the same shape. A piece's
/// invocations stay under 2^32, so each one's index in it fits a `u32`.
pub(crate) fn piece(
    (groups, max): ([u32; 3], [u32; 3]),
    invocations: u64,
    slot_bytes: u32,
    budget: u64,
) -> [u32; 3] {
    let slots = (budget / u64::from(slot_bytes)).min(u64::from(u32::MAX));
    let mut room = (slots / invocations.max(1)).max(1);
    std::array::from_fn(|axis| {
        let along = u64::from(groups[axis].min(max[axis])).min(room).max(1);
        room = (room / along).max(1);
        along as u32
    })
}

/// The block a dispatch on `core` whose slots take `len` bytes reaches its slots in, and whether
/// the dispatch took it new: the device's last, where a command buffer still keeps it and it holds
/// `len`, else a block of `len` rounded up to a power of two, within `budget` where `len` is.
///
/// The arena's lock is held only to read the device's last block and to store a new one, never
/// across the take: a take may make an allocation and run the consumer's
/// [`BlockMappings::map`](crate::BlockMappings::map) on this thread, and another thread's dispatch
/// goes on meanwhile. Two dispatches that find no block holding their slots at once each take
/// their own and keep it, held by its command buffer until it retires, even where the other's,
/// stored meanwhile, would hold their slots; the device keeps the longer of the two as its last.
/// So the arena stays within its bound: each block is its own dispatch's, at most `budget`, or
/// one workgroup's slots where those pass it, as a block taken with no race is, and a race adds
/// one such block for each dispatch that missed at once, never a longer one.
///
/// Err: what [`Core::take_memory`] returns.
pub(crate) fn take(
    core: &Arc<Core>,
    len: u64,
    budget: u64,
) -> Result<(Arc<memory::Block>, bool), Error> {
    let last = lock(&core.arena).upgrade();
    if let Some(block) = last.filter(|block| block.memory.len >= len) {
        return Ok((block, false));
    }
    let len = len
        .checked_next_power_of_two()
        .unwrap_or(len)
        .min(budget.max(len));
    let memory = core.take_memory(MemoryClass::Private, len, Site::PrivateArena)?;
    let block = Arc::new(memory::Block::new(memory, Arc::clone(core)));
    let mut last = lock(&core.arena);
    let kept = last.upgrade();
    if kept
        .as_ref()
        .is_none_or(|kept| kept.memory.len < block.memory.len)
    {
        *last = Arc::downgrade(&block);
    }
    drop(last);
    // Where it was the last holder of the block it read, the block goes back here, off the lock.
    drop(kept);
    core.counters.count(Metric::PrivateArenaBlocks);
    Ok((block, true))
}

/// What the device keeps of its last arena block ([`take`]).
pub(crate) type Last = std::sync::Mutex<Weak<memory::Block>>;
