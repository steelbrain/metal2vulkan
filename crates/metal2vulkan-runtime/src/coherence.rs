//! Shared memory's host coherence: the flushes and invalidates Vulkan asks of mapped memory, made
//! on every device whatever the flags of its Shared type.
//!
//! Shared storage lives in the first host-visible, host-cached memory type a buffer can use, a
//! host-coherent one first (`floor::choose`). Vulkan makes the host's writes to mapped memory of a
//! type without `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` available to the device only through
//! `vkFlushMappedMemoryRanges`, and the device's writes visible to the host only through
//! `vkInvalidateMappedMemoryRanges` ("Host Access to Device Memory Objects"). The runtime makes
//! both calls on every device, over the ranges commands reach, and never reads the type's flags:
//! one form, whose calls a driver answers at once where the type is coherent (Mesa 26.2's Intel
//! driver returns before it reads a range where no type of the device needs a flush, and skips a
//! coherent type's ranges where one does).
//!
//! - **Before each submission**, the Shared memory its command buffers reach is flushed: each
//!   buffer, texture over a buffer and heap a command buffer with a segment in it keeps, as its
//!   commit found them ([`Ranges::of`]), so the host's writes up to that submission reach the GPU,
//!   those made after a commit and before the host value its segment waits for among them; and
//!   the ranges each Managed copy-in it carries reads (`crate::capture`).
//! - **Before the host can see the GPU's progress**, the Shared memory a command buffer may write
//!   is flushed, then invalidated ([`settle`]): as it retires, before its status reads completed,
//!   a wait for it returns or its completed handlers run; and, for every command buffer in flight,
//!   before a shared event's value a GPU signal raised reaches a listener, `signaledValue` or a
//!   wait for it. The flush first keeps the host's own writes to those ranges, which Vulkan leaves
//!   undefined when a range written by the host is invalidated without a flush.
//! - **The runtime's own host accesses** of the Shared type: the bytes `replace_region` writes into
//!   a staging span are flushed before the copy is recorded, and so are the named bytes
//!   `upload_region` copies, at the call; those a copy into a span wrote are invalidated before
//!   `get_bytes` reads them; and the zero source is flushed once, at its make.
//!
//! What a command buffer may write ([`Writes`]): each Shared buffer a command writes, or a texture
//! over one, as the command names it (a blit's destination, a binding its function's interface
//! writes, a `use_resource` with a write, a render pass's attachment, a visibility result buffer,
//! a synchronize's target); each heap's block it uses; and each texture over a Shared buffer in an
//! image of its own, whose bytes the runtime's copies write back. A range a command reaches only
//! through an address, with no binding or declaration naming it, is neither flushed nor
//! invalidated.
//!
//! Ranges are of the pool's Shared allocations and the runtime's own, which it maps whole,
//! widened to whole multiples of the device's `nonCoherentAtomSize`: within a block, since a block
//! starts and ends at multiples of 64 KiB and the atom is a power of two of at most 256. An import
//! is never flushed: its type is host-coherent (`crate::import`), and Vulkan maps none of it.
//! Each call is counted, with its bytes, in [`Metrics`](crate::Metrics)'s `shared_flushes`,
//! `shared_flushed_bytes`, `shared_invalidates` and `shared_invalidated_bytes`; a call that fails
//! in `shared_coherence_failures`.

use crate::counters::Counters;
use crate::device::{lock, Core};
use crate::memory::{Memory, MemoryClass};
use ash::vk;
use std::collections::HashSet;
use std::sync::{Arc, Mutex, Weak};

/// A range of mapped memory: its allocation, and its first and end bytes there, each a multiple
/// of the device's `nonCoherentAtomSize` or the allocation's end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Range {
    memory: vk::DeviceMemory,
    start: u64,
    end: u64,
}

impl Range {
    /// The `len` bytes at `offset` of `memory`, widened to whole atoms of `atom` bytes and cut at
    /// `limit`, a multiple of `atom` or the allocation's end; `None` for no bytes.
    pub(crate) fn widened(
        memory: vk::DeviceMemory,
        offset: u64,
        len: u64,
        limit: u64,
        atom: u64,
    ) -> Option<Range> {
        if len == 0 {
            return None;
        }
        Some(Range {
            memory,
            start: offset - offset % atom,
            end: (offset + len).next_multiple_of(atom).min(limit),
        })
    }

    /// Of a block's memory, the `len` bytes at `offset` in the block: where the block is Shared
    /// memory of the pool, which the runtime maps whole; `None` for a Private block, an import's
    /// or an export's, or no bytes.
    pub(crate) fn of_block(memory: &Memory, offset: u64, len: u64, atom: u64) -> Option<Range> {
        if memory.class != MemoryClass::Shared || memory.own.is_some() || memory.mapping.is_none() {
            return None;
        }
        let base = memory.base();
        let len = len.min(memory.len.saturating_sub(offset));
        Range::widened(
            memory.device_memory(),
            base + offset,
            len,
            base + memory.len,
            atom,
        )
    }

    fn bytes(self) -> u64 {
        self.end - self.start
    }

    fn mapped(self) -> vk::MappedMemoryRange<'static> {
        vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(self.start)
            .size(self.end - self.start)
    }
}

/// The ranges in order, each pair that meets or touches in one allocation made one.
pub(crate) fn merged(mut ranges: Vec<Range>) -> Vec<Range> {
    ranges.sort_unstable();
    let mut out: Vec<Range> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match out.last_mut() {
            Some(last) if last.memory == range.memory && range.start <= last.end => {
                last.end = last.end.max(range.end);
            }
            _ => out.push(range),
        }
    }
    out
}

/// The Shared memory a recording's commands may write, each range once, as they name it.
#[derive(Default)]
pub(crate) struct Writes {
    ranges: HashSet<Range>,
}

impl Writes {
    /// A command may write `range`; nothing for `None`.
    pub(crate) fn add(&mut self, range: Option<Range>) {
        if let Some(range) = range {
            self.ranges.insert(range);
        }
    }
}

/// What a committed command buffer's commands reach of Shared memory, merged: every range, which
/// each submission that carries one of its segments flushes first, and those they may write,
/// which the host's sight of their progress settles first ([`settle`]).
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Ranges {
    pub(crate) reached: Vec<Range>,
    pub(crate) written: Vec<Range>,
}

impl Ranges {
    /// Of a recording: each buffer and texture over a buffer it keeps (`kept`), each block it keeps
    /// (`blocks`: of a heap it uses where Shared), and what its commands may write (`writes`);
    /// every heap's block and every texture whose image the runtime's copies write back to its
    /// Shared buffer count as written. `None` where it reaches no Shared memory.
    pub(crate) fn of(
        kept: &crate::kept::ByKind,
        blocks: &[Arc<crate::memory::Block>],
        writes: &Writes,
        atom: u64,
    ) -> Option<Ranges> {
        use crate::buffer::BufferObject;
        use crate::texture::TextureObject;
        let mut written: Vec<Range> = writes.ranges.iter().copied().collect();
        let heaps = (blocks.iter())
            .filter_map(|block| Range::of_block(&block.memory, 0, block.memory.len, atom));
        written.extend(heaps);
        let mut reached: Vec<Range> = (kept.buffers.iter())
            .filter_map(|buffer| buffer.shared_range(atom))
            .collect();
        for texture in kept.textures.iter() {
            let range = texture.shared_range(atom);
            if texture.copied_shared() {
                written.extend(range);
            }
            reached.extend(range);
        }
        reached.extend_from_slice(&written);
        if reached.is_empty() {
            return None;
        }
        Some(Ranges {
            reached: merged(reached),
            written: merged(written),
        })
    }
}

/// The device's side of coherence: the atom ranges widen to, and the written ranges of each
/// command buffer committed and not yet retired that may write Shared memory, which a shared
/// event's signal seen settles.
pub(crate) struct Coherence {
    /// The device's `nonCoherentAtomSize`.
    pub(crate) atom: u64,
    flight: Mutex<Vec<Weak<Ranges>>>,
}

impl Coherence {
    pub(crate) fn new(atom: u64) -> Coherence {
        Coherence {
            atom: atom.max(1),
            flight: Mutex::new(Vec::new()),
        }
    }

    /// A command buffer committed with `ranges`: while it lives in flight, a shared event's signal
    /// seen settles its written ranges. One that writes none is not held.
    pub(crate) fn committed(&self, ranges: &Arc<Ranges>) {
        if ranges.written.is_empty() {
            return;
        }
        let mut flight = lock(&self.flight);
        flight.retain(|held| held.strong_count() > 0);
        flight.push(Arc::downgrade(ranges));
    }

    /// Before the host sees a value a GPU signal raised: flushes, then invalidates, the written
    /// ranges of every command buffer committed and not yet retired (`settle`).
    pub(crate) fn signal_seen(&self, core: &Core) {
        let held: Vec<Arc<Ranges>> = {
            let mut flight = lock(&self.flight);
            flight.retain(|held| held.strong_count() > 0);
            flight.iter().filter_map(Weak::upgrade).collect()
        };
        if held.is_empty() {
            return;
        }
        let ranges: Vec<Range> = held
            .iter()
            .flat_map(|r| r.written.iter().copied())
            .collect();
        settle(core, &merged(ranges));
    }
}

/// Flushes `ranges` (`vkFlushMappedMemoryRanges`), counted; nothing for none.
pub(crate) fn flush(core: &Core, ranges: &[Range]) {
    if ranges.is_empty() {
        return;
    }
    let mapped: Vec<vk::MappedMemoryRange> = ranges.iter().map(|range| range.mapped()).collect();
    // SAFETY: each range is of an allocation the runtime maps whole and keeps while the range is
    // named (a block a command buffer or upload keeps, the staging region, a staging buffer, the
    // zero source), its ends at atoms or the allocation's end.
    let result = unsafe { core.device.flush_mapped_memory_ranges(&mapped) };
    count(&core.counters, Call::Flush, ranges, result.is_err());
}

/// Flushes `ranges`, then invalidates them (`vkInvalidateMappedMemoryRanges`), counted: the
/// GPU's writes there are visible to the host's reads after it, and the host's own writes made
/// before it are kept. Nothing for none.
pub(crate) fn settle(core: &Core, ranges: &[Range]) {
    if ranges.is_empty() {
        return;
    }
    flush(core, ranges);
    let mapped: Vec<vk::MappedMemoryRange> = ranges.iter().map(|range| range.mapped()).collect();
    // SAFETY: as in `flush`.
    let result = unsafe { core.device.invalidate_mapped_memory_ranges(&mapped) };
    count(&core.counters, Call::Invalidate, ranges, result.is_err());
}

/// Which call was made.
#[derive(Clone, Copy)]
pub(crate) enum Call {
    Flush,
    Invalidate,
}

fn count(counters: &Counters, call: Call, ranges: &[Range], failed: bool) {
    let bytes = ranges.iter().map(|range| range.bytes()).sum();
    counters.coherence(call, bytes, failed);
}
