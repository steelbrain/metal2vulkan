//! Shared memory's host coherence: the flushes and invalidates Vulkan asks of mapped memory of a
//! Shared type that is not host-coherent.
//!
//! Shared storage lives in the first host-visible, host-cached memory type a buffer can use, a
//! host-coherent one first (`floor::choose`). Vulkan makes the host's writes to mapped memory of a
//! type without `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` available to the device only through
//! `vkFlushMappedMemoryRanges`, and the device's writes visible to the host only through
//! `vkInvalidateMappedMemoryRanges` ("Host Access to Device Memory Objects"). The device's make
//! reads the chosen type's `HOST_COHERENT` flag once ([`Coherence::of`]): where the type is
//! host-coherent, the device holds no [`Coherence`], keeps no ranges and makes neither call, as
//! Vulkan has them "unnecessary" there; where it is not, the device makes both calls, over the
//! ranges commands reach, as below.
//!
//! - **Before each submission**, the Shared memory its command buffers reach is flushed: each
//!   buffer, texture over a buffer and heap a command buffer with a segment in it keeps, as its
//!   commit found them ([`Ranges::of`]), so the host's writes up to that submission reach the GPU,
//!   those made after a commit and before the host value its segment waits for among them; and
//!   the ranges each Managed copy-in it carries reads (`crate::capture`). A Managed buffer with a
//!   device copy counts for none of its bytes: its commands reach the copy, and the block of its
//!   bytes only its copy-ins read and its synchronize writes, each naming its own bytes there.
//! - **Before the host can see the GPU's progress**, the Shared memory a command buffer may write
//!   is flushed, then invalidated ([`Coherence::settle`]): as it retires, before its status reads
//!   completed, a wait for it returns or its completed handlers run; and, for every command buffer
//!   in flight, before a shared event's value a GPU signal raised reaches a listener,
//!   `signaledValue` or a wait for it. The flush first keeps the host's own writes to those ranges, which Vulkan leaves
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

use crate::device::{lock, Core};
use crate::memory::{Memory, MemoryClass};
use ash::vk;
use std::sync::{Arc, Mutex, Weak};

/// The most ranges a call names from an array on the stack; a call naming more collects them.
const ON_STACK: usize = 8;

/// A range of mapped memory: its allocation, and its first and end bytes there, each a multiple
/// of the device's `nonCoherentAtomSize` or the allocation's end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

/// Puts `ranges` in order, in place, each pair that meets or touches in one allocation made one.
pub(crate) fn merge(ranges: &mut Vec<Range>) {
    ranges.sort_unstable();
    ranges.dedup_by(|next, last| {
        let meets = last.memory == next.memory && next.start <= last.end;
        if meets {
            last.end = last.end.max(next.end);
        }
        meets
    });
}

/// The Shared memory a recording's commands may write, as they name it: a range named again
/// right after itself is held once, and the ranges are merged each time their count reaches a
/// power of two from 64, so a recording that names a few ranges over and over holds each about
/// once.
#[derive(Default)]
pub(crate) struct Writes {
    ranges: Vec<Range>,
}

impl Writes {
    /// A command may write `range`; nothing for `None`.
    pub(crate) fn add(&mut self, range: Option<Range>) {
        let Some(range) = range else {
            return;
        };
        if self.ranges.last() == Some(&range) {
            return;
        }
        let len = self.ranges.len();
        if len >= 64 && len.is_power_of_two() {
            merge(&mut self.ranges);
        }
        self.ranges.push(range);
    }
}

/// What a committed command buffer's commands reach of Shared memory, merged: every range, which
/// each submission that carries one of its segments flushes first, and those they may write,
/// which the host's sight of their progress settles first ([`Coherence::settle`]).
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
        let mut written = writes.ranges.clone();
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
        merge(&mut reached);
        merge(&mut written);
        Some(Ranges { reached, written })
    }
}

/// The device's side of coherence, held only where its Shared type is not host-coherent: the atom
/// ranges widen to, and the written ranges of each command buffer committed and not yet retired
/// that may write Shared memory, which a shared event's signal seen settles.
pub(crate) struct Coherence {
    /// The device's `nonCoherentAtomSize`.
    pub(crate) atom: u64,
    flight: Mutex<Vec<Weak<Ranges>>>,
}

impl Coherence {
    /// The device's coherence, where its Shared type (`types.shared` of `memory`) lacks
    /// `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`; `None` where it has it. `atom` is the device's
    /// `nonCoherentAtomSize`.
    pub(crate) fn of(
        memory: &vk::PhysicalDeviceMemoryProperties,
        types: crate::floor::MemoryTypes,
        atom: u64,
    ) -> Option<Coherence> {
        let shared = memory.memory_types_as_slice()[types.shared as usize];
        let coherent = (shared.property_flags).contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        (!coherent).then(|| Coherence {
            atom: atom.max(1),
            flight: Mutex::new(Vec::new()),
        })
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

    /// A command buffer committed with `ranges` retires, before it lets go of the blocks they lie
    /// in: once this returns, no shared event's signal seen settles them. A settle under way
    /// finishes first, as it runs under the same lock ([`signal_seen`](Self::signal_seen)), so
    /// no call names memory a retired command buffer gave back.
    pub(crate) fn retired(&self, ranges: &Arc<Ranges>) {
        if ranges.written.is_empty() {
            return;
        }
        let mut flight = lock(&self.flight);
        flight.retain(|held| held.strong_count() > 0 && held.as_ptr() != Arc::as_ptr(ranges));
    }

    /// Before the host sees a value a GPU signal raised: flushes, then invalidates, the written
    /// ranges of every command buffer committed and not yet retired (`settle`), under the lock
    /// that keeps each of them from retiring meanwhile ([`retired`](Self::retired)).
    pub(crate) fn signal_seen(&self, core: &Core) {
        let mut flight = lock(&self.flight);
        flight.retain(|held| held.strong_count() > 0);
        let mut ranges = Vec::new();
        for held in flight.iter().filter_map(Weak::upgrade) {
            ranges.extend_from_slice(&held.written);
        }
        merge(&mut ranges);
        self.settle(core, &ranges);
    }

    /// Flushes `ranges` (`vkFlushMappedMemoryRanges`), counted; nothing for none.
    pub(crate) fn flush(&self, core: &Core, ranges: &[Range]) {
        if ranges.is_empty() {
            return;
        }
        // SAFETY: each range is of an allocation the runtime maps whole and keeps while the range
        // is named (a block a command buffer or upload keeps, the staging region, a staging
        // buffer, the zero source), its ends at atoms or the allocation's end.
        let failed = with_mapped(ranges, |mapped| unsafe {
            core.device.flush_mapped_memory_ranges(mapped).is_err()
        });
        core.counters.coherence(Call::Flush, bytes(ranges), failed);
    }

    /// Flushes `ranges`, then invalidates them (`vkInvalidateMappedMemoryRanges`), counted: the
    /// GPU's writes there are visible to the host's reads after it, and the host's own writes made
    /// before it are kept. Nothing for none.
    pub(crate) fn settle(&self, core: &Core, ranges: &[Range]) {
        if ranges.is_empty() {
            return;
        }
        // SAFETY: as in `flush`.
        let (flushed, invalidated) = with_mapped(ranges, |mapped| unsafe {
            let flushed = core.device.flush_mapped_memory_ranges(mapped).is_err();
            (
                flushed,
                core.device.invalidate_mapped_memory_ranges(mapped).is_err(),
            )
        });
        let bytes = bytes(ranges);
        core.counters.coherence(Call::Flush, bytes, flushed);
        core.counters
            .coherence(Call::Invalidate, bytes, invalidated);
    }
}

/// `call` with `ranges` as Vulkan names them: from an array on the stack for up to [`ON_STACK`]
/// of them, else collected.
fn with_mapped<R>(ranges: &[Range], call: impl FnOnce(&[vk::MappedMemoryRange]) -> R) -> R {
    if ranges.len() <= ON_STACK {
        let mut mapped = [vk::MappedMemoryRange::default(); ON_STACK];
        for (slot, range) in mapped.iter_mut().zip(ranges) {
            *slot = range.mapped();
        }
        return call(&mapped[..ranges.len()]);
    }
    let mapped: Vec<vk::MappedMemoryRange> = ranges.iter().map(|range| range.mapped()).collect();
    call(&mapped)
}

/// Which call was made.
#[derive(Clone, Copy)]
pub(crate) enum Call {
    Flush,
    Invalidate,
}

/// The bytes `ranges` name.
fn bytes(ranges: &[Range]) -> u64 {
    ranges.iter().map(|range| range.bytes()).sum()
}
