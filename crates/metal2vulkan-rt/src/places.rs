//! Places: the Private blocks the runtime places its own resources in, taken as the consumer takes
//! blocks. The device has two sets: texture blocks, where textures' images are bound, and copy
//! blocks, which hold Managed buffers' device copies. Neither set holds the other's kind, so no
//! buffer lies beside an optimal image in one allocation.
//!
//! A place is a resource's bytes at a multiple of its alignment: the first that fits, in the
//! blocks in the order they were taken; where none fits, the next block comes from the pool, or,
//! for a place longer than a pooled block, a dedicated block of its own. A place is free again once
//! its resource is freed, after every command buffer and upload that used it retired. One empty
//! pooled block of a set is kept for the next place; another that empties, and a dedicated block
//! once empty, goes back to the pool, which zeroes and keeps a pooled block or frees it.
//!
//! An image in a texture block is optimal. The block's own `VkBuffer`, which spans it, aliases the
//! images (Vulkan's memory aliasing), and only the pool uses it: to zero the block before it is
//! taken and after it is given back, when no image is in it. A copy block's `VkBuffer` is what the
//! device copies are reached through.

use crate::device::{lock, Core};
use crate::error::Error;
use crate::memory::{BlockId, Memory, MemoryClass};
use std::sync::{Mutex, PoisonError};

/// One set of blocks.
pub(crate) struct Places {
    blocks: Mutex<Vec<PlaceBlock>>,
}

struct PlaceBlock {
    memory: Memory,
    free: Ranges,
    /// The places taken in it.
    live: usize,
}

/// A resource's place: `len` bytes at `offset` in a block of a set.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Place {
    block: BlockId,
    pub(crate) offset: u64,
    len: u64,
}

impl Places {
    pub(crate) fn new() -> Places {
        Places {
            blocks: Mutex::new(Vec::new()),
        }
    }

    /// A place of `len` bytes at a multiple of `align`, a power of two, and its block. Where no
    /// block has room, a Private block is taken, without the lock: from the pool, or a dedicated
    /// one for a place longer than a pooled block.
    ///
    /// Err: what [`Core::take_memory`] returns.
    pub(crate) fn place(
        &self,
        core: &Core,
        len: u64,
        align: u64,
    ) -> Result<(Place, Memory), Error> {
        let mut blocks = lock(&self.blocks);
        loop {
            for block in blocks.iter_mut() {
                if let Some(offset) = block.free.take(len, align) {
                    block.live += 1;
                    let place = Place {
                        block: block.memory.id,
                        offset,
                        len,
                    };
                    return Ok((place, block.memory));
                }
            }
            drop(blocks);
            let memory = core.take_memory(MemoryClass::Private, len)?;
            blocks = lock(&self.blocks);
            blocks.push(PlaceBlock {
                memory,
                free: Ranges::new(memory.len),
                live: 0,
            });
        }
    }

    /// Frees `place`. Where its block empties and is dedicated, or another pooled block of the
    /// set is empty, the block goes back to the pool.
    pub(crate) fn release(&self, core: &Core, place: Place) {
        let mut blocks = lock(&self.blocks);
        let Some(index) = (blocks.iter()).position(|block| block.memory.id == place.block) else {
            return;
        };
        let block = &mut blocks[index];
        block.free.give(place.offset, place.len);
        block.live -= 1;
        let empty = |block: &PlaceBlock| block.live == 0 && block.memory.pooled();
        let surplus =
            !block.memory.pooled() || blocks.iter().filter(|block| empty(block)).count() > 1;
        if blocks[index].live == 0 && surplus {
            let emptied = blocks.remove(index);
            drop(blocks);
            core.pool.give_back(emptied.memory);
        }
    }

    /// For the device tests: the blocks of the set.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn blocks(&self) -> usize {
        lock(&self.blocks).len()
    }

    /// At teardown, once every resource placed is freed: the blocks, for the core to free.
    pub(crate) fn remaining(&mut self) -> Vec<Memory> {
        let blocks = self
            .blocks
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        blocks.drain(..).map(|block| block.memory).collect()
    }
}

/// A block's free ranges, `start..end` by offset, no two adjacent.
pub(crate) struct Ranges {
    free: Vec<(u64, u64)>,
}

impl Ranges {
    /// `len` bytes, all free.
    pub(crate) fn new(len: u64) -> Ranges {
        Ranges {
            free: vec![(0, len)],
        }
    }

    /// Takes `len` bytes, more than 0, at the first offset that is a multiple of `align`, a power
    /// of two, with the bytes free from it; `None` where there is none.
    pub(crate) fn take(&mut self, len: u64, align: u64) -> Option<u64> {
        let (index, at) = (self.free.iter().enumerate()).find_map(|(index, &(start, end))| {
            let at = start.checked_next_multiple_of(align)?;
            (at.checked_add(len)? <= end).then_some((index, at))
        })?;
        let (start, end) = self.free[index];
        let stop = at + len;
        match (start < at, stop < end) {
            (false, false) => {
                self.free.remove(index);
            }
            (true, false) => self.free[index] = (start, at),
            (false, true) => self.free[index] = (stop, end),
            (true, true) => {
                self.free[index] = (start, at);
                self.free.insert(index + 1, (stop, end));
            }
        }
        Some(at)
    }

    /// Frees `offset..offset + len`, taken before, joining it to the free ranges beside it.
    pub(crate) fn give(&mut self, offset: u64, len: u64) {
        let end = offset + len;
        let index = self.free.partition_point(|&(start, _)| start < offset);
        let before = index > 0 && self.free[index - 1].1 == offset;
        let after = index < self.free.len() && self.free[index].0 == end;
        match (before, after) {
            (true, true) => {
                self.free[index - 1].1 = self.free[index].1;
                self.free.remove(index);
            }
            (true, false) => self.free[index - 1].1 = end,
            (false, true) => self.free[index].0 = offset,
            (false, false) => self.free.insert(index, (offset, end)),
        }
    }

    /// For the tests: the free ranges.
    #[cfg(test)]
    pub(crate) fn free(&self) -> &[(u64, u64)] {
        &self.free
    }
}
