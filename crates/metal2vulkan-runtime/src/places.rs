//! Places: the Private blocks the runtime places textures' images in, taken as the consumer takes
//! blocks. A texture block holds images alone, so no buffer lies beside an optimal image in one
//! block.
//!
//! A place is a resource's bytes at a multiple of its alignment. One no longer than the class's
//! block length ([`MemoryClass::block_len`]) is the first that fits in the blocks of that length,
//! in the order they were taken; where none fits, the next such block comes from the pool. A
//! longer one takes a block of its own length from the pool, which holds no other place, not even
//! in the bytes past it, so its memory goes back with it. A place is free again once its resource
//! is freed, after every command buffer and upload that used it retired. One empty block of the
//! block length is kept for the next place; another block that empties goes back to the pool,
//! whose free ranges its bytes join.
//! A place's bytes are what its last resource left; a new texture's move home zeroes its image
//! ([`crate::texture::Zero`]), unless the place answers that its bytes read zero: a place in a
//! block the pool handed out zeroed, past every place the block held before.
//!
//! An image in a texture block is optimal. The `VkBuffer` of the block's allocation, which spans
//! it, aliases the images (Vulkan's memory aliasing), and nothing records through it in a texture
//! block. Writes through it in other blocks touch no page of a texture block's: blocks start and
//! end on the pool's granule, at least the device's `bufferImageGranularity`.

use crate::counters::{Gauge, Site};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::memory::{BlockId, Memory, MemoryClass, SPARSE};
use std::sync::Mutex;

/// The length of the blocks a set takes for places no longer than it.
const BLOCK_LEN: u64 = MemoryClass::Private.block_len();

/// One set of blocks.
pub(crate) struct Places {
    blocks: Mutex<Vec<PlaceBlock>>,
}

struct PlaceBlock {
    memory: Memory,
    free: Ranges,
    /// The places taken in it.
    live: usize,
    /// Their bytes.
    placed: u64,
}

impl PlaceBlock {
    /// Whether it is sparse: it holds a place, and its places' bytes are less than a sixteenth
    /// of its length
    /// ([`Gauges::texture_block_sparse_bytes`](crate::Gauges::texture_block_sparse_bytes)).
    fn sparse(&self) -> bool {
        self.live > 0 && self.placed < self.memory.len / SPARSE
    }

    /// Takes or frees a place of `len` bytes (`taken`), moving the sparse gauges with it.
    fn count(&mut self, core: &Core, len: u64, taken: bool) {
        let was = self.sparse();
        match taken {
            true => {
                self.live += 1;
                self.placed += len;
                core.counters.rise_by(Gauge::TextureBlockBytes, len);
            }
            false => {
                self.live -= 1;
                self.placed -= len;
                core.counters.fall_by(Gauge::TextureBlockBytes, len);
            }
        }
        let counters = &core.counters;
        match (was, self.sparse()) {
            (false, true) => {
                counters.rise(Gauge::TextureBlockSparseBlocks);
                counters.rise_by(Gauge::TextureBlockSparseBytes, self.memory.len);
            }
            (true, false) => {
                counters.fall(Gauge::TextureBlockSparseBlocks);
                counters.fall_by(Gauge::TextureBlockSparseBytes, self.memory.len);
            }
            _ => {}
        }
    }
}

/// A resource's place: `len` bytes at `offset` in a block of a set.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Place {
    block: BlockId,
    pub(crate) offset: u64,
    len: u64,
}

impl Place {
    /// For the device tests: the block it is in.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn block(&self) -> BlockId {
        self.block
    }
}

impl Places {
    pub(crate) fn new() -> Places {
        Places {
            blocks: Mutex::new(Vec::new()),
        }
    }

    /// A place of `len` bytes at a multiple of `align`, a power of two, and its block: in a block
    /// of the block length, or, where `len` is longer, alone in the block taken for it. Where no
    /// block has room, a Private block is taken from the pool, without the lock: of the block
    /// length, or of `len` where that is longer.
    ///
    /// Err: what [`Core::take_memory`] returns.
    pub(crate) fn place(
        &self,
        core: &Core,
        len: u64,
        align: u64,
    ) -> Result<(Place, Memory), Error> {
        // A place no longer than the block length goes in a block of that length; a longer one
        // only in an empty block, the one taken for it, whose bytes past it then stay free.
        let serves = |block: &PlaceBlock| match len <= BLOCK_LEN {
            true => block.memory.len == BLOCK_LEN,
            false => block.live == 0,
        };
        let mut blocks = lock(&self.blocks);
        loop {
            for block in blocks.iter_mut().filter(|block| serves(block)) {
                // The image's alignment holds in the allocation the block is a range of.
                if let Some(offset) = block.free.take_from(len, align, block.memory.base()) {
                    block.count(core, len, true);
                    let place = Place {
                        block: block.memory.id,
                        offset,
                        len,
                    };
                    return Ok((place, block.memory));
                }
            }
            drop(blocks);
            let memory =
                core.take_memory(MemoryClass::Private, len.max(BLOCK_LEN), Site::TextureBlock)?;
            // A block that spilled out of device-local memory holds no place: textures then take
            // memory of their own, and a Managed buffer's device copy a Private block of its own.
            if memory.spilled() {
                core.pool.give_back(memory);
                return Err(Error::out_of_device_memory());
            }
            blocks = lock(&self.blocks);
            core.counters
                .rise_by(Gauge::TextureBlockHeldBytes, memory.len);
            blocks.push(PlaceBlock {
                memory,
                free: Ranges::new(memory.len),
                live: 0,
                placed: 0,
            });
        }
    }

    /// Frees `place`. Where its block empties and is longer than the block length, or another
    /// block of the set of that length is empty, the block goes back to the pool.
    pub(crate) fn release(&self, core: &Core, place: Place) {
        let mut blocks = lock(&self.blocks);
        let Some(index) = (blocks.iter()).position(|block| block.memory.id == place.block) else {
            core.counters.fall_by(Gauge::TextureBlockBytes, place.len);
            return;
        };
        let block = &mut blocks[index];
        block.free.give(place.offset, place.len);
        block.count(core, place.len, false);
        let standard = |block: &PlaceBlock| block.memory.len == BLOCK_LEN;
        let empty = |block: &PlaceBlock| block.live == 0 && standard(block);
        let surplus = !standard(block) || blocks.iter().filter(|block| empty(block)).count() > 1;
        if blocks[index].live == 0 && surplus {
            let emptied = blocks.remove(index);
            drop(blocks);
            (core.counters).fall_by(Gauge::TextureBlockHeldBytes, emptied.memory.len);
            core.pool.give_back(emptied.memory);
        }
    }

    /// For the device tests: the blocks of the set.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn blocks(&self) -> usize {
        lock(&self.blocks).len()
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
        self.take_from(len, align, 0)
    }

    /// The first offset that lies `base` bytes past a multiple of `align`, a power of two, less
    /// `base`, with `len` bytes, more than 0, free from it, and the index of the range it lies in.
    fn first_fit(&self, len: u64, align: u64, base: u64) -> Option<(usize, u64)> {
        (self.free.iter().enumerate()).find_map(|(index, &(start, end))| {
            let at = start.checked_add(base)?.checked_next_multiple_of(align)? - base;
            (at.checked_add(len)? <= end).then_some((index, at))
        })
    }

    /// Takes `len` bytes, more than 0, at the first offset that lies `base` bytes past a multiple
    /// of `align`, a power of two, less `base`, with the bytes free from it: the first offset
    /// aligned where the ranges start `base` bytes into something larger. `None` where there is
    /// none.
    pub(crate) fn take_from(&mut self, len: u64, align: u64, base: u64) -> Option<u64> {
        let (index, at) = self.first_fit(len, align, base)?;
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
