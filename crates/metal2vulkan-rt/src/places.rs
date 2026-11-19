//! Places: the Private blocks the runtime places textures' images in, taken as the consumer takes
//! blocks. A texture block holds images alone, so no buffer lies beside an optimal image in one
//! block.
//!
//! A place is a resource's bytes at a multiple of its alignment: the first that fits, in the
//! blocks in the order they were taken; where none fits, the next block comes from the pool: of
//! the class's block length ([`MemoryClass::block_len`]), or of the place's own length where that
//! is longer. A place is free again once its resource is freed, after every command buffer and
//! upload that used it retired. One empty block of the block length is kept for the next place;
//! another block that empties goes back to the pool, whose free ranges its bytes join.
//! A place's bytes are what its last resource left; a new texture's move home zeroes its image
//! ([`crate::texture::Zero`]), unless the place answers that its bytes read zero: a place in a
//! block the pool handed out zeroed, past every place the block held before.
//!
//! An image in a texture block is optimal. The `VkBuffer` of the block's allocation, which spans
//! it, aliases the images (Vulkan's memory aliasing), and nothing records through it in a texture
//! block. Writes through it in other blocks touch no page of a texture block's: blocks start and
//! end on the pool's granule, at least the device's `bufferImageGranularity`.

use crate::counters::Site;
use crate::device::{lock, Core};
use crate::error::Error;
use crate::memory::{BlockId, Memory, MemoryClass};
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
    /// Its bytes read zero as the pool handed it out ([`BlockInfo::zeroed`](crate::BlockInfo::zeroed)).
    zeroed: bool,
    /// The end of the furthest place taken in it: a place past it is bytes no place held.
    taken_to: u64,
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

    /// A place of `len` bytes at a multiple of `align`, a power of two, its block, and whether
    /// its bytes read zero: true where the block read zero as the pool handed it out and no place
    /// of it ever held a byte at or past the place's first byte. Where no block has room, a
    /// Private block is taken from the pool, without the lock: of the block length, or of `len`
    /// where that is longer.
    ///
    /// Err: what [`Core::take_memory`] returns.
    pub(crate) fn place(
        &self,
        core: &Core,
        len: u64,
        align: u64,
    ) -> Result<(Place, Memory, bool), Error> {
        let mut blocks = lock(&self.blocks);
        loop {
            for block in blocks.iter_mut() {
                // The image's alignment holds in the allocation the block is a range of.
                if let Some(offset) = block.free.take_from(len, align, block.memory.base()) {
                    block.live += 1;
                    let zeroed = block.zeroed && offset >= block.taken_to;
                    block.taken_to = block.taken_to.max(offset + len);
                    let place = Place {
                        block: block.memory.id,
                        offset,
                        len,
                    };
                    return Ok((place, block.memory, zeroed));
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
            blocks.push(PlaceBlock {
                memory,
                free: Ranges::new(memory.len),
                live: 0,
                zeroed: memory.info().zeroed,
                taken_to: 0,
            });
        }
    }

    /// Frees `place`. Where its block empties and is longer than the block length, or another
    /// block of the set of that length is empty, the block goes back to the pool.
    pub(crate) fn release(&self, core: &Core, place: Place) {
        let mut blocks = lock(&self.blocks);
        let Some(index) = (blocks.iter()).position(|block| block.memory.id == place.block) else {
            return;
        };
        let block = &mut blocks[index];
        block.free.give(place.offset, place.len);
        block.live -= 1;
        let standard = |block: &PlaceBlock| block.memory.len == BLOCK_LEN;
        let empty = |block: &PlaceBlock| block.live == 0 && standard(block);
        let surplus = !standard(block) || blocks.iter().filter(|block| empty(block)).count() > 1;
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
