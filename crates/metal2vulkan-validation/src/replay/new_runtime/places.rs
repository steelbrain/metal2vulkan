//! The places of the stream's own buffers in the blocks the backend takes: each memory class's
//! pooled blocks (`MemoryClass::block_len`), first fit at Metal's 256-byte buffer alignment, and
//! a block of its own for a buffer past the class's block length. A buffer's place is free again only once every command buffer
//! committed before the buffer was released has completed, as Metal keeps a released buffer for
//! the work that names it.

use metal2vulkan_rt as rt;
use std::ops::Range;

/// Where a buffer's place starts: Metal's buffers lie at multiples of 256 bytes.
const ALIGN: u64 = 256;

/// A buffer's place: its block, its bytes there, and the block's handle where the block is its
/// own.
pub(super) struct Place {
    pub(super) block: rt::BlockId,
    pub(super) range: Range<u64>,
    own: Option<rt::Block>,
}

/// One pooled block: its handle, and its free ranges, in order.
struct Block {
    info: rt::Block,
    free: Vec<Range<u64>>,
}

/// The blocks of both classes, and the places freed but not yet free.
pub(super) struct Places {
    device: rt::Device,
    blocks: Vec<Block>,
    /// A released buffer's place, and the commits before its release.
    held: Vec<(Place, u64)>,
    /// The bytes of the blocks held, pooled and own, by class (`class_index`).
    bytes: [u64; 2],
}

/// `class`'s index in a pair of per-class values: Private, then Shared.
pub(super) fn class_index(class: rt::MemoryClass) -> usize {
    match class {
        rt::MemoryClass::Private => 0,
        rt::MemoryClass::Shared => 1,
    }
}

impl Places {
    pub(super) fn new(device: rt::Device) -> Places {
        Places {
            device,
            blocks: Vec::new(),
            held: Vec::new(),
            bytes: [0; 2],
        }
    }

    /// The bytes of the blocks the places hold, by class (`class_index`): pooled blocks whole,
    /// own blocks at their buffers' lengths.
    pub(super) fn bytes(&self) -> [u64; 2] {
        self.bytes
    }

    /// A place of `len` bytes of `class`, once every place whose commits have all completed
    /// (`completed`, the commits that have) is free again.
    pub(super) fn take(
        &mut self,
        class: rt::MemoryClass,
        len: u64,
        completed: u64,
    ) -> Result<Place, rt::Error> {
        self.free_completed(completed)?;
        if len > class.block_len() {
            let block = self.device.take_block(class, len)?;
            self.bytes[class_index(class)] += len;
            return Ok(Place {
                block: block.id,
                range: 0..len,
                own: Some(block),
            });
        }
        let span = len.next_multiple_of(ALIGN);
        for block in self
            .blocks
            .iter_mut()
            .filter(|block| block.info.class == class)
        {
            if let Some(range) = take_from(&mut block.free, span) {
                return Ok(Place {
                    block: block.info.id,
                    range: range.start..range.start + len,
                    own: None,
                });
            }
        }
        let info = self.device.take_block(class, class.block_len())?;
        self.bytes[class_index(class)] += info.len;
        let mut free = Vec::new();
        free.push(0..info.len);
        let range = take_from(&mut free, span).expect("a new block holds any pooled place");
        let block = info.id;
        self.blocks.push(Block { info, free });
        Ok(Place {
            block,
            range: range.start..range.start + len,
            own: None,
        })
    }

    /// `place` given back after `commits` commits: free once they have all completed.
    pub(super) fn give(&mut self, place: Place, commits: u64) {
        self.held.push((place, commits));
    }

    fn free_completed(&mut self, completed: u64) -> Result<(), rt::Error> {
        let mut at = 0;
        while at < self.held.len() {
            if self.held[at].1 > completed {
                at += 1;
                continue;
            }
            let (place, _) = self.held.swap_remove(at);
            if let Some(own) = place.own {
                let class = own.class;
                // The runtime keeps the block for the command buffers that use it.
                self.device.release_block(own)?;
                self.bytes[class_index(class)] -= place.range.end;
                continue;
            }
            let span = (place.range.end - place.range.start).next_multiple_of(ALIGN);
            let start = place.range.start;
            if let Some(block) = self.blocks.iter_mut().find(|b| b.info.id == place.block) {
                give_to(&mut block.free, start..start + span);
            }
        }
        Ok(())
    }
}

/// The first free range of `free` that holds `span` bytes, taken from its start.
fn take_from(free: &mut Vec<Range<u64>>, span: u64) -> Option<Range<u64>> {
    let at = free
        .iter()
        .position(|range| range.end - range.start >= span)?;
    let start = free[at].start;
    free[at].start += span;
    if free[at].start == free[at].end {
        free.remove(at);
    }
    Some(start..start + span)
}

/// `range` back in `free`, merged with the ranges it touches.
fn give_to(free: &mut Vec<Range<u64>>, range: Range<u64>) {
    let at = free.partition_point(|r| r.start < range.start);
    free.insert(at, range);
    if at + 1 < free.len() && free[at].end == free[at + 1].start {
        free[at].end = free[at + 1].end;
        free.remove(at + 1);
    }
    if at > 0 && free[at - 1].end == free[at].start {
        free[at - 1].end = free[at].end;
        free.remove(at);
    }
}
