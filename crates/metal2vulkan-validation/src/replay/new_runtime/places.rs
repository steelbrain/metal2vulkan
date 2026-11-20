//! The places of the stream's own buffers in the blocks the backend takes: each memory class's
//! pooled blocks (`MemoryClass::block_len`), first fit at Metal's 256-byte buffer alignment, and
//! a block of its own for a buffer past the class's block length. A pooled block goes back to the
//! runtime once its last place is free again, as the runtime's own blocks do, and the runtime's
//! idle rule frees its allocation; a later place takes a block again. A place spans its buffer's
//! length rounded up to the buffer's recorded granule ([`granule`]), so the bytes past a buffer
//! are the slack it had in the recording. A buffer's place is free again
//! only once every command buffer that named the buffer has completed, as Metal keeps a released
//! buffer for the work that names it: those committed before its release, and those that named it
//! before its release and are committed after it (a command buffer released uncommitted runs
//! nothing, and holds nothing from then).

use metal2vulkan_runtime as rt;
use std::collections::BTreeMap;
use std::ops::Range;

/// Where a buffer's place starts, and the least granule: Metal's buffers lie at multiples of 256
/// bytes.
const ALIGN: u64 = 256;

/// The most a recorded granule is taken as: past it, an address's alignment says more about where
/// the guest's allocator started than about the granule it carves at.
const MOST_GRANULE: u64 = 64 << 10;

/// The granule of a buffer the stream recorded at `address`: the largest power of two that
/// divides the address, from Metal's 256-byte alignment up to 64 KiB, and 256 bytes for a buffer
/// with no recorded address (or the player's own). Its place spans its length rounded up to the
/// granule: a program that writes past a buffer's end writes into the slack the buffer had in the
/// recording, as the recording's program did, and not into the next buffer.
pub(super) fn granule(address: Option<u64>) -> u64 {
    match address {
        Some(address) if address != 0 => (1 << address.trailing_zeros()).clamp(ALIGN, MOST_GRANULE),
        _ => ALIGN,
    }
}

/// A buffer's place: its block, its buffer's bytes there, the bytes the place spans from their
/// start (the buffer's, then its slack), and the block's handle where the block is its own.
pub(super) struct Place {
    pub(super) block: rt::BlockId,
    pub(super) range: Range<u64>,
    span: u64,
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
    /// A released buffer's place while command buffers that named it are not yet committed.
    waiting: Waiting<Place>,
    /// The bytes of the blocks held, pooled and own, by class (`class_index`).
    bytes: [u64; 2],
    /// The places taken, by granule.
    granules: BTreeMap<u64, u64>,
    /// The bytes of slack the places hold now, and the most they held.
    slack: (u64, u64),
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
            waiting: Waiting::default(),
            bytes: [0; 2],
            granules: BTreeMap::new(),
            slack: (0, 0),
        }
    }

    /// The bytes of the blocks the places hold, by class (`class_index`): pooled blocks whole,
    /// own blocks at their buffers' lengths.
    pub(super) fn bytes(&self) -> [u64; 2] {
        self.bytes
    }

    /// The `places` line: the places taken by granule ([`granule`]), and the most bytes of slack
    /// held at once.
    pub(super) fn line(&self) -> String {
        let granules: Vec<String> = (self.granules.iter())
            .map(|(granule, count)| format!("{granule} B {count}"))
            .collect();
        format!(
            "places by recorded granule: {}; most slack held {} bytes",
            granules.join(", "),
            self.slack.1
        )
    }

    /// A place of `len` bytes of `class`, spanning `len` rounded up to `granule` ([`granule`]),
    /// once every place whose commits have all completed (`completed`, the commits that have) is
    /// free again.
    pub(super) fn take(
        &mut self,
        class: rt::MemoryClass,
        len: u64,
        granule: u64,
        completed: u64,
    ) -> Result<Place, rt::Error> {
        self.free_completed(completed)?;
        let span = len.next_multiple_of(granule.max(ALIGN));
        *self.granules.entry(granule.max(ALIGN)).or_default() += 1;
        self.slack.0 += span - len;
        self.slack.1 = self.slack.1.max(self.slack.0);
        if span > class.block_len() {
            let block = self.device.take_block(class, span)?;
            self.bytes[class_index(class)] += span;
            return Ok(Place {
                block: block.id,
                range: 0..len,
                span,
                own: Some(block),
            });
        }
        for block in self
            .blocks
            .iter_mut()
            .filter(|block| block.info.class == class)
        {
            if let Some(range) = take_from(&mut block.free, span) {
                return Ok(Place {
                    block: block.info.id,
                    range: range.start..range.start + len,
                    span,
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
            span,
            own: None,
        })
    }

    /// `place` given back after `commits` commits, while the command buffers `users` that named
    /// its buffer are not yet committed: free once each user is committed (or released
    /// uncommitted, [`ended`](Self::ended)) and every commit up to the last of them has completed.
    pub(super) fn give(&mut self, place: Place, commits: u64, users: Vec<u64>) {
        if let Some(place) = self.waiting.wait(place, users) {
            self.held.push((place, commits));
        }
    }

    /// Command buffer `command_buffer` committed, or released uncommitted, as the `commits`th
    /// commit or after it: the places that waited for it alone are free once those commits have
    /// completed.
    pub(super) fn ended(&mut self, command_buffer: u64, commits: u64) {
        for place in self.waiting.ended(command_buffer) {
            self.held.push((place, commits));
        }
    }

    /// Frees each place given back whose commits have all completed (`completed`, the commits
    /// that have), and gives back to the runtime each pooled block that then holds no place.
    pub(super) fn free_completed(&mut self, completed: u64) -> Result<(), rt::Error> {
        let mut at = 0;
        while at < self.held.len() {
            if self.held[at].1 > completed {
                at += 1;
                continue;
            }
            let (place, _) = self.held.swap_remove(at);
            self.slack.0 -= place.span - (place.range.end - place.range.start);
            if let Some(own) = place.own {
                let class = own.class;
                // The runtime keeps the block for the command buffers that use it.
                self.device.release_block(own)?;
                self.bytes[class_index(class)] -= place.span;
                continue;
            }
            let start = place.range.start;
            let Some(index) = self.blocks.iter().position(|b| b.info.id == place.block) else {
                continue;
            };
            let block = &mut self.blocks[index];
            give_to(&mut block.free, start..start + place.span);
            let len = block.info.len;
            if matches!(block.free.as_slice(), [whole] if *whole == (0..len)) {
                // In the order taken, as first fit reads them.
                let emptied = self.blocks.remove(index);
                let class = emptied.info.class;
                self.device.release_block(emptied.info)?;
                self.bytes[class_index(class)] -= len;
            }
        }
        Ok(())
    }
}

/// Things that wait for command buffers not yet committed: each is let go once every one of its
/// command buffers has ended its wait.
pub(in crate::replay) struct Waiting<T> {
    items: Vec<(T, Vec<u64>)>,
}

impl<T> Default for Waiting<T> {
    fn default() -> Self {
        Waiting { items: Vec::new() }
    }
}

impl<T> Waiting<T> {
    /// `item` waiting for `command_buffers`; back at once where there are none.
    pub(in crate::replay) fn wait(&mut self, item: T, command_buffers: Vec<u64>) -> Option<T> {
        if command_buffers.is_empty() {
            return Some(item);
        }
        self.items.push((item, command_buffers));
        None
    }

    /// `command_buffer`'s wait over: the items that waited for it last.
    pub(in crate::replay) fn ended(&mut self, command_buffer: u64) -> Vec<T> {
        let mut done = Vec::new();
        let mut at = 0;
        while at < self.items.len() {
            let users = &mut self.items[at].1;
            users.retain(|&user| user != command_buffer);
            if users.is_empty() {
                done.push(self.items.swap_remove(at).0);
            } else {
                at += 1;
            }
        }
        done
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

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod tests {
    use super::{Place, Places, ALIGN};
    use crate::replay::rt_replay::{rt_device, ScratchCache};
    use metal2vulkan_runtime as rt;

    /// Forty places of 1 MiB take three pooled blocks; each block goes back to the runtime once
    /// the commits of its places' releases have completed and it holds no place, until the
    /// runtime holds none of them; a later place takes a block again.
    #[test]
    fn an_emptied_pooled_block_goes_back_to_the_runtime() {
        let cache = ScratchCache::new().unwrap();
        let device = rt_device(cache.path());
        let counters = device.counters();
        let class = rt::MemoryClass::Private;
        let unit = class.block_len();
        let mut places = Places::new(device.clone());
        let held = |places: &Places| {
            let gauges = counters.gauges();
            (
                places.blocks.len(),
                places.bytes()[0],
                gauges.private_take_held_bytes,
            )
        };
        let taken: Vec<Place> = (0..40)
            .map(|_| places.take(class, 1 << 20, ALIGN, 0).unwrap())
            .collect();
        assert_eq!(held(&places), (3, 3 * unit, 3 * unit), "three blocks taken");

        // The first block's sixteen places and four of the second's, released by commit 1.
        let mut taken = taken.into_iter();
        for place in taken.by_ref().take(20) {
            places.give(place, 1, Vec::new());
        }
        places.free_completed(0).unwrap();
        assert_eq!(
            held(&places),
            (3, 3 * unit, 3 * unit),
            "commit 1 not completed"
        );
        places.free_completed(1).unwrap();
        assert_eq!(
            held(&places),
            (2, 2 * unit, 2 * unit),
            "the first block back"
        );

        for place in taken {
            places.give(place, 1, Vec::new());
        }
        places.free_completed(1).unwrap();
        assert_eq!(held(&places), (0, 0, 0), "every block back");

        let again = places.take(class, 1 << 20, ALIGN, 1).unwrap();
        assert_eq!(held(&places), (1, unit, unit), "a block taken again");
        places.give(again, 1, Vec::new());
        places.free_completed(1).unwrap();
        assert_eq!(held(&places), (0, 0, 0), "and back");
    }
}
