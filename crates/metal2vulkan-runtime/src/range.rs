//! A first-fit allocator of aligned ranges with coalescing frees: the one mechanism under the
//! memory sub-allocator's blocks, the Metal address space and automatic heaps, whose
//! placement it is exactly. Pure, so the CPU tests cover it.

use std::collections::BTreeMap;

/// The free ranges of `[base, end)`, by start. Every allocation is a range the caller frees with
/// the same start and size.
#[derive(Debug)]
pub(super) struct RangeAllocator {
    base: u64,
    end: u64,
    /// Disjoint, non-adjacent free ranges: start to length.
    free: BTreeMap<u64, u64>,
}

impl RangeAllocator {
    /// Everything of `[base, end)` free.
    pub(super) fn new(base: u64, end: u64) -> RangeAllocator {
        let mut free = BTreeMap::new();
        if end > base {
            free.insert(base, end - base);
        }
        RangeAllocator { base, end, free }
    }

    /// The lowest start of a free range of `size` bytes aligned to `align` (a power of two), now
    /// allocated; `None` when no free range holds it. A `size` of 0 allocates nothing.
    pub(super) fn allocate(&mut self, size: u64, align: u64) -> Option<u64> {
        debug_assert!(align.is_power_of_two());
        if size == 0 {
            return None;
        }
        let (start, len, at) = self.free.iter().find_map(|(&start, &len)| {
            let at = start.checked_next_multiple_of(align)?;
            let stop = at.checked_add(size)?;
            (stop <= start + len).then_some((start, len, at))
        })?;
        self.free.remove(&start);
        if at > start {
            self.free.insert(start, at - start);
        }
        let stop = at + size;
        if stop < start + len {
            self.free.insert(stop, start + len - stop);
        }
        Some(at)
    }

    /// `[start, start + size)`, now allocated, when every byte of it is free (`size` above 0);
    /// otherwise nothing changes and the answer is the first run of allocated bytes it meets, as
    /// its start and length (a run bounded by free ranges or the ends, not one allocation).
    pub(super) fn allocate_at(&mut self, start: u64, size: u64) -> Result<(), (u64, u64)> {
        debug_assert!(size > 0 && start >= self.base && start + size <= self.end);
        let containing = self
            .free
            .range(..=start)
            .next_back()
            .map(|(&at, &len)| (at, len))
            .filter(|&(at, len)| start < at + len);
        let Some((at, len)) = containing else {
            // `start` is allocated: its run starts where the free range before it ends.
            let from = self
                .free
                .range(..start)
                .next_back()
                .map_or(self.base, |(&at, &len)| at + len);
            return Err((from, self.run_end(start) - from));
        };
        if start + size > at + len {
            let from = at + len;
            return Err((from, self.run_end(from) - from));
        }
        self.free.remove(&at);
        if start > at {
            self.free.insert(at, start - at);
        }
        if start + size < at + len {
            self.free.insert(start + size, at + len - (start + size));
        }
        Ok(())
    }

    /// The end of the run of allocated bytes holding `at`: the next free range's start, or the
    /// end.
    fn run_end(&self, at: u64) -> u64 {
        self.free
            .range(at..)
            .next()
            .map_or(self.end, |(&start, _)| start)
    }

    /// Return `[start, start + size)`, which [`RangeAllocator::allocate`] or
    /// [`RangeAllocator::allocate_at`] gave, merging it with the free ranges beside it.
    pub(super) fn free(&mut self, start: u64, size: u64) {
        debug_assert!(start >= self.base && start + size <= self.end);
        let mut start = start;
        let mut size = size;
        if let Some((&before, &len)) = self.free.range(..start).next_back() {
            debug_assert!(before + len <= start, "a range freed twice");
            if before + len == start {
                self.free.remove(&before);
                start = before;
                size += len;
            }
        }
        if let Some(&after) = self.free.get(&(start + size)) {
            self.free.remove(&(start + size));
            size += after;
        }
        self.free.insert(start, size);
    }

    /// Whether nothing is allocated.
    pub(super) fn is_empty(&self) -> bool {
        self.free.len() == 1 && self.free.get(&self.base) == Some(&(self.end - self.base))
    }

    /// The bytes allocated.
    pub(super) fn used(&self) -> u64 {
        (self.end - self.base) - self.free.values().sum::<u64>()
    }

    /// How many free ranges it holds: what one [`RangeAllocator::longest`] visits, which is why
    /// a pooled block derives its longest free range once (`memory.rs`'s `PoolBlock`).
    pub(super) fn free_ranges(&self) -> u64 {
        self.free.len() as u64
    }

    /// The most bytes one allocation aligned to `align` (a power of two) could take now: the
    /// longest free range, less what aligning its start skips. It visits every free range
    /// ([`RangeAllocator::free_ranges`]).
    pub(super) fn longest(&self, align: u64) -> u64 {
        debug_assert!(align.is_power_of_two());
        self.free
            .iter()
            .filter_map(|(&start, &len)| {
                let at = start.checked_next_multiple_of(align)?;
                (start + len).checked_sub(at)
            })
            .max()
            .unwrap_or(0)
    }
}
