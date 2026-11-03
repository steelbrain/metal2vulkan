//! Counts that threads add to apart: each thread adds into a shard of its own, on cache lines of
//! its own, and a count is the sum of its shards. Threads that count at once then write apart,
//! where one atomic would take each of their additions in turn. The recorder's command counts
//! (`recorder`) and the counts a draw adds on the paths that take no shared lock
//! (`work_counts::DrawCounters`) are kept so.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// How many shards a count keeps. More threads than shards share them, round the shards in turn.
pub(crate) const SHARDS: usize = 16;

/// How many threads took a shard: the next takes the shard of this count.
static SHARDS_TAKEN: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The shard this thread counts into, taken at its first count; `usize::MAX` before.
    static SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// The shard the calling thread counts into: its own, taken once.
pub(crate) fn shard() -> usize {
    let taken = SHARD.try_with(|shard| {
        if shard.get() == usize::MAX {
            shard.set(SHARDS_TAKEN.fetch_add(1, Ordering::Relaxed) % SHARDS);
        }
        shard.get()
    });
    // A thread past its thread-locals' end (its last destructors' counts) counts into shard 0.
    taken.unwrap_or(0)
}

/// One thread's shard, on cache lines of its own (128 bytes, the line of Apple's and the pair of
/// lines x86's prefetchers fetch together).
#[derive(Default)]
#[repr(align(128))]
struct Shard<T>(T);

/// Counts only ever added to, which threads add to apart (the module docs): `T` holds a shard's
/// counts, and a count is its shards' sum.
pub(crate) struct Sharded<T>([Shard<T>; SHARDS]);

impl<T: Default> Default for Sharded<T> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| Shard::default()))
    }
}

impl<T> Sharded<T> {
    /// The calling thread's shard, to add its counts into.
    pub(crate) fn mine(&self) -> &T {
        &self.0[shard()].0
    }

    /// The count `count` picks of each shard so far: their sum.
    pub(crate) fn sum(&self, count: impl Fn(&T) -> &AtomicU64) -> u64 {
        self.0
            .iter()
            .map(|shard| count(&shard.0).load(Ordering::Relaxed))
            .sum()
    }
}
