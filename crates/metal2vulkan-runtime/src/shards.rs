//! Counts that threads add to apart: each thread adds into a shard it owns, on cache lines of its
//! own, and a count is the sum of its shards. Threads that count at once then write apart, where
//! one atomic would take each of their additions in turn, and a thread's add into its own shard is
//! a plain load and store, no locked read-modify-write. The recorder's command counts
//! (`recorder`) and the counts a draw adds on the paths that take no shared lock
//! (`work_counts::DrawCounters`) are kept so.
//!
//! - **Owned while the thread lives.** A thread takes the lowest free one of [`SHARDS`] shards at
//!   its first count and gives it back as it ends (its thread-local's destructor); the next thread
//!   to take it adds on top of what it holds, so a count keeps every ended thread's additions.
//!   Only its owner writes a shard, so its load and store lose no other thread's addition.
//! - **Shared past them.** A thread that finds every shard owned, or counts after its
//!   thread-locals ended (its last destructors'), adds into the one shard no thread owns, by an
//!   atomic add.
//! - **Handed over in order.** The shard's bit is given back with release ordering after its
//!   owner's last store and taken with acquire ordering, so the next owner loads what the last one
//!   stored. A reader sums each shard's count as stored: an addition is in the sum once its store
//!   is seen, and a hand-over moves no count, so no reader sees one lost or twice.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// How many shards threads own; the threads past them share one more ([`SHARED`]).
pub(crate) const SHARDS: usize = 16;

/// The index of the shard no thread owns, which the threads past the owned ones add into.
pub(crate) const SHARED: usize = SHARDS;

/// More threads than shards, so that some share ([`SHARED`]): what the tests count with.
#[cfg(test)]
pub(crate) const SHARDED_THREADS: usize = SHARDS + 3;

/// The owned shards in use: bit `i` while a live thread owns shard `i`.
static OWNED: AtomicU32 = AtomicU32::new(0);

/// A thread's shard: its index once taken, `usize::MAX` before; given back as the thread ends.
struct Taken(Cell<usize>);

impl Drop for Taken {
    fn drop(&mut self) {
        let at = self.0.get();
        if at < SHARDS {
            // After the thread's last store into it: the next owner's take acquires them.
            OWNED.fetch_and(!(1 << at), Ordering::Release);
        }
    }
}

thread_local! {
    /// The shard this thread counts into, taken at its first count.
    static SHARD: Taken = const { Taken(Cell::new(usize::MAX)) };
}

/// The lowest free owned shard, taken; [`SHARED`] where every one is owned.
fn take() -> usize {
    let mut owned = OWNED.load(Ordering::Relaxed);
    loop {
        let free = !owned & ((1 << SHARDS) - 1);
        if free == 0 {
            return SHARED;
        }
        let at = free.trailing_zeros();
        match OWNED.compare_exchange_weak(
            owned,
            owned | (1 << at),
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => return at as usize,
            Err(now) => owned = now,
        }
    }
}

/// The shard the calling thread counts into: the one it owns, taken at its first count, else
/// [`SHARED`].
pub(crate) fn shard() -> usize {
    let taken = SHARD.try_with(|shard| {
        if shard.0.get() == usize::MAX {
            shard.0.set(take());
        }
        shard.0.get()
    });
    taken.unwrap_or(SHARED)
}

/// Add `by` to `count`, a count of shard `at`, which the calling thread counts into
/// ([`shard`]): its owner's load and store, or the shared shard's atomic add.
#[inline]
pub(crate) fn add(at: usize, count: &AtomicU64, by: u64) {
    match at < SHARDS {
        true => count.store(
            count.load(Ordering::Relaxed).wrapping_add(by),
            Ordering::Relaxed,
        ),
        false => {
            count.fetch_add(by, Ordering::Relaxed);
        }
    }
}

/// One shard, on cache lines of its own (128 bytes, the line of Apple's and the pair of lines
/// x86's prefetchers fetch together).
#[derive(Default)]
#[repr(align(128))]
struct Shard<T>(T);

/// Counts only ever added to, which threads add to apart (the module docs): `T` holds a shard's
/// counts, and a count is its shards' sum.
pub(crate) struct Sharded<T>([Shard<T>; SHARDS + 1]);

impl<T: Default> Default for Sharded<T> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| Shard::default()))
    }
}

impl<T> Sharded<T> {
    /// The calling thread's shard, to add its counts into.
    pub(crate) fn mine(&self) -> Mine<'_, T> {
        let at = shard();
        Mine {
            at,
            counts: &self.0[at].0,
        }
    }

    /// The count `count` picks of each shard so far: their sum.
    pub(crate) fn sum(&self, count: impl Fn(&T) -> &AtomicU64) -> u64 {
        (self.0.iter())
            .map(|shard| count(&shard.0).load(Ordering::Relaxed))
            .sum()
    }
}

/// The calling thread's shard of a [`Sharded`], which it adds into.
pub(crate) struct Mine<'a, T> {
    at: usize,
    counts: &'a T,
}

impl<T> Mine<'_, T> {
    /// Add `by` to the count `count` picks ([`add`]).
    #[inline]
    pub(crate) fn add(&self, count: impl Fn(&T) -> &AtomicU64, by: u64) {
        add(self.at, count(self.counts), by);
    }
}
