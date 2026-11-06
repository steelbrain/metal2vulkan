//! Counts that their adders add to apart: each adder adds into a shard it owns, on cache lines of
//! its own, and a count is the sum of its shards. Adders that count at once then write apart,
//! where one atomic would take each of their additions in turn, and an add into an owned shard is
//! a plain load and store, no locked read-modify-write. Two kinds of adder own shards:
//!
//! - **A thread, while it lives** ([`Sharded`]): the recorder's command counts (`recorder`), and
//!   the counts a command adds where no encoder is at hand (`work_counts::Counters::thread`). A
//!   thread takes the lowest free one of [`SHARDS`] shards at its first count and gives it back as
//!   it ends (its thread-local's destructor).
//! - **An encoder, while it is open** ([`Tallies`]): the counts its commands add as it records
//!   them (`work_counts::Counting`). An encoder takes the lowest free one of [`TALLIES`] shards as
//!   it opens ([`Tally::take`]) and gives it back as it ends or goes (the tally's drop). It adds
//!   into it on whichever thread records it, one thread at a time (an encoder is recorded by one
//!   thread at once, and handed between threads in order), so how many threads ever counted does
//!   not bound the owned shards, only how many encoders are open at once.
//!
//! The next adder to take a shard given back adds on top of what it holds, so a count keeps every
//! ended adder's additions. Only its owner writes a shard, so its load and store lose no other
//! adder's addition.
//!
//! - **Shared past them.** An adder that finds every shard of its kind owned, or a thread that
//!   counts after its thread-locals ended (its last destructors'), adds into the one shard of its
//!   kind no adder owns, by an atomic add.
//! - **Held without a call where a call costs.** A recorder's command thunk counts into the shard
//!   its thread holds ([`held`]) and takes none: it makes no call, so it keeps the command's
//!   arguments in their registers to hand on. A thread takes its shard at its first other count
//!   ([`shard`]: an encoder's opening on it takes it, `encoder.rs`); its commands before then
//!   count into the shared one.
//! - **Handed over in order.** A shard's bit is given back with release ordering after its
//!   owner's last store and taken with acquire ordering, so the next owner loads what the last one
//!   stored. A reader sums each shard's count as stored: an addition is in the sum once its store
//!   is seen, and a hand-over moves no count, so no reader sees one lost or twice; a reader sums
//!   the shards of encoders still open too, so a count is whole whenever it is read.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// How many shards threads own; the threads past them share one more ([`SHARED`]).
pub(crate) const SHARDS: usize = 64;

/// The index of the shard no thread owns, which the threads past the owned ones add into.
pub(crate) const SHARED: usize = SHARDS;

/// How many shards open encoders own; the encoders past them share one more.
pub(crate) const TALLIES: usize = 64;

/// More threads than shards, so that some share ([`SHARED`]): what the tests count with.
#[cfg(test)]
pub(crate) const SHARDED_THREADS: usize = SHARDS + 3;

/// One kind of adder's shards in use: bit `i` while an adder owns shard `i` of the `size` it may
/// own, at most 64.
struct Owners {
    owned: AtomicU64,
    size: usize,
}

/// The shards threads own.
static THREADS: Owners = Owners::new(SHARDS);

/// The shards open encoders own.
static ENCODERS: Owners = Owners::new(TALLIES);

impl Owners {
    const fn new(size: usize) -> Owners {
        Owners {
            owned: AtomicU64::new(0),
            size,
        }
    }

    /// The lowest free owned shard, taken; `size`, the shared one, where every one is owned.
    fn take(&self) -> usize {
        let all = u64::MAX >> (64 - self.size);
        let mut owned = self.owned.load(Ordering::Relaxed);
        loop {
            let free = !owned & all;
            if free == 0 {
                return self.size;
            }
            let at = free.trailing_zeros();
            match self.owned.compare_exchange_weak(
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

    /// Give back shard `at`, where it is an owned one.
    fn give_back(&self, at: usize) {
        if at < self.size {
            self.owned.fetch_and(!(1 << at), Ordering::Release);
        }
    }
}

thread_local! {
    /// The shard this thread counts into: `usize::MAX` before its first count, [`SHARED`] once
    /// it gave its own back. No destructor of its own, so reading it checks no state: what
    /// [`shard`] reads at each count.
    static SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
    /// What gives the thread's shard back as it ends, touched once, as it takes one.
    static OWNER: Owner = const { Owner };
}

/// The thread's hold on the shard [`SHARD`] names (the module docs).
struct Owner;

impl Drop for Owner {
    fn drop(&mut self) {
        // After the thread's last store into it: its later counts (its last destructors') go to
        // the shared shard, and the next owner's take acquires its stores.
        let at = SHARD.with(|shard| shard.replace(SHARED));
        THREADS.give_back(at);
    }
}

/// The shard the calling thread counts into: the one it owns, taken at its first count
/// ([`first_shard`]), else [`SHARED`].
#[inline]
pub(crate) fn shard() -> usize {
    match SHARD.with(Cell::get) {
        usize::MAX => first_shard(),
        at => at,
    }
}

/// The shard the calling thread holds, taking none: [`SHARED`] before it takes one through
/// [`shard`].
#[inline]
pub(crate) fn held() -> usize {
    match SHARD.with(Cell::get) {
        usize::MAX => SHARED,
        at => at,
    }
}

/// A thread's first count: a shard taken, held by its [`OWNER`]; the shared one where none is
/// free, or the thread is past its thread-locals' end (its owner could give none back).
#[cold]
#[inline(never)]
fn first_shard() -> usize {
    let mut at = THREADS.take();
    if OWNER.try_with(|_| ()).is_err() {
        THREADS.give_back(at);
        at = SHARED;
    }
    SHARD.with(|shard| shard.set(at));
    at
}

/// Add `by` to `count`, a count of shard `at` of a thread's, which the calling thread counts into
/// ([`shard`]): its owner's load and store, or the shared shard's atomic add.
#[inline]
pub(crate) fn add(at: usize, count: &AtomicU64, by: u64) {
    add_into(at < SHARDS, count, by);
}

/// Add `by` to `count`: by its owner's load and store where `owned`, else by an atomic add.
#[inline]
fn add_into(owned: bool, count: &AtomicU64, by: u64) {
    match owned {
        true => count.store(
            count.load(Ordering::Relaxed).wrapping_add(by),
            Ordering::Relaxed,
        ),
        false => {
            count.fetch_add(by, Ordering::Relaxed);
        }
    }
}

/// An open encoder's hold on the shard of [`TALLIES`] its commands count into (the module docs):
/// taken as it opens, given back as it drops. Not `Clone`: one holder adds into it.
pub(crate) struct Tally(usize);

impl Tally {
    /// The lowest free shard of the encoders', else the one they share; and the calling
    /// thread's own, where it holds none yet ([`shard`]), for the recorder's counts of the
    /// commands it records.
    pub(crate) fn take() -> Tally {
        shard();
        Tally(ENCODERS.take())
    }
}

/// The shard a tally holds: what the tests look at.
#[cfg(test)]
impl Tally {
    pub(crate) fn at(&self) -> usize {
        self.0
    }
}

impl Drop for Tally {
    fn drop(&mut self) {
        // After the encoder's last store into it: the next owner's take acquires its stores.
        ENCODERS.give_back(self.0);
    }
}

/// One shard, on cache lines of its own (128 bytes, the line of Apple's and the pair of lines
/// x86's prefetchers fetch together).
#[derive(Default)]
#[repr(align(128))]
struct Shard<T>(T);

/// `N` shards of counts only ever added to, the last the one no adder owns.
struct Shards<T, const N: usize>([Shard<T>; N]);

impl<T: Default, const N: usize> Default for Shards<T, N> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| Shard::default()))
    }
}

impl<T, const N: usize> Shards<T, N> {
    /// Shard `at` to add into: owned where it is not the last.
    fn at(&self, at: usize) -> Mine<'_, T> {
        Mine {
            owned: at < N - 1,
            counts: &self.0[at].0,
        }
    }

    /// The count `count` picks of each shard so far: their sum.
    fn sum(&self, count: impl Fn(&T) -> &AtomicU64) -> u64 {
        (self.0.iter())
            .map(|shard| count(&shard.0).load(Ordering::Relaxed))
            .sum()
    }
}

/// Counts only ever added to, which threads add to apart (the module docs): `T` holds a shard's
/// counts, and a count is its shards' sum.
#[derive(Default)]
pub(crate) struct Sharded<T>(Shards<T, { SHARDS + 1 }>);

impl<T> Sharded<T> {
    /// The calling thread's shard, to add its counts into.
    #[inline]
    pub(crate) fn mine(&self) -> Mine<'_, T> {
        self.0.at(shard())
    }

    /// The count `count` picks of each shard so far: their sum.
    pub(crate) fn sum(&self, count: impl Fn(&T) -> &AtomicU64) -> u64 {
        self.0.sum(count)
    }
}

/// Counts only ever added to, which open encoders add to apart (the module docs): `T` holds a
/// shard's counts, and a count is its shards' sum.
#[derive(Default)]
pub(crate) struct Tallies<T>(Shards<T, { TALLIES + 1 }>);

impl<T> Tallies<T> {
    /// The shard `tally` holds, to add its encoder's counts into.
    #[inline]
    pub(crate) fn of(&self, tally: &Tally) -> Mine<'_, T> {
        self.0.at(tally.0)
    }

    /// The count `count` picks of each shard so far, open encoders' too: their sum.
    pub(crate) fn sum(&self, count: impl Fn(&T) -> &AtomicU64) -> u64 {
        self.0.sum(count)
    }
}

/// The shard of a [`Sharded`] or [`Tallies`] its adder adds into.
pub(crate) struct Mine<'a, T> {
    owned: bool,
    counts: &'a T,
}

impl<T> Clone for Mine<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Mine<'_, T> {}

impl<T> Mine<'_, T> {
    /// Add `by` to the count `count` picks: the owner's load and store, or the shared shard's
    /// atomic add.
    #[inline]
    pub(crate) fn add(&self, count: impl Fn(&T) -> &AtomicU64, by: u64) {
        add_into(self.owned, count(self.counts), by);
    }
}
