//! Counts by thread and by open encoder (`shards`): each thread counts into a shard it owns while
//! it lives, and each open encoder into a tally it owns while it is open, those past them into
//! one they share, so adders that count at once write apart; a shard given back keeps its counts
//! for its next owner to add on top of, and a read sums the shards of every adder, those still
//! counting too.

use crate::shards::{
    held, shard, Sharded, Tallies, Tally, SHARDED_THREADS, SHARDS, SHARED, TALLIES,
};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Barrier;

/// A recorder's thunk counts into the shard its thread holds, taking none (`shards::held`): a
/// thread that took none holds the shared one, and holding takes none; a thread that took its own
/// holds that.
#[test]
fn a_thread_holds_the_shared_shard_until_it_takes_its_own() {
    std::thread::spawn(|| {
        assert_eq!(
            held(),
            SHARED,
            "a thread that took none holds the shared shard"
        );
        assert_eq!(held(), SHARED, "holding takes none");
        let at = shard();
        assert!(at < SHARDS, "the thread takes one of its own");
        assert_eq!(held(), at, "a thread that took its own holds it");
    })
    .join()
    .unwrap();
}

/// Threads alive at once own shards apart, a thread keeping its own; the thread past them shares
/// the one no thread owns; and a thread that starts once they ended owns a shard again.
#[test]
fn threads_own_their_shards_while_they_live() {
    let first = shard();
    assert!(first < SHARDS, "the first thread owns a shard");
    assert_eq!(shard(), first, "a thread keeps its shard");
    // This thread and SHARDS more, each taking its shard before any ends.
    let barrier = Barrier::new(SHARDS);
    let taken: Vec<usize> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..SHARDS)
            .map(|_| {
                scope.spawn(|| {
                    let at = shard();
                    barrier.wait();
                    at
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    });
    let owned: BTreeSet<usize> = taken.iter().copied().filter(|&at| at < SHARDS).collect();
    assert_eq!(
        owned.len(),
        SHARDS - 1,
        "each live thread owns its own: {taken:?}"
    );
    assert!(
        !owned.contains(&first),
        "none owns this thread's: {taken:?}"
    );
    let shared = taken.iter().filter(|&&at| at == SHARED).count();
    assert_eq!(shared, 1, "the thread past them shares: {taken:?}");
    let next = std::thread::spawn(shard).join().unwrap();
    assert!(next < SHARDS, "a shard given back is owned again");
    assert_ne!(next, first, "but not this live thread's");
}

/// A sharded count is every thread's additions, summed over the shards they went into, whether
/// they owned theirs or shared one: threads alive at once, those past the shards adding into the
/// shared one together.
#[test]
fn a_sharded_count_is_every_thread_s_additions() {
    const ADDS: u64 = 100_000;
    let count = Sharded::<AtomicU64>::default();
    let barrier = Barrier::new(SHARDED_THREADS);
    std::thread::scope(|scope| {
        for at in 0..SHARDED_THREADS {
            let (count, barrier) = (&count, &barrier);
            scope.spawn(move || {
                shard();
                barrier.wait();
                for _ in 0..ADDS {
                    count.mine().add(|count| count, at as u64 + 1);
                }
            });
        }
    });
    let threads = SHARDED_THREADS as u64;
    let sum = count.sum(|count| count);
    assert_eq!(sum, ADDS * threads * (threads + 1) / 2);
}

/// A reader summing while threads count, end and give their shards back to the threads after
/// them sees each sum at least the last one it saw and at most what was added, and the sum once
/// they all ended is every addition: a hand-over loses no count and adds none twice.
#[test]
fn a_reader_summing_through_hand_overs_sees_every_count_once() {
    const ROUNDS: u64 = 40;
    const ADDS: u64 = 500;
    let count = Sharded::<AtomicU64>::default();
    let done = AtomicBool::new(false);
    let total = ROUNDS * SHARDED_THREADS as u64 * ADDS;
    std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            let (mut last, mut sums) = (0, 0u64);
            while !done.load(Ordering::Acquire) {
                let sum = count.sum(|count| count);
                assert!(sum >= last, "a sum of {sum} after one of {last}");
                assert!(sum <= total, "a sum of {sum} past the {total} added");
                (last, sums) = (sum, sums + 1);
            }
            sums
        });
        for _ in 0..ROUNDS {
            std::thread::scope(|round| {
                for _ in 0..SHARDED_THREADS {
                    round.spawn(|| {
                        for _ in 0..ADDS {
                            count.mine().add(|count| count, 1);
                        }
                    });
                }
            });
        }
        done.store(true, Ordering::Release);
        assert!(
            reader.join().unwrap() > 0,
            "the reader summed while they counted"
        );
    });
    assert_eq!(count.sum(|count| count), total);
}

/// Encoders open at once own tallies apart; the encoder past them shares the one no encoder owns;
/// and a tally given back is owned again.
#[test]
fn encoders_open_at_once_own_their_tallies_apart() {
    let open: Vec<Tally> = (0..TALLIES + 1).map(|_| Tally::take()).collect();
    let owned: BTreeSet<usize> = open
        .iter()
        .map(Tally::at)
        .filter(|&at| at < TALLIES)
        .collect();
    assert_eq!(owned.len(), TALLIES, "each open encoder owns its own");
    assert_eq!(open[TALLIES].at(), TALLIES, "the encoder past them shares");
    let given = open[3].at();
    drop(open);
    let again = Tally::take();
    assert!(again.at() < TALLIES, "a tally given back is owned again");
    assert!(
        again.at() <= given,
        "the lowest free one: {} past {given}",
        again.at()
    );
}

/// A tallied count is every encoder's additions, summed over the tallies they went into, whether
/// they owned theirs or shared one: read while every encoder is still open, and once they all
/// gave theirs back.
#[test]
fn a_tallied_count_is_every_encoder_s_additions_while_they_are_open() {
    const ADDS: u64 = 100_000;
    const ENCODERS: usize = TALLIES + 3;
    let count = Tallies::<AtomicU64>::default();
    let (counted, read) = (Barrier::new(ENCODERS + 1), Barrier::new(ENCODERS + 1));
    let encoders = ENCODERS as u64;
    let total = ADDS * encoders * (encoders + 1) / 2;
    // Read while they are open, and checked once they all went on: a failed check never leaves
    // them waiting.
    let open = std::thread::scope(|scope| {
        for at in 0..ENCODERS {
            let (count, counted, read) = (&count, &counted, &read);
            scope.spawn(move || {
                let tally = Tally::take();
                counted.wait();
                for _ in 0..ADDS {
                    count.of(&tally).add(|count| count, at as u64 + 1);
                }
                counted.wait();
                read.wait();
                drop(tally);
            });
        }
        counted.wait();
        counted.wait();
        let open = count.sum(|count| count);
        read.wait();
        open
    });
    assert_eq!(open, total, "read while they are open");
    assert_eq!(
        count.sum(|count| count),
        total,
        "read once they gave theirs back"
    );
}

/// A reader summing while encoders count, end and give their tallies back to the encoders after
/// them sees each sum at least the last one it saw and at most what was added, and the sum once
/// they all ended is every addition: a hand-over loses no count and adds none twice.
#[test]
fn a_reader_summing_through_tally_hand_overs_sees_every_count_once() {
    const ROUNDS: u64 = 40;
    const ADDS: u64 = 500;
    const ENCODERS: usize = TALLIES + 3;
    let count = Tallies::<AtomicU64>::default();
    let done = AtomicBool::new(false);
    let total = ROUNDS * ENCODERS as u64 * ADDS;
    std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            let (mut last, mut sums) = (0, 0u64);
            while !done.load(Ordering::Acquire) {
                let sum = count.sum(|count| count);
                assert!(sum >= last, "a sum of {sum} after one of {last}");
                assert!(sum <= total, "a sum of {sum} past the {total} added");
                (last, sums) = (sum, sums + 1);
            }
            sums
        });
        for _ in 0..ROUNDS {
            std::thread::scope(|round| {
                for _ in 0..ENCODERS {
                    round.spawn(|| {
                        let tally = Tally::take();
                        for _ in 0..ADDS {
                            count.of(&tally).add(|count| count, 1);
                        }
                    });
                }
            });
        }
        done.store(true, Ordering::Release);
        assert!(
            reader.join().unwrap() > 0,
            "the reader summed while they counted"
        );
    });
    assert_eq!(count.sum(|count| count), total);
}
