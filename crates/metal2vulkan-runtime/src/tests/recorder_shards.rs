//! Counts by thread (`shards`): each thread counts into a shard of its own, the threads past the
//! shards sharing them round in turn, so threads that count at once write apart.

use crate::shards::{shard, Sharded, SHARDS};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Every thread takes the next shard as it counts first, and keeps it: as many threads as shards
/// count apart, and the next shares the first's.
#[test]
fn threads_count_into_shards_of_their_own() {
    let first = shard();
    assert_eq!(shard(), first, "a thread keeps its shard");
    let others: Vec<usize> = (1..=SHARDS)
        .map(|_| std::thread::spawn(shard).join().unwrap())
        .collect();
    let apart: BTreeSet<usize> = std::iter::once(first)
        .chain(others[..SHARDS - 1].iter().copied())
        .collect();
    assert_eq!(
        apart.len(),
        SHARDS,
        "as many threads as shards, each its own"
    );
    assert_eq!(
        others[SHARDS - 1],
        first,
        "the next thread shares the first's"
    );
}

/// A sharded count is every thread's additions, summed over the shards they went into.
#[test]
fn a_sharded_count_is_every_thread_s_additions() {
    let count = std::sync::Arc::new(Sharded::<AtomicU64>::default());
    let threads: Vec<_> = (0..SHARDS + 3)
        .map(|at| {
            let count = count.clone();
            std::thread::spawn(move || {
                for _ in 0..100 {
                    count.mine().fetch_add(at as u64 + 1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    let threads = (SHARDS + 3) as u64;
    let sum = count.sum(|count| count);
    assert_eq!(sum, 100 * threads * (threads + 1) / 2);
}
