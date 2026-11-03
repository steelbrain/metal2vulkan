//! The recorder's counts by thread (`recorder`): each thread counts its commands into a shard of
//! its own, the threads past the shards sharing them round in turn, so threads that record at
//! once write apart.

use crate::recorder::{shard, SHARDS};
use std::collections::BTreeSet;

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
