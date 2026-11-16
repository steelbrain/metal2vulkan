//! A group's bound on its batches' expected runs (`command_buffer/group.rs`, `GROUP_RUN`): a
//! batch joins only where its queue's expected run is known, and the group's expected runs,
//! summed with its own, stay within 16 ms.

use crate::command_buffer::{within_run, GROUP_BATCHES, GROUP_RUN};

/// A millisecond, in nanoseconds.
const MS: u64 = 1_000_000;

#[test]
fn a_batch_joins_only_with_a_known_run_within_the_bound() {
    assert_eq!(GROUP_RUN, 16 * MS);
    // A batch of a queue none of whose batches was seen complete: none.
    assert!(!within_run(0, None));
    assert!(!within_run(4 * MS, None));
    // The first batch of a group: within the bound alone, or not.
    assert!(within_run(0, Some(16 * MS)));
    assert!(!within_run(0, Some(16 * MS + 1)));
    // A full group, a quarter of the bound each.
    let quarter = GROUP_RUN / GROUP_BATCHES as u64;
    let mut held = 0;
    for _ in 0..GROUP_BATCHES {
        assert!(within_run(held, Some(quarter)), "{held} held");
        held += quarter;
    }
    assert!(!within_run(held, Some(1)));
    // 12 ms held: a batch of 5 ms waits for the next group, one of 4 ms joins.
    assert!(!within_run(12 * MS, Some(5 * MS)));
    assert!(within_run(12 * MS, Some(4 * MS)));
    // No overflow.
    assert!(!within_run(u64::MAX, Some(1)));
}
