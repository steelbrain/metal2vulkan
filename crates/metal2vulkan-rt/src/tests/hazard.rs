//! The hazard sets: which accesses conflict, and that a barrier starts them again.

use crate::hazard::{Access, Hazards};

#[test]
fn a_read_or_write_after_a_write_conflicts_and_a_write_after_a_read_too() {
    let mut sets = Hazards::default();
    sets.mark(3, Access::Write);
    for access in [Access::Read, Access::Write, Access::ReadWrite] {
        assert!(sets.conflicts(3, access), "{access:?} after a write");
    }
    assert!(!sets.conflicts(4, Access::ReadWrite), "another resource");

    let mut sets = Hazards::default();
    sets.mark(7, Access::Read);
    assert!(!sets.conflicts(7, Access::Read), "a read after a read");
    assert!(sets.conflicts(7, Access::Write), "a write after a read");
    assert!(
        sets.conflicts(7, Access::ReadWrite),
        "a read-write after a read"
    );
}

#[test]
fn a_barrier_starts_the_sets_again() {
    let mut sets = Hazards::default();
    sets.mark(0, Access::ReadWrite);
    sets.mark(9, Access::Read);
    assert!(sets.wrote() && sets.touched());
    sets.clear();
    assert!(!sets.wrote() && !sets.touched());
    for (id, access) in [(0, Access::ReadWrite), (9, Access::Write)] {
        assert!(
            !sets.conflicts(id, access),
            "{id} {access:?} after the barrier"
        );
    }
}

#[test]
fn accesses_outside_every_resource_count_as_touched_and_writes_as_written() {
    let mut sets = Hazards::default();
    sets.mark_wild_read();
    assert!(sets.touched() && !sets.wrote());
    sets.mark_wild_write();
    assert!(sets.wrote());
}
