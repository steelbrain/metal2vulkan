//! The hazard sets: which accesses conflict, on what (a write, or a read alone), what a resource's
//! own id finds beside its shared one, and that a barrier starts them again.

use crate::hazard::{Access, Dependency, Hazards};

#[test]
fn a_read_or_write_after_a_write_conflicts_and_a_write_after_a_read_too() {
    let mut sets = Hazards::default();
    sets.mark(3, Access::Write);
    for access in [Access::Read, Access::Write, Access::ReadWrite] {
        let found = sets.dependency(3, access);
        assert_eq!(found, Some(Dependency::Write), "{access:?} after a write");
    }
    assert_eq!(
        sets.dependency(4, Access::ReadWrite),
        None,
        "another resource"
    );

    let mut sets = Hazards::default();
    sets.mark(7, Access::Read);
    assert_eq!(
        sets.dependency(7, Access::Read),
        None,
        "a read after a read"
    );
    assert_eq!(
        sets.dependency(7, Access::Write),
        Some(Dependency::Read),
        "a write after a read"
    );
    assert_eq!(
        sets.dependency(7, Access::ReadWrite),
        Some(Dependency::Read),
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
        assert_eq!(
            sets.dependency(id, access),
            None,
            "{id} {access:?} after the barrier"
        );
    }
}

#[test]
fn own_ids_place_nothing_and_tell_a_resource_from_its_heap() {
    // Two buffers of heap 5, with own ids 1 and 2: the first written.
    let mut sets = Hazards::default();
    sets.mark(5, Access::Write);
    sets.mark_own(1, Access::Write);
    for (id, found) in [(1, Some(Dependency::Write)), (2, None)] {
        let shared = sets.dependency(5, Access::Read);
        assert_eq!(
            shared,
            Some(Dependency::Write),
            "own id {id}, by the heap's"
        );
        assert_eq!(sets.own_dependency(id, Access::Read), found, "own id {id}");
    }
    // The own marks are in no set of their own ids.
    assert_eq!(
        sets.dependency(1, Access::ReadWrite),
        None,
        "own id 1 as an id"
    );
    sets.mark_own(2, Access::Read);
    let read = sets.own_dependency(2, Access::Write);
    assert_eq!(
        read,
        Some(Dependency::Read),
        "a write after a read, by its own id"
    );
    sets.clear();
    assert_eq!(
        sets.own_dependency(1, Access::Read),
        None,
        "after the barrier"
    );
}

#[test]
fn accesses_outside_every_resource_count_as_touched_and_writes_as_written() {
    let mut sets = Hazards::default();
    sets.mark_wild_read();
    assert!(sets.touched() && !sets.wrote());
    sets.mark_wild_write();
    assert!(sets.wrote());
}
