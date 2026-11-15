//! An encoder's holds (`retained.rs`): a shared object among the last ones held by reference, or
//! among those its command buffer held lately, is held once, and lives until the holds go.

use crate::retained::Retained;
use std::sync::Arc;

/// The same three objects held four times over are three holds; eight others push the first out
/// of those looked among, so it is held again; and each object lives until the entry goes.
#[test]
fn an_object_held_lately_is_held_once_until_the_holds_go() {
    let objects: Vec<Arc<u64>> = (0..9).map(Arc::new).collect();
    let mut retained = Retained::default();
    for _ in 0..4 {
        for object in &objects[..3] {
            retained.shared(object.clone());
        }
    }
    assert_eq!(retained.len(), 3);
    assert!(objects[..3]
        .iter()
        .all(|object| Arc::strong_count(object) == 2));
    for object in &objects[1..] {
        retained.shared(object.clone());
    }
    assert_eq!(retained.len(), 9);
    retained.shared(objects[0].clone());
    assert_eq!(retained.len(), 10);
    assert_eq!(Arc::strong_count(&objects[0]), 3);
    let entry = retained.into_entry();
    assert!(objects.iter().all(|object| Arc::strong_count(object) > 1));
    drop(entry);
    assert!(objects.iter().all(|object| Arc::strong_count(object) == 1));
}

/// An encoder seeded with what the encoder before it held lately holds none of those objects
/// again, and hands on the latest first: those it held, then its seed, 16 at most.
#[test]
fn an_object_the_command_buffer_held_lately_is_not_held_again() {
    let objects: Vec<Arc<u64>> = (0..17).map(Arc::new).collect();
    let mut first = Retained::default();
    for object in &objects[..8] {
        first.shared(object.clone());
    }
    assert_eq!(first.len(), 8);
    let mut second = Retained::seeded(first.lately());
    for object in &objects[..8] {
        second.shared(object.clone());
    }
    assert_eq!(second.len(), 0);
    assert!(second.into_entry().is_none());
    let mut second = Retained::seeded(first.lately());
    for object in &objects[8..16] {
        second.shared(object.clone());
    }
    assert_eq!(second.len(), 8);
    // The seventeenth pushes the first held out of the sixteen handed on.
    let mut third = Retained::seeded(second.lately());
    third.shared(objects[16].clone());
    let mut fourth = Retained::seeded(third.lately());
    for object in &objects[1..] {
        fourth.shared(object.clone());
    }
    assert_eq!(fourth.len(), 0);
    fourth.shared(objects[0].clone());
    assert_eq!(fourth.len(), 1);
    assert_eq!(Arc::strong_count(&objects[0]), 3);
    assert!(objects[1..]
        .iter()
        .all(|object| Arc::strong_count(object) == 2));
}

/// A hold by the caller's reference (`Retained::shared_of`) takes another reference only where the
/// object is not held lately: the same three objects held four times over are three holds, each
/// counted once; one held lately by either form is not held again by the other; and eight others
/// push the first out of those looked among, so it is held again, each living until the entry
/// goes.
#[test]
fn a_hold_by_reference_takes_another_only_where_none_is_held_lately() {
    let objects: Vec<Arc<u64>> = (0..9).map(Arc::new).collect();
    let mut retained = Retained::default();
    for _ in 0..4 {
        for object in &objects[..3] {
            retained.shared_of(object);
        }
    }
    assert_eq!(retained.len(), 3);
    assert!(objects[..3]
        .iter()
        .all(|object| Arc::strong_count(object) == 2));
    retained.shared(objects[3].clone());
    retained.shared_of(&objects[3]);
    retained.shared_of(&objects[4]);
    retained.shared(objects[4].clone());
    assert_eq!(retained.len(), 5);
    assert!(objects[3..5]
        .iter()
        .all(|object| Arc::strong_count(object) == 2));
    for object in &objects[5..] {
        retained.shared_of(object);
    }
    assert_eq!(retained.len(), 9);
    retained.shared_of(&objects[0]);
    assert_eq!(retained.len(), 10);
    assert_eq!(Arc::strong_count(&objects[0]), 3);
    let entry = retained.into_entry();
    assert!(objects.iter().all(|object| Arc::strong_count(object) > 1));
    drop(entry);
    assert!(objects.iter().all(|object| Arc::strong_count(object) == 1));
}
