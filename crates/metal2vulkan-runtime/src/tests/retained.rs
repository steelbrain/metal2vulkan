//! An encoder's holds (`retained.rs`): a shared object among the last ones held by reference is
//! held once, and lives until the holds go.

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
