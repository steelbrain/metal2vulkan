//! A command queue's slots (`queue.rs`): one per uncompleted command buffer, at most the queue's
//! `maxCommandBufferCount`, each counted open until its command buffer is committed, and among the
//! device's uncommitted command buffers until its commit handed its batches to the scheduler.

use crate::queue::{Held, Slots, Uncommitted};
use std::sync::Arc;

#[test]
fn a_queue_hands_out_its_slots_up_to_its_count_and_takes_each_back_when_dropped() {
    let uncommitted = Arc::new(Uncommitted::default());
    let slots = Slots::new(2, Arc::clone(&uncommitted));
    let others = Slots::new(1, Arc::clone(&uncommitted));
    let mut held = slots.lock();
    let first = slots.take(&mut held).expect("a first slot");
    let mut second = slots.take(&mut held).expect("a second slot");
    assert!(slots.take(&mut held).is_none(), "a third slot of two");
    assert_eq!(*held, Held { all: 2, open: 2 });
    drop(held);
    let other = others
        .take(&mut others.lock())
        .expect("another queue's slot");
    assert_eq!(uncommitted.count(), 3);
    // Committed, a slot is no longer open, and only its drop gives it back; it counts among the
    // uncommitted until its batches are handed to the scheduler.
    second.committed();
    second.committed();
    assert_eq!(*slots.lock(), Held { all: 2, open: 1 });
    assert_eq!(uncommitted.count(), 3);
    second.handed();
    second.handed();
    assert_eq!(uncommitted.count(), 2);
    drop(second);
    assert_eq!(uncommitted.count(), 2);
    assert_eq!(*slots.lock(), Held { all: 1, open: 1 });
    // An uncommitted command buffer's slot comes back when it is dropped.
    drop(first);
    assert_eq!(*slots.lock(), Held { all: 0, open: 0 });
    assert_eq!(uncommitted.count(), 1);
    // One committed and dropped before its batches were handed counts no more once dropped.
    let mut committed = other;
    committed.committed();
    assert_eq!(uncommitted.count(), 1);
    drop(committed);
    assert_eq!(uncommitted.count(), 0);
    let again: Vec<_> = {
        let mut held = slots.lock();
        (0..3).filter_map(|_| slots.take(&mut held)).collect()
    };
    assert_eq!(again.len(), 2);
}
