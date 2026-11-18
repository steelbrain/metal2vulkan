//! What a recording keeps (`crate::kept`): each object once, its first use the one reference
//! taken; a later use and every borrow of it change no count; clearing releases every object.

use crate::kept::{Handle, Kept};
use std::mem::ManuallyDrop;
use std::sync::Arc;

/// A handle of one `Arc`, as the runtime's objects are.
#[derive(Clone)]
struct Probe {
    inner: Arc<u32>,
}

impl Handle for Probe {
    fn address(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    unsafe fn uncounted(&self) -> ManuallyDrop<Probe> {
        // SAFETY: as the runtime's handles do.
        let inner = unsafe { Arc::from_raw(Arc::as_ptr(&self.inner)) };
        ManuallyDrop::new(Probe { inner })
    }
}

/// An object kept, then used again, held again and borrowed many times, has one reference of the
/// list's; the borrows read it; clearing the list gives the reference back, and the next
/// recording keeps the object again.
#[test]
fn a_recording_keeps_an_object_once_and_its_borrows_change_no_count() {
    let (a, b) = (Probe { inner: Arc::new(1) }, Probe { inner: Arc::new(2) });
    let mut kept = Kept::default();
    // SAFETY: every borrow ends before the list is cleared.
    let lent = unsafe { kept.keep(&a) };
    assert_eq!(
        Arc::strong_count(&a.inner),
        2,
        "the first use takes one reference"
    );
    let borrows: Vec<_> = (0..1_000)
        .map(|_| {
            // SAFETY: as above.
            unsafe { kept.keep(&a) }
        })
        .collect();
    assert!(!kept.hold(&a), "a kept object is not kept again");
    let copies: Vec<_> = (0..1_000).map(|_| lent.clone()).collect();
    assert_eq!(
        Arc::strong_count(&a.inner),
        2,
        "1,000 uses and 1,000 borrows' copies change no count"
    );
    assert!(borrows
        .iter()
        .chain(&copies)
        .all(|borrow| *borrow.inner == 1));
    assert!(kept.hold(&b), "another object is kept");
    assert_eq!(
        kept.iter().map(|probe| *probe.inner).collect::<Vec<_>>(),
        [1, 2]
    );
    drop((lent, borrows, copies));
    assert_eq!(
        Arc::strong_count(&a.inner),
        2,
        "dropping borrows releases nothing"
    );
    kept.clear();
    assert_eq!(
        (Arc::strong_count(&a.inner), Arc::strong_count(&b.inner)),
        (1, 1),
        "clearing releases each object's one reference"
    );
    assert!(kept.as_slice().is_empty());
    assert!(kept.hold(&a), "a cleared list keeps the object again");
    assert_eq!(Arc::strong_count(&a.inner), 2);
}
