//! What a recording keeps (`crate::kept`): each object once, its first use the one reference
//! taken, of the object alone, never of the device; a later use and every borrow of it change no
//! count; clearing releases every object, and frees a list grown past the bound.

use crate::kept::{Handle, Kept, KEPT_BOUND};
use std::sync::Arc;

/// A handle of an object and of its device, as the runtime's buffers, textures and sampler
/// states are.
#[derive(Clone)]
struct Probe {
    inner: Arc<u32>,
    device: Arc<()>,
}

impl Handle for Probe {
    type Kept = Arc<u32>;

    fn object(&self) -> &Arc<u32> {
        &self.inner
    }

    fn address_of(object: &Arc<u32>) -> usize {
        Arc::as_ptr(object) as usize
    }
}

/// A probe of a new object on `device`.
fn probe(value: u32, device: &Arc<()>) -> Probe {
    Probe {
        inner: Arc::new(value),
        device: Arc::clone(device),
    }
}

/// An object kept, then used again, held again and borrowed many times, has one reference of the
/// list's, and the device none; the borrows read it; clearing the list gives the reference back,
/// and the next recording keeps the object again.
#[test]
fn a_recording_keeps_an_object_once_and_its_borrows_change_no_count() {
    let device = Arc::new(());
    let (a, b) = (probe(1, &device), probe(2, &device));
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
    assert!(borrows.iter().chain(&copies).all(|borrow| ***borrow == 1));
    assert!(kept.hold(&b), "another object is kept");
    assert_eq!(kept.iter().map(|kept| **kept).collect::<Vec<_>>(), [1, 2]);
    assert_eq!(
        Arc::strong_count(&device),
        3,
        "nothing the list keeps, and no borrow, counts as a handle of the device"
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

/// A recording of the bound's objects keeps its list's room for the next; one of a single object
/// more grows it past the bound, and its clear frees it, releasing every object, so the next
/// recording starts with no room.
#[test]
fn a_list_grown_past_the_bound_is_freed_at_its_clear() {
    let device = Arc::new(());
    let probes: Vec<Probe> = (0..=KEPT_BOUND)
        .map(|index| probe(index as u32, &device))
        .collect();
    let mut kept = Kept::default();
    for probe in &probes[..KEPT_BOUND] {
        assert!(kept.hold(probe));
    }
    assert_eq!(kept.capacity(), KEPT_BOUND, "room for the bound's objects");
    kept.clear();
    assert_eq!(kept.capacity(), KEPT_BOUND, "kept within the bound");
    for probe in &probes {
        assert!(kept.hold(probe));
    }
    assert!(kept.capacity() > KEPT_BOUND, "grown past the bound");
    kept.clear();
    assert_eq!(kept.capacity(), 0, "freed past the bound");
    assert!(
        probes
            .iter()
            .all(|probe| Arc::strong_count(&probe.inner) == 1),
        "the freed list released every object"
    );
    assert!(kept.hold(&probes[0]), "the next recording keeps an object");
    assert!(kept.capacity() < KEPT_BOUND, "from no room");
}
