//! The staging region's ring: spans are aligned, inside the ring and never overlap a live one;
//! a span is free only once its submission completed and no host reads it, oldest first; the ring
//! starts again at 0 when nothing is live and wraps to the start when the end has no room; and
//! what frees the oldest span is named for a caller with no room.

use crate::staging::{At, Oldest, Ring, SPAN_ALIGN};

/// Every live span lies in the ring at an aligned offset, and no two overlap.
fn check(ring: &Ring) {
    let live = ring.live();
    for (index, &(start, end, _)) in live.iter().enumerate() {
        assert_eq!(start % SPAN_ALIGN, 0, "span at {start}");
        assert!(start < end && end <= ring.len(), "span {start}..{end}");
        for &(other_start, other_end, _) in &live[index + 1..] {
            assert!(
                end <= other_start || other_end <= start,
                "{start}..{end} overlaps {other_start}..{other_end}"
            );
        }
    }
}

#[test]
fn a_ring_takes_after_the_last_span_wraps_and_starts_again() {
    let mut ring = Ring::new(4096);
    let (a, at_a) = ring.take(1000, false).expect("room");
    let (b, at_b) = ring.take(1000, false).expect("room");
    assert_eq!((at_a, at_b), (0, 1024), "after the last span, aligned");
    let (c, at_c) = ring.take(1500, false).expect("room");
    assert_eq!(at_c, 2048);
    // 3548..4096 is too short, and the start is live.
    assert_eq!(ring.take(1000, false), None);
    assert_eq!(ring.oldest(), Some(Oldest::Busy), "a span being written");
    ring.set(a, At::Recorded);
    assert_eq!(ring.oldest(), Some(Oldest::Pending));
    ring.set(a, At::Submitted(5));
    assert_eq!(ring.oldest(), Some(Oldest::Submitted(5)));
    ring.reclaim(4);
    assert_eq!(
        ring.take(1000, false),
        None,
        "the submission has not completed"
    );
    ring.reclaim(5);
    // The end has no room; the start, freed, has.
    let (d, at_d) = ring.take(1000, false).expect("room at the start");
    assert_eq!(at_d, 0);
    check(&ring);
    // The live spans wrap: the room is between the last taken and the oldest.
    assert_eq!(ring.take(100, false), None, "1000..1024 is too short");
    for span in [b, c, d] {
        ring.set(span, At::Submitted(6));
    }
    ring.reclaim(6);
    assert!(ring.live().is_empty());
    assert_eq!(
        ring.take(4096, false).map(|(_, at)| at),
        Some(0),
        "empty starts at 0"
    );
}

#[test]
fn a_read_span_and_a_span_behind_an_unfinished_one_stay_live() {
    let mut ring = Ring::new(4096);
    let (read, _) = ring.take(256, true).expect("room");
    let (later, _) = ring.take(256, false).expect("room");
    ring.set(read, At::Submitted(1));
    ring.set(later, At::Submitted(1));
    ring.reclaim(1);
    assert_eq!(ring.live().len(), 2, "the host still reads the first");
    assert_eq!(ring.oldest(), Some(Oldest::Busy));
    ring.unread(read);
    ring.reclaim(1);
    assert!(ring.live().is_empty());
    // A span given up frees at once; one behind a span still being written waits for it.
    let (first, _) = ring.take(256, false).expect("room");
    let (second, _) = ring.take(256, false).expect("room");
    ring.set(second, At::Done);
    ring.reclaim(u64::MAX);
    assert_eq!(ring.live().len(), 2);
    ring.set(first, At::Done);
    ring.reclaim(0);
    assert!(ring.live().is_empty());
}

/// A long run of takes of every length up to the ring's, submitted and completed out of step,
/// against the invariants after every step.
#[test]
fn a_ring_under_load_never_overlaps_a_live_span() {
    let len = 64 * SPAN_ALIGN;
    let mut ring = Ring::new(len);
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut submitted, mut completed) = (0_u64, 0_u64);
    let mut recorded: Vec<u64> = Vec::new();
    let mut taken = 0;
    for _ in 0..20_000 {
        match next() % 4 {
            0 | 1 => {
                let want = 1 + next() % len;
                if let Some((number, _)) = ring.take(want, false) {
                    ring.set(number, At::Recorded);
                    recorded.push(number);
                    taken += 1;
                }
            }
            2 => {
                submitted += 1;
                for number in recorded.drain(..) {
                    ring.set(number, At::Submitted(submitted));
                }
            }
            _ => {
                completed = (completed + 1 + next() % 2).min(submitted);
                ring.reclaim(completed);
            }
        }
        check(&ring);
        // Nothing completed stays at the front.
        if let Some(Oldest::Submitted(value)) = ring.oldest() {
            assert!(
                value > completed,
                "the front's {value} completed at {completed}"
            );
        }
    }
    assert!(taken > 1_000, "the run took {taken} spans");
}
