//! A `VkQueue`'s floor of its last calls of commands (`device/submit_calls.rs`): the least of the
//! last [`CALLS`] once the shortest is set aside.

use crate::device::submit_calls::{SubmitCalls, CALLS, SLOW_NANOS};

/// A call of commands under the bound, as lavapipe's or an idle device's.
const SHORT: u64 = 5_000;
/// A call of commands past the bound.
const SLOW: u64 = 2 * SLOW_NANOS;

/// A ring that timed `calls`, in order.
fn timed(calls: impl IntoIterator<Item = u64>) -> SubmitCalls {
    let ring = SubmitCalls::default();
    for nanos in calls {
        ring.called(nanos);
    }
    ring
}

/// `CALLS` calls, each `SLOW` but those at `short`, which are `SHORT`.
fn slow_but(short: &[usize]) -> Vec<u64> {
    (0..CALLS)
        .map(|at| if short.contains(&at) { SHORT } else { SLOW })
        .collect()
}

/// No floor before [`CALLS`] calls were timed. One short call among the last [`CALLS`] slow ones,
/// wherever it fell, leaves the queue slow (the module docs, "One short call set aside"), as one
/// guest queue's smaller batches among another's slow ones; two do not, and a short call leaves
/// the ring [`CALLS`] calls after it. A driver's long call among short ones never makes the queue
/// slow, nor do a fresh device's slow first calls but the last two.
#[test]
fn a_queue_s_floor_sets_its_shortest_call_aside() {
    let early = timed(vec![SLOW; CALLS - 1]);
    assert_eq!(early.floor(), 0, "a floor before {CALLS} calls");
    assert!(!early.slow(SLOW_NANOS));
    for at in 0..CALLS {
        let ring = timed(slow_but(&[at]));
        assert_eq!(
            ring.floor(),
            SLOW,
            "the short call {at} of {CALLS} set aside"
        );
        assert!(ring.slow(SLOW_NANOS), "the short call {at}");
    }
    let two = timed(slow_but(&[3, 9]));
    assert_eq!(two.floor(), SHORT, "two short calls among slow ones");
    assert!(!two.slow(SLOW_NANOS));
    // The first of two short calls leaves the ring with the next call after these.
    let mut calls = slow_but(&[0, 1]);
    let ring = timed(calls.iter().copied());
    assert!(!ring.slow(SLOW_NANOS), "two short calls in the ring");
    calls.push(SLOW);
    assert!(
        timed(calls).slow(SLOW_NANOS),
        "one short call left in the ring"
    );
    let long = timed((0..CALLS).map(|at| if at == 7 { SLOW } else { SHORT }));
    assert_eq!(long.floor(), SHORT, "a long call among short ones");
    assert!(!long.slow(SLOW_NANOS));
    let fresh = timed(slow_but(&[CALLS - 2, CALLS - 1]));
    assert!(!fresh.slow(SLOW_NANOS), "a fresh device's slow first calls");
}
