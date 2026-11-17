//! A `VkQueue`'s cost of its last batch submissions (`device/submit_calls.rs`): the median of the
//! last [`CALLS`].

use crate::device::submit_calls::{SubmitCalls, CALLS, SLOW_NANOS};

/// A submission under the bound, as lavapipe's or an idle device's.
const SHORT: u64 = 5_000;
/// A submission past the bound.
const SLOW: u64 = 2 * SLOW_NANOS;

/// A ring that timed `calls`, in order.
fn timed(calls: impl IntoIterator<Item = u64>) -> SubmitCalls {
    let ring = SubmitCalls::default();
    for nanos in calls {
        ring.called(nanos);
    }
    ring
}

/// `CALLS` submissions, each `SLOW` but those at `short`, which are `SHORT`.
fn slow_but(short: &[usize]) -> Vec<u64> {
    (0..CALLS)
        .map(|at| if short.contains(&at) { SHORT } else { SLOW })
        .collect()
}

/// No cost before [`CALLS`] submissions were timed. Short submissions among the last [`CALLS`] slow
/// ones leave the queue slow while fewer than half of them are short, wherever they fell: one, as
/// one guest queue's smaller batches among another's slow ones; a third, as the Iris Xe's
/// compositor parts, whose submissions straddle the bound (the module docs), which a floor of the
/// least but the shortest judged fast; seven. Eight do not, and a short submission leaves the
/// ring [`CALLS`] submissions after it. A driver's long calls among short submissions never make
/// the queue slow while fewer than half are long; nor does a fresh device's slow first calls but
/// the last eight.
#[test]
fn a_queue_s_cost_is_the_median_of_its_last_submissions() {
    let early = timed(vec![SLOW; CALLS - 1]);
    assert_eq!(early.cost(), 0, "a cost before {CALLS} submissions");
    assert!(!early.slow(SLOW_NANOS));
    for at in 0..CALLS {
        let ring = timed(slow_but(&[at]));
        assert_eq!(ring.cost(), SLOW, "the short submission {at} of {CALLS}");
        assert!(ring.slow(SLOW_NANOS), "the short submission {at}");
    }
    let third = timed(slow_but(&[0, 3, 6, 9, 12]));
    assert_eq!(third.cost(), SLOW, "a third of the submissions short");
    assert!(third.slow(SLOW_NANOS));
    let seven = timed(slow_but(&[0, 2, 4, 6, 8, 10, 12]));
    assert!(seven.slow(SLOW_NANOS), "seven short submissions");
    let eight = timed(slow_but(&[0, 2, 4, 6, 8, 10, 12, 14]));
    assert_eq!(
        eight.cost(),
        SHORT,
        "eight short submissions among slow ones"
    );
    assert!(!eight.slow(SLOW_NANOS));
    // The first of eight short submissions leaves the ring with the next one after these.
    let mut calls = slow_but(&[0, 1, 2, 3, 4, 5, 6, 7]);
    let ring = timed(calls.iter().copied());
    assert!(
        !ring.slow(SLOW_NANOS),
        "eight short submissions in the ring"
    );
    calls.push(SLOW);
    assert!(
        timed(calls).slow(SLOW_NANOS),
        "seven short submissions left in the ring"
    );
    let long = timed((0..CALLS).map(|at| if at % 2 == 1 { SLOW } else { SHORT }));
    assert_eq!(
        long.cost(),
        SHORT,
        "seven long calls among short submissions"
    );
    assert!(!long.slow(SLOW_NANOS));
    let fresh = timed(slow_but(&[7, 8, 9, 10, 11, 12, 13, 14]));
    assert!(!fresh.slow(SLOW_NANOS), "a fresh device's slow first calls");
}
