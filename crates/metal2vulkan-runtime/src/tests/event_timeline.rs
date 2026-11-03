//! An event's signal indices and Metal's values (`event_timeline.rs`): monotone values, dropped
//! signals, the order of kept signals across `VkQueue`s, the host's floor, the waits a batch is
//! submitted with, and a counter read observed only once a host wait returned for it. A timeline
//! of a plain event (`MTLEvent`) is `new(false)`, one whose value the host may set
//! (`MTLSharedEvent`) `new(true)`; `host_settable_event_waits` holds the difference.

use crate::event_timeline::{EventTimeline, Signal, WaitFor};

/// Submit a GPU signal of `value` on `queue`: what it became, kept when it is.
fn submit(timeline: &mut EventTimeline<usize>, value: u64, queue: usize) -> Signal {
    let signal = timeline.signal(value, queue);
    if let Signal::Kept { index, .. } = signal {
        timeline.keep(index, value, (queue, index), None, false);
    }
    signal
}

#[test]
fn kept_signals_take_rising_indices() {
    let mut timeline = EventTimeline::new(false);
    assert_eq!(
        submit(&mut timeline, 3, 0),
        Signal::Kept {
            index: 1,
            after: None
        }
    );
    assert_eq!(
        submit(&mut timeline, 5, 0),
        Signal::Kept {
            index: 2,
            after: None
        }
    );
    assert_eq!(timeline.known(), 5);
    // Nothing reached yet.
    assert_eq!(timeline.value(), 0);
    assert!(timeline.observe(1));
    assert_eq!(timeline.value(), 3);
    assert!(timeline.observe(2));
    assert_eq!(timeline.value(), 5);
    assert_eq!(timeline.pending(), 0);
    // A later signal continues from the last index reached.
    assert_eq!(
        submit(&mut timeline, 6, 1),
        Signal::Kept {
            index: 3,
            after: None
        }
    );
}

#[test]
fn a_signal_not_above_what_is_known_is_dropped() {
    let mut timeline = EventTimeline::new(false);
    submit(&mut timeline, 7, 0);
    // Lower and equal: Metal's maximum makes them no-ops (measured).
    assert_eq!(submit(&mut timeline, 4, 1), Signal::Dropped);
    assert_eq!(submit(&mut timeline, 7, 0), Signal::Dropped);
    assert_eq!(timeline.pending(), 1);
    timeline.observe(1);
    assert_eq!(timeline.value(), 7);
}

#[test]
fn either_submission_order_of_seven_and_four_ends_at_seven() {
    // Seven first: the four is dropped.
    let mut first = EventTimeline::new(false);
    submit(&mut first, 7, 0);
    assert_eq!(submit(&mut first, 4, 1), Signal::Dropped);
    first.observe(1);
    assert_eq!(first.value(), 7);
    // Four first: the seven is kept after it, and waits for it on another queue.
    let mut second = EventTimeline::new(false);
    submit(&mut second, 4, 1);
    assert_eq!(
        submit(&mut second, 7, 0),
        Signal::Kept {
            index: 2,
            after: Some(1)
        }
    );
    second.observe(1);
    assert_eq!(second.value(), 4);
    second.observe(2);
    assert_eq!(second.value(), 7);
}

#[test]
fn a_kept_signal_waits_only_for_one_pending_on_another_queue() {
    let mut timeline = EventTimeline::new(false);
    submit(&mut timeline, 1, 0);
    // Same queue: Vulkan's signal operation order keeps them in order.
    assert_eq!(
        submit(&mut timeline, 2, 0),
        Signal::Kept {
            index: 2,
            after: None
        }
    );
    // Another queue, the previous one pending: wait for it.
    assert_eq!(
        submit(&mut timeline, 3, 1),
        Signal::Kept {
            index: 3,
            after: Some(2)
        }
    );
    // Once it was reached, no wait.
    timeline.observe(3);
    assert_eq!(
        submit(&mut timeline, 4, 0),
        Signal::Kept {
            index: 4,
            after: None
        }
    );
}

#[test]
fn waits_map_to_the_first_index_that_reaches_the_value() {
    let mut timeline = EventTimeline::new(false);
    submit(&mut timeline, 3, 0);
    submit(&mut timeline, 8, 0);
    assert_eq!(timeline.wait(0), Some(WaitFor::Reached));
    assert_eq!(timeline.wait(2), Some(WaitFor::Index(1)));
    assert_eq!(timeline.wait(3), Some(WaitFor::Index(1)));
    assert_eq!(timeline.wait(4), Some(WaitFor::Index(2)));
    assert_eq!(timeline.wait(8), Some(WaitFor::Index(2)));
    // Nothing submitted reaches 9: the batch is held.
    assert_eq!(timeline.wait(9), None);
    timeline.observe(1);
    assert_eq!(timeline.wait(3), Some(WaitFor::Reached));
}

#[test]
fn the_host_floor_takes_effect_at_once_and_never_on_the_semaphore() {
    // A shared event: only its value may be set.
    let mut timeline = EventTimeline::new(true);
    submit(&mut timeline, 7, 0);
    // A host set below the pending GPU signal: the value is 5 now.
    assert!(timeline.set(5));
    assert_eq!(timeline.value(), 5);
    // Lower and equal sets change nothing (measured).
    assert!(!timeline.set(3));
    assert!(!timeline.set(5));
    // A wait the floor reached needs no Vulkan wait, though a GPU signal above it is pending.
    assert_eq!(timeline.wait(5), Some(WaitFor::Reached));
    // One above it is held: a set the host makes next must release it, which no Vulkan wait would.
    assert_eq!(timeline.wait(6), None);
    // The GPU's seven lands above the floor.
    assert!(timeline.observe(1));
    assert_eq!(timeline.value(), 7);
}

#[test]
fn a_wait_only_the_host_floor_satisfies_is_never_a_vulkan_wait() {
    let mut timeline = EventTimeline::new(true);
    assert_eq!(timeline.wait(9), None);
    timeline.set(9);
    assert_eq!(timeline.known(), 9);
    for value in [1, 8, 9] {
        assert_eq!(timeline.wait(value), Some(WaitFor::Reached));
    }
    assert_eq!(timeline.wait(10), None);
    // A GPU signal not above the floor is dropped, so no index ever stands for the floor.
    assert_eq!(submit(&mut timeline, 9, 0), Signal::Dropped);
    assert_eq!(
        submit(&mut timeline, 12, 0),
        Signal::Kept {
            index: 1,
            after: None
        }
    );
    assert_eq!(timeline.wait(9), Some(WaitFor::Reached));
    // The twelve is pending, but the host may set ten: the batch is held, not handed a wait.
    assert_eq!(timeline.wait(10), None);
}

#[test]
fn a_host_set_above_a_pending_signal_reaches_every_wait_below_it() {
    let mut timeline = EventTimeline::new(true);
    submit(&mut timeline, 7, 0);
    // Held, never handed to Vulkan: the host may set five, and a Vulkan wait for the seven's
    // index would hold the batch until the seven's GPU work ran (`host_settable_event_waits`).
    assert_eq!(timeline.wait(5), None);
    timeline.set(9);
    assert_eq!(timeline.value(), 9);
    assert_eq!(timeline.wait(5), Some(WaitFor::Reached));
    // The GPU's seven reaches index 1, and the value stays at the floor.
    assert!(!timeline.observe(1));
    assert_eq!(timeline.value(), 9);
}

#[test]
fn the_largest_value_is_kept() {
    let mut timeline = EventTimeline::new(true);
    assert_eq!(
        submit(&mut timeline, u64::MAX, 0),
        Signal::Kept {
            index: 1,
            after: None
        }
    );
    timeline.observe(1);
    assert_eq!(timeline.value(), u64::MAX);
    assert_eq!(submit(&mut timeline, u64::MAX, 0), Signal::Dropped);
    assert!(!timeline.set(u64::MAX));
}

/// A counter read past the index last seen raises the value only once a host wait returned for
/// it (`event.rs`): a read alone orders no host operation after the signal. A read the index
/// last seen covers asks for no wait, and a failed wait observes nothing.
#[test]
fn a_counter_read_is_observed_only_once_a_host_wait_returned_for_it() {
    let mut timeline = EventTimeline::new(false);
    submit(&mut timeline, 3, 0);
    submit(&mut timeline, 5, 0);
    let mut asked = Vec::new();
    let mut observe = |timeline: &mut EventTimeline<usize>, read: u64, returned: bool| {
        timeline.observe_read::<()>(read, |index| {
            asked.push(index);
            Ok(returned)
        })
    };
    // The read shows both signals; the wait for them does not return: nothing is observed.
    assert_eq!(observe(&mut timeline, 2, false), Ok(false));
    assert_eq!(timeline.value(), 0);
    // It returns: both are.
    assert_eq!(observe(&mut timeline, 2, true), Ok(true));
    assert_eq!(timeline.value(), 5);
    // A read the index last seen covers needs no wait.
    assert_eq!(observe(&mut timeline, 2, true), Ok(false));
    assert_eq!(asked, [2, 2]);
    // A failed wait (a lost device) observes nothing.
    submit(&mut timeline, 7, 0);
    assert_eq!(timeline.observe_read(3, |_| Err("lost")), Err("lost"));
    assert_eq!(timeline.value(), 5);
}

/// A signal kept behind a merge (its batch may write a copied host buffer, merged into the pages
/// at its completion turn) is observed by that turn alone: a counter read past it observes only
/// the signals before it, no wait is handed its index or a later one's (a plain event's wait is
/// held as a shared event's is), and the completion turn's observation reaches it and those after
/// it.
#[test]
fn a_signal_behind_a_merge_is_seen_at_its_completion_turn_alone() {
    for settable in [false, true] {
        let mut timeline: EventTimeline<usize> = EventTimeline::new(settable);
        for (value, behind_merge) in [(3, false), (5, true), (7, false)] {
            let Signal::Kept { index, .. } = timeline.signal(value, 0) else {
                panic!("kept");
            };
            timeline.keep(index, value, (0, index), None, behind_merge);
        }
        // The counter shows all three: the read observes the first alone.
        assert_eq!(timeline.observe_read::<()>(3, |_| Ok(true)), Ok(true));
        assert_eq!(timeline.value(), 3, "{settable}");
        assert_eq!(timeline.observe_read::<()>(3, |_| Ok(true)), Ok(false));
        assert_eq!(timeline.value(), 3, "{settable}");
        // No wait is handed the index of the signal behind the merge, or a later one's.
        for value in [4, 5, 6, 7] {
            assert_eq!(timeline.landing(value), None, "{settable} {value}");
            assert_eq!(timeline.wait(value), None, "{settable} {value}");
        }
        assert_eq!(timeline.wait(3), Some(WaitFor::Reached));
        // The completion turn, after the merge, observes it.
        assert!(timeline.observe(2));
        assert_eq!(timeline.value(), 5, "{settable}");
        // A later signal not behind a merge is read and waited for as before.
        let want = (!settable).then_some(WaitFor::Index(3));
        assert_eq!(timeline.wait(7), want, "{settable}");
        assert_eq!(timeline.observe_read::<()>(3, |_| Ok(true)), Ok(true));
        assert_eq!(timeline.value(), 7, "{settable}");
    }
}

/// The caller's signals (`SharedEvent::reserve_vk_signal`): each takes the next index at once,
/// after the last pending signal whatever queue it is on, and a later batch's signal waits for it
/// in turn; one not above what is known is dropped; once a host wait returned for one (the
/// watcher's), every pending signal up to it is observed, but never past a signal behind a merge,
/// whose completion turn observes it and then the rest.
#[test]
fn the_caller_s_signals_take_the_next_index_and_land_through_the_watcher() {
    let mut timeline = EventTimeline::<usize>::new(true);
    assert_eq!(
        timeline.signal_apart(3),
        Signal::Kept {
            index: 1,
            after: None
        }
    );
    assert_eq!(timeline.known(), 3);
    assert_eq!(timeline.signal_apart(3), Signal::Dropped);
    // A batch's signal after it, on any queue, waits for it; and the caller's next after that.
    assert_eq!(
        submit(&mut timeline, 5, 0),
        Signal::Kept {
            index: 2,
            after: Some(1)
        }
    );
    assert_eq!(
        timeline.signal_apart(7),
        Signal::Kept {
            index: 3,
            after: Some(2)
        }
    );
    assert_eq!(
        timeline.signal_apart(8),
        Signal::Kept {
            index: 4,
            after: Some(3)
        }
    );
    assert_eq!(timeline.landing(7), Some(WaitFor::Index(3)));
    // Nothing is seen until the watcher's wait returned.
    assert_eq!(timeline.value(), 0);
    assert!(timeline.landed(3));
    assert_eq!((timeline.value(), timeline.pending()), (7, 1));
    assert_eq!(timeline.landing(7), Some(WaitFor::Reached));
    assert!(timeline.landed(4));
    assert_eq!((timeline.value(), timeline.pending()), (8, 0));
    // A merge's signal before the caller's: the caller's waits for its completion turn.
    let mut timeline = EventTimeline::<usize>::new(true);
    let Signal::Kept { index, .. } = timeline.signal(2, 0) else {
        panic!("kept");
    };
    timeline.keep(index, 2, (0, index), None, true);
    assert_eq!(
        timeline.signal_apart(4),
        Signal::Kept {
            index: 2,
            after: Some(1)
        }
    );
    assert!(!timeline.landed(2), "behind the merge's signal");
    assert_eq!(timeline.value(), 0);
    assert!(timeline.observe(1), "the merge's completion turn");
    assert_eq!((timeline.value(), timeline.pending()), (4, 0));
}

/// What a batch that waits for a value completes after (`EventTimeline::signallers`), from the
/// first kept signal that reaches it, whatever the semaphore was seen to reach: its batch, until
/// that batch's completion turn ran, and its command buffer, where the batch ends it, until that
/// one's transition to `Completed` ran up to its completed handlers; nothing where the host's set
/// or the caller's signal reached it. Signals are let go of from the first on as they are released
/// (a batch that ends no command buffer is released as its turn runs), so the record holds only
/// those not released and the released ones after them.
#[test]
fn a_wait_completes_after_the_signal_that_reaches_it() {
    let mut timeline = EventTimeline::<usize, char>::new(true);
    // Value 2 on queue 0 at its value 10, ending command buffer 'a'; then 5 on queue 1 at 20, in
    // a batch that ends no command buffer.
    for (value, queue, batch, signaller) in [(2, 0, 10, Some('a')), (5, 1, 20, None)] {
        let Signal::Kept { index, .. } = timeline.signal(value, queue) else {
            panic!("kept");
        };
        timeline.keep(index, value, (queue, batch), signaller, false);
    }
    assert_eq!(timeline.signallers(1), (Some((0, 10)), Some('a')));
    assert_eq!(timeline.signallers(2), (Some((0, 10)), Some('a')));
    assert_eq!(timeline.signallers(3), (Some((1, 20)), None));
    assert_eq!(
        timeline.signallers(6),
        (None, None),
        "no kept signal reaches it"
    );
    // A read of the counter observes them; their batches' turns have not run.
    assert!(timeline.observe(2));
    assert_eq!(timeline.signallers(4), (Some((1, 20)), None));
    // The second's turn first: it is released, ending no command buffer; the first still not.
    timeline.finished(2);
    assert_eq!(timeline.signallers(4), (None, None));
    assert_eq!(timeline.unfinished(), 2);
    // The first's turn: its batch finished, its command buffer's transition still to run.
    timeline.finished(1);
    assert_eq!(timeline.signallers(2), (None, Some('a')));
    assert_eq!(timeline.unfinished(), 2);
    // That transition reached its handlers: both let go of.
    timeline.released(1);
    assert_eq!(timeline.unfinished(), 0);
    assert_eq!(timeline.signallers(2), (None, None));
    // A later signal still to finish does not stand for one let go of that reached the value.
    let Signal::Kept { index, .. } = timeline.signal(6, 1) else {
        panic!("kept");
    };
    timeline.keep(index, 6, (1, 25), Some('b'), false);
    assert_eq!(timeline.signallers(4), (None, None));
    assert_eq!(timeline.signallers(6), (Some((1, 25)), Some('b')));
    // Released before its turn was seen (a drained failure's transition): let go of.
    timeline.released(index);
    assert_eq!(
        (timeline.signallers(6), timeline.unfinished()),
        ((None, None), 0)
    );
    // The host's set reaches 7: nothing to complete after, though a GPU signal reaches it too.
    let Signal::Kept { index, .. } = timeline.signal(8, 0) else {
        panic!("kept");
    };
    timeline.keep(index, 8, (0, 30), Some('c'), false);
    assert!(timeline.set(7));
    assert_eq!(timeline.signallers(7), (None, None));
    assert_eq!(timeline.signallers(8), (Some((0, 30)), Some('c')));
    // A caller's signal reaching it: nothing to complete after.
    assert!(matches!(timeline.signal_apart(9), Signal::Kept { .. }));
    assert_eq!(timeline.signallers(9), (None, None));
    timeline.finished(index);
    timeline.released(index);
    assert_eq!(timeline.unfinished(), 0);
}
