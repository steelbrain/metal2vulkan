//! Waits from inside the runtime's handler threads (`command_buffer/own.rs`): the marker, what
//! each wait answers in Metal's measured cases, and the graph of waits across contexts.

use crate::command_buffer::own::{
    completed_wait, running, scheduled_wait, slot_wait, Answer, Context, Marker, Running, Step,
    Target, Waits,
};
use crate::error::DeadlockCause;

const DEVICE: usize = 0x10;
const OTHER_DEVICE: usize = 0x20;
const QUEUE: u64 = 1;
const OTHER_QUEUE: u64 = 2;
const OWN: usize = 0x1000;
const LATER: usize = 0x2000;

fn context(device: usize, queue: u64) -> Context {
    Context { device, queue }
}

fn marker(step: Step) -> Option<Marker> {
    Some(Marker {
        context: context(DEVICE, QUEUE),
        buffer: OWN,
        step,
    })
}

fn target(device: usize, queue: u64, buffer: usize) -> Target {
    Target {
        context: context(device, queue),
        buffer,
    }
}

const STEPS: [Step; 3] = [
    Step::ScheduledHandlers,
    Step::CompletedHandlers,
    Step::Release,
];

#[test]
fn a_step_is_marked_until_its_marker_goes_and_the_outer_one_comes_back() {
    assert_eq!(running(), None);
    let outer = marker(Step::ScheduledHandlers).unwrap();
    let _outer = Running::enter(outer);
    {
        let inner = Marker {
            buffer: LATER,
            step: Step::Release,
            ..outer
        };
        let _inner = Running::enter(inner);
        assert_eq!(running(), Some(inner));
    }
    assert_eq!(running(), Some(outer));
    // The marker is this thread's alone.
    assert_eq!(std::thread::spawn(running).join().unwrap(), None);
}

#[test]
fn waits_off_the_handler_threads_wait() {
    for done in [false, true] {
        for buffer in [OWN, LATER] {
            let at = target(DEVICE, QUEUE, buffer);
            assert_eq!(completed_wait(None, at, done), Answer::Wait);
            assert_eq!(scheduled_wait(None, at, done), Answer::Wait);
        }
    }
}

/// Where Metal returns, the runtime returns or waits; where Metal never returns, it refuses.
#[test]
fn own_waits_answer_as_metal_s() {
    let own = target(DEVICE, QUEUE, OWN);
    // A deallocator's wait for the completion of the command buffer releasing it returns.
    assert_eq!(
        completed_wait(marker(Step::Release), own, false),
        Answer::Return
    );
    // A completed handler's wait for its command buffer's scheduling returns (that ran).
    assert_eq!(
        scheduled_wait(marker(Step::CompletedHandlers), own, true),
        Answer::Wait
    );
    // Metal never returns from these.
    for step in [Step::ScheduledHandlers, Step::CompletedHandlers] {
        assert_eq!(
            completed_wait(marker(step), own, false),
            Answer::Refuse(DeadlockCause::OwnQueue)
        );
    }
    assert_eq!(
        scheduled_wait(marker(Step::ScheduledHandlers), own, true),
        Answer::Refuse(DeadlockCause::OwnQueue)
    );
}

/// A later command buffer of the handler's own queue: its transitions come after the handler's
/// in the queue's order (Metal: never returns), unless they ran.
#[test]
fn a_wait_for_a_later_transition_of_the_own_queue_is_refused() {
    let later = target(DEVICE, QUEUE, LATER);
    for step in STEPS {
        assert_eq!(
            completed_wait(marker(step), later, false),
            Answer::Refuse(DeadlockCause::OwnQueue)
        );
        assert_eq!(
            scheduled_wait(marker(step), later, false),
            Answer::Refuse(DeadlockCause::OwnQueue)
        );
        assert_eq!(completed_wait(marker(step), later, true), Answer::Wait);
        assert_eq!(scheduled_wait(marker(step), later, true), Answer::Wait);
    }
}

/// Another queue's command buffer, on this device or another: Metal returns, so the runtime
/// waits, counted from the handler's context.
#[test]
fn a_wait_for_another_queue_blocks_from_the_handler_s_context() {
    let from = context(DEVICE, QUEUE);
    for at in [
        target(DEVICE, OTHER_QUEUE, LATER),
        target(OTHER_DEVICE, QUEUE, LATER),
    ] {
        for step in STEPS {
            assert_eq!(completed_wait(marker(step), at, false), Answer::Block(from));
            assert_eq!(scheduled_wait(marker(step), at, false), Answer::Block(from));
            assert_eq!(completed_wait(marker(step), at, true), Answer::Wait);
            assert_eq!(scheduled_wait(marker(step), at, true), Answer::Wait);
        }
    }
}

#[test]
fn a_wait_closing_a_cycle_of_waits_is_found() {
    let [a, b, c, d] = [1, 2, 3, 4].map(|queue| context(DEVICE, queue));
    let elsewhere = context(OTHER_DEVICE, 1);
    let mut waits = Waits::new();
    assert!(!waits.closes_cycle(a, b));
    // A context never waits for itself.
    assert!(waits.closes_cycle(a, a));
    waits.insert(a, b);
    assert!(waits.closes_cycle(b, a));
    waits.insert(b, c);
    // Through b: c waits for a.
    assert!(waits.closes_cycle(c, a));
    assert!(!waits.closes_cycle(d, a));
    // Across devices too.
    waits.insert(c, elsewhere);
    assert!(waits.closes_cycle(elsewhere, a));
    assert!(waits.closes_cycle(elsewhere, b));
    assert!(!waits.closes_cycle(elsewhere, d));
    // A wait that ended leaves no edge.
    waits.remove(b, c);
    assert!(!waits.closes_cycle(c, a));
    assert!(!waits.closes_cycle(elsewhere, a));
    assert!(waits.closes_cycle(b, a));
}

/// A make on a full queue: off the handler threads it waits; from another queue's context it
/// blocks, counted (Metal's returns once a slot comes back); in the queue's own context, where no
/// committed command buffer can complete first, it is refused (Metal's never returns) unless an
/// uncommitted one may still be dropped.
#[test]
fn a_make_on_a_full_queue_answers_by_where_its_slots_can_come_back() {
    let own = context(DEVICE, QUEUE);
    for open in [false, true] {
        assert_eq!(slot_wait(None, own, open), Answer::Wait);
        for step in STEPS {
            for queue in [context(DEVICE, OTHER_QUEUE), context(OTHER_DEVICE, QUEUE)] {
                assert_eq!(slot_wait(marker(step), queue, open), Answer::Block(own));
            }
            let answer = if open {
                Answer::Wait
            } else {
                Answer::Refuse(DeadlockCause::OwnQueue)
            };
            assert_eq!(slot_wait(marker(step), own, open), answer);
        }
    }
}

/// A submission handler: its command buffer's scheduling, and every later transition of its
/// queue, waits for it, and on the scheduler thread so do the device's later submissions. A
/// completion-seen handler: its command buffer's completion, and every later transition of its
/// queue, waits for it, and on the thread that sees command buffers complete so do every queue's
/// completions. So a wait in either for anything that has not happened yet, of any queue or device,
/// is refused, each with its own cause; one for what happened returns.
#[test]
fn a_submission_or_completion_seen_handler_s_waits_for_what_has_not_happened_are_refused() {
    for (step, cause) in [
        (Step::SubmissionHandlers, DeadlockCause::SubmissionHandler),
        (
            Step::CompletionSeenHandlers,
            DeadlockCause::CompletionSeenHandler,
        ),
    ] {
        let refused = Answer::Refuse(cause);
        let running = marker(step);
        for at in [
            target(DEVICE, QUEUE, OWN),
            target(DEVICE, QUEUE, LATER),
            target(DEVICE, OTHER_QUEUE, LATER),
            target(OTHER_DEVICE, QUEUE, LATER),
        ] {
            assert_eq!(completed_wait(running, at, false), refused);
            assert_eq!(scheduled_wait(running, at, false), refused);
            assert_eq!(completed_wait(running, at, true), Answer::Wait);
            assert_eq!(scheduled_wait(running, at, true), Answer::Wait);
        }
        for queue in [
            context(DEVICE, QUEUE),
            context(DEVICE, OTHER_QUEUE),
            context(OTHER_DEVICE, QUEUE),
        ] {
            assert_eq!(slot_wait(running, queue, false), refused);
            assert_eq!(slot_wait(running, queue, true), Answer::Wait);
        }
    }
}
