//! The scheduler's rule for the caller's binary semaphores (`scheduler/binary.rs`): a wait is
//! submitted only once every signal of its semaphore that this device's command buffers committed
//! before it is submitted, with the scheduler's order (`scheduler/order.rs`) choosing among the
//! ready heads, and the table holds only what is pending.

use crate::scheduler::binary::{BinarySignals, Needs};
use crate::scheduler::order::Order;

type Semaphore = u8;

const ACQUIRED: Semaphore = 1;
const RENDERED: Semaphore = 2;

#[test]
fn a_wait_for_the_callers_own_signal_is_ready_at_once() {
    // An acquire, or the caller's own submission, was submitted before the commit.
    let mut table = BinarySignals::<Semaphore>::default();
    let needs = table.commit(&[ACQUIRED], &[]);
    assert_eq!(needs, [(ACQUIRED, 0)]);
    assert!(table.ready(&needs));
    table.submitted(&needs, &[]);
    assert_eq!(table.len(), 0);
}

#[test]
fn a_wait_after_a_committed_signal_holds_until_it_is_submitted() {
    let mut table = BinarySignals::<Semaphore>::default();
    let signal = table.commit(&[], &[RENDERED]);
    assert_eq!(signal, []);
    let wait = table.commit(&[RENDERED], &[]);
    assert_eq!(wait, [(RENDERED, 1)]);
    assert!(!table.ready(&wait));
    table.submitted(&signal, &[RENDERED]);
    assert!(table.ready(&wait));
    // The signal's count stays while a wait counts on it.
    assert_eq!(table.len(), 1);
    table.submitted(&wait, &[]);
    assert_eq!(table.len(), 0);
}

#[test]
fn a_batch_waits_before_it_signals() {
    // One batch that waits for a semaphore and signals it again counts only earlier signals.
    let mut table = BinarySignals::<Semaphore>::default();
    let first = table.commit(&[RENDERED], &[RENDERED]);
    assert_eq!(first, [(RENDERED, 0)]);
    assert!(table.ready(&first));
    let second = table.commit(&[RENDERED], &[]);
    assert_eq!(second, [(RENDERED, 1)]);
    assert!(!table.ready(&second));
    table.submitted(&first, &[RENDERED]);
    assert!(table.ready(&second));
    table.submitted(&second, &[]);
    assert_eq!(table.len(), 0);
}

#[test]
fn each_wait_counts_the_signals_committed_before_it() {
    let mut table = BinarySignals::<Semaphore>::default();
    let a = table.commit(&[], &[RENDERED]);
    let wait_a = table.commit(&[RENDERED], &[]);
    let b = table.commit(&[], &[RENDERED]);
    let wait_b = table.commit(&[RENDERED], &[]);
    assert_eq!(
        (wait_a.clone(), wait_b.clone()),
        (vec![(RENDERED, 1)], vec![(RENDERED, 2)])
    );
    table.submitted(&a, &[RENDERED]);
    assert!(table.ready(&wait_a) && !table.ready(&wait_b));
    table.submitted(&wait_a, &[]);
    table.submitted(&b, &[RENDERED]);
    assert!(table.ready(&wait_b));
    table.submitted(&wait_b, &[]);
    assert_eq!(table.len(), 0);
}

#[test]
fn a_failed_batchs_signal_never_comes() {
    // Its waits leave; a later wait for its signal stays held, as it would on Vulkan.
    let mut table = BinarySignals::<Semaphore>::default();
    let failed = table.commit(&[ACQUIRED], &[RENDERED]);
    let wait = table.commit(&[RENDERED], &[]);
    table.dropped(&failed);
    assert!(!table.ready(&wait));
    table.dropped(&wait);
    // The signal stays committed and unsubmitted.
    assert_eq!(table.len(), 1);
}

/// A batch held in the order: its name, what its binary waits need, and what it signals.
type Held = (&'static str, Needs<Semaphore>, Vec<Semaphore>);

/// Submit every ready head as the scheduler does, with the batches named in `events_unmet`
/// held by their event waits: the names, in submission order.
fn schedule(
    order: &mut Order<Held>,
    table: &mut BinarySignals<Semaphore>,
    events_unmet: &[&str],
) -> Vec<&'static str> {
    let mut submitted = Vec::new();
    loop {
        let ready = &*table;
        let Some((_, _, (name, needs, signals), ())) = order.next(
            |_| false,
            |_, _| false,
            |(name, needs, _)| (ready.ready(needs) && !events_unmet.contains(name)).then_some(()),
        ) else {
            return submitted;
        };
        table.submitted(&needs, &signals);
        submitted.push(name);
    }
}

#[test]
fn a_wait_on_one_queue_follows_a_signal_held_on_another() {
    // Queue 1's first batch waits for an event, and its second signals RENDERED; queue 2
    // committed a wait for RENDERED after them. Queue 2's wait is held until the signal is
    // submitted, though its queue's head is otherwise free.
    let mut order = Order::default();
    let mut table = BinarySignals::<Semaphore>::default();
    let event_wait = table.commit(&[], &[]);
    let signal = table.commit(&[], &[RENDERED]);
    order.push(
        1,
        [
            ("event wait", event_wait, Vec::new()),
            ("signal", signal, vec![RENDERED]),
        ],
    );
    let wait = table.commit(&[RENDERED], &[]);
    order.push(2, [("wait", wait, Vec::new())]);
    assert_eq!(
        schedule(&mut order, &mut table, &["event wait"]),
        Vec::<&str>::new()
    );
    assert_eq!(order.len(), 3);
    assert_eq!(
        schedule(&mut order, &mut table, &[]),
        ["event wait", "signal", "wait"]
    );
    assert_eq!(table.len(), 0);
}

#[test]
fn a_present_chain_submits_in_order() {
    // A frame waits for the acquire, renders, and signals RENDERED, which the caller's present
    // waits for once it is scheduled. A command buffer on another queue committed after it
    // waits for RENDERED instead (the caller's own copy), and is pushed to the order first.
    let mut order = Order::default();
    let mut table = BinarySignals::<Semaphore>::default();
    let render = table.commit(&[ACQUIRED], &[RENDERED]);
    let copy = table.commit(&[RENDERED], &[]);
    order.push(2, [("copy", copy, Vec::new())]);
    order.push(1, [("render", render, vec![RENDERED])]);
    assert_eq!(schedule(&mut order, &mut table, &[]), ["render", "copy"]);
    assert_eq!(table.len(), 0);
}
