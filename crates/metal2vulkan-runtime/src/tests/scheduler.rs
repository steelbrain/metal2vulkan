//! The scheduler's order (`scheduler/order.rs`) with the events' timelines (`event_timeline.rs`),
//! run against a model of `VkQueue`s that stall at any wait not yet met: logical queues on one
//! `VkQueue` never deadlock, signals of one event from several queues end at the highest value,
//! and host sets racing pending GPU signals never deadlock.
//!
//! An event the program's host sets is a shared event ([`Model::sharing`]), whose waits are held
//! until the value is reached rather than handed to Vulkan (`host_settable_event_waits`); the
//! rest are plain events, whose waits chain on the `VkQueue`. A batch held for either reason goes
//! once what it waits for arrives, which in the runtime wakes the scheduler ([`EventShared::set`]
//! and [`EventShared::reached`], `event.rs`) and here makes a pass again ([`Model::host_set`],
//! [`Model::step`]).
//!
//! [`Order::take_first`], which a thread about to block submits with
//! (`scheduler::submit_plain_ready`), is tested here too: it takes the head committed first of
//! all and stops at it, so it passes no head over and needs no answer of
//! [`Order::is_behind`](crate::scheduler::order::Order::is_behind).
//!
//! The model is the worst case Vulkan allows: each `VkQueue` runs its batches strictly in
//! submission order, and a batch at its head whose waits are not met stalls every batch behind
//! it. A batch's work and signals happen together. Every program here has an execution that
//! completes (each wait is for a value signaled earlier in that execution, or set by the host),
//! committed in any order.

use crate::event_timeline::{EventTimeline, Signal, WaitFor};
use crate::scheduler::order::{Behind, Order};
use std::cell::Cell;
use std::collections::VecDeque;

/// One batch of a command buffer: the events it waits for and signals, with the values.
#[derive(Clone, Debug, Default)]
struct Spec {
    waits: Vec<(usize, u64)>,
    signals: Vec<(usize, u64)>,
}

fn waits(waits: &[(usize, u64)]) -> Spec {
    Spec {
        waits: waits.to_vec(),
        signals: Vec::new(),
    }
}

fn signals(signals: &[(usize, u64)]) -> Spec {
    Spec {
        waits: Vec::new(),
        signals: signals.to_vec(),
    }
}

/// A batch as a `VkQueue` holds it: Vulkan waits and signals by signal index.
#[derive(Debug)]
struct Queued {
    id: usize,
    /// The waits for the events (the work's) and for an event's earlier signal (the end's).
    waits: Vec<(usize, u64)>,
    signals: Vec<(usize, u64)>,
}

/// The scheduler and the device it submits to.
struct Model {
    events: Vec<EventTimeline<usize>>,
    /// Each event's semaphore counter.
    counters: Vec<u64>,
    /// The highest signal index kept per event: no Vulkan wait may be above it.
    kept: Vec<u64>,
    order: Order<(usize, Spec)>,
    /// The `VkQueue` of each logical queue.
    vk_queue_of: Vec<usize>,
    vk_queues: Vec<VecDeque<Queued>>,
    next: usize,
    /// Batch identities, in submission order.
    submitted: Vec<usize>,
    /// The Vulkan waits each batch was submitted with, by identity.
    submitted_waits: Vec<(usize, Vec<(usize, u64)>)>,
    executed: Vec<usize>,
}

impl Model {
    /// `events` events; logical queue `i` on `VkQueue` `vk_queue_of[i]`.
    fn new(events: usize, vk_queue_of: &[usize]) -> Model {
        let vk_queues = vk_queue_of.iter().max().map_or(0, |&last| last + 1);
        Model {
            events: (0..events).map(|_| EventTimeline::new(false)).collect(),
            counters: vec![0; events],
            kept: vec![0; events],
            order: Order::default(),
            vk_queue_of: vk_queue_of.to_vec(),
            vk_queues: (0..vk_queues).map(|_| VecDeque::new()).collect(),
            next: 0,
            submitted: Vec::new(),
            submitted_waits: Vec::new(),
            executed: Vec::new(),
        }
    }

    /// The events the program's host sets are shared events (`MTLSharedEvent`): only their value
    /// may be set.
    fn sharing(mut self, shared: &[usize]) -> Model {
        for &event in shared {
            self.events[event] = EventTimeline::new(true);
        }
        self
    }

    /// Commit a command buffer of `batches` on logical queue `queue`, then schedule: the batches'
    /// identities.
    fn commit(&mut self, queue: usize, batches: Vec<Spec>) -> Vec<usize> {
        let first = self.next;
        self.next += batches.len();
        self.order
            .push(queue as u64, (first..).zip(batches).collect::<Vec<_>>());
        self.schedule();
        (first..self.next).collect()
    }

    /// The host sets `event` to `value`, then the scheduler looks again. Only a shared event's
    /// value is set ([`Model::sharing`]).
    fn host_set(&mut self, event: usize, value: u64) {
        assert!(
            self.events[event].settable(),
            "event {event} is a plain one"
        );
        self.events[event].set(value);
        self.schedule();
    }

    /// Submit every ready batch, as the scheduler's thread does.
    fn schedule(&mut self) {
        loop {
            let events = &self.events;
            let next = self.order.next(
                |_, _| false,
                |(_, spec)| {
                    let mut waits = Vec::new();
                    for &(event, value) in &spec.waits {
                        match events[event].wait(value)? {
                            WaitFor::Reached => {}
                            WaitFor::Index(index) => waits.push((event, index)),
                        }
                    }
                    Some(waits)
                },
            );
            let Some((queue, _, (id, spec), mut waits)) = next else {
                return;
            };
            for &(event, index) in &waits {
                assert!(
                    index <= self.kept[event],
                    "batch {id} waits for event {event}'s index {index}, not yet submitted"
                );
            }
            self.submitted_waits.push((id, waits.clone()));
            let vk_queue = self.vk_queue_of[queue as usize];
            let mut signaled = Vec::new();
            for &(event, value) in &spec.signals {
                if let Signal::Kept { index, after } = self.events[event].signal(value, vk_queue) {
                    self.events[event].keep(index, value, vk_queue, false);
                    self.kept[event] = index;
                    signaled.push((event, index));
                    waits.extend(after.map(|after| (event, after)));
                }
            }
            self.vk_queues[vk_queue].push_back(Queued {
                id,
                waits,
                signals: signaled,
            });
            self.submitted.push(id);
        }
    }

    /// Run the head of `vk_queue` if its waits are met: whether it ran.
    fn step(&mut self, vk_queue: usize) -> bool {
        let counters = &self.counters;
        let ready = self.vk_queues[vk_queue]
            .front()
            .is_some_and(|head| head.waits.iter().all(|&(e, i)| counters[e] >= i));
        if !ready {
            return false;
        }
        let Some(head) = self.vk_queues[vk_queue].pop_front() else {
            return false;
        };
        for (event, index) in head.signals {
            assert!(
                index > self.counters[event],
                "a timeline semaphore must be signaled above its value"
            );
            self.counters[event] = index;
            self.events[event].observe(index);
        }
        self.executed.push(head.id);
        // The signals seen let a batch held for one of their values go, as a completion turn
        // wakes the scheduler for it (`event.rs`).
        self.schedule();
        true
    }

    /// Run the device until no head can run, `pick` choosing among the queues that can.
    fn run(&mut self, pick: &mut impl FnMut(usize) -> usize) {
        loop {
            let counters = &self.counters;
            let runnable: Vec<usize> = (0..self.vk_queues.len())
                .filter(|&q| {
                    self.vk_queues[q]
                        .front()
                        .is_some_and(|head| head.waits.iter().all(|&(e, i)| counters[e] >= i))
                })
                .collect();
            if runnable.is_empty() {
                return;
            }
            let chosen = runnable[pick(runnable.len())];
            self.step(chosen);
        }
    }

    /// Whether every batch committed was submitted and ran.
    fn finished(&self) -> bool {
        self.order.is_empty()
            && self.vk_queues.iter().all(VecDeque::is_empty)
            && self.executed.len() == self.next
    }

    fn position(&self, id: usize) -> usize {
        self.submitted
            .iter()
            .position(|&submitted| submitted == id)
            .expect("submitted")
    }
}

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as usize % bound.max(1)
    }
}

/// A line's head can be updated in place (a batch's host-step gate), and the order is kept.
#[test]
fn the_heads_update_in_place() {
    let mut order = Order::default();
    order.push(1, ['a', 'b']);
    order.push(2, ['c']);
    for head in order.heads_mut(&[]) {
        *head = head.to_ascii_uppercase();
    }
    assert_eq!(
        order.next(|_, _| false, |_| Some(())),
        Some((1, 0, 'A', ()))
    );
    assert_eq!(
        order.next(|_, _| false, |_| Some(())),
        Some((1, 1, 'b', ()))
    );
    assert_eq!(
        order.next(|_, _| false, |_| Some(())),
        Some((2, 2, 'C', ()))
    );
}

#[test]
fn the_order_takes_the_ready_head_committed_first() {
    let mut order = Order::default();
    order.push(1, ['a', 'b']);
    order.push(2, ['c']);
    order.push(1, ['d']);
    let ready = |held: &mut char| (*held != 'a').then_some(());
    // Queue 1's head is held: queue 2's goes, and queue 1's later batches wait behind its head.
    assert_eq!(order.next(|_, _| false, ready), Some((2, 2, 'c', ())));
    assert_eq!(order.next(|_, _| false, ready), None);
    assert_eq!(
        order.next(|_, _| false, |_| Some(())),
        Some((1, 0, 'a', ()))
    );
    assert_eq!(
        order.next(|_, _| false, |_| Some(())),
        Some((1, 1, 'b', ()))
    );
    assert_eq!(order.drop_head(1, |held| *held == 'd'), ['d']);
    assert_eq!(order.len(), 0);
}

/// A batch taken and put back (its first submission waits for encoders a rebind encodes again)
/// is the head of its line again with its commit sequence number: it goes before its line's
/// later batches, and before another line's head committed after it, as if never taken. A line
/// it emptied comes back.
#[test]
fn a_batch_put_back_keeps_its_place() {
    let mut order = Order::default();
    order.push(1, ['a', 'b']);
    order.push(2, ['c']);
    let (queue, sequence, held, ()) = order.next(|_, _| false, |_| Some(())).unwrap();
    assert_eq!((queue, sequence, held), (1, 0, 'a'));
    order.put_back(queue, sequence, held);
    assert_eq!(order.len(), 3);
    assert_eq!(
        order.next(|_, _| false, |_| Some(())),
        Some((1, 0, 'a', ()))
    );
    let (queue, sequence, held, ()) = order
        .next(|_, _| false, |head| (*head == 'c').then_some(()))
        .unwrap();
    assert_eq!((queue, sequence, held), (2, 2, 'c'));
    order.put_back(queue, sequence, held);
    assert_eq!(
        order.next(|_, _| false, |_| Some(())),
        Some((1, 1, 'b', ()))
    );
    assert_eq!(
        order.next(|_, _| false, |_| Some(())),
        Some((2, 2, 'c', ()))
    );
    assert!(order.is_empty());
}

#[test]
fn a_wait_before_its_signal_on_one_vk_queue_never_deadlocks() {
    // Two logical queues on one VkQueue: A commits a wait for B's signal first. Submitted in
    // commit order, A's wait would stall the VkQueue ahead of B's signal.
    let mut model = Model::new(1, &[0, 0]);
    let a = model.commit(0, vec![waits(&[(0, 1)])]);
    assert!(model.submitted.is_empty());
    let b = model.commit(1, vec![signals(&[(0, 1)])]);
    assert_eq!(model.submitted, [b[0], a[0]]);
    model.run(&mut |_| 0);
    assert!(model.finished());
    assert_eq!(model.executed, [b[0], a[0]]);
}

#[test]
fn a_held_batch_holds_back_only_its_own_queue() {
    // As measured on Metal: a command buffer waiting for an event stays committed, holds the
    // later command buffers of its queue, and lets other queues run.
    let mut model = Model::new(1, &[0, 0]).sharing(&[0]);
    let held = model.commit(0, vec![waits(&[(0, 1)])]);
    let behind = model.commit(0, vec![Spec::default()]);
    let other = model.commit(1, vec![Spec::default()]);
    assert_eq!(model.submitted, other);
    model.host_set(0, 1);
    assert_eq!(model.submitted, [other[0], held[0], behind[0]]);
    // The host reached the value: no Vulkan wait.
    assert_eq!(model.submitted_waits[1], (held[0], vec![]));
    model.run(&mut |_| 0);
    assert!(model.finished());
}

#[test]
fn three_queues_ordered_by_events_as_paging_does() {
    // The consumer's paging: blit-up signals `up`, exec waits for it and signals `exec`,
    // blit-down waits for that and signals `down`; each round is committed downstream first.
    for vk_queues in [[0, 0, 0], [0, 1, 2], [0, 1, 0]] {
        let (up, exec, down) = (0, 1, 2);
        let mut model = Model::new(3, &vk_queues);
        for round in 1..=8u64 {
            model.commit(
                2,
                vec![Spec {
                    waits: vec![(exec, round)],
                    signals: vec![(down, round)],
                }],
            );
            model.commit(
                1,
                vec![Spec {
                    waits: vec![(up, round)],
                    signals: vec![(exec, round)],
                }],
            );
            model.commit(0, vec![signals(&[(up, round)])]);
            model.run(&mut |count| count - 1);
        }
        assert!(model.finished(), "{vk_queues:?}");
        for event in [up, exec, down] {
            assert_eq!(model.events[event].value(), 8);
        }
    }
}

#[test]
fn signals_of_one_event_from_two_queues_end_at_the_highest() {
    // Two queues signal 7 and 4, in both submission orders, on one and on two VkQueues: the
    // event ends at 7, as Metal's did whichever executed last (measured on the M3 and the M4).
    for vk_queues in [[0, 0], [0, 1]] {
        for seven_first in [true, false] {
            let mut model = Model::new(1, &vk_queues);
            let (first, second) = if seven_first { (7, 4) } else { (4, 7) };
            let a = model.commit(0, vec![signals(&[(0, first)])]);
            let b = model.commit(1, vec![signals(&[(0, second)])]);
            assert_eq!(model.submitted, [a[0], b[0]]);
            // Any execution order the device may take.
            for pick_last in [false, true] {
                let mut copy = Model::new(1, &vk_queues);
                copy.commit(0, vec![signals(&[(0, first)])]);
                copy.commit(1, vec![signals(&[(0, second)])]);
                copy.run(&mut |count| if pick_last { count - 1 } else { 0 });
                assert!(copy.finished());
                assert_eq!(copy.events[0].value(), 7, "{vk_queues:?} {seven_first}");
            }
            model.run(&mut |_| 0);
            assert_eq!(model.events[0].value(), 7);
        }
    }
}

#[test]
fn a_host_set_racing_a_pending_signal_never_deadlocks() {
    // Event 0's seven is submitted on VkQueue 0 but waits there for event 1, which queue 1
    // signals on VkQueue 1. The host sets event 0 to 9 meanwhile.
    for run_first in [0, 1] {
        let mut model = Model::new(2, &[0, 1, 0, 1]).sharing(&[0]);
        model.commit(1, vec![signals(&[(1, 1)])]);
        model.commit(
            0,
            vec![Spec {
                waits: vec![(1, 1)],
                signals: vec![(0, 7)],
            }],
        );
        // Event 0 is shared, so this one is held rather than handed a wait for the seven's
        // index: the host set must release it, which no Vulkan wait would.
        let early = model.commit(3, vec![waits(&[(0, 5)])]);
        if run_first == 1 {
            model.run(&mut |_| 0);
        }
        model.host_set(0, 9);
        // Submitted after: the host's floor reached 8 and 9, so no Vulkan wait at all.
        let late = model.commit(2, vec![waits(&[(0, 8)])]);
        let later = model.commit(3, vec![waits(&[(0, 9)])]);
        for id in [late[0], later[0]] {
            let (_, waits) = &model.submitted_waits[model.position(id)];
            assert_eq!(waits, &[], "a wait only the host's floor satisfies");
        }
        let (_, waits) = &model.submitted_waits[model.position(early[0])];
        assert_eq!(waits, &[], "a wait the host's set may satisfy");
        model.run(&mut |count| count - 1);
        assert!(model.finished(), "{:?}", model.vk_queues);
        assert_eq!(model.events[0].value(), 9);
    }
}

/// A program with an execution that completes.
struct Program {
    /// The command buffers, (logical queue, batches), in the reference order.
    buffers: Vec<(usize, Vec<Spec>)>,
    /// The host sets, (event, value), in the reference order.
    sets: Vec<(usize, u64)>,
    /// Each event's highest value.
    highest: Vec<u64>,
}

/// A random program: batches in a reference order, each waiting only for values that earlier
/// batches signaled or the host set, each logical queue's command buffers in that order.
fn program(rng: &mut Rng, queues: usize, events: usize) -> Program {
    let mut highest = vec![0u64; events];
    let mut buffers = Vec::new();
    let mut sets = Vec::new();
    for _ in 0..(8 + rng.below(16)) {
        if rng.below(6) == 0 {
            let event = rng.below(events);
            let value = highest[event] + rng.below(3) as u64;
            sets.push((event, value));
            highest[event] = highest[event].max(value);
            continue;
        }
        let mut batches = Vec::new();
        for _ in 0..(1 + rng.below(3)) {
            let mut spec = Spec::default();
            for _ in 0..rng.below(3) {
                let event = rng.below(events);
                spec.waits
                    .push((event, rng.below(highest[event] as usize + 1) as u64));
            }
            for _ in 0..rng.below(3) {
                let event = rng.below(events);
                // Sometimes not above the highest: Metal's maximum drops it.
                let value = (highest[event] + rng.below(4) as u64).saturating_sub(1);
                spec.signals.push((event, value));
            }
            for &(event, value) in &spec.signals {
                highest[event] = highest[event].max(value);
            }
            batches.push(spec);
        }
        buffers.push((rng.below(queues), batches));
    }
    Program {
        buffers,
        sets,
        highest,
    }
}

#[test]
fn random_programs_complete_on_one_to_four_vk_queues() {
    for seed in 1..=400u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let queues = 1 + rng.below(5);
        let events = 1 + rng.below(3);
        let vk_queues = 1 + rng.below(4);
        let vk_queue_of: Vec<usize> = (0..queues).map(|queue| queue % vk_queues).collect();
        let Program {
            buffers,
            sets,
            highest,
        } = program(&mut rng, queues, events);
        let shared: Vec<usize> = sets.iter().map(|&(event, _)| event).collect();
        let mut model = Model::new(events, &vk_queue_of).sharing(&shared);
        // Commit in a random interleaving of the queues (each queue's own order kept), with
        // the host sets at random points and the device running at random points.
        let mut lines: Vec<VecDeque<Vec<Spec>>> = (0..queues).map(|_| VecDeque::new()).collect();
        for (queue, batches) in buffers {
            lines[queue].push_back(batches);
        }
        let mut sets: VecDeque<(usize, u64)> = sets.into();
        loop {
            let open: Vec<usize> = (0..queues).filter(|&q| !lines[q].is_empty()).collect();
            if open.is_empty() && sets.is_empty() {
                break;
            }
            match rng.below(4) {
                0 if !sets.is_empty() => {
                    if let Some((event, value)) = sets.pop_front() {
                        model.host_set(event, value);
                    }
                }
                1 => model.run(&mut |count| rng.below(count)),
                _ if !open.is_empty() => {
                    let queue = open[rng.below(open.len())];
                    if let Some(batches) = lines[queue].pop_front() {
                        model.commit(queue, batches);
                    }
                }
                _ => {}
            }
        }
        model.run(&mut |count| rng.below(count));
        assert!(
            model.finished(),
            "seed {seed}: held {}, stalled {:?}",
            model.order.len(),
            model.vk_queues
        );
        for (event, &value) in highest.iter().enumerate() {
            assert_eq!(model.events[event].value(), value, "seed {seed}");
        }
    }
}

/// A batch for the order across queues: held by the runtime (`step`: its host step or checks)
/// or by a wait for an event the program has not reached (`event`), with the resources it reads
/// and writes.
#[derive(Clone, Copy, Debug)]
struct Work {
    name: char,
    step: bool,
    event: bool,
    reads: &'static [u32],
    writes: &'static [u32],
}

fn work(name: char, reads: &'static [u32], writes: &'static [u32]) -> Work {
    Work {
        name,
        step: false,
        event: false,
        reads,
        writes,
    }
}

/// `work` held by the runtime.
fn stepping(work: Work) -> Work {
    Work { step: true, ..work }
}

/// `work` waiting for an event the program has not reached.
fn waiting(work: Work) -> Work {
    Work {
        event: true,
        ..work
    }
}

/// What a [`Work`] reads and writes.
type Uses = (&'static [u32], &'static [u32]);

/// The batch the scheduler submits next, as `submit_ready` takes it: the heads behind an earlier
/// commit that keeps its place wait.
fn take(order: &mut Order<Work>) -> Option<char> {
    let meets = |a: &[u32], b: &[u32]| a.iter().any(|key| b.contains(key));
    let mut memo = Behind::default();
    let behind = |order: &Order<Work>, sequence| {
        order.is_behind(
            sequence,
            &mut memo,
            (
                &mut |work: &Work| !work.event,
                &mut |work: &Work| (work.reads, work.writes),
                &|&(reads, writes): &Uses, &(later_reads, later_writes): &Uses| {
                    meets(writes, later_reads)
                        || meets(writes, later_writes)
                        || meets(reads, later_writes)
                },
            ),
        )
    };
    order
        .next(behind, |work| (!work.step && !work.event).then_some(()))
        .map(|(_, _, work, ())| work.name)
}

/// The runtime's step done: the head of `queue`'s line is ready.
fn stepped(order: &mut Order<Work>, name: char) {
    for head in order.heads_mut(&[]) {
        if head.name == name {
            head.step = false;
        }
    }
}

/// A batch held for its host step keeps its place: a later commit of another queue that reads
/// what it writes waits until it is submitted, while one that touches nothing of it goes, and
/// so does a read of what it only reads (the lane's steady stream, frame 0).
#[test]
fn a_later_commit_that_conflicts_waits_behind_a_batch_held_for_its_step() {
    let mut order = Order::default();
    order.push(1, [stepping(work('a', &[3], &[1]))]);
    order.push(2, [work('b', &[1], &[])]);
    order.push(3, [work('c', &[2], &[])]);
    order.push(4, [work('d', &[3], &[])]);
    assert_eq!(take(&mut order), Some('c'));
    assert_eq!(take(&mut order), Some('d'));
    assert_eq!(take(&mut order), None);
    stepped(&mut order, 'a');
    assert_eq!(take(&mut order), Some('a'));
    assert_eq!(take(&mut order), Some('b'));
    assert_eq!(take(&mut order), None);
}

/// The heads a look found waiting behind an earlier commit, as `submit_ready` counts them
/// (`WorkCounts::heads_held_for_earlier_commits`): the numbers the look asked about and answered
/// true for ([`Behind::behind`]), each looked up on its own line ([`Order::head_mut_at`]). The
/// batch taken, then those names.
fn take_and_behind(order: &mut Order<Work>) -> (Option<char>, Vec<char>) {
    let meets = |a: &[u32], b: &[u32]| a.iter().any(|key| b.contains(key));
    let mut memo = Behind::default();
    let taken = {
        let behind = |order: &Order<Work>, sequence| {
            order.is_behind(
                sequence,
                &mut memo,
                (
                    &mut |work: &Work| !work.event,
                    &mut |work: &Work| (work.reads, work.writes),
                    &|&(reads, writes): &Uses, &(later_reads, later_writes): &Uses| {
                        meets(writes, later_reads)
                            || meets(writes, later_writes)
                            || meets(reads, later_writes)
                    },
                ),
            )
        };
        order
            .next(behind, |work| (!work.step && !work.event).then_some(()))
            .map(|(_, _, work, ())| work.name)
    };
    let mut held: Vec<char> = Vec::new();
    for sequence in memo.behind() {
        if let Some(head) = order.head_mut_at(sequence) {
            held.push(head.name);
        }
    }
    held.sort_unstable();
    (taken, held)
}

/// What the scheduler counts as held behind an earlier commit is exactly the head it passed over
/// for that reason, and nothing else: of three heads behind a batch held for its step, the one
/// that reads what it writes is named and the two that touch nothing of it are not, whether a
/// batch went in the look or none did. A head the look never asked about is never named.
#[test]
fn the_heads_counted_as_behind_are_the_ones_passed_over() {
    let mut order = Order::default();
    order.push(1, [stepping(work('a', &[3], &[1]))]);
    order.push(2, [work('b', &[1], &[])]);
    order.push(3, [work('c', &[2], &[])]);
    order.push(4, [work('d', &[3], &[])]);
    assert_eq!(take_and_behind(&mut order), (Some('c'), vec!['b']));
    assert_eq!(take_and_behind(&mut order), (Some('d'), vec!['b']));
    assert_eq!(take_and_behind(&mut order), (None, vec!['b']));
    stepped(&mut order, 'a');
    // Nothing is behind the head it was behind once that head may go.
    assert_eq!(take_and_behind(&mut order), (Some('a'), vec![]));
    assert_eq!(take_and_behind(&mut order), (Some('b'), vec![]));
}

/// A batch committed before the held one goes first, whatever it touches: only later commits
/// wait.
#[test]
fn an_earlier_commit_never_waits_behind_a_later_held_one() {
    let mut order = Order::default();
    order.push(2, [work('b', &[1], &[])]);
    order.push(1, [stepping(work('a', &[], &[1]))]);
    assert_eq!(take(&mut order), Some('b'));
    assert_eq!(take(&mut order), None);
}

/// Every batch of a held line keeps its place behind the held head, and a head waiting behind
/// one holds back later commits that conflict with it in turn.
#[test]
fn a_held_line_keeps_the_place_of_every_batch_in_it() {
    let mut order = Order::default();
    order.push(1, [stepping(work('a', &[], &[])), work('e', &[], &[1])]);
    order.push(2, [work('b', &[1], &[2])]);
    order.push(3, [work('c', &[2], &[])]);
    order.push(4, [work('d', &[4], &[])]);
    assert_eq!(take(&mut order), Some('d'));
    assert_eq!(take(&mut order), None);
    stepped(&mut order, 'a');
    let taken: Vec<_> = std::iter::from_fn(|| take(&mut order)).collect();
    assert_eq!(taken, ['a', 'e', 'b', 'c']);
}

/// A batch waiting for an event the program has not reached keeps no place, nor do the batches
/// behind it on its line: Metal schedules them after the signal, so later commits go first.
#[test]
fn a_batch_waiting_for_an_event_keeps_no_place() {
    let mut order = Order::default();
    order.push(1, [waiting(work('a', &[], &[1])), work('e', &[], &[2])]);
    order.push(2, [work('b', &[1], &[])]);
    order.push(3, [work('c', &[2], &[])]);
    assert_eq!(take(&mut order), Some('b'));
    assert_eq!(take(&mut order), Some('c'));
    assert_eq!(take(&mut order), None);
}

/// [`take`], counting the footprints taken and the batches asked whether they keep their place.
fn take_counted(order: &mut Order<Work>, footprints: &Cell<usize>, keeps: &Cell<usize>) -> bool {
    let mut memo = Behind::default();
    let behind = |order: &Order<Work>, sequence| {
        order.is_behind(
            sequence,
            &mut memo,
            (
                &mut |work: &Work| {
                    keeps.set(keeps.get() + 1);
                    !work.event
                },
                &mut |work: &Work| {
                    footprints.set(footprints.get() + 1);
                    work.writes
                },
                &|earlier: &&[u32], later: &&[u32]| earlier.iter().any(|key| later.contains(key)),
            ),
        )
    };
    (order.next(behind, |work| (!work.step && !work.event).then_some(()))).is_some()
}

/// Whether a head waits behind an earlier batch is asked of the heads that would go alone: over
/// 261 lines of 2049 batches that all write one resource, a pass whose heads all wait for an event
/// asks nothing, and draining them once ready, each in commit order, takes no footprint and asks
/// no batch whether it keeps its place, as no line holds a batch committed before the head that
/// goes. Each pass once compared every head with every earlier batch of every other line, so 16385
/// command buffers in flight over 261 queues did not drain within a minute.
#[test]
fn heads_that_go_in_commit_order_take_no_footprint() {
    let (footprints, keeps) = (Cell::new(0), Cell::new(0));
    let mut waiting_order = Order::default();
    let mut ready_order = Order::default();
    for at in 0..2049u64 {
        waiting_order.push(at % 261, [waiting(work('w', &[], &[1]))]);
        ready_order.push(at % 261, [work('r', &[], &[1])]);
    }
    assert!(!take_counted(&mut waiting_order, &footprints, &keeps));
    assert_eq!((footprints.get(), keeps.get()), (0, 0));
    let mut taken = 0;
    while take_counted(&mut ready_order, &footprints, &keeps) {
        taken += 1;
    }
    assert_eq!(taken, 2049);
    assert_eq!((footprints.get(), keeps.get()), (0, 0));
    assert!(ready_order.is_empty());
}

/// The batch a thread about to block takes next ([`Order::take_first`]): the head committed first
/// of all, and only where it is ready with nothing held for the runtime.
fn take_first(order: &mut Order<Work>) -> Option<char> {
    order
        .take_first(|work| (!work.step && !work.event).then_some(()))
        .map(|(_, _, work, ())| work.name)
}

/// A thread about to block stops at the first head in commit order it may not submit, whatever
/// else is ready: a batch held for its host step, and one waiting for an event, hold back every
/// later commit, so its submissions are in commit order and no batch that keeps its place is
/// passed over. The scheduler's own pass, which has the step and the check to run, goes on past
/// them ([`take`]).
#[test]
fn a_thread_about_to_block_stops_at_the_first_head_it_may_not_submit() {
    for held in [stepping as fn(Work) -> Work, waiting] {
        let mut order = Order::default();
        order.push(1, [work('a', &[], &[1]), work('b', &[], &[1])]);
        order.push(2, [held(work('h', &[], &[2])), work('i', &[], &[2])]);
        order.push(3, [work('c', &[], &[3])]);
        // Commit order: a, b, h, i, c.
        assert_eq!(take_first(&mut order), Some('a'));
        assert_eq!(take_first(&mut order), Some('b'));
        assert_eq!(take_first(&mut order), None, "it went past the head held");
        // The scheduler's pass takes what it can, the held line's batches keeping their place.
        assert_eq!(take(&mut order), Some('c'));
        stepped(&mut order, 'h');
        let taken: Vec<_> = std::iter::from_fn(|| take_first(&mut order)).collect();
        match held(work('h', &[], &[])).event {
            // A step recorded lets it go; the line then drains in commit order.
            false => assert_eq!(taken, ['h', 'i']),
            // The event is still not reached: nothing goes, and the scheduler's pass holds it too.
            true => assert_eq!(taken, [] as [char; 0]),
        }
    }
}

/// With nothing held, it drains every line in commit order, one batch at a time.
#[test]
fn a_thread_about_to_block_drains_the_lines_in_commit_order() {
    let mut order = Order::default();
    order.push(1, [work('a', &[], &[]), work('d', &[], &[])]);
    order.push(2, [work('b', &[], &[])]);
    order.push(1, [work('e', &[], &[])]);
    order.push(3, [work('c', &[], &[])]);
    let taken: Vec<_> = std::iter::from_fn(|| take_first(&mut order)).collect();
    assert_eq!(taken, ['a', 'd', 'b', 'e', 'c']);
    assert!(order.is_empty());
    assert_eq!(take_first(&mut order), None);
}
