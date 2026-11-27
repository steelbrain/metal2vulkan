//! The orders a command buffer's completion and completed handlers keep, and a commit's time,
//! under a device cfg on every device of the cfg's kind.
//!
//! - **Across queues, after a GPU wait.** A command buffer whose work waited on the GPU for an
//!   event another queue's command buffer signals at its end completes after it: the signaller's
//!   status reads `Completed` once the waiter's wait returns and in each of the waiter's handlers,
//!   and the signaller's first handler starts before any of the waiter's.
//! - **Within a command buffer, a queue, and across queues.** A command buffer's status reads
//!   `Completed` in its handlers, its first handler returns before its second starts, and
//!   `wait_until_completed` returns after its handlers returned; an earlier command buffer's
//!   handlers returned before a later one's of its queue start; and a command buffer that waited on
//!   the GPU for another queue's signal starts its handlers after the signaller's first returned,
//!   reading it `Completed`.
//! - **The commit.** Metal's `commit` returns at once, and so does the runtime's while other queues
//!   are in use: no commit of a run's 1,800, less the committing thread's wait for a CPU where the
//!   kernel counts it, takes [`STALL`] or more.
//!
//! Metal has no completion-seen handler: where a shape names one, it is a completed handler added
//! before the command buffer's other completed handler, so it is the first to run at its
//! completion.

use crate::tests::command::on_device::{each_device, finished, PATIENCE};
use crate::{
    host_time, Block, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, Event,
    MemoryClass,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The rounds of each shape of [`a_command_buffer_that_waited_for_another_queue_completes_after_it`].
const ROUNDS: u32 = 50;

/// The bytes the signaller fills: enough that its work is still running as the waiter's batch is
/// submitted, and the waiter's batch, a fill of 256 bytes, completes just after it.
const FILL: u64 = 16 << 20;

fn queue(device: &Device) -> CommandQueue {
    device.new_command_queue(64).expect("a queue")
}

/// A Private buffer of `length` bytes over a block of its own, which it is returned with.
fn private(device: &Device, length: u64) -> (Block, Buffer) {
    let block = (device.take_block(MemoryClass::Private, length)).expect("a Private block");
    let buffer = (device.new_buffer(block.id, 0, length, None)).expect("a buffer");
    (block, buffer)
}

/// Records a fill of the whole of `buffer` with `value` in a blit encoder of `command_buffer`.
fn blit_fill(command_buffer: &CommandBuffer, buffer: &Buffer, value: u8) {
    let mut blit = command_buffer
        .blit_command_encoder()
        .expect("a blit encoder");
    (blit.fill_buffer(buffer, 0..buffer.length(), value)).expect("a fill");
    blit.end_encoding();
}

/// Waits until `lines` reads `expected` or more, for at most [`PATIENCE`].
fn wait_for_lines(name: &str, expected: usize, lines: impl Fn() -> usize) {
    let deadline = Instant::now() + PATIENCE;
    while lines() < expected {
        assert!(Instant::now() < deadline, "{name}: a handler never ran");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The handlers a command buffer of a shape carries.
#[derive(Clone, Copy, Debug)]
enum Handlers {
    /// A completed handler.
    Completed,
    /// A completion-seen handler alone.
    Seen,
    /// Both.
    Both,
}

impl Handlers {
    fn completed(self) -> bool {
        matches!(self, Handlers::Completed | Handlers::Both)
    }

    fn seen(self) -> bool {
        matches!(self, Handlers::Seen | Handlers::Both)
    }

    /// How many handlers it adds.
    fn count(self) -> usize {
        usize::from(self.completed()) + usize::from(self.seen())
    }
}

/// What a round's handlers saw, in the order they started: each one's command buffer, `'a'` or
/// `'b'`, whether it is a completion-seen handler, and the signaller's status as it started.
type Log = Arc<Mutex<Vec<(char, bool, CommandBufferStatus)>>>;

/// Add `handlers` to `buffer`, each logging `name` and `signaller`'s status as it starts: the
/// completion-seen one first.
fn log_handlers(
    buffer: &CommandBuffer,
    name: char,
    handlers: Handlers,
    signaller: &CommandBuffer,
    log: &Log,
) {
    for (wanted, seen) in [(handlers.seen(), true), (handlers.completed(), false)] {
        if !wanted {
            continue;
        }
        let (log, signaller) = (Arc::clone(log), signaller.clone());
        buffer
            .add_completed_handler(move |_| {
                let status = signaller.status();
                crate::device::lock(&log).push((name, seen, status));
            })
            .expect("a completed handler");
    }
}

/// The signaller: a fill of `big` on `queue`, then a signal of `event` to `value`.
fn signaller(queue: &CommandQueue, big: &Buffer, event: &Event, value: u64) -> CommandBuffer {
    let a = queue.command_buffer().expect("a command buffer");
    blit_fill(&a, big, value as u8);
    (a.encode_signal_event(event, value)).expect("a signal");
    a
}

/// The waiter: a wait for `event` to reach `value` on `queue`, then a fill of `word`.
fn waiter(queue: &CommandQueue, word: &Buffer, event: &Event, value: u64) -> CommandBuffer {
    let b = queue.command_buffer().expect("a command buffer");
    (b.encode_wait_for_event(event, value)).expect("a wait");
    blit_fill(&b, word, value as u8);
    b
}

/// Each commit order, kind of event and order of the two queues: B committed first, the event a
/// shared one and A on the queue made second.
fn shapes() -> impl Iterator<Item = (bool, bool, bool)> {
    let both = [false, true];
    both.into_iter().flat_map(move |b_first| {
        both.into_iter().flat_map(move |shared| {
            both.into_iter()
                .map(move |swapped| (b_first, shared, swapped))
        })
    })
}

/// What a round broke of Metal's order.
#[derive(Default)]
struct Broken {
    /// A's status was short of `Completed` as B's wait returned or one of B's handlers started.
    status: u32,
    /// One of B's handlers started before A's completion-seen handler.
    seen: u32,
    /// One of B's handlers started before A's completed handler, A having no completion-seen
    /// handler (counted, not a break).
    completed: u32,
}

/// One round: A on `of_a` fills `big` and signals `event` to `value` at its end, and B on `of_b`
/// waits for it on the GPU and fills `word`, each with its handlers, committed in that order or B
/// first. What it broke is added to `broken`.
fn round(
    name: &str,
    (of_a, of_b): (&CommandQueue, &CommandQueue),
    (big, word): (&Buffer, &Buffer),
    (event, value): (&Event, u64),
    (a_handlers, b_handlers): (Handlers, Handlers),
    (b_first, broken): (bool, &mut Broken),
) {
    let log: Log = Arc::default();
    let a = signaller(of_a, big, event, value);
    let b = waiter(of_b, word, event, value);
    log_handlers(&a, 'a', a_handlers, &a, &log);
    log_handlers(&b, 'b', b_handlers, &a, &log);
    let order = match b_first {
        false => [&a, &b],
        true => [&b, &a],
    };
    for buffer in order {
        buffer.commit().expect("a commit");
    }
    b.wait_until_completed().expect("a wait outside a handler");
    let a_completed = a.status() == CommandBufferStatus::Completed;
    a.wait_until_completed().expect("a wait outside a handler");
    // A wait returns once the handlers ran; the log is read once it holds every line.
    let expected = a_handlers.count() + b_handlers.count();
    wait_for_lines(name, expected, || crate::device::lock(&log).len());
    let log = crate::device::lock(&log);
    let completed = |&&(name, _, status): &&(char, bool, CommandBufferStatus)| {
        name == 'a' || status == CommandBufferStatus::Completed
    };
    broken.status += u32::from(!a_completed || !log.iter().all(|entry| completed(&entry)));
    let a_first = log[0].0 == 'a';
    match a_handlers.seen() {
        true => broken.seen += u32::from(!a_first || !log[0].1),
        false => broken.completed += u32::from(!a_first),
    }
}

/// Over [`ROUNDS`] rounds of each shape (the handlers of A and of B, and each of [`shapes`]), A on
/// one queue fills [`FILL`] bytes and signals the event at its end, and B on another waits for it
/// on the GPU ([`round`]): A's status reads `Completed` once B's wait returns and in each of B's
/// handlers, and A's completion-seen handler starts before any of B's.
#[test]
fn a_command_buffer_that_waited_for_another_queue_completes_after_it() {
    each_device("completion_order", |name, _, device| {
        let (first, second) = (queue(&device), queue(&device));
        let (big_block, big) = private(&device, FILL);
        let (word_block, word) = private(&device, 256);
        let shared = device.new_shared_event().expect("a shared event");
        let plain = device.new_event().expect("an event");
        let mut value = 0;
        let (mut overtaken, mut dispatched) = (Vec::new(), 0);
        for handlers in [
            (Handlers::Both, Handlers::Seen),
            (Handlers::Completed, Handlers::Seen),
            (Handlers::Completed, Handlers::Completed),
            (Handlers::Seen, Handlers::Seen),
            (Handlers::Both, Handlers::Both),
        ] {
            for (b_first, on_shared, swapped) in shapes() {
                let event = match on_shared {
                    true => shared.as_event(),
                    false => &plain,
                };
                let queues = match swapped {
                    false => (&first, &second),
                    true => (&second, &first),
                };
                let mut broken = Broken::default();
                for _ in 0..ROUNDS {
                    value += 1;
                    let shape = (b_first, &mut broken);
                    round(name, queues, (&big, &word), (event, value), handlers, shape);
                }
                dispatched += broken.completed;
                if broken.status + broken.seen > 0 {
                    overtaken.push(format!(
                        "A {:?}, B {:?}, B committed first {b_first}, shared event {on_shared}, \
                         queues swapped {swapped}: A not completed in {}, A's completion-seen \
                         handler after B's in {} of {ROUNDS}",
                        handlers.0, handlers.1, broken.status, broken.seen
                    ));
                }
            }
        }
        println!("{name}: B's handler first of two dispatched {dispatched}");
        assert!(
            overtaken.is_empty(),
            "{name}: B's handlers overtook A's: {overtaken:#?}"
        );
        drop((
            big, word, big_block, word_block, shared, plain, first, second, device,
        ));
    });
}

/// The rounds of each shape of
/// [`handlers_run_in_their_command_buffer_s_queue_s_and_signaller_s_order`].
const ORDER_ROUNDS: u64 = 40;

/// How long A's completed handler holds its thread.
const HOLD: Duration = Duration::from_millis(3);

/// One command buffer's log: the counter's value where each handler started and returned, and
/// where the test's wait for it returned; and the status it read of itself in its completion-seen
/// handler, and of `of` (A) in it.
#[derive(Clone, Copy, Debug, Default)]
struct Steps {
    seen: (u64, u64),
    completed: (u64, u64),
    waited: u64,
    own_status: Option<CommandBufferStatus>,
    a_status: Option<CommandBufferStatus>,
}

/// The shared counter and each command buffer's log.
#[derive(Default)]
struct OrderLog {
    counter: AtomicU64,
    steps: Mutex<[Steps; 3]>,
}

impl OrderLog {
    fn tick(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::SeqCst) + 1
    }
}

/// A fill of `buffer` on `queue`, after a wait for `wait`'s event to reach its value where given,
/// with a completion-seen handler and a completed handler that log to `log` as command buffer
/// `index`, the completed one holding its thread `hold`; its completion-seen handler reads `a`'s
/// status too, where given.
fn logged(
    queue: &CommandQueue,
    buffer: &Buffer,
    (log, index): (&Arc<OrderLog>, usize),
    (hold, wait): (Duration, Option<(&Event, u64)>),
    a: Option<&CommandBuffer>,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().expect("a command buffer");
    if let Some((event, value)) = wait {
        (command_buffer.encode_wait_for_event(event, value)).expect("a wait");
    }
    blit_fill(&command_buffer, buffer, index as u8);
    let (seen_log, a) = (Arc::clone(log), a.cloned());
    command_buffer
        .add_completed_handler(move |own| {
            let start = seen_log.tick();
            let own_status = own.status();
            let a_status = a.as_ref().map(CommandBuffer::status);
            let end = seen_log.tick();
            let mut steps = crate::device::lock(&seen_log.steps);
            steps[index].seen = (start, end);
            steps[index].own_status = Some(own_status);
            steps[index].a_status = a_status;
        })
        .expect("a completed handler");
    let completed_log = Arc::clone(log);
    command_buffer
        .add_completed_handler(move |_| {
            let start = completed_log.tick();
            std::thread::sleep(hold);
            let end = completed_log.tick();
            crate::device::lock(&completed_log.steps)[index].completed = (start, end);
        })
        .expect("a completed handler");
    command_buffer
}

/// What a round broke of the orders (the module docs), by order.
#[derive(Debug, Default, PartialEq, Eq)]
struct OrderBroken {
    /// A status short of `Completed` in its own completion-seen handler, a completion-seen
    /// handler not returned before its completed handler started, or a wait that returned before
    /// its completed handler did.
    within: u32,
    /// B's completion-seen handler started before A's completed handler returned.
    queue: u32,
    /// C's completion-seen handler started before A's returned, or read A short of `Completed`.
    across: u32,
}

/// One round on `first` (A, B) and, with `gpu_wait`, `second` (C, waiting for A's signal of
/// `event` to `value`).
fn order_round(
    (first, second): (&CommandQueue, &CommandQueue),
    buffer: &Buffer,
    gpu_wait: Option<(&Event, u64)>,
    broken: &mut OrderBroken,
) {
    let log = Arc::new(OrderLog::default());
    let a = logged(first, buffer, (&log, 0), (HOLD, None), None);
    if let Some((event, value)) = gpu_wait {
        (a.encode_signal_event(event, value)).expect("a signal");
    }
    let b = logged(first, buffer, (&log, 1), (Duration::ZERO, None), None);
    let c = gpu_wait.map(|wait| {
        let shape = (Duration::ZERO, Some(wait));
        logged(second, buffer, (&log, 2), shape, Some(&a))
    });
    a.commit().expect("a commit");
    b.commit().expect("a commit");
    if let Some(c) = &c {
        c.commit().expect("a commit");
    }
    let buffers: Vec<(usize, &CommandBuffer)> = [Some(&a), Some(&b), c.as_ref()]
        .into_iter()
        .enumerate()
        .filter_map(|(index, buffer)| buffer.map(|buffer| (index, buffer)))
        .collect();
    for &(index, buffer) in &buffers {
        buffer
            .wait_until_completed()
            .expect("a wait outside a handler");
        let waited = log.tick();
        crate::device::lock(&log.steps)[index].waited = waited;
    }
    let steps = *crate::device::lock(&log.steps);
    for &(index, _) in &buffers {
        let s = steps[index];
        let ordered = s.own_status == Some(CommandBufferStatus::Completed)
            && s.seen.1 < s.completed.0
            && s.completed.1 < s.waited;
        broken.within += u32::from(!ordered);
    }
    let (a_steps, b_steps) = (steps[0], steps[1]);
    broken.queue += u32::from(b_steps.seen.0 < a_steps.completed.1);
    if c.is_some() {
        let c_steps = steps[2];
        let after = c_steps.seen.0 > a_steps.seen.1
            && c_steps.a_status == Some(CommandBufferStatus::Completed);
        broken.across += u32::from(!after);
    }
}

/// Over [`ORDER_ROUNDS`] rounds with no GPU wait and as many with one, every order of the module
/// docs holds: two queues; each round commits A then B on the first, A's completed handler holding
/// its thread [`HOLD`] so that B completes meanwhile, and, in the rounds with a GPU wait, C on the
/// second, which waits on the GPU for an event A signals at its end. Every handler logs where it
/// starts and returns on one counter, and each wait's return is logged after it.
#[test]
fn handlers_run_in_their_command_buffer_s_queue_s_and_signaller_s_order() {
    each_device("handler_orders", |name, _, device| {
        let (first, second) = (queue(&device), queue(&device));
        let (block, buffer) = private(&device, 4096);
        let event = device.new_event().expect("an event");
        let mut broken = OrderBroken::default();
        for value in 1..=ORDER_ROUNDS {
            order_round((&first, &second), &buffer, None, &mut broken);
            let wait = Some((&event, value));
            order_round((&first, &second), &buffer, wait, &mut broken);
        }
        println!("{name}: HANDLER-ORDERS {broken:?}");
        assert_eq!(
            broken,
            OrderBroken::default(),
            "{name}: of {} rounds",
            2 * ORDER_ROUNDS
        );
        drop((buffer, block, event, first, second, device));
    });
}

/// How many rounds the busy queue runs in [`a_commit_returns_while_other_queues_are_waited_for`].
const COMMIT_ROUNDS: usize = 600;

/// The command buffers of a round, committed back to back.
const IN_FLIGHT: usize = 3;

/// The fills' sizes, in turn over the rounds: their GPU times move each signal's moment.
const SIZES: [u64; 7] = [
    4 << 10,
    16 << 10,
    64 << 10,
    256 << 10,
    1 << 20,
    4 << 20,
    8 << 20,
];

/// The longest a commit may take, less the committing thread's wait for a CPU.
const STALL: Duration = Duration::from_millis(50);

/// Four queues, three of them idle and in use, the fourth the one each round commits on, and the
/// buffer the rounds fill: [`CommitRounds::round`] runs one round.
struct CommitRounds {
    queues: Vec<CommandQueue>,
    buffer: Buffer,
    _block: Block,
}

impl CommitRounds {
    fn new(name: &str, device: &Device) -> CommitRounds {
        let (block, buffer) = private(device, SIZES[SIZES.len() - 1]);
        let queues: Vec<CommandQueue> = (0..4).map(|_| queue(device)).collect();
        let rounds = CommitRounds {
            queues,
            buffer,
            _block: block,
        };
        // Each idle queue runs one first.
        for (value, queue) in rounds.queues[1..].iter().enumerate() {
            let command_buffer = rounds.fill(queue, SIZES[0], value as u8);
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
        rounds
    }

    /// A short command buffer of `queue`: one fill of `size` bytes of `value`.
    fn fill(&self, queue: &CommandQueue, size: u64, value: u8) -> CommandBuffer {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut blit = command_buffer
            .blit_command_encoder()
            .expect("a blit encoder");
        (blit.fill_buffer(&self.buffer, 0..size, value)).expect("a fill");
        blit.end_encoding();
        command_buffer
    }

    /// Round `round`: [`IN_FLIGHT`] fills of its size committed back to back on the busy queue,
    /// each with a completed handler that notes the time, then a wait for each; each commit's
    /// time and the committing thread's wait for a CPU meanwhile ([`timed_commit`]).
    fn round(&self, name: &str, round: usize) -> Vec<(Duration, Duration)> {
        let size = SIZES[round % SIZES.len()];
        let command_buffers: Vec<(CommandBuffer, Arc<Mutex<Option<u64>>>)> = (0..IN_FLIGHT)
            .map(|_| {
                let command_buffer = self.fill(&self.queues[0], size, round as u8);
                let handled = Arc::new(Mutex::new(None));
                let noted = Arc::clone(&handled);
                command_buffer
                    .add_completed_handler(move |_| {
                        *crate::device::lock(&noted) = Some(host_time());
                    })
                    .expect("a completed handler");
                (command_buffer, handled)
            })
            .collect();
        let commits: Vec<(Duration, Duration)> = command_buffers
            .iter()
            .map(|(command_buffer, _)| timed_commit(command_buffer))
            .collect();
        for (command_buffer, handled) in &command_buffers {
            finished(name, command_buffer);
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert!(
                crate::device::lock(handled).is_some(),
                "{name}: its completed handler ran before its wait returned"
            );
        }
        commits
    }
}

/// The longest commit of a round's.
fn longest(commits: &[(Duration, Duration)]) -> Duration {
    (commits.iter().map(|&(commit, _)| commit).max()).expect("a round's commits")
}

/// `command_buffer`'s `commit`, timed: its wall time less the committing thread's wait for a CPU
/// meanwhile, and that wait. Linux counts each thread's wait: the second field of
/// `/proc/thread-self/schedstat`, "time spent waiting on a runqueue" in nanoseconds (the kernel's
/// `Documentation/scheduler/sched-stats.rst`); where it is not counted, the time is the wall time.
/// The instant is read before the count at the start and after it at the end, so a wait between
/// the two reads stays in the time, which never reads less than the call's time on a CPU and
/// blocked: a wait for a lock stays in it.
fn timed_commit(command_buffer: &CommandBuffer) -> (Duration, Duration) {
    let start = Instant::now();
    let before = run_queue_wait();
    command_buffer.commit().expect("a commit");
    let after = run_queue_wait();
    let wall = start.elapsed();
    let queued = match (before, after) {
        (Some(before), Some(after)) => after.saturating_sub(before),
        _ => Duration::ZERO,
    };
    (wall.saturating_sub(queued), queued)
}

/// The calling thread's wait for a CPU so far, where the kernel counts it ([`timed_commit`]).
fn run_queue_wait() -> Option<Duration> {
    let stat = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    let waited = stat.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(Duration::from_nanos(waited))
}

/// Over [`COMMIT_ROUNDS`] rounds of [`IN_FLIGHT`] fills committed back to back on one of four
/// queues, the other three idle and in use, no commit takes [`STALL`] or more, less the
/// committing thread's wait for a CPU where the kernel counts it. Where it counts none (macOS: no
/// `/proc`), a commit's time is its wall time, the host's preemption of the thread in it: there a
/// round with a commit that slow runs again, timed afresh, and the test fails only where a commit
/// of the round run again takes that long too. A summary of the commits' times and their waits
/// for a CPU is printed (`ROUNDS`), and each round with a commit that long (`SLOW-ROUND`).
#[test]
fn a_commit_returns_while_other_queues_are_waited_for() {
    each_device("commit_returns", |name, _, device| {
        let runner = CommitRounds::new(name, &device);
        let rounds: Vec<Vec<(Duration, Duration)>> = (0..COMMIT_ROUNDS)
            .map(|round| runner.round(name, round))
            .collect();
        for (n, commits) in rounds.iter().enumerate() {
            if longest(commits) >= STALL {
                println!("{name}: SLOW-ROUND {n}: (commit, wait for a CPU) {commits:?}");
            }
        }
        let summary = |mut sorted: Vec<Duration>| {
            sorted.sort();
            let at = |percent: usize| sorted[(sorted.len() - 1) * percent / 100];
            format!(
                "p50 {:?} p90 {:?} p99 {:?} max {:?}",
                at(50),
                at(90),
                at(99),
                sorted[sorted.len() - 1]
            )
        };
        let all = || rounds.iter().flatten();
        println!(
            "{name}: ROUNDS {COMMIT_ROUNDS}: commit time less the wait for a CPU: {}, {} at or \
             over {STALL:?}; the commits' waits for a CPU: {}",
            summary(all().map(|&(commit, _)| commit).collect()),
            all().filter(|&&(commit, _)| commit >= STALL).count(),
            summary(all().map(|&(_, queued)| queued).collect()),
        );
        // Where the kernel counts no wait for a CPU, a round with a commit that slow runs again,
        // timed afresh: held only where a commit takes that long again.
        let counted = run_queue_wait().is_some();
        let held: Vec<(usize, Duration, Option<Duration>)> = (rounds.iter().enumerate())
            .filter(|(_, commits)| longest(commits) >= STALL)
            .map(|(n, commits)| {
                let again = (!counted).then(|| longest(&runner.round(name, n)));
                (n, longest(commits), again)
            })
            .filter(|&(_, _, again)| again.is_none_or(|again| again >= STALL))
            .collect();
        assert!(
            held.is_empty(),
            "{name}: commits that took {STALL:?} or more, less the committing thread's wait for \
             a CPU where the kernel counts it, and again in the round run again where it does \
             not, where Metal's returns at once (round, longest, longest again): {held:?}. A \
             commit hands its command buffer to the submit thread: one this slow waited for a \
             lock or worked that long, not for a submission"
        );
        drop((runner, device));
    });
}
