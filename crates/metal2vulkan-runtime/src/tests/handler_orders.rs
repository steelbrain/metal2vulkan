//! The orders a command buffer's handlers run in, whichever thread runs them
//! (`command_buffer/mod.rs`, `CommandBuffer::add_completion_seen_handler`; `contexts/mod.rs`;
//! `command_buffer/follow.rs`), each command buffer here with a completion-seen handler and a
//! completed handler:
//!
//! 1. **Within a command buffer.** Its status reads `Completed` in its completion-seen handler,
//!    which returns before its completed handler starts, and `wait_until_completed` returns after
//!    its completed handler returned.
//! 2. **Within a queue.** An earlier command buffer's completed handler returned before a later
//!    one's completion-seen handler starts: a queue's transitions run one after another in its
//!    context, so a later one waits behind an earlier one's handlers, as Metal's does.
//! 3. **Across queues, after a GPU wait.** A command buffer whose work waited on the GPU for an
//!    event another queue's command buffer signals at its end completes after it: the signaller's
//!    status reads `Completed` in the waiter's completion-seen handler, which starts after the
//!    signaller's completion-seen handler returned.
//!
//! Two queues; each round commits A then B on the first, A's completed handler holding its thread
//! [`HOLD`] so that B completes meanwhile, and, in the rounds with a GPU wait, C on the second,
//! which waits on the GPU for an event A signals at its end. Every handler logs where it starts
//! and returns on one counter, and each wait's return is logged after it. Nothing here waits for
//! a handler to start: the orders hold whatever thread runs each (a queue's waiter, where the
//! command buffer's completion is seen, or its handler thread).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects.

use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions,
    Event, ResourceOptions,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The rounds of each shape.
const ROUNDS: u64 = 40;

/// How long A's completed handler holds its thread.
const HOLD: Duration = Duration::from_millis(3);

/// How long a test waits for a command buffer before it fails.
const WAIT: Duration = Duration::from_secs(20);

fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

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
struct Log {
    counter: AtomicU64,
    steps: Mutex<[Steps; 3]>,
}

impl Log {
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
    (log, index): (&Arc<Log>, usize),
    (hold, wait): (Duration, Option<(&Event, u64)>),
    a: Option<&CommandBuffer>,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    if let Some((event, value)) = wait {
        command_buffer
            .encode_wait_for_event(event.into(), value)
            .unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..buffer.length(), index as u8)
        .unwrap();
    blit.end_encoding().unwrap();
    let (seen_log, a) = (Arc::clone(log), a.cloned());
    command_buffer
        .add_completion_seen_handler(Box::new(move |own: &CommandBuffer| {
            let start = seen_log.tick();
            let own_status = own.status();
            let a_status = a.as_ref().map(CommandBuffer::status);
            let end = seen_log.tick();
            let mut steps = seen_log.steps.lock().unwrap();
            steps[index].seen = (start, end);
            steps[index].own_status = Some(own_status);
            steps[index].a_status = a_status;
        }))
        .unwrap();
    let completed_log = Arc::clone(log);
    command_buffer
        .add_completed_handler(Box::new(move |_: &CommandBuffer| {
            let start = completed_log.tick();
            std::thread::sleep(hold);
            let end = completed_log.tick();
            completed_log.steps.lock().unwrap()[index].completed = (start, end);
        }))
        .unwrap();
    command_buffer
}

/// What a round broke of the orders (the module docs), by order.
#[derive(Debug, Default, PartialEq, Eq)]
struct Broken {
    /// 1: a status short of `Completed` in its own completion-seen handler, a completion-seen
    /// handler not returned before its completed handler started, or a wait that returned before
    /// its completed handler did.
    within: u32,
    /// 2: B's completion-seen handler started before A's completed handler returned.
    queue: u32,
    /// 3: C's completion-seen handler started before A's returned, or read A short of
    /// `Completed`.
    across: u32,
}

/// One round on `first` (A, B) and, with `gpu_wait`, `second` (C, waiting for A's signal of
/// `event` to `value`).
fn round(
    (first, second): (&CommandQueue, &CommandQueue),
    buffer: &Buffer,
    gpu_wait: Option<(&Event, u64)>,
    broken: &mut Broken,
) {
    let log = Arc::new(Log::default());
    let a = logged(first, buffer, (&log, 0), (HOLD, None), None);
    if let Some((event, value)) = gpu_wait {
        a.encode_signal_event(event.into(), value).unwrap();
    }
    let b = logged(first, buffer, (&log, 1), (Duration::ZERO, None), None);
    let c = gpu_wait.map(|wait| {
        let shape = (Duration::ZERO, Some(wait));
        logged(second, buffer, (&log, 2), shape, Some(&a))
    });
    a.commit().unwrap();
    b.commit().unwrap();
    if let Some(c) = &c {
        c.commit().unwrap();
    }
    let buffers: Vec<(usize, &CommandBuffer)> = [Some(&a), Some(&b), c.as_ref()]
        .into_iter()
        .enumerate()
        .filter_map(|(index, buffer)| buffer.map(|buffer| (index, buffer)))
        .collect();
    for &(index, buffer) in &buffers {
        buffer.wait_until_completed_for(WAIT).unwrap();
        let waited = log.tick();
        log.steps.lock().unwrap()[index].waited = waited;
    }
    let steps = *log.steps.lock().unwrap();
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

/// Over [`ROUNDS`] rounds with no GPU wait and as many with one, every order of the module docs
/// holds; and the completion-seen handlers of a command buffer with a completed handler too ran
/// where its completion was seen at least once
/// (`WorkCounts::completion_seen_handlers_where_seen`), so the orders were checked on that path.
#[test]
fn handlers_run_in_their_command_buffer_s_queue_s_and_signaller_s_order() {
    let device = device();
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let event = device.new_event().unwrap();
    let before = device.work_counts();
    let mut broken = Broken::default();
    for value in 1..=ROUNDS {
        round((&first, &second), &buffer, None, &mut broken);
        round(
            (&first, &second),
            &buffer,
            Some((&event, value)),
            &mut broken,
        );
    }
    let after = device.work_counts();
    let where_seen =
        after.completion_seen_handlers_where_seen - before.completion_seen_handlers_where_seen;
    let run = after.completion_seen_handlers_run - before.completion_seen_handlers_run;
    eprintln!(
        "HANDLER-ORDERS: {run} completion-seen handlers, {where_seen} where seen; {broken:?}"
    );
    assert_eq!(broken, Broken::default(), "of {} rounds", 2 * ROUNDS);
    assert!(
        where_seen > 0,
        "no completion-seen handler of a command buffer with a completed handler ran where seen"
    );
}
