//! A command buffer whose work waited on the GPU for another queue's command buffer, through an
//! event that one signals at its end, completes after it, as Metal's completion follows the GPU's
//! (`command_buffer/follow.rs`): the signaller's status reads `Completed` once the waiter's wait
//! returns and in each of the waiter's handlers, and the signaller's completion-seen handlers run
//! before any of the waiter's. Its completed handlers are dispatched before the waiter's, which run
//! on another handler thread, as Metal runs two queues' handlers. A blocked handler of the
//! signaller's queue holds back the signaller's transition and not the other queue's, Metal's
//! measured order (`contexts/mod.rs`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions,
    EventRef, ResourceOptions,
};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The rounds of each shape.
const ROUNDS: u32 = 50;

/// The bytes the signaller fills: enough that its work is still running as the waiter's batch is
/// submitted, and the waiter's batch, a fill of 256 bytes, completes just after it.
const FILL: u64 = 16 << 20;

/// How long a test waits for a handler before it fails.
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

/// The handlers a command buffer of a shape carries.
#[derive(Clone, Copy, Debug)]
enum Handlers {
    /// A completed handler, which its queue's handler thread runs.
    Completed,
    /// A completion-seen handler alone, which the queue waiter that saw it complete may run.
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

/// Add `handlers` to `buffer`, each logging `name` and `signaller`'s status as it starts.
fn log_handlers(
    buffer: &CommandBuffer,
    name: char,
    handlers: Handlers,
    signaller: &CommandBuffer,
    log: &Log,
) {
    if handlers.completed() {
        let (log, signaller) = (Arc::clone(log), signaller.clone());
        buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                let status = signaller.status();
                log.lock().unwrap().push((name, false, status));
            }))
            .unwrap();
    }
    if handlers.seen() {
        let (log, signaller) = (Arc::clone(log), signaller.clone());
        buffer
            .add_completion_seen_handler(Box::new(move |_: &CommandBuffer| {
                let status = signaller.status();
                log.lock().unwrap().push((name, true, status));
            }))
            .unwrap();
    }
}

/// The signaller: a fill of `big` on `queue`, then a signal of `event` to `value`.
fn signaller(queue: &CommandQueue, big: &Buffer, event: EventRef<'_>, value: u64) -> CommandBuffer {
    let a = queue.command_buffer().unwrap();
    let mut blit = a.blit_command_encoder().unwrap();
    blit.fill_buffer(big, 0..big.length(), value as u8).unwrap();
    blit.end_encoding().unwrap();
    a.encode_signal_event(event, value).unwrap();
    a
}

/// The waiter: a wait for `event` to reach `value` on `queue`, then a fill of `word`.
fn waiter(queue: &CommandQueue, word: &Buffer, event: EventRef<'_>, value: u64) -> CommandBuffer {
    let b = queue.command_buffer().unwrap();
    b.encode_wait_for_event(event, value).unwrap();
    let mut blit = b.blit_command_encoder().unwrap();
    blit.fill_buffer(word, 0..word.length(), value as u8)
        .unwrap();
    blit.end_encoding().unwrap();
    b
}

/// Each commit order, kind of event and order of the two queues: B committed first, the event a
/// shared one (whose wait the host holds until its value is reached) and A on the queue made
/// second.
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
    /// handler: both were dispatched, A's first, to two handler threads (counted, not a break).
    completed: u32,
}

/// One round: A on `of_a` fills `big` and signals `event` to `value` at its end, and B on `of_b`
/// waits for it on the GPU and fills `word`, each with its handlers, committed in that order or B
/// first. What it broke is added to `broken`.
fn round(
    (of_a, of_b): (&CommandQueue, &CommandQueue),
    (big, word): (&Buffer, &Buffer),
    (event, value): (EventRef<'_>, u64),
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
        buffer.commit().unwrap();
    }
    b.wait_until_completed().unwrap();
    let a_completed = a.status() == CommandBufferStatus::Completed;
    a.wait_until_completed().unwrap();
    // A wait returns once the status is final, which may be before the handlers ran: wait for
    // the log to fill.
    let expected = a_handlers.count() + b_handlers.count();
    let mut waited = Duration::ZERO;
    while log.lock().unwrap().len() < expected {
        assert!(waited < WAIT, "a handler never ran");
        std::thread::sleep(Duration::from_millis(1));
        waited += Duration::from_millis(1);
    }
    let log = log.lock().unwrap();
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
    let device = device();
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let big = device.new_buffer(FILL, ResourceOptions::private()).unwrap();
    let word = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let (shared, plain) = (
        device.new_shared_event().unwrap(),
        device.new_event().unwrap(),
    );
    let before = device.work_counts();
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
            let event = || match on_shared {
                true => EventRef::from(&shared),
                false => EventRef::from(&plain),
            };
            // Each order of the queues' `VkQueue`s, where the device has several.
            let queues = match swapped {
                false => (&first, &second),
                true => (&second, &first),
            };
            let mut broken = Broken::default();
            for _ in 0..ROUNDS {
                value += 1;
                let shape = (b_first, &mut broken);
                round(queues, (&big, &word), (event(), value), handlers, shape);
            }
            dispatched += broken.completed;
            if broken.status + broken.seen > 0 {
                overtaken.push(format!(
                    "A {:?}, B {:?}, B committed first {b_first}, shared event {on_shared}, queues \
                     swapped {swapped}: A not completed in {}, A's completion-seen handler after \
                     B's in {} of {ROUNDS}",
                    handlers.0, handlers.1, broken.status, broken.seen
                ));
            }
        }
    }
    let after = device.work_counts();
    eprintln!(
        "held for another queue {}, completed after another queue {}, B's handler first of two \
         dispatched {dispatched}",
        after.batches_held_for_another_queue - before.batches_held_for_another_queue,
        after.command_buffers_completed_after_another_queue
            - before.command_buffers_completed_after_another_queue
    );
    assert!(
        overtaken.is_empty(),
        "B's handlers overtook A's: {overtaken:#?}"
    );
}

/// A's queue's earlier command buffer blocks in its completed handler, so A's transition waits
/// behind it, as Metal's does; B, which waited on the GPU for A's signal, still completes and runs
/// its completion-seen handler, A short of `Completed`: Metal runs another queue's transitions
/// while one queue's handler blocks, so B's waits for A's only where A's leads its context
/// (`WorkCounts::command_buffers_completed_after_another_queue`), never behind another item.
#[test]
fn a_blocked_handler_of_the_signaller_s_queue_holds_back_the_signaller_and_not_the_waiter() {
    let device = device();
    let (first, second) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let big = device.new_buffer(FILL, ResourceOptions::private()).unwrap();
    let word = device.new_buffer(256, ResourceOptions::private()).unwrap();
    let event = device.new_event().unwrap();
    let before = device.work_counts();
    let (release, released) = mpsc::channel::<()>();
    let earlier = first.command_buffer().unwrap();
    let mut blit = earlier.blit_command_encoder().unwrap();
    blit.fill_buffer(&word, 0..256, 1).unwrap();
    blit.end_encoding().unwrap();
    earlier
        .add_completed_handler(Box::new(move |_: &CommandBuffer| {
            released.recv_timeout(WAIT).unwrap();
        }))
        .unwrap();
    let a = signaller(&first, &big, (&event).into(), 1);
    let b = waiter(&second, &word, (&event).into(), 1);
    let (seen, saw) = mpsc::channel();
    {
        let a = a.clone();
        b.add_completion_seen_handler(Box::new(move |_: &CommandBuffer| {
            seen.send(a.status()).unwrap();
        }))
        .unwrap();
    }
    for buffer in [&earlier, &a, &b] {
        buffer.commit().unwrap();
    }
    let a_status = saw.recv_timeout(WAIT).unwrap();
    // B's wait returns as its handler starts, before its gate (if any) opens: none was closed.
    b.wait_until_completed().unwrap();
    assert_ne!(a_status, CommandBufferStatus::Completed);
    assert_ne!(a.status(), CommandBufferStatus::Completed);
    release.send(()).unwrap();
    a.wait_until_completed().unwrap();
    assert_eq!(a.status(), CommandBufferStatus::Completed);
    let after = device.work_counts();
    assert_eq!(
        after.command_buffers_completed_after_another_queue
            - before.command_buffers_completed_after_another_queue,
        0
    );
}
