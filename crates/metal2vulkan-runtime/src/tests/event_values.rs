//! An event's values, under a device cfg on every device of the cfg's kind: a signal not above
//! the highest value known is left out, whichever queue it is on; the host's value of a shared
//! event reads at once below a GPU signal still pending and never signals the semaphore, and a
//! host value above a pending GPU signal stays the event's value once that signal lands; the
//! largest value is kept; a value the host sees is one a host wait confirmed, a steady value
//! asking for no wait and the completion thread waiting on the GPU, not spinning, for a value
//! still to come; and a shared event's listeners run on one thread, not the posting one, in the
//! order posted.
//!
//! The GPU signals stay pending while the test reads the values: a [`Busy`] gate submitted ahead
//! of them holds every later command of the device's one `VkQueue` until the test opens it.

use crate::device::lock;
use crate::tests::command::on_device::{
    each_device, fill, finished, shared_event, within, Arena, Busy, PATIENCE,
};
use crate::{Buffer, CommandBuffer, CommandQueue, Device, Event};
use std::sync::mpsc;
use std::time::Duration;

fn queue(device: &Device) -> CommandQueue {
    device.new_command_queue(64).expect("a queue")
}

/// Commits `command_buffer` and returns once its submission is counted.
fn submitted(name: &str, device: &Device, command_buffer: &CommandBuffer) {
    let before = device.counters().metrics().submissions;
    command_buffer.commit().expect("a commit");
    within(name, || device.counters().metrics().submissions > before);
}

/// A command buffer of `queue` whose one command is a signal of `event` to `value`, submitted.
fn signal(
    name: &str,
    device: &Device,
    queue: &CommandQueue,
    event: &Event,
    value: u64,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().expect("a command buffer");
    (command_buffer.encode_signal_event(event, value)).expect("a signal");
    submitted(name, device, &command_buffer);
    command_buffer
}

/// The values of `event`'s signals submitted and not yet seen, lowest first.
fn pending(event: &Event) -> Vec<u64> {
    lock(&event.inner().state).pending.iter().copied().collect()
}

/// The value of `event`'s semaphore a host wait confirmed: an `MTLEvent`'s value, which no host
/// call sets.
fn counter(event: &Event) -> u64 {
    (event.inner().reached(None)).expect("the semaphore's value")
}

/// The holds for an event's value the submit thread made, of every length.
fn holds(device: &Device) -> u64 {
    device.counters().metrics().event_holds.iter().sum()
}

/// After a signal of 7, a lower one (4, on another queue) and an equal one (7) change nothing:
/// Metal's maximum makes them no-ops (measured), so one signal is pending, nothing is reached
/// while the GPU is held, and once it runs the event's value is 7. The validation layer checks
/// that the semaphore only rises.
#[test]
fn a_signal_not_above_what_is_known_is_dropped() {
    each_device("event_signal_dropped", |name, _, device| {
        let event = device.new_event().expect("an event");
        let (first, second) = (queue(&device), queue(&device));
        let busy = Busy::new(&device);
        let signals: Vec<CommandBuffer> = [(&first, 7), (&second, 4), (&first, 7)]
            .into_iter()
            .map(|(queue, value)| signal(name, &device, queue, &event, value))
            .collect();
        assert_eq!(
            pending(&event),
            [7],
            "{name}: the lower and the equal left out"
        );
        assert_eq!(counter(&event), 0, "{name}: nothing reached yet");
        busy.open();
        for command_buffer in &signals {
            finished(name, command_buffer);
        }
        assert_eq!(counter(&event), 7, "{name}");
        within(name, || pending(&event).is_empty());
        drop((signals, busy, first, second, event, device));
    });
}

/// A shared event's host value takes effect at once and never on the semaphore: with a GPU
/// signal of 7 pending, a set of 5 reads 5, and lower and equal sets change nothing (measured). A
/// wait the host's value reached goes to the GPU with no hold, though the GPU signal above it is
/// pending; a wait for 6, which only the pending 7 reaches, completes after it. Once the GPU runs,
/// the value is 7.
#[test]
fn the_host_floor_takes_effect_at_once_and_never_on_the_semaphore() {
    each_device("event_host_floor", |name, _, device| {
        let mut arena = Arena::new(&device);
        let (signalled, met, above) = (arena.buffer(), arena.buffer(), arena.buffer());
        let event = shared_event(&device);
        let (first, second) = (queue(&device), queue(&device));
        let busy = Busy::new(&device);
        let signaller = first.command_buffer().expect("a command buffer");
        fill(&signaller, &signalled, 7);
        (signaller.encode_signal_event(event.as_event(), 7)).expect("a signal");
        submitted(name, &device, &signaller);
        assert_eq!(pending(event.as_event()), [7], "{name}");
        // A host set below the pending GPU signal: the value is 5 now.
        event.set_signaled_value(5);
        assert_eq!(event.signaled_value(), 5, "{name}");
        // Lower and equal sets change nothing.
        event.set_signaled_value(3);
        event.set_signaled_value(5);
        assert_eq!(event.signaled_value(), 5, "{name}");
        assert_eq!(
            counter(event.as_event()),
            0,
            "{name}: never on the semaphore"
        );
        let held = holds(&device);
        // A wait the host's value reached goes, though a GPU signal above it is pending.
        let waiter = |value: u64, buffer: &Buffer, word: u32| {
            let command_buffer = second.command_buffer().expect("a command buffer");
            (command_buffer.encode_wait_for_event(event.as_event(), value)).expect("a wait");
            fill(&command_buffer, buffer, word);
            submitted(name, &device, &command_buffer);
            command_buffer
        };
        let at_floor = waiter(5, &met, 5);
        // One above it: only the GPU's seven reaches it.
        let past_floor = waiter(6, &above, 6);
        assert_eq!(holds(&device), held, "{name}: neither wait held");
        assert_eq!(event.signaled_value(), 5, "{name}: the GPU held");
        busy.open();
        for command_buffer in [&signaller, &at_floor, &past_floor] {
            finished(name, command_buffer);
        }
        assert!(
            past_floor.gpu_end_time() >= signaller.gpu_end_time(),
            "{name}: the wait for 6 after the signal of 7"
        );
        // The GPU's seven lands above the floor.
        assert_eq!(event.signaled_value(), 7, "{name}");
        drop((signaller, at_floor, past_floor, busy, first, second));
        drop((signalled, met, above, arena, event, device));
    });
}

/// A host value set above a pending GPU signal reaches every wait below it at once, and stays the
/// event's value once that signal lands: with a GPU signal of 7 pending, a set of 9 reads 9, a host
/// wait for 5 is met at once, and once the GPU's seven reaches the semaphore the value is still 9.
#[test]
fn a_host_set_above_a_pending_signal_reaches_every_wait_below_it() {
    each_device("event_host_above", |name, _, device| {
        let event = shared_event(&device);
        let queue = queue(&device);
        let busy = Busy::new(&device);
        let signaller = signal(name, &device, &queue, event.as_event(), 7);
        assert_eq!(pending(event.as_event()), [7], "{name}");
        event.set_signaled_value(9);
        assert_eq!(event.signaled_value(), 9, "{name}");
        assert!(
            (event.wait_until_signaled_value(5, Duration::ZERO))
                .expect("a wait outside a scheduled handler"),
            "{name}: reached at once"
        );
        busy.open();
        finished(name, &signaller);
        assert_eq!(counter(event.as_event()), 7, "{name}: the GPU's seven");
        assert_eq!(
            event.signaled_value(),
            9,
            "{name}: the value stays at the floor"
        );
        drop((signaller, busy, queue, event, device));
    });
}

/// The largest value is kept: a signal of `u64::MAX` reaches the event, which reads it, a second
/// is left out, and a host set of it changes nothing. Metal's values are `uint64_t` (MTLEvent.h's
/// `signaledValue`). The validation layer checks the semaphore's signal of it, which Vulkan bounds
/// by the device's `maxTimelineSemaphoreValueDifference` from the semaphore's value
/// (`VUID-VkSubmitInfo2-semaphore-03883`).
#[test]
fn the_largest_value_is_kept() {
    each_device("event_largest_value", |name, _, device| {
        let event = shared_event(&device);
        let queue = queue(&device);
        let first = signal(name, &device, &queue, event.as_event(), u64::MAX);
        finished(name, &first);
        assert_eq!(event.signaled_value(), u64::MAX, "{name}");
        let second = signal(name, &device, &queue, event.as_event(), u64::MAX);
        finished(name, &second);
        within(name, || pending(event.as_event()).is_empty());
        event.set_signaled_value(u64::MAX);
        assert_eq!(event.signaled_value(), u64::MAX, "{name}");
        assert_eq!(counter(event.as_event()), u64::MAX, "{name}");
        drop((first, second, queue, event, device));
    });
}

/// A value the host sees that a GPU signal raised is one a host wait confirmed: the counter read
/// finds it and a wait with a timeout of 0 confirms it (`EventInner::reached`). On a steady event,
/// a value read at or below the one confirmed asks for no wait: 1,000 reads of a value the GPU
/// reached make at most one wait. With a signal pending behind the [`Busy`] gate, the event is
/// watched and the completion thread waits on the GPU for it: over 200 ms it does not return from
/// that wait, and a read none confirmed there ("not yet") leaves it waiting, not spinning. The
/// counts are printed.
#[test]
fn a_value_the_host_sees_is_one_a_host_wait_confirmed() {
    each_device("event_confirmed_reads", |name, _, device| {
        let event = shared_event(&device);
        let queue = queue(&device);
        let counters = device.counters();
        let first = signal(name, &device, &queue, event.as_event(), 5);
        finished(name, &first);
        assert!(
            (event.wait_until_signaled_value(5, PATIENCE)).expect("a wait outside a handler"),
            "{name}"
        );
        // A steady event: every read finds the value confirmed.
        let before = counters.event_reads();
        for _ in 0..1000 {
            assert_eq!(event.signaled_value(), 5, "{name}");
        }
        let steady = counters.event_reads();
        let (reads, waits) = (steady.reads - before.reads, steady.waits - before.waits);
        println!("{name}: a steady event: {reads} reads, {waits} waits");
        assert!(reads >= 1000, "{name}: {reads} reads");
        assert!(
            waits <= 1,
            "{name}: {waits} waits for 1,000 reads of a steady value"
        );
        // A signal pending: the completion thread waits on the GPU for it.
        let busy = Busy::new(&device);
        // A value no progress value of the device's reaches, so the GPU wait's values name it
        // alone.
        let later = 1 << 40;
        let second = signal(name, &device, &queue, event.as_event(), later);
        assert_eq!(pending(event.as_event()), [later], "{name}");
        within(name, || {
            (device.core().done.gpu_wait()).is_some_and(|values| values.contains(&later))
        });
        let idle = counters.event_reads();
        std::thread::sleep(Duration::from_millis(200));
        let after = counters.event_reads();
        println!(
            "{name}: a pending signal, 200 ms: {} wake-ups of the completion thread, {} not yet \
             on it ({} in all); since the device opened {:?}",
            after.wakes - idle.wakes,
            after.not_yet_completing - idle.not_yet_completing,
            after.not_yet - idle.not_yet,
            after
        );
        assert_eq!(
            after.wakes, idle.wakes,
            "{name}: the completion thread returned from its GPU wait with nothing reached"
        );
        assert_eq!(event.signaled_value(), 5, "{name}: the GPU held");
        busy.open();
        finished(name, &second);
        assert!(
            (event.wait_until_signaled_value(later, PATIENCE)).expect("a wait outside a handler"),
            "{name}"
        );
        assert_eq!(event.signaled_value(), later, "{name}");
        drop((first, second, busy, queue, event, device));
    });
}

/// The notifications of a shared event's listeners run off the posting thread, on one thread for
/// them all, in the order posted.
#[test]
fn the_first_notification_starts_the_thread() {
    each_device("event_listener_order", |name, _, device| {
        let event = shared_event(&device);
        event.set_signaled_value(1);
        let (send, receive) = mpsc::channel();
        for n in 0..8u32 {
            let send = send.clone();
            (event.notify_listener(1, move |_| {
                let _ = send.send((n, std::thread::current().id()));
            }))
            .expect("a listener");
        }
        let ran: Vec<_> = (0..8)
            .map(|_| receive.recv_timeout(PATIENCE).expect("a notification"))
            .collect();
        let order: Vec<u32> = ran.iter().map(|&(n, _)| n).collect();
        assert_eq!(order, (0..8).collect::<Vec<_>>(), "{name}");
        let thread = ran[0].1;
        assert_ne!(thread, std::thread::current().id(), "{name}");
        assert!(
            ran.iter().all(|&(_, id)| id == thread),
            "{name}: one thread"
        );
        drop((event, device));
    });
}
