//! A signal of the caller's own submission, reserved on either kind of event
//! (`Event::reserve_vk_signal`, `SharedEvent::reserve_vk_signal`), and the command buffers that
//! wait for its value on other queues: a plain event's (`MTLEvent`, whose value no host call sets)
//! go to their `VkQueue`s with a Vulkan wait for the reserved counter, submitted (`Scheduled`)
//! before the signal is, with `WorkCounts::shared_event_waits_held` unmoved; a shared event's are
//! held on the host until the device's watcher saw the signal land, as every wait the host may
//! satisfy is (`event_timeline.rs`). One command buffer is committed before the reservation and
//! one after: the reservation lets the first go, and the second goes at its commit.
//!
//! The raw submission goes to an extra queue of the device's, which no command queue submits to,
//! as the hook's contract asks of a signal the runtime's command buffers may wait for on the GPU
//! before it is submitted; a device with none runs neither test. Each prints the host's time from
//! the raw signal's `vkQueueSubmit` to the waiting command buffer's completed handler, over timed
//! rounds after the checked one (`EVENT-VK-SIGNAL:` lines).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use super::video_interop::{device, extra, shared_families, submit_raw, CallerBuffer, RawQueue};
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, Event, EventRef,
    ResourceOptions, SharedEvent, VkEventSignal, VkEventWait,
};
use ash::vk;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The bytes the raw fill writes and the waiting command buffers copy.
const BYTES: u64 = 4096;

/// The fill's byte.
const FILL: u8 = 0x5a;

/// How long a held command buffer is watched for going early.
const HELD: Duration = Duration::from_millis(300);

/// How long a command buffer whose wait chains on its `VkQueue` may take to be submitted.
const GOES: Duration = Duration::from_secs(5);

/// The timed rounds after the checked one, for the latency figures.
const ROUNDS: u64 = 16;

/// Either kind's interop hooks, so one scenario runs on both.
trait Hooked {
    /// Whether the host may set its value.
    const SETTABLE: bool;
    /// A new event of the kind.
    fn new_of(device: &Device) -> Self;
    /// The event a command buffer waits for.
    fn event(&self) -> EventRef<'_>;
    fn vk_wait(&self, value: u64) -> VkEventWait;
    /// # Safety
    ///
    /// As `reserve_vk_signal`'s.
    unsafe fn reserve_vk_signal(&self, value: u64) -> VkEventSignal;
    /// # Safety
    ///
    /// As `vk_semaphore`'s.
    unsafe fn vk_semaphore(&self) -> vk::Semaphore;
}

impl Hooked for Event {
    const SETTABLE: bool = false;
    fn new_of(device: &Device) -> Event {
        device.new_event().unwrap()
    }
    fn event(&self) -> EventRef<'_> {
        self.into()
    }
    fn vk_wait(&self, value: u64) -> VkEventWait {
        Event::vk_wait(self, value)
    }
    unsafe fn reserve_vk_signal(&self, value: u64) -> VkEventSignal {
        // SAFETY: the caller's.
        unsafe { Event::reserve_vk_signal(self, value) }.unwrap()
    }
    unsafe fn vk_semaphore(&self) -> vk::Semaphore {
        // SAFETY: the caller's.
        unsafe { Event::vk_semaphore(self) }
    }
}

impl Hooked for SharedEvent {
    const SETTABLE: bool = true;
    fn new_of(device: &Device) -> SharedEvent {
        device.new_shared_event().unwrap()
    }
    fn event(&self) -> EventRef<'_> {
        self.into()
    }
    fn vk_wait(&self, value: u64) -> VkEventWait {
        SharedEvent::vk_wait(self, value)
    }
    unsafe fn reserve_vk_signal(&self, value: u64) -> VkEventSignal {
        // SAFETY: the caller's.
        unsafe { SharedEvent::reserve_vk_signal(self, value) }.unwrap()
    }
    unsafe fn vk_semaphore(&self) -> vk::Semaphore {
        // SAFETY: the caller's.
        unsafe { SharedEvent::vk_semaphore(self) }
    }
}

/// When a command buffer's completed handler ran.
type Completed = Arc<Mutex<Option<Instant>>>;

/// A committed command buffer of `queue` that waits for `event` to reach `value`, then copies
/// `from` into `to`; and when its completed handler ran.
fn waiter(
    queue: &CommandQueue,
    event: EventRef<'_>,
    value: u64,
    from: &Buffer,
    to: &Buffer,
) -> (CommandBuffer, Completed) {
    let buffer = queue.command_buffer().unwrap();
    buffer.encode_wait_for_event(event, value).unwrap();
    let mut blit = buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(from, 0, to, 0, BYTES)
        .unwrap();
    blit.end_encoding().unwrap();
    let completed = Completed::default();
    let at = Arc::clone(&completed);
    buffer
        .add_completed_handler(Box::new(move |_| {
            *at.lock().unwrap() = Some(Instant::now());
        }))
        .unwrap();
    buffer.commit().unwrap();
    (buffer, completed)
}

/// When `completed`'s handler ran, waited for at most [`WAIT`].
fn completed_at(completed: &Completed) -> Instant {
    let deadline = Instant::now() + WAIT;
    loop {
        if let Some(at) = *completed.lock().unwrap() {
            return at;
        }
        assert!(Instant::now() < deadline, "the completed handler never ran");
        std::thread::sleep(Duration::from_micros(100));
    }
}

/// Whether `buffer`'s status reads `Scheduled` (or later) within `within`.
fn scheduled_within(buffer: &CommandBuffer, within: Duration) -> bool {
    let deadline = Instant::now() + within;
    loop {
        if buffer.status() != CommandBufferStatus::Committed {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Signal `counter` of `semaphore` from the raw queue, after a wait for `after` where given, with
/// `record`'s commands first, returning once it completed; when it was submitted.
fn signal_raw(
    device: &Device,
    raw_queue: &RawQueue,
    semaphore: vk::Semaphore,
    (counter, after): (u64, Option<u64>),
    record: impl FnOnce(&ash::Device, vk::CommandBuffer),
) -> Instant {
    let waits: Vec<(vk::Semaphore, u64)> =
        after.map(|after| (semaphore, after)).into_iter().collect();
    submit_raw(device, raw_queue, &waits, &[(semaphore, counter)], record)
}

/// The median and the largest of `times`, in microseconds.
fn spread(mut times: Vec<Duration>) -> (u128, u128) {
    times.sort();
    (
        times[times.len() / 2].as_micros(),
        times.last().unwrap().as_micros(),
    )
}

/// The scenario on an event of kind `E`, checked as its kind's waits go: to their `VkQueue`s
/// before the signal (`chained`), or held for the host, as the device counts.
fn run<E: Hooked>(leg: &str) {
    let scratch = Scratch::new(leg);
    let messages = Arc::new(Mutex::new(Vec::new()));
    let device = device(&scratch.0, &messages);
    if extra(&device, vk::QueueFlags::COMPUTE).is_none() {
        eprintln!(
            "EVENT-VK-SIGNAL: {}: no extra queue for the raw signal, nothing run",
            device.name()
        );
        return;
    }
    let raw_queue = RawQueue::of(&device);
    let source = CallerBuffer::new(&device, BYTES, shared_families(&device, &raw_queue));
    source.set(&[0; BYTES as usize]);
    // SAFETY: the Metal buffer and the command buffers using it end before the test's own.
    let filled = unsafe { source.wrapped() };
    let reads = [(); 2].map(|()| device.new_buffer(BYTES, ResourceOptions::shared()).unwrap());
    let queues = [(); 2].map(|()| device.new_command_queue().unwrap());
    let event = E::new_of(&device);
    let held_before = device.work_counts().shared_event_waits_held;

    // The first waits before anything reaches 10; the reservation names counter 1, after nothing.
    assert_eq!(event.vk_wait(10), VkEventWait::NotYet);
    let (early, _) = waiter(&queues[0], event.event(), 10, &filled, &reads[0]);
    assert!(
        !scheduled_within(&early, HELD),
        "a wait for a value nothing reaches went"
    );
    // SAFETY: the signal is submitted below, once, after a wait for `after`, from a queue no
    // command queue submits to.
    let reserved = unsafe { event.reserve_vk_signal(10) };
    assert_eq!(
        reserved,
        VkEventSignal::Counter {
            counter: 1,
            after: None
        }
    );
    assert_eq!(event.vk_wait(10), VkEventWait::Counter(1));
    let (late, _) = waiter(&queues[1], event.event(), 10, &filled, &reads[1]);

    // A plain event's waits go to their `VkQueue`s now, before the signal is even submitted; a
    // shared event's are held until it landed. Neither completes before it.
    let within = if E::SETTABLE { HELD } else { GOES };
    let chained = [&early, &late].map(|buffer| scheduled_within(buffer, within));
    for buffer in [&early, &late] {
        assert!(
            buffer.wait_until_completed_for(HELD).is_err(),
            "a command buffer ran before the raw signal"
        );
    }
    let held = device.work_counts().shared_event_waits_held - held_before;
    let statuses = [&early, &late].map(CommandBuffer::status);
    match E::SETTABLE {
        false => {
            assert_eq!(
                chained, [true; 2],
                "a plain event's wait was held: {statuses:?}"
            );
            assert_eq!(held, 0, "a plain event's wait counted as held for the host");
        }
        true => {
            assert_eq!(
                chained, [false; 2],
                "a shared event's wait went before its signal"
            );
            assert!(held > 0, "a shared event's held wait not counted");
        }
    }

    // SAFETY: the semaphore is used while the event lives, signalled only as reserved.
    let semaphore = unsafe { event.vk_semaphore() };
    let target = source.buffer;
    signal_raw(
        &device,
        &raw_queue,
        semaphore,
        (1, None),
        |vk_device, command_buffer| {
            // SAFETY: the buffer lives and takes transfers; the command buffer is recording.
            unsafe {
                vk_device.cmd_fill_buffer(
                    command_buffer,
                    target,
                    0,
                    BYTES,
                    u32::from_ne_bytes([FILL; 4]),
                )
            };
        },
    );
    for buffer in [&early, &late] {
        assert!(
            buffer.wait_until_completed_for(WAIT).is_ok(),
            "a command buffer waiting for the raw signal never completed"
        );
        assert_eq!(buffer.status(), CommandBufferStatus::Completed);
    }
    for read in &reads {
        let contents = read.contents().unwrap().as_ptr();
        // SAFETY: the command buffers completed; nothing writes the bytes now.
        let bytes = unsafe { std::slice::from_raw_parts(contents, BYTES as usize) };
        assert!(
            bytes.iter().all(|&b| b == FILL),
            "the raw fill reached the copy"
        );
    }
    // The device's watcher saw the signal land, whatever the kind.
    let deadline = Instant::now() + WAIT;
    while event.vk_wait(10) != VkEventWait::Reached {
        assert!(
            Instant::now() < deadline,
            "the watcher never saw the raw signal"
        );
        std::thread::sleep(Duration::from_millis(1));
    }

    // Timed rounds: each reserves the next value, commits its waiter, then signals: from the
    // signal's submission to the waiter's completed handler.
    let mut times = Vec::new();
    for round in 0..ROUNDS {
        let value = 20 + round;
        // SAFETY: as above.
        let VkEventSignal::Counter { counter, after } = (unsafe { event.reserve_vk_signal(value) })
        else {
            panic!("a signal above every value is kept");
        };
        let (buffer, completed) = waiter(&queues[1], event.event(), value, &filled, &reads[1]);
        if !E::SETTABLE {
            assert!(scheduled_within(&buffer, GOES), "round {round} held");
        }
        let submitted = signal_raw(&device, &raw_queue, semaphore, (counter, after), |_, _| {});
        assert!(buffer.wait_until_completed_for(WAIT).is_ok());
        times.push(completed_at(&completed).saturating_duration_since(submitted));
    }
    let (median, most) = spread(times);
    let held = device.work_counts().shared_event_waits_held - held_before;
    eprintln!(
        "EVENT-VK-SIGNAL: {}: {leg}: chained {chained:?}, waits held for the host {held}; the \
         raw signal's submission to the waiter's completed handler, {ROUNDS} rounds: median \
         {median} us, most {most} us",
        device.name()
    );
    if !E::SETTABLE {
        assert_eq!(
            held, 0,
            "a plain event's timed wait counted as held for the host"
        );
    }
    drop((early, late, filled));
}

/// A plain event's waits for the caller's reserved signal chain on their `VkQueue`s: both command
/// buffers are `Scheduled` before the raw signal is submitted, none counted as held for the host,
/// and both read what the raw fill wrote once it signalled.
#[test]
fn a_plain_event_s_wait_for_a_reserved_vk_signal_chains_on_the_vk_queue() {
    run::<Event>("event-vk-signal-plain");
}

/// A shared event's waits for the caller's reserved signal are held on the host until the
/// watcher saw it land (its value may be set from the host, which no Vulkan wait answers), and
/// counted so.
#[test]
fn a_shared_event_s_wait_for_a_reserved_vk_signal_is_held_on_the_host() {
    run::<SharedEvent>("event-vk-signal-shared");
}
