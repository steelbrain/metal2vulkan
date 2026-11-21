//! Command queues, command buffers and events, under a device cfg on every device of the cfg's
//! kind: states, handlers and refusals; holds per segment; the host floor; host values racing GPU
//! signals; listeners; abandon; quiesce and its second call; teardown from a handler, waited for,
//! and with work held, which holds no handle of the device; refusals of what has started; a loss; a released block kept until its last
//! command buffer retires; a block whose last handle drops freed once its last command buffer
//! retires; a block's clones, one dropped giving nothing back, and a handle outliving its
//! device; the thread gauge holding the device's four threads as it is made; a command buffer
//! joining a held group only within the group's expected GPU run; and each command buffer
//! counted once in every stage histogram. Without a device: a push whose bytes the command buffer
//! holds is not recorded again; a queue's expected run rises to a longer run at once and halves
//! toward a shorter one; the join and send decisions of the group bound; the stage histograms'
//! bucket edges; the part of a span under the latest GPU spans, and the completion round's steps.

use crate::command::Pushed;
use crate::complete::Round;
use crate::counters::{stage_bucket, STAGE_BUCKETS};
use crate::queue::next_run;
use crate::submit::{apart, joins, GROUP_RUN, UNSEEN_JOINS};
use crate::times::{Spans, SPANS_KEPT};

/// A span from a scheduling to a GPU start lies under the GPU spans kept as their union clipped
/// to it: overlapping spans count once, one outside it not at all, one of no length is not kept,
/// and past `SPANS_KEPT` the oldest are no longer kept.
#[test]
fn a_span_counts_the_part_under_the_latest_gpu_spans_once() {
    let mut spans = Spans::new();
    assert_eq!(spans.covered(0, 100), 0);
    for span in [
        (10, 30),
        (20, 40),
        (60, 70),
        (90, 200),
        (50, 50),
        (300, 400),
    ] {
        spans.push(span);
    }
    // [10, 40) once, [60, 70) and [90, 100): 30 + 10 + 10.
    assert_eq!(spans.covered(0, 100), 50);
    assert_eq!(spans.covered(25, 65), 20);
    assert_eq!(spans.covered(40, 60), 0);
    assert_eq!(spans.covered(100, 100), 0);
    for at in 0..SPANS_KEPT as u64 {
        spans.push((1_000 + at, 1_001 + at));
    }
    assert_eq!(spans.covered(0, 1_000), 0, "the first spans no longer kept");
    assert_eq!(
        spans.covered(1_000, 1_000 + SPANS_KEPT as u64),
        SPANS_KEPT as u64
    );
}

/// A command buffer's span from its GPU end to its handlers splits at the round's host times,
/// the six steps adding up to the span; a step whose later time comes first is 0.
#[test]
fn a_command_buffer_s_span_to_its_handlers_splits_at_the_round_s_steps() {
    let round = Round::at(100, 110, 130, 160);
    let steps = round.steps(90, 175, 200);
    assert_eq!(steps, [10, 10, 20, 30, 15, 25]);
    assert_eq!(steps.iter().sum::<u64>(), 200 - 90);
    let none = Round::at(100, 110, 130, 130);
    assert_eq!(none.steps(120, 140, 150), [0, 10, 20, 0, 10, 10]);
}

/// The stage histograms' bucket edges: under 2 us in bucket 0, from 2^i us to under 2^(i+1) us
/// in bucket i, and from 2^22 us on in the last.
#[test]
fn a_stage_span_lands_in_its_power_of_two_bucket() {
    let us = 1_000;
    assert_eq!(stage_bucket(0), 0);
    assert_eq!(stage_bucket(2 * us - 1), 0);
    assert_eq!(stage_bucket(2 * us), 1);
    assert_eq!(stage_bucket(4 * us - 1), 1);
    assert_eq!(stage_bucket(4 * us), 2);
    assert_eq!(stage_bucket((1 << 21) * us), 21);
    assert_eq!(stage_bucket((1 << 22) * us - 1), 21);
    assert_eq!(stage_bucket((1 << 22) * us), STAGE_BUCKETS - 1);
    assert_eq!(stage_bucket(u64::MAX), STAGE_BUCKETS - 1);
}

/// The group bound's decisions. A command buffer joins a held group only while the members'
/// expected runs and its own, summed, stay within `GROUP_RUN`; one past it goes alone, the group
/// first; one whose queue has no expected run yet goes alone; and a group that holds an unknown
/// run takes no more. At most `UNSEEN_JOINS` join while none of their queue's retires. At the
/// send the bound is tested again, at the run as it then stands: a group past it, or of an
/// unknown run, goes member by member; one member never does.
#[test]
fn a_command_buffer_joins_a_group_only_within_the_bound_tested_again_at_the_send() {
    let ms = 1_000_000;
    assert!(joins(10 * ms, 1, Some(ms)), "within the bound");
    assert!(joins(10 * ms, 1, Some(GROUP_RUN - 10 * ms)), "at the bound");
    assert!(
        !joins(10 * ms, 1, Some(GROUP_RUN - 10 * ms + 1)),
        "past it: alone"
    );
    assert!(!joins(0, 0, None), "an unknown run: alone");
    assert!(
        !joins(u64::MAX, 1, Some(1)),
        "a group holding an unknown run"
    );
    assert!(joins(0, UNSEEN_JOINS - 1, Some(1)), "below the unseen cap");
    assert!(!joins(0, UNSEEN_JOINS, Some(1)), "at the unseen cap");
    assert!(!apart(4, Some(GROUP_RUN / 4)), "within at the send");
    assert!(
        apart(4, Some(GROUP_RUN / 4 + 1)),
        "past it at the send: member by member"
    );
    assert!(apart(2, None), "an unknown run at the send");
    assert!(
        !apart(1, None) && !apart(1, Some(u64::MAX)),
        "one member is one submission"
    );
}

/// A queue's expected GPU run: its first command buffer's run; a longer run raises it at once, a
/// shorter one halves the way toward it, and the "none yet" mark (`u64::MAX`) is never a run.
#[test]
fn a_queues_expected_run_rises_at_once_and_halves_toward_a_shorter_run() {
    assert_eq!(next_run(None, 5_000), 5_000);
    assert_eq!(next_run(Some(10_000), 30_000), 30_000);
    assert_eq!(next_run(Some(30_000), 10_000), 20_000);
    assert_eq!(next_run(Some(20_000), 10_000), 15_000);
    assert_eq!(next_run(None, u64::MAX), u64::MAX - 1);
}

/// A push whose bytes the command buffer already holds from offset 0 records nothing; one that
/// changes any of them records, and keeps the held bytes past its end.
#[test]
fn a_push_of_the_bytes_a_command_buffer_holds_records_nothing() {
    let mut pushed = Pushed::default();
    assert!(pushed.needed(&[1, 2, 3, 4]), "the first push");
    assert!(!pushed.needed(&[1, 2, 3, 4]), "the same bytes again");
    assert!(!pushed.needed(&[1, 2]), "a shorter push of bytes held");
    assert!(pushed.needed(&[1, 9]), "a shorter push that changes one");
    assert!(!pushed.needed(&[1, 9, 3, 4]), "the bytes past it kept");
    assert!(pushed.needed(&[1, 9, 3, 4, 5]), "a longer push");
    assert!(Pushed::default().needed(&[1, 9]), "a new command buffer's");
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod on_device {
    use crate::buffer::BufferObject;
    use crate::device::lock;
    use crate::submit::{GROUP_RUN, IN_FLIGHT, UNSEEN_JOINS};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        host_time, AllocationId, AllocationInfo, Block, BlockMappings, Buffer, CommandBuffer,
        CommandBufferError, CommandBufferStatus, CommandQueue, Device, DeviceLoss, DeviceOptions,
        DispatchType, Error, Fence, Gauges, Instance, MemoryClass, Refusal, SharedEvent,
    };
    use ash::vk;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    /// How long a test waits for what must happen.
    pub(crate) const PATIENCE: Duration = Duration::from_secs(60);

    /// Under this, a call given [`PATIENCE`] answered at once: it waited for nothing.
    const AT_ONCE: Duration = Duration::from_secs(5);

    /// The test's second meeting at a barrier a thread of the runtime is parked at, which releases
    /// it: made at [`now`](Self::now), or as the test unwinds, so a failed assertion releases the
    /// thread rather than leaving the device's teardown waiting for it.
    pub(crate) struct Release(Option<Arc<std::sync::Barrier>>);

    impl Release {
        /// The release of the thread parked at `park`, once the test has met it there once.
        pub(crate) fn of(park: &Arc<std::sync::Barrier>) -> Release {
            Release(Some(Arc::clone(park)))
        }

        pub(crate) fn now(&mut self) {
            if let Some(park) = self.0.take() {
                park.wait();
            }
        }
    }

    impl Drop for Release {
        fn drop(&mut self) {
            self.now();
        }
    }

    pub(crate) fn within(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + PATIENCE;
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// The matrix's leak records: each kind's live count once the device is gone.
    fn print_gauges_at_drop(test: &str, gauges: Gauges) {
        for (kind, value) in [
            ("buffers", gauges.buffers),
            ("blocks", gauges.blocks),
            ("command_buffers", gauges.command_buffers),
            ("events", gauges.events),
        ] {
            println!("M2V-MATRIX gauge {test}/{kind}@drop {value}");
        }
    }

    /// Opens each device of the cfg's kind, under the validation layer, waits for the block-maker
    /// thread's first work, runs `body`, which drops the device, and checks that every gauge
    /// reaches zero.
    pub(crate) fn each_device(test: &str, body: impl FnMut(&str, &Instance, Device)) {
        let instance = crate::tests::validation::instance(&[]);
        on_each_device(test, &instance, body);
    }

    /// [`each_device`] with `options` for each device.
    pub(crate) fn each_device_with(
        test: &str,
        options: impl Fn() -> DeviceOptions,
        body: impl FnMut(&str, &Instance, Device),
    ) {
        let instance = crate::tests::validation::instance(&[]);
        open_each_device(test, &instance, options, body);
    }

    /// [`each_device`] on `instance`.
    pub(crate) fn on_each_device(
        test: &str,
        instance: &Instance,
        body: impl FnMut(&str, &Instance, Device),
    ) {
        open_each_device(test, instance, DeviceOptions::default, body);
    }

    /// [`each_device_with`] on `instance`.
    fn open_each_device(
        test: &str,
        instance: &Instance,
        options: impl Fn() -> DeviceOptions,
        mut body: impl FnMut(&str, &Instance, Device),
    ) {
        let devices = devices_of_the_cfgs_kind(instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device =
                Device::new(physical, options()).unwrap_or_else(|error| panic!("{name}: {error}"));
            // The block-maker thread's first work done, so a body's counts start from a device
            // that holds its fixed-size internals.
            device.core().pool.await_internals();
            let counters = device.counters();
            body(name, instance, device);
            within("the teardown", || counters.gauges() == Gauges::default());
            print_gauges_at_drop(test, counters.gauges());
            // The matrix holds these at 0 (`readback::timestamps`, the completion thread's wait).
            let metrics = counters.metrics();
            let unavailable = metrics.stamps_not_available;
            println!("M2V-MATRIX metric {test}/stamps_not_available {unavailable}");
            let behind = metrics.waits_behind_counter;
            println!("M2V-MATRIX metric {test}/waits_behind_counter {behind}");
        }
    }

    /// Buffers of 256 bytes in one Shared block, each zeroed.
    pub(crate) struct Arena {
        device: Device,
        block: Block,
        next: u64,
    }

    impl Arena {
        pub(crate) fn new(device: &Device) -> Arena {
            let block = device
                .take_block(MemoryClass::Shared, 1)
                .expect("a Shared block");
            Arena {
                device: device.clone(),
                block,
                next: 0,
            }
        }

        pub(crate) fn buffer(&mut self) -> Buffer {
            let buffer = self
                .device
                .new_buffer(self.block.id, self.next, 256, Some(0))
                .expect("a buffer");
            self.next += 256;
            buffer
        }
    }

    /// Records a fill of `buffer` with `word`, keeping its block.
    pub(crate) fn fill(command_buffer: &CommandBuffer, buffer: &Buffer, word: u32) {
        let (target, offset) = buffer.device_range();
        let size = buffer.length();
        command_buffer
            .record_for_test(&[buffer], |device, command| {
                // SAFETY: the command buffer is recording; the range is the buffer's, in whole
                // words.
                unsafe { device.cmd_fill_buffer(command, target, offset, size, word) };
            })
            .expect("a fill");
    }

    /// A buffer's first word, as the host reads it.
    pub(crate) fn word(buffer: &Buffer) -> u32 {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive and at least 4 bytes long.
        unsafe { contents.cast::<u32>().as_ptr().read_volatile() }
    }

    /// Waits for a command buffer to complete or end, and for its handlers.
    pub(crate) fn finished(name: &str, command_buffer: &CommandBuffer) {
        within(name, || {
            matches!(
                command_buffer.status(),
                CommandBufferStatus::Completed | CommandBufferStatus::Error
            )
        });
        command_buffer
            .wait_until_completed()
            .expect("a wait outside a scheduled handler");
    }

    /// A log the handlers write to.
    #[derive(Clone, Default)]
    struct Log(Arc<Mutex<Vec<String>>>);

    impl Log {
        fn push(&self, line: impl Into<String>) {
            lock(&self.0).push(line.into());
        }

        fn lines(&self) -> Vec<String> {
            lock(&self.0).clone()
        }

        /// Adds handlers that log `label`'s scheduling and completion, with the status each sees.
        fn handlers(&self, label: &str, command_buffer: &CommandBuffer) {
            let (log, scheduled) = (self.clone(), format!("{label} scheduled"));
            command_buffer
                .add_scheduled_handler(move |cb| log.push(format!("{scheduled} {:?}", cb.status())))
                .expect("a scheduled handler");
            let (log, completed) = (self.clone(), format!("{label} completed"));
            command_buffer
                .add_completed_handler(move |cb| {
                    log.push(format!("{completed} {:?} {:?}", cb.status(), cb.error()))
                })
                .expect("a completed handler");
        }
    }

    /// The completion thread's host wait with a timeout of 0, which every retirement rests on
    /// (`complete::reached_by_wait`), reports the progress value the semaphore holds as reached,
    /// and a value no submission has yet signalled as not reached, at once.
    #[test]
    fn every_device_tells_a_reached_progress_value_from_one_to_come_without_blocking() {
        each_device("zero_timeout_wait", |name, _, device| {
            let core = Arc::clone(device.core());
            let progress = core.done.progress;
            // SAFETY: the progress semaphore is the device's.
            let held = unsafe { core.device.get_semaphore_counter_value(progress) }
                .unwrap_or_else(|error| panic!("{name}: the progress counter: {error}"));
            let wait = |value| crate::complete::reached_by_wait(&core.device, progress, value);
            assert_eq!(
                wait(held),
                Ok(true),
                "{name}: the value the semaphore holds"
            );
            // Far past any value the device's own work signals while the test runs.
            let to_come = held + (1 << 32);
            let started = Instant::now();
            assert_eq!(wait(to_come), Ok(false), "{name}: a value to come");
            assert!(
                started.elapsed() < PATIENCE,
                "{name}: the wait for a value to come took {:?}",
                started.elapsed()
            );
        });
    }

    /// A command buffer goes through Metal's states, runs its handlers in order (its scheduled
    /// handlers on the submit thread, its completed handlers on the completion thread), and runs
    /// its commands; a queue keeps enqueue order and bounds its command buffers;
    /// Metal's refusals.
    #[test]
    fn every_device_runs_command_buffers_through_their_states_and_handlers() {
        each_device("command_states", |name, instance, device| {
            let mut arena = Arena::new(&device);
            let (a, b) = (arena.buffer(), arena.buffer());
            let queue = device.new_command_queue(64).expect("a queue");
            let log = Log::default();
            let before = device.counters().metrics();

            let first = queue.command_buffer().expect("a command buffer");
            assert_eq!(first.status(), CommandBufferStatus::NotEnqueued);
            fill(&first, &a, 0x11);
            log.handlers("first", &first);
            first.enqueue().expect("an enqueue");
            first.enqueue().expect("a second enqueue changes nothing");
            assert_eq!(first.status(), CommandBufferStatus::Enqueued);
            // Committed first, the second waits behind the first, enqueued before it.
            let second = queue.command_buffer().expect("a command buffer");
            fill(&second, &b, 0x22);
            log.handlers("second", &second);
            second.commit().expect("a commit");
            thread::sleep(Duration::from_millis(50));
            assert_eq!(second.status(), CommandBufferStatus::Committed, "{name}");
            first.commit().expect("a commit");
            finished(name, &second);
            assert_eq!(first.status(), CommandBufferStatus::Completed, "{name}");
            assert_eq!(
                (second.status(), second.error()),
                (CommandBufferStatus::Completed, None)
            );
            assert_eq!((word(&a), word(&b)), (0x11, 0x22), "{name}");
            let lines = log.lines();
            let at = |line: &str| {
                lines
                    .iter()
                    .position(|logged| logged == line)
                    .unwrap_or_else(|| panic!("{name}: no {line:?} in {lines:?}"))
            };
            assert_eq!(lines.len(), 4, "{name}: {lines:?}");
            assert!(at("first scheduled Scheduled") < at("first completed Completed None"));
            assert!(at("second scheduled Scheduled") < at("second completed Completed None"));
            assert!(at("first completed Completed None") < at("second completed Completed None"));
            let after = device.counters().metrics();
            // Each segment with commands has one barrier, at its end.
            assert_eq!(after.barriers - before.barriers, 2, "{name}");
            assert!(after.submissions > before.submissions, "{name}");

            // After commit, Metal refuses.
            let refused = |result: Result<(), Error>| {
                matches!(result, Err(Error::Refused(Refusal::Committed)))
            };
            let event = device.new_event().expect("an event");
            assert!(refused(first.commit()), "{name}");
            assert!(refused(first.enqueue()), "{name}");
            assert!(refused(first.add_completed_handler(|_| {})), "{name}");
            assert!(refused(first.add_scheduled_handler(|_| {})), "{name}");
            assert!(refused(first.encode_signal_event(&event, 1)), "{name}");
            assert!(refused(first.encode_wait_for_event(&event, 1)), "{name}");
            assert!(matches!(
                device.new_command_queue(0),
                Err(Error::Refused(Refusal::NoCommandBuffers))
            ));

            // A queue of one: the next command buffer waits until the first retires.
            let single = device.new_command_queue(1).expect("a queue");
            let held = single.command_buffer().expect("a command buffer");
            let (sent, received) = mpsc::channel();
            let waiter = {
                let single = single.clone();
                thread::spawn(move || {
                    let next = single.command_buffer().expect("a command buffer");
                    sent.send(()).expect("a send");
                    next
                })
            };
            assert!(received.recv_timeout(Duration::from_millis(50)).is_err());
            held.commit().expect("a commit");
            received.recv_timeout(PATIENCE).expect("room in the queue");
            drop(waiter.join().expect("the waiter"));

            // Another device's event is refused.
            let other = Device::new(&physical_of(instance, &device), DeviceOptions::default())
                .expect("a second device");
            let foreign = other.new_event().expect("an event");
            let local = queue.command_buffer().expect("a command buffer");
            for result in [
                local.encode_signal_event(&foreign, 1),
                local.encode_wait_for_event(&foreign, 1),
            ] {
                assert!(matches!(result, Err(Error::Refused(Refusal::OtherDevice))));
            }
            drop((local, foreign, other));
            drop((a, b, arena, queue, single, event, device));
        });
    }

    /// A command buffer made and enqueued beforehand, encoded on another thread and committed after
    /// the one enqueued behind it: two command buffers made and enqueued in order on one thread,
    /// the second encoded and committed there, then the first encoded and committed on a second
    /// thread. The second waits behind the first until the first is committed, they complete in
    /// enqueue order, and each has its own status, handlers and GPU times.
    #[test]
    fn every_device_runs_enqueued_command_buffers_in_enqueue_order_across_threads() {
        each_device("command_enqueue_threads", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, b) = (arena.buffer(), arena.buffer());
            let queue = queue(&device);
            let log = Log::default();
            let first = queue.command_buffer().expect("a command buffer");
            let second = queue.command_buffer().expect("a command buffer");
            first.enqueue().expect("an enqueue");
            second.enqueue().expect("an enqueue");
            log.handlers("first", &first);
            log.handlers("second", &second);
            fill(&second, &b, 2);
            second.commit().expect("a commit");
            thread::sleep(Duration::from_millis(50));
            assert_eq!(
                (first.status(), second.status()),
                (
                    CommandBufferStatus::Enqueued,
                    CommandBufferStatus::Committed
                ),
                "{name}: the second waits behind the first"
            );
            assert!(log.lines().is_empty(), "{name}: {:?}", log.lines());
            let encoder = {
                let (first, a) = (first.clone(), a.clone());
                thread::spawn(move || {
                    fill(&first, &a, 1);
                    first.commit().expect("a commit");
                })
            };
            encoder.join().expect("the encoding thread");
            waits_return(
                name,
                "enqueued across threads",
                vec![first.clone(), second.clone()],
            );
            for command_buffer in [&first, &second] {
                assert_eq!(
                    (command_buffer.status(), command_buffer.error()),
                    (CommandBufferStatus::Completed, None),
                    "{name}"
                );
            }
            assert_eq!((word(&a), word(&b)), (1, 2), "{name}");
            let lines = log.lines();
            let at = |line: &str| {
                (lines.iter().position(|logged| logged == line))
                    .unwrap_or_else(|| panic!("{name}: no {line:?} in {lines:?}"))
            };
            assert_eq!(lines.len(), 4, "{name}: {lines:?}");
            assert!(at("first scheduled Scheduled") < at("first completed Completed None"));
            assert!(at("second scheduled Scheduled") < at("second completed Completed None"));
            assert!(
                at("first completed Completed None") < at("second completed Completed None"),
                "{name}: completions in enqueue order: {lines:?}"
            );
            let times = [&first, &second].map(|cb| (cb.gpu_start_time(), cb.gpu_end_time()));
            println!("{name}: GPU start and end of the first and the second {times:?}");
            for (start, end) in times {
                assert!(0 < start && start <= end, "{name}: {times:?}");
            }
            assert!(times[0].1 <= times[1].1, "{name}: {times:?}");
            drop((first, second, a, b, arena, queue, device));
        });
    }

    /// The physical device a device was opened on, for a second device.
    fn physical_of(instance: &Instance, device: &Device) -> crate::PhysicalDevice {
        let raw = device.raw().physical_device;
        instance
            .physical_devices()
            .expect("the physical devices")
            .into_iter()
            .find(|physical| physical.raw() == raw)
            .expect("the device's physical device")
    }

    /// A command buffer that signals E, waits for F and works on: the host sees E and the work
    /// before it while the rest is held, and sets F, and then the rest runs. Another queue goes
    /// on meanwhile; the same queue waits behind it.
    #[test]
    fn every_device_holds_a_command_buffer_from_its_waiting_segment_on() {
        each_device("command_holds", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, b, c, d) = (
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
            );
            let (e, f) = (shared_event(&device), shared_event(&device));
            let (queue, other) = (queue(&device), queue(&device));

            let held = queue.command_buffer().expect("a command buffer");
            fill(&held, &a, 1);
            held.encode_signal_event(e.as_event(), 1).expect("a signal");
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            fill(&held, &b, 2);
            held.commit().expect("a commit");
            let behind = queue.command_buffer().expect("a command buffer");
            fill(&behind, &d, 4);
            behind.commit().expect("a commit");

            assert!(
                e.wait_until_signaled_value(1, PATIENCE)
                    .expect("a wait outside a scheduled handler"),
                "{name}: E reached"
            );
            assert_eq!(e.signaled_value(), 1);
            assert_eq!(word(&a), 1, "{name}: the work before the signal");
            let beside = other.command_buffer().expect("a command buffer");
            fill(&beside, &c, 3);
            beside.commit().expect("a commit");
            finished(name, &beside);
            assert_eq!(word(&c), 3, "{name}: another queue goes on");
            assert_eq!(held.status(), CommandBufferStatus::Committed, "{name}");
            assert_eq!(behind.status(), CommandBufferStatus::Committed, "{name}");
            assert_eq!((word(&b), word(&d)), (0, 0), "{name}: held");

            f.set_signaled_value(1);
            finished(name, &held);
            finished(name, &behind);
            assert_eq!(
                (held.status(), behind.status()),
                (
                    CommandBufferStatus::Completed,
                    CommandBufferStatus::Completed
                )
            );
            assert_eq!((word(&b), word(&d)), (2, 4), "{name}");
            drop((a, b, c, d, arena, e, f, queue, other, device));
        });
    }

    /// The consumer's contract at the end of a device's life. A call that configures what has
    /// started refuses typed, never Ok for nothing: each call of a committed command buffer, and
    /// the release of a block already released. And with every consumer handle dropped while one
    /// command buffer is held behind a wait and another waits behind it, the device tears down
    /// and every gauge reaches zero (`each_device`). A dropped device is never handed out again:
    /// only `Device::new` makes a `Device`, and the runtime's threads hold its core, never one.
    #[test]
    fn every_device_refuses_configuration_after_start_and_tears_down_with_work_held() {
        each_device("command_contract", |name, _, device| {
            let event = shared_event(&device);
            let queue = queue(&device);
            let held = queue.command_buffer().expect("a command buffer");
            held.encode_wait_for_event(event.as_event(), 1)
                .expect("a wait");
            held.commit().expect("a commit");
            let behind = queue.command_buffer().expect("a command buffer");
            behind.commit().expect("a commit");
            let committed = |call: &str, result: Result<(), Error>| {
                assert!(
                    matches!(result, Err(Error::Refused(Refusal::Committed))),
                    "{name}: {call}: {result:?}"
                );
            };
            committed("commit", held.commit());
            committed("enqueue", held.enqueue());
            committed("add_completed_handler", held.add_completed_handler(|_| {}));
            committed("add_scheduled_handler", held.add_scheduled_handler(|_| {}));
            committed(
                "encode_signal_event",
                held.encode_signal_event(event.as_event(), 2),
            );
            committed(
                "encode_wait_for_event",
                held.encode_wait_for_event(event.as_event(), 2),
            );
            committed(
                "compute_command_encoder",
                held.compute_command_encoder(DispatchType::Serial).map(drop),
            );
            let block = device
                .take_block(MemoryClass::Shared, 1)
                .expect("a Shared block");
            let id = block.id;
            device.release_block(block).expect("a release");
            assert!(
                matches!(device.new_buffer(id, 0, 256, Some(0)), Err(Error::NoSuchBlock(gone)) if gone == id),
                "{name}: a released block's number"
            );
            assert_eq!(
                held.status(),
                CommandBufferStatus::Committed,
                "{name}: held"
            );
            drop((held, behind, queue, event, device));
        });
    }

    pub(crate) fn shared_event(device: &Device) -> SharedEvent {
        device.new_shared_event().expect("a shared event")
    }

    fn queue(device: &Device) -> CommandQueue {
        device.new_command_queue(64).expect("a queue")
    }

    /// Waits for each of `command_buffers` with `wait_until_completed` on a thread of its own, and
    /// fails the test where they have not all returned within [`PATIENCE`]: a completion that never
    /// comes fails the case instead of stalling the run.
    fn waits_return(name: &str, case: &str, command_buffers: Vec<CommandBuffer>) {
        let (returned, waited) = mpsc::channel();
        thread::spawn(move || {
            for command_buffer in &command_buffers {
                command_buffer
                    .wait_until_completed()
                    .expect("a wait outside a scheduled handler");
            }
            let _ = returned.send(command_buffers);
        });
        let command_buffers = (waited.recv_timeout(PATIENCE))
            .unwrap_or_else(|_| panic!("{name}: {case}: a wait_until_completed did not return"));
        for command_buffer in &command_buffers {
            // The GPU times, as a consumer reads them once it completed: one that ran has both,
            // in order; one ended before it ran has none.
            let (start, end) = (
                command_buffer.gpu_start_time(),
                command_buffer.gpu_end_time(),
            );
            match command_buffer.error() {
                None => assert!(0 < start && start <= end, "{name}: {case}: {start}..{end}"),
                Some(_) => assert_eq!((start, end), (0, 0), "{name}: {case}"),
            }
        }
    }

    /// Every chain of command buffers with nothing for the GPU, with fence updates alone, or with
    /// an event's signal or wait alone retires: each one's completed handler runs and its
    /// `wait_until_completed` returns, and its GPU times read back, whatever comes before it on its
    /// queue or on another. The cases: an empty one committed while a group is held; an empty one
    /// as the last commit before a wait; a fence-only one between two that fill; an empty one
    /// behind another, nothing submitted behind either, and the same after work that completed;
    /// those with a second queue's commits between them; a command buffer ended `NotRun` after
    /// its first segment was submitted; and one whose only segment signals an event, then one whose
    /// only segment waits on it.
    #[test]
    fn every_device_retires_every_chain_of_empty_fence_only_and_event_only_command_buffers() {
        each_device("command_chain", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, b) = (arena.buffer(), arena.buffer());
            let ran = Arc::new(AtomicUsize::new(0));
            let mut committed = 0;
            let mut commit = |command_buffer: &CommandBuffer| {
                let ran = Arc::clone(&ran);
                (command_buffer.add_completed_handler(move |_| {
                    ran.fetch_add(1, Ordering::SeqCst);
                }))
                .expect("a handler");
                command_buffer.commit().expect("a commit");
                committed += 1;
            };
            let empty = |queue: &CommandQueue| queue.command_buffer().expect("a command buffer");
            let filled = |queue: &CommandQueue, buffer: &Buffer, word: u32| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                fill(&command_buffer, buffer, word);
                command_buffer
            };
            let fence_only = |queue: &CommandQueue, fence: &Fence, wait: bool| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                match wait {
                    true => encoder.wait_for_fence(fence).expect("a fence wait"),
                    false => encoder.update_fence(fence).expect("a fence update"),
                }
                encoder.end_encoding();
                command_buffer
            };

            // An empty command buffer committed while its queue holds a group, waited for alone.
            let queue = self::queue(&device);
            let mut held = Vec::new();
            behind_busy(name, &device, &queue, |busy, _, _| {
                let work = filled(&queue, &a, 1);
                commit(&work);
                let nothing = empty(&queue);
                commit(&nothing);
                busy.open();
                waits_return(name, "an empty one in a held group", vec![nothing.clone()]);
                held.extend([work, nothing]);
            });
            waits_return(name, "the held group's work", held);

            // An empty command buffer as the last commit before a wait.
            let work = filled(&queue, &a, 2);
            commit(&work);
            let nothing = empty(&queue);
            commit(&nothing);
            waits_return(name, "an empty one last", vec![nothing, work]);

            // A fence-only command buffer between two that fill.
            let fence = device.new_fence();
            let before = filled(&queue, &a, 3);
            commit(&before);
            let update = fence_only(&queue, &fence, false);
            commit(&update);
            let wait = fence_only(&queue, &fence, true);
            fill(&wait, &b, 3);
            commit(&wait);
            waits_return(
                name,
                "a fence-only one between two",
                vec![update, wait, before],
            );
            assert_eq!((word(&a), word(&b)), (3, 3), "{name}");

            // An empty command buffer behind another, on a queue that never submitted, then the
            // same behind work that completed.
            let fresh = self::queue(&device);
            let (first, second) = (empty(&fresh), empty(&fresh));
            commit(&first);
            commit(&second);
            waits_return(name, "two empty ones, nothing before", vec![second, first]);
            let work = filled(&fresh, &a, 4);
            commit(&work);
            waits_return(name, "work alone", vec![work]);
            let (first, second) = (empty(&fresh), empty(&fresh));
            commit(&first);
            commit(&second);
            waits_return(name, "two empty ones behind work", vec![second, first]);

            // The same with a second queue's commits between them.
            let other = self::queue(&device);
            let other_fence = device.new_fence();
            let one = empty(&fresh);
            commit(&one);
            let work = filled(&other, &b, 5);
            commit(&work);
            let two = empty(&fresh);
            commit(&two);
            let update = fence_only(&other, &other_fence, false);
            commit(&update);
            let three = empty(&other);
            commit(&three);
            waits_return(
                name,
                "empty ones between two queues",
                vec![three, two, update, one, work],
            );

            // Ended `NotRun` after its first segment was submitted: the part after an event wait
            // the host never meets.
            let event = shared_event(&device);
            let partial = filled(&queue, &a, 6);
            partial
                .encode_wait_for_event(event.as_event(), 1)
                .expect("a wait");
            fill(&partial, &b, 6);
            commit(&partial);
            within("the part before the wait", || word(&a) == 6);
            queue.abandon();
            waits_return(
                name,
                "NotRun after a submitted segment",
                vec![partial.clone()],
            );
            assert_eq!(partial.error(), Some(CommandBufferError::NotRun), "{name}");

            // One whose only segment signals an event, then one whose only segment waits on it.
            let signal = empty(&queue);
            (signal.encode_signal_event(event.as_event(), 2)).expect("a signal");
            commit(&signal);
            let waits = empty(&queue);
            (waits.encode_wait_for_event(event.as_event(), 2)).expect("a wait");
            commit(&waits);
            waits_return(name, "event-only ones", vec![waits, signal]);

            within("every completed handler", || {
                ran.load(Ordering::SeqCst) == committed
            });
            println!("{name}: {committed} command buffers retired in every chain");
            drop((
                a,
                b,
                arena,
                event,
                fence,
                other_fence,
                queue,
                fresh,
                other,
                device,
            ));
        });
    }

    /// A command buffer that takes no submission of its own sends the group its queue holds, so
    /// its completion waits only on submitted work: one with nothing for the GPU committed behind
    /// the group, and one ended `NotRun` by `abandon` while it waited behind the group on an event
    /// the host never sets. Behind the busy submissions a queue keeps in flight a fill is held (no
    /// submission of its own);
    /// each such command buffer then has the group submitted before the first busy one ends on the
    /// GPU, where a group left held goes only once that one completes.
    #[test]
    fn every_device_sends_the_held_group_a_command_buffer_without_a_submission_retires_behind() {
        each_device("command_unheld", |name, _, device| {
            let mut arena = Arena::new(&device);
            let a = arena.buffer();
            let event = shared_event(&device);
            let queue = queue(&device);
            let submissions = || device.counters().metrics().submissions;
            for abandoned in [false, true] {
                let case = if abandoned { "ended NotRun" } else { "empty" };
                let mut kept = Vec::new();
                behind_busy(name, &device, &queue, |busy, before, _| {
                    let work = queue_fill(&queue, &a, 1);
                    within(name, || work.status() != CommandBufferStatus::Committed);
                    let held = submissions();
                    assert_eq!(held, before + AHEAD, "{name}: {case}: the fill is held");
                    let behind = queue.command_buffer().expect("a command buffer");
                    if abandoned {
                        (behind.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
                    }
                    behind.commit().expect("a commit");
                    if abandoned {
                        queue.abandon();
                    }
                    within(name, || submissions() > held);
                    let seen = host_time();
                    busy.open();
                    let first = &busy.running()[0];
                    finished(name, first);
                    assert!(
                        first.gpu_end_time() > seen,
                        "{name}: {case}: the group went before the first busy one ended"
                    );
                    kept.extend([work, behind]);
                });
                waits_return(name, case, kept.clone());
                let errors: Vec<_> = kept.iter().map(CommandBuffer::error).collect();
                let ended = abandoned.then_some(CommandBufferError::NotRun);
                assert!(
                    errors.chunks(2).all(|pair| pair == [None, ended]),
                    "{name}: {case}: {errors:?}"
                );
            }
            println!("{name}: a command buffer without a submission sent the held group");
            drop((a, arena, event, queue, device));
        });
    }

    /// A wait the host's value meets is submitted with no semaphore wait (the semaphore never
    /// reaches it, so one would hang), and a GPU signal not above the host's value is left out.
    #[test]
    fn every_device_meets_a_wait_by_the_host_value_without_a_semaphore_wait() {
        each_device("command_floor", |name, _, device| {
            let (f, e) = (shared_event(&device), shared_event(&device));
            let queue = queue(&device);

            let waiting = queue.command_buffer().expect("a command buffer");
            waiting
                .encode_wait_for_event(f.as_event(), 5)
                .expect("a wait");
            waiting.commit().expect("a commit");
            thread::sleep(Duration::from_millis(20));
            assert_eq!(waiting.status(), CommandBufferStatus::Committed, "{name}");
            f.set_signaled_value(5);
            f.set_signaled_value(3);
            assert_eq!(f.signaled_value(), 5, "{name}: the host value only rises");
            finished(name, &waiting);
            assert_eq!(waiting.status(), CommandBufferStatus::Completed, "{name}");

            e.set_signaled_value(5);
            let low = queue.command_buffer().expect("a command buffer");
            low.encode_signal_event(e.as_event(), 3).expect("a signal");
            low.encode_wait_for_event(e.as_event(), 4).expect("a wait");
            low.encode_signal_event(e.as_event(), 7).expect("a signal");
            low.commit().expect("a commit");
            finished(name, &low);
            assert_eq!(low.status(), CommandBufferStatus::Completed, "{name}");
            assert_eq!(e.signaled_value(), 7, "{name}");
            drop((f, e, queue, device));
        });
    }

    /// Host sets race GPU signals of the same values, with GPU waits on them from another queue:
    /// every command buffer completes, and the semaphore only rises (the validation layer checks
    /// each signal).
    #[test]
    fn every_device_takes_host_values_racing_gpu_signals() {
        each_device("event_race", |name, _, device| {
            let e = shared_event(&device);
            let (signals, waits) = (queue(&device), queue(&device));
            for value in 1..=200 {
                let waiter = waits.command_buffer().expect("a command buffer");
                waiter
                    .encode_wait_for_event(e.as_event(), value)
                    .expect("a wait");
                waiter.commit().expect("a commit");
                let signaller = signals.command_buffer().expect("a command buffer");
                signaller
                    .encode_signal_event(e.as_event(), value)
                    .expect("a signal");
                thread::scope(|scope| {
                    scope.spawn(|| signaller.commit().expect("a commit"));
                    scope.spawn(|| e.set_signaled_value(value));
                });
                finished(name, &signaller);
                finished(name, &waiter);
                assert_eq!(waiter.status(), CommandBufferStatus::Completed, "{name}");
                if value % 100 == 0 {
                    let gauges = device.counters().gauges();
                    println!(
                        "M2V-MATRIX gauge event_race/command_buffers@{value} {}",
                        gauges.command_buffers
                    );
                    println!(
                        "M2V-MATRIX gauge event_race/events@{value} {}",
                        gauges.events
                    );
                }
            }
            assert_eq!(e.signaled_value(), 200, "{name}");
            drop((e, signals, waits, device));
        });
    }

    /// A command buffer whose wait for an event's value is met only by a submitted signal not yet
    /// run (another queue's, held with every later command behind busy work) is `Scheduled` as it
    /// is submitted, its scheduled handler run, while the signal has not run, and so is the
    /// queue's next command buffer, which waits for nothing: for a plain event's value and for a
    /// shared event's alike. MTLCommandBuffer.h describes `Scheduled` as "All dependencies have
    /// been resolved and the command buffer has been scheduled for execution", and
    /// `encodeWaitForEvent:value:` as a command that "pauses execution of this command buffer until
    /// the specified event reaches a given value": execution pauses after scheduling. The M3 Max
    /// holds the shared event's waiter at `Committed` instead, an accepted difference (docs/API.md).
    #[test]
    fn every_device_schedules_a_wait_for_an_event_s_value_as_it_is_submitted() {
        each_device("event_wait_scheduled", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, b, c, d) = (
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
            );
            let (queue, signaller_queue) = (queue(&device), self::queue(&device));
            let log = Log::default();
            let submitted = |command_buffer: &CommandBuffer| {
                let before = device.counters().metrics().submissions;
                command_buffer.commit().expect("a commit");
                within(name, || device.counters().metrics().submissions > before);
            };
            // Commits a command buffer that waits for `event` at 1, where there is one, and fills
            // `buffer`, and returns it once its submission is counted.
            let waiting = |label: &str, event: Option<&crate::Event>, buffer: &Buffer, word| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                if let Some(event) = event {
                    (command_buffer.encode_wait_for_event(event, 1)).expect("a wait");
                }
                fill(&command_buffer, buffer, word);
                log.handlers(label, &command_buffer);
                submitted(&command_buffer);
                command_buffer
            };
            let plain = device.new_event().expect("an event");
            let shared = shared_event(&device);
            for (label, event, (buffer, next_buffer), word) in [
                ("plain", &plain, (&a, &b), 1),
                ("shared", shared.as_event(), (&c, &d), 3),
            ] {
                let busy = Busy::new(&device);
                let signaller = signaller_queue.command_buffer().expect("a command buffer");
                (signaller.encode_signal_event(event, 1)).expect("a signal");
                submitted(&signaller);
                let waiter = waiting(label, Some(event), buffer, word);
                let next_label = format!("{label} next");
                let next = waiting(&next_label, None, next_buffer, word + 1);
                assert_eq!(
                    (waiter.status(), next.status()),
                    (
                        CommandBufferStatus::Scheduled,
                        CommandBufferStatus::Scheduled
                    ),
                    "{name}: {label}: scheduled as submitted, the signal not yet run"
                );
                // A scheduled handler runs after its command buffer's status reads `Scheduled`.
                within(name, || log.lines().len() == 2);
                assert_eq!(
                    log.lines(),
                    [
                        format!("{label} scheduled Scheduled"),
                        format!("{next_label} scheduled Scheduled")
                    ],
                    "{name}: {label}"
                );
                busy.open();
                for command_buffer in [&signaller, &waiter, &next] {
                    finished(name, command_buffer);
                }
                within(name, || log.lines().len() == 4);
                lock(&log.0).clear();
                drop(busy);
            }
            assert_eq!(
                [&a, &b, &c, &d].map(word),
                [1, 2, 3, 4],
                "{name}: every fill ran"
            );
            drop((
                a,
                b,
                c,
                d,
                arena,
                plain,
                shared,
                queue,
                signaller_queue,
                device,
            ));
        });
    }

    /// Links of a chain on two queues, each but the first waiting for the shared event's value the
    /// link before it signals, then copying the words of every link before it from that link's
    /// buffer, writing its own after them, and signalling the next, all behind busy work: every
    /// link goes to the driver and reads `Scheduled`, its scheduled handler run, before the first
    /// has run, and the submit thread holds none of them for a value. No semaphore wait is
    /// submitted for a link: once the busy work ends, the links complete in order, and each holds
    /// the words of every link before it and its own, as the barrier ending each link's commands
    /// orders the next link's after them on the device's one queue. A link waiting for a value no
    /// submitted signal promises, committed while the submit thread is parked in a scheduled
    /// handler, stays `Committed` while the links complete; once released, the thread holds it,
    /// and the host sets the value only then: the hold is counted in `Metrics::event_holds` as it
    /// goes. A value set before the thread first looks would meet the wait there, with nothing
    /// held.
    #[test]
    fn every_device_submits_a_shared_event_chain_across_two_queues_before_its_first_link_runs() {
        const LINKS: u32 = 5;
        each_device("event_chain", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffers: Vec<Buffer> = (0..=LINKS).map(|_| arena.buffer()).collect();
            let queues = [queue(&device), queue(&device)];
            let log = Log::default();
            let chain = shared_event(&device);
            let holds = || device.counters().metrics().event_holds;
            let before = holds();
            // Link `link`'s commands: the words of every link before it, from the buffer of the
            // one before it, then its own word, `link + 1`, after them.
            let record = |command_buffer: &CommandBuffer, link: u32| {
                let at = link as usize;
                let (own, own_offset) = buffers[at].device_range();
                let earlier = at.checked_sub(1).map(|before| &buffers[before]);
                let kept: Vec<&Buffer> = earlier.into_iter().chain([&buffers[at]]).collect();
                let bytes = 4 * u64::from(link);
                command_buffer
                    .record_for_test(&kept, |device, command| {
                        if let Some(earlier) = earlier {
                            let (source, source_offset) = earlier.device_range();
                            let region = vk::BufferCopy::default()
                                .src_offset(source_offset)
                                .dst_offset(own_offset)
                                .size(bytes);
                            // SAFETY: the command buffer is recording outside any render pass;
                            // the regions are the buffers' own and do not overlap.
                            unsafe { device.cmd_copy_buffer(command, source, own, &[region]) };
                        }
                        // SAFETY: as above; the word is the buffer's, past the copy's words.
                        unsafe {
                            device.cmd_fill_buffer(command, own, own_offset + bytes, 4, link + 1)
                        };
                    })
                    .expect("a link's commands");
            };
            let words = |buffer: &Buffer, count: usize| -> Vec<u32> {
                let contents = buffer.contents().expect("a Shared buffer");
                // SAFETY: the buffer is alive and 256 bytes long, past `count` words.
                (0..count)
                    .map(|at| unsafe { contents.cast::<u32>().as_ptr().add(at).read_volatile() })
                    .collect()
            };
            let busy = Busy::new(&device);
            let links: Vec<CommandBuffer> = (0..LINKS)
                .map(|link| {
                    let queue = &queues[link as usize % 2];
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    if link > 0 {
                        (command_buffer.encode_wait_for_event(chain.as_event(), u64::from(link)))
                            .expect("a wait");
                    }
                    record(&command_buffer, link);
                    (command_buffer.encode_signal_event(chain.as_event(), u64::from(link) + 1))
                        .expect("a signal");
                    log.handlers(&format!("link {link}"), &command_buffer);
                    command_buffer.commit().expect("a commit");
                    command_buffer
                })
                .collect();
            for (link, command_buffer) in links.iter().enumerate() {
                within(
                    &format!("{name}: link {link} scheduled while the first has not run"),
                    || command_buffer.status() == CommandBufferStatus::Scheduled,
                );
            }
            within(name, || log.lines().len() == LINKS as usize);
            assert_eq!(
                chain.signaled_value(),
                0,
                "{name}: no link has run, the busy work holding the device"
            );
            assert_eq!(holds(), before, "{name}: no link held for a value");
            // The submit thread parked in a scheduled handler: it first looks at the link committed
            // next only once the test releases it, after the links have completed, as a loaded
            // host may run it late.
            let park = Arc::new(std::sync::Barrier::new(2));
            let parker = queues[1].command_buffer().expect("a command buffer");
            let parked = Arc::clone(&park);
            (parker.add_scheduled_handler(move |_| {
                parked.wait();
                parked.wait();
            }))
            .expect("a handler");
            parker.commit().expect("a commit");
            park.wait();
            let mut release = Release::of(&park);
            // A link waiting for a value no submitted signal promises: held until the host sets
            // it.
            let last = queues[0].command_buffer().expect("a command buffer");
            let value = u64::from(LINKS) + 2;
            (last.encode_wait_for_event(chain.as_event(), value)).expect("a wait");
            record(&last, LINKS);
            last.commit().expect("a commit");
            // The first link, then the rest in order.
            busy.open();
            for command_buffer in &links {
                finished(name, command_buffer);
            }
            assert_eq!(
                chain.signaled_value(),
                u64::from(LINKS),
                "{name}: every link's signal"
            );
            assert_eq!(
                last.status(),
                CommandBufferStatus::Committed,
                "{name}: the link waiting for a value nothing promises, not yet looked at"
            );
            // The host sets the value only once the submit thread, released, holds the link for
            // it: set before the thread has looked, it meets the wait at that first look, and
            // nothing is held.
            release.now();
            within(&format!("{name}: the link held for its value"), || {
                device.core().work.holding() == 1
            });
            chain.set_signaled_value(value);
            finished(name, &last);
            finished(name, &parker);
            let held: Vec<u64> = (holds().iter().zip(&before))
                .map(|(now, was)| now - was)
                .collect();
            assert_eq!(
                held.iter().sum::<u64>(),
                1,
                "{name}: one segment held for a value, by length {held:?}"
            );
            let completed: Vec<String> = (log.lines().into_iter())
                .filter(|line| line.contains("completed"))
                .collect();
            let expected: Vec<String> = (0..LINKS)
                .map(|link| format!("link {link} completed Completed None"))
                .collect();
            assert_eq!(completed, expected, "{name}: the links completed in order");
            for (at, buffer) in buffers.iter().enumerate() {
                assert_eq!(
                    words(buffer, at + 1),
                    (1..=at as u32 + 1).collect::<Vec<u32>>(),
                    "{name}: link {at}'s words: every link's before it, then its own"
                );
            }
            drop((
                last, parker, links, busy, buffers, arena, chain, queues, device,
            ));
        });
    }

    /// An event's signal on a queue with three submissions running goes to the driver at once,
    /// with the group held there, and a waiter on another queue that the submit thread holds for
    /// that value goes after it: two submissions, both before the busy work ends. A held group
    /// never keeps back a signal. For a plain event's value and for a shared event's alike; the
    /// waiter is held once, counted in `Metrics::event_holds`.
    #[test]
    fn every_device_sends_a_signal_from_a_queue_three_deep_to_a_held_waiter_on_another_queue() {
        each_device("event_signal_three_deep", |name, _, device| {
            let mut arena = Arena::new(&device);
            let plain = device.new_event().expect("an event");
            let shared = shared_event(&device);
            let holds = || device.counters().metrics().event_holds.iter().sum::<u64>();
            for (label, event) in [("plain", &plain), ("shared", shared.as_event())] {
                let (grouped, signalled, waited) = (arena.buffer(), arena.buffer(), arena.buffer());
                let (signaller_queue, waiter_queue) = (queue(&device), queue(&device));
                let held_before = holds();
                behind_busy(name, &device, &signaller_queue, |busy, before, _| {
                    let sent = || device.counters().metrics().submissions - before;
                    let waiter = waiter_queue.command_buffer().expect("a command buffer");
                    (waiter.encode_wait_for_event(event, 1)).expect("a wait");
                    fill(&waiter, &waited, 3);
                    waiter.commit().expect("a commit");
                    // Committed after the waiter, so staged once the submit thread has found the
                    // waiter's value neither set nor promised: held in a group behind the busy
                    // work.
                    let held = queue_fill(&signaller_queue, &grouped, 1);
                    within(name, || held.status() != CommandBufferStatus::Committed);
                    assert_eq!(
                        (waiter.status(), sent()),
                        (CommandBufferStatus::Committed, AHEAD),
                        "{name}: {label}: the waiter held for its value, the fill in a group"
                    );
                    let signaller = signaller_queue.command_buffer().expect("a command buffer");
                    fill(&signaller, &signalled, 2);
                    (signaller.encode_signal_event(event, 1)).expect("a signal");
                    signaller.commit().expect("a commit");
                    within(name, || sent() >= AHEAD + 2);
                    let seen = host_time();
                    assert_eq!(
                        sent(),
                        AHEAD + 2,
                        "{name}: {label}: the group with the signal, then the waiter"
                    );
                    busy.open();
                    for command_buffer in [&held, &signaller, &waiter] {
                        finished(name, command_buffer);
                    }
                    assert!(
                        busy.running()[0].gpu_end_time() > seen,
                        "{name}: {label}: both went before the busy work ended"
                    );
                });
                assert_eq!(
                    [&grouped, &signalled, &waited].map(word),
                    [1, 2, 3],
                    "{name}: {label}: every fill ran"
                );
                assert_eq!(
                    holds() - held_before,
                    1,
                    "{name}: {label}: the waiter held once"
                );
                drop((grouped, signalled, waited, signaller_queue, waiter_queue));
            }
            drop((arena, plain, shared, device));
        });
    }

    /// A listener runs when the GPU reaches its value in the middle of a command buffer, at once
    /// for a value already reached, and for a host value.
    #[test]
    fn every_device_runs_a_listener_when_the_gpu_reaches_its_value_mid_buffer() {
        each_device("event_listeners", |name, _, device| {
            let (e, f) = (shared_event(&device), shared_event(&device));
            let queue = queue(&device);
            let (sent, received) = mpsc::channel::<(u32, u64)>();
            let listen = |at: u64, label: u32| {
                let sent = sent.clone();
                e.notify_listener(at, move |value| sent.send((label, value)).expect("a send"))
                    .expect("a listener");
            };
            listen(1, 1);
            let held = queue.command_buffer().expect("a command buffer");
            held.encode_signal_event(e.as_event(), 1).expect("a signal");
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            held.commit().expect("a commit");
            assert_eq!(received.recv_timeout(PATIENCE), Ok((1, 1)), "{name}");
            assert_eq!(
                held.status(),
                CommandBufferStatus::Committed,
                "{name}: still held"
            );
            listen(1, 2);
            assert_eq!(
                received.recv_timeout(PATIENCE),
                Ok((2, 1)),
                "{name}: already reached"
            );
            listen(3, 3);
            f.set_signaled_value(1);
            finished(name, &held);
            assert!(received.try_recv().is_err(), "{name}: 3 is not reached");
            e.set_signaled_value(3);
            assert_eq!(
                received.recv_timeout(PATIENCE),
                Ok((3, 3)),
                "{name}: a host value"
            );
            drop((e, f, queue, device));
        });
    }

    /// Times, on the host clock: the kernel start at commit precedes the kernel end at scheduling;
    /// each GPU start and end fall between `host_time` read before its commit and the host time
    /// its completed handler reads, through pairs of timestamp queries its command buffers reuse,
    /// with or without commands. (Each command buffer of one submission starting no earlier than
    /// the one before ends is the group test's.)
    #[test]
    fn every_device_times_command_buffers_on_the_host_clock() {
        each_device("command_times", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            // Fewer places than command buffers, so pairs of queries are reused.
            let queue = device.new_command_queue(4).expect("a queue");
            // Each command buffer's index, GPU start and end, and the host time its completed
            // handler read.
            type Ends = Arc<Mutex<Vec<(usize, u64, u64, u64)>>>;
            let ends: Ends = Arc::default();
            let first = host_time();
            let mut committed = Vec::new();
            let mut before = Vec::new();
            for index in 0..24 {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                // Every third has no commands.
                if index % 3 != 2 {
                    fill(&command_buffer, &buffer, index as u32);
                }
                let ends = Arc::clone(&ends);
                command_buffer
                    .add_completed_handler(move |cb| {
                        let at = host_time();
                        lock(&ends).push((index, cb.gpu_start_time(), cb.gpu_end_time(), at))
                    })
                    .expect("a completed handler");
                before.push(host_time());
                command_buffer.commit().expect("a commit");
                committed.push(command_buffer);
            }
            for command_buffer in &committed {
                finished(name, command_buffer);
            }
            let ends = lock(&ends).clone();
            let order: Vec<usize> = ends.iter().map(|(index, ..)| *index).collect();
            assert_eq!(
                order,
                (0..24).collect::<Vec<_>>(),
                "{name}: completion order"
            );
            for (((_, start, end, handled), command_buffer), before) in
                ends.iter().zip(&committed).zip(&before)
            {
                let (kernel_start, kernel_end) = (
                    command_buffer.kernel_start_time(),
                    command_buffer.kernel_end_time(),
                );
                assert!(
                    first <= kernel_start && kernel_start <= kernel_end,
                    "{name}"
                );
                assert!(
                    before <= start && start <= end && end <= handled,
                    "{name}: {before} {start} {end} {handled}"
                );
                assert_eq!(
                    (
                        command_buffer.gpu_start_time(),
                        command_buffer.gpu_end_time()
                    ),
                    (*start, *end),
                    "{name}"
                );
            }
        });
    }

    /// A command buffer's GPU times are never later than the completion the runtime saw, by
    /// however much a calibration's host time trails its GPU tick, and stay apart by their
    /// timestamps' own span: here each calibration read a second late, whose placed times would
    /// fall after the completed handler ran.
    #[test]
    fn every_device_places_gpu_times_by_the_completion_seen_and_keeps_their_span() {
        each_device("command_times_late", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = device.new_command_queue(4).expect("a queue");
            let core = device.core();
            core.calibration_lateness
                .store(1_000_000_000, Ordering::Relaxed);
            // The first completion calibrates again, a period after the device's own reading.
            thread::sleep(Duration::from_nanos(
                crate::times::CALIBRATION_PERIOD * 3 / 2,
            ));
            lock(&core.tick_spans).clear();
            // Each command buffer's GPU start and end, and the host time its completed handler
            // read.
            type Ends = Arc<Mutex<Vec<(u64, u64, u64)>>>;
            let ends: Ends = Arc::default();
            let mut committed = Vec::new();
            let mut before = Vec::new();
            for index in 0..8 {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                fill(&command_buffer, &buffer, index);
                let ends = Arc::clone(&ends);
                command_buffer
                    .add_completed_handler(move |cb| {
                        let at = host_time();
                        lock(&ends).push((cb.gpu_start_time(), cb.gpu_end_time(), at))
                    })
                    .expect("a completed handler");
                before.push(host_time());
                command_buffer.commit().expect("a commit");
                committed.push(command_buffer);
            }
            for command_buffer in &committed {
                finished(name, command_buffer);
            }
            let ends = lock(&ends).clone();
            let spans = lock(&core.tick_spans).clone();
            assert_eq!(
                (ends.len(), spans.len()),
                (8, 8),
                "{name}: every one placed"
            );
            for (((start, end, handled), span), before) in ends.iter().zip(&spans).zip(&before) {
                assert!(
                    before <= start && start <= end && end <= handled,
                    "{name}: {before} {start} {end} {handled}"
                );
                assert_eq!(end - start, *span, "{name}: the span of {start} {end}");
            }
        });
    }

    /// The GPU's clock is calibrated against the host's as the device starts, then at a
    /// completion once 100 ms have passed since the last calibration, never for each command
    /// buffer: none while the device idles, and for a burst of 32 command buffers one at its first
    /// completion and at most one more for each 100 ms it ran.
    #[test]
    fn every_device_calibrates_its_clock_once_a_period_at_a_completion() {
        each_device("command_calibration", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let counters = &device.core().counters;
            within(name, || counters.calibrations() >= 1);
            let at_start = counters.calibrations();
            thread::sleep(Duration::from_millis(150));
            assert_eq!(
                counters.calibrations(),
                at_start,
                "{name}: no calibration without a completion"
            );
            let started = Instant::now();
            let burst: Vec<CommandBuffer> = (0..32)
                .map(|index| {
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    fill(&command_buffer, &buffer, index);
                    command_buffer.commit().expect("a commit");
                    command_buffer
                })
                .collect();
            for command_buffer in &burst {
                finished(name, command_buffer);
            }
            let elapsed = started.elapsed();
            let added = counters.calibrations() - at_start;
            let periods = elapsed.as_millis() as u64 / 100 + 1;
            assert!(
                (1..=periods).contains(&added),
                "{name}: {added} calibrations for 32 command buffers in {elapsed:?}"
            );
            assert_eq!(word(&buffer), 31, "{name}");
            drop((burst, buffer, arena, queue, device));
        });
    }

    /// A burst of 64 command buffers committed back to back with no wait: each submission ends at
    /// most sixteen of them, and a batch kept while three submissions run goes once they complete,
    /// with no host wait to send it.
    #[test]
    fn every_device_runs_a_burst_in_submissions_of_sixteen_at_most() {
        burst("command_burst", 64, 64);
    }

    /// The same burst through a queue of four places: each `command_buffer` past the fourth waits
    /// for one to retire, which sends nothing, and every command buffer completes in commit order.
    #[test]
    fn every_device_runs_a_burst_through_a_queue_of_four_places() {
        burst("command_places", 64, 4);
    }

    /// `count` command buffers, each filling the same word with its index, committed back to back
    /// on a queue of `places`, then read once the last completes, which the test polls for.
    fn burst(test: &str, count: u32, places: u32) {
        each_device(test, |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = device.new_command_queue(places).expect("a queue");
            let before = device.counters().metrics();
            let burst: Vec<CommandBuffer> = (0..count)
                .map(|index| {
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    fill(&command_buffer, &buffer, index);
                    command_buffer.commit().expect("a commit");
                    command_buffer
                })
                .collect();
            let last = burst.last().expect("a burst");
            within(name, || last.status() == CommandBufferStatus::Completed);
            for command_buffer in &burst {
                assert_eq!(
                    command_buffer.status(),
                    CommandBufferStatus::Completed,
                    "{name}"
                );
            }
            assert_eq!(word(&buffer), count - 1, "{name}");
            let submissions = device.counters().metrics().submissions - before.submissions;
            println!("{name}: {count} command buffers through {places} places in {submissions} submissions");
            assert!(
                submissions >= u64::from(count).div_ceil(crate::submit::GROUP as u64),
                "{name}: {submissions}"
            );
        });
    }

    /// A Private buffer of a whole block, for work that keeps the GPU busy.
    fn private(device: &Device) -> Buffer {
        let block = device
            .take_block(MemoryClass::Private, MemoryClass::Private.block_len())
            .expect("a Private block");
        device
            .new_buffer(block.id, 0, MemoryClass::Private.block_len(), None)
            .expect("a buffer")
    }

    /// Busy work a queue keeps in flight until the test lets it end. Ahead of it, the test submits
    /// on the runtime's `VkQueue`, under its lock, one batch of its own that waits for a timeline
    /// semaphore to reach 1, which only the host signals ([`Busy::open`]). A semaphore wait's
    /// second synchronization scope holds every command later in submission order, so each busy
    /// command buffer (a fill of a block) is submitted and in flight and does not complete before
    /// the test's checks are made, however fast the device or however loaded the host. A wait
    /// submitted before its signal is defined for a timeline semaphore, and the host signals it
    /// with `vkSignalSemaphore` (an event the host sets while a command buffer waiting on it is
    /// pending is not: `VUID-vkSetEvent-event-09543`). Dropped, it opens the work and waits for
    /// each busy command buffer and for its own batch, so a failed check does not leave the
    /// device's work waiting forever.
    pub(crate) struct Busy {
        device: Device,
        semaphore: vk::Semaphore,
        /// Signalled as the test's own batch completes.
        gate: vk::Fence,
        work: Buffer,
        running: Vec<CommandBuffer>,
    }

    impl Busy {
        pub(crate) fn new(device: &Device) -> Busy {
            let core = device.core();
            let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
            // SAFETY: the device is live; the semaphore and the fence are destroyed in `drop`.
            let (semaphore, gate) = unsafe {
                (
                    (core.device.create_semaphore(&info, None)).expect("a timeline semaphore"),
                    (core
                        .device
                        .create_fence(&vk::FenceCreateInfo::default(), None))
                    .expect("a fence"),
                )
            };
            let wait = [vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .value(1)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let batch = [vk::SubmitInfo2::default().wait_semaphore_infos(&wait)];
            let queue = lock(&core.queues.runtime);
            // SAFETY: the queue is held for the call; the batch only waits.
            unsafe { core.device.queue_submit2(*queue, &batch, gate) }.expect("the gate's batch");
            drop(queue);
            Busy {
                device: device.clone(),
                semaphore,
                gate,
                work: private(device),
                running: Vec::new(),
            }
        }

        /// Commits a busy command buffer on `queue`, returned once its submission is counted, so
        /// each is a submission of its own: committed after the one before it was submitted, it
        /// cannot join that one's group. A submit thread that wakes late (a loaded host) otherwise
        /// takes two commits into one group; the queue then has a submission fewer in flight than
        /// the test counts, and holds nothing behind them.
        fn commit(&mut self, name: &str, queue: &CommandQueue) -> CommandBuffer {
            let device = &self.device;
            let before = device.counters().metrics().submissions;
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let (target, offset) = self.work.device_range();
            // A page: the batch ahead of it holds it in flight, whatever it does, and its own GPU
            // run stays short on every device, as the queue's expected run reads it
            // (`QueueInner::ran`).
            let size = self.work.length().min(4096);
            command_buffer
                .record_for_test(&[&self.work], |device, command| {
                    // SAFETY: the command buffer is recording outside any render pass; the range
                    // is the buffer's, in whole words.
                    unsafe { device.cmd_fill_buffer(command, target, offset, size, 1) };
                })
                .expect("a fill");
            command_buffer.commit().expect("a commit");
            within(name, || device.counters().metrics().submissions > before);
            self.running.push(command_buffer.clone());
            command_buffer
        }

        /// The busy command buffers committed, in commit order.
        fn running(&self) -> &[CommandBuffer] {
            &self.running
        }

        /// Lets the busy work end: the host signals the semaphore to 1. A second call changes
        /// nothing.
        pub(crate) fn open(&self) {
            let device = &self.device.core().device;
            // SAFETY: the semaphore is the device's timeline semaphore.
            let value = unsafe { device.get_semaphore_counter_value(self.semaphore) }
                .expect("the semaphore's value");
            if value == 0 {
                let signal = vk::SemaphoreSignalInfo::default()
                    .semaphore(self.semaphore)
                    .value(1);
                // SAFETY: only this thread signals it, and 1 is above its value.
                unsafe { device.signal_semaphore(&signal) }.expect("a host signal");
            }
        }
    }

    impl Drop for Busy {
        fn drop(&mut self) {
            self.open();
            for command_buffer in &self.running {
                command_buffer
                    .wait_until_completed()
                    .expect("a wait outside a scheduled handler");
            }
            let device = &self.device.core().device;
            // SAFETY: the gate's batch, the only one that names the semaphore, completed once the
            // fence is signalled.
            unsafe {
                (device.wait_for_fences(&[self.gate], true, u64::MAX)).expect("the gate's batch");
                device.destroy_fence(self.gate, None);
                device.destroy_semaphore(self.semaphore, None);
            }
        }
    }

    /// The submissions a queue keeps in flight before it holds a group ([`IN_FLIGHT`]).
    const AHEAD: u64 = IN_FLIGHT as u64;

    /// The expected GPU run [`behind_busy`] sets on its queue: 1 ms, so up to sixteen held command
    /// buffers join one group within `submit::GROUP_RUN`.
    pub(crate) const BODY_RUN: u64 = 1_000_000;

    /// Runs `body` with as many busy command buffers committed on `queue` as a queue keeps in
    /// flight ([`IN_FLIGHT`]), each a submission of its own ([`Busy::commit`]), so its next commit
    /// is held while they wait for the host. `body` takes the busy work, the submissions counted
    /// before it, and the progress the completion thread had seen before it
    /// (`Done::completed`): every earlier command buffer's, so the progress stays there until the
    /// body opens the busy work ([`Busy::open`]) and the completion thread sees a busy one
    /// complete, and a held group goes no earlier than that. The busy work is opened after `body`
    /// where it did not, and waited for. First a command buffer of the queue with no commands of
    /// its own runs and retires, and the completion thread waits on the GPU again
    /// (`Done::busy`), so a commit in `body` is held rather than sent as that round ends. Then
    /// the queue's expected GPU run is set to [`BODY_RUN`], the test's own input, so the command
    /// buffers held behind the busy work join one group (`submit::GROUP_RUN`) however long that
    /// command buffer's run measured on a loaded host. A group sent after the busy work opens is
    /// sent at the run its retirements leave, which a loaded host can raise past the bound: a
    /// body that counts a group's submissions sends it first (`CommandQueue::send_now`).
    pub(crate) fn behind_busy(
        name: &str,
        device: &Device,
        queue: &CommandQueue,
        body: impl FnOnce(&Busy, u64, u64),
    ) {
        let first = queue.command_buffer().expect("a command buffer");
        (first.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
        first.commit().expect("a commit");
        finished(name, &first);
        within(name, || !device.core().done.busy());
        first.queue().set_run_for_test(BODY_RUN);
        let before = device.counters().metrics().submissions;
        let seen = device.core().done.completed();
        let mut busy = Busy::new(device);
        for _ in 0..IN_FLIGHT {
            busy.commit(name, queue);
        }
        body(&busy, before, seen);
        busy.open();
        for command_buffer in busy.running() {
            finished(name, command_buffer);
        }
    }

    /// A write on a queue with nothing running is submitted at once, so a command buffer of
    /// another queue committed after it reads the write, with no fence or event between them:
    /// every queue's submissions go to the device's one `VkQueue`, each segment ending in a barrier
    /// that reaches the commands after it. A third queue's busy work holds the GPU, so neither
    /// runs before both are submitted, and puts as many submissions in flight across the device
    /// as a queue keeps before it groups: the writer's queue groups on its own count alone.
    /// Metal's hazard tracking orders the same pair of tracked buffers where the writer's queue
    /// has nothing ahead of it, and not where its queue's own work is ahead of it (the README's
    /// "Queues, command buffers and events").
    #[test]
    fn every_device_submits_a_write_on_an_idle_queue_before_another_queues_later_read() {
        each_device("command_cross_queue_order", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (written, read) = (arena.buffer(), arena.buffer());
            let (busy_queue, writer_queue, reader_queue) =
                (queue(&device), queue(&device), queue(&device));
            behind_busy(name, &device, &busy_queue, |busy, _, _| {
                let sent = || device.counters().metrics().submissions;
                let before = sent();
                let writer = queue_fill(&writer_queue, &written, 0x5a5a_5a5a);
                within(name, || sent() > before);
                let reader = reader_queue.command_buffer().expect("a command buffer");
                let mut blit = reader.blit_command_encoder().expect("an encoder");
                (blit.copy_from_buffer_to_buffer(&written, 0, &read, 0, 256)).expect("a copy");
                blit.end_encoding();
                reader.commit().expect("a commit");
                within(name, || sent() > before + 1);
                busy.open();
                finished(name, &reader);
                finished(name, &writer);
                assert_eq!((writer.error(), reader.error()), (None, None), "{name}");
            });
            assert_eq!(word(&read), 0x5a5a_5a5a, "{name}: the read sees the write");
            drop((
                written,
                read,
                arena,
                busy_queue,
                writer_queue,
                reader_queue,
                device,
            ));
        });
    }

    /// The expected run of queue A of [`two_queues`]: past `GROUP_RUN` with one more behind its
    /// oldest submission, so it has room for two.
    const LONG_RUN: u64 = 20_000_000;

    /// Two queues behind the busy gate ([`Busy::new`]), so nothing they submit completes until
    /// it opens: A, its expected run [`LONG_RUN`], commits five command buffers, then B, its run
    /// [`BODY_RUN`], one. Each copies the cursor's word to its own word of the log, then fills
    /// the cursor with its number, 1 to 5 for A's and 6 for B's, so the log names the command
    /// buffer that ran just before each (0 for none). `encode(number, command_buffer, after)`
    /// encodes before (`after` false) and after that copy and fill. A's held work is checked
    /// before `body` runs and B commits: A1 and A2 submitted, the rest each scheduled with its
    /// scheduled handler run, `held` saying whether A holds a group and how many A holds for room,
    /// not yet in a group (`Metrics::held_for_room`): A3 in a group and A4 and A5 for room, or,
    /// where A3 signals an event, A3 to A5 for room. B's is staged, or held for an event's value,
    /// before the gate opens. Returns the log of A1 to B1 once each completed.
    pub(crate) fn two_queues(
        name: &str,
        device: &Device,
        held: (bool, u64),
        encode: impl Fn(u32, &CommandBuffer, bool),
        body: impl FnOnce(&CommandQueue, &[CommandBuffer]),
    ) -> Vec<u32> {
        let mut arena = Arena::new(device);
        let (log, cursor) = (arena.buffer(), arena.buffer());
        let (a, b) = (queue(device), queue(device));
        for (queue, run) in [(&a, LONG_RUN), (&b, BODY_RUN)] {
            let first = queue.command_buffer().expect("a command buffer");
            (first.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
            first.commit().expect("a commit");
            finished(name, &first);
            first.queue().set_run_for_test(run);
        }
        within(name, || !device.core().done.busy());
        let busy = Busy::new(device);
        let scheduled = Arc::new(AtomicUsize::new(0));
        let logged = |queue: &CommandQueue, number: u32| {
            let command_buffer = queue.command_buffer().expect("a command buffer");
            encode(number, &command_buffer, false);
            let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
            let at = u64::from(number) * 4;
            (blit.copy_from_buffer_to_buffer(&cursor, 0, &log, at, 4)).expect("a copy");
            (blit.fill_buffer(&cursor, 0..4, number as u8)).expect("a fill");
            blit.end_encoding();
            encode(number, &command_buffer, true);
            let counted = Arc::clone(&scheduled);
            (command_buffer.add_scheduled_handler(move |_| {
                counted.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("a handler");
            command_buffer.commit().expect("a commit");
            command_buffer
        };
        let (group, held_for_room) = held;
        let held = device.counters().metrics().held_for_room;
        let a_work: Vec<CommandBuffer> = (1..=5).map(|number| logged(&a, number)).collect();
        within(name, || scheduled.load(Ordering::SeqCst) == 5);
        let statuses: Vec<_> = a_work.iter().map(CommandBuffer::status).collect();
        assert_eq!(
            (statuses, device.core().work.in_flight(a_work[0].queue())),
            (vec![CommandBufferStatus::Scheduled; 5], (2, group)),
            "{name}: A's held work, each scheduled"
        );
        let counted = device.counters().metrics().held_for_room - held;
        assert_eq!(counted, held_for_room, "{name}: A's held for room");
        body(&a, &a_work);
        let b_work = logged(&b, 6);
        within(name, || {
            b_work.status() != CommandBufferStatus::Committed || device.core().work.holding() == 1
        });
        busy.open();
        for command_buffer in a_work.iter().chain([&b_work]) {
            finished(name, command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
        }
        let contents = log.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive and 256 bytes long, and no command writes it now.
        let words = |at: usize| unsafe { contents.cast::<u32>().as_ptr().add(at).read_volatile() };
        (1..=6).map(|at| words(at) & 0xff).collect()
    }

    /// One queue's command buffer committed after another queue's backlog runs ahead of all but
    /// two of it: behind A1 and A2, the two A keeps on the `VkQueue`, B1 runs, then A3 to A5.
    #[test]
    fn every_device_runs_a_queues_command_buffer_ahead_of_another_queues_backlog() {
        each_device("command_two_queues", |name, _, device| {
            let log = two_queues(name, &device, (true, 2), |_, _, _| {}, |_, _| {});
            assert_eq!(log, [0, 1, 6, 3, 4, 2], "{name}: the one run before each");
        });
    }

    /// A host wait for A4 (here its hint, `send_now`) sends A through to it, A5 staying held: B1
    /// then runs after A4.
    #[test]
    fn every_device_sends_a_queue_through_to_a_host_waits_command_buffer() {
        each_device("command_two_queues_sent", |name, _, device| {
            let body = |a: &CommandQueue, a_work: &[CommandBuffer]| {
                a.send_now(&a_work[3]);
                let work = &device.core().work;
                within(name, || work.in_flight(a_work[0].queue()) == (4, true));
            };
            let log = two_queues(name, &device, (true, 2), |_, _, _| {}, body);
            assert_eq!(log, [0, 1, 2, 3, 6, 4], "{name}: the one run before each");
        });
    }

    /// The consumer's cross-queue order, a timeline: A5, held for room, signals a shared event's
    /// value that B1 waits for. B1 is held for the value until A5's signal is submitted, so it
    /// runs after A1 to A5, which complete in order: a queue's signal never completes before
    /// that queue's earlier command buffers. That order held before the room rule too, every
    /// queue then on the `VkQueue` at once: there only A's held count (`two_queues`) fails, so
    /// the test guards the order under the rule and proves none of the rule's own. The two-queue
    /// tests and the fence's carry those.
    #[test]
    fn every_device_runs_a_wait_for_a_held_queues_signal_after_all_of_that_queue() {
        each_device("command_two_queues_event", |name, _, device| {
            let event = shared_event(&device);
            let encode = |number: u32, command_buffer: &CommandBuffer, after: bool| {
                match (number, after) {
                    (5, true) => command_buffer.encode_signal_event(event.as_event(), 1),
                    (6, false) => command_buffer.encode_wait_for_event(event.as_event(), 1),
                    _ => Ok(()),
                }
                .expect("an event's signal or wait")
            };
            let log = two_queues(name, &device, (true, 2), encode, |_, _| {});
            assert_eq!(log, [0, 1, 2, 3, 4, 5], "{name}: the one run before each");
        });
    }

    /// A queue whose every command buffer signals an event keeps on the `VkQueue` only what it
    /// has room for too: A1 to A5 each signal a shared event's value, their number, and A3 to A5,
    /// whose signals would each send a group at once, wait for room unstaged, so B1 runs after A1
    /// and A2 as it does behind work that signals nothing.
    #[test]
    fn every_device_runs_a_queues_command_buffer_ahead_of_another_queues_signals() {
        each_device("command_two_queues_signals", |name, _, device| {
            let event = shared_event(&device);
            let encode = |number: u32, command_buffer: &CommandBuffer, after: bool| {
                if number <= 5 && after {
                    let signal =
                        command_buffer.encode_signal_event(event.as_event(), number.into());
                    signal.expect("an event's signal");
                }
            };
            let log = two_queues(name, &device, (false, 3), encode, |_, _| {});
            assert_eq!(log, [0, 1, 6, 3, 4, 2], "{name}: the one run before each");
        });
    }

    /// So does one whose signal is in a later segment: each of A1 to A5 waits after its work for
    /// a value another event already holds, which ends its first segment (`encodeWaitForEvent`
    /// splits a segment with commands), then signals; A3 to A5 wait for room whole.
    #[test]
    fn every_device_holds_for_room_a_command_buffer_whose_signal_follows_a_wait() {
        each_device("command_two_queues_split_signals", |name, _, device| {
            let (gate, event) = (shared_event(&device), shared_event(&device));
            let encode = |number: u32, command_buffer: &CommandBuffer, after: bool| {
                if number <= 5 && after {
                    (command_buffer.encode_wait_for_event(gate.as_event(), 0)).expect("a wait");
                    let signal =
                        command_buffer.encode_signal_event(event.as_event(), number.into());
                    signal.expect("an event's signal");
                }
            };
            let log = two_queues(name, &device, (false, 3), encode, |_, _| {});
            assert_eq!(log, [0, 1, 6, 3, 4, 2], "{name}: the one run before each");
        });
    }

    /// A commit's wait for the submission lock is counted: one made while another thread holds
    /// the lock counts at least the time from the start of its count to the lock's release, here
    /// 20 ms or more. No clock of the test's bounds it from above.
    #[test]
    fn every_device_counts_a_commits_wait_for_the_submission_lock() {
        each_device("command_lock_wait", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill(&command_buffer, &buffer, 3);
            let held = device.core().work.lock();
            let committing = {
                let command_buffer = command_buffer.clone();
                thread::spawn(move || command_buffer.commit().expect("a commit"))
            };
            within("the commit's count", || {
                (device.core().work.counting).load(std::sync::atomic::Ordering::SeqCst)
            });
            let waiting = host_time();
            thread::sleep(Duration::from_millis(20));
            let released = host_time();
            drop(held);
            committing.join().expect("the committing thread");
            finished(name, &command_buffer);
            let metrics = device.counters().metrics();
            assert!(
                metrics.commit_lock_wait_max_nanos >= released - waiting,
                "{name}: {} counted, the lock held {} past the start of the commit's count",
                metrics.commit_lock_wait_max_nanos,
                released - waiting
            );
            assert!(metrics.commit_lock_wait_nanos >= metrics.commit_lock_wait_max_nanos);
            drop((command_buffer, buffer, arena, queue, device));
        });
    }

    /// A submission's wait for the runtime's queue lock and its `vkQueueSubmit2` calls are timed:
    /// one made while the test holds the lock counts at least the time from the start of its
    /// count to the lock's release, here 20 ms or more, and the calls count a time of their own,
    /// whose longest is at least their mean.
    #[test]
    fn every_device_times_its_submit_calls_and_their_wait_for_the_queue_lock() {
        each_device("command_queue_lock_wait", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let counters = device.counters();
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill(&command_buffer, &buffer, 3);
            let asking = || (device.core().work.asking).load(std::sync::atomic::Ordering::SeqCst);
            (device.core().work.asking).store(false, std::sync::atomic::Ordering::SeqCst);
            let (waiting, released) = device.with_runtime_queue(|_| {
                command_buffer.commit().expect("a commit");
                within("the submission's count", asking);
                let waiting = host_time();
                thread::sleep(Duration::from_millis(20));
                (waiting, host_time())
            });
            finished(name, &command_buffer);
            within("the call's time", || {
                counters.metrics().submit_call_nanos > before.submit_call_nanos
            });
            let after = counters.metrics();
            assert!(
                after.queue_lock_wait_max_nanos >= released - waiting,
                "{name}: {} counted, the lock held {} past the start of the submission's count",
                after.queue_lock_wait_max_nanos,
                released - waiting
            );
            assert!(
                after.queue_lock_wait_nanos - before.queue_lock_wait_nanos >= released - waiting,
                "{name}: the waits' total"
            );
            assert!(after.queue_lock_wait_nanos >= after.queue_lock_wait_max_nanos);
            let calls = after.submissions;
            assert!(calls > before.submissions, "{name}: a submission");
            assert!(
                after.submit_call_max_nanos > 0
                    && after.submit_call_max_nanos >= after.submit_call_nanos / calls
                    && after.submit_call_nanos >= after.submit_call_max_nanos,
                "{name}: the longest call {} against {} over {calls} calls",
                after.submit_call_max_nanos,
                after.submit_call_nanos
            );
            assert!(after.submit_calls_over_100_ms <= calls, "{name}");
            assert_eq!(command_buffer.error(), None, "{name}");
            drop((command_buffer, buffer, arena, queue, device));
        });
    }

    /// A completed handler that runs long holds no later submission: while the completion
    /// thread runs one command buffer's handler, here parked until the test releases it, the
    /// queue's next command buffers are each submitted at their commit, past [`IN_FLIGHT`] of
    /// them, as none is held in a group the busy completion thread could not send at a
    /// completion. Their handlers and their completion still wait for the parked one, in order.
    #[test]
    fn every_device_submits_while_a_completed_handler_runs() {
        each_device("command_handler_holds_no_submission", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let counters = device.counters();
            crate::tests::blit::on_device::settled(&device);
            let park = Arc::new(std::sync::Barrier::new(2));
            let first = queue.command_buffer().expect("a command buffer");
            fill(&first, &buffer, 1);
            let parked = Arc::clone(&park);
            (first.add_completed_handler(move |_| {
                parked.wait();
                parked.wait();
            }))
            .expect("a handler");
            first.commit().expect("a commit");
            park.wait();
            let mut release = Release::of(&park);
            let before = counters.metrics().submissions;
            let later: Vec<CommandBuffer> = (0..IN_FLIGHT + 2)
                .map(|index| {
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    fill(&command_buffer, &buffer, 2 + index as u32);
                    command_buffer.commit().expect("a commit");
                    command_buffer
                })
                .collect();
            let wanted = before + later.len() as u64;
            within("each later command buffer's submission", || {
                counters.metrics().submissions >= wanted
            });
            assert!(
                later
                    .iter()
                    .all(|cb| cb.status() != CommandBufferStatus::Completed),
                "{name}: no later command buffer completes before the parked handler returns"
            );
            release.now();
            for command_buffer in &later {
                finished(name, command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
            }
            finished(name, &first);
            assert_eq!(
                word(&buffer),
                2 + IN_FLIGHT as u32 + 1,
                "{name}: the last fill"
            );
            drop((later, first, buffer, arena, queue, device));
        });
    }

    /// A group held behind its queue's [`IN_FLIGHT`] submissions that goes once a completion
    /// leaves fewer running counts one in `Metrics::held_starts`, and the span from the end of the
    /// busy work before it to its command buffer's start in `held_start_gap_nanos`: it is
    /// submitted only once the host has seen that work complete. A command buffer sent at once,
    /// its queue with nothing in flight, counts in neither.
    #[test]
    fn every_device_counts_a_held_group_s_start_and_not_one_sent_at_once() {
        each_device("command_held_starts", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let counters = device.counters();
            let held = || {
                let m = counters.metrics();
                (m.held_starts, m.held_start_gap_nanos)
            };
            let before = held();
            let at_once = queue.command_buffer().expect("a command buffer");
            fill(&at_once, &buffer, 1);
            at_once.commit().expect("a commit");
            finished(name, &at_once);
            assert_eq!(held(), before, "{name}: a command buffer sent at once");
            behind_busy(name, &device, &queue, |busy, _, _| {
                let group = queue.command_buffer().expect("a command buffer");
                fill(&group, &buffer, 2);
                group.commit().expect("a commit");
                // Staged in the held group, then the pass over: the submit thread holds the work
                // lock from the staging to the pass's end unless it submits.
                within(name, || group.status() == CommandBufferStatus::Scheduled);
                drop(device.core().work.lock());
                assert_eq!(held(), before, "{name}: nothing counted while it is held");
                // Its status is read, not waited for: a host wait would send it.
                busy.open();
                finished(name, &group);
                assert_eq!(group.error(), None, "{name}");
            });
            let (starts, gap) = held();
            assert_eq!(starts - before.0, 1, "{name}: the held group's start");
            assert!(
                gap > before.1,
                "{name}: its command buffer starts after the busy work's end"
            );
            println!(
                "{name}: the held group's command buffer started {} ns after the busy work's end",
                gap - before.1
            );
            assert_eq!(word(&buffer), 2, "{name}: the held fill");
            drop((at_once, buffer, arena, queue, device));
        });
    }

    /// Each command buffer that completes lands one count in each of the four stage histograms,
    /// and its span in each stage's sum: three committed in turn, each finished before the next,
    /// add three to each histogram, and to the sums the spans their own times give (from its
    /// commit, its kernel start, to its handlers' start, which lies before it is seen finished).
    #[test]
    fn every_device_counts_each_command_buffer_once_in_every_stage_histogram() {
        each_device("command_stage_histograms", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let counters = device.counters();
            crate::tests::blit::on_device::settled(&device);
            let sums = |m: crate::Metrics| {
                [
                    m.commit_to_scheduled,
                    m.scheduled_to_gpu_start,
                    m.gpu_run,
                    m.gpu_end_to_handler,
                ]
                .map(|histogram| histogram.iter().sum::<u64>())
            };
            let nanos = |m: crate::Metrics| {
                [
                    m.commit_to_scheduled_nanos,
                    m.scheduled_to_gpu_start_nanos,
                    m.gpu_run_nanos,
                    m.gpu_end_to_handler_nanos,
                ]
            };
            let before = counters.metrics();
            let mut spans = [0; 3];
            let mut handled_by = 0;
            for word in 1..=3 {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                fill(&command_buffer, &buffer, word);
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
                let c = &command_buffer;
                let (start, end) = (c.gpu_start_time(), c.gpu_end_time());
                spans[0] += c.kernel_end_time() - c.kernel_start_time();
                spans[1] += start.saturating_sub(c.kernel_end_time());
                spans[2] += end - start;
                handled_by += host_time() - end;
            }
            let after = counters.metrics();
            // The completion thread reads its first calibration as it starts, before it sees any
            // completion, so three completions seen mean it has.
            assert!(
                after.calibrations >= 1,
                "{name}: the completion thread's starting calibration"
            );
            let added: Vec<u64> = (sums(after).iter().zip(sums(before)))
                .map(|(a, b)| a - b)
                .collect();
            assert_eq!(added, [3; 4], "{name}: each histogram's counts added");
            let added: Vec<u64> = (nanos(after).iter().zip(nanos(before)))
                .map(|(a, b)| a - b)
                .collect();
            assert_eq!(added[..3], spans, "{name}: the stages' sums added");
            assert!(
                added[3] <= handled_by,
                "{name}: GPU end to handlers {} past its bound {handled_by}",
                added[3]
            );
            // Each end is placed no later than the round saw it, so the steps add up to the span.
            let steps = |m: &crate::Metrics| m.gpu_end_to_handler_steps_nanos.iter().sum::<u64>();
            assert_eq!(
                steps(&after) - steps(&before),
                added[3],
                "{name}: the completion thread's steps add up to GPU end to handlers"
            );
            let behind = after.scheduled_to_gpu_start_behind_nanos
                - before.scheduled_to_gpu_start_behind_nanos;
            assert!(
                behind <= added[1],
                "{name}: {behind} under earlier spans of {} from scheduling to GPU start",
                added[1]
            );
            drop((buffer, arena, queue, device));
        });
    }

    /// Commits a command buffer of one fill of `buffer` with `word` on `queue`.
    fn queue_fill(queue: &CommandQueue, buffer: &Buffer, word: u32) -> CommandBuffer {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        fill(&command_buffer, buffer, word);
        command_buffer.commit().expect("a commit");
        command_buffer
    }

    /// A group is held only while three of its queue's submissions are not seen complete, and the
    /// completion that leaves fewer sends it: a command buffer committed behind three running ones,
    /// with no later commit and no host wait, runs once the first of them completes.
    #[test]
    fn every_device_sends_a_held_group_when_a_submission_completes() {
        each_device("command_release", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            behind_busy(name, &device, &queue, |busy, before, _| {
                let held = queue.command_buffer().expect("a command buffer");
                fill(&held, &buffer, 7);
                held.commit().expect("a commit");
                within(name, || held.status() != CommandBufferStatus::Committed);
                // Scheduled while the busy ones ran: it was held, then sent by a completion.
                let submissions = device.counters().metrics().submissions - before;
                assert_eq!(submissions, AHEAD, "{name}: held behind the busy work");
                busy.open();
                within(name, || held.status() == CommandBufferStatus::Completed);
                assert_eq!(word(&buffer), 7, "{name}");
                let after = device.counters().metrics().submissions - before;
                assert_eq!(after, AHEAD + 1, "{name}: one submission each");
            });
            drop((buffer, arena, queue, device));
        });
    }

    /// A command buffer joins a held group only while the group's expected GPU runs, summed, stay
    /// within `GROUP_RUN`, each its queue's expected run: behind three running submissions, three
    /// fills on a queue whose command buffers ran 1 ms go as one group, and on a queue whose
    /// command buffers ran past the bound as three submissions, each held, scheduled, until the
    /// queue sends them (`send_now`), all before the busy work ends: the queue has no room for
    /// the two the group does not take (`submit::room`). The expected run is set behind the busy work as a retiring command buffer's GPU
    /// times set it (`QueueInner::ran`); the busy work retires only after the fills were sent,
    /// and the retirements move it.
    #[test]
    fn every_device_joins_a_held_group_only_within_its_expected_run() {
        each_device("command_group_run", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            for (run, alone) in [(1_000_000, false), (GROUP_RUN + 1, true)] {
                let queue = queue(&device);
                behind_busy(name, &device, &queue, |busy, before, _| {
                    let probe = queue.command_buffer().expect("a command buffer");
                    probe.queue().ran(run);
                    drop(probe);
                    let held: Vec<CommandBuffer> = (1..=3)
                        .map(|word| queue_fill(&queue, &buffer, word))
                        .collect();
                    for command_buffer in &held {
                        within(name, || {
                            command_buffer.status() != CommandBufferStatus::Committed
                        });
                    }
                    let sent = || device.counters().metrics().submissions - before;
                    assert_eq!(sent(), AHEAD, "{name}: {run} ns, held behind the busy work");
                    // The held group goes before the busy work ends, at the run set here: sent
                    // once a busy one retires, it would go at the run their own runs leave.
                    queue.send_now(held.last().expect("a fill"));
                    let groups = if alone { 3 } else { 1 };
                    within(name, || sent() >= AHEAD + groups);
                    busy.open();
                    for command_buffer in &held {
                        finished(name, command_buffer);
                    }
                    assert_eq!(word(&buffer), 3, "{name}");
                    let all = device.counters().metrics().submissions - before;
                    assert_eq!(
                        all,
                        AHEAD + groups,
                        "{name}: {run} ns, submissions, the queue's expected run {:?} ns",
                        held[0].queue().expected_run()
                    );
                });
                // The busy work and the fills retired since, each moving the expected run toward
                // its own GPU run.
                let next = queue.command_buffer().expect("a command buffer");
                let expected = next.queue().expected_run().expect("an expected run");
                assert_ne!(
                    expected, run,
                    "{name}: the retirements moved the expected run"
                );
            }
            drop((buffer, arena, device));
        });
    }

    /// The bound is tested again as a held group goes, with the expected run as it then stands:
    /// three fills that joined one group behind three running submissions on a queue whose command
    /// buffers ran 1 ms go member by member, three submissions, once a run of 1 s raised the
    /// expected run before the busy work ended.
    #[test]
    fn every_device_sends_a_group_past_its_expected_run_member_by_member() {
        each_device("command_group_apart", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let probe = queue.command_buffer().expect("a command buffer");
            behind_busy(name, &device, &queue, |busy, before, _| {
                probe.queue().ran(1_000_000);
                let held: Vec<CommandBuffer> = (1..=3)
                    .map(|word| queue_fill(&queue, &buffer, word))
                    .collect();
                for command_buffer in &held {
                    within(name, || {
                        command_buffer.status() != CommandBufferStatus::Committed
                    });
                }
                let sent = device.counters().metrics().submissions - before;
                assert_eq!(sent, AHEAD, "{name}: the three joined one group");
                probe.queue().ran(1_000_000_000);
                busy.open();
                for command_buffer in &held {
                    finished(name, command_buffer);
                }
                assert_eq!(word(&buffer), 3, "{name}");
                let all = device.counters().metrics().submissions - before;
                assert_eq!(all, AHEAD + 3, "{name}: member by member");
            });
            drop((probe, buffer, arena, queue, device));
        });
    }

    /// At most `UNSEEN_JOINS` command buffers join a group while none of their queue's retires:
    /// behind three running submissions, on a queue whose command buffers ran 1 ms, that many
    /// fills and two more go as two groups before the busy work ends, each held until the queue
    /// sends them (`send_now`), the two the full group does not take scheduled while the queue
    /// has no room for them (`submit::room`). Sent only once the busy work
    /// ends, the second could go member by member: the busy work's long runs, retiring, raise the
    /// expected run before or after it goes.
    #[test]
    fn every_device_joins_at_most_unseen_joins_to_a_group_while_none_retires() {
        each_device("command_group_unseen", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let fills = UNSEEN_JOINS as u32 + 2;
            behind_busy(name, &device, &queue, |busy, before, _| {
                let probe = queue.command_buffer().expect("a command buffer");
                probe.queue().ran(1_000_000);
                drop(probe);
                let held: Vec<CommandBuffer> = (1..=fills)
                    .map(|word| queue_fill(&queue, &buffer, word))
                    .collect();
                for command_buffer in &held {
                    within(name, || {
                        command_buffer.status() != CommandBufferStatus::Committed
                    });
                }
                let sent = || device.counters().metrics().submissions - before;
                assert_eq!(sent(), AHEAD, "{name}: held behind the busy work");
                let last = held.last().expect("a fill");
                queue.send_now(last);
                within(name, || sent() > AHEAD + 1);
                assert_eq!(sent(), AHEAD + 2, "{name}: two groups");
                busy.open();
                for command_buffer in &held {
                    finished(name, command_buffer);
                }
                assert_eq!(word(&buffer), fills, "{name}");
                assert_eq!(sent(), AHEAD + 2, "{name}: nothing more");
            });
            drop((buffer, arena, queue, device));
        });
    }

    /// A queue keeps three submissions in flight: behind two running ones a commit goes at once,
    /// alone, and the commit after it is held, both seen before the first busy one ends on the
    /// GPU. The queue's expected run is 1 ms, known: one not yet known leaves no room for a
    /// third (`submit::room`).
    #[test]
    fn every_device_keeps_three_submissions_in_flight_and_holds_the_fourth() {
        each_device("command_in_flight", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let probe = queue.command_buffer().expect("a command buffer");
            probe.queue().set_run_for_test(BODY_RUN);
            drop(probe);
            let submissions = || device.counters().metrics().submissions;
            let before = submissions();
            let mut busy = Busy::new(&device);
            busy.commit(name, &queue);
            busy.commit(name, &queue);
            let alone = queue_fill(&queue, &buffer, 1);
            within(name, || submissions() == before + 3);
            let held = queue_fill(&queue, &buffer, 2);
            within(name, || held.status() != CommandBufferStatus::Committed);
            let sent = submissions() - before;
            let seen = host_time();
            assert_eq!(sent, 3, "{name}: three went and the fourth was held");
            busy.open();
            for command_buffer in busy.running().iter().chain([&alone, &held]) {
                finished(name, command_buffer);
            }
            assert_eq!(word(&buffer), 2, "{name}");
            assert_eq!(submissions() - before, 4, "{name}: one submission each");
            assert!(
                busy.running()[0].gpu_end_time() > seen,
                "{name}: the fourth was held before the first busy one ended"
            );
            drop((busy, buffer, arena, queue, device));
        });
    }

    /// The command buffers of one submission retire in commit order, each with its own status
    /// and GPU times: a group held behind three running submissions takes three command buffers,
    /// then one ended by an abandon while enqueued and committed after them, which sends it; they
    /// complete in that order, the ended one `NotRun` with no times. The other three, each the
    /// same work, run back to back: each starts no earlier than the one before ends, where one
    /// shared end less each one's duration would overlap them.
    #[test]
    fn every_device_retires_a_group_in_commit_order_each_with_its_own_times() {
        each_device("command_group_order", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffers = [arena.buffer(), arena.buffer(), arena.buffer()];
            let work = private(&device);
            let queue = queue(&device);
            // A word written, then the whole block filled: work long enough to time.
            let member = |label: &str, log: &Log, buffer: &Buffer, word: u32| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                fill(&command_buffer, buffer, word);
                let (target, _) = work.device_range();
                command_buffer
                    .record_for_test(&[&work], |device, command| {
                        // SAFETY: as `busy`'s.
                        unsafe {
                            device.cmd_fill_buffer(
                                command,
                                target,
                                0,
                                MemoryClass::Private.block_len(),
                                word,
                            )
                        };
                    })
                    .expect("a fill");
                log.handlers(label, &command_buffer);
                command_buffer.commit().expect("a commit");
                command_buffer
            };
            behind_busy(name, &device, &queue, |busy, before, _| {
                let log = Log::default();
                let first = member("1", &log, &buffers[0], 1);
                let rest = [
                    member("2", &log, &buffers[1], 3),
                    member("3", &log, &buffers[2], 4),
                ];
                // Joined to the group before the abandon, which ends only what is not.
                within(name, || rest[1].status() != CommandBufferStatus::Committed);
                // All three were held while the busy ones ran, so they go as one submission.
                let submissions = device.counters().metrics().submissions - before;
                assert_eq!(submissions, AHEAD, "{name}: held behind the busy work");
                let ended = queue.command_buffer().expect("a command buffer");
                log.handlers("4", &ended);
                ended.enqueue().expect("an enqueue");
                queue.abandon();
                ended.commit().expect("a commit");
                // Sent before the busy work ends, at the run `behind_busy` set: sent once a busy
                // one retires, it would go at the run their own runs leave.
                within(name, || {
                    device.counters().metrics().submissions - before > AHEAD
                });
                busy.open();
                for command_buffer in [&first].into_iter().chain(&rest).chain([&ended]) {
                    finished(name, command_buffer);
                }
                let completed: Vec<String> = (log.lines().into_iter())
                    .filter(|line| line.contains("completed"))
                    .collect();
                assert_eq!(
                    completed,
                    [
                        "1 completed Completed None",
                        "2 completed Completed None",
                        "3 completed Completed None",
                        "4 completed Error Some(NotRun)",
                    ],
                    "{name}"
                );
                assert_eq!(
                    (word(&buffers[0]), word(&buffers[1]), word(&buffers[2])),
                    (1, 3, 4),
                    "{name}"
                );
                assert_eq!(
                    (ended.gpu_start_time(), ended.gpu_end_time()),
                    (0, 0),
                    "{name}"
                );
                let times: Vec<(u64, u64)> = [&first, &rest[0], &rest[1]]
                    .iter()
                    .map(|cb| (cb.gpu_start_time(), cb.gpu_end_time()))
                    .collect();
                for &(start, end) in &times {
                    // A later timestamp is never lower; a device may give two one value (Mesa's
                    // Apple-GPU driver does, outside a render pass).
                    assert!(0 < start && start <= end, "{name}: {times:?}");
                }
                for pair in times.windows(2) {
                    assert!(
                        pair[0].1 <= pair[1].0,
                        "{name}: each starts no earlier than the one before ends: {times:?}"
                    );
                }
                let after = device.counters().metrics().submissions - before;
                assert_eq!(after, AHEAD + 1, "{name}: one group");
            });
            drop((buffers, work, arena, queue, device));
        });
    }

    /// A command buffer recorded and dropped before its commit leaves its pair's queries as it
    /// found them, reset, and the next command buffer takes the pair again while three others of
    /// the queue run, with no second reset (the validation layer reports a reset of a query not
    /// available while work naming its pool runs), and has its own GPU times.
    #[test]
    fn every_device_takes_again_the_pair_of_a_command_buffer_that_never_ran() {
        each_device("command_pair_again", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            behind_busy(name, &device, &queue, |busy, _, _| {
                let running = busy.running();
                within(name, || {
                    (running.iter()).all(|command_buffer| {
                        command_buffer.status() != CommandBufferStatus::Committed
                    })
                });
                let dropped = queue.command_buffer().expect("a command buffer");
                fill(&dropped, &buffer, 4);
                drop(dropped);
                let again = queue.command_buffer().expect("a command buffer");
                // Each was submitted and none had completed when the pair was taken again.
                let statuses: Vec<_> = running.iter().map(CommandBuffer::status).collect();
                assert!(
                    statuses
                        .iter()
                        .all(|s| *s == CommandBufferStatus::Scheduled),
                    "{name}: {statuses:?}"
                );
                fill(&again, &buffer, 5);
                again.commit().expect("a commit");
                busy.open();
                finished(name, &again);
                assert_eq!(again.status(), CommandBufferStatus::Completed, "{name}");
                assert_eq!(word(&buffer), 5, "{name}");
                let (start, end) = (again.gpu_start_time(), again.gpu_end_time());
                assert!(0 < start && start <= end, "{name}: {start} {end}");
            });
            drop((buffer, arena, queue, device));
        });
    }

    /// The device's timestamp pools, made as command buffers need them: a command buffer takes
    /// none before its first record; 2,049 command buffers in flight at once, held behind the
    /// first's event wait, take the 2,048 pairs of a first pool and one of a second; once they
    /// retire, each timed, 2,049 more take the pairs given back: the second pool's 2,047 left
    /// would not hold them, and no third pool is made. The pools live until the device drops,
    /// when the gauge falls to zero.
    #[test]
    fn every_device_takes_pairs_from_timestamp_pools_it_makes_on_demand() {
        each_device("command_pools", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let pools = || device.counters().gauges().timestamp_pools;
            let queue = device.new_command_queue(4096).expect("a queue");
            let empty = queue.command_buffer().expect("a command buffer");
            assert_eq!(pools(), 0, "{name}: no pool before a record");
            for round in 0..2 {
                let gate = shared_event(&device);
                let burst: Vec<CommandBuffer> = (0..2049)
                    .map(|index| {
                        let command_buffer = queue.command_buffer().expect("a command buffer");
                        if index == 0 {
                            command_buffer
                                .encode_wait_for_event(gate.as_event(), 1)
                                .expect("a wait");
                        }
                        fill(&command_buffer, &buffer, index);
                        command_buffer.commit().expect("a commit");
                        command_buffer
                    })
                    .collect();
                println!("{name}: round {round}: 2049 in flight in {} pools", pools());
                assert_eq!(pools(), 2, "{name}: round {round}");
                gate.set_signaled_value(1);
                for command_buffer in &burst {
                    finished(name, command_buffer);
                    assert_eq!(
                        command_buffer.status(),
                        CommandBufferStatus::Completed,
                        "{name}"
                    );
                    let (start, end) = (
                        command_buffer.gpu_start_time(),
                        command_buffer.gpu_end_time(),
                    );
                    assert!(0 < start && start <= end, "{name}: {start} {end}");
                }
                assert_eq!(word(&buffer), 2048, "{name}");
            }
            assert_eq!(
                device.counters().metrics().untimed_command_buffers,
                0,
                "{name}"
            );
            empty.commit().expect("a commit");
            finished(name, &empty);
            assert_eq!(pools(), 2, "{name}");
            drop((empty, queue, buffer, arena, device));
        });
    }

    /// 300 queues live at once make no Vulkan object for their command buffers' times: each
    /// queue's command buffer runs and is timed, and their pairs come from one pool.
    #[test]
    fn every_device_times_command_buffers_of_300_live_queues_from_one_pool() {
        each_device("command_queues_300", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queues: Vec<CommandQueue> = (0..300).map(|_| queue(&device)).collect();
            let committed: Vec<CommandBuffer> = (queues.iter().enumerate())
                .map(|(index, queue)| {
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    fill(&command_buffer, &buffer, index as u32);
                    command_buffer.commit().expect("a commit");
                    command_buffer
                })
                .collect();
            for command_buffer in &committed {
                finished(name, command_buffer);
                assert_eq!(
                    command_buffer.status(),
                    CommandBufferStatus::Completed,
                    "{name}"
                );
                let (start, end) = (
                    command_buffer.gpu_start_time(),
                    command_buffer.gpu_end_time(),
                );
                assert!(0 < start && start <= end, "{name}: {start} {end}");
            }
            assert_eq!(device.counters().gauges().timestamp_pools, 1, "{name}");
            drop((committed, queues, buffer, arena, device));
        });
    }

    /// A device that refuses a timestamp pool: with raw pools of 4,096 queries made until the
    /// device refuses one (at most 64), a command buffer that finds no pair runs untimed, its GPU
    /// times 0, and so does one whose last segment has no commands; each is counted. Where the
    /// device makes all 64 raw pools, the untimed path does not run: the command buffers are
    /// timed, and the test says which path ran.
    #[test]
    fn every_device_runs_a_command_buffer_untimed_where_no_pool_is_made() {
        each_device("command_untimed", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, b) = (arena.buffer(), arena.buffer());
            let raw = device.raw();
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(crate::times::POOL_QUERIES);
            let mut held = Vec::new();
            let mut refusal = None;
            while held.len() < 64 {
                // SAFETY: the info is complete; each pool is destroyed below.
                match unsafe { raw.device.create_query_pool(&info, None) } {
                    Ok(pool) => held.push(pool),
                    Err(error) => {
                        refusal = Some(error);
                        break;
                    }
                }
            }
            let queue = queue(&device);
            let before = device.counters().metrics().untimed_command_buffers;
            let event = device.new_event().expect("an event");
            let first = queue.command_buffer().expect("a command buffer");
            fill(&first, &a, 7);
            first.commit().expect("a commit");
            let second = queue.command_buffer().expect("a command buffer");
            fill(&second, &b, 8);
            second.encode_signal_event(&event, 1).expect("a signal");
            second.commit().expect("a commit");
            for command_buffer in [&first, &second] {
                finished(name, command_buffer);
                assert_eq!(
                    command_buffer.status(),
                    CommandBufferStatus::Completed,
                    "{name}"
                );
            }
            assert_eq!((word(&a), word(&b)), (7, 8), "{name}");
            let untimed = device.counters().metrics().untimed_command_buffers - before;
            let times = [&first, &second].map(|cb| (cb.gpu_start_time(), cb.gpu_end_time()));
            match refusal {
                Some(error) => {
                    println!(
                        "{name}: the untimed path ran: the device refused raw pool {} ({error}); \
                         {untimed} command buffers untimed, times {times:?}",
                        held.len() + 1
                    );
                    assert_eq!(untimed, 2, "{name}");
                    assert_eq!(times, [(0, 0), (0, 0)], "{name}");
                    assert_eq!(
                        device.counters().gauges().timestamp_pools,
                        0,
                        "{name}: no pool made"
                    );
                }
                None => {
                    println!(
                        "{name}: the untimed path did not run: the device made {} raw pools; \
                         times {times:?}",
                        held.len()
                    );
                    assert_eq!(untimed, 0, "{name}");
                    for (start, end) in times {
                        assert!(0 < start && start <= end, "{name}: {start} {end}");
                    }
                }
            }
            for pool in held {
                // SAFETY: no submission names a raw pool.
                unsafe { raw.device.destroy_query_pool(pool, None) };
            }
            drop((first, second, event, queue, a, b, arena, device));
        });
    }

    /// `send_now`, the wait hint, from another thread: a command buffer held in a group behind three
    /// running submissions goes at once, and the call returns while they still run. A call for a
    /// command buffer already submitted or completed, one not committed, or another queue's makes
    /// no submission.
    #[test]
    fn every_device_sends_a_held_group_at_the_wait_hint() {
        each_device("command_send_now", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            let other = device.new_command_queue(4).expect("a queue");
            behind_busy(name, &device, &queue, |busy, before, _| {
                let running = busy.running();
                let held = queue.command_buffer().expect("a command buffer");
                fill(&held, &buffer, 9);
                held.commit().expect("a commit");
                within(name, || held.status() != CommandBufferStatus::Committed);
                let submissions = device.counters().metrics().submissions - before;
                thread::scope(|scope| {
                    scope
                        .spawn(|| queue.send_now(&held))
                        .join()
                        .expect("the hint");
                });
                // Returned with the queue's work still running: it waited for none of it.
                let last = &running[running.len() - 1];
                assert_eq!(submissions, AHEAD, "{name}: held behind the busy work");
                assert_ne!(last.status(), CommandBufferStatus::Completed, "{name}");
                within(name, || {
                    device.counters().metrics().submissions - before == AHEAD + 1
                });
                // The hint for one submitted makes no submission.
                queue.send_now(last);
                busy.open();
                finished(name, &held);
                assert_eq!(word(&buffer), 9, "{name}");
                let uncommitted = queue.command_buffer().expect("a command buffer");
                for stale in [&held, &running[0], &uncommitted] {
                    queue.send_now(stale);
                }
                other.send_now(&held);
                thread::sleep(Duration::from_millis(20));
                assert_eq!(
                    device.counters().metrics().submissions - before,
                    AHEAD + 1,
                    "{name}: a stale hint makes no submission"
                );
                drop(uncommitted);
            });
            drop((buffer, arena, queue, other, device));
        });
    }

    /// Abandon ends the queue's held and unsubmitted command buffers with NotRun and runs their
    /// handlers; the part already submitted ran, the rest never does, and the queue goes on.
    #[test]
    fn every_device_abandons_a_queues_held_and_unsubmitted_command_buffers() {
        each_device("command_abandon", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, b, c, d) = (
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
                arena.buffer(),
            );
            let f = shared_event(&device);
            let queue = queue(&device);
            let log = Log::default();

            let held = queue.command_buffer().expect("a command buffer");
            fill(&held, &a, 1);
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            fill(&held, &b, 2);
            log.handlers("held", &held);
            held.commit().expect("a commit");
            let behind = queue.command_buffer().expect("a command buffer");
            fill(&behind, &c, 3);
            log.handlers("behind", &behind);
            behind.commit().expect("a commit");
            let enqueued = queue.command_buffer().expect("a command buffer");
            log.handlers("enqueued", &enqueued);
            enqueued.enqueue().expect("an enqueue");
            within("the part before the wait", || word(&a) == 1);

            queue.abandon();
            finished(name, &held);
            finished(name, &behind);
            for command_buffer in [&held, &behind] {
                assert_eq!(
                    (command_buffer.status(), command_buffer.error()),
                    (CommandBufferStatus::Error, Some(CommandBufferError::NotRun)),
                    "{name}"
                );
                // An ended command buffer has no GPU times.
                assert_eq!(
                    (
                        command_buffer.gpu_start_time(),
                        command_buffer.gpu_end_time()
                    ),
                    (0, 0),
                    "{name}"
                );
            }
            enqueued.commit().expect("a commit");
            finished(name, &enqueued);
            assert_eq!(enqueued.error(), Some(CommandBufferError::NotRun), "{name}");
            let mut lines = log.lines();
            lines.sort();
            assert_eq!(
                lines,
                [
                    "behind completed Error Some(NotRun)",
                    "behind scheduled Error",
                    "enqueued completed Error Some(NotRun)",
                    "enqueued scheduled Error",
                    "held completed Error Some(NotRun)",
                    "held scheduled Error",
                ],
                "{name}"
            );

            f.set_signaled_value(1);
            let after = queue.command_buffer().expect("a command buffer");
            fill(&after, &d, 4);
            after.commit().expect("a commit");
            finished(name, &after);
            assert_eq!(after.status(), CommandBufferStatus::Completed, "{name}");
            assert_eq!(
                (word(&b), word(&c), word(&d)),
                (0, 0, 4),
                "{name}: never ran"
            );
            drop((a, b, c, d, arena, f, queue, device));
        });
    }

    /// The consumer's hooks as these tests see them: each allocation mapped, and each unmapped
    /// with the thread its unmap ran on.
    #[derive(Default)]
    struct Hook {
        mapped: Mutex<Vec<AllocationId>>,
        unmapped: Mutex<Vec<(AllocationId, Option<String>)>>,
    }

    impl BlockMappings for Hook {
        fn map(&self, allocation: &AllocationInfo) -> bool {
            lock(&self.mapped).push(allocation.id);
            true
        }

        fn unmap(&self, allocation: AllocationId) {
            let thread = thread::current().name().map(str::to_string);
            lock(&self.unmapped).push((allocation, thread));
        }
    }

    /// Unmaps: each allocation and the name of the thread its unmap ran on.
    type Unmaps = Vec<(AllocationId, Option<String>)>;

    impl Hook {
        /// The unmaps, sorted, and what they are where each allocation mapped was unmapped once,
        /// on `thread`.
        fn unmaps(&self, thread: Option<&str>) -> (Unmaps, Unmaps) {
            let mut unmapped = lock(&self.unmapped).clone();
            unmapped.sort();
            let mut mapped: Vec<_> = (lock(&self.mapped).iter())
                .map(|&allocation| (allocation, thread.map(str::to_string)))
                .collect();
            mapped.sort();
            (unmapped, mapped)
        }
    }

    /// Quiesce, from a thread other than the one that made the device, with handles alive: held
    /// work ends NotRun with its handlers run, every allocation mapped is unmapped once on that
    /// thread, calls that need a thread are refused, memory stays mapped, and a second call
    /// answers at once with the first call's answer. The last drop, on the test's thread, then
    /// frees everything without the hook: every gauge reads zero as it returns.
    #[test]
    fn every_device_quiesces_with_handles_alive() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let options = DeviceOptions {
                block_mappings: Some(hook.clone() as Arc<dyn BlockMappings>),
                ..DeviceOptions::default()
            };
            let device = Device::new(physical, options).expect("a device");
            let counters = device.counters();
            let mut arena = Arena::new(&device);
            let a = arena.buffer();
            let private = device
                .take_block(MemoryClass::Private, 1)
                .expect("a Private block");
            let (f, e) = (shared_event(&device), shared_event(&device));
            let queue = queue(&device);
            let held = queue.command_buffer().expect("a command buffer");
            fill(&held, &a, 7);
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            held.commit().expect("a commit");
            let open = queue.command_buffer().expect("a command buffer");
            within("the part before the wait", || word(&a) == 7);

            let quiesced = thread::scope(|scope| {
                scope
                    .spawn(|| device.quiesce(PATIENCE))
                    .join()
                    .expect("the quiescing thread")
            });
            assert!(quiesced, "{name}: idle within the timeout");
            let began = Instant::now();
            assert!(device.quiesce(PATIENCE), "{name}: the first call's answer");
            assert!(
                began.elapsed() < AT_ONCE,
                "{name}: a second call answers at once"
            );
            assert_eq!(held.error(), Some(CommandBufferError::NotRun), "{name}");
            held.wait_until_completed()
                .expect("a wait outside a scheduled handler");

            let (unmapped, mapped) = hook.unmaps(None);
            assert_eq!(
                unmapped, mapped,
                "{name}: each allocation mapped once, on the quiescing thread"
            );
            for block in [arena.block.allocation, private.allocation] {
                assert!(
                    unmapped.iter().any(|(allocation, _)| *allocation == block),
                    "{name}: {block:?}, a block's allocation"
                );
            }
            let count = unmapped.len();
            assert_eq!(word(&a), 7, "{name}: memory stays mapped");
            assert!(matches!(queue.command_buffer(), Err(Error::Quiesced)));
            assert!(matches!(
                device.take_block(MemoryClass::Shared, 1),
                Err(Error::Quiesced)
            ));
            assert!(matches!(
                device.new_buffer(private.id, 0, 64, Some(0)),
                Err(Error::Quiesced)
            ));
            assert!(matches!(open.enqueue(), Err(Error::Quiesced)));
            assert!(matches!(open.commit(), Err(Error::Quiesced)));
            assert!(matches!(e.notify_listener(1, |_| {}), Err(Error::Quiesced)));

            drop((a, arena, f, e, queue, held, open, device));
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: every gauge zero as the drop returns"
            );
            print_gauges_at_drop("quiesce", counters.gauges());
            assert_eq!(
                lock(&hook.unmapped).len(),
                count,
                "{name}: no unmap after quiesce"
            );
        }
    }

    /// The last handle dropped in a completed handler: the reaper runs the teardown once the
    /// handler returns, so the block still held is unmapped there, and the instance's wait for its
    /// devices answers true once the device is destroyed, every gauge reading zero as it returns.
    /// The handler's handle is the last: `commit` gave the command buffer's up before the command
    /// buffer could complete.
    #[test]
    fn every_device_tears_down_from_a_completed_handler() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let options = DeviceOptions {
                block_mappings: Some(hook.clone() as Arc<dyn BlockMappings>),
                ..DeviceOptions::default()
            };
            let device = Device::new(physical, options).expect("a device");
            let counters = device.counters();
            let block = device
                .take_block(MemoryClass::Shared, 1)
                .expect("a Shared block");
            let queue = queue(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            drop(queue);
            let handles = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&handles);
            command_buffer
                .add_completed_handler(move |_| {
                    seen.store(device.handles(), Ordering::SeqCst);
                    drop(device);
                })
                .expect("a handler");
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(
                handles.load(Ordering::SeqCst),
                1,
                "{name}: the handler held the last handle"
            );
            assert!(
                instance.wait_until_devices_torn_down(PATIENCE),
                "{name}: torn down on the reaper"
            );
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: every gauge zero as the wait returns"
            );
            print_gauges_at_drop("handler_teardown", counters.gauges());
            let (unmapped, mapped) = hook.unmaps(Some("m2v-reaper"));
            assert_eq!(
                unmapped, mapped,
                "{name}: each allocation mapped once, by the reaper"
            );
            assert!(
                unmapped
                    .iter()
                    .any(|(allocation, _)| *allocation == block.allocation),
                "{name}: the block's allocation"
            );
        }
    }

    /// The last handle dropped in a scheduled handler, on the submit thread: the reaper runs the
    /// teardown once the handler returns, the command buffer still completes, and the instance's
    /// wait for its devices answers true once the device is destroyed, every gauge reading zero as
    /// it returns.
    #[test]
    fn every_device_tears_down_from_a_scheduled_handler() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let options = DeviceOptions {
                block_mappings: Some(hook.clone() as Arc<dyn BlockMappings>),
                ..DeviceOptions::default()
            };
            let device = Device::new(physical, options).expect("a device");
            let counters = device.counters();
            let block = device
                .take_block(MemoryClass::Shared, 1)
                .expect("a Shared block");
            let queue = queue(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            drop(queue);
            let handles = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&handles);
            command_buffer
                .add_scheduled_handler(move |_| {
                    seen.store(device.handles(), Ordering::SeqCst);
                    drop(device);
                })
                .expect("a handler");
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(
                handles.load(Ordering::SeqCst),
                1,
                "{name}: the handler held the last handle"
            );
            assert_eq!(command_buffer.error(), None, "{name}: it ran");
            assert!(
                instance.wait_until_devices_torn_down(PATIENCE),
                "{name}: torn down on the reaper"
            );
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: every gauge zero as the wait returns"
            );
            print_gauges_at_drop("scheduled_teardown", counters.gauges());
            let (unmapped, mapped) = hook.unmaps(Some("m2v-reaper"));
            assert_eq!(
                unmapped, mapped,
                "{name}: each allocation mapped once, by the reaper"
            );
            assert!(
                unmapped
                    .iter()
                    .any(|(allocation, _)| *allocation == block.allocation),
                "{name}: the block's allocation"
            );
        }
    }

    /// The last handle dropped in a scheduled handler, on the submit thread, while a completed
    /// handler holds the completion thread: the drop returns at once, and the reaper, which tears
    /// down, joins every other thread before anything of the device goes.
    /// While the completion thread is held, at least it and the reaper hold their slots of
    /// `Gauges::threads` (where the OS gives ids), no allocation is unmapped and the blocks live;
    /// once it is let go, the reaper unmaps each allocation and the device is destroyed, and every
    /// gauge reads zero, the reaper's slot among them, which it gives up only once it has dropped
    /// the device: it touches nothing of the device after that. Both command buffers have nothing
    /// for the GPU, so each is scheduled and completes as the submit thread reaches it.
    #[test]
    fn every_device_torn_down_in_a_handler_joins_the_others_before_destroying_what_they_reach() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let hook = Arc::new(Hook::default());
            let options = DeviceOptions {
                block_mappings: Some(hook.clone() as Arc<dyn BlockMappings>),
                ..DeviceOptions::default()
            };
            let device = Device::new(physical, options).expect("a device");
            let counters = device.counters();
            let four: Vec<u64> = (counters.gauges().threads.iter())
                .copied()
                .filter(|&id| id != 0)
                .collect();
            let alive = || {
                let now = counters.gauges().threads;
                four.iter().filter(|id| now.contains(id)).count()
            };
            let block = device
                .take_block(MemoryClass::Shared, 1)
                .expect("a Shared block");
            let queue = queue(&device);
            let held = queue.command_buffer().expect("a command buffer");
            let last = queue.command_buffer().expect("a command buffer");
            drop(queue);
            let (entered, inside) = mpsc::channel::<()>();
            let (release, gate) = mpsc::channel::<()>();
            let (dropped, gone) = mpsc::channel::<usize>();
            held.add_completed_handler(move |_| {
                let _ = entered.send(());
                let _ = gate.recv();
            })
            .expect("a handler");
            last.add_scheduled_handler(move |_| {
                let handles = device.handles();
                drop(device);
                let _ = dropped.send(handles);
            })
            .expect("a handler");
            held.commit().expect("a commit");
            last.commit().expect("a commit");
            let handles = (gone.recv_timeout(PATIENCE)).expect("the handler's drop returns");
            assert_eq!(handles, 1, "{name}: the handler held the last handle");
            inside
                .recv_timeout(PATIENCE)
                .expect("the completed handler holds the completion thread");
            assert!(
                alive() >= four.len().min(2),
                "{name}: the completion thread and the reaper live while the one is held"
            );
            assert!(
                lock(&hook.unmapped).is_empty(),
                "{name}: no allocation unmapped while the completion thread is held"
            );
            assert!(
                counters.gauges().blocks > 0,
                "{name}: the blocks live while the completion thread is held"
            );
            release.send(()).expect("the handler waits");
            finished(name, &held);
            finished(name, &last);
            within("the teardown", || counters.gauges() == Gauges::default());
            print_gauges_at_drop("handler_joins", counters.gauges());
            assert_eq!(alive(), 0, "{name}: every thread ended");
            let (unmapped, mapped) = hook.unmaps(Some("m2v-reaper"));
            assert_eq!(
                unmapped, mapped,
                "{name}: each allocation mapped once, by the reaper"
            );
            assert!(
                unmapped
                    .iter()
                    .any(|(allocation, _)| *allocation == block.allocation),
                "{name}: the block's allocation"
            );
        }
    }

    /// Quiesce in a scheduled handler answers false at once: the submit thread, which runs the
    /// handler, is not waited for, nor is the completion thread, which it stops as it ends. Its
    /// deadline is ten times the test's patience, so an answer within that patience was not the
    /// deadline's. The command buffer still completes, and the device then tears down.
    #[test]
    fn every_device_answers_false_to_a_quiesce_in_a_scheduled_handler() {
        each_device("scheduled_quiesce", |name, _, device| {
            let counters = device.counters();
            let mut arena = Arena::new(&device);
            let a = arena.buffer();
            let queue = queue(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill(&command_buffer, &a, 9);
            let (answered, answer) = mpsc::channel();
            let quiescing = device.clone();
            command_buffer
                .add_scheduled_handler(move |_| {
                    let _ = answered.send(quiescing.quiesce(PATIENCE * 10));
                })
                .expect("a handler");
            command_buffer.commit().expect("a commit");
            let idle =
                (answer.recv_timeout(PATIENCE)).expect("the handler's answer, before its deadline");
            assert!(!idle, "{name}: false in a scheduled handler");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}: it ran");
            assert_eq!(word(&a), 9, "{name}");
            assert!(!device.quiesce(PATIENCE), "{name}: a later call's answer");
            drop((a, arena, queue, command_buffer, device));
            within("the teardown", || counters.gauges() == Gauges::default());
        });
    }

    /// A drop after a late quiesce joins the thread the quiesce left running before anything that
    /// thread can reach goes (the device, the instance, a queue): the shape of a process that ends
    /// with a runtime thread still running. With the completion thread held in a completed
    /// handler (the command buffer retired, so its work holds no device handle and the test's are
    /// the last), a quiesce with no time to wait answers false at once and leaves that thread
    /// running, its slot of `Gauges::threads` held (its thread id, where the OS gives one), and a
    /// second call answers false at once. The device's last handle, dropped on another thread, has not returned half a second later, the
    /// blocks live, and the thread holds its slot; once the handler lets it go, the drop returns,
    /// every gauge reads zero and no thread of the device's holds a slot. Under the validation
    /// layer, as every device test.
    #[test]
    fn every_device_dropped_after_a_late_quiesce_joins_the_late_thread_first() {
        each_device("late_quiesce_drop", |name, _, device| {
            let counters = device.counters();
            let four: Vec<u64> = (counters.gauges().threads.iter())
                .copied()
                .filter(|&id| id != 0)
                .collect();
            let alive = || {
                let now = counters.gauges().threads;
                four.iter().filter(|id| now.contains(id)).count()
            };
            let mut arena = Arena::new(&device);
            let a = arena.buffer();
            let queue = queue(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill(&command_buffer, &a, 5);
            let (release, gate) = mpsc::channel::<()>();
            let (entered, inside) = mpsc::channel::<()>();
            command_buffer
                .add_completed_handler(move |_| {
                    let _ = entered.send(());
                    let _ = gate.recv();
                })
                .expect("a handler");
            command_buffer.commit().expect("a commit");
            inside
                .recv_timeout(PATIENCE)
                .expect("the handler holds the completion thread");
            assert!(
                !device.quiesce(Duration::ZERO),
                "{name}: no time to wait, an answer at once"
            );
            let began = Instant::now();
            assert!(!device.quiesce(PATIENCE), "{name}: the first call's answer");
            assert!(
                began.elapsed() < AT_ONCE,
                "{name}: a second call answers at once"
            );
            assert!(
                alive() >= four.len().min(1),
                "{name}: the held thread is left running"
            );
            let (dropped, gone) = mpsc::channel::<()>();
            let dropping = thread::spawn(move || {
                drop((a, arena, queue, device));
                let _ = dropped.send(());
            });
            assert!(
                gone.recv_timeout(Duration::from_millis(500)).is_err(),
                "{name}: the drop waits for the held thread"
            );
            assert!(
                counters.gauges().blocks > 0,
                "{name}: the blocks live while the held thread runs"
            );
            assert!(alive() >= four.len().min(1), "{name}: the held thread runs");
            release.send(()).expect("the handler waits");
            (gone.recv_timeout(PATIENCE)).expect("the drop returns once the held thread ended");
            dropping.join().expect("the dropping thread");
            finished(name, &command_buffer);
            drop(command_buffer);
            within("the teardown", || counters.gauges() == Gauges::default());
            assert_eq!(alive(), 0, "{name}: every thread ended");
        });
    }

    /// A second quiesce while the first still runs answers false at once; once the first has
    /// answered true, a later call answers true at once; and the last handle, dropped on the test's
    /// thread after that answer, tears the device down in that drop: every gauge reads zero as it
    /// returns. The first waits for the completion thread, which a completed handler holds.
    #[test]
    fn every_device_answers_a_second_quiesce_at_once() {
        each_device("second_quiesce", |name, _, device| {
            let counters = device.counters();
            let mut arena = Arena::new(&device);
            let a = arena.buffer();
            let queue = queue(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill(&command_buffer, &a, 6);
            let (release, gate) = mpsc::channel::<()>();
            let (entered, inside) = mpsc::channel::<()>();
            command_buffer
                .add_completed_handler(move |_| {
                    let _ = entered.send(());
                    let _ = gate.recv();
                })
                .expect("a handler");
            command_buffer.commit().expect("a commit");
            inside
                .recv_timeout(PATIENCE)
                .expect("the handler holds the completion thread");
            let first = thread::scope(|scope| {
                let first = scope.spawn(|| device.quiesce(PATIENCE));
                within("the first quiesce to start", || device.core().closed());
                let began = Instant::now();
                assert!(
                    !device.quiesce(PATIENCE),
                    "{name}: false while the first runs"
                );
                assert!(began.elapsed() < AT_ONCE, "{name}: at once");
                release.send(()).expect("the handler waits");
                first.join().expect("the quiescing thread")
            });
            assert!(first, "{name}: idle once the handler returned");
            let began = Instant::now();
            assert!(device.quiesce(PATIENCE), "{name}: the first call's answer");
            assert!(began.elapsed() < AT_ONCE, "{name}: at once");
            finished(name, &command_buffer);
            assert_eq!(word(&a), 6, "{name}");
            drop((a, arena, queue, command_buffer, device));
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: every gauge zero as the drop returns"
            );
        });
    }

    /// Nothing the runtime keeps for itself counts as a handle of the device: a command buffer
    /// held by an event wait keeps a buffer, a texture and a sampler state, and once the test
    /// drops its handles of them, of the queue and of the command buffer, the device's handles are
    /// the test's two (the device and the event). The test's last drop, the command buffer then
    /// free to run, tears the device down in that drop, with no wait: every gauge reads zero as it
    /// returns, and the instance's wait for its devices, with nothing pending, answers true at
    /// once.
    #[test]
    fn every_device_counts_nothing_a_command_buffer_keeps_as_a_handle() {
        use crate::{
            IndexClass, PixelFormat, SamplerDescriptor, TextureDescriptor, TextureType,
            TextureUsage,
        };
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default()).expect("a device");
            let counters = device.counters();
            let event = shared_event(&device);
            let mut arena = Arena::new(&device);
            let a = arena.buffer();
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
            d.usage = TextureUsage::RENDER_TARGET;
            let texture = device.new_texture_for_views(&d).expect("a texture");
            let _grant = (device.grant_indices(IndexClass::Sampler, 1, 1)).expect("an index");
            let state =
                (device.new_sampler_state(&SamplerDescriptor::new(), 1)).expect("a sampler state");
            let queue = queue(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            (command_buffer.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
            fill(&command_buffer, &a, 1);
            (command_buffer.keep_for_test(&[&texture], &[&state])).expect("kept");
            command_buffer.commit().expect("a commit");
            drop((a, arena, texture, state, queue, command_buffer));
            assert_eq!(
                device.handles(),
                2,
                "{name}: the test's device and event, and nothing the command buffer keeps"
            );
            event.set_signaled_value(1);
            drop((event, device));
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: every gauge zero as the drop returns"
            );
            let began = Instant::now();
            assert!(
                instance.wait_until_devices_torn_down(PATIENCE),
                "{name}: nothing pending"
            );
            assert!(began.elapsed() < AT_ONCE, "{name}: at once");
        }
    }

    /// The last handle dropped in a listener: in the listener, the instance's wait for its
    /// devices answers false at once; while the listener holds the handle, it answers false once
    /// its timeout passes, and `Gauges::handles_out` reads 1; once the listener drops the handle,
    /// the reaper tears the device down, the wait answers true, and every gauge reads zero as it
    /// returns.
    #[test]
    fn every_device_torn_down_from_a_listener_is_waited_for() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default()).expect("a device");
            let counters = device.counters();
            let event = shared_event(&device);
            let (release, gate) = mpsc::channel::<()>();
            let (answered, answer) = mpsc::channel::<(bool, Duration)>();
            let waiting = Instance::clone(&instance);
            let held = device.clone();
            event
                .notify_listener(1, move |_| {
                    let began = Instant::now();
                    let own = waiting.wait_until_devices_torn_down(PATIENCE);
                    let _ = answered.send((own, began.elapsed()));
                    let _ = gate.recv();
                    drop(held);
                })
                .expect("a listener");
            event.set_signaled_value(1);
            let (own, took) = (answer.recv_timeout(PATIENCE)).expect("the listener's answer");
            assert!(
                !own && took < AT_ONCE,
                "{name}: false at once in the listener"
            );
            drop((event, device));
            assert!(
                !instance.wait_until_devices_torn_down(Duration::from_millis(100)),
                "{name}: false while the listener holds a handle"
            );
            assert_eq!(counters.gauges().handles_out, 1, "{name}: a handle is out");
            release.send(()).expect("the listener waits");
            assert!(
                instance.wait_until_devices_torn_down(PATIENCE),
                "{name}: torn down on the reaper"
            );
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: every gauge zero as the wait returns"
            );
        }
    }

    /// In a scheduled handler, on the submit thread: the command buffer reads Scheduled with its
    /// kernel times set; each call that may wait for that thread is refused, on every device and
    /// whatever the runtime holds, and answers as before once the handler has returned; what the
    /// handler may do, enqueueing and making objects that take no upload, it does.
    #[test]
    fn every_device_refuses_each_call_that_waits_for_the_submit_thread_in_a_scheduled_handler() {
        use crate::{PixelFormat, Region, TextureDescriptor, TextureType, TextureUsage};
        each_device("scheduled_refusals", |name, _, device| {
            let mut arena = Arena::new(&device);
            let (a, b) = (arena.buffer(), arena.buffer());
            let private = device
                .take_block(MemoryClass::Private, 4096)
                .expect("a Private block");
            let shared = device
                .take_block(MemoryClass::Shared, 4096)
                .expect("a Shared block");
            let event = shared_event(&device);
            let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
            d.usage = TextureUsage::RENDER_TARGET;
            let texture = device.new_texture_for_views(&d).expect("a texture");
            let queue = queue(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill(&command_buffer, &a, 3);
            let other = queue.command_buffer().expect("a command buffer");
            fill(&other, &b, 4);
            let later = queue.command_buffer().expect("a command buffer");
            let (answered, answers) = mpsc::channel();
            let (in_device, in_other, in_later, in_event, in_texture) = (
                device.clone(),
                other.clone(),
                later.clone(),
                event.clone(),
                texture.clone(),
            );
            let private_id = private.id;
            let shared_id = shared.id;
            command_buffer
                .add_scheduled_handler(move |own| {
                    let refused = |result: Result<(), Error>| -> String { format!("{result:?}") };
                    let region = Region {
                        origin: [0, 0, 0],
                        size: [1, 1, 1],
                    };
                    let mut bytes = [0_u8; 4];
                    let lines = vec![
                        ("status", format!("{:?}", own.status())),
                        (
                            "kernel times",
                            format!(
                                "{}",
                                own.kernel_start_time() > 0
                                    && own.kernel_end_time() >= own.kernel_start_time()
                            ),
                        ),
                        ("commit", refused(in_other.commit())),
                        ("wait_until_scheduled", refused(own.wait_until_scheduled())),
                        ("wait_until_completed", refused(own.wait_until_completed())),
                        (
                            "wait_until_signaled_value",
                            refused(
                                in_event
                                    .wait_until_signaled_value(1, Duration::ZERO)
                                    .map(drop),
                            ),
                        ),
                        (
                            "replace_region",
                            refused(in_texture.replace_region(
                                region,
                                0,
                                0,
                                crate::HostBytes::from(&[0; 4][..]),
                                4,
                                4,
                            )),
                        ),
                        (
                            "upload_region",
                            refused(in_texture.upload_region(region, 0, 0)),
                        ),
                        (
                            "get_bytes",
                            refused(in_texture.get_bytes(&mut bytes, 4, 4, region, 0, 0)),
                        ),
                        (
                            "zero_block_range",
                            refused(in_device.zero_block_range(private_id, 0, 0)),
                        ),
                        (
                            "new_managed_buffer",
                            refused(
                                (in_device.new_managed_buffer((shared_id, 1024), None, 64, 0))
                                    .map(drop),
                            ),
                        ),
                        (
                            "new_buffer zeroed in a Private block",
                            refused(in_device.new_buffer(private_id, 0, 64, Some(0)).map(drop)),
                        ),
                        (
                            "new_buffer in a Private block, not zeroed",
                            refused(in_device.new_buffer(private_id, 64, 64, None).map(drop)),
                        ),
                        (
                            "new_buffer zeroed in a Shared block",
                            refused(in_device.new_buffer(shared_id, 0, 64, Some(0)).map(drop)),
                        ),
                        ("enqueue", refused(in_later.enqueue())),
                        (
                            "new_texture_for_views",
                            refused(in_device.new_texture_for_views(&d).map(drop)),
                        ),
                    ];
                    let _ = answered.send(lines);
                })
                .expect("a handler");
            command_buffer.commit().expect("a commit");
            let lines = answers
                .recv_timeout(PATIENCE)
                .expect("the handler's answers");
            let refusal = format!(
                "{:?}",
                Err::<(), Error>(Error::Refused(Refusal::InScheduledHandler))
            );
            for (call, answer) in &lines {
                let want = match *call {
                    "status" => format!("{:?}", CommandBufferStatus::Scheduled),
                    "kernel times" => "true".to_string(),
                    "new_buffer in a Private block, not zeroed"
                    | "new_buffer zeroed in a Shared block"
                    | "enqueue"
                    | "new_texture_for_views" => "Ok(())".to_string(),
                    _ => refusal.clone(),
                };
                assert_eq!(*answer, want, "{name}: {call}");
            }
            assert_eq!(lines.len(), 16, "{name}: {lines:?}");
            finished(name, &command_buffer);
            assert_eq!(later.status(), CommandBufferStatus::Enqueued, "{name}");
            // Outside a handler the same calls answer as before. The one the handler enqueued
            // goes first: the other is enqueued behind it as it commits.
            later.commit().expect("a commit outside a handler");
            other.commit().expect("a commit outside a handler");
            finished(name, &later);
            finished(name, &other);
            assert_eq!((word(&a), word(&b)), (3, 4), "{name}");
            assert!(event
                .wait_until_signaled_value(0, Duration::ZERO)
                .expect("a wait outside a scheduled handler"));
        });
    }

    /// On the completion thread, in a completed handler and in a listener, a wait for what that
    /// thread has not yet done is refused with `WaitOnOwnThread`, a timed one too, never false at
    /// its timeout, and one already met answers at once: the handler's own command buffer has completed. A command buffer asked of a full
    /// queue is refused there and in a scheduled handler, on the submit thread.
    #[test]
    fn every_device_refuses_in_a_handler_each_wait_its_own_thread_would_have_to_end() {
        each_device("own_thread_waits", |name, _, device| {
            let mut arena = Arena::new(&device);
            let a = arena.buffer();
            let event = shared_event(&device);
            let queue = device.new_command_queue(2).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill(&command_buffer, &a, 5);
            let held = queue.command_buffer().expect("a command buffer");
            let answer = |result: Result<(), Error>| format!("{result:?}");
            let (answered, answers) = mpsc::channel();
            let (in_answered, in_queue) = (answered.clone(), queue.clone());
            command_buffer
                .add_scheduled_handler(move |_| {
                    let made = in_queue.command_buffer().map(drop);
                    let _ = in_answered.send(vec![("scheduled: a full queue", answer(made))]);
                })
                .expect("a handler");
            let (in_answered, in_queue, in_held, in_event) =
                (answered.clone(), queue.clone(), held.clone(), event.clone());
            command_buffer
                .add_completed_handler(move |own| {
                    let waited = |value| in_event.wait_until_signaled_value(value, PATIENCE);
                    let made = in_queue.command_buffer();
                    let lines = vec![
                        (
                            "completed: own completed",
                            answer(own.wait_until_completed()),
                        ),
                        (
                            "completed: own scheduled",
                            answer(own.wait_until_scheduled()),
                        ),
                        (
                            "completed: held completed",
                            answer(in_held.wait_until_completed()),
                        ),
                        (
                            "completed: held scheduled",
                            answer(in_held.wait_until_scheduled()),
                        ),
                        ("completed: value 0", format!("{:?}", waited(0))),
                        ("completed: value 1", format!("{:?}", waited(1))),
                        (
                            "completed: value 1 at once",
                            format!(
                                "{:?}",
                                in_event.wait_until_signaled_value(1, Duration::ZERO)
                            ),
                        ),
                        ("completed: room", format!("{:?}", made.as_ref().map(drop))),
                        (
                            "completed: a full queue",
                            answer(in_queue.command_buffer().map(drop)),
                        ),
                    ];
                    drop(made);
                    let _ = in_answered.send(lines);
                })
                .expect("a handler");
            command_buffer.commit().expect("a commit");
            let mut lines = answers
                .recv_timeout(PATIENCE)
                .expect("the scheduled answers");
            lines.extend(
                answers
                    .recv_timeout(PATIENCE)
                    .expect("the completed answers"),
            );
            finished(name, &command_buffer);
            let (in_queue, in_held, in_done, in_event) = (
                queue.clone(),
                held.clone(),
                command_buffer.clone(),
                event.clone(),
            );
            event
                .notify_listener(1, move |_| {
                    let waited = |value| in_event.wait_until_signaled_value(value, PATIENCE);
                    let made = in_queue.command_buffer();
                    let lines = vec![
                        ("listener: value 1", format!("{:?}", waited(1))),
                        ("listener: value 2", format!("{:?}", waited(2))),
                        (
                            "listener: held completed",
                            answer(in_held.wait_until_completed()),
                        ),
                        (
                            "listener: done completed",
                            answer(in_done.wait_until_completed()),
                        ),
                        ("listener: room", format!("{:?}", made.as_ref().map(drop))),
                        (
                            "listener: a full queue",
                            answer(in_queue.command_buffer().map(drop)),
                        ),
                    ];
                    drop(made);
                    let _ = answered.send(lines);
                })
                .expect("a listener");
            event.set_signaled_value(1);
            lines.extend(
                answers
                    .recv_timeout(PATIENCE)
                    .expect("the listener's answers"),
            );
            let refusal = answer(Err(Error::Refused(Refusal::WaitOnOwnThread)));
            for (call, got) in &lines {
                let want = match *call {
                    "completed: own completed"
                    | "completed: own scheduled"
                    | "completed: room"
                    | "listener: done completed"
                    | "listener: room" => "Ok(())".to_string(),
                    "completed: value 0" | "listener: value 1" => "Ok(true)".to_string(),
                    _ => refusal.clone(),
                };
                assert_eq!(*got, want, "{name}: {call}");
            }
            assert_eq!(lines.len(), 16, "{name}: {lines:?}");
            // Outside a handler the same waits answer as before.
            assert_eq!(word(&a), 5, "{name}");
            assert_eq!(held.status(), CommandBufferStatus::NotEnqueued, "{name}");
            let more = queue.command_buffer().expect("room outside a handler");
            assert!(event
                .wait_until_signaled_value(1, Duration::ZERO)
                .expect("a wait outside a handler"));
            drop((more, held, command_buffer, queue, event));
        });
    }

    /// At a loss the consumer reports, its own `vkQueuePresentKHR`'s (`Device::report_loss`): a
    /// host wait for a pending signal ends false; the command buffer in flight ends DeviceLost in
    /// flight, the held one and every later commit DeviceLost not in flight; the device names the
    /// reported call and result, none before it, and keeps them through a second report of
    /// another call; the device tears down, and a new device opens, names no loss, and runs work.
    #[test]
    fn every_device_ends_its_work_at_a_reported_loss_and_a_new_device_opens() {
        let first = DeviceLoss {
            call: "vkQueuePresentKHR",
            result: vk::Result::ERROR_DEVICE_LOST,
            fault: None,
        };
        let second = DeviceLoss {
            call: "vkAcquireNextImageKHR",
            result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            fault: None,
        };
        each_device("device_loss", |name, instance, device| {
            assert_eq!(device.loss(), None, "{name}: no loss before the loss");
            let (f, g) = (shared_event(&device), shared_event(&device));
            let queue = queue(&device);
            let other = self::queue(&device);
            // In flight: held with every later command behind busy work.
            let busy = Busy::new(&device);
            let in_flight = queue.command_buffer().expect("a command buffer");
            in_flight
                .encode_signal_event(g.as_event(), 1)
                .expect("a signal");
            let before = device.counters().metrics().submissions;
            in_flight.commit().expect("a commit");
            let held = other.command_buffer().expect("a command buffer");
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            held.commit().expect("a commit");
            // Submitted, and `Committed` while the value it waits for is not reached.
            within("the submission", || {
                device.counters().metrics().submissions > before
            });
            let waiting = thread::scope(|scope| {
                let waiter = scope.spawn(|| {
                    g.wait_until_signaled_value(1, PATIENCE)
                        .expect("a wait outside a scheduled handler")
                });
                thread::sleep(Duration::from_millis(20));
                device.report_loss(first.clone());
                waiter.join().expect("the waiter")
            });
            assert!(
                !waiting,
                "{name}: a wait for a pending signal ends at the loss"
            );
            assert_eq!(
                device.loss(),
                Some(first.clone()),
                "{name}: the reported loss"
            );
            device.report_loss(second.clone());
            assert_eq!(
                device.loss(),
                Some(first.clone()),
                "{name}: a second report keeps the first's cause"
            );

            busy.open();
            finished(name, &in_flight);
            finished(name, &held);
            let late = queue.command_buffer().expect("a command buffer");
            late.commit().expect("a commit");
            finished(name, &late);
            for (command_buffer, in_flight) in [(&in_flight, true), (&held, false), (&late, false)]
            {
                assert_eq!(
                    command_buffer.error(),
                    Some(CommandBufferError::DeviceLost { in_flight }),
                    "{name}"
                );
            }
            let physical = physical_of(instance, &device);
            drop((busy, f, g, queue, other, device));

            let device = Device::new(&physical, DeviceOptions::default()).expect("a new device");
            assert_eq!(device.loss(), None, "{name}: the new device's loss");
            let counters = device.counters();
            let mut arena = Arena::new(&device);
            let a = arena.buffer();
            let queue = self::queue(&device);
            let after = queue.command_buffer().expect("a command buffer");
            fill(&after, &a, 9);
            after.commit().expect("a commit");
            finished(name, &after);
            assert_eq!(
                (after.status(), word(&a)),
                (CommandBufferStatus::Completed, 9)
            );
            drop((a, arena, queue, device));
            within("the new device's teardown", || {
                counters.gauges() == Gauges::default()
            });
        });
    }

    /// A reported loss is taken as given, whatever its result: a report of `SUCCESS` ends the
    /// device's work as any loss does, the device names it, and a later commit ends DeviceLost not
    /// in flight.
    #[test]
    fn every_device_takes_a_reported_loss_as_given_whatever_its_result() {
        let report = DeviceLoss {
            call: "vkQueuePresentKHR",
            result: vk::Result::SUCCESS,
            fault: None,
        };
        each_device("reported_loss_as_given", |name, _, device| {
            let queue = queue(&device);
            device.report_loss(report.clone());
            assert_eq!(
                device.loss(),
                Some(report.clone()),
                "{name}: the reported loss"
            );
            let late = queue.command_buffer().expect("a command buffer");
            late.commit().expect("a commit");
            finished(name, &late);
            assert_eq!(
                late.error(),
                Some(CommandBufferError::DeviceLost { in_flight: false }),
                "{name}"
            );
            drop((late, queue, device));
        });
    }

    /// A block released while a command buffer that uses it is held stays out of the pool until
    /// that command buffer retires.
    #[test]
    fn every_device_keeps_a_released_block_until_its_last_command_buffer_retires() {
        each_device("block_retention", |name, _, device| {
            let block = device
                .take_block(MemoryClass::Private, 1)
                .expect("a Private block");
            let buffer = device
                .new_buffer(block.id, 0, 4096, None)
                .expect("a buffer");
            let f = shared_event(&device);
            let queue = queue(&device);
            let held = queue.command_buffer().expect("a command buffer");
            fill(&held, &buffer, 5);
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            held.commit().expect("a commit");
            drop(buffer);
            let id = block.id;
            device.release_block(block).expect("a release");
            thread::sleep(Duration::from_millis(50));
            let pool = &device.core().pool;
            assert!(!pool.holds(id), "{name}: back while in use");
            f.set_signaled_value(1);
            finished(name, &held);
            within("the block's return", || pool.holds(id));
            drop((f, queue, device));
        });
    }

    /// A block whose last handle drops, with no `release_block`, while a held command buffer uses
    /// it: its number names nothing at once, and the block stays out of the pool until the
    /// command buffer completes, then comes back to it.
    #[test]
    fn every_device_gives_back_a_dropped_block_once_its_last_command_buffer_retires() {
        each_device("block_drop", |name, _, device| {
            let f = shared_event(&device);
            let queue = queue(&device);
            let block = (device.take_block(MemoryClass::Shared, 1)).expect("a Shared block");
            let id = block.id;
            let buffer = (device.new_buffer(id, 0, 4096, None)).expect("a buffer");
            let held = queue.command_buffer().expect("a command buffer");
            fill(&held, &buffer, 5);
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            held.commit().expect("a commit");
            drop((buffer, block));
            assert!(
                matches!(device.new_buffer(id, 0, 256, None), Err(Error::NoSuchBlock(gone)) if gone == id),
                "{name}: the dropped block's number"
            );
            thread::sleep(Duration::from_millis(50));
            let pool = &device.core().pool;
            assert!(!pool.holds(id), "{name}: back while in use");
            f.set_signaled_value(1);
            finished(name, &held);
            within("the dropped block's return", || pool.holds(id));
            drop((f, queue, device));
        });
    }

    /// Clones of a block's handle share one claim: dropping one of two gives nothing back, and
    /// the block's number still names it; releasing the other gives it back. A handle that
    /// outlives its device finds its block freed by the teardown, every gauge zero, and its drop
    /// does nothing.
    #[test]
    fn every_device_gives_back_a_block_at_its_last_clone_and_frees_it_at_teardown() {
        each_device("block_clones", |name, _, device| {
            let counters = device.counters();
            let pool = &device.core().pool;
            let block = (device.take_block(MemoryClass::Shared, 1)).expect("a Shared block");
            let id = block.id;
            let clone = block.clone();
            drop(block);
            thread::sleep(Duration::from_millis(50));
            assert!(
                !pool.holds(id),
                "{name}: back with one of two clones dropped"
            );
            let buffer = (device.new_buffer(clone.id, 0, 256, None))
                .unwrap_or_else(|error| panic!("{name}: the clone's number: {error}"));
            drop(buffer);
            device.release_block(clone).expect("a release");
            within("the block back at its last clone", || pool.holds(id));

            let kept = (device
                .take_block(MemoryClass::Shared, 2 * MemoryClass::Shared.block_len()))
            .expect("a long block");
            drop(device);
            within("the teardown", || counters.gauges() == Gauges::default());
            drop(kept);
            assert_eq!(
                counters.gauges(),
                Gauges::default(),
                "{name}: a handle dropped after the teardown"
            );
        });
    }

    /// The variable that makes [`teardown_child`] a consumer of the shape it names.
    const TEARDOWN_CHILD: &str = "M2V_TEARDOWN_CHILD";

    /// The child of `every_device_ends_its_teardown_before_the_process_exits`, run in a process of
    /// its own: a consumer that makes a device, drops every handle it holds, and returns from
    /// `main`, which exits the process. In the shape `listener`, a shared event's listener holds a
    /// handle of the device, and the host value that runs it is set before the consumer's drops:
    /// where the listener's drop is the last, the reaper tears the device down, and the consumer
    /// waits for that ([`Instance::wait_until_devices_torn_down`]) before it drops the instance.
    /// In the shape `kept`, a committed command buffer keeps a buffer of the device, and the
    /// consumer calls nothing: what the command buffer keeps holds no handle of the device, so the
    /// consumer's drop is the last and tears the device down in that drop. It reports whether the
    /// wait answered false or any gauge was still above zero as it returned. Without its variable,
    /// it does nothing.
    #[test]
    fn teardown_child() {
        let Ok(shape) = std::env::var(TEARDOWN_CHILD) else {
            return;
        };
        let instance = crate::tests::validation::bare(&[]);
        let (_, physical) = (devices_of_the_cfgs_kind(&instance).into_iter().next())
            .expect("a device of this cfg's kind");
        let device = Device::new(&physical, DeviceOptions::default()).expect("a device");
        let counters = device.counters();
        match shape.as_str() {
            "listener" => {
                let event = shared_event(&device);
                let held = device.clone();
                event
                    .notify_listener(1, move |_| drop(held))
                    .expect("a listener");
                event.set_signaled_value(1);
            }
            "kept" => {
                let mut arena = Arena::new(&device);
                let a = arena.buffer();
                let command_buffer = (queue(&device).command_buffer()).expect("a command buffer");
                fill(&command_buffer, &a, 1);
                command_buffer.commit().expect("a commit");
            }
            _ => panic!("{shape}: no such shape"),
        }
        drop((device, physical));
        let torn = shape != "listener" || instance.wait_until_devices_torn_down(PATIENCE);
        drop(instance);
        let live = !torn || counters.gauges() != Gauges::default();
        println!("M2V teardown child: live at return {live}");
    }

    /// In processes of their own, eight at a time, a consumer makes a device, drops every handle
    /// it holds and returns from `main` ([`teardown_child`], each shape): each exits 0, its wait,
    /// where it waits, answering true, and with every gauge at zero as it returned. `M2V_TEARDOWN_RUNS` runs each shape that many times (16 by
    /// default).
    #[test]
    fn every_device_ends_its_teardown_before_the_process_exits() {
        let runs: usize = std::env::var("M2V_TEARDOWN_RUNS")
            .ok()
            .and_then(|runs| runs.parse().ok())
            .unwrap_or(16);
        let executable = std::env::current_exe().expect("the test executable");
        let mut failed = Vec::new();
        for shape in ["listener", "kept"] {
            let (mut clean, mut live, mut died) = (0, 0, Vec::new());
            let mut left = runs;
            while left > 0 {
                let children: Vec<_> = (0..left.min(8))
                    .map(|_| {
                        std::process::Command::new(&executable)
                            .args([
                                "--exact",
                                "tests::command::on_device::teardown_child",
                                "--nocapture",
                                "--test-threads=1",
                            ])
                            .env(TEARDOWN_CHILD, shape)
                            .env_remove("VK_INSTANCE_LAYERS")
                            .stdout(std::process::Stdio::piped())
                            .stderr(std::process::Stdio::piped())
                            .spawn()
                            .expect("a child")
                    })
                    .collect();
                left -= children.len();
                for child in children {
                    let output = child.wait_with_output().expect("the child's output");
                    let text = format!(
                        "{}{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                    if !output.status.success() {
                        println!("M2V teardown {shape}: {:?}\n{text}", output.status);
                        died.push(format!("{:?}", output.status));
                    } else if text.contains("live at return false") {
                        clean += 1;
                    } else if text.contains("live at return true") {
                        live += 1;
                    } else {
                        println!("M2V teardown {shape}: no report\n{text}");
                        died.push("no report".to_string());
                    }
                }
            }
            println!(
                "M2V teardown {shape}: {clean} clean, {live} live at return, {} died of {runs}: \
                 {died:?}",
                died.len()
            );
            if clean != runs {
                failed.push(shape);
            }
        }
        assert!(failed.is_empty(), "shapes with a run not clean: {failed:?}");
    }

    /// A command buffer with nothing for the GPU takes no submission. Committed behind busy work
    /// on its queue and before more, it completes between them, its completed handler between
    /// theirs, its GPU start and end equal and at the busy one's end; committed alone, it
    /// completes at once, its times equal and not before its commit. Metal reports both so.
    #[test]
    fn every_device_completes_an_empty_command_buffer_without_a_submission() {
        each_device("command_empty", |name, _, device| {
            let work = private(&device);
            let queue = queue(&device);
            let log = Log::default();
            let before = device.counters().metrics().submissions;
            let logged = |label: &'static str, command_buffer: &CommandBuffer| {
                let log = log.clone();
                command_buffer
                    .add_completed_handler(move |_| log.push(label))
                    .expect("a completed handler");
            };
            let fills = |command_buffer: &CommandBuffer| {
                let (target, offset) = work.device_range();
                let size = work.length();
                command_buffer
                    .record_for_test(&[&work], |device, command| {
                        for word in 0..64 {
                            // SAFETY: the command buffer is recording; the range is the
                            // buffer's, in whole words.
                            unsafe { device.cmd_fill_buffer(command, target, offset, size, word) };
                        }
                    })
                    .expect("fills");
            };
            let first = queue.command_buffer().expect("a command buffer");
            fills(&first);
            logged("first", &first);
            let empty = queue.command_buffer().expect("a command buffer");
            logged("empty", &empty);
            let last = queue.command_buffer().expect("a command buffer");
            fills(&last);
            logged("last", &last);
            for command_buffer in [&first, &empty, &last] {
                command_buffer.commit().expect("a commit");
            }
            for command_buffer in [&first, &empty, &last] {
                finished(name, command_buffer);
            }
            assert_eq!(log.lines(), ["first", "empty", "last"], "{name}");
            let submissions = device.counters().metrics().submissions - before;
            assert_eq!(submissions, 2, "{name}: the empty one took no submission");
            let (start, end) = (empty.gpu_start_time(), empty.gpu_end_time());
            assert_eq!(start, end, "{name}: equal times");
            assert!(
                start > 0 && start >= first.gpu_end_time(),
                "{name}: {start}"
            );
            // Scheduled in order: its kernel start is no earlier than the first's kernel end, and
            // the last's no earlier than its own.
            for (before, after) in [(&first, &empty), (&empty, &last)] {
                let (end, start) = (before.kernel_end_time(), after.kernel_start_time());
                assert!(start >= end, "{name}: kernel start {start} before {end}");
            }
            assert_eq!(empty.status(), CommandBufferStatus::Completed, "{name}");
            let alone = queue.command_buffer().expect("a command buffer");
            alone.commit().expect("a commit");
            finished(name, &alone);
            let (start, end) = (alone.gpu_start_time(), alone.gpu_end_time());
            assert_eq!(start, end, "{name}: equal times alone");
            let kernel_end = alone.kernel_end_time();
            assert!(
                start >= kernel_end,
                "{name}: alone {start} before {kernel_end}"
            );
            println!("{name}: an empty command buffer completed in order with no submission");
            drop((first, empty, last, alone, work, queue, device));
        });
    }

    /// `Gauges::threads` holds the device's four threads (its reaper, block-maker, submit and
    /// completion threads) as `Device::new` returns, with no wait: each thread holds its slot
    /// before its start returns.
    #[test]
    fn every_device_counts_its_four_threads_as_it_is_made() {
        each_device("device_threads", |name, _, device| {
            let threads = device.counters().gauges().threads;
            let held = threads.iter().filter(|&&id| id != 0).count();
            assert_eq!(held, 4, "{name}: {threads:?}");
        });
    }

    /// A stream of command buffers with nothing for the GPU leaves nothing behind: 10,000
    /// committed, none waited for until the last, take no submission, the live command buffers
    /// never pass the queue's 64, and every gauge is back where it was once they retire. Each
    /// one's completed handler runs, in commit order, seeing it completed with its GPU start and
    /// end equal and nonzero.
    #[test]
    fn every_device_keeps_its_gauges_flat_under_ten_thousand_empty_commits() {
        each_device("command_empty_stream", |name, _, device| {
            let queue = queue(&device);
            let counters = device.counters();
            // The block-maker thread has made its ready blocks: the gauges hold still.
            crate::tests::blit::on_device::settled(&device);
            let (gauges, submissions) = (counters.gauges(), counters.metrics().submissions);
            // Handlers run so far, and those that ran out of commit order or saw a command buffer
            // not completed or with other times.
            let (ran, wrong) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
            let mut last = None;
            for at in 0..10_000 {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let (ran, wrong) = (Arc::clone(&ran), Arc::clone(&wrong));
                command_buffer
                    .add_completed_handler(move |done| {
                        let (start, end) = (done.gpu_start_time(), done.gpu_end_time());
                        let right = ran.fetch_add(1, Ordering::SeqCst) == at
                            && done.status() == CommandBufferStatus::Completed
                            && start == end
                            && start > 0;
                        if !right {
                            wrong.fetch_add(1, Ordering::SeqCst);
                        }
                    })
                    .expect("a completed handler");
                command_buffer.commit().expect("a commit");
                let live = counters.gauges().command_buffers;
                assert!(
                    live <= gauges.command_buffers + 64,
                    "{name}: {live} live at {at}"
                );
                last = Some(command_buffer);
            }
            finished(name, &last.take().expect("the last"));
            let deadline = Instant::now() + PATIENCE;
            while counters.gauges() != gauges && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(counters.gauges(), gauges, "{name}: the gauges once retired");
            assert_eq!(
                counters.metrics().submissions,
                submissions,
                "{name}: a submission"
            );
            assert_eq!(
                (ran.load(Ordering::SeqCst), wrong.load(Ordering::SeqCst)),
                (10_000, 0),
                "{name}: handlers run, and those out of order, not completed or with other times"
            );
            drop((queue, device));
        });
    }

    /// The command buffers of a group share submit infos where no event signal separates them,
    /// and record what they record alone. A mix of a fill whose segment's end barrier reaches
    /// commands, a Shared fill whose end barrier reaches the host, an event's signal (which ends
    /// its group) and a wait for it, and between them a command buffer with nothing for the GPU
    /// (no timestamps, no info) records the same barriers, passes and bytes copied and leaves the
    /// same words committed behind three busy submissions, held in groups, as committed one at a
    /// time. The groups pass two submit infos for the mix's four command buffers with commands:
    /// one for the fills up to the signal, which ends the group, and one for the waiter, in the
    /// next. One at a time passes one each. The busy work is held by the host until the submit thread has staged
    /// the whole mix: a completion seen sooner would send the held group with what it had staged
    /// by then, which splits it where the completion happened to arrive.
    #[test]
    fn every_device_shares_submit_infos_within_a_group_and_records_what_it_records_alone() {
        each_device("command_infos", |name, _, device| {
            let queue = queue(&device);
            let mut arena = Arena::new(&device);
            let (a, b) = (arena.buffer(), arena.buffer());
            // An `MTLEvent`: its waiter is scheduled once staged, where a shared event's stays
            // `Committed` until the value is reached, which the busy work holds back here.
            let event = device.new_event().expect("an event");
            let counters = device.counters();
            // The mix, each command buffer handed to `commit` as it is made: the fills write
            // `base + 1..=3` into `a` and `value` into every byte of `b`.
            let mix = |base: u32, value: u8, signal: u64, commit: &dyn Fn(&CommandBuffer)| {
                let first = queue.command_buffer().expect("a command buffer");
                fill(&first, &a, base + 1);
                commit(&first);
                let shared = queue.command_buffer().expect("a command buffer");
                let mut encoder = shared.blit_command_encoder().expect("an encoder");
                encoder.fill_buffer(&b, 0..256, value).expect("a fill");
                encoder.end_encoding();
                commit(&shared);
                let signals = queue.command_buffer().expect("a command buffer");
                fill(&signals, &a, base + 2);
                signals
                    .encode_signal_event(&event, signal)
                    .expect("a signal");
                commit(&signals);
                // After the signal, which sent the group: one committed while a group is held
                // would send it at once and end the group there.
                let empty = queue.command_buffer().expect("a command buffer");
                commit(&empty);
                let waits = queue.command_buffer().expect("a command buffer");
                waits.encode_wait_for_event(&event, signal).expect("a wait");
                fill(&waits, &a, base + 3);
                commit(&waits);
                vec![first, shared, signals, empty, waits]
            };
            let deterministic = || {
                let m = counters.metrics();
                (m.barriers, m.passes, m.bytes_copied)
            };
            let infos = || counters.metrics().submit_infos;
            let bytes = |buffer: &Buffer| {
                let contents = buffer.contents().expect("a Shared buffer");
                // SAFETY: the buffer is alive and 256 bytes long, and no command writes it now.
                unsafe { std::slice::from_raw_parts(contents.as_ptr(), 256) }.to_vec()
            };

            // Behind three busy submissions the host holds: the mix is held in groups, counted
            // before any of the busy work can complete.
            let signal = 1;
            let mut kept = Vec::new();
            let mut counted = None;
            let mark = infos();
            behind_busy(name, &device, &queue, |busy, submitted, _| {
                // The busy ones are submitted and counted, one submit info each, before the mix's
                // counts start: `behind_busy` runs the body once they are committed, which may be
                // before the submit thread counts them.
                within(name, || {
                    let m = counters.metrics();
                    m.submissions >= submitted + AHEAD && m.submit_infos >= mark + AHEAD
                });
                let (before, infos_before) = (deterministic(), infos());
                let grouped = mix(10, 0x5a, signal, &|command_buffer| {
                    command_buffer.commit().expect("a commit")
                });
                // The held group goes once the completion thread has seen a busy one complete,
                // which then sends what the submit thread had staged of the mix so far: the split
                // turns on when the completion arrives. So the busy work is opened once every one
                // of the mix is staged (no longer `Committed`): a commit returns before the submit
                // thread stages it, and one it stages after a completion goes with fewer than three
                // in flight.
                for command_buffer in &grouped {
                    within(name, || {
                        command_buffer.status() != CommandBufferStatus::Committed
                    });
                }
                busy.open();
                for command_buffer in busy.running().iter().chain(&grouped) {
                    finished(name, command_buffer);
                    assert_eq!(command_buffer.error(), None, "{name}");
                }
                counted = Some((before, (deterministic(), infos() - infos_before)));
                kept.extend(grouped);
            });
            let (before, in_groups) = counted.expect("the mix counted behind running work");
            let grouped_words = (word(&a), bytes(&b));

            // One at a time, each complete before the next is made.
            let (before_alone, infos_alone) = (deterministic(), infos());
            let alone = mix(10, 0x5a, signal + 1, &|command_buffer| {
                command_buffer.commit().expect("a commit");
                finished(name, command_buffer);
            });
            let one_at_a_time = (deterministic(), infos() - infos_alone);
            let delta = |(after, _): ((u64, u64, u64), u64), (b, p, c): (u64, u64, u64)| {
                (after.0 - b, after.1 - p, after.2 - c)
            };
            assert_eq!(
                delta(in_groups, before),
                delta(one_at_a_time, before_alone),
                "{name}: barriers, passes and bytes copied, in groups and one at a time"
            );
            assert_eq!(
                grouped_words,
                (word(&a), bytes(&b)),
                "{name}: the words, in groups and one at a time"
            );
            assert_eq!((word(&a), bytes(&b)), (13, vec![0x5a; 256]), "{name}");
            assert_eq!(
                in_groups.1, 2,
                "{name}: submit infos for four command buffers with commands, in groups"
            );
            assert_eq!(
                one_at_a_time.1, 4,
                "{name}: one submit info each, one at a time"
            );
            println!(
                "{name}: the mix passed {} submit infos in groups",
                in_groups.1
            );
            drop((kept, alone, a, b, arena, event, queue, device));
        });
    }
}
