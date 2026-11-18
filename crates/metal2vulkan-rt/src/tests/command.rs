//! Command queues, command buffers and events, under a device cfg on every device of the cfg's
//! kind: states, handlers and refusals; holds per segment; the host floor; host values racing GPU
//! signals; listeners; abandon; quiesce, also in an `exit()` handler; teardown from a handler and
//! with work held; refusals of what has started; a loss; a released block kept until its last
//! command buffer retires; a block whose last handle drops freed once its last command buffer
//! retires; a block's clones, one dropped giving nothing back, and a handle outliving its
//! device; and the thread gauge holding the device's four threads as it is made. Without a
//! device: a push whose bytes the command buffer holds is not recorded again.

use crate::command::Pushed;

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
    use crate::device::lock;
    use crate::submit::IN_FLIGHT;
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        host_time, Block, BlockId, BlockMappings, Buffer, CommandBuffer, CommandBufferError,
        CommandBufferStatus, CommandQueue, Device, DeviceOptions, DispatchType, Error, Fence,
        Gauges, Instance, MemoryClass, Refusal, SharedEvent,
    };
    use ash::vk;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};

    /// How long a test waits for what must happen.
    pub(crate) const PATIENCE: Duration = Duration::from_secs(60);

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

    /// Opens each device of the cfg's kind with `options`, runs `body`, which drops the device,
    /// and checks that every gauge reaches zero.
    pub(crate) fn each_device(test: &str, mut body: impl FnMut(&str, &Instance, Device)) {
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            body(name, &instance, device);
            within("the teardown", || counters.gauges() == Gauges::default());
            print_gauges_at_drop(test, counters.gauges());
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
    fn fill(command_buffer: &CommandBuffer, buffer: &Buffer, word: u32) {
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
        command_buffer.wait_until_completed();
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

    /// A command buffer goes through Metal's states, runs its handlers in order on the completion
    /// thread, and runs its commands; a queue keeps enqueue order and bounds its command buffers;
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
                e.wait_until_signaled_value(1, PATIENCE),
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
                command_buffer.wait_until_completed();
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
    /// its completed handler reads, through slots its command buffers reuse, with or without
    /// commands. (Each command buffer of one submission starting no earlier than the one before
    /// ends is the group test's.)
    #[test]
    fn every_device_times_command_buffers_on_the_host_clock() {
        each_device("command_times", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            // Fewer slots than command buffers, so slots and their queries are reused.
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

    /// Busy work a queue keeps in flight until the test lets it end: each command buffer's
    /// submission waits for a timeline semaphore to reach 1, which only the host signals
    /// ([`Busy::open`]), then fills a block. It is submitted and does not complete before the
    /// test's checks are made, however fast the device or however loaded the host. A wait
    /// submitted before its signal is defined for a timeline semaphore, and the host signals it
    /// with `vkSignalSemaphore` (an event the host sets while a command buffer waiting on it is
    /// pending is not: `VUID-vkSetEvent-event-09543`). Dropped, it opens the work and waits for
    /// each one, so a failed check does not leave the device's work waiting forever.
    struct Busy {
        device: Device,
        semaphore: vk::Semaphore,
        work: Buffer,
        running: Vec<CommandBuffer>,
    }

    impl Busy {
        fn new(device: &Device) -> Busy {
            let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
            // SAFETY: the device is live; the semaphore is destroyed in `drop`.
            let semaphore = unsafe { device.core().device.create_semaphore(&info, None) }
                .expect("a timeline semaphore");
            Busy {
                device: device.clone(),
                semaphore,
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
            let size = self.work.length();
            command_buffer.wait_semaphore_for_test(self.semaphore, 1);
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
        fn open(&self) {
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
                command_buffer.wait_until_completed();
            }
            // SAFETY: every submission that waits on it completed.
            unsafe { (self.device.core().device).destroy_semaphore(self.semaphore, None) };
        }
    }

    /// The submissions a queue keeps in flight before it holds a group ([`IN_FLIGHT`]).
    const AHEAD: u64 = IN_FLIGHT as u64;

    /// Runs `body` with as many busy command buffers committed on `queue` as a queue keeps in
    /// flight ([`IN_FLIGHT`]), each a submission of its own ([`Busy::commit`]), so its next commit
    /// is held while they wait for the host. `body` takes the busy work, the submissions counted
    /// before it, and the progress the completion thread had seen before it
    /// (`Done::completed`): every earlier command buffer's, so the progress stays there until the
    /// body opens the busy work ([`Busy::open`]) and the completion thread sees a busy one
    /// complete, and a held group goes no earlier than that. The busy work is opened after `body`
    /// where it did not, and waited for.
    fn behind_busy(
        name: &str,
        device: &Device,
        queue: &CommandQueue,
        body: impl FnOnce(&Busy, u64, u64),
    ) {
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

    /// What a second queue adds, counted: with the first queue's busy command buffers running
    /// ([`IN_FLIGHT`] of them), the second queue's first as many commits go alone (that many lone
    /// submissions across queues); once both queues have that many running, a commit on each is
    /// held, the second opening
    /// its group beside the first's (one group beside another queue). Both counted before any
    /// busy one completes: a held group that goes as its own queue's work retires is lone too
    /// where the other queue's is still in flight, which turns on when the completion thread
    /// looks. No group goes for a reason that is not its own queue's.
    #[test]
    fn every_device_counts_what_a_second_queue_adds() {
        each_device("command_cross_queue", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let (first, second) = (queue(&device), queue(&device));
            let before = device.counters().metrics();
            let mut busy = Busy::new(&device);
            for queue in [&first, &second] {
                for _ in 0..IN_FLIGHT {
                    busy.commit(name, queue);
                }
            }
            let held = [
                queue_fill(&first, &buffer, 1),
                queue_fill(&second, &buffer, 2),
            ];
            for command_buffer in &held {
                within(name, || {
                    command_buffer.status() != CommandBufferStatus::Committed
                });
            }
            // Read before any busy one can complete: no retirement sent anything counted in it.
            let held_now = device.counters().metrics();
            busy.open();
            for command_buffer in busy.running().iter().chain(&held) {
                finished(name, command_buffer);
            }
            let after = device.counters().metrics();
            assert_eq!(
                after.forced_flushes_across_queues - before.forced_flushes_across_queues,
                0,
                "{name}"
            );
            let counted = (
                held_now.lone_submissions_across_queues - before.lone_submissions_across_queues,
                held_now.groups_beside_another_queue - before.groups_beside_another_queue,
            );
            assert_eq!(
                counted,
                (AHEAD, 1),
                "{name}: lone submissions, groups beside"
            );
            drop((busy, buffer, arena, first, second, device));
        });
    }

    /// A commit's wait for the submission lock is counted: one made while another thread holds
    /// the lock for 20 ms waits at least half that long (the committing thread starts within the
    /// 20 ms, not before them).
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
            thread::sleep(Duration::from_millis(20));
            drop(held);
            committing.join().expect("the committing thread");
            finished(name, &command_buffer);
            let metrics = device.counters().metrics();
            assert!(
                metrics.commit_lock_wait_max_nanos >= 10_000_000,
                "{name}: {}",
                metrics.commit_lock_wait_max_nanos
            );
            assert!(metrics.commit_lock_wait_nanos >= metrics.commit_lock_wait_max_nanos);
            drop((command_buffer, buffer, arena, queue, device));
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

    /// A queue keeps three submissions in flight: behind two running ones a commit goes at once,
    /// alone, and the commit after it is held, both seen before the first busy one ends on the
    /// GPU.
    #[test]
    fn every_device_keeps_three_submissions_in_flight_and_holds_the_fourth() {
        each_device("command_in_flight", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
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

    /// A command buffer dropped before its commit leaves its slot's queries as it found them,
    /// reset, and the next command buffer takes the slot again while three others of the queue run,
    /// with no second reset (the validation layer reports a reset of a query not available while
    /// work naming its pool runs), and has its own GPU times.
    #[test]
    fn every_device_takes_again_the_slot_of_a_command_buffer_that_never_ran() {
        each_device("command_slot_again", |name, _, device| {
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
                drop(queue.command_buffer().expect("a command buffer"));
                let again = queue.command_buffer().expect("a command buffer");
                // Each was submitted and none had completed when the slot was taken again.
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

    /// The consumer's hook as these tests see it: each block unmapped, and the thread it ran on.
    #[derive(Default)]
    struct Hook(Mutex<Vec<(BlockId, Option<String>)>>);

    impl BlockMappings for Hook {
        fn unmap(&self, block: BlockId) {
            let thread = thread::current().name().map(str::to_string);
            lock(&self.0).push((block, thread));
        }
    }

    /// Quiesce, from a thread other than the one that made the device, with handles alive: held
    /// work ends NotRun with its handlers run, every handed block is unmapped once on that
    /// thread, calls that need a thread are refused, memory stays mapped, and a second call
    /// answers at once. The last drop then frees everything without the hook.
    #[test]
    fn every_device_quiesces_with_handles_alive() {
        let instance = Instance::new(&[]).expect("an instance at the floor");
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
            let started = Instant::now();
            assert!(device.quiesce(PATIENCE), "{name}: the first call's answer");
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "{name}: at once"
            );
            assert_eq!(held.error(), Some(CommandBufferError::NotRun), "{name}");
            held.wait_until_completed();

            let mut unmapped = lock(&hook.0).clone();
            unmapped.sort();
            let mut handed = vec![(arena.block.id, None), (private.id, None)];
            handed.sort();
            assert_eq!(
                unmapped, handed,
                "{name}: each handed block once, on the quiescing thread"
            );
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
            within("the teardown", || counters.gauges() == Gauges::default());
            print_gauges_at_drop("quiesce", counters.gauges());
            assert_eq!(lock(&hook.0).len(), 2, "{name}: no unmap after quiesce");
        }
    }

    /// The last handle dropped in a completed handler: the reaper runs the teardown once the
    /// handler returns, so the block still held is unmapped there, and every gauge reads zero once
    /// the device is destroyed. The handler's handle is the last: `commit` gave the command
    /// buffer's up before the command buffer could complete.
    #[test]
    fn every_device_tears_down_from_a_completed_handler() {
        let instance = Instance::new(&[]).expect("an instance at the floor");
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
            within("the teardown", || counters.gauges() == Gauges::default());
            print_gauges_at_drop("handler_teardown", counters.gauges());
            assert_eq!(
                *lock(&hook.0),
                [(block.id, Some("m2v-reaper".to_string()))],
                "{name}: unmapped once, by the reaper"
            );
        }
    }

    /// At a loss: a host wait for a pending signal ends false; the command buffer in flight ends
    /// DeviceLost in flight, the held one and every later commit DeviceLost not in flight; the
    /// device tears down, and a new device opens and runs work.
    #[test]
    fn every_device_ends_its_work_at_a_loss_and_a_new_device_opens() {
        each_device("device_loss", |name, instance, device| {
            let (e, f, g) = (
                shared_event(&device),
                shared_event(&device),
                shared_event(&device),
            );
            let queue = queue(&device);
            let other = self::queue(&device);
            // In flight: its wait is on a signal the test makes as if submitted, then gives from
            // the host.
            e.pretend_submitted_for_test(1);
            let in_flight = queue.command_buffer().expect("a command buffer");
            in_flight
                .encode_wait_for_event(e.as_event(), 1)
                .expect("a wait");
            in_flight
                .encode_signal_event(g.as_event(), 1)
                .expect("a signal");
            in_flight.commit().expect("a commit");
            let held = other.command_buffer().expect("a command buffer");
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            held.commit().expect("a commit");
            within("the scheduling", || {
                in_flight.status() == CommandBufferStatus::Scheduled
            });
            let waiting = thread::scope(|scope| {
                let waiter = scope.spawn(|| g.wait_until_signaled_value(1, PATIENCE));
                thread::sleep(Duration::from_millis(20));
                device.lose_for_test();
                waiter.join().expect("the waiter")
            });
            assert!(
                !waiting,
                "{name}: a wait for a pending signal ends at the loss"
            );

            let raw = device.raw();
            let signal = vk::SemaphoreSignalInfo::default()
                .semaphore(e.as_event().inner().semaphore)
                .value(1);
            // SAFETY: the semaphore is the device's, and 1 is above its value.
            unsafe { raw.device.signal_semaphore(&signal) }.expect("a host signal");
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
            drop((e, f, g, queue, other, device));

            let device = Device::new(&physical, DeviceOptions::default()).expect("a new device");
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

    /// The child's device, its reader thread and whether it reads, for the exit handler.
    static EXIT_DEVICE: OnceLock<Device> = OnceLock::new();
    static READER: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
    static READING: AtomicBool = AtomicBool::new(true);

    unsafe extern "C" {
        fn atexit(handler: extern "C" fn()) -> std::ffi::c_int;
        fn exit(code: std::ffi::c_int) -> !;
    }

    /// The consumer's exit handler: it parks its own reader, then quiesces twice.
    extern "C" fn quiesce_at_exit() {
        READING.store(false, Ordering::Release);
        if let Some(reader) = lock(&READER).take() {
            let _ = reader.join();
        }
        if let Some(device) = EXIT_DEVICE.get() {
            let first = device.quiesce(Duration::from_secs(10));
            let second = device.quiesce(Duration::from_secs(10));
            println!("M2V exit: quiesced {first} {second}");
        }
    }

    fn register_exit_handler() {
        // SAFETY: the handler is a plain function that lives as long as the process.
        assert_eq!(unsafe { atexit(quiesce_at_exit) }, 0, "atexit");
    }

    /// The child of `every_device_quiesces_in_an_exit_handler`, run in a process of its own:
    /// without its variable, it does nothing.
    #[test]
    fn exit_child() {
        let Ok(mode) = std::env::var("M2V_EXIT_CHILD") else {
            return;
        };
        if mode == "before-instance" {
            register_exit_handler();
        }
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let (_, physical) = devices_of_the_cfgs_kind(&instance)
            .into_iter()
            .next()
            .expect("a device of this cfg's kind");
        let device = Device::new(&physical, DeviceOptions::default()).expect("a device");
        if mode == "after-device" {
            register_exit_handler();
        }
        // Blocks stay taken; a reader thread reads one; the block-maker keeps released ones; a
        // command buffer is held.
        let block = device
            .take_block(MemoryClass::Shared, MemoryClass::Shared.block_len())
            .expect("a Shared block");
        let released: Vec<_> = (0..3)
            .map(|_| {
                (device.take_block(MemoryClass::Shared, MemoryClass::Shared.block_len()))
                    .expect("a block")
            })
            .collect();
        let address = block.contents.expect("contents").as_ptr() as usize;
        *lock(&READER) = Some(thread::spawn(move || {
            while READING.load(Ordering::Acquire) {
                for offset in (0..MemoryClass::Shared.block_len() as usize).step_by(4096) {
                    // SAFETY: the block stays taken and mapped until the process ends.
                    unsafe { ((address + offset) as *const u8).read_volatile() };
                }
            }
        }));
        let event = device.new_shared_event().expect("an event");
        let queue = device.new_command_queue(64).expect("a queue");
        let held = queue.command_buffer().expect("a command buffer");
        held.encode_wait_for_event(event.as_event(), 1)
            .expect("a wait");
        held.commit().expect("a commit");
        for block in released {
            device.release_block(block).expect("a release");
        }
        // The block a reader thread reads stays taken until the process ends.
        std::mem::forget((event, queue, held, block));
        assert!(EXIT_DEVICE.set(device).is_ok());
        // `exit()` from a thread other than the one that made the device, with nothing released.
        let exiting = thread::spawn(|| {
            // SAFETY: the C library's exit, which runs the handlers registered with `atexit`.
            unsafe { exit(0) }
        });
        let _ = exiting.join();
    }

    /// In a process of its own, a consumer's `atexit` handler quiesces a live device with blocks
    /// taken, a reader thread on one, the block-maker busy and a command buffer held, then the
    /// process exits through `exit()` with no release: exit code 0 and no validation error. The
    /// handler is registered after `Device::new` (`M2V_EXIT_MODES` adds `before-instance`), once
    /// per mode (`M2V_EXIT_RUNS` runs each more often).
    #[test]
    fn every_device_quiesces_in_an_exit_handler() {
        let modes = std::env::var("M2V_EXIT_MODES").unwrap_or_else(|_| "after-device".into());
        let runs: u32 = std::env::var("M2V_EXIT_RUNS")
            .ok()
            .and_then(|runs| runs.parse().ok())
            .unwrap_or(1);
        let executable = std::env::current_exe().expect("the test executable");
        let mut failed = Vec::new();
        for mode in modes.split_whitespace() {
            let mut held = 0;
            for run in 0..runs {
                let output = std::process::Command::new(&executable)
                    .args([
                        "--exact",
                        "tests::command::on_device::exit_child",
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env("M2V_EXIT_CHILD", mode)
                    .output()
                    .expect("the child");
                let text = format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                let clean = output.status.success()
                    && text.contains("M2V exit: quiesced true true")
                    && !text.contains("Validation Error");
                if clean {
                    held += 1;
                } else {
                    println!("M2V exit {mode} run {run}: {:?}\n{text}", output.status);
                }
            }
            println!("M2V exit {mode}: {held} of {runs} runs clean");
            if held != runs {
                failed.push(mode.to_string());
            }
        }
        assert!(
            failed.is_empty(),
            "exit modes with a failed run: {failed:?}"
        );
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
            assert_eq!(empty.status(), CommandBufferStatus::Completed, "{name}");
            let alone = queue.command_buffer().expect("a command buffer");
            alone.commit().expect("a commit");
            finished(name, &alone);
            let (start, end) = (alone.gpu_start_time(), alone.gpu_end_time());
            assert_eq!(start, end, "{name}: equal times alone");
            assert!(start >= alone.kernel_start_time(), "{name}: alone {start}");
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

    /// The command buffers of a group share submit infos where no event wait or signal separates
    /// them, and record what they record alone. A mix of a fill whose segment's end barrier
    /// reaches commands, a Shared fill whose end barrier reaches the host, an event's signal and
    /// wait (which end and lead infos), and between them a command buffer with nothing for the GPU
    /// (no timestamps, no info) records the same barriers, passes and bytes copied and leaves the
    /// same words committed behind three busy submissions, held in groups, as committed one at a
    /// time. The groups pass two submit infos for the mix's four command buffers with commands:
    /// one for the fills up to the signal, which ends the group, and one the wait leads. One at a
    /// time passes one each. The busy work is held by the host until the submit thread has staged
    /// the whole mix: a completion seen sooner would send the held group with what it had staged
    /// by then, which splits it where the completion happened to arrive.
    #[test]
    fn every_device_shares_submit_infos_within_a_group_and_records_what_it_records_alone() {
        each_device("command_infos", |name, _, device| {
            let queue = queue(&device);
            let mut arena = Arena::new(&device);
            let (a, b) = (arena.buffer(), arena.buffer());
            let event = shared_event(&device);
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
                    .encode_signal_event(event.as_event(), signal)
                    .expect("a signal");
                commit(&signals);
                // After the signal, which sent the group: one committed while a group is held
                // would send it at once and end the group there.
                let empty = queue.command_buffer().expect("a command buffer");
                commit(&empty);
                let waits = queue.command_buffer().expect("a command buffer");
                waits
                    .encode_wait_for_event(event.as_event(), signal)
                    .expect("a wait");
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
