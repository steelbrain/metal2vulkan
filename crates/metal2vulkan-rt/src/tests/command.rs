//! Command queues, command buffers and events, under a device cfg on every device of the cfg's
//! kind: states, handlers and refusals; holds per segment; the host floor; host values racing GPU
//! signals; listeners; abandon; quiesce, also in an `exit()` handler; teardown from a handler and
//! with work held; refusals of what has started; a loss; and a released block kept until its last
//! command buffer retires.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
pub(crate) mod on_device {
    use crate::device::lock;
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        host_time, BlockId, BlockMappings, Buffer, CommandBuffer, CommandBufferError,
        CommandBufferStatus, CommandQueue, Device, DeviceOptions, DispatchType, Error, Gauges,
        Instance, MemoryClass, Refusal, SharedEvent, BLOCK_LEN,
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
        block: BlockId,
        next: u64,
    }

    impl Arena {
        pub(crate) fn new(device: &Device) -> Arena {
            let block = device
                .take_block(MemoryClass::Shared, 1)
                .expect("a Shared block");
            Arena {
                device: device.clone(),
                block: block.id,
                next: 0,
            }
        }

        pub(crate) fn buffer(&mut self) -> Buffer {
            let buffer = self
                .device
                .new_buffer(self.block, self.next, 256, true)
                .expect("a buffer");
            self.next += 256;
            buffer
        }
    }

    /// Records a fill of `buffer` with `word`, keeping its block.
    fn fill(command_buffer: &CommandBuffer, buffer: &Buffer, word: u32) {
        let target = buffer.held_block().memory.buffer;
        let (offset, size) = (buffer.offset(), buffer.length());
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
            device.release_block(block.id).expect("a release");
            assert!(
                matches!(device.release_block(block.id), Err(Error::NoSuchBlock(id)) if id == block.id),
                "{name}: release_block"
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

    /// Times, on the host clock: the kernel start at commit precedes the kernel end at submission;
    /// each GPU start precedes its end, which falls between the first commit and the last
    /// completion; a queue's GPU end times never decrease in completion order, through slots its
    /// command buffers reuse, with or without commands.
    #[test]
    fn every_device_times_command_buffers_on_the_host_clock() {
        each_device("command_times", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            // Fewer slots than command buffers, so slots and their queries are reused.
            let queue = device.new_command_queue(4).expect("a queue");
            let ends: Arc<Mutex<Vec<(usize, u64, u64)>>> = Arc::default();
            let first = host_time();
            let mut committed = Vec::new();
            for index in 0..24 {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                // Every third has no commands.
                if index % 3 != 2 {
                    fill(&command_buffer, &buffer, index as u32);
                }
                let ends = Arc::clone(&ends);
                command_buffer
                    .add_completed_handler(move |cb| {
                        lock(&ends).push((index, cb.gpu_start_time(), cb.gpu_end_time()))
                    })
                    .expect("a completed handler");
                command_buffer.commit().expect("a commit");
                committed.push(command_buffer);
            }
            for command_buffer in &committed {
                finished(name, command_buffer);
            }
            let last = host_time();
            let ends = lock(&ends).clone();
            let order: Vec<usize> = ends.iter().map(|(index, _, _)| *index).collect();
            assert_eq!(
                order,
                (0..24).collect::<Vec<_>>(),
                "{name}: completion order"
            );
            for pair in ends.windows(2) {
                assert!(pair[0].2 <= pair[1].2, "{name}: ends decrease: {pair:?}");
            }
            for ((_, start, end), command_buffer) in ends.iter().zip(&committed) {
                let (kernel_start, kernel_end) = (
                    command_buffer.kernel_start_time(),
                    command_buffer.kernel_end_time(),
                );
                assert!(
                    first <= kernel_start && kernel_start <= kernel_end,
                    "{name}"
                );
                assert!(0 < *start && start <= end && *end <= last, "{name}");
                assert!(first <= *end, "{name}");
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

    /// A burst of 64 command buffers committed back to back with no wait: each submission ends at
    /// most four of them, and a batch kept while two submissions run goes once they complete, with
    /// no host wait to send it.
    #[test]
    fn every_device_runs_a_burst_in_submissions_of_four_at_most() {
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
                submissions >= u64::from(count.div_ceil(4)),
                "{name}: {submissions}"
            );
        });
    }

    /// A Private buffer of a whole block, for work that keeps the GPU busy.
    fn private(device: &Device) -> Buffer {
        let block = device
            .take_block(MemoryClass::Private, 1)
            .expect("a Private block");
        device
            .new_buffer(block.id, 0, BLOCK_LEN, false)
            .expect("a buffer")
    }

    /// Commits a command buffer of `fills` fills of `buffer`: work that runs for a while.
    fn busy(queue: &CommandQueue, buffer: &Buffer, fills: u32) -> CommandBuffer {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let target = buffer.held_block().memory.buffer;
        let (offset, size) = (buffer.offset(), buffer.length());
        command_buffer
            .record_for_test(&[buffer], |device, command| {
                for word in 0..fills {
                    // SAFETY: the command buffer is recording; the range is the buffer's, in
                    // whole words.
                    unsafe { device.cmd_fill_buffer(command, target, offset, size, word) };
                }
            })
            .expect("fills");
        command_buffer.commit().expect("a commit");
        command_buffer
    }

    /// Runs `body` with two command buffers of busy work committed on `queue` and so in flight,
    /// each of more fills each time, until `body` answers that it checked what it holds while
    /// they ran: that the work did not end first.
    fn behind_two(
        name: &str,
        device: &Device,
        queue: &CommandQueue,
        mut body: impl FnMut(&[CommandBuffer; 2], u64) -> bool,
    ) {
        let buffer = private(device);
        let checked = [16, 64, 256, 1024].into_iter().any(|fills| {
            let before = device.counters().metrics().submissions;
            let running = [busy(queue, &buffer, fills), busy(queue, &buffer, fills)];
            let checked = body(&running, before);
            for command_buffer in &running {
                finished(name, command_buffer);
            }
            checked
        });
        assert!(checked, "{name}: the busy work ended before every check");
    }

    /// A group is held only while two of its queue's submissions are not seen complete, and the
    /// completion that leaves fewer sends it: a command buffer committed behind two running ones,
    /// with no later commit and no host wait, runs once the first of them completes.
    #[test]
    fn every_device_sends_a_held_group_when_a_submission_completes() {
        each_device("command_release", |name, _, device| {
            let mut arena = Arena::new(&device);
            let buffer = arena.buffer();
            let queue = queue(&device);
            behind_two(name, &device, &queue, |_, before| {
                let held = queue.command_buffer().expect("a command buffer");
                fill(&held, &buffer, 7);
                held.commit().expect("a commit");
                within(name, || held.status() != CommandBufferStatus::Committed);
                let submissions = device.counters().metrics().submissions - before;
                within(name, || held.status() == CommandBufferStatus::Completed);
                assert_eq!(word(&buffer), 7, "{name}");
                let after = device.counters().metrics().submissions - before;
                assert_eq!(after, 3, "{name}: one submission each");
                // Scheduled while the two ran: it was held, then sent by a completion.
                submissions == 2
            });
            drop((buffer, arena, queue, device));
        });
    }

    /// The command buffers of one submission retire in commit order, each with its own status
    /// and GPU times: a group held behind two running submissions takes a command buffer, one
    /// ended by an abandon while enqueued and committed after it, and two more; they complete in
    /// that order, the ended one `NotRun` with no times. The other three, each the same work, run
    /// back to back: each starts no earlier than the one before ends, where one shared end less
    /// each one's duration would overlap them.
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
                let target = work.held_block().memory.buffer;
                command_buffer
                    .record_for_test(&[&work], |device, command| {
                        // SAFETY: as `busy`'s.
                        unsafe { device.cmd_fill_buffer(command, target, 0, BLOCK_LEN, word) };
                    })
                    .expect("a fill");
                log.handlers(label, &command_buffer);
                command_buffer.commit().expect("a commit");
                command_buffer
            };
            behind_two(name, &device, &queue, |_, before| {
                let log = Log::default();
                let first = member("1", &log, &buffers[0], 1);
                let ended = queue.command_buffer().expect("a command buffer");
                log.handlers("2", &ended);
                ended.enqueue().expect("an enqueue");
                queue.abandon();
                ended.commit().expect("a commit");
                let rest = [
                    member("3", &log, &buffers[1], 3),
                    member("4", &log, &buffers[2], 4),
                ];
                within(name, || rest[1].status() != CommandBufferStatus::Committed);
                let submissions = device.counters().metrics().submissions - before;
                for command_buffer in [&first, &ended].into_iter().chain(&rest) {
                    finished(name, command_buffer);
                }
                let completed: Vec<String> = (log.lines().into_iter())
                    .filter(|line| line.contains("completed"))
                    .collect();
                assert_eq!(
                    completed,
                    [
                        "1 completed Completed None",
                        "2 completed Error Some(NotRun)",
                        "3 completed Completed None",
                        "4 completed Completed None",
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
                    assert!(0 < start && start < end, "{name}: {times:?}");
                }
                for pair in times.windows(2) {
                    assert!(
                        pair[0].1 <= pair[1].0,
                        "{name}: each starts no earlier than the one before ends: {times:?}"
                    );
                }
                if submissions == 2 {
                    // All three were held while the two ran, so they went as one submission.
                    let after = device.counters().metrics().submissions - before;
                    assert_eq!(after, 3, "{name}: one group");
                }
                submissions == 2
            });
            drop((buffers, work, arena, queue, device));
        });
    }

    /// `send_now`, the wait hint, from another thread: a command buffer held in a group behind two
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
            behind_two(name, &device, &queue, |running, before| {
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
                let checked =
                    submissions == 2 && running[1].status() != CommandBufferStatus::Completed;
                within(name, || {
                    device.counters().metrics().submissions - before == 3
                });
                // The hint for one submitted makes no submission.
                queue.send_now(&running[1]);
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
                    3,
                    "{name}: a stale hint makes no submission"
                );
                drop(uncommitted);
                checked
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
            let mut handed = vec![(arena.block, None), (private.id, None)];
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
                device.new_buffer(private.id, 0, 64, true),
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
                .new_buffer(block.id, 0, 4096, false)
                .expect("a buffer");
            let f = shared_event(&device);
            let queue = queue(&device);
            let held = queue.command_buffer().expect("a command buffer");
            fill(&held, &buffer, 5);
            held.encode_wait_for_event(f.as_event(), 1).expect("a wait");
            held.commit().expect("a commit");
            drop(buffer);
            device.release_block(block.id).expect("a release");
            thread::sleep(Duration::from_millis(50));
            let pool = &device.core().pool;
            assert!(!pool.holds(block.id), "{name}: back while in use");
            f.set_signaled_value(1);
            finished(name, &held);
            within("the block's return", || pool.holds(block.id));
            drop((f, queue, device));
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
        // Blocks stay taken; a reader thread reads one; the block-maker zeroes released ones; a
        // command buffer is held.
        let block = device
            .take_block(MemoryClass::Shared, BLOCK_LEN)
            .expect("a Shared block");
        let released: Vec<_> = (0..3)
            .map(|_| device.take_block(MemoryClass::Shared, 1).expect("a block"))
            .collect();
        let address = block.contents.expect("contents").as_ptr() as usize;
        *lock(&READER) = Some(thread::spawn(move || {
            while READING.load(Ordering::Acquire) {
                for offset in (0..BLOCK_LEN as usize).step_by(4096) {
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
        for block in &released {
            device.release_block(block.id).expect("a release");
        }
        std::mem::forget((event, queue, held));
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
}
