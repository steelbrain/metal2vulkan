//! Reading a command buffer's kernel times, status and error waits for nothing, on the device the
//! cfg selects, under the validation layer: Metal's `kernelStartTime`, `kernelEndTime`, `status`
//! and `error` reads wait for no GPU work and for no submission in progress.
//!
//! GPU busy work is committed first (`first_use::Busy`, released by the test), then on the same
//! queue one blit encoder of [`FILLS`] fills of distinct words, whose submission lies between its
//! kernel start and its kernel end. One thread reads the two kernel times over and over, from the
//! commit until [`BEHIND_BUSY`] after the submission, while the fills wait behind the busy work,
//! and another reads `status` and `error` over the same time, between reads of the kernel start
//! and of the kernel end. The submission may come only as the busy work drains: where no kernel
//! end was read [`HELD_READS`] after the commit, the reading thread releases the busy work, and
//! the reads go on until the submission, however late. The claim is structural: each thread's
//! reads land inside the submission, begun after its kernel start and returned before its kernel
//! end on the clock both are host times of (`host_time`), where a read that waited for the
//! submission returns only after its kernel end.
//!
//! And no read of the pair answers a kernel start and then no kernel end, inside the submission
//! or out of it: the start shows with the end alone, as Metal's does (none of the 7.0 to 8.5
//! million reads of each such command buffer on the M3 Max with macOS 27.0.1 answered a start and
//! no end). Nor while a command buffer waits for an event, its batches before the wait submitted:
//! both kernel times read 0 until the signal, then both show
//! ([`kernel_times_show_neither_while_their_command_buffer_waits_for_an_event`]).
//!
//! The submission takes what the device's scheduler takes, and no least span is required: both
//! kernel times are host times of the CPU's scheduling, as Metal's are, and Metal's own span for
//! this command buffer is short. On the M3 Max with macOS 27.0.1, Metal's kernel end came 0.04 to
//! 1.8 ms after its kernel start behind the busy work, and at most 12.7 ms after it for 2^20 fills
//! alone. Its kernel end was read within 2 ms of the commit while the busy work ran 100 to 300 ms
//! more. The runtime's submission takes 3 ms on the M3 Max (MoltenVK), and 0.05 to 0.11 ms on the
//! RTX 5080 Laptop GPU and on lavapipe, where 490 to 980 of a thread's reads land inside it. A
//! thread descheduled through so short a submission reads nothing inside it, so the test commits
//! up to [`ROUNDS`] such command buffers, each committed once the last one's work completed, until
//! each thread has had reads inside one of their submissions. The claim is each thread's, so the
//! two need not land in the same one: the two threads are seldom both on a CPU through so short a
//! submission (on lavapipe on the Ryzen 7 7840HS, in each of 8 rounds one thread's reads landed
//! inside and the other's did not, the two taking turns).
//!
//! And each read is timed: at most one in a thousand of each thread's takes [`SLOW_READ`] or more,
//! a bound a host that preempts the reading thread now and then keeps, which no longest read is on
//! a loaded Mac (`PROPERTY-READS`).
//!
//! And `status` reads `Completed` before anything else shows the completion, as Metal's does: in
//! the command buffer's completed handler, once its `wait_until_completed` returned, and in the
//! deallocator of a no-copy buffer it copied from, which the program let go of while the copy
//! waited behind busy work, and which goes once the command buffer lets go of what it held.

use super::first_use::{release, Busy};
use super::host_memory::{no_copy, not_importable, on_device, Mapping, GUEST_PAGE};
use metal2vulkan_runtime::{
    host_time, Buffer, CommandBuffer, CommandBufferStatus, Deallocator, Device, EventRef,
    ResourceOptions,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest a command buffer here may take.
const WAIT: Duration = Duration::from_secs(60);

/// The fills each command buffer encodes.
const FILLS: usize = 1 << 12;

/// The most command buffers committed for each thread to have reads inside one's submission.
const ROUNDS: usize = 8;

/// How long the reads go on once the command buffer is submitted, its work behind the busy work.
const BEHIND_BUSY: Duration = Duration::from_millis(100);

/// How long after the commit the reads go on with the busy work held where no kernel end was read
/// by then; the reading thread then releases it. On the Radeon 780M (RADV) beside the full suite,
/// the fills' kernel end came 1007.5 ms after their kernel start, once the test released the busy
/// work after a second of reads; alone, and as six copies at once, they were submitted 3 to 80 ms
/// after their commit.
const HELD_READS: Duration = Duration::from_millis(100);

/// A slow read. On the M3 Max beside the full suite, a reading thread's longest reads took 20 to
/// 54 ms each round, of over a million reads, the thread descheduled (macOS counts no wait for a
/// CPU to take off).
const SLOW_READ: Duration = Duration::from_millis(10);

/// The most recent reads a thread keeps the host times of until a kernel end is first read, among
/// which lie those inside the submission, which ended before its end showed: 2^16 reads (1 MiB of
/// host times), more than a thread makes inside the runtime's submission of [`FILLS`] fills on the
/// devices measured (the module docs). Of a longer submission, the reads kept are its last ones.
const RECENT: usize = 1 << 16;

/// How long a command buffer waiting for an event is read before the signal.
const WAITING: Duration = Duration::from_millis(50);

/// What a reading thread saw.
#[derive(Debug, Default)]
struct Reads {
    /// Its reads: of the two kernel times, or of `status` and `error`, each pair one read.
    count: usize,
    /// The longest read of one property.
    longest: Duration,
    /// Reads of one property that took [`SLOW_READ`] or more.
    slow: usize,
    /// Reads begun after the kernel start and returned before the kernel end: inside the
    /// submission.
    inside: usize,
    /// Reads that answered a kernel start and then no kernel end: none, as Metal's.
    start_alone: usize,
    /// When, after the commit, a read first answered a kernel end (the kernel times' thread).
    submitted: Option<Duration>,
    /// The kernel times' thread released the busy work, no kernel end read [`HELD_READS`] after
    /// the commit.
    released: bool,
}

impl Reads {
    /// Count one read whose two properties took `first` and `second`, with the kernel times
    /// read around it, the start first.
    fn add(&mut self, (first, second): (Duration, Duration), (start, end): (f64, f64)) {
        self.count += 1;
        self.longest = self.longest.max(first).max(second);
        self.slow += usize::from(first >= SLOW_READ) + usize::from(second >= SLOW_READ);
        self.start_alone += usize::from(start != 0.0 && end == 0.0);
    }

    /// Whether at most one in a thousand of its reads was slow.
    fn rarely_slow(&self) -> bool {
        self.slow * 1000 <= self.count
    }
}

/// The host times ([`host_time`]) of a thread's most recent reads, at most [`RECENT`], each from
/// before it began to after it returned, kept until a kernel end is first read.
#[derive(Default)]
struct Recent(VecDeque<(f64, f64)>);

impl Recent {
    /// Keep a read made from `began` to `returned`.
    fn push(&mut self, began: f64, returned: f64) {
        if self.0.len() == RECENT {
            self.0.pop_front();
        }
        self.0.push_back((began, returned));
    }

    /// The reads kept that lie inside `command_buffer`'s submission, whose kernel end is known:
    /// begun at or after its kernel start and returned at or before its kernel end.
    fn inside(&self, command_buffer: &CommandBuffer) -> usize {
        let (start, end) = (
            command_buffer.kernel_start_time(),
            command_buffer.kernel_end_time(),
        );
        assert!(
            0.0 < start && start <= end,
            "kernel times {start}..{end} once the end is known"
        );
        (self.0.iter())
            .filter(|&&(began, returned)| start <= began && returned <= end)
            .count()
    }
}

/// Read `command_buffer`'s kernel times over and over from `committed` on, until [`BEHIND_BUSY`]
/// after a read first answered its kernel end, or [`WAIT`] after `committed` where none did;
/// releasing the busy work of `words` where none did [`HELD_READS`] after `committed`.
fn read(command_buffer: &CommandBuffer, committed: Instant, words: &Buffer) -> Reads {
    let mut reads = Reads::default();
    let mut recent = Recent::default();
    loop {
        let began = host_time();
        let before = Instant::now();
        let start = command_buffer.kernel_start_time();
        let between = Instant::now();
        let end = command_buffer.kernel_end_time();
        let after = Instant::now();
        let returned = host_time();
        reads.add((between - before, after - between), (start, end));
        let since = after.duration_since(committed);
        if reads.submitted.is_none() {
            recent.push(began, returned);
        }
        match reads.submitted {
            None if end != 0.0 => {
                reads.submitted = Some(since);
                reads.inside = recent.inside(command_buffer);
            }
            None if since >= WAIT => return reads,
            None if since >= HELD_READS && !reads.released => {
                release(words);
                reads.released = true;
            }
            Some(submitted) if since >= submitted + BEHIND_BUSY => return reads,
            _ => {}
        }
    }
}

/// Read `command_buffer`'s `status` and `error` over and over, each pair between a read of the
/// kernel start and one of the kernel end, until `done`, or until [`WAIT`] and [`BEHIND_BUSY`]
/// after `committed`.
fn read_status(command_buffer: &CommandBuffer, committed: Instant, done: &AtomicBool) -> Reads {
    let mut reads = Reads::default();
    let mut recent = Recent::default();
    let mut ended = false;
    while !done.load(Ordering::Acquire) && committed.elapsed() < WAIT + BEHIND_BUSY {
        let start = command_buffer.kernel_start_time();
        let began = host_time();
        let before = Instant::now();
        let _ = command_buffer.status();
        let between = Instant::now();
        let _ = command_buffer.error();
        let after = Instant::now();
        let returned = host_time();
        let end = command_buffer.kernel_end_time();
        reads.add((between - before, after - between), (start, end));
        match (ended, end != 0.0) {
            (false, false) => recent.push(began, returned),
            (false, true) => {
                ended = true;
                reads.inside = recent.inside(command_buffer);
            }
            (true, _) => {}
        }
    }
    reads
}

/// The batches `device` submitted, each held in a group for one kernel submission counted as it
/// is held (`WorkCounts::batches_with_none_pending` and the three fields after it): a sign that a
/// command buffer's first batch was submitted that, unlike its kernel start, shows before its last
/// batch is.
pub(super) fn batches_submitted(device: &Device) -> u64 {
    let counts = device.work_counts();
    counts.batches_with_none_pending
        + counts.batches_with_one_pending
        + counts.batches_with_two_or_three_pending
        + counts.batches_with_four_or_more_pending
}

/// Wait until `device` submitted a batch more than the `before` [`batches_submitted`] read: the
/// first batch of the one command buffer committed since.
pub(super) fn first_batch_submitted(device: &Device, before: u64) {
    let deadline = Instant::now() + WAIT;
    while batches_submitted(device) == before {
        assert!(
            Instant::now() < deadline,
            "the first batch was never submitted"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The module docs: kernel times, status and error read while the command buffer's submission
/// replays its fills and while its work waits behind busy work: some of each inside the
/// submission of one of at most [`ROUNDS`] command buffers, and at most one in a thousand of each
/// command buffer's slow.
#[test]
fn property_reads_wait_for_no_submission_and_no_busy_work() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let name = device.name();
        let busy = Busy::new(device, &queue);
        let mut inside = Vec::new();
        for round in 1..=ROUNDS {
            let target = (device.new_buffer(4 * FILLS as u64, ResourceOptions::private())).unwrap();
            let (work, words) = busy.commit(device, &queue);
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            for at in 0..FILLS as u64 {
                blit.fill_buffer(&target, 4 * at..4 * at + 4, at as u8)
                    .unwrap();
            }
            blit.end_encoding().unwrap();
            let committed = Instant::now();
            let done = AtomicBool::new(false);
            let (reads, status) = std::thread::scope(|scope| {
                let reader = scope.spawn(|| {
                    let reads = read(&command_buffer, committed, &words);
                    done.store(true, Ordering::Release);
                    reads
                });
                let status = scope.spawn(|| read_status(&command_buffer, committed, &done));
                command_buffer.commit().unwrap();
                (reader.join().unwrap(), status.join().unwrap())
            });
            release(&words);
            work.wait_until_completed_for(WAIT).unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(work.error(), None, "the busy work");
            assert_eq!(command_buffer.error(), None, "the fills");
            let span = Duration::from_secs_f64(
                command_buffer.kernel_end_time() - command_buffer.kernel_start_time(),
            );
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            eprintln!(
                "PROPERTY-READS {name}: round {round}, kernel span {:.3} ms, submitted {:?} after \
                 the commit (the busy work released first: {}); {} kernel time reads, {} inside \
                 the submission, {} slow, the longest {:.3} ms, {} a start alone; {} status and \
                 error reads, {} inside, {} slow, the longest {:.3} ms, {} between a start alone",
                ms(span),
                reads.submitted,
                reads.released,
                reads.count,
                reads.inside,
                reads.slow,
                ms(reads.longest),
                reads.start_alone,
                status.count,
                status.inside,
                status.slow,
                ms(status.longest),
                status.start_alone,
            );
            assert!(
                reads.submitted.is_some(),
                "the fills were not submitted within {WAIT:?} of their commit: {reads:?}"
            );
            assert!(
                reads.rarely_slow(),
                "more than one kernel time read in a thousand took {SLOW_READ:?} or more: {reads:?}"
            );
            assert!(
                status.rarely_slow(),
                "more than one status or error read in a thousand took {SLOW_READ:?} or more: \
                 {status:?}"
            );
            assert_eq!(
                (reads.start_alone, status.start_alone),
                (0, 0),
                "kernel starts read with no kernel end: {reads:?}, {status:?}"
            );
            inside.push((reads.inside, status.inside));
            if inside.iter().any(|&(reads, _)| reads > 0)
                && inside.iter().any(|&(_, status)| status > 0)
            {
                return;
            }
        }
        panic!(
            "in {ROUNDS} command buffers the reads of a thread never landed inside a submission \
             (each one's kernel time reads and status and error reads inside): {inside:?}"
        );
    });
}

/// The module docs: a command buffer that waits for an event after a blit encoder answers neither
/// kernel time while it waits, though its batch before the wait was submitted: for [`WAITING`],
/// every read of the two answers 0, and it is `Committed`. Once the event is signalled, both show
/// together, no read answering a start and then no end, the start before the signal and the end
/// after it. Metal answers so for an `MTLSharedEvent` the host signals and an `MTLEvent` another
/// queue's command buffer signals: on the M3 Max under macOS 27.0.1, 4.6 to 4.7 million reads of
/// each over 200 ms answered both 0, and both showed 50 to 160 us after the signal, the start from
/// before the wait.
#[test]
fn kernel_times_show_neither_while_their_command_buffer_waits_for_an_event() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let other = device.new_command_queue().unwrap();
        let name = device.name();
        let target = device.new_buffer(16, ResourceOptions::private()).unwrap();
        let fill = |command_buffer: &CommandBuffer, value: u8| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&target, 0..16, value).unwrap();
            blit.end_encoding().unwrap();
        };
        for shared in [true, false] {
            let (shared_event, event) = (
                device.new_shared_event().unwrap(),
                device.new_event().unwrap(),
            );
            let waited = match shared {
                true => EventRef::Shared(&shared_event),
                false => EventRef::Event(&event),
            };
            let command_buffer = queue.command_buffer().unwrap();
            fill(&command_buffer, 1);
            command_buffer.encode_wait_for_event(waited, 1).unwrap();
            fill(&command_buffer, 2);
            let before = batches_submitted(device);
            command_buffer.commit().unwrap();
            first_batch_submitted(device, before);
            let submitted = batches_submitted(device) - before;
            let (mut reads, mut shown) = (0usize, 0usize);
            let until = Instant::now() + WAITING;
            while Instant::now() < until {
                let start = command_buffer.kernel_start_time();
                let end = command_buffer.kernel_end_time();
                reads += 1;
                shown += usize::from(start != 0.0 || end != 0.0);
            }
            let waiting = command_buffer.status();
            let signalled = host_time();
            let signal = match shared {
                true => {
                    shared_event.set_signaled_value(1);
                    None
                }
                false => {
                    let signal = other.command_buffer().unwrap();
                    signal
                        .encode_signal_event(EventRef::Event(&event), 1)
                        .unwrap();
                    signal.commit().unwrap();
                    Some(signal)
                }
            };
            let (mut after, mut start_alone) = (0usize, 0usize);
            let deadline = Instant::now() + WAIT;
            loop {
                let start = command_buffer.kernel_start_time();
                let end = command_buffer.kernel_end_time();
                after += 1;
                start_alone += usize::from(start != 0.0 && end == 0.0);
                if end != 0.0 || Instant::now() >= deadline {
                    break;
                }
            }
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            if let Some(signal) = signal {
                signal.wait_until_completed_for(WAIT).unwrap();
            }
            let (start, end) = (
                command_buffer.kernel_start_time(),
                command_buffer.kernel_end_time(),
            );
            eprintln!(
                "PROPERTY-READS {name}: waiting for {}: {submitted} batches submitted, {reads} \
                 reads over {WAITING:?} while {waiting:?}, {shown} showing a kernel time; {after} \
                 reads after the signal, {start_alone} a start alone; kernel start {:.3} ms \
                 before the signal, end {:.3} ms after it",
                if shared {
                    "an MTLSharedEvent"
                } else {
                    "an MTLEvent"
                },
                (signalled - start) * 1e3,
                (end - signalled) * 1e3,
            );
            assert!(
                submitted >= 1,
                "the batch before the wait was not submitted"
            );
            assert_eq!(waiting, CommandBufferStatus::Committed, "while it waits");
            assert_eq!(shown, 0, "kernel times shown while it waits");
            assert_eq!(start_alone, 0, "a kernel start read with no kernel end");
            assert_eq!(command_buffer.error(), None);
            assert!(
                0.0 < start && start < signalled && signalled < end,
                "kernel times {start}..{end}, the signal at {signalled}"
            );
        }
    });
}

/// The module docs: `status` reads `Completed` in the completed handler, once the wait returned,
/// and in the deallocator of a buffer the command buffer held after the program let go of it.
#[test]
fn status_reads_completed_before_anything_shows_completion() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let name = device.name();
        let busy = Busy::new(device, &queue);
        let command_buffer = queue.command_buffer().unwrap();
        let mapping = Mapping::anonymous(GUEST_PAGE);
        let pointer = mapping.pointer();
        let in_deallocator = Arc::new(Mutex::new(None));
        let deallocator: Deallocator = {
            let (seen, reader) = (Arc::clone(&in_deallocator), command_buffer.clone());
            Box::new(move |_, _| {
                *seen.lock().unwrap() = Some(reader.status());
                drop(mapping);
            })
        };
        let options = ResourceOptions::shared();
        // SAFETY: the mapping's pages stay mapped until the deallocator drops it.
        let made = unsafe { no_copy(device, pointer, GUEST_PAGE, options, Some(deallocator)) };
        let source = match made {
            Ok(buffer) => buffer,
            Err(error) if not_importable(&error) => {
                eprintln!("PROPERTY-READS {name}: no buffer over host memory: {error:?}");
                return;
            }
            Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
        };
        let target = device.new_buffer(16, options).unwrap();
        let in_handler = Arc::new(Mutex::new(None));
        let seen = Arc::clone(&in_handler);
        (command_buffer.add_completed_handler(Box::new(move |command_buffer| {
            *seen.lock().unwrap() = Some(command_buffer.status());
        })))
        .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&source, 0, &target, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        let (work, words) = busy.commit(device, &queue);
        command_buffer.commit().unwrap();
        // The program's last handle goes while the copy waits behind the busy work.
        drop(source);
        let early = *in_deallocator.lock().unwrap();
        release(&words);
        work.wait_until_completed_for(WAIT).unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let after_wait = command_buffer.status();
        let handler = *in_handler.lock().unwrap();
        let deadline = Instant::now() + WAIT;
        let deallocator = loop {
            if let Some(status) = *in_deallocator.lock().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "the deallocator never ran");
            std::thread::sleep(Duration::from_millis(1));
        };
        eprintln!(
            "PROPERTY-READS {name}: status after the wait {after_wait:?}, in the completed \
             handler {handler:?}, in the deallocator {deallocator:?}"
        );
        assert_eq!(work.error(), None, "the busy work");
        assert_eq!(command_buffer.error(), None, "the copy");
        assert_eq!(
            early, None,
            "the buffer went while its command buffer held it"
        );
        assert_eq!(
            after_wait,
            CommandBufferStatus::Completed,
            "once the wait returned"
        );
        assert_eq!(
            handler,
            Some(CommandBufferStatus::Completed),
            "in the completed handler"
        );
        assert_eq!(
            deallocator,
            CommandBufferStatus::Completed,
            "in the deallocator"
        );
    });
}
