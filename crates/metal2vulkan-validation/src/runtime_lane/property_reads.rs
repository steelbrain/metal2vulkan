//! Reading a command buffer's kernel times, status and error waits for nothing, on the device the
//! cfg selects, under the validation layer: Metal's `kernelStartTime`, `kernelEndTime`, `status`
//! and `error` reads wait for no GPU work and for no submission in progress.
//!
//! GPU busy work is committed first (`first_use::Busy`, released by the test), then on the same
//! queue one blit encoder of fills of distinct words, whose submission lies between its kernel
//! start and its kernel end; its fills are doubled until
//! that span is at least [`LONG_SUBMISSION`]. One thread reads the two kernel times over and over,
//! from the commit until [`BEHIND_BUSY`] after the submission, while the fills wait behind the busy
//! work, and another reads `status` and `error` over the same time, each between a read of the
//! kernel start and a read of the kernel end. The submission may come only as the busy work drains:
//! where no kernel end was read [`HELD_READS`] after the commit, the reading thread releases the
//! busy work, and the reads go on until the submission, however late. The claim is structural: each
//! thread's reads land inside the submission, some answering between a kernel start and no kernel
//! end yet, where a read that waited for the submission returns only after its kernel end. And each
//! read is timed: at most one in a thousand of each thread's takes [`SLOW_READ`] or more, a bound a
//! host that preempts the reading thread now and then keeps, which no longest read is on a loaded
//! Mac (`PROPERTY-READS`).
//!
//! And `status` reads `Completed` before anything else shows the completion, as Metal's does: in
//! the command buffer's completed handler, once its `wait_until_completed` returned, and in the
//! deallocator of a no-copy buffer it copied from, which the program let go of while the copy
//! waited behind busy work, and which goes once the command buffer lets go of what it held.

use super::first_use::{release, Busy};
use super::host_memory::{no_copy, not_importable, on_device, Mapping, GUEST_PAGE};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, Deallocator, ResourceOptions,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest a command buffer here may take.
const WAIT: Duration = Duration::from_secs(60);

/// The fills the command buffer encodes at first, doubled until its kernel span is at least
/// [`LONG_SUBMISSION`], up to [`MOST_FILLS`].
const FILLS: usize = 1 << 12;
const MOST_FILLS: usize = 1 << 20;

/// The least kernel span (`kernelEndTime` less `kernelStartTime`) of the command buffer read.
const LONG_SUBMISSION: Duration = Duration::from_millis(40);

/// How long the reads go on once the command buffer is submitted, its work behind the busy work.
const BEHIND_BUSY: Duration = Duration::from_millis(100);

/// How long after the commit the reads go on with the busy work held where no kernel end was read
/// by then; the reading thread then releases it. On the Radeon 780M (RADV) beside the full suite,
/// the fills' kernel end came 1007.5 ms after their kernel start, once the test released the busy
/// work after a second of reads; alone, and as six copies at once, they were submitted 3 to 80 ms
/// after their commit.
const HELD_READS: Duration = Duration::from_millis(100);

/// A slow read: a quarter of [`LONG_SUBMISSION`]. On the M3 Max beside the full suite, a reading
/// thread's longest reads took 20 to 54 ms each round, of over a million reads, the thread
/// descheduled (macOS counts no wait for a CPU to take off); its reads inside the submission
/// numbered 100,000 to 480,000.
const SLOW_READ: Duration = Duration::from_millis(10);

/// What a reading thread saw.
#[derive(Debug, Default)]
struct Reads {
    /// Its reads: of the two kernel times, or of `status` and `error`, each pair one read.
    count: usize,
    /// The longest read of one property.
    longest: Duration,
    /// Reads of one property that took [`SLOW_READ`] or more.
    slow: usize,
    /// Reads made after a kernel start was read and before no kernel end was read: inside the
    /// submission.
    inside: usize,
    /// When, after the commit, a read first answered a kernel end (the kernel times' thread).
    submitted: Option<Duration>,
    /// The kernel times' thread released the busy work, no kernel end read [`HELD_READS`] after
    /// the commit.
    released: bool,
}

impl Reads {
    /// Count one read whose two properties took `first` and `second`, inside the submission or
    /// not.
    fn add(&mut self, (first, second): (Duration, Duration), inside: bool) {
        self.count += 1;
        self.longest = self.longest.max(first).max(second);
        self.slow += usize::from(first >= SLOW_READ) + usize::from(second >= SLOW_READ);
        self.inside += usize::from(inside);
    }

    /// Whether at most one in a thousand of its reads was slow.
    fn rarely_slow(&self) -> bool {
        self.slow * 1000 <= self.count
    }
}

/// Read `command_buffer`'s kernel times over and over from `committed` on, until [`BEHIND_BUSY`]
/// after a read first answered its kernel end, or [`WAIT`] after `committed` where none did;
/// releasing the busy work of `words` where none did [`HELD_READS`] after `committed`.
fn read(command_buffer: &CommandBuffer, committed: Instant, words: &Buffer) -> Reads {
    let mut reads = Reads::default();
    loop {
        let before = Instant::now();
        let start = command_buffer.kernel_start_time();
        let between = Instant::now();
        let end = command_buffer.kernel_end_time();
        let after = Instant::now();
        reads.add(
            (between - before, after - between),
            start != 0.0 && end == 0.0,
        );
        let since = after.duration_since(committed);
        match reads.submitted {
            None if end != 0.0 => reads.submitted = Some(since),
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
    while !done.load(Ordering::Acquire) && committed.elapsed() < WAIT + BEHIND_BUSY {
        let start = command_buffer.kernel_start_time();
        let before = Instant::now();
        let _ = command_buffer.status();
        let between = Instant::now();
        let _ = command_buffer.error();
        let after = Instant::now();
        let end = command_buffer.kernel_end_time();
        reads.add(
            (between - before, after - between),
            start != 0.0 && end == 0.0,
        );
    }
    reads
}

/// The module docs: kernel times, status and error read while the command buffer's submission
/// replays its fills and while its work waits behind busy work: some of each inside the
/// submission, and at most one in a thousand slow.
#[test]
fn property_reads_wait_for_no_submission_and_no_busy_work() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let name = device.name();
        let busy = Busy::new(device, &queue);
        let mut fills = FILLS;
        loop {
            let target = (device.new_buffer(4 * fills as u64, ResourceOptions::private())).unwrap();
            let (work, words) = busy.commit(device, &queue);
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            for at in 0..fills as u64 {
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
                "PROPERTY-READS {name}: {fills} fills, kernel span {:.1} ms, submitted {:?} after \
                 the commit (the busy work released first: {}); {} kernel time reads, {} inside \
                 the submission, {} slow, the longest {:.3} ms; {} status and error reads, {} \
                 inside, {} slow, the longest {:.3} ms",
                ms(span),
                reads.submitted,
                reads.released,
                reads.count,
                reads.inside,
                reads.slow,
                ms(reads.longest),
                status.count,
                status.inside,
                status.slow,
                ms(status.longest),
            );
            assert!(
                reads.submitted.is_some(),
                "the fills were not submitted within {WAIT:?} of their commit: {reads:?}"
            );
            if span < LONG_SUBMISSION && fills < MOST_FILLS {
                fills *= 2;
                continue;
            }
            assert!(
                span >= LONG_SUBMISSION,
                "{MOST_FILLS} fills spanned {span:?} from their kernel start to their kernel end, \
                 short of {LONG_SUBMISSION:?}"
            );
            assert!(
                reads.inside > 0,
                "no kernel time read landed inside the submission: {reads:?}"
            );
            assert!(
                status.inside > 0,
                "no status or error read landed inside the submission: {status:?}"
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
            break;
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
