//! Device tests of submission checks (`CommandBuffer::set_submission_check`, a runtime
//! extension) on the device the cfg selects, under the validation layer.
//!
//! The consumer emulates Apple's detach: Metal fails a submission (as it splits the command
//! buffer) that uses a resource whose memory was taken away by the time it is submitted, with
//! code 8, and runs every other (measured by the consumer on an M3 Max, macOS 26.7). Its check
//! fails a submission whose used resources hold a detached object, by `ObjectId`.
//!
//! - **Detached behind a wait.** Each probe shape (`failed_commits::SHAPES`), led by an event
//!   wait, is committed with its resource attached, then detached, then the wait is signalled:
//!   the submission that uses it fails alone, as `commit_failed` with that set fails it. The check
//!   sees each of Metal's submissions once.
//! - **Attached again behind a wait.** The same shapes committed while detached, attached again
//!   before the signal: everything runs, and the command buffer completes.
//! - **Reached through an indirect command alone.** A buffer an indirect command binds, which
//!   no encoder makes resident, is the submission's from the commit on; detached behind a wait,
//!   that submission fails.
//! - **A GPU signal in flight**, held on the GPU at the signaller's gate: a wait for a plain
//!   event's signal the test reserved (`Event::reserve_vk_signal`) and gives from the host once it
//!   looked, so the window does not hang on the GPU's speed or the host's load. A submission whose
//!   `SharedEvent` wait another command buffer's GPU signal meets is checked only once that signal
//!   landed (the check reads a companion `SharedEvent` the signaller signals first); detached while
//!   the signaller runs, it fails, whether committed before the signaller or with
//!   `commit_and_submit` while its signal is in flight (the committing thread leaves that check to
//!   the scheduler's). One whose `Event` wait such a signal meets is checked at once and submitted
//!   with its wait on the `VkQueue`, while the signal still runs (the companion reads 0 in its
//!   check), as Metal submits it (`event-wait-behind-a-running-signal`), no head counted as
//!   awaiting its checks' signals. A `SharedEvent` the host signals while a GPU signal of it is in
//!   flight is checked without waiting for the GPU.

use super::degrade_notices::on_device;
use super::failed_commits::{Shape, HELD, KERNEL, SHAPES};
use super::{
    bound_buffers, buffer_with, compute_descriptor, pipeline_state, public_case, read_buffer,
    CaseResources, PublicCase, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::ComputeOutput;
use ash::vk;
use metal2vulkan::bind::Dispatch;
use metal2vulkan::metal::IndirectCommandType;
use metal2vulkan::pipeline::LinkedFunctions;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, ComputePipelineState, Device,
    EncoderKind, Event, EventRef, IndirectCommandBufferDescriptor, ObjectId, PipelineOptions,
    ResourceOptions, SharedEvent, Size, Submission, SubmissionCheck, VkEventSignal,
};
use std::collections::{BTreeSet, HashSet};
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The code a submission that uses a detached resource fails with: `OutOfMemory`, as Metal's.
const DETACHED: u64 = 8;

/// The consumer's detach, as a check sees it: the detached objects, and each submission the
/// check saw.
#[derive(Default)]
struct Detach {
    detached: Mutex<HashSet<ObjectId>>,
    seen: Mutex<Vec<(EncoderKind, Range<usize>)>>,
    /// A companion event's value as each check ran ([`Detach::check_reading`]).
    landed: Mutex<Vec<u64>>,
}

impl Detach {
    /// A check that fails, with [`DETACHED`], a submission whose used resources hold a detached
    /// buffer or heap, noting each submission it sees.
    fn check(self: &Arc<Self>) -> SubmissionCheck {
        let detach = Arc::clone(self);
        Arc::new(move |submission: &Submission<'_>| detach.verdict(submission))
    }

    /// [`Detach::check`], which also notes `landed`'s value as it checks.
    fn check_reading(self: &Arc<Self>, landed: &SharedEvent) -> SubmissionCheck {
        let detach = Arc::clone(self);
        let landed = landed.clone();
        Arc::new(move |submission: &Submission<'_>| {
            detach.landed.lock().unwrap().push(landed.signaled_value());
            detach.verdict(submission)
        })
    }

    fn verdict(&self, submission: &Submission<'_>) -> Option<NonZeroU64> {
        self.seen
            .lock()
            .unwrap()
            .push((submission.kind(), submission.encoders()));
        let used = submission.used_resources();
        let detached: Vec<_> = self.detached.lock().unwrap().iter().copied().collect();
        let hit = used
            .buffers
            .iter()
            .map(Buffer::object_id)
            .chain(used.heaps.iter().map(|heap| heap.object_id()))
            .any(|id| detached.contains(&id));
        // The identity answer, with no listing, is the listing's.
        assert_eq!(submission.uses(&detached), hit, "{detached:?}");
        hit.then(|| NonZeroU64::new(DETACHED).unwrap())
    }

    fn detach(&self, buffer: &Buffer) {
        self.detached.lock().unwrap().insert(buffer.object_id());
    }

    fn attach(&self, buffer: &Buffer) {
        self.detached.lock().unwrap().remove(&buffer.object_id());
    }
}

/// Each probe shape led by a wait, committed with a check while its resource is attached, then
/// detached before the wait is signalled: the submission that uses it fails alone, the others
/// run, its signal fires, its handlers run, and it completes with status `Error` and code 8. The
/// check saw each of Metal's submissions once, in order.
#[test]
fn a_submission_detached_behind_a_wait_fails_alone() {
    let kernel = public_case(KERNEL).unwrap();
    on_device(|device, _| {
        let pipeline = pipeline_state(device, &kernel).unwrap();
        for (shape, failed) in SHAPES {
            let encoded = Shape::encode(device, &kernel, &pipeline, &format!("w{shape}"));
            let failed = encoded.failed_set(failed);
            let submissions = encoded.submissions();
            let detach = Arc::new(Detach::default());
            encoded
                .command_buffer
                .set_submission_check(detach.check())
                .unwrap();
            let seen = encoded.watch_handlers();
            encoded.command_buffer.commit().unwrap();
            detach.detach(&encoded.detached);
            encoded.release_waits();
            encoded.finished(&seen, Some(DETACHED), &failed, &kernel);
            assert_eq!(*detach.seen.lock().unwrap(), submissions, "{shape}");
        }
    });
}

/// Each probe shape led by a wait, committed with a check while its resource is detached, then
/// attached again before the wait is signalled: everything runs, and it completes.
#[test]
fn a_submission_attached_again_behind_a_wait_runs() {
    let kernel = public_case(KERNEL).unwrap();
    on_device(|device, _| {
        let pipeline = pipeline_state(device, &kernel).unwrap();
        for (shape, _) in SHAPES {
            let encoded = Shape::encode(device, &kernel, &pipeline, &format!("w{shape}"));
            let submissions = encoded.submissions();
            let detach = Arc::new(Detach::default());
            detach.detach(&encoded.detached);
            encoded
                .command_buffer
                .set_submission_check(detach.check())
                .unwrap();
            let seen = encoded.watch_handlers();
            encoded.command_buffer.commit().unwrap();
            detach.attach(&encoded.detached);
            encoded.release_waits();
            encoded.finished(&seen, None, &BTreeSet::new(), &kernel);
            assert_eq!(*detach.seen.lock().unwrap(), submissions, "{shape}");
        }
    });
}

/// A check is refused once the command buffer is committed, changing nothing.
#[test]
fn a_check_is_refused_once_committed() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer.commit().unwrap();
        let detach = Arc::new(Detach::default());
        assert!(command_buffer.set_submission_check(detach.check()).is_err());
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert!(detach.seen.lock().unwrap().is_empty());
    });
}

/// A commit returns at once while another command buffer's check runs on the scheduler thread,
/// held there by this test, as Metal's `commit` does, and submits nothing itself: the scheduler
/// thread makes every submission, so the command buffer committed on another queue stays
/// `Committed` until the check returned, and then runs and completes.
#[test]
fn a_commit_made_while_a_check_runs_returns_at_once_and_runs_after_the_check() {
    use std::sync::mpsc;
    on_device(|device, _| {
        let fill = |queue: &metal2vulkan_runtime::CommandQueue, buffer: &Buffer| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(buffer, 0..buffer.length(), 0x5a).unwrap();
            blit.end_encoding().unwrap();
            command_buffer
        };
        let (entered, check_entered) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let (entered, released) = (Mutex::new(entered), Mutex::new(released));
        let held_check: SubmissionCheck = Arc::new(move |_: &Submission<'_>| {
            let _ = entered.lock().unwrap().send(());
            let _ = released
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(60));
            None
        });
        let checked = device.new_command_queue().unwrap();
        let free = device.new_command_queue().unwrap();
        let (held_bytes, free_bytes) = (
            device.new_buffer(256, ResourceOptions::shared()).unwrap(),
            device.new_buffer(256, ResourceOptions::shared()).unwrap(),
        );
        let held = fill(&checked, &held_bytes);
        held.set_submission_check(held_check).unwrap();
        held.commit().unwrap();
        check_entered
            .recv_timeout(Duration::from_secs(30))
            .expect("the check runs on the scheduler thread");
        let other = fill(&free, &free_bytes);
        other.commit().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let status = other.status();
        release.send(()).unwrap();
        held.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(held.status(), CommandBufferStatus::Completed);
        assert_eq!(
            status,
            CommandBufferStatus::Committed,
            "submitted while the scheduler thread ran the check"
        );
        other.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(other.status(), CommandBufferStatus::Completed);
        assert_eq!(read_buffer(&free_bytes, 0, 256).unwrap(), vec![0x5a; 256]);
    });
}

/// [`KERNEL`]'s pipeline state, made with `supportIndirectCommandBuffers`.
fn icb_state(device: &Device, kernel: &PublicCase) -> ComputePipelineState {
    let function = device
        .new_library_with_air(&kernel.bitcode)
        .unwrap()
        .new_function_with_name_constant_values(&kernel.compute.entry, &kernel.compute.constants)
        .unwrap();
    let mut descriptor = compute_descriptor(&function, LinkedFunctions::new(), &kernel.compute);
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A buffer an indirect command binds as [`KERNEL`]'s output, which no encoder makes resident,
/// is the submission's from the commit on (`ended_encoders` lists it then, not before). Between
/// two blits, behind a wait: detached before the signal, that submission fails alone and the
/// output keeps its poison; left attached, the command runs and writes Metal's output.
#[test]
fn a_buffer_reached_only_through_an_indirect_command_fails_its_submission() {
    let kernel = public_case(KERNEL).unwrap();
    on_device(|device, _| {
        let pipeline = icb_state(device, &kernel);
        let compute = &kernel.compute;
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = compute.output
        else {
            panic!("{KERNEL}: the case's output is no buffer");
        };
        let size = |[x, y, z]: [u32; 3]| Size::new(x.into(), y.into(), z.into());
        for detached in [true, false] {
            let resources = CaseResources::new(device, &kernel).unwrap();
            let buffers = bound_buffers(device, compute, &resources).unwrap();
            let output = buffers[&binding].clone();
            let initial = read_buffer(&output, offset, length).unwrap();
            assert_ne!(initial, kernel.metal_output, "{KERNEL}: no output to see");
            let mut descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
                .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
            descriptor.max_kernel_buffer_bind_count = 31;
            let icb = device
                .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
                .unwrap();
            let command = icb.indirect_compute_command_at(0).unwrap();
            command.set_compute_pipeline_state(&pipeline).unwrap();
            for (&slot, buffer) in &buffers {
                command.set_kernel_buffer(Some(buffer), 0, slot).unwrap();
            }
            for (&slot, &threadgroup) in &compute.threadgroup_lengths {
                command
                    .set_threadgroup_memory_length(threadgroup, slot)
                    .unwrap();
            }
            match compute.dispatch {
                Dispatch::Threads {
                    threads,
                    threads_per_group,
                } => command.concurrent_dispatch_threads(size(threads), size(threads_per_group)),
                Dispatch::Threadgroups {
                    groups,
                    threads_per_group,
                } => {
                    command.concurrent_dispatch_threadgroups(size(groups), size(threads_per_group))
                }
                other => panic!("{other:?} is no dispatch of a command"),
            }
            .unwrap();

            let queue = device.new_command_queue().unwrap();
            let gate = device.new_shared_event().unwrap();
            let words = buffer_with(device, &[0; 8]).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_wait_for_event(EventRef::from(&gate), 1)
                .unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&words, 0..4, 0x11).unwrap();
            blit.end_encoding().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&words, 4..8, 0x22).unwrap();
            blit.end_encoding().unwrap();

            let lists = |command_buffer: &metal2vulkan_runtime::CommandBuffer| {
                command_buffer.ended_encoders()[1]
                    .used_resources
                    .buffers
                    .contains(&output)
            };
            assert!(!lists(&command_buffer), "listed before the commit");
            let detach = Arc::new(Detach::default());
            command_buffer.set_submission_check(detach.check()).unwrap();
            command_buffer.commit().unwrap();
            assert!(lists(&command_buffer), "not listed once committed");
            assert!(command_buffer.used_resources().buffers.contains(&output));
            if detached {
                detach.detach(&output);
            }
            assert!(command_buffer.wait_until_completed_for(HELD).is_err());
            gate.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            if detached {
                assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
                assert_eq!(
                    command_buffer.error(),
                    Some(CommandBufferError::Injected(DETACHED))
                );
            } else {
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            }
            let want = if detached {
                &initial
            } else {
                &kernel.metal_output
            };
            assert_eq!(
                &read_buffer(&output, offset, length).unwrap(),
                want,
                "detached {detached}"
            );
            let mut both = [0x11u8; 8];
            both[4..].fill(0x22);
            assert_eq!(read_buffer(&words, 0, 8).unwrap(), both);
            assert_eq!(
                *detach.seen.lock().unwrap(),
                [
                    (EncoderKind::Blit, 0..1),
                    (EncoderKind::Compute, 1..2),
                    (EncoderKind::Blit, 2..3)
                ]
            );
            drop(resources);
        }
    });
}

/// The event a waiting command buffer waits for, by its kind.
enum Waited {
    Event(Event),
    Shared(SharedEvent),
}

impl Waited {
    fn new(device: &Device, shared: bool) -> Waited {
        if shared {
            Waited::Shared(device.new_shared_event().unwrap())
        } else {
            Waited::Event(device.new_event().unwrap())
        }
    }

    fn event(&self) -> EventRef<'_> {
        match self {
            Waited::Event(event) => EventRef::from(event),
            Waited::Shared(event) => EventRef::from(event),
        }
    }
}

/// A GPU-signal test's run: a command buffer that waits for `waited`'s value 1 and fills
/// `target`'s first word, committed with a check, then a signaller on another queue that waits for
/// a plain event's value 1 (its gate), signals `landed` to 1, fills a word, and signals `waited`
/// to 1, committed and scheduled. The gate's one signal is the test's own, reserved before the
/// signaller waits for it (`Event::reserve_vk_signal`) and given from the host
/// ([`open`](InFlight::open)): a plain event's wait for a reserved signal goes to its `VkQueue` at
/// once, so the signaller is submitted and its signals are in flight, held on the GPU until the
/// test opens the gate, however fast the GPU and however loaded the host. `landed` reaches 1 no
/// later than `waited` does (one queue's signals land in order), so it reads 0 only while
/// `waited`'s signal has not landed. Where `at_commit`, the waiter is committed after the
/// signaller is scheduled instead, with `commit_and_submit`, its wait then met only by the GPU
/// signal in flight: a committing thread leaves such a check to the scheduler's thread, which
/// waits for the signal to land.
struct InFlight {
    device: Device,
    waiter: CommandBuffer,
    signaller: CommandBuffer,
    target: Buffer,
    landed: SharedEvent,
    detach: Arc<Detach>,
    /// The signaller's gate, the counter value of its reserved signal, and whether the test gave
    /// it: a run dropped with it closed (a failed assertion) gives it then, so no command buffer
    /// waits for good.
    gate: Event,
    counter: u64,
    opened: AtomicBool,
    /// Kept until the work completes.
    _words: Buffer,
    /// The threads the waiter's checks ran on.
    checked_on: Arc<Mutex<Vec<std::thread::ThreadId>>>,
    /// When the waiter and the signaller were scheduled, by their scheduled handlers.
    scheduled: [Arc<Mutex<Option<Instant>>>; 2],
}

/// A scheduled handler that notes when it ran in `at`.
fn noting_scheduled(command_buffer: &CommandBuffer) -> Arc<Mutex<Option<Instant>>> {
    let at = Arc::new(Mutex::new(None));
    let noted = Arc::clone(&at);
    command_buffer
        .add_scheduled_handler(Box::new(move |_| {
            *noted.lock().unwrap() = Some(Instant::now());
        }))
        .unwrap();
    at
}

impl InFlight {
    fn start(device: &Device, waited: &Waited, at_commit: bool) -> InFlight {
        let target = buffer_with(device, &[0; 16]).unwrap();
        let words = buffer_with(device, &[0; 4]).unwrap();
        let landed = device.new_shared_event().unwrap();
        let detach = Arc::new(Detach::default());
        let gate = device.new_event().unwrap();
        // SAFETY: the one signal of the reservation is given once, from the host, by `open`
        // (or the run's drop): a fresh event's first, so after nothing.
        let reserved = unsafe { gate.reserve_vk_signal(1) }.unwrap();
        let VkEventSignal::Counter {
            counter,
            after: None,
        } = reserved
        else {
            panic!("a fresh event's first signal reserved as {reserved:?}");
        };

        let waiter = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        waiter.encode_wait_for_event(waited.event(), 1).unwrap();
        let mut blit = waiter.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..4, 0x5a).unwrap();
        blit.end_encoding().unwrap();
        let checked_on = Arc::new(Mutex::new(Vec::new()));
        let (check, threads) = (detach.check_reading(&landed), Arc::clone(&checked_on));
        waiter
            .set_submission_check(Arc::new(move |submission: &Submission<'_>| {
                threads.lock().unwrap().push(std::thread::current().id());
                check(submission)
            }))
            .unwrap();
        let waiter_scheduled = noting_scheduled(&waiter);
        if !at_commit {
            waiter.commit().unwrap();
        }

        let signaller = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        signaller
            .encode_wait_for_event(EventRef::from(&gate), 1)
            .unwrap();
        signaller
            .encode_signal_event(EventRef::from(&landed), 1)
            .unwrap();
        let mut blit = signaller.blit_command_encoder().unwrap();
        blit.fill_buffer(&words, 0..4, 0x11).unwrap();
        blit.end_encoding().unwrap();
        signaller.encode_signal_event(waited.event(), 1).unwrap();
        let signaller_scheduled = noting_scheduled(&signaller);
        let run = InFlight {
            device: device.clone(),
            waiter,
            signaller,
            target,
            landed,
            detach,
            gate,
            counter,
            opened: AtomicBool::new(false),
            _words: words,
            checked_on,
            scheduled: [waiter_scheduled, signaller_scheduled],
        };
        run.signaller.commit().unwrap();
        run.signaller.wait_until_scheduled().unwrap();
        if at_commit {
            run.waiter.commit_and_submit().unwrap();
        }
        run
    }

    /// Give the gate's reserved signal from the host, once: the signaller's work runs on.
    fn open(&self) {
        if self.opened.swap(true, Ordering::SeqCst) {
            return;
        }
        let info = vk::SemaphoreSignalInfo::default()
            // SAFETY: the semaphore the gate holds while it lives, signalled only as reserved.
            .semaphore(unsafe { self.gate.vk_semaphore() })
            .value(self.counter);
        // SAFETY: the reservation's one signal operation, from the host, of a semaphore of the
        // live device; none of the device's objects is destroyed.
        unsafe { self.device.raw().device.signal_semaphore(&info) }.unwrap();
    }

    /// Wait, at most [`WAIT`], while the waiter is `Committed`: whether it left that status.
    fn waiter_scheduled(&self) -> bool {
        let deadline = Instant::now() + WAIT;
        while self.waiter.status() == CommandBufferStatus::Committed {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
        true
    }

    /// How long after the signaller the waiter was scheduled, by their scheduled handlers.
    fn scheduled_apart(&self) -> Option<Duration> {
        let [waiter, signaller] = &self.scheduled;
        let (waiter, signaller) = (*waiter.lock().unwrap(), *signaller.lock().unwrap());
        Some(waiter?.saturating_duration_since(signaller?))
    }

    /// Open the gate, then wait for both command buffers; the companion event's value as the
    /// waiter's one check ran.
    fn finish(&self) -> u64 {
        self.open();
        self.signaller.wait_until_completed_for(WAIT).unwrap();
        self.waiter.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(self.signaller.status(), CommandBufferStatus::Completed);
        let landed = self.detach.landed.lock().unwrap();
        assert_eq!(landed.len(), 1, "the waiter's submission is checked once");
        landed[0]
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.open();
    }
}

/// A submission whose `SharedEvent` wait another command buffer's GPU signal meets is checked
/// only once that signal landed, as Metal holds such a command buffer until then
/// (`event-wait-behind-a-running-signal`): the companion event the signaller signals first reads 1
/// in the check, and a head is counted as awaiting its checks' signals. Its buffer detached once
/// the signaller is scheduled, while its signal is held in flight at the signaller's gate (the
/// companion still reads 0 after the detach), the submission fails with code 8 and its fill does
/// not run. So too for a waiter committed with `commit_and_submit` while the signal is in flight:
/// its check is left to the scheduler's thread, never run on the committing thread before the
/// signal landed.
#[test]
fn a_submission_detached_while_a_gpu_signal_is_in_flight_fails() {
    on_device(|device, _| {
        let caller = std::thread::current().id();
        let shared = true;
        for at_commit in [false, true] {
            let waited = Waited::new(device, shared);
            let awaiting = device.work_counts().heads_awaiting_check_signals;
            let run = InFlight::start(device, &waited, at_commit);
            run.detach.detach(&run.target);
            assert_eq!(
                run.landed.signaled_value(),
                0,
                "shared {shared}, at commit {at_commit}: the signal landed before its gate opened"
            );
            let at_check = run.finish();
            let awaited = device.work_counts().heads_awaiting_check_signals - awaiting;
            assert_eq!(
                at_check, 1,
                "shared {shared}, at commit {at_commit}: checked before the GPU signal landed"
            );
            assert!(
                !run.checked_on.lock().unwrap().contains(&caller),
                "shared {shared}, at commit {at_commit}: checked on the committing thread"
            );
            assert!(
                awaited > 0,
                "at commit {at_commit}: no head counted as awaiting its checks' signals"
            );
            assert_eq!(
                run.waiter.status(),
                CommandBufferStatus::Error,
                "shared {shared}"
            );
            assert_eq!(
                run.waiter.error(),
                Some(CommandBufferError::Injected(DETACHED)),
                "shared {shared}"
            );
            assert_eq!(
                read_buffer(&run.target, 0, 4).unwrap(),
                [0; 4],
                "shared {shared}"
            );
        }
    });
}

/// A submission whose `Event` wait another command buffer's GPU signal meets, still running, is
/// checked at once and submitted with its wait on the `VkQueue`, as Metal submits (and so checks)
/// such a command buffer at once (an M3 Max under macOS 27.0.1: the conformance case
/// `event-wait-behind-a-running-signal`, the waiter `Scheduled` and its scheduled handler run while
/// the signaller still ran). The waiter reads `Scheduled` while the signaller's signals are held in
/// flight at its gate (the companion event the signaller signals first still reads 0), and the
/// companion read 0 in its check; no head is counted as awaiting its checks' signals
/// (`WorkCounts::heads_awaiting_check_signals`), its check runs once, and its fill runs, after the
/// signal. So too for a waiter committed with `commit_and_submit` while the signal is in flight.
/// Each run prints how long after the signaller the waiter was scheduled.
#[test]
fn an_event_wait_a_gpu_signal_in_flight_meets_is_checked_at_once() {
    on_device(|device, _| {
        for at_commit in [false, true] {
            let waited = Waited::new(device, false);
            let awaiting = device.work_counts().heads_awaiting_check_signals;
            let run = InFlight::start(device, &waited, at_commit);
            assert!(
                run.waiter_scheduled(),
                "at commit {at_commit}: the waiter was never scheduled while its signal was in \
                 flight"
            );
            // Read after the waiter's status, with the gate still closed.
            assert_eq!(
                run.landed.signaled_value(),
                0,
                "at commit {at_commit}: the signal landed before its gate opened"
            );
            let at_check = run.finish();
            let awaited = device.work_counts().heads_awaiting_check_signals - awaiting;
            assert_eq!(
                awaited, 0,
                "at commit {at_commit}: a head awaited its checks' signals"
            );
            assert_eq!(run.waiter.status(), CommandBufferStatus::Completed);
            assert_eq!(read_buffer(&run.target, 0, 4).unwrap(), [0x5a; 4]);
            assert_eq!(
                at_check, 0,
                "at commit {at_commit}: checked only once the GPU signal landed"
            );
            println!(
                "EVENT-IN-FLIGHT at_commit={at_commit}: the waiter was scheduled {:?} after the \
                 signaller",
                run.scheduled_apart()
            );
        }
    });
}

/// A `SharedEvent` wait that a GPU signal in flight would meet, met by the host first: the check
/// runs at the host's signal, without waiting for the GPU (the companion event reads 0 in it, the
/// signaller's signals held at its gate), and the submission runs.
#[test]
fn a_wait_the_host_meets_is_checked_without_waiting_for_the_gpu() {
    on_device(|device, _| {
        let waited = Waited::new(device, true);
        let run = InFlight::start(device, &waited, false);
        let Waited::Shared(event) = &waited else {
            unreachable!()
        };
        event.set_signaled_value(1);
        assert!(
            run.waiter_scheduled(),
            "the waiter was never scheduled once the host met its wait"
        );
        let at_check = run.finish();
        assert_eq!(run.waiter.status(), CommandBufferStatus::Completed);
        assert_eq!(read_buffer(&run.target, 0, 4).unwrap(), [0x5a; 4]);
        assert_eq!(at_check, 0, "checked only once the GPU signal landed");
    });
}
