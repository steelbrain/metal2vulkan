//! Device tests of submission handlers (`CommandBuffer::add_submission_handler`, a runtime
//! extension) and of `wait_until_scheduled`'s answer, on the device the cfg selects, under the
//! validation layer.
//!
//! A submission handler runs once the scheduler took the command buffer's last batch for its
//! `VkQueue` (submitted it, or held it for one kernel submission with later ones), on the
//! device's scheduler thread, which makes every submission (`commit` returns at once and submits
//! nothing), before the command buffer is `Scheduled`: once the commit is taken in where its waits
//! are met, once an event the host raised releases it, and once another command buffer's GPU signal
//! that releases it is submitted. Never for a command buffer that fails before its last batch is
//! taken.
//!
//! Metal's answers (an M4, macOS 27.0.1): `commit` returns before the command
//! buffer is scheduled, on Metal's own thread; its status reads `Scheduled` just before its first
//! scheduled handler starts, and Metal's wait for the scheduling returns only once every
//! scheduled handler returned.

use super::{
    at_plan, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, CommitFailure, DeadlockCause, Device,
    Error, EventRef, Handler, Invalid, Problem, ResourceOptions, SharedEvent, TimedOut,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// The name of the device's scheduler thread.
const SCHEDULER: &str = "metal2vulkan-scheduler";

/// Run `test` on a validated device, then check that once everything is dropped (the device
/// last) the layer reported nothing.
fn on_device(test: impl FnOnce(&Device)) {
    on_device_with(|_| {}, test)
}

/// [`on_device`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn on_device_with(adjust: fn(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A command buffer of `queue` filling `buffer` with `value`, not committed; `wait` leads it with
/// an event wait, after a fill too where `work_first`.
fn fill(
    queue: &CommandQueue,
    buffer: &Buffer,
    value: u8,
    wait: Option<(&SharedEvent, bool)>,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let encode = |value: u8| {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(buffer, 0..buffer.length(), value).unwrap();
        blit.end_encoding().unwrap();
    };
    if let Some((event, work_first)) = wait {
        if work_first {
            encode(value);
        }
        command_buffer
            .encode_wait_for_event(EventRef::from(event), 1)
            .unwrap();
    }
    encode(value);
    command_buffer
}

/// Wait for `command_buffer` and check it completed without error.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// One handler's call: its name, its thread, and the status it read.
#[derive(Clone, Debug)]
struct Call {
    name: &'static str,
    thread: ThreadId,
    thread_name: Option<String>,
    status: CommandBufferStatus,
}

/// The handlers' calls, in the order they were made.
#[derive(Clone, Default)]
struct Log(Arc<Mutex<Vec<Call>>>);

impl Log {
    /// A handler that notes its call as `name`.
    fn handler(&self, name: &'static str) -> Handler {
        let log = Arc::clone(&self.0);
        Box::new(move |own: &CommandBuffer| {
            let current = thread::current();
            log.lock().unwrap().push(Call {
                name,
                thread: current.id(),
                thread_name: current.name().map(str::to_owned),
                status: own.status(),
            });
        })
    }

    fn calls(&self) -> Vec<Call> {
        self.0.lock().unwrap().clone()
    }

    fn names(&self) -> Vec<&'static str> {
        self.calls().iter().map(|call| call.name).collect()
    }

    /// The call named `name`.
    fn call(&self, name: &str) -> Call {
        self.calls()
            .into_iter()
            .find(|call| call.name == name)
            .unwrap_or_else(|| panic!("{name} was not called: {:?}", self.names()))
    }

    /// Where the call named `name` is among the calls.
    fn at(&self, name: &str) -> usize {
        self.names()
            .iter()
            .position(|called| *called == name)
            .unwrap_or_else(|| panic!("{name} was not called: {:?}", self.names()))
    }
}

/// A commit whose waits are met (no wait, or a wait the event already met) leaves its batches to
/// the scheduler thread and returns (`WorkCounts::commits_handed_off`); that thread submits the last
/// batch and runs the submission handlers, once each, in the order added, each seeing the command
/// buffer `Committed`; its scheduled handlers run after them on a handler thread, seeing it
/// `Scheduled`.
#[test]
fn a_commit_s_handlers_run_in_order_on_the_scheduler_thread_before_it_is_scheduled() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let met = device.new_shared_event().unwrap();
        met.set_signaled_value(1);
        for wait in [None, Some((&met, false)), Some((&met, true))] {
            let case = format!("wait {:?}", wait.map(|(_, work_first)| work_first));
            let log = Log::default();
            let command_buffer = fill(&queue, &buffer, 1, wait);
            for name in ["first", "second", "third"] {
                command_buffer
                    .add_submission_handler(log.handler(name))
                    .unwrap();
            }
            command_buffer
                .add_scheduled_handler(log.handler("scheduled"))
                .unwrap();
            let handed_off = device.work_counts().commits_handed_off;
            command_buffer.commit().unwrap();
            assert_eq!(
                device.work_counts().commits_handed_off - handed_off,
                1,
                "{case}: the commit left its batches to the scheduler thread"
            );
            command_buffer.wait_until_scheduled().unwrap();
            let calls = log.calls();
            assert_eq!(
                log.names(),
                ["first", "second", "third", "scheduled"],
                "{case}: {calls:?}"
            );
            for call in &calls[..3] {
                assert_eq!(
                    call.thread_name.as_deref(),
                    Some(SCHEDULER),
                    "{case}: {calls:?}"
                );
                assert_eq!(call.status, CommandBufferStatus::Committed, "{case}");
            }
            let scheduled = log.call("scheduled");
            assert_ne!(scheduled.thread, thread::current().id(), "{case}");
            assert_ne!(scheduled.thread_name.as_deref(), Some(SCHEDULER), "{case}");
            assert_eq!(scheduled.status, CommandBufferStatus::Scheduled, "{case}");
            completed(&command_buffer);
            assert_eq!(log.calls().len(), 4, "{case}: each ran once");
        }
    });
}

/// A command buffer whose last batch waits for an event the host raises later (with or without
/// work before the wait) runs nothing at its commit and stays `Committed`; once the host raises
/// the event, the scheduler thread submits it and runs its submission handlers, seeing it
/// `Committed`, before its scheduled handlers.
#[test]
fn a_commit_behind_an_unmet_wait_runs_them_on_the_scheduler_thread_once_released() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        for work_first in [false, true] {
            let case = format!("work first {work_first}");
            let event = device.new_shared_event().unwrap();
            let log = Log::default();
            let command_buffer = fill(&queue, &buffer, 2, Some((&event, work_first)));
            command_buffer
                .add_submission_handler(log.handler("submitted"))
                .unwrap();
            command_buffer
                .add_scheduled_handler(log.handler("scheduled"))
                .unwrap();
            command_buffer.commit().unwrap();
            thread::sleep(Duration::from_millis(50));
            assert!(log.calls().is_empty(), "{case}: {:?}", log.calls());
            assert_eq!(command_buffer.status(), CommandBufferStatus::Committed);
            event.set_signaled_value(1);
            command_buffer.wait_until_scheduled().unwrap();
            assert_eq!(log.names(), ["submitted", "scheduled"], "{case}");
            let submitted = log.call("submitted");
            assert_eq!(submitted.thread_name.as_deref(), Some(SCHEDULER), "{case}");
            assert_eq!(submitted.status, CommandBufferStatus::Committed, "{case}");
            assert_eq!(
                log.call("scheduled").status,
                CommandBufferStatus::Scheduled,
                "{case}"
            );
            completed(&command_buffer);
        }
    });
}

/// A command buffer committed behind an earlier one of its queue that waits for an event is not
/// submitted at its commit; once the host raises the event, the scheduler thread submits both, in
/// order, and runs each one's submission handlers before that one's scheduled handlers.
#[test]
fn a_command_buffer_behind_a_held_one_of_its_queue_runs_them_once_it_goes() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let event = device.new_shared_event().unwrap();
        let log = Log::default();
        let held = fill(&queue, &buffer, 3, Some((&event, false)));
        let behind = fill(&queue, &buffer, 4, None);
        for (command_buffer, submitted, scheduled) in [
            (&held, "held submitted", "held scheduled"),
            (&behind, "behind submitted", "behind scheduled"),
        ] {
            command_buffer
                .add_submission_handler(log.handler(submitted))
                .unwrap();
            command_buffer
                .add_scheduled_handler(log.handler(scheduled))
                .unwrap();
        }
        held.commit().unwrap();
        behind.commit().unwrap();
        thread::sleep(Duration::from_millis(50));
        assert!(log.calls().is_empty(), "{:?}", log.calls());
        assert_eq!(behind.status(), CommandBufferStatus::Committed);
        event.set_signaled_value(1);
        behind.wait_until_scheduled().unwrap();
        for name in ["held submitted", "behind submitted"] {
            let call = log.call(name);
            assert_eq!(call.thread_name.as_deref(), Some(SCHEDULER), "{name}");
            assert_eq!(call.status, CommandBufferStatus::Committed, "{name}");
        }
        assert!(log.at("held submitted") < log.at("behind submitted"));
        assert!(log.at("held submitted") < log.at("held scheduled"));
        assert!(log.at("behind submitted") < log.at("behind scheduled"));
        assert!(log.at("held scheduled") < log.at("behind scheduled"));
        completed(&held);
        completed(&behind);
    });
}

/// A command buffer whose wait another queue's command buffer meets with a GPU signal is submitted
/// once that one is, in the same pass of the scheduler thread: its submission handlers run there,
/// after the signaller's own.
#[test]
fn a_command_buffer_a_gpu_signal_releases_runs_them_after_the_signaller_s() {
    on_device_with(at_plan, |device| {
        let (first, second) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let other = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let event = device.new_shared_event().unwrap();
        let log = Log::default();
        let waiting = fill(&second, &buffer, 5, Some((&event, false)));
        waiting
            .add_submission_handler(log.handler("waiting submitted"))
            .unwrap();
        waiting.commit().unwrap();
        let signaller = fill(&first, &other, 6, None);
        signaller
            .encode_signal_event(EventRef::from(&event), 1)
            .unwrap();
        signaller
            .add_submission_handler(log.handler("signaller submitted"))
            .unwrap();
        assert!(log.calls().is_empty());
        signaller.commit().unwrap();
        signaller.wait_until_scheduled().unwrap();
        waiting.wait_until_scheduled().unwrap();
        assert_eq!(log.names(), ["signaller submitted", "waiting submitted"]);
        for call in log.calls() {
            assert_eq!(
                call.thread_name.as_deref(),
                Some(SCHEDULER),
                "{}",
                call.name
            );
        }
        completed(&signaller);
        completed(&waiting);
    });
}

/// A command buffer committed as failed submits its batches: its submission handlers run, on the
/// scheduler thread, and it completes with its failure.
#[test]
fn a_command_buffer_committed_as_failed_runs_them() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let log = Log::default();
        let command_buffer = fill(&queue, &buffer, 7, None);
        command_buffer
            .add_submission_handler(log.handler("submitted"))
            .unwrap();
        command_buffer.commit_failed(CommitFailure::new(8)).unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(log.names(), ["submitted"]);
        assert_eq!(
            log.call("submitted").thread_name.as_deref(),
            Some(SCHEDULER)
        );
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
    });
}

/// A submission handler added after the commit is refused, as a scheduled handler is.
#[test]
fn a_submission_handler_after_the_commit_is_refused() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let command_buffer = fill(&queue, &buffer, 8, None);
        command_buffer.commit().unwrap();
        assert_eq!(
            command_buffer.add_submission_handler(Box::new(|_| {})),
            Err(Error::Invalid(Invalid::new(
                "add_submission_handler",
                Problem::HandlerAfterCommit
            )))
        );
        completed(&command_buffer);
    });
}

/// What a submission handler's waits answered: for its own command buffer's scheduling,
/// completion and bounded completion, then for a completed one's completion and scheduling.
type Answers = (
    Result<(), Error>,
    Result<(), Error>,
    Result<(), TimedOut>,
    Result<(), Error>,
    Result<(), Error>,
);

/// In a submission handler, on the scheduler thread, a wait for its own
/// command buffer (whose scheduling waits for the handler) is refused at once
/// (`DeadlockCause::SubmissionHandler`, `TimedOut` for the bounded wait); a wait for a command
/// buffer that completed returns.
#[test]
fn a_submission_handler_s_waits_for_what_has_not_happened_are_refused() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let done = fill(&queue, &buffer, 9, None);
        done.commit().unwrap();
        completed(&done);
        for held in [false, true] {
            let event = device.new_shared_event().unwrap();
            let wait = held.then_some((&event, false));
            let command_buffer = fill(&queue, &buffer, 10, wait);
            let (sender, answers) = mpsc::channel::<Answers>();
            let done = done.clone();
            command_buffer
                .add_submission_handler(Box::new(move |own: &CommandBuffer| {
                    sender
                        .send((
                            own.wait_until_scheduled(),
                            own.wait_until_completed(),
                            own.wait_until_completed_for(WAIT),
                            done.wait_until_completed(),
                            done.wait_until_scheduled(),
                        ))
                        .unwrap();
                }))
                .unwrap();
            command_buffer.commit().unwrap();
            if held {
                event.set_signaled_value(1);
            }
            let refused = Err(Error::WaitWouldDeadlock(DeadlockCause::SubmissionHandler));
            let answers = answers.recv_timeout(WAIT).unwrap();
            assert_eq!(
                answers,
                (refused.clone(), refused, Err(TimedOut), Ok(()), Ok(())),
                "held {held}"
            );
            completed(&command_buffer);
        }
    });
}

/// Metal's order: the status reads `Scheduled` while a scheduled handler runs,
/// and `wait_until_scheduled` returns only once it returned.
#[test]
fn wait_until_scheduled_returns_once_the_scheduled_handlers_returned() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let command_buffer = fill(&queue, &buffer, 11, None);
        let (entered, entering) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        command_buffer
            .add_scheduled_handler(Box::new(move |own: &CommandBuffer| {
                entered.send(own.status()).unwrap();
                released.recv_timeout(WAIT).unwrap();
            }))
            .unwrap();
        command_buffer.commit().unwrap();
        assert_eq!(
            entering.recv_timeout(WAIT).unwrap(),
            CommandBufferStatus::Scheduled
        );
        assert_eq!(command_buffer.status(), CommandBufferStatus::Scheduled);
        let returned = Arc::new(AtomicBool::new(false));
        let waiter = {
            let (command_buffer, returned) = (command_buffer.clone(), Arc::clone(&returned));
            thread::spawn(move || {
                let answer = command_buffer.wait_until_scheduled();
                returned.store(true, Ordering::SeqCst);
                answer
            })
        };
        thread::sleep(Duration::from_millis(100));
        assert!(
            !returned.load(Ordering::SeqCst),
            "wait_until_scheduled returned while a scheduled handler ran"
        );
        release.send(()).unwrap();
        waiter.join().unwrap().unwrap();
        assert!(returned.load(Ordering::SeqCst));
        completed(&command_buffer);
    });
}
