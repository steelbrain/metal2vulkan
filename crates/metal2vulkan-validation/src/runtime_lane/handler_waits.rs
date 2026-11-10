//! Device tests of the queues' serial contexts and of command buffer waits from the device's
//! handler threads: from a scheduled handler, a completed handler, and a deallocator run as a
//! command buffer releases what it retained, on the device the cfg selects, under the validation
//! layer.
//!
//! The answers expected are Metal's (measured on an M4, macOS 27.0.1): a queue's transitions wait
//! for its blocked handler and another queue's go on; a handler's wait returns for another
//! queue's command buffer and never returns for a later transition of its own queue, which the
//! runtime refuses at once (`WaitWouldDeadlock` of `DeadlockCause::OwnQueue`, or `TimedOut` for the
//! bounded wait). So it refuses a wait that would close a cycle of handler waits (`Cycle`), and
//! one that would leave the device's handler threads with none free (`HandlerThreadCap`). No wait here can hang: every wait in a handler is answered at once or
//! is for work that other threads do, and every wait of the test's own thread is bounded.

use super::{
    at_plan, layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, DeadlockCause, Device, Error,
    EventRef, ResourceOptions, SharedEvent, TimedOut,
};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The cause of a handler's refused wait for a later transition of its own queue (Metal's never
/// returns there).
const OWN_QUEUE: DeadlockCause = DeadlockCause::OwnQueue;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// The most handler waits Metal was measured to block at once (all returned).
const METAL_MEASURED: usize = 256;

/// The runtime's cap on a device's handler threads (`contexts::THREADS`).
const THREADS: usize = 512;

/// What one handler's waits answered.
type Answers = (
    Result<(), Error>,
    Result<(), TimedOut>,
    Result<(), Error>,
    CommandBufferStatus,
);

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

/// A command buffer of `queue` filling the first 64 bytes of `buffer` with `value`, not
/// committed.
fn fill(queue: &CommandQueue, buffer: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..64, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// The first byte of a host-visible buffer, as the device last wrote it.
fn first_byte(buffer: &Buffer) -> u8 {
    let contents = buffer.contents().expect("a host-visible buffer");
    // SAFETY: the buffer holds at least one byte at `contents`, alive with the buffer; a
    // volatile read sees a fill the device makes meanwhile.
    unsafe { std::ptr::read_volatile(contents.as_ptr()) }
}

/// The waits for `target` from a handler: `wait_until_completed`, its bounded form,
/// `wait_until_scheduled`, and the status after them.
fn waits(target: &CommandBuffer) -> Answers {
    (
        target.wait_until_completed(),
        target.wait_until_completed_for(WAIT),
        target.wait_until_scheduled(),
        target.status(),
    )
}

/// Wait for `command_buffer` and check it completed without error.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// Add `handler` to `command_buffer` as a scheduled handler, or as a completed one.
fn add_handler(
    command_buffer: &CommandBuffer,
    scheduled: bool,
    handler: impl FnOnce(&CommandBuffer) + Send + 'static,
) {
    if scheduled {
        command_buffer
            .add_scheduled_handler(Box::new(handler))
            .unwrap();
    } else {
        command_buffer
            .add_completed_handler(Box::new(handler))
            .unwrap();
    }
}

/// A command buffer's own handlers (Metal: a scheduled handler's waits for its command buffer,
/// and a completed handler's wait for completion, never return; a completed handler's wait for
/// scheduling returns). The runtime refuses the first at once and returns from the last.
#[test]
fn waits_from_a_command_buffer_s_own_handlers_answer_at_once() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let command_buffer = fill(&queue, &buffer, 0x11);
        let (scheduled_sender, scheduled_answers) = mpsc::channel();
        let (completed_sender, completed_answers) = mpsc::channel();
        add_handler(&command_buffer, true, move |own| {
            scheduled_sender.send(waits(own)).unwrap();
        });
        add_handler(&command_buffer, false, move |own| {
            completed_sender.send(waits(own)).unwrap();
        });
        command_buffer.commit().unwrap();
        completed(&command_buffer);
        let (completion, bounded, scheduling, status) =
            scheduled_answers.recv_timeout(WAIT).unwrap();
        assert_eq!(completion, Err(Error::WaitWouldDeadlock(OWN_QUEUE)));
        assert_eq!(bounded, Err(TimedOut));
        assert_eq!(scheduling, Err(Error::WaitWouldDeadlock(OWN_QUEUE)));
        assert_eq!(status, CommandBufferStatus::Scheduled);
        let (completion, bounded, scheduling, status) =
            completed_answers.recv_timeout(WAIT).unwrap();
        assert_eq!(completion, Err(Error::WaitWouldDeadlock(OWN_QUEUE)));
        assert_eq!(bounded, Err(TimedOut));
        assert_eq!(scheduling, Ok(()));
        assert_eq!(status, CommandBufferStatus::Completed);
    });
}

/// A handler's waits for a later command buffer of its own queue, committed before the handler
/// ran or inside it (Metal: never returns, even once that command buffer's work ran): refused at
/// once.
#[test]
fn a_handler_s_waits_for_a_later_command_buffer_of_its_queue_are_refused() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        for scheduled in [true, false] {
            for inside in [false, true] {
                let first = fill(&queue, &buffer, 0x21);
                let later = fill(&queue, &buffer, 0x22);
                let (sender, answers) = mpsc::channel();
                {
                    let later = later.clone();
                    add_handler(&first, scheduled, move |_| {
                        if inside {
                            later.commit().unwrap();
                        }
                        let (completion, bounded, scheduling, _) = waits(&later);
                        sender.send((completion, bounded, scheduling)).unwrap();
                    });
                }
                first.commit().unwrap();
                if !inside {
                    later.commit().unwrap();
                }
                let case = format!("scheduled handler {scheduled}, committed inside {inside}");
                let (completion, bounded, scheduling) = answers.recv_timeout(WAIT).unwrap();
                assert_eq!(
                    completion,
                    Err(Error::WaitWouldDeadlock(OWN_QUEUE)),
                    "{case}"
                );
                assert_eq!(bounded, Err(TimedOut), "{case}");
                // From a completed handler, the later one's scheduling may have run before it.
                if scheduled || inside {
                    assert_eq!(
                        scheduling,
                        Err(Error::WaitWouldDeadlock(OWN_QUEUE)),
                        "{case}"
                    );
                }
                completed(&first);
                completed(&later);
            }
        }
    });
}

/// A handler's waits for a command buffer of another queue, committed before the handler ran
/// or inside it, from a scheduled and from a completed handler (Metal: they return): they return
/// once it is scheduled or completed.
#[test]
fn a_handler_s_waits_for_another_queue_return_as_metal_s() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        let other = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        for scheduled in [true, false] {
            for inside in [false, true] {
                let first = fill(&queue, &buffer, 0x31);
                let target = fill(&other, &buffer, 0x32);
                let (sender, answers) = mpsc::channel();
                {
                    let target = target.clone();
                    add_handler(&first, scheduled, move |_| {
                        if inside {
                            target.commit().unwrap();
                        }
                        sender.send(waits(&target)).unwrap();
                    });
                }
                first.commit().unwrap();
                if !inside {
                    target.commit().unwrap();
                }
                let case = format!("scheduled handler {scheduled}, committed inside {inside}");
                let (completion, bounded, scheduling, status) = answers.recv_timeout(WAIT).unwrap();
                assert_eq!(completion, Ok(()), "{case}");
                assert_eq!(bounded, Ok(()), "{case}");
                assert_eq!(scheduling, Ok(()), "{case}");
                assert_eq!(status, CommandBufferStatus::Completed, "{case}");
                completed(&first);
                completed(&target);
            }
        }
    });
}

/// While a queue's handler blocks, a later command buffer of that queue stays `Committed` and
/// its handlers wait, although its work ran; another queue's handlers run meanwhile (Metal's
/// measured order).
#[test]
fn a_blocked_handler_holds_back_its_queue_s_transitions_and_no_other_s() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        let other = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let log = Arc::new(Mutex::new(Vec::new()));
        let first = fill(&queue, &buffer, 0x41);
        let later_buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let later = fill(&queue, &later_buffer, 0x42);
        let elsewhere = fill(&other, &buffer, 0x43);
        let (ran_elsewhere, elsewhere_ran) = mpsc::channel();
        for (command_buffer, name) in [(&later, "later"), (&elsewhere, "elsewhere")] {
            let (scheduled_log, completed_log) = (Arc::clone(&log), Arc::clone(&log));
            let ran = ran_elsewhere.clone();
            add_handler(command_buffer, true, move |_| {
                scheduled_log
                    .lock()
                    .unwrap()
                    .push(format!("{name} scheduled"));
            });
            add_handler(command_buffer, false, move |_| {
                completed_log
                    .lock()
                    .unwrap()
                    .push(format!("{name} completed"));
                if name == "elsewhere" {
                    ran.send(()).unwrap();
                }
            });
        }
        let (sender, seen) = mpsc::channel();
        {
            let (log, later, later_buffer, elsewhere) = (
                Arc::clone(&log),
                later.clone(),
                later_buffer.clone(),
                elsewhere.clone(),
            );
            add_handler(&first, false, move |_| {
                log.lock().unwrap().push("first completed".to_owned());
                later.commit().unwrap();
                elsewhere.commit().unwrap();
                // Another queue's handlers run while this one blocks.
                let elsewhere_ran = elsewhere_ran.recv_timeout(WAIT).is_ok();
                // The later command buffer's work runs; its transitions wait for this handler.
                let deadline = Instant::now() + WAIT;
                while first_byte(&later_buffer) != 0x42 && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                let later_ran = first_byte(&later_buffer) == 0x42;
                sender
                    .send((elsewhere_ran, later_ran, later.status()))
                    .unwrap();
                log.lock().unwrap().push("first returns".to_owned());
            });
        }
        first.commit().unwrap();
        let (elsewhere_ran, later_ran, later_status) = seen.recv_timeout(WAIT).unwrap();
        completed(&first);
        completed(&later);
        completed(&elsewhere);
        assert!(
            elsewhere_ran,
            "another queue's handler waited for this one's"
        );
        assert!(later_ran, "the later command buffer's work did not run");
        assert_eq!(later_status, CommandBufferStatus::Committed);
        let log = log.lock().unwrap().clone();
        let at = |entry: &str| log.iter().position(|logged| logged == entry).unwrap();
        assert!(at("elsewhere scheduled") < at("first returns"), "{log:?}");
        assert!(at("elsewhere completed") < at("first returns"), "{log:?}");
        assert!(at("first returns") < at("later scheduled"), "{log:?}");
        assert!(at("later scheduled") < at("later completed"), "{log:?}");
    });
}

/// Queue A's handler waits for queue B's command buffer, whose handler waits for a later
/// command buffer of queue A: whichever wait comes second would close the cycle and is refused,
/// and the other then returns.
#[test]
fn a_cycle_of_handler_waits_refuses_exactly_one() {
    on_device_with(at_plan, |device| {
        let a = device.new_command_queue().unwrap();
        let b = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let first = fill(&a, &buffer, 0x51);
        let later = fill(&a, &buffer, 0x52);
        let middle = fill(&b, &buffer, 0x53);
        let (sender, answers) = mpsc::channel();
        {
            let (middle, sender) = (middle.clone(), sender.clone());
            add_handler(&first, false, move |_| {
                sender
                    .send(("A waits for B", middle.wait_until_completed()))
                    .unwrap();
            });
        }
        {
            let later = later.clone();
            add_handler(&middle, false, move |_| {
                sender
                    .send(("B waits for A", later.wait_until_completed()))
                    .unwrap();
            });
        }
        for command_buffer in [&first, &later, &middle] {
            command_buffer.commit().unwrap();
        }
        let answered: Vec<_> = (0..2)
            .map(|_| answers.recv_timeout(WAIT).unwrap())
            .collect();
        for command_buffer in [&first, &later, &middle] {
            completed(command_buffer);
        }
        let refused: Vec<_> = answered
            .iter()
            .filter(|(_, answer)| *answer == Err(Error::WaitWouldDeadlock(DeadlockCause::Cycle)))
            .collect();
        let returned: Vec<_> = answered
            .iter()
            .filter(|(_, answer)| answer.is_ok())
            .collect();
        assert_eq!((refused.len(), returned.len()), (1, 1), "{answered:?}");
        // The refused one answered first: the other waited until it returned.
        assert_eq!(
            answered[0].1,
            Err(Error::WaitWouldDeadlock(DeadlockCause::Cycle)),
            "{answered:?}"
        );
    });
}

/// `count` queues' command buffers (empty), whose completed handlers each wait for one command
/// buffer of another queue that waits for an event. The event is signaled once every handler
/// entered its wait and a refused wait had time to answer. The handlers' answers.
fn many_waits(device: &Device, count: usize) -> Vec<Result<(), Error>> {
    let event: SharedEvent = device.new_shared_event().unwrap();
    let buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let target_queue = device.new_command_queue().unwrap();
    let target = fill(&target_queue, &buffer, 0x61);
    target
        .encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    target.commit().unwrap();
    let (entered_sender, entered) = mpsc::channel();
    let (sender, answers) = mpsc::channel();
    let mut command_buffers = Vec::with_capacity(count);
    for _ in 0..count {
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let (target, entered_sender, sender) =
            (target.clone(), entered_sender.clone(), sender.clone());
        add_handler(&command_buffer, false, move |_| {
            entered_sender.send(()).unwrap();
            sender.send(target.wait_until_completed()).unwrap();
        });
        command_buffer.commit().unwrap();
        command_buffers.push(command_buffer);
    }
    let all_entered = (0..count).all(|_| entered.recv_timeout(WAIT).is_ok());
    // A refused wait answers at once; the others wait for the event.
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut answered = Vec::new();
    while Instant::now() < deadline {
        match answers.recv_timeout(Duration::from_millis(50)) {
            Ok(answer) => answered.push(answer),
            Err(_) => continue,
        }
    }
    event.set_signaled_value(1);
    while answered.len() < count {
        match answers.recv_timeout(WAIT) {
            Ok(answer) => answered.push(answer),
            Err(_) => break,
        }
    }
    completed(&target);
    for command_buffer in &command_buffers {
        completed(command_buffer);
    }
    assert!(all_entered, "not every handler ran");
    answered
}

/// As many handler waits as Metal was measured to block at once, each on its own queue, all
/// for one command buffer of another queue: every one returns.
#[test]
fn as_many_blocked_handler_waits_as_metal_ran_all_return() {
    on_device_with(at_plan, |device| {
        let answered = many_waits(device, METAL_MEASURED);
        assert_eq!(answered.len(), METAL_MEASURED);
        assert!(answered.iter().all(Result::is_ok), "{answered:?}");
    });
}

/// As many handler waits as the device has handler threads: the one that would leave no thread
/// free is refused at once, and the others return.
#[test]
fn the_handler_wait_that_would_leave_no_thread_free_is_refused() {
    on_device_with(at_plan, |device| {
        let answered = many_waits(device, THREADS);
        assert_eq!(answered.len(), THREADS);
        let refused = answered
            .iter()
            .filter(|answer| {
                **answer == Err(Error::WaitWouldDeadlock(DeadlockCause::HandlerThreadCap))
            })
            .count();
        let returned = answered.iter().filter(|answer| answer.is_ok()).count();
        assert_eq!((refused, returned), (1, THREADS - 1));
    });
}

/// A completed handler's wait for the scheduling of a command buffer of another queue that is
/// scheduled already returns (Metal: a completed handler's wait for scheduling returns).
#[test]
fn a_completed_handler_waits_for_a_later_command_buffer_s_scheduling() {
    on_device_with(at_plan, |device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let event: SharedEvent = device.new_shared_event().unwrap();
        // The first command buffer completes only after the later one is committed.
        let first = queue.command_buffer().unwrap();
        first
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let later = fill(&device.new_command_queue().unwrap(), &buffer, 0x71);
        let (sender, answers) = mpsc::channel();
        {
            let later = later.clone();
            add_handler(&first, false, move |_| {
                let scheduling = later.wait_until_scheduled();
                sender.send((scheduling, later.status())).unwrap();
            });
        }
        first.commit().unwrap();
        later.commit().unwrap();
        event.set_signaled_value(1);
        completed(&first);
        let (scheduling, status) = answers.recv_timeout(WAIT).unwrap();
        assert_eq!(scheduling, Ok(()));
        assert!(
            matches!(
                status,
                CommandBufferStatus::Scheduled | CommandBufferStatus::Completed
            ),
            "{status:?}"
        );
        completed(&later);
    });
}
