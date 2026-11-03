//! Device tests of a command queue's limit on uncompleted command buffers (`queue.rs`), on the
//! device the cfg selects, under the validation layer.
//!
//! The answers expected are Metal's, measured on an M3 Max with macOS 26.7 and an M4 with macOS
//! 27.0.1, with and without API validation: a queue holds 64 uncompleted command buffers, or the
//! count it was made with; a make past them blocks until one completes (after its completed
//! handlers, not waiting for its release: probe `slotdealloc`, the M3 Max under macOS 27.0.1) or is
//! dropped uncommitted, and never answers nil; a count of 0 stops Metal's validation, where the
//! runtime refuses it typed; and another queue's makes go on. Where Metal's blocked make never
//! returns, from a handler of the full queue itself, the runtime refuses it (`WaitWouldDeadlock`).
//! No wait here can hang: each blocked make runs on a thread of its own and is answered within
//! [`WAIT`].

use super::host_memory::Mapping;
use super::{
    layer_reported_nothing_at_all, validated_device, validated_device_with, Messages, ScratchCache,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::QueueSharingPolicy;
use metal2vulkan_runtime::{
    BufferContents, BufferDescriptor, CommandBuffer, CommandQueue, DeadlockCause, Deallocator,
    Degradation, Device, Error, EventRef, HostMemory, Invalid, KeepAlive, Problem, ResourceOptions,
    SharedEvent,
};
use std::sync::mpsc;
use std::time::Duration;

/// The longest a command buffer, or a make, of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// How long a make must stay blocked to count as blocked.
const BLOCKED: Duration = Duration::from_millis(300);

/// Metal's default `maxCommandBufferCount`, measured.
const METAL_DEFAULT: usize = 64;

/// How long the completed handler of
/// [`a_command_buffer_s_slot_comes_back_as_its_completed_handlers_return`] runs.
const HANDLING: Duration = Duration::from_millis(300);

/// How long its no-copy buffer's deallocator runs.
const DEALLOCATING: Duration = Duration::from_secs(2);

/// Run `test` on a validated device, then check that once everything is dropped (the device
/// last) the layer reported nothing.
fn on_device(test: impl FnOnce(&Device)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// A make of `queue`'s next command buffer on a thread of its own; its answer comes on the
/// receiver.
fn make_aside(queue: &CommandQueue) -> mpsc::Receiver<Result<CommandBuffer, Error>> {
    let (sender, answer) = mpsc::channel();
    let queue = queue.clone();
    std::thread::spawn(move || sender.send(queue.command_buffer()).unwrap());
    answer
}

/// The make's answer is still to come after [`BLOCKED`].
fn still_blocked(answer: &mpsc::Receiver<Result<CommandBuffer, Error>>) -> bool {
    matches!(
        answer.recv_timeout(BLOCKED),
        Err(mpsc::RecvTimeoutError::Timeout)
    )
}

/// The make's answer, which must come within [`WAIT`].
fn answered(answer: &mpsc::Receiver<Result<CommandBuffer, Error>>) -> Result<CommandBuffer, Error> {
    answer
        .recv_timeout(WAIT)
        .expect("a blocked make answered within the wait")
}

/// A count of 0 is refused typed, where Metal's validation ends the process; a queue holds the
/// count it was made with, or Metal's 64, uncompleted command buffers; the next make blocks, and
/// returns once one held uncommitted is dropped; another queue's make goes on meanwhile.
#[test]
fn a_queue_holds_as_many_uncompleted_command_buffers_as_metal_s() {
    on_device(|device| {
        match device.new_command_queue_with_max_command_buffer_count(0) {
            Err(Error::Invalid(invalid)) => assert_eq!(
                invalid,
                Invalid::new(
                    "new_command_queue_with_max_command_buffer_count",
                    Problem::MaxCommandBufferCount
                )
            ),
            other => panic!("a count of 0 answered {other:?}"),
        }
        let other = device
            .new_command_queue_with_max_command_buffer_count(1)
            .unwrap();
        for count in [None, Some(1), Some(4), Some(65)] {
            let queue = match count {
                None => device.new_command_queue().unwrap(),
                Some(count) => device
                    .new_command_queue_with_max_command_buffer_count(count)
                    .unwrap(),
            };
            let count = count.unwrap_or(METAL_DEFAULT);
            let mut held: Vec<CommandBuffer> = (0..count)
                .map(|_| queue.command_buffer().unwrap())
                .collect();
            let next = make_aside(&queue);
            assert!(still_blocked(&next), "a make past {count} returned");
            // Another queue's make goes on (and its slot comes back with it).
            drop(other.command_buffer().unwrap());
            held.pop();
            answered(&next).unwrap();
        }
    });
}

/// A committed command buffer holds its slot while it waits and runs, and gives it back once it
/// completed: a make blocked behind two committed command buffers waiting for an event returns
/// once they ran, and a make after `wait_until_completed` returned never blocks.
#[test]
fn a_committed_command_buffer_gives_its_slot_back_once_completed() {
    on_device(|device| {
        let queue = device
            .new_command_queue_with_max_command_buffer_count(2)
            .unwrap();
        let event: SharedEvent = device.new_shared_event().unwrap();
        let waiting: Vec<CommandBuffer> = (0..2)
            .map(|_| {
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .encode_wait_for_event(EventRef::from(&event), 1)
                    .unwrap();
                command_buffer.commit().unwrap();
                command_buffer
            })
            .collect();
        let next = make_aside(&queue);
        assert!(still_blocked(&next), "a make past two committed returned");
        event.set_signaled_value(1);
        let made = answered(&next).unwrap();
        for command_buffer in &waiting {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        drop(made);
        let one = device
            .new_command_queue_with_max_command_buffer_count(1)
            .unwrap();
        for round in 0..50 {
            let command_buffer = one.command_buffer().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            drop(command_buffer);
            let next = make_aside(&one);
            assert!(
                answered(&next).is_ok(),
                "round {round}: the make after the wait"
            );
        }
    });
}

/// A committed command buffer gives its slot back as its last completed handler returns, before
/// what its commands used is released, as Metal's does (probe `slotdealloc`, the M3 Max under
/// macOS 27.0.1): a make blocked on its full queue returns once a long completed handler
/// returned, never while it runs, and while the deallocator of a no-copy buffer the command
/// buffer held last still runs, whether or not the program still holds the command buffer. The
/// buffer has a keep-alive, which its mapping lives in, as the consumer's do: its deallocator then
/// runs in the release. The slot's return and the make's wait are counted.
#[test]
fn a_command_buffer_s_slot_comes_back_as_its_completed_handlers_return() {
    on_device(|device| {
        let queue = device
            .new_command_queue_with_max_command_buffer_count(1)
            .unwrap();
        for kept in [false, true] {
            const LENGTH: usize = 1 << 16;
            let (sender, steps) = mpsc::channel();
            let mapping = Mapping::anonymous(LENGTH);
            let pointer = mapping.pointer();
            let in_deallocator = sender.clone();
            let deallocator: Deallocator = Box::new(move |_, _| {
                std::thread::sleep(DEALLOCATING);
                in_deallocator.send("the deallocator ended").unwrap();
            });
            let keep_alive: KeepAlive = Box::new(mapping);
            // SAFETY: the mapping lives in the keep-alive, which the import holds until its pages
            // are freed.
            let memory = unsafe { HostMemory::new(pointer) };
            let contents = BufferContents::NoCopy {
                memory,
                deallocator: Some(deallocator),
                keep_alive: Some(keep_alive),
            };
            let made = device.new_buffer_with_descriptor(BufferDescriptor::new(
                LENGTH as u64,
                ResourceOptions::default(),
                contents,
            ));
            let buffer = match made {
                Ok(buffer) => buffer,
                Err(error) => {
                    eprintln!("QUEUE slot: no buffer over host memory: {error:?}");
                    return;
                }
            };
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..LENGTH as u64, 7).unwrap();
            blit.end_encoding().unwrap();
            let in_handler = sender.clone();
            command_buffer
                .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                    std::thread::sleep(HANDLING);
                    in_handler.send("the completed handler ended").unwrap();
                }))
                .unwrap();
            let before = device.work_counts();
            command_buffer.commit().unwrap();
            drop(buffer);
            let held = kept.then_some(command_buffer);
            let next = make_aside(&queue);
            let made = answered(&next).unwrap();
            let seen: Vec<&str> = steps.try_iter().collect();
            assert_eq!(
                seen,
                ["the completed handler ended"],
                "kept {kept}: what ran before the blocked make returned"
            );
            assert_eq!(
                steps.recv_timeout(WAIT),
                Ok("the deallocator ended"),
                "kept {kept}: the deallocator ran"
            );
            drop((made, held));
            let after = device.work_counts();
            assert_eq!(
                after.slot_waits - before.slot_waits,
                1,
                "kept {kept}: the blocked make counted"
            );
            assert!(
                after.slot_returns > before.slot_returns,
                "kept {kept}: the slot's return counted"
            );
            let returning =
                Duration::from_nanos(after.slot_return_nanos - before.slot_return_nanos);
            assert!(
                returning < DEALLOCATING,
                "kept {kept}: the slot came back {returning:?} after the completed handlers"
            );
            eprintln!(
                "QUEUE slot: kept {kept}: back {returning:?} after the handlers; the make waited \
                 {:?}",
                Duration::from_nanos(after.slot_wait_nanos - before.slot_wait_nanos)
            );
        }
    });
}

/// Makes from handlers. On the handler's own full queue, where the only slot is its own command
/// buffer's, which comes back after its handlers, Metal's make never returns: the runtime refuses
/// it at once, from the scheduled and the completed handler. Where a command buffer held
/// uncommitted may still be dropped, the handler's make waits, and returns once it is. On
/// another full queue, the make blocks until that queue's command buffer completed, and returns.
#[test]
fn a_handler_s_make_on_a_full_queue_returns_where_a_slot_can_come_back() {
    on_device(|device| {
        let (sender, answers) = mpsc::channel();
        // Its own queue, full with its own command buffer.
        let own = device
            .new_command_queue_with_max_command_buffer_count(1)
            .unwrap();
        let command_buffer = own.command_buffer().unwrap();
        for scheduled in [true, false] {
            let (queue, sender) = (own.clone(), sender.clone());
            let handler = Box::new(move |_: &CommandBuffer| {
                sender.send(queue.command_buffer().map(drop)).unwrap();
            });
            if scheduled {
                command_buffer.add_scheduled_handler(handler).unwrap();
            } else {
                command_buffer.add_completed_handler(handler).unwrap();
            }
        }
        command_buffer.commit().unwrap();
        for _ in 0..2 {
            let answer = answers.recv_timeout(WAIT).unwrap();
            assert!(
                matches!(
                    answer,
                    Err(Error::WaitWouldDeadlock(DeadlockCause::OwnQueue))
                ),
                "a make from a handler of its own full queue answered {answer:?}"
            );
        }
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        drop(command_buffer);

        // Its own queue, its other slot held by a command buffer not committed.
        let shared = device
            .new_command_queue_with_max_command_buffer_count(2)
            .unwrap();
        let open = shared.command_buffer().unwrap();
        let command_buffer = shared.command_buffer().unwrap();
        let (queue, in_handler) = (shared.clone(), sender.clone());
        command_buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                in_handler.send(queue.command_buffer().map(drop)).unwrap();
            }))
            .unwrap();
        command_buffer.commit().unwrap();
        assert!(
            matches!(
                answers.recv_timeout(BLOCKED),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "the handler's make returned while every slot was held"
        );
        drop(open);
        answers.recv_timeout(WAIT).unwrap().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        drop(command_buffer);

        // Another queue, full with a command buffer waiting for an event.
        let full = device
            .new_command_queue_with_max_command_buffer_count(1)
            .unwrap();
        let event: SharedEvent = device.new_shared_event().unwrap();
        let waiting = full.command_buffer().unwrap();
        waiting
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        waiting.commit().unwrap();
        let command_buffer = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        let (queue, in_handler) = (full.clone(), sender.clone());
        command_buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                in_handler.send(queue.command_buffer().map(drop)).unwrap();
            }))
            .unwrap();
        command_buffer.commit().unwrap();
        assert!(
            matches!(
                answers.recv_timeout(BLOCKED),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "the handler's make on another full queue returned before its slot came back"
        );
        event.set_signaled_value(1);
        answers.recv_timeout(WAIT).unwrap().unwrap();
        waiting.wait_until_completed_for(WAIT).unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    });
}

/// The kind a queue made for presenting takes on a device that reserved no `VkQueue` for it
/// ([`a_presenting_queue_s_lock_holds_no_other_queue_where_the_device_reserved_one`]).
pub(super) const SHARED_PRESENT_QUEUES: [Degradation; 1] = [Degradation::SharedPresentQueue];

/// A queue made for presenting submits to a `VkQueue` no other queue shares where the device
/// reserved one (the queue family has queues past the four the round robin takes: NVIDIA's), so
/// its lock, held as a present that blocks holds it, holds no other queue's submissions: each
/// other queue's fill completes meanwhile. On a device that reserved none (lavapipe, RADV,
/// MoltenVK) it shares a `VkQueue` and the device announces `shared-present-queue` once, at the
/// first such queue only.
#[test]
fn a_presenting_queue_s_lock_holds_no_other_queue_where_the_device_reserved_one() {
    use metal2vulkan_runtime::ResourceOptions;
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let shared_kinds = || -> usize {
            messages
                .take_notices()
                .into_iter()
                .filter_map(|message| message.degradation)
                .filter(|notice| SHARED_PRESENT_QUEUES.contains(&notice.degradation))
                .count()
        };
        shared_kinds();
        let queues: Vec<CommandQueue> = (0..4)
            .map(|_| device.new_command_queue().unwrap())
            .collect();
        let presenting = device.new_command_queue_for_presenting().unwrap();
        // SAFETY: the handle is compared only, inside and after the call; nothing is submitted.
        let raw = |queue: &CommandQueue| unsafe { queue.with_vk_queue(|raw, _| raw) };
        let reserved = queues.iter().all(|queue| raw(queue) != raw(&presenting));
        let second = device.new_command_queue_for_presenting().unwrap();
        assert_eq!(
            shared_kinds(),
            usize::from(!reserved),
            "announced once, where the device reserved no VkQueue"
        );
        if reserved {
            assert!(queues.iter().all(|queue| raw(queue) != raw(&second)));
        }
        let (held, holding) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let holder = {
            let presenting = presenting.clone();
            std::thread::spawn(move || {
                // SAFETY: the closure keeps no handle and calls nothing of the runtime's; it waits
                // only for this test's release.
                unsafe {
                    presenting.with_vk_queue(|_, _| {
                        held.send(()).unwrap();
                        let _ = released.recv_timeout(WAIT);
                    })
                }
            })
        };
        holding.recv_timeout(WAIT).unwrap();
        if reserved {
            let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
            for queue in &queues {
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&buffer, 0..4096, 0x5a).unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                assert!(
                    command_buffer.wait_until_completed_for(WAIT).is_ok(),
                    "another queue's command buffer completes while the presenting queue's \
                     VkQueue is held"
                );
            }
        }
        release.send(()).unwrap();
        holder.join().unwrap();
    }
    layer_reported_nothing_at_all(&messages);
}

/// The kind a command queue takes as it moves onto a `VkQueue` with another queue's work in flight
/// ([`a_queue_joining_another_s_busy_vk_queue_is_announced_once`]).
pub(super) const QUEUE_SHARING: [Degradation; 1] = [Degradation::QueueSharing];

/// A command queue that makes a command buffer while it holds no other uncompleted one, beside a
/// queue whose short command buffers keep its `VkQueue` busy, joins that `VkQueue` where the device
/// shares (`QueueSharingPolicy::Adaptive`, set here over the lane's `exact_answers()`, on a device
/// whose check finds its GPU idle between `VkQueue`s: NVIDIA's), and the device announces
/// `queue-sharing` once however many queues join: each notice a join (`WorkCounts::queue_joins`,
/// above 0 where the device shares), never one without. On a device that shares no `VkQueue`
/// (lavapipe, RADV, MoltenVK) nothing joins and nothing is announced, and the test says why.
#[test]
fn a_queue_joining_another_s_busy_vk_queue_is_announced_once() {
    use metal2vulkan_runtime::ResourceOptions;
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.queue_sharing = QueueSharingPolicy::Adaptive;
        })
        .unwrap();
        assert!(device.settle_queue_switch_check(WAIT), "the check settles");
        let shares = device.shares_vk_queues();
        if !shares {
            println!(
                "{} shares no VkQueue, so no queue joins: {}",
                device.name(),
                device.queue_switch_check_report()
            );
        }
        let sharing_kinds = || -> usize {
            messages
                .take_notices()
                .into_iter()
                .filter_map(|message| message.degradation)
                .filter(|notice| QUEUE_SHARING.contains(&notice.degradation))
                .count()
        };
        // 16 MiB a fill: tens of microseconds each on the RTX 5080 Laptop GPU, so the busy queue's
        // oldest batch is young when a joiner looks and no command buffer runs long.
        let buffer = device
            .new_buffer(16 << 20, ResourceOptions::private())
            .unwrap();
        let fill = |queue: &CommandQueue| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..(16 << 20), 0x5a).unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
        };
        let busy = device.new_command_queue().unwrap();
        let joiners: Vec<CommandQueue> = (0..3)
            .map(|_| device.new_command_queue().unwrap())
            .collect();
        for joiner in &joiners {
            let kept: Vec<CommandBuffer> = (0..24).map(|_| fill(&busy)).collect();
            // Submitted, so the busy `VkQueue` has work in flight as the joiner makes its own.
            kept.last().unwrap().wait_until_scheduled().unwrap();
            let joined = fill(joiner);
            joined.wait_until_completed_for(WAIT).unwrap();
            for command_buffer in kept {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
            }
        }
        let joins = device.work_counts().queue_joins;
        assert_eq!(
            joins > 0,
            shares,
            "{joins} joins where the device shares: {shares}"
        );
        assert_eq!(
            sharing_kinds(),
            usize::from(joins > 0),
            "announced once, where a queue joined ({joins} joins)"
        );
    }
    layer_reported_nothing_at_all(&messages);
}
