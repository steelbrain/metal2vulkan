//! `CommandBufferShared::finish_here`: a command buffer that runs none of the program's code as it
//! completes finishes on the queue waiter that saw its work complete, with no handler thread
//! woken, as Metal's completion wakes its waiting thread itself (`command_buffer/completion.rs`);
//! one with a completed handler, and any while the runtime holds a deallocator of the program's,
//! finishes on its queue's handler thread (`WorkCounts::completed_where_seen`). One whose only
//! code of the program's is its completion-seen handlers finishes where seen too, and runs them
//! there, once its status is final (`WorkCounts::completion_seen_handlers_where_seen`); so does one
//! with a completed handler besides, whose completed handler then runs on its queue's handler
//! thread (`Transition::seen_first`). Likewise
//! `CommandBufferShared::schedule`: a command buffer with no scheduled handler is scheduled on the
//! scheduler's thread as its last batch goes, and one with a scheduled handler on its queue's
//! handler thread (`WorkCounts::scheduled_where_submitted`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    CommandBuffer, CommandBufferStatus, CommandQueue, DeadlockCause, Device, DeviceKind,
    DeviceOptions, Error, HostMemory, ResourceOptions, ShaderCacheOptions,
};
use std::alloc::Layout;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The command buffers of each kind.
const ROUNDS: u64 = 20;

/// The no-copy buffer's bytes: a whole page of any host.
const BYTES: usize = 64 << 10;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-completion-here-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Page-aligned host memory of [`BYTES`], freed when dropped: declared first, after the device
/// that used it.
struct Host(NonNull<u8>);

impl Host {
    fn layout() -> Layout {
        Layout::from_size_align(BYTES, BYTES).unwrap()
    }

    fn new() -> Host {
        // SAFETY: the layout's size is not zero.
        let pointer = unsafe { std::alloc::alloc_zeroed(Host::layout()) };
        Host(NonNull::new(pointer).unwrap())
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Host::new`.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), Host::layout()) };
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// [`ROUNDS`] command buffers of `queue`, each a fill of 16 bytes, with a completed handler
/// where `handler`, each committed and waited for before the next: how many finished where
/// their completion was seen.
fn finished_where_seen(device: &Device, queue: &CommandQueue, handler: bool) -> u64 {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let before = device.work_counts().completed_where_seen;
    for round in 0..ROUNDS {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..16, round as u8).unwrap();
        blit.end_encoding().unwrap();
        if handler {
            command_buffer
                .add_completed_handler(Box::new(|_: &CommandBuffer| {}))
                .unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed().unwrap();
    }
    device.work_counts().completed_where_seen - before
}

/// Command buffers with no handler finish where their completion is seen: at least one of
/// [`ROUNDS`] (each's transition to `Scheduled` may be running on a handler thread as its work
/// completes, and it is then posted behind it). Those with a completed handler never do, nor do
/// any while a no-copy buffer's deallocator of the program's is held.
#[test]
fn a_command_buffer_with_no_handler_finishes_where_its_completion_is_seen() {
    let host = Host::new();
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let plain = finished_where_seen(&device, &queue, false);
    println!("finished where seen: {plain} of {ROUNDS} with no handler");
    assert!(plain >= 1, "none of {ROUNDS} finished where seen");
    assert_eq!(
        finished_where_seen(&device, &queue, true),
        0,
        "a command buffer with a completed handler finished off its queue's handler thread"
    );
    // SAFETY: the memory outlives the buffer (dropped after it, below), and nothing else uses it.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(host.0),
            BYTES as u64,
            ResourceOptions::default(),
            Some(Box::new(|_: NonNull<u8>, _: u64| {})),
        )
    };
    match made {
        Ok(buffer) => {
            assert_eq!(
                finished_where_seen(&device, &queue, false),
                0,
                "a command buffer finished off its queue's handler thread while a deallocator \
                 of the program's was held"
            );
            drop(buffer);
        }
        Err(error) => println!("no buffer over host memory: {error:?}"),
    }
    drop(queue);
    drop(device);
}

/// What a completion-seen handler saw as it ran.
#[derive(Debug)]
struct Seen {
    /// The thread it ran on.
    thread: String,
    /// It ran on one of the device's handler threads.
    on_handler_thread: bool,
    /// Its command buffer's status.
    status: CommandBufferStatus,
    /// Its command buffer's GPU end: positive, or refused where the device writes no timestamps.
    gpu_end: Result<f64, Error>,
    /// Its command buffer's completed handler had run already.
    after_completed: bool,
    /// What its wait for its own command buffer's completion answered.
    own_wait: Result<(), Error>,
    /// What its wait for its own command buffer's scheduling answered.
    own_scheduled_wait: Result<(), Error>,
}

/// [`ROUNDS`] command buffers of `queue`, each a fill of 16 bytes with a completion-seen handler,
/// and a completed handler where `completed`, each committed and waited for before the next: what
/// each completion-seen handler saw (each had run by its wait's return), and whether each
/// completed handler saw its completion-seen handler run before it, and ran on a handler thread;
/// and how many of the handlers the device counted run, and run where seen.
fn seen_handlers(
    device: &Device,
    queue: &CommandQueue,
    completed: bool,
) -> (Vec<Seen>, Vec<(bool, bool)>, u64, u64) {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let before = device.work_counts();
    let (mut seen, mut ordered) = (Vec::new(), Vec::new());
    for round in 0..ROUNDS {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..16, round as u8).unwrap();
        blit.end_encoding().unwrap();
        let record = Arc::new(Mutex::new(None));
        let (seen_ran, completed_ran) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        );
        {
            let (record, seen_ran, completed_ran) = (
                Arc::clone(&record),
                Arc::clone(&seen_ran),
                Arc::clone(&completed_ran),
            );
            command_buffer
                .add_completion_seen_handler(Box::new(move |buffer: &CommandBuffer| {
                    let thread = std::thread::current();
                    *record.lock().unwrap() = Some(Seen {
                        thread: thread.name().unwrap_or("").to_owned(),
                        on_handler_thread: crate::contexts::on_handler_thread(),
                        status: buffer.status(),
                        gpu_end: buffer.gpu_end_time(),
                        after_completed: completed_ran.load(Ordering::SeqCst),
                        own_wait: buffer.wait_until_completed(),
                        own_scheduled_wait: buffer.wait_until_scheduled(),
                    });
                    seen_ran.store(true, Ordering::SeqCst);
                }))
                .unwrap();
        }
        let (saw_seen, on_handler_thread) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        );
        if completed {
            let (seen_ran, completed_ran, saw_seen, on_handler_thread) = (
                Arc::clone(&seen_ran),
                Arc::clone(&completed_ran),
                Arc::clone(&saw_seen),
                Arc::clone(&on_handler_thread),
            );
            command_buffer
                .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                    saw_seen.store(seen_ran.load(Ordering::SeqCst), Ordering::SeqCst);
                    let here = crate::contexts::on_handler_thread();
                    on_handler_thread.store(here, Ordering::SeqCst);
                    completed_ran.store(true, Ordering::SeqCst);
                }))
                .unwrap();
        }
        command_buffer.commit().unwrap();
        // Refused once committed, as the other handlers are.
        assert!(command_buffer
            .add_completion_seen_handler(Box::new(|_: &CommandBuffer| {}))
            .is_err());
        command_buffer.wait_until_completed().unwrap();
        assert!(
            seen_ran.load(Ordering::SeqCst),
            "the wait returned before the handler ran"
        );
        seen.push(record.lock().unwrap().take().unwrap());
        if completed {
            ordered.push((
                saw_seen.load(Ordering::SeqCst),
                on_handler_thread.load(Ordering::SeqCst),
            ));
        }
    }
    let after = device.work_counts();
    (
        seen,
        ordered,
        after.completion_seen_handlers_run - before.completion_seen_handlers_run,
        after.completion_seen_handlers_where_seen - before.completion_seen_handlers_where_seen,
    )
}

/// A command buffer whose only handler is a completion-seen one finishes where its completion is
/// seen, and runs the handler there, on its queue's waiter, its status `Completed` and its GPU
/// times placed: at least one of [`ROUNDS`] (as for a command buffer with no handler, an earlier
/// transition of its queue may hold the context), each counted, with no handler thread woken.
/// With a completed handler besides, so does it, at least once, before the completed handler,
/// which runs on its queue's handler thread. A wait in one for what has not happened (its own
/// completion) is refused,
/// one for what happened (its own scheduling) returns, and the command buffer's wait returns only
/// once it ran.
#[test]
fn a_completion_seen_handler_runs_where_the_completion_is_seen() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let refused = Err(Error::WaitWouldDeadlock(
        DeadlockCause::CompletionSeenHandler,
    ));
    let (seen, _, run, where_seen) = seen_handlers(&device, &queue, false);
    let on_waiters = seen.iter().filter(|seen| !seen.on_handler_thread).count() as u64;
    println!("completion-seen handlers where seen: {where_seen} of {ROUNDS}");
    assert_eq!(run, ROUNDS);
    assert!(
        where_seen >= 1,
        "none of {ROUNDS} ran where its completion was seen"
    );
    assert_eq!(where_seen, on_waiters, "{seen:#?}");
    for seen in &seen {
        if !seen.on_handler_thread {
            assert!(
                seen.thread.starts_with("metal2vulkan-queue-wait-"),
                "ran on {}",
                seen.thread
            );
        }
        assert_eq!(seen.status, CommandBufferStatus::Completed);
        assert!(
            seen.gpu_end.as_ref().map_or(true, |&end| end > 0.0),
            "{seen:?}"
        );
        assert_eq!(seen.own_wait, refused);
        assert_eq!(seen.own_scheduled_wait, Ok(()));
    }
    let (seen, ordered, run, where_seen) = seen_handlers(&device, &queue, true);
    let on_waiters = seen.iter().filter(|seen| !seen.on_handler_thread).count() as u64;
    println!(
        "with a completed handler, completion-seen handlers where seen: {where_seen} of {ROUNDS}"
    );
    assert_eq!(run, ROUNDS);
    assert!(
        where_seen >= 1,
        "none of {ROUNDS} with a completed handler ran where its completion was seen"
    );
    assert_eq!(where_seen, on_waiters, "{seen:#?}");
    assert!(seen.iter().all(|seen| !seen.after_completed), "{seen:#?}");
    assert!(
        ordered.iter().all(|&(ordered, _)| ordered),
        "a completed handler ran first"
    );
    assert!(
        ordered
            .iter()
            .all(|&(_, on_handler_thread)| on_handler_thread),
        "a completed handler ran off the handler threads"
    );
    for seen in &seen {
        assert_eq!(seen.status, CommandBufferStatus::Completed);
        assert_eq!(seen.own_wait, refused);
    }
    drop(queue);
    drop(device);
}

/// What a command buffer of [`scheduled_where_submitted`] has besides its fill.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Besides {
    Nothing,
    ScheduledHandler,
    /// A scheduled signal of the shared event, of the round's number and one.
    ScheduledSignal,
}

/// [`ROUNDS`] command buffers of `queue`, each a fill of 16 bytes with what `besides` says, each
/// committed and waited for before the next: how many were scheduled where their last batch was
/// submitted. Each scheduled signal set its value by its wait's return.
fn scheduled_where_submitted(device: &Device, queue: &CommandQueue, besides: Besides) -> u64 {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let event = device.new_shared_event().unwrap();
    let before = device.work_counts().scheduled_where_submitted;
    for round in 0..ROUNDS {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..16, round as u8).unwrap();
        blit.end_encoding().unwrap();
        match besides {
            Besides::Nothing => {}
            Besides::ScheduledHandler => command_buffer
                .add_scheduled_handler(Box::new(|_: &CommandBuffer| {}))
                .unwrap(),
            Besides::ScheduledSignal => command_buffer
                .encode_signal_event_scheduled(&event, round + 1)
                .unwrap(),
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_scheduled().unwrap();
        if besides == Besides::ScheduledSignal {
            assert_eq!(event.signaled_value(), round + 1);
        }
        command_buffer.wait_until_completed().unwrap();
    }
    device.work_counts().scheduled_where_submitted - before
}

/// Command buffers with no scheduled handler are scheduled where their last batch is submitted:
/// at least one of [`ROUNDS`] (the one before's transition to `Completed` may still hold its
/// queue's context as the next one's last batch goes, which is then posted behind it), those with
/// a scheduled signal too, which wakes no handler thread. Those with a scheduled handler never
/// are.
#[test]
fn a_command_buffer_with_no_scheduled_handler_is_scheduled_where_submitted() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let plain = scheduled_where_submitted(&device, &queue, Besides::Nothing);
    println!("scheduled where submitted: {plain} of {ROUNDS} with no handler");
    assert!(plain >= 1, "none of {ROUNDS} was scheduled where submitted");
    let signalled = scheduled_where_submitted(&device, &queue, Besides::ScheduledSignal);
    println!("scheduled where submitted: {signalled} of {ROUNDS} with a scheduled signal");
    assert!(
        signalled >= 1,
        "none of {ROUNDS} was scheduled where submitted"
    );
    assert_eq!(
        scheduled_where_submitted(&device, &queue, Besides::ScheduledHandler),
        0,
        "a command buffer with a scheduled handler was scheduled off its queue's handler thread"
    );
    drop(queue);
    drop(device);
}
