//! `Scheduler::submit_before_waiting`: a thread about to block for a command buffer's completion,
//! or for its scheduling, submits for itself what still waits in the scheduler's inbox, instead of
//! waiting for a wake and a whole pass (`WorkCounts::scheduler_passes_by_waiters`,
//! `WorkCounts::batches_submitted_by_waiters`); and it submits nothing of a command buffer with a
//! submission or a scheduled handler, which runs the program's code on the thread that takes its
//! last batch, or that is scheduled where its last batch goes: it passes and leaves every batch
//! of it to the scheduler's thread, which runs those handlers, as it does today.
//!
//! Nor does it submit a batch of a queue whose batches wait in the scheduler's group
//! (`command_buffer/group.rs`), which is the scheduler thread's to hold and to send: under the
//! default `CompletionPolicy::Grouped`, on a driver measured to merge a submission that signals
//! nothing into the next (`DriverFact::SubmissionsWithoutSignalsMerge`), that is every queue. So
//! the waiter's own submissions are counted under `CompletionPolicy::Exact`, where no queue's
//! batches group on any driver, and under the default each test checks that a waiting thread
//! submits nothing where the device's batches group, and submits as under `Exact` elsewhere.
//!
//! Which thread takes a commit in is a race between the waiting thread and the scheduler's,
//! which the commit wakes: on a loaded host the woken thread may win every one of them. So each
//! round counted for a waiter's own pass holds the scheduler's thread first ([`Held`]), in a
//! submission handler of another queue's command buffer, which holds back the thread's passes
//! (`CommandBuffer::add_submission_handler`): the waiting thread's pass is then the only one that
//! can take its commit in, and each round's pass and submissions are counted, whoever would have
//! won.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::command_buffer::merges_submissions;
use crate::{
    Buffer, CommandBuffer, CommandQueue, Device, DeviceKind, DeviceOptions, ResourceOptions,
    ShaderCacheOptions,
};
use metal2vulkan::device::{CompletionPolicy, DevicePolicy};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread::ThreadId;
use std::time::Duration;

/// The command buffers of each kind: each committed and waited for before the next, so each one's
/// wait makes a pass for its commit alone.
const ROUNDS: u64 = 32;

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-waiter-submits-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`, its policy the default's with `completion`.
fn device(cache: &Path, completion: CompletionPolicy) -> Device {
    let mut policy = DevicePolicy::default();
    policy.completion = completion;
    let options = DeviceOptions {
        policy,
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

/// The scheduler's thread held in a submission handler of a command buffer of `side`, which runs
/// on that thread once it submitted the command buffer and holds back its next pass, and with it
/// every commit made meanwhile (`CommandBuffer::add_submission_handler`). While it is held, a
/// commit is taken in only by a thread about to block for it. Dropping it lets the handler return
/// and waits until its command buffer completed.
struct Held {
    command_buffer: CommandBuffer,
    release: Option<mpsc::Sender<()>>,
}

impl Held {
    /// Hold the thread: commit a fill of `scratch` on `side` whose submission handler says it
    /// began, and then waits until it is let go; return once it began.
    fn new(side: &CommandQueue, scratch: &Buffer) -> Held {
        let command_buffer = side.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(scratch, 0..16, 0).unwrap();
        blit.end_encoding().unwrap();
        let (began, began_seen) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let handler = Box::new(move |_: &CommandBuffer| {
            began.send(()).unwrap();
            // Ends once the sender is dropped.
            let _ = released.recv();
        });
        command_buffer.add_submission_handler(handler).unwrap();
        command_buffer.commit().unwrap();
        began_seen.recv().unwrap();
        Held {
            command_buffer,
            release: Some(release),
        }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        drop(self.release.take());
        let _ = self.command_buffer.wait_until_completed();
    }
}

/// What a command buffer of [`rounds`] has besides its fill.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Besides {
    Nothing,
    /// A submission handler, which runs on the thread that takes its last batch.
    SubmissionHandler,
    /// A scheduled handler, which its queue's handler thread runs.
    ScheduledHandler,
}

/// What [`rounds`] counted: the passes threads about to block made, the batches they submitted,
/// and the threads the handlers ran on.
struct Passed {
    passes: u64,
    batches: u64,
    handler_threads: Vec<ThreadId>,
}

/// [`ROUNDS`] command buffers of `queue`, each a fill of 16 bytes with what `besides` says, each
/// committed and waited for before the next, the scheduler's thread [`Held`] by a command buffer
/// of `side`, where one is given, from before the commit until the waiting thread's pass. That
/// pass is made by `wait_until_completed_for` with no time to wait, which passes as
/// `wait_until_completed` does (`CommandBufferShared::waits_for_completion`) and returns: the
/// thread is let go before the wait for the completion, which a command buffer with a handler
/// waits for it for.
fn rounds(
    device: &Device,
    queue: &CommandQueue,
    side: Option<&CommandQueue>,
    besides: Besides,
) -> Passed {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let scratch = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let threads: &'static Mutex<Vec<ThreadId>> = Box::leak(Box::new(Mutex::new(Vec::new())));
    let before = device.work_counts();
    for round in 0..ROUNDS {
        let held = side.map(|side| Held::new(side, &scratch));
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..16, round as u8).unwrap();
        blit.end_encoding().unwrap();
        let handler = Box::new(|_: &CommandBuffer| {
            (threads.lock().unwrap()).push(std::thread::current().id());
        });
        match besides {
            Besides::Nothing => {}
            Besides::SubmissionHandler => command_buffer.add_submission_handler(handler).unwrap(),
            Besides::ScheduledHandler => command_buffer.add_scheduled_handler(handler).unwrap(),
        }
        command_buffer.commit().unwrap();
        // Completed or not, the pass was made.
        let _ = command_buffer.wait_until_completed_for(Duration::ZERO);
        drop(held);
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(command_buffer.error(), None);
    }
    let after = device.work_counts();
    Passed {
        passes: after.scheduler_passes_by_waiters - before.scheduler_passes_by_waiters,
        batches: after.batches_submitted_by_waiters - before.batches_submitted_by_waiters,
        handler_threads: threads.lock().unwrap().clone(),
    }
}

/// A thread that commits and then waits submits its own batch: with the scheduler's thread
/// [`Held`], each of [`ROUNDS`] passes and submits it (unheld, the scheduler's thread, woken by the
/// commit, may take it in first, and then the waiting thread finds the inbox empty and sleeps).
/// Under `CompletionPolicy::Exact`, so on every driver; under the default, as
/// [`under_the_default`] says.
///
/// A command buffer with a submission or a scheduled handler is never submitted by a waiting
/// thread: it passes in each round, taking the commit into the order, and submits no batch of it. Each handler
/// ran, and on none of the waiting threads: Metal runs a command buffer's handlers on its own
/// threads, and one that took a lock the waiting thread holds would deadlock where Metal's would
/// not.
#[test]
fn a_thread_about_to_block_submits_what_it_committed_unless_a_handler_must_run() {
    let scratch = Scratch::new();
    let device = device(&scratch.0, CompletionPolicy::Exact);
    let queue = device.new_command_queue().unwrap();
    let side = device.new_command_queue().unwrap();
    let waiting = std::thread::current().id();
    let plain = rounds(&device, &queue, Some(&side), Besides::Nothing);
    println!(
        "submitted by waiters: {} batches over {} passes of {ROUNDS} waits",
        plain.batches, plain.passes
    );
    assert_eq!(
        plain.passes, ROUNDS,
        "a waiting thread did not pass for its own commit while the scheduler's thread was held"
    );
    assert!(
        plain.batches >= ROUNDS,
        "a waiting thread's pass did not submit its own batch"
    );
    for besides in [Besides::SubmissionHandler, Besides::ScheduledHandler] {
        let held = rounds(&device, &queue, Some(&side), besides);
        println!(
            "with {besides:?}: {} batches over {} passes of {ROUNDS} waits",
            held.batches, held.passes
        );
        assert_eq!(
            held.batches, 0,
            "a waiting thread submitted a batch of a command buffer with {besides:?}"
        );
        assert_eq!(
            held.passes, ROUNDS,
            "a waiting thread did not pass with {besides:?}, so the batches it held out prove \
             nothing"
        );
        assert_eq!(
            held.handler_threads.len(),
            ROUNDS as usize,
            "{besides:?} did not run once a round"
        );
        assert!(
            !held.handler_threads.contains(&waiting),
            "a {besides:?} ran on the thread waiting in wait_until_completed"
        );
    }
    drop(side);
    drop(queue);
    drop(device);
    under_the_default(&scratch.0, "completion", |device, queue, side| {
        rounds(device, queue, side, Besides::Nothing).batches
    });
}

/// Under the default `CompletionPolicy::Grouped`: where the device's batches group (its driver
/// merges a submission that signals nothing into the next, `merges_submissions`), every batch of
/// a queue is the scheduler thread's to hold in its group, so a thread waiting for its `wait`
/// submits none of [`ROUNDS`] (`scheduler::plain_ready`) and the scheduler's thread submits them
/// all; elsewhere, the scheduler's thread [`Held`], it submits each, as under `Exact`. `submitted` runs the plain rounds
/// and answers the batches the waiters submitted; it is given the queue whose command buffers
/// hold the scheduler's thread, where it holds it.
fn under_the_default(
    cache: &Path,
    wait: &str,
    submitted: impl Fn(&Device, &CommandQueue, Option<&CommandQueue>) -> u64,
) {
    let device = device(cache, CompletionPolicy::Grouped);
    let queue = device.new_command_queue().unwrap();
    let side = device.new_command_queue().unwrap();
    let groups = merges_submissions(&device.properties().driver_facts);
    let batches = submitted(&device, &queue, (!groups).then_some(&side));
    println!(
        "under the default, batches grouping {groups}: submitted by waiters for the {wait}: \
         {batches} batches of {ROUNDS} waits"
    );
    match groups {
        true => assert_eq!(
            batches, 0,
            "a thread waiting for the {wait} submitted a batch of a queue whose batches group"
        ),
        false => assert!(
            batches >= ROUNDS,
            "a thread waiting for the {wait} did not submit its own batch under the default, on \
             a device whose batches do not group"
        ),
    }
    drop(side);
    drop(queue);
    drop(device);
}

/// What one round of [`scheduled_rounds`] counted, over the wait for the scheduling alone.
struct Scheduled {
    passes: u64,
    batches: u64,
    handler_threads: Vec<ThreadId>,
}

/// [`ROUNDS`] command buffers of `queue`, each a fill of 16 bytes with what `besides` says, each
/// committed and then waited for, counting the passes and batches of the wait for the
/// **scheduling** alone: the counts are read around `wait_until_scheduled`, before the wait for
/// the completion, which would make a pass of its own. Where `side` is given, the scheduler's
/// thread is [`Held`] by a command buffer of it from before the commit until the wait for the
/// scheduling returned: only for a command buffer the waiting thread may submit itself, as one
/// it leaves to the held thread is never scheduled meanwhile.
fn scheduled_rounds(
    device: &Device,
    queue: &CommandQueue,
    side: Option<&CommandQueue>,
    besides: Besides,
) -> Scheduled {
    let target = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let scratch = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let threads: &'static Mutex<Vec<ThreadId>> = Box::leak(Box::new(Mutex::new(Vec::new())));
    let (mut passes, mut batches) = (0, 0);
    for round in 0..ROUNDS {
        let held = side.map(|side| Held::new(side, &scratch));
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..16, round as u8).unwrap();
        blit.end_encoding().unwrap();
        let handler = Box::new(|_: &CommandBuffer| {
            (threads.lock().unwrap()).push(std::thread::current().id());
        });
        match besides {
            Besides::Nothing => {}
            Besides::SubmissionHandler => command_buffer.add_submission_handler(handler).unwrap(),
            Besides::ScheduledHandler => command_buffer.add_scheduled_handler(handler).unwrap(),
        }
        command_buffer.commit().unwrap();
        let before = device.work_counts();
        command_buffer.wait_until_scheduled().unwrap();
        let after = device.work_counts();
        drop(held);
        passes += after.scheduler_passes_by_waiters - before.scheduler_passes_by_waiters;
        batches += after.batches_submitted_by_waiters - before.batches_submitted_by_waiters;
        command_buffer.wait_until_completed().unwrap();
        assert_eq!(command_buffer.error(), None);
    }
    Scheduled {
        passes,
        batches,
        handler_threads: threads.lock().unwrap().clone(),
    }
}

/// A thread that commits and then waits for the **scheduling** submits its own batch too: with the
/// scheduler's thread [`Held`], each of [`ROUNDS`] does, counted over that wait alone rather than
/// over the wait for the completion that follows it. Under `CompletionPolicy::Exact`, so on every driver; under the
/// default, as [`under_the_default`] says.
///
/// Its claim on the batches behind it is the weaker one -- the scheduling is established by the
/// submission of the command buffer's last batch, which another thread's work may have to come
/// first for -- and the predicate is what holds that line: a command buffer with a scheduled
/// handler is scheduled on its queue's handler thread, and one with a submission handler has that
/// handler run on the thread that takes its last batch, so neither is submitted here. Each
/// handler ran, and on none of the waiting threads.
#[test]
fn a_thread_waiting_for_the_scheduling_submits_what_it_committed_unless_a_handler_must_run() {
    let scratch = Scratch::new();
    let device = device(&scratch.0, CompletionPolicy::Exact);
    let queue = device.new_command_queue().unwrap();
    let side = device.new_command_queue().unwrap();
    let waiting = std::thread::current().id();
    let plain = scheduled_rounds(&device, &queue, Some(&side), Besides::Nothing);
    println!(
        "submitted by waiters for the scheduling: {} batches over {} passes of {ROUNDS} waits",
        plain.batches, plain.passes
    );
    assert_eq!(
        plain.passes, ROUNDS,
        "a thread waiting for the scheduling did not pass for its own commit while the \
         scheduler's thread was held"
    );
    assert!(
        plain.batches >= ROUNDS,
        "a thread waiting for the scheduling did not submit its own batch in its pass"
    );
    // Not held: the thread a command buffer with a handler is left to must run.
    for besides in [Besides::SubmissionHandler, Besides::ScheduledHandler] {
        let held = scheduled_rounds(&device, &queue, None, besides);
        println!(
            "with {besides:?}: {} batches over {} passes of {ROUNDS} waits for the scheduling",
            held.batches, held.passes
        );
        assert_eq!(
            held.batches, 0,
            "a thread waiting for the scheduling submitted a batch of a command buffer with \
             {besides:?}"
        );
        assert_eq!(
            held.handler_threads.len(),
            ROUNDS as usize,
            "{besides:?} did not run once a round"
        );
        assert!(
            !held.handler_threads.contains(&waiting),
            "a {besides:?} ran on the thread waiting in wait_until_scheduled"
        );
    }
    drop(side);
    drop(queue);
    drop(device);
    under_the_default(&scratch.0, "scheduling", |device, queue, side| {
        scheduled_rounds(device, queue, side, Besides::Nothing).batches
    });
}
