//! The device's completion: each submitted batch finishes once its queue's timeline reached the
//! batch's value, which one waiter thread per `VkQueue` watches, and the waiter that sees the
//! value reached finishes it ([`turns`]).
//!
//! - **In each queue's order.** A `VkQueue`'s batches signal its timeline with rising values, and
//!   Vulkan's signal operation order makes them complete in that order, so each queue's batches
//!   wait in one line, and a value reached finishes every batch up to it.
//! - **What a finished batch does.** Its queue's completed value is published for the tracker
//!   (the host saw it before any later submission), the events it signaled are reached, and the
//!   batch that ends a command buffer finishes it, once the turn read the command buffer's GPU
//!   timestamps and gave their slot back (`gpu_time.rs`): the turn runs its transition to
//!   `Completed` itself where that runs none of the program's code, else posts it to its queue's
//!   context ([`contexts`](crate::contexts)). A turn runs no handler.
//! - **Finished where seen.** Metal's completion wakes a thread waiting for a command buffer with
//!   no completed handler from the thread that saw it complete, and orders it after its queue's
//!   earlier completed handlers. Measured on the Apple M3 Max (macOS 27.0.1), 40 runs each: with
//!   no handler, the status read `Completed` 91 to 137 us after `GPUEndTime` and
//!   `waitUntilCompleted` returned 3 to 10 us after the status; with a completed handler of 20 ms,
//!   the status read `Completed` before the handler began (within 10 us) and in it, and the wait
//!   returned 10 to 23 us after it ended; a command buffer with no handler behind an earlier one of
//!   its queue whose handler ran 20 ms read `Completed`, and its wait returned, 3 to 23 us after
//!   that handler ended, 21 to 30 ms after its own `GPUEndTime`. So the turn runs a command
//!   buffer's transition itself (`CommandBufferShared::finish_here`) where it is plain (no
//!   completed handler, none being added after the commit, and no notice), its queue's context
//!   runs on no handler thread and holds only plain transitions, which the turn runs first, in
//!   order, and the runtime holds none of the program's objects a release drops or runs
//!   (`program_objects.rs`): its status, release and gate then come from the waiter, which wakes
//!   the program's waiting thread itself. Everywhere else it is posted, as before. Beside the
//!   runtime lane's tests on the M3 Max at the utility clamp, the handler thread's wake between
//!   the turn and the program's thread took up to 13 to 55 ms at a light burst's 99th percentile
//!   of the queue storm (`runtime_lane::queue_storms`), over Metal's one wake from its completion;
//!   counted (`WorkCounts::completed_where_seen`).
//! - **A command buffer finishes at its fence, where it has one.** Where the driver still reads a
//!   command buffer's query pools after the value (MoltenVK), the batch that ends it also signals
//!   a fence ([`fences`](super::fences)), which the driver signals after those reads: the turn
//!   posts the transition, after which what the command buffer used is released, only once the
//!   fence is signaled too, and then resets the fence and gives it back. A wait for fences holds
//!   up the other queues' batches by as long. Elsewhere Vulkan's value is enough, and the batch
//!   has no fence. On MoltenVK every command buffer takes one, not only the ones whose query pools
//!   that driver reads late: a relayed queue's failure attribution needs one for every batch, and
//!   the host sees the fence sooner than the value there anyway (`fences.rs`, which holds the
//!   argument, the reading of MoltenVK's completed handlers and the measurement).
//! - **One waiter per `VkQueue`, each waiting for one object** (replacing the wait for any of the
//!   timelines). Each `VkQueue`'s waiter thread waits for its queue's first pending batch past the
//!   value it saw reached: for the batch's fence where it has one, else for its queue's timeline to
//!   reach the batch's value, a wait of that one fence or that one semaphore (`poll`). Then it
//!   reads the timeline and publishes it; a signaled fence's batch reached its value whatever the
//!   read shows (lavapipe can show it later). A batch
//!   submitted to an idle `VkQueue` is waited for by that queue's own waiter, so it completes when
//!   its work does, not when another queue's does. No wait names several objects: MoltenVK 1.4.1's
//!   wait for any of several timelines (`vkWaitSemaphores` with `VK_SEMAPHORE_WAIT_ANY_BIT`,
//!   `mvkWaitSemaphores`) loses a signal that lands while it is still registering a later timeline
//!   (`RuntimeDriverDefect::WaitForAnyObjectLosesASignal`, `driver_defects.rs`),
//!   so the waiter saw the batch only when its 100 ms wait ran out (the consumer's guest presented
//!   at about a quarter of Metal's rate); its wait for any of several fences has the same flaw. A
//!   wait for one timeline or one fence counts its one signal under a lock, and loses none. The
//!   value is read again after every wait, so the waiter relies on no wake for the value, only for
//!   its timing. Counted by which object the wait named (`WorkCounts::fence_waits`,
//!   `WorkCounts::timeline_waits`), and by whether the batch was watched while this waiter was
//!   parked, so the wait for it began only after a notify and a thread wake
//!   (`WorkCounts::completion_arms` and `WorkCounts::completion_arms_after_a_wake`); a read that
//!   ran past the wait's value, and so costs one wait more, by
//!   `WorkCounts::completion_reads_ran_ahead`.
//! - **Finished by the waiter that saw it, one turn at a time** ([`turns`]). A waiter whose wait
//!   returned with its queue's value reached, or with a failure, makes the turn itself where none
//!   is under way: it takes every queue's batches their waiters saw reached (or the failure) and
//!   finishes them, and looks again until a look finds none. One turn runs at a time on the device:
//!   a waiter that finds one under way leaves what it saw to it, under the lock that turn's last
//!   look takes, so nothing is left behind. So a turn keeps every ordering across the queues:
//!   events reached before the scheduler is woken, fences settled before the transitions, each
//!   queue's batches in value order, and the failure drain over every queue, whose errors so never
//!   pass a queue's earlier completions (Metal runs a queue's completed handlers in commit order).
//!   No thread is woken between a batch's wait and its transition's run or post: on an Apple M3 Max (macOS
//!   27.0.1, MoltenVK 1.4.1) beside 24 processes spinning at the utility clamp, a separate thread
//!   that the waiters woke for their batches took them 8.8 to 15.1 ms after the wake in 4 runs of 6
//!   (22 us at the median), and since a turn finishes every queue's batches, each such wait put
//!   every light queue's burst of the queue storm (`runtime_lane::queue_storms`) into its tail.
//!   While its waiter makes a turn, a queue's next batch waits for that turn to end before its wait
//!   begins; its fence and value stay signaled meanwhile.
//! - **Published once a host wait returned for it** ([`seen`]). The host sees a batch's writes once
//!   the batch completed: its last command buffer makes them available to the host
//!   (`Barrier::to_host`, or `Barrier::to_all`), before its signal in submission order, and the
//!   host's reads come after a host wait for that signal. Vulkan orders host operations after a
//!   signal only through a wait: "signal operations happening before host operations that wait on
//!   them" system-synchronize-with them (the memory model), and a fence's "execution dependency is
//!   defined by waiting for a fence to become signaled, either via `vkWaitForFences` or by polling
//!   on `vkGetFenceStatus`". It gives `vkGetSemaphoreCounterValue` no such dependency (its value
//!   "may immediately be out of date"). So a value the read shows past the one a wait returned for
//!   (later batches that completed meanwhile, or the front's after a wait that ran out) is waited
//!   for in turn, a wait that returns at once ("If the condition is satisfied when
//!   `vkWaitSemaphores` is called, then `vkWaitSemaphores` returns immediately"), before it is
//!   published: one call more, and only where the read ran ahead.
//! - **Waits are bounded.** A waiter waits at most [`WAIT_SLICE`] at a time, so a stop is seen
//!   promptly and a lost device is reported by the next wait. A wait that ran out its slice
//!   although its fence or value was signaled when the waiter looked again is a wake it missed (or
//!   a signal in the slice's last moment), counted (`WorkCounts::missed_completion_wakes`), which
//!   the completion test holds to none.
//! - **Exit.** Each wait is a section of the exit guard ([`exit_guard`]), so an exit waits at most
//!   one slice for it; once the process is exiting, the waiters end.
//! - **A drain waits for the waiters.** A failed wait or read finishes every pending batch, which
//!   may destroy their fences (`let_go`); the batches are taken only once no waiter is waiting,
//!   and none arms until they are.
//! - **Let go before the waiters return.** A turn holds a command buffer through a [`Hold`],
//!   and lets go of everything it holds for a batch (its events, then the command buffer, or
//!   hands the hold to the transition it posts) before the next; the command buffer's waiters
//!   return once no hold is left ([`gate`](super::gate)). It reaches the device through a pending
//!   command buffer only for the wait, and for a failed wait's drain.
//! - **A failed wait drains the queues first**. Its command buffers finish with its error, and let
//!   go of what they used, once the device's queues were drained (`Device::drain_queues`): MoltenVK
//!   1.4.1 reports a lost device from the first completed handler of a failed Metal command buffer,
//!   while later ones still read the command buffers' pools.
//! - **No self-join.** A turn's drop of a command buffer's last reference can drop the device's
//!   last handle on its waiter; stopping from a waiter then joins the others and only asks it to
//!   end.
//! - **Woken only where one sleeps, and only the one a change is for.** Each waiter sleeps on a
//!   condition variable of its own, and a drain ([`take_pending`]) on another. A change wakes only
//!   the sleepers it is for: a batch watched, its queue's waiter; a turn that took a queue's
//!   batches, that queue's waiter (its next batch may arm); a wait that ended, a drain waiting for
//!   it; the pending batches given back after a failure, a stop or an exit, every one. Each where
//!   it sleeps (`Pending::parked`, `Pending::draining`), read under the lock the change is made
//!   under: one that sleeps later looks at the change first. A batch watched while its waiter is
//!   in its queue's wait, or a turn's look, so costs no system call (a notify is one, sleeper or
//!   not), and the waiters of the device's idle `VkQueue`s sleep through the other queues'
//!   batches. Counted: `WorkCounts::wakes_completion_waiters`, `WorkCounts::wakes_spared`.

use super::gate::Hold;
use super::CommandBufferShared;
use crate::copied::Written;
use crate::device::lock;
use crate::error::{CommandBufferError, Error, MemoryKind, VkResultCode};
use crate::event::EventShared;
use crate::exit_guard::{self, Call};
use crate::gpu_time;
use crate::rebind::SubmittedHolds;
use crate::release;
use crate::Device;
use ash::vk;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// How long one wait on the timelines lasts.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// One device's `VkQueue`s' waiters, which finish its batches.
pub(crate) struct Completion {
    watch: Arc<Watch>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

/// What the waiters and the scheduler share.
struct Watch {
    pending: Mutex<Pending>,
    /// Per `VkQueue`: its waiter's condition variable, notified where it sleeps on it on a change
    /// it waits for (the module docs): a batch watched on its queue, its queue's batches taken by
    /// a turn, the pending batches given back after a failure, a stop, an exit.
    wakes: Box<[Condvar]>,
    /// A drain's ([`take_pending`]), notified where one sleeps on it as a waiter's wait ends, and
    /// on a stop or an exit.
    drained: Condvar,
    /// The notifies made, and those spared as no waiter slept (`WorkCounts`).
    woke: AtomicU64,
    spared: AtomicU64,
}

impl Watch {
    /// Notify `wake` where `asleep` (read under the lock the change was made under) says a thread
    /// sleeps on it, counted either way.
    fn notify(&self, wake: &Condvar, asleep: bool) {
        if asleep {
            self.woke.fetch_add(1, Ordering::Relaxed);
            wake.notify_all();
        } else {
            self.spared.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Wake every waiter and drain, wherever they sleep: the pending batches were given back after
    /// a failure, or the waiters stop or end.
    fn wake_all(&self) {
        self.woke.fetch_add(1, Ordering::Relaxed);
        for wake in &self.wakes {
            wake.notify_all();
        }
        self.drained.notify_all();
    }
}

struct Pending {
    /// Per `VkQueue`: the submitted batches not yet finished, in value order.
    queues: Vec<VecDeque<Submitted>>,
    /// Per `VkQueue`: the largest value its waiter saw its timeline reach.
    reached: Vec<u64>,
    /// Per `VkQueue`: its waiter is in a wait.
    waiting: Vec<bool>,
    /// A waiter's failed wait or read, which every pending batch finishes with.
    failed: Option<CommandBufferError>,
    /// The pending batches are being taken (`take_pending`): no waiter arms.
    holding: bool,
    /// A waiter makes a turn ([`turns`]): another leaves what it saw to that turn.
    turning: bool,
    /// The process is exiting: every waiter ends, and what is pending stays pending.
    exiting: bool,
    stopping: bool,
    /// Per `VkQueue`: its waiter sleeps on its condition variable ([`wait`]).
    parked: Vec<bool>,
    /// The drains sleeping until no waiter is in a wait ([`take_pending`]).
    draining: usize,
}

/// A submitted batch its queue's waiter waits for.
pub(crate) struct Submitted {
    /// The value its queue's timeline reaches once it completed.
    pub(crate) value: u64,
    pub(crate) buffer: Hold<CommandBufferShared>,
    /// It ends its command buffer: finishing it completes the command buffer.
    pub(crate) last: bool,
    /// The fence its submission signals after the work, when it ends its command buffer where
    /// the driver needs one, or its queue's failures are relayed: the command buffer finishes
    /// once it is signaled ([`fences`](super::fences)).
    pub(crate) fence: Option<vk::Fence>,
    /// The events it signals, with their signal indices.
    pub(crate) signals: Vec<(Arc<EventShared>, u64)>,
    /// The holds of what its event group's commands took that its completion lets go
    /// (`State::holds_until`): at once, or, where it ends its command buffer, as that command
    /// buffer's completion is processed, before its completed handlers run
    /// (`State::keep_holds`, `CommandBufferShared::complete`).
    pub(crate) holds: Vec<SubmittedHolds>,
}

impl Completion {
    /// Start the waiters of a device with `queues` `VkQueue`s, one per `VkQueue`; refused when the
    /// host cannot start one.
    pub(crate) fn start(queues: usize) -> Result<Completion, Error> {
        let watch = Arc::new(Watch {
            pending: Mutex::new(Pending {
                queues: (0..queues).map(|_| VecDeque::new()).collect(),
                reached: vec![0; queues],
                waiting: vec![false; queues],
                failed: None,
                holding: false,
                turning: false,
                exiting: false,
                stopping: false,
                parked: vec![false; queues],
                draining: 0,
            }),
            wakes: (0..queues).map(|_| Condvar::new()).collect(),
            drained: Condvar::new(),
            woke: AtomicU64::new(0),
            spared: AtomicU64::new(0),
        });
        let completion = Completion {
            watch: Arc::clone(&watch),
            threads: Mutex::new(Vec::new()),
        };
        let spawned: std::io::Result<()> = (|| {
            for queue in 0..queues {
                let watch = Arc::clone(&watch);
                let waiter = exit_guard::spawn_user_interactive(
                    std::thread::Builder::new().name(format!("metal2vulkan-queue-wait-{queue}")),
                    move || wait_for_queue(&watch, queue),
                )?;
                lock(&completion.threads).push(waiter);
            }
            Ok(())
        })();
        match spawned {
            Ok(()) => Ok(completion),
            Err(_) => {
                completion.stop();
                Err(Error::OutOfMemory(MemoryKind::Host))
            }
        }
    }

    /// Wait for `batch`, just submitted to `VkQueue` `queue` (or a failed command buffer's end,
    /// at the queue's last submitted value).
    pub(crate) fn watch(&self, queue: usize, batch: Submitted) {
        let parked = {
            let mut pending = lock(&self.watch.pending);
            pending.queues[queue].push_back(batch);
            pending.parked[queue]
        };
        self.watch.notify(&self.watch.wakes[queue], parked);
    }

    /// The notifies made to the waiters, and those spared as none slept.
    pub(crate) fn wakes(&self) -> (u64, u64) {
        (
            self.watch.woke.load(Ordering::Relaxed),
            self.watch.spared.load(Ordering::Relaxed),
        )
    }

    /// Ask the waiters to end once nothing is pending, and wait for each but this one (the
    /// device's last handle can drop on one of them, in its turn).
    pub(crate) fn stop(&self) {
        lock(&self.watch.pending).stopping = true;
        self.watch.wake_all();
        let current = std::thread::current().id();
        let threads: Vec<JoinHandle<()>> = lock(&self.threads).drain(..).collect();
        for handle in threads {
            if handle.thread().id() != current {
                let _ = handle.join();
            }
        }
    }
}

/// What a turn's look found: batches whose values their queues reached, or a waiter's failure.
enum Turn {
    Finished(Vec<(usize, Submitted)>),
    Failed(CommandBufferError),
}

/// Wait on `queue`'s waiter's condition variable up to one slice, counted as parked meanwhile.
fn wait<'a>(
    watch: &'a Watch,
    mut pending: std::sync::MutexGuard<'a, Pending>,
    queue: usize,
) -> std::sync::MutexGuard<'a, Pending> {
    pending.parked[queue] = true;
    let mut pending = watch.wakes[queue]
        .wait_timeout(pending, WAIT_SLICE)
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .0;
    pending.parked[queue] = false;
    pending
}

/// The process is exiting: every waiter ends.
fn exit(watch: &Watch) {
    lock(&watch.pending).exiting = true;
    watch.wake_all();
}

/// Turns, made by the waiter that called this once it set `turning` (the module docs): each looks
/// for the batches the queues' waiters saw reached, or a waiter's failure, and finishes them
/// ([`finish`]), or finishes every pending batch with the failure ([`fail`]); until a look finds
/// nothing, which clears `turning` under the lock it took: a waiter that saw a value reached while
/// `turning` was set left it to that look. `lists` is the waiter's own, kept from turn to turn: a
/// turn fills them again within the room the earlier ones left, so in a steady state it allocates
/// none. False once the process is exiting.
fn turns(watch: &Watch, lists: &mut FenceLists) -> bool {
    loop {
        let turn = {
            let mut pending = lock(&watch.pending);
            if pending.exiting {
                return false;
            }
            lists.woken.clear();
            if let Some(error) = pending.failed.take() {
                // No waiter arms until the pending batches are taken (`fail`).
                pending.holding = true;
                Turn::Failed(error)
            } else {
                let finished = reached(&mut pending);
                if finished.is_empty() {
                    pending.turning = false;
                    return true;
                }
                // The waiters of the queues whose batches it took, where they sleep.
                for (queue, _) in &finished {
                    if lists.woken.last() != Some(queue) {
                        lists.woken.push(*queue);
                    }
                }
                lists.woken.retain(|&queue| pending.parked[queue]);
                Turn::Finished(finished)
            }
        };
        // A waiter whose front finished may arm for its queue's next.
        if lists.woken.is_empty() {
            watch.spared.fetch_add(1, Ordering::Relaxed);
        }
        for &queue in &lists.woken {
            watch.notify(&watch.wakes[queue], true);
        }
        match turn {
            Turn::Finished(finished) => {
                if !finish(watch, lists, finished) {
                    return false;
                }
            }
            Turn::Failed(error) => fail(watch, Vec::new(), error),
        }
    }
}

/// Finish `finished`, the batches a turn took, each queue's in its value order: their events
/// reached, the scheduler told, their fences settled, then, for each that ends its command buffer,
/// its timestamps read and its GPU times placed, and its transition posted. False once the
/// process is exiting.
fn finish(watch: &Watch, lists: &mut FenceLists, mut finished: Vec<(usize, Submitted)>) -> bool {
    // The host time the batches were seen complete: an uncalibrated command buffer's GPU end
    // (`gpu_time.rs`).
    let seen = gpu_time::host_nanos();
    // The device, held while the batches finish (each holds it meanwhile).
    let Some(device) = finished
        .first()
        .map(|(_, batch)| batch.buffer.device().clone())
    else {
        return true;
    };
    // The GPU's writes into copied host buffers go back to the caller's pages before anything
    // sees the batches complete: their event signals, the scheduler, a waiter, their completed
    // handlers (`copied.rs`). Only the copies the batches' command buffers may have written are
    // compared.
    let copied = &device.shared().copied;
    if copied.any() {
        let mut written = Written::default();
        for (_, batch) in &finished {
            let buffer = &batch.buffer;
            if buffer.unnoted.load(Ordering::Relaxed) || lock(&buffer.state).reaches_unnamed {
                written = Written::all();
                break;
            }
            lock(&buffer.dirty).copies_written(&mut written);
        }
        copied.merge_back(&device.shared().counts, &written);
    }
    // So do their writes into logged host pages imported in place, to the pages' logs, again now
    // they ran (`dirty_pages.rs`, `DirtyCommit::report_host_writes`).
    if device.shared().dirty.in_place() {
        for (_, batch) in &finished {
            lock(&batch.buffer.dirty).report_host_writes();
        }
    }
    // The holds of the batches that end no command buffer go first, their work completed: a
    // detach that sees one of their signals detaches what only they held (`detach.rs`).
    for (_, batch) in &mut finished {
        if !batch.last {
            batch.holds.clear();
        }
    }
    // Their event signals are seen before the scheduler looks again: a batch whose submission
    // checks wait for a GPU signal to land may now go.
    for (_, batch) in &finished {
        for (event, index) in &batch.signals {
            event.reached(*index);
        }
    }
    // A batch held at its gate, or for a GPU signal to land, may now go (`scheduler`).
    device.shared().scheduler.completions(&device);
    // Every finished batch's fence is waited for: where a queue's failures are relayed each
    // batch has one (`device/relay.rs`), signaled after the driver reported its failure.
    let mut ending = Vec::new();
    lists.gathered.clear();
    for (queue, batch) in finished {
        let Submitted {
            buffer,
            last,
            fence,
            holds,
            ..
        } = batch;
        lists.gathered.extend(fence.map(|fence| (queue, fence)));
        if last {
            let mut state = lock(&buffer.state);
            state.times.completed = seen;
            state.keep_holds(holds);
            drop(state);
            ending.push(buffer);
        }
    }
    let settled = settle(&device, lists);
    // Their GPU times' timestamps are read, and their slots given back, and their GPU times
    // placed among their queues' in order, before their transitions are posted: a completed
    // handler reads the times. A section, as each poll.
    if let Settled::Signaled = settled {
        let Some(_busy) = exit_guard::enter() else {
            exit(watch);
            return false;
        };
        for buffer in &ending {
            buffer.read_stamps();
            buffer.place_gpu_times();
        }
    }
    drop(device);
    match settled {
        Settled::Signaled => {
            for buffer in ending {
                CommandBufferShared::finish_here(buffer, Ok(()));
            }
        }
        // What is pending stays pending, as Metal's does.
        Settled::Exiting => {
            exit(watch);
            return false;
        }
        Settled::Failed(error) => fail(watch, ending, error),
    }
    true
}

/// Every pending batch, and the command buffers `ending` whose work completed (their fences let
/// go of already), finish with `error`: a wait or a read failed. The pending batches' fences are
/// let go of ([`let_go`]).
///
/// First the device's queues are drained (`Device::drain_queues`): a failed command buffer lets go
/// of its pools as it finishes, and a driver may still read them after the failed wait returned
/// (MoltenVK 1.4.1, from its later completed handlers), until its own completion of the work ran.
fn fail(watch: &Watch, ending: Vec<Hold<CommandBufferShared>>, error: CommandBufferError) {
    let failed = take_pending(watch);
    let held = (ending.first().map(|buffer| buffer.device()))
        .or_else(|| failed.first().map(|(_, batch)| batch.buffer.device()));
    if let Some(device) = held {
        // Their writes into copied host buffers are merged back by the next turn, which compares
        // every copy (`copied.rs`).
        device.shared().copied.owe();
        device.drain_queues();
    }
    let failed = failed.into_iter().filter_map(|(queue, batch)| {
        let Submitted {
            buffer,
            last,
            fence,
            holds,
            ..
        } = batch;
        if let Some(fence) = fence {
            let_go(buffer.device(), queue, fence);
        }
        if last {
            lock(&buffer.state).keep_holds(holds);
        }
        last.then_some(buffer)
    });
    for buffer in ending.into_iter().chain(failed) {
        CommandBufferShared::post_finished(buffer, Err(error));
    }
}

/// Let go of `fence` of `device`'s `VkQueue` `queue` after a failed wait: released once the
/// device is lost, when every submission counts as completed (out of the queue's submissions in
/// flight first, `device/relay.rs`); otherwise left to the device, since a submission that
/// signals it may still be pending. The release reaches the device's functions inside a section
/// of the exit guard, whether or not the caller is in one ([`fail`] is not: it posts the
/// command buffers' transitions, which a section may not), and once the process is exiting the
/// fence is left to its end.
fn let_go(device: &Device, queue: usize, fence: vk::Fence) {
    if device.is_lost() {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = device.shared();
        if let Some(flights) = shared.flights(queue) {
            flights.let_go(fence);
        }
        // SAFETY: the fence is the device's, and the device is lost.
        unsafe { release::object(shared.vk(), fence) };
    }
}

/// How the fences of the command buffers whose work completed ended.
enum Settled {
    /// Each is signaled, reset and given back.
    Signaled,
    /// The process is exiting: the waiters end.
    Exiting,
    /// A wait or a reset failed: the command buffers finish with this error, and the fences are
    /// let go of ([`let_go`]).
    Failed(CommandBufferError),
}

/// A turn's fences, and the waiters it wakes, in lists each waiter keeps from turn to turn
/// ([`turns`]).
#[derive(Default)]
struct FenceLists {
    /// The queues whose batches a look took and whose waiters sleep, to wake.
    woken: Vec<usize>,
    /// The fences of the batches the turn finished, each with its `VkQueue`.
    gathered: Vec<(usize, vk::Fence)>,
    /// Those the turn waits for, then resets and gives back, the relayed queues' marks with them.
    waited: Vec<vk::Fence>,
    /// One relayed queue's of `gathered`.
    queue: Vec<vk::Fence>,
}

/// Wait for `lists.gathered`'s fences of `device` (each of a `VkQueue`) to be signaled, a slice at
/// a time, each wait a section of the exit guard; then reset them and give them back to the
/// device's idle fences, with the marks before them where the queue's failures are relayed
/// (`device/relay.rs`), which leave its submissions in flight first. `lists.waited` and
/// `lists.queue` are filled here.
fn settle(device: &Device, lists: &mut FenceLists) -> Settled {
    let FenceLists {
        gathered,
        waited,
        queue: ours,
        ..
    } = lists;
    if gathered.is_empty() {
        return Settled::Signaled;
    }
    waited.clear();
    waited.extend(gathered.iter().map(|&(_, fence)| fence));
    loop {
        let Some(_busy) = exit_guard::enter() else {
            return Settled::Exiting;
        };
        let shared = device.shared();
        let raw = shared.vk();
        // SAFETY: the fences are the device's, each signaled by a submission made.
        let signaled = unsafe { raw.wait_for_fences(waited, true, WAIT_SLICE.as_nanos() as u64) };
        exit_guard::called(Call::WaitForFences);
        let error = match signaled {
            Ok(()) => {
                // Out of the queues' submissions in flight before they go back: every
                // submission before a signaled fence's completed too.
                for (queue, flights) in shared.relayed_queues() {
                    ours.clear();
                    ours.extend(
                        (gathered.iter())
                            .filter(|&&(q, _)| q == queue)
                            .map(|&(_, fence)| fence),
                    );
                    flights.settled(ours, waited);
                }
                // SAFETY: each fence is signaled and its submission completed.
                let reset = unsafe { raw.reset_fences(waited) };
                exit_guard::called(Call::ResetFences);
                match reset {
                    Ok(()) => {
                        // SAFETY: each fence is unsignaled and in no pending submission; inside
                        // a section.
                        unsafe { shared.fences.give(raw, waited) };
                        return Settled::Signaled;
                    }
                    Err(error) => error,
                }
            }
            Err(vk::Result::TIMEOUT) => continue,
            Err(error) => error,
        };
        let failed = failure(device, error);
        for &(queue, fence) in gathered.iter() {
            let_go(device, queue, fence);
        }
        return Settled::Failed(failed);
    }
}

/// Each queue's batches whose values its waiter saw its timeline reach, taken from `pending`, in
/// each queue's order.
fn reached(pending: &mut Pending) -> Vec<(usize, Submitted)> {
    let Pending {
        queues, reached, ..
    } = pending;
    let mut finished = Vec::new();
    for (queue, batches) in queues.iter_mut().enumerate() {
        while batches
            .front()
            .is_some_and(|batch| batch.value <= reached[queue])
        {
            finished.extend(batches.pop_front().map(|batch| (queue, batch)));
        }
    }
    finished
}

/// Every pending batch, taken once no waiter is in a wait: a waiter may wait for a pending
/// batch's fence, which the caller may destroy (`let_go`). No waiter arms meanwhile.
fn take_pending(watch: &Watch) -> Vec<(usize, Submitted)> {
    let mut pending = lock(&watch.pending);
    pending.holding = true;
    while pending.waiting.iter().any(|&waiting| waiting) {
        pending.draining += 1;
        pending = (watch.drained)
            .wait_timeout(pending, WAIT_SLICE)
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .0;
        pending.draining -= 1;
    }
    let taken = pending
        .queues
        .iter_mut()
        .enumerate()
        .flat_map(|(queue, batches)| batches.drain(..).map(move |batch| (queue, batch)))
        .collect();
    pending.holding = false;
    drop(pending);
    watch.wake_all();
    taken
}

/// A queue's first pending batch, as [`poll`] waits for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Front {
    /// The value its queue's timeline reaches once it completed.
    value: u64,
    /// The fence its submission signals, where it has one.
    fence: Option<vk::Fence>,
}

/// `VkQueue` `queue`'s waiter (the module docs): it waits for its queue's first pending batch
/// past the value it saw reached, a slice at a time, and makes the turn that finishes what its
/// queue reached, or a failure, where no turn is under way ([`turns`]); else it leaves them to the
/// turn under way.
fn wait_for_queue(watch: &Watch, queue: usize) {
    exit_guard::serves_completion();
    // The fence lists of the turns it makes.
    let mut lists = FenceLists::default();
    loop {
        // Whether the look for a batch to wait for slept first: the batch was watched while this
        // waiter was parked, so the wait for it began only after a notify and a thread wake, which
        // sit between the submission and the wait (`WorkCounts::completion_arms_after_a_wake`).
        let mut slept = false;
        // The front to wait for, and the device, reached through it (held only for the wait); or
        // none, for a turn: its queue holds batches it reached that no turn took yet (a failed
        // command buffer's end, watched at a value its queue reached), or a waiter failed, and no
        // turn is under way.
        let armed = {
            let mut pending = lock(&watch.pending);
            loop {
                if pending.exiting {
                    return;
                }
                if !pending.turning && due(&pending, queue) {
                    pending.turning = true;
                    break None;
                }
                let armable = !pending.holding && pending.failed.is_none();
                let armed = match pending.queues[queue].front() {
                    Some(batch) if armable && batch.value > pending.reached[queue] => {
                        let front = Front {
                            value: batch.value,
                            fence: batch.fence,
                        };
                        Some((front, batch.buffer.device().clone()))
                    }
                    None if pending.stopping => return,
                    _ => None,
                };
                if let Some(armed) = armed {
                    pending.waiting[queue] = true;
                    break Some(armed);
                }
                pending = wait(watch, pending, queue);
                slept = true;
            }
        };
        let Some((front, device)) = armed else {
            if !turns(watch, &mut lists) {
                return;
            }
            continue;
        };
        device.shared().counts.completion_armed(slept);
        // The wait is a section of the exit guard; once the process is exiting the waiters end,
        // and what is pending stays pending, as Metal's does.
        let polled = match exit_guard::enter() {
            Some(busy) => {
                let polled = poll(&device, queue, front);
                drop(busy);
                Some(polled)
            }
            None => None,
        };
        // Let go of the device before the report: until then the front is pending (nothing
        // finishes or takes it while this waits), and holds the device, so this is never its
        // last handle, which goes on the program's or the handlers' threads (`teardown_threads`).
        drop(device);
        let mut pending = lock(&watch.pending);
        pending.waiting[queue] = false;
        let Some(polled) = polled else {
            pending.exiting = true;
            drop(pending);
            watch.wake_all();
            return;
        };
        match polled {
            Ok(value) => {
                let reached = &mut pending.reached[queue];
                *reached = (*reached).max(value);
            }
            Err(error) => {
                pending.failed.get_or_insert(error);
            }
        }
        // This waiter finishes what it saw, where no turn is under way; the one under way takes
        // it at its next look, which takes this lock. A drain waits for the wait to end.
        let turn = !pending.turning && due(&pending, queue);
        pending.turning |= turn;
        let draining = pending.draining > 0;
        drop(pending);
        watch.notify(&watch.drained, draining);
        if turn && !turns(watch, &mut lists) {
            return;
        }
    }
}

/// Whether a turn has work that `queue`'s waiter sees: a waiter's failure, or its queue's first
/// pending batch reached.
fn due(pending: &Pending, queue: usize) -> bool {
    pending.failed.is_some()
        || pending.queues[queue]
            .front()
            .is_some_and(|batch| batch.value <= pending.reached[queue])
}

/// Wait up to one slice for `front`, the first pending batch of `device`'s `VkQueue` `queue`: for
/// its fence where it has one, else for the queue's timeline to reach its value, a
/// wait of that one object (the module docs). Then read the timeline and publish what a host wait
/// returned for ([`seen`]): the front's where its wait returned, and the value read where a wait
/// for it returned too. A failed wait or read is the error every pending command buffer finishes
/// with.
///
/// MoltenVK signals a fence from its Metal command buffer's completion handler, where it wakes a
/// timeline wait from a shared event's listener (MoltenVK's `MVKSync.mm`; 6 to 11 us later).
///
/// A signaled fence's batch reached its value whatever its timeline's read shows: a submission's
/// fence signal follows its timeline signal in Vulkan's signal operation order. A driver may show
/// the value later than the fence: lavapipe's emulated timeline shows it once the thread that
/// submitted the batch installed its point, after the batch ran, which a loaded host can hold up
/// past the fence's signal.
fn poll(device: &Device, queue: usize, front: Front) -> Result<u64, CommandBufferError> {
    let shared = device.shared();
    let raw = shared.vk();
    let slot = &shared.queues[queue];
    // A wait of up to one slice for the queue's timeline to reach `value`: whether it did.
    let reaches = |value: u64| {
        let semaphores = [slot.timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the semaphore is the device's timeline, live while the device is (held here).
        let waited = unsafe { raw.wait_semaphores(&info, WAIT_SLICE.as_nanos() as u64) };
        exit_guard::called(Call::WaitSemaphores);
        match waited {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(error) => Err(failure(device, error)),
        }
    };
    let waited = match front.fence {
        Some(fence) => {
            // SAFETY: the fence is the device's, in a submission made and not yet reset (its
            // batch is pending, and is not taken while this waits: `take_pending`), live while
            // the device is (held here).
            shared.counts.fence_wait();
            let waited =
                unsafe { raw.wait_for_fences(&[fence], true, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitForFences);
            match waited {
                Ok(()) => true,
                Err(vk::Result::TIMEOUT) => false,
                Err(error) => return Err(failure(device, error)),
            }
        }
        None => {
            shared.counts.timeline_wait();
            reaches(front.value)?
        }
    };
    // SAFETY: the semaphore is the device's timeline, live while the device is (held here).
    let read = unsafe { raw.get_semaphore_counter_value(slot.timeline) };
    exit_guard::called(Call::GetSemaphoreCounterValue);
    let read = read.map_err(|error| failure(device, error))?;
    // A wait that ran out its slice although what it waited for was signaled by now: a wake it
    // missed, counted (`WorkCounts::missed_completion_wakes`). A timeline is looked at by its
    // read; a fence by a wait of no time: "If timeout is zero, then vkWaitForFences does not
    // wait, but simply returns the current state of the fences".
    if !waited {
        let signaled = match front.fence {
            Some(fence) => {
                // SAFETY: as the wait's above.
                let looked = unsafe { raw.wait_for_fences(&[fence], true, 0) };
                exit_guard::called(Call::WaitForFences);
                looked.is_ok()
            }
            None => read >= front.value,
        };
        if signaled {
            shared.counts.missed_completion_wake();
        }
    }
    let value = seen(
        slot.completed_value(),
        waited.then_some(front.value),
        read,
        |value| {
            // The read ran past what this wait returned for: one more wait, to put the host
            // operation after a signal ([`seen`]).
            shared.counts.completion_read_ran_ahead();
            reaches(value)
        },
    )?;
    slot.completed(value);
    // Freed memory that waits for the batches in flight goes once they completed, on a handler
    // thread (`freed_memory.rs`).
    if let Some(parking) = shared.parking.get() {
        parking.completed(queue, value);
        parking.after_completion(device);
    }
    Ok(value)
}

/// The value of a queue's timeline the host may publish as reached (the module docs): `known`,
/// the one it published before; `waited`, the one a wait just returned for (a timeline's wait,
/// or the front's fence, whose signal follows its timeline's in signal operation order, whatever
/// the read shows); `read`, the timeline's read after that wait. A read past both is a value
/// no host wait returned for yet: `reaches` waits for it, which returns at once where the read
/// was right, and it is published only where that wait returned.
pub(crate) fn seen<E>(
    known: u64,
    waited: Option<u64>,
    read: u64,
    reaches: impl FnOnce(u64) -> Result<bool, E>,
) -> Result<u64, E> {
    let confirmed = waited.map_or(known, |waited| waited.max(known));
    if read <= confirmed {
        return Ok(confirmed);
    }
    Ok(match reaches(read)? {
        true => read,
        false => confirmed,
    })
}

/// The command-buffer error of a failed wait, recording a lost device.
fn failure(device: &Device, error: vk::Result) -> CommandBufferError {
    match error {
        vk::Result::ERROR_DEVICE_LOST => {
            device.mark_lost();
            // A batch held for an event may now be refused.
            device.shared().scheduler.wake();
            CommandBufferError::DeviceLost
        }
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            CommandBufferError::OutOfMemory
        }
        other => CommandBufferError::Internal(VkResultCode(other.as_raw())),
    }
}
