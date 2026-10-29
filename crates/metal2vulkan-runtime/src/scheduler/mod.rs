//! The device's scheduler: one thread per device that submits the committed command buffers'
//! batches to the device's `VkQueue`s.
//!
//! - **Handed to the thread** ([`Inbox`]). `commit` leaves its batches in the inbox and returns
//!   at once, as Metal's `commit` returns before its command buffer is scheduled: it makes no
//!   pass and submits nothing, so no driver call holds it. This thread takes them in, in commit
//!   order, at its next pass, and submits what is ready. The commit wakes the thread where it is
//!   not making a pass; where it is, the thread looks at the inbox again once its pass ends. The
//!   order sees every commit in commit order, and every batch is planned as it is submitted. This
//!   thread stops looking at the inbox again at a pass that leaves it a check or a host step, and
//!   before each one it runs passes again for as long as the inbox holds commits, and looks at it
//!   before it sleeps ([`left`]): a commit waits for the pass under way, the submission handlers
//!   that pass leaves, and the pass that takes it in, never for a check or a step. A commit
//!   wakes this thread alone.
//!
//! - **A thread about to block submits for itself** ([`Scheduler::submit_before_waiting`]).
//!   Metal encodes and commits inline on the program's thread, and a thread that commits and then
//!   waits for the completion would otherwise wait for a wake and a whole pass before its work
//!   reached the `VkQueue`. So, where the inbox still holds commits and no pass is under way, that
//!   thread becomes the passer ([`Inbox::pass_once`]), takes them into the order and submits what
//!   needs nothing of this thread and runs none of the program's code ([`submit_plain_ready`]):
//!   batches with no host step, no submission check, no handler of any kind, no switch of storage
//!   forms, and never one of a `VkQueue` whose batches group. It takes the head committed first of
//!   all and stops there, so its submissions are in commit order; it leaves [`Awaits`], [`Left`]
//!   and the group to this thread, which it wakes for what it left; and it never looks at the
//!   inbox again, so what was left to it goes to this thread by the same fences
//!   ([`Inbox::passed_over`]). Counted: `WorkCounts::scheduler_passes_by_waiters` and
//!   `WorkCounts::batches_submitted_by_waiters`.
//! - **The order** ([`Order`]). Each logical queue's batches wait in commit order; the head
//!   committed first among those ready goes next. A batch is ready once every event it waits for
//!   is reached by a signal already submitted or by the host ([`EventTimeline`]), so no `VkQueue`
//!   ever holds a wait for a signal that is not submitted: logical queues sharing a `VkQueue`
//!   cannot deadlock through the order they wait for each other in.
//! - **Binary semaphores** ([`BinarySignals`]). A batch that waits for one of the caller's binary
//!   semaphores is ready once every signal of it that a command buffer of this device committed
//!   before is submitted; a signal from the caller was submitted before the commit.
//! - **Tracking** ([`Tracker`]). Each batch is planned as it is submitted, against the device's
//!   hazard state, in this order: the order Metal follows across queues. A batch held for its host
//!   step or its checks keeps its place among the tracked accesses: a later commit's batch on
//!   another queue that conflicts with it waits behind it, its own host step and checks too
//!   ([`Order::behind`]).
//! - **Many logical queues on few `VkQueue`s.** A logical queue keeps one `VkQueue` (round robin
//!   at creation); each `VkQueue` signals its own timeline semaphore with every batch, which is
//!   the [`Point`] the tracker and the completion turns know the batch by.
//! - **Scheduled.** A command buffer is scheduled once its last batch is submitted (its every
//!   wait was satisfied), or held in a group (below): this thread posts its transition to
//!   `Scheduled` to its queue's context
//!   ([`contexts`](super::contexts)), which runs its scheduled handlers. No submission runs a
//!   scheduled handler, so none holds back a submission. A transition with no handler to run or
//!   wait for, whose context holds nothing and runs on no thread, this thread runs itself as it
//!   publishes the batch (`CommandBufferShared::schedule`): it sets the status and wakes the
//!   command buffer's waiters, and no handler thread is woken for it.
//! - **Submission handlers** (a runtime extension). This thread runs a command buffer's
//!   submission handlers once it submitted its last batch, or held it in a group (below), in
//!   submission order among the command buffers its passes took so, once it let go of the lock
//!   and before it lets go of the command buffers, before its next pass, which they hold back.
//!   The command buffer's transition to `Scheduled` waits for them.
//! - **Host steps**. A batch that executes commands the GPU may write
//!   (`CommandBufferShared::needs_host_step`) is held, once otherwise ready, until every
//!   `VkQueue` completed what was submitted to it by then (its gate, a snapshot of the queues'
//!   values): only then are the records the work before it wrote known. The thread then records
//!   the batch's segments (`CommandBufferShared::host_step`) without its lock, each step a
//!   section of the exit guard, and submits the batch in its order. Meanwhile it submits other
//!   ready batches; the completion turns wake it as batches complete while a gate is pending. A
//!   batch whose step is a predicated loop's check alone waits for its own `VkQueue`'s work
//!   alone; while the check's thread runs the loop's laying again, the batch waits as for a
//!   making, which the thread's end wakes the thread for (`command_buffer/rerun.rs`).
//! - **Submission checks** (`command_buffer/checks.rs`). A batch of a command buffer with a check
//!   is held, once otherwise ready and its host step recorded, until the checks of Metal's
//!   submissions in it ran (`CommandBufferShared::run_checks`), which the thread runs as it runs
//!   host steps: without its lock, each pass a section of the exit guard. Its checks wait, besides,
//!   until every event value it waits for was reached on the device or by the host: a value only a
//!   submitted GPU signal reaches is held for until that signal landed, as Metal submits (and
//!   checks) only then. A completion turn sees each finished batch's event signals before it
//!   wakes the thread, as for a gate. A command buffer without a check costs one flag per batch.
//! - **Grouped submissions** (`command_buffer/group.rs`). Under `CompletionPolicy::Grouped`, on a
//!   driver whose queue merges a submission that signals nothing into the next (Mesa's), a pass
//!   holds up to four ready batches of one `VkQueue` that signal nothing but its timeline, after
//!   their first work, and submits them in one `vkQueueSubmit` whose last batch alone signals: one
//!   kernel submission where there were four. Each held batch is tracked and its queue's value
//!   applied as it is held, and its command buffer scheduled where it ends it, its submission
//!   handlers run first (a switch of storage forms riding on it waits for the group); the group
//!   goes once it holds four, with or before a batch that cannot join, or when the pass runs out of
//!   ready batches, unless two or more earlier submissions of its `VkQueue` are not seen complete
//!   and no head waits for the GPU: then it stays held until one is left (a completion turn wakes
//!   the thread for it), it fills, a later batch sends it, or a thread waits for one of its command
//!   buffers (`Group::keeps`). The submission checks of up to three batches right behind a line's
//!   head that goes run with the head's ([`check_heads`]), so a line of checked batches is found
//!   ready together.
//! - **Waits are bounded.** The thread wakes on a commit or a host signal, and at least every
//!   [`WAIT_SLICE`], so a stop is seen promptly.
//! - **The driver's stalls taken idle** (`device/submit_stalls.rs`). Where the driver stalls in
//!   one signalling submission of every 1024 to a `VkQueue` (NVIDIA's), and a queue's next stall
//!   may be due, the thread wakes once it has had nothing to submit for `IDLE_NANOS`, and
//!   submits the calls left before the stall as empty submissions, without its lock, stopping
//!   where a commit or a waker came meanwhile: the stall falls in the gap between bursts of work
//!   rather than on a batch. That wake is no [`WAIT_SLICE`]'s.
//! - **Woken by completions only when they let a batch go** ([`Awaits`]). A pass that leaves a
//!   head at its gate, its checks waiting for a GPU signal to land, or a group kept, leaves what
//!   it waits for: each `VkQueue`'s value to complete, each event value to be reached. A
//!   completion turn wakes
//!   the thread only once one is met ([`Scheduler::completions`]), not at every batch that
//!   completes meanwhile. Each waking is counted by what woke the thread, and each sleep
//!   (`WorkCounts::scheduler_sleeps` and kin).
//! - **Waits for a signal to land itself** ([`Nudge`]). Where the heads wait only for event
//!   values that submitted GPU signals reach (no gate, none reached yet; a shared event's too,
//!   whose wait is held on the host rather than handed to a `VkQueue`, `event_timeline.rs`:
//!   [`landing_waits`]), the thread sleeps in one
//!   `vkWaitSemaphores` for any of those values, or for the device's nudge semaphore to reach its
//!   next value, instead of on its condition variable: the signal wakes it from the driver, with
//!   no completion turn and no waker between, and it observes the values itself
//!   (`EventShared::value`, which waits for what it reads), then checks and submits. Every other
//!   waker (a commit, a host signal, a finished decode, a lost device, a stop) signals the nudge,
//!   under the lock, where the thread waits so; a completion turn needs not, as the thread's own
//!   wait sees the signal the turn saw. Not where the driver's wait for any of several objects
//!   loses a signal that lands while it registers a later one
//!   (`RuntimeDriverDefect::WaitForAnyObjectLosesASignal`, `driver_defects.rs`; MoltenVK's):
//!   there the thread sleeps on its condition variable, woken by the turns, and
//!   [`waits_on_gpu`] says which. Counted:
//!   `WorkCounts::scheduler_gpu_waits`; checked by `runtime_lane::event_round_trips`, a host
//!   signal included.
//! - **Exit.** Each submission is a section of the exit guard ([`exit_guard`]); once the process
//!   is exiting, the thread ends, and what it holds stays held.
//! - **Let go without the lock.** A submitted batch's hold of its command buffer goes after the
//!   lock: it may be the last reference, and the command buffer's device may go with it, whose
//!   drop stops this thread.
//! - **A few batches a pass** ([`PASS_SUBMISSIONS`]). A pass that submitted that many batches ends
//!   at the next point where no batch is held for a group (so a group is never cut short); the
//!   thread lets go of their command buffers, runs their submission handlers, and passes again at
//!   once for what is still ready, in the same order. A command buffer's waiters wait for these
//!   holds (`command_buffer/gate.rs`), so a pass of every ready batch would keep the first one's
//!   waiters until the last was submitted, however long before that its work completed: seconds,
//!   once thousands of command buffers wait for one shared event (`runtime_lane::gpu_times`).
//!
//! Locks are taken in one order: the scheduler's state, then its inbox's, then a command buffer's,
//! then an event's or a `VkQueue`'s, then the handler threads' state, then the queues' waiters'
//! list. No call made under the scheduler's lock takes it again.
//!
//! [`EventTimeline`]: super::event_timeline::EventTimeline
//! [`Point`]: super::tracker::Point

pub(super) mod binary;
pub(super) mod inbox;
pub(super) mod left;
pub(super) mod order;

use super::command_buffer::gate::Hold;
use super::command_buffer::kept_barriers::KeptBarriers;
use super::command_buffer::BatchLists;
use super::command_buffer::CommandBufferShared;
use super::command_buffer::{Group, GroupEnd, Submitting, GROUP_BATCHES};
use super::device::functions::Functions;
use super::device::lock;
use super::device::submit_stalls::{IdleCounts, IdleStalls, IDLE_NANOS};
use super::error::{CommandBufferError, Error, MemoryKind};
use super::event::EventShared;
use super::event_timeline::WaitFor;
use super::exit_guard;
use super::gpu_time;
use super::replay::SpareLogs;
use super::resource::ResourceKey;
use super::tracker::{self, Footprint, Tracker};
use super::Device;
use ash::vk;
use binary::{BinarySignals, Needs};
use inbox::Inbox;
use left::{work_through, Kind, Left};
use metal2vulkan::error::RuntimeDriverDefect;
use order::{Behind, Order};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{fence, AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long the thread sleeps at most without looking for work.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The batches one pass submits before it ends, at the next point where no batch is held for a
/// group, for the thread to let go of their command buffers (the module docs): a submitted
/// command buffer's waiters wait for about that many submissions, not for every ready batch's.
/// A multiple of a group's batches (`command_buffer/group.rs`).
const PASS_SUBMISSIONS: usize = 16;

/// Whether the thread of a device whose driver has `defects` waits for the GPU itself where its
/// heads wait only for submitted signals to land (the module docs,
/// [`Scheduler::wait_on_gpu`]): its one `vkWaitSemaphores` names several objects, which a driver
/// that loses a signal landing while it registers a later one would miss, so the thread sleeps on
/// its condition variable there instead, woken by the completion turns. The question is the
/// driver's, and the defect answers it, never the driver's name.
pub(crate) fn waits_on_gpu(defects: &BTreeSet<RuntimeDriverDefect>) -> bool {
    !defects.contains(&RuntimeDriverDefect::WaitForAnyObjectLosesASignal)
}

/// The device's nudge (the module docs): a timeline semaphore of its own, which a waker signals
/// to the value the thread waits for beside the GPU's signals.
struct Nudge {
    functions: Functions,
    semaphore: vk::Semaphore,
}

impl Nudge {
    /// Signal the semaphore to `value`, above every value signalled before (each under the
    /// scheduler's lock); false where it failed (a lost device, whose waits return) or the process
    /// is exiting (the thread ends at its slice).
    fn signal(&self, value: u64) -> bool {
        let Some(_busy) = exit_guard::enter() else {
            return false;
        };
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(value);
        // SAFETY: the device's semaphore, live while its scheduler runs; `value` is above the
        // semaphore's (the values rise under the scheduler's lock).
        let signalled = unsafe { self.functions.get().signal_semaphore(&info) };
        exit_guard::called(exit_guard::Call::SignalSemaphore);
        signalled.is_ok()
    }
}

/// How the thread slept as a waker marked the state ([`State::mark`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Asleep {
    /// It made a pass or was about to look again: no wake needed.
    No,
    /// On its condition variable: notify it.
    OnCondvar,
    /// On the GPU: the nudge was signalled, under the lock.
    Nudged,
    /// On the GPU, for what the waker's completion turn saw land: its own wait returns for it.
    OnGpu,
}

/// Why a look of the scheduler's for the next batch to submit did not submit a line's head
/// ([`Held::held_for`], `WorkCounts::heads_held_for_binary_semaphores` and kin). Every reason a
/// head keeps its place for is one of these, so the counts say what holds a device's submissions
/// back; what a batch waits for in its `vkQueueSubmit` is not a hold of the scheduler's and is
/// none of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeldFor {
    /// A binary semaphore it waits for whose signal is not submitted.
    BinarySemaphore,
    /// A rebind is encoding its event group's encoders again, or its command buffer's first
    /// submission is not resolved.
    Reencoding,
    /// A CPU decode queued into a decoded twin it may read.
    Decode,
    /// Its host step has not been recorded.
    HostStep,
    /// Its command buffer's submission checks have not run.
    Check,
    /// An event value it waits for that neither a submitted signal nor the host reaches.
    EventValue,
    /// A batch of an earlier commit on another logical queue that keeps its place and whose
    /// footprint runs in commit order with its own ([`Order::is_behind`]).
    EarlierCommit,
}

impl HeldFor {
    /// How many there are, for the counters' array.
    pub(crate) const COUNT: usize = 7;
}

/// The lists the thread's wait for the GPU fills, kept with their room.
#[derive(Default)]
struct GpuWait {
    events: Vec<(Arc<EventShared>, u64)>,
    semaphores: Vec<vk::Semaphore>,
    values: Vec<u64>,
}

/// What woke the scheduler's thread (`WorkCounts::scheduler_wakes_commit` and kin), in the order
/// [`Scheduler::sleeps_and_wakings`] gives them.
#[derive(Clone, Copy)]
enum Woken {
    /// A commit left its batches to the thread.
    Commit,
    /// A completion turn: what a head waited for from the GPU was met, or the device was lost.
    Completion,
    /// A host signal: an event's value set, a pipeline made, a decode finished, a command buffer
    /// encoded again, the device lost.
    Host,
    /// No waking: the sleep ended at [`WAIT_SLICE`].
    Slice,
}

/// What the heads a pass left waiting for the GPU wait for (the module docs): each one at its gate,
/// the `VkQueue`s' values to complete; each one whose checks wait for GPU signals to land, the
/// event values. Left under its own lock, which the scheduler's thread takes under its state's
/// and a completion turn alone; the completion turn reads events under it, so no event's lock is
/// held while it is taken.
#[derive(Default)]
struct Awaits {
    /// A head waits: read without the lock first.
    any: AtomicBool,
    list: Mutex<AwaitList>,
}

/// [`Awaits`]' lists, kept with their room.
#[derive(Default)]
struct AwaitList {
    /// The event values heads' checks wait for to be reached on the device or by the host.
    events: Vec<(Arc<EventShared>, u64)>,
    /// The gates heads' host steps wait at, one after another, each a value per `VkQueue`.
    gates: Vec<u64>,
    /// How many `VkQueue`s each gate has a value for.
    queues: usize,
}

impl Awaits {
    /// A pass begins: what heads wait for is found again.
    fn clear(&self) {
        if self.any.load(Ordering::SeqCst) {
            let mut list = lock(&self.list);
            list.events.clear();
            list.gates.clear();
        }
    }

    /// A head's checks wait for the event values of `waits` to land.
    fn events(&self, waits: &[(Arc<EventShared>, u64)]) {
        let mut list = lock(&self.list);
        list.events.extend(waits.iter().cloned());
        self.any.store(true, Ordering::SeqCst);
    }

    /// A head's host step waits for every `VkQueue` to complete its value of `gate`.
    fn gate(&self, gate: &[u64]) {
        let mut list = lock(&self.list);
        list.queues = gate.len();
        list.gates.extend_from_slice(gate);
        self.any.store(true, Ordering::SeqCst);
    }

    /// The group held waits for `VkQueue` `queue`, of `queues`, to complete `value`
    /// (`Group::keeps`): a gate of no other queue's value.
    fn gate_on(&self, queue: usize, value: u64, queues: usize) {
        let mut list = lock(&self.list);
        list.queues = queues;
        (list.gates).extend((0..queues).map(|at| if at == queue { value } else { 0 }));
        self.any.store(true, Ordering::SeqCst);
    }
}

impl AwaitList {
    /// Whether a head may go: an event value it waits for was reached on the device or by the
    /// host, or every `VkQueue` completed its gate (`completed`).
    fn met(&self, completed: impl Fn(&[u64]) -> bool) -> bool {
        (self.events.iter())
            .any(|(event, value)| matches!(event.wait(*value), Some(WaitFor::Reached)))
            || (self.queues > 0 && self.gates.chunks(self.queues).any(completed))
    }
}

/// One device's scheduler thread.
pub(crate) struct Scheduler {
    shared: Arc<SchedulerShared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

struct SchedulerShared {
    /// The state, under the lock a pass holds, with the commits a pass under way takes in.
    inbox: Inbox<State, Committed>,
    wake: Condvar,
    /// What heads wait for from the GPU: the completion turns wake the thread once one is met.
    awaits: Awaits,
    /// The thread's sleeps, its wakings by what woke it ([`Woken`]), and the notifies spared as
    /// it did not sleep, counted.
    sleeps: AtomicU64,
    wakings: [AtomicU64; 4],
    spared: AtomicU64,
    /// The sleeps on the GPU, of `sleeps` (the module docs).
    gpu_waits: AtomicU64,
    /// The passes the thread made.
    passes: AtomicU64,
    /// The device's nudge, where the thread waits for the GPU itself (the module docs).
    nudge: OnceLock<Nudge>,
    /// The device's cleared encoder logs (`replay.rs`): an encoder takes one, and its log comes
    /// back as the encoder ends, or in its command buffer's release.
    spare_logs: SpareLogs,
    /// Where the device's driver stalls in one signalling submission of every 1024 to a queue,
    /// what the thread takes those stalls idle with (the module docs).
    stalls: OnceLock<IdleStalls>,
}

impl SchedulerShared {
    /// A waker marked the state for the thread, which makes no pass: wake it where it sleeps
    /// (`asleep`, read under the lock the mark was made under; one that sleeps later reads the
    /// mark first), counted as `woken`; else the notify, a system call, is spared.
    fn woken(&self, woken: Woken, asleep: Asleep) {
        match asleep {
            Asleep::OnCondvar => {
                self.wakings[woken as usize].fetch_add(1, Ordering::Relaxed);
                self.wake.notify_all();
            }
            Asleep::Nudged => {
                self.wakings[woken as usize].fetch_add(1, Ordering::Relaxed);
            }
            Asleep::No | Asleep::OnGpu => {
                self.spared.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// The `(semaphore, index)` pairs of the event values every head waits for to land, into
    /// `wait`, where the thread may wait for them itself (the module docs): the device has a
    /// nudge, no head waits at a gate, and each value is a submitted signal's not yet observed (one
    /// reached already may let a head go, a completion turn's wake the thread's condition variable
    /// takes).
    fn gpu_waits(&self, wait: &mut GpuWait) -> bool {
        wait.events.clear();
        wait.semaphores.clear();
        wait.values.clear();
        if self.nudge.get().is_none() || !self.awaits.any.load(Ordering::SeqCst) {
            return false;
        }
        let list = lock(&self.awaits.list);
        if !list.gates.is_empty() || list.events.is_empty() {
            return false;
        }
        for (event, value) in &list.events {
            let Some(WaitFor::Index(index)) = event.landing(*value) else {
                return false;
            };
            wait.events.push((Arc::clone(event), *value));
            wait.semaphores.push(event.semaphore());
            wait.values.push(index);
        }
        true
    }
}

/// What the scheduler submits from, and the device state it submits against.
pub(crate) struct State {
    order: Order<Held>,
    /// The device's hazard state, in submission order.
    pub(crate) tracker: Tracker<ResourceKey>,
    /// Per `VkQueue`: the timeline value of the last batch submitted there.
    pub(crate) values: Vec<u64>,
    /// Per `VkQueue`, per timeline whose completed writes its batches wait for (each `VkQueue`'s,
    /// then the device's two own, `device/submit.rs`): the value a batch there last waited for,
    /// which made what completed through it visible to every later command there
    /// (`command_buffer/submit.rs`).
    pub(crate) waited: Vec<Vec<u64>>,
    /// The binary semaphore signals the device's command buffers committed and submitted.
    binary: BinarySignals<vk::Semaphore>,
    /// A commit or a host signal came since the thread last looked.
    dirty: bool,
    /// The thread sleeps on its condition variable: a waker notifies it only then.
    asleep: bool,
    /// The thread waits for the GPU, and for the nudge to reach this value: a waker signals it
    /// to that value under the lock, instead of notifying (the module docs).
    on_gpu: Option<u64>,
    /// The nudge semaphore's value: the last a waker signalled it to.
    nudge_value: u64,
    stopping: bool,
    /// The textures whose decoded twins have CPU decodes queued or running, each with how many
    /// (`texture/twin_decode.rs`): a batch whose footprint names one waits for them.
    decodes: BTreeMap<ResourceKey, usize>,
    /// How many of `decodes` are of untracked textures, whose uses no footprint names: every
    /// batch waits while one is left.
    untracked_decodes: usize,
    /// The batches' barrier command buffers, recorded once and submitted again
    /// (`command_buffer/kept_barriers.rs`).
    pub(crate) kept: KeptBarriers,
    /// The lists each batch's submission fills and empties, with their room
    /// (`command_buffer/submit.rs`).
    pub(crate) lists: BatchLists,
    /// The batches held for one kernel submission on one `VkQueue` (`command_buffer/group.rs`),
    /// submitted before the pass ends, or kept past it while that queue runs earlier work
    /// (`Group::keeps`).
    pub(crate) group: Group,
    /// Those batches as the order held them, with their logical queues, in order.
    grouped: Vec<(u64, Held)>,
    /// The earliest bound of a head whose host step waits for its draws' pipelines
    /// ([`StepWaits::until`]), found by the last look ([`gate_heads`]): the thread sleeps no
    /// later, so the step records those still being made as nothing then.
    wake_by: Option<Instant>,
}

impl State {
    /// A waker's mark (the module docs): look for work again, and how the thread sleeps. Where it
    /// waits for the GPU, `nudge` is signalled to the value it waits for, here, under the lock;
    /// none is given for a completion turn's wake, whose signal the thread's own wait sees.
    fn mark(&mut self, nudge: Option<&Nudge>) -> Asleep {
        self.dirty = true;
        if self.asleep {
            return Asleep::OnCondvar;
        }
        match (self.on_gpu, nudge) {
            (None, _) => Asleep::No,
            (Some(_), None) => Asleep::OnGpu,
            (Some(value), Some(nudge)) => {
                if nudge.signal(value) {
                    self.on_gpu = None;
                    self.nudge_value = value;
                }
                Asleep::Nudged
            }
        }
    }
}

/// A commit not yet taken into the order: the command buffer, its logical queue, its batches.
struct Committed {
    buffer: Hold<CommandBufferShared>,
    queue: u64,
    batches: Vec<BatchSync>,
}

/// A committed batch not yet submitted.
struct Held {
    buffer: Hold<CommandBufferShared>,
    /// Its index among the command buffer's batches.
    index: usize,
    /// Whether it is the command buffer's last batch.
    last: bool,
    /// The events it waits for, with the values.
    waits: Vec<(Arc<EventShared>, u64)>,
    /// What its binary semaphore waits need submitted first.
    binary_needs: Needs<vk::Semaphore>,
    /// The binary semaphores it signals.
    binary_signals: Vec<vk::Semaphore>,
    /// It holds segments, so it may need a host step.
    segments: bool,
    /// For its host step: each `VkQueue`'s value of the last batch submitted there once the batch
    /// was otherwise ready, which the queue's completed value reaches before the step.
    gate: Option<Vec<u64>>,
    /// Its command buffer has a submission check, so it may wait for checks to run.
    checked: bool,
    /// The makings of the pipelines its host step records were raised to a queue's first held
    /// batch's ([`hasten_heads`]).
    hastened: bool,
    /// Its waits held on the host where a Vulkan wait would have served were counted
    /// ([`count_waits_held`]): once per batch.
    counted_held: bool,
    /// The host nanoseconds at the first look of the scheduler's that found it held
    /// ([`held_for`](Self::held_for)), until it goes ([`hold_ended`](Self::hold_ended)).
    held_since: Option<u64>,
}

impl Held {
    /// A look of the scheduler's found it held for `reason`: counted, and the host nanoseconds
    /// noted at the first look that found it held at all, whatever the reason then. Nothing is
    /// read from the host's clock while nothing waits.
    fn held_for(&mut self, reason: HeldFor) {
        (self.buffer.device().shared().counts).head_held_for(reason);
        if self.held_since.is_none() {
            self.held_since = Some(gpu_time::host_nanos());
        }
    }

    /// It goes (submitted, or held in a group): where a look had found it held, the nanoseconds
    /// from that look to now are counted with it. A batch no look found held costs no clock read.
    fn hold_ended(&mut self) {
        if let Some(since) = self.held_since.take() {
            let nanos = gpu_time::host_nanos().saturating_sub(since);
            (self.buffer.device().shared().counts).batch_hold_ended(nanos);
        }
    }
}

/// What a batch's host step waits for (`CommandBufferShared::step_waits`).
pub(crate) struct StepWaits {
    /// The work submitted before it to complete (its gate): the GPU may write what it reads.
    pub(crate) gpu: bool,
    /// The `VkQueue` whose work alone it waits for, where the GPU writes what it reads there alone
    /// (a predicated loop's check, `command_buffer/rerun.rs`): every other queue goes on.
    pub(crate) queue_alone: Option<usize>,
    /// A pipeline it records, still being made.
    pub(crate) pipelines: bool,
    /// Where it waits for pipelines and its command buffer's bound has not passed: the bound,
    /// after which its draws' pipelines still being made are waited for no longer
    /// (`CommandBuffer::making_deadline`).
    pub(crate) until: Option<Instant>,
}

/// What one committed batch waits for and signals, as the scheduler orders it.
pub(crate) struct BatchSync {
    /// The events it waits for, with the values.
    pub(crate) waits: Vec<(Arc<EventShared>, u64)>,
    /// The caller's binary semaphores it waits for.
    pub(crate) binary_waits: Vec<vk::Semaphore>,
    /// The caller's binary semaphores it signals.
    pub(crate) binary_signals: Vec<vk::Semaphore>,
    /// It holds segments, so it may need a host step.
    pub(crate) segments: bool,
    /// Its command buffer has a submission check, so it may wait for checks to run.
    pub(crate) checked: bool,
}

impl Scheduler {
    /// Start the thread of a device with `queues` `VkQueue`s, whose batches each end with
    /// `Barrier::to_all` where `by_ends` (`batch_order.rs`); refused when the host cannot start
    /// one.
    pub(crate) fn start(queues: usize, by_ends: bool) -> Result<Scheduler, Error> {
        let shared = Arc::new(SchedulerShared {
            inbox: Inbox::new(State {
                order: Order::default(),
                tracker: match by_ends {
                    true => Tracker::by_ends(),
                    false => Tracker::default(),
                },
                values: vec![0; queues],
                waited: vec![vec![0; queues + 2]; queues],
                binary: BinarySignals::default(),
                dirty: false,
                asleep: false,
                on_gpu: None,
                nudge_value: 0,
                stopping: false,
                decodes: BTreeMap::new(),
                untracked_decodes: 0,
                kept: KeptBarriers::default(),
                wake_by: None,
                lists: BatchLists::default(),
                group: Group::default(),
                grouped: Vec::new(),
            }),
            wake: Condvar::new(),
            awaits: Awaits::default(),
            sleeps: AtomicU64::new(0),
            wakings: Default::default(),
            spared: AtomicU64::new(0),
            gpu_waits: AtomicU64::new(0),
            passes: AtomicU64::new(0),
            nudge: OnceLock::new(),
            spare_logs: SpareLogs::default(),
            stalls: OnceLock::new(),
        });
        let thread = {
            let shared = Arc::clone(&shared);
            exit_guard::spawn_user_interactive(
                std::thread::Builder::new().name("metal2vulkan-scheduler".into()),
                move || run(&shared),
            )
            .map_err(|_| Error::OutOfMemory(MemoryKind::Host))?
        };
        Ok(Scheduler {
            shared,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Wait for the GPU itself where heads wait for signals to land, with `semaphore`, a timeline
    /// semaphore at 0 of the device of `functions`, which destroys it after stopping the thread,
    /// as its nudge (the module docs): on a device whose driver loses no signal in a wait for any
    /// of several objects ([`waits_on_gpu`]).
    pub(crate) fn wait_on_gpu(&self, functions: Functions, semaphore: vk::Semaphore) {
        let _ = self.shared.nudge.set(Nudge {
            functions,
            semaphore,
        });
    }

    /// Take the driver's stalls in one signalling submission of every 1024 to a queue idle with
    /// `stalls` (the module docs): on a device whose driver has them, which releases the
    /// semaphores after stopping the thread ([`release_idle`](Self::release_idle)).
    pub(crate) fn take_stalls_idle(&self, stalls: IdleStalls) {
        let _ = self.shared.stalls.set(stalls);
    }

    /// The counts of the stalls taken idle so far (`WorkCounts::idle_submits` and kin): none on
    /// a device that takes none.
    pub(crate) fn idle_stall_counts(&self) -> IdleCounts {
        (self.shared.stalls.get()).map_or_else(IdleCounts::default, IdleStalls::counts)
    }

    /// Release what the thread took the stalls idle with: the device is being torn down, its
    /// thread stopped and its queues idle.
    pub(crate) fn release_idle(&self, raw: &ash::Device) {
        if let Some(stalls) = self.shared.stalls.get() {
            stalls.release(raw);
        }
    }

    /// Hold the batches of `buffer`, committed now on logical queue `queue`: what each batch
    /// waits for and signals, in order. They go to the thread's inbox, and `commit` returns at
    /// once, with nothing submitted: the thread takes them in at its next pass, woken for it
    /// where it is not making one (the module docs).
    pub(crate) fn commit(
        &self,
        buffer: &Arc<CommandBufferShared>,
        queue: u64,
        batches: Vec<BatchSync>,
    ) {
        let shared = &*self.shared;
        let committed = Committed {
            buffer: buffer.hold(),
            queue,
            batches,
        };
        buffer.device().shared().counts.commit_handed_off();
        let mut asleep = Asleep::No;
        let marked = (shared.inbox).post(committed, |state| {
            asleep = state.mark(shared.nudge.get());
        });
        if marked {
            shared.woken(Woken::Commit, asleep);
        }
    }

    /// A thread is about to block until a command buffer of this device completes
    /// (`CommandBufferShared::waits_for_completion`): where the batches it committed still wait
    /// in the inbox and no pass is under way, it becomes the passer ([`Inbox::pass_once`]), takes
    /// them into the order and submits what it may itself ([`submit_plain_ready`]), then leaves
    /// the rest to the thread, which it wakes.
    ///
    /// Metal encodes and commits inline on the program's thread. `commit` here submits nothing,
    /// as Metal's returns before its command buffer is scheduled ([`commit`](Self::commit)), so a
    /// thread that commits and then blocks would otherwise wait for a wake and a whole pass
    /// before its work reached the `VkQueue`. The condition that earns the submission is that
    /// this thread is about to block, which is known only here.
    ///
    /// **It runs none of the program's code.** Metal runs a command buffer's handlers on its own
    /// threads, and one that took a lock this thread holds would deadlock where Metal's would
    /// not; a release of the program's objects it ran would run where Metal runs none. So it
    /// submits only batches whose command buffer has no submission and no scheduled handler, no
    /// submission check and no host step, and no switch of a storage form riding on it
    /// ([`plain_ready`]), and the holds its pass lets go of go to a handler thread wherever one
    /// is a command buffer's last reference. Whatever it leaves is the thread's, which runs them
    /// as it does today.
    pub(crate) fn submit_before_waiting(&self, device: &Device) {
        let shared = &*self.shared;
        let mut spent = Spent::new();
        let (mut submitted, mut left_over) = (0, false);
        let passed = shared.inbox.pass_once(|state, commits| {
            take_in(state, commits, &mut spent);
            submitted = submit_plain_ready(state, &mut spent);
            left_over = !state.order.is_empty() || !state.grouped.is_empty();
        });
        if !passed {
            return;
        }
        device.shared().counts.waiter_passed(submitted);
        // The holds go outside the lock, and a command buffer's last reference to a handler
        // thread: no release of the program's objects (a no-copy buffer's deallocator among them)
        // runs on a thread blocking in `wait_until_completed`, where Metal runs none.
        for buffer in spent.drain(..) {
            if let Some(last) = buffer.let_go_short_of_last() {
                device.let_go_elsewhere(Box::new(last));
            }
        }
        // This pass never looks again (`inbox.rs`): what is left in the order, and anything
        // posted or nudged during it, goes to the thread that makes passes.
        if left_over || shared.inbox.passed_over() {
            self.wake_for(Woken::Host);
        }
    }

    /// What makes the thread look for work again from any thread: a pipeline a held batch's
    /// host step waits for was made (`render/pending.rs`). It holds the thread's state weakly,
    /// so a making it waits on keeps nothing of the scheduler alive.
    pub(crate) fn waker(&self) -> impl Fn() + Clone + Send + 'static {
        let shared = Arc::downgrade(&self.shared);
        move || {
            if let Some(shared) = shared.upgrade() {
                let asleep = lock(shared.inbox.state()).mark(shared.nudge.get());
                shared.woken(Woken::Host, asleep);
            }
        }
    }

    /// Whether every batch committed so far was submitted: none is held for an event, a binary
    /// semaphore, a host step, a check or a group, or waits in the inbox
    /// (`texture/storage_form.rs`).
    pub(crate) fn holds_none(&self) -> bool {
        let inbox = &self.shared.inbox;
        let state = lock(inbox.state());
        state.order.is_empty() && state.grouped.is_empty() && !inbox.holds_commits()
    }

    /// A CPU decode into the decoded twin of the resource `key` was queued: every batch whose
    /// footprint names it waits at its submission until [`decode_finished`](Self::decode_finished)
    /// says so of every one queued; where it is not `tracked` (its uses not recorded), every
    /// batch. A batch already submitted is not waited for.
    pub(crate) fn decode_queued(&self, key: ResourceKey, tracked: bool) {
        let mut state = lock(self.shared.inbox.state());
        *state.decodes.entry(key).or_default() += 1;
        if !tracked {
            state.untracked_decodes += 1;
        }
    }

    /// One CPU decode into the decoded twin of `key` (`tracked` as it was queued) finished: the
    /// batches waiting for its decodes go once none is left.
    pub(crate) fn decode_finished(&self, key: ResourceKey, tracked: bool) {
        let mut state = lock(self.shared.inbox.state());
        if let Some(count) = state.decodes.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                state.decodes.remove(&key);
            }
        }
        if !tracked {
            state.untracked_decodes -= 1;
        }
        let asleep = state.mark(self.shared.nudge.get());
        drop(state);
        self.shared.woken(Woken::Host, asleep);
    }

    /// Whether this thread is the scheduler's (a submission check runs on it).
    pub(crate) fn is_current(&self) -> bool {
        lock(&self.thread)
            .as_ref()
            .is_some_and(|handle| handle.thread().id() == std::thread::current().id())
    }

    /// Look for work again: an event's value rose, a pipeline was made, a rebind encoded a
    /// command buffer again, or the device was lost (each a host signal). Where a pass is under
    /// way, it passes again once done, and wakes the thread for a step or a check that finds;
    /// else the thread is woken here, under the lock, which then only a short hold or a pass made
    /// after this one's change has ([`Inbox::wake`]): the waker never waits for a pass's
    /// submissions.
    pub(crate) fn wake(&self) {
        self.wake_for(Woken::Host);
    }

    /// [`wake`](Self::wake), counted as `woken`.
    fn wake_for(&self, woken: Woken) {
        self.wake_with(woken, self.shared.nudge.get());
    }

    /// [`wake_for`](Self::wake_for), signalling `nudge` where the thread waits for the GPU.
    fn wake_with(&self, woken: Woken, nudge: Option<&Nudge>) {
        let mut asleep = Asleep::No;
        let marked = (self.shared.inbox).wake(|state| {
            asleep = state.mark(nudge);
        });
        if marked {
            self.shared.woken(woken, asleep);
        }
    }

    /// Batches of `device` completed, their queues' completed values and their event signals
    /// published: look for work again where a head waits at its gate, or for a GPU signal to land,
    /// and what it waits for is now met ([`Awaits`]), or the device is lost. The flag is read
    /// after a fence, as the thread sets it, and leaves what it waits for, before it reads the
    /// completed values or the events, so one of the two sees the other.
    pub(crate) fn completions(&self, device: &Device) {
        fence(Ordering::SeqCst);
        let awaits = &self.shared.awaits;
        if !awaits.any.load(Ordering::SeqCst) {
            return;
        }
        let shared = device.shared();
        let completed = |gate: &[u64]| {
            (shared.queues.iter())
                .zip(gate)
                .all(|(queue, &value)| queue.completed_value() >= value)
        };
        // Where the thread waits for the GPU, it waits for the signals the turn saw land: it is
        // nudged only for a lost device.
        if device.is_lost() {
            self.wake_for(Woken::Completion);
        } else if lock(&awaits.list).met(completed) {
            self.wake_with(Woken::Completion, None);
        }
    }

    /// The thread's sleeps and wakings so far: sleeps, then wakings by a commit, by a completion,
    /// by a host signal, and the sleeps that ended at [`WAIT_SLICE`]; then the notifies spared as
    /// it did not sleep; then the sleeps on the GPU, of the sleeps (`WorkCounts`).
    pub(crate) fn sleeps_and_wakings(&self) -> (u64, [u64; 4], u64, u64) {
        let shared = &self.shared;
        (
            shared.sleeps.load(Ordering::Relaxed),
            std::array::from_fn(|at| shared.wakings[at].load(Ordering::Relaxed)),
            shared.spared.load(Ordering::Relaxed),
            shared.gpu_waits.load(Ordering::Relaxed),
        )
    }

    /// The passes the thread made so far (`WorkCounts::scheduler_passes`).
    pub(crate) fn passes(&self) -> u64 {
        self.shared.passes.load(Ordering::Relaxed)
    }

    /// The device's cleared encoder logs (`replay.rs`).
    pub(crate) fn spare_logs(&self) -> &SpareLogs {
        &self.shared.spare_logs
    }

    /// Release the kept barrier command buffers (`command_buffer/kept_barriers.rs`): the device is
    /// being torn down, its queues idle.
    pub(crate) fn release_kept(&self, raw: &ash::Device) {
        lock(self.shared.inbox.state()).kept.release(raw);
    }

    /// Ask the thread to end, and wait for it unless this is the thread. The device is being
    /// dropped, so nothing is held: every held command buffer holds its device.
    pub(crate) fn stop(&self) {
        {
            let mut state = lock(self.shared.inbox.state());
            state.stopping = true;
            state.mark(self.shared.nudge.get());
        }
        self.shared.wake.notify_all();
        let mut thread = lock(&self.thread);
        let own = thread
            .as_ref()
            .is_some_and(|handle| handle.thread().id() == std::thread::current().id());
        if !own {
            if let Some(handle) = thread.take() {
                let _ = handle.join();
            }
        }
    }
}

/// The thread's loop. It ends when the device stops it, or once the process is exiting
/// ([`exit_guard`]): what is held then stays held, as Metal's pending work stays pending.
fn run(shared: &SchedulerShared) {
    exit_guard::serves_completion();
    // What its rounds let go of and the work they leave it, emptied after each waking and kept
    // with their room.
    let (mut passed, mut left) = (Passed::default(), Left::default());
    let mut gpu = GpuWait::default();
    loop {
        let mut state = lock(shared.inbox.state());
        // Since when the thread has had nothing to submit, and whether it looked for a driver's
        // stall due since (the module docs): once an idle, once it lasted `IDLE_NANOS`.
        let (idle_since, mut looked) = (Instant::now(), false);
        // A commit, or a nudge, left to this thread's pass may wait in the inbox (`inbox.rs`).
        while !state.dirty
            && !state.stopping
            && !shared.inbox.holds_commits()
            && !shared.inbox.nudged()
        {
            // A head whose step waits for its draws' pipelines waits until its bound at most
            // (`StepWaits::until`): a pass looks at it again then.
            let bound = match state.wake_by {
                None => None,
                Some(at) => match at.checked_duration_since(Instant::now()) {
                    Some(left) if !left.is_zero() => Some(left),
                    _ => {
                        state.wake_by = None;
                        state.dirty = true;
                        continue;
                    }
                },
            };
            // A stall due is taken without the lock, the empty submissions stopping where a
            // commit or a waker came; the sleep before it ends at the idle's length, no slice.
            let stalls = (shared.stalls.get()).filter(|stalls| !looked && stalls.may_be_due());
            let (slice, short) = match stalls {
                Some(stalls) => {
                    let idle = Duration::from_nanos(IDLE_NANOS);
                    match idle.checked_sub(idle_since.elapsed()) {
                        Some(left) if !left.is_zero() => (WAIT_SLICE.min(left), left < WAIT_SLICE),
                        _ => {
                            looked = true;
                            drop(state);
                            stalls.take(|| {
                                shared.inbox.holds_commits()
                                    || shared.inbox.nudged()
                                    || lock(shared.inbox.state()).dirty
                            });
                            state = lock(shared.inbox.state());
                            continue;
                        }
                    }
                }
                None => (WAIT_SLICE, false),
            };
            let slice = bound.map_or(slice, |left| left.min(slice));
            shared.sleeps.fetch_add(1, Ordering::Relaxed);
            if let Some(nudge) = shared.nudge.get().filter(|_| shared.gpu_waits(&mut gpu)) {
                state = wait_for_gpu(shared, state, nudge, &mut gpu, (slice, short));
                continue;
            }
            state.asleep = true;
            let (woke, waited) = shared
                .wake
                .wait_timeout(state, slice)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = woke;
            state.asleep = false;
            if waited.timed_out() && !short {
                shared.wakings[Woken::Slice as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
        if state.stopping {
            return;
        }
        // The pass takes the lock again, counted as a passer from before (`inbox.rs`): a commit
        // that finds it taken leaves its batches to this thread.
        drop(state);
        match work_through(
            &shared.inbox,
            &mut left,
            |left| round(shared, left, &mut passed),
            run_work,
        ) {
            None => return,
            Some(false) => {}
            // What the work did may let batches go.
            Some(true) => lock(shared.inbox.state()).dirty = true,
        }
    }
}

/// Sleep on the GPU (the module docs): for any of `gpu`'s semaphores to reach its value, or the
/// nudge its next, at most `slice` ([`WAIT_SLICE`], or less to a head's bound, or, where `short`,
/// the idle's length before a driver's stall is taken, whose end is no slice's), without the lock
/// `state` holds, which comes back taken. Where a value landed, its events observe their counters,
/// and the state is marked for a pass.
fn wait_for_gpu<'a>(
    shared: &'a SchedulerShared,
    mut state: std::sync::MutexGuard<'a, State>,
    nudge: &Nudge,
    gpu: &mut GpuWait,
    (slice, short): (Duration, bool),
) -> std::sync::MutexGuard<'a, State> {
    shared.gpu_waits.fetch_add(1, Ordering::Relaxed);
    let target = state.nudge_value + 1;
    state.on_gpu = Some(target);
    drop(state);
    gpu.semaphores.push(nudge.semaphore);
    gpu.values.push(target);
    let info = vk::SemaphoreWaitInfo::default()
        .flags(vk::SemaphoreWaitFlags::ANY)
        .semaphores(&gpu.semaphores)
        .values(&gpu.values);
    // A section, as each queue waiter's wait: once the process is exiting, no wait.
    let waited = exit_guard::enter().map(|_busy| {
        // SAFETY: the semaphores are live: the events' are held, the nudge is the device's until
        // the thread stopped; each value is a submitted signal's, or the nudge's next, which a
        // waker signals.
        let waited =
            unsafe { (nudge.functions.get()).wait_semaphores(&info, slice.as_nanos() as u64) };
        exit_guard::called(exit_guard::Call::WaitSemaphores);
        waited
    });
    let landed = matches!(waited, Some(Ok(())));
    if landed {
        // What landed is observed here, the reads waited for (`event.rs`), so the pass sees it.
        for (event, _) in &gpu.events {
            event.value();
        }
    }
    gpu.events.clear();
    let mut state = lock(shared.inbox.state());
    // Whether a waker signalled the nudge (taking `on_gpu`); where none did, the next wait waits
    // for the same value.
    let by_waker = state.on_gpu.take().is_none();
    match waited {
        Some(Ok(())) if !by_waker => {
            shared.wakings[Woken::Completion as usize].fetch_add(1, Ordering::Relaxed);
            state.dirty = true;
        }
        // Nudged: the waker counted the waking and marked the state.
        Some(Ok(())) => {}
        Some(Err(vk::Result::TIMEOUT)) if !short => {
            shared.wakings[Woken::Slice as usize].fetch_add(1, Ordering::Relaxed);
        }
        Some(Err(vk::Result::TIMEOUT)) => {}
        // A failed wait (a lost device) or an exit: a pass finds out.
        _ => state.dirty = true,
    }
    state
}

/// Run one piece of work the thread's passes left it, without the lock (a check runs the program's
/// code; a step records, and may make what a dispatch makes), a section of the exit guard; false,
/// with nothing run, once the process is exiting.
fn run_work(kind: Kind, work: Work) -> bool {
    let Some(busy) = exit_guard::enter() else {
        return false;
    };
    let (buffer, index) = &work;
    match kind {
        Kind::Step => {
            buffer.host_step(*index);
            for &call in exit_guard::HOST_STEP {
                exit_guard::called(call);
            }
        }
        Kind::Check => buffer.run_checks(*index),
    }
    drop(busy);
    // Out of the section: it may be the command buffer's last reference.
    drop(work);
    true
}

/// One round of the thread's: a pass, looking at the inbox again while the passes leave the
/// thread no work and none stopped at [`PASS_SUBMISSIONS`] ([`Inbox::pass`]; such a pass marked
/// the state for the next round), what they leave added to `left` and what they let go of to
/// `passed` (empty, kept from round to round); then, without the lock, what they let go of: the
/// switches of storage forms settled ended and the submission handlers run (on this thread,
/// before its next pass; neither once the process is exiting, as the thread then runs nothing
/// more, their command buffers
/// staying `Committed`), the holds dropped (the last reference to a command buffer may be among
/// them, and with it its device, whose drop stops this thread). False once the process is
/// exiting.
fn round(shared: &SchedulerShared, left: &mut Left<Work>, passed: &mut Passed) -> bool {
    (shared.inbox).pass(|state, commits| {
        shared.passes.fetch_add(1, Ordering::Relaxed);
        pass(state, commits, (&mut *passed, &mut *left), &shared.awaits);
        !passed.exiting && left.is_empty() && !passed.more
    });
    passed.more = false;
    if passed.exiting {
        passed.settled.clear();
        passed.handlers.clear();
    } else {
        settle(&mut passed.settled);
        run_submission_handlers(&mut passed.handlers);
    }
    passed.spent.clear();
    !passed.exiting
}

/// What passes let go of for their thread to finish without the lock ([`submit_ready`]'s
/// outputs), whether a pass stopped at [`PASS_SUBMISSIONS`] (more may be ready), and whether the
/// process is exiting.
#[derive(Default)]
struct Passed {
    spent: Spent,
    settled: Settled,
    handlers: Handlers,
    more: bool,
    exiting: bool,
}

/// Take `commits` into the order, in commit order: each command buffer's batches, each with what
/// it waits for and signals, behind the batches its logical queue already holds. Each commit's own
/// hold goes to `spent`, for the caller to drop without the lock: its batches hold the command
/// buffer now.
fn take_in(state: &mut State, commits: &mut Vec<Committed>, spent: &mut Spent) {
    for Committed {
        buffer,
        queue,
        batches,
    } in commits.drain(..)
    {
        let count = batches.len();
        let State { order, binary, .. } = &mut *state;
        let held = batches.into_iter().enumerate().map(|(index, batch)| Held {
            buffer: buffer.hold(),
            index,
            last: index + 1 == count,
            binary_needs: binary.commit(&batch.binary_waits, &batch.binary_signals),
            waits: batch.waits,
            binary_signals: batch.binary_signals,
            segments: batch.segments,
            gate: None,
            checked: batch.checked,
            hastened: false,
            counted_held: false,
            held_since: None,
        });
        order.push(queue, held);
        spent.push(buffer);
    }
}

/// A pass of the thread's: take `commits` into the order, in commit order, then submit what is
/// ready ([`submit_ready`]) into `passed` and `left`. A pass that stopped at
/// [`PASS_SUBMISSIONS`] marks the state, so the thread passes again at once after its round let
/// go of them.
fn pass(
    state: &mut State,
    commits: &mut Vec<Committed>,
    (passed, left): (&mut Passed, &mut Left<Work>),
    awaits: &Awaits,
) {
    take_in(state, commits, &mut passed.spent);
    state.dirty = false;
    match submit_ready(
        state,
        (&mut passed.spent, &mut passed.settled),
        left,
        &mut passed.handlers,
        awaits,
    ) {
        Ok(More(true)) => {
            state.dirty = true;
            passed.more = true;
        }
        Ok(More(false)) => {}
        Err(Exiting) => passed.exiting = true,
    }
}

/// Whether a pass of [`submit_ready`] stopped at [`PASS_SUBMISSIONS`], with batches perhaps still
/// ready.
struct More(bool);

/// What a pass of [`submit_ready`] lets go of, for the caller to drop without the lock: the holds
/// of the command buffers its batches came from, the last of which may be a command buffer's last
/// reference. Nothing else: what a command buffer's commands and encoders took (what it retained,
/// its encoders' logs) goes in its release, on its queue's handler thread
/// (`CommandBufferShared::complete`), never here. A batch is published before its pass ends, so
/// its command buffer may complete, and run its release, before these holds go; a handle let go
/// of here after that would be the last, and a no-copy buffer's deallocator would run on this
/// thread, whose own hold keeps the command buffer's gate shut against the deallocator's wait for
/// it.
type Spent = Vec<Hold<CommandBufferShared>>;

/// Submit every batch that is ready, in order, until none is; each submission is a section of the
/// exit guard. A batch is published (or its failure posted) only after its section ended, so the
/// exit hook's registration at the section's end comes before anything the program can learn of it
/// (`exit_guard`). A batch's events are let go of before its command buffer (`gate`), whose holds
/// go to `spent`, for the caller to drop without the lock. A batch whose host step may run now goes
/// to `left` as a step (once), for the caller to record without the lock; `awaits` holds what the
/// heads still waiting at their gates wait for. A batch whose submission checks may run now goes to
/// `left` as a check (once), for the caller to run without the lock; `awaits` holds the GPU signals
/// those still waiting to land wait for. A command buffer whose last batch was submitted, or that
/// failed, and on which a fetched target's storage form rides (`texture/storage_form.rs`) is
/// settled and goes to `settled`, for the caller to announce without the lock ([`settle`]). A
/// command buffer whose last batch was submitted, or held in a group (`command_buffer/group.rs`),
/// with submission handlers goes to `handlers`, in that order, for the caller to run them without
/// the lock ([`run_submission_handlers`]). Once it let go of [`PASS_SUBMISSIONS`] batches' holds,
/// it stops where no batch is held for a group, answering `More(true)`, for the caller to let go of
/// them and pass again. `Err` once the process is exiting.
fn submit_ready(
    state: &mut State,
    (spent, settled): (&mut Spent, &mut Settled),
    left: &mut Left<Work>,
    handlers: &mut Handlers,
    awaits: &Awaits,
) -> Result<More, Exiting> {
    let before = spent.len();
    loop {
        if spent.len() - before >= PASS_SUBMISSIONS && state.grouped.is_empty() {
            return Ok(More(true));
        }
        let busy = exit_guard::enter().ok_or(Exiting)?;
        // What the heads wait for from the GPU is found again by this pass's looks.
        awaits.clear();
        // The heads that wait behind a batch committed before them that keeps its place: no step,
        // check or submission of theirs before it is submitted (`order.rs`).
        // Of those whose host step or checks may start (`gate_heads`, `check_heads`), and of the
        // ready ones in commit order until one goes (`Order::next`); `memo` holds what this pass
        // found.
        // With one line no head waits behind another line's batch.
        let mut memo = Behind::default();
        let behind: Vec<u64> = match state.order.several_lines() {
            false => Vec::new(),
            true => (state
                .order
                .heads_where(|held| held.segments || held.checked))
            .into_iter()
            .filter(|&sequence| is_behind(&state.order, &state.binary, &mut memo, sequence))
            .collect(),
        };
        hasten_heads(&mut state.order);
        let waiting = gate_heads(state, &behind, left, awaits);
        let landing = check_heads(state, &behind, left, awaits);
        let binary = &state.binary;
        let decodes = &state.decodes;
        let untracked_decodes = state.untracked_decodes;
        let behind = |order: &Order<Held>, sequence| is_behind(order, binary, &mut memo, sequence);
        let next = state.order.next(behind, |held| {
            if !binary.ready(&held.binary_needs) {
                held.held_for(HeldFor::BinarySemaphore);
                return None;
            }
            // The submission of its event group waits while a rebind encodes encoders of it again
            // (`command_buffer/reencode.rs`).
            if held.buffer.reencode_pending() {
                held.held_for(HeldFor::Reencoding);
                return None;
            }
            // Held for a decoded twin's CPU decode it may read: any batch, while an untracked
            // texture's is left.
            if untracked_decodes > 0 || (!decodes.is_empty() && reads_decoding(held, decodes)) {
                held.held_for(HeldFor::Decode);
                return None;
            }
            // Held for its host step.
            if held.segments && held.buffer.needs_host_step(held.index) {
                held.held_for(HeldFor::HostStep);
                return None;
            }
            // Held for its submission checks.
            if held.checked && held.buffer.needs_check(held.index) {
                held.held_for(HeldFor::Check);
                return None;
            }
            let waits = event_waits(held);
            if waits.is_none() {
                held.held_for(HeldFor::EventValue);
            }
            waits
        });
        // The heads this look found waiting behind a batch of an earlier commit on another logical
        // queue ([`is_behind`]), counted on each: the one hold that crosses logical queues and
        // `VkQueue`s. The memo holds every head the look asked about, those whose host step or
        // checks it held back among them, so none is looked up that was not asked.
        for sequence in memo.behind() {
            if let Some(held) = state.order.head_mut_at(sequence) {
                held.held_for(HeldFor::EarlierCommit);
            }
        }
        let Some((queue, sequence, mut held, waits)) = next else {
            // The batches held go before the pass ends, unless they stay held while their
            // `VkQueue` runs two or more earlier submissions and no head waits for the GPU
            // (`Group::keeps`): a completion turn wakes the thread once one is left, a commit or
            // a waiter for one of their command buffers sooner. Their submission makes no batch
            // ready: none of them signals a binary semaphore, and their events' signals counted
            // as submitted as they were held.
            drop(busy);
            if !waiting && !landing {
                if let Some(value) = state.group.keeps() {
                    awaits.gate_on(state.group.queue(), value, state.values.len());
                    return Ok(More(false));
                }
            }
            let end = match (state.order.is_empty(), left.is_empty()) {
                (true, _) => GroupEnd::Drained,
                (false, false) => GroupEnd::Checks,
                (false, true) => GroupEnd::Held,
            };
            submit_group(state, (spent, settled), handlers, end)?;
            if !waiting && !landing {
                awaits.any.store(false, Ordering::SeqCst);
            }
            return Ok(More(false));
        };
        let submitted = held
            .buffer
            .submit_batch(held.index, held.last, &waits, state);
        // Its waits for values found reached go: it waits for none of them.
        if matches!(
            submitted,
            Ok(Submitting::Submitted(_) | Submitting::Grouped)
        ) {
            held.hold_ended();
            let met = held.waits.len() - waits.len();
            if met > 0 {
                held.buffer.device().shared().counts.event_waits_met(met);
            }
        }
        let submitted = match submitted {
            Ok(Submitting::Submitted(submission)) => Ok(submission),
            Err(error) => Err(error),
            // A rebind marked it since it was found ready: back at its place, which it keeps,
            // until the rebind encoded its encoders again (`command_buffer/reencode.rs`), which
            // wakes the thread.
            Ok(Submitting::Later) => {
                drop(busy);
                state.order.put_back(queue, sequence, held);
                continue;
            }
            // The batches held go first: back at its place, then looked for again.
            Ok(Submitting::AfterGroup(end)) => {
                drop(busy);
                state.order.put_back(queue, sequence, held);
                submit_group(state, (spent, settled), handlers, end)?;
                continue;
            }
            // Held for one kernel submission with the batches after it: published once that is
            // made (`command_buffer/group.rs`), at once where the group is full.
            Ok(Submitting::Grouped) => {
                drop(busy);
                // Scheduled as it is held, out of the section, its submission handlers run first
                // (`group.rs`).
                if state.group.schedule_held() {
                    handlers.push(held.buffer.hold());
                }
                held.buffer.held_in_group();
                state.grouped.push((queue, held));
                if state.group.full() {
                    submit_group(state, (spent, settled), handlers, GroupEnd::Full)?;
                }
                continue;
            }
        };
        let Held {
            buffer,
            last,
            waits: events,
            binary_needs,
            binary_signals,
            ..
        } = held;
        drop(events);
        drop(busy);
        // Where batches were held, they went first, in this batch's `vkQueueSubmit` (first work it
        // took sent it at once).
        if let Some(outcome) = state.group.take_outcome() {
            settle_group(state, outcome, (spent, settled), handlers);
        }
        match submitted {
            Ok(submission) => {
                state.binary.submitted(&binary_needs, &binary_signals);
                if submission.runs_handlers() {
                    handlers.push(buffer.hold());
                }
                buffer.publish(submission);
                if last && buffer.settle_storage_forms() {
                    settled.push(buffer.hold());
                }
            }
            Err(error) => {
                // The rest of the command buffer never runs (dropped here, before it can
                // finish).
                state.binary.dropped(&binary_needs);
                let rest = state
                    .order
                    .drop_head(queue, |next| next.buffer.same(&buffer));
                for held in &rest {
                    state.binary.dropped(&held.binary_needs);
                }
                spent.extend(rest.into_iter().map(|held| held.buffer));
                buffer.failed(error, &state.values);
                if buffer.settle_storage_forms() {
                    settled.push(buffer.hold());
                }
            }
        }
        spent.push(buffer);
    }
}

/// The process is exiting ([`exit_guard`]): the thread submits no more.
struct Exiting;

/// Submit, into `spent`, the batches a thread about to block may submit itself
/// ([`Scheduler::submit_before_waiting`]), at most [`PASS_SUBMISSIONS`] of them: the head
/// committed first of all while [`plain_ready`] holds for it. How many it submitted.
///
/// It takes the first head rather than the first *ready* one ([`Order::take_first`]), so it stops
/// at the first head it may not submit: submission order stays commit order, and no batch that
/// keeps its place is passed over by one committed after it ([`is_behind`] has nothing to answer).
/// That is why it is a path of its own and not a condition in [`submit_ready`]'s head selection.
///
/// **It touches none of the thread's own state.** [`Awaits`] is what the thread's last pass left
/// it asleep on, its `vkWaitSemaphores` among them, and [`Left`] is the work that pass handed it:
/// a waiting thread that cleared the first would leave the thread asleep with work ready, and one
/// handed a piece of the second would run a host step, or the program's code in a submission
/// check, where Metal runs neither. So it reads `order`, `binary` and `decodes`, submits against
/// `tracker`, `values`, `waited`, `kept` and `lists` as a pass does, and leaves `awaits`, `left`,
/// the group, `dirty` and the sleeps alone. It makes no pass at all while the group holds a batch.
fn submit_plain_ready(state: &mut State, spent: &mut Spent) -> usize {
    let mut submitted = 0;
    while submitted < PASS_SUBMISSIONS && state.grouped.is_empty() {
        let Some(busy) = exit_guard::enter() else {
            return submitted;
        };
        let taken = {
            let State {
                order,
                binary,
                decodes,
                untracked_decodes,
                ..
            } = &mut *state;
            let (binary, decodes, untracked) = (&*binary, &*decodes, *untracked_decodes);
            order.take_first(|held| plain_ready(held, binary, decodes, untracked))
        };
        let Some((queue, sequence, mut held, waits)) = taken else {
            drop(busy);
            return submitted;
        };
        let answer = held
            .buffer
            .submit_batch(held.index, held.last, &waits, state);
        // Its waits for values found reached go: it waits for none of them.
        if matches!(answer, Ok(Submitting::Submitted(_))) {
            held.hold_ended();
            let met = held.waits.len() - waits.len();
            if met > 0 {
                held.buffer.device().shared().counts.event_waits_met(met);
            }
        }
        match answer {
            Ok(Submitting::Submitted(submission)) => {
                drop(busy);
                // The predicate holds out every command buffer with submission handlers and
                // every switch of storage forms, so there is nothing for the submitting thread
                // to run once the lock is free, and nothing to settle and announce.
                debug_assert!(
                    !submission.runs_handlers(),
                    "a waiting thread submitted a batch whose handlers it must not run"
                );
                let Held {
                    buffer,
                    binary_needs,
                    binary_signals,
                    waits: events,
                    ..
                } = held;
                drop(events);
                state.binary.submitted(&binary_needs, &binary_signals);
                buffer.publish(submission);
                spent.push(buffer);
                submitted += 1;
            }
            // Nothing submitted or changed: a rebind encodes encoders of it again
            // (`command_buffer/reencode.rs`), or the group held must go first. It goes back at
            // its place, which it keeps, for the thread that makes passes, which the caller wakes.
            Ok(Submitting::Later | Submitting::AfterGroup(_)) => {
                drop(busy);
                state.order.put_back(queue, sequence, held);
                return submitted;
            }
            // Unreachable: `groups` stops holding for a queue but never starts (its `flights` are
            // set once, `device/relay.rs`), so a batch [`plain_ready`] found ungrouped has
            // `joins` false and submits alone (`command_buffer/group.rs`), and this pass runs
            // only while the group holds no batch, so `admits` holds for it. Held as the thread's
            // own pass holds one all the same, and this pass ends there: the thread, which the
            // caller wakes, sends the group at once, as it does for any group one of whose
            // command buffers a thread waits for (`Group::keeps`).
            Ok(Submitting::Grouped) => {
                drop(busy);
                let handlers = state.group.schedule_held();
                debug_assert!(
                    !handlers,
                    "a waiting thread held a batch whose handlers it must not run"
                );
                held.buffer.held_in_group();
                state.grouped.push((queue, held));
                return submitted;
            }
            // As a pass fails a batch it could not submit: the rest of the command buffer never
            // runs. Its storage forms are none to settle, and its handlers never run, as they
            // never do for a command buffer that fails before its last batch is taken.
            Err(error) => {
                drop(busy);
                let Held {
                    buffer,
                    binary_needs,
                    waits: events,
                    ..
                } = held;
                drop(events);
                state.binary.dropped(&binary_needs);
                let rest = state
                    .order
                    .drop_head(queue, |next| next.buffer.same(&buffer));
                for held in &rest {
                    state.binary.dropped(&held.binary_needs);
                }
                spent.extend(rest.into_iter().map(|held| held.buffer));
                buffer.failed(error, &state.values);
                spent.push(buffer);
                return submitted;
            }
        }
    }
    submitted
}

/// The Vulkan waits for `held`'s events' signal indices where a thread about to block may submit
/// it itself ([`Scheduler::submit_before_waiting`]); `None` where it may not.
///
/// It is ready as [`submit_ready`] asks of a head: its binary semaphores' signals submitted, no
/// rebind encoding encoders of its event group again, no CPU decode queued into a twin it may
/// read, and every event value it waits for reached by a submitted signal or by the host. And
/// neither the thread nor the program has anything to do for it:
///
/// - no segments, so no host step the thread records, and none in its [`Left`] work;
/// - no submission check, which runs the program's code on the thread that runs it;
/// - no submission handler and no scheduled handler, so its transition to `Scheduled` is plain
///   and runs none of the program's code where it is published
///   (`CommandBufferShared::submits_without_the_program`);
/// - no switch of a fetched target's storage form riding on it, which the submitting thread
///   settles and announces (`texture/storage_form.rs`);
/// - and a `VkQueue` whose batches never wait in a group (`command_buffer/group.rs`), which is
///   the thread's to hold and to send.
fn plain_ready(
    held: &Held,
    binary: &BinarySignals<vk::Semaphore>,
    decodes: &BTreeMap<ResourceKey, usize>,
    untracked_decodes: usize,
) -> Option<Vec<(vk::Semaphore, u64)>> {
    if held.segments || held.checked {
        return None;
    }
    if !binary.ready(&held.binary_needs) {
        return None;
    }
    if held.buffer.reencode_pending() {
        return None;
    }
    if untracked_decodes > 0 || (!decodes.is_empty() && reads_decoding(held, decodes)) {
        return None;
    }
    if held.buffer.groups() || !held.buffer.submits_without_the_program() {
        return None;
    }
    event_waits(held)
}

/// Submit the batches held for one kernel submission (`command_buffer/group.rs`), sent by `end`
/// (counted), a section of the exit guard, then publish them, or fail their command buffers
/// ([`settle_group`]). `Err` once the process is exiting: they stay held.
fn submit_group(
    state: &mut State,
    (spent, settled): (&mut Spent, &mut Settled),
    handlers: &mut Handlers,
    end: GroupEnd,
) -> Result<(), Exiting> {
    let Some(device) = (state.grouped.first()).map(|(_, held)| held.buffer.device().clone()) else {
        return Ok(());
    };
    let busy = exit_guard::enter().ok_or(Exiting)?;
    device.shared().counts.group_ended(end);
    let queue = state.group.queue();
    let State { group, waited, .. } = &mut *state;
    // The call's own result is a batch's submitted after the group, of which there is none.
    let _ = group.submit(&device, &mut waited[queue], None);
    drop(busy);
    if let Some(outcome) = state.group.take_outcome() {
        settle_group(state, outcome, (spent, settled), handlers);
    }
    Ok(())
}

/// Publish the batches held for a kernel submission that was made, in order, as [`submit_ready`]
/// publishes one; or, where it failed, fail each one's command buffer, once, and drop the rest of
/// its batches, as a batch whose submission failed. Their holds go to `spent`.
fn settle_group(
    state: &mut State,
    outcome: Result<(), CommandBufferError>,
    (spent, settled): (&mut Spent, &mut Settled),
    handlers: &mut Handlers,
) {
    let State {
        order,
        binary,
        values,
        grouped,
        group,
        ..
    } = state;
    match outcome {
        Ok(()) => {
            for ((_, held), submission) in grouped.drain(..).zip(group.made()) {
                let Held {
                    buffer,
                    last,
                    waits: events,
                    binary_needs,
                    binary_signals,
                    ..
                } = held;
                drop(events);
                binary.submitted(&binary_needs, &binary_signals);
                if submission.runs_handlers() {
                    handlers.push(buffer.hold());
                }
                buffer.left_group();
                buffer.publish(submission);
                if last && buffer.settle_storage_forms() {
                    settled.push(buffer.hold());
                }
                spent.push(buffer);
            }
        }
        Err(error) => {
            let mut failed: Vec<Hold<CommandBufferShared>> = Vec::new();
            for (queue, held) in grouped.drain(..) {
                let Held {
                    buffer,
                    waits: events,
                    binary_needs,
                    ..
                } = held;
                drop(events);
                binary.dropped(&binary_needs);
                buffer.left_group();
                if !failed.iter().any(|done| done.same(&buffer)) {
                    // The rest of the command buffer never runs (dropped here, before it can
                    // finish).
                    let rest = order.drop_head(queue, |next| next.buffer.same(&buffer));
                    for held in &rest {
                        binary.dropped(&held.binary_needs);
                    }
                    spent.extend(rest.into_iter().map(|held| held.buffer));
                    buffer.failed(error, values);
                    if buffer.settle_storage_forms() {
                        settled.push(buffer.hold());
                    }
                    failed.push(buffer.hold());
                }
                spent.push(buffer);
            }
        }
    }
}

/// Whether `held` may read a texture of `decodes`: its footprint names one, or it has none it can
/// know before its submission (sparse work, a segment whose uses its host step finds).
fn reads_decoding(held: &Held, decodes: &BTreeMap<ResourceKey, usize>) -> bool {
    held.buffer
        .footprint(held.index)
        .is_none_or(|footprint| footprint.names_any(|key| decodes.contains_key(key)))
}

/// End the switches of storage forms riding on the `settled` command buffers
/// (`texture/storage_form.rs`) on this thread, with the scheduler's lock free (the device's switch
/// lock is taken before it, never under it), before their transitions to `Scheduled` go on
/// ([`run_submission_handlers`]): the program sees each swap, and its announcement, by the time it
/// can see its command buffer scheduled. A thread that made a command buffer meanwhile may have
/// ended it first. Each end is a section of the exit guard (it makes Vulkan calls and announces):
/// once the process is exiting, the switch stays as it is.
fn settle(settled: &mut Settled) {
    for buffer in settled.iter() {
        let Some(busy) = exit_guard::enter() else {
            break;
        };
        buffer.device().finish_storage_forms();
        drop(busy);
    }
    settled.clear();
}

/// The command buffers whose storage forms the scheduler settled, to end without its lock.
type Settled = Vec<Hold<CommandBufferShared>>;

/// The command buffers whose last batch a pass submitted or held in a group with submission
/// handlers, in that order, for the submitting thread to run without the lock.
type Handlers = Vec<Hold<CommandBufferShared>>;

/// Run the submission handlers of `handlers`' command buffers on this thread, which holds none of
/// the runtime's locks, in submission order; each one's transition to `Scheduled` goes on after
/// its own (`CommandBufferShared::run_submission_handlers`), and after its switch of storage forms
/// ended ([`settle`], run first). Each hold goes after its handlers, outside every lock: it may be
/// its command buffer's last reference.
fn run_submission_handlers(handlers: &mut Handlers) {
    for buffer in handlers.drain(..) {
        buffer.run_submission_handlers();
    }
}

/// A batch for the thread to work on without its lock: its command buffer with the batch's index.
type Work = (Hold<CommandBufferShared>, usize);

/// Look at each line's head of a command buffer with a submission check that is otherwise
/// ready (its binary semaphores' signals submitted, its events' values reached, its host step
/// recorded, not `behind` an earlier commit) and waits for its checks: once every value it waits
/// for was reached on the device or by the host, not only by a submitted GPU signal (or the device
/// is lost), it goes to `left` as a check, unless there already. What it waits for goes to
/// `awaits` before the events are read again, as [`Scheduler::completions`] says. Whether a head
/// still waits for a GPU signal to land.
///
/// Where the line's batches may go in one kernel submission (`command_buffer/group.rs`), the
/// batches right behind a head that goes (its checks in `left` or none due, otherwise ready, its
/// events' values reached on the device or by the host) are looked at too, at most
/// [`GROUP_BATCHES`] less one and none committed after another line's batch
/// ([`Order::leading_mut`]), each that goes taking its checks to `left` likewise: so the next pass
/// finds them ready together. Each check still runs in commit order before its own batch's
/// submission, within the pass before it, where a head's alone runs after the batch before it
/// was submitted.
fn check_heads(state: &mut State, behind: &[u64], left: &mut Left<Work>, awaits: &Awaits) -> bool {
    let State { order, binary, .. } = state;
    let mut landing = false;
    order.leading_mut(behind, GROUP_BATCHES - 1, |held, head| {
        let otherwise_ready = binary.ready(&held.binary_needs)
            && !(held.segments && held.buffer.needs_host_step(held.index));
        let due = held.checked
            && held.buffer.needs_check(held.index)
            && !left.holds(Kind::Check, |(buffer, index)| {
                buffer.same(&held.buffer) && *index == held.index
            });
        // Whether it goes as far as its checks know: the batches after it may be checked now.
        let goes = |held: &Held| {
            otherwise_ready
                && held.buffer.groups()
                && event_waits(held).is_some_and(|waits| waits.is_empty())
        };
        if !head {
            if !goes(held) {
                return false;
            }
            if due {
                left.push(Kind::Check, (held.buffer.hold(), held.index));
            }
            return true;
        }
        count_waits_held(held);
        if !held.checked || !otherwise_ready || !due {
            return !due && goes(held);
        }
        match landing_waits(held) {
            None => return false,
            Some(waits) if !waits.is_empty() && !held.buffer.device().is_lost() => {
                awaits.events(&held.waits);
                fence(Ordering::SeqCst);
                if landing_waits(held).is_some_and(|waits| !waits.is_empty()) {
                    (held.buffer.device().shared().counts).head_awaiting_check_signals();
                    landing = true;
                    return false;
                }
            }
            Some(_) => {}
        }
        left.push(Kind::Check, (held.buffer.hold(), held.index));
        goes(held)
    });
    landing
}

/// The waits of `held`, a line's head, for a shared event's value that a submitted GPU signal
/// reaches: each chained on the `VkQueue` before and is held on the host now, so each cost moved
/// (`WorkCounts::shared_event_waits_held`, `event_timeline.rs`). Counted once per batch, at the
/// pass that first finds it leading its line: a batch behind another waits for that one, not for
/// its own events.
fn count_waits_held(held: &mut Held) {
    if held.counted_held {
        return;
    }
    let held_for_the_host = (held.waits.iter())
        .filter(|(event, value)| event.held_for_the_host(*value))
        .count();
    if held_for_the_host > 0 {
        held.counted_held = true;
        (held.buffer.device().shared().counts).shared_event_waits_held(held_for_the_host);
    }
}

/// Raise the makings of the pipelines each command queue's first held batch's host step records
/// to a head's (`budget::Urgency::Head`), once per batch, whatever else it waits for: every batch
/// committed after it on its queue waits behind it, and once a queue's command buffers are all
/// held, so does the program's thread. Metal makes a pipeline state at its creation, so no
/// submission of Metal's waits for a making; a head's making goes before the other committed
/// batches' and those nothing waits for, aged or not, whose tickets are earlier. Each making's
/// end wakes the thread (`CommandBuffer::commit` registers it, `when_made`).
fn hasten_heads(order: &mut Order<Held>) {
    for held in order.heads_mut(&[]) {
        if held.segments && !held.hastened {
            held.hastened = true;
            held.buffer.hasten_head(held.index);
        }
    }
}

/// Look at each line's head that is otherwise ready (not `behind` an earlier commit either,
/// its gate then covers that commit's work) and needs a host step: its gate is taken (the queues'
/// submitted values, the first time), and once every `VkQueue`'s completed value reached it (or the
/// device is lost), its step goes to `left`, unless there already. The gate goes to `awaits`
/// before the completed values are read, as [`Scheduler::completions`] says. Whether a head still
/// waits at its gate.
fn gate_heads(state: &mut State, behind: &[u64], left: &mut Left<Work>, awaits: &Awaits) -> bool {
    let State {
        order,
        values,
        binary,
        wake_by,
        ..
    } = state;
    *wake_by = None;
    let mut waiting = false;
    for held in order.heads_mut(behind) {
        if !held.segments
            || !binary.ready(&held.binary_needs)
            || event_waits(held).is_none()
            || !held.buffer.needs_host_step(held.index)
            || left.holds(Kind::Step, |(buffer, index)| {
                buffer.same(&held.buffer) && *index == held.index
            })
        {
            continue;
        }
        // A step waits for the pipelines its segments record, whose makings wake the thread as
        // they finish; one for them alone waits for no work on the GPU.
        let waits = held.buffer.step_waits(held.index);
        if waits.pipelines {
            // Its bound wakes the thread where its makings do not finish first.
            if let Some(at) = waits.until {
                *wake_by = Some(wake_by.map_or(at, |by| by.min(at)));
            }
            continue;
        }
        if !waits.gpu {
            left.push(Kind::Step, (held.buffer.hold(), held.index));
            continue;
        }
        let gate = held.gate.get_or_insert_with(|| values.clone());
        match waits.queue_alone {
            Some(queue) => awaits.gate_on(queue, gate[queue], gate.len()),
            None => awaits.gate(gate),
        }
        fence(Ordering::SeqCst);
        let device = held.buffer.device();
        let shared = device.shared();
        let reached = device.is_lost()
            || match waits.queue_alone {
                Some(queue) => shared.queues[queue].completed_value() >= gate[queue],
                None => shared
                    .queues
                    .iter()
                    .zip(gate.iter())
                    .all(|(queue, &value)| queue.completed_value() >= value),
            };
        if reached {
            left.push(Kind::Step, (held.buffer.hold(), held.index));
        } else {
            (shared.counts).head_at_its_gate();
            waiting = true;
        }
    }
    waiting
}

/// Whether the head of commit sequence number `sequence` waits behind a batch committed before it
/// that keeps its place ([`Order::is_behind`]): one whose binary semaphores' signals are submitted
/// and whose event values are reached, as is every batch before it on its line, and whose footprint
/// and the head's run in commit order ([`tracker::ordered`]). `memo` holds what this pass found.
fn is_behind(
    order: &Order<Held>,
    binary: &BinarySignals<vk::Semaphore>,
    memo: &mut Behind<Option<Footprint<ResourceKey>>>,
    sequence: u64,
) -> bool {
    order.is_behind(
        sequence,
        memo,
        (
            &mut |held: &Held| binary.ready(&held.binary_needs) && events_reachable(held),
            &mut |held: &Held| held.buffer.footprint(held.index),
            &|earlier: &Option<Footprint<ResourceKey>>, later: &Option<Footprint<ResourceKey>>| {
                tracker::ordered(earlier.as_ref(), later.as_ref())
            },
        ),
    )
}

/// Whether every event value `held` waits for is reached by a submitted signal or the host:
/// [`event_waits`] would answer, for a batch the order does not take yet.
fn events_reachable(held: &Held) -> bool {
    held.waits
        .iter()
        .all(|(event, value)| event.wait(*value).is_some())
}

/// The waits of `held`, a line's head whose checks wait for its events' values to land, for the
/// signal indices of the submitted GPU signals that reach the values not yet reached, a shared
/// event's whose wait is held on the host included (`EventShared::landing`): what the scheduler's
/// thread may wait for on the GPU itself (the module docs). `None` while a value is reached by
/// neither a submitted signal nor the host.
fn landing_waits(held: &Held) -> Option<Vec<(vk::Semaphore, u64)>> {
    let mut waits = Vec::new();
    for (event, value) in &held.waits {
        match event.landing(*value)? {
            WaitFor::Reached => {}
            WaitFor::Index(index) => waits.push((event.semaphore(), index)),
        }
    }
    Some(waits)
}

/// The Vulkan waits of `held` for its events' signal indices, when every value it waits for is
/// reached by a submitted signal or the host; `None` while one is not.
fn event_waits(held: &Held) -> Option<Vec<(vk::Semaphore, u64)>> {
    let mut waits = Vec::new();
    for (event, value) in &held.waits {
        match event.wait(*value)? {
            WaitFor::Reached => {}
            WaitFor::Index(index) => waits.push((event.semaphore(), index)),
        }
    }
    Some(waits)
}
