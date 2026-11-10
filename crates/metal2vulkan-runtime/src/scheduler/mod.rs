//! The device's scheduler: one thread per device that submits the committed command buffers'
//! batches to the device's `VkQueue`s.
//!
//! - **Handed to the thread** ([`Inbox`]). `commit` leaves its batches in the inbox and returns
//!   at once, as Metal's `commit` returns before its command buffer is scheduled: it makes no
//!   pass and submits nothing, so no driver call holds it. This thread takes them in, in commit
//!   order, at its next pass, and submits what is ready. The commit wakes the thread where it
//!   sleeps, once a sleep: the waker that finds it asleep marks it awake as it notifies it, under
//!   the thread's own lock for its sleep ([`Sleep`]), so a commit made before the thread ran finds
//!   it awake and makes no system call ([`Sleep::mark`]). No pass takes that lock, so a commit
//!   never waits for the scheduler's lock, which a pass holds across its submissions. Where it is
//!   making a pass, the thread looks at the inbox again once its pass ends. The
//!   order sees every commit in commit order, and every batch is planned as it is submitted. This
//!   thread stops looking at the inbox again at a pass that leaves it a check or a host step to
//!   hand over, and before each one it runs passes again for as long as the inbox holds commits,
//!   and looks at it before it sleeps ([`left`]): a commit waits for the pass under way, the
//!   submission handlers that pass leaves, and the pass that takes it in, never for a check or a
//!   step. A commit wakes this thread alone.
//!
//! - **A committing thread may submit for itself** ([`Scheduler::submit_at_commit`],
//!   `CommandBuffer::commit_and_submit`, a runtime extension). Its commit is posted with no wake,
//!   and where no passer is counted it makes one pass as a thread about to block does: it takes
//!   in what the inbox holds and submits, in commit order, the line heads that are ready and need
//!   nothing of this thread (no host step, no switch of a storage form, no rebind's encoding; a
//!   batch that may wait in a group goes alone where its `VkQueue` runs at most one earlier
//!   submission not seen complete, as the group would go at once, and is left to this thread to
//!   hold where two or more run) and wait behind no earlier commit, as this thread's own pass picks them
//!   ([`Order::next`], [`is_behind`]), up to [`PASS_SUBMISSIONS`]; their submission handlers then
//!   run on it, in their turn ([`Turns`]), and their transitions to `Scheduled` run on it where
//!   plain, else go to their queues' contexts. The first head in commit order ready but for its
//!   submission checks, every value it waits for reached on the device or by the host, and behind
//!   no earlier commit, has its checks run there too, without the lock, the head marked so this
//!   thread leaves them ([`claim_check`]), and is submitted by a pass of the committing thread's
//!   right after. A head another queue holds back (its checks waiting for a signal to land, or run
//!   by this thread) so holds back no later commit that does not wait behind it. What it leaves
//!   goes to this thread, which it wakes, and a pass under way takes the commit in instead. So no
//!   wake and no thread hop lies on a command buffer's way to its `VkQueue`, its submission's cost
//!   on the committing thread. Where it submitted alone a batch that may wait in a group, which
//!   every other pass leaves to this thread, it wakes this thread after its submissions all the
//!   same, as that commit's hand-off would have, so this thread sleeps and wakes as it does where
//!   such batches are its own. A thread left asleep across those commits would sleep longer, and
//!   where the host's cores idle deeper the longer they sleep, its next wake and pass, for a
//!   `commit`, take several times longer (`tests::completion_latency`, RADV).
//! - **Submission handlers run in submission order, whichever thread runs them** ([`Turns`]).
//!   Each pass that took batches with submission handlers takes a ticket under the lock, in the
//!   order the passes ran; the handlers of a ticket run once those of every earlier one ran, on
//!   the thread that made the pass, without the lock.
//! - **A thread about to block submits for itself** ([`Scheduler::submit_before_waiting`]).
//!   Metal encodes and commits inline on the program's thread, and a thread that commits and then
//!   waits for the completion would otherwise wait for a wake and a whole pass before its work
//!   reached the `VkQueue`. So, where the inbox still holds commits and no pass is under way, that
//!   thread becomes the passer ([`Inbox::pass_once`]), takes them into the order and submits what
//!   needs nothing of this thread and runs none of the program's code ([`submit_plain_ready`]):
//!   batches with no host step, no submission check, no handler of any kind, no switch of storage
//!   forms, and never one of a `VkQueue` whose batches group. It takes the ready heads in commit
//!   order, passing over a held one only for a head that waits behind no earlier commit, as this
//!   thread's own pass does; it leaves [`Awaits`], [`Left`] and the group to this thread, which
//!   it wakes for what it left; and it never looks at the
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
//!   ([`Order::is_behind`]).
//! - **Many logical queues on few `VkQueue`s.** A logical queue keeps its `VkQueue` from a
//!   command buffer's making to its completion, and moves only where it holds no other
//!   uncompleted one (`queue_share.rs`); each `VkQueue` signals its own timeline semaphore with
//!   every batch, which is the [`Point`] the tracker and the completion turns know the batch by.
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
//!   The command buffer's transition to `Scheduled` waits for them. They hold back no completion
//!   of what the passes took: a command buffer taken with no submission whose `VkQueue` has no
//!   work in flight, whose completion turn is this thread's (`Completion::turn_owed`), is
//!   finished before they run.
//! - **Host steps**. A batch that executes commands the GPU may write
//!   (`CommandBufferShared::needs_host_step`) is held, once otherwise ready, until every
//!   `VkQueue` completed what was submitted to it by then (its gate, a snapshot of the queues'
//!   values): only then are the records the work before it wrote known. The thread then hands
//!   the batch's step to the steps' threads ([`steps`], a step a thread, up to eight at once),
//!   one of which records its segments (`CommandBufferShared::host_step`), each step a section of
//!   the exit guard, and wakes this one, which submits the batch in its order. Meanwhile it
//!   submits other ready batches, and hands over other queues' steps, those of other queues
//!   whose footprint shares nothing with the held batch's (`CommandBufferShared::footprint`: a
//!   step's, what `commit` bounded its uses by): one queue's long step holds back no other
//!   queue's that Metal does not order after it. The
//!   completion turns wake it as batches complete while a gate is pending. A
//!   batch whose step is a predicated loop's check alone waits for its own `VkQueue`'s work
//!   alone; while the check's thread runs the loop's laying again, the batch waits as for a
//!   making, which the thread's end wakes the thread for (`command_buffer/rerun.rs`).
//! - **Submission checks** (`command_buffer/checks.rs`). A batch of a command buffer with a check
//!   is held, once otherwise ready and its host step recorded, until the checks of Metal's
//!   submissions in it ran (`CommandBufferShared::run_checks`), which the thread runs as it runs
//!   host steps: without its lock, each pass a section of the exit guard; or a committing thread,
//!   for the head it submits next ([`claim_check`]). Its checks wait, besides, until every event
//!   value it waits for was reached on the device or by the host: a value only a submitted GPU
//!   signal reaches is held for until that signal landed, as Metal submits (and checks) only
//!   then. A completion turn sees each finished batch's event signals before it
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
//! - **Idle, it sleeps until it is woken.** The thread wakes on a commit, a host signal or a
//!   stop, each of which marks its state under its lock and notifies it where it sleeps
//!   ([`Inbox::post`], [`Scheduler::waker`], `Scheduler::stop`). With no batch held in its order
//!   or its group, no head's bound and no driver's stall to take, nothing else can give it work,
//!   so it sleeps with no timeout: an idle device's scheduler does not wake at all
//!   (`WorkCounts::scheduler_sleep_slices`). With a batch held it still wakes at least every
//!   [`WAIT_SLICE`], beside the completion that lets its batch go.
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
//!   values that submitted GPU signals reach (no gate, none reached yet; a shared event's, whose
//!   wait is held on the host rather than handed to a `VkQueue`, `event_timeline.rs`, and a
//!   checked head's for a shared event's value: [`landing_waits`]), the thread sleeps in one
//!   `vkWaitSemaphores` for any of those values, or for the device's nudge semaphore to reach its
//!   next value, instead of on its condition variable: the signal wakes it from the driver, with
//!   no completion turn and no waker between, and it observes the values itself
//!   (`EventShared::value`, which waits for what it reads), then checks and submits. Every other
//!   waker (a commit, a host signal, a finished decode, a lost device, a stop) signals the nudge,
//!   under the lock of the thread's sleep ([`Sleep`]), where the thread waits so; a completion
//!   turn needs not, as the thread's own
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
mod steps;

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
use super::command_buffer::completion::trace;
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
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
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
        // semaphore's (the values rise under the lock of the thread's sleep, [`Sleep`]).
        let signalled = unsafe { self.functions.get().signal_semaphore(&info) };
        exit_guard::called(exit_guard::Call::SignalSemaphore);
        signalled.is_ok()
    }
}

/// How the thread slept as a waker woke it ([`Sleep::mark`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Asleep {
    /// It made a pass, was about to look again, or a waker before woke it: no wake needed.
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
    /// A commit left its batches to the thread, or a committing thread's pass submitted alone a
    /// batch that may wait in a group ([`Scheduler::submit_at_commit`]).
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
    /// How the thread sleeps, under a lock of its own, which no pass takes ([`Sleep`]).
    sleep: Mutex<Sleep>,
    /// The thread's sleep on its condition variable, under `sleep`'s lock.
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
    /// The commits the passes took into the order, any thread's, each counted as the pass that
    /// took it in ends, after its submissions ([`SchedulerShared::took_in`]).
    taken: AtomicU64,
    /// The device's nudge, where the thread waits for the GPU itself (the module docs).
    nudge: OnceLock<Nudge>,
    /// The device's cleared encoder logs (`replay.rs`): an encoder takes one, and its log comes
    /// back as the encoder ends, or in its command buffer's release.
    spare_logs: SpareLogs,
    /// Where the device's driver stalls in one signalling submission of every 1024 to a queue,
    /// what the thread takes those stalls idle with (the module docs).
    stalls: OnceLock<IdleStalls>,
    /// The passes' turns to run their submission handlers (the module docs).
    turns: Turns,
    /// The host steps handed to the steps' threads (`steps.rs`).
    steps: steps::Steps,
}

/// The passes' turns to run the submission handlers they took (the module docs): a pass that took
/// batches with handlers takes a ticket under the scheduler's lock (`State::handler_ticket`), and
/// runs them once every earlier ticket's ran.
#[derive(Default)]
struct Turns {
    /// The next ticket to run, and how many threads wait for theirs.
    served: Mutex<(u64, usize)>,
    turned: Condvar,
}

impl Turns {
    /// Run `handlers`' submission handlers on this thread, which holds none of the runtime's
    /// locks, each segment of them (`segments`: where it ends in `handlers`, and its ticket) in
    /// its turn, in submission order within it; none once the process is exiting, as the
    /// scheduler's thread then runs nothing more. Each one's transition to `Scheduled` goes on
    /// after its own (`CommandBufferShared::run_submission_handlers`), and after its switch of
    /// storage forms ended ([`settle`], run first). Each hold goes after its handlers, outside
    /// every lock: it may be its command buffer's last reference. Both are emptied.
    fn run(&self, handlers: &mut Handlers, segments: &mut Vec<(usize, u64)>) {
        let mut start = 0;
        let mut held = handlers.drain(..);
        for (end, ticket) in segments.drain(..) {
            let turn = self.wait(ticket);
            for buffer in held.by_ref().take(end - start) {
                if turn {
                    buffer.run_submission_handlers();
                }
            }
            start = end;
            self.served(ticket);
        }
        debug_assert!(held.next().is_none(), "a submission handler with no turn");
    }

    /// Wait for `ticket`'s turn; false once the process is exiting.
    fn wait(&self, ticket: u64) -> bool {
        let mut served = lock(&self.served);
        while served.0 < ticket {
            if exit_guard::closed() {
                return false;
            }
            served.1 += 1;
            served = (self.turned)
                .wait_timeout(served, exit_guard::WAKE)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
            served.1 -= 1;
        }
        true
    }

    /// `ticket`'s turn ended: the next one's comes.
    fn served(&self, ticket: u64) {
        let mut served = lock(&self.served);
        served.0 = served.0.max(ticket + 1);
        if served.1 > 0 {
            self.turned.notify_all();
        }
    }
}

impl SchedulerShared {
    /// A pass that took in `taken` commits ended, its submissions made and counted, under the
    /// lock: counted, released after them (`WorkCounts::commits_taken_in`).
    fn took_in(&self, taken: usize) {
        if taken > 0 {
            self.taken.fetch_add(taken as u64, Ordering::Release);
        }
    }

    /// Wake the thread, which makes no pass, for what the caller wrote before (a commit posted, a
    /// nudge, the state marked under its lock), signalling `nudge` where the thread waits for the
    /// GPU: under the lock of its sleep alone ([`Sleep`]), which no pass holds, so the waker never
    /// waits for a pass's submissions.
    fn wake_thread(&self, woken: Woken, nudge: Option<&Nudge>) {
        let asleep = lock(&self.sleep).mark(nudge);
        self.woken(woken, asleep);
    }

    /// A waker found the thread making no pass: wake it where it sleeps (`asleep`, read under the
    /// lock of its sleep; one that sleeps later reads what the waker wrote first), counted as
    /// `woken`; else the notify, a system call, is spared.
    fn woken(&self, woken: Woken, asleep: Asleep) {
        match asleep {
            // The thread alone waits on it.
            Asleep::OnCondvar => {
                self.wakings[woken as usize].fetch_add(1, Ordering::Relaxed);
                self.wake.notify_one();
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
    /// A host signal marked the state since the thread last looked (a commit and a nudge are
    /// the inbox's, `inbox.rs`).
    dirty: bool,
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
    /// The next pass's ticket to run its submission handlers ([`Turns`]).
    handler_tickets: u64,
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
    /// The ticket of a pass that took batches with submission handlers ([`Turns`]), taken under
    /// the lock in the order the passes ran.
    fn handler_ticket(&mut self) -> u64 {
        let ticket = self.handler_tickets;
        self.handler_tickets += 1;
        ticket
    }
}

/// How the scheduler's thread sleeps, under a lock of its own, which no pass takes: a waker that
/// finds no pass under way wakes the thread through it alone ([`SchedulerShared::wake_thread`]),
/// having written what it asks for first (a commit or a nudge in the inbox, or the state marked
/// under the scheduler's lock), so it never waits for a pass's submissions. The thread takes this
/// lock under the scheduler's as it goes to sleep, reads the inbox again, and lets go of the
/// scheduler's lock before it sleeps: a waker's write is read there, or the waker finds it asleep.
#[derive(Default)]
struct Sleep {
    /// The thread sleeps on its condition variable and no waker notified it yet: a waker
    /// notifies it only then, and clears this as it does ([`Sleep::mark`]).
    asleep: bool,
    /// The thread waits for the GPU, and for the nudge to reach this value: a waker signals it
    /// to that value under this lock, instead of notifying (the module docs).
    on_gpu: Option<u64>,
    /// The nudge semaphore's value: the last a waker signalled it to.
    nudge_value: u64,
}

impl Sleep {
    /// A waker's wake (the module docs), for what it wrote before: how the thread sleeps. Where it
    /// waits for the GPU, `nudge` is signalled to the value it waits for, here, under this lock;
    /// none is given for a completion turn's wake, whose signal the thread's own wait sees. Where
    /// it sleeps on its condition variable, it is marked awake here, for this waker to notify: it
    /// then passes for what the waker wrote, so a later waker before it ran notifies nothing (as
    /// the nudge, once signalled, is not signalled again).
    fn mark(&mut self, nudge: Option<&Nudge>) -> Asleep {
        if self.asleep {
            self.asleep = false;
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
    /// Who took its checks to run, once.
    checks_taken: Option<Taken>,
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
                stopping: false,
                decodes: BTreeMap::new(),
                untracked_decodes: 0,
                kept: KeptBarriers::default(),
                handler_tickets: 0,
                wake_by: None,
                lists: BatchLists::default(),
                group: Group::default(),
                grouped: Vec::new(),
            }),
            sleep: Mutex::new(Sleep::default()),
            wake: Condvar::new(),
            awaits: Awaits::default(),
            sleeps: AtomicU64::new(0),
            wakings: Default::default(),
            spared: AtomicU64::new(0),
            gpu_waits: AtomicU64::new(0),
            passes: AtomicU64::new(0),
            taken: AtomicU64::new(0),
            nudge: OnceLock::new(),
            spare_logs: SpareLogs::default(),
            stalls: OnceLock::new(),
            turns: Turns::default(),
            steps: steps::Steps::default(),
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
        hand: Hand,
    ) {
        let shared = &*self.shared;
        let committed = Committed {
            buffer: buffer.hold(),
            queue,
            batches,
        };
        buffer.device().shared().counts.commit_handed_off();
        if hand == Hand::Pass {
            shared.inbox.post_quietly(committed);
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            trace::submitting("posted");
            return;
        }
        if shared.inbox.post(committed) {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            trace::submitting("posted");
            shared.wake_thread(Woken::Commit, shared.nudge.get());
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            trace::submitting("woke the thread");
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
            let taken = take_in(state, commits, &mut spent);
            let mut none = Handlers::new();
            let waiting = (Passer::Waiting, &mut none);
            submitted = submit_plain_ready(state, &mut spent, waiting, PASS_SUBMISSIONS).batches;
            debug_assert!(
                none.is_empty(),
                "a waiting thread took a submission handler"
            );
            left_over = !state.order.is_empty() || !state.grouped.is_empty();
            shared.took_in(taken);
        });
        if !passed {
            return;
        }
        device.shared().counts.waiter_passed(submitted);
        // The holds go outside the lock, and a command buffer's last reference to a handler
        // thread: no release of the program's objects (a no-copy buffer's deallocator among them)
        // runs on a thread blocking in `wait_until_completed`, where Metal runs none.
        for buffer in spent.drain(..) {
            if let Err(last) = buffer.let_go_short_of_last() {
                device.let_go_elsewhere(last);
            }
        }
        // A command buffer that needs no GPU work, taken with no submission behind no work in
        // flight, is finished by this thread, which holds no lock: it runs only the transitions
        // that run none of the program's code, and posts the rest (`Completion::turn_owed`).
        device.shared().completion.turn_owed();
        // This pass never looks again (`inbox.rs`): what is left in the order, and anything
        // posted or nudged during it, goes to the thread that makes passes.
        if left_over || shared.inbox.passed_over() {
            self.wake_for(Woken::Host);
        }
    }

    /// The committing thread's own pass, right after its commit was posted with no wake
    /// ([`Hand::Pass`], `CommandBuffer::commit_and_submit`; the module docs): where no passer is
    /// counted it becomes the passer ([`Inbox::pass_once_then`]), takes in what the inbox holds,
    /// and submits the heads [`plain_ready`] holds for, in commit order, each behind no earlier
    /// commit ([`submit_plain_ready`], [`Passer::Committing`]); then, still counted and without the
    /// lock, it runs their submission handlers in their turn ([`Turns`]) and lets go of what the
    /// pass held, a command buffer's last reference on a handler thread. Where a head left waits
    /// only for its submission checks and this thread may run them ([`claim_check`]), it runs them
    /// then, as the scheduler's thread does (`CommandBufferShared::run_checks`), and passes again
    /// under the lock ([`Inbox::pass_again`]), which submits that head; and so on, up to
    /// [`PASS_SUBMISSIONS`] batches in all. What it leaves, and anything committed or nudged
    /// meanwhile, goes to the thread, which it wakes ([`Woken::Host`]); else, where it submitted
    /// alone a batch that may wait in a group ([`Plain::alone`]), it wakes the thread as that
    /// batch's hand-off would have ([`Woken::Commit`], the module docs). Where a passer is counted,
    /// that pass takes the commit in. After each pass, before its submission handlers and its
    /// check, it makes the completion turn a command buffer that needs no GPU work left it
    /// (`Completion::turn_owed`), so neither waits for that completion.
    ///
    /// It runs the program's code only in the submission checks of the heads it checks and the
    /// submission handlers of the batches it submitted: a scheduled handler is run on its queue's
    /// handler thread, as the thread's passes leave it.
    pub(crate) fn submit_at_commit(&self, device: &Device) {
        let shared = &*self.shared;
        let passed = shared.inbox.pass_once_then(
            |state, commits| {
                let mut spent = Spent::new();
                let taken = take_in(state, commits, &mut spent);
                let pass = AtCommit::pass(state, spent, PASS_SUBMISSIONS);
                shared.took_in(taken);
                pass
            },
            |mut pass| {
                let (mut submitted, mut alone, mut checked) = (0, 0, 0);
                loop {
                    submitted += pass.submitted;
                    alone += pass.alone;
                    let left_over = pass.left_over;
                    // The completion turn a command buffer the pass took with no submission left
                    // this thread, before the handlers and the check, as the scheduler's thread
                    // makes it before its handlers ([`round`]).
                    device.shared().completion.turn_owed();
                    let Some(check) = pass.let_go(&shared.turns, device) else {
                        return (submitted, alone, checked, left_over);
                    };
                    // None once the process is exiting, as the thread then runs nothing more.
                    if !run_work(&self.shared, Kind::Check, check) {
                        return (submitted, alone, checked, left_over);
                    }
                    checked += 1;
                    let room = PASS_SUBMISSIONS.saturating_sub(submitted);
                    pass = (shared.inbox)
                        .pass_again(|state| AtCommit::pass(state, Spent::new(), room));
                }
            },
        );
        let Some((submitted, alone, checked, left_over)) = passed else {
            return;
        };
        device.shared().counts.committer_passed(submitted, checked);
        // After the submissions, so none waits for the wake.
        if left_over || shared.inbox.passed_over() {
            self.wake_for(Woken::Host);
        } else if alone > 0 {
            self.wake_for(Woken::Commit);
        }
    }

    /// What makes the thread look for work again from any thread: a pipeline a held batch's
    /// host step waits for was made (`render/pending.rs`). It holds the thread's state weakly,
    /// so a making it waits on keeps nothing of the scheduler alive.
    pub(crate) fn waker(&self) -> impl Fn() + Clone + Send + 'static {
        let shared = Arc::downgrade(&self.shared);
        move || {
            if let Some(shared) = shared.upgrade() {
                lock(shared.inbox.state()).dirty = true;
                shared.wake_thread(Woken::Host, shared.nudge.get());
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
        state.dirty = true;
        drop(state);
        (self.shared).wake_thread(Woken::Host, self.shared.nudge.get());
    }

    /// Whether this thread is the scheduler's (a submission check runs on it).
    pub(crate) fn is_current(&self) -> bool {
        lock(&self.thread)
            .as_ref()
            .is_some_and(|handle| handle.thread().id() == std::thread::current().id())
    }

    /// `f`'s answer under the state's lock, which every submission holds from an event signal's
    /// index to its keep (`command_buffer/submit.rs`): an index `f` takes of an event's timeline
    /// (`EventShared::reserve_signal`) is never one a batch takes meanwhile. `f` takes no lock the
    /// scheduler takes before the state's.
    pub(crate) fn under_lock<R>(&self, f: impl FnOnce() -> R) -> R {
        let _state = lock(self.shared.inbox.state());
        f()
    }

    /// Look for work again: an event's value rose, a pipeline was made, a rebind encoded a
    /// command buffer again, or the device was lost (each a host signal). Where a pass is under
    /// way, it passes again once done, and wakes the thread for a step or a check that finds;
    /// else the thread is woken here, under the lock of its sleep alone
    /// ([`SchedulerShared::wake_thread`]): the waker never waits for a pass's submissions.
    pub(crate) fn wake(&self) {
        self.wake_for(Woken::Host);
    }

    /// [`wake`](Self::wake), counted as `woken`.
    fn wake_for(&self, woken: Woken) {
        self.wake_with(woken, self.shared.nudge.get());
    }

    /// [`wake_for`](Self::wake_for), signalling `nudge` where the thread waits for the GPU.
    fn wake_with(&self, woken: Woken, nudge: Option<&Nudge>) {
        if !self.shared.inbox.nudge() {
            self.shared.wake_thread(woken, nudge);
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

    /// Whether the thread sleeps, on its condition variable or on the GPU, and no waker woke it
    /// since ([`Sleep`]): a test that wakes it waits for this, rather than for a time.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn asleep(&self) -> bool {
        let sleep = lock(&self.shared.sleep);
        sleep.asleep || sleep.on_gpu.is_some()
    }

    /// The passes the thread made so far (`WorkCounts::scheduler_passes`).
    pub(crate) fn passes(&self) -> u64 {
        self.shared.passes.load(Ordering::Relaxed)
    }

    /// The commits the passes took in so far, each counted as its pass ended
    /// (`WorkCounts::commits_taken_in`): read after the pass's own counts, which it makes visible.
    pub(crate) fn commits_taken_in(&self) -> u64 {
        self.shared.taken.load(Ordering::Acquire)
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
        lock(self.shared.inbox.state()).stopping = true;
        lock(&self.shared.sleep).mark(self.shared.nudge.get());
        self.shared.wake.notify_all();
        self.shared.steps.stop();
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
fn run(shared: &Arc<SchedulerShared>) {
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
            // Nothing held, no bound, no stall to take: only a commit, a host signal or a stop
            // gives the thread work, and each notifies it (the module docs).
            let idle = state.order.is_empty()
                && state.grouped.is_empty()
                && bound.is_none()
                && stalls.is_none();
            let slice = bound.map_or(slice, |left| left.min(slice));
            let on_gpu = shared.nudge.get().filter(|_| shared.gpu_waits(&mut gpu));
            // The lock of the sleep ([`Sleep`]), under the scheduler's: a waker that found no
            // pass under way wrote what it asks for before it took it, so its commit or nudge is
            // read here, or it finds the thread asleep and wakes it.
            let mut sleep = lock(&shared.sleep);
            if shared.inbox.holds_commits() || shared.inbox.nudged() {
                continue;
            }
            shared.sleeps.fetch_add(1, Ordering::Relaxed);
            if let Some(nudge) = on_gpu {
                state = wait_for_gpu(shared, (state, sleep), nudge, &mut gpu, (slice, short));
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                trace::submitting("thread woke");
                continue;
            }
            sleep.asleep = true;
            // A waker that marks the state under the scheduler's lock takes it once this lets
            // go of it, and then the sleep's, which the wait lets go of.
            drop(state);
            let timed_out = match idle {
                true => {
                    sleep = (shared.wake.wait(sleep))
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    false
                }
                false => {
                    let (woke, waited) = (shared.wake.wait_timeout(sleep, slice))
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    sleep = woke;
                    waited.timed_out()
                }
            };
            sleep.asleep = false;
            drop(sleep);
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            trace::submitting("thread woke");
            if timed_out && !short {
                shared.wakings[Woken::Slice as usize].fetch_add(1, Ordering::Relaxed);
            }
            state = lock(shared.inbox.state());
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
            |kind, work| run_work(shared, kind, work),
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
/// the idle's length before a driver's stall is taken, whose end is no slice's), without the locks
/// `state` and `sleep` hold, the scheduler's coming back taken. Where a value landed, its events
/// observe their counters, and the state is marked for a pass.
fn wait_for_gpu<'a>(
    shared: &'a SchedulerShared,
    (state, mut sleep): (MutexGuard<'a, State>, MutexGuard<'a, Sleep>),
    nudge: &Nudge,
    gpu: &mut GpuWait,
    (slice, short): (Duration, bool),
) -> MutexGuard<'a, State> {
    shared.gpu_waits.fetch_add(1, Ordering::Relaxed);
    let target = sleep.nudge_value + 1;
    sleep.on_gpu = Some(target);
    drop(sleep);
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
    let by_waker = lock(&shared.sleep).on_gpu.take().is_none();
    match waited {
        Some(Ok(())) if !by_waker => {
            shared.wakings[Woken::Completion as usize].fetch_add(1, Ordering::Relaxed);
            state.dirty = true;
        }
        // Nudged: the waker counted the waking, and the next look reads what it wrote.
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

/// Run one piece of work the thread's passes left it, or a committing thread's pass took for it
/// ([`claim_check`]), without the lock, a section of the exit guard: a check, which runs the
/// program's code; or hand a step, which records and may make what a dispatch makes, to the steps'
/// thread (`steps.rs`). False, with nothing run, once the process is exiting.
fn run_work(shared: &Arc<SchedulerShared>, kind: Kind, work: Work) -> bool {
    if kind == Kind::Step {
        return steps::post(shared, work);
    }
    let Some(busy) = exit_guard::enter() else {
        return false;
    };
    let (buffer, index) = &work;
    buffer.run_checks(*index);
    drop(busy);
    // Out of the section: it may be the command buffer's last reference.
    drop(work);
    true
}

/// One round of the thread's: a pass, looking at the inbox again while the passes leave the
/// thread no work and none stopped at [`PASS_SUBMISSIONS`] ([`Inbox::pass`]; such a pass marked
/// the state for the next round), what they leave added to `left` and what they let go of to
/// `passed` (empty, kept from round to round); then, without the lock, what they let go of: the
/// switches of storage forms settled ended, the completion turn a command buffer they took with
/// no submission left this thread made (`Completion::turn_owed`), and the submission handlers
/// run (on this thread, before its next pass; none of these once the process is exiting, as the
/// thread then runs nothing more, their command buffers staying `Committed`), the holds dropped
/// (the last reference to a command buffer may be among them, and with it its device, whose drop
/// stops this thread). False once the process is exiting.
fn round(shared: &SchedulerShared, left: &mut Left<Work>, passed: &mut Passed) -> bool {
    (shared.inbox).pass(|state, commits| {
        shared.passes.fetch_add(1, Ordering::Relaxed);
        let taken = pass(state, commits, (&mut *passed, &mut *left), &shared.awaits);
        shared.took_in(taken);
        !passed.exiting && left.is_empty() && !passed.more
    });
    passed.more = false;
    if passed.exiting {
        passed.settled.clear();
        passed.handlers.clear();
        for (_, ticket) in passed.turns.drain(..) {
            shared.turns.served(ticket);
        }
    } else {
        settle(&mut passed.settled);
        // A command buffer that needs no GPU work, taken with no submission behind no work in
        // flight, is finished by this thread, holding no lock (`Completion::turn_owed`), before
        // it runs a submission handler: a handler that waits for its completion (another
        // queue's, as Metal delivers) would otherwise wait for its own return. Every command
        // buffer the passes took is in `spent`.
        if let Some(buffer) = passed.spent.first() {
            buffer.device().shared().completion.turn_owed();
        }
        shared.turns.run(&mut passed.handlers, &mut passed.turns);
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
    /// Where each pass's submission handlers end in `handlers`, with the pass's ticket
    /// ([`Turns`]).
    turns: Vec<(usize, u64)>,
    more: bool,
    exiting: bool,
}

/// Take `commits` into the order, in commit order: each command buffer's batches, each with what
/// it waits for and signals, behind the batches its logical queue already holds. Each commit's own
/// hold goes to `spent`, for the caller to drop without the lock: its batches hold the command
/// buffer now. How many it took in, for the caller to count once its pass ends
/// ([`SchedulerShared::took_in`]).
fn take_in(state: &mut State, commits: &mut Vec<Committed>, spent: &mut Spent) -> usize {
    let taken = commits.len();
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if taken > 0 {
        trace::submitting("taken in");
    }
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
            checks_taken: None,
            hastened: false,
            counted_held: false,
            held_since: None,
        });
        order.push(queue, held);
        spent.push(buffer);
    }
    taken
}

/// A pass of the thread's: take `commits` into the order, in commit order, then submit what is
/// ready ([`submit_ready`]) into `passed` and `left`. A pass that stopped at
/// [`PASS_SUBMISSIONS`] marks the state, so the thread passes again at once after its round let
/// go of them. How many commits it took in.
fn pass(
    state: &mut State,
    commits: &mut Vec<Committed>,
    (passed, left): (&mut Passed, &mut Left<Work>),
    awaits: &Awaits,
) -> usize {
    let taken = take_in(state, commits, &mut passed.spent);
    state.dirty = false;
    let before = passed.handlers.len();
    let ready = submit_ready(
        state,
        (&mut passed.spent, &mut passed.settled),
        left,
        &mut passed.handlers,
        awaits,
    );
    // Its submission handlers run in this pass's turn ([`Turns`]).
    if passed.handlers.len() > before {
        let ticket = state.handler_ticket();
        passed.turns.push((passed.handlers.len(), ticket));
    }
    match ready {
        Ok(More(true)) => {
            state.dirty = true;
            passed.more = true;
        }
        Ok(More(false)) => {}
        Err(Exiting) => passed.exiting = true,
    }
    taken
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
/// the lock ([`Turns::run`]). Once it let go of [`PASS_SUBMISSIONS`] batches' holds,
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
            // a waiter for one of their command buffers sooner. Never behind the submission
            // handlers of another logical queue's command buffer, which the thread runs before
            // its next pass ([`holds_back_another_queue`]). Their submission makes no batch
            // ready: none of them signals a binary semaphore, and their events' signals counted
            // as submitted as they were held.
            drop(busy);
            if !waiting && !landing && !holds_back_another_queue(&state.grouped, handlers) {
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
            .submit_batch(held.index, held.last, &waits, false, state);
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

/// Whether the batches held in the group (`grouped`, with their logical queues) would wait, kept
/// past the pass, behind the submission handlers of a command buffer of another logical queue:
/// `handlers` holds the command buffers whose handlers the thread runs once its round's passes
/// end ([`round`]), held in the group or submitted. Such a group goes before they run
/// (`command_buffer/group.rs`, "Kept while its queue runs earlier work"), as Metal schedules a
/// queue's command buffers whatever a handler of another queue's does: on an M3 Max under macOS
/// 27.0.1, a command buffer's completed handler reached another queue's scheduled or completed
/// handler that waited for it in each of 30 runs, 7.6 to 9.3 ms after the wait began (probe
/// `cross_queue_handler_waits`). Handlers of the held batches' own queue keep it held: on one
/// queue Metal delivered no completed handler to a scheduled handler that waited for it (the
/// probe's `scheduled-one-queue`, the 5 s bound in each of 10 runs), so the runtime promises
/// nothing there.
fn holds_back_another_queue(grouped: &[(u64, Held)], handlers: &Handlers) -> bool {
    (grouped.iter())
        .any(|&(queue, _)| (handlers.iter()).any(|buffer| buffer.logical_queue() != queue))
}

/// The process is exiting ([`exit_guard`]): the thread submits no more.
struct Exiting;

/// One pass of a committing thread's under the lock ([`Scheduler::submit_at_commit`]), and what it
/// leaves that thread to do without the lock.
struct AtCommit {
    /// The holds it let go of.
    spent: Spent,
    /// The command buffers whose submission handlers it took, in the turn of its ticket.
    handlers: Handlers,
    segments: Vec<(usize, u64)>,
    /// The batches it submitted.
    submitted: usize,
    /// Of those, the batches it submitted alone where they may wait in a group
    /// ([`Plain::alone`]).
    alone: usize,
    /// The checks of a head it left, taken for the thread to run ([`claim_check`]).
    check: Option<Work>,
    /// It left batches in the order or the group.
    left_over: bool,
}

impl AtCommit {
    /// Submit the heads [`plain_ready`] holds for, in commit order, at most `room`
    /// ([`submit_plain_ready`]); then, with room for one more, take the checks of the first head
    /// left whose checks the committing thread may run ([`claim_check`]). Its holds go to `spent`.
    fn pass(state: &mut State, mut spent: Spent, room: usize) -> AtCommit {
        let mut handlers = Handlers::new();
        let committing = (Passer::Committing, &mut handlers);
        let submitted = submit_plain_ready(state, &mut spent, committing, room);
        let mut segments = Vec::new();
        if !handlers.is_empty() {
            segments.push((handlers.len(), state.handler_ticket()));
        }
        let check = match submitted.batches < room && state.grouped.is_empty() {
            true => claim_check(state),
            false => None,
        };
        let left_over = !state.order.is_empty() || !state.grouped.is_empty();
        AtCommit {
            spent,
            handlers,
            segments,
            submitted: submitted.batches,
            alone: submitted.alone,
            check,
            left_over,
        }
    }

    /// Without the lock: run its submission handlers in their turn ([`Turns`]) and let go of its
    /// holds, a command buffer's last reference on a handler thread; then answer the checks it
    /// took, for the thread to run.
    fn let_go(self, turns: &Turns, device: &Device) -> Option<Work> {
        let AtCommit {
            mut spent,
            mut handlers,
            mut segments,
            check,
            ..
        } = self;
        turns.run(&mut handlers, &mut segments);
        for buffer in spent.drain(..) {
            if let Err(last) = buffer.let_go_short_of_last() {
                device.let_go_elsewhere(last);
            }
        }
        check
    }
}

/// Submit, into `spent`, the batches a thread other than the scheduler's may submit itself
/// ([`Scheduler::submit_before_waiting`], [`Scheduler::submit_at_commit`]), at most `room` of
/// them: each the line head committed first of those [`plain_ready`] holds for that wait behind
/// no earlier commit, as the thread's own pass picks its next ([`Order::next`], [`is_behind`]). A
/// head held (for its checks, a host step, an event value) holds back no later commit of another
/// queue but one whose footprint runs in commit order with it while it keeps its place, as in the
/// thread's own pass. How many it submitted, and how many of them alone ([`Plain`]).
///
/// **It touches none of the thread's own state.** [`Awaits`] is what the thread's last pass left
/// it asleep on, its `vkWaitSemaphores` among them, and [`Left`] is the work that pass handed it:
/// a waiting thread that cleared the first would leave the thread asleep with work ready, and one
/// handed a piece of the second would run a host step, or the program's code in a submission
/// check, where Metal runs neither. So it reads `order`, `binary` and `decodes`, submits against
/// `tracker`, `values`, `waited`, `kept` and `lists` as a pass does, and leaves `awaits`, `left`,
/// the group, `dirty` and the sleeps alone. It makes no pass at all while the group holds a batch.
fn submit_plain_ready(
    state: &mut State,
    spent: &mut Spent,
    (passer, handlers): (Passer, &mut Handlers),
    room: usize,
) -> Plain {
    let mut submitted = Plain::default();
    while submitted.batches < room && state.grouped.is_empty() {
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
            let mut memo = Behind::default();
            order.next(
                |order, sequence| is_behind(order, binary, &mut memo, sequence),
                |held| plain_ready(held, binary, decodes, untracked, passer),
            )
        };
        let Some((queue, sequence, mut held, waits)) = taken else {
            drop(busy);
            return submitted;
        };
        let answer = held
            .buffer
            .submit_batch(held.index, held.last, &waits, true, state);
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
                // The predicate holds out every switch of storage forms, so there is nothing to
                // settle and announce, and for a waiting thread every command buffer with
                // submission handlers: those it takes are a committing thread's to run once the
                // lock is free.
                if submission.runs_handlers() {
                    debug_assert!(
                        passer == Passer::Committing,
                        "a waiting thread submitted a batch whose handlers it must not run"
                    );
                    handlers.push(held.buffer.hold());
                }
                let Held {
                    buffer,
                    binary_needs,
                    binary_signals,
                    waits: events,
                    ..
                } = held;
                drop(events);
                state.binary.submitted(&binary_needs, &binary_signals);
                submitted.batches += 1;
                if buffer.groups() {
                    submitted.alone += 1;
                }
                buffer.publish(submission);
                spent.push(buffer);
            }
            // Nothing submitted or changed: a rebind encodes encoders of it again
            // (`command_buffer/reencode.rs`), or the group held must go first. It goes back at
            // its place, which it keeps, for the thread that makes passes, which the caller wakes.
            Ok(Submitting::Later | Submitting::AfterGroup(_)) => {
                drop(busy);
                state.order.put_back(queue, sequence, held);
                return submitted;
            }
            // Unreachable: this pass submits each batch alone (`joins` false,
            // `command_buffer/group.rs`), and runs only while the group holds no batch, so
            // `admits` holds for it. Held as the thread's
            // own pass holds one all the same, and this pass ends there: the thread, which the
            // caller wakes, sends the group at once, as it does for any group one of whose
            // command buffers a thread waits for (`Group::keeps`).
            Ok(Submitting::Grouped) => {
                drop(busy);
                if state.group.schedule_held() {
                    debug_assert!(
                        passer == Passer::Committing,
                        "a waiting thread held a batch whose handlers it must not run"
                    );
                    handlers.push(held.buffer.hold());
                }
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

/// What [`submit_plain_ready`] submitted.
#[derive(Clone, Copy, Debug, Default)]
struct Plain {
    /// The batches.
    batches: usize,
    /// Of those, the batches of a `VkQueue` whose batches may wait in a group, which a committing
    /// thread submits alone where the group would go at once ([`plain`]) and every other pass
    /// leaves to the scheduler's thread: a waiting thread's submits none.
    alone: usize,
}

/// Who took a held batch's submission checks to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Taken {
    /// Left to the scheduler's thread ([`check_heads`]), which runs its checks in the order it
    /// took them.
    Left,
    /// A committing thread, which runs them before it submits the batch ([`claim_check`]).
    AtCommit,
}

/// Which thread other than the scheduler's makes a pass for itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Passer {
    /// A thread about to block for a command buffer ([`Scheduler::submit_before_waiting`]),
    /// which runs none of the program's code.
    Waiting,
    /// A thread that just committed ([`Scheduler::submit_at_commit`]), which runs the submission
    /// handlers of what it submits.
    Committing,
}

/// How a commit hands its batches to the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hand {
    /// Posted, and the thread woken where it sleeps ([`Inbox::post`]).
    Wake,
    /// Posted with no wake for the committing thread's own pass, which follows
    /// ([`Scheduler::submit_at_commit`]).
    Pass,
}

/// The Vulkan waits for `held`'s events' signal indices where a thread other than the
/// scheduler's, `passer`, may submit it itself ([`Scheduler::submit_before_waiting`],
/// [`Scheduler::submit_at_commit`]); `None` where it may not.
///
/// It is ready as [`submit_ready`] asks of a head: its binary semaphores' signals submitted, no
/// rebind encoding encoders of its event group again, no CPU decode queued into a twin it may
/// read, and every event value it waits for reached by a submitted signal or by the host. And
/// neither the thread nor the program has anything to do for it:
///
/// - no segments, so no host step the thread records, and none in its [`Left`] work;
/// - no submission check left to run, which runs the program's code on the thread that runs it
///   (a committing thread runs the checks of a head left itself first, where
///   [`claim_check`] takes them);
/// - for a waiting thread, no submission handler and no scheduled handler, so its transition to
///   `Scheduled` is plain and runs none of the program's code where it is published
///   (`CommandBufferShared::submits_without_the_program`); a committing thread runs the
///   submission handlers itself, and a transition with scheduled handlers goes to its queue's
///   context (`CommandBufferShared::submits_at_commit`);
/// - no switch of a fetched target's storage form riding on it, which the submitting thread
///   settles and announces (`texture/storage_form.rs`);
/// - and a `VkQueue` whose batches never wait in a group (`command_buffer/group.rs`), which is
///   the thread's to hold and to send.
fn plain_ready(
    held: &Held,
    binary: &BinarySignals<vk::Semaphore>,
    decodes: &BTreeMap<ResourceKey, usize>,
    untracked_decodes: usize,
    passer: Passer,
) -> Option<Vec<(vk::Semaphore, u64)>> {
    if held.checked && held.buffer.needs_check(held.index) {
        return None;
    }
    if !plain(held, binary, decodes, untracked_decodes, passer) {
        return None;
    }
    event_waits(held)
}

/// Whether nothing holds `held` back from `passer` but its submission checks and its event waits
/// ([`plain_ready`]'s other conditions).
fn plain(
    held: &Held,
    binary: &BinarySignals<vk::Semaphore>,
    decodes: &BTreeMap<ResourceKey, usize>,
    untracked_decodes: usize,
    passer: Passer,
) -> bool {
    if held.segments || !binary.ready(&held.binary_needs) || held.buffer.reencode_pending() {
        return false;
    }
    if untracked_decodes > 0 || (!decodes.is_empty() && reads_decoding(held, decodes)) {
        return false;
    }
    // A batch that may wait in the group is the scheduler thread's to hold, but for a committing
    // thread's where its `VkQueue` would see the group go at once (`goes_alone_at_commit`).
    match passer {
        Passer::Waiting => held.buffer.submits_without_the_program() && !held.buffer.groups(),
        Passer::Committing => {
            held.buffer.submits_at_commit()
                && (!held.buffer.groups() || held.buffer.goes_alone_at_commit())
        }
    }
}

/// The checks of the first line head in commit order whose checks a committing thread may run
/// itself, without the lock and right before it submits that head
/// ([`Scheduler::submit_at_commit`]), as Metal checks a submission as it submits it: where they are still to run and no other thread
/// took them, nothing else holds the head back from that thread ([`plain`]), every event value it
/// waits for was reached on the device or by the host, or is one its `VkQueue` waits for
/// ([`landing_waits`]), and it waits behind no earlier commit ([`is_behind`]: no check of a head
/// before the batch it waits behind is submitted, as in the thread's own pass, [`check_heads`]). A
/// head whose checks wait for a submitted GPU signal to land (a shared event's) is left to the
/// scheduler's thread, which waits for it, and so is one whose checks that thread took; a later
/// commit's head of another queue may be taken past either. Marked taken, so that thread leaves
/// them.
fn claim_check(state: &mut State) -> Option<Work> {
    let State {
        order,
        binary,
        decodes,
        untracked_decodes,
        ..
    } = state;
    let passer = Passer::Committing;
    let mut claimable = order.heads_where(|held| {
        held.checked
            && held.checks_taken.is_none()
            && held.buffer.needs_check(held.index)
            && plain(held, binary, decodes, *untracked_decodes, passer)
            && landing_waits(held).is_some_and(|waits| waits.is_empty())
    });
    claimable.sort_unstable();
    let mut memo = Behind::default();
    let sequence =
        (claimable.into_iter()).find(|&sequence| !is_behind(order, binary, &mut memo, sequence))?;
    let held = order.head_mut_at(sequence)?;
    held.checks_taken = Some(Taken::AtCommit);
    Some((held.buffer.hold(), held.index))
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
/// ([`Turns::run`]): the program sees each swap, and its announcement, by the time it
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

/// A batch for the thread to work on without its lock: its command buffer with the batch's index.
type Work = (Hold<CommandBufferShared>, usize);

/// Look at each line's head of a command buffer with a submission check that is otherwise
/// ready (its binary semaphores' signals submitted, its events' values reached, its host step
/// recorded, not `behind` an earlier commit) and waits for its checks: once every value it waits
/// for was reached on the device or by the host, or is one its `VkQueue` waits for (a plain
/// event's for a submitted or reserved signal, [`landing_waits`]), not one a shared event's
/// submitted GPU signal alone reaches (or the device is lost), it goes to `left` as a check,
/// unless its checks were taken already (to `left`, or by a committing thread, [`claim_check`]).
/// What it waits for goes to `awaits` before the events are read again, as
/// [`Scheduler::completions`] says. Whether a head still waits for a GPU signal to land.
///
/// Where the line's batches may go in one kernel submission (`command_buffer/group.rs`), the
/// batches right behind a head that goes (its checks in `left` or none due, otherwise ready, its
/// events' values reached on the device or by the host; not one whose checks a committing thread
/// runs, as those behind it would run on this thread meanwhile) are looked at too, at most
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
        let due =
            held.checked && held.checks_taken.is_none() && held.buffer.needs_check(held.index);
        // A committing thread runs its checks ([`claim_check`]), so no check behind it may start.
        let elsewhere =
            held.checks_taken == Some(Taken::AtCommit) && held.buffer.needs_check(held.index);
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
                held.checks_taken = Some(Taken::Left);
                left.push(Kind::Check, (held.buffer.hold(), held.index));
            }
            return true;
        }
        count_waits_held(held);
        if !held.checked || !otherwise_ready || !due {
            return !due && !elsewhere && goes(held);
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
        held.checks_taken = Some(Taken::Left);
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
            || held.buffer.stepping(held.index)
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
    memo: &mut Behind<Option<Arc<Footprint<ResourceKey>>>>,
    sequence: u64,
) -> bool {
    order.is_behind(
        sequence,
        memo,
        (
            &mut |held: &Held| binary.ready(&held.binary_needs) && events_reachable(held),
            &mut |held: &Held| held.buffer.footprint(held.index),
            &|earlier: &Option<Arc<Footprint<ResourceKey>>>,
              later: &Option<Arc<Footprint<ResourceKey>>>| {
                tracker::ordered(earlier.as_deref(), later.as_deref())
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

/// The waits of `held`, a line's head whose checks are due, that its checks wait for to land:
/// the signal indices of the submitted GPU signals that reach the values not yet reached of the
/// waits the batch cannot hand to its `VkQueue` (`EventShared::wait` answers none: a shared
/// event's, whose wait is held on the host, and one a signal behind a merge reaches), from
/// `EventShared::landing`. What the scheduler's thread may wait for on the GPU itself (the module
/// docs). `None` while such a value is reached by neither a submitted signal nor the host.
///
/// A wait the batch hands to its `VkQueue` (a plain event's, `MTLEvent`, for a signal submitted or
/// reserved before it) holds no check: Metal submits a command buffer whose `MTLEvent` wait only
/// a signal still running on the GPU meets, and so checks it, at once, where it holds one whose
/// `MTLSharedEvent` wait such a signal meets until the signal lands (an M3 Max under macOS 27.0.1:
/// the conformance case `event-wait-behind-a-running-signal`, the waiter's status and scheduled
/// handler read while the signaller still ran). Such a head is checked once otherwise ready, and
/// its waits chain on the `VkQueue` as an unchecked head's do ([`event_waits`]).
fn landing_waits(held: &Held) -> Option<Vec<(vk::Semaphore, u64)>> {
    let mut waits = Vec::new();
    for (event, value) in &held.waits {
        if event.wait(*value).is_some() {
            continue;
        }
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
