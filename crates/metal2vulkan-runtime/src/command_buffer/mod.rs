//! `CommandBuffer` (`MTLCommandBuffer`).
//!
//! A command buffer owns a command pool of its own: each encoder records into a `VkCommandBuffer`
//! from it when its commands are encoded ([`encoder`](super::encoder)). The pool is a spare of
//! the device's where it has one, which a command buffer that completed gave back
//! ([`spares`](super::spares)). Its event waits and signals split its encoders into batches
//! ([`batches`]); `commit` hands them to the device's scheduler ([`scheduler`](super::scheduler)),
//! which submits each once the events it waits for are reached ([`submit`]), and their
//! `VkQueue`'s waiter ([`completion`]) sees them complete. One encoder is open at a time; an
//! encoder dropped without `end_encoding` loses its commands, so the command buffer can no longer
//! be committed.
//!
//! Statuses follow Metal's: `NotEnqueued`, then `Committed` at `commit`, `Scheduled`, and
//! `Completed` or `Error`. The transitions after `Committed` run in the queue's serial context on
//! the device's handler threads ([`contexts`](super::contexts)), in the queue's order: to
//! `Scheduled` once its last batch is submitted (every wait of it satisfied), with its scheduled
//! handlers; to `Completed` once its work completed, with its completed handlers and the release of
//! what it retained. A transition that runs none of the program's code runs where it is seen
//! instead, where its context is free: to `Scheduled` on the scheduler's thread
//! (`CommandBufferShared::schedule`), to `Completed` on the queue waiter
//! (`CommandBufferShared::finish_here`). As on Metal, a queue's command buffer whose work completed
//! stays `Committed` while an earlier transition of its queue, a handler, still runs. Its
//! submission handlers (a runtime extension) run before its transition to `Scheduled`, on the
//! thread that took its last batch for its queue (submitted it, or held it for one kernel
//! submission with later batches: `group.rs`), which the transition waits for.
//!
//! Encoders write their commands' own data (tables, neutral buffers) into the command buffer's
//! upload arena ([`upload`](super::upload)), and the execution status words its commands share; the
//! command buffer decodes them when it completes ([`CommandBuffer::execution_report`]) and releases
//! the arena and what it retained.

pub(super) mod batches;
pub(super) mod checks;
pub(super) mod completion;
pub(super) mod fences;
mod first_submission;
pub(super) mod gate;
mod group;
pub(super) mod hang;
pub(super) mod kept_barriers;
pub(super) mod own;
mod protection;
pub(super) mod rebound_use;
pub(super) mod recovery;
mod reencode;
pub(super) mod rerun;
pub(super) mod revocation;
mod storage_form;
mod submit;

pub(super) use completion::Completion;
pub(super) use first_submission::{NotedTable, ReplayTable};
/// Whether a driver's facts merge a submission that signals nothing into the next, for the test
/// that holds every committed profile to the fact
/// (`tests::batches::only_drivers_measured_to_merge_their_submissions_group_their_batches`).
#[cfg(test)]
pub(crate) use group::merges_submissions;
pub(super) use group::{Group, GroupEnd, GROUP_BATCHES};
use rebound_use::{ReboundUse, Unlogged};
pub(super) use submit::{BatchLists, Submitting};

use super::apple_only::AppleOnly;
use super::compute::EncoderState;
use super::contexts::{self, Run};
use super::deferred::{Deferred, Prepared, StepInputs};
use super::degradation::Degradation;
use super::descriptor_arena::DescriptorArena;
use super::device::lock;
use super::device::relay::Relayed;
use super::dirty_pages::DirtyCommit;
use super::encoder::{Recorded, Segment};
use super::error::{
    internal, invalid, vulkan, CommandBufferError, DeadlockCause, EncoderCall, Error,
    FailedPipeline, FailedState, MeshRecordsOverflow, PipelineCreationCause, Problem,
    TessellationRecordsOverflow, TimedOut,
};
use super::event::{EventRef, EventShared, SharedEvent};
use super::execution::Execution;
use super::exit_guard::{self, Call};
use super::function_table::TableSnapshot;
use super::gpu_pools::{IndexPool, PoolKind, PoolLease};
use super::gpu_time::{self, KernelTimes, Life, Placement, Stamped, Times};
use super::heap::Heap;
use super::heap_sets::HeapUse;
use super::live_buffers::LiveTable;
use super::pipeline::budget::Urgency;
use super::pipeline::kept::StepNumbers;
use super::predication::{Event, Predication};
use super::program_objects;
use super::queue::Slot;
use super::rebind::{Holds, SubmittedHolds, Taken};
use super::render::EndInFlight;
use super::replay::Replay;
use super::residency::Residency;
use super::resource::{ObjectId, ResourceKey};
use super::scheduler::{BatchSync, StepWaits};
use super::shadow::ShadowCarry;
use super::spares::CommandPool;
use super::sparse::encoder::SparseUpdate;
use super::sparse::indirect;
use super::texture::storage_form::Riding;
use super::texture::Texture;
use super::tracker::{Footprint, Use};
use super::unique_identifier::ScheduledWork;
use super::upload::{Upload, Uploads};
use super::used_resources::{names_any, EncoderKind, EncoderRecord, Reached, ReadRecord};
use super::values::CommandBufferStatus;
use super::{Buffer, CommandQueue, Device};
use crate::release;
use ash::vk;
use batches::Batches;
use checks::{Checks, SubmissionCheck};
use completion::Submitted;
use first_submission::{Listing, Live};
use gate::{Gate, Hold};
use hang::{Hang, Held, Later};
use metal2vulkan::bind::{
    ExecutionStatus, IndexBoundsPass, ReplayFaults, ResidentRange, ResidentSet, RootTable,
    StatusReport,
};
use metal2vulkan::device::PipelineWaitPolicy;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::DispatchType;
use own::{
    completed_wait, running, scheduled_wait, Answer, Context, Marker, Running, Step, Target,
};
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// A scheduled or completed handler (`MTLCommandBufferHandler`): called once with the command buffer, on one of the
/// device's handler threads, in its queue's order. A panic inside it is caught. A submission
/// handler ([`CommandBuffer::add_submission_handler`]) is one too, called on the thread that
/// submits the command buffer's last batch.
///
/// - **macOS.** The handler threads run at the user-interactive quality of service, whatever the
///   class of the thread that committed, as Metal runs its handlers. A thread a handler starts
///   starts at the default class, as a new thread does (`<sys/qos.h>`).
/// - **Linux**, which Metal does not run on. Each handler thread asks the kernel to treat it as
///   fully busy, as the user-interactive class does on macOS: a utilization floor of 1024
///   (`sched_setattr(2)`), its policy and nice value as they were. A kernel built without the
///   clamps refuses it, and the thread runs as before. The thread also has
///   `SCHED_FLAG_RESET_ON_FORK`, so a thread a handler starts begins with no floor, and at
///   `SCHED_OTHER` and nice 0 where the handler thread runs at a real-time policy or a negative
///   nice value; a handler cannot clear the setting without `CAP_SYS_NICE`. The first such floor
///   since the host booted turns the kernel's clamp accounting on for every task until it
///   reboots.
pub type Handler = Box<dyn FnOnce(&CommandBuffer) + Send + 'static>;

/// A Metal command buffer (`MTLCommandBuffer`). Cheap to clone (a retain); `Send + Sync`.
#[derive(Clone)]
pub struct CommandBuffer(Arc<CommandBufferShared>);

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBuffer")
            .field("status", &self.status())
            .finish()
    }
}

/// How a commit runs ([`CommandBuffer::commit_now`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum CommitRun {
    /// Inside `commit` (or `commit_failed`).
    AtCall,
    /// After `commit` returned, its own work having waited for a render encoder's end
    /// (`render/queued_commits.rs`).
    Queued,
    /// Again after that work refused it (`State::refuse_commit`): nothing of the command buffer
    /// runs, and nothing that may fail is made for it (its predication tables, its root tables'
    /// pointing, its function-table snapshots, its argument samplers' listing), so it commits.
    Refused,
}

/// A command buffer as the device's commits that wait hold it (`render/queued_commits.rs`): a
/// trait object, whose `Send` and `Sync` its bounds give, made with no allocation. A program's
/// crate proves the device's `Send` and `Sync` within its own recursion limit (rustc's 128 by
/// default), which a path through the command buffer once more passes (`E0275` in the validation
/// crate's statics).
pub(crate) trait Waiting: Send + Sync {
    /// The command buffer.
    fn command_buffer(self: Arc<Self>) -> CommandBuffer;

    /// The command buffer's address, which names it ([`CommandBuffer::is`]).
    fn address(&self) -> usize;
}

impl Waiting for CommandBufferShared {
    fn command_buffer(self: Arc<Self>) -> CommandBuffer {
        CommandBuffer(self)
    }

    fn address(&self) -> usize {
        CommandBufferShared::address(self)
    }
}

/// A command buffer that a holder does not keep ([`CommandBuffer::downgrade`]).
pub(crate) struct WeakCommandBuffer(std::sync::Weak<CommandBufferShared>);

impl WeakCommandBuffer {
    /// The command buffer, where something still holds it.
    pub(crate) fn upgrade(&self) -> Option<CommandBuffer> {
        self.0.upgrade().map(CommandBuffer)
    }
}

pub(super) struct CommandBufferShared {
    device: Device,
    queue: CommandQueue,
    state: Mutex<State>,
    /// Signaled when the command buffer's status changes (`wait_until_scheduled` waits on it).
    changed: Condvar,
    /// What `wait_until_completed` waits for: the command buffer finished, and the runtime's
    /// threads let go of it.
    gate: Arc<Gate>,
    /// The `MTLCommandBufferError` code the driver relayed for one of its batches, with Metal's
    /// description, empty until one (`device/relay.rs`).
    relayed: Arc<Relayed>,
    /// The numbers its host steps give the variant instances they make, drawn at its commit in
    /// the program's call (`pipeline/kept.rs`).
    step_numbers: OnceLock<StepNumbers>,
    /// Until when its recordings wait for its draws' and dispatches' pipelines still being made:
    /// its commit plus the policy's `pipeline_commit_wait`, set at the commit under
    /// `PipelineWaitPolicy::Bounded`; none under `PipelineWaitPolicy::Exact`, where they wait
    /// until each is made (`render/pending.rs`, `pending_dispatch.rs`). A cold compile of
    /// MoltenVK's on a cold Metal compiler service was measured at 11.6 s inside a state creation
    /// a guest's command buffer encoding made (the guest's 10 s FIFO check ended its task);
    /// docs/FAST-PATHS.md holds the row.
    making_deadline: OnceLock<Instant>,
    /// The textures its encoders made for their own commands (`Device::new_planned_texture`),
    /// held from their making until it completes, whichever of its commands use them (its
    /// first batch takes them, `submit.rs`). The device's next submission moves each one's image
    /// from `UNDEFINED` to `GENERAL` (`texture/first_move.rs`). Apart from the state, so a texture
    /// is made under no lock of the command buffer's.
    own_textures: Mutex<Vec<Texture>>,
    /// The bytes a color fill of a color Metal packs no color in writes (`blit_fill.rs`): the
    /// last color fill's packed texel over the ones before, from zeros. Apart from the state,
    /// as `own_textures`.
    fill_slot: Mutex<[u8; 16]>,
    /// Rebinds since its commit, of buffers its replayable encoders named, whose encoders are
    /// not yet encoded again (`reencode.rs`): its first submission waits until none is. Raised
    /// under the device's lock of its buffers over host pages (`Rebinds::replace`), which its
    /// first submission reads it under; apart from the state, which that lock comes after.
    reencode: AtomicU64,
    /// Held while its encoders are encoded again, one rebind's turn at a time, and while a host
    /// step records: the recordings after its commit into its pool take turns, as Vulkan
    /// synchronizes a pool's command buffers' recording externally.
    reencoding: Mutex<()>,
    /// What its commands did that the device's logs of host pages hear of at its commit
    /// (`dirty_pages.rs`), and its copied host buffers' merges back (`copied.rs`): collected
    /// only while the device copied a shadow over logged pages in, imports logged pages in place,
    /// or keeps a copied host buffer. Apart from the state, which no encoder holds while it
    /// records.
    pub(super) dirty: Mutex<DirtyCommit>,
    /// A command's uses went unnoted, nothing collecting them then (no copied host buffer, and
    /// no log of host pages, lived): the merge back of copied host buffers takes every copy as
    /// written by it (`copied.rs`).
    pub(super) unnoted: AtomicBool,
    /// The `VkCommandBuffer`s of its pool its recordings ended ([`CommandBuffer::end_raw`]):
    /// where they are every one the pool handed out and it completed, it gives the pool back
    /// with none recording (`spares.rs`).
    raws_ended: AtomicU64,
    /// Its kernel times, apart from the state, which the scheduler holds while it records and
    /// submits a batch: a read waits for neither (`gpu_time.rs`).
    kernel: KernelTimes,
    /// Its status as `status` answers it, apart from the state for the same reason: each
    /// transition of the state's status stores it here under the state's lock
    /// ([`CommandBufferShared::answer`]), its raw value with release ordering, and a read loads
    /// it with acquire ordering and takes no lock. Metal's reads wait for no submission.
    status: AtomicU8,
    /// Its error as `error` answers it: set once, by the completion that fails it, before that
    /// stores the `Error` status, so a read that loads `Error` finds it. A commit takes nothing
    /// here: no lock is taken first at a commit (on Apple's hosts a mutex allocates at its first
    /// lock).
    error: OnceLock<CommandBufferError>,
    /// Whether a batch of it is held in the scheduler's group, and whether a thread waits for its
    /// completion: a group kept past a pass goes at once once one does (`group.rs`).
    group_hold: AtomicU8,
}

/// The word [`CommandBufferShared::status`] holds for `status`: its raw value.
fn status_word(status: CommandBufferStatus) -> u8 {
    status.raw() as u8
}

/// The status whose word ([`status_word`]) `word` is: every word stored is one of a status, the
/// last of them `Error`'s.
fn status_of(word: u8) -> CommandBufferStatus {
    match word {
        0 => CommandBufferStatus::NotEnqueued,
        1 => CommandBufferStatus::Enqueued,
        2 => CommandBufferStatus::Committed,
        3 => CommandBufferStatus::Scheduled,
        4 => CommandBufferStatus::Completed,
        _ => CommandBufferStatus::Error,
    }
}

/// How [`CommandBuffer::commit_failed`] fails a command buffer: a runtime extension, no Metal
/// type.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitFailure {
    /// The `MTLCommandBufferError` code it fails with: 2 for a GPU hang (`Timeout`), 8 for out
    /// of memory, 1 or 16 for a protection fault, among others. 0 for none: the encoders it names
    /// run nothing, and it completes with status `Completed` and no error, as Metal completes a
    /// command buffer of a queue whose GPU hang is in flight.
    pub code: u64,
    /// The encoders that run nothing, each by its place among those the command buffer ended,
    /// in the order they ended, from 0 (the index of [`CommandBuffer::ended_encoders`]); the
    /// others run.
    pub encoders: FailedEncoders,
    /// Whether its event signals (`encode_signal_event`) fire, as a GPU hang's do; none fires
    /// where false, as for a protection fault.
    pub signals: bool,
}

impl CommitFailure {
    /// A failure of `code` in which no encoder runs and the event signals fire.
    pub fn new(code: u64) -> CommitFailure {
        CommitFailure {
            code,
            encoders: FailedEncoders::From(0),
            signals: true,
        }
    }

    /// Why `commit_failed` refuses the failure, the command buffer having ended `ended`
    /// encoders: a set that names an encoder past them.
    pub(super) fn problem(&self, ended: usize) -> Option<Problem> {
        let FailedEncoders::Set(set) = &self.encoders else {
            return None;
        };
        let past = set.range(ended..).next()?;
        let saturated = |count: usize| u32::try_from(count).unwrap_or(u32::MAX);
        Some(Problem::IndexOutOfRange {
            index: saturated(*past),
            limit: saturated(ended),
        })
    }
}

/// How a committing command buffer fails: as `commit_failed` asks, from an encoder Metal fails
/// it from (`State::fails_from`), or both.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Failure {
    /// The encoders that run nothing.
    pub(super) encoders: FailedEncoders,
    /// The event signals that fire.
    pub(super) signals: FailedSignals,
    /// The error it completes with, if any.
    pub(super) error: Option<CommandBufferError>,
}

/// Which event signals of a failing command buffer fire.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum FailedSignals {
    /// Every one (`CommitFailure::signals`).
    Every,
    /// None.
    Dropped,
    /// Those encoded before ended encoder `n`: Metal fails a command buffer from a render pass
    /// without attachments past 16384 a side, and signals what was encoded before the pass,
    /// not after it.
    From(usize),
}

impl Failure {
    /// The failure of a command buffer committed on a queue Metal revoked (`revocation.rs`): no
    /// encoder runs, no event signal fires, and it completes with
    /// `CommandBufferError::AccessRevoked`.
    pub(super) fn revoked() -> Failure {
        Failure {
            encoders: FailedEncoders::From(0),
            signals: FailedSignals::Dropped,
            error: Some(CommandBufferError::AccessRevoked),
        }
    }

    /// Whether ended encoder `encoder` runs under `failure`: every one does but those it fails.
    pub(super) fn runs(failure: Option<&Failure>, encoder: usize) -> bool {
        failure.is_none_or(|failure| !failure.encoders.fails(encoder))
    }

    /// The failure of a command buffer of `ended` ended encoders that `commit_failed` fails as
    /// `injected` asks (none for `commit`), and that Metal fails from ended encoder `from` with
    /// its error: the encoders either fails, and `injected`'s code where it is given, else
    /// `from`'s error. `None` where neither fails it.
    pub(super) fn of(
        injected: Option<CommitFailure>,
        from: Option<(usize, CommandBufferError)>,
        ended: usize,
    ) -> Option<Failure> {
        let injected_signals = |signals: bool| match signals {
            true => FailedSignals::Every,
            false => FailedSignals::Dropped,
        };
        // A code of 0 is no error: the command buffer completes without one.
        let injected_error = |code: u64| (code != 0).then_some(CommandBufferError::Injected(code));
        Some(match (injected, from) {
            (None, None) => return None,
            (Some(failure), None) => Failure {
                encoders: failure.encoders,
                signals: injected_signals(failure.signals),
                error: injected_error(failure.code),
            },
            (None, Some((from, error))) => Failure {
                encoders: FailedEncoders::From(from),
                signals: from_signals(from, error),
                error: Some(error),
            },
            (Some(failure), Some((from, error))) => Failure {
                encoders: match failure.encoders {
                    FailedEncoders::From(first) => FailedEncoders::From(first.min(from)),
                    FailedEncoders::Set(mut set) => {
                        set.extend(from..ended);
                        FailedEncoders::Set(set)
                    }
                },
                signals: match failure.signals {
                    true => from_signals(from, error),
                    false => FailedSignals::Dropped,
                },
                error: injected_error(failure.code).or(Some(error)),
            },
        })
    }

    /// `failure` (`Failure::of`'s) with the encoders Metal's GPU faults, `faulted`, each with
    /// its error, of a command buffer of `ended` ended encoders: from the first faulted encoder on
    /// they run nothing too, as Metal's GPU's recovery loses them on a GPU that ran just before
    /// (`hang.rs`, the fault's hold), and, where nothing else fails it, the command buffer
    /// completes with that encoder's error (`CommandBufferError::PageFault`, or
    /// `GpuRecoveryVictim`) and every event signal fires, as Metal's does. Where something else
    /// fails it too, its error and its signals stand (Metal's error there is not measured).
    pub(super) fn with_faults(
        failure: Option<Failure>,
        faulted: &BTreeMap<usize, CommandBufferError>,
        ended: usize,
    ) -> Option<Failure> {
        let Some((&first, &faulted_error)) = faulted.iter().next() else {
            return failure;
        };
        Some(match failure {
            None => Failure {
                encoders: FailedEncoders::From(first),
                signals: FailedSignals::Every,
                error: Some(faulted_error),
            },
            Some(failure) => Failure {
                // Under a failure of no error, a faulted encoder it lets run faults it.
                error: failure
                    .error
                    .or_else(|| (!failure.encoders.fails(first)).then_some(faulted_error)),
                encoders: match failure.encoders {
                    FailedEncoders::From(from) => FailedEncoders::From(from.min(first)),
                    FailedEncoders::Set(mut set) => {
                        set.extend(first..ended);
                        FailedEncoders::Set(set)
                    }
                },
                ..failure
            },
        })
    }
}

/// The event signals that fire of a command buffer Metal fails from ended encoder `from` on with
/// `error`: every one at a GPU hang (`hang.rs`), those encoded before the encoder otherwise.
fn from_signals(from: usize, error: CommandBufferError) -> FailedSignals {
    match error {
        CommandBufferError::GpuHang => FailedSignals::Every,
        _ => FailedSignals::From(from),
    }
}

/// The encoders a [`CommitFailure`] fails, each by its place among those the command buffer
/// ended, from 0: a runtime extension, no Metal type.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailedEncoders {
    /// This encoder and every one after it, as a GPU hang fails the hung encoder's: the
    /// encoders before it run. One past the ended encoders fails none.
    From(usize),
    /// These encoders alone: every other encoder runs, before and after them, as Metal runs
    /// every submission but the one that failed (a submission that uses a resource whose
    /// memory was taken away, measured by a consumer on an M3 Max, macOS 26.7). Each names an
    /// ended encoder; an empty set fails none.
    Set(BTreeSet<usize>),
}

impl FailedEncoders {
    /// Whether encoder `encoder` runs nothing.
    pub(super) fn fails(&self, encoder: usize) -> bool {
        match self {
            FailedEncoders::From(first) => encoder >= *first,
            FailedEncoders::Set(set) => set.contains(&encoder),
        }
    }
}

/// A command buffer's mutable state.
pub(super) struct State {
    status: CommandBufferStatus,
    error: Option<CommandBufferError>,
    /// An encoder is open.
    pub(super) encoder_open: bool,
    /// An encoder was dropped without `end_encoding`.
    pub(super) abandoned: bool,
    /// A render encoder's end, which its call left to the device's lowering thread
    /// (`render/deferral.rs`): the encoder stays open until it is lowered, and what follows the
    /// end in the command buffer lowers it first ([`CommandBuffer::settle_end`]).
    ending: Option<EndInFlight>,
    /// The device's lowering thread lowered the end it holds (`ending`).
    end_lowered: bool,
    /// The commits' own work fails this many more times (a device test's injection of an out of
    /// memory, `tests::deferred_calls`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    failures_injected: u32,
    /// The error the making of the next draw's pipeline its recording waits for answers, in place
    /// of the making's own (a device test's injection, `tests::deferred_calls`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    making_failure_injected: Option<Error>,
    /// `commit` returned with its own work left to run once a render encoder's end is lowered
    /// (`render/queued_commits.rs`), which has not run: to the program, the command buffer is
    /// committed.
    queued_commit: bool,
    /// A render pass of it has a draw call over a `Mutable` rasterization-rate map: its commit
    /// lets a later pass of no draw call over such a map run (`rate_mapped_pass.rs`).
    pub(super) called_over_mutable_map: bool,
    /// `commit` is recording the encoders' segments (`encoder.rs`), outside this lock: nothing
    /// else may encode or commit meanwhile.
    committing: bool,
    /// The pool every `VkCommandBuffer` of this command buffer comes from, with those made from
    /// it; `None` until the first, and once given back to the device's spares.
    pool: Option<CommandPool>,
    /// The pool of a predicated loop's check while it records again the work its laying left
    /// apart (`rerun.rs`): every `VkCommandBuffer` the command buffer hands out meanwhile comes
    /// from it, and goes back to it for the next run again.
    rerun_pool: Option<Arc<Mutex<CommandPool>>>,
    /// The ended encoders' recordings, in encoding order.
    recorded: Vec<Recorded>,
    /// The last ended encoder's last `VkCommandBuffer`, still recording: its batch's end goes
    /// into it as the batch ends ([`CommandBufferShared::close_tail`]), or it ends as it is where
    /// a later encoder joins its batch.
    pub(super) tail: Option<vk::CommandBuffer>,
    /// The batches whose end went into their last `VkCommandBuffer` (`submit.rs`): each batch's
    /// index, that command buffer, and whether it wrote the command buffer's end timestamp.
    folds: Vec<Fold>,
    /// The first encoder's first `VkCommandBuffer`, where its first commands write the start into
    /// the slot's third query (`gpu_time.rs`): the first batch writes the start there alone where
    /// it plans nothing before that encoder and the command buffer reaches the queue as recorded
    /// (`submit.rs`).
    pub(super) prefix: Option<vk::CommandBuffer>,
    /// The encoders split at the event and binary semaphore waits and signals.
    batches: Batches<Arc<EventShared>, vk::Semaphore>,
    /// Its scheduled signals, in the order encoded
    /// ([`CommandBuffer::encode_signal_event_scheduled`]).
    scheduled_signals: Vec<(Arc<EventShared>, u64)>,
    scheduled: VecDeque<Handler>,
    /// Its submission handlers ([`CommandBuffer::add_submission_handler`]): run on the thread
    /// that takes its last batch for its queue (submits it, or holds it in a group), before its
    /// transition to `Scheduled`.
    submission: VecDeque<Handler>,
    /// The thread that took its last batch has its submission handlers to run, or the
    /// switch of storage forms riding on it to end (`texture/storage_form.rs`): its transition to
    /// `Scheduled` waits until it did.
    submitting: bool,
    /// Its scheduled handlers ran: `wait_until_scheduled` returns, as Metal's does only once
    /// they ran.
    scheduled_ran: bool,
    /// What its encoders did that Metal's identifier allocations follow as it is scheduled
    /// (`unique_identifier.rs`).
    scheduled_work: ScheduledWork,
    /// The threads sleeping on its `changed` condition variable: a change wakes it only where
    /// one does ([`CommandBufferShared::wake_parked`]).
    parked: usize,
    /// Why a batch could not be submitted: the command buffer finishes with it, and its
    /// scheduled handlers never run. Or the failure `commit_failed` chose, with which it
    /// finishes once its batches completed, its scheduled handlers run.
    failure: Option<CommandBufferError>,
    completed: VecDeque<Handler>,
    /// The handlers of its executions' records as the runtime reads them
    /// (`executed_commands.rs`).
    pub(crate) executed_handlers: Vec<crate::executed_commands::ExecutedCommandsHandler>,
    /// What the ended encoders' commands use, held until the command buffer completed and its
    /// completed handlers ran. A drop may run the caller's code (a no-copy buffer's deallocator),
    /// which may use this command buffer, so each is released outside this state's lock.
    pub(super) retained: Vec<Box<dyn Any + Send + Sync>>,
    /// What the ended encoders' dispatches and draws executed (`execution.rs`), kept after the
    /// command buffer completed.
    pub(super) executions: Vec<Execution>,
    /// The commands' own data, released when the command buffer completes.
    pub(super) uploads: Uploads,
    /// The device's heap sets its commands bound (`heap_sets.rs`), released with the uploads.
    pub(super) heaps: HeapUse,
    /// Its commands' function-table snapshots, written again as it commits.
    table_snapshots: Vec<TableSnapshot>,
    /// A command left a live sampler state out of its argument samplers
    /// (`State::unlisted_argument_sampler`): counted once
    /// (`WorkCounts::unlisted_argument_sampler_buffers`).
    unlisted_argument_samplers: bool,
    /// The first ended encoder Metal fails the command buffer from, with the error (a render
    /// pass without attachments past 16384 a side, `CommandBufferError::RenderTargetSize`, or
    /// at a default sample count Metal fails it at, `DefaultRasterSampleCount`): as it commits,
    /// that encoder and every one after it run nothing, and it completes with that error.
    fails_from: Option<(usize, CommandBufferError)>,
    /// The hold of the encoder Metal's GPU hangs at, where one does (`hang.rs`).
    hang: Option<Hang>,
    /// Its transition to `Scheduled` around a GPU hang of its queue, from its commit on
    /// (`hang.rs`): its own hang's, which the hang's thread makes, or one committed during a
    /// hang, which the hang's thread makes early unless its last batch did.
    hang_scheduling: Option<hang::Scheduling>,
    /// It was committed with an event wait while a hang of its queue was in flight: the verdict
    /// its first batch's submission applies (`hang.rs`, `checks.rs`).
    behind_hang: Option<hang::Verdict>,
    /// The ended encoders Metal's GPU faults, each with the error it fails the command buffer
    /// with (an execution of commands that inherit a pipeline state made without
    /// `supportIndirectCommandBuffers`, `CommandBufferError::PageFault`; an indexed or indirect
    /// draw of primitive type 9): as it commits, they run nothing, and it completes with the first
    /// one's error.
    faulted: BTreeMap<usize, CommandBufferError>,
    /// The shadowed textures its ended encoders' passes took as attachments
    /// (`render/shadows.rs`), each once.
    shadow_attachments: Vec<ResourceKey>,
    /// The shadowed textures its ended encoders copied in first (`shadow.rs`), each once.
    shadows_copied_in: Vec<ResourceKey>,
    /// The shadowed textures its ended encoders left holding their truth, which its later
    /// encoders take instead of copying it in again (`shadow::ShadowCarry`): handed to each
    /// encoder as it opens and taken back as it ends, so it is `None` while an encoder is open.
    /// An event or semaphore wait drops it, after which Metal shows the work encoded next
    /// another writer's bytes; Metal refuses such a wait while an encoder is open, so one never
    /// finds it lent out.
    shadow_carry: Option<ShadowCarry>,
    /// The pools of its commands' set 0 (`descriptor_arena.rs`), given back or released with
    /// the uploads.
    pub(super) descriptors: DescriptorArena,
    /// The execution status words the commands share, once a command reaches them.
    status_words: Option<Upload>,
    /// A direct mesh draw's object grid lay past what runs (`render/mesh_arguments.rs`): the
    /// command buffer fails as for `ExecutionStatus`'s `MESH_GRID_PAST_LIMITS`.
    mesh_grid_past: bool,
    /// The index pool of the commands' index bounds and rewrite passes (`index_bounds.rs`,
    /// `index_rewrite.rs`), once a draw needs it: its counter in the arena, starting at 0, and
    /// its lease of its queue's block, which its indices lie in (`gpu_pools.rs`).
    index_pool: Option<(Upload, PoolLease)>,
    /// The records pool of the commands' scanned emulated mesh draws (`render/mesh.rs`), once a
    /// draw needs it: its lease of its queue's block (`gpu_pools.rs`); the scans' cursor is a
    /// status word.
    mesh_pool: Option<PoolLease>,
    /// The emulated mesh draws numbered so far, each by its order.
    mesh_draws: u32,
    /// The resident-set misses the segments' recordings found on the host, which the report adds to
    /// the words' count.
    resident_misses: u32,
    /// Why some of the segments' commands ran nothing, which the report states.
    faults: ReplayFaults,
    /// The dispatches of the segments placed so far, each among the executions.
    placed: usize,
    /// Why a host step failed: the batch it recorded for is not submitted, and the command buffer
    /// finishes with it.
    step_failure: Option<CommandBufferError>,
    /// The refusal of a host step that failed so (`CommandBufferError::Refused`).
    refusal: Option<Error>,
    /// The first dispatch's pipeline whose making failed, which a host step met
    /// (`CommandBufferError::PipelineCreation`).
    failed_pipeline: Option<FailedPipeline>,
    /// The first draw's pipeline whose making failed, which its recording dropped
    /// (`render/pending.rs`).
    dropped_draw: Option<FailedPipeline>,
    /// Metal's description of the failure the driver relayed, where the command buffer failed
    /// with it (`CommandBuffer::error_description`).
    relayed_description: Option<String>,
    /// The decoded status words, once the command buffer completed.
    report: Option<StatusReport>,
    /// Its GPU and kernel times (`gpu_time.rs`).
    times: Times,
    /// The slot of its queue it holds until it completes, or is dropped uncommitted
    /// (`queue.rs`).
    slot: Option<Slot>,
    /// The heaps its ended encoders' commands used, which its commit marks used
    /// (`Heap::set_purgeable_state`).
    used_heaps: Vec<Heap>,
    /// Its ended encoders' holds on the buffers over host pages their commands used
    /// (`rebind.rs`): let go at its commit, or, where a host step records commands after it, as
    /// it completes.
    pub(super) holds: Vec<Holds>,
    /// What its ended encoders made resident, every encoder's joined (`residency.rs`): the
    /// residency the work its commit and host steps record resolves and binds against. Released
    /// with what it retained, as it completes.
    residency: Option<Residency>,
    /// Its ended encoders, in the order they ended, each with what its commands use
    /// (`used_resources.rs`): the keys its commands' uses and bindings named, its placed
    /// segments' among them, and what it made resident. Their uses are let go as it completes.
    ended: Vec<EncoderRecord>,
    /// Its root tables, until its first submission points them at its resident set as the
    /// live buffers stand then (`first_submission.rs`); the commit points them first.
    root_tables: Vec<NotedTable>,
    /// The ranges of its root tables' own resident sets, each once.
    root_ranges: HashSet<ResidentRange>,
    /// The resident set its root tables search, once made; released as it completes.
    resident: Option<Resident>,
    /// The live buffers' tables its host steps resolved against (`CommandBuffer::residency`),
    /// each once ([`State::keep_stepped_table`]); released as it completes.
    stepped_tables: Vec<Arc<LiveTable>>,
    /// What its commit listed among the live buffers (`LiveBuffers::reach`), and the list's
    /// generation then: listed again at its first submission where the list changed since.
    listing: Option<Listing>,
    /// Its first batch was submitted: from then on it holds what its commands took and the
    /// live buffers' table its resident set was made of until it completes.
    submitted: bool,
    /// The batches before this one lie in event groups judged as they were submitted
    /// (`State::first_submission`, `State::later_group`): a rebind encodes again only encoders
    /// of the groups from here (`reencode.rs`).
    judged_to: usize,
    /// What `set_protection_options` left (`protection.rs`).
    protection: protection::Protection,
    /// Metal's submissions (`checks::metal_submissions`, by their ended encoders) its first
    /// submission found to name a buffer rebound since, or to hold a detached one, each with
    /// its failure: each fails as its own batch is submitted, after its check
    /// (`first_submission.rs`).
    failing: Vec<(Range<usize>, CommandBufferError)>,
    /// Where its first submission, or a later batch's, found the first use rebound, for the
    /// notice (`rebound_use.rs`).
    rebound_use: Option<ReboundUse>,
    /// The resident root tables of its work on the GPU alone recorded before its first
    /// submission, which that submission points at the set as it lies then
    /// (`first_submission.rs`).
    replay_tables: Vec<ReplayTable>,
    /// A command of it may write memory no use it noted names: a root table (a function that
    /// reaches buffers through addresses), a root table of work on the GPU alone, or work its
    /// host steps resolve from records (`Deferred::reaches_unnamed`). The merge back of copied
    /// host buffers then takes every copy as written by it (`copied.rs`).
    pub(super) reaches_unnamed: bool,
    /// The ended encoders whose submissions failed so: no part of one goes to a queue with work
    /// (`submit.rs` holds it to that).
    unrunnable: BTreeSet<usize>,
    /// What its batches' commands took of buffers over host pages, held from the submission of
    /// their event group's first batch (`rebind::SubmittedHolds`, `State::event_group`), each with
    /// the batch whose submission takes it ([`State::holds_until`]): let go as that batch's work
    /// completed, and, for its last batch's, as its completion is processed, before its completed
    /// handlers run ([`State::keep_holds`], `CommandBufferShared::complete`). A group of its never
    /// submitted keeps its holds until then too, since nothing of it will run.
    submitted_holds: Vec<(usize, SubmittedHolds)>,
    /// The encoders its commit's failure failed (`commit_failed`'s, Metal's own from a render
    /// pass, or a faulted encoder's: [`State::fail`]): a submission none of whose encoders
    /// runs so is spared its rebound verdict (`first_submission.rs`).
    failed_at_commit: Option<FailedEncoders>,
    /// The program's submission check (`checks.rs`), released as it completes.
    check: Option<SubmissionCheck>,
    /// Metal's submissions of a committed command buffer with a check, and how far they were
    /// checked.
    checks: Option<Checks>,
    /// The fetched render targets' switch to their storage form riding on its commit
    /// (`texture/storage_form.rs`), until the scheduler settled it and a program thread finished
    /// it.
    storage_form: Option<Arc<Riding>>,
    /// Its units of command predication (`predication.rs`).
    pub(super) predication: Predication,
    /// The predication of an encoder a rebind encodes again now, which encodes again exactly
    /// (`Predication::replaying`, `reencode.rs`): its records and dispatches step and gate there.
    pub(super) replaying_predication: Option<Predication>,
    /// An encoder of it used command predication: its commit lets go of the logs of its compute
    /// encoders their unit of predication reaches past, or that laid a loop or left work out
    /// (`State::keep_replays`).
    pub(super) predicated: bool,
    /// The device's trail mark as it completed (`trails.rs`): its waiters return once no trail
    /// numbered below is open. 0 before, which waits for none.
    trail_mark: u64,
}

/// A batch whose end went into its last `VkCommandBuffer` ([`State::folds`]).
struct Fold {
    /// The batch's index.
    batch: usize,
    /// The command buffer.
    raw: vk::CommandBuffer,
    /// It wrote the command buffer's end timestamp.
    stamped: bool,
}

/// The resident set a command buffer's root tables search (`live_buffers.rs`): the device's live
/// buffers at its commit, and what its encoders bound and made resident beyond them (`merged`,
/// uploaded to its arena): `count` entries at `address` (`None` for none).
struct Resident {
    live: Live,
    merged: Option<ResidentSet>,
    address: Option<u64>,
    count: u64,
}

impl State {
    /// Whether it was committed, its commit's own work waiting or not
    /// (`render/queued_commits.rs`): a handler added now would not run
    /// (`Problem::HandlerAfterCommit`).
    pub(crate) fn enqueued(&self) -> bool {
        self.status != CommandBufferStatus::NotEnqueued || self.queued_commit
    }

    /// An encoder of `kind` ended, its commands' uses and bindings having named `keys`, having
    /// made `residency` resident (a compute or render encoder's): it joins the ended encoders,
    /// with the event waits and signals encoded before it, and what it made resident joins the
    /// command buffer's residency (`Residency::join_encoder`). Its place among the ended
    /// encoders, which each of its recordings' parts carries.
    pub(super) fn encoder_ended(
        &mut self,
        kind: EncoderKind,
        keys: BTreeSet<ResourceKey>,
        residency: Option<Residency>,
        taken: Taken,
    ) -> usize {
        if let Some(ended) = &residency {
            self.residency
                .get_or_insert_with(Residency::default)
                .join_encoder(ended, kind == EncoderKind::Compute);
        }
        let (event_waits_before, event_signals_before) = self.batches.events();
        self.ended.push(EncoderRecord {
            kind,
            event_waits_before,
            event_signals_before,
            keys,
            residency,
            taken,
            replay: None,
            executions: 0..0,
            held: Vec::new(),
            unlogged: None,
            refused: None,
        });
        self.ended.len() - 1
    }

    /// Note `table`, the resident root table of work recorded on the GPU alone: its first
    /// submission points it at the resident set as it lies then (`first_submission.rs`). One
    /// recorded after the first submission searches the set as it lay as it was recorded, which
    /// the submitted command buffer holds.
    pub(super) fn replay_table(&mut self, table: ReplayTable) {
        self.reaches_unnamed = true;
        if !self.submitted {
            self.replay_tables.push(table);
        }
    }

    /// Ended encoder `encoder` kept no log a rebind encodes it again from (`replay.rs`), as
    /// `why` says, for the rebound notice (`rebound_use.rs`).
    pub(super) fn encoder_unlogged(&mut self, encoder: usize, why: Unlogged) {
        if let Some(record) = self.ended.get_mut(encoder) {
            record.unlogged = Some(why);
        }
    }

    /// Ended encoder `encoder` holds the images `held` in their attachment layout
    /// (`texture/held_depth.rs`).
    pub(super) fn encoder_holds(&mut self, encoder: usize, held: Vec<ResourceKey>) {
        if let Some(record) = self.ended.get_mut(encoder) {
            record.held = held;
        }
    }

    /// Keep `replay`, the log of ended encoder `encoder`, whose executions take `executions`
    /// among the command buffer's. One of no such encoder comes back, for the caller to let go
    /// of without this lock (what it holds may be the last hold of a no-copy buffer).
    pub(super) fn keep_replay(
        &mut self,
        encoder: usize,
        replay: Option<Replay>,
        executions: Range<usize>,
    ) -> Option<Replay> {
        let replay = replay?;
        let Some(record) = self.ended.get_mut(encoder) else {
            return Some(replay);
        };
        record.replay = Some(Arc::new(replay));
        record.executions = executions;
        None
    }

    /// Its ended encoders `encoders` (all of them where `None`), each as `used_resources.rs`
    /// lists it: its kind, its event counts, every key its commands use and reach, and what it
    /// made resident. The keys are its own commands' and bindings', its placed segments', and
    /// those of each segment recorded and not yet placed (the commit's recording, or the host
    /// step's that replaced it): what the commands of an indirect command buffer it executes
    /// reach, from the commit on.
    pub(super) fn read_ended(&self, encoders: Option<Range<usize>>) -> Vec<ReadRecord> {
        let range = encoders.unwrap_or(0..self.ended.len());
        let Some(records) = self.ended.get(range.clone()) else {
            return Vec::new();
        };
        let mut read: Vec<ReadRecord> = records
            .iter()
            .map(|record| ReadRecord {
                kind: record.kind,
                event_waits_before: record.event_waits_before,
                event_signals_before: record.event_signals_before,
                keys: record.keys.clone(),
                residency: record.residency.clone(),
            })
            .collect();
        let unplaced = self
            .recorded
            .iter()
            .filter(|part| range.contains(&part.encoder))
            .flat_map(|part| {
                let keys = part
                    .segments
                    .iter()
                    .filter_map(|segment| segment.recorded.as_ref())
                    .flat_map(|done| done.keys.iter().copied());
                keys.map(move |key| (part.encoder - range.start, key))
            });
        for (at, key) in unplaced {
            if let Some(record) = read.get_mut(at) {
                record.keys.insert(key);
            }
        }
        read
    }

    /// Whether what its ended encoders `encoders` (all of them where `None`) use, as
    /// [`read_ended`](Self::read_ended) reads it, has an owner among `objects`, as
    /// `used_resources.rs` would list it: read in place, under this lock, with no handle cloned
    /// or held (`used_resources::names_any`). False for a range past its ended encoders, as
    /// `read_ended` reads none there.
    pub(super) fn uses(
        &self,
        shared: &super::device::DeviceShared,
        encoders: Option<Range<usize>>,
        objects: &[ObjectId],
    ) -> bool {
        let range = encoders.unwrap_or(0..self.ended.len());
        let Some(records) = self.ended.get(range.clone()) else {
            return false;
        };
        let ended = records
            .iter()
            .any(|record| names_any(shared, &record.keys, record.residency.as_ref(), objects));
        ended
            || (self.recorded.iter())
                .filter(|part| range.contains(&part.encoder))
                .any(|part| {
                    let keys = (part.segments.iter())
                        .filter_map(|segment| segment.recorded.as_ref())
                        .flat_map(|done| done.keys.iter());
                    names_any(shared, keys, None, objects)
                })
    }

    /// What the commands of its ended encoders that `runs` (by their place among them) use and a
    /// command reaches unnamed only once used, gathered weakly under this lock ([`Reached`]): what
    /// their keys name, as [`read_ended`](Self::read_ended) reads them (each segment's recorded
    /// and not yet placed with its encoder's), and what they made resident.
    fn reached(
        &self,
        shared: &super::device::DeviceShared,
        runs: impl Fn(usize) -> bool,
    ) -> Reached {
        let mut reached = Reached::default();
        for (_, record) in (self.ended.iter().enumerate()).filter(|(at, _)| runs(*at)) {
            reached.gather(shared, &record.keys, record.residency.as_ref());
        }
        for part in self.recorded.iter().filter(|part| runs(part.encoder)) {
            let segments = part.segments.iter();
            for done in segments.filter_map(|segment| segment.recorded.as_ref()) {
                reached.gather(shared, &done.keys, None);
            }
        }
        reached
    }

    /// Hold `table`, the live buffers' table a host step resolves against, each once: so the
    /// step's own hold, let go of on the scheduler's thread, is never the last of a released
    /// buffer's pages, whose release runs the program's code (`exit_guard::program_release`).
    /// From the first submission on the commands the step records may reach the table's
    /// buffers, so the command buffer holds it until it completes, as it holds its first
    /// submission's. One a step took before then (a step of commands the GPU may write never
    /// runs before it, its batch never the first, `Batches::host_step`; another kind's may) goes
    /// at the first submission, at once, as the set is made of the list as it stands then
    /// (`first_submission.rs`). `commit` holds none.
    fn keep_stepped_table(&mut self, table: &Arc<LiveTable>) {
        let kept = self
            .stepped_tables
            .iter()
            .any(|kept| Arc::ptr_eq(kept, table));
        if self.status != CommandBufferStatus::NotEnqueued && !kept {
            self.stepped_tables.push(Arc::clone(table));
        }
    }

    /// Let go of what its ended encoders' commands use, as it completes: their keys, and what
    /// they made resident, for the caller to release without this lock.
    fn let_go_of_uses(&mut self) -> Vec<Residency> {
        self.ended
            .iter_mut()
            .filter_map(|record| {
                record.keys.clear();
                record.residency.take()
            })
            .collect()
    }

    /// Note a root table its commands read, uploaded at `table` (`upload`), whose own resident
    /// set is `ranges`: the commit points it at the command buffer's resident set. One a host
    /// step uploads after the commit is pointed now, at that set where it holds `ranges`, else
    /// at a set of both made for it.
    pub(super) fn root_table(
        &mut self,
        device: &Device,
        noted: NotedTable,
        ranges: Vec<ResidentRange>,
    ) -> Result<(), Error> {
        self.reaches_unnamed = true;
        if self.status == CommandBufferStatus::NotEnqueued {
            self.root_ranges.extend(ranges);
            self.root_tables.push(noted);
            return Ok(());
        }
        // A host step's before the first submission is noted too, which points it again where
        // the live buffers changed by then; the set it is pointed at now is made of the list as
        // it stands.
        if !self.submitted {
            self.root_ranges.extend(ranges.iter().copied());
            self.root_tables.push(noted.clone());
            self.resident(device)?;
            self.hold_live(device)?;
        }
        let NotedTable {
            upload,
            address: table,
        } = noted;
        let resident = self.resident(device)?;
        // Held from the first submission, and from a host step's before it (above).
        let set = resident
            .merged
            .as_ref()
            .or(resident.live.set())
            .ok_or_else(|| internal("a host step's root table before its live table is held"))?;
        if ranges.iter().all(|range| set.holds(range)) {
            return self.point(&upload, table);
        }
        let both = set.with(ranges);
        let (address, count) = self.upload_set(device, &both)?;
        write_fields(&upload, table, address, count)
    }

    /// Point every root table noted before the commit at the command buffer's resident set.
    fn point_root_tables(&mut self, device: &Device) -> Result<(), Error> {
        if self.root_tables.is_empty() {
            return Ok(());
        }
        self.resident(device)?;
        for noted in &self.root_tables {
            self.point(&noted.upload, noted.address)?;
        }
        Ok(())
    }

    /// The command buffer's resident set, made on first need: the device's live buffers, and
    /// the ranges its root tables noted and its encoders made resident beyond them.
    fn resident(&mut self, device: &Device) -> Result<&Resident, Error> {
        if self.resident.is_none() {
            let resident = self.make_resident(device)?;
            self.resident = Some(resident);
        }
        self.resident
            .as_ref()
            .ok_or_else(|| internal("a command buffer's resident set not made"))
    }

    fn make_resident(&mut self, device: &Device) -> Result<Resident, Error> {
        let live = device.shared().live_buffers.table(device)?;
        let set = live.set();
        let mut beyond: Vec<ResidentRange> = self.root_ranges.iter().copied().collect();
        if let Some(residency) = &self.residency {
            beyond.extend(residency.resident_ranges());
        }
        beyond.retain(|range| !set.holds(range));
        let (merged, address, count) = match beyond.is_empty() {
            true => (None, live.address(), set.len()),
            false => {
                let merged = set.with(beyond);
                let (address, count) = self.upload_set(device, &merged)?;
                (Some(merged), Some(address), count)
            }
        };
        // Held weakly until the first submission, which makes the set again where the list
        // changed since.
        Ok(Resident {
            live: Live::of(live, self.submitted),
            merged,
            address,
            count,
        })
    }

    /// `set` uploaded to the arena at a 32-byte-aligned device address: that address and its
    /// count.
    fn upload_set(&mut self, device: &Device, set: &ResidentSet) -> Result<(u64, u64), Error> {
        let bytes = set.bytes();
        let upload = self.uploads.upload_aligned(
            device,
            &bytes,
            bytes.len() as u64,
            RootTable::ENTRY_ALIGNMENT,
        )?;
        let address = upload
            .device_address()
            .ok_or_else(|| internal("a resident set without a device address"))?;
        Ok((address, set.len()))
    }

    /// Point the root table at `table` (`upload`) at the command buffer's resident set, once
    /// made.
    fn point(&self, upload: &Upload, table: u64) -> Result<(), Error> {
        match &self.resident {
            Some(resident) => write_fields(
                upload,
                table,
                resident.address.unwrap_or(table),
                resident.count,
            ),
            None => Ok(()),
        }
    }

    /// Note `heaps`, which an ended encoder's commands used.
    pub(super) fn use_heaps(&mut self, heaps: Vec<Heap>) {
        for heap in heaps {
            if !self.used_heaps.contains(&heap) {
                self.used_heaps.push(heap);
            }
        }
    }

    /// Why nothing more can be encoded or committed now: the command buffer was committed (its
    /// commit's own work waiting or not, `render/queued_commits.rs`), an encoder is open, or one
    /// was abandoned.
    pub(super) fn refusal(&self) -> Option<Problem> {
        match self.queued_commit {
            true => Some(Problem::AlreadyCommitted),
            false => self.commit_refusal(),
        }
    }

    /// [`refusal`](Self::refusal), but of a commit whose own work waited
    /// (`render/queued_commits.rs`), as it runs.
    fn commit_refusal(&self) -> Option<Problem> {
        if self.status != CommandBufferStatus::NotEnqueued || self.committing {
            Some(Problem::AlreadyCommitted)
        } else if self.encoder_open {
            Some(Problem::EncoderOpen)
        } else if self.abandoned {
            Some(Problem::EncoderAbandoned)
        } else {
            None
        }
    }

    /// Keep a command's function-table `snapshot` to write again as the command buffer commits. A
    /// snapshot a host step writes after the commit already holds the words as they stand then.
    pub(super) fn snapshot_tables(&mut self, snapshot: TableSnapshot) {
        if self.status == CommandBufferStatus::NotEnqueued {
            self.table_snapshots.push(snapshot);
        }
    }

    /// A command left a live sampler state out of its argument samplers: its variant's set-1
    /// sampler arrays hold fewer elements than the live states (`heap_sets::fit_samplers`), so
    /// its function reads the neutral sampler where Metal's reaches the state by its ID. The
    /// command buffer is counted once (`WorkCounts::unlisted_argument_sampler_buffers`).
    pub(super) fn unlisted_argument_sampler(&mut self, device: &Device) {
        if !self.unlisted_argument_samplers {
            self.unlisted_argument_samplers = true;
            device.shared().counts.unlisted_argument_sampler_buffer();
        }
    }

    /// The place among the ended encoders the open encoder takes as it ends: one encoder is open
    /// at a time, and one dropped before it ended abandons the command buffer, which then
    /// commits nothing.
    pub(super) fn open_encoder_place(&self) -> usize {
        self.ended.len()
    }

    /// The encoders, or parts of them, ended so far.
    pub(super) fn parts(&self) -> usize {
        self.recorded.len()
    }

    /// The predication the open encoder's records and dispatches use: that of an encoder a rebind
    /// encodes again now ([`replaying_predication`](Self::replaying_predication)), else the
    /// command buffer's.
    pub(super) fn predication_mut(&mut self) -> &mut Predication {
        match &mut self.replaying_predication {
            Some(replaying) => replaying,
            None => &mut self.predication,
        }
    }

    /// [`predication_mut`](Self::predication_mut), with the upload arena.
    pub(super) fn predication_and_uploads(&mut self) -> (&mut Predication, &mut Uploads) {
        let predication = match &mut self.replaying_predication {
            Some(replaying) => replaying,
            None => &mut self.predication,
        };
        (predication, &mut self.uploads)
    }

    /// A compute encoder of `dispatch_type` opened: it joins the unit of command predication of
    /// the compute encoder that ended just before it, of its dispatch type with no event wait or
    /// signal between and no fence update in it, taking the bodies of the loops it left open, or
    /// starts the next (`predication.rs`).
    pub(super) fn enter_predication(
        &mut self,
        dispatch_type: DispatchType,
    ) -> Vec<Vec<(EncoderState, Event)>> {
        let place = self.open_encoder_place();
        self.predication
            .enter(place, self.batches.events(), dispatch_type)
    }

    /// The failure of the making of a draw's pipeline a device test injected
    /// ([`CommandBuffer::fail_draw_making`]), taken here.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn injected_making_failure(&mut self) -> Option<Error> {
        self.making_failure_injected.take()
    }

    /// Keep `error`, a render encoder's call's refused as it was lowered
    /// (`CommandBufferError::CallRefused`): the first such call's.
    pub(super) fn refuse_call(&mut self, error: Error) {
        self.refusal.get_or_insert(error);
    }

    /// The open render encoder's end, lowered after its call, refused for `error`
    /// (`render/deferral.rs`): the encoder, which its end's refusal dropped unended, ends here
    /// with nothing of it run, and the command buffer fails from it on as a refusal of one of its
    /// calls does (`CommandBufferError::CallRefused` of `call`, its error kept), commits, and
    /// runs nothing of it or after it.
    pub(super) fn refuse_end(&mut self, call: EncoderCall, error: Error) {
        self.encoder_open = false;
        self.abandoned = false;
        let place = self.open_encoder_place();
        self.fail_from(place, CommandBufferError::CallRefused(call));
        self.refuse_call(error);
    }

    /// The commit's own work, which waited (`render/queued_commits.rs`), refused it for
    /// `error`, found after `commit` returned: nothing of the command buffer runs, and it fails
    /// as its first refused call names it (`CommandBufferError::CallRefused`): the commit, as
    /// `commit` at index 0, where no call of its encoders was refused before. The refusal's
    /// error is kept, as a call's is.
    fn refuse_commit(&mut self, error: Error) {
        let commit = CommandBufferError::CallRefused(EncoderCall::new("commit", 0));
        let first = self.fails_from.map_or(commit, |(_, first)| first);
        self.fails_from = Some((0, first));
        self.refuse_call(error);
    }

    /// Ended encoder `encoder` fails the command buffer from it on with `error` (`fails_from`).
    pub(super) fn fail_from(&mut self, encoder: usize, error: CommandBufferError) {
        self.fails_from.get_or_insert((encoder, error));
    }

    /// Ended encoder `encoder`, the last, fails the command buffer with `error` from the first
    /// encoder of the submission Metal holds it in on (`checks::metal_submissions`: a render pass
    /// alone, a run of blit or of compute encoders that no event command splits), as Metal fails
    /// a submission that uses a memoryless texture (`CommandBufferError::InvalidResource`); an
    /// earlier failure keeps its error, from the earlier of the two encoders.
    pub(super) fn fail_submission_from(&mut self, encoder: usize, error: CommandBufferError) {
        let records = self.ended.iter().map(|record| {
            (
                record.kind,
                record.event_waits_before,
                record.event_signals_before,
            )
        });
        let first = checks::submission_start(records, encoder);
        self.fails_from = Some(match self.fails_from {
            Some((from, earlier)) => (from.min(first), earlier),
            None => (first, error),
        });
    }

    /// Note the shadowed textures an ended encoder's pass took as attachments: how many of them
    /// no pass of the command buffer took before.
    pub(super) fn shadows_attached(&mut self, keys: &[ResourceKey]) -> usize {
        let mut first = 0;
        for &key in keys {
            if !self.shadow_attachments.contains(&key) {
                self.shadow_attachments.push(key);
                first += 1;
            }
        }
        first
    }

    /// Note the shadowed textures an ended encoder copied in first, each copy's texture and
    /// texels: the texels of those an earlier encoder of the command buffer copied in too.
    pub(super) fn shadows_copied_in(&mut self, copies: &[(ResourceKey, u64)]) -> u64 {
        let earlier = self.shadows_copied_in.len();
        let mut again = 0;
        for &(key, texels) in copies {
            if self.shadows_copied_in[..earlier].contains(&key) {
                again += texels;
            } else if !self.shadows_copied_in.contains(&key) {
                self.shadows_copied_in.push(key);
            }
        }
        again
    }

    /// What its ended encoders left shadowed textures' images holding, for the encoder opening
    /// now (`shadow::ShadowCarry`): the command buffer carries nothing until it is handed back
    /// by [`shadows_carried`](Self::shadows_carried), so an encoder abandoned without ending
    /// leaves it carrying none and every texture copies in again.
    pub(super) fn shadows_to_carry(&mut self) -> ShadowCarry {
        self.shadow_carry.take().unwrap_or_default()
    }

    /// Take back what an ended encoder leaves its command buffer's later encoders
    /// (`shadow::ShadowSync::ended`).
    pub(super) fn shadows_carried(&mut self, carry: ShadowCarry) {
        self.shadow_carry = Some(carry);
    }

    /// Carry no shadowed texture's image on: the work encoded next reads what another writer the
    /// wait orders before it left in the truth, which every texture copies in again to hold
    /// (`shadow::ShadowCarry`).
    pub(super) fn shadows_not_carried(&mut self) {
        if let Some(carry) = &mut self.shadow_carry {
            carry.clear();
        }
    }

    /// Metal's GPU faults ended encoder `encoder`, failing the command buffer with `error` from
    /// it on (`faulted`, `Failure::with_faults`); where nothing failed it from an earlier encoder
    /// and no earlier encoder faulted, the work before the encoder signals the fault's hold and
    /// what follows waits for it (`hang.rs`), so the command buffer completes when Metal's does.
    /// Where an earlier encoder faulted, the GPU reaches this one once it recovered: the earlier
    /// fault's hold counts it (`Hang::fails_later`).
    pub(super) fn fault(
        &mut self,
        encoder: usize,
        error: CommandBufferError,
        device: &Device,
    ) -> Result<(), Error> {
        if self.fails_from.is_none() && self.hang.is_none() {
            self.hold(device, encoder, Held::Fault(Later::default()))?;
        } else if !self.faulted.contains_key(&encoder) {
            if let Some(hang) = &mut self.hang {
                hang.fails_later(false);
            }
        }
        self.faulted.entry(encoder).or_insert(error);
        Ok(())
    }

    /// Ended encoder `encoder` drew primitive type 9 directly, at which Metal's GPU hangs
    /// (`hang.rs`): the command buffer fails from it on with `CommandBufferError::GpuHang`, and,
    /// where nothing failed it from an earlier encoder, the work before the encoder signals the
    /// hang's event and what follows waits for the hang's hold. Where an earlier encoder faulted,
    /// it fails from the fault with the fault's error (`State::fault`), and the GPU reaches it
    /// once it recovered: the fault's hold counts it (`Hang::fails_later`).
    pub(super) fn hang_from(&mut self, encoder: usize, device: &Device) -> Result<(), Error> {
        if !self.faulted.is_empty() {
            if let Some(hang) = &mut self.hang {
                hang.fails_later(true);
            }
            return Ok(());
        }
        if self.fails_from.is_none() && self.hang.is_none() {
            self.hold(device, encoder, Held::Hang)?;
        }
        self.fail_from(encoder, CommandBufferError::GpuHang);
        Ok(())
    }

    /// Hold what follows ended encoder `encoder`, the last, of the hold `held`: its batch signals
    /// the hold's event, and the next batch waits for its release (`hang.rs`).
    fn hold(&mut self, device: &Device, encoder: usize, held: Held) -> Result<(), Error> {
        let hang = Hang::new(device, (self.batches.all().len() - 1, encoder), held)?;
        let event = || Arc::clone(EventRef::Shared(hang.event()).shared());
        self.batches.hold_signal(event(), Hang::REACHED);
        self.batches.hold(event(), Hang::RELEASED);
        self.hang = Some(hang);
        Ok(())
    }

    /// The holds of what its commands took that batch `index`'s work lets go as it completes
    /// (`State::submitted_holds`), for its submission to carry (`completion::Submitted`).
    pub(super) fn holds_until(&mut self, index: usize) -> Vec<SubmittedHolds> {
        if self.submitted_holds.is_empty() {
            return Vec::new();
        }
        let (now, later): (Vec<_>, Vec<_>) = self
            .submitted_holds
            .drain(..)
            .partition(|(until, _)| *until <= index);
        self.submitted_holds = later;
        now.into_iter().map(|(_, holds)| holds).collect()
    }

    /// The holds of its last batch, whose work completed: kept until its completion is processed,
    /// which lets go of every hold it still counts before its completed handlers run
    /// (`State::submitted_holds`, `CommandBufferShared::complete`).
    pub(super) fn keep_holds(&mut self, holds: Vec<SubmittedHolds>) {
        for held in holds {
            self.submitted_holds.push((usize::MAX, held));
        }
    }

    /// The encoders (or parts of them) of batch `index`.
    pub(super) fn batch_range(&self, index: usize) -> Range<usize> {
        let batches = self.batches.all();
        let start = index.checked_sub(1).map_or(0, |before| batches[before].end);
        start..batches[index].end
    }

    /// The next encoder, a part of one that executes commands the GPU may write, starts a batch
    /// of its own, which waits for a host step.
    pub(super) fn host_step(&mut self) {
        self.batches.host_step();
    }

    /// A resource state encoder of `command_buffer` ended with `recorded`: it starts a batch of
    /// its own, unless the batch so far holds only such encoders, so its updates bind before
    /// every encoder after it.
    pub(super) fn end_sparse_encoder(
        &mut self,
        command_buffer: &CommandBuffer,
        recorded: Recorded,
    ) {
        self.batches.sparse();
        self.end_encoder(command_buffer, recorded);
    }

    /// An encoder of `command_buffer` ended with `recorded`: it joins the current batch, or a
    /// new one after the batch's signals. The tail before it ends: with its batch's end where
    /// that batch ended before this encoder ([`CommandBufferShared::close_tail`]).
    pub(super) fn end_encoder(&mut self, command_buffer: &CommandBuffer, recorded: Recorded) {
        self.recorded.push(recorded);
        self.batches.encoder();
        self.encoder_open = false;
        let joined = self.recorded.len() - 1;
        command_buffer.0.close_tail(self, Some(joined));
    }

    /// The batch that holds part `part` of the ended encoders.
    fn batch_of(&self, part: usize) -> usize {
        let batches = self.batches.all();
        batches
            .iter()
            .position(|batch| batch.end > part)
            .unwrap_or(batches.len() - 1)
    }

    /// Whether batch `index`'s end may go into its last `VkCommandBuffer` (`submit.rs`): its
    /// parts leave no segment or sparse update for its submission (whose recordings go among or
    /// after the parts' own), nor its last part a write into an ASTC carrier (whose second twin's
    /// decode its submission may record after it, `texture/decoded_twin.rs`), and name no texture
    /// over the caller's image that moves between layouts (whose move back its submission plans
    /// before the end). An event's kept signal that waits for another queue's still goes after
    /// that wait, in an end of its own at the submission (`submit.rs`).
    fn may_fold(&self, device: &Device, index: usize) -> bool {
        let parts = &self.recorded[self.batch_range(index)];
        parts
            .iter()
            .all(|part| part.segments.is_empty() && part.sparse.is_empty())
            && parts.last().is_none_or(|part| part.twin_writes.is_empty())
            && device
                .shared()
                .interop
                .wrapped
                .among(parts.iter().flat_map(|part| part.summary.keys()))
                .is_empty()
    }

    /// The `VkCommandBuffer` batch `index`'s end went into, and whether it wrote the command
    /// buffer's end timestamp; `None` where it went into none.
    pub(super) fn fold(&self, index: usize) -> Option<(vk::CommandBuffer, bool)> {
        self.folds
            .iter()
            .find(|fold| fold.batch == index)
            .map(|fold| (fold.raw, fold.stamped))
    }

    /// Fail the command buffer as `failure` says ([`CommandBuffer::commit_failed`], or an
    /// encoder Metal fails it from, `fails_from`), before `commit` hands its work on the segments
    /// of the encoders that run to them: the recordings of the encoders `failure.encoders` names
    /// (every part of each) are released unrun, each part keeping its place among the batches
    /// with its fences' waits and updates alone; the event signals `failure.signals` drops are
    /// not signaled; and the command buffer finishes with `failure.error`, where it has one
    /// (a code-0 `commit_failed` has none).
    fn fail(&mut self, failure: Failure) {
        let State {
            recorded, retained, ..
        } = self;
        for part in recorded
            .iter_mut()
            .filter(|part| failure.encoders.fails(part.encoder))
        {
            State::fail_part(part, retained);
        }
        match failure.signals {
            FailedSignals::Every => {}
            FailedSignals::Dropped => self.batches.drop_signals(),
            FailedSignals::From(encoder) => {
                // The parts before the encoder's first, which the batches count.
                let parts = self
                    .recorded
                    .iter()
                    .take_while(|part| part.encoder < encoder)
                    .count();
                self.batches.drop_signals_after(parts);
            }
        }
        if let Some(error) = failure.error {
            self.failure = Some(error);
        }
        self.failed_at_commit = Some(failure.encoders);
    }

    /// A command-predication record dropped what the ended encoders from `first` on recorded,
    /// its unit's earlier encoders (`predication.rs`): each of their parts runs nothing, as a
    /// failed encoder's ([`fail_part`](Self::fail_part)).
    pub(super) fn leave_out_ended(&mut self, first: usize) {
        let State {
            recorded, retained, ..
        } = self;
        for part in recorded.iter_mut().filter(|part| part.encoder >= first) {
            State::fail_part(part, retained);
        }
    }

    /// A failed encoder's recorded `part` runs nothing: its segments (with their recordings),
    /// sparse updates and writes into ASTC carriers go to `retained`, released with what the
    /// command buffer retained without this lock, its `VkCommandBuffer`s are left unsubmitted,
    /// and it keeps its place among the batches with its fences' waits and updates alone.
    pub(super) fn fail_part(part: &mut Recorded, retained: &mut Vec<Box<dyn Any + Send + Sync>>) {
        for segment in part.segments.drain(..) {
            retained.push(Box::new((segment.work, segment.inputs)));
            if let Some(mut done) = segment.recorded {
                retained.extend(done.retained.into_entry());
                done.taken.retain_into(retained);
            }
        }
        for update in part.sparse.drain(..) {
            retained.push(Box::new(update));
        }
        // Its writes into ASTC carriers ran nothing, so nothing of them is decoded.
        for write in part.twin_writes.drain(..) {
            retained.push(Box::new(write));
        }
        part.raws.clear();
        part.summary = part.summary.fences_only();
    }

    /// Hand `commit`'s work on the segments, one per segment in encoding order, to them: the
    /// recording of a segment `commit` made, for good where no host step records it, and what a
    /// host step needs.
    fn prepare(&mut self, prepared: Vec<Prepared>) {
        let segments = self
            .recorded
            .iter_mut()
            .flat_map(|encoder| encoder.segments.iter_mut());
        let mut checks = Vec::new();
        for (segment, prepared) in segments.zip(prepared) {
            let Prepared {
                recorded,
                for_good,
                inputs,
                retained,
                pipelines,
            } = prepared;
            segment.gpu_alone = for_good && recorded.is_some();
            segment.recorded = recorded;
            segment.stepped |= for_good;
            segment.pipelines |= pipelines;
            segment.inputs = Some(inputs);
            self.retained.extend(retained);
            if let Deferred::Loop(check) = &segment.work {
                checks.push(Arc::clone(check));
            }
        }
        // A loop's check learns how a run again records the work its laying left apart, from
        // the segments the laying's parts hold, each prepared now (`rerun.rs`).
        for check in checks {
            check.learn(self.recorded.get(check.from()..).unwrap_or_default());
        }
    }

    /// Put the segments of encoders (or parts of them) `range`, each recorded, in their places as
    /// their batch is submitted: each `VkCommandBuffer` among its part's, its uses joined to the
    /// part's summary, its dispatches among the executions in encoding order, and what it used
    /// retained. Refused, changing nothing, where a segment is not recorded.
    fn place(&mut self, range: Range<usize>) -> Result<(), Error> {
        let State {
            recorded,
            executions,
            retained,
            resident_misses,
            faults,
            placed,
            ended,
            ..
        } = self;
        let encoders = &mut recorded[range];
        if encoders
            .iter()
            .flat_map(|encoder| &encoder.segments)
            .any(|segment| segment.recorded.is_none())
        {
            return Err(internal("a segment submitted unrecorded"));
        }
        // A segment that recorded nothing for a command (a buffer it named was detached) is
        // never placed: its submission failed (`State::failing_at`).
        let dead = encoders
            .iter()
            .flat_map(|encoder| &encoder.segments)
            .filter_map(|segment| segment.recorded.as_ref())
            .any(|done| done.taken.dead());
        debug_assert!(!dead, "a segment that named a detached buffer submitted");
        if dead {
            return Err(internal("a segment that named a detached buffer submitted"));
        }
        for encoder in encoders {
            // The segments' command buffers placed so far, each segment's after the part's own
            // before it and the segments' before it.
            let mut before = 0;
            for segment in encoder.segments.drain(..) {
                let Segment {
                    at,
                    executions_at,
                    work,
                    inputs,
                    recorded,
                    ..
                } = segment;
                // Released with what the command buffer retained, without this lock.
                retained.push(Box::new((work, inputs)));
                let Some(mut done) = recorded else {
                    continue;
                };
                // The pages its commands took, held until the command buffer completes.
                done.taken.retain_into(retained);
                let raws = done.raws.len();
                encoder.raws.splice(at + before..at + before, done.raws);
                before += raws;
                encoder.summary.join(done.summary);
                let at = executions_at + *placed;
                *placed += done.executions.len();
                executions.splice(at..at, done.executions);
                retained.extend(done.retained.into_entry());
                // The segment's keys are its encoder's.
                if let Some(record) = ended.get_mut(encoder.encoder) {
                    record.keys.extend(done.keys);
                }
                *resident_misses = resident_misses.wrapping_add(done.resident_misses);
                *faults = ReplayFaults(faults.0 | done.faults.0);
            }
        }
        Ok(())
    }

    /// What the segments a failed command buffer never placed retained, for its release.
    fn unplaced(&mut self) {
        let leftover: Vec<_> = self
            .recorded
            .iter_mut()
            .flat_map(|encoder| encoder.segments.drain(..))
            .collect();
        for segment in leftover {
            let Segment {
                work,
                inputs,
                recorded,
                ..
            } = segment;
            self.retained.push(Box::new((work, inputs)));
            if let Some(mut done) = recorded {
                self.retained.extend(done.retained.into_entry());
                done.taken.retain_into(&mut self.retained);
            }
        }
    }

    /// The device address of the command buffer's execution status words, uploaded (all zero,
    /// `ExecutionStatus::initial`) when a command first reaches them; `None` on a device without
    /// device addresses.
    pub(super) fn status_address(&mut self, device: &Device) -> Result<Option<u64>, Error> {
        Ok(self.status_words(device)?.device_address())
    }

    /// A direct mesh draw's object grid lies past what the device launches
    /// (`render/mesh_arguments.rs`): the command buffer fails.
    pub(crate) fn mesh_grid_past_limits(&mut self) {
        self.mesh_grid_past = true;
    }

    /// The command buffer's execution status words, uploaded (all zero, `ExecutionStatus::initial`)
    /// when a command first reaches them.
    pub(super) fn status_words(&mut self, device: &Device) -> Result<Upload, Error> {
        if let Some(words) = &self.status_words {
            return Ok(words.clone());
        }
        let words = self.uploads.upload(
            device,
            &ExecutionStatus::initial(),
            ExecutionStatus::BYTES as u64,
        )?;
        self.status_words = Some(words.clone());
        Ok(words)
    }

    /// The command buffer's index pool (`index_bounds.rs`, `index_rewrite.rs`), taken when a
    /// draw first needs it: its counter, `IndexBoundsPass::POOL_HEADER_BYTES` of the arena, zero;
    /// and the block its indices lie in, from the block's
    /// `IndexBoundsPass::POOL_HEADER_BYTES` on, its queue's (`queue`, the command buffer's
    /// `CommandQueue::id`), leased from the device (`gpu_pools.rs`): the block's `VkBuffer` and
    /// bytes, the header's included.
    pub(super) fn index_pool(&mut self, device: &Device, queue: u64) -> Result<IndexPool, Error> {
        if let Some((counter, block)) = &self.index_pool {
            return Ok(IndexPool::of(counter, block));
        }
        let header = IndexBoundsPass::POOL_HEADER_BYTES;
        let counter = self
            .uploads
            .upload(device, &vec![0; header as usize], header)?;
        let block = device.lease_pool(PoolKind::Index, queue)?;
        let pool = IndexPool::of(&counter, &block);
        self.index_pool = Some((counter, block));
        Ok(pool)
    }

    /// The command buffer's mesh records pool (`render/mesh.rs`): a block of the device's
    /// budget, leased when a scanned emulated mesh draw first needs it, a host step's replay of
    /// one among them (`gpu_pools.rs`): its `VkBuffer` and bytes. Its words need no value before
    /// the scans reserve them.
    pub(super) fn mesh_pool(
        &mut self,
        device: &Device,
        queue: u64,
    ) -> Result<(vk::Buffer, u64), Error> {
        if let Some(pool) = &self.mesh_pool {
            return Ok((pool.raw(), pool.bytes()));
        }
        let pool = device.lease_pool(PoolKind::MeshRecords, queue)?;
        let leased = (pool.raw(), pool.bytes());
        self.mesh_pool = Some(pool);
        Ok(leased)
    }

    /// The number of the command buffer's next emulated mesh draw: its order among them, from 0
    /// (`MeshDispatch::DRAW_ID`).
    pub(super) fn next_mesh_draw(&mut self) -> u32 {
        let number = self.mesh_draws;
        self.mesh_draws = number.saturating_add(1);
        number
    }
}

impl Drop for CommandBufferShared {
    fn drop(&mut self) {
        // A command buffer that still holds its pool was never submitted, or ended before its
        // completion gave the pool back: a committed command buffer is held by the scheduler until
        // its batches were submitted, and by its queue's waiter until they completed, so
        // nothing of its pool is pending. The device's spares keep it (`spares.rs`), which
        // release it only with the device. A timestamp slot it still holds goes back with its
        // state.
        let pool = lock(&self.state).pool.take();
        if let Some(pool) = pool {
            let shared = self.device.shared();
            shared.spares.give_command_pool(shared, pool, false);
        }
    }
}

/// Point the root table at `table` (`upload`) at `count` entries of a resident set at `set`.
fn write_fields(upload: &Upload, table: u64, set: u64, count: u64) -> Result<(), Error> {
    let fields = RootTable::resident_fields(table, set, count).ok_or_else(|| {
        internal("a root table not a whole number of entries from its resident set")
    })?;
    upload.write_at(u64::from(RootTable::RESIDENT_FIELDS_OFFSET), &fields);
    Ok(())
}

impl CommandBuffer {
    /// A new command buffer of `queue`, holding `slot` of it.
    pub(super) fn new(queue: &CommandQueue, slot: Slot) -> CommandBuffer {
        CommandBuffer(Arc::new(CommandBufferShared {
            device: queue.device().clone(),
            queue: queue.clone(),
            state: Mutex::new(State {
                status: CommandBufferStatus::NotEnqueued,
                error: None,
                encoder_open: false,
                abandoned: false,
                ending: None,
                end_lowered: false,
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                failures_injected: 0,
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                making_failure_injected: None,
                queued_commit: false,
                called_over_mutable_map: false,
                committing: false,
                pool: None,
                rerun_pool: None,
                recorded: Vec::new(),
                tail: None,
                folds: Vec::new(),
                prefix: None,
                batches: Batches::default(),
                scheduled_signals: Vec::new(),
                scheduled: VecDeque::new(),
                submission: VecDeque::new(),
                submitting: false,
                scheduled_ran: false,
                scheduled_work: ScheduledWork::default(),
                parked: 0,
                failure: None,
                completed: VecDeque::new(),
                executed_handlers: Vec::new(),
                retained: Vec::new(),
                executions: Vec::new(),
                uploads: Uploads::new(),
                heaps: HeapUse::default(),
                table_snapshots: Vec::new(),
                unlisted_argument_samplers: false,
                fails_from: None,
                faulted: BTreeMap::new(),
                hang: None,
                hang_scheduling: None,
                behind_hang: None,
                shadow_attachments: Vec::new(),
                shadows_copied_in: Vec::new(),
                shadow_carry: Some(ShadowCarry::default()),
                descriptors: DescriptorArena::default(),
                status_words: None,
                mesh_grid_past: false,
                index_pool: None,
                mesh_pool: None,
                mesh_draws: 0,
                resident_misses: 0,
                faults: ReplayFaults::default(),
                placed: 0,
                step_failure: None,
                refusal: None,
                failed_pipeline: None,
                dropped_draw: None,
                relayed_description: None,
                report: None,
                times: Times::default(),
                slot: Some(slot),
                used_heaps: Vec::new(),
                holds: Vec::new(),
                residency: None,
                ended: Vec::new(),
                root_tables: Vec::new(),
                root_ranges: HashSet::new(),
                resident: None,
                stepped_tables: Vec::new(),
                listing: None,
                submitted: false,
                judged_to: 0,
                protection: protection::Protection::default(),
                failing: Vec::new(),
                rebound_use: None,
                replay_tables: Vec::new(),
                reaches_unnamed: false,
                unrunnable: BTreeSet::new(),
                submitted_holds: Vec::new(),
                failed_at_commit: None,
                check: None,
                checks: None,
                storage_form: None,
                predication: Predication::default(),
                replaying_predication: None,
                predicated: false,
                trail_mark: 0,
            }),
            changed: Condvar::new(),
            gate: Arc::default(),
            relayed: Arc::default(),
            step_numbers: OnceLock::new(),
            making_deadline: OnceLock::new(),
            own_textures: Mutex::new(Vec::new()),
            fill_slot: Mutex::new([0; 16]),
            reencode: AtomicU64::new(0),
            reencoding: Mutex::new(()),
            dirty: Mutex::new(DirtyCommit::default()),
            unnoted: AtomicBool::new(false),
            raws_ended: AtomicU64::new(0),
            kernel: KernelTimes::default(),
            status: AtomicU8::new(status_word(CommandBufferStatus::NotEnqueued)),
            error: OnceLock::new(),
            group_hold: AtomicU8::new(0),
        }))
    }

    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// The color fill slot (`fill_slot`), after a color fill packed `packed` (`None` for one
    /// that packs nothing): its first bytes become the texel's.
    pub(super) fn fill_slot(&self, packed: Option<&[u8]>) -> [u8; 16] {
        let mut slot = lock(&self.0.fill_slot);
        if let Some(texel) = packed {
            let n = texel.len().min(slot.len());
            slot[..n].copy_from_slice(&texel[..n]);
        }
        *slot
    }

    /// Hold `texture`, which one of its encoders made for its own commands, as its
    /// `own_textures` says.
    pub(super) fn hold_own_texture(&self, texture: Texture) {
        lock(&self.0.own_textures).push(texture);
    }

    /// `commandQueue`.
    pub fn command_queue(&self) -> &CommandQueue {
        &self.0.queue
    }

    /// The state, locked.
    /// Note `work` of an encoder it ended, which its scheduling hands to the device's identifier
    /// allocations ([`UniqueIdentifiers::scheduled`](crate::unique_identifier::UniqueIdentifiers::scheduled)).
    pub(super) fn note_scheduled_work(&self, work: ScheduledWork) {
        let mut state = self.state();
        state.scheduled_work.dispatch |= work.dispatch;
        state.scheduled_work.render |= work.render;
    }

    pub(super) fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.0.state)
    }

    /// Note a command's `uses` for the device's logs of host pages, where it keeps any
    /// (`dirty_pages.rs`): its writes let go, at the commit, of the holds of the shadows they meet,
    /// and those into logged pages imported in place are reported to their logs; and for its
    /// copied host buffers, where any lives, whose merges back compare those its writes reach
    /// (`copied.rs`).
    pub(super) fn note_dirty(&self, uses: &[Use<ResourceKey>]) {
        let shared = self.0.device.shared();
        match shared.dirty.collects() || shared.copied.any() {
            true => lock(&self.0.dirty).note(uses, &shared.dirty),
            false => self.0.unnoted.store(true, Ordering::Relaxed),
        }
    }

    /// What its commands did so far that the device's logs of host pages hear of at its commit.
    pub(super) fn dirty(&self) -> MutexGuard<'_, DirtyCommit> {
        lock(&self.0.dirty)
    }

    /// Whether a buffer over host pages a command of its ended encoder at `encoder` named lies
    /// elsewhere now than the command took it (`Rebinds::rebound_now`): a rebind since its commit
    /// that no encoding again reached. The submission holding that encoder runs nothing and the
    /// command buffer fails `CommandBufferError::EncodedUseRebound` (`first_submission.rs`), so
    /// work of the encoder a host step records resolved as it was encoded records nothing
    /// (`pending_dispatch.rs`): the descriptors it would record name a `VkBuffer` the rebind
    /// freed. Read without the state's lock held, as the lock each submission reads it under is
    /// taken inside this one.
    pub(super) fn encoder_rebound(&self, encoder: usize) -> bool {
        let rebinds = &self.0.device.shared().rebinds;
        if !rebinds.any() {
            return false;
        }
        let state = self.state();
        let Some(record) = state.ended.get(encoder) else {
            return false;
        };
        rebinds.rebound_now(&record.taken)
    }

    /// This command buffer as the device's commits that wait hold it ([`Waiting`]).
    pub(crate) fn waiting(self) -> Arc<dyn Waiting> {
        self.0
    }

    /// Fail the work of its next `count` commits (not one run again with nothing of it run), as
    /// out of memory: a device test's injection (`tests::deferred_calls`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn fail_commit_work(&self, count: u32) {
        self.state().failures_injected = count;
    }

    /// Answer the making of the next draw's pipeline its recording waits for
    /// (`render/pending.rs`) with `error`, as the driver's failure: a device test's injection
    /// (`tests::deferred_calls`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn fail_draw_making(&self, error: Error) {
        self.state().making_failure_injected = Some(error);
    }

    /// The bytes its upload arena's regions took: a device test's measure (`tests::rerun_arenas`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn upload_bytes(&self) -> u64 {
        self.state().uploads.bytes()
    }

    /// Whether `waiting` holds this command buffer.
    pub(crate) fn is(&self, waiting: &dyn Waiting) -> bool {
        self.0.address() == waiting.address()
    }

    /// A handle that does not keep the command buffer: a render encoder's, which its end in
    /// flight hands the command buffer (`render/deferral.rs`), so that hold makes no cycle.
    pub(crate) fn downgrade(&self) -> WeakCommandBuffer {
        WeakCommandBuffer(Arc::downgrade(&self.0))
    }

    /// Hold `ending`, the open render encoder's end, which its call left to the device's
    /// lowering thread, until what follows it in the command buffer lowers it
    /// ([`settle_end`](Self::settle_end)).
    pub(crate) fn hold_end(&self, ending: EndInFlight) {
        let mut state = self.state();
        let held = state.ending.replace(ending);
        debug_assert!(held.is_none(), "two render encoders' ends in flight");
        state.end_lowered = false;
    }

    /// The render encoder's end the command buffer holds was lowered (`render/deferral.rs`), on
    /// the device's lowering thread (`by_thread`) or the program's: a commit whose own work
    /// waited for it runs now, on this thread, with those ready behind it
    /// (`render/queued_commits.rs`). Then this handle goes, off the lowering thread where it is
    /// the last ([`let_go_off_thread`](Self::let_go_off_thread)).
    pub(crate) fn end_lowered(self, by_thread: bool) {
        let queued = {
            let mut state = self.state();
            state.end_lowered = true;
            state.queued_commit
        };
        let commits = self.0.device.shared().deferred.commits();
        if let Some(commits) = commits.filter(|commits| queued && commits.ready(&self)) {
            commits.run(by_thread);
        }
        if by_thread {
            self.let_go_off_thread();
        }
    }

    /// Let go of this handle on the device's lowering thread, which has just lowered a render
    /// encoder's end (`render/deferral.rs`): where it is the command buffer's last, the command
    /// buffer goes on one of the device's handler threads (`Device::let_go_elsewhere`), as it would
    /// go on the program's thread that let go of it last, since its drop gives the encoder's slot
    /// back, whose logged calls may hold the program's objects' last handles.
    pub(crate) fn let_go_off_thread(self) {
        let device = self.0.device.clone();
        if let Some(shared) = Arc::into_inner(self.0) {
            device.let_go_elsewhere(Box::new(shared));
        }
    }

    /// Lower, on this thread, the end of the render encoder that ended last, where its call
    /// left it to the device's lowering thread and the thread has not lowered it (with whatever
    /// of its calls the thread has not lowered): first in each call that follows an encoder's end
    /// in the command buffer (another encoder's opening, an event or a semaphore, the commit, a
    /// read of what its encoders left), as if the end had been lowered inside its call.
    pub(crate) fn settle_end(&self) {
        let ending = self.state().ending.take();
        if let Some(ending) = ending {
            ending.finish();
        }
    }

    /// The residency a new compute or render (`render`) encoder starts with
    /// (`Residency::shadows`): the shadowed textures its ended encoders made resident, those in
    /// the heaps they used made since included.
    pub(super) fn inherited_residency(&self, render: bool) -> Residency {
        let mut state = self.state();
        let Some(residency) = &mut state.residency else {
            return Residency::default();
        };
        residency.sync_heaps();
        residency.shadows(render)
    }

    /// What its ended encoders made resident, every encoder's joined (`residency.rs`): what the
    /// work its commit and host steps record resolves and binds against.
    pub(super) fn residency(&self) -> Result<Residency, Error> {
        let residency = self.state().residency.clone().unwrap_or_default();
        // With the device's live buffers, which an address in no resource made resident
        // resolves against (`live_buffers.rs`): a host step's, the command buffer holds until it
        // completes.
        let device = &self.0.device;
        let live = device.shared().live_buffers.table(device)?;
        self.state().keep_stepped_table(&live);
        Ok(residency.with_live(live))
    }

    /// An encoder's primary command buffer, from this command buffer's pool, made on first use:
    /// the encoder records into it at each call, on the program's thread. The caller holds
    /// `state` (the pool is externally synchronized).
    pub(super) fn allocate_raw(&self, state: &mut State) -> Result<vk::CommandBuffer, Error> {
        self.0
            .allocate_encoder(state, vk::CommandBufferLevel::PRIMARY)
    }

    /// An encoder's own primary `VkCommandBuffer` from this command buffer's pool, and whether it
    /// is begun: the pool's primary begun ahead where it holds one (`spares.rs`), else as
    /// [`allocate_raw`](Self::allocate_raw) hands one out, to begin. The caller holds `state`.
    pub(super) fn encoder_primary(
        &self,
        state: &mut State,
    ) -> Result<(vk::CommandBuffer, bool), Error> {
        self.0.encoder_primary(state)
    }

    /// Write the start into `raw`, the first encoder's first `VkCommandBuffer`, just begun, where
    /// the command buffer is timed and the encoder opens its first batch (`gpu_time.rs`).
    pub(super) fn open_start(&self, state: &mut State, raw: vk::CommandBuffer) {
        self.0.open_start(state, raw);
    }

    /// End `raw`, a `VkCommandBuffer` of this command buffer's pool, counted among those its
    /// recordings ended: every `vkEndCommandBuffer` of one goes through here
    /// (`CommandBufferShared::end_raw`).
    ///
    /// # Safety
    /// `raw` is recording, and only the caller records it, or any command buffer of its pool:
    /// Vulkan synchronizes both externally.
    pub(super) unsafe fn end_raw(&self, raw: vk::CommandBuffer) -> Result<(), vk::Result> {
        // SAFETY: the caller's.
        unsafe { self.0.end_raw(raw) }
    }

    /// Give up `raw`, a secondary command buffer recording that nothing will execute (a render
    /// pass instance its pass replaced or gave up before running it, or the one instance of a
    /// pass that changes nothing): it ends ([`end_raw`](Self::end_raw)), so its pool goes back
    /// to the device's spares unreset (`spares.rs`). Nothing executes it, so the GPU does the
    /// same work and every byte it writes is the same.
    ///
    /// # Safety
    /// As [`end_raw`](Self::end_raw)'s, and no command buffer executes `raw`.
    pub(super) unsafe fn give_up_raw(&self, raw: vk::CommandBuffer) -> Result<(), vk::Result> {
        // SAFETY: the caller's.
        unsafe { self.end_raw(raw) }
    }

    /// A secondary command buffer (a render encoder's draws), from this command buffer's pool as
    /// [`allocate_raw`](Self::allocate_raw)'s. The caller holds `state`.
    pub(super) fn allocate_secondary(&self, state: &mut State) -> Result<vk::CommandBuffer, Error> {
        self.0
            .allocate_encoder(state, vk::CommandBufferLevel::SECONDARY)
    }

    /// `addScheduledHandler:`: call `handler` once the command buffer is scheduled, on one of the
    /// device's handler threads, at the user-interactive class on macOS as in Metal, and with a
    /// utilization floor on Linux ([`Handler`]). Refused once the command buffer is committed.
    pub fn add_scheduled_handler(&self, handler: Handler) -> Result<(), Error> {
        let mut state = self.state();
        if state.enqueued() {
            return Err(invalid(
                "add_scheduled_handler",
                Problem::HandlerAfterCommit,
            ));
        }
        state.scheduled.push_back(handler);
        Ok(())
    }

    /// Call `handler` once the runtime took the command buffer's last batch for its `VkQueue`,
    /// on the thread that took it, before the command buffer becomes `Scheduled`: a runtime
    /// extension, no Metal call (the crate's README, "Runtime extensions"). The batch is taken
    /// where it is submitted, or held for one kernel submission with the batches
    /// committed after it (`CompletionPolicy::Grouped`, on a driver that merges submissions): its
    /// commands and its place in its queue's order are fixed then, so nothing committed later on
    /// the queue runs before it. Metal submits on its own thread and tells a program of it only
    /// through `Scheduled`; a program that models what Metal does as it
    /// submits takes that point here. A command buffer whose GPU hangs (a draw of primitive type
    /// 9, `hang.rs`) runs them as it becomes `Scheduled` before its last batch, which the hang
    /// holds: once the batch before the hang is taken and Metal's time to read `Scheduled` came.
    ///
    /// - **The thread.** The device's scheduler thread: `commit` returns at once, as Metal's does,
    ///   and submits nothing itself. It runs with no lock of the runtime's held, after the
    ///   submission's call returned. A thread about to block for a command buffer's completion
    ///   submits what still waits in the scheduler's inbox itself, but never a batch of a command
    ///   buffer with a handler of any kind, which stays the scheduler thread's to submit: no
    ///   handler of the program's ever runs on a thread blocking in
    ///   [`wait_until_completed`](Self::wait_until_completed). Behind a GPU hang, the hang's own
    ///   thread where the time came after the batch was taken.
    /// - **Before.** Its handlers run, in the order added, before the status reads `Scheduled`,
    ///   before [`wait_until_scheduled`](Self::wait_until_scheduled) returns, before its
    ///   scheduled handlers run, and before it can complete. The GPU may run the work meanwhile:
    ///   a value of an event its work signals can rise before they ran.
    /// - **Never** for a command buffer that fails before its last batch is taken (a submission
    ///   or host step that fails, a lost device); they are dropped with it. One whose last batch
    ///   was held ran them, and fails from `Scheduled` where the group's submission fails. A
    ///   command buffer committed as failed ([`commit_failed`](Self::commit_failed)) submits its
    ///   batches and runs them.
    /// - **Keep it short.** It holds back its command buffer's scheduling and the device's other
    ///   submissions, among them those of a commit made meanwhile, which the thread takes in only
    ///   after its handlers (`scheduler/inbox.rs`). A wait in it
    ///   for a command buffer, or for a slot of a full queue, that has not happened yet is refused
    ///   (`WaitWouldDeadlock` of [`DeadlockCause::SubmissionHandler`]); it must not wait for GPU
    ///   work otherwise.
    ///
    /// Refused once the command buffer is committed (`Invalid`, `Problem::HandlerAfterCommit`),
    /// as [`add_scheduled_handler`](Self::add_scheduled_handler) is.
    pub fn add_submission_handler(&self, handler: Handler) -> Result<(), Error> {
        let mut state = self.state();
        if state.enqueued() {
            return Err(invalid(
                "add_submission_handler",
                Problem::HandlerAfterCommit,
            ));
        }
        state.submission.push_back(handler);
        Ok(())
    }

    /// `encodeSignalEvent:value:`: once the work encoded before it completed, the event's value
    /// becomes at least `value`. Refused while an encoder is open (Metal asserts), after commit,
    /// after an abandoned encoder, and for another device's event.
    pub fn encode_signal_event(&self, event: EventRef<'_>, value: u64) -> Result<(), Error> {
        self.event_command("encode_signal_event", event, |batches, event| {
            batches.signal(event, value);
        })
    }

    /// `encodeSignalEventScheduled:value:`, private: once the command buffer is scheduled, the
    /// event's value becomes at least `value`, set as [`SharedEvent::set_signaled_value`] sets it:
    /// before its status reads `Scheduled`, before any of its scheduled handlers runs (one added
    /// before this call too) and before [`wait_until_scheduled`](Self::wait_until_scheduled)
    /// returns, as Metal's (an M3 Max, macOS 27.0.1); several in the order encoded. No handler
    /// thread is woken for it: a command buffer whose transition to `Scheduled` has nothing else
    /// to run makes it on the thread that submitted its last batch. One that fails before it is
    /// scheduled sets nothing, as it runs no scheduled handler.
    ///
    /// Called once the command buffer is committed and before it completed, it does nothing, as
    /// Metal's (measured before and after the command buffer was scheduled). Refused once it
    /// completed or failed (`Invalid`, `Problem::ScheduledSignalAfterCompletion`), where Metal's
    /// process ends; while an encoder is open (Metal asserts), after an abandoned encoder, and for another
    /// device's event, as [`encode_signal_event`](Self::encode_signal_event) is. Metal ends the
    /// process for an `MTLEvent`, which this call does not take.
    pub fn encode_signal_event_scheduled(
        &self,
        event: &SharedEvent,
        value: u64,
    ) -> Result<(), Error> {
        const CALL: &str = "encode_signal_event_scheduled";
        let event = EventRef::Shared(event);
        let event = event.shared();
        if !event.device().same(&self.0.device) {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        self.settle_end();
        let mut state = self.state();
        match state.refusal() {
            None => state.scheduled_signals.push((Arc::clone(event), value)),
            Some(Problem::AlreadyCommitted)
                if matches!(
                    state.status,
                    CommandBufferStatus::Completed | CommandBufferStatus::Error
                ) =>
            {
                return Err(invalid(CALL, Problem::ScheduledSignalAfterCompletion))
            }
            Some(Problem::AlreadyCommitted) => {}
            Some(problem) => return Err(invalid(CALL, problem)),
        }
        Ok(())
    }

    /// `encodeWaitForEvent:value:`: the work encoded after it starts once the event's value
    /// reached `value`. Refused as [`CommandBuffer::encode_signal_event`] is.
    pub fn encode_wait_for_event(&self, event: EventRef<'_>, value: u64) -> Result<(), Error> {
        self.event_command("encode_wait_for_event", event, |batches, event| {
            batches.wait(event, value);
        })
        .inspect(|()| self.state().shadows_not_carried())
    }

    /// Wait for the caller's binary semaphore `semaphore` before the work encoded after it: the
    /// acquire of a swapchain image, or a signal of the caller's own submission. It splits the
    /// command buffer's batches as [`encode_wait_for_event`](Self::encode_wait_for_event) does, and
    /// the batch waits in all its stages. The device's scheduler submits the batch in its order:
    /// never before the signals of `semaphore` that command buffers of this device committed before
    /// this one are submitted.
    ///
    /// Refused as [`encode_signal_event`](Self::encode_signal_event) is (an encoder open, after
    /// commit, after an abandoned encoder), and for what the runtime can tell is no binary
    /// semaphore of the device: a null handle, or one of its own timeline semaphores (`Invalid`
    /// with a [`SemaphoreProblem`](super::SemaphoreProblem)).
    ///
    /// # Safety
    ///
    /// - `semaphore` is a binary semaphore of this device, and lives until the command buffer
    ///   completed.
    /// - The signal this wait consumes was submitted to the device before `commit` (by the
    ///   caller: `vkAcquireNextImageKHR`, or its own `vkQueueSubmit`), or is the last one an
    ///   [`encode_signal_vk_semaphore`](Self::encode_signal_vk_semaphore) of a command buffer of
    ///   this device committed before this one encodes. No other wait consumes it.
    pub unsafe fn encode_wait_for_vk_semaphore(
        &self,
        semaphore: vk::Semaphore,
    ) -> Result<(), Error> {
        self.binary_command("encode_wait_for_vk_semaphore", semaphore, |batches| {
            batches.binary_wait(semaphore);
        })
        .inspect(|()| self.state().shadows_not_carried())
    }

    /// Signal the caller's binary semaphore `semaphore` once the work encoded before it
    /// completed: the semaphore a present waits for. It closes the batch as
    /// [`encode_signal_event`](Self::encode_signal_event) does; a wrapped image the batch uses
    /// is back in its home layout before the signal
    /// ([`Device::new_texture_with_vk_image`](super::Device::new_texture_with_vk_image)).
    ///
    /// The signal is submitted with its batch, by the device's scheduler: a wait the caller
    /// submits for it (`vkQueuePresentKHR`, its own `vkQueueSubmit`) comes after the command
    /// buffer is scheduled ([`wait_until_scheduled`](Self::wait_until_scheduled), or a
    /// scheduled handler), which Vulkan requires of a binary semaphore.
    ///
    /// Refused as [`encode_wait_for_vk_semaphore`](Self::encode_wait_for_vk_semaphore) is.
    ///
    /// # Safety
    ///
    /// `semaphore` is a binary semaphore of this device, unsignaled with no signal pending when
    /// the batch runs (every earlier signal of it waited for), and lives until the command buffer
    /// completed.
    pub unsafe fn encode_signal_vk_semaphore(&self, semaphore: vk::Semaphore) -> Result<(), Error> {
        self.binary_command("encode_signal_vk_semaphore", semaphore, |batches| {
            batches.binary_signal(semaphore);
        })
    }

    /// Add a binary semaphore wait or signal to the batches, after the runtime's checks.
    fn binary_command(
        &self,
        call: &'static str,
        semaphore: vk::Semaphore,
        add: impl FnOnce(&mut Batches<Arc<EventShared>, vk::Semaphore>),
    ) -> Result<(), Error> {
        if let Some(problem) = self.0.device.binary_semaphore_problem(semaphore) {
            return Err(invalid(call, Problem::Semaphore(problem)));
        }
        self.settle_end();
        let mut state = self.state();
        if let Some(problem) = state.refusal() {
            return Err(invalid(call, problem));
        }
        add(&mut state.batches);
        Ok(())
    }

    /// Add an event wait or signal to the batches, after the checks Metal makes.
    fn event_command(
        &self,
        call: &'static str,
        event: EventRef<'_>,
        add: impl FnOnce(&mut Batches<Arc<EventShared>, vk::Semaphore>, Arc<EventShared>),
    ) -> Result<(), Error> {
        let event = event.shared();
        if !event.device().same(&self.0.device) {
            return Err(invalid(call, Problem::ForeignObject));
        }
        self.settle_end();
        let mut state = self.state();
        if let Some(problem) = state.refusal() {
            return Err(invalid(call, problem));
        }
        add(&mut state.batches, Arc::clone(event));
        Ok(())
    }

    /// `addCompletedHandler:`: call `handler` once the command buffer completed or failed, on one
    /// of the device's handler threads, at the user-interactive class on macOS as in Metal, and
    /// with a utilization floor on Linux ([`Handler`]). Refused once the command buffer is
    /// committed.
    pub fn add_completed_handler(&self, handler: Handler) -> Result<(), Error> {
        let mut state = self.state();
        if state.enqueued() {
            return Err(invalid(
                "add_completed_handler",
                Problem::HandlerAfterCommit,
            ));
        }
        state.completed.push_back(handler);
        Ok(())
    }

    /// `commit`: hand the batches to the device's scheduler, which submits them in the queue's
    /// commit order, each once the events it waits for are reached. It returns at once, as
    /// Metal's does: the scheduler's thread makes every submission, so no driver call holds it
    /// (`scheduler/inbox.rs`).
    ///
    /// The indirect command buffers' commands the encoders execute are recorded first, from
    /// their records as they are now: each through the plan path, as a dispatch of the encoder
    /// would be. A command such a dispatch would refuse refuses the commit with that error, and
    /// nothing is submitted; a command that runs nothing on Metal (no dispatch, an empty or
    /// oversized threadgroup) is left out.
    ///
    /// Refused (and nothing submitted) when the command buffer was committed already, an
    /// encoder is open, or an encoder was abandoned (`Invalid`). A submission the device fails
    /// completes the command buffer with that error ([`CommandBuffer::error`]), as Metal reports
    /// it on the command buffer. A command buffer that encoded a render pass without attachments
    /// past 16384 a side fails from that pass on, as Metal's does
    /// ([`CommandBufferError::RenderTargetSize`]): the encoders before it run and the event
    /// signals encoded before it fire, and nothing from the pass on runs or signals. One with an
    /// encoder Metal's GPU faults fails as Metal's does ([`CommandBufferError::PageFault`]): that
    /// encoder runs nothing, and the others run and every signal fires.
    ///
    /// Under `RenderLoweringPolicy::Worker`, where the device's lowering thread has not lowered
    /// the last render encoder's end, or a commit of the device waits for one, it makes the
    /// checks above that refuse at the call (committed already, an encoder open, one abandoned)
    /// and returns, its status `Committed`: its own work runs
    /// once the end is lowered, after the commits that wait, on the thread that lowers it
    /// (docs/API.md section 1). A refusal only that work finds (an executed indirect command's,
    /// out of memory) fails the command buffer with nothing of it run
    /// ([`CommandBufferError::CallRefused`] of `commit`, where no call of it was refused before;
    /// [`refusal`](Self::refusal) keeps the error): it completes in its queue's order, its
    /// completed handlers run.
    pub fn commit(&self) -> Result<(), Error> {
        self.commit_as("commit", None)
    }

    /// Commit the command buffer as failed with `failure`: a runtime extension, no Metal call
    /// (docs/API.md, "A command buffer committed as failed"), for layered validation, error
    /// injection and paravirtualization. It fails as Metal's failed command buffers do (measured
    /// by a consumer on an M3 Max, macOS 26.7): the encoders `failure.encoders` names run
    /// nothing, not even a render pass's load, clear or store actions, and every other encoder
    /// runs, before and after them ([`FailedEncoders`]: a GPU hang's suffix, or the set of a
    /// failed submission); its event waits hold it as they hold one committed; its event signals
    /// fire where `failure.signals` says so, and none fires where not (the caller's binary
    /// semaphores are signaled either way); its scheduled handlers run; and it completes in its
    /// queue's order, with status `Error` and [`CommandBufferError::Injected`] of
    /// `failure.code`, which its completed handlers see. A code of 0 completes it with status
    /// `Completed` and no error instead, as Metal completes a command buffer of a queue whose GPU
    /// hang is in flight (measured by the consumer on an M3 Max, 11 hangs at a draw of primitive
    /// type 9: it runs nothing, its event signals fire and its completed handlers see `Completed`
    /// and no error). The indirect command buffers the failing encoders execute are neither
    /// recorded nor checked. [`ended_encoders`](Self::ended_encoders) tells the encoders apart:
    /// each one's kind, its place among the event waits and signals, and what it uses. A command
    /// buffer Metal fails from a render pass on ([`commit`](Self::commit)) fails from there too,
    /// whatever `failure` names, and signals no event after the pass; it completes with
    /// `failure.code`.
    ///
    /// Refused, changing nothing, as [`commit`](Self::commit) is; and for a set naming an
    /// encoder past the ended ones (`Invalid`, `Problem::IndexOutOfRange`, its `limit` the
    /// number ended). (A code of 0 was refused, `Problem::UndefinedValue`.)
    pub fn commit_failed(&self, failure: CommitFailure) -> Result<(), Error> {
        self.commit_as("commit_failed", Some(failure))
    }

    /// [`commit`](Self::commit) for `call`, or, with `failure`,
    /// [`commit_failed`](Self::commit_failed).
    fn commit_as(&self, call: &'static str, failure: Option<CommitFailure>) -> Result<(), Error> {
        // Left to wait for the last render encoder's end, where its call left it to the device's
        // lowering thread and the thread has not lowered it, or for the commits left so.
        if failure.is_none() && self.queue_commit(call)? {
            return Ok(());
        }
        // The last render encoder's end, where its call left it to the device's lowering thread.
        self.settle_end();
        self.commit_settled(call, failure, CommitRun::AtCall)
    }

    /// Under `RenderLoweringPolicy::Worker`, where the command buffer holds a render encoder's
    /// end the device's lowering thread has not lowered, or a commit of the device waits for one
    /// (`render/queued_commits.rs`): make Metal's checks of `commit` at the call, of the command
    /// buffer as the program sees it (its last encoder ended), returning their refusal, and leave
    /// the commit's own work to run after the end, and after the commits that wait. Whether it
    /// did.
    fn queue_commit(&self, call: &'static str) -> Result<bool, Error> {
        let Some(commits) = self.0.device.shared().deferred.commits() else {
            return Ok(false);
        };
        let mut state = self.state();
        let end_in_flight = state.ending.is_some() && !state.end_lowered;
        let queued = commits.lock();
        if !end_in_flight && queued.idle() {
            return Ok(false);
        }
        // The encoder open is the one whose end is in flight, which ended.
        let refusal = match state.refusal() {
            Some(Problem::EncoderOpen) if end_in_flight => None,
            refusal => refusal,
        };
        if let Some(problem) = refusal {
            return Err(invalid(call, problem));
        }
        state.queued_commit = true;
        self.0.status.store(
            status_word(CommandBufferStatus::Committed),
            Ordering::Release,
        );
        let runs = queued.push(self.clone(), !end_in_flight);
        drop(state);
        if runs {
            commits.run(false);
        }
        Ok(true)
    }

    /// Run the commit's own work, which waited for a render encoder's end
    /// (`render/queued_commits.rs`), the end lowered: on the device's lowering thread
    /// (`by_thread`) or the program's. A refusal it finds fails the command buffer
    /// (`State::refuse_commit`), which then commits again with nothing of it run and nothing that
    /// may fail made for it ([`CommitRun::Refused`]): it completes, in its queue's order, with
    /// that error, and its completed handlers run, as every committed command buffer of Metal's
    /// does.
    pub(crate) fn commit_queued(&self, by_thread: bool) {
        let counts = &self.0.device.shared().counts;
        counts.render_log_commit_run(by_thread);
        self.settle_end();
        let Err(error) = self.commit_settled("commit", None, CommitRun::Queued) else {
            return;
        };
        self.state().refuse_commit(error);
        let committed = self.commit_settled("commit", None, CommitRun::Refused);
        debug_assert!(
            committed.is_ok(),
            "a refused commit's commit of nothing failed: {committed:?}"
        );
    }

    /// Before a wait for the command buffer: where its commit's own work waits
    /// (`render/queued_commits.rs`), lower on this thread the ends it and the commits before it
    /// wait for, which runs it here where the device's lowering thread has not.
    fn settle_commit(&self) {
        if !self.state().queued_commit {
            return;
        }
        let Some(commits) = self.0.device.shared().deferred.commits() else {
            return;
        };
        for command_buffer in commits.unready_through(self) {
            command_buffer.settle_end();
        }
    }

    /// The commit, the last render encoder's end lowered, run as `run` says.
    fn commit_settled(
        &self,
        call: &'static str,
        failure: Option<CommitFailure>,
        run: CommitRun,
    ) -> Result<(), Error> {
        // The waiting fetched render targets take their storage form at this commit where it
        // allows (`storage_form.rs`): one atomic read where none waits. The scheduler's thread
        // ends the switch once it submitted the command buffer. None where nothing of it runs.
        let injected = failure.is_some() || run == CommitRun::Refused;
        if !self.take_storage_forms(injected) {
            return self.commit_now(call, failure, run);
        }
        let committed = self.commit_now(call, failure, run);
        if committed.is_err() {
            self.give_up_storage_forms();
        }
        committed
    }

    /// [`commit_settled`](Self::commit_settled) itself.
    fn commit_now(
        &self,
        call: &'static str,
        failure: Option<CommitFailure>,
        run: CommitRun,
    ) -> Result<(), Error> {
        let device = &self.0.device;
        let committed = Instant::now();
        // From here its draws' pipelines still being made are waited for a bound at most: a
        // driver's cold compile can outlast a guest's watchdog (`PipelineWaitPolicy`).
        let policy = device.compile_device().policy();
        if policy.pipeline_waits == PipelineWaitPolicy::Bounded {
            let _ = self
                .0
                .making_deadline
                .set(committed + policy.pipeline_commit_wait);
        }
        // Textures' host shadows hold their images' texels only until the program gives the device
        // GPU work of its own, which is what a commit is: raised here, before anything of this
        // command buffer can run, so a `getBytes` after the commit reads the image and not a
        // shadow this command buffer may write over (`texture/host_shadow.rs`). Raised even where
        // the checks below refuse the commit: the cost of a shadow read again is a submission, and
        // a refusal that left the count behind would be a correctness question.
        device.shared().program_gave_gpu_work();
        let (segments, places, failure, reached, hangs): (
            Vec<(Deferred, bool)>,
            Vec<usize>,
            _,
            _,
            _,
        ) = {
            let mut state = self.state();
            let refusal = match run {
                CommitRun::AtCall => state.refusal(),
                CommitRun::Queued | CommitRun::Refused => state.commit_refusal(),
            };
            if let Some(problem) = refusal {
                return Err(invalid(call, problem));
            }
            let ended = state.ended.len();
            if let Some(problem) = failure.as_ref().and_then(|failure| failure.problem(ended)) {
                return Err(invalid(call, problem));
            }
            // A pass with a draw call over a mutable map lets later passes of none run on Metal's
            // GPU (`rate_mapped_pass.rs`).
            if state.called_over_mutable_map {
                (device.shared().mutable_rate_map_called).store(true, Ordering::Release);
            }
            // Metal's GPU fails a command buffer an encoder of which was made under protection
            // options as a failure given to `commit_failed` fails it, with no signal firing
            // (`protection.rs`); one given such a failure besides is not measured.
            let protected = state.protection.failure();
            let failure = match (failure, &protected) {
                (Some(_), Some(_)) => return Err(Error::AppleOnly(AppleOnly::ProtectedContent)),
                (Some(failure), None) => Some(failure),
                (None, Some((encoders, _))) => Some(CommitFailure {
                    code: 0,
                    encoders: encoders.clone(),
                    signals: false,
                }),
                (None, None) => None,
            };
            // Its last unit of command predication ends, and each step's gates are listed for it.
            if run != CommitRun::Refused {
                let held = &mut *state;
                held.predication.commit(&mut held.uploads, device)?;
            }
            // One committed on its queue while a hang of it is in flight runs nothing, decided
            // here without an event wait and at its first batch's submission with one; its own
            // hang hangs where its encoder runs and its signals fire under what it was committed
            // as (`hang.rs`).
            // Either reads `Scheduled` early, from the hang's on, once its waits are met.
            // One committed on a queue Metal revoked runs nothing and completes at once, its
            // signals dropped (`revocation.rs`).
            let revoked = run != CommitRun::Refused && self.0.queue.errors().revoked();
            let event_waits = state.batches.events().0;
            let discarded = match failure {
                None if run != CommitRun::Refused && !revoked => {
                    hang::discarded(&self.0.queue, event_waits)
                }
                _ => None,
            };
            let joins = discarded.is_some();
            let failure = failure.or(discarded);
            let behind = run != CommitRun::Refused
                && !revoked
                && failure.is_none()
                && event_waits > 0
                && hang::in_flight(&self.0.queue);
            let waits: Vec<(Arc<EventShared>, u64)> = match behind {
                true => (state.batches.all().iter())
                    .flat_map(|batch| batch.waits.iter().cloned())
                    .collect(),
                false => Vec::new(),
            };
            state.behind_hang = match behind {
                true => hang::behind(&self.0.queue, &waits),
                false => None,
            };
            // One committed while a fault's hold of the queue is in flight runs, and reads
            // `Scheduled` early, from the fault's on, once its waits are met.
            let joins_hold = !joins
                && !behind
                && run != CommitRun::Refused
                && !revoked
                && hang::holding(&self.0.queue);
            let waits = match joins_hold {
                true => (state.batches.all().iter())
                    .flat_map(|batch| batch.waits.iter().cloned())
                    .collect(),
                false => waits,
            };
            if joins || joins_hold || state.behind_hang.is_some() {
                hang::join(&self.0, &mut state, waits);
            }
            // Its hold runs where its encoder runs and its signals fire under what it was
            // committed as.
            let hangs = run != CommitRun::Refused
                && !revoked
                && state.hang.as_ref().is_some_and(|hang| {
                    (failure.as_ref())
                        .is_none_or(|f| f.signals && !f.encoders.fails(hang.encoder()))
                });
            if let (true, Some(hang)) = (hangs, &state.hang) {
                let event = Arc::clone(EventRef::Shared(hang.event()).shared());
                state.batches.hold_signal(event, Hang::DONE);
            }
            let mut failure = Failure::of(failure, state.fails_from, ended);
            if let (Some(failure), Some((_, error))) = (&mut failure, protected) {
                failure.error = Some(error);
            }
            let failure = Failure::with_faults(failure, &state.faulted, ended);
            let failure = match revoked {
                true => Some(Failure::revoked()),
                false => failure,
            };
            // The encoders that run: all but those `failure` fails.
            let runs = |encoder: usize| Failure::runs(failure.as_ref(), encoder);
            state.committing = true;
            // Its heaps reach every texture they hold now, for the work recorded below and by its
            // host steps.
            if let Some(residency) = &mut state.residency {
                residency.sync_heaps();
            }
            let (segments, places): (Vec<(Deferred, bool)>, Vec<usize>) = state
                .recorded
                .iter()
                .filter(|recorded| runs(recorded.encoder))
                .flat_map(|recorded| (recorded.segments.iter()).map(|at| (at, recorded.encoder)))
                .map(|(segment, encoder)| ((segment.work.clone(), segment.needs_step()), encoder))
                .unzip();
            if segments.iter().any(|(work, _)| work.reaches_unnamed()) {
                state.reaches_unnamed = true;
            }
            // What of it the live buffers list once used, held weakly: the device's registries'
            // locks, which take no other, inside this one.
            let reached = state.reached(device.shared(), runs);
            // The device's logs of host pages hear of its writes and gated copies in, in commit
            // order, and of the shadows its end left holding their truth (`dirty_pages.rs`).
            // One that waits for an event or a semaphore may run after work committed later,
            // whose writes no commit before it noted: its copies in count on no hold.
            if device.shared().dirty.collects() {
                let held = match (&failure, &state.shadow_carry) {
                    (None, Some(carry)) => carry.held_whole(),
                    _ => Vec::new(),
                };
                let waits = (state.batches.all().iter())
                    .any(|batch| !batch.waits.is_empty() || !batch.binary_waits.is_empty());
                device
                    .shared()
                    .dirty
                    .committed(&lock(&self.0.dirty), &held, waits);
            }
            (segments, places, failure, reached, hangs)
        };
        // What the commands of its encoders that run use is resident from its commit on, as
        // Metal's is: a no-copy buffer or a heap among it is reached unnamed from here, by its
        // own commands too, the work recorded below included (`live_buffers.rs`). Listed without
        // the lock: what it upgrades may be a resource's last hold.
        let used = reached.list();
        device.shared().live_buffers.reach(&used);
        let shared = device.shared();
        (shared.counts).commit_used_host_imports(Buffer::distinct_host_imports(&used.buffers));
        // Listed again at its first submission where the list changed since (a rebind ends a
        // use).
        let listing = Listing::new(device, used);
        // Recorded, or settled for their host steps, with the lock free: their dispatches take it,
        // and announce degradations.
        let prepared: Result<Vec<_>, Error> = segments
            .iter()
            .zip(&places)
            .map(|((work, host), _)| work.prepare(self, *host))
            .collect();
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                self.state().committing = false;
                return Err(error);
            }
        };
        // The first segment a host step records, which the device announces (a loop's check
        // records nothing).
        let announced = segments
            .iter()
            .zip(&prepared)
            .find(|((work, host), prepared)| {
                *host && !prepared.for_good && !matches!(work, Deferred::Loop(_))
            })
            .map(|((work, _), _)| work.clone());
        // A host step may record commands after the commit (a segment not recorded for good).
        let stepped = prepared.iter().any(|prepared| !prepared.for_good);
        // Until the command buffer completes, the buffers its host steps' commands use keep their
        // pages, where a step resolves what it records as it records it, and a rebind of one is
        // refused meanwhile (`RebindProblem::EncodedUses`). Work whose step records only what its
        // calls resolved holds nothing past the commit (`Deferred::resolved_at_its_calls`): a
        // dispatch whose pipelines were being made (`pending_dispatch.rs`), and a render pass left
        // for the pipelines its draws and tile dispatches record
        // (`render/pending_tile.rs`). Metal's replace fails for neither, and a rebind encodes
        // their encoder again instead (`command_buffer/reencode.rs`); where it cannot, their
        // recording records nothing of the encoder (`DeferredPass::record_forms`).
        let steps_hold = (segments.iter().zip(&prepared)).any(|((work, _), prepared)| {
            !prepared.for_good && !work.resolved_at_its_calls(prepared.pipelines)
        });
        // The segments a host step records once the pipelines they record are made.
        let made_later: Vec<Deferred> = segments
            .iter()
            .zip(&prepared)
            .filter(|(_, prepared)| prepared.pipelines)
            .map(|((work, _), _)| work.clone())
            .collect();
        // Every byte the CPU changed in a copied host buffer's pages is in its copy before any of
        // this command buffer's work is submitted (`copied.rs`): copied in with no lock of the
        // command buffer's held, as a copy over logged pages waits for the log's publication.
        shared.copied.copy_in(&shared.counts, shared.dirty.tally());
        let (waits, away, reaches, unreplayable, reset) = {
            let mut state = self.state();
            state.committing = false;
            if let Some(failure) = failure {
                state.fail(failure);
            }
            state.prepare(prepared);
            state.plan_checks();
            // A refused commit's commands run nothing: none of what follows is made for them.
            if run == CommitRun::Refused {
                state.table_snapshots.clear();
            }
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            if run != CommitRun::Refused && state.failures_injected > 0 {
                state.failures_injected -= 1;
                return Err(vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
            }
            // Its root tables search every live buffer of its own on the device, and what its
            // encoders bound and made resident (`live_buffers.rs`).
            if run != CommitRun::Refused {
                state.point_root_tables(device)?;
            }
            // The function-table snapshots read the tables' words as they stand now.
            for snapshot in &state.table_snapshots {
                snapshot.refresh()?;
            }
            state.table_snapshots.clear();
            // Until it completes, an image retired from the device's heap waits for it, where
            // its commands, or its host steps', may read the heap (`heap_sets.rs`). The ticket
            // taken here holds what a rebind or a detach replaces; its first submission takes the
            // one gone textures wait for (`submit.rs`).
            if state.heaps.bound() || stepped {
                let ticket = state.heaps.commit(device);
                state.retained.push(Box::new(ticket));
            }
            state.status = CommandBufferStatus::Committed;
            // Its set-0 cache is shut here, before anything records again: from the commit on it
            // holds what its commands took only weakly (below), so no handle a kept set holds is
            // held against another object taking it -- which is the cache's own precondition
            // (`binding::SetCache`). A rebind frees the import a kept set was written for and
            // records the commands over the new one in the same call (`reencode.rs`), and the new
            // import's `VkBuffer` may take the freed one's handle, all the cache compares; so
            // every command recorded from here -- an encoder encoded again, a host step's --
            // allocates and writes its own set. The commands recorded before the commit keep the
            // sets they bound.
            state.descriptors.shut();
            state.queued_commit = false;
            if let Some(hang) = state.hang.take() {
                match hangs {
                    true => hang.start(&self.0, &mut state, committed),
                    false => hang.release(),
                }
            }
            self.0.answer(&state);
            for heap in state.used_heaps.drain(..) {
                heap.mark_used();
            }
            // A rebind before its first submission of a buffer its replayable encoders, or its
            // work on the GPU alone, named encodes or records them again (`reencode.rs`): noted
            // while their holds still refuse one. So does a reset of a rasterization-rate map
            // their passes are over (`rate_mapped_pass.rs`): noted first, then a reset before the
            // commit found here, marked before the scheduler sees the command buffer.
            let kept = state.keep_replays();
            if !kept.keys.is_empty() || !kept.maps.is_empty() {
                device
                    .shared()
                    .rebinds
                    .wait_for_replay(&self.0, kept.keys, kept.maps);
            }
            let unreplayable = kept.gone;
            let reset = state.rate_maps_reset();
            if reset.0 {
                self.0.mark_reencode();
            }
            if let Some(slot) = &mut state.slot {
                slot.committed();
            }
            self.0.time(&mut state);
            // The last encoder's last command buffer ends, with its batch's end where it may
            // (`submit.rs`), the slot taken.
            self.0.close_tail(&mut state, None);
            // Its submissions record their barriers from its pool, inside sections, where no
            // spare is taken: one that encoded nothing takes a spare now, if one is free.
            self.0.take_spare_pool(&mut state);
            // Until its first submission it holds what its commands took weakly: a rebind or a
            // release meanwhile lets the pages go, and the submission finds them moved. Only once
            // its last command buffer ended: what lay in pages a detach replaced while an encoder
            // held them goes with them (`detach.rs`), and its tail's commands recorded above
            // name the views of an attachment recorded inline (`render/inline_first.rs`).
            state.wait_for_submission(listing);
            // Committed, its commands recorded for good: a rebind of what they used takes
            // effect after them (`rebind.rs`); one a host step records holds until completion
            // (`steps_hold`).
            if steps_hold {
                let held = &mut *state;
                let kept = held
                    .holds
                    .drain(..)
                    .map(|holds| Box::new(holds) as Box<dyn Any + Send + Sync>);
                held.retained.extend(kept);
            } else {
                state.holds.clear();
            }
            let waits = state
                .batches
                .all()
                .iter()
                .enumerate()
                .map(|(index, batch)| BatchSync {
                    waits: batch.waits.clone(),
                    binary_waits: batch.binary_waits.clone(),
                    binary_signals: batch.binary_signals.clone(),
                    // A batch holds segments, or may come to: a rebind before its submission
                    // encodes an encoder of it again from its log, and the encoding again leaves
                    // a segment of its own where a pipeline it records is being made, though the
                    // first encoding left none (`command_buffer/reencode.rs`,
                    // `render/pending_tile.rs`). The scheduler reads this as a gate alone and
                    // asks the command buffer itself whether a step is due
                    // ([`needs_host_step`](Self::needs_host_step)), so a batch that gains none
                    // waits for nothing.
                    segments: state.recorded[state.batch_range(index)]
                        .iter()
                        .any(|encoder| {
                            !encoder.segments.is_empty()
                                || encoder.sparse.iter().any(SparseUpdate::needs_step)
                                || (state.ended.get(encoder.encoder))
                                    .is_some_and(|record| record.replay.is_some())
                        }),
                    checked: state.checks.is_some(),
                })
                .collect();
            // A texture over the caller's image that rests out of `GENERAL`: each batch that
            // uses it moves it.
            let away = device
                .shared()
                .interop
                .wrapped
                .among(state.recorded.iter().flat_map(|r| {
                    let segments = r.segments.iter().filter_map(|s| s.recorded.as_ref());
                    r.summary
                        .keys()
                        .chain(segments.flat_map(|done| done.summary.keys()))
                }))
                .into_iter()
                .find(|image| image.rests_out_of_general());
            // Its commands may reach every texture through its resource ID where one bound a
            // heap set.
            (waits, away, state.heaps.bound(), unreplayable, reset)
        };
        // Each making the batches wait for wakes the scheduler as it finishes, with no lock
        // held (`render/pending.rs`), registered before the scheduler first looks at the batches:
        // a making that finished between that look and a later registration left the batch held
        // with nothing to wake the scheduler (17 lane tests timed out).
        if !made_later.is_empty() {
            let waker = device.shared().scheduler.waker();
            for work in &made_later {
                work.when_made(&waker);
                // Its batch waits for them now: they go before the makings nothing waits for.
                work.hasten(Urgency::Committed);
            }
        }
        drop(unreplayable);
        // The storage-sink degrade's trigger decides, in the program's call, before the
        // scheduler sees the command buffer (`detach.rs`).
        device.reach_storage_sinks(reaches);
        // Its host steps' instances take their numbers from here, in the program's call, before
        // the scheduler can run one.
        let _ = self.0.step_numbers.set(StepNumbers::reserve());
        // The scheduler's lock is taken before a command buffer's, never under it.
        device
            .shared()
            .scheduler
            .commit(&self.0, self.0.queue.id(), waits);
        // Its passes that took a rasterization-rate map since reset, encoded again in this call
        // as a reset's are; its first submission waits for them (marked above). Those it cannot
        // encode again are announced.
        let (again, unencoded) = reset;
        if again {
            self.0.reencode();
        }
        for unlogged in unencoded {
            crate::rate_mapped_pass::degrade_unencoded(device, unlogged);
        }
        if let Some(image) = away {
            device.degrade(Degradation::WrappedImageLayoutTransitions, || {
                image.notice()
            });
        }
        if let Some(work) = announced {
            work.degrade(device);
        }
        Ok(())
    }

    /// `status`. Like Metal's, the read waits for no submission in progress and no GPU work.
    pub fn status(&self) -> CommandBufferStatus {
        status_of(self.0.status.load(Ordering::Acquire))
    }

    /// `error`: why the command buffer failed, once its status is `Error`. Like Metal's, the
    /// read waits for no submission in progress and no GPU work.
    pub fn error(&self) -> Option<CommandBufferError> {
        match self.status() {
            CommandBufferStatus::Error => self.0.error.get().copied(),
            _ => None,
        }
    }

    /// `error.localizedDescription`, where the driver relayed Metal's failure of the command
    /// buffer's work ([`CommandBufferError::Execution`], and the `PageFault` and `DeviceLost` it
    /// maps codes 3 and 11 to): Metal's description, as MoltenVK's message gave it after the
    /// code (`device/relay.rs`), empty where the message had none; and Metal's for
    /// [`CommandBufferError::PredicationOpenAtUnitEnd`], as measured. `None` for every other
    /// failure, and before one.
    pub fn error_description(&self) -> Option<String> {
        self.state().relayed_description.clone()
    }

    /// Why a host step refused the command buffer's commands, once it failed with
    /// [`CommandBufferError::Refused`]: the refusal, typed as [`commit`](Self::commit) types it
    /// where the CPU wrote the records (`Invalid`, `Unsupported`, `NotImplemented` and the
    /// others). Once it failed with [`CommandBufferError::CallRefused`]: the error the render
    /// encoder's call refused as it was lowered would have returned at the call. No Metal
    /// counterpart.
    pub fn refusal(&self) -> Option<Error> {
        self.settle_end();
        self.state().refusal.clone()
    }

    /// Once the command buffer failed with [`CommandBufferError::PipelineCreation`]: the
    /// dispatch whose `VkPipeline` the driver did not make, by its encoder and its order there,
    /// the state and the variant it ran, and the driver's error, which the dispatch returned
    /// where the runtime made the pipeline at the dispatch (`FailedPipeline`). The first such
    /// dispatch, in encoding order: Metal reports one error per command buffer. No Metal
    /// counterpart.
    pub fn failed_pipeline(&self) -> Option<FailedPipeline> {
        self.settle_end();
        self.state().failed_pipeline.clone()
    }

    /// The first draw, in encoding order, whose `VkPipeline` the driver did not make after the
    /// draw returned (the runtime makes a draw's pipeline while the draw goes on,
    /// `render/pending.rs`): the draw ran nothing, the command buffer running on, without error.
    /// Named as [`failed_pipeline`](Self::failed_pipeline) names a dispatch. Counted, each such
    /// draw: `WorkCounts::draws_dropped_unmade`. Under `PipelineWaitPolicy::Bounded` (the
    /// default), also the first draw or dispatch (a compute dispatch, a render pass's tile
    /// dispatch) whose `VkPipeline` was still being made once the command buffer's bound passed,
    /// with the error `VK_TIMEOUT`: `WorkCounts::draws_dropped_still_making` and
    /// `dispatches_dropped_still_making`. No Metal counterpart: Metal makes a state's pipelines as
    /// it makes the state.
    pub fn dropped_draw(&self) -> Option<FailedPipeline> {
        self.settle_end();
        self.state().dropped_draw.clone()
    }

    /// Keep `failed`, a dispatch's pipeline whose making failed, which a host step met
    /// (`pending_dispatch.rs`, `render/pending_tile.rs`), unless one was kept already: the steps
    /// record in encoding order and stop at the first failure, so the one kept is the first.
    pub(super) fn fail_pipeline(&self, failed: FailedPipeline) {
        let mut held = self.state();
        if held.failed_pipeline.is_none() {
            held.failed_pipeline = Some(failed);
        }
    }

    /// Note `failed`, a draw's pipeline whose making failed, whose recording drops the draw
    /// (`render/pending.rs`): kept where it is the first, counted
    /// (`WorkCounts::draws_dropped_unmade`).
    pub(super) fn drop_draw(&self, failed: FailedPipeline) {
        let device = self.device();
        device.shared().counts.draw_dropped_unmade();
        device.degrade(Degradation::DrawDroppedUnmade, || {
            let state = match &failed.state {
                FailedState::Render(state) => format!(
                    "render pipeline state {}{}",
                    state.unique_identifier(),
                    state
                        .label()
                        .map(|label| format!(" `{label}`"))
                        .unwrap_or_default()
                ),
                FailedState::Compute(state) => {
                    format!("compute pipeline state {}", state.unique_identifier())
                }
            };
            let variant = {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                failed.variant.hash(&mut hasher);
                failed.specialization.hash(&mut hasher);
                hasher.finish()
            };
            (
                format!(
                    "draw {} of render encoder {} of a command buffer, of {state}, variant and \
                     specialization {variant:016x}",
                    failed.draw, failed.encoder
                ),
                format!("the driver failed to make its pipeline: {}", failed.error),
            )
        });
        self.keep_dropped(failed);
    }

    /// Note `late`, a draw whose pipeline was still being made once the command buffer's bound
    /// passed ([`making_deadline`](Self::making_deadline)), whose recording drops the draw
    /// (`render/pending.rs`): kept where it is the first, its error `VK_TIMEOUT`, which no
    /// driver's creation returns; counted (`WorkCounts::draws_dropped_still_making`), and
    /// announced (`draw-dropped-still-making`) where its recording runs, in the program's
    /// `commit` or a host step.
    pub(super) fn drop_late_draw(&self, late: FailedPipeline) {
        let device = self.device();
        device.shared().counts.draw_dropped_still_making();
        let (encoder, draw) = (late.encoder, late.draw);
        device.degrade(Degradation::DrawDroppedStillMaking, || {
            (
                format!("draw {draw} of render encoder {encoder} of a command buffer"),
                format!(
                    "its pipeline was still being made {} ms after the command buffer's commit",
                    device
                        .compile_device()
                        .policy()
                        .pipeline_commit_wait
                        .as_millis()
                ),
            )
        });
        self.keep_dropped(late);
    }

    /// Note `late`, a dispatch whose pipeline was still being made once the command buffer's
    /// bound passed ([`making_deadline`](Self::making_deadline)): a compute encoder's
    /// (`pending_dispatch.rs`), or a render pass's tile dispatch where `tile`
    /// (`render/pending_tile.rs`), whose recording records no dispatch. Kept as
    /// [`drop_late_draw`](Self::drop_late_draw) keeps a draw, its error `VK_TIMEOUT`; counted
    /// (`WorkCounts::dispatches_dropped_still_making`), and announced
    /// (`dispatch-dropped-still-making`) where its recording runs, in the program's `commit` or a
    /// host step.
    pub(crate) fn drop_late_dispatch(&self, late: FailedPipeline, tile: bool) {
        let device = self.device();
        device.shared().counts.dispatch_dropped_still_making();
        let (encoder, dispatch) = (late.encoder, late.draw);
        device.degrade(Degradation::DispatchDroppedStillMaking, || {
            let what = match tile {
                true => "tile dispatch",
                false => "dispatch",
            };
            let of = match tile {
                true => "render encoder",
                false => "compute encoder",
            };
            (
                format!("{what} {dispatch} of {of} {encoder} of a command buffer"),
                format!(
                    "its pipeline was still being made {} ms after the command buffer's commit",
                    device
                        .compile_device()
                        .policy()
                        .pipeline_commit_wait
                        .as_millis()
                ),
            )
        });
        self.keep_dropped(late);
    }

    /// Keep `dropped` as the command buffer's first draw or dispatch that ran nothing, unless one
    /// was kept already: the recordings run in encoding order.
    fn keep_dropped(&self, dropped: FailedPipeline) {
        let mut held = self.state();
        if held.dropped_draw.is_none() {
            held.dropped_draw = Some(dropped);
        }
    }

    /// Until when its recordings wait for its draws' and dispatches' pipelines still being made:
    /// set at its commit under `PipelineWaitPolicy::Bounded`, none before it or under `Exact`.
    pub(crate) fn making_deadline(&self) -> Option<Instant> {
        self.0.making_deadline.get().copied()
    }

    /// What the command buffer's execution status words report, once it completed: resident-set
    /// misses of argument-buffer pointers, and the other conditions [`StatusReport`] names. They
    /// are facts of the run, not errors; `None` before completion and when the command buffer
    /// failed. No Metal counterpart.
    pub fn execution_report(&self) -> Option<StatusReport> {
        self.state().report
    }

    /// What the ended encoders' dispatches and draws executed, in encoding order (`execution.rs`):
    /// validation's executor reads it; no Metal call answers it. Empty unless the device was made
    /// with [`DeviceOptions::record_executions`](crate::DeviceOptions::record_executions): its
    /// encoders build no record otherwise.
    #[doc(hidden)]
    pub fn executions(&self) -> Vec<Execution> {
        self.settle_end();
        self.state().executions.clone()
    }

    /// `GPUStartTime`: the host time, in seconds of [`host_time`](super::host_time)'s clock, at
    /// which the GPU began the command buffer's work, once it completed; 0 before, and when it
    /// failed (`gpu_time.rs`).
    ///
    /// On a device with calibrated timestamps (`VK_KHR_calibrated_timestamps` or
    /// `VK_EXT_calibrated_timestamps`, enabled where listed) it is Metal's answer, within the
    /// calibration's deviation. Without, it is the end below less the GPU's duration, which
    /// stays exact, and the device announces `Degradation::UncalibratedGpuTimes` once; so is it
    /// where the calibration would place the times outside the command buffer's life, from its
    /// first submission to its completion, by more than its deviation and the clocks' drift (a
    /// calibration that disagrees with the device's own timestamps, `gpu_time.rs`). A device
    /// whose queue family writes no timestamps refuses it (`Unsupported`, `Feature::GpuTimes`).
    /// So does a command buffer committed while others held every slot of the device's
    /// timestamp pools and no pool more could be made (2048 on MoltenVK; elsewhere only where
    /// the device refused a pool, `gpu_time.rs`), and the device announces
    /// `Degradation::UntimedCommandBuffer` once.
    ///
    /// A command buffer that encoded nothing and waits for nothing writes no timestamps: its GPU
    /// start and end are where its queue reached it, its kernel end time or the GPU end of the
    /// queue's command buffer before it where later, a zero duration, as Metal answers for one.
    /// On a device made with `gpu_times_from_host` ([`DeviceOptions`](crate::DeviceOptions),
    /// [`RawDevice`](crate::RawDevice)), and on a driver whose timestamps wait for every
    /// queue's (MoltenVK 1.4.1 on Apple's GPUs), none writes them: the start is the kernel end
    /// time (or the queue's latest GPU end before it, where later) and the end the host time the
    /// runtime saw it complete, and the device announces `Degradation::GpuTimesFromHost` once.
    pub fn gpu_start_time(&self) -> Result<f64, Error> {
        self.gpu_times().map(|(start, _)| gpu_time::seconds(start))
    }

    /// `GPUEndTime`: the host time at which the GPU finished the command buffer's work, as
    /// [`gpu_start_time`](Self::gpu_start_time) says. Without calibrated timestamps, it is the
    /// host time at which the runtime saw the command buffer complete: late by that latency.
    pub fn gpu_end_time(&self) -> Result<f64, Error> {
        self.gpu_times().map(|(_, end)| gpu_time::seconds(end))
    }

    /// `kernelStartTime`: the host time, in seconds of [`host_time`](super::host_time)'s clock,
    /// at which the device's scheduler began submitting the command buffer's work to Vulkan; 0
    /// before. Like Metal's, the read waits for no submission in progress and no GPU work.
    pub fn kernel_start_time(&self) -> f64 {
        gpu_time::seconds(self.0.kernel.start())
    }

    /// `kernelEndTime`: the host time at which the device's scheduler finished submitting the
    /// command buffer's work to Vulkan, or held its last batch for one kernel submission with
    /// later ones (`CompletionPolicy::Grouped`); it is then `Scheduled`; 0 before. Like Metal's,
    /// the read waits for no submission in progress and no GPU work.
    pub fn kernel_end_time(&self) -> f64 {
        gpu_time::seconds(self.0.kernel.end())
    }

    /// The GPU start and end, in nanoseconds of the host's clock: placed from the command
    /// buffer's timestamps once it completed, then kept; 0 and 0 before, when it failed, or when
    /// the GPU wrote none. One that wrote no timestamps answers the times placed as it completed
    /// (`place_gpu_times`, `gpu_time.rs`). Refused where the device writes no timestamps (and
    /// does not answer from the host), and for a command buffer that found no slot of the
    /// device's timestamp pools free at its commit, and no pool more.
    fn gpu_times(&self) -> Result<(u64, u64), Error> {
        let device = &self.0.device;
        let shared = device.shared();
        let untimed = || Error::Unsupported(Unsupported::not_representable(Feature::GpuTimes));
        let clock = shared.timing.clock();
        if clock.is_none() && !shared.timing.answers_from_host() {
            return Err(untimed());
        }
        let mut state = self.state();
        if let Some(times) = state.times.gpu {
            return Ok(times);
        }
        if state.status != CommandBufferStatus::Completed {
            return Ok((0, 0));
        }
        let (start, end) = match state.times.stamped {
            // Placed as it completed (`place_gpu_times`).
            Stamped::NoWork => return Ok(state.times.placed.unwrap_or((0, 0))),
            Stamped::FromHost => {
                let Some(times) = state.times.placed else {
                    return Ok((0, 0));
                };
                state.times.gpu = Some(times);
                drop(state);
                let because = shared
                    .timing
                    .host_clock_reason()
                    .map_or("", |from| from.because());
                device.degrade(Degradation::GpuTimesFromHost, || {
                    (
                        "a command buffer's GPU times".to_owned(),
                        because.to_owned(),
                    )
                });
                return Ok(times);
            }
            Stamped::Read(Some(ticks)) => ticks,
            Stamped::Read(None) | Stamped::Unread => return Ok((0, 0)),
            Stamped::NoSlot => {
                drop(state);
                let slots = shared.timing.slots();
                device.degrade(Degradation::UntimedCommandBuffer, || {
                    let why = if slots == 0 {
                        "the device could make no timestamp pool at its commit".to_owned()
                    } else {
                        format!(
                            "every one of the device's {slots} timestamp slots was held by a \
                             command buffer in flight at its commit, and it could make no pool \
                             more"
                        )
                    };
                    ("a command buffer's GPU times".to_owned(), why)
                });
                return Err(untimed());
            }
            Stamped::Failed(error) => return Err(device.failed(error)),
        };
        // Read timestamps came from the device's clock.
        let clock = clock.ok_or_else(untimed)?;
        let (times, placement) = clock.span(
            state.times.pair,
            start,
            end,
            life(&self.0.kernel, &state.times),
        );
        state.times.gpu = Some(times);
        drop(state);
        let reason = match placement {
            Placement::Calibrated => return Ok(times),
            Placement::Uncalibrated if clock.calibration.is_some() => {
                "the device's calibrated timestamps could not be read as it completed".to_owned()
            }
            Placement::Uncalibrated => {
                "the device calibrates its timestamps against no host clock the runtime reads"
                    .to_owned()
            }
            Placement::Disagreed { by, deviation } => format!(
                "the device's calibration placed them {by} ns outside the command buffer's life, \
                 between its first submission and its completion, where it stated a deviation of \
                 {deviation} ns: its timestamps count on another clock than its calibrated one"
            ),
        };
        device.degrade(Degradation::UncalibratedGpuTimes, || {
            (
                "a command buffer's GPU times".to_owned(),
                format!(
                    "{reason} ({}-bit timestamps of {} ns)",
                    clock.valid_bits, clock.period
                ),
            )
        });
        Ok(times)
    }

    /// `waitUntilScheduled`: block until the command buffer's scheduled handlers ran, or it
    /// failed. As Metal's (measured on an M4, macOS 27.0.1), it returns only
    /// once every scheduled handler returned, though [`status`](Self::status) reads `Scheduled`
    /// just before they start.
    ///
    /// As [`wait_until_completed`](Self::wait_until_completed) does, it submits for itself what
    /// still waits in the scheduler's inbox before it sleeps, where no handler of the program's
    /// would run on this thread for it (`Scheduler::submit_before_waiting`). It sends no group
    /// held: a batch held for one kernel submission makes its command buffer `Scheduled` as it is
    /// held, so this wait is already answered by the hold.
    ///
    /// From a handler or a release on the device's handler threads it answers as Metal's does
    /// where measured: once the command buffer's transition to `Scheduled` ran, it returns. Its own
    /// scheduled handlers' wait, and a wait for a later command buffer of the handler's queue, are
    /// refused (`WaitWouldDeadlock` of [`DeadlockCause::OwnQueue`]): that transition comes after
    /// the handler in the queue's order, and Metal's never returns there. A wait for another
    /// queue's command buffer waits, unless it would deadlock (as
    /// [`CommandBuffer::wait_until_completed`] says). From a submission handler
    /// ([`add_submission_handler`](Self::add_submission_handler)), a wait for one not yet scheduled
    /// is refused ([`DeadlockCause::SubmissionHandler`]). Once the process is exiting, it answers
    /// as that one does.
    pub fn wait_until_scheduled(&self) -> Result<(), Error> {
        let scheduled_now = scheduled(&self.state());
        let _blocked = match scheduled_wait(running(), self.0.target(), scheduled_now) {
            Answer::Refuse(cause) => return Err(cause.into()),
            Answer::Return | Answer::Wait => None,
            Answer::Block(from) => Some(contexts::block(from, self.0.context())?),
        };
        self.settle_commit();
        self.0.waits_for_scheduling();
        let mut state = self.state();
        while !scheduled(&state) {
            if exit_guard::closed() {
                return Err(DeadlockCause::Exiting.into());
            }
            state.parked += 1;
            state = self
                .0
                .changed
                .wait_timeout(state, exit_guard::WAKE)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
            state.parked -= 1;
        }
        Ok(())
    }

    /// `waitUntilCompleted`: block until the command buffer completed or failed and its
    /// completed handlers ran. It returns on device loss (the status is then `Error`). Where a
    /// batch of it is held for one kernel submission with later ones while its queue's earlier
    /// work runs (`CompletionPolicy::Grouped`), the wait sends it at once.
    ///
    /// It returns only once the runtime's threads let go of the command buffer, and of what they
    /// held to make what its work waited for (a pipeline made on the device's pool, a texture's
    /// CPU decode, `trails.rs`), so dropping the last handles after it destroys the device on the
    /// dropping thread, before the drop returns. Such a thread lets go a few drops after its
    /// signal, so this adds to the wait only where it lost its CPU in between; the completion,
    /// the status and the completed handlers do not wait for it. A program that learns of the
    /// completion otherwise (a completed handler, an event the command buffer signals, its
    /// status) may drop them while a runtime thread still holds the command buffer; the device
    /// then goes on that thread. An exit meanwhile waits for that teardown, or leaves the device
    /// to the process's end.
    ///
    /// Once the process is exiting (its exit handlers run), the runtime's threads that finish
    /// command buffers have ended, and no Vulkan call is made: a wait for a command buffer that
    /// did not finish answers `WaitWouldDeadlock` of [`DeadlockCause::Exiting`] at once, on any
    /// thread, and one already blocked within 50 ms, so that nothing the exit runs waits for
    /// those threads ([`process_exiting`](crate::process_exiting) answers whether it is so).
    ///
    /// From a handler or a release on the device's handler threads it answers as Metal's does
    /// where measured:
    /// - a deallocator that runs as this command buffer releases what it retained gets `Ok` at
    ///   once (the status is `Completed` or `Error`);
    /// - a handler's wait for its own command buffer, or for a later one of its queue, is
    ///   refused (`WaitWouldDeadlock` of [`DeadlockCause::OwnQueue`]): that transition comes
    ///   after the handler in the queue's order, and Metal's never returns there;
    /// - a wait for another queue's command buffer blocks the handler's thread until it
    ///   completed, as Metal's does. It is refused (`WaitWouldDeadlock`) only where it would
    ///   deadlock: when that queue's handlers wait, through the queues they wait for, for this
    ///   one ([`DeadlockCause::Cycle`]), or when it would leave the device's handler threads
    ///   with none that is not blocked so ([`DeadlockCause::HandlerThreadCap`]: the 512th, where
    ///   Metal's handler threads run out and it hangs).
    pub fn wait_until_completed(&self) -> Result<(), Error> {
        let _blocked = match self.completed_wait() {
            Answer::Return => {
                return self
                    .trailed(None)
                    .map_err(|_| DeadlockCause::Exiting.into())
            }
            Answer::Refuse(cause) => return Err(cause.into()),
            Answer::Wait => None,
            Answer::Block(from) => Some(contexts::block(from, self.0.context())?),
        };
        self.settle_commit();
        self.0.waits_for_completion();
        let waited = self.0.gate.wait()?;
        self.0.device.shared().counts.gate_waited(waited);
        self.trailed(None)
            .map_err(|_| DeadlockCause::Exiting.into())
    }

    /// [`CommandBuffer::wait_until_completed`] for at most `timeout`, which bounds the whole call;
    /// a timeout past what the host's clock holds waits until the completion comes. Where that is
    /// refused, it ends at once with `TimedOut`, as it does once the process is exiting.
    pub fn wait_until_completed_for(&self, timeout: Duration) -> Result<(), TimedOut> {
        let deadline = Instant::now().checked_add(timeout);
        let _blocked = match self.completed_wait() {
            Answer::Return => return self.trailed(deadline).map_err(|_| TimedOut),
            Answer::Refuse(_) => return Err(TimedOut),
            Answer::Wait => None,
            Answer::Block(from) => {
                Some(contexts::block(from, self.0.context()).map_err(|_| TimedOut)?)
            }
        };
        self.settle_commit();
        self.0.waits_for_completion();
        let waited = match deadline {
            Some(deadline) => self.0.gate.wait_until(deadline)?,
            // Only an exiting process ends a wait with no deadline: that wait's `TimedOut`.
            None => self.0.gate.wait().map_err(|_| TimedOut)?,
        };
        self.0.device.shared().counts.gate_waited(waited);
        self.trailed(deadline).map_err(|_| TimedOut)
    }

    /// Once its gate opened, wait until the trails opened before it completed closed
    /// (`trails.rs`): what the runtime's threads held for the work it waited for is let go. At
    /// most until `deadline` where one is given. A wait answered before the gate opened (a
    /// deallocator's, as it releases what it retained) waits for none.
    fn trailed(&self, deadline: Option<Instant>) -> Result<(), crate::trails::Ended> {
        if !self.0.gate.is_open() {
            return Ok(());
        }
        let mark = self.state().trail_mark;
        let shared = self.0.device.shared();
        let waited = shared.trails.wait_below(mark, deadline)?;
        shared.counts.trail_waited(waited);
        Ok(())
    }

    /// What a wait for completion answers on this thread.
    fn completed_wait(&self) -> Answer {
        completed_wait(running(), self.0.target(), self.0.gate.is_open())
    }
}

impl CommandBufferShared {
    /// The device.
    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    /// This command buffer's address, which names it to [`own`]'s marker.
    fn address(&self) -> usize {
        self as *const CommandBufferShared as usize
    }

    /// Its queue's serial context.
    fn context(&self) -> Context {
        self.queue.context()
    }

    /// This command buffer as the target of a wait.
    fn target(&self) -> Target {
        Target {
            context: self.context(),
            buffer: self.address(),
        }
    }

    /// The marker of `step` of this command buffer.
    fn marker(&self, step: Step) -> Marker {
        Marker {
            context: self.context(),
            buffer: self.address(),
            step,
        }
    }

    /// A reference for a runtime thread: the command buffer's waiters wait until it is gone.
    pub(crate) fn hold(self: &Arc<Self>) -> Hold<CommandBufferShared> {
        Hold::new(self, &self.gate)
    }

    /// Whether batch `index` waits for a host step: a segment of it executes commands the GPU may
    /// write, not recorded since, or an indirect sparse update of it reads regions the GPU may
    /// write, not read since.
    pub(crate) fn needs_host_step(&self, index: usize) -> bool {
        let state = lock(&self.state);
        // After a failed step the batch goes to its submission, which fails.
        if state.step_failure.is_some() {
            return false;
        }
        let range = state.batch_range(index);
        let recorded = &state.recorded[range];
        recorded
            .iter()
            .flat_map(|encoder| &encoder.segments)
            .any(Segment::needs_step)
            || recorded
                .iter()
                .flat_map(|encoder| &encoder.sparse)
                .any(SparseUpdate::needs_step)
    }

    /// What the host step of batch `index` waits for ([`needs_host_step`](Self::needs_host_step)):
    /// the work submitted before it, where the GPU may write what a segment or an indirect
    /// sparse update of it reads, its own `VkQueue`'s alone where those are loops' checks; and the
    /// pipelines its segments record, while one is being made, and what a loop's check runs again.
    pub(crate) fn step_waits(&self, index: usize) -> StepWaits {
        let state = lock(&self.state);
        let recorded = &state.recorded[state.batch_range(index)];
        let segments = || recorded.iter().flat_map(|encoder| &encoder.segments);
        let sparse = recorded
            .iter()
            .flat_map(|encoder| &encoder.sparse)
            .any(SparseUpdate::needs_step);
        // A loop's check reads only what its laying, on its own `VkQueue`, wrote (`rerun.rs`).
        let checks_alone = !sparse
            && segments()
                .filter(|segment| segment.steps_after_the_gpu())
                .all(|segment| matches!(segment.work, Deferred::Loop(_)));
        // Past its bound its step records the draws and dispatches whose pipelines are still
        // being made as nothing (`render/pending.rs`, `pending_dispatch.rs`), waiting for no
        // making.
        let deadline = self.making_deadline.get().copied();
        let late = deadline.is_some_and(|at| Instant::now() >= at);
        let pipelines = match late {
            false => segments().any(Segment::steps_after_pipelines),
            true => segments().any(Segment::holds),
        };
        StepWaits {
            gpu: segments().any(Segment::steps_after_the_gpu) || sparse,
            queue_alone: checks_alone.then(|| self.queue.vk_queue()),
            pipelines,
            until: deadline.filter(|_| pipelines && !late),
        }
    }

    /// Raise the makings of the pipelines batch `index`'s host step records, still being made,
    /// to a queue's first held batch's (`budget::Urgency::Head`): the scheduler found the batch
    /// first in its queue's line, every batch after it there waiting behind it. The makings are
    /// read under the command buffer's lock and hastened with it released, as `commit` hastens
    /// them.
    pub(crate) fn hasten_head(&self, index: usize) {
        let works: Vec<Deferred> = {
            let state = lock(&self.state);
            state.recorded[state.batch_range(index)]
                .iter()
                .flat_map(|encoder| &encoder.segments)
                .filter(|segment| segment.steps_after_pipelines())
                .map(|segment| segment.work.clone())
                .collect()
        };
        for work in &works {
            work.hasten(Urgency::Head);
        }
    }

    /// What batch `index` uses, for the scheduler's order across queues (`scheduler/order.rs`):
    /// its encoders' summaries and its segments' recordings, and for a segment its host step
    /// records for the pipelines it waits for alone, what its work uses, known at encode time
    /// (`Segment::known_summary`): Metal makes the pipeline at `newRenderPipelineState`, so only a
    /// batch that shares a resource with it waits for it. `None` while some of it is not known: a
    /// segment its host step records (or records again) for what the GPU may write, one its
    /// submission records (a scrub), or a resource state encoder's update, whose uses its
    /// submission adds. A loop's check adds what its laying's parts use, which it may run again.
    pub(crate) fn footprint(&self, index: usize) -> Option<Footprint<ResourceKey>> {
        let state = lock(&self.state);
        let range = state.batch_range(index);
        let recorded = &state.recorded[range.clone()];
        let mut summaries = Vec::new();
        let mut known = Vec::new();
        for (at, encoder) in recorded.iter().enumerate() {
            if !encoder.sparse.is_empty() {
                return None;
            }
            summaries.push(&encoder.summary);
            for segment in &encoder.segments {
                // A loop's check may run its laying again: what the laying's parts use (`rerun.rs`).
                if let Deferred::Loop(check) = &segment.work {
                    let laying = state.recorded.get(check.from()..range.start + at);
                    summaries.extend(laying.into_iter().flatten().map(|part| &part.summary));
                    continue;
                }
                match &segment.recorded {
                    // One recorded again on the GPU alone uses what its recording does; one a
                    // route outgrew, what its host step finds.
                    Some(recording)
                        if !segment.needs_step()
                            || (segment.rerecords() && !segment.work.outgrown()) =>
                    {
                        summaries.push(&recording.summary)
                    }
                    _ => known.push(segment.known_summary(&self.device)?),
                }
            }
        }
        Some(Footprint::of(summaries.into_iter().chain(&known)))
    }

    /// The host step of batch `index` (`deferred.rs`): on the scheduler's thread, without its lock,
    /// inside a section of the exit guard, once every batch submitted before it completed. Each
    /// segment of the batch that needs it is recorded from what its work reads as it stands now and
    /// what `commit` left for the step (`Deferred::step`); the recording `commit` made of one, if
    /// any, is dropped. A failure is kept for the batch's submission, which fails the command
    /// buffer with it.
    pub(crate) fn host_step(self: &Arc<Self>, index: usize) {
        // A rebind's encoding again of its encoders records into its pool too: one at a time.
        let _turn = lock(&self.reencoding);
        // Its instances numbered from its commit's place among the program's calls.
        let step = || exit_guard::host_step(|| self.record_host_step(index));
        match self.step_numbers.get() {
            Some(numbers) => numbers.numbering(step),
            None => step(),
        }
    }

    /// [`host_step`](Self::host_step)'s recording, marked: every notice it announces, those
    /// asked for at every occurrence included, is announced here, on this thread, before the
    /// batch is submitted, completes or is scheduled.
    fn record_host_step(self: &Arc<Self>, index: usize) {
        let device = &self.device;
        let steps: Vec<_> = {
            let state = lock(&self.state);
            let range = state.batch_range(index);
            let first = range.start;
            state.recorded[range]
                .iter()
                .enumerate()
                .flat_map(|(encoder, recorded)| {
                    recorded
                        .segments
                        .iter()
                        .enumerate()
                        .filter(|(_, segment)| segment.needs_step())
                        .map(move |(at, segment)| {
                            (
                                (first + encoder, at),
                                segment.work.clone(),
                                segment.inputs.clone(),
                            )
                        })
                })
                .collect()
        };
        self.read_mappings(index);
        if steps.is_empty() {
            return;
        }
        // The first segment a host step records for what the GPU may write, which the device
        // announces: a recording on the GPU alone made again announces nothing
        // (`gpu_execute::Rerecord`), unless a route outgrew it (`Deferred::outgrown`), which the
        // step records from the records; nor does work recorded again for the CPU's writes since
        // the commit alone (`Deferred::rewritten`), or for its pipelines alone; nor a loop's check,
        // which records nothing.
        let announced = steps
            .iter()
            .find(|(_, work, inputs)| {
                work.gpu_written()
                    && (!matches!(inputs, Some(StepInputs::Rerecord(_))) || work.outgrown())
                    && !matches!(inputs, Some(StepInputs::Loop))
            })
            .map(|(_, work, _)| work.clone());
        let buffer = CommandBuffer(Arc::clone(self));
        for ((encoder, at), work, inputs) in steps {
            let Some(inputs) = inputs else {
                let error = internal("a committed segment without what its host step needs");
                lock(&self.state).step_failure = Some(recording_error(device, error));
                break;
            };
            let recorded = work.step(&buffer, &inputs);
            let mut state = lock(&self.state);
            match recorded {
                Ok(done) => {
                    let segment = &mut state.recorded[encoder].segments[at];
                    segment.stepped = true;
                    // Its recording is the step's now, which a rebind does not record again.
                    segment.gpu_alone = false;
                    let before = segment.recorded.replace(done);
                    if let Some(mut before) = before {
                        state.retained.extend(before.retained.into_entry());
                        before.taken.retain_into(&mut state.retained);
                    }
                }
                Err(error) => {
                    // A draw's pipeline whose making failed fails the command buffer so.
                    state.step_failure = Some(match &state.failed_pipeline {
                        Some(failed) => CommandBufferError::PipelineCreation(
                            PipelineCreationCause::of(&failed.error),
                        ),
                        None => step_error(device, error, &mut state.refusal),
                    });
                    break;
                }
            }
            drop(state);
            // A loop's check that runs its laying again holds the batch: the segments after it
            // read what the laying leaves, and step once it ends (`rerun.rs`).
            let holds = work.holds();
            // Dropped without the lock: what it holds may be a state's last handle.
            drop((work, inputs));
            if holds {
                break;
            }
        }
        drop(buffer);
        if let Some(announced) = announced {
            announced.degrade(device);
        }
    }

    /// The host step's reading of batch `index`'s indirect sparse updates (`sparse/indirect.rs`),
    /// before its segments' steps: each is replaced, in its place, by the updates its arguments
    /// state now, and held until the command buffer completes. A refusal is kept for the batch's
    /// submission, which fails the command buffer with it, and the rest stay unread.
    fn read_mappings(self: &Arc<Self>, index: usize) {
        let device = &self.device;
        let unread: Vec<(usize, usize, SparseUpdate)> = {
            let state = lock(&self.state);
            if state.step_failure.is_some() {
                return;
            }
            let range = state.batch_range(index);
            let first = range.start;
            state.recorded[range]
                .iter()
                .enumerate()
                .flat_map(|(encoder, recorded)| {
                    recorded
                        .sparse
                        .iter()
                        .enumerate()
                        .filter(|(_, update)| update.needs_step())
                        .map(move |(at, update)| (first + encoder, at, update.clone()))
                })
                .collect()
        };
        if unread.is_empty() {
            return;
        }
        // Read without the lock, the last of an encoder first, so the places of those before
        // it stay where they were.
        for (encoder, at, update) in unread.into_iter().rev() {
            let SparseUpdate::Indirect {
                texture,
                mode,
                buffer,
                offset,
            } = &update
            else {
                continue;
            };
            let read = indirect::resolve(device, texture, *mode, buffer, *offset);
            let mut state = lock(&self.state);
            match read {
                Ok(updates) => {
                    // The update read goes to `retained`: its texture and buffer are held until
                    // the command buffer completes, as Metal's command buffer holds what its
                    // encoders were given, though its arguments map nothing; and it is released
                    // outside this lock (the buffer's last release may run a deallocator).
                    let read: Vec<SparseUpdate> = state.recorded[encoder]
                        .sparse
                        .splice(at..at + 1, updates)
                        .collect();
                    state
                        .retained
                        .extend(read.into_iter().map(|update| Box::new(update) as Box<_>));
                }
                Err(error) => {
                    state.step_failure = Some(recording_error(device, error));
                    break;
                }
            }
        }
        indirect::degrade(device);
    }

    /// A primary `VkCommandBuffer` from this command buffer's pool, made on first use. The
    /// caller holds `state` (the pool is externally synchronized).
    fn allocate_raw(&self, state: &mut State) -> Result<vk::CommandBuffer, Error> {
        self.allocate(state, vk::CommandBufferLevel::PRIMARY)
    }

    /// End `state`'s tail, if it has one ([`State::tail`]). Where its batch ended (before part
    /// `joined`, which just ended, or at the commit, `joined` `None`) and may end there
    /// ([`State::may_fold`]), the batch's end goes into it first, as `submit.rs` would record it
    /// in a command buffer of its own: the barrier to the host, then, at the commit of a timed
    /// command buffer whose last batch it ends, the end timestamp. Elsewhere it ends as it is.
    /// A failure to end it abandons the command buffer as an encoder's does (`encoder.rs`); at
    /// the commit, it fails the command buffer's first submission, before any batch goes to the
    /// queue.
    fn close_tail(&self, state: &mut State, joined: Option<usize>) {
        let Some(raw) = state.tail.take() else {
            return;
        };
        let part = joined.unwrap_or(state.recorded.len()).saturating_sub(1);
        let batch = state.batch_of(part);
        let ended = joined.is_none_or(|joined| state.batch_of(joined) != batch);
        let fold = ended && state.may_fold(&self.device, batch);
        let last = joined.is_none() && batch + 1 == state.batches.all().len();
        let stamp = match (fold && last, &state.times.slot) {
            (true, Some(slot)) => Some((slot.queries(), slot.first())),
            _ => None,
        };
        let device = self.device.shared().vk();
        if fold {
            // SAFETY: the tail is recording, outside any render pass (its encoder's passes began
            // and ended in its own recordings), and only this call records it, under the state
            // lock; the slot's queries were reset by the batch's start, before it in submission
            // order (`submit.rs`).
            unsafe {
                crate::batch_order::end(self.device.shared().by_ends).record(device, raw);
                if let Some((queries, first)) = stamp {
                    let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                    device.cmd_write_timestamp(raw, bottom, queries, first + 1);
                }
            }
        }
        // SAFETY: as above.
        match unsafe { self.end_raw(raw) } {
            Ok(()) if fold => state.folds.push(Fold {
                batch,
                raw,
                stamped: stamp.is_some(),
            }),
            Ok(()) => {}
            Err(_) if joined.is_some() => state.abandoned = true,
            Err(error) => {
                let error = recording_error(&self.device, vulkan(error));
                state.step_failure.get_or_insert(error);
            }
        }
    }

    /// End `raw`, a `VkCommandBuffer` of its pool, counted where it ended: its pool goes back to
    /// the device's spares unreset only where every one the pool handed out ended (`spares.rs`).
    ///
    /// # Safety
    /// As [`CommandBuffer::end_raw`].
    unsafe fn end_raw(&self, raw: vk::CommandBuffer) -> Result<(), vk::Result> {
        // SAFETY: the caller's.
        let ended = unsafe { self.device.shared().vk().end_command_buffer(raw) };
        if ended.is_ok() {
            self.raws_ended.fetch_add(1, Ordering::Relaxed);
        }
        ended
    }

    /// Give a command buffer without a pool yet (one that encoded nothing) a spare pool of the
    /// device's where one is free, on the committing thread: its submissions, which record its
    /// barriers inside sections where no spare is taken, then make no `VkCommandPool` of their
    /// own (one made and destroyed per such command buffer before). Where none is free, or the
    /// spare's reset fails, the submission makes one as before and answers any failure then.
    fn take_spare_pool(&self, state: &mut State) {
        if state.pool.is_some() {
            return;
        }
        let shared = self.device.shared();
        let family = shared.queue_family_of(self.queue.vk_queue());
        if let Ok(Some(spare)) = shared.spares.command_pool(shared, family) {
            state.pool = Some(spare);
        }
    }

    /// An encoder's command buffer of `level`, a driver's the encoder records into at each call:
    /// one from this command buffer's pool ([`allocate`](Self::allocate)), or one of a
    /// predicated loop's check's pool while it records again (`rerun.rs`).
    fn allocate_encoder(
        &self,
        state: &mut State,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer, Error> {
        if let Some(pool) = &state.rerun_pool {
            return hand_out(&self.device, &mut lock(pool), level);
        }
        self.allocate(state, level)
    }

    /// A `VkCommandBuffer` of `level` from this command buffer's pool, made on first use: a
    /// spare of the device's, or a new one. A command buffer the pool made before this command
    /// buffer took it is handed out again before a new one is made; its begin resets it (the
    /// pool's `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`, `spares.rs`). The caller holds
    /// `state` (the pool is externally synchronized).
    fn allocate(
        &self,
        state: &mut State,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer, Error> {
        hand_out(&self.device, self.pool(state)?, level)
    }

    /// [`CommandBuffer::encoder_primary`].
    fn encoder_primary(&self, state: &mut State) -> Result<(vk::CommandBuffer, bool), Error> {
        if let Some(buffer) = self.pool(state)?.take_begun() {
            return Ok((buffer, true));
        }
        Ok((self.allocate_raw(state)?, false))
    }

    /// This command buffer's pool, taken on first use: a spare of the device's, or a new one.
    /// The caller holds `state` (the pool is externally synchronized).
    fn pool<'a>(&self, state: &'a mut State) -> Result<&'a mut CommandPool, Error> {
        if state.pool.is_none() {
            let family = self.device.shared().queue_family_of(self.queue.vk_queue());
            state.pool = Some(command_pool(&self.device, family)?);
        }
        state
            .pool
            .as_mut()
            .ok_or_else(|| internal("a command buffer without its command pool"))
    }

    /// Ready `state`'s times at its commit, on the program's thread: where the device writes
    /// timestamps, the slot of the device's timestamp pools the batches write its start and end
    /// into. Without a slot free (and no pool more), it has no timestamps. Its calibration pair
    /// is taken as it completes (`place_gpu_times`): no reading of the GPU's clock here. One
    /// that encoded nothing and waits for nothing, and every one where the device answers GPU
    /// times from the host, writes none (`gpu_time.rs`).
    fn time(&self, state: &mut State) {
        let shared = self.device.shared();
        let timing = &shared.timing;
        if timing.clock().is_none() && !timing.answers_from_host() {
            return;
        }
        let waits = state
            .batches
            .all()
            .iter()
            .any(|batch| !batch.waits.is_empty() || !batch.binary_waits.is_empty());
        if state.recorded.is_empty() && !waits {
            state.times.stamped = Stamped::NoWork;
            state.times.slot = None;
            return;
        }
        if timing.answers_from_host() {
            state.times.stamped = Stamped::FromHost;
            return;
        }
        if !self.take_slot(state) {
            state.times.stamped = Stamped::NoSlot;
        }
    }

    /// Whether `state` holds a slot of the device's timestamp pools: the one its first encoder
    /// took as it opened, or one taken now; `false` where every slot is held and no pool more can
    /// be made.
    fn take_slot(&self, state: &mut State) -> bool {
        if state.times.slot.is_some() {
            return true;
        }
        let shared = self.device.shared();
        // SAFETY: the device is live.
        let Some(slot) = (unsafe { shared.timing.slot(shared.vk(), &shared.counts) }) else {
            return false;
        };
        shared.counts.timestamp_slot_taken();
        state.times.slot = Some(slot);
        true
    }

    /// Where the device writes timestamps into slots of three queries and `raw`, just begun, is the
    /// first `VkCommandBuffer` of the command buffer's first encoder, which opens its first batch:
    /// take the slot now, and write the start into its third query as `raw`'s first commands, the
    /// reset of the end and of that query (where the device does not reset the slot on the host as
    /// it takes it), then the timestamp at the top of the pipe (the bottom where batches are
    /// ordered by their ends: `batch_order.rs`; `gpu_time.rs`). The first batch's submission
    /// decides which start is read (`submit.rs`).
    fn open_start(&self, state: &mut State, raw: vk::CommandBuffer) {
        let timing = &self.device.shared().timing;
        let first_batch = match state.batches.all() {
            [batch] => batch.signals.is_empty() && batch.binary_signals.is_empty(),
            _ => false,
        };
        if !state.recorded.is_empty()
            || !first_batch
            || !timing.prefixed()
            || timing.clock().is_none()
            || timing.answers_from_host()
            || !self.take_slot(state)
        {
            return;
        }
        let Some((queries, prefix)) = (state.times.slot.as_ref())
            .and_then(|slot| slot.prefix().map(|prefix| (slot.queries(), prefix)))
        else {
            return;
        };
        let device = self.device.shared().vk();
        // SAFETY: `raw` was just begun, under the state lock, outside any render pass, and only
        // this call records it now; the end's query is the one before the prefix's, and both are
        // reset before the timestamp is written, on the host as the slot was taken or here
        // (`vkCmdResetQueryPool`: "sets the status of query indices [firstQuery, firstQuery +
        // queryCount - 1] to unavailable").
        unsafe {
            if !timing.resets_on_host() {
                device.cmd_reset_query_pool(raw, queries, prefix - 1, 2);
                exit_guard::called(Call::CmdResetQueryPool);
            }
            let stage = crate::batch_order::start_stage(self.device.shared().by_ends);
            device.cmd_write_timestamp(raw, stage, queries, prefix);
            exit_guard::called(Call::CmdWriteTimestamp);
        }
        state.prefix = Some(raw);
    }

    /// Read the command buffer's timestamps, now that its work completed, and give its slot
    /// back: in the completion turn, inside its section.
    pub(super) fn read_stamps(&self) {
        let mut state = lock(&self.state);
        let Some(slot) = state.times.slot.take() else {
            return;
        };
        // The start its first batch wrote: the prefix's where its first encoder wrote it alone,
        // else the one before the barriers (`gpu_time.rs`); then the end, after the first start.
        let start = match (state.times.prefixed, slot.prefix()) {
            (true, Some(prefix)) => prefix,
            _ => slot.first(),
        };
        let device = self.device.shared().vk();
        // Each query's value, then its availability.
        let read = |query: u32| {
            let mut result = [[0u64; 2]; 1];
            // SAFETY: the query was reset and written by work that completed; the result is
            // 64-bit with availability, two words.
            let read = unsafe {
                device.get_query_pool_results(
                    slot.queries(),
                    query,
                    &mut result,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
            };
            exit_guard::called(Call::GetQueryPoolResults);
            match read {
                Ok(()) | Err(vk::Result::NOT_READY) => {
                    let [[value, available]] = result;
                    Ok((available != 0).then_some(value))
                }
                Err(error) => Err(error),
            }
        };
        let read = read(start).and_then(|start| Ok(start.zip(read(slot.first() + 1)?)));
        state.times.stamped = match read {
            Ok(both) => Stamped::Read(both),
            Err(error) => Stamped::Failed(error),
        };
    }

    /// Publish the state's status, and its error where it failed, for `status` and `error` to
    /// read ([`CommandBufferShared::status`]), under the state's lock, so in the order of its
    /// transitions: the error first, set once (only the completion fails it, and nothing
    /// follows that), then the status.
    fn answer(&self, state: &State) {
        if let Some(error) = state.error {
            // Set at most once: a second set would be of the same completion's error.
            let _ = self.error.set(error);
        }
        self.status
            .store(status_word(state.status), Ordering::Release);
    }

    /// Place its GPU times among its queue's, now that it completed: in the completion turn,
    /// after its timestamps were read, in its queue's order (`gpu_time.rs`). One that wrote no
    /// timestamps takes its times here: encoder-less, where its queue reached it; answered from
    /// the host, its kernel end (or its queue's latest end) and its completion. Every one moves
    /// its queue's latest GPU end: its own, or its completion where it has none (an upper bound
    /// of its GPU end).
    pub(super) fn place_gpu_times(&self) {
        let shared = self.device.shared();
        let mut state = lock(&self.state);
        let times = &mut state.times;
        // The device's calibration pair, taken now, where the GPU that ran the work is awake
        // (`gpu_time.rs`).
        if let Stamped::Read(Some(_)) = times.stamped {
            times.pair = shared.timing.pair(&shared.counts);
        }
        self.queue.place_gpu_times(|latest_end| {
            let (kernel_end, completed) = (self.kernel.end(), times.completed);
            match times.stamped {
                Stamped::NoWork => {
                    let at = kernel_end.max(latest_end);
                    times.placed = Some((at, at));
                    at
                }
                // One after another, as Metal's serial queue can run them (`gpu_time.rs`).
                Stamped::FromHost => {
                    let start = kernel_end.max(latest_end);
                    let end = completed.max(start);
                    times.placed = Some((start, end));
                    end
                }
                Stamped::Read(Some((start, end))) => match shared.timing.clock() {
                    Some(clock) => {
                        let ((_, end), placement) =
                            clock.span(times.pair, start, end, life(&self.kernel, times));
                        if let Placement::Disagreed { .. } = placement {
                            shared.counts.calibration_disagreed();
                        }
                        end
                    }
                    None => completed,
                },
                Stamped::Unread | Stamped::NoSlot | Stamped::Read(None) | Stamped::Failed(_) => {
                    completed
                }
            }
        });
    }

    /// The scheduler submitted the last batch, or held it in a group (`group.rs`): in the queue's
    /// context, once the submitting thread ran its submission handlers, set the scheduled signals'
    /// values, in the order encoded, then become `Scheduled`, then run the scheduled handlers on
    /// this thread, in the order added, then let `wait_until_scheduled` return. Metal's status
    /// reads `Scheduled` just before its first scheduled handler starts, and its wait returns
    /// once the last returned (an M4, macOS 27.0.1); its scheduled signals' values show before
    /// its status reads `Scheduled` (an M3 Max, macOS 27.0.1).
    fn run_scheduled(self: &Arc<Self>) {
        {
            let mut state = lock(&self.state);
            while state.submitting {
                // The submitting thread runs no more once the process is exiting: what is
                // held stays held.
                if exit_guard::closed() {
                    return;
                }
                state.parked += 1;
                state = self
                    .changed
                    .wait_timeout(state, exit_guard::WAKE)
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
                state.parked -= 1;
            }
            if !state.scheduled_signals.is_empty() {
                // Set without the command buffer's lock: a set wakes the scheduler, whose lock
                // comes first.
                let signals = std::mem::take(&mut state.scheduled_signals);
                drop(state);
                for (event, value) in signals {
                    event.set(value);
                }
                state = lock(&self.state);
            }
            // Metal's dispatch and render work take their identifier allocations as the command
            // buffer is scheduled, before its status reads so (`unique_identifier.rs`).
            self.device
                .shared()
                .identifiers
                .scheduled(state.scheduled_work);
            state.status = CommandBufferStatus::Scheduled;
            self.answer(&state);
        }
        let buffer = CommandBuffer(Arc::clone(self));
        {
            let _running = Running::enter(self.marker(Step::ScheduledHandlers));
            loop {
                let Some(handler) = lock(&self.state).scheduled.pop_front() else {
                    break;
                };
                exit_guard::program(|| handler(&buffer));
            }
        }
        drop(buffer);
        let parked = {
            let mut state = lock(&self.state);
            state.scheduled_ran = true;
            state.parked > 0
        };
        self.wake_parked(parked);
    }

    /// Wake the threads sleeping on `changed`, where `parked` (read under the state's lock the
    /// change was made under) says one does: a thread that sleeps later looks at the change
    /// first, and a notify is a system call, sleeper or not. Counted:
    /// `WorkCounts::wakes_command_buffer_waiters`, `WorkCounts::wakes_spared`.
    fn wake_parked(&self, parked: bool) {
        self.device.shared().counts.command_buffer_wake(parked);
        if parked {
            self.changed.notify_all();
        }
    }

    /// Its last batch was submitted, or held in a group, on this thread, which holds no lock of
    /// the runtime's: run its submission handlers here, in the order added, then let its
    /// transition to `Scheduled` run ([`CommandBuffer::add_submission_handler`]).
    pub(crate) fn run_submission_handlers(self: &Arc<Self>) {
        let buffer = CommandBuffer(Arc::clone(self));
        {
            let _running = Running::enter(self.marker(Step::SubmissionHandlers));
            loop {
                let Some(handler) = lock(&self.state).submission.pop_front() else {
                    break;
                };
                exit_guard::program(|| handler(&buffer));
            }
        }
        drop(buffer);
        let parked = {
            let mut state = lock(&self.state);
            state.submitting = false;
            state.parked > 0
        };
        self.wake_parked(parked);
    }

    /// A batch could not be submitted (the scheduler dropped the rest): the command buffer
    /// finishes with `error` once what it submitted before completed, at its `VkQueue`'s last
    /// submitted value in `values`, and, where a command buffer ends at a fence
    /// ([`fences::ends_at_fence`]), at one submitted after it, as a command buffer whose last
    /// batch was submitted finishes; at the value alone when the queue takes no such fence (a
    /// lost device). Its submission and scheduled handlers never run; they are dropped with it,
    /// unless it became `Scheduled` as its last batch was held in a group (`group.rs`), or before
    /// it behind its GPU hang (`hang.rs`), which then fails from `Scheduled`.
    pub(crate) fn failed(self: &Arc<Self>, error: CommandBufferError, values: &[u64]) {
        let vk_queue = self.queue.vk_queue();
        {
            let mut state = lock(&self.state);
            state.failure = Some(error);
            // Its hold's batch before the wait, not taken yet, never is (`hang.rs`).
            if matches!(
                state.hang_scheduling,
                Some(hang::Scheduling::Hang { taken: false, .. })
            ) {
                state.hang_scheduling = None;
                self.changed.notify_all();
            }
        }
        let fence = fences::ends_at_fence(&self.device, vk_queue)
            .then(|| self.fence_after_submitted(vk_queue))
            .flatten();
        self.device.shared().completion.watch(
            vk_queue,
            Submitted {
                value: values[vk_queue],
                buffer: self.hold(),
                last: true,
                fence,
                signals: Vec::new(),
                holds: Vec::new(),
            },
        );
    }

    /// A fence of the device's, submitted to `VkQueue` `queue` with no work: it is signaled once
    /// everything submitted there before completed. `None` when the process is exiting or the
    /// submission fails. A section of the exit guard of its own.
    fn fence_after_submitted(&self, queue: usize) -> Option<vk::Fence> {
        let _busy = exit_guard::enter()?;
        let shared = self.device.shared();
        let raw = shared.vk();
        // SAFETY: the device's fences, inside a section.
        let fence = unsafe { shared.fences.take(raw) }.ok()?;
        // SAFETY: a submission of no batch with an unsignaled fence in no pending submission;
        // the `VkQueue` is used under its lock.
        shared.counts.queue_submitted(0, 0);
        let submitted = shared.submit_logged(
            (queue, None),
            Some(fence),
            &self.relayed,
            |vk_queue| unsafe { raw.queue_submit(vk_queue, &[], fence) },
        );
        exit_guard::called(Call::QueueSubmit);
        match submitted {
            Ok(()) => Some(fence),
            Err(error) => {
                // SAFETY: as for a failed batch's fence (`submit.rs`).
                unsafe { release::object(raw, fence) };
                let _ = self.device.failed(error);
                None
            }
        }
    }

    /// The command buffer's last batch is submitted, or held in a group (`group.rs`), on the
    /// scheduler's thread under its lock: its transition to `Scheduled` runs here where it is [`plain`](Transition::plain) (it has no
    /// scheduled handler, none is added after the commit, and it waits for no submission handler)
    /// and its queue's context runs on no thread and holds nothing (`Contexts::run_alone`): its
    /// status reads `Scheduled`, and `wait_until_scheduled` returns, as its last batch goes, with
    /// no handler thread woken for it. Else it is posted to that context. The scheduler's own
    /// holds of the command buffer outlive the transition's, so it never lets go of the last one
    /// here. Counted: `WorkCounts::scheduled_where_submitted`.
    pub(crate) fn schedule(self: &Arc<Self>) {
        let plain = plain_transition(&lock(&self.state));
        let shared = self.device.shared();
        let item = Item::Transition(Transition {
            buffer: self.hold(),
            to: To::Scheduled,
            plain,
        });
        match shared.contexts.run_alone(self.queue.id(), item) {
            Ok(()) => shared.counts.scheduled_where_submitted(),
            Err(item) => shared.contexts.post(self.queue.id(), item),
        }
    }

    /// Whether a thread about to block for the command buffer's completion may submit a batch of
    /// it itself (`scheduler::plain_ready`): nothing of the program's runs where it is submitted
    /// and published. Its transition to `Scheduled` is [`plain`](plain_transition), it has no
    /// submission handler for the submitting thread to run once the scheduler's lock is free, and
    /// no switch of a fetched target's storage form rides on it, which that thread settles and
    /// announces (`texture/storage_form.rs`). Handlers are refused once it is committed, so none
    /// comes after this read ([`CommandBuffer::add_submission_handler`]).
    pub(crate) fn submits_without_the_program(&self) -> bool {
        let state = lock(&self.state);
        plain_transition(&state) && state.submission.is_empty() && state.storage_form.is_none()
    }

    /// The last batch of the command buffer `buffer` holds finished with `result` (or a batch
    /// failed before): post its transition to `Completed` or `Error` to its queue's context. The
    /// hold moves into the transition, and the caller keeps no other reference: once the
    /// transition ran, the command buffer may go, with its device, on the handler thread.
    pub(crate) fn post_finished(
        buffer: Hold<CommandBufferShared>,
        result: Result<(), CommandBufferError>,
    ) {
        let (contexts, queue) = (buffer.device.shared().contexts.clone(), buffer.queue.id());
        contexts.post(
            queue,
            Item::Transition(Transition::finished(buffer, result)),
        );
    }

    /// The last batch of the command buffer `buffer` holds finished with `result`, as the queue
    /// waiter on this thread saw: its transition to `Completed` or `Error` runs here, where it is
    /// [`plain`](Transition::plain), its queue's context runs on no thread and holds only plain
    /// items, which run here before it (`Contexts::run_here`), and the runtime holds none of the
    /// program's objects a release drops or runs (`program_objects.rs`), so its release runs none
    /// of the program's code. A thread waiting for it is then woken from here, as Metal's
    /// completion wakes it, with no handler thread woken between. Else it is posted, as
    /// [`post_finished`](Self::post_finished) posts it.
    ///
    /// The waiter takes no handle of the device here, and lets each transition's hold go short of
    /// its command buffer's last reference ([`Transition::run`]): the device's last handle never
    /// goes here, and none is left here once the gate opens.
    pub(crate) fn finish_here(
        buffer: Hold<CommandBufferShared>,
        result: Result<(), CommandBufferError>,
    ) {
        let (contexts, queue) = (buffer.device.shared().contexts.clone(), buffer.queue.id());
        let item = Item::Transition(Transition::finished(buffer, result));
        let item = match program_objects::Unheld::enter() {
            Some(unheld) => {
                let ran = contexts.run_here(queue, item);
                drop(unheld);
                match ran {
                    Ok(()) => return,
                    Err(item) => item,
                }
            }
            None => item,
        };
        contexts.post(queue, item);
    }

    /// Finish the command buffer with `result` (or the failure of a batch): take the device's
    /// trail mark (`trails.rs`), set its status, decode its status words, run its completed
    /// handlers on this thread, give its queue's slot back, release its uploads, pools and heap
    /// versions, what it retained and its encoders' logs, then open its gate, which releases its
    /// waiters once the transition's [`Hold`] (and every other) is gone, and they wait for the
    /// trails below the mark. Waits from inside these steps answer as [`own`] says.
    fn complete(self: &Arc<Self>, result: Result<(), CommandBufferError>) {
        // A thread sleeps on `changed`, as the state's lock below reads.
        let parked;
        // Its command pool, where it settled, whose primary the device's spares begin ahead once
        // its completed handlers ran (below).
        let mut settled_pool = None;
        // The heap versions its commands bound, let go of once its completed handlers ran.
        let released_versions;
        // The holds its batches' commands took of buffers over host pages, let go of here, before
        // its completed handlers run: dropped without the state's lock (`detach.rs`).
        let submitted_holds;
        // No rebind encodes it again from here: a group that was never submitted runs nothing.
        self.device.shared().rebinds.no_replays(self);
        {
            let mut state = lock(&self.state);
            // Every trail of the work it waited for (a pipeline's making, a twin's decode) was
            // opened before its batches went, so before now: its waiters wait for those to close
            // (`trails.rs`).
            state.trail_mark = self.device.shared().trails.mark();
            // A failure the driver relayed for one of its batches is Metal's answer for its work
            // (`device/relay.rs`).
            let result = match (state.failure, self.relayed.failure()) {
                (Some(error), _) => Err(error),
                (None, Some((code, description))) => {
                    state.relayed_description = description;
                    Err(CommandBufferError::of_metal_code(code))
                }
                (None, None) => result,
            };
            // The logs of host pages hear of its shadows' copies back and its writes in place
            // again, now they ran, and let go of the holds a failed one's copies in set
            // (`dirty_pages.rs`).
            if self.device.shared().dirty.collects() {
                let dirty = lock(&self.dirty);
                (self.device.shared().dirty).completed(&dirty, result.is_err());
            }
            match result {
                Ok(()) => {
                    state.status = CommandBufferStatus::Completed;
                    // The queue's timeline reached the last batch's value, and every batch ends
                    // with a barrier to the host.
                    let words = state
                        .status_words
                        .as_ref()
                        .map_or(ExecutionStatus::initial(), Upload::read);
                    let mut report = ExecutionStatus::decode(&words);
                    if state.resident_misses != 0 {
                        report.resident_miss = true;
                        report.resident_misses =
                            report.resident_misses.wrapping_add(state.resident_misses);
                    }
                    if state.faults != ReplayFaults::default() {
                        report.replay_fault = true;
                        report.replay_faults =
                            ReplayFaults(report.replay_faults.0 | state.faults.0);
                    }
                    // An index bounds pass drew nothing of a draw its pool could not hold
                    // (`index_bounds.rs`): the command buffer failed.
                    if report.index_pool_overflow {
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::IndexPoolOverflow);
                    }
                    // A unit of command predication ended with a live block open where its
                    // commands skipped (`predication.rs`): Metal's GPU recovery failed the command
                    // buffer.
                    if let (Some(record), None) =
                        (state.predication.open_at_unit_end(), state.error)
                    {
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::PredicationOpenAtUnitEnd { record });
                        state.relayed_description = Some(
                            CommandBufferError::PREDICATION_OPEN_AT_UNIT_END_DESCRIPTION.to_owned(),
                        );
                    }
                    // An indirect mesh draw's object grid was one Metal's GPU discards
                    // (`render/mesh_arguments.rs`): the command buffer failed as Metal's.
                    if report.mesh_grid_discarded && state.error.is_none() {
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::GpuRecoveryVictim);
                    }
                    // A mesh draw's grid lay past what the device launches
                    // (`render/mesh_arguments.rs`): the command buffer failed.
                    if (report.mesh_grid_past_limits || state.mesh_grid_past)
                        && state.error.is_none()
                    {
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::MeshGridPastLimits);
                    }
                    // A scanned emulated mesh draw drew nothing of records its pool could not
                    // hold (`render/mesh.rs`): the command buffer failed.
                    if let Some(fault) = report.mesh_records_overflow {
                        let budget = state.mesh_pool.as_ref().map_or(0, PoolLease::bytes);
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::MeshRecordsOverflow(
                            MeshRecordsOverflow {
                                draw: fault.draw,
                                object_threadgroups: fault.object_threadgroups,
                                mesh_threadgroups: fault.mesh_threadgroups,
                                budget,
                            },
                        ));
                    }
                    // A scanned emulated tessellation draw drew nothing of records the same pool
                    // could not hold (`render/tessellation_scan.rs`): the command buffer failed.
                    if let Some(fault) = report.tessellation_records_overflow {
                        let budget = state.mesh_pool.as_ref().map_or(0, PoolLease::bytes);
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::TessellationRecordsOverflow(
                            TessellationRecordsOverflow {
                                draw: fault.draw,
                                records: fault.records,
                                budget,
                            },
                        ));
                    }
                    // A function that read an argument its function constants removed took zero
                    // for it, Metal's answer under its debug layer: the command buffer completes,
                    // and its report states the read. A function sampled through a cubic sampler an
                    // argument buffer named, whose bicubic filter it does not emulate there: no
                    // result is then Metal's.
                    if report.cubic_argument_sampler && state.error.is_none() {
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::CubicArgumentSampler);
                    }
                    state.report = Some(report);
                }
                Err(error) => {
                    state.status = CommandBufferStatus::Error;
                    state.error = Some(error);
                }
            }
            // A GPU error the command buffer caused counts against its queue, which Metal revokes
            // at the second, from before its status reads final (`revocation.rs`); a relayed one
            // is the driver's queue's.
            self.queue.errors().completed(state.failure);
            // An injected page fault or hang stands for it: the GPU recovers from it now, which
            // ends a predicated loop running meanwhile (`recovery.rs`).
            if let Some(recovered) = state.failure.and_then(recovery::injected) {
                self.device.shared().recoveries.raise(recovered);
            }
            // The final status, with its error, once: a read sees no `Completed` that a decoded
            // failure turns to `Error`.
            self.answer(&state);
            state.status_words = None;
            // Its heap versions, uploads, pools and descriptor pools go once its completed
            // handlers ran (below).
            released_versions = state.heaps.clear();
            // Its work ended, so nothing of it holds a buffer over host pages any longer: every
            // hold goes as its completion is processed, before its completed handlers run, as
            // Metal releases what a completion processes (`detach.rs`). A buffer's detach waiting
            // for its work answers `false`; one made from here on detaches. Dropped below,
            // without this lock, which no hold may be let go under (`Rebinds::wait_ran`).
            submitted_holds = state.holds_until(usize::MAX);
            // Its work ended (its fences were signaled, or its queues drained after a failed
            // wait), so nothing of its pools is pending: its command pool goes to the device's
            // spares whatever its status, never released on this thread (`spares.rs`); its
            // descriptor pools go once its completed handlers ran (below).
            if let Some(pool) = state.pool.take() {
                // None of the pool's command buffers is recording where it completed and ended
                // every one the pool handed out: a recording it left unused leaves one so.
                let ended = self.raws_ended.load(Ordering::Relaxed);
                let settled = state.status == CommandBufferStatus::Completed
                    && u64::try_from(pool.handed_out()).is_ok_and(|handed| handed == ended);
                settled_pool = settled.then_some(pool.raw);
                let shared = self.device.shared();
                shared.spares.give_command_pool(shared, pool, settled);
            }
            state.unplaced();
            parked = state.parked > 0;
        }
        // The holds taken above go here, with no lock of the command buffer's held, before any
        // handler of its runs: a detach from one of them detaches (`detach.rs`).
        drop(submitted_holds);
        self.wake_parked(parked);
        let buffer = CommandBuffer(Arc::clone(self));
        // Each handler leaves the queue under the lock and runs without it, in the order added
        // (a handler may query the command buffer).
        {
            let _running = Running::enter(self.marker(Step::CompletedHandlers));
            loop {
                let Some(handler) = lock(&self.state).completed.pop_front() else {
                    break;
                };
                exit_guard::program(|| handler(&buffer));
            }
        }
        let handled = Instant::now();
        drop(buffer);
        // Its queue's slot comes back as its last completed handler returns, as Metal's does: a
        // make blocked on its full queue returns then, never while a completed handler runs, nor
        // from one on its own full queue, and does not wait for what the commands used to be
        // released below (a no-copy buffer's deallocator may run long there); a make after
        // `wait_until_completed` returned never blocks (probe `slotdealloc`:
        // `queue.rs`).
        let slot = lock(&self.state).slot.take();
        drop(slot);
        let nanos = u64::try_from(handled.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.device.shared().counts.slot_returned(nanos);
        // What the commands used goes after the completed handlers, as Metal releases it (a
        // no-copy buffer's deallocator runs after them), and before the gate opens, so a waiter
        // returns after it. Each leaves under the lock and drops without it.
        {
            let _running = Running::enter(self.marker(Step::Release));
            // Its live buffers' tables first: a buffer whose last hold goes below then counts
            // only the tables of other command buffers, which may still reach its pages, as
            // holding them past its release (`HostPages::released_listed`); this one's wait ends
            // here, at its completion, where Metal's hold of what it retained ends too.
            let resident = lock(&self.state).resident.take();
            drop(resident);
            loop {
                let Some(table) = lock(&self.state).stepped_tables.pop() else {
                    break;
                };
                drop(table);
            }
            loop {
                let Some(resource) = lock(&self.state).retained.pop() else {
                    break;
                };
                drop(resource);
            }
            // Its encoders' logs go here too, back to the device's spares (`replay.rs`): they
            // hold what their calls named, which Metal holds until the command buffer completed
            // (a binding replaced before any command among it), and no thread but this one may
            // hold such a handle past the completion.
            let logs: Vec<Arc<Replay>> = (lock(&self.state).ended.iter_mut())
                .filter_map(|record| record.replay.take())
                .collect();
            let spares = self.device.shared().scheduler.spare_logs();
            for log in logs {
                spares.give_last(log);
            }
            let residency = lock(&self.state).residency.take();
            drop(residency);
            let residencies = lock(&self.state).let_go_of_uses();
            drop(residencies);
            let check = lock(&self.state).check.take();
            drop(check);
            // Then the runtime's own: its uploads, its index and mesh records pools' blocks, its
            // heap versions and its descriptor pools, none of them the program's, last. Each may
            // take a driver's lock as it goes (MoltenVK's `vkDestroyBuffer`, `vkFreeMemory` and
            // `vkDestroyDescriptorPool` take the lock of its device's residency set, which its
            // `vkBindBufferMemory` holds across a commit that took 24 to 39 ms for a 1 GiB block
            // on the M3 Max, `gpu_pools.rs`), so they go after the completed handlers, which wait
            // for none of them, and after what the program's commands used.
            let (uploads, index_pool, mesh_pool, mut descriptors, completed) = {
                let mut state = lock(&self.state);
                (
                    state.uploads.take(),
                    state.index_pool.take(),
                    state.mesh_pool.take(),
                    state.descriptors.take(),
                    state.status == CommandBufferStatus::Completed,
                )
            };
            drop((uploads, index_pool, mesh_pool, released_versions));
            // A command buffer that completed gives its descriptor pools back to the device's
            // spares; one that failed releases them.
            if completed {
                descriptors.give_back();
            } else {
                descriptors.release();
            }
        }
        // The gate notifies its waiters only where it opened here, which this transition's own
        // hold holds it from: the notify for them is that hold's, as it goes (`gate.rs`).
        let notified = self.gate.finish();
        self.device.shared().counts.command_buffer_wake(notified);
        // Its pool's primary begun ahead, where no command buffer took the pool since
        // (`spares.rs`), and no thread waits for the gate, which opens only once this
        // transition's hold goes: a waiter's return is never delayed by the begin, but for one
        // that came after this check.
        if let Some(pool) = settled_pool.filter(|_| !self.gate.awaited()) {
            let shared = self.device.shared();
            shared.spares.begin_ahead(shared, pool);
        }
    }
}

/// A command buffer's transition, run in its queue's context.
pub(crate) struct Transition {
    /// The command buffer, held until the transition ran.
    buffer: Hold<CommandBufferShared>,
    to: To,
    /// It runs none of the program's code but what a release of the program's objects runs
    /// (`program_objects.rs`): no handler and no notice.
    plain: bool,
}

/// The status a transition goes to.
enum To {
    Scheduled,
    /// `Completed`, or `Error`.
    Finished(Result<(), CommandBufferError>),
}

impl Transition {
    /// The transition of the command buffer `buffer` holds to `Completed` or `Error`, with
    /// `result`: plain where it has no completed handler (none is added after the commit).
    fn finished(buffer: Hold<CommandBufferShared>, result: Result<(), CommandBufferError>) -> Self {
        let plain = lock(&buffer.state).completed.is_empty();
        Transition {
            buffer,
            to: To::Finished(result),
            plain,
        }
    }

    fn run(self) {
        let on_handler_thread = contexts::on_handler_thread();
        match self.to {
            To::Scheduled => self.buffer.run_scheduled(),
            To::Finished(result) => {
                if !on_handler_thread {
                    self.buffer.device.shared().counts.completed_where_seen();
                }
                self.buffer.complete(result);
            }
        }
        // The hold goes here, after the transition: the gate opens then. On a queue's waiter
        // (`CommandBufferShared::finish_here`) it goes short of the command buffer's last
        // reference, which holds the device: that last one goes on a handler thread, so the
        // device's teardown, which stops the waiters, never runs on one, and the waiter holds
        // nothing of the device once the gate opened, so a program's drop after its wait is the
        // device's last.
        if !on_handler_thread {
            if let Some(last) = self.buffer.let_go_short_of_last() {
                let (contexts, context) = last.device.elsewhere();
                contexts.post(context, Item::Release(Box::new(last)));
            }
        }
    }
}

/// An item of the device's handler threads (`contexts`).
pub(crate) enum Item {
    /// A command buffer's transition, in its queue's context.
    Transition(Transition),
    /// What a thread that may run none of the program's code let go of, dropped in a context of
    /// its own (`Device::let_go_elsewhere`): it holds no command buffer, so nothing waits for it
    /// and it waits for nothing.
    Release(Box<dyn Send>),
}

impl Run for Item {
    fn run(self) {
        match self {
            Item::Transition(transition) => transition.run(),
            Item::Release(held) => drop(held),
        }
    }

    fn plain(&self) -> bool {
        match self {
            Item::Transition(transition) => transition.plain,
            Item::Release(_) => false,
        }
    }
}

/// Whether a command buffer's transition runs none of the program's code ([`Transition::plain`]):
/// it has no scheduled handler, none is added after its commit, and it waits for no submission
/// handler.
fn plain_transition(state: &State) -> bool {
    state.scheduled.is_empty() && !state.submitting
}

/// Whether `wait_until_scheduled` returns: the command buffer's scheduled handlers ran (Metal's
/// wait returns only then), or it finished, or failed without being scheduled.
fn scheduled(state: &State) -> bool {
    state.scheduled_ran
        || matches!(
            state.status,
            CommandBufferStatus::Completed | CommandBufferStatus::Error
        )
}

/// A command pool of `device` for queue family `family`: a spare of the device's (outside a
/// section of the exit guard), or a new one, whose command buffers are reset as they are begun
/// again (`spares.rs`).
pub(super) fn command_pool(device: &Device, family: u32) -> Result<CommandPool, Error> {
    let shared = device.shared();
    if let Some(spare) = shared.spares.command_pool(shared, family)? {
        return Ok(spare);
    }
    let info = vk::CommandPoolCreateInfo::default()
        .flags(
            vk::CommandPoolCreateFlags::TRANSIENT
                | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )
        .queue_family_index(family);
    // SAFETY: the device is live and `info` valid.
    let pool = unsafe { shared.vk().create_command_pool(&info, None) };
    exit_guard::called(Call::CreateCommandPool);
    shared.counts.command_pool_made();
    Ok(CommandPool::new(pool.map_err(vulkan)?, family))
}

/// A `VkCommandBuffer` of `level` from `pool`, of `device`: one it made before and has not
/// handed out since it was taken, whose begin resets it, else a new one. The caller holds the
/// pool alone (Vulkan synchronizes it externally).
pub(super) fn hand_out(
    device: &Device,
    pool: &mut CommandPool,
    level: vk::CommandBufferLevel,
) -> Result<vk::CommandBuffer, Error> {
    if let Some(buffer) = pool.reuse(level) {
        return Ok(buffer);
    }
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool.raw)
        .level(level)
        .command_buffer_count(1);
    let buffer = device.allocate_command_buffers(&info)?[0];
    pool.made(level, buffer);
    Ok(buffer)
}

/// The command-buffer error of a barrier command buffer that could not be allocated, or of a
/// recording that failed otherwise (a heap texture's moves' images, `EncoderCore::sample_moves`),
/// recording a lost device. `allocate_raw` fails only with a Vulkan result; any other error is
/// internal.
pub(super) fn recording_error(device: &Device, error: Error) -> CommandBufferError {
    match error {
        Error::DeviceLost => submission_error(device, vk::Result::ERROR_DEVICE_LOST),
        Error::OutOfMemory(_) => CommandBufferError::OutOfMemory,
        Error::Vulkan(code) => CommandBufferError::Internal(code),
        _ => CommandBufferError::Internal(super::error::VkResultCode(
            vk::Result::ERROR_UNKNOWN.as_raw(),
        )),
    }
}

/// The command-buffer error of a host step whose recording failed with `error`: a refusal, what
/// `commit` refuses where the CPU wrote the records, is [`CommandBufferError::Refused`], kept in
/// `refusal`; a Vulkan failure, a lost device, memory, and the runtime's own broken invariant fail
/// it as a recording does ([`recording_error`]).
fn step_error(device: &Device, error: Error, refusal: &mut Option<Error>) -> CommandBufferError {
    let refused = match &error {
        Error::Compile(metal2vulkan::error::Error::Internal(_)) => false,
        Error::Invalid(_)
        | Error::Unsupported(_)
        | Error::Bind(_)
        | Error::Compile(_)
        | Error::NotImplemented(_)
        | Error::AppleOnly(_)
        | Error::SamplerLimit(_) => true,
        _ => false,
    };
    if !refused {
        return recording_error(device, error);
    }
    *refusal = Some(error);
    CommandBufferError::Refused
}

/// The command-buffer error of a failed submission, recording a lost device.
fn submission_error(device: &Device, error: vk::Result) -> CommandBufferError {
    match device.failed(error) {
        Error::DeviceLost => CommandBufferError::DeviceLost,
        Error::OutOfMemory(_) => CommandBufferError::OutOfMemory,
        _ => CommandBufferError::Internal(super::error::VkResultCode(error.as_raw())),
    }
}

/// The life of the command buffer whose kernel times are `kernel` and other times `times`, as its
/// timestamps are checked against it (`gpu_time.rs`, `GpuClock::span`).
fn life(kernel: &KernelTimes, times: &Times) -> Life {
    Life {
        kernel_start: kernel.start(),
        completed: times.completed,
    }
}
