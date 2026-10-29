//! The check after a predicated loop laid down as copies (`predication.rs`): where the step past
//! its last copy halted its unit (the loop would run another pass), the check runs the loop's
//! laying again, from the halted loop's copy on (the halt word, `bind::PredicationOp::WhileFirst`),
//! until the loop ends, as Metal runs a loop on; then the command buffer goes on. Metal never ends
//! a loop that runs without bound (an M3 Max with macOS 27.0.1 ran an endless While and DoWhile for
//! 180 s, each, with no error), and neither does the check.
//!
//! - **The laying** is its own run of pieces ([`Piece`]), in encoding order: the
//!   `VkCommandBuffer`s the encoders recorded it into, each begun to be submitted again, and the
//!   work they left apart between them (`Deferred`: an indirect dispatch or an indirect command
//!   buffer's execution a host step records, or a dispatch whose variant records in a command
//!   buffer of its own). The encoder cut its command buffer at the outermost loop's start
//!   (`EncoderCore::cut_reusable`), at each later encoder of its unit and after each work apart
//!   (`EncoderCore::continuation`), each behind a barrier from everything before it, and ends it at
//!   the check (`EncoderCore::end_laying`), behind one to every later command and to the host; the
//!   textures it shadows are copied back at each cut. A run again so repeats nothing encoded before
//!   the loop (a render pass's load and store actions, a blit, an event's signal or wait) and
//!   copies in again what it shadows; every dispatch in it is gated, and every step holds until
//!   the halted loop's copy starts.
//! - **The work apart runs again as it first ran** ([`LoopCheck::learn`], at the commit, from the
//!   segment each left): a run again submits the laying's command buffers up to it, and where its
//!   first run waited for the work before it to complete (a host step that reads what the GPU
//!   wrote: an indirect dispatch's arguments, an execution's records or its block's counter), waits
//!   for them too, then records it again as its host step did (`Deferred::step`), or as the commit
//!   did on the GPU alone (`Deferred::record_on_gpu`), and goes on. Those recordings come from a
//!   command pool of the check's own, and take their descriptor sets and uploads from arenas of
//!   its own, lent to the command buffer meanwhile ([`RunArenas`]); once the run again completed,
//!   the pool is taken back, the sets' pools reset and the uploads let go (their chunks parked
//!   for the next run, `recycle.rs`), so a loop that runs on holds no more of them than one run.
//! - **The check** is a host step (`Deferred::Loop`), its segment the first of the batch after the
//!   laying: once the work before it completed, it reads the unit's halt word. 0: the batch goes.
//!   Else a thread of its own ([`run_again`]) runs the laying again on the command buffer's
//!   `VkQueue`, each submission with a fence of the device's, logged for the command buffer's
//!   relayed failures (`device/relay.rs`), and reads the word again once the run completed, until
//!   it is 0; then it wakes the scheduler. Meanwhile the batch waits as for a pipeline's making
//!   (`Segment::steps_after_pipelines`), and the scheduler goes on with every other line; a batch
//!   committed later on another queue that conflicts with the laying's uses waits behind it
//!   (`CommandBufferShared::footprint`), as behind any host step. The batch's other segments step
//!   once the check ends: they read what the laying leaves.
//! - **Ends.** As Metal's GPU runs a loop without bound on after the program let go of every
//!   handle of its queue and its device, so does the thread, which holds the command buffer, and
//!   through it the queue and the device. It stops only at a lost device, the GPU's recovery
//!   from a fault or a hang anywhere on the device, as Metal's loop does (`recovery.rs`), a
//!   failed submission, wait or recording (each failing the command buffer as its submission or
//!   host step would), or the process's exit, which it never holds up: it waits in slices of
//!   [`WAIT_SLICE`], each inside a section of the exit guard, and does nothing more once the
//!   guard closes. Each run again is counted (`WorkCounts::predicated_loop_runs_again`).

use super::gate::Hold;
use super::recovery::{self, Recovery};
use super::{
    command_pool, recording_error, step_error, submission_error, CommandBuffer, CommandBufferShared,
};
use crate::deferred::{Deferred, StepInputs};
use crate::descriptor_arena::DescriptorArena;
use crate::device::lock;
use crate::encoder::{Recorded, RecordedSegment, Segment};
use crate::error::{internal, CommandBufferError, Error};
use crate::exit_guard::{self, Call};
use crate::retained::Retained;
use crate::spares::CommandPool;
use crate::tracker::EncoderUses;
use crate::upload::{Upload, Uploads};
use ash::vk;
use metal2vulkan::bind::ReplayFaults;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// How long one wait for a run again's fence lasts before the thread looks for an exit.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// One piece of a loop's laying, in encoding order (the module docs).
pub(crate) enum Piece {
    /// A `VkCommandBuffer` an encoder recorded the laying into, begun to be submitted again.
    Raw(vk::CommandBuffer),
    /// Work the encoder left apart there, which each run again records again.
    Apart(Deferred),
}

/// The arenas a check's runs again take their descriptor sets and uploads from, each lent to the
/// command buffer while a run's work apart records (`Uploads::lend`, `DescriptorArena::lend`),
/// and emptied once the run completed ([`settle`](Self::settle)).
struct RunArenas {
    uploads: Arc<Mutex<Uploads>>,
    descriptors: Arc<Mutex<DescriptorArena>>,
}

impl RunArenas {
    /// Empty arenas.
    fn new() -> RunArenas {
        RunArenas {
            uploads: Arc::new(Mutex::new(Uploads::new())),
            descriptors: Arc::new(Mutex::new(DescriptorArena::default())),
        }
    }

    /// Lend the arenas to the command buffer of `state`, or (`false`) take them back.
    fn lend(&self, state: &mut super::State, lent: bool) {
        state.uploads.lend(lent.then(|| Arc::clone(&self.uploads)));
        state
            .descriptors
            .lend(lent.then(|| Arc::clone(&self.descriptors)));
    }

    /// The run again completed, so nothing uses what it took: the sets' pools reset for the next
    /// run, and the uploads let go, without a lock of the command buffer (a chunk's last holder
    /// parks it, `recycle.rs`). Inside a section of the exit guard; refused as the reset is.
    fn settle(&self) -> Result<(), Error> {
        let uploads = lock(&self.uploads).take();
        drop(uploads);
        lock(&self.descriptors).reset()
    }
}

/// How a run again records a piece of work apart: as its first run did.
enum Again {
    /// As its host step did, from `inputs`, after the work before it completed where `waits`.
    Step { inputs: StepInputs, waits: bool },
    /// As the commit did, on the GPU alone.
    OnGpu,
}

impl Again {
    /// How a run again records the work of `segment`, prepared by the commit: `None` where the
    /// commit left it nothing a step takes.
    fn of(segment: &Segment) -> Option<Again> {
        if segment.gpu_alone {
            return Some(Again::OnGpu);
        }
        let inputs = segment.inputs.clone()?;
        Some(Again::Step {
            inputs,
            waits: segment.host || segment.work.gpu_written(),
        })
    }
}

/// The check after a loop's laying (the module docs).
pub(crate) struct LoopCheck {
    /// The unit's skip counter: its halt word lies 4 bytes past it.
    counter: Upload,
    /// The laying, as encoded.
    pieces: Vec<Piece>,
    /// How each of its pieces of work apart runs again, in order: learned at the commit.
    agains: OnceLock<Vec<Option<Again>>>,
    /// The command buffer's parts before the laying's first: the laying lies in the parts from
    /// there to the check's.
    from: usize,
    rounds: Mutex<Rounds>,
}

/// Where a check stands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rounds {
    /// Its host step has not read the halt word yet.
    Unchecked,
    /// Its thread runs the laying again.
    Running,
    /// The loop ended, or the check stopped.
    Ended,
}

/// Why a run again stopped short of reading the halt word.
enum Stop {
    /// A failure the driver relayed for it: the command buffer fails with it at its completion.
    Relayed,
    /// The command buffer fails so.
    Failed(CommandBufferError),
    /// The process is exiting: the thread does nothing more.
    Exiting,
}

/// How one run again ended.
enum Round {
    /// The loop would run on.
    Again,
    /// The loop ended, or a relayed failure ends the command buffer.
    Ended,
    /// The command buffer fails so.
    Failed(CommandBufferError),
    /// The loop stopped with no error, and nothing after it runs (`recovery.rs`).
    Cut,
    /// The process is exiting: the thread does nothing more.
    Exiting,
}

impl LoopCheck {
    /// The check of the laying `pieces`, after `from` of the command buffer's parts, of the loop
    /// of the unit of skip counter `counter`.
    pub(crate) fn new(counter: Upload, pieces: Vec<Piece>, from: usize) -> LoopCheck {
        LoopCheck {
            counter,
            pieces,
            agains: OnceLock::new(),
            from,
            rounds: Mutex::new(Rounds::Unchecked),
        }
    }

    /// The command buffer's parts before the laying's first.
    pub(crate) fn from(&self) -> usize {
        self.from
    }

    /// At the commit, with its segments prepared: learn how each piece of work apart runs again
    /// from the segment it left among `parts`, the laying's parts and those after it.
    pub(crate) fn learn(&self, parts: &[Recorded]) {
        let segments = || parts.iter().flat_map(|part| &part.segments);
        let agains = (self.pieces.iter())
            .filter_map(|piece| match piece {
                Piece::Apart(work) => Some(work),
                Piece::Raw(_) => None,
            })
            .map(|work| segments().find(|segment| segment.work.same(work)))
            .map(|segment| segment.and_then(Again::of))
            .collect();
        drop(self.agains.set(agains));
    }

    /// Whether its thread runs the laying again: its batch waits.
    pub(crate) fn running(&self) -> bool {
        *lock(&self.rounds) == Rounds::Running
    }

    /// Whether the unit is halted: its halt word is not 0. Read once the work that writes it
    /// completed.
    fn halted(&self) -> bool {
        self.counter.read::<8>()[4..] != [0; 4]
    }

    /// Its host step, once the laying completed: nothing to record; where the loop halted its
    /// unit, its thread starts running the laying again. Refused where the thread cannot start.
    pub(crate) fn step(
        self: &Arc<Self>,
        command_buffer: &CommandBuffer,
    ) -> Result<RecordedSegment, Error> {
        if !self.halted() {
            *lock(&self.rounds) = Rounds::Ended;
            return Ok(nothing());
        }
        *lock(&self.rounds) = Rounds::Running;
        let check = Arc::clone(self);
        let buffer = command_buffer.0.hold();
        // One of the threads that take a command buffer to its handlers (`exit_guard.rs`).
        let builder = std::thread::Builder::new().name("metal2vulkan-loop".to_owned());
        let spawned =
            exit_guard::spawn_user_interactive(builder, move || run_again(&check, buffer));
        if spawned.is_err() {
            *lock(&self.rounds) = Rounds::Ended;
            return Err(internal(
                "a predicated loop's check could not start its thread",
            ));
        }
        Ok(nothing())
    }

    /// Run the laying again on `buffer`'s `VkQueue`, its work apart recorded from `pool` and
    /// `arenas`, and read the halt word.
    fn run_once(
        &self,
        buffer: &Hold<CommandBufferShared>,
        (pool, arenas): (&Arc<Mutex<CommandPool>>, &RunArenas),
    ) -> Round {
        let mut recordings = Vec::new();
        let ran = self.run_pieces(buffer, (pool, arenas), &mut recordings);
        // Its recordings' command buffers go back to the pool for the next run: they ended there,
        // none of the command buffer's own pool (`CommandBufferShared::end_raw` counted them).
        let ended: usize = recordings.iter().map(|done| done.raws.len()).sum();
        buffer.raws_ended.fetch_sub(ended as u64, Ordering::Relaxed);
        {
            let mut state = lock(&buffer.state);
            for done in &recordings {
                state.faults = ReplayFaults(state.faults.0 | done.faults.0);
                state.resident_misses = state.resident_misses.wrapping_add(done.resident_misses);
            }
        }
        if ran.is_ok() {
            lock(pool).taken();
        }
        // Dropped without a lock: what it holds may be a state's last handle.
        drop(recordings);
        match ran {
            Ok(()) => {}
            Err(Stop::Relayed) => return Round::Ended,
            Err(Stop::Failed(error)) => return Round::Failed(error),
            Err(Stop::Exiting) => return Round::Exiting,
        }
        {
            let Some(_busy) = exit_guard::enter() else {
                return Round::Exiting;
            };
            if let Err(error) = arenas.settle() {
                return Round::Failed(recording_error(&buffer.device, error));
            }
        }
        buffer.device.shared().counts.predicated_loop_ran_again();
        match self.halted() {
            true => Round::Again,
            false => Round::Ended,
        }
    }

    /// The laying's pieces, in order: its command buffers submitted, up to each piece of work
    /// apart whose first run waited for the work before it, which completes first; each piece of
    /// work apart recorded again, its recording kept in `recordings` until the run completed.
    fn run_pieces(
        &self,
        buffer: &Hold<CommandBufferShared>,
        sources: (&Arc<Mutex<CommandPool>>, &RunArenas),
        recordings: &mut Vec<RecordedSegment>,
    ) -> Result<(), Stop> {
        let device = &buffer.device;
        let unlearned = || {
            let error = internal("a predicated loop's work apart without how it runs again");
            Stop::Failed(recording_error(device, error))
        };
        let agains = self.agains.get().ok_or_else(unlearned)?;
        let mut agains = agains.iter();
        let mut submits = Vec::new();
        for piece in &self.pieces {
            match piece {
                Piece::Raw(raw) => submits.push(*raw),
                Piece::Apart(work) => {
                    let again = (agains.next().and_then(Option::as_ref)).ok_or_else(unlearned)?;
                    if matches!(again, Again::Step { waits: true, .. }) && !submits.is_empty() {
                        self.submit(buffer, &submits)?;
                        submits.clear();
                    }
                    let done = self.record(buffer, sources, work, again)?;
                    submits.extend(&done.raws);
                    recordings.push(done);
                }
            }
        }
        self.submit(buffer, &submits)
    }

    /// Record `work` again as `again` says, its command buffers from `pool` and its sets and
    /// uploads from `arenas`, as a host step: on this thread, inside a section of the exit guard,
    /// with no lock of the command buffer held.
    fn record(
        &self,
        buffer: &Hold<CommandBufferShared>,
        (pool, arenas): (&Arc<Mutex<CommandPool>>, &RunArenas),
        work: &Deferred,
        again: &Again,
    ) -> Result<RecordedSegment, Stop> {
        let device = &buffer.device;
        let Some(_busy) = exit_guard::enter() else {
            return Err(Stop::Exiting);
        };
        let _turn = lock(&buffer.reencoding);
        let command_buffer = CommandBuffer(Arc::clone(buffer));
        {
            let mut state = lock(&buffer.state);
            state.rerun_pool = Some(Arc::clone(pool));
            arenas.lend(&mut state, true);
        }
        let record = || {
            exit_guard::host_step(|| match again {
                Again::OnGpu => (work.record_on_gpu(&command_buffer)?)
                    .map(|(recorded, _)| recorded)
                    .ok_or_else(|| internal("a predicated loop's work apart off the GPU")),
                Again::Step { inputs, .. } => work.step(&command_buffer, inputs),
            })
        };
        let recorded = match buffer.step_numbers.get() {
            Some(numbers) => numbers.numbering(record),
            None => record(),
        };
        let mut state = lock(&buffer.state);
        state.rerun_pool = None;
        arenas.lend(&mut state, false);
        match recorded {
            // Its first run named no detached buffer, or the command buffer failed there.
            Ok(done) if done.taken.dead() => {
                drop(state);
                let error = internal("a predicated loop's run again named a detached buffer");
                Err(Stop::Failed(recording_error(device, error)))
            }
            Ok(done) => Ok(done),
            Err(error) => Err(Stop::Failed(step_error(device, error, &mut state.refusal))),
        }
    }

    /// Submit `raws` (ended, in order) on `buffer`'s `VkQueue` and wait for them to complete.
    fn submit(&self, buffer: &CommandBufferShared, raws: &[vk::CommandBuffer]) -> Result<(), Stop> {
        let device = &buffer.device;
        let Some(busy) = exit_guard::enter() else {
            return Err(Stop::Exiting);
        };
        let shared = device.shared();
        let raw = shared.vk();
        // SAFETY: the device's fences, inside a section.
        let fence = unsafe { shared.fences.take(raw) }
            .map_err(|error| Stop::Failed(submission_error(device, error)))?;
        let queue = buffer.queue.vk_queue();
        let submits = [vk::SubmitInfo::default().command_buffers(raws)];
        shared.counts.queue_submitted(1, raws.len());
        // SAFETY: the `VkQueue` is used under its lock; each command buffer was begun to be
        // submitted again (the laying's) or recorded for this submission alone (a run again's),
        // ended, and is in no pending submission: the laying's batch completed before the check's
        // step, and each submission of a run again before the next; the fence is unsignaled and
        // in no pending submission.
        let submitted = shared.submit_logged(
            (queue, None),
            Some(fence),
            &buffer.relayed,
            |vk_queue| unsafe { raw.queue_submit(vk_queue, &submits, fence) },
        );
        exit_guard::called(Call::QueueSubmit);
        if let Err(error) = submitted {
            // SAFETY: a fence no submission took.
            unsafe { crate::release::object(raw, fence) };
            return Err(Stop::Failed(submission_error(device, error)));
        }
        drop(busy);
        loop {
            let Some(_busy) = exit_guard::enter() else {
                return Err(Stop::Exiting);
            };
            // SAFETY: the device's fence, submitted above.
            let waited =
                unsafe { raw.wait_for_fences(&[fence], true, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitForFences);
            match waited {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => continue,
                // The fence stays in the relay's log, as a failed batch's does, never released.
                Err(error) => return Err(Stop::Failed(submission_error(device, error))),
            }
        }
        let Some(_busy) = exit_guard::enter() else {
            return Err(Stop::Exiting);
        };
        // Out of the relay's log before it goes back: its submission completed.
        if let Some(flights) = shared.flights(queue) {
            flights.let_go(fence);
        }
        // SAFETY: the fence is signaled and its submission completed.
        let reset = unsafe { raw.reset_fences(&[fence]) };
        exit_guard::called(Call::ResetFences);
        let mut fences = vec![fence];
        match reset {
            // SAFETY: unsignaled and in no pending submission; inside a section.
            Ok(()) => unsafe { shared.fences.give(raw, &mut fences) },
            Err(_) => {
                // SAFETY: in no pending submission.
                unsafe { crate::release::object(raw, fence) };
            }
        }
        // A failure the driver relayed for it fails the command buffer at its completion.
        match buffer.relayed.failure() {
            Some(_) => Err(Stop::Relayed),
            None => Ok(()),
        }
    }
}

/// A recording of nothing: a check records no command.
fn nothing() -> RecordedSegment {
    RecordedSegment {
        raws: Vec::new(),
        summary: EncoderUses::new(true).finish(),
        retained: Retained::default(),
        executions: Vec::new(),
        keys: Default::default(),
        taken: Default::default(),
        resident_misses: 0,
        faults: ReplayFaults::default(),
    }
}

/// The loop's decision before each run again: `None` where it runs again; else how it ends
/// without running. It stops once its device is lost (the module docs), and at the GPU's
/// recovery from a fault (as a victim) or a hang (cut) since its command buffer's first
/// submission (`recovery.rs`).
fn before_round(buffer: &CommandBufferShared) -> Option<Round> {
    if buffer.device.is_lost() {
        return Some(Round::Failed(CommandBufferError::DeviceLost));
    }
    match recovery::recovered(buffer)? {
        Recovery::Fault => Some(Round::Failed(CommandBufferError::GpuRecoveryVictim)),
        Recovery::Hang => Some(Round::Cut),
    }
}

/// The check's thread (the module docs): run the laying again until the loop ends, or the check
/// stops ([`before_round`]); then the scheduler takes the batch again.
fn run_again(check: &LoopCheck, buffer: Hold<CommandBufferShared>) {
    let device = &buffer.device;
    let waker = device.shared().scheduler.waker();
    // A new one, made inside a section as every Vulkan call of the runtime's threads is (a spare
    // of the device's is taken outside one alone), given to the device's spares at the end.
    let family = device.shared().queue_family_of(buffer.queue.vk_queue());
    let pool = {
        let Some(_busy) = exit_guard::enter() else {
            // What it holds stays held to the process's end, as the scheduler's does.
            return std::mem::forget(buffer);
        };
        command_pool(device, family).map(|pool| Arc::new(Mutex::new(pool)))
    };
    let arenas = RunArenas::new();
    let (failure, settled) = match &pool {
        Ok(pool) => loop {
            let round =
                before_round(&buffer).unwrap_or_else(|| check.run_once(&buffer, (pool, &arenas)));
            match round {
                Round::Again => continue,
                Round::Ended => break (None, true),
                // A recording it left unended leaves the pool to be reset.
                Round::Failed(error) => break (Some(error), false),
                Round::Cut => {
                    recovery::cut(&buffer, check);
                    break (None, true);
                }
                // What it holds stays held to the process's end, as the scheduler's does.
                Round::Exiting => {
                    std::mem::forget(arenas);
                    return std::mem::forget(buffer);
                }
            }
        },
        Err(error) => (Some(recording_error(device, error.clone())), true),
    };
    if let Some(error) = failure {
        lock(&buffer.state).step_failure.get_or_insert(error);
    }
    // Nothing of it is pending: each submission completed before the thread went on.
    if let Ok(pool) = pool {
        if let Ok(pool) = Arc::try_unwrap(pool) {
            let pool = pool
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let shared = device.shared();
            shared.spares.give_command_pool(shared, pool, settled);
        }
    }
    *lock(&check.rounds) = Rounds::Ended;
    waker();
    drop(buffer);
}
