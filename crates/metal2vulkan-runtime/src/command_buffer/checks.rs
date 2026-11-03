//! A command buffer's submission checks (a runtime extension): a check the program sets runs for
//! each submission of the command buffer as Metal splits it, once its event waits are met, just
//! before the runtime submits it, and may fail it with a code.
//!
//! - **Metal's submissions** ([`metal_submissions`]). Metal splits a command buffer into
//!   submissions (measured by a consumer on an M3 Max, macOS 26.7): each render pass is its own;
//!   consecutive blit encoders share one, as do consecutive compute encoders; an event wait or
//!   signal between two encoders splits a run. Resource state encoders join their own runs, as
//!   blits do (unmeasured). The runtime's batches split at every event command too, so a submission
//!   begins in one batch; it may run on into later batches, split at a host step, a resource state
//!   encoder or a binary semaphore, which Metal does not split at.
//! - **When.** The scheduler holds a batch of a command buffer with a check, as it holds one for
//!   its host step, once it is otherwise ready: its binary semaphores' signals submitted, every
//!   event value it waits for reached by the host or by a submitted signal, and its host step
//!   recorded. It holds it further while a value it waits for is reached only by a GPU signal
//!   submitted but not yet landed (an `Event`'s, or a `SharedEvent`'s the GPU signals): Metal
//!   submits the submission, and checks it, once the signal happened, so a resource detached
//!   meanwhile fails it. A value the host set (or the device reached) needs no wait. On its
//!   thread, without its own lock or the command buffer's, it then runs the check of each
//!   submission that begins in the batch, and again of each that a host step of the batch
//!   recorded commands for (what they reach is known only then), then submits the batch in its
//!   order. A thread in `CommandBuffer::commit_and_submit` that makes the submission pass itself
//!   runs those checks instead, the same way, just before it submits the batch, where the batch
//!   is the one committed first of all, ready but for them, with no host step and no group, and
//!   every value it waits for reached on the device or by the host (`scheduler::claim_check`);
//!   the rest are the scheduler thread's. Where the scheduler groups batches (`CompletionPolicy::Grouped` on a driver that
//!   merges submissions, `group.rs`), it runs the checks of up to three batches right behind a
//!   head that goes with the head's, in commit order, each once its batch is otherwise ready and
//!   its events' values reached on the device or by the host, none committed after another
//!   queue's batch: so they go to the driver together. Such a check runs before the previous
//!   batch's `vkQueueSubmit`, in the pass that submits its own batch, where alone it runs after
//!   it. `CompletionPolicy::Exact` checks each head alone.
//! - **Stricter than Metal: the re-check after a host step.** Metal checks a submission once,
//!   with what it knows as it submits it, and cannot know what the commands the GPU writes into
//!   an indirect command buffer reach before they run (an earlier encoder of the same submission
//!   may write them). The runtime checks such a submission as it begins and again once the host
//!   step recorded those commands; a second check that fails it fails its encoders from that
//!   batch on, its first part having run. So a resource reached only through such commands, which
//!   Metal would not see, fails what is left of its submission.
//! - **A failed submission.** The encoders of a submission whose check answers a code run
//!   nothing from that batch on, with the mechanics of `CommitFailure`'s set
//!   (`State::fail_part`): every other encoder runs, the event signals fire, the scheduled and
//!   completed handlers run, and the command buffer completes with status `Error` and
//!   `CommandBufferError::Injected` of the first failure's code (a `commit_failed`'s before any
//!   check's). A check that panics fails its submission with code 1 (`Internal`).
//! - **Behind a hang** (`hang.rs`). A command buffer committed with an event wait while a hang
//!   of its queue (a draw of primitive type 9) was in flight is held for this pass too, with or
//!   without a check: its first batch's pass, once its waits are met and the batches before it
//!   in its queue's order are submitted, runs nothing of it where a hang of its queue is still in
//!   flight then, and the command buffer completes with no error, its event signals firing. No
//!   check of the program's runs for it then.
//! - **Zero cost without a check.** A command buffer without one holds no submission and its
//!   batches are never held for one: the scheduler reads one flag per batch.

use super::{lock, CommandBuffer, CommandBufferShared, State};
use crate::error::{invalid, CommandBufferError, Error, Problem};
use crate::exit_guard;
use crate::resource::ObjectId;
use crate::used_resources::{union_of, EncoderKind, UsedResources};
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

/// A command buffer's submission check ([`CommandBuffer::set_submission_check`]): a runtime
/// extension, no Metal type. Called with each submission of the command buffer as Metal splits
/// it, on the device's scheduler thread, or on the thread of a
/// [`CommandBuffer::commit_and_submit`] that submits it, it answers `None` for the submission to
/// run, or the `MTLCommandBufferError` code to fail it with (8, `OutOfMemory`, for one that uses
/// a resource whose memory was taken away, as Metal fails it).
///
/// It runs with no lock of the runtime's held, and holds back the submissions of the thread that
/// runs it: a commit made meanwhile returns at once, and its batches wait in the inbox for the
/// scheduler thread's next pass, after this check (`scheduler/inbox.rs`). It must take no lock a
/// caller of `commit_and_submit` holds and must not wait for GPU work, a command buffer or a
/// handler, and should be quick. It may call the runtime otherwise.
pub type SubmissionCheck = Arc<dyn Fn(&Submission<'_>) -> Option<NonZeroU64> + Send + Sync>;

/// The `MTLCommandBufferError` code a submission whose check panicked fails with: `Internal`.
const PANICKED: u64 = 1;

/// One submission of a command buffer as Metal splits it, as its [`SubmissionCheck`] sees it,
/// just before the runtime submits it: a runtime extension, no Metal type.
pub struct Submission<'a> {
    command_buffer: &'a CommandBuffer,
    kind: EncoderKind,
    encoders: Range<usize>,
}

impl Submission<'_> {
    /// The command buffer.
    pub fn command_buffer(&self) -> &CommandBuffer {
        self.command_buffer
    }

    /// Its encoders' kind: every encoder of a submission is of one kind.
    pub fn kind(&self) -> EncoderKind {
        self.kind
    }

    /// Its encoders, by their places among the command buffer's ended encoders (the index of
    /// [`CommandBuffer::ended_encoders`]).
    pub fn encoders(&self) -> Range<usize> {
        self.encoders.clone()
    }

    /// What its encoders' commands use and reach, listed as
    /// [`CommandBuffer::used_resources`] lists the command buffer's: what they bound, made
    /// resident, copied, attached or mapped, and what the commands of the indirect command
    /// buffers they execute reach (recorded at the commit, or by the host step before this
    /// check). Listed when asked: a check that does not ask pays nothing for it.
    pub fn used_resources(&self) -> UsedResources {
        let records = self
            .command_buffer
            .state()
            .read_ended(Some(self.encoders.clone()));
        let used = union_of(self.command_buffer.device().shared(), &records);
        // Released without the lock: what it holds may be a resource's last handle.
        drop(records);
        used
    }

    /// Whether its encoders' commands use any of `objects`, the identities of what
    /// [`used_resources`](Self::used_resources) would list, answered without listing it, as
    /// [`CommandBuffer::uses`] answers: no handle is cloned or held.
    pub fn uses(&self, objects: &[ObjectId]) -> bool {
        let shared = self.command_buffer.device().shared();
        (self.command_buffer.state()).uses(shared, Some(self.encoders.clone()), objects)
    }
}

/// One of Metal's submissions of a committed command buffer with a check.
pub(super) struct Planned {
    kind: EncoderKind,
    encoders: Range<usize>,
    /// Its check ran (once at least).
    checked: bool,
    /// Its check failed it.
    failed: bool,
}

/// The submissions due for checks before a batch, each by its place among Metal's, its kind and
/// its encoders, with the program's check where it set one (`State::due`).
type Due = (
    Option<SubmissionCheck>,
    Vec<(usize, EncoderKind, Range<usize>)>,
);

/// What a committed command buffer with a check keeps for it.
pub(super) struct Checks {
    /// Metal's submissions of its ended encoders, in order, made at the commit.
    planned: Vec<Planned>,
    /// The batches whose checks ran, a prefix of them.
    batches: usize,
}

/// Metal's submissions of encoders of `kinds`, each with the event waits and signals encoded
/// before it, in order (the module docs): each submission's kind and encoders.
pub(crate) fn metal_submissions(
    encoders: impl IntoIterator<Item = (EncoderKind, usize, usize)>,
) -> Vec<(EncoderKind, Range<usize>)> {
    let mut submissions: Vec<(EncoderKind, Range<usize>)> = Vec::new();
    let mut before: Option<(EncoderKind, usize, usize)> = None;
    for (at, encoder) in encoders.into_iter().enumerate() {
        let joins =
            before.is_some_and(|before| before == encoder && encoder.0 != EncoderKind::Render);
        match submissions.last_mut() {
            Some((_, range)) if joins => range.end = at + 1,
            _ => submissions.push((encoder.0, at..at + 1)),
        }
        before = Some(encoder);
    }
    submissions
}

/// The first encoder of the submission among Metal's submissions of `encoders`
/// ([`metal_submissions`]) that holds encoder `at`: `at` itself past them.
pub(crate) fn submission_start(
    encoders: impl IntoIterator<Item = (EncoderKind, usize, usize)>,
    at: usize,
) -> usize {
    metal_submissions(encoders)
        .into_iter()
        .find(|(_, range)| range.contains(&at))
        .map_or(at, |(_, range)| range.start)
}

impl State {
    /// Metal's submissions of its ended encoders, for its check, as it commits; nothing without
    /// a check, or a decision of its first batch's submission behind a hang (`hang.rs`).
    pub(super) fn plan_checks(&mut self) {
        if self.check.is_none() && self.behind_hang.is_none() {
            return;
        }
        let encoders = self.ended.iter().map(|record| {
            (
                record.kind,
                record.event_waits_before,
                record.event_signals_before,
            )
        });
        let planned = metal_submissions(encoders)
            .into_iter()
            .map(|(kind, encoders)| Planned {
                kind,
                encoders,
                checked: false,
                failed: false,
            })
            .collect();
        self.checks = Some(Checks {
            planned,
            batches: 0,
        });
    }

    /// The submissions to check before batch `index` is submitted: each that begins in it, and
    /// each that a host step of it recorded commands for; each marked checked; with the program's
    /// check, where it set one. `None` when the batch needs no check pass.
    fn due(&mut self, index: usize) -> Option<Due> {
        let range = self.batch_range(index);
        let parts = &self.recorded[range];
        let encoders: Vec<usize> = parts.iter().map(|part| part.encoder).collect();
        let stepped: Vec<usize> = parts
            .iter()
            .filter(|part| {
                part.segments
                    .iter()
                    .any(|segment| segment.stepped && segment.recorded.is_some())
            })
            .map(|part| part.encoder)
            .collect();
        let check = self.check.clone();
        let checks = self.checks.as_mut()?;
        if checks.batches > index {
            return None;
        }
        let mut due = Vec::new();
        for (at, planned) in checks.planned.iter_mut().enumerate() {
            if planned.failed {
                continue;
            }
            let here = |list: &[usize]| list.iter().any(|e| planned.encoders.contains(e));
            if (!planned.checked && here(&encoders)) || (planned.checked && here(&stepped)) {
                planned.checked = true;
                due.push((at, planned.kind, planned.encoders.clone()));
            }
        }
        Some((check, due))
    }

    /// The check of submission `at` (`encoders`) failed it with `code`, before batch `index`
    /// is submitted: its encoders' parts from that batch on run nothing (`fail_part`), and the
    /// command buffer fails with the code unless it failed already.
    fn fail_submission(&mut self, index: usize, at: usize, encoders: &Range<usize>, code: u64) {
        if let Some(planned) = self
            .checks
            .as_mut()
            .and_then(|checks| checks.planned.get_mut(at))
        {
            planned.failed = true;
        }
        let from = self.batch_range(index).start;
        let State {
            recorded, retained, ..
        } = self;
        for part in recorded[from..]
            .iter_mut()
            .filter(|part| encoders.contains(&part.encoder))
        {
            State::fail_part(part, retained);
        }
        if self.failure.is_none() {
            self.failure = Some(CommandBufferError::Injected(code));
        }
    }

    /// Its first batch's submission, batch `index`, found a hang of its queue in flight
    /// (`hang.rs`): from that batch on every encoder runs nothing, and it completes with no
    /// error, its event signals firing, as one committed during the hang with no event wait
    /// does. No check of the program's runs for it.
    fn discard_behind_hang(&mut self, index: usize) {
        if let Some(checks) = &mut self.checks {
            for planned in &mut checks.planned {
                planned.checked = true;
                planned.failed = true;
            }
        }
        let from = self.batch_range(index).start;
        let State {
            recorded, retained, ..
        } = self;
        for part in &mut recorded[from..] {
            State::fail_part(part, retained);
        }
    }

    /// Whether the check of Metal's submission of `encoders` failed it.
    pub(super) fn check_failed(&self, encoders: &Range<usize>) -> bool {
        self.checks.as_ref().is_some_and(|checks| {
            checks
                .planned
                .iter()
                .any(|planned| planned.failed && planned.encoders == *encoders)
        })
    }

    /// Batch `index`'s checks ran.
    fn checked(&mut self, index: usize) {
        if let Some(checks) = &mut self.checks {
            checks.batches = checks.batches.max(index + 1);
        }
    }
}

impl CommandBuffer {
    /// Check each submission of the command buffer, as Metal splits it, with `check` just before
    /// the runtime submits it (the module docs of `command_buffer/checks.rs`, docs/API.md "A
    /// submission checked as it is submitted"): a runtime extension, no Metal call. Metal fails a
    /// submission that uses a resource whose memory was taken away by the time it is submitted
    /// (Apple's detach, measured by a consumer on an M3 Max, macOS 26.7): its encoders run
    /// nothing, every other submission runs, the event signals fire, the handlers run, and the
    /// command buffer completes with status `Error` and code 8. A check that answers a code
    /// does that to its submission, and [`Submission::used_resources`] says what the submission
    /// uses and reaches. Setting another check replaces it. The check runs on the device's
    /// scheduler thread, or on the thread of a [`commit_and_submit`](Self::commit_and_submit)
    /// that submits the submission ([`SubmissionCheck`]).
    ///
    /// Refused, changing nothing, once the command buffer is committed (`Invalid`,
    /// `Problem::AlreadyCommitted`).
    pub fn set_submission_check(&self, check: SubmissionCheck) -> Result<(), Error> {
        let mut state = self.state();
        if state.enqueued() || state.committing {
            return Err(invalid("set_submission_check", Problem::AlreadyCommitted));
        }
        state.check = Some(check);
        Ok(())
    }
}

impl CommandBufferShared {
    /// Whether batch `index` waits for its submission checks to run.
    pub(crate) fn needs_check(&self, index: usize) -> bool {
        let state = lock(&self.state);
        state.step_failure.is_none()
            && state
                .checks
                .as_ref()
                .is_some_and(|checks| checks.batches <= index)
    }

    /// The submission checks before batch `index` is submitted: on the scheduler's thread, or a
    /// committing one that submits the batch next (`scheduler::claim_check`), without the
    /// scheduler's lock or this command buffer's, inside a section of the exit guard, once the
    /// batch is otherwise ready and its host step recorded. Each due check runs as the program's
    /// code (`exit_guard::program`); a code it answers, or a panic, fails its submission.
    pub(crate) fn run_checks(self: &Arc<Self>, index: usize) {
        let due = {
            let mut state = lock(&self.state);
            // Committed with an event wait during a hang of its queue: applied once, here.
            let behind_hang = state.behind_hang.take();
            if behind_hang.is_some_and(|verdict| verdict.discards(&self.queue)) {
                state.discard_behind_hang(index);
                state.checked(index);
                return;
            }
            state.due(index)
        };
        let Some((Some(check), due)) = due else {
            if due.is_some() {
                lock(&self.state).checked(index);
            }
            return;
        };
        let buffer = CommandBuffer(Arc::clone(self));
        let mut failed = Vec::new();
        for (at, kind, encoders) in due {
            let submission = Submission {
                command_buffer: &buffer,
                kind,
                encoders,
            };
            let mut verdict = NonZeroU64::new(PANICKED);
            exit_guard::program(|| verdict = check(&submission));
            if let Some(code) = verdict {
                failed.push((at, submission.encoders, code.get()));
            }
        }
        let mut state = lock(&self.state);
        for (at, encoders, code) in &failed {
            state.fail_submission(index, *at, encoders, *code);
        }
        state.checked(index);
        drop(state);
        // Released without the lock: the check may hold a resource's last handle.
        drop(check);
        drop(buffer);
    }
}
