//! What a program's commit calls spend, phase by phase (`WorkCounts::commit_nanos` and kin): the
//! time a call of `CommandBuffer::commit`, `commit_and_submit` or `commit_failed` holds the
//! program's thread, and how much of it each phase took. Not part of the API.
//!
//! - **One clock a call.** Each such call runs under a [`CommitClock`], this thread's while it
//!   lasts: its whole time, and each phase's own time as the call enters it ([`phase`]). Each
//!   phase is timed apart from the phases inside it: a phase entered inside another pauses that
//!   one until it ends, so the phases' times never overlap and sum to at most the call's. A
//!   commit inside a call (a submission handler's) runs under a clock of its own, its whole time
//!   counted in the phase of the outer call that ran it.
//! - **Nowhere else.** Outside such a call (the scheduler's thread, the lowering thread, a thread
//!   that waits) a phase costs one read of this thread's clock and is timed by nothing, so the
//!   paths it marks pay nothing more. A call pays two `Instant` reads for itself and two for each
//!   phase it enters: none per draw or per command.

use std::cell::Cell;
use std::time::{Duration, Instant};

use crate::device::Device;

/// A phase of a commit call, in [`Times`]'s order (`WorkCounts::commit_lowering_nanos` and kin).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// The encoders' calls left to the device's lowering thread, lowered on this one.
    Lowering,
    /// The work its encoders left to the commit, recorded or settled for its host steps.
    Recording,
    /// That recording's waits for its draws' pipelines being made.
    PipelineWait,
    /// Taking the scheduler's lock for the pass at commit.
    LockWait,
    /// That pass, but its submissions.
    Pass,
    /// Its submissions, with the submission checks and handlers it runs.
    Submit,
}

/// How many phases there are.
const PHASES: usize = 6;

impl Phase {
    /// Its place in [`Times`].
    fn at(self) -> usize {
        match self {
            Phase::Lowering => 0,
            Phase::Recording => 1,
            Phase::PipelineWait => 2,
            Phase::LockWait => 3,
            Phase::Pass => 4,
            Phase::Submit => 5,
        }
    }
}

/// Nanoseconds a call spent in each phase, by [`Phase`]'s place.
pub(crate) type Times = [u64; PHASES];

/// A call's clock: its phases' times so far, and the phase it is in, since when.
#[derive(Clone, Copy)]
struct Clock {
    times: Times,
    current: Option<(Phase, Instant)>,
}

impl Clock {
    /// `phase` took `spent` more.
    fn add(&mut self, phase: Phase, spent: Duration) {
        self.times[phase.at()] += nanos(spent);
    }
}

thread_local! {
    /// The clock of the commit call this thread runs, where it runs one.
    static CLOCK: Cell<Option<Clock>> = const { Cell::new(None) };
}

/// `spent` in nanoseconds, saturated.
fn nanos(spent: Duration) -> u64 {
    u64::try_from(spent.as_nanos()).unwrap_or(u64::MAX)
}

/// A commit call's clock on this thread, from its making until it goes, which counts the call
/// and its phases' times in its device's work counts (the module docs).
pub(crate) struct CommitClock<'a> {
    device: &'a Device,
    started: Instant,
    /// The clock of the call this one runs inside, where it does, which it puts back.
    outer: Option<Clock>,
}

impl CommitClock<'_> {
    /// A commit call of `device`'s starts now on this thread.
    pub(crate) fn start(device: &Device) -> CommitClock<'_> {
        let fresh = Clock {
            times: [0; PHASES],
            current: None,
        };
        let outer = CLOCK.replace(Some(fresh));
        CommitClock {
            device,
            started: Instant::now(),
            outer,
        }
    }
}

impl Drop for CommitClock<'_> {
    /// The call returns: counted, with its phases' times, and the outer call's clock back.
    fn drop(&mut self) {
        let whole = nanos(self.started.elapsed());
        let clock = CLOCK.replace(self.outer.take());
        let times = clock.map_or([0; PHASES], |clock| clock.times);
        self.device.shared().counts.commit_timed(whole, &times);
    }
}

/// Run `run`, the commit call's `phase`, timed apart from the phase it runs inside, which it
/// pauses meanwhile; timed by nothing outside a commit call (the module docs).
pub(crate) fn phase<R>(phase: Phase, run: impl FnOnce() -> R) -> R {
    let Some(mut clock) = CLOCK.get() else {
        return run();
    };
    let now = Instant::now();
    let outer = clock.current.replace((phase, now));
    if let Some((was, since)) = outer {
        clock.add(was, now - since);
    }
    CLOCK.set(Some(clock));
    let ran = run();
    let end = Instant::now();
    // The clock as `run` left it: its phases inside added, `phase` current again.
    if let Some(mut clock) = CLOCK.get() {
        if let Some((was, since)) = clock.current {
            clock.add(was, end - since);
        }
        clock.current = outer.map(|(was, _)| (was, end));
        CLOCK.set(Some(clock));
    }
    ran
}
