//! The times half: what a call costs on native Metal, beside what the same call costs on our
//! stack, over the same case.
//!
//! The suite is the right place for this because **a case is one write and both legs run it from
//! the identical transcript**: [`crate::metal_side::MetalSide`] drives native Metal and
//! [`crate::runtime_side::RuntimeSide`] drives ours, from the same [`crate::case::Case`], on one
//! machine, back to back. Nothing else in the project compares the two over the same work.
//!
//! # What a difference here is, and what it is not
//!
//! **Read this before quoting a number.** On this Mac our leg is translation, then our runtime,
//! then MoltenVK, then Metal underneath; the Metal leg is Metal. A row's difference is therefore
//! the cost of **that whole chain against native Metal**, and it does **not** attribute to a layer
//! of it. The suite cannot say what MoltenVK's share is, and no figure here licenses a sentence of
//! the form "MoltenVK costs X" or "our runtime costs Y". Separating the shares needs either
//! counters inside each layer or the same case on a non-Apple driver to difference against, and
//! this round is one machine. The ratio is honest and the attribution to a layer would not be.
//!
//! What a row **does** license: this case's calls cost N times what Metal's do, here, today, over
//! this many rounds. That is the number the project has never had.
//!
//! # Three attributions, kept apart, because moving work between them is not a win
//!
//! A call that returns at once and hands its work to a thread the caller must then wait on has
//! saved nothing. So each measurement is three numbers, never one:
//!
//! - **on cpu** — the calling thread's own CPU time inside the call. What the caller burned.
//! - **blocked** — the call's elapsed time less that. What the caller waited for.
//! - **elsewhere** — CPU the rest of the process burned over the case. What was handed off.
//!
//! The first two are per call; the third is per case, because the process clock's step is a
//! microsecond and a per-call reading of it would be a number about the clock
//! ([`crate::clocks`]). `on cpu + blocked` is exactly what the caller paid at the call boundary;
//! `elsewhere` is work the machine did that the caller may pay for in a later call.
//!
//! The quantity a margin is judged on is **`wall + elsewhere` summed over the case**: everything
//! the machine spent on the case's calls, however attributed. Nothing can lower it by moving work
//! from one attribution to another, which is the whole point of keeping the three apart.
//!
//! # Reported, not gated
//!
//! A time is not assertable: this machine runs other work, and the counters half exists precisely
//! because counts are the things a gate can hold ([`crate::cost`]). So every figure here is
//! reported. [`MARGIN`] is the one judgement the plan allows — an order of magnitude, which is
//! this project's actual failure mode — and it is computed and printed as a verdict
//! ([`TimeVerdict`]) while deliberately **not** deciding any exit status yet: the measurement
//! discipline a gate needs (a quiet machine, the contrast between a case alone and in company,
//! enough rounds for a rare arm) has not been paid for times the way it was for counts. Turning it
//! on is a later, separate commit, and `README.md` says what that commit has to show first.
//!
//! # The degraded-set rule, which matters more here than for counts
//!
//! A case whose run announced a degrade notice did a **different amount of work** from Metal's
//! leg: the device lacked the exact form and the runtime did something cheaper and said so. Its
//! cost is therefore not comparable with Metal's, and the row is
//! [`TimeVerdict::Unmeasured`] rather than a fast row. Without that rule a device's honest gap
//! reads as a cost win, which is the most flattering lie the suite could tell.

use crate::watch::Watcher;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Instant;

/// The margin a row is judged against: our leg may cost an order of magnitude more than Metal's
/// before the row reads [`TimeVerdict::Over`].
///
/// Ten, and not a tighter number, for two reasons the project has evidence for. A tighter margin
/// is not measurable on a machine running other work. And an order of magnitude is the size of
/// every per-call cost bug this project has actually had: 77.3 us a draw against 2.39, and 57.8
/// against 4.11. A margin that fires on 40% would fire on load; this one fires on the shape of bug
/// we keep finding.
pub const MARGIN: u64 = 10;

/// Which leg a measurement is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Leg {
    /// Native Metal, through `objc2-metal`.
    Metal,
    /// Our stack: the translator, the runtime, and whatever Vulkan implementation the venue is.
    Ours,
}

impl Leg {
    /// The word the report and the record use.
    pub fn label(self) -> &'static str {
        match self {
            Leg::Metal => "metal",
            Leg::Ours => "ours",
        }
    }
}

impl std::fmt::Display for Leg {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(self.label())
    }
}

/// One call, measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Observation {
    /// The [`crate::case::Side`] call's name, as [`crate::watch`] names it.
    pub call: &'static str,
    /// Elapsed nanoseconds inside the call.
    pub wall: u64,
    /// The calling thread's own CPU nanoseconds inside the call. Zero where the host answers no
    /// thread clock, which [`Measurement::cpu_clock`] says rather than implying.
    pub on_cpu: u64,
}

impl Observation {
    /// The nanoseconds the caller waited rather than computed: elapsed less its own CPU time.
    ///
    /// Floored at zero. The two clocks are not read at the same instant and the thread clock's
    /// step is 125 ns, so a call that computes throughout can read a hair over its own elapsed
    /// time; that is instrument noise and never negative waiting.
    pub fn blocked(&self) -> u64 {
        self.wall.saturating_sub(self.on_cpu)
    }
}

/// What a leg cost over one case in one round.
#[derive(Clone, Debug, Default)]
pub struct CaseRound {
    /// Every call the case made, in order.
    pub observed: Vec<Observation>,
    /// Elapsed nanoseconds over the whole case, calls and the harness's work between them.
    pub window_wall: u64,
    /// The calling thread's CPU nanoseconds over the whole case.
    pub window_on_cpu: u64,
    /// The whole process's CPU nanoseconds over the whole case, every thread of it.
    pub window_process_cpu: u64,
    /// Why the run stopped before the case finished, where it did.
    pub stopped: Option<String>,
    /// The degrade-notice codes the run announced, sorted.
    pub degraded: Vec<String>,
}

impl CaseRound {
    /// CPU the process burned **off** the calling thread over the case: the handoff.
    ///
    /// Floored at zero, because the two clocks are read a few hundred nanoseconds apart.
    pub fn elsewhere(&self) -> u64 {
        self.window_process_cpu.saturating_sub(self.window_on_cpu)
    }

    /// The nanoseconds spent inside the case's calls, summed.
    pub fn in_calls(&self) -> u64 {
        self.observed.iter().map(|one| one.wall).sum()
    }

    /// What a margin is judged on: everything the machine spent on the case's calls, however
    /// attributed. See the [module docs](self).
    pub fn spend(&self) -> u64 {
        self.in_calls().saturating_add(self.elsewhere())
    }
}

/// The log a [`Timer`] writes into, shared with every span it has handed out.
#[derive(Clone, Debug, Default)]
pub struct Log {
    observed: Vec<Observation>,
}

/// A [`Watcher`] that measures each call.
///
/// It holds its log through an [`Rc`] rather than owning it, because a span must not borrow the
/// watcher: the wrapper holds the span across the forwarded call, and that call needs the wrapped
/// side mutably.
#[derive(Clone, Debug, Default)]
pub struct Timer {
    log: Rc<RefCell<Log>>,
}

impl Timer {
    /// A timer and the log it writes into.
    pub fn new() -> Self {
        Self::default()
    }

    /// Everything measured so far, and the log emptied.
    pub fn take(&self) -> Vec<Observation> {
        std::mem::take(&mut self.log.borrow_mut().observed)
    }

    /// How many calls have been measured.
    pub fn len(&self) -> usize {
        self.log.borrow().observed.len()
    }

    /// Whether nothing has been measured.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Watcher for Timer {
    type Span = Span;

    fn enter(&self, call: &'static str) -> Span {
        // Read the CPU clock first and the wall clock last, so the wall span is the inner of the
        // two and the CPU clock's own cost falls outside it rather than inside the call's elapsed
        // time. The reverse at the other end, for the same reason.
        Span {
            log: Rc::clone(&self.log),
            call,
            on_cpu: crate::clocks::thread_cpu_nanos(),
            started: Instant::now(),
        }
    }
}

/// One call's measurement, open until the call returns.
pub struct Span {
    log: Rc<RefCell<Log>>,
    call: &'static str,
    on_cpu: Option<u64>,
    started: Instant,
}

impl Drop for Span {
    fn drop(&mut self) {
        let wall = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let on_cpu = match (self.on_cpu, crate::clocks::thread_cpu_nanos()) {
            (Some(before), Some(after)) => after.saturating_sub(before),
            _ => 0,
        };
        self.log.borrow_mut().observed.push(Observation {
            call: self.call,
            wall,
            on_cpu,
        });
    }
}

/// What the instrument itself costs on this host, in nanoseconds a call, measured now.
///
/// It is the whole span machinery — both clock reads at each end and the push into the log — so it
/// is the number to subtract mentally from a very short call. It is **not** subtracted from any
/// figure ([`crate::clocks`]), and it is paid identically on both legs, so a ratio carries it top
/// and bottom.
pub fn instrument_nanos() -> u64 {
    const ROUNDS: usize = 20_000;
    let timer = Timer::new();
    let started = Instant::now();
    for _ in 0..ROUNDS {
        drop(timer.enter("calibration"));
    }
    let took = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    timer.take();
    took / ROUNDS as u64
}

/// The middle value of `values`, which is sorted in place. Zero for nothing.
///
/// The lower middle for an even count, so a median is always one of the readings taken rather than
/// a mean of two.
fn median(values: &mut [u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[(values.len() - 1) / 2]
}

/// One call's cost, over every round and every case of the selection that made it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallTimes {
    /// The [`crate::case::Side`] call.
    pub call: String,
    /// Calls measured in a round: the least and the most any round saw. The two differ where a
    /// case's work is not ordered, which is itself a fact about the measurement.
    pub least_calls: u64,
    pub most_calls: u64,
    /// The median of the rounds' own medians, in nanoseconds: elapsed, the caller's own CPU, and
    /// what it waited for. A median of medians, so one noisy round moves nothing.
    pub wall: u64,
    pub on_cpu: u64,
    pub blocked: u64,
    /// The least and the most a round's median elapsed time reached: the spread behind `wall`.
    pub wall_least: u64,
    pub wall_most: u64,
    /// The median round's **sum** over this call: every call of this name added up, in
    /// nanoseconds.
    ///
    /// A median says what one call costs and a sum says what the call costs the case, and a report
    /// that carries only the first cannot tell a 30x on a call made once from a 30x on a call made
    /// a thousand times. The paired table ratios on the sums for exactly that reason.
    pub wall_total: u64,
    pub on_cpu_total: u64,
    pub blocked_total: u64,
}

/// What a leg cost over one case, over every round.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseTimes {
    pub case: String,
    /// Calls the case made in a round, the median across rounds.
    pub calls: u64,
    /// The median round's nanoseconds inside the case's calls.
    pub in_calls: u64,
    /// The median round's three attributions, in nanoseconds. The first two are summed over the
    /// case's calls; `elsewhere` is over the whole case window ([module docs](self)).
    pub on_cpu: u64,
    pub blocked: u64,
    pub elsewhere: u64,
    /// The median round's [`CaseRound::spend`]: what a margin is judged on.
    pub spend: u64,
    /// The least and most any round's spend reached.
    pub spend_least: u64,
    pub spend_most: u64,
    /// Why a round stopped before the case finished, where one did. Its calls are a part of the
    /// case, so nothing is compared from them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped: Option<String>,
    /// The degrade-notice codes the rounds announced, sorted: how much of the case this leg did
    /// exactly, and so whether its cost is comparable at all.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degraded: Vec<String>,
    /// This case's own per-call figures.
    pub per_call: Vec<CallTimes>,
}

impl CaseTimes {
    /// Fold one case's rounds into its figures.
    pub fn of(case: &str, rounds: &[&CaseRound]) -> Self {
        let mut calls: Vec<u64> = Vec::new();
        let mut in_calls: Vec<u64> = Vec::new();
        let mut on_cpu: Vec<u64> = Vec::new();
        let mut blocked: Vec<u64> = Vec::new();
        let mut elsewhere: Vec<u64> = Vec::new();
        let mut spend: Vec<u64> = Vec::new();
        let mut stopped = None;
        let mut degraded: Vec<String> = Vec::new();
        for round in rounds {
            calls.push(round.observed.len() as u64);
            in_calls.push(round.in_calls());
            on_cpu.push(round.observed.iter().map(|one| one.on_cpu).sum());
            blocked.push(round.observed.iter().map(|one| one.blocked()).sum());
            elsewhere.push(round.elsewhere());
            spend.push(round.spend());
            // A round that stopped where another did not is a fact about the measurement, so the
            // first reason any round gave is kept rather than the last round's silence.
            if stopped.is_none() {
                stopped = round.stopped.clone();
            }
            for code in &round.degraded {
                if !degraded.contains(code) {
                    degraded.push(code.clone());
                }
            }
        }
        degraded.sort();
        let spend_least = spend.iter().copied().min().unwrap_or(0);
        let spend_most = spend.iter().copied().max().unwrap_or(0);
        Self {
            case: case.to_owned(),
            calls: median(&mut calls),
            in_calls: median(&mut in_calls),
            on_cpu: median(&mut on_cpu),
            blocked: median(&mut blocked),
            elsewhere: median(&mut elsewhere),
            spend: median(&mut spend),
            spend_least,
            spend_most,
            stopped,
            degraded,
            per_call: per_call(rounds),
        }
    }
}

/// The per-call figures over a set of rounds: each round's own median for a call, then the median
/// of those.
///
/// Pooling the rounds' observations into one list instead would let a round that made ten times as
/// many calls decide the figure, which is the shape of noise a median of medians exists to refuse.
pub fn per_call(rounds: &[&CaseRound]) -> Vec<CallTimes> {
    /// Every reading of one call within one round.
    #[derive(Default)]
    struct Readings {
        wall: Vec<u64>,
        on_cpu: Vec<u64>,
        blocked: Vec<u64>,
    }

    /// What one round says about one call: the calls it made, the middle of each reading, and
    /// what the call cost the round in total.
    struct Round {
        calls: u64,
        wall: u64,
        on_cpu: u64,
        blocked: u64,
        wall_total: u64,
        on_cpu_total: u64,
        blocked_total: u64,
    }

    let mut rounds_by_call: BTreeMap<&str, Vec<Round>> = BTreeMap::new();
    for round in rounds {
        let mut seen: BTreeMap<&str, Readings> = BTreeMap::new();
        for one in &round.observed {
            let held = seen.entry(one.call).or_default();
            held.wall.push(one.wall);
            held.on_cpu.push(one.on_cpu);
            held.blocked.push(one.blocked());
        }
        for (call, mut readings) in seen {
            rounds_by_call.entry(call).or_default().push(Round {
                calls: readings.wall.len() as u64,
                wall_total: readings.wall.iter().sum(),
                on_cpu_total: readings.on_cpu.iter().sum(),
                blocked_total: readings.blocked.iter().sum(),
                wall: median(&mut readings.wall),
                on_cpu: median(&mut readings.on_cpu),
                blocked: median(&mut readings.blocked),
            });
        }
    }
    rounds_by_call
        .into_iter()
        .map(|(call, readings)| {
            let counts: Vec<u64> = readings.iter().map(|one| one.calls).collect();
            let mut wall: Vec<u64> = readings.iter().map(|one| one.wall).collect();
            let mut on_cpu: Vec<u64> = readings.iter().map(|one| one.on_cpu).collect();
            let mut blocked: Vec<u64> = readings.iter().map(|one| one.blocked).collect();
            let mut wall_total: Vec<u64> = readings.iter().map(|one| one.wall_total).collect();
            let mut on_cpu_total: Vec<u64> = readings.iter().map(|one| one.on_cpu_total).collect();
            let mut blocked_total: Vec<u64> =
                readings.iter().map(|one| one.blocked_total).collect();
            let wall_least = wall.iter().copied().min().unwrap_or(0);
            let wall_most = wall.iter().copied().max().unwrap_or(0);
            CallTimes {
                wall_total: median(&mut wall_total),
                on_cpu_total: median(&mut on_cpu_total),
                blocked_total: median(&mut blocked_total),
                call: call.to_owned(),
                least_calls: counts.iter().copied().min().unwrap_or(0),
                most_calls: counts.iter().copied().max().unwrap_or(0),
                wall: median(&mut wall),
                on_cpu: median(&mut on_cpu),
                blocked: median(&mut blocked),
                wall_least,
                wall_most,
            }
        })
        .collect()
}

/// A whole leg's measurement over a selection.
#[derive(Clone, Debug)]
pub struct Measurement {
    pub leg: Leg,
    /// The venue the leg ran on ([`crate::case::Venue::label`]).
    pub venue: String,
    /// The rounds taken.
    pub rounds: u32,
    /// What the instrument itself cost on this host, in nanoseconds a call: the middle of the two
    /// readings [`Measurement::calibration`] holds.
    pub instrument: u64,
    /// What the instrument cost before this leg ran and again after it, which is what the gate
    /// judges the measurement's own conditions by
    /// ([`crate::slow_calls::Calibration`]).
    pub calibration: crate::slow_calls::Calibration,
    /// Whether the host answered the thread CPU clock. Where it did not, `on_cpu` reads zero
    /// everywhere and `blocked` equals elapsed, and the report says so rather than letting a
    /// reader take the zeros for a fact.
    pub cpu_clock: bool,
    /// Per case, in the selection's order.
    pub cases: Vec<CaseTimes>,
    /// Per call, pooled over the whole selection: the table a reader acts on.
    pub calls: Vec<CallTimes>,
}

impl Measurement {
    /// One case's figures, where the selection held it.
    pub fn case(&self, name: &str) -> Option<&CaseTimes> {
        self.cases.iter().find(|one| one.case == name)
    }

    /// One call's pooled figures, where the selection made it.
    pub fn call(&self, name: &str) -> Option<&CallTimes> {
        self.calls.iter().find(|one| one.call == name)
    }
}

/// How one case's cost stands against Metal's over the same case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TimeVerdict {
    /// Our leg's spend is within [`MARGIN`] times Metal's.
    Within { ours: u64, metal: u64 },
    /// Our leg's spend is over [`MARGIN`] times Metal's: an order of magnitude, the shape of every
    /// per-call cost bug this project has had.
    Over { ours: u64, metal: u64 },
    /// The two legs cannot be compared, and why.
    Unmeasured { why: String },
}

impl TimeVerdict {
    /// The word the report prints.
    pub fn word(&self) -> &'static str {
        match self {
            TimeVerdict::Within { .. } => "within",
            TimeVerdict::Over { .. } => "over",
            TimeVerdict::Unmeasured { .. } => "unmeasured",
        }
    }

    /// How many times our leg's spend is Metal's, where both were measured.
    pub fn ratio(&self) -> Option<f64> {
        match self {
            TimeVerdict::Within { ours, metal } | TimeVerdict::Over { ours, metal } => {
                (*metal > 0).then(|| *ours as f64 / *metal as f64)
            }
            TimeVerdict::Unmeasured { .. } => None,
        }
    }
}

/// Judge one case's two legs against each other.
///
/// Every reason a row cannot be compared is checked before the margin, and each is its own
/// sentence in the report:
///
/// - either leg's run **stopped** before the case finished, so its calls are a part of the case;
/// - our leg **announced a degrade notice**, so it did a different amount of work from Metal's and
///   a lower cost is the device's gap rather than our saving ([module docs](self));
/// - the Metal leg announced one, which cannot happen today and would mean the same thing;
/// - Metal's spend reads **zero**, which is a measurement that did not happen rather than a free
///   case.
pub fn judge(metal: &CaseTimes, ours: &CaseTimes) -> TimeVerdict {
    if let Some(why) = &metal.stopped {
        return TimeVerdict::Unmeasured {
            why: format!("the Metal leg stopped before the case finished: {why}"),
        };
    }
    if let Some(why) = &ours.stopped {
        return TimeVerdict::Unmeasured {
            why: format!("our leg stopped before the case finished: {why}"),
        };
    }
    if !ours.degraded.is_empty() {
        return TimeVerdict::Unmeasured {
            why: format!(
                "our leg announced {}, so it did a different amount of work from Metal's and its \
                 cost is not Metal's to compare with",
                ours.degraded.join(", ")
            ),
        };
    }
    if !metal.degraded.is_empty() {
        return TimeVerdict::Unmeasured {
            why: format!(
                "the Metal leg announced {}, so the two legs did different work",
                metal.degraded.join(", ")
            ),
        };
    }
    if metal.spend == 0 {
        return TimeVerdict::Unmeasured {
            why: "the Metal leg's spend reads zero, which is a measurement that did not happen \
                  rather than a case Metal answers for nothing"
                .to_owned(),
        };
    }
    let (ours, metal) = (ours.spend, metal.spend);
    if ours > metal.saturating_mul(MARGIN) {
        TimeVerdict::Over { ours, metal }
    } else {
        TimeVerdict::Within { ours, metal }
    }
}

/// One leg's cost for a case, as its recording keeps it.
///
/// **Why a recording holds this at all**: Metal's own cost on a given macOS build is the one half
/// that cannot be re-taken later. A measurement of ours is a command away on any host; what Metal
/// charged for a `setBytes:` on 26A434 is gone the moment the machine updates. That is the plan's
/// standing property — a year of Metal measurements should stop evaporating — applied to cost.
///
/// It is **not** a Metal fact anything compares against: a stored figure and a live one come from
/// different moments and, off this Mac, different hardware. A comparison is only ever made between
/// two legs measured in the same run ([`judge`]), and the stored record is history.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeRecord {
    /// Which leg ([`Leg`]).
    pub leg: Leg,
    /// The venue the leg ran on.
    pub venue: String,
    /// The chip, as the suite names it ([`crate::recording::ORACLE_CHIP`]).
    pub chip: String,
    /// The macOS build. Two records are of one machine only where these two fields agree.
    pub os_build: String,
    /// The rounds behind the figures.
    pub rounds: u32,
    /// The measurement that justified the figures, in words: what was run, where, and how the
    /// figures were taken from it. Every bound in this suite names its measurement, and a time is
    /// held to that rule too.
    pub measured: String,
    /// What the instrument cost on the host that took the measurement, in nanoseconds a call.
    pub instrument: u64,
    /// The case's figures.
    pub times: CaseTimes,
}

/// Fold a leg's rounds into its measurement.
///
/// `taken` is one entry per round, each holding one [`CaseRound`] per case of the selection, in the
/// selection's order. A case's figures are the median across rounds of its own per-round figures.
///
/// `calibration` is what [`instrument_nanos`] read before this leg's rounds and again after them,
/// bracketed rather than sampled once because a leg is minutes long and a reading from its first
/// five milliseconds says nothing about the rest ([`crate::slow_calls::Calibration`]).
///
/// The pooled per-call table ([`Measurement::calls`]) is taken over the cases this leg did
/// **exactly, start to finish**: a round that stopped part way did a part of the case, and a round
/// that announced a notice did something cheaper than the exact form, so neither belongs in a
/// figure a reader will read as the cost of a call. Each row of the report's paired table prints
/// both legs' call counts, so a row whose two legs pooled different work is visible rather than
/// implied.
pub fn fold(
    leg: Leg,
    venue: String,
    rounds: u32,
    calibration: crate::slow_calls::Calibration,
    cases: &[Box<dyn crate::case::Case>],
    taken: &[Vec<CaseRound>],
) -> Measurement {
    let mut folded = Vec::with_capacity(cases.len());
    for (index, case) in cases.iter().enumerate() {
        let of_case: Vec<&CaseRound> = taken.iter().filter_map(|round| round.get(index)).collect();
        folded.push(CaseTimes::of(case.name(), &of_case));
    }
    let pooled: Vec<CaseRound> = taken
        .iter()
        .map(|round| CaseRound {
            observed: round
                .iter()
                .filter(|one| one.stopped.is_none() && one.degraded.is_empty())
                .flat_map(|one| one.observed.iter().copied())
                .collect(),
            ..Default::default()
        })
        .collect();
    Measurement {
        leg,
        venue,
        rounds,
        instrument: calibration.typical().unwrap_or_default(),
        calibration,
        cpu_clock: crate::clocks::thread_cpu_nanos().is_some(),
        cases: folded,
        calls: per_call(&pooled.iter().collect::<Vec<_>>()),
    }
}

/// One call's cost on one leg, added up over a set of cases.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallTotal {
    /// The calls of this name those cases made, in the median round.
    pub calls: u64,
    /// What they cost in total, in nanoseconds, and how that splits.
    pub wall: u64,
    pub on_cpu: u64,
    pub blocked: u64,
    /// The middle case's own median for this call: what one such call costs, as against what they
    /// cost together.
    pub typical: u64,
}

/// One call, on both legs, over the cases **both legs did exactly**.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paired {
    pub call: String,
    pub metal: CallTotal,
    pub ours: CallTotal,
}

impl Paired {
    /// How many times our leg's total is Metal's, where Metal's is not zero.
    pub fn ratio(&self) -> Option<f64> {
        (self.metal.wall > 0).then(|| self.ours.wall as f64 / self.metal.wall as f64)
    }
}

/// How many cases both legs did exactly, so the paired table's denominator is printed rather
/// than implied.
pub fn comparable(metal: &Measurement, ours: &Measurement) -> usize {
    ours.cases
        .iter()
        .filter(|one| {
            metal
                .case(&one.case)
                .is_some_and(|theirs| !matches!(judge(theirs, one), TimeVerdict::Unmeasured { .. }))
        })
        .count()
}

/// The per-call table, paired over the cases both legs did exactly.
///
/// **Why the pairing is not optional.** Each leg's own pooled table is over the cases *that leg*
/// finished without a notice, and the two sets are not the same: 107 of this suite's 383 cases
/// announce a notice on our leg and none do on Metal's. A table built from the two pools compares
/// `render_pipeline` over 132 Metal calls against 85 of ours — a ratio of different work, printed
/// with the authority of a measurement. Measured on this Mac, not supposed: that is what the first
/// whole-set run produced, and it is why this function exists.
///
/// So a call's row is summed over the cases where [`judge`] returns a verdict at all. Both legs
/// then cover one case set, and the row's two call counts are printed anyway, because a case whose
/// two legs make a call a different number of times is still comparing different work and only the
/// counts can say so.
///
/// The ratio is on the **totals**, so a call is weighted by how often it runs: a 30x on a call
/// made once is not the same finding as a 30x on a call made a thousand times.
pub fn paired(metal: &Measurement, ours: &Measurement) -> Vec<Paired> {
    let mut by_call: BTreeMap<String, (Vec<&CallTimes>, Vec<&CallTimes>)> = BTreeMap::new();
    for one in &ours.cases {
        let Some(theirs) = metal.case(&one.case) else {
            continue;
        };
        if matches!(judge(theirs, one), TimeVerdict::Unmeasured { .. }) {
            continue;
        }
        for call in &one.per_call {
            by_call.entry(call.call.clone()).or_default().1.push(call);
        }
        for call in &theirs.per_call {
            by_call.entry(call.call.clone()).or_default().0.push(call);
        }
    }
    let total = |of: &[&CallTimes]| {
        let mut typical: Vec<u64> = of.iter().map(|one| one.wall).collect();
        CallTotal {
            calls: of.iter().map(|one| one.most_calls).sum(),
            wall: of.iter().map(|one| one.wall_total).sum(),
            on_cpu: of.iter().map(|one| one.on_cpu_total).sum(),
            blocked: of.iter().map(|one| one.blocked_total).sum(),
            typical: median(&mut typical),
        }
    };
    by_call
        .into_iter()
        .map(|(call, (theirs, ours))| Paired {
            call,
            metal: total(&theirs),
            ours: total(&ours),
        })
        .collect()
}

/// The layers under our runtime on this host, named so the honest-framing line is true wherever
/// the suite runs.
///
/// It was not, once: the line read "MoltenVK and Metal underneath" on every venue, and the first
/// run in the Linux container printed it over llvmpipe, where neither is present. A caution that
/// overstates the stack is the same defect as a figure that overstates a result.
fn chain(venue: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("MoltenVK over Metal on {venue}")
    } else {
        format!("the venue's own Vulkan implementation, {venue}")
    }
}

/// Microseconds, three places, from nanoseconds: the unit a per-call figure reads in.
fn us(nanos: u64) -> String {
    format!("{:.3}", nanos as f64 / 1_000.0)
}

/// Milliseconds, three places, from nanoseconds: the unit a per-case figure reads in.
fn ms(nanos: u64) -> String {
    format!("{:.3}", nanos as f64 / 1_000_000.0)
}

/// Where one call's cost sits, case by case, worst first: the triage for a row of the per-call
/// table.
///
/// **Why this exists.** The first whole-set run put `replace_region` at 105x over 108 calls, and
/// running the three cases that obviously make it gave 1.2x over 3 — so the figure was real and
/// almost none of it was there. A per-call total over a case set says *that* a call is expensive
/// and never *where*, and the counters half's hardest lesson is that a reading over one selection
/// is not evidence about another. Without this a reader has only the bisection by hand that the
/// plan's UX bar exists to abolish.
///
/// Only the cases both legs did exactly are listed, for the same reason the paired table lists
/// only those.
pub fn render_one_call(metal: &Measurement, ours: &Measurement, call: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\nWHERE `{call}` COSTS, case by case over the cases both legs did exactly, worst ratio \
         of totals first (us)\n"
    ));
    out.push_str(&format!(
        "{:<44} {:>7} {:>11} {:>11} {:>8}   {:>11} {:>11}\n",
        "case", "calls", "metal tot", "ours tot", "ratio", "ours cpu", "ours blkd"
    ));
    let mut rows: Vec<(&str, u64, &CallTimes, Option<&CallTimes>)> = Vec::new();
    for one in &ours.cases {
        let Some(theirs) = metal.case(&one.case) else {
            continue;
        };
        if matches!(judge(theirs, one), TimeVerdict::Unmeasured { .. }) {
            continue;
        }
        let Some(mine) = one.per_call.iter().find(|held| held.call == call) else {
            continue;
        };
        rows.push((
            &one.case,
            mine.most_calls,
            mine,
            theirs.per_call.iter().find(|held| held.call == call),
        ));
    }
    rows.sort_by_key(|(_, _, mine, theirs)| {
        let metal = theirs.map(|theirs| theirs.wall_total).unwrap_or(0).max(1);
        std::cmp::Reverse(mine.wall_total.saturating_mul(1_000) / metal)
    });
    for (case, calls, mine, theirs) in &rows {
        let (their_total, ratio) = match theirs {
            Some(theirs) if theirs.wall_total > 0 => (
                us(theirs.wall_total),
                format!("{:.1}x", mine.wall_total as f64 / theirs.wall_total as f64),
            ),
            Some(theirs) => (us(theirs.wall_total), "-".to_owned()),
            None => ("-".to_owned(), "-".to_owned()),
        };
        out.push_str(&format!(
            "{:<44} {:>7} {:>11} {:>11} {:>8}   {:>11} {:>11}\n",
            case,
            calls,
            their_total,
            us(mine.wall_total),
            ratio,
            us(mine.on_cpu_total),
            us(mine.blocked_total)
        ));
    }
    if rows.is_empty() {
        out.push_str("  no case both legs did exactly makes that call\n");
    }
    out
}

/// The table a `times` run prints: native Metal beside our stack, per call and per case.
///
/// `metal` is absent off macOS, where there is no Metal leg to compare with; the table then
/// reports our own figures alone and says so, which is the honest shape for a venue the suite can
/// measure but has no oracle on.
pub fn render(metal: Option<&Measurement>, ours: &Measurement, show: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "times: our stack on {} against native Metal, over {} round(s)\n",
        ours.venue, ours.rounds
    ));
    out.push_str(&format!(
        "the instrument costs {} us a call on this host, paid identically on both legs and never \
         subtracted from a figure\n",
        us(ours.instrument)
    ));
    if !ours.cpu_clock {
        out.push_str(
            "this host answers no thread CPU clock, so the `on cpu` column reads zero everywhere \
             and `blocked` is the whole elapsed time: an attribution that was not measured, not \
             one that is zero\n",
        );
    }
    out.push_str(&format!(
        "\nWHAT A DIFFERENCE HERE IS: our leg is translation, then our runtime, then {}.\nA ratio \
         is the cost of that whole chain against native Metal and attributes to no layer of \
         it.\nNothing here licenses a sentence about what any one of those layers costs on its \
         own.\n",
        chain(&ours.venue)
    ));

    let Some(metal) = metal else {
        out.push_str("\nno Metal leg on this host, so every row below is ours alone\n\n");
        out.push_str(&format!(
            "{:<38} {:>7} {:>11} {:>11} {:>11}\n",
            "call", "calls", "wall us", "on cpu us", "blocked us"
        ));
        for one in &ours.calls {
            out.push_str(&format!(
                "{:<38} {:>7} {:>11} {:>11} {:>11}\n",
                one.call,
                one.most_calls,
                us(one.wall),
                us(one.on_cpu),
                us(one.blocked)
            ));
        }
        return out;
    };

    // Per call, paired over the cases both legs did exactly, worst ratio first. The question a
    // reader brings is how much more a call costs us; the counts are printed for both legs
    // because a row whose legs make a call a different number of times is comparing different
    // work and nothing but the counts can say so.
    //
    // Measured, not supposed: on this Mac `library` costs Metal about 23 ms and us about 0.1 ms,
    // while `compute_pipeline` costs Metal 0.09 ms and us 2.1 ms. Neither row is a result. Metal
    // compiles the AIR when the library is made and we translate it when the pipeline is made, so
    // the *same work* sits at a different call on each leg, and the two rows are only honest read
    // together. That is the per-call table's standing trap and it belongs in its heading.
    let rows = {
        let mut rows = paired(metal, ours);
        rows.sort_by(|one, two| {
            let ratio = |row: &Paired| row.ratio().unwrap_or(f64::INFINITY);
            ratio(two)
                .partial_cmp(&ratio(one))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows
    };
    out.push_str(&format!(
        "\nPER CALL, over the {} case(s) BOTH legs did exactly, worst ratio of totals first\n",
        comparable(metal, ours)
    ));
    out.push_str(
        "  a per-call ratio is only honest read beside the per-case total: the two legs put the \
         same work at different\n  calls, and the measured example is right here -- Metal \
         compiles at `library`, we translate at `compute_pipeline`\n",
    );
    // The reproducibility of the thing being printed, beside it rather than in a report nobody
    // opens. Measured over the two whole-set passes this tree has recorded.
    out.push_str(
        "  A RATIO HERE IS NOT REPRODUCIBLE TO BETTER THAN A FACTOR THE MARGIN CARES ABOUT. Over \
         the two whole-set passes this\n  tree has recorded, the 98 rows common to both moved a \
         median 2.62x, 14.96x at the 90th percentile and 63.9x at\n  worst; the rows resolved \
         above four instrument costs moved a median 1.87x and 6.92x at the 90th. The cause is \
         the\n  CPU the measurement ran on: one binary and one case set inside one minute read \
         38.8x on a call at the ordinary\n  scheduling class and 75.5x under a background one, \
         with the load average unmoved. The gate below withholds a\n  verdict where it cannot \
         stand behind one; a row of this table on its own is a figure to localise, not to quote.\n",
    );
    // A reader meeting a ratio hugely in our favour needs the reason beside it, not in a report
    // they may never open. Both sentences are measurements of this tree, not cautions in general.
    out.push_str(
        "  A RATIO FAR IN OUR FAVOUR IS A SETUP DIFFERENCE UNTIL SOMEBODY SHOWS OTHERWISE. The \
         suite's recordings make 733\n  shader references over 152 distinct (shader, entry) \
         pairs, so 90% are to a shader another case already used,\n  and our leg has a shader \
         cache while Metal's newLibraryWithData recompiles every call. Our counters sum to 518\n           shader modules and 1,040 pipelines for 122 ms of measured compile -- about 78 us each, \
         which is not a price at\n  which any compile to machine code happens, so something below \
         us is handing back work already done. Use\n  --cold-shaders for the only setting under \
         which the compile rows compare like with like.\n",
    );
    out.push_str(&format!(
        "{:<34} {:>7} {:>10} {:>10} {:>7} {:>10} {:>10} {:>7}   {:>9} {:>9}\n",
        "call",
        "metal n",
        "metal tot",
        "metal one",
        "ours n",
        "ours tot",
        "ours one",
        "ratio",
        "ours cpu",
        "ours blkd"
    ));
    for row in &rows {
        let ratio = row
            .ratio()
            .map(|ratio| format!("{ratio:.1}x"))
            .unwrap_or_else(|| "-".to_owned());
        out.push_str(&format!(
            "{:<34} {:>7} {:>10} {:>10} {:>7} {:>10} {:>10} {:>7}   {:>9} {:>9}\n",
            row.call,
            row.metal.calls,
            us(row.metal.wall),
            us(row.metal.typical),
            row.ours.calls,
            us(row.ours.wall),
            us(row.ours.typical),
            ratio,
            us(row.ours.on_cpu),
            us(row.ours.blocked)
        ));
    }

    // Per case, the three attributions kept apart. A case is judged on `spend`, which is
    // everything the machine spent on its calls however attributed, so nothing here can be
    // improved by moving work from one column to another.
    out.push_str(
        "\nPER CASE, the three attributions apart, in ms; `spend` is wall-in-calls + elsewhere \
         and is what the margin judges\n",
    );
    out.push_str(&format!(
        "{:<42} {:>8} {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8} {:>8} {:>7}\n",
        "case",
        "m spend",
        "m cpu",
        "m blkd",
        "m elsew",
        "o spend",
        "o cpu",
        "o blkd",
        "o elsew",
        "ratio"
    ));
    let mut within = 0;
    let mut over = 0;
    let mut unmeasured = 0;
    let mut notes: Vec<String> = Vec::new();
    for one in &ours.cases {
        let verdict = match metal.case(&one.case) {
            Some(theirs) => judge(theirs, one),
            None => TimeVerdict::Unmeasured {
                why: "the Metal leg holds no figures for the case".to_owned(),
            },
        };
        match &verdict {
            TimeVerdict::Within { .. } => within += 1,
            TimeVerdict::Over { .. } => over += 1,
            TimeVerdict::Unmeasured { why } => {
                unmeasured += 1;
                notes.push(format!("{}: unmeasured, {why}", one.case));
            }
        }
        let interesting = !matches!(verdict, TimeVerdict::Within { .. });
        if !show && !interesting {
            continue;
        }
        let theirs = metal.case(&one.case);
        let column = |of: Option<u64>| of.map(ms).unwrap_or_else(|| "-".to_owned());
        let ratio = verdict
            .ratio()
            .map(|ratio| format!("{ratio:.1}x"))
            .unwrap_or_else(|| verdict.word().to_owned());
        out.push_str(&format!(
            "{:<42} {:>8} {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8} {:>8} {:>7}\n",
            one.case,
            column(theirs.map(|theirs| theirs.spend)),
            column(theirs.map(|theirs| theirs.on_cpu)),
            column(theirs.map(|theirs| theirs.blocked)),
            column(theirs.map(|theirs| theirs.elsewhere)),
            ms(one.spend),
            ms(one.on_cpu),
            ms(one.blocked),
            ms(one.elsewhere),
            ratio
        ));
    }
    if !show && (within > 0) {
        out.push_str(&format!(
            "({within} case(s) within the margin are not listed; pass --show for every row)\n"
        ));
    }
    for note in &notes {
        out.push_str(&format!("  {note}\n"));
    }
    out.push_str(&format!(
        "\n{within} within {MARGIN}x, {over} over it, {unmeasured} unmeasured, over {} round(s)\n",
        ours.rounds
    ));
    out.push_str(
        "no exit status turns on any of this: a time is reported, never gated. README.md says what \
         the commit that gates it has to show first.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A case's figures with its three attributions split however the test wants, folded through
    /// the real aggregation.
    ///
    /// **It goes through [`CaseRound`] and [`CaseTimes::of`] on purpose.** Setting the fields of a
    /// `CaseTimes` by hand would make every test below a test of nothing but `judge` reading one
    /// field: the first version of this helper computed `spend` itself, and the guard that the
    /// margin cannot be dodged by moving work then **passed with a `spend` that ignored the
    /// handoff entirely**. Caught by breaking it, which is the only way such a test is ever
    /// caught.
    fn case_of(case: &str, on_cpu: u64, blocked: u64, elsewhere: u64) -> CaseTimes {
        let round = CaseRound {
            // One call that burned `on_cpu` and waited `blocked`.
            observed: vec![seen("the_call", on_cpu + blocked, on_cpu)],
            window_wall: on_cpu + blocked,
            window_on_cpu: on_cpu,
            // The process burned the calling thread's own CPU plus `elsewhere` on other threads.
            window_process_cpu: on_cpu + elsewhere,
            stopped: None,
            degraded: Vec::new(),
        };
        CaseTimes::of(case, &[&round])
    }

    /// One call, measured, for the aggregation tests.
    fn seen(call: &'static str, wall: u64, on_cpu: u64) -> Observation {
        Observation { call, wall, on_cpu }
    }

    /// A median is one of the readings taken, and the lower of the two middles for an even count,
    /// so no figure in a report is a number nobody measured.
    #[test]
    fn a_median_is_a_reading_and_never_a_mean() {
        assert_eq!(median(&mut []), 0);
        assert_eq!(median(&mut [5]), 5);
        assert_eq!(median(&mut [9, 1]), 1);
        assert_eq!(median(&mut [9, 1, 5]), 5);
        assert_eq!(median(&mut [1, 2, 3, 4]), 2);
    }

    /// **The property the whole three-way split exists for**: the margin is judged on the case's
    /// whole spend, so moving work from the caller's thread to our own threads, or from computing
    /// to waiting, changes no verdict.
    ///
    /// Without this a call that returns at once and queues 50 us of scheduler work would score as
    /// a win, which is the known handoff problem the plan names.
    #[test]
    fn moving_work_between_the_attributions_changes_no_verdict() {
        let metal = case_of("m", 100, 0, 0);
        assert_eq!(metal.spend, 100, "the fold did not take Metal's own spend");
        let splits = [
            case_of("o", 900, 0, 0),
            case_of("o", 0, 900, 0),
            case_of("o", 0, 0, 900),
            case_of("o", 300, 300, 300),
            case_of("o", 450, 0, 450),
        ];
        for ours in &splits {
            assert_eq!(
                ours.spend, 900,
                "a split of {}/{}/{} did not fold to the same spend, so the verdict below would \
                 be comparing different totals",
                ours.on_cpu, ours.blocked, ours.elsewhere
            );
            assert_eq!(
                judge(&metal, ours),
                TimeVerdict::Within {
                    ours: 900,
                    metal: 100
                },
                "a split of {}/{}/{} changed the verdict",
                ours.on_cpu,
                ours.blocked,
                ours.elsewhere
            );
        }
    }

    /// The margin is an order of magnitude, and the boundary is inclusive: ten times Metal is
    /// within, one nanosecond more is over.
    #[test]
    fn the_margin_is_an_order_of_magnitude() {
        let metal = case_of("m", 100, 0, 0);
        assert_eq!(
            judge(&metal, &case_of("o", 1_000, 0, 0)),
            TimeVerdict::Within {
                ours: 1_000,
                metal: 100
            }
        );
        assert_eq!(
            judge(&metal, &case_of("o", 1_001, 0, 0)),
            TimeVerdict::Over {
                ours: 1_001,
                metal: 100
            }
        );
        assert_eq!(MARGIN, 10);
    }

    /// A leg that announced a degrade notice did a **different amount of work**, so its cost is
    /// not Metal's to compare with. This is the rule that stops a device's honest gap reading as a
    /// cost win, and it fires before the margin.
    #[test]
    fn a_degraded_leg_is_unmeasured_rather_than_fast() {
        let metal = case_of("m", 1_000_000, 0, 0);
        let mut ours = case_of("o", 1, 0, 0);
        ours.degraded = vec!["some-notice".to_owned()];
        let verdict = judge(&metal, &ours);
        match &verdict {
            TimeVerdict::Unmeasured { why } => {
                assert!(why.contains("some-notice"), "{why}");
                assert!(why.contains("different amount of work"), "{why}");
            }
            other => panic!("a degraded leg a millionth of Metal's cost read {other:?}"),
        }
        assert_eq!(verdict.ratio(), None);
    }

    /// A run that stopped part way did a part of the case, on either leg, so nothing is compared
    /// from it — and the leg that stopped is named, because the two mean different things to fix.
    #[test]
    fn a_stopped_leg_is_unmeasured_and_says_which_one() {
        let good = case_of("m", 100, 0, 0);
        let mut stopped = case_of("o", 1, 0, 0);
        stopped.stopped = Some("the venue lacks it".to_owned());
        let ours = judge(&good, &stopped);
        assert!(
            matches!(&ours, TimeVerdict::Unmeasured { why } if why.starts_with("our leg stopped")),
            "{ours:?}"
        );
        let metal = judge(&stopped, &good);
        assert!(
            matches!(&metal, TimeVerdict::Unmeasured { why } if why.starts_with("the Metal leg stopped")),
            "{metal:?}"
        );
    }

    /// Metal costing nothing is a measurement that did not happen, never a free case, so it is
    /// never the denominator of a ratio.
    #[test]
    fn a_metal_leg_of_zero_is_unmeasured_rather_than_infinitely_worse() {
        let verdict = judge(&case_of("m", 0, 0, 0), &case_of("o", 500, 0, 0));
        assert!(
            matches!(&verdict, TimeVerdict::Unmeasured { why } if why.contains("reads zero")),
            "{verdict:?}"
        );
    }

    /// CPU read off the calling thread floors at zero: the process and thread clocks are read a
    /// few hundred nanoseconds apart, and a negative handoff is instrument noise rather than a
    /// fact.
    #[test]
    fn a_negative_handoff_reads_as_none_rather_than_wrapping() {
        let round = CaseRound {
            window_on_cpu: 500,
            window_process_cpu: 100,
            ..Default::default()
        };
        assert_eq!(round.elsewhere(), 0);
        let one = seen("library", 10, 400);
        assert_eq!(one.blocked(), 0);
    }

    /// The per-call figure is a median of the **rounds'** medians, so one wild round moves it by
    /// nothing. Pooling every observation into one list instead would let the round that made the
    /// most calls decide the figure.
    #[test]
    fn a_wild_round_does_not_move_a_per_call_figure() {
        let quiet = |wall| CaseRound {
            observed: vec![seen("dispatch", wall, wall)],
            ..Default::default()
        };
        // Four honest rounds at 100 ns and one round of a thousand calls at 100 us.
        let wild = CaseRound {
            observed: (0..1_000).map(|_| seen("dispatch", 100_000, 0)).collect(),
            ..Default::default()
        };
        let rounds = [quiet(100), quiet(100), quiet(100), quiet(100), wild];
        let folded = per_call(&rounds.iter().collect::<Vec<_>>());
        let [one] = folded.as_slice() else {
            panic!("one call: {folded:?}");
        };
        assert_eq!(one.wall, 100, "the wild round decided the figure");
        // The spread is still reported, so the wild round is visible rather than discarded.
        assert_eq!((one.wall_least, one.wall_most), (100, 100_000));
        assert_eq!((one.least_calls, one.most_calls), (1, 1_000));
    }

    /// A case's figure is the **median** round and its spread is the least and most any round
    /// reached, so a reader can see how settled the measurement is.
    ///
    /// The spread is the times half's version of the counters half's hardest lesson: a reading
    /// that moves between two runs of an unchanged tree is not a fact, and the only way a report
    /// can say so is by carrying both ends.
    #[test]
    fn a_cases_figure_is_the_median_round_and_carries_its_spread() {
        let round = |on_cpu: u64| CaseRound {
            observed: vec![seen("the_call", on_cpu, on_cpu)],
            window_wall: on_cpu,
            window_on_cpu: on_cpu,
            window_process_cpu: on_cpu,
            stopped: None,
            degraded: Vec::new(),
        };
        let rounds = [round(10), round(900), round(20), round(30), round(40)];
        let folded = CaseTimes::of("o", &rounds.iter().collect::<Vec<_>>());
        assert_eq!(folded.spend, 30, "the figure is not the median round");
        assert_eq!(
            (folded.spend_least, folded.spend_most),
            (10, 900),
            "the spread does not carry the rounds the measurement actually saw"
        );
        assert_eq!(folded.calls, 1);
    }

    /// A round that stopped, and the notices any round announced, both survive the fold: a
    /// measurement whose rounds did different amounts of work must not read as one clean figure.
    #[test]
    fn a_fold_keeps_the_reasons_a_round_was_not_comparable() {
        let clean = CaseRound {
            observed: vec![seen("the_call", 10, 10)],
            ..Default::default()
        };
        let stopped = CaseRound {
            observed: vec![seen("the_call", 10, 10)],
            stopped: Some("the venue lacks it".to_owned()),
            degraded: vec!["second-code".to_owned()],
            ..Default::default()
        };
        let noticed = CaseRound {
            observed: vec![seen("the_call", 10, 10)],
            degraded: vec!["first-code".to_owned()],
            ..Default::default()
        };
        let folded = CaseTimes::of("o", &[&clean, &stopped, &noticed]);
        assert_eq!(folded.stopped.as_deref(), Some("the venue lacks it"));
        assert_eq!(folded.degraded, ["first-code", "second-code"]);
    }

    /// A measurement of one leg over the cases given, for the pairing test.
    fn measurement(leg: Leg, cases: Vec<CaseTimes>) -> Measurement {
        Measurement {
            leg,
            venue: "test".to_owned(),
            rounds: 1,
            instrument: 1,
            calibration: crate::slow_calls::Calibration {
                before: 1,
                after: 1,
            },
            cpu_clock: true,
            cases,
            calls: Vec::new(),
        }
    }

    /// A case's figures from a given number of calls of one name, each costing `each`.
    fn calling(case: &str, call: &'static str, times: usize, each: u64) -> CaseTimes {
        let round = CaseRound {
            observed: (0..times).map(|_| seen(call, each, each)).collect(),
            window_wall: each * times as u64,
            window_on_cpu: each * times as u64,
            window_process_cpu: each * times as u64,
            stopped: None,
            degraded: Vec::new(),
        };
        CaseTimes::of(case, &[&round])
    }

    /// **The per-call table is paired over the cases both legs did exactly.**
    ///
    /// This is the guard for a flaw the first whole-set run actually had: each leg's own pool was
    /// over the cases *that leg* finished cleanly, and because 107 of this suite's cases announce
    /// a notice on our leg and none do on Metal's, the table compared `render_pipeline` over 132
    /// Metal calls against 85 of ours and printed a ratio of different work.
    #[test]
    fn a_case_only_one_leg_did_exactly_is_left_out_of_the_per_call_table() {
        let clean_metal = calling("clean", "render_pipeline", 1, 100);
        let clean_ours = calling("clean", "render_pipeline", 1, 200);
        let noisy_metal = calling("noisy", "render_pipeline", 1_000, 100);
        let mut noisy_ours = calling("noisy", "render_pipeline", 1_000, 100);
        noisy_ours.degraded = vec!["a-notice".to_owned()];
        let metal = measurement(Leg::Metal, vec![clean_metal, noisy_metal]);
        let ours = measurement(Leg::Ours, vec![clean_ours, noisy_ours]);
        assert_eq!(
            comparable(&metal, &ours),
            1,
            "the degraded case was compared"
        );
        let rows = paired(&metal, &ours);
        let [row] = rows.as_slice() else {
            panic!("one call: {rows:?}");
        };
        assert_eq!(row.call, "render_pipeline");
        assert_eq!(
            (row.metal.calls, row.ours.calls),
            (1, 1),
            "the 1,000 calls of a case only one leg did exactly reached the paired table"
        );
        assert_eq!((row.metal.wall, row.ours.wall), (100, 200));
        assert_eq!(row.ratio(), Some(2.0));
    }

    /// The ratio is on the totals, so a call is weighted by how often it runs. A 30x on a call
    /// made once must not read the same as a 30x on a call made a thousand times.
    #[test]
    fn the_per_call_ratio_weights_a_call_by_how_often_it_runs() {
        // A case of its own for each call, and the names must differ: a case is keyed by name
        // throughout the suite (its recording is a file of that name) and the pairing looks one up
        // by it, so two cases sharing a name pair the same Metal figures twice. Caught by writing
        // this test with both named `c` and watching the hot row come out at double.
        let metal = measurement(
            Leg::Metal,
            vec![
                calling("hot-case", "hot", 1_000, 10),
                calling("cold-case", "cold", 1, 10),
            ],
        );
        let ours = measurement(
            Leg::Ours,
            vec![
                calling("hot-case", "hot", 1_000, 100),
                calling("cold-case", "cold", 1, 100),
            ],
        );
        let rows = paired(&metal, &ours);
        let hot = rows.iter().find(|row| row.call == "hot").unwrap();
        assert_eq!((hot.metal.wall, hot.ours.wall), (10_000, 100_000));
        assert_eq!(hot.metal.typical, 10, "the typical call lost its median");
        assert_eq!(hot.ratio(), Some(10.0));
    }

    /// A span measures what it was asked to and nothing else: the call's name reaches the
    /// observation, and the instrument's own cost is a positive number this host can state.
    #[test]
    fn the_instrument_states_its_own_cost() {
        let cost = instrument_nanos();
        assert!(cost > 0, "the instrument measured itself at zero");
        // Generous: the point is that the figure is real, not that this machine is fast. A reading
        // this far out would mean the calibration loop measured something other than the span.
        assert!(
            cost < 100_000,
            "the instrument measured itself at {cost} ns a call"
        );
    }
}

/// Every case of the suite carries a time record, and this is what makes that structural.
///
/// **The requirement is not that a command can time a selection, it is that a suite cannot exist
/// without its timing being recorded.** A case added tomorrow is timed with no author action and
/// no opt-in, because this fails until `conformance times --write` has been run over it. Nothing
/// here needs Metal or a device: it reads the committed recordings, so it holds on every host and
/// in CI.
///
/// A leg that could not finish the case is a **recorded hole**, not an absence: `record_times`
/// writes its record with the reason in [`CaseTimes::stopped`], and [`judge`] refuses such a leg
/// before the margin. So the check is unconditional and has no allowlist — an allowlist is exactly
/// the thing that would let a case be added and silently untimed.
#[cfg(test)]
mod every_case_is_timed {
    /// Each case's recording holds at least one [`super::TimeRecord`].
    #[test]
    fn every_case_carries_its_timing() {
        let mut missing: Vec<&str> = Vec::new();
        let mut legs = 0;
        let mut holes = 0;
        for case in crate::cases::all() {
            let recording = crate::recording::read(case.name())
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            if recording.times.is_empty() {
                missing.push(case.name());
                continue;
            }
            legs += recording.times.len();
            holes += recording
                .times
                .iter()
                .filter(|held| held.times.stopped.is_some())
                .count();
        }
        // A screen's worth, never the whole list: this failed with 542 names in it once, which is
        // a wall of text a reader scrolls past rather than a failure they act on.
        let shown: Vec<&str> = missing.iter().take(10).copied().collect();
        let rest = missing.len().saturating_sub(shown.len());
        assert!(
            missing.is_empty(),
            "{} case(s) carry no timing at all, so the suite's timing is not recorded for every \
             case. Run `conformance times --runs 8 --write` on a Mac over the whole set; a leg \
             that cannot run records its reason rather than nothing. First: {shown:?}{}",
            missing.len(),
            if rest > 0 {
                format!(" and {rest} more")
            } else {
                String::new()
            }
        );
        // Not a bound, a floor: a tree where this reached zero would mean the reader above counted
        // nothing, which is how a test like this rots into always passing.
        assert!(legs > 0, "no leg of any case carries a figure");
        println!(
            "{legs} leg record(s) over {} case(s), {holes} of them recorded holes",
            crate::cases::all().len()
        );
    }

    /// A recorded hole says **why**, so the reader is never left to guess whether a leg was
    /// refused, stopped, or simply never asked.
    #[test]
    fn a_recorded_hole_names_its_reason() {
        for case in crate::cases::all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                continue;
            };
            for held in &recording.times {
                if let Some(why) = &held.times.stopped {
                    assert!(
                        !why.trim().is_empty(),
                        "{}: its {} leg records a hole with no reason",
                        case.name(),
                        held.leg
                    );
                }
            }
        }
    }
}
