//! The times half: what a call costs on native Metal, over the same case the suite records.
//!
//! Metal's own cost on a given macOS build is the one figure that cannot be re-taken later, so
//! `conformance times` measures it per call and per case and keeps it in the case's recording
//! ([`TimeRecord`]). The figures gate nothing.
//!
//! # Three attributions, kept apart
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
//! A case's **spend** is `wall + elsewhere` summed over the case: everything the machine spent on
//! the case's calls, however attributed.
//!
//! # The degraded-set rule
//!
//! A case whose run announced a degrade notice did a **different amount of work** from the exact
//! form, so its calls are pooled apart from the cases done exactly
//! ([`Measurement::noticed_calls`]). A note of the device's native answer
//! ([`crate::replay::NATIVE_NOTES`]) is not such a notice: the native form is the only one there
//! is, so it marks no cheaper work, and the rule counts every notice but those
//! ([`crate::replay::degrading`]).

use crate::watch::Watcher;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Instant;

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
    /// The degrade-notice codes the run announced, sorted, but the device's native answers
    /// ([`crate::replay::degrading`]).
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

    /// Everything the machine spent on the case's calls, however attributed. See the
    /// [module docs](self).
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
    /// a thousand times.
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
    /// The median round's [`CaseRound::spend`].
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

/// Metal's measurement over a selection.
#[derive(Clone, Debug)]
pub struct Measurement {
    /// The venue the leg ran on.
    pub venue: String,
    /// The rounds taken.
    pub rounds: u32,
    /// What the instrument itself cost on this host, in nanoseconds a call: the mean of one
    /// reading before the leg and one after it.
    pub instrument: u64,
    /// Whether the host answered the thread CPU clock. Where it did not, `on_cpu` reads zero
    /// everywhere and `blocked` equals elapsed, and the report says so rather than letting a
    /// reader take the zeros for a fact.
    pub cpu_clock: bool,
    /// Per case, in the selection's order.
    pub cases: Vec<CaseTimes>,
    /// Per call, pooled over the cases the leg did **exactly**.
    pub calls: Vec<CallTimes>,
    /// Per call, pooled over the cases the leg ran to the end while announcing a **notice**. A
    /// case whose run **stopped** is in neither pool: its calls are a part of the case.
    pub noticed_calls: Vec<CallTimes>,
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

/// Metal's cost for a case, as its recording keeps it.
///
/// **Why a recording holds this at all**: Metal's own cost on a given macOS build cannot be
/// re-taken later. What Metal charged for a `setBytes:` on 26A434 is gone the moment the machine
/// updates, so a measurement of it is kept with the case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeRecord {
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
/// `instrument` is what [`instrument_nanos`] read over the leg.
///
/// The pooled per-call table ([`Measurement::calls`]) is taken over the cases the leg did
/// **exactly, start to finish**: a round that stopped part way did a part of the case, and a round
/// that announced a notice did something cheaper than the exact form, so neither belongs in a
/// figure a reader will read as the cost of a call.
pub fn fold(
    venue: String,
    rounds: u32,
    instrument: u64,
    cases: &[Box<dyn crate::case::Case>],
    taken: &[Vec<CaseRound>],
) -> Measurement {
    let mut folded = Vec::with_capacity(cases.len());
    for (index, case) in cases.iter().enumerate() {
        let of_case: Vec<&CaseRound> = taken.iter().filter_map(|round| round.get(index)).collect();
        folded.push(CaseTimes::of(case.name(), &of_case));
    }
    // Two pools, not one filtered pool. A case that stopped part way is in neither, because its
    // calls are a part of the case; a case that ran to the end while announcing a notice is in the
    // second, because its figures are real.
    let pool = |keep: fn(&CaseRound) -> bool| -> Vec<CaseRound> {
        taken
            .iter()
            .map(|round| CaseRound {
                observed: round
                    .iter()
                    .filter(|one| one.stopped.is_none() && keep(one))
                    .flat_map(|one| one.observed.iter().copied())
                    .collect(),
                ..Default::default()
            })
            .collect()
    };
    let exactly = pool(|one| one.degraded.is_empty());
    let noticed = pool(|one| !one.degraded.is_empty());
    Measurement {
        venue,
        rounds,
        instrument,
        cpu_clock: crate::clocks::thread_cpu_nanos().is_some(),
        cases: folded,
        calls: per_call(&exactly.iter().collect::<Vec<_>>()),
        noticed_calls: per_call(&noticed.iter().collect::<Vec<_>>()),
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

/// The table a `times` run prints: Metal's figures per call and, with `show`, per case.
pub fn render(metal: &Measurement, show: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "times: {}, over {} round(s)\n",
        metal.venue, metal.rounds
    ));
    out.push_str(&format!(
        "the instrument costs {} us a call on this host, never subtracted from a figure\n",
        us(metal.instrument)
    ));
    if !metal.cpu_clock {
        out.push_str(
            "this host answers no thread CPU clock, so the `on cpu` column reads zero everywhere \
             and `blocked` is the whole elapsed time: an attribution that was not measured, not \
             one that is zero\n",
        );
    }
    let table = |heading: &str, calls: &[CallTimes]| {
        let mut out = String::from(heading);
        out.push_str(&format!(
            "{:<38} {:>7} {:>11} {:>11} {:>11}\n",
            "call", "calls", "wall us", "on cpu us", "blocked us"
        ));
        for one in calls {
            out.push_str(&format!(
                "{:<38} {:>7} {:>11} {:>11} {:>11}\n",
                one.call,
                one.most_calls,
                us(one.wall),
                us(one.on_cpu),
                us(one.blocked)
            ));
        }
        if calls.is_empty() {
            out.push_str("  no case of this class made a call\n");
        }
        out
    };
    out.push_str(&table(
        "\nover the cases the leg did exactly\n",
        &metal.calls,
    ));
    out.push_str(&table(
        "\nover the cases the leg ran to the end while announcing a notice\n",
        &metal.noticed_calls,
    ));
    if show {
        out.push_str(
            "\nPER CASE, the three attributions apart, in ms; `spend` is wall-in-calls + \
             elsewhere\n",
        );
        out.push_str(&format!(
            "{:<42} {:>8} {:>8} {:>8} {:>8}\n",
            "case", "spend", "cpu", "blkd", "elsew"
        ));
        for one in &metal.cases {
            out.push_str(&format!(
                "{:<42} {:>8} {:>8} {:>8} {:>8}\n",
                one.case,
                ms(one.spend),
                ms(one.on_cpu),
                ms(one.blocked),
                ms(one.elsewhere)
            ));
            if let Some(why) = &one.stopped {
                out.push_str(&format!("  stopped: {why}\n"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A case's figures with its three attributions split however the test wants, folded through
    /// the real aggregation.
    ///
    /// **It goes through [`CaseRound`] and [`CaseTimes::of`] on purpose.** Setting the fields of a
    /// `CaseTimes` by hand would make the spend test below a test of nothing: the first version of
    /// this helper computed `spend` itself, and the guard then **passed with a `spend` that ignored
    /// the handoff entirely**.
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

    /// **The property the whole three-way split exists for**: a case's spend is its whole cost,
    /// so moving work from the caller's thread to other threads, or from computing to waiting,
    /// changes no spend.
    #[test]
    fn moving_work_between_the_attributions_changes_no_spend() {
        let splits = [
            case_of("o", 900, 0, 0),
            case_of("o", 0, 900, 0),
            case_of("o", 0, 0, 900),
            case_of("o", 300, 300, 300),
            case_of("o", 450, 0, 450),
        ];
        for one in &splits {
            assert_eq!(
                one.spend, 900,
                "a split of {}/{}/{} did not fold to the same spend",
                one.on_cpu, one.blocked, one.elsewhere
            );
        }
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

    /// The leg's pools are two pools and not one filtered pool, so a noticed case's calls are
    /// printed rather than dropped.
    #[test]
    fn the_single_leg_pool_keeps_a_noticed_case_apart_rather_than_dropping_it() {
        let round = |degraded: &[&str], stopped: Option<&str>, wall: u64| CaseRound {
            observed: vec![seen("dispatch", wall, wall)],
            window_wall: wall,
            window_on_cpu: wall,
            window_process_cpu: wall,
            stopped: stopped.map(ToOwned::to_owned),
            degraded: degraded.iter().map(|code| (*code).to_owned()).collect(),
        };
        let cases: Vec<Box<dyn crate::case::Case>> =
            crate::cases::all().into_iter().take(3).collect();
        assert_eq!(cases.len(), 3, "the suite holds fewer than three cases");
        let taken = vec![vec![
            round(&[], None, 10),
            round(&["a-notice"], None, 700),
            round(&[], Some("the venue lacks it"), 50_000),
        ]];
        let folded = fold("test".to_owned(), 1, 1, &cases, &taken);
        let one = |calls: &[CallTimes]| calls.iter().find(|held| held.call == "dispatch").cloned();
        assert_eq!(
            one(&folded.calls).map(|held| held.wall_total),
            Some(10),
            "the exact pool is not the clean case alone"
        );
        assert_eq!(
            one(&folded.noticed_calls).map(|held| held.wall_total),
            Some(700),
            "the noticed case's calls reached no pool"
        );
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
/// writes its record with the reason in [`CaseTimes::stopped`]. So the check is unconditional and
/// has no allowlist — an allowlist is exactly
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
                        held.venue
                    );
                }
            }
        }
    }
}
