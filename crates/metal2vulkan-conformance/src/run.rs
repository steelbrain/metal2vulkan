//! Running the suite: record a selection's Metal answers, and replay a selection on a venue.
//!
//! One function per leg, so the command, a test and a device lane all drive the same code.

use crate::case::{Case, Side};
use crate::cost::{self, CaseCost, Counts};
use crate::device::{self, Notices, Policy, ScratchCache, Venues};
use crate::replay::replay;
use crate::report::{CostRow, Report};
use crate::runtime_side::RuntimeSide;

/// The runs record mode takes a case's answer over by default, so a case whose behaviour varies
/// records its outcomes with their frequencies rather than whichever run came last.
///
/// A case whose work nothing orders asks for more through
/// [`crate::case::Case::record_runs`].
pub const RECORD_RUNS: u32 = 3;

/// The cases `names` selects, or every case where `names` is empty.
pub fn selected(names: &[String]) -> Result<Vec<Box<dyn Case>>, String> {
    if names.is_empty() {
        return Ok(crate::cases::all());
    }
    names
        .iter()
        .map(|name| {
            crate::cases::named(name).ok_or_else(|| format!("the suite holds no case {name}"))
        })
        .collect()
}

/// Replay `cases` on the venue `venues` names, with exact answers, as a report.
pub fn replay_on(venues: Venues, cases: &[Box<dyn Case>]) -> Result<Report, String> {
    replay_on_under(venues, Policy::Exact, cases)
}

/// [`replay_on`] under `policy`: either form of every speed-for-accuracy choice.
pub fn replay_on_under(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn Case>],
) -> Result<Report, String> {
    let mut notices = Notices::default();
    let cache = ScratchCache::new().map_err(|error| error.to_string())?;
    let device = device::open_under(venues, policy, &notices, &cache)?;
    // The notices a device announces as it is made are the venue's, not any case's: they say what
    // this host's driver and release cost, once, before a case runs. They go in the report's venue
    // line; no case's scope was open for them, so none of them can reach a case.
    let mut report = Report::new();
    report.venue_announced(device.name(), notices.venue());
    // The cost half's gate: the counts the runtime moved over each case, held to the bounds the
    // case's own recording carries for this venue. It rides along with the correctness replay
    // because the two readings must be of the same run, and it is cheap: two reads of
    // `WorkCounts` a case. Nothing here enters a correctness verdict.
    let whole_set = cases.len() == crate::cases::all().len();
    let venue_label = device.name().to_owned();
    for case in cases {
        let before = read_counts(&device);
        let started = std::time::Instant::now();
        // The case's own notice scope, open for exactly as long as its side lives. Nothing it
        // announces can reach another case, and nothing another case announced can reach it.
        let mut side = RuntimeSide::new(&device, notices.open(case.name()));
        let replayed = replay(case.as_ref(), &mut side);
        drop(side);
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let moved = match (before, read_counts(&device)) {
            (Ok(before), Ok(after)) => Ok(after.since(&before)),
            (Err(why), _) | (_, Err(why)) => Err(why),
        };
        let mut degraded: Vec<String> = match &replayed {
            Ok(replayed) => replayed.ours.announced_notices().into_keys().collect(),
            Err(_) => Vec::new(),
        };
        degraded.sort();
        degraded.dedup();
        report.add_cost(cost_row(
            case.as_ref(),
            &venue_label,
            whole_set,
            moved,
            nanos,
            &degraded,
            &replayed,
        ));
        report.add_result(replayed);
    }
    // A notice that arrived with no case's scope open is the harness's own gap in attribution, not
    // any case's difference, so it is reported rather than charged to whichever case drained next.
    report.unattributed_notices(notices.unattributed());
    Ok(report)
}

/// The device's counts, or nothing where they could not be read as counts.
///
/// A reading that does not parse is the cost half's own failure, never the case's: it is reported
/// as an unmeasured row naming the reason, so a runtime field of a new shape shows up as a hole in
/// the gate instead of as a correctness result.
fn read_counts(device: &metal2vulkan_runtime::Device) -> Result<Counts, String> {
    Counts::of(&device.work_counts())
}

/// How one case's counts stand against its recording's bounds.
fn cost_row(
    case: &dyn Case,
    venue: &str,
    whole_set: bool,
    moved: Result<Counts, String>,
    nanos: u64,
    degraded: &[String],
    replayed: &Result<crate::replay::Replayed, crate::replay::ReplayError>,
) -> CostRow {
    let counts = match moved {
        Ok(counts) => counts,
        Err(why) => {
            return CostRow {
                case: case.name().to_owned(),
                verdict: cost::CostVerdict::Unmeasured {
                    why: format!("the runtime's counts could not be read: {why}"),
                },
                counts: Counts::default(),
                nanos,
                reported: [0; 3],
            }
        }
    };
    // A run the venue stopped — it lacks the feature, or the case could not be replayed at all —
    // did a part of the case, so its counts are not the ones any bound was measured over.
    let stopped = match replayed {
        Ok(row) => match &row.verdict {
            crate::verdict::Verdict::LacksFeature { needs } => {
                Some(format!("the venue lacks {needs}"))
            }
            _ => None,
        },
        Err(error) => Some(error.to_string()),
    };
    let records = crate::recording::read(case.name())
        .map(|recording| recording.costs)
        .unwrap_or_default();
    let verdict = cost::judge(
        &records,
        venue,
        whole_set,
        &counts,
        stopped.as_deref(),
        degraded,
    );
    let reported = records
        .iter()
        .find(|record| record.venue == venue)
        .map(|record| {
            let held = record.held();
            [
                held.iter().filter(|one| one.racy.is_some()).count(),
                held.iter()
                    .filter(|one| one.racy.is_none() && one.range.is_some())
                    .count(),
                held.iter()
                    .filter(|one| {
                        one.racy.is_none() && one.range.is_none() && one.composed.is_some()
                    })
                    .count(),
            ]
        })
        .unwrap_or([0; 3]);
    CostRow {
        case: case.name().to_owned(),
        verdict,
        counts,
        nanos,
        reported,
    }
}

/// Measure what `cases` cost on the venue `venues` names, over `runs` rounds.
///
/// This is the cost half's own leg: it runs our side alone, with no recording read and no
/// comparison, because a count has no Metal counterpart to compare with. Each round runs the whole
/// selection once and a count seen at two values over the rounds is reported rather than gated.
///
/// Two properties of the measurement's shape are load-bearing, and both were learned by measuring:
///
/// - **The whole selection in order**, not one case repeatedly. The runtime pools command pools,
///   memory blocks and descriptor sets across cases, so a case's counts depend on what ran before
///   it. A measurement must have the shape the gate will have.
/// - **A fresh device each round**, because the gate's own run is a fresh one. Reusing a device
///   would measure a *warm* recycler and a warm memory pool: in a first measurement here, 41 cases
///   made a `VkImage` on the first round and took a parked one on the second, which read as a
///   non-deterministic count when it was simply a second round. A bound taken from a warm round
///   would be below what a replay on a cold device does, and the gate would fire on honest work.
pub fn cost_on_under(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn Case>],
    runs: u32,
) -> Result<(String, Vec<CaseCost>), String> {
    let mut held: Vec<CaseCost> = Vec::new();
    let mut venue = String::new();
    for round in 0..runs.max(1) {
        let mut notices = Notices::default();
        let cache = ScratchCache::new().map_err(|error| error.to_string())?;
        let device = device::open_under(venues, policy, &notices, &cache)?;
        // The device's own notices, read off so no case's scope ever holds them.
        notices.venue();
        venue = device.name().to_owned();
        for (index, case) in cases.iter().enumerate() {
            let before = read_counts(&device)?;
            let started = std::time::Instant::now();
            let mut side = RuntimeSide::new(&device, notices.open(case.name()));
            let mut sites = crate::transcript::Sites::new();
            let inputs = case
                .inputs()
                .map_err(|why| format!("{}: {why}", case.name()))?;
            sites.push(crate::case::inputs_node(&inputs));
            let abort = case.run(&mut side, &mut sites).err();
            let mut announced = side.announced();
            drop(side);
            let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let moved = read_counts(&device)?.since(&before);
            let stopped = abort.map(|abort| {
                let what = abort
                    .failure
                    .lacks
                    .as_deref()
                    .unwrap_or("a call did not answer");
                format!("{what}, at {}", abort.at)
            });
            announced.extend(sites.finish().announced_notices().into_keys());
            announced.sort();
            announced.dedup();
            if round == 0 {
                held.push(CaseCost {
                    case: case.name().to_owned(),
                    counts: moved.clone(),
                    least: moved,
                    runs: 1,
                    stopped,
                    degraded: announced,
                    nanos,
                });
            } else {
                let held = &mut held[index];
                held.counts = held.counts.largest(&moved);
                held.least = held.least.smallest(&moved);
                held.runs += 1;
                held.nanos = held.nanos.max(nanos);
                // A run that stopped where an earlier one did not is itself a fact about the
                // measurement: the case is not recorded from a measurement that ran only
                // sometimes.
                if held.stopped.is_none() {
                    held.stopped = stopped;
                }
                // The notices a run announced say how much of the case the device could do
                // exactly. A measurement whose runs announced different sets is not one
                // measurement, and the union is what makes that visible.
                for code in announced {
                    if !held.degraded.contains(&code) {
                        held.degraded.push(code);
                    }
                }
                held.degraded.sort();
            }
        }
    }
    Ok((venue, held))
}

/// Measure each of `cases` on its own: a selection of one, on a device no other case has touched,
/// over `runs` rounds.
///
/// This is the other shape a cost measurement can have, and it answers a different question.
/// [`cost_on_under`] measures a case where the gate will run it, after everything before it in the
/// set; this measures the case with nothing before it at all. The difference matters for every
/// count the runtime pools, because a case that runs alone makes what a case in company finds
/// already made: alone is where a `*_made` count is at its largest and a `*_shared` count at its
/// smallest.
///
/// A bound taken alone is therefore an upper bound that no composition can exceed — and so it does
/// not go stale when a case lands beside it, which a bound taken over one particular set does. It
/// buys that with looseness: for a count that pooling lowers a lot, a bound measured alone sits
/// well above what the case does in company, and a gate holding it will not notice a smaller
/// regression.
pub fn cost_alone_on_under(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn Case>],
    runs: u32,
) -> Result<(String, Vec<CaseCost>), String> {
    // This is the leg that must not be chained onto a whole-set leg inside one process, however
    // convenient that looks: it inherits the other leg's threads and reads a different number.
    // `bc1-rgba` reads `commands_replayed.cmd_dispatch` 4 and `command_buffers_deferred` 6 in
    // three invocations out of three on its own, and 0 for both when this leg follows a whole-set
    // one, because whether a deferred replay lands inside a case's window is not the same question
    // after 393 cases have run. Measured that way the contrast called 6,843 counts
    // composition-dependent where two processes find 3,442. `cost --alone --write` folds into the
    // record the other command wrote, which is what makes two commands unnecessary to coordinate.
    let mut venue = String::new();
    let mut held = Vec::with_capacity(cases.len());
    for case in cases {
        let (name, measured) = cost_on_under(venues, policy, std::slice::from_ref(case), runs)?;
        venue = name;
        held.extend(measured);
    }
    Ok((venue, held))
}

/// Write what `measured` says each case cost on `venue` into the cases' own recordings, answering
/// where each went or why it could not be written.
///
/// A case whose run stopped is **not** recorded: its counts are of a part of the case.
pub fn record_costs(
    venue: &str,
    shape: crate::cost::Shape,
    measured: &[CaseCost],
    words: &str,
) -> Vec<(String, Result<std::path::PathBuf, String>)> {
    measured
        .iter()
        .map(|cost| {
            let written = if let Some(why) = &cost.stopped {
                Err(format!(
                    "the run stopped before the case finished ({why}), so its counts are of a \
                     part of the case"
                ))
            } else {
                record_one_cost(venue, shape, cost, words)
            };
            (cost.case.clone(), written)
        })
        .collect()
}

/// Put one case's cost record in its recording, replacing the venue's own record and keeping every
/// other venue's.
///
/// The bounds are the new measurement's, so a case that now does less work shows it. The set of
/// counts a measurement saw **move** is the union of the new measurement's and the record's,
/// because a count once seen to take two values on this venue is not deterministic there, and a
/// later measurement that happened not to catch it moving has not made it so. Three measurements
/// of this case set found 24, 38, 47 and 68 moving readings of the same ~6,300, all of them the
/// rows that say which thread arrived first, so a single measurement's set is a floor rather than
/// the answer. The union only ever stops a count being gated; it never raises a bound.
fn record_one_cost(
    venue: &str,
    shape: crate::cost::Shape,
    cost: &CaseCost,
    words: &str,
) -> Result<std::path::PathBuf, String> {
    let mut recording = crate::recording::read(&cost.case)?;
    // A measurement of the cases alone does not replace the gate's own shape: it folds into it,
    // and what the two read differently becomes the counts the selection decides.
    if shape == crate::cost::Shape::Alone {
        let held = recording
            .costs
            .iter()
            .find(|held| held.venue == venue)
            .ok_or_else(|| {
                format!(
                    "no cost record for {venue} to contrast with: measure the whole case set \
                     first, then the cases alone"
                )
            })?;
        let folded = cost.fold_alone_into(held, words.to_owned());
        recording.costs.retain(|held| held.venue != venue);
        recording.costs.push(folded);
        recording
            .costs
            .sort_by(|one, two| one.venue.cmp(&two.venue));
        return crate::recording::write(&recording);
    }
    let mut record = cost.record(venue, words.to_owned());
    if let Some(held) = recording.costs.iter().find(|held| held.venue == venue) {
        record = record.carrying_the_contrast_of(held);
    }
    recording.costs.retain(|held| held.venue != venue);
    recording.costs.push(record);
    recording
        .costs
        .sort_by(|one, two| one.venue.cmp(&two.venue));
    crate::recording::write(&recording)
}

/// Take `cases`' Metal answers on this Mac and write their recordings, answering where each
/// recording went or why it could not be written.
///
/// A case record mode must not run on Metal is refused there before anything is encoded; a case
/// whose answer is carried writes what its own measurement states. So one command fills a whole
/// selection's recordings without walking into a case whose Metal leg is not safe.
#[cfg(target_os = "macos")]
pub fn record_all(cases: &[Box<dyn Case>]) -> Vec<(String, Result<std::path::PathBuf, String>)> {
    cases
        .iter()
        .map(|case| (case.name().to_owned(), record_one(case.as_ref())))
        .collect()
}

/// Take one case's answer and write its recording.
///
/// A case record mode runs on Metal is run **twice**: plain, which is the behavioural oracle and
/// the transcript comparison uses, and once more under Metal's own API validation layer, in a
/// child process of its own ([`crate::validated`]). The second run is a fact of its own and is
/// never compared with our behaviour.
///
/// A case whose answer is carried was never run here, so its validated answer is
/// [`Validated::NotRun`]: the absence is recorded, not guessed at.
#[cfg(target_os = "macos")]
fn record_one(case: &dyn Case) -> Result<std::path::PathBuf, String> {
    use crate::recording::Validated;
    let level = metal2vulkan_runtime::MetalLevel::NEWEST;
    let answer = match case.metal_source() {
        crate::case::MetalSource::Run => {
            let validated = crate::validated::take(case).map_err(|error| error.to_string())?;
            let mut side = crate::metal_side::MetalSide::new()?;
            crate::record::record(case, &mut side, case.record_runs(), level, validated)
                .map_err(|error| error.to_string())?
        }
        // Never run, and never written from here either: the call this row names is the call
        // Metal does not come back from, so there is nothing for record mode to take. Its
        // recording is written by hand from the probe that died
        // ([`crate::record::never_returned`]).
        crate::case::MetalSource::DidNotReturn { call, says, .. } => {
            return Err(format!(
                "{}: record mode must not run {call}: {says}. Its recording is written by hand \
                 from the probe that died, never by running the shape",
                case.name()
            ))
        }
        crate::case::MetalSource::Carried {
            from, measured_on, ..
        } => {
            let (runs, transcript) = case
                .carried_transcript()?
                .ok_or_else(|| format!("{}: it carries no answer", case.name()))?;
            crate::record::carried(
                level,
                from,
                measured_on,
                crate::recording::ORACLE_CHIP,
                runs,
                transcript,
                Validated::NotRun,
            )
        }
    };
    crate::record::write(case, vec![answer])
}

/// Run one case on native Metal once, announcing each call, and say nothing of what it observed.
///
/// This is the body of the child process [`crate::validated::take`] spawns: the parent reads what
/// this run's Metal said on its standard error. A call that did not answer is not a failure here —
/// the point is what the validation layer printed, whatever the call went on to do.
#[cfg(target_os = "macos")]
pub fn validated_leg(case: &dyn Case) -> Result<(), String> {
    use crate::transcript::Sites;
    let inputs = case.inputs()?;
    let mut side = crate::validated::Announcing::new(crate::metal_side::MetalSide::new()?);
    let mut sites = Sites::new();
    sites.push(crate::case::inputs_node(&inputs));
    let _ = case.run(&mut side, &mut sites);
    Ok(())
}

/// Whether our leg's shader cache carries over between the cases of a round.
///
/// **This exists because the first whole-set run produced a figure nobody could defend.** Metal's
/// `library` cost 46 ms a call against our 0.2 ms, which read as our compiling 200 times faster.
/// It is not a compiler comparison. The suite's 547 recordings make 733 shader references over
/// **152 distinct (shader, entry) pairs**, so 90% of the references are to a shader another case
/// already used, and our leg has a shader cache while Metal's `newLibraryWithData` recompiles
/// every call. Warm is what a consumer sees across a session and is the right default; cold is the
/// only setting under which the compile rows compare like with like.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shaders {
    /// One device and one shader cache for the whole round, so a shader another case compiled is
    /// already there. What a consumer sees, and what the counters half measures.
    Warm,
    /// A device and a shader cache of its own for every case, so no case is handed another's
    /// compile.
    ///
    /// **This setting is one-sided and a reader has to know it.** It turns off *our* reuse and
    /// leaves Metal's leg exactly as it was: there is no public call that empties Metal's own
    /// shader cache, and this host's on-disk one is not writable under SIP. So a cold ratio is
    /// our cold cost against Metal's cost as it comes, and the claim that Metal is recompiling
    /// too is a measurement and not an assumption -- over the warm and cold runs of one twelve
    /// case selection Metal's per-call totals moved 3.0%, and its `newLibraryWithData` costs
    /// about 24.6 ms a call in both, which is a price at which a compile is happening. The
    /// comparison still has a floor that nobody here controls; say so beside any figure from it.
    Cold,
}

/// A device open for a measurement, with the sink its notices go to.
struct Open {
    device: metal2vulkan_runtime::Device,
    notices: Notices,
    /// Dropped with the device, which removes the directory.
    _cache: ScratchCache,
}

/// A fresh device, its own shader-cache directory, and its notices drained of the venue's own.
fn open(venues: Venues, policy: Policy) -> Result<Open, String> {
    let notices = Notices::default();
    let cache = ScratchCache::new().map_err(|error| error.to_string())?;
    let device = device::open_under(venues, policy, &notices, &cache)?;
    // The notices the device announced as it was made are the venue's, not any case's.
    notices.venue();
    Ok(Open {
        device,
        notices,
        _cache: cache,
    })
}

/// Measure what `cases` cost **on our stack** on the venue `venues` names, over `rounds` rounds.
///
/// The measurement's shape is the cost half's, for the same reasons measured there: the whole
/// selection in its own order, because the runtime pools command pools, memory blocks and
/// descriptor sets across cases, and **a fresh device each round**, because a warm recycler is not
/// what a first run meets.
///
/// This leg and [`metal_times`] are run as two phases rather than case by case, and the reason is
/// a measurement hazard rather than tidiness: the `elsewhere` attribution reads the **whole
/// process's** CPU, so interleaving the legs would put Metal's own threads inside our cases'
/// readings and ours inside Metal's. Phases confine that to the first case or two of a phase. The
/// per-call elapsed and calling-thread figures — the table a reader acts on — are of the calling
/// thread alone and are not affected either way. `--leg` runs one leg in a process of its own,
/// which is the only way to read `elsewhere` with nothing of the other leg in it.
pub fn times_on_under(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn Case>],
    rounds: u32,
    shaders: Shaders,
) -> Result<crate::times::Measurement, String> {
    // Bracketed, not sampled once: a leg is minutes long and the reading from its first five
    // milliseconds is of a freshly-spawned thread, which is the case a scheduler treats best
    // ([`crate::slow_calls::Calibration`]).
    let mut calibration = crate::slow_calls::Calibration {
        before: crate::times::instrument_nanos(),
        after: 0,
    };
    let mut venue = String::new();
    let mut taken: Vec<Vec<crate::times::CaseRound>> = Vec::new();
    for _ in 0..rounds.max(1) {
        let mut round = Vec::with_capacity(cases.len());
        // A device and a shader cache per round is the measurement's own shape, matching the gate's
        // ([`cost_on_under`]). `Shaders::Cold` gives each case its own of both instead, which is
        // the only way to read our compile cost per shader rather than per *first* use of a shader
        // — see [`Shaders`].
        let mut warm = match shaders {
            Shaders::Warm => Some(open(venues, policy)?),
            Shaders::Cold => None,
        };
        for case in cases {
            let mut cold = match shaders {
                Shaders::Warm => None,
                Shaders::Cold => Some(open(venues, policy)?),
            };
            let open = warm.as_mut().or(cold.as_mut()).expect("one of the two");
            venue = open.device.name().to_owned();
            let Open {
                device, notices, ..
            } = open;
            let side = RuntimeSide::new(device, notices.open(case.name()));
            round.push(timed(case.as_ref(), side)?);
        }
        taken.push(round);
    }
    calibration.after = crate::times::instrument_nanos();
    Ok(crate::times::fold(
        crate::times::Leg::Ours,
        venue,
        rounds.max(1),
        calibration,
        cases,
        &taken,
    ))
}

/// Measure what `cases` cost **on native Metal** on this Mac, over `rounds` rounds.
///
/// A case record mode must not run on Metal is not run here either, and for the same reason: the
/// safety rule is about what the suite asks Metal to do, not about which command asks it. A case
/// whose Metal leg would end the process or fault the device is left with no Metal figures, which
/// reads as an unmeasured row rather than as a fast one.
#[cfg(target_os = "macos")]
pub fn metal_times(
    cases: &[Box<dyn Case>],
    rounds: u32,
) -> Result<crate::times::Measurement, String> {
    let mut calibration = crate::slow_calls::Calibration {
        before: crate::times::instrument_nanos(),
        after: 0,
    };
    let mut venue = String::new();
    let mut taken: Vec<Vec<crate::times::CaseRound>> = Vec::new();
    for _ in 0..rounds.max(1) {
        let mut round = Vec::with_capacity(cases.len());
        for case in cases {
            if let Some(why) = metal_leg_refused(case.as_ref()) {
                round.push(crate::times::CaseRound {
                    stopped: Some(why),
                    ..Default::default()
                });
                continue;
            }
            // Every Metal call that answers an object answers an **autoreleased** one, and this
            // process drains no pool of its own, so a case's objects would live to the end of the
            // command and a few hundred cases of them would hold every queue slot Metal has. The
            // pool is drained after the case's window is read, so its own cost is outside the
            // measurement.
            let (side, measured) = objc2::rc::autoreleasepool(|_| {
                let side = crate::metal_side::MetalSide::new();
                match side {
                    Ok(side) => {
                        let taken = timed(case.as_ref(), side);
                        (Ok(()), taken)
                    }
                    Err(why) => (Err(why), Ok(crate::times::CaseRound::default())),
                }
            });
            match side {
                Ok(()) => round.push(measured?),
                Err(why) => {
                    return Err(format!("{}: no Metal side: {why}", case.name()));
                }
            }
        }
        venue = "native Metal".to_owned();
        taken.push(round);
    }
    calibration.after = crate::times::instrument_nanos();
    Ok(crate::times::fold(
        crate::times::Leg::Metal,
        venue,
        rounds.max(1),
        calibration,
        cases,
        &taken,
    ))
}

/// Why the Metal leg must not run `case`, where it must not.
///
/// The two sources the suite already holds for this: a row Metal never returned from, and a
/// carried answer whose case says re-recording it on Metal is not safe.
#[cfg(target_os = "macos")]
fn metal_leg_refused(case: &dyn Case) -> Option<String> {
    match case.metal_source() {
        crate::case::MetalSource::Run => None,
        crate::case::MetalSource::Carried { re_record, .. } => match re_record {
            crate::case::ReRecord::Allowed => None,
            crate::case::ReRecord::Forbidden { reason } => Some(format!(
                "the case forbids running its Metal leg here: {reason}"
            )),
        },
        crate::case::MetalSource::DidNotReturn { call, says, .. } => {
            Some(format!("Metal does not return from {call}: {says}"))
        }
    }
}

/// Run one case on `side`, measuring every call it makes.
///
/// The span is the [`crate::watch`] wrapper's, so both legs are measured at one boundary: around
/// the Metal or Vulkan call and the side's own handle bookkeeping, and outside the transcript, the
/// fixtures and the input fingerprint. A cost comparison of two legs measured in different places
/// would not be a comparison, and one wrapper is how that cannot happen.
///
/// The case's window is closed **before** the side is dropped, so tearing a device down is not
/// charged to the case's calls on either leg.
fn timed<S: Side>(case: &dyn Case, side: S) -> Result<crate::times::CaseRound, String> {
    use crate::times::Timer;
    let timer = Timer::new();
    let mut side = crate::watch::Watching::with(side, timer.clone());
    let mut sites = crate::transcript::Sites::new();
    let inputs = case
        .inputs()
        .map_err(|why| format!("{}: {why}", case.name()))?;
    sites.push(crate::case::inputs_node(&inputs));
    let process_cpu_before = crate::clocks::process_cpu_nanos();
    let on_cpu_before = crate::clocks::thread_cpu_nanos();
    let started = std::time::Instant::now();
    let abort = case.run(&mut side, &mut sites).err();
    let window_wall = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let on_cpu_after = crate::clocks::thread_cpu_nanos();
    let process_cpu_after = crate::clocks::process_cpu_nanos();
    let observed = timer.take();
    let mut degraded = side.announced();
    degraded.extend(sites.finish().announced_notices().into_keys());
    degraded.sort();
    degraded.dedup();
    let stopped = abort.map(|abort| {
        let what = abort
            .failure
            .lacks
            .as_deref()
            .unwrap_or("a call did not answer");
        format!("{what}, at {}", abort.at)
    });
    Ok(crate::times::CaseRound {
        observed,
        window_wall,
        window_on_cpu: span(on_cpu_before, on_cpu_after),
        window_process_cpu: span(process_cpu_before, process_cpu_after),
        stopped,
        degraded,
    })
}

/// The nanoseconds between two readings of one clock, or zero where the host answered neither.
fn span(before: Option<u64>, after: Option<u64>) -> u64 {
    match (before, after) {
        (Some(before), Some(after)) => after.saturating_sub(before),
        _ => 0,
    }
}

/// Write what `metal` and `ours` measured into the cases' own recordings, answering where each
/// went or why it could not be written.
///
/// A leg whose round **stopped** before the case finished is not written: its calls are a part of
/// the case. A leg that announced a degrade notice **is** written, with the codes beside the
/// figures, because the figures are a true measurement of what that device did — it is the
/// *comparison* that the notices rule out, and [`crate::times::judge`] is where that happens.
pub fn record_times(
    metal: Option<&crate::times::Measurement>,
    ours: Option<&crate::times::Measurement>,
    words: &str,
) -> Vec<(String, Result<std::path::PathBuf, String>)> {
    let mut names: Vec<&str> = Vec::new();
    for measured in [metal, ours].into_iter().flatten() {
        for case in &measured.cases {
            if !names.contains(&case.case.as_str()) {
                names.push(&case.case);
            }
        }
    }
    names
        .into_iter()
        .map(|name| (name.to_owned(), record_one_time(name, metal, ours, words)))
        .collect()
}

/// Put one case's time records in its recording, replacing this machine's records for the legs
/// measured and keeping every other machine's.
fn record_one_time(
    case: &str,
    metal: Option<&crate::times::Measurement>,
    ours: Option<&crate::times::Measurement>,
    words: &str,
) -> Result<std::path::PathBuf, String> {
    use crate::times::TimeRecord;
    let mut recording = crate::recording::read(case)?;
    let mut written = 0;
    for measured in [metal, ours].into_iter().flatten() {
        let Some(times) = measured.case(case) else {
            continue;
        };
        // **A leg that did not finish the case still writes its record**, carrying the reason in
        // `CaseTimes::stopped`. The absence of a record and a recorded hole mean different things
        // to a reader and to a test: every case carrying its timing is what makes "every suite has
        // its timing recorded" a structural property instead of a habit, and a case whose Metal leg
        // the safety rule refuses must say so in its own file rather than by having nothing there.
        // Such a record's figures are not a measurement of the case and nothing compares them:
        // `crate::times::judge` refuses a stopped leg before it reaches the margin.
        let record = TimeRecord {
            leg: measured.leg,
            venue: measured.venue.clone(),
            chip: crate::recording::ORACLE_CHIP.to_owned(),
            os_build: crate::recording::ORACLE_OS_BUILD.to_owned(),
            rounds: measured.rounds,
            measured: words.to_owned(),
            instrument: measured.instrument,
            times: times.clone(),
        };
        recording.times.retain(|held| {
            held.leg != record.leg
                || held.venue != record.venue
                || held.chip != record.chip
                || held.os_build != record.os_build
        });
        recording.times.push(record);
        written += 1;
    }
    if written == 0 {
        return Err(
            "the measurement holds no leg for this case at all, so nothing can be written; a case \
             the run did not reach cannot carry its timing"
                .to_owned(),
        );
    }
    recording
        .times
        .sort_by(|one, two| (one.leg, &one.venue).cmp(&(two.leg, &two.venue)));
    crate::recording::write(&recording)
}

/// The suite's integrity property, on a device: **every case reads the same verdict alone as it
/// reads inside a selection.**
///
/// This is the whole-run form of what `device::tests` and
/// `tests::a_notice_one_case_provoked_does_not_excuse_the_nexts_difference` hold piece by piece.
/// Those two are deterministic and need no device; this one runs the real legs, so it catches any
/// other state the suite carries from case to case — the device, its queue, its pools, its shader
/// cache, a lazily-built fixture — wherever that state reaches a verdict.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
mod every_case_reads_the_same_verdict_alone {
    use super::{replay_on_under, Policy, Venues};
    use crate::allowance::Justification;
    use crate::case::Case;

    const VENUE: Venues = if cfg!(m2v_gpu) {
        Venues::Gpu
    } else {
        Venues::Cpu
    };

    /// How many of the notice-sensitive cases the guard takes. A bound, because each case is
    /// replayed twice and the alone leg opens a device apiece: the property is a property of the
    /// machinery, so a sample of the family that can express it is the whole evidence, and the
    /// deterministic tests cover the mechanism exhaustively.
    const TAKE: usize = 8;

    /// The cases whose verdict a notice can move: the ones that cite a degrade notice to justify
    /// something. Selected by shape rather than by name, so a case added to the family joins the
    /// guard and a case whose notice is fixed leaves it.
    fn notice_sensitive() -> Vec<Box<dyn Case>> {
        crate::cases::all()
            .into_iter()
            .filter(|case| {
                case.allowances().iter().any(|allowance| {
                    matches!(allowance.justified_by, Justification::Degradation(_))
                })
            })
            .take(TAKE)
            .collect()
    }

    #[test]
    fn a_cases_verdict_is_the_same_alone_as_in_a_selection() {
        let selection = notice_sensitive();
        assert!(
            !selection.is_empty(),
            "no case cites a degrade notice, so the guard has nothing to hold"
        );

        let together = replay_on_under(VENUE, Policy::Exact, &selection).unwrap();
        let mut announced_anywhere = 0usize;
        for row in together.rows() {
            announced_anywhere += row.ours.announced_notices().len();
        }

        for case in &selection {
            let alone = replay_on_under(VENUE, Policy::Exact, std::slice::from_ref(case)).unwrap();
            let alone_row = alone
                .rows()
                .iter()
                .find(|row| row.case == case.name())
                .unwrap_or_else(|| panic!("{} replayed alone gave no row", case.name()));
            let in_set_row = together
                .rows()
                .iter()
                .find(|row| row.case == case.name())
                .unwrap_or_else(|| panic!("{} in the selection gave no row", case.name()));
            assert_eq!(
                alone_row.verdict,
                in_set_row.verdict,
                "{} read one verdict alone and another inside a selection of {}",
                case.name(),
                selection.len()
            );
        }

        // A guard that holds a property nothing on this venue exercises has to say so, rather than
        // reading as evidence. The verdicts above are still held either way.
        if announced_anywhere == 0 {
            println!(
                "no case of the selection announced a degrade notice on this venue, so the \
                 notice half of the property was not exercised here"
            );
        } else {
            println!("{announced_anywhere} notice(s) announced across the selection");
        }
        assert_eq!(
            together.unattributed(),
            &[],
            "the selection announced notices no case's scope was open for"
        );
    }
}
