//! Running the suite: record a selection's Metal answers, and replay a selection on a venue.
//!
//! One function per leg, so the command, a test and a device lane all drive the same code.

use crate::case::Case;
#[cfg(target_os = "macos")]
use crate::case::Side;
use crate::device::Venues;
use crate::replay::replay;
use crate::report::Report;
use crate::stability::Stability;

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

/// Replay `cases` on the runtime (`metal2vulkan-runtime`), on the device `venues` names, as a report,
/// and say whether the same selection read the same thing in each of `runs` passes
/// ([`crate::stability`]). The runtime has no policy and counts no work for the cost half, so the
/// report holds no cost rows.
///
/// Every pass is in **this** process and on **this** device, which is what makes the comparison
/// mean anything: a second process would differ in its caches, its pools and its shader compiles.
/// `runs` of 1 is the default replay: one pass, nothing compared.
///
/// A case that made exported surfaces prints how many of each form it made
/// (`Metrics::surfaces_in_place`, `Metrics::surfaces_copied`), which is the evidence of the form a
/// venue's surfaces took, and a case whose whole grids ran an observing kernel's
/// partial-threadgroups variant prints how many (`Metrics::partial_variant_whole_grids`), and a
/// case whose dispatches and draws pushed a storage texture's level fields prints how many
/// (`Metrics::level_field_commands`), and each pass prints its dispatches by the form their kernels
/// reached their buffers in (`Metrics::bound_dispatches`, `Metrics::address_dispatches`): nothing
/// in a verdict reads any of them.
pub fn replay_rt_on_under(
    venues: Venues,
    cases: &[Box<dyn Case>],
    runs: u32,
) -> Result<(Report, Stability), String> {
    let device = crate::rt_side::RtDevice::open(venues)?;
    let counters = device.device().counters();
    let pass = || {
        let mut report = Report::new();
        let start = counters.metrics();
        for case in cases {
            let before = counters.metrics();
            let mut side = crate::rt_side::RtSide::new(&device);
            report.add_result(replay(case.as_ref(), &mut side));
            drop(side);
            let after = counters.metrics();
            let in_place = after.surfaces_in_place - before.surfaces_in_place;
            let copied = after.surfaces_copied - before.surfaces_copied;
            if in_place + copied > 0 {
                println!(
                    "surface forms: {} on {}: {in_place} in place, {copied} copied",
                    case.name(),
                    device.venue().device
                );
            }
            let whole = after.partial_variant_whole_grids - before.partial_variant_whole_grids;
            if whole > 0 {
                println!(
                    "dispatch forms: {} on {}: {whole} whole grids on the partial-threadgroups variant",
                    case.name(),
                    device.venue().device
                );
            }
            let levels = after.level_field_commands - before.level_field_commands;
            if levels > 0 {
                println!(
                    "level fields: {} on {}: {levels} dispatches and draws pushed a texture's level fields",
                    case.name(),
                    device.venue().device
                );
            }
        }
        let end = counters.metrics();
        println!(
            "buffer forms on {}: {} dispatches bound, {} at their device addresses",
            device.venue().device,
            end.bound_dispatches - start.bound_dispatches,
            end.address_dispatches - start.address_dispatches
        );
        report
    };
    let report = pass();
    let mut stability = Stability::new(runs);
    for at in 2..=runs {
        stability.observe(at, &report, &pass());
    }
    Ok((report, stability))
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
/// [`crate::recording::Validated::NotRun`]: the absence is recorded, not guessed at.
#[cfg(target_os = "macos")]
fn record_one(case: &dyn Case) -> Result<std::path::PathBuf, String> {
    use crate::recording::Validated;
    let answer = match case.metal_source() {
        crate::case::MetalSource::Run => {
            let validated = crate::validated::take(case).map_err(|error| error.to_string())?;
            let mut side = crate::metal_side::MetalSide::new()?;
            crate::record::record(case, &mut side, case.record_runs(), validated)
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
    let before = crate::times::instrument_nanos();
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
    // Read again after the leg, because a leg is minutes long and a reading from its first
    // milliseconds says nothing about the rest: the figure kept is the mean of the two.
    let instrument = (before + crate::times::instrument_nanos()) / 2;
    Ok(crate::times::fold(
        venue,
        rounds.max(1),
        instrument,
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
#[cfg(target_os = "macos")]
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
    let mut announced = side.announced();
    announced.extend(sites.finish().announced_notices().into_keys());
    let degraded = crate::replay::degrading(announced);
    let stopped = abort.map(|abort| abort.stopped());
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
#[cfg(target_os = "macos")]
fn span(before: Option<u64>, after: Option<u64>) -> u64 {
    match (before, after) {
        (Some(before), Some(after)) => after.saturating_sub(before),
        _ => 0,
    }
}

/// Write what `metal` measured into the cases' own recordings, answering where each went or why
/// it could not be written.
///
/// A case whose run announced a degrade notice **is** written, with the codes beside the figures,
/// because the figures are a true measurement of what that device did.
pub fn record_times(
    metal: &crate::times::Measurement,
    words: &str,
) -> Vec<(String, Result<std::path::PathBuf, String>)> {
    metal
        .cases
        .iter()
        .map(|times| (times.case.clone(), record_one_time(times, metal, words)))
        .collect()
}

/// Put one case's time record in its recording, replacing this machine's record and keeping every
/// other machine's.
fn record_one_time(
    times: &crate::times::CaseTimes,
    metal: &crate::times::Measurement,
    words: &str,
) -> Result<std::path::PathBuf, String> {
    use crate::times::TimeRecord;
    let mut recording = crate::recording::read(&times.case)?;
    // **A leg that did not finish the case still writes its record**, carrying the reason in
    // `CaseTimes::stopped`. The absence of a record and a recorded hole mean different things to a
    // reader and to a test: every case carrying its timing is what makes "every suite has its
    // timing recorded" a structural property instead of a habit, and a case whose Metal leg the
    // safety rule refuses must say so in its own file rather than by having nothing there. Such a
    // record's figures are not a measurement of the case.
    let record = TimeRecord {
        venue: metal.venue.clone(),
        chip: crate::recording::ORACLE_CHIP.to_owned(),
        os_build: crate::recording::ORACLE_OS_BUILD.to_owned(),
        rounds: metal.rounds,
        measured: words.to_owned(),
        instrument: metal.instrument,
        times: times.clone(),
    };
    recording.times.retain(|held| {
        held.venue != record.venue || held.chip != record.chip || held.os_build != record.os_build
    });
    recording.times.push(record);
    recording
        .times
        .sort_by(|one, two| one.venue.cmp(&two.venue));
    crate::recording::write(&recording)
}
