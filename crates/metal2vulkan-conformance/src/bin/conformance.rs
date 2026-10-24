//! The conformance suite's command: `record` takes Metal's answers on a Mac, `replay` runs our side
//! on a venue and reports.
//!
//! ```text
//! conformance record [case ...]
//! conformance replay [--cpu] [--default-policy] [case ...]
//! conformance cost [--cpu] [--default-policy] [--runs N] [--write] [--show] [--alone] [--check] [case ...]
//! conformance times [--cpu] [--default-policy] [--runs N] [--leg metal|ours] [--write] [--show]
//!                    [--call NAME]... [--cold-shaders] [case ...]
//! conformance cases
//! ```
//!
//! `record` runs each case on Metal twice: plain, which is the behavioural oracle, and once under
//! Metal's own API validation layer, in a child of this command ([`metal2vulkan_conformance::validated`]).
//!
//! With no case named, the whole case set is selected. `--cpu` replays on a CPU Vulkan
//! implementation (lavapipe), which is how the no-Metal leg is proven. A replay holds the runtime
//! to exact answers by default; `--default-policy` replays under the runtime's own defaults, the
//! fast form of every speed-for-accuracy choice, which is what a consumer runs. Comparing a case
//! under both is how the suite says whether a difference is the policy's or the implementation's.
//!
//! The command's exit status is 0 only where the selection holds no surprise: no unrecorded case,
//! no stale recording, and exactly the gaps `known-gaps.tsv` expects — a refusal or a divergence
//! it does not list, and a row it lists that the run no longer holds on a venue it names, each
//! fail the run. A venue that lacks a feature a case needs does not spoil it.
//!
//! The ledger never changes a verdict. A gap it expects is still printed as `Refused` or
//! `Divergent`, with our answer and Metal's, and still counted, because its rows are open bugs
//! rather than accepted differences.
//!
//! `times` is the cost half's other face ([`metal2vulkan_conformance::times`]): it runs **both**
//! legs over the same selection on this Mac, back to back, and prints what each call cost on
//! native Metal beside what it cost on our stack. The figures are reported and gate nothing — a
//! time is not assertable on a machine running other work — and a ratio is the cost of our whole
//! chain (translation, runtime, MoltenVK, Metal) against native Metal, attributing to no layer of
//! it. `--leg` runs one leg alone, which is the only way to read the off-thread attribution with
//! nothing of the other leg's threads in it; `--write` keeps the figures in the cases' own
//! recordings, because Metal's cost on a given macOS build is the one half that cannot be re-taken
//! later.
//!
//! `cost` is the other half ([`metal2vulkan_conformance::cost`]): it runs our side alone and says
//! what the runtime did to answer each case, by count. `--runs N` runs the whole selection N times
//! (eight by default), each round on a fresh device, so a count that is not deterministic is seen
//! to move rather than assumed; `--write` puts the bounds in the cases' own recordings, and
//! `--show` prints every count a case moved, which is what a contrast run reads. `replay` then
//! holds each case
//! to them and reports the cost verdicts in their own section. Counts are ours and not Metal's, so
//! they never touch a correctness verdict — but a case going over its bounds does fail the run, on
//! its own line of the exit status.

use metal2vulkan_conformance::device::{Policy, Venues};
use metal2vulkan_conformance::known_gaps::Expectation;
use metal2vulkan_conformance::run;

/// What the command answers a bare invocation with. Printed rather than acted on: the verbs each
/// take the device, and `replay` with no case named selects every case, so defaulting to it meant
/// running the whole set against the GPU for anyone who typed the command to find out what it took.
const USAGE: &str = "\
conformance record [case ...]
conformance replay [--cpu] [--default-policy] [case ...]
conformance cost [--cpu] [--default-policy] [--runs N] [--write] [--show] [--alone] [--check] [case ...]
conformance times [--cpu] [--default-policy] [--runs N] [--leg metal|ours] [--write] [--show]
                   [--call NAME]... [--cold-shaders] [case ...]
conformance cases

A verb is required. `cases` lists the case set and takes no device; every other verb runs work on
a GPU, and with no case named the whole set is selected.";

fn main() -> std::process::ExitCode {
    match command() {
        Ok(compliant) if compliant => std::process::ExitCode::SUCCESS,
        Ok(_) => std::process::ExitCode::FAILURE,
        Err(why) => {
            eprintln!("conformance: {why}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Run the command, answering whether the selection is compliant.
fn command() -> Result<bool, String> {
    let mut words = std::env::args().skip(1);
    // No verb means no work, and the return is before anything reaches a device. Defaulting to
    // `replay` here cost two whole-set runs against the only Metal venue in one day, both from
    // reading the command's usage by invoking it bare; the GPU is shared with another project's
    // measurements, so taking it is something to ask for rather than to receive by default.
    let Some(verb) = words.next() else {
        eprintln!("{USAGE}");
        return Ok(false);
    };
    let mut venues = Venues::Gpu;
    let mut policy = Policy::Exact;
    let mut names = Vec::new();
    // The rounds `cost` takes when nothing says otherwise. Eight, measured: a count the
    // measurement never sees move is gated, and `host_texel_copies` in
    // `chroma-pair-texel-reads-bgrg422` held at 2 over three rounds and over two whole-set
    // replays, then reached 3 — and running that case alone for eight rounds moved it 2..3. A rare
    // second arm is the one thing a thin measurement turns into a false regression.
    let mut runs = 8;
    let mut how = How::default();
    let mut legs = Legs::Both;
    let mut calls = Vec::new();
    let mut shaders = run::Shaders::Warm;
    let mut words = words.peekable();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--cpu" => venues = Venues::Cpu,
            "--gpu" => venues = Venues::Gpu,
            "--default-policy" => policy = Policy::Default,
            "--exact-policy" => policy = Policy::Exact,
            "--write" => how.write = true,
            "--show" => how.show = true,
            "--alone" => how.alone = true,
            "--check" => how.check = true,
            "--cold-shaders" => shaders = run::Shaders::Cold,
            "--call" => {
                calls.push(words.next().ok_or("--call wants a Side call's name")?);
            }
            "--leg" => {
                let value = words.next().ok_or("--leg wants metal or ours")?;
                legs = match value.as_str() {
                    "metal" => Legs::Metal,
                    "ours" => Legs::Ours,
                    "both" => Legs::Both,
                    other => return Err(format!("--leg {other}: expected metal, ours or both")),
                };
            }
            "--runs" => {
                let value = words.next().ok_or("--runs wants a number")?;
                runs = value
                    .parse()
                    .map_err(|_| format!("--runs {value}: not a number"))?;
            }
            flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}")),
            name => names.push(name.to_owned()),
        }
    }
    let cases = run::selected(&names)?;
    match verb.as_str() {
        "cases" => {
            for case in &cases {
                println!("{}  {:?}", case.name(), case.metal_source());
            }
            Ok(true)
        }
        "record" => record(&cases),
        "cost" => cost(venues, policy, &cases, runs, how),
        "times" => times(
            &cases,
            Asked {
                venues,
                policy,
                runs,
                legs,
                write: how.write,
                show: how.show,
                calls: &calls,
                shaders,
            },
        ),
        "timing-coverage" => timing_coverage(venues, policy, &cases),
        metal2vulkan_conformance::validated::LEG_VERB => validated_leg(&cases),
        "replay" => {
            let report = run::replay_on_under(venues, policy, &cases)?;
            print!("{}", report.render());
            // The known-gaps ledger decides the exit status and nothing else: each of its rows
            // is still printed above as a `Refused` or `Divergent` row and still counted there.
            // It is held in both directions and per venue, so a new gap fails the run and a row
            // the run no longer holds fails it too.
            let ledger = metal2vulkan_conformance::known_gaps::ledger()?;
            let unexpected = report.unexpected(&ledger);
            // The cost half is a gate of its own, so it has its own line in the exit status: a
            // count going over its bound is never a conformance difference, and a conformance
            // difference is never a cost result. Both must hold for the run to be green.
            let within = report.cost_is_within();
            if !within {
                println!(
                    "\n{} case(s) did more work than the measurement their bounds name; the \
                     counts are listed above",
                    report.cost_overruns().len()
                );
            }
            if unexpected.is_empty() && within {
                let expected = ledger
                    .iter()
                    .filter(|known| {
                        report
                            .rows()
                            .iter()
                            .any(|row| row.case == known.case && known.covers_venue(&row.venue))
                    })
                    .count();
                if expected > 0 {
                    println!(
                        "\nthe {expected} row(s) above are exactly the open gaps \
                         known-gaps.tsv expects on this venue, each one a bug with a todo behind \
                         it"
                    );
                }
                return Ok(true);
            }
            println!();
            for why in &unexpected {
                println!("not what known-gaps.tsv expects: {why}");
            }
            // The rows to add, in the ledger's own format, each on one line and all of them under
            // one unambiguous heading. The report above is prose for a person; this is for
            // whoever has to edit the file, and it removes the step where a site or an answer is
            // reconstructed by hand or by somebody's parser.
            let rows: Vec<String> = unexpected.iter().filter_map(Expectation::row).collect();
            if !rows.is_empty() {
                println!(
                    "\nknown-gaps.tsv rows for the {} gap(s) above, one per line, tab-separated. \
                     Replace the last field with the open bug, and narrow the venue field to a \
                     stable part of this venue's name (an LLVM or driver version in it will move) \
                     or widen it to * if the gap is every venue's:",
                    rows.len()
                );
                for row in rows {
                    println!("{row}");
                }
            }
            Ok(false)
        }
        other => Err(format!(
            "unknown command {other}: expected record, replay, cost, times, timing-coverage or \
             cases"
        )),
    }
}

/// Measure what the selection costs on a venue and print it, writing the bounds into the cases'
/// own recordings with `--write`.
///
/// The whole selection runs `--runs` times in order, because a case's counts depend on what ran
/// before it in the same process: the runtime pools command pools, memory blocks and descriptor
/// sets across cases, so the measurement must have the shape the gate will have.
///
/// `--alone` measures the other shape instead ([`run::cost_alone_on_under`]): each case on a
/// device nothing else has touched. That is where a pooled count is at its largest, so it is the
/// bound a later landing cannot invalidate — and the contrast between the two shapes is what says
/// whether a count is the case's own work or the selection's.
/// How one `cost` invocation runs: which shape it measures and what it does with the result.
///
/// `times` takes the same `write` and `show` through its own [`Asked`], because the two commands
/// mean the same thing by them: keep the measurement in the recordings, and print every figure
/// rather than the ones that decide the run.
///
/// The four switches travel together because they are one decision made at the command line, and
/// because four separate `bool` parameters are four chances for a call site to transpose two of
/// them with nothing to say so. They arrived from different branches and the merged signature was
/// the one clippy refused.
#[derive(Clone, Copy, Default)]
struct How {
    /// Keep the measurement in the cases' own recordings.
    write: bool,
    /// Print every count, not only the ones that decide the run.
    show: bool,
    /// Measure each case on a device nothing else has touched, rather than in the whole selection.
    alone: bool,
    /// Compare the committed records against the measurement instead of measuring.
    check: bool,
}

fn cost(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn metal2vulkan_conformance::case::Case>],
    runs: u32,
    how: How,
) -> Result<bool, String> {
    let How {
        write,
        show,
        alone,
        check,
    } = how;
    // **Do not run both shapes in one process.** It reads as the obvious simplification -- one
    // `cost --write` taking the whole set and then each case alone, instead of two commands a
    // reader has to know to run in order -- and it silently measures the wrong thing. The second
    // leg inherits the first leg's threads: `bc1-rgba` run on its own reads
    // `commands_replayed.cmd_dispatch` 4 and `command_buffers_deferred` 6 in three invocations out
    // of three, and 0 for both when the same leg follows a whole-set leg in the same process,
    // because whether a deferred replay lands inside a case's window is not the same question
    // after 393 cases have run. Built that way, the contrast called 6,843 counts
    // composition-dependent where two processes find 3,442, and the gated population fell to
    // 6,584 from 9,974 for no reason but the instrument. The record accumulates across commands
    // precisely so this does not have to happen in one.
    let shape = if alone {
        metal2vulkan_conformance::cost::Shape::Alone
    } else {
        metal2vulkan_conformance::cost::Shape::InSet
    };
    let (venue, measured) = if alone {
        run::cost_alone_on_under(venues, policy, cases, runs)?
    } else {
        run::cost_on_under(venues, policy, cases, runs)?
    };
    let words = if alone {
        format!(
            "{runs} run(s) of each case on its own on {venue}, nothing before it on the device, \
             the largest each count reached"
        )
    } else {
        format!(
            "{runs} run(s) of the whole selection on {venue}, in the case set's own order, the \
             largest each count reached"
        )
    };
    println!(
        "cost on {venue}, over {runs} run(s) of {} case(s)",
        cases.len()
    );
    println!("{words}\n");
    let mut moved = 0;
    let mut raced = 0;
    let mut composed = 0;
    let mut stopped = 0;
    for one in &measured {
        let regions = metal2vulkan_conformance::cost::by_region(&one.counts);
        let region = |region: metal2vulkan_conformance::cost::Region| {
            regions.get(&region).copied().unwrap_or(0)
        };
        use metal2vulkan_conformance::cost::Region;
        println!(
            "{:<44} {:>4} counts: transl {:>3}  api {:>4}  submit {:>4}   {:>8.3} ms",
            one.case,
            one.counts.len(),
            region(Region::Translation),
            region(Region::DeviceApi),
            region(Region::Submission),
            one.nanos as f64 / 1e6,
        );
        if let Some(why) = &one.stopped {
            stopped += 1;
            println!("    the run stopped: {why}");
        }
        // With the cases measured alone, the contrast against the record the gate's own shape
        // wrote is what decides which counts it holds, so it is printed where it is read.
        if alone {
            let held = metal2vulkan_conformance::recording::read(&one.case)
                .map(|recording| recording.costs)
                .unwrap_or_default();
            if let Some(held) = held.iter().find(|held| held.venue == venue) {
                let apart = one.apart_from(held);
                if !apart.is_empty() {
                    composed += apart.len();
                    println!(
                        "    {} count(s) read differently from the whole-set measurement, so the \
                         selection decides them and no bound of this case holds them:",
                        apart.len()
                    );
                    for (count, inside, alone) in apart {
                        println!("      {count} {inside} in the case set, {alone} alone");
                    }
                }
            }
        }
        let spread = one.moved();
        if !spread.is_empty() {
            moved += spread.len();
            println!(
                "    {} count(s) moved between runs for no reason the race rule names, so they \
                 are reported and not gated:",
                spread.len()
            );
            for (count, least, most) in spread {
                println!("      {count} {least}..{most}");
            }
        }
        raced += one.moved_by_a_race().len();
        // `--show` prints every count the case moved. What a contrast run needs: the same case in
        // a selection of one and inside the whole set, count by count.
        if show {
            for (count, value) in one.counts.iter() {
                println!("    = {count} {value}");
            }
        }
    }
    // How much of the runtime's counted surface the selection reaches: the cost half's own
    // coverage number, tracked the way "unrecorded" is rather than left to a feeling.
    let kept = metal2vulkan_conformance::cost::Counts::of(&Default::default())
        .map(|all| all.len())
        .unwrap_or_default();
    let mut touched: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for one in &measured {
        touched.extend(one.counts.iter().map(|(name, _)| name));
    }
    println!(
        "\n{} case(s) measured, {stopped} stopped before finishing, {moved} count reading(s) \
         moved between runs for a reason the race rule does not name, {raced} for one it does, \
         {composed} read differently from the whole-set measurement",
        measured.len()
    );
    println!(
        "the selection moves {} of the {kept} counts the runtime keeps",
        touched.len()
    );
    // `--check` asks the one question a diff of the files cannot: would re-measuring change
    // anything the gate reads? Most of a record legitimately differs between two runs of the same
    // binary over the same cases, so the signal sits under the noise unless the comparison is of
    // the gate's own facts alone.
    if check {
        let mut drifted = 0;
        let mut fatal = 0;
        // The two reach directions are counted apart, by `cost::reach`, because they are not
        // equally safe and because a tally written here could swap them with nothing to catch it.
        let mut reach = metal2vulkan_conformance::cost::Reach::default();
        let mut per_case = Vec::new();
        // A case landed since the last whole-set measurement has no bound, so this comparison has
        // nothing to compare and passes it over. That is right -- inventing a bound from the run
        // that is checking it would gate on itself -- but passing it over in silence makes the
        // gate's coverage shrink invisibly with every landing. The count is printed so the bill
        // is read rather than discovered.
        let mut uncovered = Vec::new();
        for one in &measured {
            if one.stopped.is_some() {
                continue;
            }
            let held = metal2vulkan_conformance::recording::read(&one.case)
                .map(|recording| recording.costs)
                .unwrap_or_default();
            let Some(held) = held.iter().find(|held| held.venue == venue) else {
                uncovered.push(one.case.clone());
                continue;
            };
            let fresh = one.record(&venue, words.clone());
            let fresh = if alone {
                one.fold_alone_into(held, words.clone())
            } else {
                fresh.carrying_the_contrast_of(held)
            };
            let moved = metal2vulkan_conformance::cost::drift(held, &fresh);
            if moved.is_empty() {
                continue;
            }
            println!("\n{}", one.case);
            let here = metal2vulkan_conformance::cost::reach(&moved);
            reach.given_up += here.given_up;
            reach.taken_on += here.taken_on;
            per_case.push(moved.clone());
            for one in &moved {
                println!("  {one}");
                drifted += 1;
                if one.what.is_load_bearing() {
                    fatal += 1;
                }
            }
        }
        println!(
            "\n{drifted} load-bearing or reach difference(s) between the records and this \
             measurement, {fatal} of them a number the gate reads"
        );
        println!(
            "reach: {} count(s) the gate gives up, {} it takes on. A count given up stops \
             gating; a count taken on is a bound this measurement is the only evidence for, so \
             {} is the number to watch.",
            reach.given_up, reach.taken_on, reach.taken_on
        );
        // Case by case answers "what moved"; by count it answers "does this belong to the thing
        // that changed", which is the question a re-bless has to pass before it is blessed.
        let by_count = metal2vulkan_conformance::cost::counts_that_moved(&per_case);
        if !by_count.is_empty() {
            println!("\nthe counts that moved, most cases first:");
            for one in &by_count {
                println!(
                    "  {:<48} {:>4} case(s), {:>4} of them gated",
                    one.count, one.cases, one.gated
                );
            }
        }
        if !uncovered.is_empty() {
            println!(
                "{} of the {} case(s) measured hold no bound on {venue}, so this comparison \
                 passed them over; each is a case the gate does not cover until a whole-set \
                 measurement writes one:",
                uncovered.len(),
                measured.len()
            );
            for case in &uncovered {
                println!("  {case}");
            }
        }
        if fatal == 0 {
            println!(
                "no gated bound and no ceiling moved: re-measuring would change nothing the gate \
                 reads"
            );
        }
        return Ok(fatal == 0);
    }
    if !write {
        println!("nothing written; pass --write to put these bounds in the recordings");
        return Ok(true);
    }
    let mut all = true;
    for (case, written) in run::record_costs(&venue, shape, &measured, &words) {
        match written {
            Ok(path) => println!("{case}: {}", path.display()),
            Err(why) => {
                all = false;
                println!("{case}: not recorded: {why}");
            }
        }
    }
    Ok(all)
}

/// Take the selection's Metal answers, printing where each recording went.
#[cfg(target_os = "macos")]
fn record(cases: &[Box<dyn metal2vulkan_conformance::case::Case>]) -> Result<bool, String> {
    let mut all = true;
    for (case, written) in run::record_all(cases) {
        match written {
            Ok(path) => println!("{case}: {}", path.display()),
            Err(why) => {
                all = false;
                println!("{case}: {why}");
            }
        }
    }
    Ok(all)
}

/// There is no Metal to record from off macOS.
#[cfg(not(target_os = "macos"))]
fn record(_cases: &[Box<dyn metal2vulkan_conformance::case::Case>]) -> Result<bool, String> {
    Err("record mode runs on macOS, where Metal is".to_owned())
}

/// Take one case's validated leg: the body of the child process record mode spawns with Metal's
/// API validation layer on. It is not part of the command's documented surface.
#[cfg(target_os = "macos")]
fn validated_leg(cases: &[Box<dyn metal2vulkan_conformance::case::Case>]) -> Result<bool, String> {
    let [case] = cases else {
        return Err(format!(
            "{}: one case at a time",
            metal2vulkan_conformance::validated::LEG_VERB
        ));
    };
    run::validated_leg(case.as_ref())?;
    Ok(true)
}

/// There is no Metal to validate against off macOS.
#[cfg(not(target_os = "macos"))]
fn validated_leg(_cases: &[Box<dyn metal2vulkan_conformance::case::Case>]) -> Result<bool, String> {
    Err("Metal's API validation layer is on macOS, where Metal is".to_owned())
}

/// Which legs a `times` run measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Legs {
    /// Both, in two phases, which is what a comparison needs.
    Both,
    /// Native Metal alone.
    Metal,
    /// Our stack alone.
    Ours,
}

/// What a `times` run was asked for: the flags, gathered so the run takes one argument rather
/// than eight.
struct Asked<'a> {
    venues: Venues,
    policy: Policy,
    runs: u32,
    legs: Legs,
    write: bool,
    show: bool,
    /// The calls `--call` names, each printed case by case. It repeats, because the contrast a
    /// reader wants is usually two calls at once and a whole-set run is minutes.
    calls: &'a [String],
    /// Whether our leg's shader cache carries over between cases ([`run::Shaders`]). Warm is what
    /// a consumer sees; cold is the only setting under which the compile rows compare like with
    /// like.
    shaders: run::Shaders,
}

/// Measure what the selection costs in **time** on each leg and print the table.
///
/// The two legs run as two phases of one process rather than case by case, because the off-thread
/// attribution reads the whole process's CPU and interleaving would put each leg's threads in the
/// other's readings. `--leg` runs one leg in a process of its own, which is the only way to read
/// that attribution with nothing of the other leg in it.
fn times(
    cases: &[Box<dyn metal2vulkan_conformance::case::Case>],
    asked: Asked<'_>,
) -> Result<bool, String> {
    let Asked {
        venues,
        policy,
        runs,
        legs,
        write,
        show,
        calls,
        shaders,
    } = asked;
    // Each leg is bracketed by its own pair of load readings. The legs run one after the other,
    // so what corrupts a ratio is the host moving between their two windows -- and bracketing the
    // whole run cannot see that, because the run's endpoints average over exactly the interval
    // where the damage happens. These two brackets measure it.
    use metal2vulkan_conformance::clocks::host_load;
    use metal2vulkan_conformance::slow_calls::LegLoad;
    let mut metal_load = LegLoad::default();
    let mut our_load = LegLoad::default();
    metal_load.before = host_load();
    let metal = match legs {
        Legs::Both | Legs::Metal => metal_times(cases, runs)?,
        Legs::Ours => None,
    };
    metal_load.after = host_load();
    our_load.before = host_load();
    let ours = match legs {
        Legs::Both | Legs::Ours => Some(run::times_on_under(venues, policy, cases, runs, shaders)?),
        Legs::Metal => None,
    };
    our_load.after = host_load();
    // The host's load belongs in the words a record carries, because our leg is CPU-bound where
    // Metal's largest calls are GPU-blocked: a figure taken on a busy host is not the figure a
    // quiet one gives, and a reader of the recording cannot otherwise tell which they have.
    let words = format!(
        "{runs} round(s) of the whole selection, in the case set's own order, each round on a \
         fresh device, our shader cache {}; a figure is the median of the rounds' own medians; \
         {}",
        match shaders {
            run::Shaders::Warm => "carried over between the cases of a round",
            run::Shaders::Cold => "fresh for every case",
        },
        match (metal_load.typical(), our_load.typical()) {
            (Some(metal_at), Some(ours_at)) => format!(
                "Metal's leg ran at a one-minute load average of {metal_at:.2} and ours at \
                 {ours_at:.2}; the legs run one after the other, so the per-call margin is judged \
                 only when both are {:.1} or under and within {:.1}x of each other",
                metal2vulkan_conformance::slow_calls::QUIET_ENOUGH,
                metal2vulkan_conformance::slow_calls::LOAD_MAY_MOVE
            ),
            _ => "this host answers no load average".to_owned(),
        }
    );
    match (&metal, &ours) {
        (metal, Some(ours)) => print!(
            "{}",
            metal2vulkan_conformance::times::render(metal.as_ref(), ours, show)
        ),
        // The Metal leg alone: there is nothing to compare it with, so it prints as its own
        // measurement rather than as half a table.
        (Some(metal), None) => print!(
            "{}",
            metal2vulkan_conformance::times::render(None, metal, show)
        ),
        (None, None) => return Err("--leg selected neither leg".to_owned()),
    }
    // The triage for a row of the per-call table: a total over a case set says that a call is
    // expensive, never where.
    if let (Some(metal), Some(ours)) = (&metal, &ours) {
        for call in calls {
            print!(
                "{}",
                metal2vulkan_conformance::times::render_one_call(metal, ours, call)
            );
        }
    }
    // The per-call gate, which is why this command has a meaningful exit status at all. It is held
    // against the paired rows, so it needs both legs; a one-leg run reports and judges nothing.
    // It also needs a measurement whose readings localise: a warm one rather than a cold one, and
    // one taken on a quiet host rather than a busy one -- see `slow_calls::judges`.
    let mut green = true;
    // The instrument's own cost on the CPU each leg ran on, and what this chip's recordings say it
    // costs: the measurement of the measurement, which the load average cannot stand in for.
    let cpu = metal2vulkan_conformance::slow_calls::Calibrations {
        metal: metal
            .as_ref()
            .map(|one| one.calibration)
            .unwrap_or_default(),
        ours: ours.as_ref().map(|one| one.calibration).unwrap_or_default(),
        reference: recorded_instrument(cases),
    };
    if let Err(why) =
        metal2vulkan_conformance::slow_calls::judges(shaders, metal_load, our_load, cpu)
    {
        println!("\n{why}");
    } else if let (Some(metal), Some(ours)) = (&metal, &ours) {
        let ledger = metal2vulkan_conformance::slow_calls::ledger()?;
        let held = metal2vulkan_conformance::slow_calls::held(
            &metal2vulkan_conformance::times::paired(metal, ours),
            &ledger,
            ours.instrument,
        );
        println!();
        for one in &held {
            println!("{one}");
        }
        let failing: Vec<_> = held.iter().filter(|one| one.fails()).collect();
        let withheld = held.iter().filter(|one| one.withheld()).count();
        if failing.is_empty() {
            println!(
                "every call over the {}x margin that this measurement can resolve is one \
                 known-slow-calls.tsv lists, each a bug with a todo behind it",
                metal2vulkan_conformance::times::MARGIN
            );
        } else {
            green = false;
            println!(
                "\n{} per-call cost result(s) the ledger does not expect",
                failing.len()
            );
        }
        if withheld > 0 {
            println!(
                "{withheld} row(s) withheld: the measurement does not resolve the call above its \
                 own instrument, so they neither add a ledger row nor retire one"
            );
        }
    }
    if !write {
        println!("nothing written; pass --write to keep these figures in the recordings");
        return Ok(green);
    }
    let mut written_all = true;
    let mut wrote = 0;
    for (case, written) in run::record_times(metal.as_ref(), ours.as_ref(), &words) {
        match written {
            Ok(_) => wrote += 1,
            Err(why) => {
                written_all = false;
                println!("{case}: not recorded: {why}");
            }
        }
    }
    println!("{wrote} case(s) carry their timing");
    Ok(green && written_all)
}

/// What this chip's own recordings say the instrument costs, in nanoseconds a call: the median over
/// the selection's recorded legs taken on this chip and this macOS build.
///
/// **It is the only part of the gate that reaches outside the run, and it is there for the one
/// condition a run cannot see from inside itself.** A measurement degraded from its first
/// instruction to its last has four calibration readings that all agree, and nothing but a figure
/// from another day says they are all wrong together. Measured on this host: under a background
/// scheduling class the instrument reads 0.647 us a call against the 0.244 us the recordings
/// carry, and the per-call ratio it produces doubles, while the one-minute load average holds at
/// 6.1.
///
/// A selection whose recordings carry no reading for this chip and build answers `None`, and the
/// gate then judges on what it does have: a missing reading is not evidence of a degraded host,
/// the same rule the load readings follow.
fn recorded_instrument(cases: &[Box<dyn metal2vulkan_conformance::case::Case>]) -> Option<u64> {
    use metal2vulkan_conformance::recording::{ORACLE_CHIP, ORACLE_OS_BUILD};
    let mut seen: Vec<u64> = Vec::new();
    for case in cases {
        let Ok(recording) = metal2vulkan_conformance::recording::read(case.name()) else {
            continue;
        };
        for held in &recording.times {
            if held.chip == ORACLE_CHIP && held.os_build == ORACLE_OS_BUILD && held.instrument > 0 {
                seen.push(held.instrument);
            }
        }
    }
    if seen.is_empty() {
        return None;
    }
    seen.sort_unstable();
    Some(seen[(seen.len() - 1) / 2])
}

/// The Metal leg's figures, where this host has Metal.
#[cfg(target_os = "macos")]
fn metal_times(
    cases: &[Box<dyn metal2vulkan_conformance::case::Case>],
    runs: u32,
) -> Result<Option<metal2vulkan_conformance::times::Measurement>, String> {
    run::metal_times(cases, runs).map(Some)
}

/// There is no Metal to time off macOS, so the run reports our leg alone rather than refusing.
#[cfg(not(target_os = "macos"))]
fn metal_times(
    _cases: &[Box<dyn metal2vulkan_conformance::case::Case>],
    _runs: u32,
) -> Result<Option<metal2vulkan_conformance::times::Measurement>, String> {
    Ok(None)
}

/// Print the timing-coverage census: which calls the selection times and which it does not.
fn timing_coverage(
    venues: Venues,
    policy: Policy,
    cases: &[Box<dyn metal2vulkan_conformance::case::Case>],
) -> Result<bool, String> {
    // One round is enough: the census asks which calls a case makes, not what they cost. A call a
    // case makes only sometimes is the one thing a single round can miss, so the census says how
    // many rounds it took.
    // Warm, because the census asks which calls a case makes and a cold run would cost a device
    // per case to answer the same question.
    let measured = run::times_on_under(venues, policy, cases, 1, run::Shaders::Warm)?;
    let census = metal2vulkan_conformance::coverage::Census::of(&measured)?;
    print!("{}", census.render());
    Ok(true)
}
