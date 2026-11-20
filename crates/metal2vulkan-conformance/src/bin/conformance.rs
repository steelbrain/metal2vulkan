//! The conformance suite's command: `record` takes Metal's answers on a Mac, `replay` runs the
//! runtime's side on a venue and reports.
//!
//! ```text
//! conformance record [case ...]
//! conformance replay [--cpu] [--runs N] [case ...]
//! conformance times [--runs N] [--write] [--show] [case ...]
//! conformance cases
//! ```
//!
//! `record` runs each case on Metal twice: plain, which is the behavioural oracle, and once under
//! Metal's own API validation layer, in a child of this command ([`metal2vulkan_conformance::validated`]).
//!
//! With no case named, the whole case set is selected. `--cpu` replays on a CPU Vulkan
//! implementation (lavapipe), which is how the no-Metal leg is proven.
//!
//! `replay` replays through the runtime (`metal2vulkan-rt`), on its public API
//! ([`metal2vulkan_conformance::rt_side`]), held to the ledger, `known-gaps-rt.tsv`.
//!
//! The command's exit status is 0 only where the selection holds no surprise: no unrecorded case,
//! no stale recording, and exactly the gaps `known-gaps-rt.tsv` expects — a refusal or a
//! divergence it does not list, and a row it lists that the run no longer holds on a venue it
//! names, each fail the run. A venue that lacks a feature a case needs does not spoil it.
//!
//! The ledger never changes a verdict. A gap it expects is still printed as `Refused` or
//! `Divergent`, with our answer and Metal's, and still counted, because its rows are open bugs
//! rather than accepted differences.
//!
//! `replay --runs N` replays the selection N times in **this** process and on **this** device, and
//! flags every verdict and site that is not the same in all N
//! ([`metal2vulkan_conformance::stability`]). It is off by default — one pass, nothing compared —
//! because the point of a replay is one honest pass over a shared venue. It has its own line of
//! the exit status: a reading that moves run to run is not a difference from Metal, and a one-shot
//! gate reads it as a regression because that is the only reading it has.
//!
//! `times` takes what each call of the selection costs on native Metal on this Mac
//! ([`metal2vulkan_conformance::times`]) and prints it; `--write` keeps the figures in the cases'
//! own recordings, because Metal's cost on a given macOS build is the one half that cannot be
//! re-taken later. The figures gate nothing.

use metal2vulkan_conformance::device::Venues;
use metal2vulkan_conformance::known_gaps::Expectation;
use metal2vulkan_conformance::run;

/// What the command answers a bare invocation with. Printed rather than acted on: the verbs each
/// take the device, and `replay` with no case named selects every case, so defaulting to it meant
/// running the whole set against the GPU for anyone who typed the command to find out what it took.
const USAGE: &str = "\
conformance record [case ...]
conformance replay [--cpu] [--runs N] [case ...]
conformance times [--runs N] [--write] [--show] [case ...]
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
    let mut names = Vec::new();
    // The rounds `times` takes when nothing says otherwise.
    let mut runs = 8;
    // Whether `--runs` was stated, so `replay` can default to one pass while `times` keeps its
    // eight. A sentinel in `runs` itself would make "the default" a number a reader has to look up
    // in two places.
    let mut asked_runs = false;
    let mut write = false;
    let mut show = false;
    let mut words = words.peekable();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--cpu" => venues = Venues::Cpu,
            "--gpu" => venues = Venues::Gpu,
            "--write" => write = true,
            "--show" => show = true,
            "--runs" => {
                let value = words.next().ok_or("--runs wants a number")?;
                runs = value
                    .parse()
                    .map_err(|_| format!("--runs {value}: not a number"))?;
                if runs == 0 {
                    return Err("--runs 0: a run that happens no times reads nothing".to_owned());
                }
                asked_runs = true;
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
        "times" => times(&cases, runs, write, show),
        metal2vulkan_conformance::validated::LEG_VERB => validated_leg(&cases),
        "replay" => {
            // `--runs` defaults to one pass for `replay`, not to `times`' eight: the point of a
            // replay is one honest pass over a venue, and a venue is shared. More than one pass
            // is asked for, and what it buys is the run-to-run comparison below.
            let passes = if asked_runs { runs } else { 1 };
            let (report, stability) = run::replay_rt_on_under(venues, &cases, passes)?;
            print!("{}", report.render());
            print!("{}", stability.render());
            // The known-gaps ledger decides the exit status and nothing else: each of its rows
            // is still printed above as a `Refused` or `Divergent` row and still counted there.
            // It is held in both directions and per venue, so a new gap fails the run and a row
            // the run no longer holds fails it too.
            let ledger = metal2vulkan_conformance::known_gaps::ledger()?;
            let file = metal2vulkan_conformance::known_gaps::LEDGER_FILE;
            let unexpected = report.unexpected(&ledger);
            // The ledger's typed reasons, counted on every run: the rows not built yet are open
            // work, and the count is printed whether or not the run is green.
            println!(
                "\n{}",
                metal2vulkan_conformance::known_gaps::open_work(&ledger, file)
            );
            // A reading that moves between two runs of one unchanged selection is its own gate,
            // kept apart from the ledger: it is not a difference from Metal, and a one-shot gate
            // cannot tell it from one.
            let steady = stability.is_stable();
            if !steady {
                println!(
                    "\n{} reading(s) moved between runs of the same selection; they are listed \
                     above",
                    stability.moved().len()
                );
            }
            if unexpected.is_empty() && steady {
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
                        "\nthe {expected} row(s) above are exactly the open gaps {file} \
                         expects on this venue, each one a bug with a todo behind it"
                    );
                }
                return Ok(true);
            }
            println!();
            for why in &unexpected {
                println!("not what {file} expects: {why}");
            }
            // The rows to add, in the ledger's own format, each on one line and all of them under
            // one unambiguous heading. The report above is prose for a person; this is for
            // whoever has to edit the file, and it removes the step where a site or an answer is
            // reconstructed by hand or by somebody's parser.
            let rows: Vec<String> = unexpected.iter().filter_map(Expectation::row).collect();
            if !rows.is_empty() {
                println!(
                    "\n{file} rows for the {} gap(s) above, one per line, tab-separated. \
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
            "unknown command {other}: expected record, replay, times or cases"
        )),
    }
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

/// Measure what the selection costs in **time** on native Metal and print the table, writing the
/// figures into the cases' own recordings with `--write`.
#[cfg(target_os = "macos")]
fn times(
    cases: &[Box<dyn metal2vulkan_conformance::case::Case>],
    runs: u32,
    write: bool,
    show: bool,
) -> Result<bool, String> {
    use metal2vulkan_conformance::clocks::host_load;
    use metal2vulkan_conformance::slow_calls::LegLoad;
    let before = host_load();
    let metal = run::metal_times(cases, runs)?;
    let load = LegLoad {
        before,
        after: host_load(),
    };
    // The host's load belongs in the words a record carries: a figure taken on a busy host is not
    // the figure a quiet one gives, and a reader of the recording cannot otherwise tell which they
    // have.
    let words = format!(
        "{runs} round(s) of the whole selection, in the case set's own order; a figure is the \
         median of the rounds' own medians; {}",
        match load.typical() {
            Some(at) => format!("Metal's leg ran at a one-minute load average of {at:.2}"),
            None => "this host answers no load average".to_owned(),
        }
    );
    print!(
        "{}",
        metal2vulkan_conformance::times::render(None, &metal, show)
    );
    if !write {
        println!("nothing written; pass --write to keep these figures in the recordings");
        return Ok(true);
    }
    let mut written_all = true;
    let mut wrote = 0;
    for (case, written) in run::record_times(Some(&metal), None, &words) {
        match written {
            Ok(_) => wrote += 1,
            Err(why) => {
                written_all = false;
                println!("{case}: not recorded: {why}");
            }
        }
    }
    println!("{wrote} case(s) carry their timing");
    Ok(written_all)
}

/// There is no Metal to time off macOS.
#[cfg(not(target_os = "macos"))]
fn times(
    _cases: &[Box<dyn metal2vulkan_conformance::case::Case>],
    _runs: u32,
    _write: bool,
    _show: bool,
) -> Result<bool, String> {
    Err("times measures native Metal, on macOS".to_owned())
}
