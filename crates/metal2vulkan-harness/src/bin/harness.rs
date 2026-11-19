//! The harness's command: the authored workloads, or the workloads a description names, on one
//! runtime or several, checked, then timed (the crate's docs).
//!
//! ```text
//! harness --runtime old|new[,old|new...] [--runs N] [--warmup N] [--no-settle]
//!     [--only NAME,NAME...] [--skip NAME,NAME...] [--size NAME=N]... [--store DIRECTORY]
//!     [--workloads DIRECTORY --description FILE]
//! ```
//!
//! `--only` runs the workloads it names and `--skip` leaves out those it names, of all of them or
//! of `--only`'s. A name ending in `*` names every workload whose name starts with what comes
//! before it: `--skip 'create_*'` runs every workload but the creation ones, and
//! `--only 'create_*'` those alone. A name that names no workload is an error.
//!
//! Each runtime's device keeps its pipelines (the old runtime's shader cache, the new runtime's
//! cache directory) in a store of its own: `<DIRECTORY>/<place>-<runtime>` under `--store`, so a
//! second run on the same directory is warm, or a directory made for the run under the OS temp
//! directory and removed after it, so every run is cold. Each runtime's line says whether its
//! store was fresh or held files, and so does each `create_state` row.
//!
//! A workload whose size follows the device runs at the size `--size` gives it, timing no base;
//! without one, the first runtime sizes it. Each such workload's size prints as its row's `size`
//! record, so runtimes timed in different processes can be checked for the same work.
//!
//! Each workload runs on each runtime in turn, in the order named; a runtime named twice runs it
//! twice. A workload's rows print once its first run matched its CPU reference on every runtime
//! and every runtime's outputs matched the first runtime's. The creation workloads run after the
//! authored ones (not with a description); their rows print once every buffer each runtime made
//! started zero. The rows of a runtime's own run last (not with a description either), each
//! printing its lines once its check passed, or `not compared` on a runtime without the row. A
//! workload a runtime refuses with a typed refusal prints `REFUSED` and the
//! refusal on that runtime's row, and the other runtimes' outputs are compared among
//! themselves. The command exits 0 when every workload matched or was refused, 1 when one did
//! not match or failed, and 2 when it could not start.

use metal2vulkan_harness::authored::{Create, Makes, Own, FIRST_THREADGROUP};
use metal2vulkan_harness::plan::Plan;
use metal2vulkan_harness::run::{
    count_instructions, matrix_time, percentile, run_each, steady, CreateReport, Harness, Report,
    Runs, Session, Warmup, FROM_IDLE,
};
use metal2vulkan_harness::{authored, description, old, rt, Error, Failed, Result, Waits};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The command's arguments.
struct Args {
    runtimes: Vec<String>,
    runs: Runs,
    only: Option<Vec<String>>,
    /// The workloads `--skip` leaves out.
    skip: Vec<String>,
    /// The sizes `--size` gives, by workload.
    sizes: Vec<(String, u32)>,
    /// The directory of the runtimes' stores, `--store`.
    store: Option<PathBuf>,
    workloads: Option<PathBuf>,
    description: Option<PathBuf>,
}

const USAGE: &str = "usage: harness --runtime old|new[,old|new...] [--runs N] [--warmup N] \
                     [--no-settle] [--only NAME,NAME...] [--skip NAME,NAME...] [--size NAME=N]... \
                     [--store DIRECTORY] [--workloads DIRECTORY --description FILE]";

fn main() -> ExitCode {
    // Before any device starts its threads, so that theirs count too.
    count_instructions();
    match parse(std::env::args().skip(1)).and_then(|args| run_all(&args)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(why) => {
            eprintln!("harness: {why}");
            ExitCode::from(2)
        }
    }
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Args> {
    let mut parsed = Args {
        runtimes: Vec::new(),
        runs: Runs {
            warmup: 5,
            settle: true,
            timed: 200,
        },
        only: None,
        skip: Vec::new(),
        sizes: Vec::new(),
        store: None,
        workloads: None,
        description: None,
    };
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("{flag} needs a value; {USAGE}"))
        };
        match flag.as_str() {
            "--runtime" => parsed.runtimes = value()?.split(',').map(str::to_owned).collect(),
            "--runs" => parsed.runs.timed = value()?.parse()?,
            "--warmup" => parsed.runs.warmup = value()?.parse()?,
            "--no-settle" => parsed.runs.settle = false,
            "--only" => parsed.only = Some(value()?.split(',').map(str::to_owned).collect()),
            "--skip" => parsed.skip = value()?.split(',').map(str::to_owned).collect(),
            "--size" => {
                let given = value()?;
                let (name, size) = (given.split_once('='))
                    .ok_or_else(|| format!("--size takes NAME=N, not {given}; {USAGE}"))?;
                parsed.sizes.push((name.to_owned(), size.parse()?));
            }
            "--store" => parsed.store = Some(value()?.into()),
            "--workloads" => parsed.workloads = Some(value()?.into()),
            "--description" => parsed.description = Some(value()?.into()),
            _ => return Err(format!("unknown argument {flag}; {USAGE}").into()),
        }
    }
    if parsed.runtimes.is_empty() || parsed.workloads.is_some() != parsed.description.is_some() {
        return Err(USAGE.into());
    }
    Ok(parsed)
}

/// The directory of the runtimes' stores: `--store`'s, or one made under the OS temp directory
/// for this run, removed when the run ends.
struct Stores {
    root: PathBuf,
    made: bool,
}

impl Stores {
    fn of(store: Option<&Path>) -> Stores {
        match store {
            Some(root) => Stores {
                root: root.to_owned(),
                made: false,
            },
            None => Stores {
                root: std::env::temp_dir().join(format!("m2v-harness-{}", std::process::id())),
                made: true,
            },
        }
    }
}

impl Drop for Stores {
    fn drop(&mut self) {
        if self.made {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

/// How many files `directory` holds, in it and below; 0 where it does not exist.
fn files(directory: &Path) -> usize {
    (std::fs::read_dir(directory).into_iter().flatten().flatten())
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => files(&entry.path()),
            Ok(_) => 1,
            Err(_) => 0,
        })
        .sum()
}

/// The session of the runtime named `name`, its device's pipelines kept in `store`.
fn session(name: &str, authored: &[u8], store: &Path) -> Result<Box<dyn Session>> {
    match name {
        "old" => Ok(Box::new(Harness::new(
            old::system_default(Some(store))?,
            authored,
        )?)),
        "new" => Ok(Box::new(Harness::new(
            rt::system_default(Some(store))?,
            authored,
        )?)),
        other => Err(format!("no runtime {other}; the runtimes are: old, new").into()),
    }
}

/// A workload's name and its plan's maker: a plan holds its buffers' starting contents, so each
/// is made when it runs.
type Workload = (String, Box<dyn Fn() -> Result<Plan>>);

/// The workloads the arguments select, and the creation workloads and the rows of a runtime's own
/// (with the authored ones only).
fn workloads(args: &Args) -> Result<(Vec<Workload>, Vec<Create>, Vec<Own>)> {
    let (all, creates, owns): (Vec<Workload>, Vec<Create>, Vec<Own>) =
        match (&args.workloads, &args.description) {
            (Some(directory), Some(file)) => {
                let text = std::fs::read_to_string(file)
                    .map_err(|why| format!("read {}: {why}", file.display()))?;
                let described = (description::parse(&text, directory)?.into_iter())
                    .map(|described| {
                        let name = described.name.clone();
                        (name, Box::new(move || described.plan()) as Box<_>)
                    })
                    .collect();
                (described, Vec::new(), Vec::new())
            }
            _ => {
                let authored = (authored::WORKLOADS.iter())
                    .map(|workload| {
                        let plan = workload.plan;
                        (
                            workload.name.to_owned(),
                            Box::new(move || Ok(plan())) as Box<_>,
                        )
                    })
                    .collect();
                (
                    authored,
                    authored::CREATES.to_vec(),
                    authored::OWN_ROWS.to_vec(),
                )
            }
        };
    if let Some((name, _)) =
        (args.sizes.iter()).find(|(name, _)| !all.iter().any(|(known, _)| known == name))
    {
        return Err(format!("no workload {name} to size").into());
    }
    let known: Vec<&str> = (all.iter().map(|(known, _)| known.as_str()))
        .chain(creates.iter().map(|create| create.name))
        .chain(owns.iter().map(|own| own.name))
        .collect();
    if let Some(given) = (args.only.iter().flatten())
        .chain(&args.skip)
        .find(|given| !known.iter().any(|name| names(given, name)))
    {
        return Err(format!(
            "no workload {given}; the workloads are: {}",
            known.join(", ")
        )
        .into());
    }
    let selected = |name: &str| {
        (args.only.as_ref()).is_none_or(|only| only.iter().any(|given| names(given, name)))
            && !args.skip.iter().any(|given| names(given, name))
    };
    Ok((
        (all.into_iter())
            .filter(|(name, _)| selected(name))
            .collect(),
        (creates.into_iter())
            .filter(|create| selected(create.name))
            .collect(),
        (owns.into_iter())
            .filter(|own| selected(own.name))
            .collect(),
    ))
}

/// Whether `given`, a name `--only` or `--skip` takes, names the workload `name`: the same name,
/// or, ending in `*`, a start of it.
fn names(given: &str, name: &str) -> bool {
    match given.strip_suffix('*') {
        Some(start) => name.starts_with(start),
        None => given == name,
    }
}

/// Every selected workload on every runtime named; whether each matched.
fn run_all(args: &Args) -> Result<bool> {
    let (workloads, creates, owns) = workloads(args)?;
    let air = authored::library_air()?;
    let stores = Stores::of(args.store.as_deref());
    let mut stored = Vec::with_capacity(args.runtimes.len());
    let mut sessions = Vec::with_capacity(args.runtimes.len());
    for runtime in &args.runtimes {
        let store = stores.root.join(format!("{}-{runtime}", sessions.len()));
        let state = match files(&store) {
            0 => "fresh".to_owned(),
            held => format!("warm, {held} files"),
        };
        let session = session(runtime, &air, &store)?;
        println!(
            "runtime {runtime}: device {}; store {state}",
            session.device_name()
        );
        stored.push(state);
        sessions.push((runtime.as_str(), session));
    }
    println!(
        "warmup runs {}{}{}, timed runs {}",
        if args.runs.settle { "at least " } else { "" },
        args.runs.warmup,
        if args.runs.settle {
            " and until the time settles"
        } else {
            ""
        },
        args.runs.timed
    );
    if !workloads.is_empty() {
        println!(
            "{:<17} {:<8} {:>10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>18} {:>15} {:>11} {:>11} {:>11} {:>11} {:>11}  gauges @100 -> @200",
            "workload",
            "runtime",
            "dispatches",
            "warmup",
            "gpu_us_p10",
            "gpu_us_p50",
            "gpu_us_p90",
            "steady",
            "encode_ns/dispatch",
            "cpu_ns/dispatch",
            "submits/run",
            "c2c_us_p50",
            "c2c_us_p99",
            "late_us_p50",
            "late_us_p99",
        );
    }
    let set = if args.description.is_some() {
        "described"
    } else {
        "authored"
    };
    let before: Vec<(u64, Option<Waits>)> = (sessions.iter())
        .map(|(_, session)| (session.pipelines_created(), session.first_use_waits()))
        .collect();
    let mut refused = vec![0u64; sessions.len()];
    let mut matched = true;
    for (name, plan) in &workloads {
        let size = (args.sizes.iter().rev())
            .find(|(sized, _)| sized == name)
            .map(|&(_, size)| size);
        match plan().and_then(|plan| run_each(&plan, &mut sessions, args.runs, size)) {
            Ok(runs) => {
                for (at, ((runtime, _), run)) in sessions.iter().zip(&runs).enumerate() {
                    match run {
                        Ok(report) => print(runtime, name, report),
                        Err(why) => {
                            println!("{name:<17} {runtime:<8} REFUSED: {why}");
                            refused[at] += 1;
                        }
                    }
                }
            }
            Err(why) => {
                println!("{name}: FAILED: {why}");
                failed(&why, &sessions, name);
                matched = false;
            }
        }
    }
    if !workloads.is_empty() {
        for (((runtime, session), (pipelines, waits)), refused) in
            sessions.iter().zip(before).zip(refused)
        {
            let made = session.pipelines_created() - pipelines;
            println!(
                "runtime {runtime}: the {set} workloads created {made} pipelines; it refused {refused} of them"
            );
            println!("M2V-MATRIX metric harness-{runtime}/{set}/pipelines_created {made}");
            println!("M2V-MATRIX metric harness-{runtime}/{set}/refused {refused}");
            if let (Some(before), Some(after)) = (waits, session.first_use_waits()) {
                let count = after.count - before.count;
                let total = (after.nanos - before.nanos) as f64 / 1e3;
                let longest = after.longest_nanos as f64 / 1e3;
                println!(
                    "runtime {runtime}: the {set} workloads waited {count} times while a pipeline was made, {total:.1} us in all; the longest wait since the device was made took {longest:.1} us"
                );
                println!("M2V-MATRIX metric harness-{runtime}/{set}/first_use_waits {count}");
                println!("M2V-MATRIX time harness-{runtime}/{set}/first_use_wait_total {total:.3}");
                println!(
                    "M2V-MATRIX time harness-{runtime}/{set}/first_use_wait_longest {longest:.3}"
                );
            }
        }
    }
    if !creates.is_empty() {
        println!(
            "{:<17} {:<8} {:>10} {:>11} {:>11} {:>11} {:>11} {:>11}  gauges @100 -> @200",
            "create", "runtime", "bytes", "warmup", "us_p10", "us_p50", "us_p90", "steady",
        );
    }
    for create in &creates {
        let reports = (sessions.iter_mut())
            .map(|(runtime, session)| {
                (session.create(create.makes, args.runs)).map_err(|why| -> Error {
                    Box::new(Failed {
                        runtime: runtime.to_string(),
                        why,
                    })
                })
            })
            .collect::<Result<Vec<CreateReport>>>();
        match reports {
            Ok(reports) => {
                for (((runtime, _), report), store) in sessions.iter().zip(&reports).zip(&stored) {
                    print_create(runtime, create, report, store);
                }
            }
            Err(why) => {
                println!("{}: FAILED: {why}", create.name);
                failed(&why, &sessions, create.name);
                matched = false;
            }
        }
    }
    for own in &owns {
        for (runtime, session) in &mut sessions {
            let name = own.name;
            match session.own_row(own.row) {
                None => println!("{name:<17} {runtime:<8} not compared: not a row of its own"),
                Some(Ok(report)) => {
                    for line in &report.lines {
                        println!("{name:<17} {runtime:<8} {line}");
                    }
                    for (key, us) in &report.times {
                        println!("M2V-MATRIX time harness-{runtime}/{name}/{key} {us:.3}");
                    }
                }
                Some(Err(why)) => {
                    println!("{name}: FAILED: {runtime}: {why}");
                    println!("M2V-FAILED harness-{runtime}/{name}");
                    matched = false;
                }
            }
        }
    }
    for (runtime, session) in sessions {
        for (total, value) in session.totals() {
            println!("M2V-MATRIX metric harness-{runtime}/device/{total} {value}");
        }
        if let Some(gauges) = session.finish() {
            let list: Vec<String> = (gauges.iter())
                .map(|(gauge, value)| format!("{gauge} {value}"))
                .collect();
            println!(
                "runtime {runtime}: after the device dropped: {}",
                list.join(", ")
            );
            for (gauge, value) in gauges {
                println!("M2V-MATRIX gauge harness-{runtime}/device/{gauge}@drop {value}");
            }
        }
    }
    Ok(matched)
}

/// The matrix's line of a workload that failed, `M2V-FAILED harness-<runtime>/<workload>`: for
/// the runtime it failed on ([`Failed`]), or for each runtime of the command where the failure
/// names none (a plan that could not be made, or outputs that differ between runtimes).
fn failed(why: &Error, sessions: &[(&str, Box<dyn Session>)], name: &str) {
    match why.downcast_ref::<Failed>() {
        Some(failed) => println!("M2V-FAILED harness-{}/{name}", failed.runtime),
        None => {
            for (runtime, _) in sessions {
                println!("M2V-FAILED harness-{runtime}/{name}");
            }
        }
    }
}

/// The column of a workload's [`steady`] time.
fn time(times: &[f64]) -> String {
    steady(times).map_or("-".to_owned(), |us| format!("{us:.1}"))
}

/// The column of `warmup`: its runs, marked where the time did not settle.
fn warmup(warmup: &Warmup) -> String {
    let unsettled = if warmup.settled { "" } else { "~" };
    format!("{unsettled}{}", warmup.runs)
}

/// The matrix lines of a workload's gauges at the timed runs they were read after, and of its
/// [`steady`] time where it has one.
fn matrix(
    runtime: &str,
    name: &str,
    time: Option<f64>,
    gauges: &[(u32, Vec<(&'static str, u64)>)],
) {
    if let Some(time) = time {
        println!("M2V-MATRIX time harness-{runtime}/{name} {time:.3}");
    }
    for (point, values) in gauges {
        for (gauge, value) in values {
            println!("M2V-MATRIX gauge harness-{runtime}/{name}/{gauge}@{point} {value}");
        }
    }
}

/// The gauges after the 100th and the 200th timed runs, each marked where it grew between them.
fn gauges(gauges: &[(u32, Vec<(&'static str, u64)>)]) -> String {
    match gauges {
        [] => "-".to_owned(),
        [(_, only)] => (only.iter())
            .map(|(gauge, value)| format!("{gauge} {value}"))
            .collect::<Vec<_>>()
            .join(", "),
        [(_, first), (_, last), ..] => (first.iter().zip(last))
            .map(|((gauge, first), (_, last))| {
                let grew = if last > first { " GREW" } else { "" };
                format!("{gauge} {first} -> {last}{grew}")
            })
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// A threadgroup size as `64x1x1`.
fn size([width, height, depth]: [u32; 3]) -> String {
    format!("{width}x{height}x{depth}")
}

/// `report`'s row, and its matrix lines.
fn print_create(runtime: &str, create: &Create, report: &CreateReport, store: &str) {
    let us = |p| percentile(&report.create_us, p).map_or("-".to_owned(), |us| format!("{us:.1}"));
    println!(
        "{:<17} {runtime:<8} {:>10} {:>11} {:>11} {:>11} {:>11} {:>11}  {}{}{}{}",
        create.name,
        match create.makes {
            Makes::Buffers { length } | Makes::PrivateBuffers { length } => length.to_string(),
            Makes::States { .. } | Makes::RenderStates { .. } => "state".to_owned(),
        },
        warmup(&report.warmup),
        us(10),
        us(50),
        us(90),
        time(&report.create_us),
        gauges(&report.gauges),
        match (
            report.first_us,
            report.first_create_us,
            report.again_us,
            &create.makes
        ) {
            (Some(us), Some(_), _, Makes::RenderStates { .. }) => {
                format!("; first {us:.1} us, its creation alone; store {store}")
            }
            (Some(us), Some(create), _, _) => format!(
                "; first {us:.1} us with one dispatch of {}, its creation {create:.1} us; store {store}",
                size(FIRST_THREADGROUP)
            ),
            (None, Some(first), Some(again), _) => {
                format!("; first take {first:.1} us, the next {again:.1} us")
            }
            _ => String::new(),
        },
        report
            .device_first_us
            .map_or(String::new(), |us| format!("; its warm-up {us:.1} us")),
        report.one_thread_first_us.map_or(String::new(), |us| format!(
            "; another content's first with one dispatch of 1x1x1 {us:.1} us"
        )),
    );
    if let Some(waits) = report.first_allocations {
        println!(
            "{:<17} {runtime:<8} the first's allocations on this thread: {}, {:.1} us in all; the longest since the device was made took {:.1} us",
            create.name,
            waits.count,
            waits.nanos as f64 / 1e3,
            waits.longest_nanos as f64 / 1e3,
        );
        println!(
            "M2V-MATRIX metric harness-{runtime}/{}/first_allocations {}",
            create.name, waits.count
        );
        println!(
            "M2V-MATRIX time harness-{runtime}/{}/first_allocation_total {:.3}",
            create.name,
            waits.nanos as f64 / 1e3
        );
    }
    matrix(
        runtime,
        create.name,
        steady(&report.create_us),
        &report.gauges,
    );
    if let Some(us) = report.device_first_us {
        println!(
            "M2V-MATRIX time harness-{runtime}/{}/device_first {us:.3}",
            create.name
        );
    }
    if let Some(us) = report.first_us {
        println!(
            "M2V-MATRIX time harness-{runtime}/{}/first {us:.3}",
            create.name
        );
    }
    // A compute state's first sample holds its first dispatch; the creation call alone beside it.
    if let (Some(_), Some(call), Makes::States { .. }) =
        (report.first_us, report.first_create_us, &create.makes)
    {
        println!(
            "M2V-MATRIX time harness-{runtime}/{}/first_call {call:.3}",
            create.name
        );
    }
    if let Some(us) = report.one_thread_first_us {
        println!(
            "M2V-MATRIX time harness-{runtime}/{}/first_one_thread {us:.3}",
            create.name
        );
    }
    if let (None, Some(first), Some(again)) =
        (report.first_us, report.first_create_us, report.again_us)
    {
        println!(
            "M2V-MATRIX time harness-{runtime}/{}/first_take {first:.3}",
            create.name
        );
        println!(
            "M2V-MATRIX time harness-{runtime}/{}/next_take {again:.3}",
            create.name
        );
    }
}

/// `report`'s row, and its matrix lines.
fn print(runtime: &str, name: &str, report: &Report) {
    let gpu = |p| percentile(&report.gpu_us, p).map_or("-".to_owned(), |us| format!("{us:.1}"));
    let us =
        |values: &[f64], p| percentile(values, p).map_or("-".to_owned(), |us| format!("{us:.1}"));
    let every: Vec<f64> = report.commit_us.concat();
    let commit = |p| us(&every, p);
    let late_every: Vec<f64> = report.late_us.concat();
    let late = |p| us(&late_every, p);
    let per_dispatch = |values: &[f64]| {
        percentile(values, 50).map_or("-".to_owned(), |ns| {
            format!("{:.0}", ns / report.dispatches as f64)
        })
    };
    println!(
        "{name:<17} {runtime:<8} {:>10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>18} {:>15} {:>11} {:>11} {:>11} {:>11} {:>11}  {}",
        report.dispatches,
        warmup(&report.warmup),
        gpu(10),
        gpu(50),
        gpu(90),
        time(&report.gpu_us),
        per_dispatch(&report.encode_ns),
        per_dispatch(&report.cpu_ns),
        report.submissions,
        commit(50),
        commit(99),
        late(50),
        late(99),
        gauges(&report.gauges),
    );
    println!(
        "{name:<17} {runtime:<8} timed runs: {:.2} submissions and {} submit infos a run",
        report.timed_submissions,
        (report.timed_submit_infos).map_or("-".to_owned(), |infos| format!("{infos:.2}")),
    );
    if report.commit_us.len() > 1 {
        let places: Vec<String> = (report.commit_us.iter().enumerate())
            .map(|(place, times)| format!("#{} {} / {}", place + 1, us(times, 50), us(times, 99)))
            .collect();
        println!(
            "{name:<17} {runtime:<8} c2c_us p50 / p99 by place in the run: {}",
            places.join(", ")
        );
        let places: Vec<String> = (report.late_us.iter().enumerate())
            .map(|(place, times)| format!("#{} {} / {}", place + 1, us(times, 50), us(times, 99)))
            .collect();
        println!(
            "{name:<17} {runtime:<8} late_us p50 / p99 by place in the run: {}",
            places.join(", ")
        );
    }
    let judged = if !report.throughput_us.is_empty() {
        "the frames' throughput, from one's last completed handler to the next's, two in flight"
    } else if report.commit_us.is_empty() {
        "the host wall time a run adds"
    } else {
        "a run's last command buffer's commit-to-complete time"
    };
    let time = matrix_time(report);
    println!(
        "{name:<17} {runtime:<8} matrix time {} us: {judged}; wall p50 {} us",
        time.map_or("-".to_owned(), |us| format!("{us:.1}")),
        us(&report.wall_us, 50),
    );
    // A run's GPU time at the 50th percentile, beside the matrix time and on its scale; none for
    // a workload whose runs wrote no GPU time.
    let gpu_p50 = percentile(&report.gpu_us, 50).filter(|&us| us > 0.0);
    let at_base = report.scaled_to.map_or(1.0, |scaled| {
        f64::from(scaled.base) / f64::from(scaled.size)
    });
    if let Some(us) = gpu_p50 {
        println!(
            "M2V-MATRIX time harness-gpu-{runtime}/{name} {:.3}",
            us * at_base
        );
    }
    match report.scaled_to {
        None => matrix(runtime, name, time, &report.gauges),
        Some(scaled) => {
            let from = scaled.base_us.map_or_else(
                || "as given".to_owned(),
                |us| format!("from {} at {us:.1} us a command buffer", scaled.base),
            );
            println!(
                "{name:<17} {runtime:<8} sized to {} {from}; its matrix time is a run's at {}",
                scaled.size, scaled.base
            );
            println!("M2V-MATRIX size harness-{runtime}/{name} {}", scaled.size);
            matrix(runtime, name, time.map(|us| us * at_base), &report.gauges);
        }
    }
    if !report.throughput_us.is_empty() {
        // The frames' latency from idle, a row of its own beside the throughput that is their
        // row's time.
        let idle = steady(&report.from_idle_us);
        let show = idle.map_or("-".to_owned(), |us| format!("{us:.1}"));
        println!("{name:<17} {runtime:<8} latency from idle {show} us, one run in flight");
        if let Some(us) = idle {
            println!("M2V-MATRIX time harness-{runtime}/{name}{FROM_IDLE} {us:.3}");
        }
    }
    for (count, value) in &report.counts {
        println!("M2V-MATRIX metric harness-{runtime}/{name}/{count}_per_run {value}");
    }
    // User-space instructions per timed run at the 50th percentile, where the kernel offered the
    // counter: the whole process's, the runtime's own threads' with the harness's where the
    // runtime names its threads, and the rest, the threads a driver started for itself. None for a
    // row the device sizes, whose work follows the size.
    if report.scaled_to.is_none() {
        let p50 = |mut values: Vec<u64>| {
            values.sort_unstable();
            values.get(values.len() / 2).copied()
        };
        if let Some(whole) = p50(report.instructions.clone()) {
            println!("M2V-MATRIX instructions harness-{runtime}/{name}/per_run {whole}");
        }
        if report.own_instructions.len() == report.instructions.len() {
            let others = (report.instructions.iter().zip(&report.own_instructions))
                .map(|(whole, own)| whole.saturating_sub(*own))
                .collect();
            if let (Some(own), Some(others)) = (p50(report.own_instructions.clone()), p50(others)) {
                println!("M2V-MATRIX instructions harness-{runtime}/{name}/own_per_run {own}");
                println!(
                    "M2V-MATRIX instructions harness-{runtime}/{name}/others_per_run {others}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{names, parse, workloads};
    use metal2vulkan_harness::authored;

    /// The workloads, creation rows and rows of a runtime's own that ARGS select, by name.
    fn selected(args: &[&str]) -> (Vec<String>, Vec<&'static str>, Vec<&'static str>) {
        let args = parse(args.iter().map(|arg| arg.to_string())).expect("the arguments");
        let (all, creates, owns) = workloads(&args).expect("a selection");
        (
            all.into_iter().map(|(name, _)| name).collect(),
            creates.iter().map(|create| create.name).collect(),
            owns.iter().map(|own| own.name).collect(),
        )
    }

    #[test]
    fn a_name_ending_in_a_star_names_the_workloads_it_starts() {
        assert!(names("create_*", "create_64MiB"));
        assert!(!names("create_*", "alu"));
        assert!(names("alu", "alu"));
        assert!(!names("alu", "alu_wide"));
    }

    #[test]
    fn skip_leaves_out_what_only_would_run_and_the_creation_rows_split_apart() {
        let every: Vec<String> = (authored::WORKLOADS.iter())
            .map(|workload| workload.name.to_owned())
            .collect();
        let creates: Vec<&str> = authored::CREATES.iter().map(|create| create.name).collect();
        let owns: Vec<&str> = authored::OWN_ROWS.iter().map(|own| own.name).collect();
        assert_eq!(
            selected(&["--runtime", "new", "--skip", "create_*"]),
            (every.clone(), Vec::new(), owns.clone())
        );
        assert_eq!(
            selected(&["--runtime", "new", "--only", "create_*"]),
            (Vec::new(), creates.clone(), Vec::new())
        );
        assert_eq!(
            selected(&["--runtime", "new", "--only", "idle_frees*"]),
            (
                Vec::new(),
                Vec::new(),
                vec!["idle_frees", "idle_frees_under_commits"]
            )
        );
        assert_eq!(selected(&["--runtime", "new"]), (every, creates, owns));
        let first = authored::WORKLOADS[0].name;
        assert_eq!(
            selected(&["--runtime", "new", "--only", first, "--skip", first]),
            (Vec::new(), Vec::new(), Vec::new())
        );
        // A name that names no workload is an error, in either flag.
        for flag in ["--only", "--skip"] {
            let args = parse(
                ["--runtime", "new", flag, "no_such_*"]
                    .iter()
                    .map(|arg| arg.to_string()),
            )
            .expect("the arguments");
            assert!(workloads(&args).is_err(), "{flag}");
        }
    }
}
