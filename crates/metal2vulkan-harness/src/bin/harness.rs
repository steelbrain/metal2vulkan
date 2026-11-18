//! The harness's command: the authored workloads, or the workloads a description names, on one
//! runtime or several, checked, then timed (the crate's docs).
//!
//! ```text
//! harness --runtime old|new[,old|new...] [--runs N] [--warmup N] [--no-settle]
//!     [--only NAME,NAME...] [--size NAME=N]... [--store DIRECTORY]
//!     [--workloads DIRECTORY --description FILE]
//! ```
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
//! started zero. A workload a runtime refuses with a typed refusal prints `REFUSED` and the
//! refusal on that runtime's row, and the other runtimes' outputs are compared among
//! themselves. The command exits 0 when every workload matched or was refused, 1 when one did
//! not match or failed, and 2 when it could not start.

use metal2vulkan_harness::authored::{Create, Makes};
use metal2vulkan_harness::plan::Plan;
use metal2vulkan_harness::run::{
    matrix_time, percentile, run_each, steady, CreateReport, Harness, Report, Runs, Session, Warmup,
};
use metal2vulkan_harness::{authored, description, old, rt, Error, Failed, Result, Waits};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The command's arguments.
struct Args {
    runtimes: Vec<String>,
    runs: Runs,
    only: Option<Vec<String>>,
    /// The sizes `--size` gives, by workload.
    sizes: Vec<(String, u32)>,
    /// The directory of the runtimes' stores, `--store`.
    store: Option<PathBuf>,
    workloads: Option<PathBuf>,
    description: Option<PathBuf>,
}

const USAGE: &str = "usage: harness --runtime old|new[,old|new...] [--runs N] [--warmup N] \
                     [--no-settle] [--only NAME,NAME...] [--size NAME=N]... [--store DIRECTORY] \
                     [--workloads DIRECTORY --description FILE]";

fn main() -> ExitCode {
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

/// The workloads the arguments select, and the creation workloads (with the authored ones only).
fn workloads(args: &Args) -> Result<(Vec<Workload>, Vec<Create>)> {
    let (all, creates): (Vec<Workload>, Vec<Create>) = match (&args.workloads, &args.description) {
        (Some(directory), Some(file)) => {
            let text = std::fs::read_to_string(file)
                .map_err(|why| format!("read {}: {why}", file.display()))?;
            let described = (description::parse(&text, directory)?.into_iter())
                .map(|described| {
                    let name = described.name.clone();
                    (name, Box::new(move || described.plan()) as Box<_>)
                })
                .collect();
            (described, Vec::new())
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
            (authored, authored::CREATES.to_vec())
        }
    };
    let Some(only) = &args.only else {
        if let Some((name, _)) =
            (args.sizes.iter()).find(|(name, _)| !all.iter().any(|(known, _)| known == name))
        {
            return Err(format!("no workload {name} to size").into());
        }
        return Ok((all, creates));
    };
    let known: Vec<&str> = (all.iter().map(|(known, _)| known.as_str()))
        .chain(creates.iter().map(|create| create.name))
        .collect();
    if let Some(name) = (only.iter())
        .chain(args.sizes.iter().map(|(name, _)| name))
        .find(|name| !known.contains(&name.as_str()))
    {
        return Err(format!(
            "no workload {name}; the workloads are: {}",
            known.join(", ")
        )
        .into());
    }
    Ok((
        (all.into_iter())
            .filter(|(name, _)| only.contains(name))
            .collect(),
        (creates.into_iter())
            .filter(|create| only.iter().any(|name| name == create.name))
            .collect(),
    ))
}

/// Every selected workload on every runtime named; whether each matched.
fn run_all(args: &Args) -> Result<bool> {
    let (workloads, creates) = workloads(args)?;
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
    for (runtime, session) in sessions {
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

/// `report`'s row, and its matrix lines.
fn print_create(runtime: &str, create: &Create, report: &CreateReport, store: &str) {
    let us = |p| percentile(&report.create_us, p).map_or("-".to_owned(), |us| format!("{us:.1}"));
    println!(
        "{:<17} {runtime:<8} {:>10} {:>11} {:>11} {:>11} {:>11} {:>11}  {}{}",
        create.name,
        match create.makes {
            Makes::Buffers { length } => length.to_string(),
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
                "; first {us:.1} us with one dispatch, its creation {create:.1} us; store {store}"
            ),
            (None, Some(first), Some(again), _) => {
                format!("; first take {first:.1} us, the next {again:.1} us")
            }
            _ => String::new(),
        },
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
    if let Some(us) = report.first_us {
        println!(
            "M2V-MATRIX time harness-{runtime}/{}/first {us:.3}",
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
    let judged = if report.commit_us.is_empty() {
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
            let at_base = f64::from(scaled.base) / f64::from(scaled.size);
            matrix(runtime, name, time.map(|us| us * at_base), &report.gauges);
        }
    }
    for (count, value) in &report.counts {
        println!("M2V-MATRIX metric harness-{runtime}/{name}/{count}_per_run {value}");
    }
}
