//! The harness's command: the authored workloads, or the workloads a description names, on one
//! runtime or several, checked, then timed (the crate's docs).
//!
//! ```text
//! harness --runtime old[,RUNTIME...] [--runs N] [--warmup N] [--only NAME,NAME...]
//!     [--workloads DIRECTORY --description FILE]
//! ```
//!
//! Each workload runs on each runtime in turn, in the order named; a runtime named twice runs it
//! twice. A workload's rows print once its first run matched its CPU reference on every runtime
//! and every runtime's outputs matched the first runtime's. The command exits 0 when every
//! workload did, 1 when one did not or failed, and 2 when it could not start.

use metal2vulkan_harness::plan::Plan;
use metal2vulkan_harness::run::{compare_outputs, percentile, Harness, Report, Runs, Session};
use metal2vulkan_harness::{authored, description, old, Result};
use std::path::PathBuf;
use std::process::ExitCode;

/// The command's arguments.
struct Args {
    runtimes: Vec<String>,
    runs: Runs,
    only: Option<Vec<String>>,
    workloads: Option<PathBuf>,
    description: Option<PathBuf>,
}

const USAGE: &str = "usage: harness --runtime old[,RUNTIME...] [--runs N] [--warmup N] \
                     [--only NAME,NAME...] [--workloads DIRECTORY --description FILE]";

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
            timed: 200,
        },
        only: None,
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
            "--only" => parsed.only = Some(value()?.split(',').map(str::to_owned).collect()),
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

/// The session of the runtime named `name`.
fn session(name: &str, authored: &[u8]) -> Result<Box<dyn Session>> {
    match name {
        "old" => Ok(Box::new(Harness::new(old::system_default()?, authored)?)),
        other => Err(format!("no runtime {other}; the runtimes are: old").into()),
    }
}

/// A workload's name and its plan's maker: a plan holds its buffers' starting contents, so each
/// is made when it runs.
type Workload = (String, Box<dyn Fn() -> Result<Plan>>);

/// The workloads the arguments select.
fn workloads(args: &Args) -> Result<Vec<Workload>> {
    let all: Vec<Workload> = match (&args.workloads, &args.description) {
        (Some(directory), Some(file)) => {
            let text = std::fs::read_to_string(file)
                .map_err(|why| format!("read {}: {why}", file.display()))?;
            (description::parse(&text, directory)?.into_iter())
                .map(|described| {
                    let name = described.name.clone();
                    (name, Box::new(move || described.plan()) as Box<_>)
                })
                .collect()
        }
        _ => (authored::WORKLOADS.iter())
            .map(|workload| {
                let plan = workload.plan;
                (
                    workload.name.to_owned(),
                    Box::new(move || Ok(plan())) as Box<_>,
                )
            })
            .collect(),
    };
    let Some(only) = &args.only else {
        return Ok(all);
    };
    if let Some(name) = only
        .iter()
        .find(|name| !all.iter().any(|(known, _)| known == *name))
    {
        let known: Vec<&str> = all.iter().map(|(known, _)| known.as_str()).collect();
        return Err(format!(
            "no workload {name}; the workloads are: {}",
            known.join(", ")
        )
        .into());
    }
    Ok(all
        .into_iter()
        .filter(|(name, _)| only.contains(name))
        .collect())
}

/// Every selected workload on every runtime named; whether each matched.
fn run_all(args: &Args) -> Result<bool> {
    let workloads = workloads(args)?;
    let air = authored::library_air()?;
    let mut sessions = Vec::with_capacity(args.runtimes.len());
    for runtime in &args.runtimes {
        let session = session(runtime, &air)?;
        println!("runtime {runtime}: device {}", session.device_name());
        sessions.push((runtime.as_str(), session));
    }
    println!(
        "warmup runs {}, timed runs {}",
        args.runs.warmup, args.runs.timed
    );
    println!(
        "{:<17} {:<8} {:>10} {:>11} {:>11} {:>11} {:>18} {:>15}  gauges @100 -> @200",
        "workload",
        "runtime",
        "dispatches",
        "gpu_us_p10",
        "gpu_us_p50",
        "gpu_us_p90",
        "encode_ns/dispatch",
        "cpu_ns/dispatch",
    );
    let mut matched = true;
    for (name, plan) in &workloads {
        let checked = plan().and_then(|plan| {
            let reports = (sessions.iter_mut())
                .map(|(_, session)| session.run(&plan, args.runs))
                .collect::<Result<Vec<Report>>>()?;
            for report in &reports[1..] {
                compare_outputs(&plan, &reports[0], report)?;
            }
            Ok(reports)
        });
        match checked {
            Ok(reports) => {
                for ((runtime, _), report) in sessions.iter().zip(&reports) {
                    print(runtime, name, report);
                }
            }
            Err(why) => {
                println!("{name}: FAILED: {why}");
                matched = false;
            }
        }
    }
    Ok(matched)
}

/// `report`'s row, and its matrix line where it timed a run.
fn print(runtime: &str, name: &str, report: &Report) {
    let gpu = |p| percentile(&report.gpu_us, p).map_or("-".to_owned(), |us| format!("{us:.1}"));
    let per_dispatch = |values: &[f64]| {
        percentile(values, 50).map_or("-".to_owned(), |ns| {
            format!("{:.0}", ns / report.dispatches as f64)
        })
    };
    let gauges = match report.gauges.as_slice() {
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
    };
    println!(
        "{name:<17} {runtime:<8} {:>10} {:>11} {:>11} {:>11} {:>18} {:>15}  {gauges}",
        report.dispatches,
        gpu(10),
        gpu(50),
        gpu(90),
        per_dispatch(&report.encode_ns),
        per_dispatch(&report.cpu_ns),
    );
    if let Some(median) = percentile(&report.gpu_us, 50) {
        println!("M2V-MATRIX time harness-{runtime}/{name} {median:.3}");
    }
}
