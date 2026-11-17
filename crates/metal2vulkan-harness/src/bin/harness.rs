//! The harness's command: the authored workloads on one runtime, checked, then timed (the crate's
//! docs).
//!
//! ```text
//! harness --runtime old [--runs N] [--warmup N] [--only NAME,NAME...]
//! ```
//!
//! It exits 0 when every workload it ran matched its reference, 1 when one did not or failed,
//! and 2 when it could not start.

use metal2vulkan_harness::run::{percentile, Harness, Report, Runs};
use metal2vulkan_harness::{authored, old, Code, Device, Result};
use std::process::ExitCode;

/// The command's arguments.
struct Args {
    runtime: String,
    runs: Runs,
    only: Option<Vec<String>>,
}

const USAGE: &str = "usage: harness --runtime old [--runs N] [--warmup N] [--only NAME,NAME...]";

fn main() -> ExitCode {
    let ran = parse(std::env::args().skip(1)).and_then(|args| match args.runtime.as_str() {
        "old" => run_all(&old::system_default()?, &args),
        other => Err(format!("no runtime {other}; the runtimes are: old").into()),
    });
    match ran {
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
        runtime: String::new(),
        runs: Runs {
            warmup: 5,
            timed: 200,
        },
        only: None,
    };
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("{flag} needs a value; {USAGE}"))
        };
        match flag.as_str() {
            "--runtime" => parsed.runtime = value()?,
            "--runs" => parsed.runs.timed = value()?.parse()?,
            "--warmup" => parsed.runs.warmup = value()?.parse()?,
            "--only" => {
                let names: Vec<String> = value()?.split(',').map(str::to_owned).collect();
                for name in &names {
                    if !authored::WORKLOADS.iter().any(|known| known.name == name) {
                        let known = authored::WORKLOADS.map(|known| known.name).join(", ");
                        return Err(
                            format!("no workload {name}; the workloads are: {known}").into()
                        );
                    }
                }
                parsed.only = Some(names);
            }
            _ => return Err(format!("unknown argument {flag}; {USAGE}").into()),
        }
    }
    if parsed.runtime.is_empty() {
        return Err(USAGE.into());
    }
    Ok(parsed)
}

/// Every selected workload on `device`; whether each matched its reference.
fn run_all<D: Device>(device: &D, args: &Args) -> Result<bool> {
    let runtime = &args.runtime;
    println!("device: {}", device.name());
    println!(
        "runtime: {runtime}; warmup runs {}, timed runs {}",
        args.runs.warmup, args.runs.timed
    );
    let air = authored::library_air()?;
    let library = device.new_library(Code::Air(&air))?;
    let harness = Harness::new(device, &library)?;
    println!(
        "{:<17} {:>10} {:>11} {:>11} {:>11} {:>18} {:>15}  gauges @100 -> @200",
        "workload",
        "dispatches",
        "gpu_us_p10",
        "gpu_us_p50",
        "gpu_us_p90",
        "encode_ns/dispatch",
        "cpu_ns/dispatch",
    );
    let mut matched = true;
    for workload in authored::WORKLOADS {
        let name = workload.name;
        if (args.only.as_ref()).is_some_and(|only| !only.iter().any(|selected| selected == name)) {
            continue;
        }
        match harness.run(&library, &(workload.plan)(), args.runs) {
            Ok(report) => print(runtime, name, &report),
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
        "{name:<17} {:>10} {:>11} {:>11} {:>11} {:>18} {:>15}  {gauges}",
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
