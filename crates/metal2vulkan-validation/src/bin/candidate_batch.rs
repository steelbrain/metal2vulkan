//! `corpus-vulkan --batch`, `--batch-report`, and `--batch-install`: the command line of
//! [`metal2vulkan_validation::vulkan_batch`], which runs this executable once per case.

use metal2vulkan_validation::gpu::tripwire::TripwirePaths;
use metal2vulkan_validation::gpu::wait::parse_wait_timeout;
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::observation::Backend;
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use metal2vulkan_validation::vulkan_batch::{
    self, plan, qualified_cases, read_done_list, report, requested_cases, run_batch, BatchBuild,
    BatchConfig, BatchDevice, BatchOutput, Class, Outcome, Pass, DONE_LIST,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

/// The per-run timeout when none is given: well past the compile worker's 20 s and the device
/// waits' bound, so only a run stuck outside both reaches it.
const DEFAULT_RUN_TIMEOUT_SECONDS: u64 = 300;

const USAGE: &str =
    "usage: corpus-vulkan --batch --device-type cpu|gpu --out DIR \\\n\
    --metal-environment-id ID --environment-id ID [--corpus DIR] \\\n\
    [--jobs N] [--run-timeout SECONDS] [--limit CASES] \\\n\
    [--case-id HASH]... [--gpu-wait-timeout SECONDS] [--profiles-layer P] [--source-index FILE]\n\
  corpus-vulkan --batch-report --out DIR --metal-environment-id ID --environment-id ID \\\n\
    [--corpus DIR] [--case-id HASH]... [--profiles-layer P]\n\
  corpus-vulkan --batch-install DIR --out DIR\n\
Each mode records, reports, and installs the Vulkan backend's rows, never Metal's. With --profiles-layer P\n\
every run is the executor's --profiles-layer P run (the device restricted to the committed profile\n\
P, which it must hold), its rows in the ID@engine@P slots; the build record names P.\n\
--batch runs every case with a current Metal row in --metal-environment-id (or each --case-id,\n\
which must have one) through this executable (the engine)\n\
with the --device-type, one process per case, each in its own session killed whole after\n\
--run-timeout seconds (default 300) and in its own scratch store (links to --corpus, which is\n\
never written). On a cpu device --jobs\n\
run at a time (default: the available parallelism, in a container its CPU share); on a gpu one\n\
at a time (--jobs 1, the default): the batch refuses to start a run while the host's GPU hang\n\
tripwire exists, and stops after a run whose device wait timed out, whose device worker crashed,\n\
or that it killed at --run-timeout (writing the tripwire if no one did). DIR/build.json records\n\
the build and venue of DIR's runs; a batch refuses a DIR of another build or venue, or one with\n\
runs and no record.\n\
Each finished run appends a line to DIR/done.tsv (case, pass, outcome, exit, seconds, detail),\n\
each row it recorded to DIR/rows.jsonl, and each refusal it recorded after a device opened to\n\
DIR/refusals.jsonl; a case done.tsv lists is skipped, so a rerun resumes. --limit bounds\n\
the cases of one invocation. Non-matching runs keep their output in DIR/logs/. Each run reads its\n\
case from the one case shard its AIR selects and its Metal row from its own Metal shard, and its\n\
private sources through --source-index (default: --corpus's .index.sqlite, when it has one) by\n\
byte location, read only, falling back to the source's shard wherever the index is missing,\n\
stale, or wrong.\n\
--batch-report prints DIR's build record against this build (batch-build), then, over\n\
the cases --batch selects with the same --case-id list (other cases' runs are not counted), the\n\
runs by class (match, mismatch, refusal, failure), then the same per family (batch-family: the\n\
stage and first resource family) and per scope (batch-scope: public or private AIR), and every\n\
run that did not match (batch-run, with its scope and family, and a refusal's recorded class).\n\
A last line of done.tsv, rows.jsonl, or refusals.jsonl without its newline (an append cut\n\
short) is dropped. --batch-install writes into the store at DIR, for each case, the\n\
evidence of its last run that recorded any (its row, or its refusal) through the store's\n\
validated writes, and prints batch-install rows=N refusals=N store=DIR this_build=yes|no|unrecorded:\n\
whether DIR/build.json is this build (executor, product, semantics, oracle, and validation\n\
sources), the only build whose rows are current once installed.";

/// Run a batch mode if the command line names one (its first argument): `true` when it did (and
/// it exited the process on an error), `false` for a single-case invocation, the workers' included.
pub fn serve(backend: Backend) -> bool {
    let mut args = std::env::args().skip(1);
    let Some(mode) = args.next().filter(|arg| {
        matches!(
            arg.as_str(),
            "--batch" | "--batch-report" | "--batch-install"
        )
    }) else {
        return false;
    };
    if let Err(error) = main(backend, &mode, args.collect()) {
        eprintln!("corpus-{} {mode}: {error}", backend.directory());
        std::process::exit(1);
    }
    true
}

/// Run the batch mode `mode` (`--batch`, `--batch-report`, `--batch-install`) of `backend` with
/// `args`.
fn main(backend: Backend, mode: &str, args: Vec<String>) -> Result<(), String> {
    let mut corpus = corpus_root();
    let mut out = None;
    let mut install = None;
    let mut metal_environment_id = None;
    let mut environment_id = None;
    let mut jobs = None;
    let mut run_timeout = DEFAULT_RUN_TIMEOUT_SECONDS;
    let mut limit = None;
    let mut case_ids = Vec::new();
    let mut gpu_wait_timeout = None;
    let mut device_type = None;
    let mut restricted_to = None;
    let mut source_index = None;
    let mut args = args.into_iter();
    if mode == "--batch-install" {
        install = Some(PathBuf::from(required(&mut args, mode)?));
    }
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => corpus = PathBuf::from(required(&mut args, "--corpus")?),
            "--out" => out = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--metal-environment-id" => {
                metal_environment_id = Some(required(&mut args, "--metal-environment-id")?)
            }
            "--environment-id" => environment_id = Some(required(&mut args, "--environment-id")?),
            "--jobs" => jobs = Some(positive(&required(&mut args, "--jobs")?, "--jobs")?),
            "--run-timeout" => {
                run_timeout =
                    positive(&required(&mut args, "--run-timeout")?, "--run-timeout")? as u64
            }
            "--limit" => limit = Some(positive(&required(&mut args, "--limit")?, "--limit")?),
            "--case-id" => case_ids.push(required(&mut args, "--case-id")?),
            "--gpu-wait-timeout" => {
                let value = required(&mut args, "--gpu-wait-timeout")?;
                gpu_wait_timeout = Some(parse_wait_timeout(&value)?.as_secs());
            }
            "--device-type" => device_type = Some(required(&mut args, "--device-type")?),
            "--profiles-layer" => restricted_to = Some(required(&mut args, "--profiles-layer")?),
            "--source-index" => {
                source_index = Some(PathBuf::from(required(&mut args, "--source-index")?))
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}\n{USAGE}")),
        }
    }
    let out = out.ok_or("--out DIR is required")?;
    let store = CorpusStore::new(&corpus);
    match mode {
        "--batch-install" => {
            let target = install.ok_or("--batch-install DIR is required")?;
            let installed = vulkan_batch::install(&out, &CorpusStore::new(&target), backend)?;
            println!(
                "batch-install rows={} refusals={} store={} this_build={}",
                installed.rows,
                installed.refusals,
                target.display(),
                match &installed.build {
                    Some(build) if build.is_this_build() => "yes",
                    Some(_) => "no",
                    None => "unrecorded",
                }
            );
            Ok(())
        }
        "--batch-report" => {
            let metal_environment_id =
                metal_environment_id.ok_or("--metal-environment-id is required")?;
            let environment_id = environment_id.ok_or("--environment-id is required")?;
            let recorded = BatchBuild::read(&out)?;
            // The report reads no device: the recorded device type is compared as it is.
            let this_build = match &recorded {
                Some(recorded) => Some(
                    *recorded
                        == BatchBuild::current(
                            backend.into(),
                            &metal_environment_id,
                            &environment_id,
                            restricted_to.as_deref(),
                            DeviceType::parse(&recorded.device_type)?,
                        ),
                ),
                None => None,
            };
            println!(
                "batch-build this_build={}{}",
                match this_build {
                    Some(true) => "yes",
                    Some(false) => "no",
                    None => "unrecorded",
                },
                recorded
                    .map(|recorded| format!(
                        " {}",
                        serde_json::to_string(&recorded).unwrap_or_default()
                    ))
                    .unwrap_or_default()
            );
            let metals = store.read_metal()?;
            let selected = requested_cases(
                qualified_cases(store.read_all_cases()?, &metals, &metal_environment_id),
                &case_ids,
                &metal_environment_id,
            )?;
            for line in report(
                &selected,
                &metals,
                &metal_environment_id,
                &environment_id,
                restricted_to.as_deref(),
                &BatchOutput::read(&out)?,
            ) {
                println!("{line}");
            }
            Ok(())
        }
        "--batch" => {
            // A profile the layer cannot present refuses every run alike: refused here, before
            // any, so no run fails and none is retried.
            if let Some(profile) = &restricted_to {
                metal2vulkan_validation::candidate::check_presentable(profile)?;
            }
            let device = match DeviceType::parse(
                device_type
                    .as_deref()
                    .ok_or("--batch needs --device-type cpu or gpu")?,
            )? {
                DeviceType::Cpu => BatchDevice::Cpu,
                DeviceType::Gpu => BatchDevice::Gpu(TripwirePaths::for_host()?),
            };
            let metal_environment_id =
                metal_environment_id.ok_or("--metal-environment-id is required")?;
            let environment_id = environment_id.ok_or("--environment-id is required")?;
            let qualified = qualified_cases(
                store.read_all_cases()?,
                &store.read_metal()?,
                &metal_environment_id,
            );
            let selected = qualified.len();
            let cases = requested_cases(qualified, &case_ids, &metal_environment_id)?;
            let done = read_done_list(&out.join(DONE_LIST))?;
            let runs = plan(&cases, &[Pass::Engine], &done, limit);
            let jobs = match (jobs, &device) {
                (Some(jobs), _) => jobs,
                (None, BatchDevice::Gpu(_)) => 1,
                (None, BatchDevice::Cpu) => std::thread::available_parallelism()
                    .map_err(|error| format!("available parallelism: {error}"))?
                    .get(),
            };
            // The corpus's own index, when no other is named and it has one.
            let source_index = source_index.or_else(|| {
                Some(metal2vulkan_validation::index::default_index_path(&corpus))
                    .filter(|index| index.is_file())
            });
            let config = BatchConfig {
                executor: std::env::current_exe()
                    .map_err(|error| format!("this executable: {error}"))?,
                backend: backend.into(),
                corpus,
                out,
                metal_environment_id,
                environment_id,
                restricted_to,
                device,
                jobs,
                timeout: Duration::from_secs(run_timeout),
                gpu_wait_timeout,
                source_index,
            };
            eprintln!(
                "batch: selected={selected} cases={} done_before={} runs={} device={} jobs={jobs} \
                 run_timeout={run_timeout}s",
                cases.len(),
                done.len(),
                runs.len(),
                config.device.device_type().as_str(),
            );
            let finished = run_batch(&config, &runs, |run| {
                println!(
                    "batch-run {} {} {} {} {:.3}s {}",
                    run.case_id,
                    run.pass.label(),
                    run.outcome.label(),
                    run.ended,
                    run.seconds,
                    run.detail
                );
            })?;
            let mut counts = BTreeMap::<Outcome, usize>::new();
            for run in &finished {
                *counts.entry(run.outcome).or_default() += 1;
            }
            let of = |class: Class| -> usize {
                counts
                    .iter()
                    .filter(|(outcome, _)| outcome.class() == class)
                    .map(|(_, count)| count)
                    .sum()
            };
            println!(
                "batch-summary ran={} match={} mismatch={} refusal={} failure={}",
                of(Class::Match) + of(Class::Mismatch) + of(Class::Refusal) + of(Class::Failure),
                of(Class::Match),
                of(Class::Mismatch),
                of(Class::Refusal),
                of(Class::Failure)
            );
            if let Some(slowest) = finished
                .iter()
                .max_by(|left, right| left.seconds.total_cmp(&right.seconds))
            {
                println!(
                    "batch-slowest {} {} {:.3}s {}",
                    slowest.case_id,
                    slowest.pass.label(),
                    slowest.seconds,
                    slowest.outcome.label()
                );
            }
            let pending = plan(
                &cases,
                &[Pass::Engine],
                &read_done_list(&config.out.join(DONE_LIST))?,
                None,
            )
            .len();
            println!("batch-pending runs={pending}");
            Ok(())
        }
        _ => Err(format!("unknown batch mode {mode:?}")),
    }
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn positive(value: &str, flag: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(number) if number > 0 => Ok(number),
        _ => Err(format!(
            "{flag} {value:?}: expected a positive whole number"
        )),
    }
}
