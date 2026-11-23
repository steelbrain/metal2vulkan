//! `corpus-triage --audit specialization --profile P...`: plan every indexed row's
//! function-constant value sets, then, for each profile, translate each row under each planned
//! set, deriving the pipeline inputs its AIR states
//! ([`metal2vulkan_validation::specialization::attempt`]), each attempt in its own killable worker
//! under the 20-s and 500-MiB ceilings. Plans and attempts go to the lane's own index tables
//! ([`metal2vulkan_validation::specialization::audit`]); no other census moves.

use super::{
    print_budget_report, report_transient, run_worker_process, single_line, sweep_translation_rows,
    write_translation_worker_input, Lane, SweepRow, WorkerLane,
};
use metal2vulkan::library::FunctionConstantValue;
use metal2vulkan::Budget;
use metal2vulkan_air::budget::Meter;
use metal2vulkan_validation::bitcode_audit::peak_rss_bytes;
use metal2vulkan_validation::budgets::{BudgetAccumulator, BudgetReport};
use metal2vulkan_validation::engine::{self, BitcodeError, FailureKind, Outcome};
use metal2vulkan_validation::source::{for_each_indexed_source_with_blob, SourceRow};
use metal2vulkan_validation::specialization::attempt::{
    translate_specialized, Linkage, SpecAttempt,
};
use metal2vulkan_validation::specialization::audit::{self, PendingSet, PlanSummary, SpecResult};
use metal2vulkan_validation::specialization::{AuthoredLinks, RowPlan};
use metal2vulkan_validation::store::CorpusStore;
use metal2vulkan_validation::translation_audit::{
    engine_fingerprint, CostRow, RecordedCost, WorkerMeasure,
};
use metal2vulkan_validation::worker;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

/// The worker argument that plans one row.
pub(super) const PLAN_WORKER: &str = "--specialization-plan";
/// The worker argument that attempts one row under one set; the profile follows it.
pub(super) const SET_WORKER: &str = "--specialization-set";

/// How one run selects and reports: its workers (`--jobs`), its report's detail
/// (`--summary-only`), its limit (`--limit`), and its rows (`--hash-file`; every row without it).
#[derive(Clone, Copy)]
pub(super) struct Run<'a> {
    pub(super) jobs: usize,
    pub(super) summary_only: bool,
    pub(super) limit: usize,
    pub(super) only: Option<&'a BTreeSet<String>>,
}

/// Plan at most `limit` rows that need it, then attempt each profile's pending sets of the planned
/// rows, at most `limit` attempts per profile (whole rows); with `only`, of its rows alone, each
/// reported on a `specialization-row` line per profile. Fails when a plan or an attempt was
/// stopped at, or ran above, the 20-s or 500-MiB ceilings, or an attempt's ledger peak is below
/// the floor of its allocator peak, after reporting everything. The lane has no separate budgets
/// audit, so its own exit is the budget gate of its attempts.
pub(super) fn audit_specialization(
    root: &Path,
    index: &Path,
    run: Run<'_>,
    profiles: &[String],
) -> Result<(), String> {
    // An attempt resolves a visible function's module through the library-module index, which
    // this keeps current (incrementally: a warm one opens no module shard).
    let linked = metal2vulkan_validation::library_module::sync_library_module_index(root, index)?;
    println!(
        "specialization-library-modules\tshards_scanned={}\tbytes_scanned={}",
        linked.shards_scanned, linked.bytes_scanned
    );
    let mut breached = Vec::new();
    let (plan_breaches, unplanned) = plan_rows(root, index, run)?;
    if !plan_breaches.is_empty() {
        breached.push(format!("planning ({})", plan_breaches.len()));
    }
    let planner = audit::planner_key();
    let plans = audit::plans(index, &planner)?;
    print_plan_summary(&planner, &plans, unplanned, run.summary_only);
    for profile in profiles {
        let breaches = attempt_profile(root, index, run, profile, &plans)?;
        if !breaches.is_empty() {
            breached.push(format!("{profile} ({})", breaches.len()));
        }
    }
    if breached.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "specialization work failed its budget gates (the 20-s/500-MiB ceilings, the ledger \
             floor) for {} (specialization-*-budgets-failure lines name them)",
            breached.join(", ")
        ))
    }
}

/// One authored case's function-constant values, by index.
type AuthoredSet = BTreeMap<u32, FunctionConstantValue>;

/// What a planning worker reads: the row, its authored cases' value sets, and the functions they
/// link (kept with the plan).
#[derive(Serialize, Deserialize)]
struct PlanInput<'a> {
    row: engine::RowInput<'a>,
    authored: Vec<AuthoredSet>,
    #[serde(default)]
    links: AuthoredLinks,
}

/// How a row's planning ended.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Planned {
    /// The plan, recorded (a row that gives no bitcode or does not decode has a refused plan).
    Plan(RowPlan),
    /// A stop that is not a property of the row (a deadline, a worker stop, tooling): reported
    /// and not recorded, so the next run plans the row again.
    Stopped { outcome: Outcome, detail: String },
}

#[derive(Serialize, Deserialize)]
struct PlanOutput {
    key: String,
    planned: Planned,
}

/// A row to plan as the parent holds it.
struct PlanRow {
    input: PlanInput<'static>,
    cases: Vec<String>,
    lane: Lane,
    size: usize,
}

impl SweepRow for PlanRow {
    fn key(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(&self.input.row.air_sha256)
    }

    fn lane(&self) -> Lane {
        self.lane
    }

    fn size(&self) -> usize {
        self.size
    }
}

struct PlanResult {
    key: String,
    cases: Vec<String>,
    planned: Planned,
    worker: Option<WorkerMeasure>,
}

/// The value sets of the authored cases `ids` that run `row` (the index's case ids for its key)
/// and the functions they link, read from the case shard of its module. A shard that does not
/// hold exactly those cases means the index is stale: an error, never a silently different plan.
fn authored_inputs(
    store: &CorpusStore,
    row: &SourceRow,
    ids: &[String],
) -> Result<(Vec<AuthoredSet>, AuthoredLinks), String> {
    let cases = store.find_cases_for_row(row)?;
    let mut read = cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect::<Vec<_>>();
    read.sort();
    if read != ids {
        return Err(format!(
            "row {}: the index lists cases {ids:?}, its case shard holds {read:?}; sync the index",
            row.key()
        ));
    }
    let sets = cases
        .iter()
        .map(|case| {
            case.function_constants
                .iter()
                .map(|constant| {
                    let ty = constant
                        .scalar_type
                        .metal_data_type(constant.lanes)
                        .ok_or_else(|| {
                            format!(
                                "case {}: constant {} of {:?}x{} has no MTLDataType",
                                case.case_id, constant.index, constant.scalar_type, constant.lanes
                            )
                        })?;
                    let bytes = base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD,
                        &constant.bytes_b64,
                    )
                    .map_err(|error| {
                        format!(
                            "case {}: constant {} bytes: {error}",
                            case.case_id, constant.index
                        )
                    })?;
                    Ok((constant.index, FunctionConstantValue::new(ty, bytes)))
                })
                .collect::<Result<BTreeMap<_, _>, String>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((sets, AuthoredLinks::of(&cases)))
}

/// Plan at most `limit` rows without a current plan (or whose authored cases changed), in hash
/// order, each in a killable worker, checkpointing the plans. Returns the ceiling breaches and how
/// many rows are still to plan (a later run plans them; until then no set of theirs is selected).
fn plan_rows(root: &Path, index: &Path, run: Run<'_>) -> Result<(Vec<String>, usize), String> {
    let Run {
        jobs,
        summary_only,
        limit,
        only,
    } = run;
    let planner = audit::planner_key();
    let started = Instant::now();
    let cases = audit::case_ids(index)?;
    let keys = audit::rows_to_plan(index, &planner, &cases, only, limit)?;
    let selection_ms = started.elapsed().as_millis();
    let store = CorpusStore::new(root);
    let mut budgets = BudgetAccumulator::new();
    let (mut planned, mut stopped) = (0usize, 0usize);
    let sweep = sweep_translation_rows(
        &keys,
        jobs,
        "specialization-plan",
        &|keys, consume| {
            for_each_indexed_source_with_blob(root, index, keys, |row| {
                let ids = cases.get(row.key().as_ref()).cloned().unwrap_or_default();
                let (authored, links) = if ids.is_empty() {
                    (Vec::new(), AuthoredLinks::default())
                } else {
                    authored_inputs(&store, &row, &ids)?
                };
                let (lane, size) = (row.lane(), row.size());
                consume(PlanRow {
                    input: PlanInput {
                        row: engine::RowInput::from(row),
                        authored,
                        links,
                    },
                    cases: ids,
                    lane,
                    size,
                })
            })
        },
        &|row| plan_in_worker(root, index, row),
        &mut |result: &PlanResult, _| {
            let outcome = match &result.planned {
                Planned::Plan(_) => {
                    planned += 1;
                    ("planned".to_string(), String::new())
                }
                Planned::Stopped { outcome, detail } => {
                    stopped += 1;
                    println!(
                        "specialization-plan-stopped\t{}\t{}\t{}\t{}",
                        result.key,
                        outcome.label(),
                        outcome.key(),
                        engine::bounded_detail(detail)
                    );
                    (outcome.label().to_string(), outcome.key())
                }
            };
            budgets.add(&CostRow {
                air_sha256: result.key.clone(),
                outcome,
                cost: RecordedCost {
                    worker: result.worker,
                    ..RecordedCost::default()
                },
            });
        },
        &mut |checkpoint: &[PlanResult]| {
            let plans = checkpoint
                .iter()
                .filter_map(|result| match &result.planned {
                    Planned::Plan(plan) => {
                        Some((result.key.clone(), result.cases.clone(), plan.clone()))
                    }
                    Planned::Stopped { .. } => None,
                })
                .collect::<Vec<_>>();
            audit::write_plans(index, &planner, &plans)
        },
    )?;
    let unplanned = audit::rows_to_plan(index, &planner, &cases, only, usize::MAX)?.len();
    let (slowest_row, slowest_ms) = sweep.slowest();
    println!(
        "specialization-plan-batch\tplanner={planner}\tselected={}\tplanned={planned}\tunplanned_remaining={unplanned}\tstopped_not_recorded={stopped}\tindex_select_ms={selection_ms}\tplan_ms={}\tslowest_air_sha256={slowest_row}\tslowest_ms={slowest_ms}\tparent_peak_rss_bytes={}\t{}\tindexed_rows={}\treread_rows={}\tsource_shards_opened={}\tsource_bytes_read={}",
        sweep.rows,
        sweep.wall.as_millis(),
        peak_rss_bytes(),
        sweep.parent,
        sweep.read.rows,
        sweep.reread,
        sweep.read.source_shards_opened,
        sweep.read.source_bytes_read,
    );
    let report = budgets.finish();
    print_budget_report("specialization-plan-budgets", "all", &report, summary_only);
    Ok((report.failures(), unplanned))
}

/// One row's planning through the killable worker; a worker-boundary stop is reported, not
/// recorded.
fn plan_in_worker(root: &Path, index: &Path, row: PlanRow) -> PlanResult {
    let PlanRow { input, cases, .. } = row;
    let key = input.row.air_sha256.to_string();
    let stopped =
        |outcome: Outcome, detail: String, measured: Option<worker::Measured>| PlanResult {
            key: key.clone(),
            cases: cases.clone(),
            planned: Planned::Stopped { outcome, detail },
            worker: measured.map(WorkerMeasure::from),
        };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_worker_process::<PlanOutput>(root, index, WorkerLane::SpecializationPlan, |stdin| {
            write_translation_worker_input(stdin, &input)
        })
    })) {
        Ok(Ok(run)) if run.result.key == key => {
            let measured = run.measured();
            PlanResult {
                key: key.clone(),
                cases: cases.clone(),
                planned: run.result.planned,
                worker: Some(WorkerMeasure::from(measured)),
            }
        }
        Ok(Ok(run)) => stopped(
            Outcome::Failed {
                failure: FailureKind::WorkerExit,
            },
            format!("worker planned row {} for {key}", run.result.key),
            Some(run.measured()),
        ),
        Ok(Err(failure)) => stopped(failure.outcome(), failure.message(), failure.measured()),
        Err(_) => stopped(
            Outcome::Failed {
                failure: FailureKind::Tooling,
            },
            "the parent panicked while supervising the planning worker".into(),
            None,
        ),
    }
}

/// The `--specialization-plan` worker: decode the row under the deadline and plan it.
pub(super) fn plan_worker(started: Instant) -> Result<(), String> {
    let input: PlanInput<'static> = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|error| format!("decode planning worker input: {error}"))?;
    worker::report(&plan_row(input, started.checked_add(worker::DEADLINE)))
}

/// The plan of one row: its bitcode decoded and its declared constants planned with its authored
/// sets. A row whose text `llvm-as` rejects, or whose bitcode does not decode, has a refused plan
/// with the reason; a transient stop (deadline, tooling, a corrupt blob) is reported unrecorded.
fn plan_row(input: PlanInput<'_>, deadline: Option<Instant>) -> PlanOutput {
    let key = input.row.air_sha256.to_string();
    let refused = |why: String| {
        Planned::Plan(RowPlan {
            refused: Some(engine::bounded_detail(&why)),
            ..RowPlan::default()
        })
    };
    let planned = match input.row.source.bitcode() {
        Err(error @ BitcodeError::Rejected(_)) => refused(error.to_string()),
        Err(error) => Planned::Stopped {
            outcome: error.attempt().outcome,
            detail: error.to_string(),
        },
        Ok(bitcode) => {
            let mut meter = Meter::new(&Budget::with_deadline(deadline));
            match metal2vulkan_air::decode(&bitcode, &mut meter)
                .map_err(metal2vulkan::error::Error::from)
            {
                Ok(decoded) => Planned::Plan(RowPlan {
                    links: input.links,
                    ..RowPlan::of(&decoded, &input.row.entry, &input.authored)
                }),
                Err(error) => {
                    let outcome = Outcome::from(&error);
                    if outcome.is_transient() {
                        Planned::Stopped {
                            outcome,
                            detail: format!("decode: {error}"),
                        }
                    } else {
                        refused(format!("decode: {error}"))
                    }
                }
            }
        }
    };
    PlanOutput { key, planned }
}

/// What one set's worker reads, as the parent writes it.
#[derive(Serialize)]
struct SetRequest<'a> {
    row: &'a engine::RowInput<'a>,
    libraries: &'a [String],
    links: &'a AuthoredLinks,
    set: &'a PendingSet,
}

/// What one set's worker reads.
#[derive(Deserialize)]
struct SetInput {
    row: engine::RowInput<'static>,
    libraries: Vec<String>,
    #[serde(default)]
    links: AuthoredLinks,
    set: PendingSet,
}

/// A row to attempt as the parent holds it: its worker input, its parent libraries and the
/// functions its authored cases link (where a visible function's module is found), and its
/// pending sets.
struct SetRow {
    input: engine::RowInput<'static>,
    libraries: Vec<String>,
    links: AuthoredLinks,
    sets: Vec<PendingSet>,
    lane: Lane,
    size: usize,
}

impl SweepRow for SetRow {
    fn key(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(&self.input.air_sha256)
    }

    fn lane(&self) -> Lane {
        self.lane
    }

    fn size(&self) -> usize {
        self.size
    }
}

/// A row's attempts, one per pending set, and the lane they ran in.
struct RowResults {
    lane: Lane,
    results: Vec<SpecResult>,
}

/// Per lane: rows, attempts, and the summed wall time of its rows.
#[derive(Default)]
struct LaneTally {
    rows: usize,
    attempts: usize,
    row_ms: u128,
}

/// Attempt `profile_stem`'s pending sets, at most `limit` (whole rows), and report the profile's
/// recorded attempts. Returns the batch's budget failures: its ceiling breaches and its ledger
/// shortfalls.
fn attempt_profile(
    root: &Path,
    index: &Path,
    run: Run<'_>,
    profile_stem: &str,
    plans: &BTreeMap<String, RowPlan>,
) -> Result<Vec<String>, String> {
    let Run {
        jobs,
        summary_only,
        limit,
        only,
    } = run;
    let profile = engine::load_profile(profile_stem)?;
    let fingerprint = audit::fingerprint(profile_stem, &profile);
    let plain = engine_fingerprint(profile_stem, &profile);
    drop(profile);
    let started = Instant::now();
    let input_rows = audit::plain_input_rows(index, &plain)?;
    let pending = audit::pending(index, &fingerprint, plans, &input_rows, only, limit)?;
    let selection_ms = started.elapsed().as_millis();
    let keys = pending.keys().cloned().collect::<Vec<_>>();
    let selected_attempts = pending.values().map(Vec::len).sum::<usize>();
    let mut budgets = BudgetAccumulator::new();
    let mut lanes = BTreeMap::<&'static str, LaneTally>::new();
    let mut outcomes = BTreeMap::<(String, String), usize>::new();
    let (mut translated, mut defects, mut transient) = (0usize, 0usize, 0usize);
    let sweep = sweep_translation_rows(
        &keys,
        jobs,
        "specialization-audit",
        &|keys, consume| {
            for_each_indexed_source_with_blob(root, index, keys, |row| {
                let key = row.key().into_owned();
                let sets = pending
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| format!("row {key} was read but not selected"))?;
                let (lane, size) = (row.lane(), row.size());
                let libraries = row.lib_sha256s.clone();
                let links = plans
                    .get(&key)
                    .map(|plan| plan.links.clone())
                    .unwrap_or_default();
                consume(SetRow {
                    input: engine::RowInput::from(row),
                    libraries,
                    links,
                    sets,
                    lane,
                    size,
                })
            })
        },
        &|row| attempt_row(root, index, row, profile_stem),
        &mut |row: &RowResults, elapsed_ms| {
            let tally = lanes.entry(lane_name(row.lane)).or_default();
            tally.rows += 1;
            tally.attempts += row.results.len();
            tally.row_ms += elapsed_ms;
            for result in &row.results {
                let outcome = &result.result.attempt.outcome;
                let (label, key) = (outcome.label().to_string(), outcome.key());
                translated += usize::from(*outcome == Outcome::Translated);
                defects += usize::from(outcome.is_defect());
                if !summary_only {
                    println!(
                        "specialization-audit\t{}\t{}\t{}\t{label}\t{key}",
                        result.air_sha256, result.spec_id, result.kinds
                    );
                }
                if outcome.is_transient() {
                    report_transient(
                        "specialization-audit",
                        profile_stem,
                        &attempt_name(result),
                        (&label, &key),
                        result.result.attempt.detail.as_deref(),
                        result.worker,
                    );
                }
                budgets.add(&CostRow {
                    air_sha256: attempt_name(result),
                    outcome: (label.clone(), key.clone()),
                    cost: RecordedCost::of(&result.as_plain()).unwrap_or_default(),
                });
                *outcomes.entry((label, key)).or_default() += 1;
            }
        },
        &mut |checkpoint: &[RowResults]| {
            transient += audit::write_results(
                index,
                profile_stem,
                &fingerprint,
                checkpoint.iter().flat_map(|row| &row.results),
            )?;
            Ok(())
        },
    )?;
    let (slowest_row, slowest_ms) = sweep.slowest();
    println!(
        "specialization-audit-summary\tprofile={profile_stem}\tfingerprint={fingerprint}\tplain_fingerprint={plain}\tselected_rows={}\tselected_attempts={selected_attempts}\ttranslated={translated}\tdefects={defects}\ttransient_not_recorded={transient}\tindex_select_ms={selection_ms}\twall_ms={}\tslowest_row_air_sha256={slowest_row}\tslowest_row_ms={slowest_ms}\tparent_peak_rss_bytes={}\t{}\tindexed_rows={}\treread_rows={}\tsource_shards_opened={}\tsource_bytes_read={}",
        sweep.rows,
        sweep.wall.as_millis(),
        peak_rss_bytes(),
        sweep.parent,
        sweep.read.rows,
        sweep.reread,
        sweep.read.source_shards_opened,
        sweep.read.source_bytes_read,
    );
    for (lane, tally) in &lanes {
        println!(
            "specialization-audit-lane\tprofile={profile_stem}\tlane={lane}\trows={}\tattempts={}\trow_ms_sum={}",
            tally.rows, tally.attempts, tally.row_ms
        );
    }
    for ((label, key), count) in &outcomes {
        println!("specialization-audit-outcome\t{profile_stem}\t{count}\t{label}\t{key}");
    }
    let batch = budgets.finish();
    print_budget_report(
        "specialization-audit-budgets",
        profile_stem,
        &batch,
        summary_only,
    );
    print_profile_report(index, profile_stem, &fingerprint, &plain, run)?;
    Ok(batch.failures())
}

fn lane_name(lane: Lane) -> &'static str {
    match lane {
        Lane::Small => "small",
        Lane::Serialized => "serialized",
        Lane::Large => "large",
    }
}

/// An attempt's name in budget lines: its row and a prefix of its set's id.
fn attempt_name(result: &SpecResult) -> String {
    let id = result.spec_id.get(..12).unwrap_or(&result.spec_id);
    format!("{}/{id}", result.air_sha256)
}

/// Each pending set of `row`, one after another, each in its own killable worker.
fn attempt_row(root: &Path, index: &Path, row: SetRow, profile: &str) -> RowResults {
    let results = row
        .sets
        .iter()
        .map(|set| attempt_set(root, index, &row, set, profile))
        .collect();
    RowResults {
        lane: row.lane,
        results,
    }
}

/// One set's attempt through the killable worker; a worker-boundary stop is a typed outcome with
/// what the parent measured up to it.
fn attempt_set(
    root: &Path,
    index: &Path,
    row: &SetRow,
    set: &PendingSet,
    profile: &str,
) -> SpecResult {
    let key = row.input.air_sha256.to_string();
    let failed =
        |outcome: Outcome, detail: String, measured: Option<worker::Measured>| SpecResult {
            air_sha256: key.clone(),
            spec_id: set.id.clone(),
            kinds: set.kinds_key(),
            result: SpecAttempt {
                attempt: engine::Attempt {
                    outcome,
                    detail: Some(engine::bounded_detail(&detail)),
                    words: None,
                    cost: None,
                },
                supplied: Vec::new(),
                unsupplied: None,
                ray_tracing: false,
            },
            worker: measured.map(WorkerMeasure::from),
        };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_worker_process::<SpecResult>(
            root,
            index,
            WorkerLane::SpecializationSet(profile),
            |stdin| {
                write_translation_worker_input(
                    stdin,
                    &SetRequest {
                        row: &row.input,
                        libraries: &row.libraries,
                        links: &row.links,
                        set,
                    },
                )
            },
        )
    })) {
        Ok(Ok(run)) if run.result.air_sha256 == key && run.result.spec_id == set.id => {
            let measured = run.measured();
            SpecResult {
                worker: Some(WorkerMeasure::from(measured)),
                ..run.result
            }
        }
        Ok(Ok(run)) => failed(
            Outcome::Failed {
                failure: FailureKind::WorkerExit,
            },
            format!(
                "worker reported {}/{} for {key}/{}",
                run.result.air_sha256, run.result.spec_id, set.id
            ),
            Some(run.measured()),
        ),
        Ok(Err(failure)) => failed(failure.outcome(), failure.message(), failure.measured()),
        Err(_) => failed(
            Outcome::Failed {
                failure: FailureKind::Tooling,
            },
            "the parent panicked while supervising the specialization worker".into(),
            None,
        ),
    }
}

/// The `--specialization-set PROFILE` worker: [`attempt_input`] of its input under the deadline.
pub(super) fn set_worker(
    corpus_root: &Path,
    index: &Path,
    profile: &str,
    started: Instant,
) -> Result<(), String> {
    let input: SetInput = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|error| format!("decode specialization worker input: {error}"))?;
    let profile = engine::load_profile(profile)?;
    let (result, pending) = attempt_input(
        input,
        corpus_root,
        index,
        &profile,
        started.checked_add(worker::DEADLINE),
    );
    match pending {
        // `spirv-val` runs in a fresh image of this executable, which reports.
        Some(pending) => worker::l3::report_after_l3(SET_STAGE, &result, pending),
        None => worker::report(&result),
    }
}

/// The L3 stage of a `--specialization-set` worker ([`worker::l3`]).
pub(super) const SET_STAGE: &str = "specialization-set";

/// The L3 stage of a `--specialization-set` worker on the pending file `pending`.
pub(super) fn set_stage(pending: &Path) -> Result<(), String> {
    worker::l3::serve(pending, |result: &mut SpecResult| {
        &mut result.result.attempt
    })
}

/// One set's attempt inside the worker up to L3's `spirv-val`: the row's bitcode translated
/// under the set, with the row's parent libraries in the index for the visible functions it
/// references, and what L3 still has to run on a module it linked; a row that gives no bitcode is
/// that attempt ([`BitcodeError::attempt`]).
fn attempt_input(
    input: SetInput,
    corpus_root: &Path,
    index: &Path,
    profile: &metal2vulkan::device::DeviceProfile,
    deadline: Option<Instant>,
) -> (SpecResult, Option<engine::PendingL3>) {
    let (result, pending) = match input.row.source.bitcode() {
        Ok(bitcode) => translate_specialized(
            &bitcode,
            &input.row.entry,
            &input.set.values(),
            profile,
            Some(Linkage {
                corpus_root,
                index,
                parent_libraries: &input.libraries,
                authored: &input.links,
            }),
            deadline,
        ),
        Err(error) => (
            SpecAttempt {
                attempt: error.attempt(),
                supplied: Vec::new(),
                unsupplied: None,
                ray_tracing: false,
            },
            None,
        ),
    };
    let result = SpecResult {
        air_sha256: input.row.air_sha256.into_owned(),
        kinds: input.set.kinds_key(),
        spec_id: input.set.id,
        result,
        worker: None,
    };
    (result, pending)
}

/// The plans' summary: rows, sets by kind, refusals, the declarations' facts (the fifth
/// operand's agreement with "required"), and every plan over the 16-set bound.
fn print_plan_summary(
    planner: &str,
    plans: &BTreeMap<String, RowPlan>,
    unplanned: usize,
    summary_only: bool,
) {
    let summary = PlanSummary::of(plans);
    let facts = summary.facts;
    println!(
        "specialization-plan-summary\tplanner={planner}\tunplanned_rows={unplanned}\trows={}\trows_with_constants={}\tsets={}\tover_cap_rows={}\trefused_rows={}\tdeclared_constants={}\toptional_constants={}\tflag_agrees={}\tflag_disagrees={}\tflag_absent={}",
        summary.rows,
        summary.with_constants,
        summary.sets,
        summary.over_cap.len(),
        summary.refused.values().sum::<usize>(),
        facts.declared,
        facts.optional,
        facts.flag_agrees,
        facts.flag_disagrees,
        facts.flag_absent,
    );
    for (kind, sets) in &summary.kinds {
        println!("specialization-plan-kind\t{}\tsets={sets}", kind.as_str());
    }
    // Sets under which two live arguments share a slot: planned apart, never attempted.
    println!(
        "specialization-plan-aliased\trows={}\tsets={}",
        summary.aliased_rows,
        summary.aliased.values().sum::<usize>()
    );
    for (table, sets) in &summary.aliased {
        println!("specialization-plan-aliased-table\t{table}\tsets={sets}");
    }
    // A plan over the bound is attempted whole; each one is named.
    for (row, sets) in &summary.over_cap {
        println!("specialization-plan-over-cap\t{row}\tsets={sets}");
    }
    let mut refused = summary.refused.iter().collect::<Vec<_>>();
    refused.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let shown = if summary_only { 20 } else { usize::MAX };
    for (why, rows) in refused.iter().take(shown) {
        println!(
            "specialization-plan-refused\trows={rows}\t{}",
            single_line(why)
        );
    }
}

/// The profile's report over every attempt recorded under `fingerprint`: rows by plain outcome,
/// outcomes by kind, the inputs derived and left, the refusal classes ranked by rows (ray tracing
/// listed apart, out of scope for the launch), the recorded attempts' budgets, and each selected
/// row's verdict ([`audit::row_verdicts`]).
fn print_profile_report(
    index: &Path,
    profile_stem: &str,
    fingerprint: &str,
    plain: &str,
    run: Run<'_>,
) -> Result<(), String> {
    let Run {
        summary_only, only, ..
    } = run;
    let attempts = audit::recorded(index, profile_stem, fingerprint)?;
    let report = audit::report(&attempts);
    println!(
        "specialization-rows\tprofile={profile_stem}\trows={}\ttranslated_under_some_set={}\ttranslated_under_every_set={}\tattempts={}\tdefects={}",
        report.rows,
        report.rows_translated_some,
        report.rows_translated_all,
        report.total.attempts,
        report.total.defects,
    );
    let plain_outcomes = audit::plain_outcomes(index, plain)?;
    for ((label, key), class) in audit::rows_by_plain_outcome(&attempts, &plain_outcomes) {
        println!(
            "specialization-plain-class\tprofile={profile_stem}\tplain={label}\tplain_key={key}\trows={}\ttranslated_under_some_set={}\ttranslated_under_every_set={}",
            class.rows, class.translated_some, class.translated_all
        );
    }
    for (kind, counts) in &report.kinds {
        let labels = counts
            .labels
            .iter()
            .map(|(label, n)| format!("{label}={n}"))
            .collect::<Vec<_>>()
            .join("\t");
        println!(
            "specialization-kind\tprofile={profile_stem}\tkind={}\tattempts={}\tdefects={}\t{labels}",
            kind.as_str(),
            counts.attempts,
            counts.defects
        );
    }
    for (kind, (attempts, rows)) in &report.inputs.supplied {
        println!(
            "specialization-input-supplied\tprofile={profile_stem}\tinput={kind}\tattempts={attempts}\trows={}",
            rows.len()
        );
    }
    for ((key, why), rows) in &report.inputs.unsupplied {
        println!(
            "specialization-input-unsupplied\tprofile={profile_stem}\tinput={key}\trows={}\t{}",
            rows.len(),
            single_line(why)
        );
    }
    let shown = if summary_only { 40 } else { usize::MAX };
    let (ray_tracing, holes): (Vec<_>, Vec<_>) =
        report.holes.iter().partition(|hole| hole.ray_tracing);
    for (rank, hole) in holes.iter().enumerate().take(shown) {
        print_hole("specialization-hole", profile_stem, rank + 1, hole);
    }
    if holes.len() > shown {
        println!(
            "specialization-hole\tprofile={profile_stem}\t... {} more (run without --summary-only)",
            holes.len() - shown
        );
    }
    for (rank, hole) in ray_tracing.iter().enumerate() {
        print_hole("specialization-ray-tracing", profile_stem, rank + 1, hole);
    }
    let rows = attempts
        .iter()
        .map(|attempt| CostRow {
            air_sha256: format!(
                "{}/{}",
                attempt.air_sha256,
                attempt.spec_id.get(..12).unwrap_or(&attempt.spec_id)
            ),
            outcome: attempt.outcome.clone(),
            cost: attempt.cost.clone().unwrap_or_default(),
        })
        .collect::<Vec<_>>();
    let recorded = BudgetReport::from_rows(&rows);
    print_budget_report("specialization-budgets", profile_stem, &recorded, true);
    let verdicts = only.map(|rows| audit::row_verdicts(rows, &attempts, &plain_outcomes));
    for (air, row) in verdicts.unwrap_or_default() {
        let refusals = row
            .refusals
            .iter()
            .map(|((label, key), sets)| format!("{label} {key} ({sets})"))
            .collect::<Vec<_>>()
            .join("; ");
        println!(
            "specialization-row\tprofile={profile_stem}\tair={}\tplain={}\tplain_key={}\tsets={}\ttranslated={}\tunder={}\trefusals={refusals}",
            air.get(..12).unwrap_or(&air),
            row.plain.0,
            row.plain.1,
            row.attempts,
            row.translated,
            row.under(),
        );
    }
    Ok(())
}

fn print_hole(label: &str, profile_stem: &str, rank: usize, hole: &audit::Hole) {
    let kinds = hole
        .kinds
        .iter()
        .map(|kind| kind.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let (example_row, example_detail) =
        hole.example.as_ref().map_or(("none", ""), |(row, detail)| {
            (row.as_str(), detail.as_str())
        });
    println!(
        "{label}\tprofile={profile_stem}\trank={rank}\trows={}\tattempts={}\toutcome={}\tkey={}\tkinds={kinds}\tvalue_dependent_rows={}\tzero_only_rows={}\tdefect={}\texample={example_row}\t{}",
        hole.rows.len(),
        hole.attempts,
        hole.label,
        hole.key,
        hole.value_dependent_rows,
        hole.zero_only_rows,
        hole.defect,
        single_line(example_detail),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan_devtools::fixtures_dir;
    use metal2vulkan_validation::specialization::SpecKind;

    fn row(text: String, blob_b64: Option<String>) -> SourceRow {
        SourceRow {
            air_sha256: "12".repeat(32),
            stage: "Kernel".into(),
            entry: "fcfold".into(),
            air_ll: text,
            blob_b64,
            lib_sha256s: vec!["34".repeat(32)],
            label: "public/synthetic.ll".into(),
        }
    }

    /// A row's planning input crosses the worker boundary as JSON and plans the row: a row with
    /// a required constant gets the planned sets; text `llvm-as` rejects is a recorded refused
    /// plan; a blob that is not base64 is a transient stop, not recorded.
    #[test]
    fn planning_crosses_the_worker_boundary() {
        let text = fs_text(&fixtures_dir().join("kernel_function_constant_gate_fold.ll"));
        let plan = |source: SourceRow| {
            let input = PlanInput {
                row: engine::RowInput::from(source),
                authored: Vec::new(),
                links: AuthoredLinks::default(),
            };
            let json = serde_json::to_vec(&input).unwrap();
            let input: PlanInput<'static> = serde_json::from_slice(&json).unwrap();
            let output = plan_row(input, None);
            assert_eq!(output.key, "12".repeat(32));
            output.planned
        };
        match plan(row(text, None)) {
            Planned::Plan(plan) => {
                assert!(plan.refused.is_none(), "{plan:?}");
                assert!(plan.specializations.len() >= 2, "{plan:?}");
                assert!(plan
                    .specializations
                    .iter()
                    .any(|set| set.kinds.contains(&SpecKind::One)));
            }
            Planned::Stopped { outcome, detail } if outcome.is_transient() => {
                eprintln!("SKIPPED: {detail}");
                return;
            }
            other => panic!("{other:?}"),
        }
        match plan(row("this is not LLVM assembly".into(), None)) {
            Planned::Plan(plan) => {
                assert!(plan.specializations.is_empty());
                assert!(plan.refused.unwrap().contains("llvm-as rejected"));
            }
            other => panic!("{other:?}"),
        }
        match plan(row(String::new(), Some("not base64!".into()))) {
            Planned::Stopped { outcome, .. } => assert!(outcome.is_transient()),
            other => panic!("{other:?}"),
        }
    }

    /// One set's input crosses the worker boundary as JSON: the attempt names its row and set,
    /// and the set of ones translates the row its undefined constant stops.
    #[test]
    fn a_set_crosses_the_worker_boundary() {
        let text = fs_text(&fixtures_dir().join("kernel_function_constant_gate_fold.ll"));
        let Planned::Plan(plan) = plan_row(
            PlanInput {
                row: engine::RowInput::from(row(text.clone(), None)),
                authored: Vec::new(),
                links: AuthoredLinks::default(),
            },
            None,
        )
        .planned
        else {
            eprintln!("SKIPPED: the row could not be assembled");
            return;
        };
        let one = plan
            .specializations
            .iter()
            .find(|set| set.kinds.contains(&SpecKind::One))
            .unwrap();
        let pending = audit::pending_set(one);
        let source = row(text, None);
        let input = engine::RowInput::from(source.clone());
        let links = AuthoredLinks {
            functions: BTreeMap::from([("f".to_string(), "ab".repeat(32))]),
            ..AuthoredLinks::default()
        };
        let json = serde_json::to_vec(&SetRequest {
            row: &input,
            libraries: &source.lib_sha256s,
            links: &links,
            set: &pending,
        })
        .unwrap();
        let input: SetInput = serde_json::from_slice(&json).unwrap();
        assert_eq!(input.libraries, source.lib_sha256s);
        assert_eq!(input.links, links);
        let dir = metal2vulkan_validation::ScratchDir::new("specialization-set-test").unwrap();
        let profile = engine::load_profile("runtime-floor").unwrap();
        let (mut result, l3) = attempt_input(
            input,
            dir.path(),
            &dir.path().join("index.sqlite"),
            &profile,
            None,
        );
        // What the worker's L3 stage settles, here in this process.
        let l3 = l3.expect("a module L3 has yet to check");
        result.result.attempt.settle_l3(l3.check());
        assert_eq!(result.air_sha256, "12".repeat(32));
        assert_eq!(result.spec_id, pending.id);
        assert_eq!(result.kinds, pending.kinds_key());
        assert_eq!(
            result.result.attempt.outcome,
            Outcome::Translated,
            "{result:?}"
        );
    }

    /// A case shard that does not hold exactly the cases the index lists for a row is an error
    /// that names the row, never a plan from other cases.
    #[test]
    fn authored_inputs_refuse_a_stale_index() {
        let dir = metal2vulkan_validation::ScratchDir::new("specialization-cases-test").unwrap();
        let store = CorpusStore::new(dir.path());
        let source = row(String::new(), None);
        let (sets, links) = authored_inputs(&store, &source, &[]).unwrap();
        assert!(sets.is_empty() && links.is_empty());
        let error = authored_inputs(&store, &source, &["case-1".to_string()]).unwrap_err();
        assert!(error.contains("sync the index"), "{error}");
    }

    fn fs_text(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap()
    }
}
