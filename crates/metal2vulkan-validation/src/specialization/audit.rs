//! The specialization audit's index tables, keys, selection, and report.
//!
//! Two tables of the disposable index hold the lane, apart from every other census:
//! - `specialization_plan`: each row's [`RowPlan`] under the planner key ([`planner_key`]: the
//!   product fingerprint, [`SPECIALIZATION_AUDIT_VERSION`], and the census tools), with the
//!   authored case ids its authored sets came from. It does not depend on a profile.
//! - `specialization_audit`: one attempt per `(row, set, fingerprint)`, the fingerprint being the
//!   profile's engine census key extended by [`SPECIALIZATION_AUDIT_VERSION`]
//!   ([`fingerprint`]).
//!
//! Nothing here writes `new_engine_translation_audit`: the plain census, C1, and the budgets audit
//! read only that table, so this lane moves none of their counts. Selection
//! reads the plain census only to find the rows whose plain compile needed another input (the
//! [`SpecKind::Inputs`] attempts), and refuses to run when that census is not complete for the
//! profile. A warm pass (every plan current, every planned set recorded) selects nothing.

use super::attempt::SpecAttempt;
use super::{RowPlan, SpecKind, Specialization};
use crate::engine::Outcome;
use crate::translation_audit::{
    engine_fingerprint, EngineAuditResult, RecordedCost, WorkerMeasure,
};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The specialization lane's own version: everything under `validation/` that can change a recorded
/// plan or attempt (the planner's sets, the inputs an attempt derives, what is recorded). Bump it
/// with any change to one of those.
///
/// 1: the first lane. 2: a planning or attempt worker's group member is charged only the memory of
/// its own, never its parent's while it still runs in it. 3: a set that leaves two color outputs at
/// one attachment and blend index is kept apart, as Metal refuses it. 4: a post-tessellation
/// function's patch fetches are the compile's own projection of the attributes its specialized
/// entry reads (`DeviceProfile::vertex_entry_inputs`), not one over the numeric classes of the
/// derived descriptor's formats. 5: an attempt worker runs `spirv-val` in its L3 stage, a fresh
/// image of itself, so the attempt's peak is the larger of its translation's and its validation's,
/// not their sum.
pub const SPECIALIZATION_AUDIT_VERSION: u32 = 5;

const PRODUCT_FINGERPRINT: &str = env!("METAL2VULKAN_PRODUCT_FINGERPRINT");

fn field(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

/// The key every plan is recorded under: the product's source fingerprint (its decoder states the
/// constants), [`SPECIALIZATION_AUDIT_VERSION`], and the census tools (`llvm-as` assembles a
/// public row's bitcode).
pub fn planner_key() -> String {
    let mut bytes = Vec::new();
    field(&mut bytes, b"metal2vulkan-specialization-plan-key");
    field(&mut bytes, PRODUCT_FINGERPRINT.as_bytes());
    field(&mut bytes, &SPECIALIZATION_AUDIT_VERSION.to_le_bytes());
    for (tool, identity) in crate::translation_audit::census_tools() {
        field(&mut bytes, tool.as_bytes());
        field(&mut bytes, identity.as_bytes());
    }
    crate::hash::sha256_bytes(&bytes)
}

/// The key a profile's attempts are recorded under: the profile's engine census key (the
/// product, `SEMANTICS_VERSION`, `ENGINE_CENSUS_VERSION`, the profile, the tools), which this
/// lane's attempt shares, and [`SPECIALIZATION_AUDIT_VERSION`].
pub fn fingerprint(profile_stem: &str, profile: &metal2vulkan::device::DeviceProfile) -> String {
    let mut bytes = Vec::new();
    field(&mut bytes, b"metal2vulkan-specialization-census-key");
    field(
        &mut bytes,
        engine_fingerprint(profile_stem, profile).as_bytes(),
    );
    field(&mut bytes, &SPECIALIZATION_AUDIT_VERSION.to_le_bytes());
    crate::hash::sha256_bytes(&bytes)
}

fn open(index: &Path) -> Result<Connection, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS specialization_plan (
               air_sha256 TEXT NOT NULL,
               planner TEXT NOT NULL,
               cases TEXT NOT NULL,
               plan TEXT NOT NULL,
               PRIMARY KEY(air_sha256, planner)
             );
             CREATE TABLE IF NOT EXISTS specialization_audit (
               air_sha256 TEXT NOT NULL,
               spec_id TEXT NOT NULL,
               fingerprint TEXT NOT NULL,
               profile TEXT NOT NULL,
               kinds TEXT NOT NULL,
               outcome TEXT NOT NULL,
               outcome_key TEXT NOT NULL,
               detail TEXT,
               supplied TEXT,
               unsupplied TEXT,
               ray_tracing INTEGER NOT NULL,
               cost TEXT,
               PRIMARY KEY(air_sha256, spec_id, fingerprint)
             );
             CREATE INDEX IF NOT EXISTS specialization_audit_census
               ON specialization_audit(fingerprint, profile);",
        )
        .map_err(|error| format!("create specialization-audit tables: {error}"))?;
    Ok(connection)
}

/// Each indexed row with authored cases, and their case ids, sorted.
pub fn case_ids(index: &Path) -> Result<BTreeMap<String, Vec<String>>, String> {
    let connection = open(index)?;
    let mut statement = connection
        .prepare(
            "SELECT c.air_sha256, c.case_id FROM cases c JOIN sources s USING (air_sha256)
             ORDER BY c.air_sha256, c.case_id",
        )
        .map_err(|error| format!("prepare case-id selection: {error}"))?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| format!("query case ids: {error}"))?;
    let mut out = BTreeMap::<String, Vec<String>>::new();
    for row in rows {
        let (air, case) = row.map_err(|error| format!("read case id: {error}"))?;
        out.entry(air).or_default().push(case);
    }
    Ok(out)
}

/// The rows to plan, in hash order, at most `limit`: every indexed row without a plan under
/// `planner`, and every row whose authored case ids (`cases`, [`case_ids`]) are not the ones its
/// plan was made with.
pub fn rows_to_plan(
    index: &Path,
    planner: &str,
    cases: &BTreeMap<String, Vec<String>>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let connection = open(index)?;
    let mut unplanned = connection
        .prepare(
            "SELECT s.air_sha256 FROM sources s WHERE NOT EXISTS (
               SELECT 1 FROM specialization_plan p
               WHERE p.air_sha256=s.air_sha256 AND p.planner=?1
             ) ORDER BY s.air_sha256",
        )
        .map_err(|error| format!("prepare plan selection: {error}"))?;
    let mut out = unplanned
        .query_map([planner], |row| row.get::<_, String>(0))
        .map_err(|error| format!("query plan selection: {error}"))?
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(|error| format!("read plan selection: {error}"))?;
    let mut planned_with_cases = connection
        .prepare(
            "SELECT p.air_sha256, p.cases FROM specialization_plan p JOIN sources s
               USING (air_sha256) WHERE p.planner=?1 AND p.cases<>'[]'",
        )
        .map_err(|error| format!("prepare planned-case selection: {error}"))?;
    let recorded = planned_with_cases
        .query_map([planner], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| format!("query planned cases: {error}"))?
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map_err(|error| format!("read planned cases: {error}"))?;
    let none = Vec::new();
    for air in recorded.keys().chain(cases.keys()) {
        let now = serde_json::to_string(cases.get(air).unwrap_or(&none))
            .map_err(|error| format!("encode case ids: {error}"))?;
        let then = recorded.get(air).map(String::as_str).unwrap_or("[]");
        if now != then && !out.contains(air) {
            // A row without a plan is already selected; this one's case set changed.
            let has_plan: bool = connection
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM specialization_plan WHERE air_sha256=?1 AND planner=?2)",
                    params![air, planner],
                    |row| row.get(0),
                )
                .map_err(|error| format!("query plan of {air}: {error}"))?;
            if has_plan {
                out.insert(air.clone());
            }
        }
    }
    Ok(out.into_iter().take(limit).collect())
}

/// Record each row's plan under `planner`, with the case ids its authored sets came from. A row
/// whose recorded plan changes (its authored cases changed) loses its recorded attempts: an
/// attempt also depends on the functions the plan links, which its set's id does not name.
pub fn write_plans(
    index: &Path,
    planner: &str,
    plans: &[(String, Vec<String>, RowPlan)],
) -> Result<(), String> {
    let mut connection = open(index)?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin plan update: {error}"))?;
    {
        let mut recorded = transaction
            .prepare("SELECT plan FROM specialization_plan WHERE air_sha256=?1 AND planner=?2")
            .map_err(|error| format!("prepare recorded-plan read: {error}"))?;
        let mut forget = transaction
            .prepare("DELETE FROM specialization_audit WHERE air_sha256=?1")
            .map_err(|error| format!("prepare attempt invalidation: {error}"))?;
        let mut insert = transaction
            .prepare(
                "INSERT INTO specialization_plan (air_sha256, planner, cases, plan)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(air_sha256, planner) DO UPDATE SET
                   cases=excluded.cases, plan=excluded.plan",
            )
            .map_err(|error| format!("prepare plan update: {error}"))?;
        for (air, cases, plan) in plans {
            let cases = serde_json::to_string(cases)
                .map_err(|error| format!("encode case ids of {air}: {error}"))?;
            let plan = serde_json::to_string(plan)
                .map_err(|error| format!("encode plan of {air}: {error}"))?;
            let then = recorded
                .query_row(params![air, planner], |row| row.get::<_, String>(0))
                .optional()
                .map_err(|error| format!("read recorded plan of {air}: {error}"))?;
            if then.is_some_and(|then| then != plan) {
                forget
                    .execute([air])
                    .map_err(|error| format!("forget the attempts of {air}: {error}"))?;
            }
            insert
                .execute(params![air, planner, cases, plan])
                .map_err(|error| format!("record plan of {air}: {error}"))?;
        }
    }
    transaction
        .commit()
        .map_err(|error| format!("commit plan update: {error}"))
}

/// Every current plan, by row. It only reads the index; a planning run creates the table first.
pub fn plans(index: &Path, planner: &str) -> Result<BTreeMap<String, RowPlan>, String> {
    let connection = Connection::open_with_flags(
        index,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| format!("open index {} read-only: {error}", index.display()))?;
    let mut statement = connection
        .prepare(
            "SELECT p.air_sha256, p.plan FROM specialization_plan p JOIN sources s
               USING (air_sha256) WHERE p.planner=?1",
        )
        .map_err(|error| format!("prepare plan read: {error}"))?;
    let rows = statement
        .query_map([planner], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| format!("query plans: {error}"))?;
    let mut out = BTreeMap::new();
    for row in rows {
        let (air, plan) = row.map_err(|error| format!("read plan: {error}"))?;
        let plan: RowPlan = serde_json::from_str(&plan)
            .map_err(|error| format!("decode plan of {air}: {error}"))?;
        out.insert(air, plan);
    }
    Ok(out)
}

/// One set a row still has to be attempted under.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingSet {
    pub id: String,
    pub kinds: BTreeSet<SpecKind>,
    pub constants: BTreeMap<u32, FunctionConstantValue>,
}

/// The set of id `spec_id` a row was attempted under: the `inputs` set, or a set of the row's
/// plan; `None` when neither has that id (the plan changed since).
pub fn recorded_set(plan: Option<&RowPlan>, spec_id: &str) -> Option<PendingSet> {
    let inputs = PendingSet::inputs();
    if inputs.id == spec_id {
        return Some(inputs);
    }
    plan?
        .specializations
        .iter()
        .map(pending_set)
        .find(|set| set.id == spec_id)
}

/// `set` as a set to attempt.
pub fn pending_set(set: &Specialization) -> PendingSet {
    PendingSet {
        id: set.id(),
        kinds: set.kinds.clone(),
        constants: set.constants.clone(),
    }
}

impl PendingSet {
    /// The set with no constants, attempted for a row whose plain compile needed another input.
    pub fn inputs() -> PendingSet {
        PendingSet {
            id: super::id_of(&BTreeMap::new()),
            kinds: BTreeSet::from([SpecKind::Inputs]),
            constants: BTreeMap::new(),
        }
    }

    pub fn kinds_key(&self) -> String {
        self.kinds
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The values as the product takes them; a constant absent from the set is undefined.
    pub fn values(&self) -> FunctionConstantValues {
        super::values_of(&self.constants)
    }
}

/// Each row's plain outcome (label, key) recorded under `plain`, the profile's engine census
/// key: what the report groups the specialized rows by.
pub fn plain_outcomes(
    index: &Path,
    plain: &str,
) -> Result<BTreeMap<String, (String, String)>, String> {
    let connection = open(index)?;
    let exists: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table'
               AND name='new_engine_translation_audit')",
            [],
            |row| row.get(0),
        )
        .map_err(|error| format!("query index schema: {error}"))?;
    if !exists {
        return Ok(BTreeMap::new());
    }
    let mut statement = connection
        .prepare(
            "SELECT a.air_sha256, a.outcome, a.outcome_key FROM new_engine_translation_audit a
               JOIN sources s USING (air_sha256) WHERE a.fingerprint=?1",
        )
        .map_err(|error| format!("prepare plain-outcome read: {error}"))?;
    let rows = statement
        .query_map([plain], |row| {
            Ok((
                row.get::<_, String>(0)?,
                (row.get::<_, String>(1)?, row.get::<_, String>(2)?),
            ))
        })
        .map_err(|error| format!("query plain outcomes: {error}"))?
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map_err(|error| format!("read plain outcomes: {error}"))?;
    Ok(rows)
}

/// The rows of one plain outcome class that the lane attempted: how many, how many translate
/// under at least one set, and under every set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassRows {
    pub rows: usize,
    pub translated_some: usize,
    pub translated_all: usize,
}

/// [`ClassRows`] of `attempts` by each row's plain outcome (`plain`, [`plain_outcomes`]); a row
/// the plain census has no outcome for is grouped under `("unrecorded", "")`.
pub fn rows_by_plain_outcome(
    attempts: &[RecordedAttempt],
    plain: &BTreeMap<String, (String, String)>,
) -> BTreeMap<(String, String), ClassRows> {
    let mut by_row = BTreeMap::<&str, (usize, usize)>::new();
    for attempt in attempts {
        let entry = by_row.entry(&attempt.air_sha256).or_default();
        entry.0 += 1;
        entry.1 += usize::from(attempt.outcome.0 == "translated");
    }
    let unrecorded = ("unrecorded".to_string(), String::new());
    let mut out = BTreeMap::<(String, String), ClassRows>::new();
    for (air, (attempted, translated)) in by_row {
        let class = out
            .entry(plain.get(air).unwrap_or(&unrecorded).clone())
            .or_default();
        class.rows += 1;
        class.translated_some += usize::from(translated != 0);
        class.translated_all += usize::from(translated == attempted);
    }
    out
}

/// The rows whose plain compile, recorded under `plain` (the profile's engine census key),
/// needed a pipeline input other than function constants; or why the plain census cannot be
/// used: it does not cover every indexed row.
pub fn plain_input_rows(index: &Path, plain: &str) -> Result<BTreeSet<String>, String> {
    let connection = open(index)?;
    let exists: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table'
               AND name='new_engine_translation_audit')",
            [],
            |row| row.get(0),
        )
        .map_err(|error| format!("query index schema: {error}"))?;
    let missing: usize = if exists {
        connection
            .query_row(
                "SELECT count(*) FROM sources s WHERE NOT EXISTS (
                   SELECT 1 FROM new_engine_translation_audit a
                   WHERE a.air_sha256=s.air_sha256 AND a.fingerprint=?1)",
                [plain],
                |row| row.get(0),
            )
            .map_err(|error| format!("query plain census coverage: {error}"))?
    } else {
        connection
            .query_row("SELECT count(*) FROM sources", [], |row| row.get(0))
            .map_err(|error| format!("count sources: {error}"))?
    };
    if missing != 0 {
        return Err(format!(
            "the plain engine census at {plain} lacks {missing} indexed rows: run \
             `--audit translation --profile P --current-fingerprint` first"
        ));
    }
    let mut statement = connection
        .prepare(
            "SELECT a.air_sha256 FROM new_engine_translation_audit a JOIN sources s
               USING (air_sha256)
             WHERE a.fingerprint=?1 AND a.outcome='needs_input'
               AND a.outcome_key<>'function_constant'",
        )
        .map_err(|error| format!("prepare input-row selection: {error}"))?;
    let rows = statement
        .query_map([plain], |row| row.get::<_, String>(0))
        .map_err(|error| format!("query input rows: {error}"))?
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(|error| format!("read input rows: {error}"))?;
    Ok(rows)
}

/// The sets each row still has to be attempted under `fingerprint`: every set of its plan, and
/// the [`PendingSet::inputs`] set of each `input_rows` row, less those recorded; at most `limit`
/// attempts, whole rows in hash order.
pub fn pending(
    index: &Path,
    fingerprint: &str,
    plans: &BTreeMap<String, RowPlan>,
    input_rows: &BTreeSet<String>,
    limit: usize,
) -> Result<BTreeMap<String, Vec<PendingSet>>, String> {
    let connection = open(index)?;
    let mut statement = connection
        .prepare("SELECT air_sha256, spec_id FROM specialization_audit WHERE fingerprint=?1")
        .map_err(|error| format!("prepare recorded-set read: {error}"))?;
    let recorded = statement
        .query_map([fingerprint], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| format!("query recorded sets: {error}"))?
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(|error| format!("read recorded sets: {error}"))?;
    let rows = plans
        .keys()
        .chain(input_rows)
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut out = BTreeMap::new();
    let mut attempts = 0usize;
    for air in rows {
        let mut sets = plans
            .get(&air)
            .map(|plan| {
                plan.specializations
                    .iter()
                    .map(pending_set)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if input_rows.contains(&air) {
            sets.push(PendingSet::inputs());
        }
        sets.retain(|set| !recorded.contains(&(air.clone(), set.id.clone())));
        if sets.is_empty() {
            continue;
        }
        if attempts + sets.len() > limit && attempts != 0 {
            break;
        }
        attempts += sets.len();
        out.insert(air, sets);
    }
    Ok(out)
}

/// One recorded attempt: the row, the set, the attempt, and what the parent measured around its
/// worker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecResult {
    pub air_sha256: String,
    pub spec_id: String,
    pub kinds: String,
    pub result: SpecAttempt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerMeasure>,
}

impl SpecResult {
    /// The result as the plain census's cost accounting reads one ([`RecordedCost::of`]).
    pub fn as_plain(&self) -> EngineAuditResult {
        EngineAuditResult {
            air_sha256: self.air_sha256.clone(),
            attempt: self.result.attempt.clone(),
            worker: self.worker,
        }
    }
}

/// Record `results` for `profile` under `fingerprint`. Transient outcomes are never recorded (as
/// in the plain census), so a later run selects them again; returns how many were skipped.
pub fn write_results<'a>(
    index: &Path,
    profile: &str,
    fingerprint: &str,
    results: impl IntoIterator<Item = &'a SpecResult>,
) -> Result<usize, String> {
    let mut connection = open(index)?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin specialization-audit update: {error}"))?;
    let mut skipped = 0;
    {
        let mut insert = transaction
            .prepare(
                "INSERT INTO specialization_audit (air_sha256, spec_id, fingerprint, profile,
                   kinds, outcome, outcome_key, detail, supplied, unsupplied, ray_tracing, cost)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                 ON CONFLICT(air_sha256, spec_id, fingerprint) DO UPDATE SET
                   profile=excluded.profile, kinds=excluded.kinds, outcome=excluded.outcome,
                   outcome_key=excluded.outcome_key, detail=excluded.detail,
                   supplied=excluded.supplied, unsupplied=excluded.unsupplied,
                   ray_tracing=excluded.ray_tracing, cost=excluded.cost",
            )
            .map_err(|error| format!("prepare specialization-audit update: {error}"))?;
        for result in results {
            let outcome = &result.result.attempt.outcome;
            if outcome.is_transient() {
                skipped += 1;
                continue;
            }
            let supplied = (!result.result.supplied.is_empty())
                .then(|| serde_json::to_string(&result.result.supplied))
                .transpose()
                .map_err(|error| format!("encode inputs of {}: {error}", result.air_sha256))?;
            let cost = RecordedCost::of(&result.as_plain())
                .map(|cost| serde_json::to_string(&cost))
                .transpose()
                .map_err(|error| format!("encode cost of {}: {error}", result.air_sha256))?;
            insert
                .execute(params![
                    result.air_sha256,
                    result.spec_id,
                    fingerprint,
                    profile,
                    result.kinds,
                    outcome.label(),
                    outcome.key(),
                    result.result.attempt.detail,
                    supplied,
                    result.result.unsupplied,
                    result.result.ray_tracing,
                    cost,
                ])
                .map_err(|error| {
                    format!(
                        "record specialization attempt of {}: {error}",
                        result.air_sha256
                    )
                })?;
        }
    }
    transaction
        .commit()
        .map_err(|error| format!("commit specialization-audit update: {error}"))?;
    Ok(skipped)
}

/// One recorded attempt as the report reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedAttempt {
    pub air_sha256: String,
    pub spec_id: String,
    pub kinds: BTreeSet<SpecKind>,
    pub outcome: (String, String),
    pub detail: Option<String>,
    pub supplied: Vec<super::attempt::Supplied>,
    pub unsupplied: Option<String>,
    pub ray_tracing: bool,
    pub cost: Option<RecordedCost>,
}

/// Every attempt recorded for `profile` under `fingerprint`, in row and set order.
pub fn recorded(
    index: &Path,
    profile: &str,
    fingerprint: &str,
) -> Result<Vec<RecordedAttempt>, String> {
    let connection = open(index)?;
    let mut statement = connection
        .prepare(
            "SELECT a.air_sha256, a.spec_id, a.kinds, a.outcome, a.outcome_key, a.detail,
                    a.supplied, a.unsupplied, a.ray_tracing, a.cost
             FROM specialization_audit a JOIN sources s USING (air_sha256)
             WHERE a.fingerprint=?1 AND a.profile=?2
             ORDER BY a.air_sha256, a.spec_id",
        )
        .map_err(|error| format!("prepare recorded-attempt read: {error}"))?;
    let rows = statement
        .query_map(params![fingerprint, profile], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                (row.get::<_, String>(3)?, row.get::<_, String>(4)?),
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, bool>(8)?,
                row.get::<_, Option<String>>(9)?,
            ))
        })
        .map_err(|error| format!("query recorded attempts: {error}"))?;
    let mut out = Vec::new();
    for row in rows {
        let (air, spec_id, kinds, outcome, detail, supplied, unsupplied, ray_tracing, cost) =
            row.map_err(|error| format!("read recorded attempt: {error}"))?;
        let kinds = kinds
            .split(',')
            .map(|kind| {
                SpecKind::parse(kind).ok_or_else(|| format!("{air}: unknown kind {kind:?}"))
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let supplied = supplied
            .map(|json| serde_json::from_str(&json))
            .transpose()
            .map_err(|error| format!("decode inputs of {air}: {error}"))?
            .unwrap_or_default();
        let cost = cost
            .map(|json| serde_json::from_str(&json))
            .transpose()
            .map_err(|error| format!("decode cost of {air}: {error}"))?;
        out.push(RecordedAttempt {
            air_sha256: air,
            spec_id,
            kinds,
            outcome,
            detail,
            supplied,
            unsupplied,
            ray_tracing,
            cost,
        });
    }
    Ok(out)
}

/// What the plans say, over every current plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanSummary {
    pub rows: usize,
    pub with_constants: usize,
    pub sets: usize,
    pub over_cap: Vec<(String, usize)>,
    pub refused: BTreeMap<String, usize>,
    pub facts: super::ConstantFacts,
    /// Sets by each kind that produced them (a set of several kinds counts under each).
    pub kinds: BTreeMap<SpecKind, usize>,
    /// The sets kept apart because two bindings share a slot or a color under them, by
    /// [`Shared::kind`](super::Shared::kind) (an argument table, or `color`), and the rows that
    /// have one.
    pub aliased: BTreeMap<String, usize>,
    pub aliased_rows: usize,
}

impl PlanSummary {
    pub fn of(plans: &BTreeMap<String, RowPlan>) -> PlanSummary {
        let mut out = PlanSummary::default();
        for (air, plan) in plans {
            out.rows += 1;
            out.with_constants += usize::from(plan.facts.declared != 0);
            out.sets += plan.specializations.len();
            for set in &plan.specializations {
                for kind in &set.kinds {
                    *out.kinds.entry(*kind).or_default() += 1;
                }
            }
            if plan.over_cap() {
                out.over_cap.push((air.clone(), plan.specializations.len()));
            }
            for set in &plan.aliased {
                *out.aliased
                    .entry(set.shared.kind().to_string())
                    .or_default() += 1;
            }
            out.aliased_rows += usize::from(!plan.aliased.is_empty());
            if let Some(why) = &plan.refused {
                *out.refused.entry(why.clone()).or_default() += 1;
            }
            out.facts.add(&plan.facts);
        }
        out
    }
}

/// Outcome counts of one kind (or of every attempt).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KindCounts {
    pub attempts: usize,
    /// By outcome label (`translated`, `needs_input`, ...).
    pub labels: BTreeMap<String, usize>,
    pub defects: usize,
}

impl KindCounts {
    fn add(&mut self, label: &str, key: &str) {
        self.attempts += 1;
        *self.labels.entry(label.to_string()).or_default() += 1;
        self.defects += usize::from(Outcome::recorded_is_defect(label, key));
    }
}

/// One refusal class over a profile's attempts: the rows and attempts it stops, which kinds of
/// set reached it, and how many of its rows translate or refuse otherwise under another set
/// (value-dependent), or reach it only in an all-zero set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Hole {
    pub label: String,
    pub key: String,
    pub rows: BTreeSet<String>,
    pub attempts: usize,
    pub kinds: BTreeSet<SpecKind>,
    pub value_dependent_rows: usize,
    pub zero_only_rows: usize,
    pub ray_tracing: bool,
    pub defect: bool,
    pub example: Option<(String, String)>,
}

/// The inputs attempts derived, and the ones they could not, by input kind and reason.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InputCounts {
    /// Input kind -> (attempts, rows) that used a derived input of the kind.
    pub supplied: BTreeMap<String, (usize, BTreeSet<String>)>,
    /// (`needs_input` key, reason) -> rows whose attempt still stopped there.
    pub unsupplied: BTreeMap<(String, String), BTreeSet<String>>,
}

/// A profile's report over its recorded attempts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub kinds: BTreeMap<SpecKind, KindCounts>,
    pub total: KindCounts,
    pub rows: usize,
    pub rows_translated_some: usize,
    pub rows_translated_all: usize,
    pub inputs: InputCounts,
    /// Every non-translated class except `needs_input` (the inputs section), ranked by rows.
    pub holes: Vec<Hole>,
}

/// The bounded, one-line form of a refusal detail for an example column.
fn example_detail(detail: Option<&str>) -> String {
    detail
        .unwrap_or("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(200)
        .collect()
}

/// The report of `attempts` (one profile's, [`recorded`]).
pub fn report(attempts: &[RecordedAttempt]) -> Report {
    let mut out = Report::default();
    let mut by_row = BTreeMap::<&str, Vec<&RecordedAttempt>>::new();
    for attempt in attempts {
        by_row.entry(&attempt.air_sha256).or_default().push(attempt);
    }
    let mut holes = BTreeMap::<(String, String), Hole>::new();
    for (air, row) in &by_row {
        out.rows += 1;
        let translated = row.iter().filter(|a| a.outcome.0 == "translated").count();
        out.rows_translated_some += usize::from(translated != 0);
        out.rows_translated_all += usize::from(translated == row.len());
        let outcomes = row.iter().map(|a| &a.outcome).collect::<BTreeSet<_>>();
        let mut row_holes = BTreeMap::<(String, String), Vec<&RecordedAttempt>>::new();
        for attempt in row {
            let (label, key) = &attempt.outcome;
            out.total.add(label, key);
            for kind in &attempt.kinds {
                out.kinds.entry(*kind).or_default().add(label, key);
            }
            for supplied in &attempt.supplied {
                let entry = out
                    .inputs
                    .supplied
                    .entry(supplied.kind().to_string())
                    .or_default();
                entry.0 += 1;
                entry.1.insert(air.to_string());
            }
            if label == "needs_input" {
                out.inputs
                    .unsupplied
                    .entry((key.clone(), attempt.unsupplied.clone().unwrap_or_default()))
                    .or_default()
                    .insert(air.to_string());
            } else if label != "translated" {
                row_holes
                    .entry(attempt.outcome.clone())
                    .or_default()
                    .push(attempt);
            }
        }
        for ((label, key), hit) in row_holes {
            let hole = holes
                .entry((label.clone(), key.clone()))
                .or_insert_with(|| Hole {
                    defect: Outcome::recorded_is_defect(&label, &key),
                    label: label.clone(),
                    key: key.clone(),
                    ..Hole::default()
                });
            hole.rows.insert(air.to_string());
            hole.attempts += hit.len();
            hole.kinds
                .extend(hit.iter().flat_map(|a| a.kinds.iter().copied()));
            hole.ray_tracing |= hit.iter().any(|a| a.ray_tracing);
            hole.value_dependent_rows += usize::from(outcomes.len() > 1);
            hole.zero_only_rows += usize::from(
                outcomes.len() > 1
                    && hit.iter().all(|a| {
                        a.kinds.contains(&SpecKind::Zero)
                            || a.kinds.contains(&SpecKind::ZeroDefined)
                    }),
            );
            if hole.example.is_none() {
                hole.example = Some((air.to_string(), example_detail(hit[0].detail.as_deref())));
            }
        }
    }
    let mut holes = holes.into_values().collect::<Vec<_>>();
    holes.sort_by(|a, b| {
        b.rows
            .len()
            .cmp(&a.rows.len())
            .then_with(|| (&a.label, &a.key).cmp(&(&b.label, &b.key)))
    });
    out.holes = holes;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Attempt, FailureKind};
    use crate::specialization::attempt::Supplied;
    use metal2vulkan::metal::DataType;

    /// An index with `sources` rows, the cases table, and a plain census at `plain` (each row's
    /// outcome as given), as the corpus index holds them.
    fn index(
        dir: &Path,
        rows: &[(&str, &str, &str)],
        cases: &[(&str, &str)],
    ) -> std::path::PathBuf {
        let path = dir.join("index.sqlite");
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT, entry TEXT);
                 CREATE TABLE cases (case_id TEXT PRIMARY KEY, air_sha256 TEXT NOT NULL);
                 CREATE TABLE new_engine_translation_audit (
                   air_sha256 TEXT NOT NULL, fingerprint TEXT NOT NULL, profile TEXT NOT NULL,
                   outcome TEXT NOT NULL, outcome_key TEXT NOT NULL, detail TEXT, cost TEXT,
                   PRIMARY KEY(air_sha256, fingerprint));",
            )
            .unwrap();
        for (air, outcome, key) in rows {
            connection
                .execute("INSERT INTO sources VALUES (?1, 'Kernel', 'main')", [air])
                .unwrap();
            connection
                .execute(
                    "INSERT INTO new_engine_translation_audit VALUES (?1, 'plain', 'p', ?2, ?3,
                       NULL, NULL)",
                    params![air, outcome, key],
                )
                .unwrap();
        }
        for (case, air) in cases {
            connection
                .execute("INSERT INTO cases VALUES (?1, ?2)", params![case, air])
                .unwrap();
        }
        path
    }

    fn row_plan(values: &[u8]) -> RowPlan {
        RowPlan {
            facts: super::super::ConstantFacts {
                declared: 1,
                ..Default::default()
            },
            specializations: values
                .iter()
                .map(|v| Specialization {
                    kinds: BTreeSet::from([SpecKind::Covering]),
                    constants: BTreeMap::from([(
                        0,
                        FunctionConstantValue::new(DataType::UChar, [*v]),
                    )]),
                })
                .collect(),
            ..RowPlan::default()
        }
    }

    fn result(air: &str, set: &PendingSet, outcome: Outcome) -> SpecResult {
        SpecResult {
            air_sha256: air.into(),
            spec_id: set.id.clone(),
            kinds: set.kinds_key(),
            result: SpecAttempt {
                attempt: Attempt::uncosted(outcome, Some("why".into())),
                supplied: Vec::new(),
                unsupplied: None,
                ray_tracing: false,
            },
            worker: None,
        }
    }

    /// The cold pass plans every row and selects each planned set and each input row's set; once
    /// recorded, a warm pass selects nothing; a row whose authored cases change is planned
    /// again, and only it, and loses its attempts when its plan changes. A transient outcome is
    /// not recorded, so it is selected again.
    #[test]
    fn a_warm_pass_selects_nothing_and_a_changed_case_set_replans_one_row() {
        let dir = crate::ScratchDir::new("specialization-audit-test").unwrap();
        let a = "aa".repeat(32);
        let b = "bb".repeat(32);
        let c = "cc".repeat(32);
        let index = index(
            dir.path(),
            &[
                (&a, "needs_input", "function_constant"),
                (&b, "needs_input", "LinkedFunction"),
                (&c, "translated", ""),
            ],
            &[("case-1", &a)],
        );
        let planner = "planner";
        let cases = case_ids(&index).unwrap();
        assert_eq!(cases.get(&a).unwrap(), &vec!["case-1".to_string()]);
        let selected = rows_to_plan(&index, planner, &cases, usize::MAX).unwrap();
        assert_eq!(selected, vec![a.clone(), b.clone(), c.clone()]);
        write_plans(
            &index,
            planner,
            &[
                (a.clone(), vec!["case-1".into()], row_plan(&[0, 1])),
                (b.clone(), Vec::new(), RowPlan::default()),
                (c.clone(), Vec::new(), RowPlan::default()),
            ],
        )
        .unwrap();
        assert!(rows_to_plan(&index, planner, &cases, usize::MAX)
            .unwrap()
            .is_empty());
        let plans = plans(&index, planner).unwrap();
        let input_rows = plain_input_rows(&index, "plain").unwrap();
        assert_eq!(input_rows, BTreeSet::from([b.clone()]));
        let todo = pending(&index, "fp", &plans, &input_rows, usize::MAX).unwrap();
        assert_eq!(todo.len(), 2);
        assert_eq!(todo[&a].len(), 2);
        assert_eq!(todo[&b], vec![PendingSet::inputs()]);
        let results = vec![
            result(&a, &todo[&a][0], Outcome::Translated),
            result(
                &a,
                &todo[&a][1],
                Outcome::Failed {
                    failure: FailureKind::Timeout,
                },
            ),
            result(&b, &todo[&b][0], Outcome::Translated),
        ];
        assert_eq!(write_results(&index, "p", "fp", &results).unwrap(), 1);
        let todo = pending(&index, "fp", &plans, &input_rows, usize::MAX).unwrap();
        assert_eq!(todo.keys().collect::<Vec<_>>(), vec![&a]);
        assert_eq!(
            todo[&a].len(),
            1,
            "only the transient attempt is selected again"
        );
        write_results(
            &index,
            "p",
            "fp",
            &[result(&a, &todo[&a][0], Outcome::Translated)],
        )
        .unwrap();
        assert!(pending(&index, "fp", &plans, &input_rows, usize::MAX)
            .unwrap()
            .is_empty());
        // Another fingerprint selects everything again.
        assert_eq!(
            pending(&index, "other", &plans, &input_rows, usize::MAX)
                .unwrap()
                .len(),
            2
        );
        // A new case of row c re-plans c alone.
        let connection = Connection::open(&index).unwrap();
        connection
            .execute("INSERT INTO cases VALUES ('case-2', ?1)", [&c])
            .unwrap();
        let cases = case_ids(&index).unwrap();
        assert_eq!(
            rows_to_plan(&index, planner, &cases, usize::MAX).unwrap(),
            vec![c.clone()]
        );
        // The same plan again keeps a row's attempts; a changed one (a function its cases now
        // link) forgets them, so the row's sets are selected again.
        write_plans(
            &index,
            planner,
            &[(a.clone(), vec!["case-1".into()], row_plan(&[0, 1]))],
        )
        .unwrap();
        assert!(pending(&index, "fp", &plans, &input_rows, usize::MAX)
            .unwrap()
            .is_empty());
        let linked = RowPlan {
            links: crate::specialization::AuthoredLinks {
                functions: BTreeMap::from([("f".to_string(), "dd".repeat(32))]),
                ..Default::default()
            },
            ..RowPlan::default()
        };
        write_plans(&index, planner, &[(b.clone(), Vec::new(), linked)]).unwrap();
        let plans = super::plans(&index, planner).unwrap();
        let todo = pending(&index, "fp", &plans, &input_rows, usize::MAX).unwrap();
        assert_eq!(todo.keys().collect::<Vec<_>>(), vec![&b]);
        // A plain census that does not cover every row refuses.
        connection
            .execute(
                "DELETE FROM new_engine_translation_audit WHERE air_sha256=?1",
                [&c],
            )
            .unwrap();
        assert!(plain_input_rows(&index, "plain")
            .unwrap_err()
            .contains("lacks 1 indexed rows"));
    }

    /// The plan summary counts rows, sets by kind, refusals, and every plan over the bound.
    #[test]
    fn the_plan_summary_counts_sets_by_kind_and_names_plans_over_the_bound() {
        let values = (0..=16).collect::<Vec<u8>>();
        let plans = BTreeMap::from([
            ("a".to_string(), row_plan(&[0, 1])),
            ("b".to_string(), row_plan(&values)),
            (
                "c".to_string(),
                RowPlan {
                    refused: Some("why".into()),
                    ..RowPlan::default()
                },
            ),
        ]);
        let summary = PlanSummary::of(&plans);
        assert_eq!(summary.rows, 3);
        assert_eq!(summary.with_constants, 2);
        assert_eq!(summary.sets, 19);
        assert_eq!(summary.kinds[&SpecKind::Covering], 19);
        assert_eq!(summary.over_cap, vec![("b".to_string(), 17)]);
        assert_eq!(summary.refused["why"], 1);
        assert_eq!(summary.facts.declared, 2);
    }

    /// The limit takes whole rows, at least one.
    #[test]
    fn the_limit_takes_whole_rows() {
        let dir = crate::ScratchDir::new("specialization-audit-limit").unwrap();
        let a = "aa".repeat(32);
        let b = "bb".repeat(32);
        let index = index(
            dir.path(),
            &[(&a, "translated", ""), (&b, "translated", "")],
            &[],
        );
        let plans = BTreeMap::from([(a.clone(), row_plan(&[0, 1, 2])), (b, row_plan(&[0]))]);
        let todo = pending(&index, "fp", &plans, &BTreeSet::new(), 1).unwrap();
        assert_eq!(todo.keys().collect::<Vec<_>>(), vec![&a]);
        assert_eq!(todo[&a].len(), 3);
    }

    fn recorded_attempt(
        air: &str,
        kinds: &[SpecKind],
        outcome: (&str, &str),
        ray_tracing: bool,
    ) -> RecordedAttempt {
        RecordedAttempt {
            air_sha256: air.into(),
            spec_id: format!("{air}{kinds:?}"),
            kinds: kinds.iter().copied().collect(),
            outcome: (outcome.0.into(), outcome.1.into()),
            detail: Some("detail".into()),
            supplied: Vec::new(),
            unsupplied: None,
            ray_tracing,
            cost: None,
        }
    }

    /// Holes rank by rows; a hole whose row translates under another set is value-dependent, and
    /// zero-only when only all-zero sets reach it; ray-tracing refusals are marked; needs_input
    /// goes to the inputs section with its reason.
    #[test]
    fn the_report_ranks_holes_and_marks_value_dependence() {
        let unsupported = ("unsupported", "not_implemented Linking");
        let attempts = vec![
            recorded_attempt("r1", &[SpecKind::Zero], unsupported, false),
            recorded_attempt("r1", &[SpecKind::One], ("translated", ""), false),
            recorded_attempt("r2", &[SpecKind::Covering], unsupported, false),
            recorded_attempt("r2", &[SpecKind::OneDefined], unsupported, false),
            recorded_attempt(
                "r3",
                &[SpecKind::Covering],
                ("unsupported", "not_implemented RayTracing"),
                true,
            ),
            RecordedAttempt {
                supplied: vec![Supplied::StageInputDescriptor],
                ..recorded_attempt("r4", &[SpecKind::Inputs], ("translated", ""), false)
            },
            RecordedAttempt {
                unsupplied: Some("no format".into()),
                ..recorded_attempt(
                    "r5",
                    &[SpecKind::Inputs],
                    ("needs_input", "StorageTextureFormat"),
                    false,
                )
            },
            recorded_attempt("r6", &[SpecKind::Covering], ("failed", "panic"), false),
        ];
        let report = report(&attempts);
        assert_eq!(report.rows, 6);
        assert_eq!(report.rows_translated_some, 2);
        assert_eq!(report.rows_translated_all, 1);
        assert_eq!(report.total.attempts, 8);
        assert_eq!(report.total.defects, 1);
        assert_eq!(report.kinds[&SpecKind::Covering].attempts, 3);
        let linking = &report.holes[0];
        assert_eq!(linking.key, "not_implemented Linking");
        assert_eq!(linking.rows.len(), 2);
        assert_eq!(linking.attempts, 3);
        assert_eq!(linking.value_dependent_rows, 1);
        assert_eq!(linking.zero_only_rows, 1);
        assert!(report
            .holes
            .iter()
            .any(|h| h.key == "not_implemented RayTracing" && h.ray_tracing));
        assert!(report.holes.iter().any(|h| h.key == "panic" && h.defect));
        assert!(!report.holes.iter().any(|h| h.label == "needs_input"));
        assert_eq!(
            report.inputs.supplied["StageInputDescriptor"].1,
            BTreeSet::from(["r4".to_string()])
        );
        assert!(report
            .inputs
            .unsupplied
            .contains_key(&("StorageTextureFormat".into(), "no format".into())));
    }

    /// Rows group by their plain outcome; a row translates under some or every set; a row the
    /// plain census lacks is `unrecorded`. The plain outcomes are read from the index.
    #[test]
    fn rows_group_by_their_plain_outcome() {
        let dir = crate::ScratchDir::new("specialization-audit-plain").unwrap();
        let a = "aa".repeat(32);
        let b = "bb".repeat(32);
        let index = index(
            dir.path(),
            &[
                (&a, "needs_input", "function_constant"),
                (&b, "translated", ""),
            ],
            &[],
        );
        let plain = plain_outcomes(&index, "plain").unwrap();
        assert_eq!(plain.len(), 2);
        assert!(plain_outcomes(&index, "other").unwrap().is_empty());
        let attempts = vec![
            recorded_attempt(&a, &[SpecKind::Zero], ("translated", ""), false),
            recorded_attempt(&a, &[SpecKind::One], ("unsupported", "x"), false),
            recorded_attempt(&b, &[SpecKind::Zero], ("translated", ""), false),
            recorded_attempt("r9", &[SpecKind::Zero], ("unsupported", "x"), false),
        ];
        let classes = rows_by_plain_outcome(&attempts, &plain);
        assert_eq!(
            classes[&("needs_input".into(), "function_constant".into())],
            ClassRows {
                rows: 1,
                translated_some: 1,
                translated_all: 0
            }
        );
        assert_eq!(
            classes[&("translated".into(), String::new())],
            ClassRows {
                rows: 1,
                translated_some: 1,
                translated_all: 1
            }
        );
        assert_eq!(classes[&("unrecorded".into(), String::new())].rows, 1);
    }
}
