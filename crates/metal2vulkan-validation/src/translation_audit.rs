//! Resumable exact-source translation census stored in the disposable corpus index.
//!
//! Discovery remembers a successful source across product edits while retrying prior failures with
//! the current translator. A final current-fingerprint sweep remains available to prove that later
//! fixes did not regress earlier successes.
//!
//! Rows live in `new_engine_translation_audit`, keyed by a per-profile census key
//! ([`engine_fingerprint`]) and carrying the typed outcome class of
//! [`crate::engine::Outcome`]; their per-profile summary is the shadow census. Each key hashes
//! exactly the inputs of a [`CensusKey`]. The old engine's census (`translation_audit`) is deleted
//! with that engine; an index that holds its table keeps it unread.

use crate::engine::Outcome;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::OnceLock;

/// The product's source fingerprint (`validation/product_fingerprint.rs`): the translator's
/// manifest, build inputs, and exactly the files it can compile or read (`build_support::
/// product_hash`; a test-only module is not one), then the dev-tools crate's sources, whose `llvm-as` assembler gives a public row its bitcode. No file
/// under `validation/` feeds it.
const PRODUCT_FINGERPRINT: &str = env!("METAL2VULKAN_PRODUCT_FINGERPRINT");

/// The engine census lane's own version. It stands for everything under `validation/` that can
/// change a recorded engine row (outcome class and key, detail, cost): how a row's bitcode is
/// prepared ([`crate::engine::RowSource`]: the retained blob, or `llvm-as` of a public row's text),
/// the attempt ([`crate::engine::translate_new`]: request, standalone link, relocation, L3), the
/// outcome mapping ([`crate::engine::Outcome`]), the worker boundary's recorded stops and
/// measurements, and which outcomes are recorded. Bump it with any change to one of those; nothing
/// else under `validation/` feeds a new census key. One exemption: a new optional measurement field
/// that older rows decode as absent (unmeasured) and that no outcome, key, or stop depends on, such
/// as `WorkerMeasure::allocator_peak_bytes`, changes no recorded row's meaning, so it needs no
/// bump; reports count the rows that carry it.
///
/// 1: the first scoped key. 2: a public row is assembled from its text. 3: an allocation failure is
/// transient `allocation_failed`, never recorded. 4: a kernel reading stage inputs is translated
/// under the stage-input descriptor its AIR implies. 5: a worker group member is charged only the
/// memory of its own, never its parent's while it still runs in it. 6: the worker runs `spirv-val`
/// in its L3 stage, a fresh image of itself, so an attempt's peak is the larger of its
/// translation's and its validation's, not their sum. 7: a fragment function refused for writing
/// more colors than the profile draws at once is translated in color groups, one module a group,
/// each checked by L3 (`engine::color_groups`).
pub const ENGINE_CENSUS_VERSION: u32 = 7;

/// The external tools a census row can depend on: `llvm-as` (a public row's bitcode) and
/// `spirv-val` (L3). Each is keyed by its `--version` output.
pub const CENSUS_TOOLS: [&str; 2] = ["llvm-as", "spirv-val"];

/// Every input of one translation census's key, and nothing else (AGENTS.md "Cache validity follows
/// semantics").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CensusKey {
    /// The product crate's source fingerprint.
    pub product: String,
    /// The lane's own version ([`ENGINE_CENSUS_VERSION`]).
    pub lane_version: u32,
    /// The engine's `SEMANTICS_VERSION`.
    pub semantics: u32,
    /// The committed profile's stem and derived fingerprint (description and policy).
    pub profile: (String, [u8; 32]),
    /// Each [`CENSUS_TOOLS`] entry with its identity (its `--version` output, or `unavailable`).
    pub tools: Vec<(String, String)>,
}

impl CensusKey {
    /// The engine's key for committed profile `profile_stem`, derived as `profile`, in this
    /// build, on this host.
    pub fn engine(profile_stem: &str, profile: &metal2vulkan::device::DeviceProfile) -> CensusKey {
        CensusKey {
            product: PRODUCT_FINGERPRINT.to_string(),
            lane_version: ENGINE_CENSUS_VERSION,
            semantics: metal2vulkan::cache::SEMANTICS_VERSION,
            profile: (profile_stem.to_string(), profile.fingerprint()),
            tools: census_tools().to_vec(),
        }
    }

    /// SHA-256 over a domain tag and each input, length-prefixed. The engine's name, `new`, is
    /// hashed after the tag as every key has hashed it since keys named one of two engines, so a
    /// key is the one its census recorded rows under.
    pub fn fingerprint(&self) -> String {
        fn field(bytes: &mut Vec<u8>, value: &[u8]) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value);
        }
        let mut bytes = Vec::new();
        field(&mut bytes, b"metal2vulkan-translation-census-key");
        field(&mut bytes, b"new");
        field(&mut bytes, self.product.as_bytes());
        field(&mut bytes, &self.lane_version.to_le_bytes());
        field(&mut bytes, &self.semantics.to_le_bytes());
        let (stem, fingerprint) = &self.profile;
        field(&mut bytes, stem.as_bytes());
        field(&mut bytes, fingerprint);
        for (tool, identity) in &self.tools {
            field(&mut bytes, tool.as_bytes());
            field(&mut bytes, identity.as_bytes());
        }
        crate::hash::sha256_bytes(&bytes)
    }
}

/// Each [`CENSUS_TOOLS`] entry with its identity, probed once per process.
pub(crate) fn census_tools() -> &'static [(String, String)] {
    static TOOLS: OnceLock<Vec<(String, String)>> = OnceLock::new();
    TOOLS.get_or_init(|| {
        CENSUS_TOOLS
            .iter()
            .map(|tool| {
                let identity = metal2vulkan_devtools::tool_version(tool)
                    .unwrap_or_else(|_| "unavailable".to_string());
                (tool.to_string(), identity)
            })
            .collect()
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode {
    Discovery,
    CurrentFingerprint,
    RetryCurrentFailures,
}

fn open(index: &Path) -> Result<Connection, String> {
    Connection::open(index).map_err(|error| format!("open index {}: {error}", index.display()))
}

/// A read-only connection for analysis that may run concurrently with an index writer: it never
/// creates schema, so readers treat an absent audit table as empty.
fn open_read_only(index: &Path) -> Result<Connection, String> {
    Connection::open_with_flags(
        index,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| format!("open index {} read-only: {error}", index.display()))
}

fn table_exists(connection: &Connection, table: &str) -> Result<bool, String> {
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
            [table],
            |row| row.get::<_, bool>(0),
        )
        .map_err(|error| format!("query index schema for {table}: {error}"))
}

const ENGINE_TABLE: &str = "new_engine_translation_audit";

/// The column names of `table`.
fn columns(connection: &Connection, table: &str) -> Result<Vec<String>, String> {
    connection
        .prepare(&format!("PRAGMA table_info({table})"))
        .and_then(|mut statement| {
            statement
                .query_map([], |row| row.get::<_, String>(1))?
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|error| format!("inspect {table} schema: {error}"))
}

/// Add `column` to `table` when an index written before it lacks it. Rows written before carry
/// NULL, which readers treat as "not measured".
fn add_column(connection: &Connection, table: &str, column: &str, ty: &str) -> Result<(), String> {
    if !columns(connection, table)?
        .iter()
        .any(|name| name == column)
    {
        connection
            .execute(&format!("ALTER TABLE {table} ADD COLUMN {column} {ty}"), [])
            .map_err(|error| format!("add {table}.{column}: {error}"))?;
    }
    Ok(())
}

/// The engine's census key for one committed profile (the shadow-census fingerprint;
/// [`CensusKey::engine`]): the product fingerprint, `SEMANTICS_VERSION`,
/// [`ENGINE_CENSUS_VERSION`], the profile stem and derived fingerprint, and the
/// [`CENSUS_TOOLS`]. A change to one of them selects every row again; any other validation edit
/// does not.
pub fn engine_fingerprint(
    profile_stem: &str,
    profile: &metal2vulkan::device::DeviceProfile,
) -> String {
    CensusKey::engine(profile_stem, profile).fingerprint()
}

/// One engine census row as a worker reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineAuditResult {
    pub air_sha256: String,
    pub attempt: crate::engine::Attempt,
    /// What the parent measured around the worker, when the worker reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerMeasure>,
}

/// What a sweep's parent measures around one killable worker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerMeasure {
    /// From preparing the worker's input through decoding its result.
    pub elapsed_ms: u64,
    /// The worker process group's peak sampled RSS (worker, `spirv-val`, and any descendant).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_resident_bytes: Option<u64>,
    /// The worker's counting-allocator peak, as its report states it: exact where the 10-ms RSS
    /// sampler can miss a short peak. Absent when the worker reported no result, and in rows
    /// recorded before it was measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocator_peak_bytes: Option<u64>,
}

impl WorkerMeasure {
    /// The larger of the sampled RSS and the allocator peak ([`crate::worker::Measured::peak_bytes`]):
    /// the measure the 500-MiB ceiling is checked against.
    pub fn peak_bytes(&self) -> Option<u64> {
        self.peak_resident_bytes.max(self.allocator_peak_bytes)
    }
}

impl From<crate::worker::Measured> for WorkerMeasure {
    fn from(measured: crate::worker::Measured) -> WorkerMeasure {
        WorkerMeasure {
            elapsed_ms: u64::try_from(measured.elapsed.as_millis()).unwrap_or(u64::MAX),
            peak_resident_bytes: measured.peak_resident_bytes,
            allocator_peak_bytes: measured.allocator_peak_bytes,
        }
    }
}

/// The measured budget facts of one recorded engine row, persisted as the row's `cost`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedCost {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile: Option<crate::engine::AttemptCost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerMeasure>,
}

impl RecordedCost {
    /// The cost a result carries, or `None` when it measured nothing.
    pub fn of(result: &EngineAuditResult) -> Option<RecordedCost> {
        let cost = RecordedCost {
            words: result.attempt.words,
            compile: result.attempt.cost.clone(),
            worker: result.worker,
        };
        (cost != RecordedCost::default()).then_some(cost)
    }
}

/// The engine census of one profile at one fingerprint.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EngineCensus {
    pub total_sources: usize,
    pub attempted: usize,
    pub remaining: usize,
    /// `(outcome label, outcome key) -> rows`.
    pub outcomes: BTreeMap<OutcomeLabel, usize>,
}

impl EngineCensus {
    /// Rows whose outcome has class `label`.
    pub fn count(&self, label: &str) -> usize {
        self.outcomes
            .iter()
            .filter(|((l, _), _)| l == label)
            .map(|(_, n)| n)
            .sum()
    }

    /// Recorded rows whose outcome is a defect ([`Outcome::is_defect`]).
    pub fn defects(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|((label, key), _)| Outcome::recorded_is_defect(label, key))
            .map(|(_, n)| n)
            .sum()
    }
}

/// A row's recorded outcome as `(label, key)` ([`crate::engine::Outcome::label`] and `key`).
pub type OutcomeLabel = (String, String);

/// Select up to `limit` indexed sources to translate with the engine for `profile` under
/// `fingerprint`, in hash order after `after`. `Discovery` skips rows already translated for the
/// profile under any fingerprint and rows attempted under this one; `CurrentFingerprint` selects
/// every row not attempted under this one; `RetryCurrentFailures` selects this fingerprint's
/// non-translated rows.
pub fn select_engine_batch(
    index: &Path,
    profile: &str,
    fingerprint: &str,
    mode: SelectionMode,
    after: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let predicate = match mode {
        SelectionMode::Discovery => {
            "NOT EXISTS (
               SELECT 1 FROM new_engine_translation_audit a
               WHERE a.air_sha256=s.air_sha256 AND (
                 a.fingerprint=?1 OR (a.profile=?2 AND a.outcome='translated')
               )
             )"
        }
        SelectionMode::CurrentFingerprint => {
            "NOT EXISTS (
               SELECT 1 FROM new_engine_translation_audit a
               WHERE a.air_sha256=s.air_sha256 AND a.fingerprint=?1
             )"
        }
        SelectionMode::RetryCurrentFailures => {
            "EXISTS (
               SELECT 1 FROM new_engine_translation_audit a
               WHERE a.air_sha256=s.air_sha256 AND a.fingerprint=?1
                 AND a.outcome<>'translated'
             )"
        }
    };
    let connection = open(index)?;
    ensure_engine_table(&connection)?;
    let sql = format!(
        "SELECT s.air_sha256 FROM sources s WHERE {predicate}
           AND (?3 IS NULL OR s.air_sha256 > ?3)
         ORDER BY s.air_sha256
         LIMIT ?4"
    );
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| format!("prepare engine audit selection: {error}"))?;
    let hashes = statement
        .query_map(params![fingerprint, profile, after, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| format!("query engine audit selection: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read engine audit selection: {error}"))?;
    Ok(hashes)
}

/// Persist engine rows for `profile` under `fingerprint`. Transient outcomes (`Retry::Later`:
/// deadline, cancellation, an allocation failure, tooling) are never recorded, so a later sweep
/// selects those rows again; returns how many were skipped.
pub fn write_engine_results(
    index: &Path,
    profile: &str,
    fingerprint: &str,
    results: &[EngineAuditResult],
) -> Result<usize, String> {
    let mut connection = open(index)?;
    ensure_engine_table(&connection)?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin engine audit update: {error}"))?;
    let mut skipped = 0;
    {
        let mut insert = transaction
            .prepare(
                "INSERT INTO new_engine_translation_audit
                   (air_sha256, fingerprint, profile, outcome, outcome_key, detail, cost)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(air_sha256, fingerprint) DO UPDATE SET
                   profile=excluded.profile,
                   outcome=excluded.outcome,
                   outcome_key=excluded.outcome_key,
                   detail=excluded.detail,
                   cost=excluded.cost",
            )
            .map_err(|error| format!("prepare engine audit update: {error}"))?;
        for result in results {
            let outcome = &result.attempt.outcome;
            if outcome.is_transient() {
                skipped += 1;
                continue;
            }
            insert
                .execute(params![
                    result.air_sha256,
                    fingerprint,
                    profile,
                    outcome.label(),
                    outcome.key(),
                    result.attempt.detail,
                    RecordedCost::of(result)
                        .map(|cost| serde_json::to_string(&cost))
                        .transpose()
                        .map_err(|error| format!(
                            "encode cost of {}: {error}",
                            result.air_sha256
                        ))?,
                ])
                .map_err(|error| {
                    format!("record engine audit for {}: {error}", result.air_sha256)
                })?;
        }
    }
    transaction
        .commit()
        .map_err(|error| format!("commit engine audit update: {error}"))?;
    Ok(skipped)
}

/// The census of `profile` at `fingerprint` over every indexed source.
pub fn engine_census(
    index: &Path,
    profile: &str,
    fingerprint: &str,
) -> Result<EngineCensus, String> {
    let connection = open_read_only(index)?;
    let total_sources = connection
        .query_row("SELECT count(*) FROM sources", [], |row| {
            row.get::<_, usize>(0)
        })
        .map_err(|error| format!("query engine census total: {error}"))?;
    if !table_exists(&connection, ENGINE_TABLE)? {
        return Ok(EngineCensus {
            total_sources,
            remaining: total_sources,
            ..EngineCensus::default()
        });
    }
    let mut statement = connection
        .prepare(
            "SELECT a.outcome, a.outcome_key, count(*)
               FROM sources s JOIN new_engine_translation_audit a USING (air_sha256)
              WHERE a.fingerprint=?1 AND a.profile=?2
              GROUP BY a.outcome, a.outcome_key
              ORDER BY a.outcome, a.outcome_key",
        )
        .map_err(|error| format!("prepare engine census: {error}"))?;
    let outcomes = statement
        .query_map(params![fingerprint, profile], |row| {
            Ok((
                (row.get::<_, String>(0)?, row.get::<_, String>(1)?),
                row.get::<_, usize>(2)?,
            ))
        })
        .map_err(|error| format!("query engine census: {error}"))?
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map_err(|error| format!("read engine census: {error}"))?;
    let attempted = outcomes.values().sum::<usize>();
    Ok(EngineCensus {
        total_sources,
        attempted,
        remaining: total_sources.saturating_sub(attempted),
        outcomes,
    })
}

/// The profiles with any recorded engine row, sorted.
pub fn engine_recorded_profiles(index: &Path) -> Result<Vec<String>, String> {
    let connection = open_read_only(index)?;
    if !table_exists(&connection, ENGINE_TABLE)? {
        return Ok(Vec::new());
    }
    let mut statement = connection
        .prepare("SELECT DISTINCT profile FROM new_engine_translation_audit ORDER BY profile")
        .map_err(|error| format!("prepare engine profiles: {error}"))?;
    let profiles = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| format!("query engine profiles: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read engine profiles: {error}"))?;
    Ok(profiles)
}

/// The engine outcomes of `hashes` recorded under `fingerprint`. Rows without one are absent.
pub fn engine_outcomes(
    index: &Path,
    fingerprint: &str,
    hashes: &[String],
) -> Result<HashMap<String, OutcomeLabel>, String> {
    let connection = open_read_only(index)?;
    if !table_exists(&connection, ENGINE_TABLE)? {
        return Ok(HashMap::new());
    }
    let mut statement = connection
        .prepare(
            "SELECT outcome, outcome_key FROM new_engine_translation_audit
              WHERE air_sha256=?1 AND fingerprint=?2",
        )
        .map_err(|error| format!("prepare engine outcomes: {error}"))?;
    let mut outcomes = HashMap::new();
    for hash in hashes {
        let row = statement
            .query_row(params![hash, fingerprint], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .optional()
            .map_err(|error| format!("query engine outcome {hash}: {error}"))?;
        if let Some(row) = row {
            outcomes.insert(hash.clone(), row);
        }
    }
    Ok(outcomes)
}

fn ensure_engine_table(connection: &Connection) -> Result<(), String> {
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS new_engine_translation_audit (
               air_sha256 TEXT NOT NULL,
               fingerprint TEXT NOT NULL,
               profile TEXT NOT NULL,
               outcome TEXT NOT NULL,
               outcome_key TEXT NOT NULL,
               detail TEXT,
               PRIMARY KEY(air_sha256, fingerprint)
             );
             CREATE INDEX IF NOT EXISTS new_engine_translation_audit_census
               ON new_engine_translation_audit(fingerprint, profile, outcome);",
        )
        .map_err(|error| format!("create engine translation-audit cache: {error}"))?;
    add_column(connection, ENGINE_TABLE, "cost", "TEXT")
}

/// One recorded engine row of a profile at a fingerprint with its measured cost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CostRow {
    pub air_sha256: String,
    pub outcome: OutcomeLabel,
    pub cost: RecordedCost,
}

/// Every recorded row of `profile` at `fingerprint` with its cost: read-only, one query, in hash
/// order.
pub fn engine_cost_rows(
    index: &Path,
    profile: &str,
    fingerprint: &str,
) -> Result<Vec<CostRow>, String> {
    let connection = open_read_only(index)?;
    if !table_exists(&connection, ENGINE_TABLE)?
        || !columns(&connection, ENGINE_TABLE)?
            .iter()
            .any(|name| name == "cost")
    {
        return Ok(Vec::new());
    }
    let mut statement = connection
        .prepare(
            "SELECT a.air_sha256, a.outcome, a.outcome_key, a.cost
               FROM new_engine_translation_audit a
              WHERE a.fingerprint=?1 AND a.profile=?2
              ORDER BY a.air_sha256",
        )
        .map_err(|error| format!("prepare engine costs: {error}"))?;
    let rows = statement
        .query_map(params![fingerprint, profile], |row| {
            Ok((
                row.get::<_, String>(0)?,
                (row.get::<_, String>(1)?, row.get::<_, String>(2)?),
                row.get::<_, Option<String>>(3)?,
            ))
        })
        .map_err(|error| format!("query engine costs: {error}"))?;
    let mut out = Vec::new();
    for row in rows {
        let (air_sha256, outcome, cost) =
            row.map_err(|error| format!("read engine cost: {error}"))?;
        let cost = match cost {
            Some(json) => serde_json::from_str(&json)
                .map_err(|error| format!("decode cost of {air_sha256}: {error}"))?,
            None => RecordedCost::default(),
        };
        out.push(CostRow {
            air_sha256,
            outcome,
            cost,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScratchDir;

    fn index() -> (ScratchDir, std::path::PathBuf) {
        let scratch = ScratchDir::new("translation-audit-cache").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY);
                 INSERT INTO sources VALUES ('a'), ('b'), ('c');",
            )
            .unwrap();
        (scratch, index)
    }

    fn new_result(hash: &str, outcome: Outcome) -> EngineAuditResult {
        EngineAuditResult {
            air_sha256: hash.into(),
            attempt: crate::engine::Attempt {
                outcome,
                detail: Some("detail".into()),
                words: None,
                cost: None,
            },
            worker: None,
        }
    }

    #[test]
    fn engine_rows_are_keyed_by_profile_fingerprint_and_skip_transient_outcomes() {
        use crate::engine::{FailureKind, UnsupportedReason};
        let (_scratch, index) = index();
        let skipped = write_engine_results(
            &index,
            "vk12-floor",
            "fp1",
            &[
                new_result("a", Outcome::Translated),
                new_result(
                    "b",
                    Outcome::Unsupported {
                        reason: UnsupportedReason::NotImplemented,
                        feature: "Lowering(Texture)".into(),
                    },
                ),
                new_result(
                    "c",
                    Outcome::Failed {
                        failure: FailureKind::Timeout,
                    },
                ),
            ],
        )
        .unwrap();
        assert_eq!(skipped, 1, "a deadline is transient and never recorded");
        assert_eq!(
            engine_census(&index, "vk12-floor", "fp1")
                .unwrap()
                .defects(),
            0
        );
        let census = engine_census(&index, "vk12-floor", "fp1").unwrap();
        assert_eq!(
            (census.total_sources, census.attempted, census.remaining),
            (3, 2, 1)
        );
        assert_eq!(census.count("translated"), 1);
        assert_eq!(
            census.outcomes[&(
                "unsupported".to_string(),
                "not_implemented Lowering(Texture)".to_string()
            )],
            1
        );
        // The timed-out row is selected again by every mode that selects unattempted rows.
        let select = |mode, after| {
            select_engine_batch(&index, "vk12-floor", "fp1", mode, after, 10).unwrap()
        };
        assert_eq!(select(SelectionMode::CurrentFingerprint, None), ["c"]);
        assert_eq!(select(SelectionMode::Discovery, None), ["c"]);
        assert_eq!(select(SelectionMode::RetryCurrentFailures, None), ["b"]);
        assert!(select(SelectionMode::CurrentFingerprint, Some("c")).is_empty());
        // A new fingerprint re-selects everything for a fresh sweep; discovery remembers the
        // profile's earlier success.
        let fresh =
            |mode| select_engine_batch(&index, "vk12-floor", "fp2", mode, None, 10).unwrap();
        assert_eq!(fresh(SelectionMode::CurrentFingerprint), ["a", "b", "c"]);
        assert_eq!(fresh(SelectionMode::Discovery), ["b", "c"]);
        // Another profile's success is not this profile's.
        assert_eq!(
            select_engine_batch(
                &index,
                "moltenvk-m3",
                "fp3",
                SelectionMode::Discovery,
                None,
                10
            )
            .unwrap(),
            ["a", "b", "c"]
        );
        assert_eq!(engine_recorded_profiles(&index).unwrap(), ["vk12-floor"]);
        let outcomes =
            engine_outcomes(&index, "fp1", &["a".into(), "b".into(), "c".into()]).unwrap();
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes["a"], ("translated".to_string(), String::new()));
    }

    /// Budget breaches count as census defects; structural caps do not.
    #[test]
    fn census_counts_budget_limit_breaches_as_defects() {
        use crate::engine::LimitKind;
        let (_scratch, index) = index();
        let limit = |kind, limit: &str| Outcome::LimitExceeded {
            kind,
            limit: limit.into(),
        };
        write_engine_results(
            &index,
            "vk12-floor",
            "fp1",
            &[
                new_result("a", limit(LimitKind::Budget, "WorkUnits")),
                new_result("b", limit(LimitKind::Budget, "LedgerBytes")),
                new_result("c", limit(LimitKind::Structural, "SpirvWords")),
            ],
        )
        .unwrap();
        let census = engine_census(&index, "vk12-floor", "fp1").unwrap();
        assert_eq!(census.count("limit_exceeded"), 3);
        assert_eq!(census.defects(), 2);
    }

    /// The census and matrix readers may run beside an index writer: on an index without the
    /// audit tables they report empty results and create no schema.
    #[test]
    fn readers_never_create_audit_schema() {
        let (_scratch, index) = index();
        let census = engine_census(&index, "vk12-floor", "fp1").unwrap();
        assert_eq!(
            (census.total_sources, census.attempted, census.remaining),
            (3, 0, 3)
        );
        assert!(engine_recorded_profiles(&index).unwrap().is_empty());
        let hashes = ["a".to_string()];
        assert!(engine_outcomes(&index, "fp1", &hashes).unwrap().is_empty());
        let connection = Connection::open(&index).unwrap();
        assert!(!table_exists(&connection, ENGINE_TABLE).unwrap());
    }

    #[test]
    fn engine_fingerprint_separates_profiles() {
        let floor = crate::engine::load_profile("vk12-floor").unwrap();
        let mvk = crate::engine::load_profile("moltenvk-m3").unwrap();
        let a = engine_fingerprint("vk12-floor", &floor);
        assert_eq!(a, engine_fingerprint("vk12-floor", &floor));
        assert_ne!(a, engine_fingerprint("moltenvk-m3", &mvk));
    }

    /// A census key follows exactly its inputs. A validation-only edit leaves the product
    /// fingerprint, and so every key, unchanged; a product, runtime or dev-tools edit, a
    /// `SEMANTICS_VERSION` or lane-version bump, another profile, or another tool changes it. The
    /// product fingerprint is the build's own function over a synthetic tree. A key hashes the
    /// layout every census key has hashed since keys named one of two engines (the engine's name
    /// `new` after the tag), so no recorded row's key moved when the old engine's census was
    /// deleted.
    #[test]
    fn census_keys_follow_their_inputs_and_no_validation_source() {
        let scratch = ScratchDir::new("census-key-scope").unwrap();
        let root = scratch.path();
        let write = |path: &str, text: &str| {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        };
        write("metal2vulkan/Cargo.toml", "[package]\nname = \"product\"\n");
        write("metal2vulkan/build.rs", "fn main() {}\n");
        write("metal2vulkan/build_support.rs", "// fingerprint inputs\n");
        write("metal2vulkan/src/lib.rs", "pub fn translate() {}\n");
        write(
            "metal2vulkan-devtools/Cargo.toml",
            "[package]\nname = \"devtools\"\n",
        );
        write("metal2vulkan-devtools/src/lib.rs", "pub fn assemble() {}\n");
        write(
            "metal2vulkan-validation/Cargo.toml",
            "[package]\nname = \"validation\"\n",
        );
        write("metal2vulkan-validation/src/lib.rs", "pub fn census() {}\n");
        let product = || crate::product_fingerprint::product_fingerprint(root);
        let key = |product: String| CensusKey {
            product,
            lane_version: ENGINE_CENSUS_VERSION,
            semantics: metal2vulkan::cache::SEMANTICS_VERSION,
            profile: ("vk12-floor".into(), [1; 32]),
            tools: vec![
                ("llvm-as".into(), "LLVM version 23.1.2".into()),
                ("spirv-val".into(), "SPIRV-Tools v2026.3".into()),
            ],
        };
        let before = key(product());
        let mut layout = Vec::new();
        for field in [
            &b"metal2vulkan-translation-census-key"[..],
            b"new",
            before.product.as_bytes(),
            &ENGINE_CENSUS_VERSION.to_le_bytes(),
            &metal2vulkan::cache::SEMANTICS_VERSION.to_le_bytes(),
            b"vk12-floor",
            &[1; 32],
            b"llvm-as",
            b"LLVM version 23.1.2",
            b"spirv-val",
            b"SPIRV-Tools v2026.3",
        ] {
            layout.extend_from_slice(&(field.len() as u64).to_le_bytes());
            layout.extend_from_slice(field);
        }
        assert_eq!(before.fingerprint(), crate::hash::sha256_bytes(&layout));

        write(
            "metal2vulkan-validation/src/lib.rs",
            "pub fn census() { /* edited */ }\n",
        );
        write(
            "metal2vulkan-validation/src/new_lane.rs",
            "pub fn lane() {}\n",
        );
        assert_eq!(
            key(product()).fingerprint(),
            before.fingerprint(),
            "a validation edit is not an input"
        );

        let changed = |edit: fn(&mut CensusKey)| {
            let mut key = before.clone();
            edit(&mut key);
            key.fingerprint()
        };
        let variants = [
            changed(|key| key.semantics += 1),
            changed(|key| key.lane_version += 1),
            changed(|key| key.profile = ("moltenvk-m3".into(), [1; 32])),
            changed(|key| key.profile = ("vk12-floor".into(), [2; 32])),
            changed(|key| key.tools[0].1 = "LLVM version 24.0.0".into()),
            changed(|key| key.tools[1].1 = "unavailable".into()),
        ];
        let mut distinct = std::collections::BTreeSet::from([before.fingerprint()]);
        for variant in &variants {
            assert!(distinct.insert(variant.clone()), "{variant} repeats a key");
        }

        write(
            "metal2vulkan-devtools/src/lib.rs",
            "pub fn assemble() { /* edited */ }\n",
        );
        assert_ne!(
            key(product()).fingerprint(),
            before.fingerprint(),
            "a dev-tools edit is an input"
        );
        write("metal2vulkan-devtools/src/lib.rs", "pub fn assemble() {}\n");
        assert_eq!(key(product()).fingerprint(), before.fingerprint());
        write(
            "metal2vulkan/src/lib.rs",
            "pub fn translate() { /* edited */ }\n",
        );
        assert_ne!(
            key(product()).fingerprint(),
            before.fingerprint(),
            "a product edit is an input"
        );

        // This build's keys take the product fingerprint.
        let floor = crate::engine::load_profile("vk12-floor").unwrap();
        let current = CensusKey::engine("vk12-floor", &floor);
        assert_eq!(current.product, PRODUCT_FINGERPRINT);
        assert_eq!(
            (current.lane_version, current.semantics),
            (
                ENGINE_CENSUS_VERSION,
                metal2vulkan::cache::SEMANTICS_VERSION
            )
        );
        assert_eq!(current.profile, ("vk12-floor".into(), floor.fingerprint()));
        let tools: Vec<_> = current
            .tools
            .iter()
            .map(|(tool, _)| tool.as_str())
            .collect();
        assert_eq!(tools, CENSUS_TOOLS);
        assert_eq!(
            engine_fingerprint("vk12-floor", &floor),
            current.fingerprint()
        );
    }

    /// A row's recorded cost reads back, also from an index that still holds the deleted old
    /// engine's census table, which nothing reads.
    #[test]
    fn recorded_costs_read_back_beside_an_unread_old_census() {
        let (_scratch, index) = index();
        let mut a = new_result("a", Outcome::Translated);
        a.attempt.words = Some(150);
        write_engine_results(&index, "vk12-floor", "fp1", &[a]).unwrap();
        let rows = engine_cost_rows(&index, "vk12-floor", "fp1").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].cost.words, Some(150));
        Connection::open(&index)
            .unwrap()
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS translation_audit (
                   air_sha256 TEXT NOT NULL, translator_fingerprint TEXT NOT NULL,
                   status TEXT NOT NULL);",
            )
            .unwrap();
        assert_eq!(engine_cost_rows(&index, "vk12-floor", "fp1").unwrap(), rows);
    }
}
