use crate::candidate::engine::{engine_fingerprint_of, is_engine_slot, ENGINE_EXECUTOR_ABI};
use crate::metal::ORACLE_ABI;
use crate::observation::{Backend, CandidateDependencies, MetalStatus};
use crate::source::{
    private_store_present, public_source_set, shard_index_from_path, source_shards_dir,
    PublicSources, SourceIndexLocation, SourceRow,
};
use crate::store::CorpusStore;
use memchr::memmem;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueRow {
    pub air_sha256: String,
    pub stage: String,
    pub entry: String,
    pub label: String,
    pub state: QueueState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueState {
    Unplanned,
    Authored,
    MetalQualified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceState {
    Current,
    Failed,
    Missing,
    Stale,
    /// A Metal slot whose row is a current reference gap: the reference device cannot run the case,
    /// so requalifying it there records the same gap.
    ReferenceGap,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexSyncStats {
    pub rebuilt: bool,
    pub source_shards_scanned: usize,
    pub source_bytes_scanned: u64,
    /// Public rows the sync retired and added to make the index's public rows the checkout's
    /// ([`reconcile_checkout_rows`]); a rebuild adds every one.
    pub public_rows_retired: usize,
    pub public_rows_added: usize,
    /// Which committed rows the index holds: all, or only those of public sources.
    pub scope: IndexScope,
}

/// Which of a corpus root's committed rows (cases and their observations) its index
/// holds, decided by whether the root has a private store at all
/// ([`crate::source::private_store_present`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IndexScope {
    /// The root has a private store, so every committed row is indexed; one whose source the
    /// store lacks (a missing source shard) is an error.
    #[default]
    Complete,
    /// The root has no private store (a public checkout): committed rows reference private AIR
    /// by design, and those are expected and unavailable there. The index holds the rows of public
    /// sources and counts the rest.
    PublicOnly(UnavailablePrivateRows),
}

/// The committed rows a public checkout's index leaves out ([`IndexScope::PublicOnly`]): cases
/// whose row is no public source, and the Metal and candidate observations of those cases.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnavailablePrivateRows {
    pub cases: usize,
    pub metal_observations: usize,
    pub candidate_observations: usize,
}

/// One public identity of the index: a public fixture's `air_sha256` and its label.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PublicIdentity {
    pub air_sha256: String,
    pub label: String,
}

/// What reconciling the index's public rows with a checkout's public fixtures changed: the
/// identities it retired (public rows of the index that the checkout's fixtures do not have, such
/// as a re-keyed fixture's old identity), the identities it added, and how many rows were already
/// the checkout's. A warm reconciliation retires and adds nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PublicReconciliation {
    pub kept: usize,
    pub retired: Vec<PublicIdentity>,
    pub added: Vec<PublicIdentity>,
}

impl PublicReconciliation {
    /// Whether the index's public rows were already the checkout's.
    pub fn is_unchanged(&self) -> bool {
        self.retired.is_empty() && self.added.is_empty()
    }

    /// The lines a tool prints: the counts, then one line per retired and per added identity.
    pub fn report_lines(&self) -> Vec<String> {
        std::iter::once(format!(
            "index-public-rows\tkept={}\tretired={}\tadded={}",
            self.kept,
            self.retired.len(),
            self.added.len()
        ))
        .chain(self.retired.iter().map(|identity| {
            format!(
                "index-public-row-retired\t{}\t{}",
                identity.air_sha256, identity.label
            )
        }))
        .chain(self.added.iter().map(|identity| {
            format!(
                "index-public-row-added\t{}\t{}",
                identity.air_sha256, identity.label
            )
        }))
        .collect()
    }
}

/// What making the index's case rows a checkout's changed ([`reconcile_checkout_rows`]): the case
/// shards whose content differed from the content the index's rows were made from (each read once),
/// and the cases retired from and added to the index from them. A warm reconciliation reads no case
/// shard.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaseReconciliation {
    pub shards_read: usize,
    pub retired: usize,
    pub added: usize,
}

impl CaseReconciliation {
    /// Whether the index's case rows were already the checkout's.
    pub fn is_unchanged(&self) -> bool {
        self.retired == 0 && self.added == 0
    }

    /// The line a tool prints.
    pub fn report_line(&self) -> String {
        format!(
            "index-case-rows\tshards_read={}\tretired={}\tadded={}",
            self.shards_read, self.retired, self.added
        )
    }
}

/// What [`reconcile_checkout_rows`] changed: the public rows and the case rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckoutReconciliation {
    pub public: PublicReconciliation,
    pub cases: CaseReconciliation,
}

impl CheckoutReconciliation {
    /// Whether the index's public and case rows were already the checkout's.
    pub fn is_unchanged(&self) -> bool {
        self.public.is_unchanged() && self.cases.is_unchanged()
    }

    /// The public rows' lines, then the case rows' line.
    pub fn report_lines(&self) -> Vec<String> {
        let mut lines = self.public.report_lines();
        lines.push(self.cases.report_line());
        lines
    }
}

impl EvidenceState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Failed => "failed",
            Self::Missing => "missing",
            Self::Stale => "stale",
            Self::ReferenceGap => "reference_gap",
        }
    }
}

/// One authored case's evidence. `moltenvk` and `vulkan` summarize the case's engine slots
/// (`ENV@engine`) in each backend, judged by the one currency rule. The old engine's slots are
/// not evidence: its executor is deleted, and the census omits them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRow {
    pub case_id: String,
    pub name: String,
    pub metal: EvidenceState,
    pub moltenvk: EvidenceState,
    pub vulkan: EvidenceState,
    pub translation_error: Option<String>,
    pub slots: Vec<EvidenceSlot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceSlot {
    pub backend: &'static str,
    pub environment_id: String,
    pub state: EvidenceState,
}

impl QueueState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unplanned => "unplanned",
            Self::Authored => "authored",
            Self::MetalQualified => "metal-qualified",
        }
    }
}

/// The Metal rows that are reference gaps, apart from `metal_observations`, which holds observed
/// outputs. Added without a rebuild: every path that writes or reads it creates it first
/// ([`ensure_reference_gap_table`]). It names no foreign key, so a build that predates it deletes
/// and replaces case rows under it unhindered (and never reads it); the next sync by a build that
/// knows it replaces its rows.
const REFERENCE_GAP_TABLE: &str = "CREATE TABLE IF NOT EXISTS metal_reference_gaps (
       case_id TEXT NOT NULL,
       environment_id TEXT NOT NULL,
       input_sha256 TEXT NOT NULL,
       apple_family INTEGER NOT NULL,
       oracle_abi TEXT NOT NULL,
       PRIMARY KEY(case_id, environment_id)
     );";

/// Create [`REFERENCE_GAP_TABLE`] if the index predates it. Until a sync fills it, the index
/// reads as an older build's did: a gap's case is unqualified.
fn ensure_reference_gap_table(connection: &Connection) -> Result<(), String> {
    connection
        .execute_batch(REFERENCE_GAP_TABLE)
        .map_err(|error| format!("create the reference-gap table: {error}"))
}

pub fn default_index_path(root: &Path) -> PathBuf {
    root.join(".index.sqlite")
}

/// Keep the index at `index`, open on `connection`, in write-ahead-log mode.
///
/// A lane's workers read the index (their linkage lookups) while its parent commits their results
/// every few rows. With a rollback journal each commit locks every reader out until its journal
/// and the database are synced, and on a loaded host the workers' reads waited past their busy
/// timeout and failed. With a write-ahead log a reader reads the last committed state and never
/// waits on a writer, nor a writer on readers. The mode belongs to the file: an index built before
/// it changes once, the first time a tool that selects rows opens it, and an older binary opens a
/// write-ahead-log index as any SQLite does. A copy of the index must go through SQLite (its
/// `.backup`, or [`checkpoint_write_ahead_log`] before copying the file), since committed pages
/// can still be in the log beside it.
pub fn keep_write_ahead_log(connection: &Connection, index: &Path) -> Result<(), String> {
    let mode: String = connection
        .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
        .map_err(|error| {
            format!(
                "switch index {} to write-ahead logging: {error}",
                index.display()
            )
        })?;
    if mode.eq_ignore_ascii_case("wal") {
        Ok(())
    } else {
        Err(format!(
            "index {} kept journal mode {mode}: switching it to write-ahead logging needs a \
             moment when no other process has it open",
            index.display()
        ))
    }
}

/// Move every committed page of `index`'s write-ahead log into the database file and empty the
/// log, so that the file alone is the whole index (a copy of it misses nothing). An index still in
/// rollback-journal mode has no log, and this changes nothing.
pub fn checkpoint_write_ahead_log(index: &Path) -> Result<(), String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    checkpoint(&connection, index)
}

/// [`checkpoint_write_ahead_log`] on an open `connection` to `index`.
fn checkpoint(connection: &Connection, index: &Path) -> Result<(), String> {
    let busy: i64 = connection
        .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))
        .map_err(|error| format!("checkpoint index {}: {error}", index.display()))?;
    if busy == 0 {
        Ok(())
    } else {
        Err(format!(
            "checkpoint index {}: another connection kept part of its write-ahead log",
            index.display()
        ))
    }
}

/// The write-ahead log and its shared-memory index beside the database at `index`.
fn write_ahead_log_files(index: &Path) -> [PathBuf; 2] {
    ["-wal", "-shm"].map(|suffix| {
        let mut name = index.as_os_str().to_owned();
        name.push(suffix);
        PathBuf::from(name)
    })
}

/// Remove the write-ahead log files beside `index`. A log left by a process that stopped before
/// its last checkpoint holds pages of the database it was written for, and SQLite would apply
/// them to whatever database file later takes that name, so a rebuilt index never inherits one.
fn remove_write_ahead_log_files(index: &Path) -> Result<(), String> {
    for path in write_ahead_log_files(index) {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("remove {}: {error}", path.display())),
        }
    }
    Ok(())
}

/// Rebuild the index of corpus root `root` at `destination`, returning which committed rows it
/// holds ([`IndexScope`]).
pub fn rebuild_index(root: &Path, destination: &Path) -> Result<IndexScope, String> {
    rebuild_index_with(root, destination, &public_source_set()?)
}

/// [`rebuild_index`] with `public` as the checkout's public fixtures: the caller's
/// [`public_source_set`], or a synthetic set in a test.
pub fn rebuild_index_with(
    root: &Path,
    destination: &Path,
    public: &PublicSources,
) -> Result<IndexScope, String> {
    rebuild_index_counting(root, destination, public).map(|(_, scope)| scope)
}

/// [`rebuild_index_with`], also returning how many public rows it added: the fixtures whose module
/// no private shard holds ([`PublicRows`]).
fn rebuild_index_counting(
    root: &Path,
    destination: &Path,
    public: &PublicSources,
) -> Result<(usize, IndexScope), String> {
    let temporary = destination.with_extension(format!("sqlite.{}.tmp", std::process::id()));
    if temporary.exists() {
        fs::remove_file(&temporary)
            .map_err(|error| format!("remove {}: {error}", temporary.display()))?;
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    remove_write_ahead_log_files(&temporary)?;
    let mut cleanup = RemoveOnDrop::new(temporary.clone());
    let mut connection = Connection::open(&temporary)
        .map_err(|error| format!("open {}: {error}", temporary.display()))?;
    keep_write_ahead_log(&connection, &temporary)?;
    // The schema. Every `air_sha256` column that names a source row (`sources`, its library
    // memberships, `cases`, `triage_analysis`, and the audit tables keyed the same way)
    // holds the row's key, [`SourceRow::key`]: the module's content hash when the module declares
    // one entry point, so every such row keeps its identity, and [`crate::source::entry_row_key`]
    // of the module and entry for each entry of a public module that declares several. No reader
    // may take the column for a module's hash: one that locates a source or matches a case's module
    // goes through `SourceRow::key`, [`PublicSources::key_of`], or
    // [`crate::source::find_case_source`]. A private row is its module, so its key is the hash that
    // names its shard.
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             PRAGMA synchronous=FULL;
             CREATE TABLE sources (
               air_sha256 TEXT PRIMARY KEY,
               stage TEXT NOT NULL,
               entry TEXT NOT NULL,
               label TEXT NOT NULL,
               shard INTEGER,
               source_offset INTEGER,
               source_length INTEGER
             );
             CREATE TABLE source_library_memberships (
               air_sha256 TEXT NOT NULL REFERENCES sources(air_sha256) ON DELETE CASCADE,
               lib_sha256 TEXT NOT NULL,
               PRIMARY KEY(air_sha256, lib_sha256)
             );
             CREATE TABLE cases (
               case_id TEXT PRIMARY KEY,
               air_sha256 TEXT NOT NULL REFERENCES sources(air_sha256),
               name TEXT NOT NULL,
               input_sha256 TEXT NOT NULL,
               case_shard INTEGER,
               UNIQUE(air_sha256, name)
             );
             CREATE TABLE indexed_case_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL,
               sha256 TEXT NOT NULL
             );
             CREATE TABLE metal_observations (
               case_id TEXT NOT NULL REFERENCES cases(case_id),
               environment_id TEXT NOT NULL,
               input_sha256 TEXT NOT NULL,
               output_sha256 TEXT NOT NULL,
               oracle_abi TEXT NOT NULL,
               PRIMARY KEY(case_id, environment_id)
             );
             CREATE TABLE candidate_observations (
               case_id TEXT NOT NULL REFERENCES cases(case_id),
               backend TEXT NOT NULL,
               environment_id TEXT NOT NULL,
               input_sha256 TEXT NOT NULL,
               golden_output_sha256 TEXT NOT NULL,
               spv_sha256 TEXT NOT NULL,
               translator_fingerprint TEXT NOT NULL,
               executor_abi TEXT NOT NULL,
               status TEXT NOT NULL,
               PRIMARY KEY(case_id, backend, environment_id)
             );
             CREATE TABLE indexed_source_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL
             );
             CREATE TABLE index_metadata (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             INSERT INTO index_metadata (key, value)
               VALUES ('source_shard_tracking', 'complete');
             CREATE TABLE triage_analysis (
               air_sha256 TEXT PRIMARY KEY,
               analyzer_abi TEXT NOT NULL,
               result_json TEXT NOT NULL
             );
             CREATE INDEX cases_air_sha256 ON cases(air_sha256);
             CREATE INDEX cases_case_shard ON cases(case_shard);
             CREATE INDEX metal_case_dependencies ON metal_observations(case_id, input_sha256, oracle_abi);",
        )
        .map_err(|error| format!("create index schema: {error}"))?;
    ensure_reference_gap_table(&connection)?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin index transaction: {error}"))?;

    let mut public_rows_added = 0;
    {
        // The private rows first: a module a private shard holds is that shard's row, and a public
        // fixture of the same module adds nothing.
        let mut private_rows = PrivateRows::new(&transaction)?;
        let source_dir = source_shards_dir(root);
        if let Ok(entries) = fs::read_dir(&source_dir) {
            let mut paths = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.extension()
                        .is_some_and(|extension| extension == "jsonl")
                })
                .collect::<Vec<_>>();
            paths.sort();
            let mut private_hashes = HashSet::new();
            for path in paths {
                let expected_shard = shard_index_from_path(&path)?;
                read_source_index_shard(&path, |source| {
                    let actual_shard = crate::source::shard_index_for_hash(&source.air_sha256)?;
                    if actual_shard != expected_shard {
                        return Err(format!(
                            "source {} belongs in shard {}, not {}",
                            source.air_sha256, actual_shard, expected_shard
                        ));
                    }
                    if !private_hashes.insert(source.air_sha256.clone()) {
                        return Err(format!("duplicate private source {}", source.air_sha256));
                    }
                    private_rows.write(&source, actual_shard)
                })?;
                let stamp = source_shard_stamp(&path)?;
                transaction
                    .execute(
                        "INSERT INTO indexed_source_shards (shard, size, modified_ns) VALUES (?1, ?2, ?3)",
                        params![expected_shard, stamp.size, stamp.modified_ns],
                    )
                    .map_err(|error| format!("index source shard stamp: {error}"))?;
            }
        }
        let mut public_rows = PublicRows::new(&transaction)?;
        for source in &public.rows {
            public_rows_added += usize::from(public_rows.add(source)?);
        }
    }

    let scope = insert_compact_rows(&transaction, root, public)?;
    transaction
        .commit()
        .map_err(|error| format!("commit index: {error}"))?;
    connection
        .execute_batch("PRAGMA optimize;")
        .map_err(|error| format!("optimize index: {error}"))?;
    // Every committed page goes into the file before it takes the index's name, so the file
    // alone is the index; a log some stopped process left beside `destination` goes first.
    checkpoint(&connection, &temporary)?;
    connection
        .close()
        .map_err(|(_, error)| format!("close {}: {error}", temporary.display()))?;
    remove_write_ahead_log_files(&temporary)?;
    remove_write_ahead_log_files(destination)?;
    fs::rename(&temporary, destination).map_err(|error| {
        format!(
            "rename {} to {}: {error}",
            temporary.display(),
            destination.display()
        )
    })?;
    cleanup.disarm();
    Ok((public_rows_added, scope))
}

/// Incrementally synchronize the disposable index.
///
/// Source bodies dominate corpus size, so unchanged source shards are identified by their file
/// stamp and never opened. Authored cases and evidence are compact and are refreshed atomically on
/// every sync.
pub fn sync_index(root: &Path, destination: &Path) -> Result<(), String> {
    sync_index_with_stats(root, destination).map(|_| ())
}

pub fn sync_index_with_stats(root: &Path, destination: &Path) -> Result<IndexSyncStats, String> {
    sync_index_with(root, destination, &public_source_set()?)
}

/// [`sync_index_with_stats`] with `public` as the checkout's public fixtures: the caller's
/// [`public_source_set`], or a synthetic set in a test.
pub fn sync_index_with(
    root: &Path,
    destination: &Path,
    public: &PublicSources,
) -> Result<IndexSyncStats, String> {
    if !destination.is_file() {
        let paths = source_shard_paths(root)?;
        let source_bytes_scanned = paths.iter().try_fold(0u64, |total, path| {
            source_shard_stamp(path).and_then(|stamp| {
                total
                    .checked_add(stamp.size)
                    .ok_or_else(|| "source shard byte count overflow".to_string())
            })
        })?;
        let (public_rows_added, scope) = rebuild_index_counting(root, destination, public)?;
        return Ok(IndexSyncStats {
            rebuilt: true,
            source_shards_scanned: paths.len(),
            source_bytes_scanned,
            public_rows_retired: 0,
            public_rows_added,
            scope,
        });
    }
    let mut stats = IndexSyncStats::default();
    let mut connection = Connection::open(destination)
        .map_err(|error| format!("open index {}: {error}", destination.display()))?;
    keep_write_ahead_log(&connection, destination)?;
    ensure_source_location_columns(&connection)?;
    ensure_candidate_fingerprint_column(&connection)?;
    ensure_case_tracking(&connection)?;
    ensure_reference_gap_table(&connection)?;
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS indexed_source_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS source_library_memberships (
               air_sha256 TEXT NOT NULL REFERENCES sources(air_sha256) ON DELETE CASCADE,
               lib_sha256 TEXT NOT NULL,
               PRIMARY KEY(air_sha256, lib_sha256)
             );
             CREATE TABLE IF NOT EXISTS index_metadata (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS triage_analysis (
               air_sha256 TEXT PRIMARY KEY,
               analyzer_abi TEXT NOT NULL,
               result_json TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS cases_air_sha256 ON cases(air_sha256);
             CREATE INDEX IF NOT EXISTS metal_case_dependencies ON metal_observations(case_id, input_sha256, oracle_abi);",
        )
        .map_err(|error| format!("upgrade index schema: {error}"))?;
    // Do not infer migration completion from a nonempty stamp table. Harvest can publish one
    // changed shard before the first sync of a legacy index, leaving a deliberately partial table.
    // Only this marker, committed after all shard stamps, proves missing stamps mean new data.
    let tracking_complete = connection
        .query_row(
            "SELECT 1 FROM index_metadata WHERE key='source_shard_tracking' AND value='complete'",
            [],
            |_| Ok(()),
        )
        .optional()
        .map_err(|error| format!("inspect source-shard migration marker: {error}"))?
        .is_some();
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin index sync: {error}"))?;

    transaction
        .execute_batch(
            "DELETE FROM candidate_observations;
             DELETE FROM metal_reference_gaps;
             DELETE FROM metal_observations;
             DELETE FROM cases;",
        )
        .map_err(|error| format!("clear compact index rows: {error}"))?;

    let paths = source_shard_paths(root)?;
    let current_shards = paths
        .iter()
        .map(|path| shard_index_from_path(path))
        .collect::<Result<HashSet<_>, _>>()?;
    let tracked_shards = {
        let mut statement = transaction
            .prepare("SELECT shard FROM indexed_source_shards")
            .map_err(|error| format!("prepare tracked-shard query: {error}"))?;
        let shards = statement
            .query_map([], |row| row.get::<_, usize>(0))
            .map_err(|error| format!("query tracked shards: {error}"))?
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|error| format!("read tracked shards: {error}"))?;
        shards
    };
    for removed in tracked_shards.difference(&current_shards) {
        transaction
            .execute("DELETE FROM sources WHERE shard=?1", [removed])
            .map_err(|error| format!("remove source shard {removed}: {error}"))?;
        transaction
            .execute(
                "DELETE FROM indexed_source_shards WHERE shard=?1",
                [removed],
            )
            .map_err(|error| format!("remove source shard stamp {removed}: {error}"))?;
    }

    let mut private_rows = PrivateRows::new(&transaction)?;
    for path in paths {
        let shard = shard_index_from_path(&path)?;
        let stamp = source_shard_stamp(&path)?;
        let tracked = transaction
            .query_row(
                "SELECT size, modified_ns FROM indexed_source_shards WHERE shard=?1",
                [shard],
                |row| {
                    Ok(SourceShardStamp {
                        size: row.get(0)?,
                        modified_ns: row.get(1)?,
                    })
                },
            )
            .optional()
            .map_err(|error| format!("read source shard {shard} stamp: {error}"))?;
        // A legacy index already contains the source projection. Treat it as the migration
        // baseline and install per-shard stamps from metadata alone. Reading every source body to
        // add bookkeeping would defeat the index; missing byte locations are reconciled lazily by
        // scanning only the hash-derived shard on first lookup.
        let bootstrap_unchanged = !tracking_complete;
        // Harvest rewrites only buckets that received a hash, so the complete file stamp is a
        // reliable shard-local invalidation key. This also notices same-size replacements.
        let tracked_content_unchanged = tracked == Some(stamp);
        if tracked_content_unchanged || bootstrap_unchanged {
            transaction
                .execute(
                    "INSERT OR REPLACE INTO indexed_source_shards (shard, size, modified_ns) VALUES (?1, ?2, ?3)",
                    params![shard, stamp.size, stamp.modified_ns],
                )
                .map_err(|error| format!("record source shard {shard} stamp: {error}"))?;
            continue;
        }
        transaction
            .execute("DELETE FROM sources WHERE shard=?1", [shard])
            .map_err(|error| format!("replace source shard {shard}: {error}"))?;
        stats.source_shards_scanned += 1;
        stats.source_bytes_scanned = stats
            .source_bytes_scanned
            .checked_add(stamp.size)
            .ok_or_else(|| "source shard byte count overflow".to_string())?;
        read_source_index_shard(&path, |source| {
            let actual = crate::source::shard_index_for_hash(&source.air_sha256)?;
            if actual != shard {
                return Err(format!(
                    "source {} belongs in shard {}, not {}",
                    source.air_sha256, actual, shard
                ));
            }
            private_rows.write(&source, shard)
        })?;
        transaction
            .execute(
                "INSERT OR REPLACE INTO indexed_source_shards (shard, size, modified_ns) VALUES (?1, ?2, ?3)",
                params![shard, stamp.size, stamp.modified_ns],
            )
            .map_err(|error| format!("record source shard {shard} stamp: {error}"))?;
    }
    drop(private_rows);
    // The checkout owns its public rows: retire the index's public identities it does not have and
    // add the ones it does. After the changed private shards, whose rows replace a public row of
    // the same module, so a fixture whose private row a shard just dropped is a public row again.
    let public_rows = reconcile_public_rows(&transaction, &public.rows)?;
    stats.public_rows_retired = public_rows.retired.len();
    stats.public_rows_added = public_rows.added.len();
    transaction
        .execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('source_shard_tracking', 'complete')",
            [],
        )
        .map_err(|error| format!("record source-shard migration completion: {error}"))?;
    stats.scope = insert_compact_rows(&transaction, root, public)?;
    transaction
        .commit()
        .map_err(|error| format!("commit index sync: {error}"))?;
    Ok(stats)
}

/// Make the index's public rows and case rows the checkout's at `root`: the public rows those of
/// `public`, the checkout's public fixtures ([`public_source_set`]), without opening a private
/// source shard, and the case rows those of the checkout's case shards, reading only the case
/// shards that changed.
///
/// Public fixtures and case shards are tracked files of a checkout, so the checkout, not the
/// index's history, owns which public identities and which authored cases exist. An index shared
/// by checkouts at different commits holds the rows of whichever one synchronized it last: a
/// fixture the other re-keyed is then a row this checkout cannot read (it is no fixture of this
/// checkout and no private shard holds it), and a case the other added is one this checkout's
/// case shards do not hold. Every tool that selects rows from the index therefore reconciles
/// first, and every sync makes both the checkout's.
///
/// Only what differs is touched. An index public row the checkout lacks, or whose projection
/// (stage, entry, label, parent libraries) differs, is retired, and each checkout row the index
/// lacks is added, with no source location (a public row is read from the checkout). A retired
/// identity's cached facts stay keyed by its content hash, so a checkout that has it again reuses
/// them; the rows that are no longer sources are not counted by any census. The cases are
/// reconciled shard by shard: the index records each case shard's file stamp and content digest,
/// opens no shard whose stamp is unchanged, and reads only one whose digest changed, retiring the
/// cases it no longer holds (with their observation rows) and adding the ones it gained.
///
/// Observations are the corpus root's compact rows, which the next sync replaces as a whole, so a
/// retired identity a case's observation still names is left to that sync, which names it. The connection therefore does not enforce foreign keys.
pub fn reconcile_checkout_rows(
    root: &Path,
    index: &Path,
    public: &PublicSources,
) -> Result<CheckoutReconciliation, String> {
    if !index.is_file() {
        return Err(format!(
            "index {} does not exist; build it with corpus-index",
            index.display()
        ));
    }
    let mut connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    keep_write_ahead_log(&connection, index)?;
    ensure_source_location_columns(&connection)?;
    ensure_source_membership_table(&connection)?;
    ensure_case_tracking(&connection)?;
    ensure_reference_gap_table(&connection)?;
    connection
        .execute_batch("PRAGMA foreign_keys=OFF;")
        .map_err(|error| format!("prepare checkout-row reconciliation: {error}"))?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin checkout-row reconciliation: {error}"))?;
    let public_rows = reconcile_public_rows(&transaction, &public.rows)?;
    let cases = reconcile_case_rows(&transaction, root, public)?;
    transaction
        .commit()
        .map_err(|error| format!("commit checkout-row reconciliation: {error}"))?;
    Ok(CheckoutReconciliation {
        public: public_rows,
        cases,
    })
}

/// Give an index made before case-shard tracking its case-shard column and table. Its case rows
/// have no shard and no shard is tracked, so the first reconciliation or sync replaces them all,
/// once.
fn ensure_case_tracking(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(cases)")
        .map_err(|error| format!("inspect case index columns: {error}"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| format!("query case index columns: {error}"))?
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|error| format!("read case index columns: {error}"))?;
    drop(statement);
    if !columns.contains("case_shard") {
        connection
            .execute("ALTER TABLE cases ADD COLUMN case_shard INTEGER", [])
            .map_err(|error| format!("add case_shard index column: {error}"))?;
    }
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS indexed_case_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL,
               sha256 TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS index_metadata (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS cases_case_shard ON cases(case_shard);",
        )
        .map_err(|error| format!("upgrade case-shard tracking: {error}"))
}

/// The marker a complete case-shard tracking table carries: every case row has its shard, and
/// every case shard the rows were made from is tracked with its content's digest.
const CASE_SHARD_TRACKING: &str = "case_shard_tracking";

/// The keys of the rows a case may be indexed under at `root`: every key when the root has a
/// private store, else only the public rows' ([`insert_compact_rows`]).
fn available_case_keys(root: &Path, public: &PublicSources) -> Option<HashSet<String>> {
    (!private_store_present(root)).then(|| {
        public
            .rows
            .iter()
            .map(|row| row.key().into_owned())
            .collect()
    })
}

/// The SHA-256 of a case shard's bytes.
fn case_shard_digest(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| crate::hash::sha256_bytes(&bytes))
        .map_err(|error| format!("read case shard {}: {error}", path.display()))
}

/// Track every case shard of `store` with its file stamp and digest, as the case rows of the
/// whole store are written ([`insert_compact_rows`]), and mark the tracking complete. Each digest
/// is taken before the rows are read, so a shard rewritten in between is read again by the next
/// reconciliation rather than missed.
fn track_every_case_shard(
    transaction: &Transaction<'_>,
    store: &CorpusStore,
) -> Result<(), String> {
    transaction
        .execute("DELETE FROM indexed_case_shards", [])
        .map_err(|error| format!("clear case-shard tracking: {error}"))?;
    for path in store.case_shard_paths()? {
        let shard = shard_index_from_path(&path)?;
        let stamp = source_shard_stamp(&path)?;
        let digest = case_shard_digest(&path)?;
        track_case_shard(transaction, shard, stamp, &digest)?;
    }
    transaction
        .execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?1, 'complete')",
            [CASE_SHARD_TRACKING],
        )
        .map_err(|error| format!("record case-shard tracking: {error}"))?;
    Ok(())
}

fn track_case_shard(
    transaction: &Transaction<'_>,
    shard: usize,
    stamp: SourceShardStamp,
    digest: &str,
) -> Result<(), String> {
    transaction
        .execute(
            "INSERT OR REPLACE INTO indexed_case_shards (shard, size, modified_ns, sha256) \
             VALUES (?1, ?2, ?3, ?4)",
            params![shard, stamp.size, stamp.modified_ns, digest],
        )
        .map_err(|error| format!("record case shard {shard}: {error}"))?;
    Ok(())
}

/// Make the index's case rows those of the case shards at `root`, inside `transaction`. A shard
/// whose file stamp is the tracked one is not opened; one whose stamp differs is hashed, and read
/// only when its digest differs too (a checkout at another path holds the same bytes under another
/// stamp). From a read shard, the index's cases of that shard that it no longer holds are retired
/// and the ones it holds that the index lacks are added; a case another shard holds is not touched.
/// A tracked shard the checkout no longer has retires all its cases. An index whose tracking is not
/// complete (one made before it) has all its case rows replaced, once. The observations of a
/// retired case are removed with it, since only a case's rows say which observations the checkout
/// holds.
fn reconcile_case_rows(
    transaction: &Transaction<'_>,
    root: &Path,
    public: &PublicSources,
) -> Result<CaseReconciliation, String> {
    let store = CorpusStore::new(root);
    let complete = transaction
        .query_row(
            "SELECT 1 FROM index_metadata WHERE key=?1 AND value='complete'",
            [CASE_SHARD_TRACKING],
            |_| Ok(()),
        )
        .optional()
        .map_err(|error| format!("inspect case-shard tracking: {error}"))?
        .is_some();
    let mut out = CaseReconciliation::default();
    if !complete {
        out.retired += transaction
            .execute("DELETE FROM cases", [])
            .map_err(|error| format!("replace untracked case rows: {error}"))?;
        transaction
            .execute("DELETE FROM indexed_case_shards", [])
            .map_err(|error| format!("replace untracked case rows: {error}"))?;
    }
    let tracked = {
        let mut statement = transaction
            .prepare("SELECT shard, size, modified_ns, sha256 FROM indexed_case_shards")
            .map_err(|error| format!("prepare case-shard tracking query: {error}"))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, usize>(0)?,
                    (
                        SourceShardStamp {
                            size: row.get(1)?,
                            modified_ns: row.get(2)?,
                        },
                        row.get::<_, String>(3)?,
                    ),
                ))
            })
            .map_err(|error| format!("query case-shard tracking: {error}"))?
            .collect::<Result<std::collections::HashMap<_, _>, _>>()
            .map_err(|error| format!("read case-shard tracking: {error}"))?;
        rows
    };
    let paths = store.case_shard_paths()?;
    let current = paths
        .iter()
        .map(|path| shard_index_from_path(path))
        .collect::<Result<HashSet<_>, _>>()?;
    for shard in tracked.keys().filter(|shard| !current.contains(shard)) {
        out.retired += transaction
            .execute("DELETE FROM cases WHERE case_shard=?1", [shard])
            .map_err(|error| format!("retire the cases of case shard {shard}: {error}"))?;
        transaction
            .execute("DELETE FROM indexed_case_shards WHERE shard=?1", [shard])
            .map_err(|error| format!("forget case shard {shard}: {error}"))?;
    }
    let available = available_case_keys(root, public);
    for path in paths {
        let shard = shard_index_from_path(&path)?;
        let stamp = source_shard_stamp(&path)?;
        let known = tracked.get(&shard);
        if known.is_some_and(|(tracked_stamp, _)| *tracked_stamp == stamp) {
            continue;
        }
        let digest = case_shard_digest(&path)?;
        if known.is_none_or(|(_, tracked_digest)| *tracked_digest != digest) {
            out.shards_read += 1;
            let (retired, added) =
                replace_case_shard(transaction, &store, shard, public, available.as_ref())?;
            out.retired += retired;
            out.added += added;
        }
        track_case_shard(transaction, shard, stamp, &digest)?;
    }
    if out.retired > 0 {
        transaction
            .execute_batch(
                "DELETE FROM metal_observations WHERE case_id NOT IN (SELECT case_id FROM cases);
                 DELETE FROM metal_reference_gaps WHERE case_id NOT IN (SELECT case_id FROM cases);
                 DELETE FROM candidate_observations WHERE case_id NOT IN (SELECT case_id FROM cases);",
            )
            .map_err(|error| format!("remove the observations of retired cases: {error}"))?;
    }
    transaction
        .execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?1, 'complete')",
            [CASE_SHARD_TRACKING],
        )
        .map_err(|error| format!("record case-shard tracking: {error}"))?;
    Ok(out)
}

/// Make the index's cases of case shard `shard` the ones `store`'s shard holds (those whose row
/// `available` admits): retire the rest, then add the new. Returns (retired, added).
fn replace_case_shard(
    transaction: &Transaction<'_>,
    store: &CorpusStore,
    shard: usize,
    public: &PublicSources,
    available: Option<&HashSet<String>>,
) -> Result<(usize, usize), String> {
    let indexed = {
        let mut statement = transaction
            .prepare("SELECT case_id FROM cases WHERE case_shard=?1")
            .map_err(|error| format!("prepare case shard {shard} query: {error}"))?;
        let ids = statement
            .query_map([shard], |row| row.get::<_, String>(0))
            .map_err(|error| format!("query case shard {shard}: {error}"))?
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|error| format!("read case shard {shard}: {error}"))?;
        ids
    };
    let mut held = Vec::new();
    for case in store.read_case_shard(shard)? {
        let key = public.key_of(&case.air_sha256, &case.entry);
        if available.is_none_or(|keys| keys.contains(&key)) {
            held.push((key, case));
        }
    }
    let held_ids = held
        .iter()
        .map(|(_, case)| case.case_id.as_str())
        .collect::<HashSet<_>>();
    let mut retired = 0;
    for id in indexed.iter().filter(|id| !held_ids.contains(id.as_str())) {
        retired += transaction
            .execute("DELETE FROM cases WHERE case_id=?1", [id])
            .map_err(|error| format!("retire case {id}: {error}"))?;
    }
    let mut added = 0;
    for (key, case) in held
        .iter()
        .filter(|(_, case)| !indexed.contains(&case.case_id))
    {
        transaction
            .execute(
                "INSERT INTO cases (case_id, air_sha256, name, input_sha256, case_shard) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    case.case_id,
                    key,
                    case.name,
                    case.computed_input_sha256()?,
                    shard
                ],
            )
            .map_err(|error| format!("index case {}: {error}", case.case_id))?;
        added += 1;
    }
    Ok((retired, added))
}

/// The public half of [`reconcile_checkout_rows`], inside `transaction`: retire the index's public
/// rows (rows without a shard) that are not exactly a row of `public`, then add each row of
/// `public` the index lacks. A private row is never touched, whatever its label: a row of `public`
/// whose module a private shard holds stays that shard's row, as a rebuild indexes it.
fn reconcile_public_rows(
    transaction: &Transaction<'_>,
    public: &[SourceRow],
) -> Result<PublicReconciliation, String> {
    // The projection a public row is indexed with.
    type Projection = (String, String, String, std::collections::BTreeSet<String>);
    let projection = |source: &SourceRow| -> Projection {
        (
            source.stage.clone(),
            source.entry.clone(),
            source.label.clone(),
            source.lib_sha256s.iter().cloned().collect(),
        )
    };
    let mut indexed = std::collections::BTreeMap::<String, Projection>::new();
    {
        let mut statement = transaction
            .prepare("SELECT air_sha256, stage, entry, label FROM sources WHERE shard IS NULL")
            .map_err(|error| format!("prepare indexed public rows: {error}"))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(|error| format!("query indexed public rows: {error}"))?;
        for row in rows {
            let (hash, stage, entry, label) =
                row.map_err(|error| format!("read indexed public rows: {error}"))?;
            indexed.insert(hash, (stage, entry, label, Default::default()));
        }
        let mut statement = transaction
            .prepare(
                "SELECT m.air_sha256, m.lib_sha256 FROM source_library_memberships m \
                 JOIN sources s USING (air_sha256) WHERE s.shard IS NULL",
            )
            .map_err(|error| format!("prepare indexed public memberships: {error}"))?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| format!("query indexed public memberships: {error}"))?;
        for row in rows {
            let (hash, library) =
                row.map_err(|error| format!("read indexed public memberships: {error}"))?;
            if let Some((_, _, _, libraries)) = indexed.get_mut(&hash) {
                libraries.insert(library);
            }
        }
    }
    let wanted = public
        .iter()
        .map(|source| (source.key(), source))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut reconciliation = PublicReconciliation::default();
    {
        let mut forget_memberships = transaction
            .prepare("DELETE FROM source_library_memberships WHERE air_sha256=?1")
            .map_err(|error| format!("prepare public-row retirement: {error}"))?;
        let mut forget = transaction
            .prepare("DELETE FROM sources WHERE air_sha256=?1")
            .map_err(|error| format!("prepare public-row retirement: {error}"))?;
        for (hash, indexed_projection) in &indexed {
            let current = wanted
                .get(hash.as_str())
                .is_some_and(|source| projection(source) == *indexed_projection);
            if current {
                reconciliation.kept += 1;
                continue;
            }
            forget_memberships
                .execute([hash])
                .map_err(|error| format!("retire public row {hash}: {error}"))?;
            forget
                .execute([hash])
                .map_err(|error| format!("retire public row {hash}: {error}"))?;
            reconciliation.retired.push(PublicIdentity {
                air_sha256: hash.clone(),
                label: indexed_projection.2.clone(),
            });
        }
    }
    let retired = reconciliation
        .retired
        .iter()
        .map(|identity| identity.air_sha256.as_str())
        .collect::<HashSet<_>>();
    let mut public_rows = PublicRows::new(transaction)?;
    for (hash, source) in wanted {
        let hash = hash.as_ref();
        if indexed.contains_key(hash) && !retired.contains(hash) {
            continue;
        }
        if !public_rows.add(source)? {
            // A private shard holds the same module, and its row keeps its location.
            reconciliation.kept += 1;
            continue;
        }
        reconciliation.added.push(PublicIdentity {
            air_sha256: hash.to_string(),
            label: source.label.clone(),
        });
    }
    Ok(reconciliation)
}

/// Publish the compact projection of a source shard that was just atomically replaced.
///
/// The shard writer already serialized every row, so reopening a multi-gigabyte bucket merely to
/// rediscover metadata and byte offsets is wasted work. If this update fails or the process exits
/// between the shard rename and this transaction, the file stamp remains different and the normal
/// synchronizer repairs exactly that shard on its next run.
pub(crate) fn record_source_shard_write(
    root: &Path,
    shard: usize,
    locations: &[SourceIndexLocation],
) -> Result<(), String> {
    let destination = default_index_path(root);
    if !destination.is_file() {
        return Ok(());
    }
    let stamp = source_shard_stamp(&crate::source::source_shard_path(root, shard))?;
    let mut connection = Connection::open(&destination)
        .map_err(|error| format!("open index {}: {error}", destination.display()))?;
    ensure_source_location_columns(&connection)?;
    ensure_source_membership_table(&connection)?;
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS indexed_source_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL
             );",
        )
        .map_err(|error| format!("upgrade source-shard tracking: {error}"))?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin source-shard index update: {error}"))?;
    transaction
        .execute_batch(
            "CREATE TEMP TABLE source_shard_replacement_hashes (
               air_sha256 TEXT PRIMARY KEY
             );",
        )
        .map_err(|error| format!("prepare source-shard replacement set: {error}"))?;
    transaction
        .execute(
            "DELETE FROM source_library_memberships WHERE air_sha256 IN (
               SELECT air_sha256 FROM sources WHERE shard=?1
             )",
            [shard],
        )
        .map_err(|error| format!("clear source memberships for shard {shard}: {error}"))?;
    {
        let mut private_rows = PrivateRows::new(&transaction)?;
        let mut retain = transaction
            .prepare("INSERT INTO source_shard_replacement_hashes VALUES (?1)")
            .map_err(|error| format!("prepare retained source hash insert: {error}"))?;
        for location in locations {
            retain
                .execute([&location.air_sha256])
                .map_err(|error| format!("retain source {}: {error}", location.air_sha256))?;
            private_rows.write(
                &SourceIndexRow {
                    air_sha256: location.air_sha256.clone(),
                    stage: location.stage.clone(),
                    entry: location.entry.clone(),
                    lib_sha256s: location.lib_sha256s.clone(),
                    label: location.label.clone(),
                    source_offset: Some(location.offset),
                    source_length: Some(location.length),
                },
                shard,
            )?;
        }
    }
    transaction
        .execute(
            "DELETE FROM sources
             WHERE shard=?1
               AND NOT EXISTS (
                 SELECT 1 FROM source_shard_replacement_hashes replacement
                 WHERE replacement.air_sha256=sources.air_sha256
               )",
            [shard],
        )
        .map_err(|error| format!("remove stale sources from shard {shard}: {error}"))?;
    transaction
        .execute(
            "INSERT OR REPLACE INTO indexed_source_shards (shard, size, modified_ns)
             VALUES (?1, ?2, ?3)",
            params![shard, stamp.size, stamp.modified_ns],
        )
        .map_err(|error| format!("record source shard {shard} stamp: {error}"))?;
    transaction
        .commit()
        .map_err(|error| format!("commit source-shard index update: {error}"))
}

struct SourceIndexRow {
    air_sha256: String,
    stage: String,
    entry: String,
    lib_sha256s: Vec<String>,
    label: String,
    source_offset: Option<i64>,
    source_length: Option<i64>,
}

impl SourceIndexRow {
    /// The projection of a public fixture row: no shard and no byte location.
    fn public(source: &SourceRow) -> Self {
        Self {
            air_sha256: source.key().into_owned(),
            stage: source.stage.clone(),
            entry: source.entry.clone(),
            lib_sha256s: source.lib_sha256s.clone(),
            label: source.label.clone(),
            source_offset: None,
            source_length: None,
        }
    }

    fn validate_metadata(&self) -> Result<(), String> {
        crate::source::shard_index_for_hash(&self.air_sha256)?;
        if self
            .air_sha256
            .bytes()
            .any(|byte| byte.is_ascii_uppercase())
        {
            return Err(format!("source {} hash must be lowercase", self.air_sha256));
        }
        if crate::case::Stage::from_metadata_label(&self.stage).is_none() {
            return Err(format!(
                "source {} has invalid stage {:?}",
                self.label, self.stage
            ));
        }
        if self.entry.is_empty() {
            return Err(format!("source {} has empty entry", self.label));
        }
        if self.label.trim().is_empty() {
            return Err("source label must not be empty".into());
        }
        if self.lib_sha256s.is_empty() {
            return Err(format!("source {} has no parent library", self.label));
        }
        let mut previous = None;
        for hash in &self.lib_sha256s {
            if hash != "owned-synthetic"
                && (hash.len() != 64
                    || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                    || hash.bytes().any(|byte| byte.is_ascii_uppercase()))
            {
                return Err(format!(
                    "source {} parent library must be lowercase SHA-256",
                    self.label
                ));
            }
            if previous.is_some_and(|value: &String| value >= hash) {
                return Err(format!(
                    "source {} parent libraries must be sorted and unique",
                    self.label
                ));
            }
            previous = Some(hash);
        }
        Ok(())
    }
}

fn read_source_index_shard(
    path: &Path,
    mut consume: impl FnMut(SourceIndexRow) -> Result<(), String>,
) -> Result<(), String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    let mut line_number = 0usize;
    loop {
        line.clear();
        if reader
            .read_until(b'\n', &mut line)
            .map_err(|error| format!("read {}: {error}", path.display()))?
            == 0
        {
            break;
        }
        line_number += 1;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let source_offset = reader
            .stream_position()
            .map_err(|error| format!("locate {}:{line_number}: {error}", path.display()))?
            .checked_sub(line.len() as u64)
            .ok_or_else(|| format!("invalid source offset at {}:{line_number}", path.display()))?;
        let mut row = source_index_row_from_canonical_json(&line)
            .map_err(|error| format!("parse {}:{line_number}: {error}", path.display()))?;
        row.source_offset = Some(source_offset.try_into().map_err(|_| {
            format!(
                "source offset at {}:{line_number} is too large",
                path.display()
            )
        })?);
        row.source_length = Some(line.len().try_into().map_err(|_| {
            format!(
                "source row at {}:{line_number} is too large",
                path.display()
            )
        })?);
        row.validate_metadata()
            .map_err(|error| format!("{}:{line_number}: {error}", path.display()))?;
        consume(row).map_err(|error| format!("{}:{line_number}: {error}", path.display()))?;
    }
    Ok(())
}

/// Extract the small index projection without decoding the potentially very large `air_ll` and
/// `blob_b64` JSON strings. Source shards are written by `write_source_shards` with sorted keys;
/// ingestion performs the full source/hash/metadata validation before that canonical write.
fn source_index_row_from_canonical_json(line: &[u8]) -> Result<SourceIndexRow, String> {
    const METADATA_START: &[u8] = b"\",\"air_sha256\":";
    let start = memmem::find(line, METADATA_START)
        .ok_or_else(|| "canonical source row has no air_sha256 boundary".to_string())?
        + 2;
    let metadata = &line[start..];
    Ok(SourceIndexRow {
        air_sha256: extract_json_string(metadata, b"air_sha256")?,
        stage: extract_json_string(metadata, b"stage")?,
        entry: extract_json_string(metadata, b"entry")?,
        lib_sha256s: extract_json_string_array(metadata, b"lib_sha256s")?,
        label: extract_json_string(metadata, b"label")?,
        source_offset: None,
        source_length: None,
    })
}

fn extract_json_string_array(bytes: &[u8], array_key: &[u8]) -> Result<Vec<String>, String> {
    let mut pattern = Vec::with_capacity(array_key.len() + 4);
    pattern.push(b'"');
    pattern.extend_from_slice(array_key);
    pattern.extend_from_slice(b"\":[");
    let start = memmem::find(bytes, &pattern)
        .map(|offset| offset + pattern.len() - 1)
        .ok_or_else(|| {
            format!(
                "canonical source row requires array {}",
                String::from_utf8_lossy(array_key)
            )
        })?;
    let end = bytes[start..]
        .iter()
        .position(|byte| *byte == b']')
        .map(|offset| start + offset + 1)
        .ok_or_else(|| {
            format!(
                "canonical source row has unterminated {}",
                String::from_utf8_lossy(array_key)
            )
        })?;
    serde_json::from_slice(&bytes[start..end]).map_err(|error| {
        format!(
            "canonical source row has invalid {}: {error}",
            String::from_utf8_lossy(array_key)
        )
    })
}

fn extract_json_string(bytes: &[u8], key: &[u8]) -> Result<String, String> {
    let mut pattern = Vec::with_capacity(key.len() + 4);
    pattern.push(b'"');
    pattern.extend_from_slice(key);
    pattern.extend_from_slice(b"\":\"");
    let value_start = memmem::find(bytes, &pattern).ok_or_else(|| {
        format!(
            "canonical source row has no {}",
            String::from_utf8_lossy(key)
        )
    })? + pattern.len();
    let value = &bytes[value_start..];
    let mut escaped = false;
    let end = value
        .iter()
        .position(|byte| {
            if escaped {
                escaped = false;
                false
            } else if *byte == b'\\' {
                escaped = true;
                false
            } else {
                *byte == b'"'
            }
        })
        .ok_or_else(|| {
            format!(
                "canonical source row has unterminated {}",
                String::from_utf8_lossy(key)
            )
        })?;
    let mut quoted = Vec::with_capacity(end + 2);
    quoted.push(b'"');
    quoted.extend_from_slice(&value[..end]);
    quoted.push(b'"');
    serde_json::from_slice(&quoted).map_err(|error| {
        format!(
            "canonical source row has invalid {}: {error}",
            String::from_utf8_lossy(key)
        )
    })
}

/// The writer of private rows, shared by every path that indexes a private shard's rows: a
/// rebuild, a sync of a changed shard, a shard writer's publication, and a shard's location repair.
///
/// A private row replaces a public row of the same module, with the private row's parent
/// libraries. A module a private shard holds is therefore that shard's row in every checkout,
/// whatever public fixtures the checkout has: a sync and a rebuild index it alike, and reconciling
/// the public rows, which touches only rows without a shard, never removes it.
struct PrivateRows<'connection> {
    forget_memberships: rusqlite::Statement<'connection>,
    upsert: rusqlite::Statement<'connection>,
    insert_membership: rusqlite::Statement<'connection>,
}

impl<'connection> PrivateRows<'connection> {
    fn new(connection: &'connection Connection) -> Result<Self, String> {
        let prepare = |sql: &str| {
            connection
                .prepare(sql)
                .map_err(|error| format!("prepare private source index write: {error}"))
        };
        Ok(Self {
            forget_memberships: prepare(
                "DELETE FROM source_library_memberships WHERE air_sha256=?1",
            )?,
            upsert: prepare(
                "INSERT INTO sources \
                   (air_sha256, stage, entry, label, shard, source_offset, source_length) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) \
                 ON CONFLICT(air_sha256) DO UPDATE SET \
                   stage=excluded.stage, entry=excluded.entry, label=excluded.label, \
                   shard=excluded.shard, source_offset=excluded.source_offset, \
                   source_length=excluded.source_length",
            )?,
            insert_membership: prepare(
                "INSERT OR IGNORE INTO source_library_memberships (air_sha256, lib_sha256) \
                 VALUES (?1, ?2)",
            )?,
        })
    }

    /// Index `source` as the row private shard `shard` holds.
    fn write(&mut self, source: &SourceIndexRow, shard: usize) -> Result<(), String> {
        let hash = &source.air_sha256;
        self.forget_memberships
            .execute([hash])
            .map_err(|error| format!("index source {hash}: {error}"))?;
        self.upsert
            .execute(params![
                hash,
                source.stage,
                source.entry,
                source.label,
                shard,
                source.source_offset,
                source.source_length,
            ])
            .map_err(|error| format!("index source {hash}: {error}"))?;
        for library in &source.lib_sha256s {
            self.insert_membership
                .execute(params![hash, library])
                .map_err(|error| format!("index source {hash} parent library: {error}"))?;
        }
        Ok(())
    }
}

/// The writer of public rows: a public fixture is a row without a shard or a location (it is read
/// from the checkout), added only where no private shard's row holds its module ([`PrivateRows`]).
struct PublicRows<'connection> {
    insert: rusqlite::Statement<'connection>,
    insert_membership: rusqlite::Statement<'connection>,
}

impl<'connection> PublicRows<'connection> {
    fn new(connection: &'connection Connection) -> Result<Self, String> {
        let prepare = |sql: &str| {
            connection
                .prepare(sql)
                .map_err(|error| format!("prepare public source index write: {error}"))
        };
        Ok(Self {
            insert: prepare(
                "INSERT OR IGNORE INTO sources \
                   (air_sha256, stage, entry, label, shard, source_offset, source_length) \
                 VALUES (?1, ?2, ?3, ?4, NULL, NULL, NULL)",
            )?,
            insert_membership: prepare(
                "INSERT OR IGNORE INTO source_library_memberships (air_sha256, lib_sha256) \
                 VALUES (?1, ?2)",
            )?,
        })
    }

    /// Add `source` as a public row; `false` when the index already has a row of its module.
    fn add(&mut self, source: &SourceRow) -> Result<bool, String> {
        let row = SourceIndexRow::public(source);
        let hash = &row.air_sha256;
        let inserted = self
            .insert
            .execute(params![hash, row.stage, row.entry, row.label])
            .map_err(|error| format!("add public row {hash} ({}): {error}", row.label))?;
        if inserted == 0 {
            return Ok(false);
        }
        for library in &row.lib_sha256s {
            self.insert_membership
                .execute(params![hash, library])
                .map_err(|error| format!("add public row {hash} parent library: {error}"))?;
        }
        Ok(true)
    }
}

fn ensure_source_location_columns(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(sources)")
        .map_err(|error| format!("inspect source index columns: {error}"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| format!("query source index columns: {error}"))?
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|error| format!("read source index columns: {error}"))?;
    drop(statement);
    if !columns.contains("source_offset") {
        connection
            .execute("ALTER TABLE sources ADD COLUMN source_offset INTEGER", [])
            .map_err(|error| format!("add source_offset index column: {error}"))?;
    }
    if !columns.contains("source_length") {
        connection
            .execute("ALTER TABLE sources ADD COLUMN source_length INTEGER", [])
            .map_err(|error| format!("add source_length index column: {error}"))?;
    }
    Ok(())
}

fn ensure_source_membership_table(connection: &Connection) -> Result<(), String> {
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS source_library_memberships (
               air_sha256 TEXT NOT NULL REFERENCES sources(air_sha256) ON DELETE CASCADE,
               lib_sha256 TEXT NOT NULL,
               PRIMARY KEY(air_sha256, lib_sha256)
             );",
        )
        .map_err(|error| format!("upgrade source-membership index: {error}"))
}

fn ensure_candidate_fingerprint_column(connection: &Connection) -> Result<(), String> {
    let mut statement = connection
        .prepare("PRAGMA table_info(candidate_observations)")
        .map_err(|error| format!("inspect candidate index columns: {error}"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| format!("query candidate index columns: {error}"))?
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|error| format!("read candidate index columns: {error}"))?;
    drop(statement);
    if !columns.contains("translator_fingerprint") {
        connection
            .execute(
                "ALTER TABLE candidate_observations ADD COLUMN translator_fingerprint TEXT NOT NULL DEFAULT ''",
                [],
            )
            .map_err(|error| format!("add translator_fingerprint index column: {error}"))?;
    }
    Ok(())
}

/// Lazily backfill byte locations for one private source shard.
///
/// Older indexes already know each source's shard and never need a corpus-wide migration. The first
/// lookup into a shard scans that shard alone; subsequent lookups seek directly to the recorded row.
pub(crate) fn index_source_shard_locations(
    root: &Path,
    index_path: &Path,
    shard: usize,
) -> Result<u64, String> {
    if !index_path.is_file() {
        return Ok(0);
    }
    let path = crate::source::source_shard_path(root, shard);
    let stamp = source_shard_stamp(&path)?;
    let mut connection = Connection::open(index_path)
        .map_err(|error| format!("open index {}: {error}", index_path.display()))?;
    ensure_source_location_columns(&connection)?;
    ensure_source_membership_table(&connection)?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin source-location update: {error}"))?;
    transaction
        .execute(
            "UPDATE sources SET source_offset=NULL, source_length=NULL WHERE shard=?1",
            [shard],
        )
        .map_err(|error| format!("clear source locations for shard {shard}: {error}"))?;
    transaction
        .execute(
            "DELETE FROM source_library_memberships WHERE air_sha256 IN (
               SELECT air_sha256 FROM sources WHERE shard=?1
             )",
            [shard],
        )
        .map_err(|error| format!("clear source memberships for shard {shard}: {error}"))?;
    {
        let mut private_rows = PrivateRows::new(&transaction)?;
        read_source_index_shard(&path, |source| private_rows.write(&source, shard))?;
    }
    transaction
        .execute(
            "INSERT OR REPLACE INTO indexed_source_shards (shard, size, modified_ns) \
             VALUES (?1, ?2, ?3)",
            params![shard, stamp.size, stamp.modified_ns],
        )
        .map_err(|error| format!("record source shard {shard} stamp: {error}"))?;
    transaction
        .commit()
        .map_err(|error| format!("commit source-location update: {error}"))?;
    Ok(stamp.size)
}

/// Index the corpus root's compact rows: its cases and observations. A case is indexed under the
/// key of the row it runs ([`PublicSources::key_of`] its module and entry).
///
/// A root with no private store ([`private_store_present`]) is a public checkout, whose committed
/// rows reference private AIR by design: there a case whose row is no public source, and the
/// observations of such a case, are left out and counted ([`IndexScope::PublicOnly`]). That is
/// decided by the store's absence, never by a row's: with a private store every row is indexed, and
/// the `sources` foreign keys refuse a row whose source it lacks.
fn insert_compact_rows(
    transaction: &Transaction<'_>,
    root: &Path,
    public: &PublicSources,
) -> Result<IndexScope, String> {
    let store = CorpusStore::new(root);
    let public_keys = available_case_keys(root, public);
    let available = |key: &str| public_keys.as_ref().is_none_or(|keys| keys.contains(key));
    let mut unavailable = UnavailablePrivateRows::default();
    // The case shards are tracked as they are read, so a later reconciliation
    // reads only the ones that change.
    track_every_case_shard(transaction, &store)?;
    let mut unavailable_cases = HashSet::new();
    for case in store.read_all_cases()? {
        let key = public.key_of(&case.air_sha256, &case.entry);
        if !available(&key) {
            unavailable_cases.insert(case.case_id);
            continue;
        }
        transaction
            .execute(
                "INSERT INTO cases (case_id, air_sha256, name, input_sha256, case_shard) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    case.case_id,
                    key,
                    case.name,
                    case.computed_input_sha256()?,
                    crate::source::shard_index_for_hash(&case.air_sha256)?
                ],
            )
            .map_err(|error| format!("index case: {error}"))?;
    }
    unavailable.cases = unavailable_cases.len();
    for observation in store.read_metal()? {
        if unavailable_cases.contains(&observation.case_id) {
            unavailable.metal_observations += 1;
            continue;
        }
        // A reference gap observed no output: it has its own table, so its case is never
        // Metal-qualified, and is not work to qualify either.
        if let MetalStatus::ReferenceGap(gap) = &observation.status {
            transaction
                .execute(
                    "INSERT INTO metal_reference_gaps (case_id, environment_id, input_sha256, apple_family, oracle_abi) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![observation.case_id, observation.environment_id, observation.input_sha256, gap.apple_family, observation.oracle_abi],
                )
                .map_err(|error| format!("index Metal reference gap: {error}"))?;
            continue;
        }
        transaction
            .execute(
                "INSERT INTO metal_observations (case_id, environment_id, input_sha256, output_sha256, oracle_abi) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![observation.case_id, observation.environment_id, observation.input_sha256, observation.metal_output_sha256, observation.oracle_abi],
            )
            .map_err(|error| format!("index Metal observation: {error}"))?;
    }
    for backend in [Backend::Moltenvk, Backend::Vulkan] {
        for observation in store.read_candidates(backend)? {
            if unavailable_cases.contains(&observation.case_id) {
                unavailable.candidate_observations += 1;
                continue;
            }
            transaction
                .execute(
                    "INSERT INTO candidate_observations (case_id, backend, environment_id, input_sha256, golden_output_sha256, spv_sha256, translator_fingerprint, executor_abi, status) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![observation.case_id, backend.directory(), observation.environment_id, observation.input_sha256, observation.golden_output_sha256, observation.spv_sha256.as_deref().unwrap_or_default(), observation.translator_fingerprint, observation.executor_abi, format!("{:?}", observation.status).to_ascii_lowercase()],
                )
                .map_err(|error| format!("index {} observation: {error}", backend.directory()))?;
        }
    }
    Ok(match public_keys {
        None => IndexScope::Complete,
        Some(_) => IndexScope::PublicOnly(unavailable),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SourceShardStamp {
    size: u64,
    modified_ns: i64,
}

fn source_shard_stamp(path: &Path) -> Result<SourceShardStamp, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("read metadata for {}: {error}", path.display()))?;
    Ok(SourceShardStamp {
        size: metadata.len(),
        modified_ns: metadata
            .modified()
            .map_err(|error| format!("read modification time for {}: {error}", path.display()))?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| format!("invalid modification time for {}: {error}", path.display()))?
            .as_nanos()
            .try_into()
            .map_err(|_| format!("modification time for {} is out of range", path.display()))?,
    })
}

fn source_shard_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let directory = source_shards_dir(root);
    let Ok(entries) = fs::read_dir(&directory) else {
        return Ok(Vec::new());
    };
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("shard_") && name.ends_with(".jsonl"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

struct RemoveOnDrop {
    path: PathBuf,
    armed: bool,
}

impl RemoveOnDrop {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub fn check_index(root: &Path, index: &Path) -> Result<(), String> {
    if !index.is_file() {
        return Err(format!("index {} does not exist", index.display()));
    }
    let rebuilt = index.with_extension(format!("sqlite.{}.check", std::process::id()));
    let result = (|| {
        checkpoint_write_ahead_log(index)?;
        fs::copy(index, &rebuilt).map_err(|error| {
            format!(
                "copy index {} to {}: {error}",
                index.display(),
                rebuilt.display()
            )
        })?;
        sync_index(root, &rebuilt)?;
        let current = logical_snapshot(index)?;
        let expected = logical_snapshot(&rebuilt)?;
        if current != expected {
            return Err(format!(
                "index {} differs from canonical corpus shards; rebuild it",
                index.display()
            ));
        }
        Ok(())
    })();
    if rebuilt.exists() {
        fs::remove_file(&rebuilt)
            .map_err(|error| format!("remove {}: {error}", rebuilt.display()))?;
    }
    remove_write_ahead_log_files(&rebuilt)?;
    result
}

fn logical_snapshot(index: &Path) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    connection
        .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
        .map_err(|error| format!("check index integrity: {error}"))
        .and_then(|result| {
            if result == "ok" {
                Ok(())
            } else {
                Err(format!("index integrity check failed: {result}"))
            }
        })?;
    let queries = [
        "SELECT 'source|' || quote(air_sha256) || '|' || quote(stage) || '|' || quote(entry) || '|' || quote(label) || '|' || quote(shard) || '|' || quote(source_offset) || '|' || quote(source_length) FROM sources ORDER BY air_sha256",
        "SELECT 'case|' || quote(case_id) || '|' || quote(air_sha256) || '|' || quote(name) || '|' || quote(input_sha256) FROM cases ORDER BY case_id",
        "SELECT 'metal|' || quote(case_id) || '|' || quote(environment_id) || '|' || quote(input_sha256) || '|' || quote(output_sha256) || '|' || quote(oracle_abi) FROM metal_observations ORDER BY case_id, environment_id",
        "SELECT 'metal-gap|' || quote(case_id) || '|' || quote(environment_id) || '|' || quote(input_sha256) || '|' || quote(apple_family) || '|' || quote(oracle_abi) FROM metal_reference_gaps ORDER BY case_id, environment_id",
        "SELECT 'candidate|' || quote(case_id) || '|' || quote(backend) || '|' || quote(environment_id) || '|' || quote(input_sha256) || '|' || quote(golden_output_sha256) || '|' || quote(spv_sha256) || '|' || quote(translator_fingerprint) || '|' || quote(executor_abi) || '|' || quote(status) FROM candidate_observations ORDER BY case_id, backend, environment_id",
        "SELECT 'source-shard|' || quote(shard) || '|' || quote(size) || '|' || quote(modified_ns) FROM indexed_source_shards ORDER BY shard",
    ];
    let mut snapshot = Vec::new();
    for query in queries {
        let mut statement = connection
            .prepare(query)
            .map_err(|error| format!("prepare index check: {error}"))?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| format!("query index check: {error}"))?;
        snapshot.extend(
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("read index check: {error}"))?,
        );
    }
    Ok(snapshot)
}

pub fn select_queue(
    index: &Path,
    state: QueueState,
    limit: usize,
) -> Result<Vec<QueueRow>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_reference_gap_table(&connection)?;
    let matching_metal = format!(
        "m.case_id=c.case_id AND m.input_sha256=c.input_sha256 AND m.oracle_abi='{ORACLE_ABI}'"
    );
    let condition = match state {
        QueueState::Unplanned => {
            "NOT EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256)".to_string()
        }
        // A case with a current reference gap is not work to qualify.
        QueueState::Authored => format!(
            "EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256) AND EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256 AND NOT EXISTS (SELECT 1 FROM metal_observations m WHERE {matching_metal}) AND NOT EXISTS (SELECT 1 FROM metal_reference_gaps m WHERE {matching_metal}))"
        ),
        QueueState::MetalQualified => format!(
            "EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256) AND NOT EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256 AND NOT EXISTS (SELECT 1 FROM metal_observations m WHERE {matching_metal}))"
        ),
    };
    let sql = format!(
        "SELECT s.air_sha256, s.stage, s.entry, s.label FROM sources s WHERE {condition} ORDER BY s.air_sha256 LIMIT ?1"
    );
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| format!("prepare queue query: {error}"))?;
    let rows = statement
        .query_map([limit as i64], |row| {
            Ok(QueueRow {
                air_sha256: row.get(0)?,
                stage: row.get(1)?,
                entry: row.get(2)?,
                label: row.get(3)?,
                state,
            })
        })
        .map_err(|error| format!("query queue: {error}"))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read queue: {error}"))
}

pub fn status_counts(index: &Path) -> Result<Vec<(String, u64)>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_reference_gap_table(&connection)?;
    let matching_metal = format!(
        "m.case_id=c.case_id AND m.input_sha256=c.input_sha256 AND m.oracle_abi='{ORACLE_ABI}'"
    );
    // A source whose every case is Metal-qualified or a current reference gap, with at least one
    // gap, is counted apart: no qualification run on the reference device completes it.
    let sql = format!(
            "SELECT state, COUNT(*) FROM (
               SELECT CASE
                 WHEN NOT EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256) THEN 'unplanned'
                 WHEN EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256 AND NOT EXISTS (SELECT 1 FROM metal_observations m WHERE {matching_metal}) AND NOT EXISTS (SELECT 1 FROM metal_reference_gaps m WHERE {matching_metal})) THEN 'authored'
                 WHEN EXISTS (SELECT 1 FROM cases c WHERE c.air_sha256=s.air_sha256 AND NOT EXISTS (SELECT 1 FROM metal_observations m WHERE {matching_metal})) THEN 'reference-gap'
                 ELSE 'metal-qualified'
               END AS state
               FROM sources s
             ) GROUP BY state ORDER BY state"
        );
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| format!("prepare status query: {error}"))?;
    let result = statement
        .query_map([], |row| Ok((row.get(0)?, row.get::<_, i64>(1)? as u64)))
        .map_err(|error| format!("query status: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read status: {error}"));
    result
}

/// The evidence census of the store at `root`: each case's Metal and candidate slots judged by
/// the one currency rule (`candidate_slot_state`), venue records held to `venues`, and each case's
/// Metal dependency the one reference row `references` judges it by (`c3::reference`).
pub fn evidence_status(
    root: &Path,
    venues: &crate::c3::VenueKeys,
    references: &crate::c3::reference::References,
) -> Result<Vec<EvidenceRow>, String> {
    let store = CorpusStore::new(root);
    let mut cases = store.read_all_cases()?;
    cases.sort_by(|left, right| {
        left.air_sha256
            .cmp(&right.air_sha256)
            .then_with(|| left.name.cmp(&right.name))
    });
    let metals = store.read_metal()?;
    // The engine slots only: the old engine's rows are no evidence (cutover X2b).
    let engine_rows = |backend| {
        store.read_candidates(backend).map(|rows| {
            rows.into_iter()
                .filter(|row| is_engine_slot(&row.environment_id))
                .collect::<Vec<_>>()
        })
    };
    let moltenvk = engine_rows(Backend::Moltenvk)?;
    let vulkan = engine_rows(Backend::Vulkan)?;
    let (judge, _) = crate::c3::reference::Judge::new(references, &cases, &metals);
    let mut rows = Vec::new();
    for case in cases {
        let case_metals = metals
            .iter()
            .filter(|observation| observation.case_id == case.case_id)
            .collect::<Vec<_>>();
        let current = case_metals
            .iter()
            .copied()
            .filter(|observation| observation.dependency_matches(&case, ORACLE_ABI))
            .collect::<Vec<_>>();
        // The case's one reference row: every candidate's Metal dependency.
        let current_metals = judge.judging(&current).into_iter().collect::<Vec<_>>();
        let metal_state = |observation: &crate::observation::MetalObservation| {
            if observation.dependency_matches(&case, ORACLE_ABI) {
                EvidenceState::Current
            } else if observation.reference_gap_for(&case, ORACLE_ABI).is_some() {
                EvidenceState::ReferenceGap
            } else {
                EvidenceState::Stale
            }
        };
        // Missing without a row; stale while any row is; else current where a reference row
        // judges the case, a reference gap where the primary recorded one, and stale otherwise: no
        // row C3 counts.
        let states = case_metals
            .iter()
            .map(|observation| metal_state(observation))
            .collect::<Vec<_>>();
        let metal = if states.is_empty() {
            EvidenceState::Missing
        } else if states.contains(&EvidenceState::Stale) {
            EvidenceState::Stale
        } else if !current_metals.is_empty() {
            EvidenceState::Current
        } else if judge
            .reference_gap(&case, case_metals.iter().copied())
            .is_some()
        {
            EvidenceState::ReferenceGap
        } else {
            EvidenceState::Stale
        };
        let state = |backend, observations| {
            candidate_state(&case, backend, &current_metals, observations, venues)
        };
        let moltenvk_state = state(Backend::Moltenvk, &moltenvk);
        let vulkan_state = state(Backend::Vulkan, &vulkan);
        let mut slots = case_metals
            .iter()
            .map(|observation| EvidenceSlot {
                backend: "metal",
                environment_id: observation.environment_id.clone(),
                state: metal_state(observation),
            })
            .collect::<Vec<_>>();
        for (backend, observations) in [
            (Backend::Moltenvk, moltenvk.as_slice()),
            (Backend::Vulkan, vulkan.as_slice()),
        ] {
            slots.extend(
                observations
                    .iter()
                    .filter(|observation| observation.case_id == case.case_id)
                    .map(|observation| EvidenceSlot {
                        backend: backend.directory(),
                        environment_id: observation.environment_id.clone(),
                        state: candidate_slot_state(
                            &case,
                            backend,
                            &current_metals,
                            observation,
                            venues,
                        ),
                    }),
            );
        }
        slots.sort_by(|left, right| {
            left.backend
                .cmp(right.backend)
                .then_with(|| left.environment_id.cmp(&right.environment_id))
        });
        rows.push(EvidenceRow {
            case_id: case.case_id,
            name: case.name,
            metal,
            moltenvk: moltenvk_state,
            vulkan: vulkan_state,
            translation_error: None,
            slots,
        });
    }
    Ok(rows)
}

/// The summary of `case`'s `backend` slots among `candidates`.
fn candidate_state(
    case: &crate::case::AuthoredCase,
    backend: Backend,
    metals: &[&crate::observation::MetalObservation],
    candidates: &[crate::observation::CandidateObservation],
    venues: &crate::c3::VenueKeys,
) -> EvidenceState {
    let candidates = candidates
        .iter()
        .filter(|observation| observation.case_id == case.case_id)
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return EvidenceState::Missing;
    }
    let states = candidates
        .iter()
        .map(|candidate| candidate_slot_state(case, backend, metals, candidate, venues))
        .collect::<Vec<_>>();
    summarize_existing_states(&states)
}

fn summarize_existing_states(states: &[EvidenceState]) -> EvidenceState {
    if states.contains(&EvidenceState::Failed) {
        EvidenceState::Failed
    } else if states.contains(&EvidenceState::Stale) {
        EvidenceState::Stale
    } else if states.is_empty() {
        EvidenceState::Missing
    } else {
        EvidenceState::Current
    }
}

/// The engine's translator fingerprint of the profile whose `profile_fingerprint` a row's or
/// refusal's `environment` records, under this product: what that record carries when this product
/// compiled it. `None` when the fingerprint is missing or malformed.
fn engine_translator_fingerprint(environment: &serde_json::Value) -> Option<String> {
    let hex = environment.get("profile_fingerprint")?.as_str()?;
    Some(engine_fingerprint_of(&decode_hash256(hex)?))
}

/// Whether a row's or refusal's `translator_fingerprint`, with the `environment` it records, is
/// this product's: the engine's fingerprint of the profile it records
/// ([`engine_translator_fingerprint`]), or, for a run through the runtime, which records no
/// profile, the runtime's ([`runtime_fingerprint`](crate::candidate::engine::runtime_fingerprint)).
fn this_build(environment: &serde_json::Value, translator_fingerprint: &str) -> bool {
    engine_translator_fingerprint(environment).as_deref() == Some(translator_fingerprint)
        || translator_fingerprint == crate::candidate::engine::runtime_fingerprint()
}

/// The digest a lowercase 64-digit hexadecimal SHA-256 spells, or `None` for anything else.
fn decode_hash256(hex: &str) -> Option<[u8; 32]> {
    let digits = hex.as_bytes();
    if digits.len() != 64 {
        return None;
    }
    let nibble = |digit: u8| match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        _ => None,
    };
    let mut out = [0u8; 32];
    for (byte, pair) in out.iter_mut().zip(digits.as_chunks::<2>().0.iter()) {
        *byte = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(out)
}

/// One dependency a record's currency is judged on. Each kind of record lists one check per
/// dependency (`candidate_staleness`, `refusal_staleness`), and [`Staleness::judge`] is the one
/// decision over the list: a new dependency is a variant here and one check in each list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dependency {
    /// The executor ABI: the engine executor's. A row of the old engine's executor, deleted,
    /// holds another and is stale in it.
    Executor,
    /// The product build: the translator fingerprint this product derives from the device facts
    /// the record holds (missing or malformed facts count as another), and the product its census
    /// names. A venue's engine record of another build holds it while what it executed is
    /// this build's artifact of its case on its profile-leg, under this product's runtime version
    /// ([`crate::c3::VenueKeys::artifact_is_current`]).
    Product,
    /// The venue profile: a record in a venue's native slot recorded the key of the committed
    /// profile its venue holds that slot to ([`crate::c3::VenueKeys::holds`]). A record in any
    /// other slot is no venue's, and holds it.
    Venue,
    /// Every other dependency: its content or comparison class, its case's AIR or inputs, its
    /// backend, or, for a row, a current Metal row it compared against with the status it
    /// records.
    Other,
}

impl Dependency {
    /// Every dependency, in the order a [`Staleness`] lists them.
    pub const ALL: [Dependency; 4] = [
        Dependency::Executor,
        Dependency::Product,
        Dependency::Venue,
        Dependency::Other,
    ];

    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// What a candidate row or a refusal record recorded that this build does not: the verdict of the
/// one currency rule that the evidence census, `corpus-status`, the C3 report, and the coverage
/// report all judge records by, the set of [`Dependency`]s whose check failed. A record is current
/// exactly when it differs in nothing ([`Staleness::is_current`]). Execution coverage is therefore
/// a claim about what a build executes: a row is current under the product, translation semantics,
/// and executor that recorded it, so a rows-only commit at the recorded product keeps it current,
/// and a product change after it makes it stale unless the row is a venue's whose artifact this
/// build compiles byte for byte.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Staleness {
    /// One bit per stale [`Dependency`] ([`Dependency::bit`]).
    stale: u8,
}

impl Staleness {
    /// The one decision of the currency rule: given each dependency's check (whether the record
    /// holds it under this build), the record is stale in exactly the dependencies whose check
    /// fails.
    pub fn judge(checks: impl IntoIterator<Item = (Dependency, bool)>) -> Staleness {
        Staleness::of(
            checks
                .into_iter()
                .filter(|&(_, holds)| !holds)
                .map(|(dependency, _)| dependency),
        )
    }

    /// Stale in exactly `dependencies`.
    pub fn of(dependencies: impl IntoIterator<Item = Dependency>) -> Staleness {
        Staleness {
            stale: dependencies
                .into_iter()
                .fold(0, |stale, dependency| stale | dependency.bit()),
        }
    }

    /// Whether the record differs from this build in `dependency`.
    pub fn is_stale_in(self, dependency: Dependency) -> bool {
        self.stale & dependency.bit() != 0
    }

    /// The dependencies the record differs in, in [`Dependency::ALL`]'s order.
    pub fn dependencies(self) -> impl Iterator<Item = Dependency> {
        Dependency::ALL
            .into_iter()
            .filter(move |&dependency| self.is_stale_in(dependency))
    }

    /// Whether the record differs from this build in nothing.
    pub fn is_current(self) -> bool {
        self.stale == 0
    }
}

impl std::fmt::Debug for Staleness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.dependencies()).finish()
    }
}

/// [`Staleness`] of one candidate row of `case` in `backend` under this build. `metals` are the
/// case's current Metal rows (`MetalObservation::dependency_matches` under the current oracle):
/// the row's Metal dependency holds when one of them is the row it compared against and the
/// recorded status is the verdict of its class on that row's output. `venues` holds each venue's
/// native slots to its profile's key, and holds a venue's row of another build current for its
/// product while it executed this build's artifact.
pub(crate) fn candidate_staleness(
    case: &crate::case::AuthoredCase,
    backend: Backend,
    metals: &[&crate::observation::MetalObservation],
    candidate: &crate::observation::CandidateObservation,
    venues: &crate::c3::VenueKeys,
) -> Staleness {
    let census_product = candidate
        .census
        .as_ref()
        .is_none_or(|census| census.product == crate::observation::TRANSLATOR_FINGERPRINT);
    let same_build =
        census_product && this_build(&candidate.environment, &candidate.translator_fingerprint);
    let executed = candidate
        .census
        .as_ref()
        .and_then(|census| census.artifact.as_ref());
    Staleness::judge([
        (
            Dependency::Executor,
            candidate.executor_abi == ENGINE_EXECUTOR_ABI,
        ),
        (
            Dependency::Product,
            same_build
                || venues.artifact_is_current(
                    backend,
                    &candidate.environment_id,
                    &case.case_id,
                    executed,
                ),
        ),
        (
            Dependency::Venue,
            venues.holds(backend, &candidate.environment_id, &candidate.environment),
        ),
        // Every other dependency, the build's own facts taken as the row records them.
        (
            Dependency::Other,
            metals.iter().any(|metal| {
                candidate.dependency_matches(&CandidateDependencies {
                    case,
                    metal,
                    spv_sha256: candidate.spv_sha256.as_deref(),
                    translator_fingerprint: &candidate.translator_fingerprint,
                    backend,
                    environment_id: &candidate.environment_id,
                    executor_abi: &candidate.executor_abi,
                })
            }),
        ),
    ])
}

/// The state of one candidate row of `case` under this build ([`candidate_staleness`]): `Current`
/// when it matched and is current, `Failed` when it mismatched and is current, `Stale` otherwise.
/// `metals` are the case's current Metal rows.
pub(crate) fn candidate_slot_state(
    case: &crate::case::AuthoredCase,
    backend: Backend,
    metals: &[&crate::observation::MetalObservation],
    candidate: &crate::observation::CandidateObservation,
    venues: &crate::c3::VenueKeys,
) -> EvidenceState {
    if !candidate_staleness(case, backend, metals, candidate, venues).is_current() {
        return EvidenceState::Stale;
    }
    match candidate.status {
        crate::observation::CandidateStatus::Match => EvidenceState::Current,
        crate::observation::CandidateStatus::Mismatch => EvidenceState::Failed,
    }
}

/// [`Staleness`] of a refusal record of `case` (`crate::refusal`) in `backend` under this build, by
/// the rule a row is held to: the new executor's ABI, the translator fingerprint this product
/// derives from the profile fingerprint it recorded (or, for a venue's refusal made after it
/// compiled, this build's artifact), for a venue's refusal the venue profile `venues` holds its
/// slot to, and a well-formed record of this case's AIR and inputs in `backend`. A refusal compares
/// nothing with Metal, so no Metal row is a dependency.
pub(crate) fn refusal_staleness(
    case: &crate::case::AuthoredCase,
    backend: Backend,
    refusal: &crate::refusal::RefusalObservation,
    venues: &crate::c3::VenueKeys,
) -> Staleness {
    Staleness::judge([
        (
            Dependency::Executor,
            refusal.executor_abi == ENGINE_EXECUTOR_ABI,
        ),
        (
            Dependency::Product,
            this_build(&refusal.environment, &refusal.translator_fingerprint)
                || venues.artifact_is_current(
                    backend,
                    &refusal.environment_id,
                    &case.case_id,
                    refusal.artifact.as_ref(),
                ),
        ),
        (
            Dependency::Venue,
            venues.holds(backend, &refusal.environment_id, &refusal.environment),
        ),
        (
            Dependency::Other,
            refusal.validate_content().is_ok()
                && refusal.case_id == case.case_id
                && refusal.air_sha256 == case.air_sha256
                && case
                    .computed_input_sha256()
                    .is_ok_and(|digest| digest == refusal.input_sha256)
                && refusal.backend == backend,
        ),
    ])
}

/// Whether a refusal record of `case` is current under this build ([`refusal_staleness`]). The C3
/// report judges refusals by it (`c3`).
pub(crate) fn refusal_is_current(
    case: &crate::case::AuthoredCase,
    backend: Backend,
    refusal: &crate::refusal::RefusalObservation,
    venues: &crate::c3::VenueKeys,
) -> bool {
    refusal_staleness(case, backend, refusal, venues).is_current()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::engine::engine_environment_id;
    use crate::case::{
        AuthoredCase, BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection,
        ResourceRole, Stage,
    };
    use crate::observation::{
        CandidateObservation, CandidateStatus, MetalObservation, MetalStatus,
    };
    use crate::observation_contract::ComparisonClass;
    use crate::source::public_sources;
    use crate::ScratchDir;

    fn synthetic_public_case() -> AuthoredCase {
        let source = public_sources().unwrap().remove(0);
        let mut case = AuthoredCase {
            air_sha256: source.air_sha256,
            case_id: String::new(),
            name: "index-test".into(),
            entry: source.entry,
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("AAAAAA==".into()),
                rasterization_rate_map: None,
            }],
            argument_buffer_buffers: vec![],
            device_buffer_arrays: vec![],
            command_buffers: vec![],
            pipeline_states: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            argument_buffer_samplers: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            varying_values: Vec::new(),
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
                stage_in_region: None,
            }),
            draw: None,
            tessellation: None,
            mesh_draw: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            fuel: None,
            out_of_bounds: None,
            rationale: None,
            authored_by: None,
            render_pass_rate_map: None,
            raster_sample_count: None,
            public_source: None,
            no_effect: None,
            creation_bound: None,
        };
        case.case_id = case.computed_case_id().unwrap();
        case
    }

    fn synthetic_observations(case: &AuthoredCase) -> (MetalObservation, CandidateObservation) {
        let input_sha256 = case.computed_input_sha256().unwrap();
        let output_b64 = "KgAAAA==";
        let output_sha256 = crate::hash::sha256_bytes(&[42, 0, 0, 0]);
        let metal = MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: input_sha256.clone(),
            metal_output_sha256: output_sha256.clone(),
            output_b64: output_b64.into(),
            environment_id: "metal-env".into(),
            environment: serde_json::json!({}),
            oracle_abi: ORACLE_ABI.into(),
            status: MetalStatus::Qualified,
        };
        let candidate = CandidateObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256,
            golden_output_sha256: output_sha256.clone(),
            spv_sha256: Some("33".repeat(32)),
            translator_fingerprint: engine_fingerprint_of(&[7; 32]),
            candidate_output_sha256: output_sha256,
            output_b64: output_b64.into(),
            backend: Backend::Moltenvk,
            environment_id: engine_environment_id("moltenvk-env"),
            environment: serde_json::json!({
                "profile_fingerprint": crate::hash::hex_encode([7; 32]),
            }),
            executor_abi: ENGINE_EXECUTOR_ABI.into(),
            comparison: ComparisonClass::Exact,
            census: None,
            status: CandidateStatus::Match,
        };
        (metal, candidate)
    }

    #[test]
    fn rebuilding_index_twice_is_logically_equivalent() {
        let scratch = ScratchDir::new("index-rebuild").unwrap();
        let case = synthetic_public_case();
        let (metal, candidate) = synthetic_observations(&case);
        let store = CorpusStore::new(scratch.path());
        store.put_case(case.clone()).unwrap();
        store.upsert_metal(metal).unwrap();
        store.upsert_candidate(candidate).unwrap();

        let public = public_sources().unwrap();
        // The scratch corpus holds only the public sources, so this limit selects every row.
        let every = public.len();
        let source = public
            .into_iter()
            .find(|source| source.air_sha256 != case.air_sha256)
            .unwrap();
        let first = scratch.path().join("first.sqlite");
        let second = scratch.path().join("second.sqlite");
        rebuild_index(scratch.path(), &first).unwrap();
        rebuild_index(scratch.path(), &second).unwrap();
        assert_eq!(
            status_counts(&first).unwrap(),
            status_counts(&second).unwrap()
        );
        assert_eq!(
            select_queue(&first, QueueState::Unplanned, every).unwrap(),
            select_queue(&second, QueueState::Unplanned, every).unwrap()
        );
        assert!(select_queue(&first, QueueState::Unplanned, every)
            .unwrap()
            .iter()
            .any(|row| row.air_sha256 == source.air_sha256));
        check_index(scratch.path(), &first).unwrap();
    }

    /// A private source row: the first public fixture's module with comment `tag` appended, so
    /// its hash is new and no public fixture holds it.
    fn synthetic_private_source(tag: &str) -> SourceRow {
        let mut source = public_sources().unwrap().into_iter().next().unwrap();
        source
            .air_ll
            .push_str(&format!("\n; private index fixture {tag}\n"));
        source.air_sha256 = crate::hash::sha256_bytes(source.air_ll.as_bytes());
        source.lib_sha256s = vec!["owned-synthetic".into()];
        source.label = format!("private/{tag}");
        source
    }

    /// A private checkout: a public case and a case over one private source,
    /// each with a Metal and a MoltenVK row. Returns the root, the public case, and the private
    /// case.
    fn private_checkout(name: &str) -> (ScratchDir, AuthoredCase, AuthoredCase) {
        let scratch = ScratchDir::new(name).unwrap();
        let authored = synthetic_private_source("authored");
        crate::source::write_source_shards(scratch.path(), [authored.clone()]).unwrap();
        let store = CorpusStore::new(scratch.path());
        let public_case = synthetic_public_case();
        let mut private_case = synthetic_public_case();
        private_case.air_sha256 = authored.air_sha256;
        private_case.case_id = private_case.computed_case_id().unwrap();
        for case in [&public_case, &private_case] {
            let (metal, candidate) = synthetic_observations(case);
            store.put_case(case.clone()).unwrap();
            store.upsert_metal(metal).unwrap();
            store.upsert_candidate(candidate).unwrap();
        }
        (scratch, public_case, private_case)
    }

    #[test]
    fn a_public_checkout_indexes_its_public_rows_and_counts_those_of_private_air() {
        let (scratch, public_case, private_case) = private_checkout("index-public-checkout");
        let root = scratch.path();
        let index = root.join("index.sqlite");
        assert_eq!(rebuild_index(root, &index).unwrap(), IndexScope::Complete);
        let complete = logical_snapshot(&index).unwrap();
        assert!(complete
            .iter()
            .any(|line| line.contains(&private_case.case_id)));

        // The same committed rows in a checkout with no private store at all: those of private AIR
        // are expected and unavailable, so a warm sync and a rebuild both index the rest.
        fs::remove_dir_all(root.join("local")).unwrap();
        let expected = IndexScope::PublicOnly(UnavailablePrivateRows {
            cases: 1,
            metal_observations: 1,
            candidate_observations: 1,
        });
        assert_eq!(sync_index_with_stats(root, &index).unwrap().scope, expected);
        let rebuilt = root.join("rebuilt.sqlite");
        assert_eq!(rebuild_index(root, &rebuilt).unwrap(), expected);
        let public_only = logical_snapshot(&index).unwrap();
        assert_eq!(public_only, logical_snapshot(&rebuilt).unwrap());
        for kind in ["case|", "metal|", "candidate|"] {
            assert!(public_only
                .iter()
                .any(|line| line.starts_with(kind) && line.contains(&public_case.case_id)));
        }
        assert!(!public_only
            .iter()
            .any(|line| line.contains(&private_case.case_id)));
        check_index(root, &index).unwrap();
    }

    #[test]
    fn a_private_checkout_missing_a_source_shard_still_fails_to_index() {
        // The shard of the private case's source.
        let (scratch, _, private_case) = private_checkout("index-missing-case-shard");
        let root = scratch.path();
        let index = root.join("index.sqlite");
        rebuild_index(root, &index).unwrap();
        fs::remove_file(crate::source::source_shard_path(
            root,
            crate::source::shard_index_for_hash(&private_case.air_sha256).unwrap(),
        ))
        .unwrap();
        let error = rebuild_index(root, &root.join("rebuilt.sqlite")).unwrap_err();
        assert!(error.starts_with("index case: "), "{error}");
        let error = sync_index_with_stats(root, &index).unwrap_err();
        assert!(error.starts_with("index case: "), "{error}");
    }

    /// The index's case rows are the checkout's: an index synchronized while a case shard held a
    /// case the checkout's no longer holds (an index made at a newer commit) lists that case for
    /// its row until a reconciliation, which reads only the changed case shard, retires the case
    /// with its observations, and leaves the index the audit's view of the case shard. A shard
    /// rewritten with the same bytes is hashed and not read; a warm pass reads none.
    #[test]
    fn a_checkout_reconciliation_retires_the_cases_its_case_shards_no_longer_hold() {
        let scratch = ScratchDir::new("index-case-reconciliation").unwrap();
        let root = scratch.path();
        let store = CorpusStore::new(root);
        let kept = synthetic_public_case();
        let mut dropped = synthetic_public_case();
        dropped.name = "index-test-dropped".into();
        dropped.buffers[0].initial_bytes_b64 = Some("AQAAAA==".into());
        dropped.case_id = dropped.computed_case_id().unwrap();
        for case in [&kept, &dropped] {
            let (metal, candidate) = synthetic_observations(case);
            store.put_case(case.clone()).unwrap();
            store.upsert_metal(metal).unwrap();
            store.upsert_candidate(candidate).unwrap();
        }
        let index = root.join("index.sqlite");
        rebuild_index(root, &index).unwrap();
        let public = public_source_set().unwrap();
        let row = public
            .rows
            .iter()
            .find(|row| row.air_sha256 == kept.air_sha256 && row.entry == kept.entry)
            .unwrap()
            .clone();
        // What `corpus-triage --audit specialization` compares: the index's case ids for the row
        // against the ones its case shard holds.
        let agree = || {
            let mut held = store
                .find_cases_for_row(&row)
                .unwrap()
                .into_iter()
                .map(|case| case.case_id)
                .collect::<Vec<_>>();
            held.sort();
            let listed = crate::specialization::audit::case_ids(&index)
                .unwrap()
                .remove(row.key().as_ref())
                .unwrap_or_default();
            (listed == held, listed)
        };
        assert!(agree().0);
        let unchanged = reconcile_checkout_rows(root, &index, &public).unwrap();
        assert!(unchanged.is_unchanged(), "{unchanged:?}");
        assert_eq!(unchanged.cases.shards_read, 0);

        // The checkout's case shard no longer holds `dropped`.
        store
            .delete_named_case(&dropped.air_sha256, Some(&dropped.entry), &dropped.name)
            .unwrap();
        let (agrees, listed) = agree();
        assert!(!agrees && listed.contains(&dropped.case_id), "{listed:?}");
        let reconciled = reconcile_checkout_rows(root, &index, &public).unwrap();
        assert_eq!(
            reconciled.cases,
            CaseReconciliation {
                shards_read: 1,
                retired: 1,
                added: 0,
            }
        );
        assert!(agree().0, "{:?}", agree().1);
        assert!(!logical_snapshot(&index)
            .unwrap()
            .iter()
            .any(|line| line.contains(&dropped.case_id)));
        check_index(root, &index).unwrap();

        // The same bytes under a new stamp are hashed, not read; a warm pass opens nothing.
        let shard = root
            .join("cases")
            .join(crate::source::shard_name_for_hash(&kept.air_sha256).unwrap());
        let bytes = fs::read(&shard).unwrap();
        fs::remove_file(&shard).unwrap();
        fs::write(&shard, &bytes).unwrap();
        let rewritten = reconcile_checkout_rows(root, &index, &public).unwrap();
        assert_eq!(rewritten.cases, CaseReconciliation::default());
        let warm = reconcile_checkout_rows(root, &index, &public).unwrap();
        assert_eq!(warm.cases, CaseReconciliation::default());

        // A case the shard gains is added.
        store.put_case(dropped.clone()).unwrap();
        let regained = reconcile_checkout_rows(root, &index, &public).unwrap();
        assert_eq!(
            regained.cases,
            CaseReconciliation {
                shards_read: 1,
                retired: 0,
                added: 1,
            }
        );
        assert!(agree().0);
        check_index(root, &index).unwrap();

        // An index made before case-shard tracking has its case rows replaced once.
        Connection::open(&index)
            .unwrap()
            .execute_batch(
                "DELETE FROM index_metadata WHERE key='case_shard_tracking';
                 UPDATE cases SET case_shard=NULL;",
            )
            .unwrap();
        let migrated = reconcile_checkout_rows(root, &index, &public).unwrap();
        assert_eq!((migrated.cases.retired, migrated.cases.added), (2, 2));
        assert!(agree().0);
        let warm = reconcile_checkout_rows(root, &index, &public).unwrap();
        assert_eq!(warm.cases, CaseReconciliation::default());
        check_index(root, &index).unwrap();
    }

    #[test]
    fn incremental_sync_reindexes_only_a_changed_source_shard() {
        let scratch = ScratchDir::new("index-incremental").unwrap();
        let mut source = public_sources().unwrap().into_iter().next().unwrap();
        source.air_ll.push_str("\n; private index fixture\n");
        source.air_sha256 = crate::hash::sha256_bytes(source.air_ll.as_bytes());
        source.lib_sha256s = vec!["owned-synthetic".into()];
        source.label = "private/before".into();
        crate::source::write_source_shards(scratch.path(), [source.clone()]).unwrap();
        let index = scratch.path().join("index.sqlite");
        rebuild_index(scratch.path(), &index).unwrap();

        source.label = "private/after-a-changed-shard".into();
        source.lib_sha256s = vec!["00".repeat(32)];
        crate::source::merge_source_shards(scratch.path(), [source.clone()]).unwrap();
        let stats = sync_index_with_stats(scratch.path(), &index).unwrap();
        assert_eq!(stats.source_shards_scanned, 1);
        assert_eq!(
            stats.source_bytes_scanned,
            fs::metadata(crate::source::source_shard_path(
                scratch.path(),
                crate::source::shard_index_for_hash(&source.air_sha256).unwrap()
            ))
            .unwrap()
            .len()
        );

        let row = select_queue(&index, QueueState::Unplanned, usize::MAX)
            .unwrap()
            .into_iter()
            .find(|row| row.air_sha256 == source.air_sha256)
            .unwrap();
        assert_eq!(row.label, source.label);
        check_index(scratch.path(), &index).unwrap();
    }

    #[test]
    fn source_merge_publishes_locations_without_a_followup_body_scan() {
        let scratch = ScratchDir::new("index-merge-handoff").unwrap();
        let mut source = public_sources().unwrap().into_iter().next().unwrap();
        source
            .air_ll
            .push_str("\n; indexed merge handoff fixture\n");
        source.air_sha256 = crate::hash::sha256_bytes(source.air_ll.as_bytes());
        source.lib_sha256s = vec!["owned-synthetic".into()];
        source.label = "private/before".into();
        crate::source::write_source_shards(scratch.path(), [source.clone()]).unwrap();
        let index = default_index_path(scratch.path());
        rebuild_index(scratch.path(), &index).unwrap();
        let connection = Connection::open(&index).unwrap();
        connection
            .execute(
                "INSERT INTO cases (case_id, air_sha256, name, input_sha256) \
                 VALUES ('retained-case', ?1, 'retained', 'input')",
                [&source.air_sha256],
            )
            .unwrap();
        drop(connection);

        source.label = "private/after-direct-handoff".into();
        source.lib_sha256s = vec!["00".repeat(32)];
        crate::source::merge_source_shards(scratch.path(), [source.clone()]).unwrap();

        let connection = Connection::open(&index).unwrap();
        connection
            .execute("DELETE FROM cases WHERE case_id='retained-case'", [])
            .unwrap();
        drop(connection);

        let stats = sync_index_with_stats(scratch.path(), &index).unwrap();
        assert_eq!(stats.source_shards_scanned, 0);
        assert_eq!(stats.source_bytes_scanned, 0);
        let connection = Connection::open(&index).unwrap();
        let indexed = connection
            .query_row(
                "SELECT label, source_offset, source_length FROM sources WHERE air_sha256=?1",
                [&source.air_sha256],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<i64>>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(indexed.0, source.label);
        assert!(indexed.1.is_some());
        assert!(indexed.2.is_some());
        assert_eq!(
            connection
                .prepare(
                    "SELECT lib_sha256 FROM source_library_memberships \
                     WHERE air_sha256=?1 ORDER BY lib_sha256",
                )
                .unwrap()
                .query_map([&source.air_sha256], |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec!["00".repeat(32), "owned-synthetic".into()]
        );
        let read_stats = crate::source::for_each_indexed_source_analysis_with_stats(
            scratch.path(),
            &index,
            std::slice::from_ref(&source.air_sha256),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(read_stats.rows, 1);
        assert_eq!(read_stats.source_shards_opened, 1);
        assert_eq!(read_stats.source_bytes_read, indexed.2.unwrap() as u64);
        assert_eq!(read_stats.repair_shards_scanned, 0);
        assert_eq!(read_stats.repair_bytes_scanned, 0);
        check_index(scratch.path(), &index).unwrap();
    }

    #[test]
    fn source_shard_handoff_removes_rows_absent_from_the_replacement() {
        let scratch = ScratchDir::new("index-source-replacement").unwrap();
        let mut retained = public_sources().unwrap().into_iter().next().unwrap();
        retained.air_ll.push_str("\n; retained private fixture\n");
        retained.air_sha256 = crate::hash::sha256_bytes(retained.air_ll.as_bytes());
        retained.lib_sha256s = vec!["11".repeat(32), "22".repeat(32)];
        retained.label = "private/retained".into();
        let shard = crate::source::shard_index_for_hash(&retained.air_sha256).unwrap();
        let mut stale = retained.clone();
        for nonce in 0..u32::MAX {
            stale.air_ll = format!("{}\n; stale private fixture {nonce}\n", retained.air_ll);
            stale.air_sha256 = crate::hash::sha256_bytes(stale.air_ll.as_bytes());
            if crate::source::shard_index_for_hash(&stale.air_sha256).unwrap() == shard {
                break;
            }
        }
        assert_ne!(stale.air_sha256, retained.air_sha256);
        stale.lib_sha256s = vec!["33".repeat(32)];
        stale.label = "private/stale".into();
        crate::source::write_source_shards(scratch.path(), [retained.clone(), stale.clone()])
            .unwrap();
        let index = default_index_path(scratch.path());
        rebuild_index(scratch.path(), &index).unwrap();

        let source_path = crate::source::source_shard_path(scratch.path(), shard);
        let encoded = format!(
            "{}\n",
            crate::jsonl::to_sorted_json_string(&retained).unwrap()
        );
        fs::write(&source_path, encoded.as_bytes()).unwrap();
        record_source_shard_write(
            scratch.path(),
            shard,
            &[crate::source::SourceIndexLocation {
                air_sha256: retained.air_sha256.clone(),
                stage: retained.stage.clone(),
                entry: retained.entry.clone(),
                label: retained.label.clone(),
                lib_sha256s: retained.lib_sha256s.clone(),
                offset: 0,
                length: encoded.len().try_into().unwrap(),
            }],
        )
        .unwrap();

        let connection = Connection::open(&index).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM sources WHERE air_sha256=?1",
                    [&stale.air_sha256],
                    |row| row.get::<_, usize>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM source_library_memberships WHERE air_sha256=?1",
                    [&stale.air_sha256],
                    |row| row.get::<_, usize>(0),
                )
                .unwrap(),
            0
        );
        check_index(scratch.path(), &index).unwrap();
    }

    #[test]
    fn legacy_migration_and_lookup_never_open_an_unrelated_shard() {
        let scratch = ScratchDir::new("index-lazy-locations").unwrap();
        let mut source = public_sources().unwrap().into_iter().next().unwrap();
        source
            .air_ll
            .push_str("\n; private lazy-location fixture\n");
        source.air_sha256 = crate::hash::sha256_bytes(source.air_ll.as_bytes());
        source.lib_sha256s = vec!["owned-synthetic".into()];
        source.label = "private/lazy".into();
        let source_shard = crate::source::shard_index_for_hash(&source.air_sha256).unwrap();
        let mut other = source.clone();
        for nonce in 0..u32::MAX {
            other.air_ll = format!("{}\n; unrelated shard {nonce}\n", source.air_ll);
            other.air_sha256 = crate::hash::sha256_bytes(other.air_ll.as_bytes());
            if crate::source::shard_index_for_hash(&other.air_sha256).unwrap() != source_shard {
                break;
            }
        }
        other.label = "private/unrelated".into();
        crate::source::write_source_shards(scratch.path(), [source.clone(), other.clone()])
            .unwrap();
        let index = scratch.path().join(".index.sqlite");
        rebuild_index(scratch.path(), &index).unwrap();
        let connection = Connection::open(&index).unwrap();
        connection
            .execute(
                "UPDATE sources SET source_offset=NULL, source_length=NULL WHERE air_sha256 IN (?1, ?2)",
                [&source.air_sha256, &other.air_sha256],
            )
            .unwrap();
        connection
            .execute("DROP TABLE indexed_source_shards", [])
            .unwrap();
        connection.execute("DROP TABLE index_metadata", []).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE indexed_source_shards (
                   shard INTEGER PRIMARY KEY,
                   size INTEGER NOT NULL,
                   modified_ns INTEGER NOT NULL
                 );
                 INSERT INTO indexed_source_shards (shard, size, modified_ns)
                   VALUES (63, 1, 1);",
            )
            .unwrap();
        drop(connection);

        // Preserve readable metadata but make any attempted body scan fail. Both migration and a
        // lookup in `source_shard` must leave this unrelated shard unopened.
        let unrelated_path = crate::source::source_shard_path(
            scratch.path(),
            crate::source::shard_index_for_hash(&other.air_sha256).unwrap(),
        );
        let unrelated_len = fs::metadata(&unrelated_path).unwrap().len() as usize;
        fs::write(&unrelated_path, vec![b'x'; unrelated_len]).unwrap();

        let stats = sync_index_with_stats(scratch.path(), &index).unwrap();
        assert_eq!(stats.source_shards_scanned, 0);
        assert_eq!(stats.source_bytes_scanned, 0);
        let connection = Connection::open(&index).unwrap();
        let before = connection
            .query_row(
                "SELECT source_offset FROM sources WHERE air_sha256=?1",
                [&source.air_sha256],
                |row| row.get::<_, Option<i64>>(0),
            )
            .unwrap();
        assert_eq!(before, None);
        drop(connection);

        assert_eq!(
            crate::source::find_source(scratch.path(), &source.air_sha256)
                .unwrap()
                .unwrap()
                .label,
            source.label
        );
        let connection = Connection::open(&index).unwrap();
        let after = connection
            .query_row(
                "SELECT source_offset FROM sources WHERE air_sha256=?1",
                [&source.air_sha256],
                |row| row.get::<_, Option<i64>>(0),
            )
            .unwrap();
        assert!(after.is_some());
        let unrelated = connection
            .query_row(
                "SELECT source_offset FROM sources WHERE air_sha256=?1",
                [&other.air_sha256],
                |row| row.get::<_, Option<i64>>(0),
            )
            .unwrap();
        assert_eq!(unrelated, None, "lookup must not scan an unrelated shard");
    }

    /// An engine row as the executor records it: its slot, ABI, and the fingerprint of the
    /// profile in its environment.
    fn engine_observation(
        case: &AuthoredCase,
        profile: [u8; 32],
    ) -> (MetalObservation, CandidateObservation) {
        let (metal, mut candidate) = synthetic_observations(case);
        candidate.translator_fingerprint = engine_fingerprint_of(&profile);
        candidate.environment["profile_fingerprint"] = crate::hash::hex_encode(profile).into();
        (metal, candidate)
    }

    /// An engine row recorded previously carries the old engine's Vulkan target in its
    /// environment (`vulkan_target`, which the executor no longer builds): such a row still parses
    /// from its shard's line, checks, and is judged by the one currency rule as the same row
    /// without the field is.
    #[test]
    fn a_row_recording_the_retired_vulkan_target_parses_and_stays_current() {
        let case = synthetic_public_case();
        let (metal, without) = engine_observation(&case, [7; 32]);
        let mut recorded = without.clone();
        recorded.environment["vulkan_target"] = serde_json::json!({
            "version": [1, 2],
            "max_bound_descriptor_sets": 8,
            "denorm_behavior_independence": "Bit32Only",
            "rounding_mode_independence": "All",
            "extensions": ["ExtShaderStencilExport", "KhrShaderFloatControls2"],
            "features": ["BufferDeviceAddress", "SampleRateShading", "ShaderInt64"],
            "properties": ["ShaderSignedZeroInfNanPreserveFloat32"],
        });
        let line = serde_json::to_string(&recorded).unwrap();
        let parsed: CandidateObservation = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed, recorded);
        parsed.validate_content().unwrap();
        let state = |candidate: &CandidateObservation| {
            candidate_slot_state(
                &case,
                Backend::Moltenvk,
                &[&metal],
                candidate,
                &crate::c3::VenueKeys::default(),
            )
        };
        assert_eq!(state(&without), EvidenceState::Current);
        assert_eq!(state(&parsed), EvidenceState::Current);
    }

    /// A row the old engine's executor (deleted in cutover X2a) recorded: its executor ABI and
    /// target facts, in its environment's bare slot.
    fn old_engine_row(case: &AuthoredCase) -> CandidateObservation {
        let (_, mut row) = synthetic_observations(case);
        row.environment_id = "moltenvk-env".into();
        row.executor_abi = "vulkan-executed-pipeline-v64".into();
        row.translator_fingerprint = "55".repeat(32);
        row.environment = serde_json::json!({"vulkan_target": {"version": [1, 2]}});
        row
    }

    /// A synthetic `vulkaninfo` venue profile of a Vulkan 1.3 device, edited by `edit`: what
    /// [`crate::c3::VenueProfile::parse`] reads, owned by the test.
    fn venue_profile(edit: impl FnOnce(&mut serde_json::Value)) -> crate::c3::VenueProfile {
        let mut document = serde_json::json!({
            "provenance": {"source": "vulkaninfo"},
            "capabilities": {"device": {
                "extensions": {"VK_KHR_format_feature_flags2": 2},
                "features": {
                    "VkPhysicalDeviceFeatures": {"shaderInt16": true},
                    "VkPhysicalDeviceVulkan12Features": {"bufferDeviceAddress": false},
                },
                "properties": {
                    "VkPhysicalDeviceProperties": {
                        "deviceName": "Toy GPU", "vendorID": 4098, "deviceID": 5567,
                        "driverVersion": 109060100,
                        "limits": {"maxComputeWorkGroupInvocations": 1024},
                    },
                    "VkPhysicalDeviceDriverProperties": {
                        "driverID": "VK_DRIVER_ID_MESA_RADV", "driverInfo": "Mesa 26.2.4",
                    },
                },
                "formats": {"VK_FORMAT_R8_UNORM": {
                    "VkFormatProperties": {
                        "linearTilingFeatures": [],
                        "optimalTilingFeatures": [
                            "VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT",
                            "VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT",
                        ],
                        "bufferFeatures": [],
                    },
                    "VkFormatProperties3": {
                        "linearTilingFeatures": [],
                        "optimalTilingFeatures": [
                            "VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT",
                            "VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT",
                        ],
                        "bufferFeatures": [],
                    },
                }},
            }},
            "profiles": {"VP_M2V_toy": {"api-version": "1.3.250", "capabilities": ["device"]}},
        });
        edit(&mut document);
        crate::c3::VenueProfile::parse("toy", document.to_string().as_bytes()).unwrap()
    }

    /// The one currency rule holds a venue's record to its venue's committed profile: the key a
    /// venue run records is the fingerprint of the device profile its committed profile derives,
    /// which every member the translator reads moves (a feature, an extension, a format's 32-bit
    /// bit, and one only `VkFormatProperties3` states), so a record of the old profile reads stale,
    /// row and refusal alike; a member the translator never reads moves nothing, and a record
    /// outside the venue's slots is asked nothing.
    #[test]
    fn a_venue_record_is_current_only_under_its_venue_profiles_key() {
        let base = venue_profile(|_| {});
        fn block(document: &mut serde_json::Value) -> &mut serde_json::Value {
            &mut document["capabilities"]["device"]
        }
        fn r8<'a>(
            document: &'a mut serde_json::Value,
            structure: &str,
        ) -> &'a mut serde_json::Value {
            &mut block(document)["formats"]["VK_FORMAT_R8_UNORM"][structure]
                ["optimalTilingFeatures"]
        }
        type Edit = fn(&mut serde_json::Value);
        let edits: [(&str, Edit); 4] = [
            ("a feature", |document| {
                block(document)["features"]["VkPhysicalDeviceFeatures"]
                    ["fragmentStoresAndAtomics"] = true.into();
            }),
            ("an extension", |document| {
                block(document)["extensions"]["VK_EXT_shader_stencil_export"] = 1.into();
            }),
            ("a 32-bit format bit", |document| {
                for (structure, bit) in [
                    (
                        "VkFormatProperties",
                        "VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT",
                    ),
                    (
                        "VkFormatProperties3",
                        "VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT",
                    ),
                ] {
                    r8(document, structure)
                        .as_array_mut()
                        .unwrap()
                        .push(bit.into());
                }
            }),
            ("a 64-bit-only format bit", |document| {
                r8(document, "VkFormatProperties3")
                    .as_array_mut()
                    .unwrap()
                    .push("VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT".into());
            }),
        ];
        let case = synthetic_public_case();
        let (metal, mut row) = engine_observation(&case, [7; 32]);
        row.environment[crate::c3::VENUE_PROFILE] = base.record();
        let keys = |profile: &crate::c3::VenueProfile| {
            let mut keys = crate::c3::VenueKeys::default();
            keys.insert(
                "moltenvk-env",
                Backend::Moltenvk,
                &profile.fingerprint,
                "toy",
            );
            keys
        };
        let state = |row: &CandidateObservation, keys: &crate::c3::VenueKeys| {
            candidate_slot_state(&case, Backend::Moltenvk, &[&metal], row, keys)
        };
        assert_eq!(state(&row, &keys(&base)), EvidenceState::Current);
        let mut refusal = crate::refusal::RefusalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            translator_fingerprint: row.translator_fingerprint.clone(),
            backend: Backend::Moltenvk,
            environment_id: row.environment_id.clone(),
            environment: row.environment.clone(),
            executor_abi: ENGINE_EXECUTOR_ABI.into(),
            refusal: crate::refusal::Refusal::of(
                &crate::check::admission::NotSupported::NotImplemented("a family".into()),
            ),
            artifact: None,
        };
        refusal.environment["robustness"] = serde_json::json!({"leg": "stated"});
        assert!(refusal_is_current(
            &case,
            Backend::Moltenvk,
            &refusal,
            &keys(&base)
        ));
        for (what, edit) in edits {
            let moved = venue_profile(edit);
            assert_ne!(moved.fingerprint, base.fingerprint, "{what} moves no key");
            assert_eq!(
                state(&row, &keys(&moved)),
                EvidenceState::Stale,
                "a row of the profile before {what} changed"
            );
            assert!(
                !refusal_is_current(&case, Backend::Moltenvk, &refusal, &keys(&moved)),
                "a refusal of the profile before {what} changed"
            );
        }
        // A member the translator never reads, and the file's provenance: the same key.
        let unread = venue_profile(|document| {
            block(document)["properties"]["VkPhysicalDeviceDiscardRectanglePropertiesEXT"] =
                serde_json::json!({"maxDiscardRectangles": 4});
            document["provenance"]["captured"] = "another day".into();
        });
        assert_ne!(unread.sha256, base.sha256);
        assert_eq!(unread.fingerprint, base.fingerprint);
        assert_eq!(state(&row, &keys(&unread)), EvidenceState::Current);
        // Nor a 64-bit-only bit of a device read at Vulkan 1.2 without
        // `VK_KHR_format_feature_flags2`: the venue's executor reads no `VkFormatProperties3`
        // there, and neither does translation.
        let without = |edit: Edit| {
            venue_profile(|document| {
                block(document)["extensions"] = serde_json::json!({});
                edit(document);
            })
            .fingerprint
        };
        assert_eq!(without(|_| {}), without(edits[3].1));
        assert_ne!(without(|_| {}), without(edits[2].1));
        // A venue's record without the key, or of another backend's slot.
        let mut unkeyed = row.clone();
        unkeyed
            .environment
            .as_object_mut()
            .unwrap()
            .remove(crate::c3::VENUE_PROFILE);
        assert_eq!(state(&unkeyed, &keys(&base)), EvidenceState::Stale);
        let mut elsewhere = crate::c3::VenueKeys::default();
        elsewhere.insert("moltenvk-env", Backend::Vulkan, &base.fingerprint, "toy");
        assert_eq!(state(&unkeyed, &elsewhere), EvidenceState::Current);
        assert_eq!(
            state(&unkeyed, &crate::c3::VenueKeys::default()),
            EvidenceState::Current
        );
    }

    #[test]
    fn engine_evidence_is_judged_by_its_recorded_profile_and_abi() {
        let case = synthetic_public_case();
        let (metal, mut candidate) = engine_observation(&case, [7; 32]);
        let state = |candidate: &CandidateObservation| {
            candidate_slot_state(
                &case,
                Backend::Moltenvk,
                &[&metal],
                candidate,
                &crate::c3::VenueKeys::default(),
            )
        };
        assert_eq!(state(&candidate), EvidenceState::Current);
        candidate.environment["profile_fingerprint"] = crate::hash::hex_encode([0xab; 32]).into();
        assert_eq!(state(&candidate), EvidenceState::Stale);
        candidate.translator_fingerprint = engine_fingerprint_of(&[0xab; 32]);
        assert_eq!(state(&candidate), EvidenceState::Current);
        for malformed in [
            serde_json::Value::Null,
            "ab".repeat(31).into(),
            "0G".repeat(32).into(),
            crate::hash::hex_encode([0xab; 32])
                .to_ascii_uppercase()
                .into(),
        ] {
            let mut malformed_row = candidate.clone();
            malformed_row.environment["profile_fingerprint"] = malformed;
            assert_eq!(state(&malformed_row), EvidenceState::Stale);
        }
        // The old engine's rows are held to the same one rule, and are stale in its executor and
        // its product (cutover X2b).
        let old = old_engine_row(&case);
        assert_eq!(state(&old), EvidenceState::Stale);
        assert_eq!(
            candidate_staleness(
                &case,
                Backend::Moltenvk,
                &[&metal],
                &old,
                &crate::c3::VenueKeys::default()
            ),
            Staleness::of([Dependency::Executor, Dependency::Product])
        );
        // A current row whose output differs from Metal's is a failure, not staleness.
        candidate.status = CandidateStatus::Mismatch;
        candidate.output_b64 = "KwAAAA==".into();
        candidate.candidate_output_sha256 = crate::hash::sha256_bytes(&[43, 0, 0, 0]);
        assert_eq!(state(&candidate), EvidenceState::Failed);
    }

    /// The one currency rule names what a record differs from this build in: a row of this build is
    /// current; one at another executor version is stale for the executor, one at another product
    /// (its translator fingerprint, or the product its census names) for the product, one in a
    /// venue's slot without the key of the profile its venue holds the slot to for the venue
    /// profile, and one whose Metal row is not current for another dependency. A refusal is held to
    /// the same rule.
    #[test]
    fn the_currency_rule_names_what_a_record_differs_in() {
        use crate::coverage::{ExecutedCensus, ExecutionClass};
        let case = synthetic_public_case();
        let (metal, candidate) = engine_observation(&case, [7; 32]);
        let none = crate::c3::VenueKeys::default();
        // The candidate's own slot held to a key it did not record.
        let mut keyed = crate::c3::VenueKeys::default();
        keyed.insert("moltenvk-env", Backend::Moltenvk, &"77".repeat(32), "toy");
        assert_eq!(
            keyed.expected(Backend::Moltenvk, &candidate.environment_id),
            Some("77".repeat(32).as_str())
        );
        let staleness = |candidate: &CandidateObservation| {
            candidate_staleness(&case, Backend::Moltenvk, &[&metal], candidate, &none)
        };
        let only = |executor: bool, product: bool, venue: bool, other: bool| {
            Staleness::judge([
                (Dependency::Executor, !executor),
                (Dependency::Product, !product),
                (Dependency::Venue, !venue),
                (Dependency::Other, !other),
            ])
        };
        assert!(staleness(&candidate).is_current());
        assert_eq!(
            candidate_slot_state(&case, Backend::Moltenvk, &[&metal], &candidate, &none),
            EvidenceState::Current
        );
        assert_eq!(
            candidate_staleness(&case, Backend::Moltenvk, &[&metal], &candidate, &keyed),
            only(false, false, true, false)
        );
        let mut executor = candidate.clone();
        executor.executor_abi = "vulkan-engine-v1".into();
        assert_eq!(staleness(&executor), only(true, false, false, false));
        let mut product = candidate.clone();
        product.translator_fingerprint = "66".repeat(32);
        assert_eq!(staleness(&product), only(false, true, false, false));
        let mut both = product.clone();
        both.executor_abi = "vulkan-engine-v1".into();
        assert_eq!(staleness(&both), only(true, true, false, false));
        for row in [&executor, &product, &both] {
            assert_eq!(
                candidate_slot_state(&case, Backend::Moltenvk, &[&metal], row, &none),
                EvidenceState::Stale
            );
        }
        // A census naming another product: that product's row, whatever its fingerprint says.
        let census = |product: &str| ExecutedCensus {
            device: Some(ExecutionClass::CpuIcd),
            product: product.into(),
            restricted_to: None,
            leg: None,
            modules: Vec::new(),
            out_of_bounds: None,
            rate_map_pass: None,
            artifact: None,
        };
        let mut censused = candidate.clone();
        censused.census = Some(census(crate::observation::TRANSLATOR_FINGERPRINT));
        assert!(staleness(&censused).is_current());
        censused.census = Some(census("an older build"));
        assert_eq!(staleness(&censused), only(false, true, false, false));
        // No current Metal row: another dependency.
        assert_eq!(
            candidate_staleness(&case, Backend::Moltenvk, &[], &candidate, &none),
            only(false, false, false, true)
        );

        let mut environment = candidate.environment.clone();
        environment["robustness"] = serde_json::json!({"leg": "stated"});
        let refusal = crate::refusal::RefusalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            translator_fingerprint: candidate.translator_fingerprint.clone(),
            backend: Backend::Moltenvk,
            environment_id: candidate.environment_id.clone(),
            environment,
            executor_abi: ENGINE_EXECUTOR_ABI.into(),
            refusal: crate::refusal::Refusal::of(
                &crate::check::admission::NotSupported::NotImplemented("a family".into()),
            ),
            artifact: None,
        };
        let judged = |refusal: &crate::refusal::RefusalObservation| {
            refusal_staleness(&case, Backend::Moltenvk, refusal, &none)
        };
        assert!(judged(&refusal).is_current());
        assert!(refusal_is_current(
            &case,
            Backend::Moltenvk,
            &refusal,
            &none
        ));
        assert_eq!(
            refusal_staleness(&case, Backend::Moltenvk, &refusal, &keyed),
            only(false, false, true, false)
        );
        let mut executor = refusal.clone();
        executor.executor_abi = "vulkan-engine-v1".into();
        assert_eq!(judged(&executor), only(true, false, false, false));
        let mut product = refusal.clone();
        product.translator_fingerprint = "66".repeat(32);
        assert_eq!(judged(&product), only(false, true, false, false));
        // A refusal recorded in another backend is another backend's record.
        assert_eq!(
            refusal_staleness(&case, Backend::Vulkan, &refusal, &none),
            only(false, false, false, true)
        );
    }

    /// Artifact-keyed currency, over a synthetic store and its census file at two builds. A venue's
    /// row and refusal recorded at another build hold the product while this build's file records
    /// the digest they executed, on their slot's profile-leg, under this runtime version. A moved
    /// digest, another runtime version, no recorded digest, a file at another build (a branch's
    /// copy of master's), no file, and a slot no venue keys all leave the record to the build that
    /// recorded it: stale for its product, and current only there.
    #[test]
    fn a_venue_record_of_another_build_is_current_while_it_executed_this_builds_artifact() {
        use crate::authored_compile::{
            expectations_path, ArtifactBuild, Cell, CellOutcome, CurrentArtifacts, Expectations,
            ProfileLeg,
        };
        use crate::candidate::engine::artifact::ExecutedArtifact;
        use crate::candidate::engine::engine_slot;
        use crate::coverage::{ExecutedCensus, ExecutionClass, Leg};
        use metal2vulkan::cache::RUNTIME_VERSION;
        let scratch = ScratchDir::new("index-artifact-currency").unwrap();
        let root = scratch.path();
        let case = synthetic_public_case();
        let profile = venue_profile(|_| {});
        let (mut metal, mut this_build) = engine_observation(&case, [7; 32]);
        this_build.environment[crate::c3::VENUE_PROFILE] = profile.record();
        // Recorded by the primary Metal reference, the row each candidate is judged against.
        let primary = crate::c3::reference::testing::primary();
        metal.environment_id = primary.environment_id.clone();
        metal.environment = crate::c3::reference::testing::recorded(&primary);
        let references = crate::c3::reference::testing::references();
        let digest = "ab".repeat(32);
        let moved = "cd".repeat(32);
        let executed = |sha256: Option<&str>, runtime_version| {
            Some(ExecutedArtifact {
                runtime_version,
                sha256: sha256.map(Into::into),
            })
        };
        // The venue's row as another build recorded it: another product in its census and in
        // its translator fingerprint.
        let at_another_build = |artifact: Option<ExecutedArtifact>| {
            let mut row = this_build.clone();
            row.translator_fingerprint = "66".repeat(32);
            row.census = Some(ExecutedCensus {
                device: Some(ExecutionClass::CpuIcd),
                product: "an older build".into(),
                restricted_to: None,
                leg: None,
                modules: Vec::new(),
                out_of_bounds: None,
                rate_map_pass: None,
                artifact,
            });
            row
        };
        let row = at_another_build(executed(Some(&digest), RUNTIME_VERSION));
        let store = CorpusStore::new(root);
        store.put_case(case.clone()).unwrap();
        store.upsert_metal(metal.clone()).unwrap();
        store.upsert_candidate(row.clone()).unwrap();

        let stated = ProfileLeg {
            stem: "toy".into(),
            leg: Leg::Stated,
        };
        let off = ProfileLeg {
            stem: "toy".into(),
            leg: Leg::RobustnessOff,
        };
        let another_build = ArtifactBuild {
            product: "77".repeat(32),
            executor: ENGINE_EXECUTOR_ABI.into(),
        };
        // The corpus's census file at `build`, with `stated` and `off` the case's digests.
        let bless = |build: &ArtifactBuild, stated_digest: &str, off_digest: &str| {
            let cell = |digest: &str| Cell {
                outcome: CellOutcome::Compiled,
                uncovered: 0,
                artifact: Some(digest.into()),
            };
            Expectations {
                build: Some(build.clone()),
                profiles: vec![stated.clone(), off.clone()],
                cases: std::collections::BTreeMap::from([(
                    case.case_id.clone(),
                    vec![cell(stated_digest), cell(off_digest)],
                )]),
            }
            .write(&expectations_path(root))
            .unwrap();
        };
        // What the binaries judge by: the venue's key, and the corpus's file at this build.
        let keys = || {
            let mut keys = crate::c3::VenueKeys::default();
            keys.insert(
                "moltenvk-env",
                Backend::Moltenvk,
                &profile.fingerprint,
                "toy",
            );
            keys.with_artifacts(CurrentArtifacts::load(root).unwrap())
        };
        let judged = || {
            evidence_status(root, &keys(), &references)
                .unwrap()
                .remove(0)
                .moltenvk
        };
        let staleness = |row: &CandidateObservation| {
            candidate_staleness(&case, Backend::Moltenvk, &[&metal], row, &keys())
        };
        let product = Staleness::of([Dependency::Product]);

        // No file: the build that recorded it.
        assert_eq!(
            CurrentArtifacts::load(root).unwrap(),
            CurrentArtifacts::default()
        );
        assert_eq!(judged(), EvidenceState::Stale);
        assert_eq!(staleness(&row), product);
        // This build's file records what it executed: current.
        bless(&ArtifactBuild::this(), &digest, &moved);
        assert_eq!(judged(), EvidenceState::Current);
        assert!(staleness(&row).is_current());
        // Under another runtime version, or without a digest: its build's.
        for artifact in [
            executed(Some(&digest), RUNTIME_VERSION + 1),
            executed(None, RUNTIME_VERSION),
            None,
        ] {
            assert_eq!(staleness(&at_another_build(artifact)), product);
        }
        // Another executor is the executor's, whatever it executed.
        let mut executor = row.clone();
        executor.executor_abi = "vulkan-engine-v1".into();
        assert_eq!(staleness(&executor), Staleness::of([Dependency::Executor]));
        // The digest moved: stale.
        bless(&ArtifactBuild::this(), &moved, &digest);
        assert_eq!(judged(), EvidenceState::Stale);
        assert_eq!(staleness(&row), product);
        // The file is at another build, with the very digest: its digests are not this build's,
        // and the row is its own build's.
        bless(&another_build, &digest, &digest);
        assert_eq!(
            CurrentArtifacts::load(root).unwrap(),
            CurrentArtifacts::default()
        );
        assert_eq!(judged(), EvidenceState::Stale);
        // A row of this build is current whatever the file says.
        store.upsert_candidate(this_build.clone()).unwrap();
        assert_eq!(judged(), EvidenceState::Current);
        bless(&ArtifactBuild::this(), &moved, &moved);
        assert_eq!(judged(), EvidenceState::Current);

        // Each slot is judged on its own leg's digest; a slot no venue keys, on its build alone.
        bless(&ArtifactBuild::this(), &digest, &moved);
        let off_slot = engine_slot("moltenvk-env", None, Leg::RobustnessOff);
        let stated_slot = engine_environment_id("moltenvk-env");
        let current = |slot: &str, backend: Backend, sha256: &str| {
            keys().artifact_is_current(
                backend,
                slot,
                &case.case_id,
                executed(Some(sha256), RUNTIME_VERSION).as_ref(),
            )
        };
        assert!(current(&stated_slot, Backend::Moltenvk, &digest));
        assert!(!current(&stated_slot, Backend::Moltenvk, &moved));
        assert!(current(&off_slot, Backend::Moltenvk, &moved));
        assert!(!current(&off_slot, Backend::Moltenvk, &digest));
        assert!(!current(&stated_slot, Backend::Vulkan, &digest));
        assert!(!current("another-env@engine", Backend::Moltenvk, &digest));

        // A refusal the venue's run made after it compiled is held to the same digest.
        let mut refusal = crate::refusal::RefusalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            translator_fingerprint: "66".repeat(32),
            backend: Backend::Moltenvk,
            environment_id: stated_slot.clone(),
            environment: this_build.environment.clone(),
            executor_abi: ENGINE_EXECUTOR_ABI.into(),
            refusal: crate::refusal::Refusal::of(
                &crate::check::admission::NotSupported::NotImplemented("a family".into()),
            ),
            artifact: executed(Some(&digest), RUNTIME_VERSION),
        };
        refusal.environment["robustness"] = serde_json::json!({"leg": "stated"});
        assert!(refusal_is_current(
            &case,
            Backend::Moltenvk,
            &refusal,
            &keys()
        ));
        refusal.artifact = executed(Some(&moved), RUNTIME_VERSION);
        assert_eq!(
            refusal_staleness(&case, Backend::Moltenvk, &refusal, &keys()),
            product
        );
        bless(&another_build, &moved, &moved);
        assert!(!refusal_is_current(
            &case,
            Backend::Moltenvk,
            &refusal,
            &keys()
        ));
    }

    /// The currency rule's one decision: a record is stale in exactly the dependencies whose check
    /// fails, whatever the order or number of checks, and current when every check holds.
    #[test]
    fn the_currency_decision_is_stale_in_each_failed_check() {
        use Dependency::{Executor, Other, Product};
        assert!(Staleness::judge([(Executor, true), (Product, true), (Other, true)]).is_current());
        assert!(Staleness::judge([]).is_current());
        let stale = Staleness::judge([(Other, false), (Executor, false), (Product, true)]);
        assert!(!stale.is_current());
        assert_eq!(stale, Staleness::of([Executor, Other]));
        assert_eq!(stale.dependencies().collect::<Vec<_>>(), [Executor, Other]);
        assert!(stale.is_stale_in(Executor) && !stale.is_stale_in(Product));
        assert_eq!(format!("{stale:?}"), "{Executor, Other}");
        // A check listed twice is stale when either fails.
        assert_eq!(
            Staleness::judge([(Product, true), (Product, false)]),
            Staleness::of([Product])
        );
    }

    /// The census keeps the engines apart: an engine slot neither stales nor stands in for the
    /// old engine's summary, and each slot names its engine.
    /// The evidence census summarizes the engine slots and omits the old engine's rows
    /// (cutover X2b).
    #[test]
    fn evidence_status_omits_the_old_engines_rows() {
        let scratch = ScratchDir::new("index-evidence-engines").unwrap();
        let case = synthetic_public_case();
        let (mut metal, _) = synthetic_observations(&case);
        let old = old_engine_row(&case);
        let primary = crate::c3::reference::testing::primary();
        metal.environment_id = primary.environment_id.clone();
        metal.environment = crate::c3::reference::testing::recorded(&primary);
        let references = crate::c3::reference::testing::references();
        let (_, mut new) = engine_observation(&case, [7; 32]);
        let store = CorpusStore::new(scratch.path());
        store.put_case(case.clone()).unwrap();
        store.upsert_metal(metal).unwrap();
        store.upsert_candidate(old).unwrap();
        store.upsert_candidate(new.clone()).unwrap();
        let row = evidence_status(
            scratch.path(),
            &crate::c3::VenueKeys::default(),
            &references,
        )
        .unwrap()
        .remove(0);
        assert_eq!(row.metal, EvidenceState::Current);
        assert_eq!(row.moltenvk, EvidenceState::Current);
        assert_eq!(row.vulkan, EvidenceState::Missing);
        let slots = row
            .slots
            .iter()
            .map(|slot| (slot.backend, slot.environment_id.as_str(), slot.state))
            .collect::<Vec<_>>();
        assert_eq!(
            slots,
            [
                (
                    "metal",
                    primary.environment_id.as_str(),
                    EvidenceState::Current
                ),
                (
                    "moltenvk",
                    new.environment_id.as_str(),
                    EvidenceState::Current
                ),
            ]
        );

        // A stale engine row makes the backend's summary stale.
        new.translator_fingerprint = engine_fingerprint_of(&[9; 32]);
        store.upsert_candidate(new).unwrap();
        let row = evidence_status(
            scratch.path(),
            &crate::c3::VenueKeys::default(),
            &references,
        )
        .unwrap()
        .remove(0);
        assert_eq!(row.moltenvk, EvidenceState::Stale);
    }

    /// A reference gap is indexed in its own table: its source counts as `reference-gap`, its case
    /// is out of the qualification queue and the refresh selection, and the evidence census names
    /// its Metal slot. An index that predates the table reads as an older build's did until a sync
    /// fills it.
    #[test]
    fn a_reference_gap_is_indexed_and_counted_apart() {
        let scratch = ScratchDir::new("index-reference-gap").unwrap();
        let case = synthetic_public_case();
        let (metal, _) = synthetic_observations(&case);
        let store = CorpusStore::new(scratch.path());
        store.put_case(case.clone()).unwrap();
        let message =
            "simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later";
        // Recorded by the primary reference, the only one whose gap is read.
        let primary = crate::c3::reference::testing::primary();
        store
            .upsert_metal(MetalObservation {
                metal_output_sha256: crate::hash::sha256_bytes(&[]),
                output_b64: String::new(),
                status: MetalStatus::ReferenceGap(crate::observation::ReferenceGap {
                    apple_family: 10,
                    message: message.into(),
                }),
                environment_id: primary.environment_id.clone(),
                environment: crate::c3::reference::testing::recorded(&primary),
                ..metal
            })
            .unwrap();
        let index = scratch.path().join("index.sqlite");
        rebuild_index(scratch.path(), &index).unwrap();
        let count = |index: &Path, state: &str| {
            status_counts(index)
                .unwrap()
                .into_iter()
                .find(|(counted, _)| counted == state)
                .map_or(0, |(_, count)| count)
        };
        assert_eq!(count(&index, "reference-gap"), 1);
        assert_eq!(count(&index, "authored"), 0);
        assert_eq!(count(&index, "metal-qualified"), 0);
        let queued = |index: &Path| {
            select_queue(index, QueueState::Authored, usize::MAX)
                .unwrap()
                .into_iter()
                .any(|row| row.air_sha256 == case.air_sha256)
        };
        assert!(!queued(&index));
        let row = evidence_status(
            scratch.path(),
            &crate::c3::VenueKeys::default(),
            &crate::c3::reference::testing::references(),
        )
        .unwrap()
        .remove(0);
        assert_eq!(row.metal, EvidenceState::ReferenceGap);
        assert_eq!(row.slots[0].state, EvidenceState::ReferenceGap);
        assert_eq!(EvidenceState::ReferenceGap.as_str(), "reference_gap");

        // The index as a build before the table left it: readable, the gap's case unqualified.
        Connection::open(&index)
            .unwrap()
            .execute_batch("DROP TABLE metal_reference_gaps;")
            .unwrap();
        assert_eq!(count(&index, "authored"), 1);
        assert!(queued(&index));
        // A sync fills it again.
        sync_index_with(
            scratch.path(),
            &index,
            &crate::source::public_source_set().unwrap(),
        )
        .unwrap();
        assert_eq!(count(&index, "reference-gap"), 1);
        assert!(!queued(&index));
    }

    #[test]
    fn status_summary_does_not_hide_failed_or_stale_slots() {
        assert_eq!(
            summarize_existing_states(&[EvidenceState::Current, EvidenceState::Stale]),
            EvidenceState::Stale
        );
        assert_eq!(
            summarize_existing_states(&[EvidenceState::Current, EvidenceState::Failed]),
            EvidenceState::Failed
        );
        assert_eq!(summarize_existing_states(&[]), EvidenceState::Missing);
    }

    /// The index keeps a write-ahead log. A rebuild makes one and never inherits a log that a
    /// stopped process left beside its destination (SQLite would apply another database's pages to
    /// it); a sync and a public-row reconciliation each switch an index an older binary built in
    /// rollback-journal mode; and `check_index` compares a copy that misses no committed page.
    #[test]
    fn the_index_keeps_a_write_ahead_log_and_never_inherits_a_stale_one() {
        let scratch = ScratchDir::new("index-write-ahead-log").unwrap();
        let mut source = public_sources().unwrap().into_iter().next().unwrap();
        source.air_ll.push_str("\n; private index fixture\n");
        source.air_sha256 = crate::hash::sha256_bytes(source.air_ll.as_bytes());
        source.label = "private/write-ahead-log".into();
        crate::source::write_source_shards(scratch.path(), [source.clone()]).unwrap();
        let index = scratch.path().join("index.sqlite");
        let journal_mode = |path: &Path| -> String {
            Connection::open(path)
                .unwrap()
                .query_row("PRAGMA journal_mode", [], |row| row.get(0))
                .unwrap()
        };
        let labelled = |path: &Path| {
            select_queue(path, QueueState::Unplanned, usize::MAX)
                .unwrap()
                .into_iter()
                .any(|row| row.air_sha256 == source.air_sha256 && row.label == source.label)
        };

        // Another database's log, holding committed pages, under the destination's log name.
        let other = scratch.path().join("other.sqlite");
        let writer = Connection::open(&other).unwrap();
        writer
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA wal_autocheckpoint=0;
                 CREATE TABLE sources (unrelated INTEGER);
                 INSERT INTO sources VALUES (1);",
            )
            .unwrap();
        let [log, _] = write_ahead_log_files(&index);
        fs::copy(&write_ahead_log_files(&other)[0], &log).unwrap();
        assert!(fs::metadata(&log).unwrap().len() > 0);
        drop(writer);
        rebuild_index(scratch.path(), &index).unwrap();
        assert_eq!(journal_mode(&index), "wal");
        assert!(labelled(&index));

        // An older binary's index, in rollback-journal mode.
        let older = |path: &Path| {
            Connection::open(path)
                .unwrap()
                .execute_batch("PRAGMA journal_mode=DELETE")
                .unwrap();
            assert_eq!(journal_mode(path), "delete");
        };
        older(&index);
        sync_index_with_stats(scratch.path(), &index).unwrap();
        assert_eq!(journal_mode(&index), "wal");
        older(&index);
        reconcile_checkout_rows(scratch.path(), &index, &public_source_set().unwrap()).unwrap();
        assert_eq!(journal_mode(&index), "wal");

        // `check_index` copies the file: a committed change still only in the log must be in it.
        // The file holds a wrong label, the log its correction; a copy without the log would
        // keep the wrong label, which a sync of unchanged shards never rewrites.
        Connection::open(&index)
            .unwrap()
            .execute(
                "UPDATE sources SET label='private/wrong' WHERE air_sha256=?1",
                [&source.air_sha256],
            )
            .unwrap();
        let held = Connection::open(&index).unwrap();
        held.execute_batch("PRAGMA wal_autocheckpoint=0").unwrap();
        held.execute(
            "UPDATE sources SET label=?1 WHERE air_sha256=?2",
            params![source.label, source.air_sha256],
        )
        .unwrap();
        assert!(fs::metadata(&log).unwrap().len() > 0);
        check_index(scratch.path(), &index).unwrap();
        drop(held);
        assert!(labelled(&index));
        let leftovers = fs::read_dir(scratch.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".check") || name.contains(".tmp"))
            .collect::<Vec<_>>();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn failed_index_rebuild_removes_its_temporary_database() {
        let scratch = ScratchDir::new("index-failure-cleanup").unwrap();
        fs::create_dir_all(scratch.path().join("cases")).unwrap();
        fs::write(scratch.path().join("cases/shard_000.jsonl"), b"{}\n").unwrap();
        let destination = scratch.path().join("index.sqlite");
        assert!(rebuild_index(scratch.path(), &destination).is_err());
        assert!(!destination
            .with_extension(format!("sqlite.{}.tmp", std::process::id()))
            .exists());
    }
}
