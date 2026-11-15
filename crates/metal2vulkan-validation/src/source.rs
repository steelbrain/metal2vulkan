use crate::hash::sha256_bytes;
use crate::jsonl::to_sorted_json_string;
use base64::Engine as _;
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const SHARD_COUNT: usize = 64;
pub const SHARD_BITS: u8 = 6;
pub const DEFAULT_CORPUS_REL: &str = "corpus";
pub const CORPUS_DIR_ENV: &str = "METAL2VULKAN_CORPUS_DIR";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceMergeStats {
    pub affected_shards: usize,
    pub inserted: usize,
    pub merged_memberships: usize,
    pub replaced: usize,
    pub duplicates: usize,
}

/// Physical I/O performed while resolving an indexed, bounded source selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexedSourceReadStats {
    pub rows: usize,
    pub source_shards_opened: usize,
    pub source_bytes_read: u64,
    /// Legacy indexes can lack byte offsets. Only selected hash-derived shards are repaired.
    pub repair_shards_scanned: usize,
    pub repair_bytes_scanned: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SourceIndexLocation {
    pub air_sha256: String,
    pub stage: String,
    pub entry: String,
    pub label: String,
    pub lib_sha256s: Vec<String>,
    pub offset: i64,
    pub length: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRow {
    pub air_sha256: String,
    pub stage: String,
    pub entry: String,
    pub air_ll: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_b64: Option<String>,
    pub lib_sha256s: Vec<String>,
    pub label: String,
}

#[derive(Deserialize)]
struct AnalysisRow {
    air_sha256: String,
    stage: String,
    entry: String,
    air_ll: String,
    lib_sha256s: Vec<String>,
    label: String,
}

/// The identity of the source row for entry `entry` of the module `air_sha256` when that module
/// declares several entry points: the SHA-256 of the sorted JSON object `{"air_sha256": ..,
/// "entry": ..}`. A module that declares one entry is its own row, keyed by its `air_sha256`
/// unchanged, so no single-entry row, case, or observation moves.
pub fn entry_row_key(air_sha256: &str, entry: &str) -> String {
    #[derive(Serialize)]
    struct EntryRow<'a> {
        air_sha256: &'a str,
        entry: &'a str,
    }
    let canonical = to_sorted_json_string(EntryRow { air_sha256, entry })
        .expect("two strings serialize to JSON");
    sha256_bytes(canonical.as_bytes())
}

impl SourceRow {
    /// The row's identity, which every index table keyed by a row holds in its `air_sha256` column:
    /// the module's `air_sha256` when the module declares one entry point, and [`entry_row_key`] of
    /// the module and this row's entry when it declares several. Code that keys, selects, or
    /// records a row goes through this, never through `air_sha256`, which is always the module's
    /// content hash.
    pub fn key(&self) -> Cow<'_, str> {
        if self.is_entry_of_several() {
            Cow::Owned(entry_row_key(&self.air_sha256, &self.entry))
        } else {
            Cow::Borrowed(&self.air_sha256)
        }
    }

    /// Whether the row is one entry of a module that declares several entry points, so its key is
    /// [`entry_row_key`] rather than its module's hash.
    pub fn is_entry_of_several(&self) -> bool {
        crate::air::entry_count_from_ll(&self.air_ll) > 1
    }

    pub fn validate(&self) -> Result<(), String> {
        let computed = sha256_bytes(self.air_ll.as_bytes());
        if self.air_sha256 != computed {
            return Err(format!(
                "source {} hash mismatch: row={} computed={computed}",
                self.label, self.air_sha256
            ));
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
        let entries = crate::air::entry_count_from_ll(&self.air_ll);
        if entries > 1 {
            return Err(format!(
                "source {} declares {entries} entry points: a stored source row is one module with \
                 one entry, so a private module with several is refused (public fixtures \
                 are a row per entry)",
                self.label
            ));
        }
        let (metadata_stage, metadata_entry) = crate::air::stage_entry_from_ll(&self.air_ll)
            .ok_or_else(|| format!("source {} has no stable AIR entry metadata", self.label))?;
        if metadata_stage != self.stage {
            return Err(format!(
                "source {} stage mismatch: row={} metadata={metadata_stage}",
                self.label, self.stage
            ));
        }
        if metadata_entry != self.entry {
            return Err(format!(
                "source {} entry mismatch: row={} metadata={metadata_entry}",
                self.label, self.entry
            ));
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
        if let Some(blob) = &self.blob_b64 {
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(blob)
                .map_err(|error| format!("source {} has invalid blob_b64: {error}", self.label))?;
            if decoded.is_empty() {
                return Err(format!("source {} has empty blob_b64", self.label));
            }
        }
        Ok(())
    }
}

pub fn corpus_root() -> PathBuf {
    std::env::var_os(CORPUS_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_CORPUS_REL))
}

pub fn shard_index_for_hash(hash: &str) -> Result<usize, String> {
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("invalid SHA-256 {hash:?}"));
    }
    let byte = u8::from_str_radix(&hash[..2], 16)
        .map_err(|error| format!("invalid SHA-256 {hash:?}: {error}"))?;
    Ok((byte >> (8 - SHARD_BITS)) as usize)
}

pub fn shard_name(index: usize) -> String {
    format!("shard_{index:03}.jsonl")
}

pub fn shard_name_for_hash(hash: &str) -> Result<String, String> {
    shard_index_for_hash(hash).map(shard_name)
}

pub fn shard_index_from_path(path: &Path) -> Result<usize, String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid shard path {}", path.display()))?;
    let digits = name
        .strip_prefix("shard_")
        .and_then(|name| name.strip_suffix(".jsonl"))
        .ok_or_else(|| format!("invalid shard filename {name:?}"))?;
    if digits.len() != 3 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("invalid shard filename {name:?}"));
    }
    let index = digits
        .parse::<usize>()
        .map_err(|error| format!("invalid shard filename {name:?}: {error}"))?;
    if index >= SHARD_COUNT {
        return Err(format!(
            "shard index {index} exceeds permanent {}-bucket mapping",
            SHARD_COUNT
        ));
    }
    Ok(index)
}

/// The corpus root whose `local/` holds this process's private data, when a binary was given one
/// with `--private-corpus` ([`use_private_corpus_root`]). Library code and tests never set it, so a
/// corpus root's private data is otherwise its own `local/`.
static PRIVATE_CORPUS_ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Read every corpus root's private data (`local/`: the source shards and library modules) from
/// `root`'s for the rest of the process: a binary's `--private-corpus DIR`, set once while it
/// parses its arguments, before it touches a corpus.
///
/// The tracked corpus (cases, reviews, observations) is a commit's data, written in that commit's
/// schema, while the private corpus is untracked and the same for every checkout. A checkout's
/// tools therefore read the tracked corpus of the checkout itself (`--corpus`, by default its own
/// `validation/corpus`) and the private sources another checkout holds from `root`, so they never
/// parse shards a newer commit wrote. Writers of private data (harvest, normalization) take no such
/// flag.
pub fn use_private_corpus_root(root: PathBuf) -> Result<(), String> {
    set_private_corpus_root("--private-corpus", root)
}

/// [`use_private_corpus_root`] for the root `source` names: refused unless it holds `local/`, and
/// set once per process.
fn set_private_corpus_root(source: &str, root: PathBuf) -> Result<(), String> {
    if !root.join("local").is_dir() {
        return Err(format!(
            "{source} {}: no local/ directory of private sources and library modules",
            root.display()
        ));
    }
    PRIVATE_CORPUS_ROOT.set(root).map_err(|root| {
        format!(
            "{source}: the private corpus root is already {}",
            root.display()
        )
    })
}

/// The environment variable that hands a process's private corpus root to the workers it
/// spawns. A worker is a new process that never parsed its parent's `--private-corpus`, so
/// [`MonitoredChild::spawn`](crate::worker::process::MonitoredChild::spawn), through which every
/// validation worker and tool starts, sets it to the parent's [`private_corpus_root`] (and clears
/// it when the parent has none), and every worker's entry adopts it
/// ([`adopt_parent_private_corpus`]). It is that handoff, not a setting: a tool run on its own
/// never reads it, so a stray export changes nothing.
pub const PRIVATE_CORPUS_ROOT_ENV: &str = "M2V_PRIVATE_CORPUS_ROOT";

/// A worker's first step, once its arguments selected worker mode: adopt the private corpus root
/// its parent handed it ([`PRIVATE_CORPUS_ROOT_ENV`]), refused as `--private-corpus` is when it
/// holds no `local/`. Without one, the worker reads each corpus root's own private data, as its
/// parent did. Every worker entry calls it: the shared harness (`worker::serve_process`), the
/// device worker, and the translation and site workers the corpus binaries serve themselves.
pub fn adopt_parent_private_corpus() -> Result<(), String> {
    match std::env::var_os(PRIVATE_CORPUS_ROOT_ENV) {
        Some(root) => set_private_corpus_root(PRIVATE_CORPUS_ROOT_ENV, PathBuf::from(root)),
        None => Ok(()),
    }
}

/// The corpus root whose `local/` holds this process's private data: its `--private-corpus`
/// ([`use_private_corpus_root`]), or, in a worker, the one its parent handed it
/// ([`adopt_parent_private_corpus`]). `None` when each corpus root's private data is its own.
pub fn private_corpus_root() -> Option<PathBuf> {
    PRIVATE_CORPUS_ROOT.get().cloned()
}

/// The directory of corpus root `root`'s private data: its own `local/`, or that of the process's
/// private corpus root ([`private_corpus_root`]).
pub fn private_data_dir(root: &Path) -> PathBuf {
    private_corpus_root()
        .unwrap_or_else(|| root.to_path_buf())
        .join("local")
}

/// Whether corpus root `root` has a private store at all: its private data directory
/// ([`private_data_dir`]) exists. A public checkout (CI's, or a worktree given no
/// `--private-corpus`) has none, so the committed reviews, cases, and observations of private AIR
/// are expected and unavailable there, never an error: its index holds only the compact rows of its
/// public sources ([`crate::index::IndexScope::PublicOnly`]). A root that has one is complete by
/// contract, so a source shard a row needs and it lacks stays an error.
pub fn private_store_present(root: &Path) -> bool {
    private_data_dir(root).is_dir()
}

pub fn source_shards_dir(root: &Path) -> PathBuf {
    private_data_dir(root).join("sources")
}

pub fn source_shard_path(root: &Path, index: usize) -> PathBuf {
    source_shards_dir(root).join(shard_name(index))
}

/// Return only hashes absent from the disposable source index.
///
/// A reharvest often rediscovers thousands of already indexed entry modules while collecting newly
/// retained non-entry dependency modules. Consulting SQLite before the shard merger prevents those
/// duplicates from reopening multi-gigabyte source buckets. A missing index conservatively returns
/// every requested hash; the canonical shard merger remains the trust boundary in that case.
pub fn unindexed_source_hashes(
    root: &Path,
    hashes: impl IntoIterator<Item = String>,
) -> Result<std::collections::HashSet<String>, String> {
    let hashes = hashes.into_iter().collect::<std::collections::HashSet<_>>();
    let index = root.join(".index.sqlite");
    if !index.is_file() || hashes.is_empty() {
        return Ok(hashes);
    }
    let connection = rusqlite::Connection::open(&index)
        .map_err(|error| format!("open source index {}: {error}", index.display()))?;
    let mut statement = connection
        .prepare("SELECT 1 FROM sources WHERE air_sha256=?1")
        .map_err(|error| format!("prepare indexed-source membership query: {error}"))?;
    let mut missing = std::collections::HashSet::new();
    for hash in hashes {
        let indexed = statement
            .query_row([&hash], |_| Ok(()))
            .optional()
            .map_err(|error| format!("query indexed source {hash}: {error}"))?
            .is_some();
        if !indexed {
            missing.insert(hash);
        }
    }
    Ok(missing)
}

/// Load the compact set of indexed entry AIR identities without opening source storage.
pub fn indexed_source_hashes(root: &Path) -> Result<std::collections::HashSet<String>, String> {
    let index = root.join(".index.sqlite");
    if !index.is_file() {
        return Ok(std::collections::HashSet::new());
    }
    let connection = rusqlite::Connection::open(&index)
        .map_err(|error| format!("open source index {}: {error}", index.display()))?;
    let mut statement = connection
        .prepare("SELECT air_sha256 FROM sources")
        .map_err(|error| format!("prepare indexed-source identity query: {error}"))?;
    let hashes = statement
        .query_map([], |row| row.get(0))
        .map_err(|error| format!("query indexed source identities: {error}"))?
        .collect::<Result<_, _>>()
        .map_err(|error| format!("read indexed source identities: {error}"))?;
    Ok(hashes)
}

/// Load indexed parent-library memberships without opening source storage.
///
/// A legacy index can return no memberships for an existing AIR identity. Harvest treats that as
/// unknown and merges the rediscovered row through its exact hash-derived shard, which fills the
/// compact membership facts without a corpus scan.
pub fn indexed_source_memberships(
    root: &Path,
) -> Result<std::collections::HashMap<String, std::collections::HashSet<String>>, String> {
    let index = root.join(".index.sqlite");
    if !index.is_file() {
        return Ok(std::collections::HashMap::new());
    }
    let connection = rusqlite::Connection::open(&index)
        .map_err(|error| format!("open source index {}: {error}", index.display()))?;
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS source_library_memberships (
               air_sha256 TEXT NOT NULL REFERENCES sources(air_sha256) ON DELETE CASCADE,
               lib_sha256 TEXT NOT NULL,
               PRIMARY KEY(air_sha256, lib_sha256)
             );",
        )
        .map_err(|error| format!("upgrade source-membership index: {error}"))?;
    let mut statement = connection
        .prepare("SELECT air_sha256, lib_sha256 FROM source_library_memberships")
        .map_err(|error| format!("prepare indexed source memberships: {error}"))?;
    let mut memberships =
        std::collections::HashMap::<String, std::collections::HashSet<String>>::new();
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| format!("query indexed source memberships: {error}"))?;
    for row in rows {
        let (air_sha256, lib_sha256) =
            row.map_err(|error| format!("read indexed source membership: {error}"))?;
        memberships
            .entry(air_sha256)
            .or_default()
            .insert(lib_sha256);
    }
    Ok(memberships)
}

pub fn write_source_shards(
    root: &Path,
    rows: impl IntoIterator<Item = SourceRow>,
) -> Result<(), String> {
    let mut buckets = vec![Vec::new(); SHARD_COUNT];
    for row in rows {
        row.validate()?;
        buckets[shard_index_for_hash(&row.air_sha256)?].push(row);
    }
    fs::create_dir_all(source_shards_dir(root))
        .map_err(|error| format!("create source shards: {error}"))?;
    remove_stale_source_temporaries(root)?;
    for (index, bucket) in buckets.iter_mut().enumerate() {
        write_source_bucket(root, index, bucket)?;
    }
    Ok(())
}

/// Merge harvested AIR into only the hash-aligned shards that receive rows.
///
/// The corpus may be several gigabytes, while one harvest batch commonly adds only a handful of
/// hashes. Reading and rewriting all 64 shards would turn the persistent index into dead weight.
/// Hash alignment makes each affected shard an independent canonical merge unit.
pub fn merge_source_shards(
    root: &Path,
    rows: impl IntoIterator<Item = SourceRow>,
) -> Result<SourceMergeStats, String> {
    let mut additions = std::collections::BTreeMap::<usize, Vec<SourceRow>>::new();
    for row in rows {
        row.validate()?;
        additions
            .entry(shard_index_for_hash(&row.air_sha256)?)
            .or_default()
            .push(row);
    }
    if additions.is_empty() {
        return Ok(SourceMergeStats::default());
    }
    fs::create_dir_all(source_shards_dir(root))
        .map_err(|error| format!("create source shards: {error}"))?;
    remove_stale_source_temporaries(root)?;

    let mut stats = SourceMergeStats {
        affected_shards: additions.len(),
        ..SourceMergeStats::default()
    };
    for (index, new_rows) in additions {
        let path = source_shard_path(root, index);
        let existing = if path.is_file() {
            read_source_shard(&path)?
        } else {
            Vec::new()
        };
        let mut by_hash = existing
            .into_iter()
            .map(|row| (row.air_sha256.clone(), row))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut changed = false;
        for mut row in new_rows {
            match by_hash.get_mut(&row.air_sha256) {
                None => {
                    stats.inserted += 1;
                    changed = true;
                    by_hash.insert(row.air_sha256.clone(), row);
                }
                Some(previous) => {
                    let before = previous.lib_sha256s.len();
                    previous.lib_sha256s.append(&mut row.lib_sha256s);
                    previous.lib_sha256s.sort();
                    previous.lib_sha256s.dedup();
                    let added_memberships = previous.lib_sha256s.len() - before;
                    stats.merged_memberships += added_memberships;
                    let blob_changed = source_blob_is_preferred(
                        previous.blob_b64.as_deref(),
                        row.blob_b64.as_deref(),
                    );
                    if blob_changed {
                        previous.blob_b64 = row.blob_b64;
                    }
                    let label_changed = row.label < previous.label;
                    if label_changed {
                        previous.label = row.label;
                    }
                    if added_memberships > 0 || blob_changed || label_changed {
                        stats.replaced += usize::from(blob_changed || label_changed);
                        changed = true;
                    } else {
                        stats.duplicates += 1;
                    }
                }
            }
        }
        if !changed {
            continue;
        }
        let mut merged = by_hash.into_values().collect::<Vec<_>>();
        let locations = write_source_bucket(root, index, &mut merged)?;
        crate::index::record_source_shard_write(root, index, &locations)?;
    }
    Ok(stats)
}

/// Return whether a candidate optional AIR blob preserves more deterministic source information.
pub fn source_blob_is_preferred(current: Option<&str>, candidate: Option<&str>) -> bool {
    match (current, candidate) {
        (None, Some(_)) => true,
        (Some(current), Some(candidate)) => candidate < current,
        (None, None) | (Some(_), None) => false,
    }
}

fn write_source_bucket(
    root: &Path,
    index: usize,
    bucket: &mut [SourceRow],
) -> Result<Vec<SourceIndexLocation>, String> {
    bucket.sort_by(|left, right| left.air_sha256.cmp(&right.air_sha256));
    let path = source_shard_path(root, index);
    let temporary =
        source_shards_dir(root).join(format!(".{}.{}.tmp", shard_name(index), std::process::id()));
    let result = (|| {
        let mut file = File::create(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?;
        let mut locations = Vec::with_capacity(bucket.len());
        let mut offset = 0u64;
        for row in bucket {
            let line = to_sorted_json_string(&*row)
                .map_err(|error| format!("serialize {}: {error}", row.air_sha256))?;
            let length = line
                .len()
                .checked_add(1)
                .ok_or_else(|| format!("source row {} length overflow", row.air_sha256))?;
            writeln!(file, "{line}")
                .map_err(|error| format!("write {}: {error}", temporary.display()))?;
            locations.push(SourceIndexLocation {
                air_sha256: row.air_sha256.clone(),
                stage: row.stage.clone(),
                entry: row.entry.clone(),
                label: row.label.clone(),
                lib_sha256s: row.lib_sha256s.clone(),
                offset: offset
                    .try_into()
                    .map_err(|_| format!("source offset in {} is too large", path.display()))?,
                length: length
                    .try_into()
                    .map_err(|_| format!("source row {} is too large", row.air_sha256))?,
            });
            offset = offset
                .checked_add(length as u64)
                .ok_or_else(|| format!("source shard {} size overflow", path.display()))?;
        }
        file.sync_all()
            .map_err(|error| format!("fsync {}: {error}", temporary.display()))?;
        fs::rename(&temporary, &path).map_err(|error| {
            format!(
                "rename {} to {}: {error}",
                temporary.display(),
                path.display()
            )
        })?;
        sync_directory(&source_shards_dir(root))?;
        Ok(locations)
    })();
    match result {
        Ok(locations) => Ok(locations),
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error)
        }
    }
}

fn remove_stale_source_temporaries(root: &Path) -> Result<(), String> {
    let directory = source_shards_dir(root);
    let entries = fs::read_dir(&directory)
        .map_err(|error| format!("read source shards {}: {error}", directory.display()))?;
    for path in entries.filter_map(Result::ok).map(|entry| entry.path()) {
        let stale = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(".shard_") && name.ends_with(".tmp"));
        if stale {
            fs::remove_file(&path)
                .map_err(|error| format!("remove stale {}: {error}", path.display()))?;
        }
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("fsync directory {}: {error}", path.display()))
}

pub fn read_source_shard(path: &Path) -> Result<Vec<SourceRow>, String> {
    let mut rows = Vec::new();
    for_each_source_shard(path, |row| {
        rows.push(row);
        Ok(())
    })?;
    Ok(rows)
}

/// Stream fully validated source rows while retaining at most one decoded row at a time.
pub fn for_each_source_shard(
    path: &Path,
    mut consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<(), String> {
    let expected_shard = shard_index_from_path(path)?;
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("read {}:{}: {error}", path.display(), index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: SourceRow = serde_json::from_str(&line)
            .map_err(|error| format!("parse {}:{}: {error}", path.display(), index + 1))?;
        row.validate()
            .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1))?;
        let actual_shard = shard_index_for_hash(&row.air_sha256)?;
        if actual_shard != expected_shard {
            return Err(format!(
                "{}:{}: source {} belongs in shard {}, not {}",
                path.display(),
                index + 1,
                row.air_sha256,
                actual_shard,
                expected_shard
            ));
        }
        consume(row)?;
    }
    Ok(())
}

/// Stream the authoring-relevant projection of a canonical harvested shard.
///
/// Harvest performs full hash, AIR metadata, and blob validation before writing canonical shards.
/// Corpus-wide analysis does not need to allocate or base64-decode each retained AIR blob again;
/// it reads the AIR text and indexed identity while Serde skips `blob_b64`. Callers process one row
/// before the reader advances, so memory is bounded by the largest source row rather than corpus
/// size. Use [`read_source_shard`] at ingestion or other trust boundaries.
pub fn for_each_source_shard_analysis(
    path: &Path,
    mut consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<(), String> {
    let expected_shard = shard_index_from_path(path)?;
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
        consume(
            analysis_source_from_json(&line, expected_shard)
                .map_err(|error| format!("parse {}:{line_number}: {error}", path.display()))?,
        )
        .map_err(|error| format!("{}:{line_number}: {error}", path.display()))?;
    }
    Ok(())
}

/// Read selected source projections through one SQLite connection and at most one open handle per
/// selected shard.
///
/// The source index supplies exact JSONL byte locations. Legacy or stale locations are repaired by
/// scanning only the hash-derived selected shards before retrying. Harvest already validated the
/// canonical rows, so analysis skips the retained binary blob and does not repeat ingestion work.
pub fn for_each_indexed_source_analysis(
    root: &Path,
    index_path: &Path,
    hashes: &[String],
    consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<(), String> {
    for_each_indexed_source_analysis_with_stats(root, index_path, hashes, consume).map(|_| ())
}

pub fn for_each_indexed_source_analysis_with_stats(
    root: &Path,
    index_path: &Path,
    hashes: &[String],
    consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<IndexedSourceReadStats, String> {
    for_each_indexed_source_analysis_in(root, index_path, &public_source_set()?, hashes, consume)
}

/// [`for_each_indexed_source_analysis_with_stats`] with `public` as the checkout's public fixtures:
/// the caller's [`public_source_set`], read once, or a synthetic set in a test.
pub fn for_each_indexed_source_analysis_in(
    root: &Path,
    index_path: &Path,
    public: &PublicSources,
    hashes: &[String],
    consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<IndexedSourceReadStats, String> {
    for_each_indexed_source_row(
        root,
        index_path,
        public,
        hashes,
        analysis_source_from_json,
        consume,
    )
}

/// Like [`for_each_indexed_source_analysis_with_stats`], but keeps each row's retained bitcode
/// blob (`blob_b64`) for analyses that decode the bitcode itself. Public fixtures carry none.
pub fn for_each_indexed_source_with_blob(
    root: &Path,
    index_path: &Path,
    hashes: &[String],
    consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<IndexedSourceReadStats, String> {
    for_each_indexed_source_row(
        root,
        index_path,
        &public_source_set()?,
        hashes,
        full_source_from_json,
        consume,
    )
}

fn for_each_indexed_source_row(
    root: &Path,
    index_path: &Path,
    public: &PublicSources,
    hashes: &[String],
    parse: fn(&[u8], usize) -> Result<SourceRow, String>,
    mut consume: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<IndexedSourceReadStats, String> {
    enum Location<'a> {
        Public(&'a SourceRow),
        Private {
            hash: String,
            shard: usize,
            offset: u64,
            length: usize,
        },
    }

    let public = public
        .rows
        .iter()
        .map(|source| (source.key(), source))
        .collect::<std::collections::HashMap<_, _>>();
    let mut locations = Vec::new();
    let mut stats = IndexedSourceReadStats::default();
    for attempt in 0..2 {
        locations.clear();
        let connection = rusqlite::Connection::open(index_path)
            .map_err(|error| format!("open index {}: {error}", index_path.display()))?;
        let mut statement = connection
            .prepare(
                "SELECT s.shard, s.source_offset, s.source_length, t.size, s.label FROM sources s \
                 LEFT JOIN indexed_source_shards t ON t.shard=s.shard WHERE s.air_sha256=?1",
            )
            .map_err(|error| format!("prepare indexed source analysis: {error}"))?;
        let mut shard_sizes = std::collections::HashMap::new();
        let mut refresh = std::collections::BTreeSet::new();
        // Rows no repair can give a source: the index's public identities this checkout's
        // fixtures do not have. No private shard holds a public row, so scanning one would only
        // cost its bytes.
        let mut stale_public = Vec::new();
        let mut missing = Vec::new();
        for hash in hashes {
            if let Some(source) = public.get(hash.as_str()) {
                locations.push(Location::Public(source));
                continue;
            }
            let expected_shard = shard_index_for_hash(hash)?;
            let location = statement
                .query_row([hash], |row| {
                    Ok((
                        row.get::<_, Option<usize>>(0)?,
                        row.get::<_, Option<i64>>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                        row.get::<_, Option<u64>>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                })
                .optional()
                .map_err(|error| format!("read indexed source {hash}: {error}"))?;
            let (shard, offset, length, indexed_size) = match location {
                Some((None, _, _, _, label)) => {
                    stale_public.push(format!(
                        "{hash} (shard {expected_shard}): the index records it as the public \
                         fixture {label}, which this checkout's public fixtures do not include (a \
                         stale public identity from another checkout; reconcile the index's \
                         public rows, which every tool that selects rows does first)"
                    ));
                    continue;
                }
                Some((Some(shard), Some(offset), Some(length), Some(size), _)) => {
                    (shard, offset, length, size)
                }
                Some(_) | None => {
                    if attempt != 0 {
                        missing.push(format!(
                            "{hash} (shard {expected_shard}): private source shard \
                             {expected_shard} ({}) does not hold it after its repair scan (a \
                             missing private source)",
                            source_shard_path(root, expected_shard).display()
                        ));
                    }
                    refresh.insert(expected_shard);
                    continue;
                }
            };
            if shard != expected_shard || offset < 0 || length <= 0 {
                refresh.insert(expected_shard);
                continue;
            }
            let current_size = *shard_sizes.entry(shard).or_insert_with(|| {
                fs::metadata(source_shard_path(root, shard))
                    .map(|metadata| metadata.len())
                    .unwrap_or(u64::MAX)
            });
            if current_size != indexed_size {
                refresh.insert(expected_shard);
                continue;
            }
            locations.push(Location::Private {
                hash: hash.clone(),
                shard,
                offset: offset as u64,
                length: length as usize,
            });
        }
        drop(statement);
        drop(connection);
        if !stale_public.is_empty() {
            return Err(format!(
                "{} selected source(s) have no source in this checkout: {}",
                stale_public.len(),
                stale_public.join("; ")
            ));
        }
        if refresh.is_empty() {
            break;
        }
        if attempt != 0 {
            return Err(format!(
                "source locations remain unavailable after repairing shards {}: {}",
                refresh
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                if missing.is_empty() {
                    "a repaired shard changed again during the read".to_string()
                } else {
                    missing.join("; ")
                }
            ));
        }
        for shard in refresh {
            stats.repair_shards_scanned += 1;
            stats.repair_bytes_scanned = stats
                .repair_bytes_scanned
                .checked_add(crate::index::index_source_shard_locations(
                    root, index_path, shard,
                )?)
                .ok_or_else(|| "source repair byte count overflow".to_string())?;
        }
    }
    if locations.len() != hashes.len() {
        return Err(format!(
            "indexed analysis resolved {} of {} requested sources",
            locations.len(),
            hashes.len()
        ));
    }

    let mut files = std::collections::HashMap::<usize, File>::new();
    for location in locations {
        let source = match location {
            Location::Public(source) => source.clone(),
            Location::Private {
                hash,
                shard,
                offset,
                length,
            } => {
                stats.source_bytes_read = stats
                    .source_bytes_read
                    .checked_add(length as u64)
                    .ok_or_else(|| "indexed source byte count overflow".to_string())?;
                let path = source_shard_path(root, shard);
                let file = match files.entry(shard) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                        File::open(&path)
                            .map_err(|error| format!("open {}: {error}", path.display()))?,
                    ),
                };
                file.seek(SeekFrom::Start(offset)).map_err(|error| {
                    format!("seek {} to source {hash}: {error}", path.display())
                })?;
                let mut bytes = vec![0; length];
                file.read_exact(&mut bytes).map_err(|error| {
                    format!(
                        "read indexed source {hash} from {}: {error}",
                        path.display()
                    )
                })?;
                let source = parse(&bytes, shard)
                    .map_err(|error| format!("parse indexed source {hash}: {error}"))?;
                if source.key() != hash {
                    return Err(format!(
                        "indexed source location for {hash} resolves to {}",
                        source.key()
                    ));
                }
                source
            }
        };
        consume(source)?;
        stats.rows += 1;
    }
    stats.source_shards_opened = files.len();
    Ok(stats)
}

fn analysis_source_from_json(bytes: &[u8], expected_shard: usize) -> Result<SourceRow, String> {
    let row: AnalysisRow =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid source JSON: {error}"))?;
    let actual_shard = shard_index_for_hash(&row.air_sha256)?;
    if actual_shard != expected_shard {
        return Err(format!(
            "source {} belongs in shard {}, not {}",
            row.air_sha256, actual_shard, expected_shard
        ));
    }
    Ok(SourceRow {
        air_sha256: row.air_sha256,
        stage: row.stage,
        entry: row.entry,
        air_ll: row.air_ll,
        blob_b64: None,
        lib_sha256s: row.lib_sha256s,
        label: row.label,
    })
}

fn full_source_from_json(bytes: &[u8], expected_shard: usize) -> Result<SourceRow, String> {
    let row: SourceRow =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid source JSON: {error}"))?;
    let actual_shard = shard_index_for_hash(&row.air_sha256)?;
    if actual_shard != expected_shard {
        return Err(format!(
            "source {} belongs in shard {}, not {}",
            row.air_sha256, actual_shard, expected_shard
        ));
    }
    Ok(row)
}

pub fn read_all_private_sources(root: &Path) -> Result<Vec<SourceRow>, String> {
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
    let mut rows = Vec::new();
    for path in paths {
        rows.extend(read_source_shard(&path)?);
    }
    let mut seen = std::collections::HashSet::new();
    for row in &rows {
        if !seen.insert(&row.air_sha256) {
            return Err(format!("duplicate private source {}", row.air_sha256));
        }
    }
    Ok(rows)
}

/// Find the row keyed `hash` ([`SourceRow::key`]) among the public fixtures and the private source
/// shards under `root`. A private row is its module, so its key is the hash that names its shard.
///
/// `Ok(None)` means the hash's source shard exists and does not hold it; a missing shard is an
/// error naming its path.
pub fn find_source(root: &Path, hash: &str) -> Result<Option<SourceRow>, String> {
    if let Some(source) = find_public_source(hash)? {
        return Ok(Some(source));
    }
    find_private_source(root, hash)
}

/// The private row keyed `hash`, read from its hash-derived shard ([`find_source`]), through this
/// process's source index when it has one ([`use_source_index`]).
fn find_private_source(root: &Path, hash: &str) -> Result<Option<SourceRow>, String> {
    find_private_source_through(root, hash, SOURCE_INDEX.get().map(PathBuf::as_path))
}

/// [`find_private_source`] through the read-only source index `source_index`, if any: the row its
/// location holds when that is the row ([`read_located_source`]), else the row the whole shard
/// holds; without one, through the corpus root's own index.
fn find_private_source_through(
    root: &Path,
    hash: &str,
    source_index: Option<&Path>,
) -> Result<Option<SourceRow>, String> {
    let index = shard_index_for_hash(hash)?;
    let located = match source_index {
        Some(source_index) => read_located_source(root, source_index, hash, index),
        None => find_indexed_private_source(root, hash, index)?,
    };
    if let Some(source) = located {
        SOURCE_READS_BY_RANGE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(Some(source));
    }
    SOURCE_READS_BY_SHARD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // A missing shard is a corpus root without this bucket's private sources (for example a
    // worktree's own `validation/corpus`), never an empty bucket: callers would report every hash
    // in it as absent.
    let path = source_shard_path(root, index);
    if !path.is_file() {
        return Err(format!(
            "source shard {} for AIR {hash} does not exist",
            path.display()
        ));
    }
    Ok(read_source_shard(&path)?
        .into_iter()
        .find(|row| row.air_sha256 == hash))
}

/// The index whose source locations this process reads private sources through, when it was
/// given one ([`use_source_index`]): read only, never repaired.
static SOURCE_INDEX: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Private sources this process found by an index's byte range, and by parsing their whole shard
/// ([`source_reads`]).
static SOURCE_READS_BY_RANGE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SOURCE_READS_BY_SHARD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How this process found the private sources it looked up ([`find_source`],
/// [`find_case_source`]): by an index's byte range, or by parsing the source's whole shard.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceReads {
    pub by_range: u64,
    pub by_shard: u64,
}

/// The private source lookups of this process so far.
pub fn source_reads() -> SourceReads {
    SourceReads {
        by_range: SOURCE_READS_BY_RANGE.load(std::sync::atomic::Ordering::Relaxed),
        by_shard: SOURCE_READS_BY_SHARD.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// Read every private source this process looks up through the source locations of the index at
/// `path` for the rest of the process, in place of its corpus root's own `.index.sqlite`: the
/// authored compile census's case worker, handed the census's `--index`, and an executor a batch
/// hands its corpus's index (`--source-index`). The index is opened read only and never repaired,
/// since many workers read it at once while another process writes it, and it is never trusted: a
/// source it does not locate, locates in a shard whose size changed since, or locates at bytes that
/// are not that source (they must parse as a valid row whose AIR text hashes to the hash looked up)
/// is read by parsing its whole shard, and [`source_reads`] counts it. A missing index file is such
/// an index. Set once, before any lookup.
pub fn use_source_index(path: PathBuf) -> Result<(), String> {
    SOURCE_INDEX
        .set(path)
        .map_err(|path| format!("the source index is already {}", path.display()))
}

/// The index this process reads private sources through ([`use_source_index`]), if it was given
/// one: an executor hands it to its device worker.
pub fn source_index() -> Option<PathBuf> {
    SOURCE_INDEX.get().cloned()
}

/// The private source `hash` at the location the index at `index_path` records for it, read only:
/// `None` when the index cannot be read, records no location in `expected_shard`, records one for
/// a shard whose size changed since, or records one whose bytes are not the source `hash` (a stale
/// or damaged location: they do not read, parse, or validate as a row, or the row's AIR text
/// hashes to another hash). The caller then parses the whole shard, so a bad index costs time,
/// never a lookup.
fn read_located_source(
    root: &Path,
    index_path: &Path,
    hash: &str,
    expected_shard: usize,
) -> Option<SourceRow> {
    let Ok(connection) = rusqlite::Connection::open_with_flags(
        index_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        return None;
    };
    let location = connection
        .query_row(
            "SELECT s.shard, s.source_offset, s.source_length, t.size \
             FROM sources s JOIN indexed_source_shards t ON t.shard=s.shard \
             WHERE s.air_sha256=?1",
            [hash],
            |row| {
                Ok((
                    row.get::<_, usize>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, u64>(3)?,
                ))
            },
        )
        .optional();
    let Ok(Some((shard, Some(offset), Some(length), indexed_size))) = location else {
        return None;
    };
    if shard != expected_shard || offset < 0 || length <= 0 {
        return None;
    }
    match fs::metadata(source_shard_path(root, shard)) {
        Ok(metadata) if metadata.len() == indexed_size => {}
        _ => return None,
    }
    // `read_source_range` validates the row it reads, hashing its AIR text, and requires that
    // hash to be `hash`: any failure there is a location that is not the source's.
    read_source_range(root, hash, shard, offset as u64, length as usize).ok()
}

/// The source row `hash` at byte `offset`, `length` bytes long, of private source shard `shard`
/// under `root`, checked to be that row.
fn read_source_range(
    root: &Path,
    hash: &str,
    shard: usize,
    offset: u64,
    length: usize,
) -> Result<SourceRow, String> {
    let path = source_shard_path(root, shard);
    let mut file =
        File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek {} to source {hash}: {error}", path.display()))?;
    let mut bytes = vec![0u8; length];
    file.read_exact(&mut bytes).map_err(|error| {
        format!(
            "read indexed source {hash} from {}: {error}",
            path.display()
        )
    })?;
    let row: SourceRow = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "parse indexed source {hash} from {}: {error}",
            path.display()
        )
    })?;
    row.validate()
        .map_err(|error| format!("indexed source {hash} from {}: {error}", path.display()))?;
    if row.air_sha256 != hash {
        return Err(format!(
            "indexed source location for {hash} resolves to {}",
            row.air_sha256
        ));
    }
    Ok(row)
}

fn find_indexed_private_source(
    root: &Path,
    hash: &str,
    expected_shard: usize,
) -> Result<Option<SourceRow>, String> {
    find_indexed_private_source_with_refresh(root, hash, expected_shard, true)
}

fn find_indexed_private_source_with_refresh(
    root: &Path,
    hash: &str,
    expected_shard: usize,
    refresh_missing_location: bool,
) -> Result<Option<SourceRow>, String> {
    let index_path = root.join(".index.sqlite");
    if !index_path.is_file() {
        return Ok(None);
    }
    let connection = match rusqlite::Connection::open(&index_path) {
        Ok(connection) => connection,
        Err(_) => return Ok(None),
    };
    let location = match connection
        .query_row(
            "SELECT s.shard, s.source_offset, s.source_length, t.size \
             FROM sources s JOIN indexed_source_shards t ON t.shard=s.shard \
             WHERE s.air_sha256=?1",
            [hash],
            |row| {
                Ok((
                    row.get::<_, usize>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, u64>(3)?,
                ))
            },
        )
        .optional()
    {
        Ok(Some(location)) => location,
        Ok(None) => {
            drop(connection);
            if refresh_missing_location {
                crate::index::index_source_shard_locations(root, &index_path, expected_shard)?;
                return find_indexed_private_source_with_refresh(root, hash, expected_shard, false);
            }
            return Ok(None);
        }
        Err(_) => return Ok(None),
    };
    let (shard, offset, length, indexed_size) = location;
    let (Some(offset), Some(length)) = (offset, length) else {
        drop(connection);
        if refresh_missing_location {
            crate::index::index_source_shard_locations(root, &index_path, expected_shard)?;
            return find_indexed_private_source_with_refresh(root, hash, expected_shard, false);
        }
        return Ok(None);
    };
    if shard != expected_shard || offset < 0 || length <= 0 {
        return Ok(None);
    }
    let current_size = match fs::metadata(source_shard_path(root, shard)) {
        Ok(metadata) => metadata.len(),
        Err(_) => return Ok(None),
    };
    if current_size != indexed_size {
        drop(connection);
        if refresh_missing_location {
            crate::index::index_source_shard_locations(root, &index_path, expected_shard)?;
            return find_indexed_private_source_with_refresh(root, hash, expected_shard, false);
        }
        return Ok(None);
    }
    read_source_range(root, hash, shard, offset as u64, length as usize).map(Some)
}

/// The directories under [`PUBLIC_FIXTURE_DIR`] whose `.ll` files are public source rows:
/// the top level (hand-written and compiled fixtures) and `engine/` (owned MSL compiled by
/// `scripts/regen-air-fixtures`). Each is enumerated without descending, so a subdirectory such as
/// `engine/companions/` holds no rows.
const PUBLIC_SOURCE_DIRS: [&str; 2] = ["", "engine"];

/// The repository-relative directory of the public fixtures: the prefix of an authored case's
/// `public_source`.
pub const PUBLIC_FIXTURE_DIR: &str = "crates/metal2vulkan-validation/fixtures/public";

/// The repository-relative path (`crates/metal2vulkan-validation/fixtures/public/...`) of the public module labelled
/// `label` (`public/...`); `None` for a label that is not a public module's.
pub fn public_source_path(label: &str) -> Option<String> {
    label
        .strip_prefix("public/")
        .map(|rest| format!("{PUBLIC_FIXTURE_DIR}/{rest}"))
}

/// The file Metal compiles the public module labelled `label` from on a Mac (`public/...`, the
/// oracle's `shader_air`): its typed AIR text, `<name>.air.ll`, which `metal-as` assembles, or
/// else its Metal source, `<name>.metal`, which `metal -c` compiles; and whether it is the typed
/// text. `None` for a label that is not a public module's.
pub fn public_metal_input(label: &str) -> Option<(PathBuf, bool)> {
    let name = label.strip_prefix("public/")?;
    let root = metal2vulkan_devtools::fixtures_dir();
    let typed = root.join(name).with_extension("air.ll");
    Some(if typed.exists() {
        (typed, true)
    } else {
        (root.join(name).with_extension("metal"), false)
    })
}

/// Whether the public module at repository-relative `path` (an authored case's `public_source`)
/// is the AIR `air_sha256`: `path` must name a module the public enumeration reads (a `.ll`, not
/// an `.air.ll`, directly in one of its directories), and its text must hash to `air_sha256`.
/// Otherwise the error names the path and why, so an edited fixture whose case was not re-keyed
/// is named by the fixture.
pub fn check_public_source(path: &str, air_sha256: &str) -> Result<(), String> {
    let relative = path
        .strip_prefix(PUBLIC_FIXTURE_DIR)
        .and_then(|rest| rest.strip_prefix('/'))
        .ok_or_else(|| format!("public_source {path} is not below {PUBLIC_FIXTURE_DIR}"))?;
    let (directory, name) = relative.rsplit_once('/').unwrap_or(("", relative));
    if !PUBLIC_SOURCE_DIRS.contains(&directory)
        || !name.ends_with(".ll")
        || name.ends_with(".air.ll")
    {
        return Err(format!(
            "public_source {path} names no public fixture module (a `.ll` directly in \
             {PUBLIC_FIXTURE_DIR} or its engine/)"
        ));
    }
    let file = metal2vulkan_devtools::fixtures_dir().join(relative);
    let text = fs::read(&file).map_err(|error| {
        format!(
            "public_source {path} does not exist: read {}: {error}",
            file.display()
        )
    })?;
    let actual = sha256_bytes(&text);
    if actual != air_sha256 {
        return Err(format!(
            "public_source {path} hashes to {actual}, not the case's AIR {air_sha256}: the \
             fixture changed without its case being re-keyed"
        ));
    }
    Ok(())
}

/// The public fixtures' source rows: one per entry point of each module, so a module that declares
/// several is several rows, each keyed by its entry ([`SourceRow::key`]).
#[derive(Clone, Debug, Default)]
pub struct PublicSources {
    pub rows: Vec<SourceRow>,
}

impl PublicSources {
    /// The key ([`SourceRow::key`]) of the row for entry `entry` of the module `air_sha256`, the
    /// identity an authored case over that entry is indexed under: [`entry_row_key`] when the
    /// module is a public fixture that declares several entries, and `air_sha256` otherwise (a
    /// private module is one row).
    pub fn key_of(&self, air_sha256: &str, entry: &str) -> String {
        match self.rows.iter().find(|row| row.air_sha256 == air_sha256) {
            Some(row) if row.is_entry_of_several() => entry_row_key(air_sha256, entry),
            _ => air_sha256.to_string(),
        }
    }

    /// The repository-relative path ([`public_source_path`]) of the public module whose text
    /// hashes to `air_sha256`, whichever of its entries' rows names it.
    pub fn path_of(&self, air_sha256: &str) -> Option<String> {
        self.rows
            .iter()
            .find(|row| row.air_sha256 == air_sha256)
            .and_then(|row| public_source_path(&row.label))
    }

    /// The report `corpus-index` and the authoring-capability census print: the rows and the
    /// modules they come from.
    pub fn report_lines(&self) -> Vec<String> {
        let modules = self
            .rows
            .iter()
            .map(|row| row.air_sha256.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
        vec![format!(
            "public-sources\trows={}\tmodules={modules}",
            self.rows.len()
        )]
    }
}

/// Every public fixture that is a source row, sorted by path. A row's label is `public/` followed
/// by its path below [`PUBLIC_FIXTURE_DIR`] (`public/engine/<name>.ll` for an engine
/// fixture), which is where the Metal oracle finds its `.air.ll` or `.metal` half.
pub fn public_sources() -> Result<Vec<SourceRow>, String> {
    Ok(public_source_set()?.rows)
}

/// The public source rows, one per entry point of each public module ([`PublicSources`]), of the
/// fixtures where the test runner names the checkout ([`metal2vulkan_devtools::fixtures_dir`]).
pub fn public_source_set() -> Result<PublicSources, String> {
    public_sources_under(&metal2vulkan_devtools::fixtures_dir())
}

pub(crate) fn public_sources_under(root: &Path) -> Result<PublicSources, String> {
    let mut paths = Vec::new();
    for directory in PUBLIC_SOURCE_DIRS.map(|relative| root.join(relative)) {
        paths.extend(
            fs::read_dir(&directory)
                .map_err(|error| format!("read {}: {error}", directory.display()))?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|extension| extension == "ll"))
                // `<name>.air.ll` is the typed-pointer Metal half of a fixture (the dialect `xcrun
                // metal-as` assembles): hand-written AIR no MSL can spell, or an engine
                // fixture's compiled text. The translator reads `<name>.ll`; this one is not a
                // source of its own.
                .filter(|path| {
                    !path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(".air.ll"))
                }),
        );
    }
    paths.sort();
    let mut sources = PublicSources::default();
    for path in paths {
        let air_ll = fs::read_to_string(&path)
            .map_err(|error| format!("read public source {}: {error}", path.display()))?;
        let label = public_label(root, &path)?;
        sources
            .rows
            .extend(public_module_rows(&path, label, air_ll)?);
    }
    Ok(sources)
}

/// The rows of the public module at `path`, labelled `label`, with text `air_ll`: one per entry
/// of a module that declares several ([`public_entry_rows`]), else its single row.
fn public_module_rows(
    path: &Path,
    label: String,
    air_ll: String,
) -> Result<Vec<SourceRow>, String> {
    if crate::air::entry_count_from_ll(&air_ll) > 1 {
        public_entry_rows(path, label, air_ll)
    } else {
        Ok(vec![public_source_row(path, label, air_ll)?])
    }
}

/// The row of entry `entry` of the public module at `path`, labelled `label`, with text `air_ll`,
/// built as [`public_source_set`] builds it; `None` when the module declares no such entry.
pub fn public_entry_row(
    path: &Path,
    label: String,
    air_ll: String,
    entry: &str,
) -> Result<Option<SourceRow>, String> {
    Ok(public_module_rows(path, label, air_ll)?
        .into_iter()
        .find(|row| row.entry == entry))
}

/// The rows of a public module that declares several entry points: one per entry, in the order
/// [`crate::air::module_entries`] lists them, each with the whole module's text and hash and keyed
/// by its entry ([`SourceRow::key`]).
fn public_entry_rows(path: &Path, label: String, air_ll: String) -> Result<Vec<SourceRow>, String> {
    let air_sha256 = sha256_bytes(air_ll.as_bytes());
    Ok(crate::air::module_entries(&air_ll)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .into_iter()
        .map(|entry| SourceRow {
            air_sha256: air_sha256.clone(),
            stage: entry.stage.into(),
            entry: entry.name,
            air_ll: air_ll.clone(),
            blob_b64: None,
            lib_sha256s: vec!["owned-synthetic".into()],
            label: label.clone(),
        })
        .collect())
}

fn find_public_source(hash: &str) -> Result<Option<SourceRow>, String> {
    Ok(public_sources()?.into_iter().find(|row| row.key() == hash))
}

/// [`public_source_set`], read once per process. The committed fixtures do not change while a
/// tool runs, and a loop that checks cases ([`crate::check::check_case`], each of which finds its
/// sources here) would otherwise read and hash every public fixture once per case.
fn committed_public_sources() -> Result<&'static PublicSources, String> {
    static SOURCES: std::sync::OnceLock<Result<PublicSources, String>> = std::sync::OnceLock::new();
    SOURCES
        .get_or_init(public_source_set)
        .as_ref()
        .map_err(Clone::clone)
}

/// The source row an authored case over entry `entry` of the module `air_sha256` runs: the row
/// keyed [`PublicSources::key_of`] the two ([`find_source`]).
pub fn find_case_source(
    root: &Path,
    air_sha256: &str,
    entry: &str,
) -> Result<Option<SourceRow>, String> {
    let public = committed_public_sources()?;
    let key = public.key_of(air_sha256, entry);
    if let Some(row) = public.rows.iter().find(|row| row.key() == key) {
        return Ok(Some(row.clone()));
    }
    // A public module is read from the checkout: an entry it does not declare has no row, and no
    // private shard holds one.
    let declared = public
        .rows
        .iter()
        .filter(|row| row.air_sha256 == air_sha256)
        .map(|row| row.entry.as_str())
        .collect::<Vec<_>>();
    if let Some(row) = public.rows.iter().find(|row| row.air_sha256 == air_sha256) {
        return Err(format!(
            "the public module {} ({air_sha256}) declares no entry {entry:?}; it declares \
             {declared:?}",
            row.label
        ));
    }
    find_private_source(root, &key)
}

/// `public/` plus `path`'s components below `root`, joined by `/`.
fn public_label(root: &Path, path: &Path) -> Result<String, String> {
    let relative = path
        .strip_prefix(root)
        .ok()
        .and_then(|relative| {
            relative
                .components()
                .map(|component| component.as_os_str().to_str())
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| {
            format!(
                "{} is not a UTF-8 path below {}",
                path.display(),
                root.display()
            )
        })?;
    Ok(format!("public/{}", relative.join("/")))
}

fn public_source_row(path: &Path, label: String, air_ll: String) -> Result<SourceRow, String> {
    let (stage, entry) = crate::air::stage_entry_from_ll(&air_ll)
        .ok_or_else(|| format!("{} has no stable AIR entry metadata", path.display()))?;
    Ok(SourceRow {
        air_sha256: sha256_bytes(air_ll.as_bytes()),
        stage: stage.into(),
        entry,
        air_ll,
        blob_b64: None,
        lib_sha256s: vec!["owned-synthetic".into()],
        label,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScratchDir;

    /// A case's `public_source` is checked against the fixture it names: the path of a public
    /// module whose text hashes to the case's AIR passes, and every other path or hash is refused
    /// naming the path.
    #[test]
    fn a_public_source_must_name_a_public_module_with_the_cases_air() {
        let set = public_source_set().unwrap();
        let row = set
            .rows
            .iter()
            .find(|row| row.label == "public/kernel_copy_word.ll")
            .expect("the copy_word fixture");
        let path = "crates/metal2vulkan-validation/fixtures/public/kernel_copy_word.ll";
        assert_eq!(set.path_of(&row.air_sha256).as_deref(), Some(path));
        assert_eq!(check_public_source(path, &row.air_sha256), Ok(()));
        let stale = check_public_source(path, &"00".repeat(32)).unwrap_err();
        assert!(
            stale.contains(path) && stale.contains("re-keyed"),
            "{stale}"
        );
        for bad in [
            "crates/metal2vulkan-validation/fixtures/public/engine/companions/x.ll",
            "crates/metal2vulkan-validation/fixtures/public/kernel_copy_word.air.ll",
            "crates/metal2vulkan-validation/fixtures/public/missing_fixture.ll",
            "crates/metal2vulkan-validation/fixtures/kernel_copy_word.ll",
            // The path before the crates/ layout.
            "validation/fixtures/public/kernel_copy_word.ll",
            "fixtures/public/kernel_copy_word.ll",
        ] {
            let error = check_public_source(bad, &row.air_sha256).unwrap_err();
            assert!(error.contains(bad), "{error}");
        }
    }

    #[test]
    fn leading_six_bits_select_stable_aligned_shards() {
        assert_eq!(
            shard_index_for_hash(&format!("00{}", "0".repeat(62))).unwrap(),
            0
        );
        assert_eq!(
            shard_index_for_hash(&format!("03{}", "0".repeat(62))).unwrap(),
            0
        );
        assert_eq!(
            shard_index_for_hash(&format!("04{}", "0".repeat(62))).unwrap(),
            1
        );
        assert_eq!(
            shard_index_for_hash(&format!("ff{}", "0".repeat(62))).unwrap(),
            63
        );
        assert_eq!(shard_name(7), "shard_007.jsonl");
    }

    #[test]
    fn source_memberships_require_the_canonical_array_schema() {
        let row = serde_json::json!({
            "air_ll": "", "air_sha256": "00".repeat(32), "entry": "main",
            "label": "schema/test", "lib_sha256s": ["11".repeat(32)], "stage": "Kernel"
        });
        assert!(serde_json::from_value::<SourceRow>(row.clone()).is_ok());
        assert!(serde_json::from_value::<AnalysisRow>(row.clone()).is_ok());
        for key in ["lib_sha256", "lib_sha256s"] {
            let mut invalid = row.clone();
            invalid.as_object_mut().unwrap().remove("lib_sha256s");
            invalid[key] = serde_json::json!("11".repeat(32));
            assert!(serde_json::from_value::<SourceRow>(invalid.clone()).is_err());
            assert!(serde_json::from_value::<AnalysisRow>(invalid).is_err());
        }
    }

    #[test]
    fn analysis_reader_skips_blob_decoding_and_streams_air_projection() {
        let scratch = ScratchDir::new("source-analysis-reader").unwrap();
        let directory = source_shards_dir(scratch.path());
        fs::create_dir_all(&directory).unwrap();
        let path = source_shard_path(scratch.path(), 0);
        let row = serde_json::json!({
            "air_ll": "define void @main() { ret void }",
            "air_sha256": "00".repeat(32),
            "blob_b64": "deliberately not base64",
            "entry": "main",
            "label": "analysis/test",
            "lib_sha256s": ["11".repeat(32)],
            "stage": "Kernel"
        });
        fs::write(
            &path,
            format!("{}\n", crate::jsonl::to_sorted_json_string(&row).unwrap()),
        )
        .unwrap();
        let mut seen = Vec::new();
        for_each_source_shard_analysis(&path, |source| {
            seen.push(source);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].air_ll, "define void @main() { ret void }");
        assert_eq!(seen[0].blob_b64, None);
    }

    #[test]
    fn source_shard_output_is_byte_deterministic() {
        let scratch = ScratchDir::new("source-shards").unwrap();
        let ll = "define void @main() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @main}";
        let row = SourceRow {
            air_sha256: sha256_bytes(ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: ll.into(),
            blob_b64: Some("YmxvYg==".into()),
            lib_sha256s: vec!["22".repeat(32)],
            label: "local/test.ll".into(),
        };
        write_source_shards(scratch.path(), [row.clone()]).unwrap();
        let first = fs::read_dir(source_shards_dir(scratch.path()))
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| (entry.file_name(), fs::read(entry.path()).unwrap()))
            .collect::<Vec<_>>();
        fs::write(
            source_shards_dir(scratch.path()).join(".shard_000.jsonl.interrupted.tmp"),
            b"partial",
        )
        .unwrap();
        write_source_shards(scratch.path(), [row]).unwrap();
        let second = fs::read_dir(source_shards_dir(scratch.path()))
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| (entry.file_name(), fs::read(entry.path()).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(first, second);
    }

    #[test]
    fn incremental_merge_never_opens_or_rewrites_an_unrelated_shard() {
        let scratch = ScratchDir::new("source-shard-merge").unwrap();
        let mut selected = public_sources().unwrap().remove(0);
        selected.air_ll.push_str("\n; selected private source\n");
        selected.air_sha256 = sha256_bytes(selected.air_ll.as_bytes());
        selected.lib_sha256s = vec!["22".repeat(32)];
        selected.label = "local/selected.ll".into();
        let selected_shard = shard_index_for_hash(&selected.air_sha256).unwrap();
        let mut unrelated = selected.clone();
        for nonce in 0..u32::MAX {
            unrelated.air_ll = format!("{}\n; unrelated {nonce}\n", selected.air_ll);
            unrelated.air_sha256 = sha256_bytes(unrelated.air_ll.as_bytes());
            if shard_index_for_hash(&unrelated.air_sha256).unwrap() != selected_shard {
                break;
            }
        }
        unrelated.label = "local/unrelated.ll".into();
        let unrelated_shard = shard_index_for_hash(&unrelated.air_sha256).unwrap();
        write_source_shards(scratch.path(), [selected.clone(), unrelated]).unwrap();

        let unrelated_path = source_shard_path(scratch.path(), unrelated_shard);
        let unrelated_bytes = fs::read(&unrelated_path).unwrap();
        let unreadable_canonical_data = vec![b'x'; unrelated_bytes.len()];
        fs::write(&unrelated_path, &unreadable_canonical_data).unwrap();

        selected.lib_sha256s = vec!["11".repeat(32)];
        let stats = merge_source_shards(scratch.path(), [selected.clone()]).unwrap();
        assert_eq!(
            stats,
            SourceMergeStats {
                affected_shards: 1,
                inserted: 0,
                merged_memberships: 1,
                replaced: 0,
                duplicates: 0,
            }
        );
        assert_eq!(
            fs::read(&unrelated_path).unwrap(),
            unreadable_canonical_data
        );
        assert_eq!(
            read_source_shard(&source_shard_path(scratch.path(), selected_shard))
                .unwrap()
                .remove(0)
                .lib_sha256s,
            vec!["11".repeat(32), "22".repeat(32)]
        );
    }

    #[test]
    fn indexed_membership_filters_reharvest_before_source_shards_are_opened() {
        let scratch = ScratchDir::new("source-index-membership").unwrap();
        let index = scratch.path().join(".index.sqlite");
        let connection = rusqlite::Connection::open(&index).unwrap();
        connection
            .execute_batch("CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY);")
            .unwrap();
        let known = "11".repeat(32);
        let missing = "22".repeat(32);
        connection
            .execute("INSERT INTO sources VALUES (?1)", [&known])
            .unwrap();
        drop(connection);

        assert_eq!(
            unindexed_source_hashes(scratch.path(), [known, missing.clone()]).unwrap(),
            [missing].into_iter().collect()
        );
        assert_eq!(
            indexed_source_hashes(scratch.path()).unwrap(),
            ["11".repeat(32)].into_iter().collect()
        );
    }

    #[test]
    fn indexed_parent_libraries_are_compact_and_legacy_safe() {
        let scratch = ScratchDir::new("source-library-memberships").unwrap();
        let index = scratch.path().join(".index.sqlite");
        let hash = "11".repeat(32);
        let connection = rusqlite::Connection::open(&index).unwrap();
        connection
            .execute_batch("CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY);")
            .unwrap();
        connection
            .execute("INSERT INTO sources VALUES (?1)", [&hash])
            .unwrap();
        drop(connection);

        assert!(indexed_source_memberships(scratch.path())
            .unwrap()
            .is_empty());
        let connection = rusqlite::Connection::open(&index).unwrap();
        connection
            .execute(
                "INSERT INTO source_library_memberships VALUES (?1, ?2)",
                rusqlite::params![hash, "22".repeat(32)],
            )
            .unwrap();
        drop(connection);
        assert_eq!(
            indexed_source_memberships(scratch.path()).unwrap()[&hash],
            ["22".repeat(32)].into_iter().collect()
        );
    }

    #[test]
    fn duplicate_merge_does_not_replace_an_unchanged_shard_file() {
        use std::os::unix::fs::MetadataExt as _;

        let scratch = ScratchDir::new("source-duplicate-no-rewrite").unwrap();
        let mut row = public_sources().unwrap().remove(0);
        row.air_ll.push_str("\n; duplicate no-rewrite fixture\n");
        row.air_sha256 = sha256_bytes(row.air_ll.as_bytes());
        row.lib_sha256s = vec!["11".repeat(32)];
        row.label = "local/duplicate.ll".into();
        write_source_shards(scratch.path(), [row.clone()]).unwrap();
        let path = source_shard_path(
            scratch.path(),
            shard_index_for_hash(&row.air_sha256).unwrap(),
        );
        let inode = fs::metadata(&path).unwrap().ino();

        let stats = merge_source_shards(scratch.path(), [row]).unwrap();
        assert_eq!(stats.inserted, 0);
        assert_eq!(stats.replaced, 0);
        assert_eq!(stats.duplicates, 1);
        assert_eq!(fs::metadata(path).unwrap().ino(), inode);
    }

    #[test]
    fn duplicate_merge_preserves_an_available_original_blob() {
        let scratch = ScratchDir::new("source-blob-preservation").unwrap();
        let mut row = public_sources().unwrap().remove(0);
        row.air_ll
            .push_str("\n; source blob preservation fixture\n");
        row.air_sha256 = sha256_bytes(row.air_ll.as_bytes());
        row.lib_sha256s = vec!["11".repeat(32)];
        row.label = "local/blob.ll".into();
        row.blob_b64 = None;
        write_source_shards(scratch.path(), [row.clone()]).unwrap();

        row.blob_b64 = Some("YmxvYg==".into());
        let stats = merge_source_shards(scratch.path(), [row.clone()]).unwrap();
        assert_eq!(stats.replaced, 1);
        assert_eq!(
            read_source_shard(&source_shard_path(
                scratch.path(),
                shard_index_for_hash(&row.air_sha256).unwrap(),
            ))
            .unwrap()
            .remove(0)
            .blob_b64,
            row.blob_b64
        );
    }

    #[test]
    fn public_source_lookup_survives_an_unrelated_private_bucket_file() {
        let scratch = ScratchDir::new("public-source-fallback").unwrap();
        let public = public_sources().unwrap().remove(0);
        let path = source_shard_path(
            scratch.path(),
            shard_index_for_hash(&public.air_sha256).unwrap(),
        );
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"").unwrap();
        assert_eq!(
            find_source(scratch.path(), &public.air_sha256)
                .unwrap()
                .unwrap(),
            public
        );
    }

    /// The census's source index (`use_source_index`) is read only: a located source is read by
    /// its byte range, and a source it cannot locate (no index, no row, a shard whose size changed)
    /// is left to the whole-shard read, with the index unchanged.
    #[test]
    fn a_source_index_is_read_by_range_and_never_repaired() {
        let scratch = ScratchDir::new("source-index-read-only").unwrap();
        let mut source = public_sources().unwrap().remove(0);
        source.air_ll.push_str("\n; private by-range fixture\n");
        source.air_sha256 = sha256_bytes(source.air_ll.as_bytes());
        source.lib_sha256s = vec!["owned-synthetic".into()];
        source.label = "private/by-range".into();
        write_source_shards(scratch.path(), [source.clone()]).unwrap();
        let shard = shard_index_for_hash(&source.air_sha256).unwrap();
        let index = scratch.path().join("census.sqlite");
        let read = || read_located_source(scratch.path(), &index, &source.air_sha256, shard);

        // No index file, then an index without the source tables: nothing located, nothing made.
        assert_eq!(read(), None);
        assert!(!index.exists());
        rusqlite::Connection::open(&index)
            .unwrap()
            .execute_batch("CREATE TABLE authored_compile_census (case_id TEXT)")
            .unwrap();
        assert_eq!(read(), None);

        crate::index::rebuild_index(scratch.path(), &index).unwrap();
        assert_eq!(read(), Some(source.clone()));
        // Another hash in the same shard is not located there.
        assert_eq!(
            read_located_source(scratch.path(), &index, &"0".repeat(64), shard),
            None
        );

        // The shard grew since the index recorded it: not read by range, and not repaired.
        let path = source_shard_path(scratch.path(), shard);
        let mut grown = fs::read(&path).unwrap();
        grown.extend_from_slice(b"\n");
        fs::write(&path, grown).unwrap();
        let recorded = || {
            rusqlite::Connection::open(&index)
                .unwrap()
                .query_row(
                    "SELECT size FROM indexed_source_shards WHERE shard=?1",
                    [shard],
                    |row| row.get::<_, u64>(0),
                )
                .unwrap()
        };
        let before = recorded();
        assert_eq!(read(), None);
        assert_eq!(recorded(), before);
    }

    /// A source index is never trusted: a missing index file, and a stale location (the shard at
    /// its recorded size, but another valid row at the recorded range) each fall back to the
    /// whole-shard read, which finds the row; neither fails the lookup, and neither index is
    /// written. A fresh location is read by its range.
    #[test]
    fn a_source_index_that_is_wrong_or_missing_falls_back_to_the_shard() {
        let scratch = ScratchDir::new("source-index-fallback").unwrap();
        let base = public_sources().unwrap().remove(0);
        let variant = |n: usize| {
            let mut source = base.clone();
            // Rows of equal length: their shard lines can trade places at the same size.
            source
                .air_ll
                .push_str(&format!("\n; private fallback fixture {n:04}\n"));
            source.air_sha256 = sha256_bytes(source.air_ll.as_bytes());
            source.lib_sha256s = vec!["owned-synthetic".into()];
            source.label = format!("private/fallback-{n:04}");
            source
        };
        // Two variants whose hashes select one shard.
        let mut by_shard = std::collections::BTreeMap::<usize, SourceRow>::new();
        let (first, second) = (0..1000)
            .find_map(|n| {
                let source = variant(n);
                let shard = shard_index_for_hash(&source.air_sha256).unwrap();
                by_shard
                    .insert(shard, source.clone())
                    .map(|earlier| (earlier, source))
            })
            .unwrap();
        write_source_shards(scratch.path(), [first.clone(), second.clone()]).unwrap();
        let shard = shard_index_for_hash(&first.air_sha256).unwrap();
        let index = scratch.path().join("batch.sqlite");
        crate::index::rebuild_index(scratch.path(), &index).unwrap();
        let find = |index: &Path, source: &SourceRow| {
            find_private_source_through(scratch.path(), &source.air_sha256, Some(index)).unwrap()
        };
        for source in [&first, &second] {
            assert_eq!(
                read_located_source(scratch.path(), &index, &source.air_sha256, shard).as_ref(),
                Some(source)
            );
            assert_eq!(find(&index, source).as_ref(), Some(source));
        }

        // A missing index file: the shard is read, and no index is made.
        let missing = scratch.path().join("no-such-index.sqlite");
        assert_eq!(find(&missing, &first), Some(first.clone()));
        assert!(!missing.exists());

        // A stale index: the shard's two lines trade places, so it keeps its recorded size while
        // each recorded range now holds the other row. Neither location is its source's, and the
        // lookups read the shard.
        let path = source_shard_path(scratch.path(), shard);
        let text = fs::read_to_string(&path).unwrap();
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].len(), lines[1].len());
        let swapped = format!("{}\n{}\n", lines[1], lines[0]);
        assert_eq!(swapped.len(), text.len());
        fs::write(&path, &swapped).unwrap();
        let before = fs::read(&index).unwrap();
        for source in [&first, &second] {
            assert_eq!(
                read_located_source(scratch.path(), &index, &source.air_sha256, shard),
                None,
                "a range holding another row is no location of {}",
                source.label
            );
            assert_eq!(find(&index, source).as_ref(), Some(source));
        }
        assert_eq!(
            fs::read(&index).unwrap(),
            before,
            "a read-only index is never repaired"
        );
    }

    #[test]
    fn private_source_lookup_names_a_missing_source_shard() {
        let scratch = ScratchDir::new("missing-source-shard").unwrap();
        let hash = "22".repeat(32);
        let path = source_shard_path(scratch.path(), shard_index_for_hash(&hash).unwrap());
        let error = find_source(scratch.path(), &hash).unwrap_err();
        assert!(
            error.contains(&path.display().to_string()) && error.contains(&hash),
            "{error}"
        );

        // An existing shard without the hash is a plain miss.
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"").unwrap();
        assert_eq!(find_source(scratch.path(), &hash).unwrap(), None);
    }

    #[test]
    fn source_reader_rejects_a_misaligned_row() {
        let scratch = ScratchDir::new("misaligned-source").unwrap();
        let ll = "define void @main() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @main}";
        let row = SourceRow {
            air_sha256: sha256_bytes(ll.as_bytes()),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: ll.into(),
            blob_b64: None,
            lib_sha256s: vec!["22".repeat(32)],
            label: "local/test.ll".into(),
        };
        let actual = shard_index_for_hash(&row.air_sha256).unwrap();
        let wrong = (actual + 1) % SHARD_COUNT;
        let path = scratch.path().join(shard_name(wrong));
        fs::write(&path, format!("{}\n", to_sorted_json_string(row).unwrap())).unwrap();
        assert!(read_source_shard(&path)
            .unwrap_err()
            .contains("belongs in shard"));
    }

    /// A row's key: a module that declares one entry is its own row under its content hash,
    /// unchanged, and each entry of a module that declares several is keyed by the module and
    /// entry, the SHA-256 of their sorted JSON. A case finds its row's key from its module and
    /// entry.
    #[test]
    fn a_row_is_keyed_by_its_entry_only_when_its_module_declares_several() {
        let single = "define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @k}\n";
        let pair = "define void @a() { ret void }\ndefine void @b() { ret void }\n\
                    !air.kernel = !{!0, !1}\n!0 = !{ptr @a}\n!1 = !{ptr @b}\n";
        let row = |ll: &str, entry: &str| SourceRow {
            air_sha256: sha256_bytes(ll.as_bytes()),
            stage: "Kernel".into(),
            entry: entry.into(),
            air_ll: ll.into(),
            blob_b64: None,
            lib_sha256s: vec!["owned-synthetic".into()],
            label: "public/synthetic.ll".into(),
        };
        let single_row = row(single, "k");
        assert!(!single_row.is_entry_of_several());
        assert!(matches!(single_row.key(), Cow::Borrowed(key) if key == single_row.air_sha256));

        let module = sha256_bytes(pair.as_bytes());
        let (a, b) = (row(pair, "a"), row(pair, "b"));
        assert!(a.is_entry_of_several() && b.is_entry_of_several());
        let spelled = |entry: &str| {
            sha256_bytes(
                format!("{{\"air_sha256\":\"{module}\",\"entry\":\"{entry}\"}}").as_bytes(),
            )
        };
        assert_eq!(a.key(), spelled("a"));
        assert_eq!(b.key(), spelled("b"));
        assert_eq!(entry_row_key(&module, "a"), spelled("a"));
        assert_ne!(a.key(), module);

        let public = PublicSources {
            rows: vec![single_row.clone(), a.clone(), b.clone()],
        };
        assert_eq!(
            public.key_of(&single_row.air_sha256, "k"),
            single_row.air_sha256
        );
        assert_eq!(public.key_of(&module, "a"), a.key());
        assert_eq!(public.key_of(&module, "b"), b.key());
        // A module no public fixture declares is a private row: its own key.
        let private = "11".repeat(32);
        assert_eq!(public.key_of(&private, "k"), private);
    }

    /// A case's row is found by its module and entry; an entry its public module does not
    /// declare is an error naming the module, read from no shard (the scratch root has none).
    #[test]
    fn a_case_finds_its_entry_row_and_an_undeclared_entry_is_named() {
        let scratch = ScratchDir::new("find-case-source").unwrap();
        let public = public_source_set().unwrap();
        let row = public
            .rows
            .iter()
            .find(|row| {
                row.label == "public/engine/sampling_completion.ll"
                    && row.entry == "sample_pixel_zero"
            })
            .unwrap();
        let found = find_case_source(scratch.path(), &row.air_sha256, "sample_pixel_zero")
            .unwrap()
            .unwrap();
        assert_eq!(
            (found.entry.as_str(), found.key()),
            ("sample_pixel_zero", row.key())
        );
        let error = find_case_source(scratch.path(), &row.air_sha256, "no_such_entry").unwrap_err();
        assert!(
            error.contains("sampling_completion.ll") && error.contains("declares no entry"),
            "{error}"
        );
    }

    #[test]
    fn public_sources_are_the_top_level_and_engine_ll_files_labelled_by_path() {
        let scratch = ScratchDir::new("public-source-dirs").unwrap();
        let root = scratch.path();
        let kernel = |name: &str| {
            format!("define void @{name}() {{ ret void }}\n!air.kernel = !{{!0}}\n!0 = !{{ptr @{name}}}\n")
        };
        let two_stages = "define void @v() { ret void }\ndefine void @f() { ret void }\n\
                          !air.vertex = !{!0}\n!air.fragment = !{!1}\n\
                          !0 = !{ptr @v}\n!1 = !{ptr @f}\n";
        for (path, text) in [
            ("top.ll", kernel("top")),
            ("top.air.ll", kernel("typed_top")),
            ("top.metal", String::new()),
            ("engine/owned.ll", kernel("owned")),
            ("engine/owned.air.ll", kernel("typed_owned")),
            ("engine/owned.metal", String::new()),
            ("engine/pair.ll", two_stages.to_string()),
            ("engine/companions/linked.ll", kernel("linked")),
            ("elsewhere/other.ll", kernel("other")),
        ] {
            let path = root.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
        let sources = public_sources_under(root).unwrap();
        let labels = sources
            .rows
            .iter()
            .map(|row| (row.label.as_str(), row.entry.as_str()))
            .collect::<Vec<_>>();
        // Neither directory is descended into, typed-pointer halves are not rows, the label names
        // the row's own path, and a module with two entries (here a vertex and a fragment) is a
        // row per entry, in the order its entry lists declare them.
        assert_eq!(
            labels,
            [
                ("public/engine/owned.ll", "owned"),
                ("public/engine/pair.ll", "v"),
                ("public/engine/pair.ll", "f"),
                ("public/top.ll", "top")
            ]
        );
        assert_eq!(
            sources.rows[0].air_sha256,
            sha256_bytes(kernel("owned").as_bytes())
        );
        // A single-entry module's row is keyed by its text's hash, unchanged.
        assert_eq!(sources.rows[0].key(), sources.rows[0].air_sha256);
        // Each entry row carries the whole module and its hash, its own stage, and its own key.
        let pair = sha256_bytes(two_stages.as_bytes());
        for (row, (stage, entry)) in sources.rows[1..3]
            .iter()
            .zip([("Vertex", "v"), ("Fragment", "f")])
        {
            assert_eq!(
                (
                    row.air_sha256.as_str(),
                    row.air_ll.as_str(),
                    row.stage.as_str()
                ),
                (pair.as_str(), two_stages, stage)
            );
            assert_eq!(row.key(), entry_row_key(&pair, entry));
            assert_eq!(sources.key_of(&pair, entry), row.key());
            // A stored row is one module with one entry: an entry row is never written to a
            // shard.
            assert!(row
                .validate()
                .unwrap_err()
                .contains("declares 2 entry points"));
        }
        assert_eq!(
            sources.path_of(&pair).as_deref(),
            Some("crates/metal2vulkan-validation/fixtures/public/engine/pair.ll")
        );
        assert_eq!(
            sources.report_lines(),
            ["public-sources\trows=4\tmodules=3"]
        );
    }

    /// The public rows of each committed module are exactly the entries the product's typed decode
    /// finds in it, so the text scanner the enumeration uses lists no entry the product lacks and
    /// misses none it has.
    #[test]
    fn every_committed_public_module_is_a_row_per_entry_the_product_decodes() {
        use metal2vulkan_devtools::{llvm_assemble, AssembleError};
        let meter = || metal2vulkan_air::budget::Meter::new(&metal2vulkan::Budget::default());
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/public");
        let sources = public_source_set().unwrap();
        let (mut modules, mut compared, mut undecoded) = (0, 0, Vec::new());
        for directory in PUBLIC_SOURCE_DIRS.map(|relative| root.join(relative)) {
            for path in fs::read_dir(&directory).unwrap().map(|e| e.unwrap().path()) {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                if !name.ends_with(".ll") || name.ends_with(".air.ll") {
                    continue;
                }
                modules += 1;
                let label = public_label(&root, &path).unwrap();
                // The product's typed decode is the reference for a module's entries, not the
                // text scanner the enumeration uses.
                let entries = match llvm_assemble(&path) {
                    Ok(bitcode) => metal2vulkan_air::decode(&bitcode, &mut meter())
                        .map(|decoded| {
                            let mut names = decoded
                                .abi
                                .entries
                                .iter()
                                .map(|entry| entry.name.to_string())
                                .collect::<Vec<_>>();
                            names.sort();
                            names
                        })
                        .map_err(|error| error.to_string()),
                    Err(AssembleError::Rejected(why)) => Err(why),
                    Err(AssembleError::Missing(why)) => panic!("llvm-as: {why}"),
                };
                let Ok(entries) = entries else {
                    undecoded.push(label);
                    continue;
                };
                compared += 1;
                let mut rows = sources
                    .rows
                    .iter()
                    .filter(|row| row.label == label)
                    .map(|row| row.entry.clone())
                    .collect::<Vec<_>>();
                rows.sort();
                assert_eq!(rows, entries, "{label}");
            }
        }
        let labels = sources
            .rows
            .iter()
            .map(|row| row.label.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(labels.len(), modules);
        // Only hand-written top-level fixtures may fail to assemble or decode (the two the compiled
        // ones replaced); every engine fixture is compiled by Metal and must decode.
        assert!(
            undecoded
                .iter()
                .all(|label| !label.starts_with("public/engine/")),
            "{undecoded:?}"
        );
        assert!(compared > modules / 2, "{compared} of {modules} compared");
    }

    #[test]
    fn every_engine_source_row_has_the_typed_half_the_metal_oracle_loads() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/public");
        let rows = public_sources().unwrap();
        let engine = rows
            .iter()
            .filter_map(|row| row.label.strip_prefix("public/engine/"))
            .collect::<Vec<_>>();
        assert!(!engine.is_empty(), "engine/ fixtures are indexed");
        for name in engine {
            let ll = root.join("engine").join(name);
            assert!(ll.is_file(), "{} is a fixture", ll.display());
            // `scripts/regen-air-fixtures` writes it; the oracle maps the label to it.
            let typed = ll.with_extension("air.ll");
            assert!(typed.is_file(), "{} has its typed half", ll.display());
        }
    }
}
