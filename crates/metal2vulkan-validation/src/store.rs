use crate::case::{AuthoredCase, OutputSelection};
use crate::jsonl::to_sorted_json_string;
use crate::observation::{CandidateObservation, MetalObservation};
use crate::refusal::RefusalObservation;
use crate::source::{shard_index_for_hash, shard_index_from_path, shard_name};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct CorpusStore {
    root: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PutCaseResult {
    Added,
    Replaced {
        old_case_id: String,
    },
    /// The slot already held this identity: only fields outside it (name excepted: the rationale,
    /// the no-effect declaration, the author, the public source) were rewritten, and its
    /// observations stand.
    Annotated,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransactionManifest {
    entries: Vec<TransactionEntry>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransactionEntry {
    staged: String,
    target: String,
    #[serde(default)]
    delete: bool,
}

impl CorpusStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn recover_transactions(&self) -> Result<(), String> {
        let transactions = self.root.join(".transactions");
        let Ok(entries) = fs::read_dir(&transactions) else {
            return Ok(());
        };
        let mut paths = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            let manifest_path = path.join("manifest.json");
            if !manifest_path.is_file() {
                fs::remove_dir_all(&path)
                    .map_err(|error| format!("remove incomplete {}: {error}", path.display()))?;
                continue;
            }
            let manifest: TransactionManifest = serde_json::from_slice(
                &fs::read(&manifest_path)
                    .map_err(|error| format!("read {}: {error}", manifest_path.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
            apply_transaction(&path, &self.root, &manifest)?;
        }
        Ok(())
    }

    pub fn read_all_cases(&self) -> Result<Vec<AuthoredCase>, String> {
        let mut rows = Vec::new();
        for path in shard_paths(&self.root.join("cases"))? {
            let index = shard_index_from_path(&path)?;
            let shard_rows = read_jsonl_if_exists(&path)?;
            validate_cases(&shard_rows, Some(index))?;
            rows.extend(shard_rows);
        }
        validate_cases(&rows, None)?;
        Ok(rows)
    }

    /// Every place a committed shard leaves the layout this store writes: in the case, Metal and
    /// Vulkan shards, a row in a shard its AIR's hash does not select, or a
    /// row whose key (`ShardRow::order_key`) does not strictly follow the previous row's. The
    /// store, the one writer of these shards, sorts the shard it writes by that key and the key is one slot, so a shard that
    /// passes is in the one order the store would write it in, with no slot twice. A tool that
    /// appends rows instead of installing them through the store is what this catches.
    pub fn layout_errors(&self) -> Result<Vec<String>, String> {
        let mut errors = Vec::new();
        shard_layout_errors::<AuthoredCase>(&self.root, "cases", &mut errors)?;
        shard_layout_errors::<MetalObservation>(&self.root, "observations/metal", &mut errors)?;
        shard_layout_errors::<CandidateObservation>(&self.root, CANDIDATE_DIRECTORY, &mut errors)?;
        shard_layout_errors::<RefusalObservation>(&self.root, REFUSAL_DIRECTORY, &mut errors)?;
        Ok(errors)
    }

    /// The paths of the case shards the store holds, in shard order.
    pub fn case_shard_paths(&self) -> Result<Vec<PathBuf>, String> {
        shard_paths(&self.root.join("cases"))
    }

    pub fn read_case_shard(&self, index: usize) -> Result<Vec<AuthoredCase>, String> {
        Ok(self
            .read_stored_case_shard(index)?
            .into_iter()
            .map(|stored| stored.row)
            .collect())
    }

    /// [`Self::read_case_shard`], keeping each case's line for a writer ([`Stored`]).
    fn read_stored_case_shard(&self, index: usize) -> Result<Vec<Stored<AuthoredCase>>, String> {
        let rows = read_stored_if_exists(&self.root.join("cases").join(shard_name(index)))?;
        validate_cases(rows_of(&rows), Some(index))?;
        Ok(rows)
    }

    /// Read authored cases for one AIR identity from only its hash-derived case shard.
    pub fn find_cases_for_air(&self, air_sha256: &str) -> Result<Vec<AuthoredCase>, String> {
        let shard = shard_index_for_hash(air_sha256)?;
        Ok(self
            .read_case_shard(shard)?
            .into_iter()
            .filter(|case| case.air_sha256 == air_sha256)
            .collect())
    }

    /// The authored cases that run the source row `source`: those over its module and its entry,
    /// read from only the module's hash-derived case shard.
    pub fn find_cases_for_row(
        &self,
        source: &crate::source::SourceRow,
    ) -> Result<Vec<AuthoredCase>, String> {
        let mut cases = self.find_cases_for_air(&source.air_sha256)?;
        cases.retain(|case| case.entry == source.entry);
        Ok(cases)
    }

    /// The case `case_id` among every case of the store: each shard is read and validated, and the
    /// whole set too (identities unique across shards). A caller that knows the case's AIR reads
    /// only its shard ([`Self::find_case_of_air`]).
    pub fn find_case(&self, case_id: &str) -> Result<Option<AuthoredCase>, String> {
        Ok(self
            .read_all_cases()?
            .into_iter()
            .find(|case| case.case_id == case_id))
    }

    /// The case `case_id` read from only the case shard `air_sha256` selects, that shard validated
    /// as [`Self::read_case_shard`] validates it: the exact lookup of a caller that knows the
    /// case's AIR, such as a batch whose plan holds it. `None` when the shard holds no such case;
    /// an error when it holds one over another AIR than `air_sha256`.
    pub fn find_case_of_air(
        &self,
        case_id: &str,
        air_sha256: &str,
    ) -> Result<Option<AuthoredCase>, String> {
        let Some(case) = self
            .read_case_shard(shard_index_for_hash(air_sha256)?)?
            .into_iter()
            .find(|case| case.case_id == case_id)
        else {
            return Ok(None);
        };
        if case.air_sha256 != air_sha256 {
            return Err(format!(
                "case {case_id} is over AIR {}, not the given {air_sha256}",
                case.air_sha256
            ));
        }
        Ok(Some(case))
    }

    pub fn put_case(&self, case: AuthoredCase) -> Result<PutCaseResult, String> {
        self.recover_transactions()?;
        let computed = case.computed_case_id()?;
        if case.case_id != computed {
            return Err(format!(
                "case {} case_id mismatch: manifest={} computed={computed}",
                case.name, case.case_id
            ));
        }
        case.validate_literal_resources()
            .map_err(|errors| errors.join("; "))?;
        let declaration = crate::no_effect::declaration_errors(&case);
        if !declaration.is_empty() {
            return Err(format!("case {}: {}", case.name, declaration.join("; ")));
        }
        let index = shard_index_for_hash(&case.air_sha256)?;
        let mut cases = self.read_stored_case_shard(index)?;
        if let Some(same) = cases.iter_mut().find(|row| row.row.case_id == case.case_id) {
            // The same identity in the same slot is an annotation of it; under another name it
            // would be the same semantics twice.
            if same.row.name != case.name {
                return Err(format!("duplicate semantic case_id {}", case.case_id));
            }
            *same = Stored::new(case);
            sort_stored(&mut cases);
            // Nothing else in the bucket changes: only the case shard is written.
            let case_path = PathBuf::from("cases").join(shard_name(index));
            self.commit_files([(case_path, stored_jsonl(&cases)?)])?;
            return Ok(PutCaseResult::Annotated);
        }
        // A case's slot is its row and its name: its module, its entry, and its name, so a
        // same-name case over another entry of the module is another slot.
        let old = cases
            .iter()
            .position(|stored| {
                stored.row.air_sha256 == case.air_sha256
                    && stored.row.entry == case.entry
                    && stored.row.name == case.name
            })
            .map(|position| cases.remove(position).row);
        let old_case_id = old.as_ref().map(|row| row.case_id.as_str());
        cases.push(Stored::new(case));
        self.rewrite_aligned(index, &mut cases, old_case_id)?;
        Ok(match old {
            Some(old) => PutCaseResult::Replaced {
                old_case_id: old.case_id,
            },
            None => PutCaseResult::Added,
        })
    }

    /// Delete the case named `name` over the module `air_sha256` and, when given, its `entry` (a
    /// case's slot is its module, entry, and name), with only that identity's observations. Without
    /// `entry`, the name must select one case: a name the module's cases use over several entries
    /// is an error naming them.
    pub fn delete_named_case(
        &self,
        air_sha256: &str,
        entry: Option<&str>,
        name: &str,
    ) -> Result<String, String> {
        self.recover_transactions()?;
        let index = shard_index_for_hash(air_sha256)?;
        let mut cases = self.read_stored_case_shard(index)?;
        let matches = cases
            .iter()
            .map(|stored| &stored.row)
            .enumerate()
            .filter(|(_, row)| {
                row.air_sha256 == air_sha256
                    && row.name == name
                    && entry.is_none_or(|entry| row.entry == entry)
            })
            .map(|(position, row)| (position, row.entry.clone()))
            .collect::<Vec<_>>();
        let position = match matches.as_slice() {
            [] => {
                return Err(match entry {
                    Some(entry) => {
                        format!("no case named {name:?} for AIR {air_sha256} entry {entry:?}")
                    }
                    None => format!("no case named {name:?} for AIR {air_sha256}"),
                })
            }
            [(position, _)] => *position,
            several => {
                return Err(format!(
                    "case name {name:?} for AIR {air_sha256} names cases over the entries {:?}; \
                     name the entry",
                    several.iter().map(|(_, entry)| entry).collect::<Vec<_>>()
                ))
            }
        };
        let old = cases.remove(position).row;
        self.rewrite_aligned(index, &mut cases, Some(&old.case_id))?;
        Ok(old.case_id)
    }

    /// Write bucket `index`: `cases`, and its observation shards less the rows of
    /// `cascaded_case_id`. Every shard is written in its key order ([`sort_stored`]), including
    /// rows a tool left out of order, and every row it keeps as it was read ([`Stored`]).
    fn rewrite_aligned(
        &self,
        index: usize,
        cases: &mut [Stored<AuthoredCase>],
        cascaded_case_id: Option<&str>,
    ) -> Result<(), String> {
        let name = shard_name(index);
        let case_path = PathBuf::from("cases").join(&name);
        let metal_path = PathBuf::from("observations/metal").join(&name);
        let vulkan_path = PathBuf::from(CANDIDATE_DIRECTORY).join(&name);
        let mut metal: Vec<Stored<MetalObservation>> =
            read_stored_if_exists(&self.root.join(&metal_path))?;
        let mut vulkan: Vec<Stored<CandidateObservation>> =
            read_stored_if_exists(&self.root.join(&vulkan_path))?;
        validate_aligned_observation_rows(rows_of(&metal), index)?;
        validate_aligned_observation_rows(rows_of(&vulkan), index)?;
        reject_duplicate_metal_slots(rows_of(&metal))?;
        reject_duplicate_candidate_slots(rows_of(&vulkan))?;
        if let Some(case_id) = cascaded_case_id {
            metal.retain(|stored| stored.row.case_id != case_id);
            vulkan.retain(|stored| stored.row.case_id != case_id);
        }
        sort_stored(cases);
        sort_stored(&mut metal);
        sort_stored(&mut vulkan);
        let mut files = vec![
            (case_path, stored_jsonl(cases)?),
            (metal_path, stored_jsonl(&metal)?),
            (vulkan_path, stored_jsonl(&vulkan)?),
        ];
        let mut deletions = Vec::new();
        // A refusal follows its case as a row does; a refusal shard the bucket does not have is
        // not created.
        let path = PathBuf::from(REFUSAL_DIRECTORY).join(&name);
        if self.root.join(&path).exists() {
            let mut refusals = self.read_refusal_shard(&path, index)?;
            if let Some(case_id) = cascaded_case_id {
                refusals.retain(|stored| stored.row.case_id != case_id);
            }
            sort_stored(&mut refusals);
            match refusals.is_empty() {
                true => deletions.push(path),
                false => files.push((path, stored_jsonl(&refusals)?)),
            }
        }
        self.commit_changes(files, deletions)
    }

    /// The refusal shard `relative` (shard `index` of the refusals), validated.
    fn read_refusal_shard(
        &self,
        relative: &Path,
        index: usize,
    ) -> Result<Vec<Stored<RefusalObservation>>, String> {
        let rows: Vec<Stored<RefusalObservation>> =
            read_stored_if_exists(&self.root.join(relative))?;
        validate_aligned_observation_rows(rows_of(&rows), index)?;
        reject_duplicate_refusal_slots(rows_of(&rows))?;
        Ok(rows)
    }

    pub fn upsert_metal(&self, row: MetalObservation) -> Result<(), String> {
        self.recover_transactions()?;
        row.validate_content()?;
        require_timing("Metal", &row.case_id, &row.environment_id, row.timing)?;
        let index = shard_index_for_hash(&row.air_sha256)?;
        self.validate_observation_case(
            index,
            &row.case_id,
            &row.air_sha256,
            &row.input_sha256,
            row.observed_output(),
        )?;
        let relative = PathBuf::from("observations/metal").join(shard_name(index));
        let mut rows: Vec<Stored<MetalObservation>> =
            read_stored_if_exists(&self.root.join(&relative))?;
        validate_aligned_observation_rows(rows_of(&rows), index)?;
        reject_duplicate_metal_slots(rows_of(&rows))?;
        rows.retain(|old| {
            !(old.row.case_id == row.case_id && old.row.environment_id == row.environment_id)
        });
        rows.push(Stored::new(row));
        sort_stored(&mut rows);
        self.commit_files([(relative, stored_jsonl(&rows)?)])
    }

    pub fn upsert_candidate(&self, row: CandidateObservation) -> Result<(), String> {
        self.recover_transactions()?;
        row.validate_content()?;
        require_timing("candidate", &row.case_id, &row.environment_id, row.timing)?;
        let index = shard_index_for_hash(&row.air_sha256)?;
        self.validate_observation_case(
            index,
            &row.case_id,
            &row.air_sha256,
            &row.input_sha256,
            Some(&row.output_b64),
        )?;
        let relative = PathBuf::from(CANDIDATE_DIRECTORY).join(shard_name(index));
        let mut rows: Vec<Stored<CandidateObservation>> =
            read_stored_if_exists(&self.root.join(&relative))?;
        validate_aligned_observation_rows(rows_of(&rows), index)?;
        reject_duplicate_candidate_slots(rows_of(&rows))?;
        rows.retain(|old| {
            !(old.row.case_id == row.case_id && old.row.environment_id == row.environment_id)
        });
        // A slot holds a row or a refusal, never both: the row replaces the slot's refusal.
        let refusal_path = PathBuf::from(REFUSAL_DIRECTORY).join(shard_name(index));
        let mut refusals = self.read_refusal_shard(&refusal_path, index)?;
        let refused = refusals.len();
        refusals.retain(|old| {
            !(old.row.case_id == row.case_id && old.row.environment_id == row.environment_id)
        });
        let mut deletions = Vec::new();
        rows.push(Stored::new(row));
        sort_stored(&mut rows);
        let mut files = vec![(relative, stored_jsonl(&rows)?)];
        if refusals.len() != refused {
            match refusals.is_empty() {
                true => deletions.push(refusal_path),
                false => files.push((refusal_path, stored_jsonl(&refusals)?)),
            }
        }
        self.commit_changes(files, deletions)
    }

    /// Store `refusal` in its slot, replacing the slot's refusal or row: a slot holds the last
    /// run's outcome, a row or a refusal.
    pub fn upsert_refusal(&self, refusal: RefusalObservation) -> Result<(), String> {
        self.recover_transactions()?;
        refusal.validate_content()?;
        require_timing(
            "refusal",
            &refusal.case_id,
            &refusal.environment_id,
            refusal.timing,
        )?;
        let index = shard_index_for_hash(&refusal.air_sha256)?;
        self.validate_observation_identity(
            index,
            &refusal.case_id,
            &refusal.air_sha256,
            &refusal.input_sha256,
        )?;
        let refusal_path = PathBuf::from(REFUSAL_DIRECTORY).join(shard_name(index));
        let mut refusals = self.read_refusal_shard(&refusal_path, index)?;
        refusals.retain(|old| {
            !(old.row.case_id == refusal.case_id
                && old.row.environment_id == refusal.environment_id)
        });
        let row_path = PathBuf::from(CANDIDATE_DIRECTORY).join(shard_name(index));
        let mut rows: Vec<Stored<CandidateObservation>> =
            read_stored_if_exists(&self.root.join(&row_path))?;
        validate_aligned_observation_rows(rows_of(&rows), index)?;
        reject_duplicate_candidate_slots(rows_of(&rows))?;
        let stored = rows.len();
        rows.retain(|old| {
            !(old.row.case_id == refusal.case_id
                && old.row.environment_id == refusal.environment_id)
        });
        refusals.push(Stored::new(refusal));
        sort_stored(&mut refusals);
        let mut files = vec![(refusal_path, stored_jsonl(&refusals)?)];
        if rows.len() != stored {
            files.push((row_path, stored_jsonl(&rows)?));
        }
        self.commit_changes(files, Vec::new())
    }

    /// Rewrite every venue row and refusal that recorded a rekey's `from` to
    /// record its `to` ([`crate::c3::VenueRekey`], which the caller has checked), in one
    /// recoverable transaction. Every other line of every shard keeps its bytes; a rewritten one is
    /// serialized as the store writes rows. Returns how many rows and refusals each rekey rewrote.
    /// A second run with the same list rewrites nothing.
    pub fn rekey_venue_records(
        &self,
        rekeys: &[crate::c3::VenueRekey],
    ) -> Result<Vec<(usize, usize)>, String> {
        self.recover_transactions()?;
        let mut counts = vec![(0, 0); rekeys.len()];
        let mut files = Vec::new();
        let rows_dir = PathBuf::from(CANDIDATE_DIRECTORY);
        for path in shard_paths(&self.root.join(&rows_dir))? {
            let mut rows: Vec<Stored<CandidateObservation>> = read_stored_if_exists(&path)?;
            let rewritten = rekey_stored(&mut rows, rekeys, |row| &mut row.environment);
            if rewritten.iter().any(|&n| n > 0) {
                for (count, n) in counts.iter_mut().zip(&rewritten) {
                    count.0 += n;
                }
                files.push((
                    rows_dir.join(path.file_name().unwrap_or_default()),
                    stored_jsonl(&rows)?,
                ));
            }
        }
        let refusals_dir = PathBuf::from(REFUSAL_DIRECTORY);
        for path in shard_paths(&self.root.join(&refusals_dir))? {
            let mut refusals: Vec<Stored<RefusalObservation>> = read_stored_if_exists(&path)?;
            let rewritten = rekey_stored(&mut refusals, rekeys, |row| &mut row.environment);
            if rewritten.iter().any(|&n| n > 0) {
                for (count, n) in counts.iter_mut().zip(&rewritten) {
                    count.1 += n;
                }
                files.push((
                    refusals_dir.join(path.file_name().unwrap_or_default()),
                    stored_jsonl(&refusals)?,
                ));
            }
        }
        if !files.is_empty() {
            self.commit_changes(files, Vec::new())?;
        }
        Ok(counts)
    }

    /// Every refusal, each of a known case with its AIR and inputs, no slot twice.
    pub fn read_refusals(&self) -> Result<Vec<RefusalObservation>, String> {
        let rows: Vec<RefusalObservation> =
            read_aligned_observations(&self.root.join(REFUSAL_DIRECTORY))?;
        reject_duplicate_refusal_slots(&rows)?;
        let cases = self.read_all_cases()?;
        let cases = cases
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect::<std::collections::HashMap<_, _>>();
        for row in &rows {
            let case = cases
                .get(row.case_id.as_str())
                .ok_or_else(|| format!("refusal references unknown case_id {}", row.case_id))?;
            if case.air_sha256 != row.air_sha256 {
                return Err(format!(
                    "refusal AIR {} does not match case AIR {}",
                    row.air_sha256, case.air_sha256
                ));
            }
        }
        Ok(rows)
    }

    /// An observation's case, AIR, and inputs, and its output against the case's selection;
    /// `output_b64` is `None` for a row that records no output (a Metal reference gap).
    fn validate_observation_case(
        &self,
        index: usize,
        case_id: &str,
        air_sha256: &str,
        input_sha256: &str,
        output_b64: Option<&String>,
    ) -> Result<(), String> {
        let case = self.validate_observation_identity(index, case_id, air_sha256, input_sha256)?;
        output_b64.map_or(Ok(()), |output| validate_observation_output(&case, output))
    }

    /// The case `case_id` of bucket `index`, whose AIR and inputs an observation names.
    fn validate_observation_identity(
        &self,
        index: usize,
        case_id: &str,
        air_sha256: &str,
        input_sha256: &str,
    ) -> Result<AuthoredCase, String> {
        let case = self
            .read_case_shard(index)?
            .into_iter()
            .find(|case| case.case_id == case_id)
            .ok_or_else(|| format!("observation references unknown case_id {case_id}"))?;
        if case.air_sha256 != air_sha256 {
            return Err(format!(
                "observation AIR {air_sha256} does not match case AIR {}",
                case.air_sha256
            ));
        }
        let expected = case.computed_input_sha256()?;
        if expected != input_sha256 {
            return Err(format!(
                "observation input {input_sha256} does not match case input {expected}"
            ));
        }
        Ok(case)
    }

    pub fn read_metal(&self) -> Result<Vec<MetalObservation>, String> {
        let rows: Vec<MetalObservation> =
            read_aligned_observations(&self.root.join("observations/metal"))?;
        validate_observations_of(
            &self.read_all_cases()?,
            rows.iter().map(|row| {
                (
                    &row.case_id,
                    &row.air_sha256,
                    &row.input_sha256,
                    row.observed_output(),
                )
            }),
        )?;
        reject_duplicate_metal_slots(&rows)?;
        Ok(rows)
    }

    /// The Metal rows of the shard `case`'s AIR selects, where every row of `case` is: that shard
    /// read and validated as [`Self::read_metal`] validates every shard, each row against the cases
    /// of its case shard. The rows `MetalObservation::current` reads for `case` are exactly those
    /// [`Self::read_metal`] would give it.
    pub fn read_metal_of_case(&self, case: &AuthoredCase) -> Result<Vec<MetalObservation>, String> {
        let index = shard_index_for_hash(&case.air_sha256)?;
        let rows: Vec<MetalObservation> =
            read_jsonl_if_exists(&self.root.join("observations/metal").join(shard_name(index)))?;
        validate_aligned_observation_rows(&rows, index)?;
        validate_observations_of(
            &self.read_case_shard(index)?,
            rows.iter().map(|row| {
                (
                    &row.case_id,
                    &row.air_sha256,
                    &row.input_sha256,
                    row.observed_output(),
                )
            }),
        )?;
        reject_duplicate_metal_slots(&rows)?;
        Ok(rows)
    }

    pub fn read_candidates(&self) -> Result<Vec<CandidateObservation>, String> {
        let rows: Vec<CandidateObservation> =
            read_aligned_observations(&self.root.join(CANDIDATE_DIRECTORY))?;
        for row in &rows {
            row.validate_content()?;
        }
        self.validate_observation_cases(rows.iter().map(|row| {
            (
                &row.case_id,
                &row.air_sha256,
                &row.input_sha256,
                Some(&row.output_b64),
            )
        }))?;
        reject_duplicate_candidate_slots(&rows)?;
        Ok(rows)
    }

    /// [`Self::validate_observation_case`] of every row, reading the case shards once.
    fn validate_observation_cases<'a>(
        &self,
        rows: impl IntoIterator<Item = (&'a String, &'a String, &'a String, Option<&'a String>)>,
    ) -> Result<(), String> {
        validate_observations_of(&self.read_all_cases()?, rows)
    }

    fn commit_files<const N: usize>(&self, files: [(PathBuf, Vec<u8>); N]) -> Result<(), String> {
        self.commit_changes(files.into(), Vec::new())
    }

    /// Write `files` and delete `deletions` (relative corpus paths) in one recoverable transaction.
    fn commit_changes(
        &self,
        files: Vec<(PathBuf, Vec<u8>)>,
        deletions: Vec<PathBuf>,
    ) -> Result<(), String> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock: {error}"))?
            .as_nanos();
        let transaction = self
            .root
            .join(".transactions")
            .join(format!("{}-{nonce}", std::process::id()));
        fs::create_dir_all(&transaction)
            .map_err(|error| format!("create {}: {error}", transaction.display()))?;
        let mut entries = Vec::new();
        for (index, (target, bytes)) in files.into_iter().enumerate() {
            let staged = format!("file-{index:03}");
            let path = transaction.join(&staged);
            let mut file = File::create(&path)
                .map_err(|error| format!("create {}: {error}", path.display()))?;
            file.write_all(&bytes)
                .map_err(|error| format!("write {}: {error}", path.display()))?;
            file.sync_all()
                .map_err(|error| format!("fsync {}: {error}", path.display()))?;
            entries.push(TransactionEntry {
                staged,
                target: target
                    .to_str()
                    .ok_or_else(|| format!("non-UTF-8 target {}", target.display()))?
                    .into(),
                delete: false,
            });
        }
        for target in deletions {
            entries.push(TransactionEntry {
                staged: String::new(),
                target: target
                    .to_str()
                    .ok_or_else(|| format!("non-UTF-8 target {}", target.display()))?
                    .into(),
                delete: true,
            });
        }
        let manifest = TransactionManifest { entries };
        let manifest_path = transaction.join("manifest.json");
        let manifest_bytes = serde_json::to_vec(&manifest)
            .map_err(|error| format!("serialize transaction: {error}"))?;
        let mut file = File::create(&manifest_path)
            .map_err(|error| format!("create {}: {error}", manifest_path.display()))?;
        file.write_all(&manifest_bytes)
            .map_err(|error| format!("write {}: {error}", manifest_path.display()))?;
        file.sync_all()
            .map_err(|error| format!("fsync {}: {error}", manifest_path.display()))?;
        sync_directory(&transaction)?;
        apply_transaction(&transaction, &self.root, &manifest)
    }
}

fn apply_transaction(
    transaction: &Path,
    root: &Path,
    manifest: &TransactionManifest,
) -> Result<(), String> {
    for entry in &manifest.entries {
        let target = root.join(&entry.target);
        if entry.delete {
            if target.exists() {
                fs::remove_file(&target)
                    .map_err(|error| format!("delete {}: {error}", target.display()))?;
                if let Some(parent) = target.parent() {
                    sync_directory(parent)?;
                }
            }
            continue;
        }
        let staged = transaction.join(&entry.staged);
        if !staged.exists() {
            if target.exists() {
                continue;
            }
            return Err(format!(
                "transaction {} lost both staged and target file {}",
                transaction.display(),
                target.display()
            ));
        }
        let parent = target
            .parent()
            .ok_or_else(|| format!("target {} has no parent", target.display()))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
        fs::rename(&staged, &target).map_err(|error| {
            format!(
                "rename {} to {}: {error}",
                staged.display(),
                target.display()
            )
        })?;
        sync_directory(parent)?;
    }
    fs::remove_dir_all(transaction)
        .map_err(|error| format!("remove {}: {error}", transaction.display()))?;
    if let Some(parent) = transaction.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), String> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|error| format!("open directory {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("fsync directory {}: {error}", path.display()))
}

/// A shard row and the line it was read from. A store writer writes every row it leaves untouched
/// back as that line and serializes only the rows it adds, so a rewrite never re-spells a row it
/// did not change. Re-serializing would: the schema reads fields it does not keep (an
/// executor-v34 row's `containment.device_addresses`, which `coverage::ContainmentRecord` reads
/// and drops), and a default can be spelled otherwise than the row was written.
struct Stored<T> {
    row: T,
    /// `None` for a row added since the shard was read.
    line: Option<String>,
}

impl<T> Stored<T> {
    /// A row to add: it is serialized when its shard is written.
    fn new(row: T) -> Self {
        Stored { row, line: None }
    }
}

/// Apply the first rekey that matches each stored record's environment (the list's keys are
/// distinct, so at most one does), dropping its read line so the shard serializes it anew; how
/// many records each rekey rewrote.
fn rekey_stored<T>(
    stored: &mut [Stored<T>],
    rekeys: &[crate::c3::VenueRekey],
    environment: impl Fn(&mut T) -> &mut serde_json::Value,
) -> Vec<usize> {
    let mut counts = vec![0; rekeys.len()];
    for record in stored.iter_mut() {
        let env = environment(&mut record.row);
        if let Some(i) = rekeys.iter().position(|rekey| rekey.matches(env)) {
            rekeys[i].apply(env);
            record.line = None;
            counts[i] += 1;
        }
    }
    counts
}

/// The rows of `stored`, in order.
fn rows_of<T>(stored: &[Stored<T>]) -> impl Iterator<Item = &T> {
    stored.iter().map(|stored| &stored.row)
}

/// A shard's bytes: each row as the line it was read from, or serialized when it was added
/// ([`Stored`]).
fn stored_jsonl<T: Serialize>(rows: &[Stored<T>]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    for stored in rows {
        match &stored.line {
            Some(line) => bytes.extend_from_slice(line.as_bytes()),
            None => {
                let line = to_sorted_json_string(&stored.row)
                    .map_err(|error| format!("serialize row: {error}"))?;
                bytes.extend_from_slice(line.as_bytes());
            }
        }
        bytes.push(b'\n');
    }
    Ok(bytes)
}

pub fn read_jsonl_if_exists<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
    Ok(read_stored_if_exists(path)?
        .into_iter()
        .map(|stored| stored.row)
        .collect())
}

/// [`read_jsonl_if_exists`], keeping each row's line ([`Stored`]).
fn read_stored_if_exists<T: DeserializeOwned>(path: &Path) -> Result<Vec<Stored<T>>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("read {}:{}: {error}", path.display(), index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(&line).map_err(|error| {
            format!(
                "parse {}:{}: {error}{}",
                path.display(),
                index + 1,
                schema_mismatch_remedy(&error)
            )
        })?;
        rows.push(Stored {
            row,
            line: Some(line),
        });
    }
    Ok(rows)
}

/// What to do about a tracked row this checkout's schema cannot read because a newer commit wrote
/// it (a field or a variant the schema does not know), or nothing for any other parse error.
fn schema_mismatch_remedy(error: &serde_json::Error) -> &'static str {
    let message = error.to_string();
    if error.classify() == serde_json::error::Category::Data
        && (message.starts_with("unknown field") || message.starts_with("unknown variant"))
    {
        "; the row uses a field or variant this checkout's schema does not know, so a newer \
         commit wrote this shard: read the tracked corpus of the checkout that runs the tool \
         (--corpus, by default its own validation/corpus) and take the private sources from the \
         other corpus with --private-corpus DIR"
    } else {
        ""
    }
}

fn shard_paths(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let Ok(entries) = fs::read_dir(directory) else {
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

/// A row of an aligned shard: the AIR whose hash places it in its shard, and the key its shard is
/// ordered by. Every writer of case and observation shards sorts a shard it writes by
/// [`ShardRow::order_key`] through [`sort_stored`]. The key names one slot (a case's name within
/// its AIR, or a Metal or candidate experiment slot), so a shard's
/// keys strictly increase.
pub(crate) trait ShardRow: DeserializeOwned {
    fn air_sha256(&self) -> &str;
    fn order_key(&self) -> (&str, &str);
}

impl ShardRow for AuthoredCase {
    fn air_sha256(&self) -> &str {
        &self.air_sha256
    }

    fn order_key(&self) -> (&str, &str) {
        (&self.air_sha256, &self.name)
    }
}

impl ShardRow for MetalObservation {
    fn air_sha256(&self) -> &str {
        &self.air_sha256
    }

    fn order_key(&self) -> (&str, &str) {
        (&self.case_id, &self.environment_id)
    }
}

/// A refusal's slot is its case and its environment, as a candidate row's is.
impl ShardRow for RefusalObservation {
    fn air_sha256(&self) -> &str {
        &self.air_sha256
    }

    fn order_key(&self) -> (&str, &str) {
        (&self.case_id, &self.environment_id)
    }
}

impl ShardRow for CandidateObservation {
    fn air_sha256(&self) -> &str {
        &self.air_sha256
    }

    fn order_key(&self) -> (&str, &str) {
        (&self.case_id, &self.environment_id)
    }
}

/// Sort stored `rows` into the key order a shard is written in ([`ShardRow::order_key`]).
fn sort_stored<T: ShardRow>(rows: &mut [Stored<T>]) {
    rows.sort_by(|left, right| left.row.order_key().cmp(&right.row.order_key()));
}

/// Where the shards under `directory` leave the layout the store writes: a row whose AIR places it
/// in another shard, or a row whose key does not follow the previous row's (out of order, or the
/// same slot twice). Rows are read as typed rows but not otherwise validated.
fn shard_layout_errors<T: ShardRow>(
    root: &Path,
    directory: &str,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    for path in shard_paths(&root.join(directory))? {
        let index = shard_index_from_path(&path)?;
        let shard = path
            .file_name()
            .map(|name| format!("{directory}/{}", name.to_string_lossy()))
            .unwrap_or_default();
        let rows: Vec<T> = read_jsonl_if_exists(&path)?;
        for (line, row) in rows.iter().enumerate() {
            let actual = shard_index_for_hash(row.air_sha256())?;
            if actual != index {
                errors.push(format!(
                    "{shard}: row {} (AIR {}) belongs in shard {actual}",
                    line + 1,
                    row.air_sha256()
                ));
            }
        }
        for (line, pair) in rows.windows(2).enumerate() {
            let (previous, next) = (pair[0].order_key(), pair[1].order_key());
            if next <= previous {
                errors.push(format!(
                    "{shard}: row {} {next:?} does not follow row {} {previous:?}",
                    line + 2,
                    line + 1
                ));
            }
        }
    }
    Ok(())
}

trait AlignedObservation: ShardRow {
    fn validate_content(&self) -> Result<(), String>;
}

impl AlignedObservation for RefusalObservation {
    fn validate_content(&self) -> Result<(), String> {
        RefusalObservation::validate_content(self)
    }
}

impl AlignedObservation for MetalObservation {
    fn validate_content(&self) -> Result<(), String> {
        MetalObservation::validate_content(self)
    }
}

impl AlignedObservation for CandidateObservation {
    fn validate_content(&self) -> Result<(), String> {
        CandidateObservation::validate_content(self)
    }
}

fn read_aligned_observations<T: DeserializeOwned + AlignedObservation>(
    directory: &Path,
) -> Result<Vec<T>, String> {
    let mut rows = Vec::new();
    for path in shard_paths(directory)? {
        let expected = shard_index_from_path(&path)?;
        let shard_rows: Vec<T> = read_jsonl_if_exists(&path)?;
        validate_aligned_observation_rows(&shard_rows, expected)?;
        rows.extend(shard_rows);
    }
    Ok(rows)
}

fn validate_aligned_observation_rows<'a, T: AlignedObservation + 'a>(
    rows: impl IntoIterator<Item = &'a T>,
    expected: usize,
) -> Result<(), String> {
    for row in rows {
        row.validate_content()?;
        let actual = shard_index_for_hash(row.air_sha256())?;
        if actual != expected {
            return Err(format!(
                "observation for AIR {} belongs in shard {}, not {}",
                row.air_sha256(),
                actual,
                expected
            ));
        }
    }
    Ok(())
}

/// Every observation of `rows` (case ID, AIR, input digest, and output, each) against its case
/// among `cases`: a known case, its AIR and input digest, and an output of the case's kind (a row
/// with no output, a Metal reference gap, has none to check).
fn validate_observations_of<'a>(
    cases: &[AuthoredCase],
    rows: impl IntoIterator<Item = (&'a String, &'a String, &'a String, Option<&'a String>)>,
) -> Result<(), String> {
    let cases = cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect::<std::collections::HashMap<_, _>>();
    for (case_id, air_sha256, input_sha256, output_b64) in rows {
        let case = cases
            .get(case_id.as_str())
            .ok_or_else(|| format!("observation references unknown case_id {case_id}"))?;
        if case.air_sha256 != *air_sha256 {
            return Err(format!(
                "observation AIR {air_sha256} does not match case AIR {}",
                case.air_sha256
            ));
        }
        let expected = case.computed_input_sha256()?;
        if expected != *input_sha256 {
            return Err(format!(
                "observation input {input_sha256} does not match case input {expected}"
            ));
        }
        if let Some(output_b64) = output_b64 {
            validate_observation_output(case, output_b64)?;
        }
    }
    Ok(())
}

fn validate_observation_output(case: &AuthoredCase, output_b64: &str) -> Result<(), String> {
    let expects_empty = matches!(&case.output, OutputSelection::None);
    if output_b64.is_empty() != expects_empty {
        return Err(if expects_empty {
            "output.kind none requires an exact empty observation".into()
        } else {
            "an exact empty observation requires output.kind none".into()
        });
    }
    Ok(())
}

/// A row this build stores records what its case cost in time, or it is not stored.
///
/// The owner's rule is that every suite records timing for every case it runs, not a selection, and
/// this is where that holds by construction: the paths that produce a row — a Metal qualification,
/// a candidate run on a device, and an attempt that prepared the case and then refused — each hand
/// their own measurement to the store, and a path that measured nothing cannot write a row at all.
/// Rows already on disk are kept as they are: a row recorded before the figure existed carries
/// none, and reading one, or rewriting its shard around a new row, never goes through here.
fn require_timing(
    side: &str,
    case_id: &str,
    environment_id: &str,
    timing: Option<crate::observation::CaseTiming>,
) -> Result<(), String> {
    match timing {
        Some(_) => Ok(()),
        None => Err(format!(
            "the {side} row of case {case_id} in {environment_id} records no timing; a run this \
             build stores measures what its case cost"
        )),
    }
}

fn reject_duplicate_metal_slots<'a>(
    rows: impl IntoIterator<Item = &'a MetalObservation>,
) -> Result<(), String> {
    let mut slots = std::collections::HashSet::new();
    for row in rows {
        if !slots.insert((&row.case_id, &row.environment_id)) {
            return Err(format!(
                "duplicate Metal experiment slot ({}, {})",
                row.case_id, row.environment_id
            ));
        }
    }
    Ok(())
}

fn reject_duplicate_refusal_slots<'a>(
    rows: impl IntoIterator<Item = &'a RefusalObservation>,
) -> Result<(), String> {
    let mut slots = std::collections::HashSet::new();
    for row in rows {
        if !slots.insert((&row.case_id, &row.environment_id)) {
            return Err(format!(
                "duplicate refusal slot ({}, {})",
                row.case_id, row.environment_id
            ));
        }
    }
    Ok(())
}

/// Where the candidate rows are stored, relative to the corpus root, sharded by AIR.
pub const CANDIDATE_DIRECTORY: &str = "observations/vulkan";

/// Where the refusals are stored, relative to the corpus root, sharded by AIR as the rows are.
pub const REFUSAL_DIRECTORY: &str = "observations/refusals/vulkan";

fn reject_duplicate_candidate_slots<'a>(
    rows: impl IntoIterator<Item = &'a CandidateObservation>,
) -> Result<(), String> {
    let mut slots = std::collections::HashSet::new();
    for row in rows {
        if !slots.insert((&row.case_id, &row.environment_id)) {
            return Err(format!(
                "duplicate candidate experiment slot ({}, {})",
                row.case_id, row.environment_id
            ));
        }
    }
    Ok(())
}

fn validate_cases<'a>(
    rows: impl IntoIterator<Item = &'a AuthoredCase>,
    expected_shard: Option<usize>,
) -> Result<(), String> {
    let mut identities = std::collections::HashSet::new();
    let mut names = std::collections::HashSet::new();
    for row in rows {
        row.validate_literal_resources()
            .map_err(|errors| format!("case {}: {}", row.case_id, errors.join("; ")))?;
        let computed = row.computed_case_id()?;
        if computed != row.case_id {
            return Err(format!(
                "case {} identity mismatch: computed {computed}",
                row.case_id
            ));
        }
        if !identities.insert(&row.case_id) {
            return Err(format!("duplicate semantic case_id {}", row.case_id));
        }
        if !names.insert((&row.air_sha256, &row.entry, &row.name)) {
            return Err(format!(
                "duplicate case name {:?} for AIR {} entry {:?}",
                row.name, row.air_sha256, row.entry
            ));
        }
        if let Some(expected) = expected_shard {
            let actual = shard_index_for_hash(&row.air_sha256)?;
            if actual != expected {
                return Err(format!(
                    "case {} is in shard {}, expected {}",
                    row.case_id, expected, actual
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{
        BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection, ResourceRole, Stage,
    };
    use crate::observation::{Backend, CandidateStatus, MetalStatus};
    use crate::observation_contract::ComparisonClass;
    use crate::ScratchDir;
    use std::collections::BTreeMap;

    fn make_case(name: &str, initial: &str) -> AuthoredCase {
        let mut case = AuthoredCase {
            air_sha256: "11".repeat(32),
            case_id: String::new(),
            name: name.into(),
            entry: "main".into(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some(initial.into()),
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

    #[test]
    fn exact_air_case_lookup_does_not_open_unrelated_case_shards() {
        let scratch = ScratchDir::new("exact-air-case-lookup").unwrap();
        let store = CorpusStore::new(scratch.path());
        let first = make_case("first", "q6urqw==");
        let mut unrelated = make_case("unrelated", "u7u7uw==");
        unrelated.air_sha256 = "22".repeat(32);
        unrelated.case_id = unrelated.computed_case_id().unwrap();
        store.put_case(first.clone()).unwrap();
        store.put_case(unrelated.clone()).unwrap();
        let unrelated_shard = shard_index_for_hash(&unrelated.air_sha256).unwrap();
        fs::write(
            scratch
                .path()
                .join("cases")
                .join(shard_name(unrelated_shard)),
            b"not json\n",
        )
        .unwrap();

        assert_eq!(
            store
                .find_cases_for_air(&first.air_sha256)
                .unwrap()
                .as_slice(),
            std::slice::from_ref(&first)
        );
        let mut pipeline = crate::case::indirect_commands::PipelineResource {
            buffer_binding: 0,
            field_offset: 0,
            token: 1,
            air_sha256: first.air_sha256.clone(),
            case_id: first.case_id.clone(),
        };
        assert_eq!(pipeline.resolve_template(&store).unwrap(), first);
        pipeline.air_sha256 = "33".repeat(32);
        assert!(pipeline
            .resolve_template(&store)
            .unwrap_err()
            .contains("does not exist"));
    }

    /// A case found by its ID and AIR reads only the case shard the AIR selects, and a case's
    /// Metal rows only its Metal shard: damaged shards of other AIRs are never opened, the rows are
    /// those the whole-store read gives the case, and a damaged row of the case's own shard still
    /// fails the read. A case named with another AIR is not found there, or is refused where its
    /// shard holds it over another AIR.
    #[test]
    fn exact_case_and_metal_lookups_read_only_the_cases_shards() {
        let scratch = ScratchDir::new("exact-case-metal-lookup").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("first", "q6urqw==");
        let mut unrelated = make_case("unrelated", "u7u7uw==");
        unrelated.air_sha256 = "22".repeat(32);
        unrelated.case_id = unrelated.computed_case_id().unwrap();
        store.put_case(case.clone()).unwrap();
        store.put_case(unrelated.clone()).unwrap();
        let metal = |case: &AuthoredCase, environment_id: &str| MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: crate::hash::sha256_bytes(&[42, 0, 0, 0]),
            output_b64: "KgAAAA==".into(),
            environment_id: environment_id.into(),
            environment: serde_json::json!({}),
            oracle_abi: "v1".into(),
            status: MetalStatus::Qualified,
        };
        for environment_id in ["env-a", "env-b"] {
            store.upsert_metal(metal(&case, environment_id)).unwrap();
        }
        store.upsert_metal(metal(&unrelated, "env-a")).unwrap();
        let whole = store
            .read_metal()
            .unwrap()
            .into_iter()
            .filter(|row| row.case_id == case.case_id)
            .collect::<Vec<_>>();
        assert_eq!(whole.len(), 2);

        // The unrelated AIR's case and Metal shards are damaged: neither exact read opens them.
        let shard = shard_index_for_hash(&case.air_sha256).unwrap();
        let unrelated_shard = shard_index_for_hash(&unrelated.air_sha256).unwrap();
        assert_ne!(shard, unrelated_shard);
        for directory in ["cases", "observations/metal"] {
            fs::write(
                scratch
                    .path()
                    .join(directory)
                    .join(shard_name(unrelated_shard)),
                b"not json\n",
            )
            .unwrap();
        }
        assert!(store.read_all_cases().is_err() && store.read_metal().is_err());
        assert_eq!(
            store
                .find_case_of_air(&case.case_id, &case.air_sha256)
                .unwrap(),
            Some(case.clone())
        );
        assert_eq!(store.read_metal_of_case(&case).unwrap(), whole);

        // Another AIR: its shard does not hold the case, or holds it over another AIR.
        assert_eq!(
            store
                .find_case_of_air(&case.case_id, &"33".repeat(32))
                .unwrap(),
            None
        );
        let same_shard = (0u32..)
            .map(|n| crate::hash::sha256_bytes(&n.to_le_bytes()))
            .find(|hash| hash != &case.air_sha256 && shard_index_for_hash(hash).unwrap() == shard)
            .unwrap();
        assert!(store
            .find_case_of_air(&case.case_id, &same_shard)
            .unwrap_err()
            .contains("not the given"));

        // A damaged row in the case's own Metal shard still fails the read.
        let path = scratch
            .path()
            .join("observations/metal")
            .join(shard_name(shard));
        let mut text = fs::read_to_string(&path).unwrap();
        text.push_str("not json\n");
        fs::write(&path, text).unwrap();
        assert!(store.read_metal_of_case(&case).is_err());
    }

    fn candidate(case: &AuthoredCase, environment_id: &str) -> CandidateObservation {
        let output_sha256 = crate::hash::sha256_bytes(&[42, 0, 0, 0]);
        CandidateObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            golden_output_sha256: output_sha256.clone(),
            spv_sha256: Some("33".repeat(32)),
            translator_fingerprint: "44".repeat(32),
            candidate_output_sha256: output_sha256,
            output_b64: "KgAAAA==".into(),
            backend: Backend::Vulkan,
            environment_id: environment_id.into(),
            environment: serde_json::json!({}),
            executor_abi: "v1".into(),
            comparison: ComparisonClass::Exact,
            census: None,
            status: CandidateStatus::Match,
        }
    }

    /// A row this build stores records what its case cost in time, and the store refuses one that
    /// does not. This is where the owner's rule — every suite records timing for every case it
    /// runs, not a selection — holds by construction rather than by review: the two paths that
    /// produce a row (a Metal qualification and a candidate run on a device) hand the store their
    /// own measurement, and a path that lost its clock cannot write a row at all.
    ///
    /// Rows already on disk are untouched by this. One recorded before the figure existed carries
    /// none, is read as it always was, and keeps its own bytes when a new row rewrites its shard.
    #[test]
    fn a_stored_row_records_what_its_case_cost() {
        let scratch = ScratchDir::new("stored-row-timing").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("timed", "KgAAAA==");
        store.put_case(case.clone()).unwrap();
        let metal = |timing| MetalObservation {
            timing,
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: crate::hash::sha256_bytes(&[42, 0, 0, 0]),
            output_b64: "KgAAAA==".into(),
            environment_id: "metal-env".into(),
            environment: serde_json::json!({}),
            oracle_abi: "v1".into(),
            status: MetalStatus::Qualified,
        };
        let refused = store.upsert_metal(metal(None)).unwrap_err();
        assert!(
            refused.contains("records no timing") && refused.contains(&case.case_id),
            "{refused}"
        );
        assert!(store.read_metal().unwrap().is_empty());
        store
            .upsert_metal(metal(Some(crate::observation::test_timing())))
            .unwrap();

        let mut rows = store.read_metal().unwrap();
        assert_eq!(rows.len(), 1);
        let stored = rows.pop().unwrap().timing.expect("a stored Metal timing");
        assert_eq!(stored.runs, crate::observation::test_timing().runs);
        assert_eq!(
            stored.total_micros,
            crate::observation::test_timing().total_micros
        );
        assert_eq!(
            stored.least_micros,
            crate::observation::test_timing().least_micros
        );

        let mut untimed = candidate(&case, "vk-env");
        untimed.timing = None;
        let refused = store.upsert_candidate(untimed).unwrap_err();
        assert!(
            refused.contains("records no timing") && refused.contains("vk-env"),
            "{refused}"
        );
        assert!(store.read_candidates().unwrap().is_empty());
        store.upsert_candidate(candidate(&case, "vk-env")).unwrap();
        let mut rows = store.read_candidates().unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows.pop().unwrap().timing.is_some());

        // A refused attempt is a case the suite ran: it prepared the case and compiled before it
        // refused, so its row records what that cost on the same terms.
        let mut untimed = refusal(&case, "env@engine");
        untimed.timing = None;
        let refused = store.upsert_refusal(untimed).unwrap_err();
        assert!(
            refused.contains("records no timing") && refused.contains("env@engine"),
            "{refused}"
        );
        assert!(store.read_refusals().unwrap().is_empty());
        store.upsert_refusal(refusal(&case, "env@engine")).unwrap();
        let mut refusals = store.read_refusals().unwrap();
        assert_eq!(refusals.len(), 1);
        assert!(refusals.pop().unwrap().timing.is_some());

        // A row a shard already holds without a figure is read as it is, and stays byte for byte
        // as it is when another row joins its shard.
        let shard = shard_index_for_hash(&case.air_sha256).unwrap();
        let path = scratch
            .path()
            .join("observations/metal")
            .join(shard_name(shard));
        let mut old = metal(None);
        old.environment_id = "old-env".into();
        let line = serde_json::to_string(&old).unwrap();
        assert!(!line.contains("timing"));
        let mut text = fs::read_to_string(&path).unwrap();
        text.insert_str(0, &format!("{line}\n"));
        fs::write(&path, &text).unwrap();
        let read = store.read_metal().unwrap();
        assert_eq!(read.len(), 2);
        assert!(read
            .iter()
            .any(|row| row.environment_id == "old-env" && row.timing.is_none()));
        let mut joined = metal(Some(crate::observation::test_timing()));
        joined.environment_id = "second-env".into();
        store.upsert_metal(joined).unwrap();
        let written = fs::read_to_string(&path).unwrap();
        assert!(
            written.lines().any(|written| written == line),
            "the untimed row was rewritten"
        );
    }

    /// The store's own writes keep every shard in key order with one row per slot, and
    /// `layout_errors` names a shard whose rows were appended out of order, a slot written twice,
    /// and a row in a shard its AIR does not select (23b3bbbc appended MoltenVK rows).
    #[test]
    fn layout_errors_name_appended_duplicated_and_misplaced_rows() {
        let scratch = ScratchDir::new("store-layout").unwrap();
        let store = CorpusStore::new(scratch.path());
        let (first, second) = (
            make_case("b-slot", "AAAAAA=="),
            make_case("a-slot", "AQAAAA=="),
        );
        store.put_case(first.clone()).unwrap();
        store.put_case(second.clone()).unwrap();
        for case in [&first, &second] {
            for environment in ["env-b", "env-a"] {
                store
                    .upsert_candidate(candidate(case, environment))
                    .unwrap();
            }
        }
        assert_eq!(store.layout_errors().unwrap(), Vec::<String>::new());

        let shard = shard_name(shard_index_for_hash(&first.air_sha256).unwrap());
        let shard_file = scratch.path().join("observations/vulkan").join(&shard);
        let sorted = std::fs::read_to_string(&shard_file).unwrap();
        let lines = sorted.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 4);
        // Appended out of order, then one slot twice.
        let appended = [lines[1], lines[2], lines[3], lines[0]].join("\n") + "\n";
        std::fs::write(&shard_file, appended).unwrap();
        let errors = store.layout_errors().unwrap();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].starts_with(&format!("observations/vulkan/{shard}: row 4 ")));
        let twice = [lines[0], lines[1], lines[1], lines[2], lines[3]].join("\n") + "\n";
        std::fs::write(&shard_file, twice).unwrap();
        let errors = store.layout_errors().unwrap();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("row 3 "), "{errors:?}");
        std::fs::write(&shard_file, &sorted).unwrap();

        // A case row in a shard its AIR does not select.
        let other = shard_name((shard_index_for_hash(&first.air_sha256).unwrap() + 1) % 64);
        let cases = std::fs::read_to_string(scratch.path().join("cases").join(&shard)).unwrap();
        std::fs::write(
            scratch.path().join("cases").join(&other),
            cases.lines().next().unwrap(),
        )
        .unwrap();
        let errors = store.layout_errors().unwrap();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].starts_with(&format!("cases/{other}: row 1 "))
                && errors[0].contains("belongs"),
            "{errors:?}"
        );
    }

    /// Installing an identity its slot already holds rewrites only the fields outside the
    /// identity and keeps its observations; the same identity under another name is refused.
    #[test]
    fn reinstalling_an_identity_annotates_it_and_keeps_its_observations() {
        let scratch = ScratchDir::new("store-annotate").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("slot", "AAAAAA==");
        store.put_case(case.clone()).unwrap();
        store.upsert_candidate(candidate(&case, "mvk-env")).unwrap();
        // A valid row in a form a rewrite would not keep (a trailing space), as legacy rows are.
        let shard = shard_name(shard_index_for_hash(&case.air_sha256).unwrap());
        let shard_file = scratch.path().join("observations/vulkan").join(&shard);
        let row = std::fs::read_to_string(&shard_file).unwrap();
        std::fs::write(&shard_file, format!("{} \n", row.trim_end())).unwrap();
        let observations = std::fs::read(&shard_file).unwrap();
        let mut annotated = case.clone();
        annotated.public_source =
            Some("crates/metal2vulkan-validation/fixtures/public/x.ll".into());
        annotated.rationale = Some("why".into());
        assert_eq!(
            store.put_case(annotated.clone()).unwrap(),
            PutCaseResult::Annotated
        );
        assert_eq!(store.read_all_cases().unwrap(), vec![annotated]);
        // The observation shards are not rewritten at all.
        assert_eq!(std::fs::read(&shard_file).unwrap(), observations);
        assert_eq!(store.read_candidates().unwrap().len(), 1);
        let mut renamed = case.clone();
        renamed.name = "other".into();
        let error = store.put_case(renamed).unwrap_err();
        assert!(error.contains("duplicate semantic case_id"), "{error}");
    }

    /// Every store writer writes back the rows it does not change exactly as they were read. The
    /// fast-math branch's install rewrote two committed MoltenVK rows of executor v34 and dropped
    /// their `containment.device_addresses`, a field the schema reads and does not keep
    /// (`coverage::ContainmentRecord`).
    #[test]
    fn a_store_write_leaves_every_row_it_does_not_change_byte_for_byte() {
        use crate::coverage::{
            BufferEncoding, Containment, ExecutedCensus, ImageEncoding, ModuleCensus,
        };
        let scratch = ScratchDir::new("store-byte-stable").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("slot", "AAAAAA==");
        store.put_case(case.clone()).unwrap();
        let mut old = candidate(&case, "env-old");
        old.census = Some(ExecutedCensus {
            device: None,
            product: "p".into(),
            restricted_to: None,
            modules: vec![ModuleCensus {
                stage: Stage::Kernel,
                subgroup: Default::default(),
                strategies: Default::default(),
                precision: Default::default(),
                containment: Some(Containment {
                    buffers: BufferEncoding::Checked,
                    images: ImageEncoding::Unused,
                }),
            }],
            out_of_bounds: None,
            rate_map_pass: None,
            artifact: None,
        });
        store.upsert_candidate(old).unwrap();
        store
            .upsert_metal(MetalObservation {
                timing: Some(crate::observation::test_timing()),
                case_id: case.case_id.clone(),
                air_sha256: case.air_sha256.clone(),
                input_sha256: case.computed_input_sha256().unwrap(),
                metal_output_sha256: crate::hash::sha256_bytes(&[42, 0, 0, 0]),
                output_b64: "KgAAAA==".into(),
                environment_id: "metal-old".into(),
                environment: serde_json::json!({}),
                oracle_abi: "v1".into(),
                status: MetalStatus::Qualified,
            })
            .unwrap();
        // The rows as an older executor wrote them: the containment states `device_addresses`,
        // and the Metal row's keys are out of the order a write would sort them into.
        let shard = shard_name(shard_index_for_hash(&case.air_sha256).unwrap());
        let shard_file = scratch.path().join("observations/vulkan").join(&shard);
        let written = fs::read_to_string(&shard_file).unwrap();
        let legacy = written.replace(
            r#""buffers":"checked","images""#,
            r#""buffers":"checked","device_addresses":false,"images""#,
        );
        assert_ne!(legacy, written);
        fs::write(&shard_file, &legacy).unwrap();
        let metal = scratch.path().join("observations/metal").join(&shard);
        let row: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(fs::read_to_string(&metal).unwrap().trim_end()).unwrap();
        let reversed = format!(
            "{{{}}}\n",
            row.iter()
                .rev()
                .map(|(key, value)| format!("{}:{value}", serde_json::Value::from(key.as_str())))
                .collect::<Vec<_>>()
                .join(",")
        );
        fs::write(&metal, &reversed).unwrap();
        assert_eq!(store.read_candidates().unwrap().len(), 1);
        assert_eq!(store.read_metal().unwrap().len(), 1);

        // A row of another slot in the same shard: the legacy row stays as it was.
        store.upsert_candidate(candidate(&case, "env-new")).unwrap();
        let lines = fs::read_to_string(&shard_file).unwrap();
        assert!(
            lines.lines().any(|line| line == legacy.trim_end()),
            "{lines}"
        );
        assert_eq!(lines.lines().count(), 2);
        store
            .upsert_metal(MetalObservation {
                environment_id: "metal-new".into(),
                ..store.read_metal().unwrap().remove(0)
            })
            .unwrap();
        let metal_lines = fs::read_to_string(&metal).unwrap();
        assert_eq!(metal_lines.lines().count(), 2);
        assert!(metal_lines.contains(&reversed), "{metal_lines}");

        // A refusal, a case added to the bucket (which rewrites every shard of it),
        // and the refusal's removal by a row in its slot: the legacy rows never change.
        store
            .upsert_refusal(RefusalObservation {
                backend: Backend::Vulkan,
                ..refusal(&case, "env@engine")
            })
            .unwrap();
        store.put_case(make_case("other", "AQAAAA==")).unwrap();
        store
            .upsert_candidate(candidate(&case, "env@engine"))
            .unwrap();
        let lines = fs::read_to_string(&shard_file).unwrap();
        assert!(
            lines.lines().any(|line| line == legacy.trim_end()),
            "{lines}"
        );
        assert!(fs::read_to_string(&metal).unwrap().contains(&reversed));
        assert!(store.layout_errors().unwrap().is_empty());
    }

    #[test]
    fn replacing_named_case_cascades_only_old_identity() {
        let scratch = ScratchDir::new("store-cascade").unwrap();
        let store = CorpusStore::new(scratch.path());
        let first = make_case("slot", "AAAAAA==");
        let sibling = make_case("sibling", "AQAAAA==");
        store.put_case(first.clone()).unwrap();
        store.put_case(sibling.clone()).unwrap();
        for case in [&first, &sibling] {
            store
                .upsert_metal(MetalObservation {
                    timing: Some(crate::observation::test_timing()),
                    case_id: case.case_id.clone(),
                    air_sha256: case.air_sha256.clone(),
                    input_sha256: case.computed_input_sha256().unwrap(),
                    metal_output_sha256: crate::hash::sha256_bytes(&[42, 0, 0, 0]),
                    output_b64: "KgAAAA==".into(),
                    environment_id: "env".into(),
                    environment: serde_json::json!({}),
                    oracle_abi: "v1".into(),
                    status: MetalStatus::Qualified,
                })
                .unwrap();
            store.upsert_candidate(candidate(case, "vk-env")).unwrap();
        }

        let replacement = make_case("slot", "AgAAAA==");
        assert_eq!(
            store.put_case(replacement.clone()).unwrap(),
            PutCaseResult::Replaced {
                old_case_id: first.case_id.clone()
            }
        );
        let cases = store.read_all_cases().unwrap();
        assert!(cases.iter().any(|row| row.case_id == replacement.case_id));
        assert!(cases.iter().any(|row| row.case_id == sibling.case_id));
        assert!(!cases.iter().any(|row| row.case_id == first.case_id));
        let metal = store.read_metal().unwrap();
        assert_eq!(metal.len(), 1);
        assert_eq!(metal[0].case_id, sibling.case_id);
        let rows = store.read_candidates().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].case_id, sibling.case_id);
    }

    #[test]
    fn exact_slot_upsert_replaces_instead_of_accumulating() {
        let scratch = ScratchDir::new("slot-upsert").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("slot", "AAAAAA==");
        store.put_case(case.clone()).unwrap();
        let mut row = MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: crate::hash::sha256_bytes(&[42, 0, 0, 0]),
            output_b64: "KgAAAA==".into(),
            environment_id: "env".into(),
            environment: serde_json::json!({}),
            oracle_abi: "v1".into(),
            status: MetalStatus::Qualified,
        };
        store.upsert_metal(row.clone()).unwrap();
        row.metal_output_sha256 = crate::hash::sha256_bytes(&[43, 0, 0, 0]);
        row.output_b64 = "KwAAAA==".into();
        store.upsert_metal(row.clone()).unwrap();
        assert_eq!(store.read_metal().unwrap(), vec![row]);

        let mut candidate = candidate(&case, "env");
        store.upsert_candidate(candidate.clone()).unwrap();
        candidate.spv_sha256 = Some("44".repeat(32));
        store.upsert_candidate(candidate.clone()).unwrap();
        assert_eq!(store.read_candidates().unwrap(), vec![candidate]);
    }

    /// A context the profiles layer restricts without a request opens and runs, but the row it
    /// would write is refused: nothing is stored, and the native row is.
    #[test]
    fn a_row_under_an_unrequested_profiles_layer_is_never_stored() {
        let scratch = ScratchDir::new("unrequested-profiles-layer").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("restricted", "AAAAAA==");
        store.put_case(case.clone()).unwrap();
        let mut row = candidate(&case, "env");
        row.environment = serde_json::json!({
            "tools": [
                {"layer": "VK_LAYER_KHRONOS_validation"},
                {"layer": "VK_LAYER_KHRONOS_profiles"},
            ],
        });
        let refused = store.upsert_candidate(row.clone()).unwrap_err();
        assert!(refused.contains("requested no restriction"), "{refused}");
        assert_eq!(store.read_candidates().unwrap(), vec![]);
        row.environment = serde_json::json!({
            "tools": [{"layer": "VK_LAYER_KHRONOS_validation"}],
        });
        store.upsert_candidate(row.clone()).unwrap();
        assert_eq!(store.read_candidates().unwrap(), vec![row]);
    }

    #[test]
    fn empty_observation_is_coupled_to_none_output_selection() {
        let scratch = ScratchDir::new("store-empty-observation").unwrap();
        let store = CorpusStore::new(scratch.path());
        let mut empty_case = make_case("empty", "AAAAAA==");
        empty_case.buffers.clear();
        empty_case.output = OutputSelection::None;
        empty_case.case_id = empty_case.computed_case_id().unwrap();
        store.put_case(empty_case.clone()).unwrap();

        let empty_sha256 = crate::hash::sha256_bytes(&[]);
        let empty_metal = MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: empty_case.case_id.clone(),
            air_sha256: empty_case.air_sha256.clone(),
            input_sha256: empty_case.computed_input_sha256().unwrap(),
            metal_output_sha256: empty_sha256.clone(),
            output_b64: String::new(),
            environment_id: "env".into(),
            environment: serde_json::json!({}),
            oracle_abi: "v1".into(),
            status: MetalStatus::Qualified,
        };
        store.upsert_metal(empty_metal.clone()).unwrap();
        assert_eq!(store.read_metal().unwrap(), vec![empty_metal]);

        let mut nonempty_for_none = candidate(&empty_case, "vk-env");
        nonempty_for_none.golden_output_sha256 = nonempty_for_none.candidate_output_sha256.clone();
        assert!(store
            .upsert_candidate(nonempty_for_none)
            .unwrap_err()
            .contains("requires an exact empty observation"));

        let nonempty_case = make_case("nonempty", "AAAAAA==");
        store.put_case(nonempty_case.clone()).unwrap();
        let nonempty_input_sha256 = nonempty_case.computed_input_sha256().unwrap();
        let empty_for_buffer = MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: nonempty_case.case_id.clone(),
            air_sha256: nonempty_case.air_sha256,
            input_sha256: nonempty_input_sha256,
            metal_output_sha256: empty_sha256,
            output_b64: String::new(),
            environment_id: "env".into(),
            environment: serde_json::json!({}),
            oracle_abi: "v1".into(),
            status: MetalStatus::Qualified,
        };
        assert!(store
            .upsert_metal(empty_for_buffer)
            .unwrap_err()
            .contains("requires output.kind none"));
    }

    /// A reference gap records no output whatever the case selects: the store keeps it for a case
    /// that selects a buffer, and still checks the row's case, AIR, and inputs.
    #[test]
    fn a_reference_gap_row_is_stored_without_output() {
        let scratch = ScratchDir::new("store-reference-gap").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("gap", "AAAAAA==");
        store.put_case(case.clone()).unwrap();
        let message =
            "simdgroup_matrix<T,16,16x16> operations are supported by GPUFamily10 and later";
        let gap = MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: crate::hash::sha256_bytes(&[]),
            output_b64: String::new(),
            environment_id: "env".into(),
            environment: serde_json::json!({}),
            oracle_abi: "v1".into(),
            status: MetalStatus::ReferenceGap(crate::observation::ReferenceGap {
                apple_family: 10,
                message: message.into(),
            }),
        };
        store.upsert_metal(gap.clone()).unwrap();
        assert_eq!(store.read_metal().unwrap(), vec![gap.clone()]);
        let other_inputs = MetalObservation {
            input_sha256: "22".repeat(32),
            environment_id: "env-2".into(),
            ..gap.clone()
        };
        assert!(store.upsert_metal(other_inputs).is_err());
        let qualified_empty = MetalObservation {
            environment_id: "env-3".into(),
            status: MetalStatus::Qualified,
            ..gap
        };
        assert!(store
            .upsert_metal(qualified_empty)
            .unwrap_err()
            .contains("requires output.kind none"));
    }

    /// A case's slot is its module, entry, and name: a same-name case over another entry of the
    /// module is added beside the first, whose identity and observations stand, and deleting by
    /// name alone refuses until the entry is named.
    #[test]
    fn a_same_name_case_over_another_entry_is_another_slot() {
        let scratch = ScratchDir::new("store-entry-slots").unwrap();
        let store = CorpusStore::new(scratch.path());
        let over = |entry: &str, initial: &str| {
            let mut case = make_case("same-name", initial);
            case.entry = entry.into();
            case.case_id = case.computed_case_id().unwrap();
            case
        };
        let first = over("first", "AAAAAA==");
        let second = over("second", "AQAAAA==");
        assert_eq!(store.put_case(first.clone()).unwrap(), PutCaseResult::Added);
        store.upsert_candidate(candidate(&first, "env")).unwrap();
        assert_eq!(
            store.put_case(second.clone()).unwrap(),
            PutCaseResult::Added
        );
        let mut cases = store.read_all_cases().unwrap();
        cases.sort_by(|left, right| left.entry.cmp(&right.entry));
        assert_eq!(cases, [first.clone(), second.clone()]);
        assert_eq!(store.read_candidates().unwrap(), [candidate(&first, "env")]);

        // The same slot again is a replacement of that entry's case alone.
        let replacement = over("second", "AgAAAA==");
        assert_eq!(
            store.put_case(replacement.clone()).unwrap(),
            PutCaseResult::Replaced {
                old_case_id: second.case_id.clone()
            }
        );

        let error = store
            .delete_named_case(&first.air_sha256, None, "same-name")
            .unwrap_err();
        assert!(
            error.contains("first") && error.contains("second"),
            "{error}"
        );
        assert_eq!(
            store
                .delete_named_case(&first.air_sha256, Some("second"), "same-name")
                .unwrap(),
            replacement.case_id
        );
        assert_eq!(
            store.read_all_cases().unwrap(),
            std::slice::from_ref(&first)
        );
        assert_eq!(store.read_candidates().unwrap(), [candidate(&first, "env")]);
    }

    #[test]
    fn deleting_named_case_cascades_only_its_evidence() {
        let scratch = ScratchDir::new("store-delete").unwrap();
        let store = CorpusStore::new(scratch.path());
        let removed = make_case("removed", "AAAAAA==");
        let kept = make_case("kept", "AQAAAA==");
        store.put_case(removed.clone()).unwrap();
        store.put_case(kept.clone()).unwrap();
        store.upsert_candidate(candidate(&removed, "env")).unwrap();
        store.upsert_candidate(candidate(&kept, "env")).unwrap();

        assert_eq!(
            store
                .delete_named_case(&removed.air_sha256, None, &removed.name)
                .unwrap(),
            removed.case_id
        );
        assert_eq!(store.read_all_cases().unwrap(), vec![kept.clone()]);
        assert_eq!(
            store.read_candidates().unwrap(),
            vec![candidate(&kept, "env")]
        );
    }

    fn refusal(case: &AuthoredCase, slot: &str) -> RefusalObservation {
        RefusalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            translator_fingerprint: "55".repeat(32),
            backend: Backend::Vulkan,
            environment_id: slot.into(),
            environment: serde_json::json!({"robustness": {"enabled": [], "unmet": []}}),
            executor_abi: "v1".into(),
            refusal: crate::refusal::Refusal::of(
                &crate::check::admission::NotSupported::NotEnabled(vec!["rayQuery".into()]),
            ),
            artifact: None,
        }
    }

    /// A venue rekey rewrites exactly the records that recorded its stem at its `from` key, rows
    /// and refusals: a record of the same profile at another key, one of another
    /// profile at the same key, and one that recorded no venue keep their bytes, and a second run
    /// rewrites nothing.
    #[test]
    fn a_venue_rekey_rewrites_exactly_the_records_at_its_from_key() {
        use crate::c3::{RecordedKey, VenueRekey};
        let scratch = ScratchDir::new("store-rekey").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("rekeyed", "AAAAAA==");
        store.put_case(case.clone()).unwrap();
        let key = |sha256: &str, fingerprint: &str| RecordedKey {
            sha256: sha256.repeat(32),
            fingerprint: fingerprint.repeat(32),
        };
        let (old, other, new) = (key("aa", "01"), key("aa", "02"), key("bb", "03"));
        let venue = |stem: &str, recorded: &RecordedKey| {
            serde_json::json!({
                crate::c3::VENUE_PROFILE: {
                    "profile": stem,
                    "sha256": recorded.sha256,
                    "fingerprint": recorded.fingerprint,
                },
            })
        };
        for (slot, environment) in [
            ("a@engine", venue("profile-a", &old)),
            ("a2@engine", venue("profile-a", &other)),
            ("b@engine", venue("profile-b", &old)),
            ("plain", serde_json::json!({})),
            ("m@engine", venue("profile-a", &old)),
        ] {
            store
                .upsert_candidate(CandidateObservation {
                    environment,
                    ..candidate(&case, slot)
                })
                .unwrap();
        }
        store
            .upsert_refusal(RefusalObservation {
                environment: serde_json::json!({
                    crate::c3::VENUE_PROFILE: venue("profile-a", &old)[crate::c3::VENUE_PROFILE],
                }),
                ..refusal(&case, "r@engine")
            })
            .unwrap();
        let lines = || {
            let mut lines = Vec::new();
            for dir in ["observations/vulkan", "observations/refusals/vulkan"] {
                for path in shard_paths(&scratch.path().join(dir)).unwrap() {
                    lines.extend(fs::read_to_string(path).unwrap().lines().map(String::from));
                }
            }
            lines
        };
        let before = lines();
        assert_eq!(before.len(), 6);
        let rekeys = [VenueRekey {
            stem: "profile-a".into(),
            from: old.clone(),
            to: new.clone(),
        }];
        assert_eq!(store.rekey_venue_records(&rekeys).unwrap(), [(2, 1)]);
        let after = lines();
        let kept = before.iter().filter(|line| after.contains(line)).count();
        assert_eq!(kept, 3, "{after:#?}");
        let fingerprint = |environment: &serde_json::Value| {
            crate::c3::VenueProfile::recorded_fingerprint(environment).map(String::from)
        };
        let mut recorded = BTreeMap::new();
        for row in store.read_candidates().unwrap() {
            recorded.insert(row.environment_id.clone(), fingerprint(&row.environment));
        }
        for refused in store.read_refusals().unwrap() {
            recorded.insert(
                refused.environment_id.clone(),
                fingerprint(&refused.environment),
            );
        }
        assert_eq!(
            recorded,
            BTreeMap::from([
                ("a@engine".to_string(), Some(new.fingerprint.clone())),
                ("a2@engine".to_string(), Some(other.fingerprint.clone())),
                ("b@engine".to_string(), Some(old.fingerprint.clone())),
                ("m@engine".to_string(), Some(new.fingerprint.clone())),
                ("plain".to_string(), None),
                ("r@engine".to_string(), Some(new.fingerprint.clone())),
            ])
        );
        assert_eq!(store.layout_errors().unwrap(), Vec::<String>::new());
        assert_eq!(store.rekey_venue_records(&rekeys).unwrap(), [(0, 0)]);
        assert_eq!(lines(), after);
        let transactions = scratch.path().join(".transactions");
        assert!(fs::read_dir(&transactions)
            .into_iter()
            .flatten()
            .next()
            .is_none());
    }

    /// A slot holds the last run's outcome, a row or a refusal, never both; a refusal follows its
    /// case as a row does when the case is replaced or deleted; a refusal shard that empties is
    /// removed; and the shards stay in the store's layout.
    #[test]
    fn a_slot_holds_a_row_or_a_refusal_and_refusals_follow_their_case() {
        let scratch = ScratchDir::new("store-refusals").unwrap();
        let store = CorpusStore::new(scratch.path());
        let refused = make_case("refused", "AAAAAA==");
        let kept = make_case("kept", "AQAAAA==");
        store.put_case(refused.clone()).unwrap();
        store.put_case(kept.clone()).unwrap();
        let slot = "env@engine";
        let shard = shard_name(shard_index_for_hash(&refused.air_sha256).unwrap());
        let refusal_shard = scratch.path().join(REFUSAL_DIRECTORY).join(&shard);

        store.upsert_candidate(candidate(&refused, slot)).unwrap();
        assert!(!refusal_shard.exists(), "a row creates no refusal shard");
        // A refusal replaces the slot's row, and only that slot's.
        store.upsert_candidate(candidate(&kept, slot)).unwrap();
        store.upsert_refusal(refusal(&refused, slot)).unwrap();
        assert_eq!(
            store.read_candidates().unwrap(),
            vec![candidate(&kept, slot)]
        );
        assert_eq!(
            store.read_refusals().unwrap(),
            vec![refusal(&refused, slot)]
        );
        // A refusal in another slot is its own.
        let off = "env@engine@runtime-floor";
        store.upsert_refusal(refusal(&refused, off)).unwrap();
        store.upsert_refusal(refusal(&kept, off)).unwrap();
        assert_eq!(store.read_refusals().unwrap().len(), 3);
        assert!(store.layout_errors().unwrap().is_empty());
        // A row replaces the slot's refusal.
        store.upsert_candidate(candidate(&refused, slot)).unwrap();
        let mut expected = vec![refusal(&kept, off), refusal(&refused, off)];
        expected.sort_by(|a, b| a.order_key().cmp(&b.order_key()));
        assert_eq!(store.read_refusals().unwrap(), expected);
        assert_eq!(store.read_candidates().unwrap().len(), 2);
        // A refusal of an unknown case is not stored.
        let unknown = make_case("unknown", "AgAAAA==");
        assert!(store
            .upsert_refusal(refusal(&unknown, slot))
            .unwrap_err()
            .contains("unknown case_id"));

        // Replacing a case drops its old identity's refusals; deleting one drops its own.
        let replacement = make_case("refused", "AwAAAA==");
        store.put_case(replacement).unwrap();
        assert_eq!(store.read_refusals().unwrap(), vec![refusal(&kept, off)]);
        store
            .delete_named_case(&kept.air_sha256, None, &kept.name)
            .unwrap();
        assert!(store.read_refusals().unwrap().is_empty());
        assert!(
            !refusal_shard.exists(),
            "an emptied refusal shard is removed"
        );
        assert!(store.layout_errors().unwrap().is_empty());
    }

    /// A run restricted to a profile the device cannot take is a typed device refusal,
    /// `not_supported{profile_exceeds_device}` naming each member, stored in the restricted slot,
    /// read back as written, and kept apart from the venue's own slots.
    #[test]
    fn a_refused_restriction_is_stored_in_its_restricted_slot() {
        use crate::check::admission::NotSupported;
        let scratch = ScratchDir::new("store-restriction-refusal").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("restricted", "AAAAAA==");
        store.put_case(case.clone()).unwrap();
        let refused = NotSupported::Restriction {
            profile: "runtime-floor".into(),
            gaps: vec![
                "formats.VK_FORMAT_R8_SNORM.optimal_tiling 0xdd83 beyond the device's 0x1dc83"
                    .into(),
            ],
        };
        let restricted = |slot: &str| RefusalObservation {
            environment: serde_json::json!({
                "profiles_layer": {"profile": "runtime-floor"},
            }),
            refusal: crate::refusal::Refusal::of(&refused),
            ..refusal(&case, slot)
        };
        let slot = "env@engine@runtime-floor";
        store.upsert_refusal(restricted(slot)).unwrap();
        store
            .upsert_candidate(candidate(&case, "env@engine"))
            .unwrap();
        let stored = store.read_refusals().unwrap();
        assert_eq!(stored, vec![restricted(slot)]);
        assert_eq!(
            stored[0].refusal.class,
            crate::refusal::RefusalClass::Device
        );
        assert_eq!(
            stored[0].refusal.outcome,
            "not_supported{profile_exceeds_device}"
        );
        assert!(stored[0]
            .refusal
            .detail
            .contains("profile runtime-floor states 1 capabilities the device lacks: formats.VK_FORMAT_R8_SNORM"));
        // The venue's own slot keeps its row.
        assert_eq!(store.read_candidates().unwrap().len(), 1);
        assert!(store.layout_errors().unwrap().is_empty());
    }

    /// A product refusal record keeps its bytes through a read and a canonical write, whether it
    /// records the binder's phase (`at_bind` true or false, as this build writes it) or not (an
    /// earlier build's record, which reads as phase unrecorded): an install rewriting a shard never
    /// changes a record it did not write.
    #[test]
    fn a_product_refusal_shard_keeps_its_bytes_with_and_without_its_phase() {
        use crate::check::admission::NotSupported;
        let scratch = ScratchDir::new("store-refusal-phase").unwrap();
        let store = CorpusStore::new(scratch.path());
        let case = make_case("phase", "AAAAAA==");
        store.put_case(case.clone()).unwrap();
        let product = |at_bind| RefusalObservation {
            refusal: crate::refusal::Refusal::of(&NotSupported::Product {
                outcome: crate::engine::Outcome::Unsupported {
                    reason: crate::engine::UnsupportedReason::NotImplemented,
                    feature: "SparseTexture".into(),
                },
                detail: "unsupported: SparseTexture".into(),
                at_bind,
            }),
            ..refusal(&case, "env@engine")
        };
        let bound = product(true);
        let compiled = RefusalObservation {
            environment_id: "env@engine@runtime-floor".into(),
            ..product(false)
        };
        let mut earlier = serde_json::to_value(RefusalObservation {
            environment_id: "env@engine@lavapipe".into(),
            ..product(false)
        })
        .unwrap();
        earlier["refusal"]["product"]
            .as_object_mut()
            .unwrap()
            .remove("at_bind");
        let earlier: RefusalObservation = serde_json::from_value(earlier).unwrap();
        assert_eq!(earlier.refusal.product.as_ref().unwrap().at_bind, None);
        // Each row serialized afresh, as a writer serializes a row it adds ([`Stored`]).
        let serialized = |rows: &[RefusalObservation]| {
            stored_jsonl(&rows.iter().cloned().map(Stored::new).collect::<Vec<_>>()).unwrap()
        };
        let mut rows = vec![bound, compiled, earlier];
        rows.sort_by(|a, b| a.order_key().cmp(&b.order_key()));
        let bytes = serialized(&rows);
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert_eq!(text.matches("\"at_bind\":true").count(), 1, "{text}");
        assert_eq!(text.matches("\"at_bind\":false").count(), 1, "{text}");
        let shard = scratch
            .path()
            .join(REFUSAL_DIRECTORY)
            .join(shard_name(shard_index_for_hash(&case.air_sha256).unwrap()));
        fs::create_dir_all(shard.parent().unwrap()).unwrap();
        fs::write(&shard, &bytes).unwrap();
        let read = store.read_refusals().unwrap();
        assert_eq!(read, rows);
        assert_eq!(serialized(&read), bytes);
        assert!(store.layout_errors().unwrap().is_empty());
        // A write through the store keeps the other records' bytes.
        store.upsert_refusal(product(true)).unwrap();
        assert_eq!(fs::read(&shard).unwrap(), bytes);
    }

    #[test]
    fn prepared_transaction_is_recovered() {
        let scratch = ScratchDir::new("store-recovery").unwrap();
        let store = CorpusStore::new(scratch.path());
        let transaction = scratch.path().join(".transactions/interrupted");
        fs::create_dir_all(&transaction).unwrap();
        fs::write(transaction.join("file-000"), b"recovered").unwrap();
        fs::write(
            transaction.join("manifest.json"),
            serde_json::to_vec(&TransactionManifest {
                entries: vec![TransactionEntry {
                    staged: "file-000".into(),
                    target: "cases/recovered.txt".into(),
                    delete: false,
                }],
            })
            .unwrap(),
        )
        .unwrap();

        store.recover_transactions().unwrap();
        assert_eq!(
            fs::read(scratch.path().join("cases/recovered.txt")).unwrap(),
            b"recovered"
        );
        assert!(!transaction.exists());
    }

    #[test]
    fn canonical_readers_reject_misaligned_case_and_observation_rows() {
        let case_scratch = ScratchDir::new("misaligned-case").unwrap();
        let case_store = CorpusStore::new(case_scratch.path());
        let case = make_case("slot", "AAAAAA==");
        case_store.put_case(case.clone()).unwrap();
        let actual = shard_index_for_hash(&case.air_sha256).unwrap();
        let wrong = (actual + 1) % crate::source::SHARD_COUNT;
        fs::rename(
            case_scratch.path().join("cases").join(shard_name(actual)),
            case_scratch.path().join("cases").join(shard_name(wrong)),
        )
        .unwrap();
        assert!(case_store
            .read_all_cases()
            .unwrap_err()
            .contains("expected"));

        let observation_scratch = ScratchDir::new("misaligned-observation").unwrap();
        let observation_store = CorpusStore::new(observation_scratch.path());
        observation_store.put_case(case.clone()).unwrap();
        observation_store
            .upsert_metal(MetalObservation {
                timing: Some(crate::observation::test_timing()),
                case_id: case.case_id.clone(),
                air_sha256: case.air_sha256.clone(),
                input_sha256: case.computed_input_sha256().unwrap(),
                metal_output_sha256: crate::hash::sha256_bytes(&[42, 0, 0, 0]),
                output_b64: "KgAAAA==".into(),
                environment_id: "env".into(),
                environment: serde_json::json!({}),
                oracle_abi: "v1".into(),
                status: MetalStatus::Qualified,
            })
            .unwrap();
        fs::rename(
            observation_scratch
                .path()
                .join("observations/metal")
                .join(shard_name(actual)),
            observation_scratch
                .path()
                .join("observations/metal")
                .join(shard_name(wrong)),
        )
        .unwrap();
        assert!(observation_store
            .read_metal()
            .unwrap_err()
            .contains("belongs in shard"));
    }
}
