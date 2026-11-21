//! The Metal reference C3 judges each case against.
//!
//! **The reference environment is data** ([`References`], `validation/cutover/c3-references.json`):
//! the Metal environment whose answers define C3 (the Apple M3 Max), with its declared identity.
//! An environment's ID is produced from its identity ([`MetalIdentity::environment_id`]), so
//! identical hosts share one environment, and another host of the same identity is one more host,
//! never another entry.
//!
//! **Which row judges a case** ([`Judge::judging`]): the reference environment's current row, when
//! it recorded the declared identity. Every reader of a candidate row's Metal dependency (C3, the
//! evidence census, coverage) judges by this one row, through the one currency rule
//! (`index::candidate_staleness`): the rule checks the Metal row's input and output, not its
//! environment.
//!
//! **The declared identity is evidence about the machine**: the device, GPU cores, macOS version
//! and build, Xcode build and Metal toolchain. A host of another identity cannot record under the
//! reference's ID ([`admit_recording`]). The reference's rows recorded before the oracle recorded
//! its toolchain state only the device and the OS; they are held to those.

use super::AnswerGap;
use crate::case::AuthoredCase;
use crate::cutover::list_path;
use crate::metal::ORACLE_ABI;
use crate::observation::{MetalObservation, ReferenceGap};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Where a reader finds the answer gaps a case's structure meets ([`AnswerGap::met_by`]).
pub trait CaseFacts {
    /// The answer gaps the case's structure meets, `None` where its structure is unknown.
    fn answer_gaps(&self, case: &AuthoredCase) -> Option<BTreeSet<AnswerGap>>;
}

/// A case's structural facts, derived from one check of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Facts {
    answer_gaps: Option<BTreeSet<AnswerGap>>,
}

/// The key of the case facts the index keeps (table `case_reference_facts`): a case's facts are a
/// function of its content, which its ID hashes, of the content-addressed modules it names, and
/// of the code that checks it and derives them: the product the check calls (its sources'
/// fingerprint) and this crate (every source of it). An edit to either reads every kept fact as
/// absent, and the next reader derives it again.
pub const CASE_FACTS_ABI: &str = concat!(
    "case-reference-facts-v2:",
    env!("METAL2VULKAN_PRODUCT_FINGERPRINT"),
    ":",
    env!("METAL2VULKAN_VALIDATION_FINGERPRINT")
);

/// The index's table of case facts ([`CheckedFacts`]), added without a rebuild: a reader
/// creates it. Only facts of the current [`CASE_FACTS_ABI`] are kept: a reader deletes the rest,
/// so the table holds at most one row per case.
const CASE_FACTS_TABLE: &str = "CREATE TABLE IF NOT EXISTS case_reference_facts (
       case_id TEXT NOT NULL,
       abi TEXT NOT NULL,
       facts TEXT NOT NULL,
       PRIMARY KEY(case_id, abi)
     );";

/// The answer gaps each case meets, derived by checking it against its source under the corpus
/// root `root` ([`crate::check::check_case`], [`AnswerGap::met_by_case`]), each case at most once,
/// when a reader first asks. A case that does not check, or whose source the root lacks (a
/// checkout without the private store), is unknown.
///
/// The facts of a case that checks are kept in the index the reader is given (table
/// `case_reference_facts`, under [`CASE_FACTS_ABI`]), so a later reader of the same build reads
/// them instead of checking the case again. A reader given no index, or one that does not exist or
/// cannot be opened, keeps nothing, and a case that does not check is not kept (its source may only
/// be absent here).
pub struct CheckedFacts {
    root: PathBuf,
    cache: RefCell<HashMap<String, Facts>>,
    index: Option<rusqlite::Connection>,
    /// Facts checked here and not yet kept: (case ID, facts as JSON), written in one transaction
    /// when the reader is dropped.
    unkept: RefCell<Vec<(String, String)>>,
    checked: std::cell::Cell<usize>,
}

impl CheckedFacts {
    /// A reader of the cases under `root`, keeping their facts in the index at `index` (the
    /// corpus's, [`crate::index::default_index_path`]) when one is given.
    pub fn new(root: &Path, index: Option<&Path>) -> CheckedFacts {
        CheckedFacts {
            root: root.into(),
            cache: RefCell::new(HashMap::new()),
            index: index.and_then(open_case_facts),
            unkept: RefCell::new(Vec::new()),
            checked: std::cell::Cell::new(0),
        }
    }

    /// How many of them were checked here, not read from the index.
    pub fn checked(&self) -> usize {
        self.checked.get()
    }

    /// The facts of `case`: from this reader, else the index, else checking it.
    fn facts(&self, case: &AuthoredCase) -> Facts {
        if let Some(facts) = self.cache.borrow().get(&case.case_id) {
            return facts.clone();
        }
        let facts = match self.kept(case) {
            Some(facts) => facts,
            None => self.check(case),
        };
        self.cache
            .borrow_mut()
            .insert(case.case_id.clone(), facts.clone());
        facts
    }

    /// The facts the index keeps of `case` under this build's key.
    fn kept(&self, case: &AuthoredCase) -> Option<Facts> {
        let facts: String = self
            .index
            .as_ref()?
            .query_row(
                "SELECT facts FROM case_reference_facts WHERE case_id=?1 AND abi=?2",
                [case.case_id.as_str(), CASE_FACTS_ABI],
                |row| row.get(0),
            )
            .ok()?;
        serde_json::from_str(&facts).ok()
    }

    /// Check `case` and derive its facts, keeping those of a case that checks in the index.
    fn check(&self, case: &AuthoredCase) -> Facts {
        self.checked.set(self.checked.get() + 1);
        match crate::check::check_case(&self.root, case.clone()) {
            Ok(checked) => {
                let facts = Facts {
                    answer_gaps: Some(AnswerGap::met_by_case(&checked)),
                };
                if let (Some(_), Ok(json)) = (&self.index, serde_json::to_string(&facts)) {
                    self.unkept.borrow_mut().push((case.case_id.clone(), json));
                }
                facts
            }
            Err(_) => Facts { answer_gaps: None },
        }
    }
}

impl Drop for CheckedFacts {
    /// Keep the facts checked here in the index, in one transaction. A fact that cannot be kept
    /// is checked again by the next reader.
    fn drop(&mut self) {
        let unkept = self.unkept.get_mut();
        let Some(index) = self.index.as_mut().filter(|_| !unkept.is_empty()) else {
            return;
        };
        let Ok(transaction) = index.transaction() else {
            return;
        };
        for (case_id, facts) in unkept.iter() {
            let _ = transaction.execute(
                "INSERT OR REPLACE INTO case_reference_facts (case_id, abi, facts) \
                 VALUES (?1, ?2, ?3)",
                [case_id.as_str(), CASE_FACTS_ABI, facts.as_str()],
            );
        }
        let _ = transaction.commit();
    }
}

/// The index at `index`, opened to keep case facts ([`CASE_FACTS_TABLE`]) with the facts of
/// other builds deleted, or `None` where there is no index or it cannot be opened or written.
fn open_case_facts(index: &Path) -> Option<rusqlite::Connection> {
    if !index.is_file() {
        return None;
    }
    let connection = rusqlite::Connection::open(index).ok()?;
    connection
        .busy_timeout(std::time::Duration::from_secs(30))
        .ok()?;
    connection.execute_batch(CASE_FACTS_TABLE).ok()?;
    connection
        .execute(
            "DELETE FROM case_reference_facts WHERE abi != ?1",
            [CASE_FACTS_ABI],
        )
        .ok()?;
    Some(connection)
}

impl CaseFacts for CheckedFacts {
    fn answer_gaps(&self, case: &AuthoredCase) -> Option<BTreeSet<AnswerGap>> {
        self.facts(case).answer_gaps
    }
}

/// A Metal environment's identity: the device, the OS, and the toolchain the oracle compiles with.
/// The Metal oracle records it in each row's `environment` (`metal::platform::environment`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetalIdentity {
    /// `MTLDevice.name`.
    pub device: String,
    /// The GPU's core count (the AGX accelerator's `gpu-core-count`).
    pub gpu_cores: u32,
    /// macOS's version (`sw_vers -productVersion`).
    pub os: String,
    /// macOS's build (`sw_vers -buildVersion`).
    pub os_build: String,
    /// Xcode's build (`xcodebuild -version`).
    pub xcode_build: String,
    /// The Metal toolchain (`xcrun metal --version`, its first line).
    pub metal_toolchain: String,
}

/// The identity members a legacy row records, which every row must record: the device and the OS.
const RECORDED_ALWAYS: [&str; 2] = ["device", "os"];

impl MetalIdentity {
    /// The environment ID the identity produces: the device's words in lower case joined by `-`,
    /// then `-macos` and the OS version (`Apple M3 Max`, `26.7`: `apple-m3-max-macos26.7`).
    pub fn environment_id(&self) -> String {
        let device = self
            .device
            .split_whitespace()
            .map(str::to_ascii_lowercase)
            .collect::<Vec<_>>()
            .join("-");
        format!("{device}-macos{}", self.os)
    }

    fn members(&self) -> [(&'static str, Value); 6] {
        [
            ("device", Value::from(self.device.clone())),
            ("gpu_cores", Value::from(self.gpu_cores)),
            ("os", Value::from(self.os.clone())),
            ("os_build", Value::from(self.os_build.clone())),
            ("xcode_build", Value::from(self.xcode_build.clone())),
            ("metal_toolchain", Value::from(self.metal_toolchain.clone())),
        ]
    }

    /// The identity a row's `environment` records, when it records every member.
    pub fn recorded(environment: &Value) -> Option<MetalIdentity> {
        let text = |member: &str| Some(environment.get(member)?.as_str()?.to_string());
        Some(MetalIdentity {
            device: text("device")?,
            gpu_cores: u32::try_from(environment.get("gpu_cores")?.as_u64()?).ok()?,
            os: text("os")?,
            os_build: text("os_build")?,
            xcode_build: text("xcode_build")?,
            metal_toolchain: text("metal_toolchain")?,
        })
    }

    /// The identity a row's `environment` records, as the JSON members the oracle writes.
    pub fn record(&self) -> serde_json::Map<String, Value> {
        self.members()
            .into_iter()
            .map(|(member, value)| (member.to_string(), value))
            .collect()
    }

    /// Each member a row's `environment` records otherwise than `self` declares, as `member:
    /// declared (recorded)`; a member it does not record is a difference when `strict`, and the
    /// device and the OS always (every row records them).
    pub fn differences(&self, environment: &Value, strict: bool) -> Vec<String> {
        self.members()
            .into_iter()
            .filter_map(|(member, declared)| match environment.get(member) {
                Some(recorded) if *recorded == declared => None,
                Some(recorded) => Some(format!("{member}: {declared} ({recorded})")),
                None if strict || RECORDED_ALWAYS.contains(&member) => {
                    Some(format!("{member}: {declared} (unrecorded)"))
                }
                None => None,
            })
            .collect()
    }
}

/// The GPU's core count in what `ioreg -r -c AGXAccelerator -d 1` prints (`"gpu-core-count" =
/// 40`): one count, however many accelerator entries state it.
pub fn parse_gpu_cores(ioreg: &str) -> Result<u32, String> {
    let counts = ioreg
        .lines()
        .filter_map(|line| line.trim().strip_prefix("\"gpu-core-count\""))
        .map(|rest| {
            rest.trim()
                .strip_prefix('=')
                .map(str::trim)
                .and_then(|count| count.parse::<u32>().ok())
                .ok_or_else(|| format!("ioreg states gpu-core-count as {rest:?}"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    match counts.into_iter().collect::<Vec<_>>()[..] {
        [count] => Ok(count),
        [] => Err("ioreg states no AGX accelerator's gpu-core-count".into()),
        ref counts => Err(format!("ioreg states several GPU core counts {counts:?}")),
    }
}

/// The Xcode build in an Xcode's `version.plist` (its `ProductBuildVersion`).
pub fn parse_xcode_build(plist: &str) -> Result<String, String> {
    let (_, after) = plist
        .split_once("<key>ProductBuildVersion</key>")
        .ok_or("version.plist states no ProductBuildVersion")?;
    after
        .trim_start()
        .strip_prefix("<string>")
        .and_then(|value| value.split_once("</string>"))
        .map(|(build, _)| build.trim().to_string())
        .filter(|build| !build.is_empty())
        .ok_or_else(|| "version.plist's ProductBuildVersion is not a string".into())
}

/// The Metal toolchain `metal --version` names: its first line (`Apple metal version 32023.921
/// (metalfe-32023.921.6)`). The lines after it name the host's target triple, which the OS
/// version already states.
pub fn parse_metal_toolchain(version: &str) -> Result<String, String> {
    version
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .filter(|line| line.contains("metal version"))
        .map(str::to_string)
        .ok_or_else(|| format!("metal --version printed no version: {version:?}"))
}

/// Refuse to record a Metal row under `environment_id` on a host whose identity `live` is not the
/// one `references` declares for it, member for member: a reference environment's rows come from
/// hosts of its identity alone, so no row of one host stands for another. An environment the list
/// does not name is never a reference, and records on any host.
pub fn admit_recording(
    references: &References,
    environment_id: &str,
    live: &MetalIdentity,
) -> Result<(), String> {
    let Some(declared) = references.declared(environment_id) else {
        return Ok(());
    };
    let differences = declared.differences(&Value::Object(live.record()), true);
    if differences.is_empty() {
        return Ok(());
    }
    Err(format!(
        "this host is not the Metal reference environment {environment_id} that the C3 reference \
         list declares (declared (this host)): {}; record under this host's own environment ID {} \
         or update the list with a new measurement",
        differences.join(", "),
        live.environment_id()
    ))
}

/// A Metal environment the reference list names: its ID and its declared identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceEnvironment {
    pub environment_id: String,
    pub identity: MetalIdentity,
}

impl ReferenceEnvironment {
    fn check(&self) -> Result<(), String> {
        let id = &self.environment_id;
        if id.trim().is_empty() || id.contains(['@', '+']) || id.contains(char::is_whitespace) {
            return Err(format!(
                "{id:?}: an environment ID has no whitespace, '@', or '+'"
            ));
        }
        let produced = self.identity.environment_id();
        if *id != produced {
            return Err(format!(
                "{id:?}: its identity produces the ID {produced:?}; an environment's ID is its \
                 identity's"
            ));
        }
        Ok(())
    }
}

/// The committed reference list (`validation/cutover/c3-references.json`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct References {
    /// The reference environment.
    pub primary: ReferenceEnvironment,
}

/// The reference list's file under `validation/cutover/`.
pub const REFERENCES_FILE: &str = "c3-references.json";

impl References {
    /// The reference environment's ID is its identity's.
    pub fn check(&self) -> Result<(), String> {
        self.primary
            .check()
            .map_err(|error| format!("C3 reference list: {error}"))
    }

    /// `validation/cutover/c3-references.json`.
    pub fn committed_path() -> PathBuf {
        list_path(REFERENCES_FILE)
    }

    /// The reference list at `path`, checked.
    pub fn load(path: &Path) -> Result<References, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        let references: References = serde_json::from_str(&text)
            .map_err(|error| format!("{}: C3 reference list: {error}", path.display()))?;
        references
            .check()
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(references)
    }

    /// The declared identity of the environment `environment_id`, when it is the reference.
    pub fn declared(&self, environment_id: &str) -> Option<&MetalIdentity> {
        (self.primary.environment_id == environment_id).then_some(&self.primary.identity)
    }
}

/// What the reference made of the store: its rows, and the current rows of other environments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceReport {
    pub environment_id: String,
    /// The reference's rows current under this oracle.
    pub current_rows: usize,
    /// Of them, rows that recorded another identity than the declared one: they judge nothing.
    pub other_identity: usize,
    /// Cases a row of the reference judges.
    pub judging: usize,
    /// Current Metal rows of environments the list does not name, by environment: never a
    /// reference.
    pub unlisted: BTreeMap<String, usize>,
}

/// The reference evaluated over a store: which row judges each case ([`Judge::judging`]).
pub struct Judge<'a> {
    references: &'a References,
}

impl<'a> Judge<'a> {
    /// Evaluate `references` over `cases` and their Metal rows `metals`.
    pub fn new(
        references: &'a References,
        cases: &[AuthoredCase],
        metals: &[MetalObservation],
    ) -> (Judge<'a>, ReferenceReport) {
        let by_case: HashMap<&str, &AuthoredCase> = cases
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect();
        let judge = Judge { references };
        let primary = &references.primary;
        let mut report = ReferenceReport {
            environment_id: primary.environment_id.clone(),
            ..ReferenceReport::default()
        };
        for metal in metals {
            let Some(case) = by_case.get(metal.case_id.as_str()) else {
                continue;
            };
            if !metal.dependency_matches(case, ORACLE_ABI) {
                continue;
            }
            if metal.environment_id != primary.environment_id {
                *report
                    .unlisted
                    .entry(metal.environment_id.clone())
                    .or_default() += 1;
                continue;
            }
            report.current_rows += 1;
            if judge.judging(&[metal]).is_some() {
                report.judging += 1;
            } else {
                report.other_identity += 1;
            }
        }
        (judge, report)
    }

    /// The Metal row a case is judged against, among `current`, its rows current under this
    /// oracle: the reference's, when it recorded the declared identity, held to its device and OS
    /// (rows recorded before the oracle recorded the toolchain state only those).
    pub fn judging<'m>(&self, current: &[&'m MetalObservation]) -> Option<&'m MetalObservation> {
        let primary = &self.references.primary;
        current.iter().copied().find(|row| {
            row.environment_id == primary.environment_id
                && primary
                    .identity
                    .differences(&row.environment, false)
                    .is_empty()
        })
    }

    /// The reference gap `case` records among its Metal rows `metals`: the reference's, from a row
    /// that recorded its declared identity; no unlisted environment's.
    pub fn reference_gap<'m>(
        &self,
        case: &AuthoredCase,
        metals: impl IntoIterator<Item = &'m MetalObservation>,
    ) -> Option<&'m ReferenceGap> {
        let primary = &self.references.primary;
        metals
            .into_iter()
            .filter(|row| {
                row.environment_id == primary.environment_id
                    && primary
                        .identity
                        .differences(&row.environment, false)
                        .is_empty()
            })
            .find_map(|row| row.reference_gap_for(case, ORACLE_ABI))
    }
}

/// Synthetic reference environments for tests that own their inputs.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// A Metal identity of `device` under macOS `os`, with fixed toolchain members.
    pub(crate) fn identity(device: &str, os: &str) -> MetalIdentity {
        MetalIdentity {
            device: device.into(),
            gpu_cores: 10,
            os: os.into(),
            os_build: "25A1".into(),
            xcode_build: "27A1".into(),
            metal_toolchain: "Apple metal version 1 (metalfe-1)".into(),
        }
    }

    /// The synthetic reference every test's Metal rows are recorded under unless they say
    /// otherwise.
    pub(crate) fn primary() -> ReferenceEnvironment {
        let identity = identity("Test GPU", "1.0");
        ReferenceEnvironment {
            environment_id: identity.environment_id(),
            identity,
        }
    }

    /// The reference list of [`primary`].
    pub(crate) fn references() -> References {
        References { primary: primary() }
    }

    /// The `environment` a row of `environment` records.
    pub(crate) fn recorded(environment: &ReferenceEnvironment) -> Value {
        Value::Object(environment.identity.record())
    }

    /// No case's facts: every case's answer gaps are unknown.
    pub(crate) struct NoFacts;

    impl CaseFacts for NoFacts {
        fn answer_gaps(&self, _: &AuthoredCase) -> Option<BTreeSet<AnswerGap>> {
            None
        }
    }
}

#[cfg(test)]
mod tests;
