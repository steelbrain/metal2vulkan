//! The parity records (DESIGN D308 item 7): what a scenario observed, record by record, one line
//! per Metal environment in `corpus/observations/parity/<scenario>.jsonl`, and the exact
//! comparison of a run's records with them.
//!
//! A line carries the [`Fingerprint`] of what produced it. A compare against a line whose
//! fingerprint differs from the current one fails as stale: the scenario, the harness's Metal
//! side or a fixture changed since Metal answered.

use crate::hash::sha256_bytes;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The most bytes a [`Value::Bytes`] record also keeps in hex, for a readable difference.
const INLINE: usize = 256;

/// What one record says.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub(crate) enum Value {
    /// The call was taken.
    Done,
    /// The call was refused. `detail` (Metal's assertion, the runtime's error) is not compared.
    Refused { refusal: Refusal, detail: String },
    /// Bytes read back: their length and SHA-256, and up to [`INLINE`] bytes in hex.
    Bytes {
        len: u64,
        sha256: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hex: Option<String>,
    },
    /// A command buffer's status once it completed: `completed` or `error`.
    Completed { status: String },
}

/// How a call was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Refusal {
    /// Metal's API validation stops the call; the runtime refuses it as `Invalid`.
    Stopped,
    /// The runtime lacks what the call needs (a device gap, a later step, an Apple-only call): a
    /// gap, never a match.
    Gap,
    /// Anything else: Metal answered nil, or the runtime failed otherwise.
    Failed,
}

impl Value {
    /// The record of `bytes`.
    pub(crate) fn bytes(bytes: &[u8]) -> Value {
        Value::Bytes {
            len: bytes.len() as u64,
            sha256: sha256_bytes(bytes),
            hex: (bytes.len() <= INLINE).then(|| hex(bytes)),
        }
    }

    /// Whether `self` says what `other` says: the details of a refusal and the hex of bytes
    /// aside, which their other fields determine or which are not compared. A gap matches
    /// nothing.
    fn same(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Done, Value::Done) => true,
            (Value::Refused { refusal: a, .. }, Value::Refused { refusal: b, .. }) => {
                a == b && *a != Refusal::Gap
            }
            (
                Value::Bytes {
                    len: a, sha256: x, ..
                },
                Value::Bytes {
                    len: b, sha256: y, ..
                },
            ) => a == b && x == y,
            (Value::Completed { status: a }, Value::Completed { status: b }) => a == b,
            _ => false,
        }
    }
}

/// Lowercase hex of `bytes`.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// What a line's records depend on. Any change stales the line.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Fingerprint {
    /// The Metal environment that answered (`MetalIdentity::environment_id`).
    pub(crate) metal_environment: String,
    pub(crate) scenario: String,
    pub(crate) scenario_version: u32,
    /// SHA-256 of the scenario's source and the harness sources its records depend on.
    pub(crate) sources_sha256: String,
    /// Each fixture the scenario loads, by its committed case.
    pub(crate) fixtures: Vec<FixtureDigest>,
}

/// A fixture a scenario loads: its committed case's name, the SHA-256 of its AIR (what the
/// runtime translates), and the SHA-256 of the file Metal compiles it from (its typed AIR text or
/// its Metal source, `source::public_metal_input`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FixtureDigest {
    pub(crate) case: String,
    pub(crate) air_sha256: String,
    pub(crate) metal_input_sha256: String,
}

/// One line of a records file: one Metal environment's answers to one scenario.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Line {
    pub(crate) fingerprint: Fingerprint,
    /// Each record's name (`case/call`) and value, in the order the scenario made them.
    pub(crate) records: Vec<(String, Value)>,
}

/// The lines of the records file at `path`; none when it does not exist.
pub(crate) fn read_lines(path: &Path) -> Result<Vec<Line>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("read {}: {error}", path.display())),
    };
    text.lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line)
                .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1))
        })
        .collect()
}

/// Put `line` in the records file at `path`, in place of its environment's line, the lines
/// sorted by environment, written to a new file that then replaces the old.
pub(crate) fn write_line(path: &Path, line: Line) -> Result<(), String> {
    let mut lines: Vec<Line> = read_lines(path)?
        .into_iter()
        .filter(|kept| kept.fingerprint.metal_environment != line.fingerprint.metal_environment)
        .collect();
    lines.push(line);
    lines.sort_by(|a, b| {
        a.fingerprint
            .metal_environment
            .cmp(&b.fingerprint.metal_environment)
    });
    let mut text = String::new();
    for line in &lines {
        text.push_str(&serde_json::to_string(line).map_err(|error| error.to_string())?);
        text.push('\n');
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    }
    let new = path.with_extension("jsonl.new");
    std::fs::write(&new, text).map_err(|error| format!("write {}: {error}", new.display()))?;
    std::fs::rename(&new, path).map_err(|error| format!("rename {}: {error}", new.display()))
}

/// Compare `records`, a run's, with `lines`' line for `current`'s environment: it must exist,
/// carry `current` as its fingerprint, and say record by record what `records` say, in order.
/// Each difference is named.
pub(crate) fn compare(
    lines: &[Line],
    current: &Fingerprint,
    records: &[(String, Value)],
) -> Result<(), String> {
    let Some(line) = lines
        .iter()
        .find(|line| line.fingerprint.metal_environment == current.metal_environment)
    else {
        return Err(format!(
            "{}: no record of {}",
            current.scenario, current.metal_environment
        ));
    };
    let stale = stale(&line.fingerprint, current);
    if !stale.is_empty() {
        return Err(format!(
            "{}: the record of {} is stale ({}): record it again",
            current.scenario,
            current.metal_environment,
            stale.join(", ")
        ));
    }
    let mut differences = Vec::new();
    let longest = line.records.len().max(records.len());
    for index in 0..longest {
        match (line.records.get(index), records.get(index)) {
            (Some((name, metal)), Some((ran, value))) if name == ran => {
                if !metal.same(value) {
                    differences.push(format!("{name}: Metal {metal:?}, here {value:?}"));
                }
            }
            (Some((name, metal)), Some((ran, value))) => differences.push(format!(
                "record {index}: Metal {name} {metal:?}, here {ran} {value:?}"
            )),
            (Some((name, metal)), None) => {
                differences.push(format!("{name}: Metal {metal:?}, here no record"));
            }
            (None, Some((ran, value))) => {
                differences.push(format!("{ran}: no Metal record, here {value:?}"));
            }
            (None, None) => {}
        }
    }
    if differences.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} differs from {}:\n{}",
            current.scenario,
            current.metal_environment,
            differences.join("\n")
        ))
    }
}

/// What differs between a line's fingerprint and the current one, by name.
pub(crate) fn stale(recorded: &Fingerprint, current: &Fingerprint) -> Vec<&'static str> {
    let mut stale = Vec::new();
    if recorded.scenario != current.scenario {
        stale.push("scenario");
    }
    if recorded.scenario_version != current.scenario_version {
        stale.push("scenario_version");
    }
    if recorded.sources_sha256 != current.sources_sha256 {
        stale.push("sources");
    }
    if recorded.fixtures != current.fixtures {
        stale.push("fixtures");
    }
    stale
}
