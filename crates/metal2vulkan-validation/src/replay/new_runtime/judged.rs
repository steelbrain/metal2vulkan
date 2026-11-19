//! The matrix's judged metrics of a play on the new runtime (`docs/REPLAY-STREAM.md`, section 10):
//! counts two plays of one stream at one commit on one card give alike, summed over the whole
//! play as the runtime executed it, each an `M2V-MATRIX metric` record that the matrix holds equal
//! to its base's (`docs/VALIDATION.md`, "Records and rules"). A play that did not finish its range
//! gives none, so the base's records it lacks fail it.

use crate::replay::player::{PlayError, Report};
use crate::replay::range::stem;
use std::path::Path;

/// Each judged metric: its name, the executed counters it adds and the one it takes away. A new
/// texture is zeroed by a clear or made zero-initialized, and the split between the two follows
/// where the pool places it, which may differ between two plays: the two are counted as one, and
/// the barriers leave out the one after an upload's clears (`texture_clear_barriers`).
const JUDGED: [(&str, &[&str], Option<&str>); 7] = [
    ("submissions", &["submissions"], None),
    ("passes", &["passes"], None),
    ("barriers", &["barriers"], Some("texture_clear_barriers")),
    ("bytes_copied", &["bytes_copied"], None),
    ("pipelines_created", &["pipelines_created"], None),
    (
        "new_textures_zeroed",
        &["texture_clears", "zeroed_textures"],
        None,
    ),
    (
        "new_texture_zeroed_bytes",
        &["texture_clear_bytes", "zeroed_texture_bytes"],
        None,
    ),
];

/// The judged metrics' records of the play of the stream whose first segment is `first`, with
/// `report` and end `ended`, through frame `last` (`None`: the whole stream), one line each:
/// `M2V-MATRIX metric replay-rt/STEM/RANGE/NAME N`, `STEM` the first segment's (whitespace made
/// `_`), `RANGE` `through-F` or `whole`, and `N` the play's executed total
/// ([`Report::total`]). A play that stopped, one that ended before frame `last`'s present, and one
/// whose backend skipped anything (no gate result) give none.
pub fn judged_records(
    first: &Path,
    report: &Report,
    ended: &Result<(), PlayError>,
    last: Option<usize>,
) -> Vec<String> {
    let finished = ended.is_ok()
        && report.first_skip.is_none()
        && last.is_none_or(|last| report.frames.len() > last);
    if !finished {
        return Vec::new();
    }
    let stream = stem(first).replace(char::is_whitespace, "_");
    let range = match last {
        Some(last) => format!("through-{last}"),
        None => "whole".to_owned(),
    };
    let executed = &report.total.executed;
    let count = |name: &str| executed.get(name).copied().unwrap_or(0);
    let records = JUDGED.iter().map(|(name, added, taken)| {
        let added: u64 = added.iter().map(|&counter| count(counter)).sum();
        let value = added.saturating_sub(taken.map_or(0, count));
        format!("M2V-MATRIX metric replay-rt/{stream}/{range}/{name} {value}")
    });
    records.collect()
}
