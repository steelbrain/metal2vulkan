//! Whether an unchanged selection answers the same thing twice.
//!
//! Every other reading the suite takes is of **one** run. A verdict, a transcript's site and a
//! cost count are each read once and then held against a recording or a bound, so a reading that
//! moves run to run is indistinguishable from a reading that moved because the implementation
//! changed — and the gate gives the second answer, because that is the only one it can give. The
//! cost half's instability was found only because a stream re-ran a tree it had not touched and
//! noticed two different numbers; nothing in the suite was looking.
//!
//! This looks. `conformance replay --runs N` replays the selection N times in the one process and
//! flags every reading that is not the same in all N. It is off by default — `--runs 1` flattens
//! nothing and costs nothing — because the point of the default replay is one pass over a venue.
//!
//! # What a reading is
//!
//! One run of one case offers three kinds, and all three are taken:
//!
//! - its **verdict**, by its word and in full, which is what the ledger and the exit status read;
//! - every **site** of the transcript it emitted, by its path from the root, which is where a
//!   value that moves shows up;
//! - every **count** the cost half read over it, by name, with that half's own verdict.
//!
//! What is deliberately not a reading: how long the case took. A time moves on every host under
//! every load, so flagging it would bury the three readings that mean something.
//!
//! # Why the readings are held short
//!
//! A whole-set run emits sites whose value is every byte of a heap, and holding the printed form
//! of all of them for a run would cost more than the run. So a reading longer than
//! [`READING_CHARS`] is held by its length and its digest instead of its text. That is enough to
//! act on — the row names the case and the site, and the next step is to replay that one case —
//! and it keeps the flattening to a line a site whatever the sites observed.

use crate::report::Report;
use crate::transcript::{Digest, Node, Observed};
use std::collections::{BTreeMap, BTreeSet};

/// The longest reading held as its own text. Longer ones are held by their length and digest.
pub const READING_CHARS: usize = 120;

/// The reading a run that emitted no such site at all offers, so a site appearing or vanishing
/// between two runs of one selection reads as the move it is rather than as nothing.
const NO_SUCH_SITE: &str = "(the run read nothing here)";

/// One reading of one case that is not the same in every run.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Moved {
    pub case: String,
    /// What was read: `verdict`, a site by its path from the transcript's root, or a count by its
    /// name.
    pub reading: String,
    /// What each run read, run 1 first, one entry a run.
    pub runs: Vec<String>,
}

impl std::fmt::Display for Moved {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{}: {} moved", self.case, self.reading)?;
        for (at, read) in self.runs.iter().enumerate() {
            write!(out, "\n    run {}: {read}", at + 1)?;
        }
        Ok(())
    }
}

/// What several runs of one unchanged selection read, and which of those readings moved.
#[derive(Clone, Debug, Default)]
pub struct Stability {
    /// How many runs the selection was replayed. 1 means nothing was compared.
    runs: u32,
    /// The readings that moved, by the case and the reading's name, each holding one entry a run.
    ///
    /// Only the moved ones are kept, and that is what bounds this: a reading enters here the
    /// first run it differs from run 1's, and the runs before that are filled in from run 1's own
    /// reading, which they all equalled by construction.
    moved: BTreeMap<(String, String), Vec<String>>,
}

impl Stability {
    /// Nothing read yet, over a selection to be replayed `runs` times.
    pub fn new(runs: u32) -> Self {
        Self {
            runs,
            moved: BTreeMap::new(),
        }
    }

    /// Compare run `at` of the selection against run 1, where `at` counts from 2.
    ///
    /// `first` is run 1's report, read again each time rather than flattened once and kept: the
    /// report holds the transcripts anyway, so reading it costs time and no memory, while keeping
    /// a second flattened copy of a whole-set run would cost memory the run has better uses for.
    pub fn observe(&mut self, at: u32, first: &Report, again: &Report) {
        let before = readings(first);
        let now = readings(again);
        for key in before.keys().chain(now.keys()).collect::<BTreeSet<_>>() {
            let was = before.get(key).map_or(NO_SUCH_SITE, String::as_str);
            let is = now.get(key).map_or(NO_SUCH_SITE, String::as_str);
            if let Some(held) = self.moved.get_mut(key) {
                // Already moving: the runs between the one it moved at and this one read what
                // run 1 read, since a run that differed would have been appended here.
                while held.len() + 1 < at as usize {
                    held.push(was.to_owned());
                }
                held.push(is.to_owned());
                continue;
            }
            if was == is {
                continue;
            }
            // The runs up to this one all read run 1's reading: this is the first run that did
            // not, or the key would be held above.
            let mut held = vec![was.to_owned(); at as usize - 1];
            held.push(is.to_owned());
            self.moved.insert(key.clone(), held);
        }
    }

    /// Every reading that moved, with what each run read.
    ///
    /// A reading that moved before the last run and then held is filled out to the full number of
    /// runs with run 1's reading, so every row prints one entry a run.
    pub fn moved(&self) -> Vec<Moved> {
        self.moved
            .iter()
            .map(|((case, reading), read)| {
                let mut runs = read.clone();
                let last = runs.last().cloned().unwrap_or_default();
                while runs.len() < self.runs as usize {
                    runs.push(last.clone());
                }
                Moved {
                    case: case.clone(),
                    reading: reading.clone(),
                    runs,
                }
            })
            .collect()
    }

    /// Whether every reading of every run was the same.
    ///
    /// A selection replayed once is stable by this: nothing was compared, and saying otherwise
    /// would be a claim no run supports.
    pub fn is_stable(&self) -> bool {
        self.moved.is_empty()
    }

    /// What the run prints of the comparison: the rows that moved, or that none did.
    pub fn render(&self) -> String {
        if self.runs < 2 {
            return String::new();
        }
        let moved = self.moved();
        if moved.is_empty() {
            return format!(
                "\n{} runs of the selection in one process: every verdict, site and count read \
                 the same in all of them\n",
                self.runs
            );
        }
        let mut out = format!(
            "\n{} runs of the selection in one process: {} reading(s) moved between runs. A \
             reading that moves is read by a one-shot gate as a regression, so each of these is \
             a defect of the measurement or of the implementation and not of the run that \
             happened to see it:\n",
            self.runs,
            moved.len()
        );
        for row in &moved {
            out.push_str(&format!("  {row}\n"));
        }
        out
    }
}

/// Every reading one run of a selection offers, by the case and the reading's name.
fn readings(report: &Report) -> BTreeMap<(String, String), String> {
    let mut read = BTreeMap::new();
    for row in report.rows() {
        let case = row.case.clone();
        read.insert(
            (case.clone(), "verdict".to_owned()),
            held(row.verdict.word()),
        );
        read.insert(
            (case.clone(), "the verdict in full".to_owned()),
            held(&row.verdict.to_string()),
        );
        read.insert(
            (case.clone(), "allowances nothing needed".to_owned()),
            held(&row.allowances.unfired.len().to_string()),
        );
        sites(&case, "", &row.ours.root, &mut read);
    }
    // A case that could not be replayed at all is a reading too: a recording that reads stale on
    // one run and fresh on another is exactly the shape this is for.
    for error in report.errors() {
        read.insert(
            (
                error.to_string(),
                "the case could not be replayed".to_owned(),
            ),
            held("yes"),
        );
    }
    for row in report.cost_rows() {
        let case = row.case.clone();
        read.insert(
            (case.clone(), "the cost verdict".to_owned()),
            held(row.verdict.word()),
        );
        for (name, count) in row.counts.iter() {
            read.insert(
                (case.clone(), format!("the count {name}")),
                held(&count.to_string()),
            );
        }
    }
    read
}

/// Add a reading for `node` and every site under it, under `parent`'s path.
fn sites(case: &str, parent: &str, node: &Node, read: &mut BTreeMap<(String, String), String>) {
    let path = if parent.is_empty() {
        node.name.to_string()
    } else {
        format!("{parent}/{}", node.name)
    };
    read.insert(
        (case.to_owned(), format!("the site {path}")),
        held(&observed(&node.observed)),
    );
    for child in &node.children {
        sites(case, &path, child, read);
    }
}

/// What one site observed, faithfully: the serialization the recordings themselves are written
/// in, so two readings are equal exactly where the sites observed the same thing.
///
/// Not [`crate::diff::print`], which is a **summary** for a failure message and says "4096 bytes"
/// of any 4096 bytes there are. A comparison built on it would have read every pair of differing
/// byte arrays as equal, which is the one reading this exists to take.
fn observed(site: &Observed) -> String {
    match serde_json::to_string(site) {
        Ok(written) => written,
        // A site whose value cannot be written down is itself a reading: it must not read as
        // equal to the next run's, whatever that one managed.
        Err(why) => format!("this site could not be written down: {why}"),
    }
}

/// How a reading is held: its own text where that is short enough to print beside the others, and
/// its length with its digest where it is not ([`READING_CHARS`]).
fn held(text: &str) -> String {
    if text.chars().count() <= READING_CHARS {
        return text.to_owned();
    }
    format!(
        "{} characters, sha256 {}",
        text.len(),
        Digest::sha256_prefix16(text.as_bytes()).hex
    )
}

#[cfg(test)]
mod tests;
