//! The report over a selection: the verdict counts, and the matrix of case against venue.
//!
//! Errors are counted apart from verdicts. A stale recording is not a verdict, and printing it as
//! one would let it read as coverage.

use crate::compare::Unfired;
use crate::cost::{self, CostVerdict, Counts, Region};
use crate::known_gaps::{against, Expectation, KnownGap};
use crate::replay::{ReplayError, Replayed};
use crate::verdict::Verdict;
use std::collections::{BTreeMap, BTreeSet};

/// What the cost half says of one case on the venue the run was on: its own verdict, kept apart
/// from the correctness one, and the counts behind it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CostRow {
    pub case: String,
    pub verdict: CostVerdict,
    /// The counts the case's run moved, which the verdict was taken over.
    pub counts: Counts,
    /// How long the case's run took, in nanoseconds. **Advisory**: gate on counters, report on
    /// times — a time is not assertable on any host under any load, so none is ever a verdict
    /// here. Reported so a reader can see where the counts are large
    /// ([`cost::CaseCost::nanos`]).
    pub nanos: u64,
    /// How many of the case's bounds are reported rather than gated, and why, in three groups: the
    /// race rule's, the ones whose readings moved between runs, and the ones the selection decides
    /// ([`cost::CostRecord::composed`]). The gate's own honesty number: it says how much of the
    /// counted surface no bound holds.
    pub reported: [usize; 3],
}

/// A report over a selection of cases and venues.
#[derive(Clone, Debug, Default)]
pub struct Report {
    rows: Vec<Replayed>,
    errors: Vec<ReplayError>,
    /// What each venue announced as its device was made: the venue's own notices, which belong to
    /// no case. They are reported, never compared: a case's notices are the ones its own work
    /// announced.
    venue_notices: Vec<(String, Vec<String>)>,
    /// Every degrade notice that arrived with no case's notice scope open
    /// ([`crate::device::Notices::unattributed`]).
    ///
    /// **The harness's own gap, never a case's difference.** A notice the runtime delivers off the
    /// calling thread, or as a case's last resource is let go, can arrive after the case that
    /// provoked it has been compared. Handing it to whichever case asked next is what made a
    /// verdict depend on the selection, so it is reported here instead: the owner's rule is that
    /// an accuracy note is logged and announced, never dropped, and no case is charged for it.
    unattributed: Vec<crate::device::Unattributed>,
    /// What the cost half says of each case ([`crate::cost`]). Its own list and its own verdicts:
    /// a count has no Metal counterpart, so it never touches a correctness row.
    costs: Vec<CostRow>,
}

impl Report {
    /// An empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record what `venue` announced as its device was made.
    pub fn venue_announced(&mut self, venue: &str, codes: Vec<String>) {
        self.venue_notices.push((venue.to_owned(), codes));
    }

    /// Record the notices the run could not attribute to a case.
    pub fn unattributed_notices(&mut self, notices: Vec<crate::device::Unattributed>) {
        self.unattributed = notices;
    }

    /// The notices the run could not attribute to a case.
    pub fn unattributed(&self) -> &[crate::device::Unattributed] {
        &self.unattributed
    }

    /// Add what replaying one case on one venue came to.
    pub fn add(&mut self, row: Replayed) {
        self.rows.push(row);
    }

    /// Add a case that could not be replayed.
    pub fn add_error(&mut self, error: ReplayError) {
        self.errors.push(error);
    }

    /// Add either.
    pub fn add_result(&mut self, result: Result<Replayed, ReplayError>) {
        match result {
            Ok(row) => self.add(row),
            Err(error) => self.add_error(error),
        }
    }

    /// Add what the cost half says of one case.
    pub fn add_cost(&mut self, row: CostRow) {
        self.costs.push(row);
    }

    /// What the cost half says of each case.
    pub fn cost_rows(&self) -> &[CostRow] {
        &self.costs
    }

    /// How many cost rows hold each of the three cost verdicts.
    pub fn cost_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for row in &self.costs {
            *counts.entry(row.verdict.word()).or_default() += 1;
        }
        counts
    }

    /// Every case that did more work than its bounds allow.
    pub fn cost_overruns(&self) -> Vec<&CostRow> {
        self.costs
            .iter()
            .filter(|row| !row.verdict.is_within())
            .collect()
    }

    /// Whether every cost row is within its bounds.
    ///
    /// Its own gate, kept apart from [`Report::is_compliant`]: a count is ours and not Metal's, so
    /// going over a bound is never a conformance difference, and a conformance difference is never
    /// a cost result. An unmeasured row does not fail this; it is counted as the gap it is.
    pub fn cost_is_within(&self) -> bool {
        self.costs.iter().all(|row| row.verdict.is_within())
    }

    /// The rows.
    pub fn rows(&self) -> &[Replayed] {
        &self.rows
    }

    /// The cases that could not be replayed.
    pub fn errors(&self) -> &[ReplayError] {
        &self.errors
    }

    /// How many rows hold each verdict, by the verdict's word.
    pub fn counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for row in &self.rows {
            *counts.entry(row.verdict.word()).or_default() += 1;
        }
        counts
    }

    /// Whether the selection is compliant: no divergence, no refusal, no unrecorded case and no
    /// error. A venue that lacks a feature does not spoil it; a case nobody recorded does.
    pub fn is_compliant(&self) -> bool {
        self.errors.is_empty() && self.rows.iter().all(|row| row.verdict.is_compliant())
    }

    /// How the run's refusals stand against the known-gaps `ledger`: nothing where the refused
    /// set is exactly the ledger's rows over the cases the run selected, and otherwise every way
    /// they differ ([`crate::known_gaps`]).
    ///
    /// This is what the command exits by. It does not touch a verdict: a refusal the ledger
    /// expects is still a `Refused` row in the counts and is still printed in full, because the
    /// ledger's rows are open bugs rather than accepted differences.
    pub fn unexpected(&self, ledger: &[KnownGap]) -> Vec<Expectation> {
        let mut found = against(&self.rows, ledger);
        for error in &self.errors {
            found.push(Expectation::NotLedgerable {
                case: "a case that could not be replayed".to_owned(),
                why: error.to_string(),
            });
        }
        found
    }

    /// Every row whose verdict is not one a compliant run may hold, for a test's failure message.
    pub fn failures(&self) -> Vec<&Replayed> {
        self.rows
            .iter()
            .filter(|row| !row.verdict.is_compliant())
            .collect()
    }

    /// How many allowances the selection named, and how many of them no row needed.
    ///
    /// A tolerance that never fires is the suite's blind spot made visible. The up-front check
    /// over a case's allowances cannot catch it — a degrade notice is announced once per device
    /// per kind, so an allowance naming a notice this run did not announce has to be allowed to
    /// stand — which means a case could name a tolerance no venue ever justifies and no run would
    /// say a word. The day one did fire, the row would read "announced, justified" with nobody
    /// having checked whether the tolerance was right.
    ///
    /// It does not fail a run, because an allowance may legitimately be venue-specific. What the
    /// report calls out is the stronger shape: [`Report::unfired_everywhere`].
    pub fn allowance_counts(&self) -> (usize, usize) {
        let named = self.rows.iter().map(|row| row.allowances.named).sum();
        let unfired = self
            .rows
            .iter()
            .map(|row| row.allowances.unfired.len())
            .sum();
        (named, unfired)
    }

    /// The allowances no row of the report needed, by case and path: those that fired on no venue
    /// of this run at all, and what the case named to justify each.
    ///
    /// An allowance that fires on lavapipe and not on MoltenVK is doing its job; one that fires
    /// nowhere is a claim this run cannot vouch for. The key is (case, path, justification),
    /// because the same case may name several tolerances at one site — a notice is announced once
    /// per device per kind, so which of them a run sees depends on what ran before it — and each
    /// is its own claim.
    pub fn unfired_everywhere(&self) -> Vec<(&str, &Unfired)> {
        // A row says only which of its case's allowances went **unused**, so a claim fired on a
        // row where that row compared at all and the claim is absent from its unfired list. A row
        // that compared nothing — a venue that lacks the feature, a case with no answer — settles
        // nothing either way, which is why `named` is read and not just the list's length.
        let fired: BTreeSet<(&str, &str, &str)> = self
            .rows
            .iter()
            .filter(|row| row.allowances.named > 0)
            .flat_map(|row| {
                let unused: BTreeSet<(&str, &str)> = row
                    .allowances
                    .unfired
                    .iter()
                    .map(|claim| (claim.path.as_str(), claim.justified_by.as_str()))
                    .collect();
                self.rows
                    .iter()
                    .filter(move |other| other.case == row.case)
                    .flat_map(|other| &other.allowances.unfired)
                    .filter(move |claim| {
                        !unused.contains(&(claim.path.as_str(), claim.justified_by.as_str()))
                    })
                    .map(move |claim| {
                        (
                            row.case.as_str(),
                            claim.path.as_str(),
                            claim.justified_by.as_str(),
                        )
                    })
            })
            .collect();
        let mut everywhere: Vec<(&str, &Unfired)> = Vec::new();
        for row in &self.rows {
            for claim in &row.allowances.unfired {
                let key = (
                    row.case.as_str(),
                    claim.path.as_str(),
                    claim.justified_by.as_str(),
                );
                let already = everywhere.iter().any(|(case, held)| {
                    (*case, held.path.as_str(), held.justified_by.as_str()) == key
                });
                if !fired.contains(&key) && !already {
                    everywhere.push((row.case.as_str(), claim));
                }
            }
        }
        everywhere
    }

    /// The allowances no row needed **and** no row could justify: the case names a tolerance, no
    /// site of the run differed inside it, and the notice or entry it names did not hold either.
    ///
    /// This is the speculative shape at its plainest, and the one worth a stronger word than the
    /// rest. Nothing in the run connects the tolerance to anything: it is a sentence about a
    /// device that was not here. Two allowances of exactly this shape were written and deleted by
    /// hand in this round, which is why the suite now counts them.
    ///
    /// It is still **reported, not fatal**. A degrade notice fires on the device that needs it, so
    /// a tolerance for a notice no venue of this run announces may be right about another venue;
    /// what the suite can say honestly is that no run has checked it. Making it fatal would
    /// instead force a case to carry only the tolerances this Mac happens to need, which is a
    /// worse rule: it would read as proof of a device nobody tested.
    pub fn unchecked_everywhere(&self) -> Vec<(&str, &Unfired)> {
        self.unfired_everywhere()
            .into_iter()
            .filter(|(case, claim)| {
                // A row that did justify the claim settles it, whether or not it fired there: the
                // notice was announced, so the tolerance is about something that happened.
                self.rows.iter().all(|row| {
                    row.case != *case
                        || row.allowances.unfired.iter().all(|held| {
                            held.path != claim.path
                                || held.justified_by != claim.justified_by
                                || held.why.is_some()
                        })
                })
            })
            .collect()
    }

    /// Each case's allowances, as (fired, named, never fired), for the cases that named any.
    ///
    /// Per case rather than only in total, because the total says a number and this says which
    /// case's claim it is. A case appearing once per venue is summed over its venues.
    pub fn allowances_by_case(&self) -> BTreeMap<&str, (usize, usize, usize)> {
        let mut by_case: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
        for row in &self.rows {
            if row.allowances.named == 0 {
                continue;
            }
            let held = by_case.entry(row.case.as_str()).or_default();
            held.0 += row.allowances.fired();
            held.1 += row.allowances.named;
            held.2 += row.allowances.unfired.len();
        }
        by_case
    }

    /// How many of the report's rows were answered from a chip that is not the oracle's: a carried
    /// measurement taken on a machine this round cannot re-record on.
    pub fn carried_from_another_chip(&self) -> BTreeMap<&str, usize> {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for row in &self.rows {
            if let Some(chip) = &row.measured_on {
                *counts.entry(chip.as_str()).or_default() += 1;
            }
        }
        counts
    }

    /// Every row that differed inside an allowance, or that the venue could not run.
    ///
    /// These are compliant, and they are still the report's most important lines: the claim under
    /// test is that a difference from Metal is a bug *unless we announced it*, so a run whose
    /// counts are all green while something was loosened has to say what. A reader who cannot see
    /// which field was allowed to differ cannot check that the allowance was the right one.
    pub fn noted(&self) -> Vec<&Replayed> {
        self.rows
            .iter()
            .filter(|row| {
                matches!(
                    row.verdict,
                    Verdict::Announced { .. }
                        | Verdict::MetalNeverReturned { .. }
                        | Verdict::LacksFeature { .. }
                )
            })
            .collect()
    }

    /// The report: the counts, the matrix of case against venue, then everything that was
    /// loosened or could not be run, and then each failure in full.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("verdicts\n");
        if self.rows.is_empty() {
            out.push_str("  none\n");
        }
        for (word, count) in self.counts() {
            out.push_str(&format!("  {word:<14} {count}\n"));
        }
        if !self.errors.is_empty() {
            out.push_str(&format!("  {:<14} {}\n", "errors", self.errors.len()));
        }
        for (venue, codes) in &self.venue_notices {
            if !codes.is_empty() {
                out.push_str(&format!("\n{venue} announced as it was made: {codes:?}\n"));
            }
        }
        if !self.unattributed.is_empty() {
            out.push_str("\nnotices no case's scope was open for, which no case is charged for\n");
            for notice in &self.unattributed {
                out.push_str(&format!("  {notice}\n"));
            }
        }
        out.push_str(&self.allowances_section());
        let carried = self.carried_from_another_chip();
        if !carried.is_empty() {
            out.push_str("\ncarried from another chip\n");
            for (chip, count) in carried {
                out.push_str(&format!(
                    "  {count} row(s) answered from {chip}, which this round cannot re-record on\n"
                ));
            }
        }
        out.push_str(&self.matrix());
        // Every justified difference is printed, not only the failures. A tolerance that is never
        // read is a tolerance nobody is checking, and a refusal that Metal's own API validation
        // justifies must not read as a bare "announced" cell: the owner's rule is that a refusal
        // stays visible.
        for row in &self.rows {
            if let Verdict::Announced { fields } = &row.verdict {
                out.push_str(&format!(
                    "\n{} on {} differed, justified\n",
                    row.case, row.venue
                ));
                for field in fields {
                    out.push_str(&format!("  {field}\n"));
                }
            }
        }
        for error in &self.errors {
            out.push_str(&format!("\nerror: {error}\n"));
        }
        for row in self.noted() {
            out.push_str(&format!(
                "\n{} on {}: {}\n",
                row.case, row.venue, row.verdict
            ));
        }
        for row in self.failures() {
            out.push_str(&format!(
                "\n{} on {}: {}\n",
                row.case, row.venue, row.verdict
            ));
        }
        out.push_str(&self.render_cost());
        out
    }

    /// The cost half of the report: its own verdict counts, where the work fell by region, and
    /// every case that went over a bound in full.
    ///
    /// Empty where nothing measured the cost half, so a report of the correctness half alone reads
    /// as it did.
    pub fn render_cost(&self) -> String {
        if self.costs.is_empty() {
            return String::new();
        }
        let mut out = String::from("\ncost verdicts (ours, never Metal's: no Metal counterpart)\n");
        for (word, count) in self.cost_counts() {
            out.push_str(&format!("  {word:<14} {count}\n"));
        }
        // How far the gate reaches, so a green cost half cannot read as more than it is.
        let held: usize = self
            .costs
            .iter()
            .filter_map(|row| match &row.verdict {
                CostVerdict::Within { held } => Some(*held),
                _ => None,
            })
            .sum();
        out.push_str(&format!(
            "  {:<14} {held} count(s) held over the cases within their bounds\n",
            "gated"
        ));
        // And how much of the counted surface no bound holds, in its three groups, so a green
        // cost half is read for what it is rather than for the whole of what the runtime counts.
        let mut reported = [0usize; 3];
        for row in &self.costs {
            for (group, many) in row.reported.iter().enumerate() {
                reported[group] += many;
            }
        }
        if reported.iter().any(|many| *many > 0) {
            out.push_str(&format!(
                "  {:<14} {} more no bound holds: {} the race rule names, {} the runs disagreed \
                 on, {} the selection decides\n",
                "reported",
                reported.iter().sum::<usize>(),
                reported[0],
                reported[1],
                reported[2],
            ));
        }
        let mut sums: BTreeMap<Region, usize> = BTreeMap::new();
        for row in &self.costs {
            for (region, many) in cost::by_region(&row.counts) {
                *sums.entry(region).or_default() += many;
            }
        }
        out.push_str("\nwhere the work fell: count readings a case moved, by region\n");
        for region in Region::ALL {
            out.push_str(&format!(
                "  {:<14} {}\n",
                region.word(),
                sums.get(&region).copied().unwrap_or(0)
            ));
        }
        // An unmeasured row is a hole in the gate, and holes are tracked the way `Unrecorded` is:
        // one line each, with the reason, never a silent skip.
        let unmeasured: Vec<&CostRow> = self
            .costs
            .iter()
            .filter(|row| matches!(row.verdict, CostVerdict::Unmeasured { .. }))
            .collect();
        if !unmeasured.is_empty() {
            out.push_str(&format!(
                "\n{} case(s) have no cost bound on this venue\n",
                unmeasured.len()
            ));
            let mut whys: BTreeMap<String, Vec<&str>> = BTreeMap::new();
            for row in unmeasured {
                if let CostVerdict::Unmeasured { why } = &row.verdict {
                    whys.entry(why.clone()).or_default().push(&row.case);
                }
            }
            for (why, cases) in whys {
                out.push_str(&format!("  {} case(s): {why}\n", cases.len()));
                for case in cases {
                    out.push_str(&format!("    {case}\n"));
                }
            }
        }
        for row in self.cost_overruns() {
            match &row.verdict {
                CostVerdict::Over { over } => {
                    out.push_str(&format!("\n{} cost more than its bounds\n", row.case));
                    for one in over {
                        out.push_str(&format!("  {one}\n"));
                    }
                }
                // The opposite finding, printed in its own words: this case stopped avoiding work
                // it avoided every run of the measurement, which no upper bound can see.
                CostVerdict::Under { under } => {
                    out.push_str(&format!("\n{} saved less than its bounds\n", row.case));
                    for one in under {
                        out.push_str(&format!("  {one}\n"));
                    }
                }
                CostVerdict::Within { .. } | CostVerdict::Unmeasured { .. } => {}
            }
        }
        out
    }

    /// What the selection's allowances did: the counts, what each justification went unused for,
    /// and the per-case tally.
    ///
    /// The whole point of the section is that an allowance which never fires is otherwise
    /// invisible, so it is written to be read: the totals first, then one line per justification
    /// rather than one per site, because 575 sites naming eleven notices is eleven facts and not
    /// 575.
    pub fn allowances_section(&self) -> String {
        let (named, unfired) = self.allowance_counts();
        if named == 0 {
            return String::new();
        }
        let mut out = format!(
            "\nallowances: {named} named, {} fired, {unfired} never fired\n",
            named - unfired
        );
        let everywhere = self.unfired_everywhere();
        if everywhere.is_empty() {
            out.push_str("  every allowance named fired on some venue of this run\n");
            return out;
        }
        let unchecked = self.unchecked_everywhere();
        out.push_str(&format!(
            "  {} of them fired on no venue of this run, and of those {} name nothing this run \
             justifies either, so no run of this suite has checked them:\n",
            everywhere.len(),
            unchecked.len()
        ));
        // One line per justification: eleven notices named by sixty cases is eleven claims.
        let mut by_justification: BTreeMap<&str, (usize, &str, &str)> = BTreeMap::new();
        for (case, claim) in &unchecked {
            let held = by_justification
                .entry(claim.justified_by.as_str())
                .or_insert((0, case, claim.path.as_str()));
            held.0 += 1;
        }
        for (justification, (count, case, path)) in &by_justification {
            out.push_str(&format!(
                "    {justification}: {count} site(s), first {case} at {path}\n"
            ));
        }
        let justified = everywhere.len() - unchecked.len();
        if justified > 0 {
            out.push_str(&format!(
                "  the other {justified} name something the run did justify — the device \
                 announced the notice, for another case's work — so they are tolerances this run \
                 did not need rather than claims nothing stands behind\n"
            ));
        }
        out.push_str("  by case, fired of named:\n");
        for (case, (fired, named, never)) in self.allowances_by_case() {
            if never == 0 {
                continue;
            }
            out.push_str(&format!("    {case:<44} {fired} of {named}\n"));
        }
        out
    }

    /// The matrix of case against venue, one row per case and one column per venue.
    pub fn matrix(&self) -> String {
        let venues: BTreeSet<String> = self.rows.iter().map(|row| row.venue.label()).collect();
        let cases: BTreeSet<&str> = self.rows.iter().map(|row| row.case.as_str()).collect();
        if venues.is_empty() {
            return String::new();
        }
        let mut cells: BTreeMap<(&str, String), &'static str> = BTreeMap::new();
        for row in &self.rows {
            cells.insert((row.case.as_str(), row.venue.label()), row.verdict.word());
        }
        let case_width = cases
            .iter()
            .map(|case| case.len())
            .max()
            .unwrap_or(4)
            .max(4);
        let widths: Vec<usize> = venues
            .iter()
            .map(|venue| venue.len().max("never-returned".len()))
            .collect();
        let mut out = String::from("\nmatrix\n");
        out.push_str(&format!("  {:<case_width$}", "case"));
        for (venue, width) in venues.iter().zip(&widths) {
            out.push_str(&format!("  {venue:<width$}"));
        }
        out.push('\n');
        for case in &cases {
            out.push_str(&format!("  {case:<case_width$}"));
            for (venue, width) in venues.iter().zip(&widths) {
                let cell = cells.get(&(case, venue.clone())).copied().unwrap_or("-");
                out.push_str(&format!("  {cell:<width$}"));
            }
            out.push('\n');
        }
        out
    }
}

/// Whether `verdict` is one the suite's own tests may let stand, with its words for a failure.
pub fn must_hold(verdict: &Verdict) -> Result<(), String> {
    if verdict.is_compliant() {
        return Ok(());
    }
    Err(verdict.to_string())
}
