//! Field-by-field comparison of two transcripts, exact by default.
//!
//! The walk pairs the sides' nodes by name and descends. At a leaf it compares the value by its own
//! kind: bytes element by element under their layout, lines in order, sets without order, a variant
//! by its name and then its payload. A site a case loosened compares under its allowance, and only
//! once the allowance's justification holds ([`crate::allowance`]).
//!
//! Where Metal answered a call and we returned a typed error, the site is a **refusal**, and no
//! allowance can make it anything else. A refusal is its own verdict so that it stays visible
//! rather than being buried in a failure count, and it is never compliant, because plain Metal is
//! the behavioural oracle and the owner's standing rule is that what Metal allows, we must allow.
//! A case may name what *documents* its refusal — Metal's own API validation refusing the same
//! call is the one thing that can — and the justification is still checked, but it buys an
//! explanation in the report and not a pass. See [`crate::validated`].
//!
//! The walk also counts the allowances it **did not** use ([`Unfired`]). A tolerance no site needed
//! is not a failure — it may be venue-specific — but it is a claim no run has checked, and until it
//! was counted a case could name one indefinitely without anybody noticing.
//!
//! One rule is checked beside the walk rather than in it: **every degrade notice our side announced
//! must be one the case names**, reported as a
//! [`DivergenceKind::UndeclaredNotice`]. The notices
//! site holds the whole announced set as one value, and a declaration unchecks that site, so the
//! walk alone let one declaration excuse every other notice. The subset check is what makes the
//! suite's central sentence — a difference is a bug unless we announced it — able to fail where it
//! is stated. The same check also marks a declaration **used** where its own code was announced,
//! because the walk marked only the first covering allowance and a case's second declaration could
//! therefore never fire.

use crate::allowance::{Allowance, Loosening, Offered, Unjustified};
use crate::diff::{Divergence, DivergenceKind, ElementAt, GapAt, LineAt, Side};
use crate::transcript::{ByteLayout, Gaps, Node, Observed, Shape, Transcript, Value};

/// A site that differed inside an allowance that holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Announced {
    /// The site, as node names from the root.
    pub path: String,
    /// What the case loosened there.
    pub how: Loosening,
    /// What justifies it, in words.
    pub justified_by: String,
    /// How many sites, or elements of one site, the allowance covered.
    pub count: usize,
    /// The largest difference the allowance let through, where the values are numbers: so a bound
    /// is read off a run rather than guessed, and a bound far above what the device needs shows as
    /// one.
    pub largest: Option<f64>,
}

/// A site where Metal answered a call and we returned a typed error.
///
/// Never compliant, whatever the case names at the site. A refusal of something plain Metal does
/// is a gap of ours, and the suite's job is to say so rather than to launder it.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    /// The site, as node names from the root.
    pub path: String,
    /// Our typed error, as the report prints it.
    pub ours: String,
    /// What the case named to **document** the refusal, where it named something and that thing
    /// holds ([`Allowance::justified_by`]): the words of Metal's own API validation refusing the
    /// same call, typically. It explains the row; it does not excuse it. `None` where the case
    /// named nothing, which is a refusal nobody has accounted for at all.
    pub documented_by: Option<String>,
}

/// A site where plain Metal answered a call and we returned a typed error that is the **venue's**
/// honest gap rather than ours ([`crate::verdict::device_gap`]).
///
/// Neither a pass nor a failure, and never a [`Refusal`]: the typed error names capabilities a
/// device could enable, or a driver measured to run the thing wrong, so the device truly lacks it
/// and the owner's rule about refusing what Metal allows does not reach it. It is counted apart,
/// printed with what it lacks, and the rest of the case still compares — which is what the
/// whole-case [`LacksFeature`](crate::verdict::Verdict::LacksFeature) verdict cannot do, since
/// that one comes from the case stopping.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceGap {
    /// The site, as node names from the root.
    pub path: String,
    /// What the venue lacks, in words.
    pub needs: String,
    /// Our typed error, as the report prints it.
    pub ours: String,
}

impl std::fmt::Display for DeviceGap {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{}: the venue lacks {} ({})",
            self.path, self.needs, self.ours
        )
    }
}

/// A site where Metal did not return from the call and we gave a typed error.
///
/// Its own outcome, because it is neither of the two it would otherwise be squeezed into. It is not
/// exact: there is no value on Metal's side to match. It is not divergent: there is no behaviour of
/// Metal's to differ from — the call ended Metal's process. And it is not a refusal, which is
/// reserved for a call plain Metal **answered**.
///
/// **No allowance is consulted here**, in either direction. A case cannot loosen its way into this
/// outcome, and it cannot loosen its way out of the divergence it becomes when we answer where
/// Metal did not ([`crate::diff::DivergenceKind::AnsweredWhereMetalDidNotReturn`]).
#[derive(Clone, Debug, PartialEq)]
pub struct NeverReturned {
    /// The site, as node names from the root.
    pub path: String,
    /// Our typed error, as the report prints it.
    pub ours: String,
}

/// An allowance that does not hold: the case names it, and why it cannot be used.
#[derive(Clone, Debug, PartialEq)]
pub struct BadAllowance {
    /// The allowance's path, as the case wrote it.
    pub path: String,
    pub why: Unjustified,
}

/// An allowance nothing on the run needed: the case names a tolerance and every site it covers
/// matched exactly.
///
/// This is counted because an allowance that never fires is otherwise **invisible**. The up-front
/// check over a case's allowances exempts a degrade notice that was not announced — it has to,
/// since a notice is announced once per device per kind and the run's own notices are checked
/// where the allowance fires — so a case may name a tolerance no venue ever justifies and no run
/// says a word. The day some venue does fire it, the row reads "announced, justified" with nobody
/// ever having looked at whether the tolerance was the right one. Two such speculative allowances
/// were written and deleted by hand in this round; the suite now says so itself.
///
/// It is **reported, not failed**: an allowance may be legitimately venue-specific, firing on
/// lavapipe where MoltenVK is exact. What the report says instead is how many fired nowhere at
/// all ([`crate::report::Report::unfired_everywhere`]), which is the suspicious shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Unfired {
    /// The allowance's path, as the case wrote it.
    pub path: String,
    /// What the case loosened there.
    pub how: Loosening,
    /// What the case named to justify it, in words.
    pub justified_by: String,
    /// Why that justification does not hold on this run, where it does not. An unfired allowance
    /// whose justification is also missing is the speculative shape at its plainest: the case
    /// names a notice the run never announced *and* no site needed the tolerance, so nothing
    /// anywhere connects the two.
    pub why: Option<String>,
}

impl std::fmt::Display for Unfired {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{} ({})", self.path, self.justified_by)?;
        match &self.why {
            Some(why) => write!(out, ", and {why}"),
            None => Ok(()),
        }
    }
}

/// What comparing one run against its recording found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Comparison {
    pub announced: Vec<Announced>,
    pub divergences: Vec<Divergence>,
    pub refusals: Vec<Refusal>,
    /// Sites the venue honestly cannot run ([`DeviceGap`]). Compliant, counted apart, and never
    /// among the refusals.
    pub device_gaps: Vec<DeviceGap>,
    /// Sites where Metal did not return from the call and we gave a typed error.
    pub never_returned: Vec<NeverReturned>,
    /// Allowances whose justification is gone. Each of these fails the case on its own: a
    /// difference nobody is accounting for.
    pub bad_allowances: Vec<BadAllowance>,
    /// Allowances nothing on this run needed ([`Unfired`]). Reported, never fatal.
    pub unfired: Vec<Unfired>,
    /// How many allowances the case named, so a report can say "none of 2" rather than only "2".
    pub named: usize,
}

impl Comparison {
    /// Note that the allowance at `path` does not hold, unless the same fault is already held.
    ///
    /// The same allowance is examined twice — once before the walk, so an allowance naming a
    /// justification that is gone fails whether or not it fires, and again where it does fire —
    /// and one fault reported twice reads as two.
    fn note_bad(&mut self, path: &str, why: Unjustified) {
        if self
            .bad_allowances
            .iter()
            .any(|held| held.path == path && held.why == why)
        {
            return;
        }
        self.bad_allowances.push(BadAllowance {
            path: path.to_owned(),
            why,
        });
    }

    /// Whether every field matched exactly and every allowance held unused.
    ///
    /// A site the venue honestly lacks is not exact: Metal answered the call and we did not, so
    /// the case did not compare there. It is compliant, which is a different question.
    pub fn is_exact(&self) -> bool {
        self.announced.is_empty()
            && self.divergences.is_empty()
            && self.refusals.is_empty()
            && self.device_gaps.is_empty()
            && self.never_returned.is_empty()
            && self.bad_allowances.is_empty()
    }
}

/// Compare `ours` against the recorded `metal`, under a case's `allowances`, with `offered` what
/// the run and its recording offer to justify a difference.
pub fn compare(
    metal: &Transcript,
    ours: &Transcript,
    allowances: &[Allowance],
    offered: Offered<'_>,
) -> Comparison {
    compare_run(metal, ours, allowances, offered, false)
}

/// [`compare`] where `stopped` says the run stopped at a call that did not answer.
///
/// A run that stopped reached fewer sites than Metal did, and the sites it never reached are not
/// differences of their own: the refused call is the fact, and burying it under one divergence per
/// site it never got to would hide it.
pub fn compare_run(
    metal: &Transcript,
    ours: &Transcript,
    allowances: &[Allowance],
    offered: Offered<'_>,
    stopped: bool,
) -> Comparison {
    let mut found = Comparison::default();
    // Every allowance names a justification that must exist, whether or not the allowance fires.
    // A case that keeps an allowance for a notice kind, a documented difference or a Metal
    // validation check that has since gone fails here rather than comparing under it.
    for allowance in allowances {
        let missing = match &allowance.justified_by {
            justification @ crate::allowance::Justification::Degradation(code) => {
                let announced = std::slice::from_ref(code);
                match justification.unjustified(&Offered::new(announced, offered.metal_validation))
                {
                    // The run's own notices are checked where the allowance fires, not here.
                    Some(Unjustified::NoticeNotAnnounced(_)) | None => None,
                    other => other,
                }
            }
            justification => justification.unjustified(&offered),
        };
        if let Some(why) = missing {
            found.note_bad(&allowance.path, why);
        }
    }
    let mut fired = vec![false; allowances.len()];
    // The complement of the walk at the notices site, and the suite's central claim rests on it:
    // every notice our side announced must be one the case names, and a declaration fires on its
    // own code. Without it a case that declared any single notice could not fail at that site at
    // all, and its second declaration could never fire (see [`notice_declarations`]).
    for undeclared in notice_declarations(ours, allowances, &mut fired) {
        found.divergences.push(undeclared);
    }
    let mut walk = Walk {
        allowances,
        offered,
        stopped,
        device_gaps: &ours.device_gaps,
        fired: &mut fired,
        found: &mut found,
    };
    walk.node("", &metal.root, &ours.root);
    // An allowance no site needed is the case naming a tolerance this run cannot vouch for. It is
    // counted rather than failed, and the justification is re-read here without the exemption the
    // up-front check gives a notice, because "the tolerance fired nowhere and the notice was never
    // announced" is the one shape nothing else in the suite can say.
    found.named = allowances.len();
    for (allowance, fired) in allowances.iter().zip(&fired) {
        if *fired {
            continue;
        }
        found.unfired.push(Unfired {
            path: allowance.path.clone(),
            how: allowance.loosening.clone(),
            justified_by: allowance.justified_by.describe(),
            why: allowance
                .justified_by
                .unjustified(&offered)
                .map(|why| why.to_string()),
        });
    }
    found
}

/// Hold the case's notice declarations to the notices our run announced, two ways: a divergence for
/// every code we announced that the case names nowhere, and `fired` set for every declaration whose
/// own code was announced.
///
/// **Why this is a check of its own, and not something the walk does.** A notice declaration is an
/// allowance at `case/notices/*` loosened [`Loosening::Unchecked`]
/// ([`crate::replay::notice_allowance`]), and the notices site holds the whole announced **set** as
/// one value. So the moment one declaration's justification holds, [`Walk::allowed`] passes that
/// site whatever is in the set, and marks only *that* allowance fired. Both halves were defects:
///
/// - **One declaration excused every other notice.** Measured at `f2fc351534`: a case declaring
///   `carried-decoded-twin` while the run announced it *plus* `device-texel-conversion` *plus*
///   `native-division` compared with **zero** divergences. The honest rule is a subset — what we
///   announce is covered by what the case names — and no allowance can reach it, because a
///   declaration of one code must not excuse another's.
/// - **A case's second declaration could never fire.** A case naming two notices, both announced,
///   read "3 named, 2 fired, 1 never fired", because the walk stops at the first covering
///   allowance that holds. So the obvious way to state two notices separately produced a
///   permanently dead allowance, and the only thing that said so was the unfired counter. A
///   declaration now fires on **its own** code being announced, which is what it claims.
///
/// `fired` is only set for an allowance that **covers the site that announced the code**: a
/// tolerance at a value site justified by a notice is a claim about a value, and whether it fired
/// is the walk's question, not this one. So a notice-level allowance whose code no run announces
/// still reads as never fired ([`Unfired`]).
///
/// A **declaration**, for the subset rule, is any allowance justified by a
/// [`crate::allowance::Justification::Degradation`], whatever its path: a case that loosens a value
/// by a notice's code has named that code and had it checked against the catalogue, and making it
/// write a second allowance would be bookkeeping rather than evidence. What is **not** required is
/// that a declared notice fired — a notice is announced once per device per kind.
fn notice_declarations(
    ours: &Transcript,
    allowances: &[Allowance],
    fired: &mut [bool],
) -> Vec<Divergence> {
    let code_of = |allowance: &Allowance| match &allowance.justified_by {
        crate::allowance::Justification::Degradation(code) => Some(code.clone()),
        _ => None,
    };
    let declared: Vec<String> = allowances.iter().filter_map(code_of).collect();
    let mut found: Vec<Divergence> = Vec::new();
    for announced in ours.announced_at() {
        for (at, allowance) in allowances.iter().enumerate() {
            if code_of(allowance).as_deref() == Some(announced.code.as_str())
                && allowance.covers(&announced.path)
            {
                fired[at] = true;
            }
        }
        if declared.contains(&announced.code) {
            continue;
        }
        // One row per code, however many sites announced it: the code is the fact, and a row per
        // site would report one undeclared notice as several.
        let held = |divergence: &Divergence| matches!(&divergence.kind, DivergenceKind::UndeclaredNotice { code } if *code == announced.code);
        if found.iter().any(held) {
            continue;
        }
        found.push(Divergence::new(
            &announced.path,
            DivergenceKind::UndeclaredNotice {
                code: announced.code,
            },
        ));
    }
    found
}

/// Whether the sides' answers at one site are Metal answering a call and us refusing it.
///
/// Our side's answer has one shape: a typed error, which a [`crate::case::Failure`] can only record
/// as an `Err`. Metal's answer has **every other** shape there is, and the rule is written that way
/// round on purpose: a value, a tree of facts whose own value is
/// [`Value::Absent`](crate::transcript::Value::Absent), a digest with no value beside it, or a
/// negative fact ([`crate::transcript::Negative`]) are all Metal answering the call. Only two of
/// Metal's answers are not: its **own** error, where the two sides agree and the difference is
/// between the errors; and the one fact that says Metal never came back from the call, which is
/// [`NeverReturned`] and not a gap of ours at all.
///
/// Reading `Ok` against `Err` alone was how a refusal went invisible three ways over. A site whose
/// answer is a tree — a queue's slots, an event's order, a buffer-backed texture's facts — records
/// as a branch whose own value is absent, and the comparison **skips** an absent value, so the
/// typed error was dropped and the report held one "only Metal observed this site" divergence per
/// child instead. A site whose answer is a bare number did read, but as two sides observing
/// different kinds of value. Neither reached the refused count, and neither could be written down:
/// the known-gaps ledger keys a refusal row by its site and its typed answer.
fn refused(metal: &Value, ours: &Value) -> bool {
    ours.is_error()
        && !metal.is_error()
        && !matches!(
            metal,
            Value::Nothing(crate::transcript::Negative::NeverReturned)
        )
}

/// The comparison's state as it descends.
struct Walk<'a> {
    allowances: &'a [Allowance],
    /// What the run and its recording offer to justify a difference.
    offered: Offered<'a>,
    /// Whether the run stopped before the end: a site only Metal reached is then no divergence.
    stopped: bool,
    /// The sites our side said the venue honestly lacks ([`Transcript::device_gaps`]), so a
    /// refusal there is read as the device's gap and not as ours.
    device_gaps: &'a [(String, String)],
    /// One flag per allowance: whether the walk used it to justify a difference, or named it as
    /// what documents a refusal. What stays false is an allowance this run never needed.
    fired: &'a mut [bool],
    found: &'a mut Comparison,
}

impl Walk<'_> {
    /// Compare one pair of sites, under `parent`'s path, and descend.
    fn node(&mut self, parent: &str, metal: &Node, ours: &Node) {
        let path = if parent.is_empty() {
            metal.name.clone()
        } else {
            format!("{parent}/{}", metal.name)
        };
        if self.observed(&path, &metal.observed, &ours.observed) {
            // The site is one we refused, so our side never reached what is under it. The refused
            // call is the fact: one divergence per site we never got to would bury it, which is
            // the reasoning [`compare_run`]'s `stopped` rests on, applied at one site rather than
            // at a whole case. Without this, a site whose answer is a tree reported the refusal
            // nowhere and its children everywhere.
            return;
        }
        // Before pairing anything: a name held twice under one parent is a pair this walk cannot
        // settle, and `Node::child` below would answer with the first namesake and compare the
        // second against nothing. Reported on whichever side holds it, because a case builds both
        // and either side's repeat loses the same comparison.
        for (side, node) in [(Side::Metal, metal), (Side::Ours, ours)] {
            for (name, held) in node.repeated_children() {
                self.found.divergences.push(Divergence::new(
                    format!("{path}/{name}"),
                    DivergenceKind::DuplicateSite {
                        held_by: side,
                        held,
                    },
                ));
            }
        }
        for child in &metal.children {
            match ours.child(&child.name) {
                Some(theirs) => self.node(&path, child, theirs),
                None if self.stopped => {}
                None => self.found.divergences.push(Divergence::new(
                    format!("{path}/{}", child.name),
                    DivergenceKind::Site {
                        held_by: Side::Metal,
                    },
                )),
            }
        }
        for child in &ours.children {
            if metal.child(&child.name).is_none() {
                self.found.divergences.push(Divergence::new(
                    format!("{path}/{}", child.name),
                    DivergenceKind::Site {
                        held_by: Side::Ours,
                    },
                ));
            }
        }
    }

    /// Compare what one site observed, and say whether the site is one **we refused**, whose
    /// children our side therefore never reached.
    fn observed(&mut self, path: &str, metal: &Observed, ours: &Observed) -> bool {
        // A refusal is settled before anything else about the site: before the digests, because a
        // refusal takes no digest; before the branch's own absent value is skipped over, which is
        // where a refusal at a site whose answer is a tree used to disappear; and before any
        // allowance, because no allowance may turn a refusal into a pass.
        if refused(&metal.value, &ours.value) {
            self.refusal(path, &ours.value);
            return true;
        }
        if let (Some(recorded), Some(taken)) = (&metal.digest, &ours.digest) {
            if recorded != taken {
                // A site a case loosened compares under its allowance here too. Without this a
                // digest-only site could never be loosened, and a row whose Metal answer is a
                // digest the device announced it cannot match -- a fill of depths outside
                // [0, 1] on a device whose depth copies go by a draw -- would have no honest
                // shape in the suite at all.
                let hex =
                    |digest: &crate::transcript::Digest| Value::Lines(vec![digest.hex.clone()]);
                if !self.allowed(path, &hex(recorded), &hex(taken)) {
                    self.found.divergences.push(Divergence::new(
                        path,
                        DivergenceKind::Digest {
                            metal: recorded.hex.clone(),
                            ours: taken.hex.clone(),
                        },
                    ));
                }
                return false;
            }
        }
        if metal.is_absent() {
            // A recording taken digest-first holds the digest alone: the digests matched above,
            // and there is no recorded value to compare against. A branch's own value is absent
            // too, and its children are compared by the caller.
            return false;
        }
        if ours.is_absent() && metal.digest.is_some() {
            return false;
        }
        match (&metal.shape, &ours.shape) {
            (Shape::Bytes(layout), _) | (_, Shape::Bytes(layout)) => {
                self.bytes(path, layout, &metal.value, &ours.value);
            }
            _ => self.value(path, &metal.value, &ours.value),
        }
        false
    }

    /// Compare a byte region element by element, and its gaps where the layout asks.
    fn bytes(&mut self, path: &str, layout: &ByteLayout, metal: &Value, ours: &Value) {
        // Only a difference consults an allowance. A site that matched must not fire one: an
        // allowance that fires where nothing moved would fail every exact run whose device
        // announced no notice.
        if metal != ours && self.allowed(path, metal, ours) {
            return;
        }
        let (Value::Bytes(recorded), Value::Bytes(read)) = (metal, ours) else {
            self.mismatched_kind(path, metal, ours);
            return;
        };
        if recorded.len() != read.len() {
            self.found.divergences.push(Divergence::new(
                path,
                DivergenceKind::Length {
                    metal: recorded.len(),
                    ours: read.len(),
                },
            ));
            return;
        }
        let size = layout.element.size() as usize;
        // The whole region is walked, not just up to the first mismatch, because the row this
        // feeds names how much moved: of twelve rotated-target divergences measured in this round
        // every one differs by a single code at its first mismatch and four reach two codes well
        // past it, so stopping early would have reported a bound that is not one.
        let mut count = 0usize;
        let mut largest: Option<f64> = None;
        let mut first: Option<ElementAt> = None;
        for image in 0..layout.images {
            for row in 0..layout.rows {
                for index in 0..layout.row_elements {
                    let at = layout.at(image, row, index);
                    let Some((expected, actual)) = recorded
                        .get(at..at + size)
                        .zip(read.get(at..at + size))
                        .filter(|(expected, actual)| expected != actual)
                    else {
                        continue;
                    };
                    let Some((expected, actual)) = layout
                        .element
                        .read(expected)
                        .zip(layout.element.read(actual))
                    else {
                        continue;
                    };
                    if self.allowed(path, &expected, &actual) {
                        continue;
                    }
                    count += 1;
                    largest = match (largest, crate::allowance::difference(&expected, &actual)) {
                        (Some(held), Some(found)) => Some(held.max(found)),
                        (held, found) => held.or(found),
                    };
                    first = first.or(Some(ElementAt {
                        image,
                        row,
                        index,
                        at,
                        metal: expected,
                        ours: actual,
                    }));
                }
            }
        }
        if let Some(first) = first {
            self.found.divergences.push(Divergence::new(
                path,
                DivergenceKind::Element {
                    element: layout.element,
                    count,
                    largest,
                    first,
                },
            ));
            return;
        }
        if layout.gaps == Gaps::Compared {
            self.gaps(path, layout, recorded, read);
        }
    }

    /// Compare the bytes a layout's elements do not cover: the padding after each row's elements
    /// and after each image's rows.
    fn gaps(&mut self, path: &str, layout: &ByteLayout, metal: &[u8], ours: &[u8]) {
        let covered = |at: usize| -> bool {
            let image = at % layout.image_pitch.max(1) as usize;
            let row = image % layout.row_pitch.max(1) as usize;
            let first = layout.row_offset as usize;
            image < layout.rows as usize * layout.row_pitch as usize
                && row >= first
                && row < first + layout.row_elements as usize * layout.element.size() as usize
        };
        let mut count = 0usize;
        let mut largest: Option<f64> = None;
        let mut first: Option<GapAt> = None;
        for (at, (expected, actual)) in metal.iter().zip(ours).enumerate() {
            if expected != actual && !covered(at) {
                count += 1;
                let difference = f64::from(expected.abs_diff(*actual));
                largest = Some(largest.map_or(difference, |held: f64| held.max(difference)));
                first = first.or(Some(GapAt {
                    at,
                    metal: *expected,
                    ours: *actual,
                }));
            }
        }
        if let Some(first) = first {
            self.found.divergences.push(Divergence::new(
                path,
                DivergenceKind::Gap {
                    count,
                    largest,
                    first,
                },
            ));
        }
    }

    /// Compare two values by their kind.
    fn value(&mut self, path: &str, metal: &Value, ours: &Value) {
        if metal == ours {
            return;
        }
        // Metal not coming back from the call is settled first, and without consulting an
        // allowance at all. There is no value here to be within a tolerance of, and the one fact
        // the row holds -- that we refuse where Metal went away -- must not be loosenable.
        if let Value::Nothing(crate::transcript::Negative::NeverReturned) = metal {
            match ours {
                Value::Variant { name, .. } if name == "Err" => {
                    self.found.never_returned.push(NeverReturned {
                        path: path.to_owned(),
                        ours: crate::diff::print(ours),
                    });
                }
                _ => self.found.divergences.push(Divergence::new(
                    path,
                    DivergenceKind::AnsweredWhereMetalDidNotReturn {
                        ours: crate::diff::print(ours),
                    },
                )),
            }
            return;
        }
        // A refusal **nested inside** a site's value: the site's own pair was settled in
        // [`Walk::observed`], and this is the one under a list's element, a variant's payload or a
        // field, where a case records one answer per entry. Still before any allowance, because no
        // allowance may turn a refusal into a pass: Metal answered the call and we did not.
        if refused(metal, ours) {
            self.refusal(path, ours);
            return;
        }
        // An allowance over the whole site holds whatever kind of value moved there, so a case can
        // justify a notice it announced or a set that gained a member, not only a number that
        // drifted.
        if self.allowed(path, metal, ours) {
            return;
        }
        match (metal, ours) {
            (Value::Nothing(_), _) | (_, Value::Nothing(_)) => {
                let fact = |value: &Value| match value {
                    Value::Nothing(fact) => Some(*fact),
                    _ => None,
                };
                self.found.divergences.push(Divergence::new(
                    path,
                    DivergenceKind::Negative {
                        metal: fact(metal),
                        ours: fact(ours),
                        // The side that observed a value, printed, so the site names what it was.
                        observed: match (fact(metal), fact(ours)) {
                            (Some(_), None) => Some(crate::diff::print(ours)),
                            (None, Some(_)) => Some(crate::diff::print(metal)),
                            _ => None,
                        },
                    },
                ));
            }
            (Value::Lines(recorded), Value::Lines(read)) => {
                // Every differing line is counted, for the same reason the element walk counts
                // every element: the count is the stable part, and a row says how much moved.
                let lines = recorded.len().max(read.len());
                let differ = |at: usize| recorded.get(at) != read.get(at);
                let count = (0..lines).filter(|at| differ(*at)).count();
                let at = (0..lines).find(|at| differ(*at)).unwrap_or(lines);
                self.found.divergences.push(Divergence::new(
                    path,
                    DivergenceKind::Line {
                        count,
                        first: LineAt {
                            at,
                            metal: recorded.get(at).cloned(),
                            ours: read.get(at).cloned(),
                        },
                    },
                ));
            }
            (Value::Set(recorded), Value::Set(read)) => {
                self.found.divergences.push(Divergence::new(
                    path,
                    DivergenceKind::Set {
                        missing: recorded.difference(read).cloned().collect(),
                        extra: read.difference(recorded).cloned().collect(),
                        common: recorded.intersection(read).count(),
                    },
                ));
            }
            (Value::List(recorded), Value::List(read)) => {
                if recorded.len() != read.len() {
                    self.found.divergences.push(Divergence::new(
                        path,
                        DivergenceKind::ListLength {
                            metal: recorded.len(),
                            ours: read.len(),
                        },
                    ));
                    return;
                }
                for (at, (expected, actual)) in recorded.iter().zip(read).enumerate() {
                    self.value(&format!("{path}/{at}"), expected, actual);
                }
            }
            (
                Value::Variant {
                    name: recorded,
                    payload: expected,
                },
                Value::Variant {
                    name: read,
                    payload: actual,
                },
            ) => {
                if recorded == read {
                    self.value(&format!("{path}/{read}"), expected, actual);
                } else {
                    self.found.divergences.push(Divergence::new(
                        path,
                        DivergenceKind::Variant {
                            metal: recorded.clone(),
                            ours: read.clone(),
                        },
                    ));
                }
            }
            (Value::Fields(recorded), Value::Fields(read)) => {
                for (name, expected) in recorded {
                    match read.get(name) {
                        Some(actual) => self.value(&format!("{path}/{name}"), expected, actual),
                        None => self.found.divergences.push(Divergence::new(
                            path,
                            DivergenceKind::Field {
                                name: name.clone(),
                                held_by: Side::Metal,
                            },
                        )),
                    }
                }
                for name in read.keys() {
                    if !recorded.contains_key(name) {
                        self.found.divergences.push(Divergence::new(
                            path,
                            DivergenceKind::Field {
                                name: name.clone(),
                                held_by: Side::Ours,
                            },
                        ));
                    }
                }
            }
            _ if metal.kind() != ours.kind() => self.mismatched_kind(path, metal, ours),
            _ => {
                if !self.allowed(path, metal, ours) {
                    self.found.divergences.push(Divergence::new(
                        path,
                        DivergenceKind::Value {
                            metal: metal.clone(),
                            ours: ours.clone(),
                        },
                    ));
                }
            }
        }
    }

    /// Note that we refused a call Metal answered, with whatever the case named to document it.
    ///
    /// The case's own allowances at the site are still held to their justifications: a site may
    /// name several, so the documentation is the first whose justification **holds**, as
    /// [`Walk::allowed`] picks one. Where none holds, the first that does not is noted as a fault
    /// of its own, because a case must not claim an explanation it cannot show. What a
    /// justification cannot do is remove the row from [`Comparison::refusals`].
    fn refusal(&mut self, path: &str, ours: &Value) {
        // A device's honest gap is settled before anything else, and consults no allowance: the
        // typed error names capabilities a device could enable or a driver measured to run the
        // thing wrong, so the venue truly cannot run the call. It is not ours to answer for and
        // it is not a tolerance either.
        //
        // Read from the value **first** and from the path-keyed list second. The value is where
        // [`crate::case::Failure::recorded`] puts the fact, so it is there whatever path a case
        // named in its own abort; the list carries the sites a case stated by hand
        // ([`crate::transcript::Sites::device_gap`]), which no failure of ours passes through.
        // Reading only the list was how a device's gap read as a refusal of ours at a site whose
        // case named its abort one path and pushed its node at another.
        let lacks = crate::case::lacks_recorded(ours)
            .map(str::to_owned)
            .or_else(|| {
                self.device_gaps
                    .iter()
                    .find(|(site, _)| site == path)
                    .map(|(_, needs)| needs.clone())
            });
        if let Some(needs) = lacks {
            self.found.device_gaps.push(DeviceGap {
                path: path.to_owned(),
                needs,
                ours: crate::diff::print(ours),
            });
            return;
        }
        let mut documented_by = None;
        let mut first_unjustified = None;
        for (at, allowance) in self
            .allowances
            .iter()
            .enumerate()
            .filter(|(_, allowance)| allowance.covers(path))
        {
            match allowance.justified_by.unjustified(&self.offered) {
                None => {
                    self.fired[at] = true;
                    documented_by = Some(allowance.justified_by.describe());
                    break;
                }
                Some(why) => {
                    first_unjustified = first_unjustified.or(Some((allowance.path.clone(), why)));
                }
            }
        }
        if documented_by.is_none() {
            if let Some((path, why)) = first_unjustified {
                self.found.note_bad(&path, why);
            }
        }
        self.found.refusals.push(Refusal {
            path: path.to_owned(),
            ours: crate::diff::print(ours),
            documented_by,
        });
    }

    /// Note that the sides observed different kinds of value at one site.
    fn mismatched_kind(&mut self, path: &str, metal: &Value, ours: &Value) {
        self.found.divergences.push(Divergence::new(
            path,
            DivergenceKind::Kind {
                metal: metal.kind(),
                ours: ours.kind(),
            },
        ));
    }

    /// Whether an allowance of the case covers `path` and holds for this difference.
    ///
    /// A site may carry several allowances, and must: a notice is announced once per device per
    /// kind, so which of a case's notices its own run sees depends on what ran before it. The
    /// difference is justified where **any** of the covering allowances holds; the case fails only
    /// where none does, and then by the first of them, so the report names a justification the
    /// case actually claimed.
    fn allowed(&mut self, path: &str, metal: &Value, ours: &Value) -> bool {
        let mut covering = self
            .allowances
            .iter()
            .enumerate()
            .filter(|(_, allowance)| allowance.covers(path) && allowance.within(metal, ours))
            .peekable();
        if covering.peek().is_none() {
            return false;
        }
        let mut first_unjustified = None;
        let mut held = None;
        for (at, allowance) in covering {
            match allowance.justified_by.unjustified(&self.offered) {
                None => {
                    held = Some((at, allowance));
                    break;
                }
                Some(why) => {
                    first_unjustified = first_unjustified.or(Some((allowance, why)));
                }
            }
        }
        let Some((at, allowance)) = held else {
            let (allowance, why) = first_unjustified.expect("a covering allowance");
            let path = allowance.path.clone();
            self.found.note_bad(&path, why);
            return false;
        };
        self.fired[at] = true;
        let difference = crate::allowance::difference(metal, ours);
        match self.found.announced.iter_mut().find(|held| {
            held.path == path && held.justified_by == allowance.justified_by.describe()
        }) {
            Some(held) => {
                held.count += 1;
                held.largest = match (held.largest, difference) {
                    (Some(held), Some(found)) => Some(held.max(found)),
                    (held, found) => held.or(found),
                };
            }
            None => self.found.announced.push(Announced {
                path: path.to_owned(),
                how: allowance.loosening.clone(),
                justified_by: allowance.justified_by.describe(),
                count: 1,
                largest: difference,
            }),
        }
        true
    }
}
