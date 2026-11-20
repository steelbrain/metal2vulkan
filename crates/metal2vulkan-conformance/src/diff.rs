//! What a divergence is and how it prints.
//!
//! A failure names the site that moved by its path from the root and says, in one line, **how
//! much** of it moved: how many of the site's leaves differ and the largest difference where the
//! values are numbers. Then it prints the first differing leaf ([`DivergenceKind::detail`]) for a
//! reader who wants somewhere to start. It never prints two whole values and leaves the reader to
//! bisect them.
//!
//! # Why the summary leads and the position follows
//!
//! That first line is what a `known-gaps-rt.tsv` row's `answer` field holds, so it has to be both
//! honest and **stable**.
//!
//! Honest: a bound read off the first mismatch is not a bound. Of twelve rotated-target
//! divergences measured in this round, every one differs by a single code at the first mismatch
//! and four reach **two** codes well past it -- one of them at row 1, byte 905 -- so a row taken
//! from the first mismatch would have told a reader the gap was one code.
//!
//! Stable: a typed error is a property of the code path, but a first-mismatch position is a
//! property of the **data walk**, so changing any earlier leaf renames the row without the gap
//! changing at all. A count of differing leaves and the largest difference are properties of the
//! comparison, so they do not move for reasons unrelated to the bug.
//!
//! The shape is `site`, `count`, and then whatever that kind can say. It degrades cleanly: a
//! count exists for every kind, and only the largest difference is numbers-only, so a texel
//! divergence reads "8419 elements differ, the largest difference 2" and a missing field reads as
//! the one fact it is. Where a site has exactly one leaf the two values **are** the whole fact and
//! are printed, because they are stable too.

use crate::transcript::{Element, Negative, Value};
use std::collections::BTreeSet;

/// A site whose observation differs, with what moved there.
#[derive(Clone, Debug, PartialEq)]
pub struct Divergence {
    /// The site, as node names from the root, `/`-separated.
    pub path: String,
    pub kind: DivergenceKind,
}

impl Divergence {
    /// A divergence at `path`.
    pub fn new(path: impl Into<String>, kind: DivergenceKind) -> Self {
        Self {
            path: path.into(),
            kind,
        }
    }
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{}: {}", self.path, self.kind)
    }
}

/// Where one differing element of a byte region sits, and what the two sides read there.
///
/// It is the exemplar a report prints, never part of what a [`DivergenceKind`] prints: a walk
/// position is not a property of the bug.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementAt {
    pub image: u32,
    pub row: u32,
    pub index: u32,
    /// The byte offset of the element inside the region, for a reader who wants the raw bytes.
    pub at: usize,
    pub metal: Value,
    pub ours: Value,
}

/// Where one differing byte between the rows sits, and what the two sides read there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GapAt {
    pub at: usize,
    pub metal: u8,
    pub ours: u8,
}

/// Which line of an ordered list differs first, and what the two sides hold there.
#[derive(Clone, Debug, PartialEq)]
pub struct LineAt {
    pub at: usize,
    pub metal: Option<String>,
    pub ours: Option<String>,
}

/// What moved at a site.
#[derive(Clone, Debug, PartialEq)]
pub enum DivergenceKind {
    /// Elements of a byte region differ.
    Element {
        element: Element,
        /// How many of the region's elements differ: a property of the comparison, so it is
        /// stable where a first-mismatch position is not.
        count: usize,
        /// The largest difference over them, where the values are numbers
        /// ([`crate::allowance::difference`]). `None` where the element's values are not numbers,
        /// which is the one thing this shape cannot say.
        largest: Option<f64>,
        /// The first of them, for a reader who wants somewhere to start. **Not** part of what the
        /// kind prints, and so not part of a ledger row's identity.
        first: ElementAt,
    },
    /// Bytes between the rows or the images of a region differ, where the case compares the gaps.
    Gap {
        /// How many such bytes differ.
        count: usize,
        /// The largest difference over them.
        largest: Option<f64>,
        /// The first of them.
        first: GapAt,
    },
    /// A region's bytes are a different length, so there is no element to name.
    Length { metal: usize, ours: usize },
    /// Lines of an ordered list differ, or one side has lines the other does not.
    Line {
        /// How many lines differ.
        count: usize,
        /// The first of them.
        first: LineAt,
    },
    /// A set differs: what Metal held and we did not, what we held and Metal did not, and how many
    /// both held.
    Set {
        missing: BTreeSet<String>,
        extra: BTreeSet<String>,
        common: usize,
    },
    /// A list is a different length.
    ListLength { metal: usize, ours: usize },
    /// A variant differs: the name, not the payload.
    Variant { metal: String, ours: String },
    /// Two values of one kind differ, and the kind has no finer diff.
    Value { metal: Value, ours: Value },
    /// The two sides observed different kinds of value at one site.
    Kind {
        metal: &'static str,
        ours: &'static str,
    },
    /// Metal recorded a fact about what did not happen, and we observed something; or the other
    /// way about.
    Negative {
        metal: Option<Negative>,
        ours: Option<Negative>,
        /// What the side that observed a value observed, in one line, where exactly one side did.
        ///
        /// It is here because the notices site is this shape: Metal announces nothing and we
        /// announce a code, so without the value the difference reads "we observed a value" and
        /// names no notice — although at that site **the notice is the whole difference**. A
        /// known-gaps row's answer is that printed line, so the row would record a difference
        /// without saying which notice it was waiting on.
        observed: Option<String>,
    },
    /// The digest both sides took differs, where the recording carries no value to diff.
    Digest { metal: String, ours: String },
    /// A site one side holds and the other does not.
    Site { held_by: Side },
    /// One side holds more than one site of this name under one parent, so the comparison cannot
    /// say which answer belongs to which.
    ///
    /// A case authoring defect rather than a difference from Metal, and reported because the
    /// alternative is silence: the walk finds a site by name
    /// ([`crate::transcript::Node::child`]) and answers with the first namesake, so the second
    /// site's observation is compared against nothing at all and the case passes having measured
    /// less than it says. **Not** loosenable, for the same reason
    /// [`DivergenceKind::UndeclaredNotice`] is not: an allowance names a site by its path, and
    /// where a path names two sites there is nothing for it to name.
    DuplicateSite { held_by: Side, held: usize },
    /// The case loosened this site, and the allowance does not hold: the difference it would have
    /// covered is one nobody is accounting for.
    Unjustified { why: String },
    /// The site's fields differ in which fields they hold.
    Field { name: String, held_by: Side },
    /// Our side announced a degrade notice the case declares nowhere.
    ///
    /// Metal announces none, so every notice of ours is a difference, and the one thing that may
    /// account for it is the case naming its code
    /// ([`crate::replay::notice_allowance`]). A notice the case names nowhere is therefore a
    /// difference reaching a consumer with nothing said about it, which is the suite's own
    /// definition of a bug — and it is **not** loosenable, because a declaration of one code must
    /// not excuse another's.
    UndeclaredNotice { code: String },
    /// Metal did not return from the call, and we answered it.
    ///
    /// The case's recording says Metal never came back from this call: the probe that found that
    /// out ended Metal's process, so there is no value and no behaviour to compare. What the case
    /// pins is that *we* refuse there. Answering instead is a divergence from the one fact the row
    /// holds, and it is the shape that keeps
    /// [`Verdict::MetalNeverReturned`](crate::verdict::Verdict::MetalNeverReturned) able to fail.
    AnsweredWhereMetalDidNotReturn { ours: String },
}

/// Which side holds something the other does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Metal,
    Ours,
}

impl std::fmt::Display for Side {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Metal => out.write_str("Metal"),
            Side::Ours => out.write_str("ours"),
        }
    }
}

impl DivergenceKind {
    /// Where the first differing leaf sits, for a reader who wants somewhere to start.
    ///
    /// Kept out of [`Display`](std::fmt::Display) on purpose: a walk position is not a property of
    /// the bug, and the printed line is what a `known-gaps-rt.tsv` row's `answer` field holds, so a
    /// position in it would rename the row whenever an earlier leaf moved. The report prints this
    /// beneath the summary, where it helps and binds nothing.
    pub fn detail(&self) -> Option<String> {
        match self {
            DivergenceKind::Element { element, first, .. } => Some(format!(
                "first at [image {}, row {}, {}] ({element:?}, byte {}): Metal {}, ours {}",
                first.image,
                first.row,
                first.index,
                first.at,
                print(&first.metal),
                print(&first.ours)
            )),
            DivergenceKind::Gap { first, .. } => Some(format!(
                "first at byte {}: {:#04x} on Metal and {:#04x} here",
                first.at, first.metal, first.ours
            )),
            DivergenceKind::Line { first, .. } => Some(match (&first.metal, &first.ours) {
                (Some(metal), Some(ours)) => {
                    format!("first at line {}: Metal {metal:?}, ours {ours:?}", first.at)
                }
                (Some(metal), None) => {
                    format!("line {} is missing; Metal has {metal:?}", first.at)
                }
                (None, Some(ours)) => format!("line {} is extra: {ours:?}", first.at),
                (None, None) => format!("first at line {}", first.at),
            }),
            // Every other kind is one leaf, so what it prints is already the whole fact and there
            // is no position to add.
            _ => None,
        }
    }
}

impl std::fmt::Display for DivergenceKind {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DivergenceKind::Element {
                element,
                count,
                largest,
                first,
            } => {
                // One element: its two values are the whole fact, and they are stable. More: the
                // count and the largest difference, because a bound read off one element is not a
                // bound.
                if *count == 1 {
                    return write!(
                        out,
                        "1 element differs ({element:?}): Metal {}, ours {}",
                        print(&first.metal),
                        print(&first.ours)
                    );
                }
                write!(out, "{count} elements differ ({element:?})")?;
                match largest {
                    Some(largest) => write!(out, ", the largest difference {largest}"),
                    None => Ok(()),
                }
            }
            DivergenceKind::Gap {
                count,
                largest,
                first,
            } => {
                if *count == 1 {
                    return write!(
                        out,
                        "1 byte between the rows differs: {:#04x} on Metal and {:#04x} here",
                        first.metal, first.ours
                    );
                }
                write!(out, "{count} bytes between the rows differ")?;
                match largest {
                    Some(largest) => write!(out, ", the largest difference {largest}"),
                    None => Ok(()),
                }
            }
            DivergenceKind::Length { metal, ours } => {
                write!(out, "Metal wrote {metal} bytes, we read {ours}")
            }
            DivergenceKind::Line { count, first } => {
                if *count > 1 {
                    return write!(out, "{count} lines differ");
                }
                match (&first.metal, &first.ours) {
                    (Some(metal), Some(ours)) => {
                        write!(out, "1 line differs: Metal {metal:?}, ours {ours:?}")
                    }
                    (Some(metal), None) => {
                        write!(out, "1 line is missing; Metal has {metal:?}")
                    }
                    (None, Some(ours)) => write!(out, "1 line is extra: {ours:?}"),
                    (None, None) => write!(out, "1 line differs"),
                }
            }
            DivergenceKind::Set {
                missing,
                extra,
                common,
            } => {
                write!(out, "{common} in both")?;
                if !missing.is_empty() {
                    write!(out, ", missing {missing:?}")?;
                }
                if !extra.is_empty() {
                    write!(out, ", extra {extra:?}")?;
                }
                Ok(())
            }
            DivergenceKind::ListLength { metal, ours } => {
                write!(out, "Metal listed {metal} values, we listed {ours}")
            }
            DivergenceKind::Variant { metal, ours } => {
                write!(out, "Metal answered {metal}, we answered {ours}")
            }
            DivergenceKind::Value { metal, ours } => {
                write!(out, "Metal {}, ours {}", print(metal), print(ours))
            }
            DivergenceKind::Kind { metal, ours } => {
                write!(out, "Metal observed a {metal}, we observed a {ours}")
            }
            DivergenceKind::Negative {
                metal,
                ours,
                observed,
            } => {
                let value = observed.as_deref().unwrap_or("a value");
                match (metal, ours) {
                    (Some(fact), None) => {
                        write!(out, "Metal recorded {fact:?}, we observed {value}")
                    }
                    (None, Some(fact)) => {
                        write!(out, "we recorded {fact:?}, Metal observed {value}")
                    }
                    (Some(metal), Some(ours)) => write!(out, "Metal {metal:?}, ours {ours:?}"),
                    (None, None) => write!(out, "a negative fact"),
                }
            }
            DivergenceKind::Digest { metal, ours } => {
                write!(out, "the digest is {metal} on Metal and {ours} here")
            }
            DivergenceKind::Site { held_by } => write!(out, "only {held_by} observed this site"),
            DivergenceKind::DuplicateSite { held_by, held } => write!(
                out,
                "{held_by} holds {held} sites of this name under one parent, so the comparison \
                 cannot say which answer is which"
            ),
            DivergenceKind::Unjustified { why } => {
                write!(out, "the case loosens this site, and {why}")
            }
            DivergenceKind::Field { name, held_by } => {
                write!(out, "only {held_by} holds the field {name}")
            }
            DivergenceKind::UndeclaredNotice { code } => write!(
                out,
                "we announced the degrade notice {code}, which the case declares nowhere"
            ),
            DivergenceKind::AnsweredWhereMetalDidNotReturn { ours } => write!(
                out,
                "Metal did not return from this call, and we answered it: {ours}"
            ),
        }
    }
}

/// `value` in one line, short enough for a failure message.
pub fn print(value: &Value) -> String {
    match value {
        Value::Absent => "absent".to_owned(),
        Value::Nothing(fact) => format!("{fact:?}"),
        Value::Bool(word) => word.to_string(),
        Value::Uint(word) => format!("{word} ({word:#x})"),
        Value::Int(word) => word.to_string(),
        Value::Float { bits, width } => format!("{width:?} {bits:#x}"),
        Value::Bytes(bytes) => format!("{} bytes", bytes.len()),
        // One line is printed, because a typed error's payload is one line and a refusal's row is
        // a bug report: "1 lines" told a reader nothing they could act on. More than one stays a
        // count, so a long probe transcript does not fill the report.
        Value::Lines(lines) => match lines.as_slice() {
            [only] => format!("{only:?}"),
            many => format!("{} lines", many.len()),
        },
        Value::Set(words) => format!("{words:?}"),
        Value::List(values) => format!("{} values", values.len()),
        Value::Variant { name, payload } => match **payload {
            Value::Absent => name.clone(),
            _ => format!("{name}({})", print(payload)),
        },
        // The fields are printed with their names, for the same reason one line is: a refusal's
        // row is a bug report, and "2 fields" names neither the typed error a call gave nor the
        // step of a program it gave it at. Every value under a failure's fields is itself one
        // line, so the whole stays one — these are a typed error's payload, never a case's own
        // observation, which the comparison descends into field by field and never prints whole.
        Value::Fields(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(name, held)| format!("{name}: {}", print(held)))
                .collect::<Vec<String>>()
                .join(", ")
        ),
    }
}
