//! What comparing one case on one venue says.
//!
//! The verdicts are kept apart on purpose. A refusal is never buried in a failure count, because
//! the owner's rule is that a refusal of something Metal does is a bug of its own kind; a case with
//! no Metal answer is counted rather than skipped; and a device that honestly lacks a feature the
//! case needs is neither a pass nor a failure, because conflating the two makes coverage fiction.

use crate::compare::{Comparison, DeviceGap};
use crate::diff::Divergence;
use crate::recording::Validated;
use metal2vulkan::error::{Needs, Unsupported};

/// What one case's comparison on one venue says.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Every field matched, and the case used no allowance.
    Exact,
    /// Fields differed only inside allowances that hold, and the notices they name were
    /// announced.
    Announced { fields: Vec<String> },
    /// A field differs with nothing to justify it: the bug class.
    Divergent {
        found: Vec<Divergence>,
        /// The sites we also refused, as their paths and our errors, in the form
        /// [`Verdict::Refused`] prints.
        ///
        /// Carried here because a case may hold both, and a divergence must not launder a
        /// refusal. The verdict is `Divergent`, since that is the more severe of the two, but the
        /// refused sites are still printed, and the known-gaps ledger still holds its refusal rows
        /// against them in both directions. Without this a refusal row of a case that also
        /// diverges would read as "no longer refused" — the ledger failing for the wrong reason —
        /// or, worse, be dropped on that reading while the gap was still there.
        refused: Vec<String>,
    },
    /// We returned a typed error where Metal answered the call.
    ///
    /// **Never compliant, and no allowance can change that.** Plain Metal is the behavioural
    /// oracle, because that is how a consumer runs the runtime, and the owner's standing rule is
    /// that what Metal allows we must allow; so a refusal of something plain Metal does is a gap
    /// of ours. A case may name what documents the refusal — Metal's own API validation refusing
    /// the same call is the one thing that can — and that is printed beside the row, but the row
    /// still counts here. Metal's validation stopping a call is not Metal refusing it.
    Refused {
        /// The sites we refused, as their paths and our errors.
        at: Vec<String>,
        /// What the recording's second oracle says of the same case
        /// ([`Validated::describe`]): a refusal Metal's own validation also refuses is a
        /// different fact from one its validation let through, and both differ again from a case
        /// whose validated leg was never taken. The refusal is a failure in all three; what
        /// changes is what a reader should do about it.
        metal_validation: String,
    },
    /// **Metal did not return from the call, and we gave a typed error there.**
    ///
    /// Its own verdict, because neither of the others fits and squeezing it into one would say
    /// something false. `Exact` would claim a value matched where Metal produced none. `Divergent`
    /// would claim a behaviour of Metal's to differ from where the call ended Metal's process.
    /// `Refused` is for a call plain Metal **answered**, which is the gap the owner's rule is
    /// about, and counting this there would bury a real gap among rows that are not one.
    ///
    /// It is **compliant**, and it loosens nothing: no allowance can reach it
    /// ([`crate::compare::NeverReturned`]), the recording is written by hand from a probe that died
    /// rather than by running the shape, and record mode refuses the row. Answering where Metal
    /// did not return is a `Divergent` row, not this one, so the verdict can still fail.
    MetalNeverReturned {
        /// The sites, as their paths and our errors.
        at: Vec<String>,
        /// What the row still loosened, where it loosened anything: a difference inside a
        /// justified allowance at another site is reported here rather than lost, because a row
        /// does not stop being a row about tolerances just because one of its calls killed Metal.
        justified: Vec<String>,
    },
    /// No Metal answer stands for this run: none was recorded at the runtime's release, or the
    /// inputs the run fed are not the ones the recording's answer was taken from. Counted, never
    /// skipped and never a pass.
    Unrecorded { why: String },
    /// The venue honestly lacks what the case needs, by the typed refusal it gave: neither a pass
    /// nor a failure.
    ///
    /// Two shapes reach it, and both are the device's own gap. The case may have **stopped** at
    /// the call, which is what `replay` reads from the abort. Or the case may have carried on and
    /// one of its sites be the venue's gap ([`DeviceGap`]), in which case `needs` names each site
    /// and what it lacks, and the rest of the case compared normally — a divergence or a refusal
    /// anywhere else in it still fails.
    LacksFeature { needs: String },
}

impl Verdict {
    /// The verdict `found` says, with `validated` what the recording holds of Metal's validated
    /// run.
    pub fn of(found: &Comparison, validated: &Validated) -> Self {
        if !found.divergences.is_empty() || !found.bad_allowances.is_empty() {
            let mut diverged = found.divergences.clone();
            for bad in &found.bad_allowances {
                diverged.push(Divergence::new(
                    bad.path.clone(),
                    crate::diff::DivergenceKind::Unjustified {
                        why: bad.why.to_string(),
                    },
                ));
            }
            return Verdict::Divergent {
                found: diverged,
                refused: refused_sites(found),
            };
        }
        if !found.refusals.is_empty() {
            return Verdict::Refused {
                at: refused_sites(found),
                metal_validation: validated.describe(),
            };
        }
        // After the refusals, because a refusal of something plain Metal answered is a gap of
        // ours and must not be hidden behind a call Metal died on; before the announced and exact
        // verdicts, because a row where Metal went away is not a row where everything matched.
        if !found.never_returned.is_empty() {
            return Verdict::MetalNeverReturned {
                at: found
                    .never_returned
                    .iter()
                    .map(|site| format!("{}: {}", site.path, site.ours))
                    .collect(),
                justified: announced_fields(found),
            };
        }
        // After everything that fails, and before the verdicts that claim the case compared: a
        // site the venue honestly cannot run is neither a pass nor a failure, so a case holding
        // one must not read as `Exact` — that would claim coverage the venue did not give — while
        // a divergence or a refusal elsewhere in the same case still fails, which is what makes
        // this safe to be compliant.
        if !found.device_gaps.is_empty() {
            return Verdict::LacksFeature {
                needs: found
                    .device_gaps
                    .iter()
                    .map(DeviceGap::to_string)
                    .collect::<Vec<String>>()
                    .join("; "),
            };
        }
        if found.announced.is_empty() {
            return Verdict::Exact;
        }
        Verdict::Announced {
            fields: announced_fields(found),
        }
    }

    /// The word the report's matrix prints.
    pub fn word(&self) -> &'static str {
        match self {
            Verdict::Exact => "exact",
            Verdict::Announced { .. } => "announced",
            Verdict::Divergent { .. } => "divergent",
            Verdict::Refused { .. } => "refused",
            Verdict::MetalNeverReturned { .. } => "never-returned",
            Verdict::Unrecorded { .. } => "unrecorded",
            Verdict::LacksFeature { .. } => "lacks-feature",
        }
    }

    /// Whether the verdict is one a compliant run may hold: exact, announced inside a justified
    /// allowance, or a venue that lacks the feature. A divergence, a refusal and an unrecorded
    /// case are not.
    pub fn is_compliant(&self) -> bool {
        matches!(
            self,
            Verdict::Exact
                | Verdict::Announced { .. }
                | Verdict::MetalNeverReturned { .. }
                | Verdict::LacksFeature { .. }
        )
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Exact => out.write_str("exact"),
            Verdict::Announced { fields } => {
                write!(out, "announced: {}", fields.join("; "))
            }
            Verdict::Divergent { found, refused } => {
                out.write_str("divergent")?;
                for divergence in found {
                    // The summary first, which is the line a `known-gaps-rt.tsv` row's answer
                    // holds, then where to start looking. The position is a line of its own
                    // precisely so that it is not part of the row's identity.
                    write!(out, "\n  {divergence}")?;
                    if let Some(detail) = divergence.kind.detail() {
                        write!(out, "\n    {detail}")?;
                    }
                }
                // A refusal is never hidden behind a divergence: plain Metal answered the call
                // and we did not, which is a gap of its own kind however the rest of the case
                // compared.
                for site in refused {
                    write!(out, "\n  and refused {site}")?;
                }
                Ok(())
            }
            Verdict::Refused {
                at,
                metal_validation,
            } => {
                out.write_str("refused")?;
                for site in at {
                    write!(out, "\n  {site}")?;
                }
                write!(out, "\n  plain Metal answered the call")?;
                write!(out, "\n  Metal's API validation: {metal_validation}")
            }
            Verdict::MetalNeverReturned { at, justified } => {
                out.write_str("Metal did not return from the call, and we refused it")?;
                for site in at {
                    write!(out, "\n  {site}")?;
                }
                for field in justified {
                    write!(out, "\n  and differed, justified: {field}")?;
                }
                Ok(())
            }
            Verdict::Unrecorded { why } => write!(out, "unrecorded: {why}"),
            Verdict::LacksFeature { needs } => write!(out, "lacks the device feature: {needs}"),
        }
    }
}

/// How the report prints each site where plain Metal answered a call and we returned a typed
/// error, with whatever the case named to document it.
///
/// One function, because the same form is read back by the known-gaps ledger, and a refusal
/// printed two ways would be two rows for one gap.
fn refused_sites(found: &Comparison) -> Vec<String> {
    found
        .refusals
        .iter()
        .map(|refusal| match &refusal.documented_by {
            Some(documented) => format!(
                "{}: {} (documented, not excused, by {documented})",
                refusal.path, refusal.ours
            ),
            None => format!("{}: {}", refusal.path, refusal.ours),
        })
        .collect()
}

/// How the report prints each difference a run made inside a justified allowance.
fn announced_fields(found: &Comparison) -> Vec<String> {
    found
        .announced
        .iter()
        .map(|announced| {
            let largest = match announced.largest {
                Some(largest) => format!(", the largest difference {largest}"),
                None => String::new(),
            };
            format!(
                "{} x{} ({}{largest})",
                announced.path, announced.count, announced.justified_by
            )
        })
        .collect()
}

/// Whether a typed `Unsupported` is a device's honest gap rather than ours: it names
/// capabilities a device could enable. A refusal the translator or the
/// runtime has not built, and one Metal itself gives, is ours to answer for.
pub fn device_gap_of(unsupported: &Unsupported) -> Option<String> {
    match &unsupported.needs {
        Needs::AnyOf(alternatives) => Some(format!(
            "{:?} needs any of {alternatives:?}",
            unsupported.feature
        )),
        // Everything else is ours to answer for: a form the translator or the runtime has not
        // built, and one Metal itself refuses, are not the device's gap.
        _ => None,
    }
}
