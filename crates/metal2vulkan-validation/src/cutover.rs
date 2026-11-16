//! What the cutover criteria's reviewed lists share.
//!
//! A reviewed list is data a person reviewed: rows or cases a criterion does not count, each with
//! the evidence that it need not, a reviewer's note, and the user's sign-off. Only the criterion's
//! report reads a list; translation, classification, and the censuses never do. Each list keys its
//! entries by what was reviewed (C1: a profile, the engine's typed outcome, and the digest of
//! the rows that had it; C3: a case, a slot, and a candidate's hashes), so that when what was
//! reviewed changes, the entry is stale and the rows it covered count again until it is reviewed
//! again. This module holds what both lists share, one vocabulary for both: [`Review`] and its
//! [`SignOff`], serialized as each entry's `review`; where the committed lists live
//! ([`list_path`]); and how a list is read, checked, and written ([`List`]). Both criteria also
//! share what the launch leaves out of scope ([`scope`]): such a row or case is reported apart,
//! never listed.

pub mod scope;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// The committed reviewed list `file` (`validation/cutover/<file>`).
pub fn list_path(file: &str) -> PathBuf {
    metal2vulkan_devtools::validation_dir()
        .join("cutover")
        .join(file)
}

/// A committed reviewed list. A list states its file, its name, its checks, and the order it is
/// committed in; reading, checking, and writing it are the same for every list.
pub trait List: Clone + Serialize + DeserializeOwned {
    /// The list's file under `validation/cutover/`.
    const FILE: &'static str;
    /// The list's name in errors (`C1 reviewed list`).
    const NAME: &'static str;

    /// The list's own checks: what its entries are keyed by, and each entry's [`Review`].
    fn check(&self) -> Result<(), String>;

    /// The list in its committed order (by default, the order given).
    fn committed(&self) -> Cow<'_, Self> {
        Cow::Borrowed(self)
    }

    /// The committed list (`validation/cutover/<FILE>`).
    fn committed_path() -> PathBuf {
        list_path(Self::FILE)
    }

    /// Parse and check a list ([`List::check`]).
    fn parse(json: &str) -> Result<Self, String> {
        let list: Self =
            serde_json::from_str(json).map_err(|error| format!("{}: {error}", Self::NAME))?;
        list.check()?;
        Ok(list)
    }

    /// Read and check the list at `path`; an error names the file.
    fn load(path: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        Self::parse(&json).map_err(|error| format!("{}: {error}", path.display()))
    }

    /// The list as committed: in its committed order, two-space JSON, one trailing newline.
    fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&*self.committed())
            .map(|json| json + "\n")
            .map_err(|error| format!("encode {}: {error}", Self::NAME))
    }
}

/// The user's decision on one reviewed entry. A criterion counts an entry only once it is signed
/// off; until then the report shows its rows as listed and awaiting the decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum SignOff {
    /// The entry awaits the user's decision `decision` (for example `sign off as a device limit`).
    AwaitingUser { decision: String },
    /// The user signed the entry off under `decision` on `date` (`YYYY-MM-DD`).
    SignedOff { decision: String, date: String },
}

impl SignOff {
    pub fn is_signed_off(&self) -> bool {
        matches!(self, SignOff::SignedOff { .. })
    }

    /// `awaiting_user(<decision>)` or `signed_off(<decision> <date>)`.
    pub fn label(&self) -> String {
        match self {
            SignOff::AwaitingUser { decision } => format!("awaiting_user({decision})"),
            SignOff::SignedOff { decision, date } => format!("signed_off({decision} {date})"),
        }
    }

    fn check(&self) -> Result<(), String> {
        let decision = match self {
            SignOff::AwaitingUser { decision } => decision,
            SignOff::SignedOff { decision, date } => {
                if !is_date(date) {
                    return Err(format!("sign-off date {date:?} is not YYYY-MM-DD"));
                }
                decision
            }
        };
        if decision.trim().is_empty() {
            return Err("the sign-off names no decision".into());
        }
        Ok(())
    }
}

fn is_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 10
        && bytes.iter().enumerate().all(|(index, byte)| match index {
            4 | 7 => *byte == b'-',
            _ => byte.is_ascii_digit(),
        })
}

/// The reviewed part of one entry: what shows the entry's rows need not count, the reviewer's
/// note, and the user's sign-off.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Review {
    /// The evidence: the measurements, code and profile facts behind the review. At least one, none
    /// empty.
    pub evidence: Vec<String>,
    /// The reviewer's note.
    pub note: String,
    pub sign_off: SignOff,
}

impl Review {
    /// Every entry needs its evidence, a note, and a well-formed sign-off.
    pub fn check(&self) -> Result<(), String> {
        if self.evidence.is_empty() || self.evidence.iter().any(|entry| entry.trim().is_empty()) {
            return Err("needs its evidence entries".into());
        }
        if self.note.trim().is_empty() {
            return Err("needs a note".into());
        }
        self.sign_off.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn review(sign_off: SignOff) -> Review {
        Review {
            evidence: vec!["a device probe".into()],
            note: "the device lacks it".into(),
            sign_off,
        }
    }

    #[test]
    fn a_review_needs_evidence_a_note_and_a_well_formed_sign_off() {
        let awaiting = SignOff::AwaitingUser {
            decision: "a device limit".into(),
        };
        assert!(review(awaiting.clone()).check().is_ok());
        assert!(!awaiting.is_signed_off());
        assert_eq!(awaiting.label(), "awaiting_user(a device limit)");
        let signed = SignOff::SignedOff {
            decision: "a device limit".into(),
            date: "2026-10-02".into(),
        };
        assert!(review(signed.clone()).check().is_ok());
        assert!(signed.is_signed_off());
        assert_eq!(signed.label(), "signed_off(a device limit 2026-10-02)");

        let mut bad = review(awaiting.clone());
        bad.evidence.clear();
        assert!(bad.check().is_err());
        let mut bad = review(awaiting.clone());
        bad.evidence.push(" ".into());
        assert!(bad.check().is_err());
        let mut bad = review(awaiting);
        bad.note = "  ".into();
        assert!(bad.check().is_err());
        assert!(review(SignOff::AwaitingUser {
            decision: String::new()
        })
        .check()
        .is_err());
        for date in ["2026-1-02", "02-10-2026", "2026/10/02", ""] {
            assert!(
                review(SignOff::SignedOff {
                    decision: "a device limit".into(),
                    date: date.into(),
                })
                .check()
                .is_err(),
                "{date:?}"
            );
        }
    }

    #[test]
    fn a_sign_off_reads_as_a_tagged_state_and_rejects_unknown_fields() {
        let awaiting: SignOff =
            serde_json::from_str(r#"{"state":"awaiting_user","decision":"a device limit"}"#)
                .unwrap();
        assert_eq!(
            awaiting,
            SignOff::AwaitingUser {
                decision: "a device limit".into()
            }
        );
        let signed: SignOff = serde_json::from_str(
            r#"{"state":"signed_off","decision":"a device limit","date":"2026-10-02"}"#,
        )
        .unwrap();
        assert!(signed.is_signed_off());
        assert!(serde_json::from_str::<SignOff>(
            r#"{"state":"awaiting_user","decision":"a device limit","by":"x"}"#
        )
        .is_err());
        assert!(serde_json::from_str::<SignOff>(
            r#"{"state":"approved","decision":"a device limit"}"#
        )
        .is_err());
        assert!(serde_json::from_str::<Review>(
            r#"{"evidence":["e"],"note":"n","sign_off":{"state":"awaiting_user","decision":"a device limit"},"extra":1}"#
        )
        .is_err());
    }

    #[test]
    fn committed_lists_live_under_validation_cutover() {
        let path = list_path("c1-reviewed.json");
        assert!(path.ends_with("cutover/c1-reviewed.json"));
        assert_eq!(
            path.parent().and_then(Path::parent),
            Some(metal2vulkan_devtools::validation_dir().as_path())
        );
        assert_eq!(crate::c1::reviewed::ReviewedList::committed_path(), path);
        assert_eq!(
            crate::c3::ReviewedList::committed_path(),
            list_path("c3-reviewed.json")
        );
    }

    /// Both committed lists read every entry's review through this module's [`Review`]: each
    /// entry's `review` object, and nothing else, is one.
    #[test]
    fn both_committed_lists_carry_the_shared_review() {
        let reviews = |file: &str, parts: &[&str]| {
            let json: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(list_path(file)).unwrap()).unwrap();
            parts
                .iter()
                .flat_map(|part| json[part].as_array().unwrap().clone())
                .map(|entry| {
                    for flat in ["evidence", "note", "sign_off"] {
                        assert!(
                            entry.get(flat).is_none(),
                            "{file}: {flat} outside the review"
                        );
                    }
                    let review: Review = serde_json::from_value(entry["review"].clone()).unwrap();
                    review.check().unwrap();
                    review
                })
                .collect::<Vec<_>>()
        };
        let c1 = reviews("c1-reviewed.json", &["entries"]);
        let c3 = reviews("c3-reviewed.json", &["readings", "old_mismatches"]);
        assert!(!c1.is_empty() && !c3.is_empty());
    }
}
