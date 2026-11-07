//! The justification a loosened field must name.
//!
//! Comparison is exact by default. A case that loosens a field names either a degrade notice's code
//! or an entry of `docs/API.md`'s accepted differences, and the comparison refuses the allowance
//! where that code is not a [`Degradation`]'s, where the entry is not in the document, or where the
//! notice the code names was not announced on the run being compared. So an allowance cannot decay
//! into "we match except where we don't": the degrade catalogue and the accepted-difference list
//! are checked by execution, not by discipline.

use crate::recording::Validated;
use crate::transcript::{FloatWidth, Value};
use metal2vulkan_runtime::Degradation;
use serde::{Deserialize, Serialize};

/// `docs/API.md`, compiled in: the accepted-difference entries are read from the document itself,
/// so an allowance that names a removed entry fails to compare rather than passing quietly.
const API_DOC: &str = include_str!("../../../docs/API.md");

/// The heading of the accepted-difference section of [`API_DOC`].
const ACCEPTED_HEADING: &str = "## 10. Accepted differences";

/// What justifies loosening a field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Justification {
    /// A degrade notice, by its code (`Degradation::code`). The code must be a kind's, and the
    /// run must have announced it.
    Degradation(String),
    /// An entry of `docs/API.md`'s accepted differences, by the slug of its bold lead-in
    /// ([`accepted_slug`]). The entry must be in the document.
    AcceptedDifference(String),
    /// **Metal's own API validation refuses the call**, which is the recording's second Metal
    /// fact ([`Validated::Stopped`]). It is the one justification that rests on the validated
    /// oracle rather than on anything of ours.
    ///
    /// **It documents a refusal; it never excuses one.** Metal's API validation stopping a call is
    /// not Metal refusing it: plain Metal is the behavioural oracle, because that is how a
    /// consumer runs the runtime, and the owner's standing rule is that what Metal allows we must
    /// allow. So a site where plain Metal answered and we returned a typed error stays
    /// [`Verdict::Refused`](crate::verdict::Verdict::Refused) and stays non-compliant, with these
    /// words printed beside it. Letting it pass would be the suite laundering a refusal.
    ///
    /// What the justification is for, then, is the report and the record: a refusal Metal's own
    /// validation also refuses reads differently from one Metal is perfectly happy with, and the
    /// first is the better bug report. It is still held to its evidence, which is what keeps the
    /// validated leg load-bearing rather than decorative: the case names the words Metal's
    /// validation used, the recording's message must hold them, and if Apple's check moves or the
    /// leg was never taken the claim fails and the case fails with it.
    MetalValidation {
        /// A fragment of the validation's own message, which the recording must contain.
        says: String,
    },
    /// **The venue presents a Mac that keeps no texture compressed**, as a Mac with an Intel or
    /// AMD GPU is, and an entry of `docs/API.md`'s accepted differences, by its slug, says what a
    /// device so presented answers otherwise than the Apple silicon Mac every recording was taken
    /// on.
    ///
    /// **It is scoped to the venue, and held both ways.** It holds only on a run whose device
    /// presents no such compression (`DeviceProperties::lossless_compression` false, the venue's
    /// [`crate::case::Venue::lossless_compression`]): on one that presents it, as the M3 Max
    /// does, it justifies nothing, so the sites it covers compare exactly there and any difference
    /// at them is a divergence. lavapipe and the RTX 5080 present a Mac with an Intel or AMD GPU,
    /// so it holds there. Where it is named and no venue without the compression
    /// ran, the report counts it as a claim nothing checked
    /// ([`crate::report::Report::unchecked_everywhere`]), as it does a notice no run announced.
    WithoutCompression(String),
}

impl Justification {
    /// A degrade notice's justification.
    pub fn notice(kind: Degradation) -> Self {
        Self::Degradation(kind.code().to_owned())
    }

    /// An accepted difference's justification, by its entry's slug.
    pub fn accepted(slug: impl Into<String>) -> Self {
        Self::AcceptedDifference(slug.into())
    }

    /// Metal's API validation refusing the call, by the words it used.
    pub fn metal_validation(says: impl Into<String>) -> Self {
        Self::MetalValidation { says: says.into() }
    }

    /// The answer of a venue that presents no texture compression, by its accepted difference's
    /// slug.
    pub fn without_compression(slug: impl Into<String>) -> Self {
        Self::WithoutCompression(slug.into())
    }

    /// What the report prints for it.
    pub fn describe(&self) -> String {
        match self {
            Justification::Degradation(code) => format!("the degrade notice {code}"),
            Justification::AcceptedDifference(slug) => {
                format!("the accepted difference {slug} (docs/API.md section 10)")
            }
            Justification::MetalValidation { says } => {
                format!("Metal's own API validation refuses the call: {says:?}")
            }
            Justification::WithoutCompression(slug) => format!(
                "a venue presenting no texture compression, the accepted difference {slug} \
                 (docs/API.md section 10)"
            ),
        }
    }

    /// Why the justification does not hold: the code names no degrade kind, the slug names no
    /// entry, the notice the code names was not announced among `offered`'s notices, the
    /// recording's validated oracle does not say what the case claims it says, or the venue
    /// presents the compression a venue's own answer needs it not to.
    pub fn unjustified(&self, offered: &Offered<'_>) -> Option<Unjustified> {
        match self {
            Justification::Degradation(code) => {
                if !Degradation::ALL.iter().any(|kind| kind.code() == code) {
                    return Some(Unjustified::NoSuchDegradation(code.clone()));
                }
                if !offered.announced.iter().any(|notice| notice == code) {
                    return Some(Unjustified::NoticeNotAnnounced(code.clone()));
                }
                None
            }
            Justification::AcceptedDifference(slug) => {
                (!accepted_differences().iter().any(|entry| entry == slug))
                    .then(|| Unjustified::NoSuchAcceptedDifference(slug.clone()))
            }
            Justification::MetalValidation { says } => match offered.metal_validation {
                Validated::Stopped { message, .. } if message.contains(says) => None,
                Validated::Stopped { message, .. } => {
                    Some(Unjustified::MetalValidationSaysOtherwise {
                        wanted: says.clone(),
                        said: message.clone(),
                    })
                }
                Validated::Ran => Some(Unjustified::MetalValidationRan),
                Validated::NotRun => Some(Unjustified::MetalValidationNotRun),
            },
            Justification::WithoutCompression(slug) => {
                if !accepted_differences().iter().any(|entry| entry == slug) {
                    return Some(Unjustified::NoSuchAcceptedDifference(slug.clone()));
                }
                offered
                    .lossless_compression
                    .then_some(Unjustified::VenueCompresses)
            }
        }
    }
}

/// What a run and its recording offer to justify a loosened field: the degrade notices our side
/// announced, and what Metal's API validation did with the same case.
///
/// The two are kept in one place because they are the only two things that may justify a
/// difference, and because a case that names one must not be able to reach the other by accident.
#[derive(Clone, Copy, Debug)]
pub struct Offered<'a> {
    /// The degrade-notice codes our run announced.
    pub announced: &'a [String],
    /// The recording's second Metal fact. On a host without Metal this is what the file says, so
    /// the validated oracle is as portable as the rest of the recording.
    pub metal_validation: &'a Validated,
    /// Whether the run's venue presents a Mac that keeps textures losslessly compressed, as the
    /// Apple silicon Mac every recording was taken on does ([`Justification::WithoutCompression`]).
    /// `true` unless the run says otherwise ([`Offered::on_venue`]), so a venue's answer justifies
    /// nothing by default.
    pub lossless_compression: bool,
}

impl<'a> Offered<'a> {
    /// What a run with `announced` offers, where the recording holds no validated answer.
    pub fn notices(announced: &'a [String]) -> Self {
        Self {
            announced,
            metal_validation: &Validated::NotRun,
            lossless_compression: true,
        }
    }

    /// What a run with `announced` offers, beside the recording's `metal_validation`.
    pub fn new(announced: &'a [String], metal_validation: &'a Validated) -> Self {
        Self {
            announced,
            metal_validation,
            lossless_compression: true,
        }
    }

    /// What it offers on a venue that presents Metal's compression where `lossless_compression`.
    pub fn on_venue(self, lossless_compression: bool) -> Self {
        Self {
            lossless_compression,
            ..self
        }
    }
}

/// Why an allowance does not hold. Each of these fails the case: an allowance whose justification
/// is gone is a divergence nobody is accounting for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unjustified {
    /// No `Degradation` has the code.
    NoSuchDegradation(String),
    /// The code is a kind's, but the run announced no such notice, so nothing was degraded here.
    NoticeNotAnnounced(String),
    /// `docs/API.md`'s accepted differences hold no such entry.
    NoSuchAcceptedDifference(String),
    /// Metal's API validation refused the call, but not with the words the case names: Apple's
    /// check moved, so what the case claims is justified is no longer what Metal objects to.
    MetalValidationSaysOtherwise { wanted: String, said: String },
    /// Metal's API validation let the call run, so nothing of Metal's justifies the difference.
    MetalValidationRan,
    /// The case was never run under Metal's API validation, so the recording holds no such fact.
    /// This is the one the whole axis turns on: an unexercised validated leg justifies nothing.
    MetalValidationNotRun,
    /// The venue presents a Mac that keeps textures compressed, as the recordings' Apple silicon
    /// Mac does, so the answer of a venue that keeps none justifies nothing here.
    VenueCompresses,
}

impl Unjustified {
    /// Whether an allowance that does not hold for this reason faults the case. Every reason does
    /// but one: a venue's own answer ([`Justification::WithoutCompression`]) on a venue that
    /// presents Metal's compression holds nowhere, as on the Mac that recorded the case, so naming
    /// it is no fault there. The site then compares as if the case loosened nothing, and a
    /// difference there is the site's own and nothing more.
    pub fn faults_the_case(&self) -> bool {
        !matches!(self, Unjustified::VenueCompresses)
    }
}

impl std::fmt::Display for Unjustified {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unjustified::NoSuchDegradation(code) => {
                write!(out, "no degrade notice has the code {code}")
            }
            Unjustified::NoticeNotAnnounced(code) => write!(
                out,
                "the run announced no {code} notice, so nothing justifies the difference"
            ),
            Unjustified::NoSuchAcceptedDifference(slug) => write!(
                out,
                "docs/API.md section 10 holds no accepted difference {slug}"
            ),
            Unjustified::MetalValidationSaysOtherwise { wanted, said } => write!(
                out,
                "Metal's API validation does not say {wanted:?} any more; it says {said:?}"
            ),
            Unjustified::MetalValidationRan => out.write_str(
                "Metal's API validation let the call run, so nothing of Metal's justifies the \
                 difference",
            ),
            Unjustified::MetalValidationNotRun => out.write_str(
                "the recording holds no validated Metal answer, so Metal's API validation \
                 justifies nothing: re-record the case on a Mac",
            ),
            Unjustified::VenueCompresses => out.write_str(
                "the venue presents a Mac that keeps textures compressed, as the recordings' Mac \
                 does, so the answer of a venue that keeps none justifies nothing here",
            ),
        }
    }
}

/// How far a field may differ.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Loosening {
    /// Floats may differ by this many units in the last place of their width.
    Ulps(u32),
    /// Numbers may differ by this much in absolute value.
    Absolute(f64),
    /// The value may be any of these.
    AnyOf(Vec<Value>),
    /// An integer zero exactly where the recorded one is: any nonzero value stands for any other.
    /// For a value whose contract states only whether it is zero, such as a `Boolean` visibility
    /// result, which is the device's own nonzero value where Metal's M3 Max writes 1
    /// (`device-boolean-visibility`).
    Nonzero,
    /// The value is not compared. The justification still has to hold, and the report says the
    /// field went unchecked.
    Unchecked,
}

/// A field of a case's transcript that may differ, and what justifies it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Allowance {
    /// The site it loosens, as a path of node names from the root, `/`-separated. A path ending in
    /// `/*` loosens every site under it.
    pub path: String,
    pub loosening: Loosening,
    pub justified_by: Justification,
}

impl Allowance {
    /// An allowance of `loosening` at `path`, justified by `justified_by`.
    pub fn new(path: impl Into<String>, loosening: Loosening, justified_by: Justification) -> Self {
        Self {
            path: path.into(),
            loosening,
            justified_by,
        }
    }

    /// Whether the allowance covers the site at `path` (node names from the root, `/`-separated).
    ///
    /// An element of a byte region is addressed as its site's path and its byte offset in the
    /// region, `<site>/<offset>` in decimal (`case/pass-0/words/24`, the eight bytes from offset 24
    /// of a `U64` region): an allowance there covers that element alone, where one at the site
    /// covers every element.
    pub fn covers(&self, path: &str) -> bool {
        match self.path.strip_suffix("/*") {
            Some(prefix) => path == prefix || path.starts_with(&format!("{prefix}/")),
            None => self.path == path,
        }
    }

    /// Whether `actual` is within the allowance of `expected`.
    pub fn within(&self, expected: &Value, actual: &Value) -> bool {
        match &self.loosening {
            Loosening::Unchecked => true,
            Loosening::AnyOf(values) => values.contains(actual),
            Loosening::Nonzero => {
                let zero = |value: &Value| match *value {
                    Value::Uint(word) => Some(word == 0),
                    Value::Int(word) => Some(word == 0),
                    _ => None,
                };
                matches!((zero(expected), zero(actual)), (Some(a), Some(b)) if a == b)
            }
            Loosening::Ulps(ulps) => within_ulps(expected, actual, *ulps),
            Loosening::Absolute(bound) => within_absolute(expected, actual, *bound),
        }
    }
}

/// Whether two floats of one width are within `ulps` of each other, counting across zero: a NaN is
/// within only of a NaN, and an infinity only of itself.
fn within_ulps(expected: &Value, actual: &Value, ulps: u32) -> bool {
    let (
        Value::Float { bits: a, width },
        Value::Float {
            bits: b,
            width: other,
        },
    ) = (expected, actual)
    else {
        return false;
    };
    if width != other {
        return false;
    }
    let (mantissa, exponent) = match width {
        FloatWidth::Half => (10, 5),
        FloatWidth::Single => (23, 8),
        FloatWidth::Double => (52, 11),
    };
    let sign = 1u64 << (mantissa + exponent);
    let magnitude = sign - 1;
    let finite = |bits: u64| (bits & magnitude) >> mantissa != (1u64 << exponent) - 1;
    if !finite(*a) || !finite(*b) {
        return a == b;
    }
    // Map each float onto a monotone ordering so the count crosses zero once.
    let ordered = |bits: u64| -> i128 {
        let value = i128::from(bits & magnitude);
        if bits & sign == 0 {
            value
        } else {
            -value
        }
    };
    (ordered(*a) - ordered(*b)).unsigned_abs() <= u128::from(ulps)
}

/// How far apart two numbers of one kind are, where both are numbers.
pub fn difference(expected: &Value, actual: &Value) -> Option<f64> {
    let number = |value: &Value| match *value {
        Value::Uint(word) => Some(word as f64),
        Value::Int(word) => Some(word as f64),
        Value::Float { bits, width } => Some(match width {
            FloatWidth::Half => half_to_f64(bits as u16),
            FloatWidth::Single => f32::from_bits(bits as u32) as f64,
            FloatWidth::Double => f64::from_bits(bits),
        }),
        _ => None,
    };
    let (a, b) = (number(expected)?, number(actual)?);
    Some((a - b).abs())
}

/// Whether two numbers of one kind are within `bound` of each other.
fn within_absolute(expected: &Value, actual: &Value, bound: f64) -> bool {
    let number = |value: &Value| match *value {
        Value::Uint(word) => Some(word as f64),
        Value::Int(word) => Some(word as f64),
        Value::Float { bits, width } => Some(match width {
            FloatWidth::Half => half_to_f64(bits as u16),
            FloatWidth::Single => f32::from_bits(bits as u32) as f64,
            FloatWidth::Double => f64::from_bits(bits),
        }),
        _ => None,
    };
    match (number(expected), number(actual)) {
        (Some(a), Some(b)) => (a - b).abs() <= bound,
        _ => false,
    }
}

/// A half's value, as a double.
fn half_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f64::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f64.powi(-24),
        0x1f if mantissa == 0.0 => sign * f64::INFINITY,
        0x1f => f64::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15),
    }
}

/// The slugs of `docs/API.md`'s accepted-difference entries: each bullet's bold lead-in, slugged
/// ([`accepted_slug`]).
pub fn accepted_differences() -> &'static [String] {
    static ENTRIES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    ENTRIES.get_or_init(|| {
        let section = API_DOC
            .split_once(ACCEPTED_HEADING)
            .map(|(_, rest)| rest.split("\n## ").next().unwrap_or(rest))
            .unwrap_or_default();
        // An entry's bold lead-in may wrap onto the next lines, so the words are taken until the
        // closing `**` rather than from the first line alone. Reading only the first line made
        // every wrapped entry invisible, and an allowance naming one of those could never be
        // justified.
        let mut entries = Vec::new();
        let mut lead_in: Option<String> = None;
        for line in section.lines() {
            let rest = match (&mut lead_in, line.strip_prefix("- **")) {
                (None, Some(start)) => start,
                (Some(_), _) => line.trim_start(),
                (None, None) => continue,
            };
            let words = lead_in.take().map_or_else(String::new, |mut so_far| {
                so_far.push(' ');
                so_far
            });
            match rest.split_once("**") {
                Some((last, _)) => entries.push(accepted_slug(&(words + last))),
                None => lead_in = Some(words + rest),
            }
        }
        entries
    })
}

/// The slug of an accepted difference's bold lead-in: its words lowercased, anything but a letter
/// or a digit a hyphen, and the hyphens collapsed.
pub fn accepted_slug(lead_in: &str) -> String {
    let mut slug = String::new();
    for character in lead_in.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.trim_matches('-').to_owned()
}
