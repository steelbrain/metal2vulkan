//! Cutover criterion C3's report.
//!
//! C3 asks that the engine on MoltenVK match every authored case the old engine matched, with
//! no new mismatch, that each of the old engine's mismatches be fixed or explained, and that the
//! cases every other device must cover match there: the public cases on lavapipe, and every
//! Metal-qualified case on the real GPUs (the committed [`Venues`]). Some cases read a value Metal
//! only bounds: an ulp or absolute bound, a formula over bounded parts, an operation Metal states
//! no accuracy for, or a grouping the AIR's fast-math flags leave open. Metal's answer is then one
//! reading inside the bound, and an exact match with it is not a property of the translation. The
//! user decided that such cases are listed, each with its bound and cause, keyed by case and
//! candidate, so that a changed candidate leaves the list and is reviewed again. A row that ran a
//! recorded near-equivalent no exact path replaces yet ([`DEFERRED_NEAR_EQUIVALENTS`]: a
//! rasterization-rate map's decoder, a sparse texture's filtered access, an explicit cube
//! gradient's footprint, a nearest cube texel from a correctly rounded reciprocal, a simdgroup
//! operation's lanes before a discard, a depth comparison the device makes, a bicubic sum's
//! unpinned arithmetic, a varying a device interpolates otherwise than Metal, a vertex simdgroup
//! the device groups otherwise than Apple's GPUs) is never such a reading: its difference stays
//! open, deferred until the exact path exists (the user's decision of 2026-10-03).
//!
//! The list ([`ReviewedList`], `validation/cutover/c3-reviewed.json`) is data a person reviewed.
//! Only this report reads it; translation, classification, and the evidence census never do. It is
//! read, checked, and written as C1's list is ([`List`]), and every entry carries the review both
//! cutover lists share ([`Review`]: its evidence, the reviewer's note, and the user's
//! [`SignOff`]). It has two parts:
//!
//! - **Readings** ([`Reading`]): per case, the cause, the bound Metal states, and the reviewed
//!   observations. Each observation names an engine slot, the candidate (`spv_sha256`), its
//!   output, and the Metal output it was compared with. An `apple_reading` is keyed by candidate:
//!   a current mismatching row whose slot and three hashes equal a reviewed observation is
//!   [`Verdict::Reviewed`], not a mismatch, and when the slot holds another candidate or output,
//!   the reviewed observation is [`ReadingState::Changed`]: the row is judged unreviewed, and the
//!   entry must be reviewed again. A `fast_math_regrouping` reading is keyed by its cause: it
//!   reviews every current mismatching row of its case, in every engine slot, that shows the
//!   cause ([`Cause::shown_by`]: the row's comparison is unbounded by `FastMathRewrites`, the
//!   product's record that it left the AIR's fast-flagged arithmetic to the device's grouping),
//!   whatever candidate it executed and whatever words differ, since the user accepted that
//!   difference with no bound; its observations are the evidence the review read, and each reads
//!   [`ReadingState::Changed`] once its slot's row no longer shows the cause.
//! - **Causes of the old engine's mismatches** ([`OldMismatch`]): per case, why its recorded
//!   old-engine rows mismatched. The ledger itself ([`LedgerRow`]) is generated from the store: one
//!   row per old-engine row recorded as a mismatch, with the case's engine verdicts beside it.
//!
//! An `apple_reading` reading also declares its bounded lanes ([`bound::BoundedLane`]): what each
//! output word computes, from which of the case's inputs. The report recomputes every word the
//! device differs from Metal in against Apple's stated accuracy ([`bound::check`]), and the user's
//! decision of 2026-10-03 ("Accept only within Apple's bound") lets such a reading be signed off
//! only while every one is within it; otherwise the report refuses the sign-off
//! ([`C3Report::refused`]) and names the words. A word where Apple states no accuracy is within
//! Apple's promise when it is the device's own answer, and open when the row records the product's
//! lowering of the word's operation as a near-equivalent of Metal's
//! ([`bound::WordVerdict::NearEquivalent`]). A word within Apple's bound, or where Apple states
//! none, is open too where it shows the difference of a near-equivalent the row records that can
//! be told word by word: a NaN encoding, a denormal, or a `half` rounding the device chose where
//! Metal's answer is reproducible ([`crate::observation_contract::value_difference`]).
//!
//! The old engine's recorded rows are C3's baseline whatever their executor ABI: its GPU runs stay
//! restricted, so its rows are not re-recorded. The engine's rows count only when current under
//! this build, by the evidence census's rule (`index::candidate_slot_state`): recorded at it, or at
//! another build while what they executed is this build's artifact
//! ([`VenueKeys::artifact_is_current`]). And they count only when they ran on a device and driver
//! their venue allows ([`DeviceIdentity`], [`Venue::devices`]): C3 is measured fresh.
//!
//! **The Metal reference** is data too ([`mod@reference`],
//! `validation/cutover/c3-references.json`): each case is judged against one Metal row, the
//! reference environment's (the Apple M3 Max). A case is Metal-qualified when it has such a row,
//! and every candidate row's Metal dependency is that row.
//!
//! The devices are data ([`Venues`], `validation/cutover/c3-venues.json`): each venue's kind, its
//! operator environment, the profile that describes it, the cases it must cover and in which
//! legs, and its declared device and driver. The report judges every venue by the same rules.
//!
//! **Refusals**. A case the executor refused on a venue's device holds a refusal record in the slot
//! (`crate::refusal`), judged current by the rule a row is. A current refusal is
//! [`Verdict::Refused`], open work, not `missing`; or [`Verdict::DeviceGap`] when it is the
//! product's typed device-capability refusal and C1's reviewed list holds, for the venue's profile,
//! that outcome's bucket as a device gap (`device_lacks` or `near_equivalent`): C1's list is the
//! one source of the user's device-gap sign-offs, and each entry a device gap reads counts among
//! C3's entries awaiting or holding sign-off ([`DeviceGapRow`]). A device-capability refusal C1's
//! list does not hold is a proposed C1 entry ([`GapProposal`]). A product refusal of a family the
//! launch leaves out of scope (ray tracing, by its typed outcome: [`OutOfScope::of_recorded`]) is
//! [`Verdict::OutOfScope`], counted per slot in its own column: never refused work, never a device
//! gap or a proposal, and never keeping the slot from holding. Every other refusal (the binder's, a
//! feature the device did not enable, an uncontained form, a product or executor refusal) reads
//! `refused` with no sign-off path.
//!
//! **Reference gaps**. A case whose current Metal row is a reference gap
//! (`MetalStatus::ReferenceGap`: Metal's pipeline creation named an Apple GPU family the reference
//! device lacks) has no Metal output, so no row of it can be current. Where a slot would read it
//! missing or stale it reads [`Verdict::ReferenceGap`], counted in its own column and not keeping
//! the slot from holding; it is never `missing_without_metal`. Each such case is listed in the
//! report ([`ReferenceGapRow`]) and C3 holds only once the list's third part
//! ([`ReferenceGapEntry`]) holds it with the user's sign-off.

pub mod bound;
pub mod reference;

use crate::c1::reviewed::{ListClass, ReviewedList as C1List};
use crate::candidate::engine::{engine_slot, is_engine_slot};
use crate::case::{AuthoredCase, Stage};
use crate::check::CheckedCase;
use crate::companion::Companion;
use crate::coverage::Leg;
use crate::cutover::scope::OutOfScope;
use crate::cutover::{List, Review, SignOff};
use crate::index::EvidenceState;
use crate::metal::ORACLE_ABI;
use crate::observation::{
    validate_hash, Backend, CandidateObservation, CandidateStatus, MetalObservation,
};
use crate::observation_contract::ComparisonClass;
use crate::refusal::{Refusal, RefusalClass, RefusalObservation};
use crate::source::public_sources;
use crate::source::SourceRow;
use crate::store::CorpusStore;
use base64::Engine as _;
use bound::{BoundCheck, BoundedLane, UNSTATED_WORDS};
use metal2vulkan::device::{DeviceDescription, DeviceProfile, DriverWorkaround};
use metal2vulkan::pipeline::PrecisionNote;
use metal2vulkan::vk::DriverId;
use metal2vulkan_air::abi::{Builtin, ParamRole};
use reference::{CaseFacts, Judge, ReferenceReport, References};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// Why a case's reading differs from Metal's, or may (one vocabulary for the readings and the old
/// engine's mismatches).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cause {
    /// Metal only bounds the value: an ulp or absolute bound, a formula over bounded parts,
    /// "larger otherwise" outside a stated range, or no stated accuracy. Metal's answer is one
    /// reading of Apple's implementation; another device's answer within Vulkan's precision is
    /// as valid, and the product records `DeviceTranscendentals`.
    AppleReading,
    /// The AIR's fast-math flags (`reassoc`, `contract`) allow any grouping of the arithmetic:
    /// Metal's value is one grouping, which moves between Metal releases, and the device's is
    /// another (`FastMathRewrites`).
    FastMathRegrouping,
}

impl Cause {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AppleReading => "apple_reading",
            Self::FastMathRegrouping => "fast_math_regrouping",
        }
    }

    /// Whether a [`Reading`] of this cause is keyed by its cause rather than by its candidate. An
    /// [`Cause::AppleReading`] is accepted only within Apple's stated accuracy, word by word
    /// ([`bound::check`]), so it reviews only the candidates whose words were checked. A
    /// [`Cause::FastMathRegrouping`] is accepted with no bound (the user's answer of 2026-10-03):
    /// any grouping the AIR's fast-math flags allow is Metal's latitude, on every device and at
    /// every candidate, so the reading reviews each row that shows the cause ([`Cause::shown_by`]).
    pub fn keyed_by_cause(self) -> bool {
        match self {
            Self::AppleReading => false,
            Self::FastMathRegrouping => true,
        }
    }

    /// Whether `row` shows this cause, for a cause a reading is keyed by
    /// ([`Cause::keyed_by_cause`]): a [`Cause::FastMathRegrouping`] row mismatches Metal and its
    /// comparison is unbounded by `PrecisionNote::FastMathRewrites`, the product's record that a
    /// module it executed leaves an operation the AIR flags `reassoc` and `contract` to the
    /// device's grouping, or by `PrecisionNote::HalfOutputFastMathBlend`, its record that a
    /// fast-math function's `half` color output meets a blend Metal's compiler fuses and regroups
    /// per pipeline (the owner's acceptance extended to blends, 2026-10-11). A row whose module no
    /// longer records it (its grouping pinned, or the arithmetic gone) no longer shows the cause.
    /// An [`Cause::AppleReading`] is keyed by candidate, so no row shows it by itself.
    pub fn shown_by(self, row: &CandidateObservation) -> bool {
        match self {
            Self::AppleReading => false,
            Self::FastMathRegrouping => {
                row.status == CandidateStatus::Mismatch
                    && matches!(
                        &row.comparison,
                        ComparisonClass::Unbounded(notes)
                            if notes.contains(&PrecisionNote::FastMathRewrites)
                                || notes.contains(&PrecisionNote::HalfOutputFastMathBlend)
                    )
            }
        }
    }
}

/// One case whose engine readings were reviewed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reading {
    pub case_id: String,
    /// The case's name, checked against the store.
    pub name: String,
    pub cause: Cause,
    /// The bound Metal guarantees for what the case reads, and Vulkan's where it matters, with the
    /// tables or decisions that state them.
    pub bound: String,
    /// The readings reviewed. For a cause keyed by candidate, each one is the key a row must
    /// equal to be reviewed; for a cause keyed by its cause ([`Cause::keyed_by_cause`]), the
    /// evidence the review read, whose slot's row must still show the cause.
    pub observations: Vec<ReviewedObservation>,
    /// For an [`Cause::AppleReading`]: what each output word computes, so that the report can
    /// judge every differing word against Apple's stated accuracy ([`bound::check`]). A reading
    /// is signed off only with lanes, and only while every differing word is within them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lanes: Vec<BoundedLane>,
    /// The evidence that measured and explained the reading, the reviewer's note, and the user's
    /// sign-off.
    pub review: Review,
}

/// One reviewed observation: the reading of one candidate in one engine slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedObservation {
    /// The engine store slot (`ENV@engine`, with `+robustness-off` for that leg): host,
    /// engine, profile restriction, and leg.
    pub environment_id: String,
    /// The candidate: the SHA-256 of the pipeline's SPIR-V the row executed.
    pub spv_sha256: String,
    /// The reading reviewed: the candidate's output.
    pub candidate_output_sha256: String,
    /// Metal's reading it was compared with.
    pub golden_output_sha256: String,
    /// The output's length in 32-bit words.
    pub words: usize,
    /// The 32-bit words in which the two readings differ.
    pub differing_words: usize,
    /// The observed values against Metal's, in words.
    pub values: String,
}

/// Why one case's recorded old-engine rows mismatched.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OldMismatch {
    pub case_id: String,
    pub name: String,
    pub cause: Cause,
    /// The evidence that explains the mismatch, the reviewer's note, and the user's sign-off.
    pub review: Review,
}

/// A case the reference device cannot run: its current Metal row is a reference gap of
/// `apple_family`, and the user's sign-off accepts that C3 holds without it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceGapEntry {
    pub case_id: String,
    pub name: String,
    /// The Apple GPU family the case's Metal row names (`GPUFamily<N>`).
    pub apple_family: u32,
    /// The evidence that measured the gap, the reviewer's note, and the user's sign-off.
    pub review: Review,
}

/// A device gap a case's answer meets: no refusal, the case runs, and its answer is one Metal's API
/// permits, not the reference's, where no Vulkan mechanism gives the reference's. Whether a venue's
/// device has it is decided from the device's description ([`AnswerGap::of_device`]), never from
/// its name, so any device of that shape has it; where the case's structure decides whether its
/// answer can meet it ([`AnswerGap::met_by`]), C3 names the gap only for a case that meets it. The
/// user's decision of 2026-10-06: every answer gap stays open until an exact emulation exists. A
/// mismatch it explains is still counted as a mismatch, with the gap named as its reason
/// ([`OpenReason::AnswerGap`]), and no entry is ever signed off.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerGap {
    /// The device may run a fragment subgroup of fewer than 8 lanes, two quads. Metal's
    /// simdgroup is 32 lanes, and Metal permits a partly occupied one: the product runs a narrower
    /// fragment subgroup as such a simdgroup (`Tier::PartialFragment`), so the device's answer is
    /// a valid Metal answer, but not the reference's where the case counts quads of its draw
    /// that share a simdgroup on Apple's GPU and cannot share a subgroup of one quad. Vulkan has
    /// no way to widen a fragment subgroup: a device with the gap states one size for its
    /// fragment stage (`[minSubgroupSize, maxSubgroupSize]` is `[4, 4]` on arm64 lavapipe, the
    /// measured one), so no required size reaches 32 lanes, and nothing relates the invocations
    /// of two subgroups. A function whose answer reads the simdgroup's members records
    /// `PrecisionNote::FragmentSimdgroupGrouping`, which the runtime announces
    /// (`metal2vulkan-degrade-fragment-simdgroup-grouping`): a device takes that note unless its
    /// driver runs the fragment on Metal (`DriverDeviation::MetalFragmentSimdgroups`), and such a
    /// device runs Metal's own 32-lane simdgroups and lacks the gap, so the gap's answer is
    /// always announced.
    FragmentSubgroupUnderTwoQuads,
    /// Apple's rasterizer draws a triangle it clips as pieces, and a pixel a shared edge crosses
    /// runs its fragments once per piece, each reading that piece's coverage as `[[sample_mask]]`
    /// (measured on the M3 Max and the M4). A Vulkan device runs the pixel's fragments with the
    /// whole primitive's coverage, Metal's answer for a primitive it does not clip; no Vulkan
    /// mechanism follows Apple's clipper. A device that draws through Metal (MoltenVK) splits as
    /// Metal does and lacks the gap. A case meets it when its answer can read a split: a fragment
    /// case drawn by the generated full-screen triangle, which extends past the viewport, at more
    /// than one sample, whose function reads `[[sample_mask]]`.
    ClippedPrimitiveSplit,
    /// The device shades a triangle it clips as several triangles
    /// (`DriverWorkaround::ClippedTrianglesSplitPixelFragments`, which its profile carries:
    /// lavapipe, whose clipper emits the clipped polygon as a fan of triangles), so a quad a seam
    /// between the pieces crosses runs once per piece, each run holding only its own piece's
    /// pixels as active lanes, where Metal's quad holds all four. Metal permits a partly occupied
    /// quad, so the device's answer is a valid Metal answer, not the reference's; no Vulkan
    /// mechanism keeps the clipped triangle one primitive (a guard band emulated by scaling the
    /// viewport and the position is exact only within the viewport limits). The module records
    /// `PrecisionNote::QuadsSplitByClipping`, which the runtime announces. A case meets it when its
    /// answer can read a split quad: a fragment case drawn by the generated full-screen triangle,
    /// which extends past the viewport, whose function runs a quad or simdgroup operation over
    /// its active lanes (`GroupOp::reads_active_lanes`).
    QuadsSplitByClipping,
}

impl AnswerGap {
    /// Every gap.
    pub const ALL: [AnswerGap; 3] = [
        AnswerGap::FragmentSubgroupUnderTwoQuads,
        AnswerGap::ClippedPrimitiveSplit,
        AnswerGap::QuadsSplitByClipping,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FragmentSubgroupUnderTwoQuads => "fragment_subgroup_under_two_quads",
            Self::ClippedPrimitiveSplit => "clipped_primitive_split",
            Self::QuadsSplitByClipping => "quads_split_by_clipping",
        }
    }

    /// Whether a device of `description`, from which the product derives `profile`, has the
    /// gap. The subgroup gap: the fragment stage runs subgroups, and the smallest it may run one
    /// with (the product's subgroup model: the pinnable range, else the API size) holds fewer
    /// than two quads. The clipped-primitive split: the device does not draw through Metal (its
    /// driver is not MoltenVK). The split quads: the profile carries the driver's
    /// `ClippedTrianglesSplitPixelFragments`.
    pub fn has(self, description: &DeviceDescription, profile: &DeviceProfile) -> bool {
        match self {
            Self::FragmentSubgroupUnderTwoQuads => {
                let fragment = &profile.subgroups().fragment;
                fragment.supported && fragment.sizes.min < 8
            }
            Self::ClippedPrimitiveSplit => description.driver_id != Some(DriverId::MOLTENVK),
            Self::QuadsSplitByClipping => profile
                .workarounds()
                .contains(&DriverWorkaround::ClippedTrianglesSplitPixelFragments),
        }
    }

    /// The gaps a device of `description`, from which the product derives `profile`, has.
    pub fn of_device(
        description: &DeviceDescription,
        profile: &DeviceProfile,
    ) -> BTreeSet<AnswerGap> {
        Self::ALL
            .into_iter()
            .filter(|gap| gap.has(description, profile))
            .collect()
    }

    /// Whether the case's structure decides whether its answer can meet the gap
    /// ([`Self::met_by`]): the clipped-primitive split's and the split quads'. The subgroup gap is
    /// held by the list's entry alone.
    pub fn decided_by_case(self) -> bool {
        match self {
            Self::FragmentSubgroupUnderTwoQuads => false,
            Self::ClippedPrimitiveSplit | Self::QuadsSplitByClipping => true,
        }
    }

    /// Whether the answer of `checked` can meet the gap, by its structure: for the
    /// clipped-primitive split, a fragment case at more than one sample, drawn by the generated
    /// passthrough vertex function's full-screen triangle, whose fragment function has a live
    /// `[[sample_mask]]` argument; for the split quads, a fragment case drawn by that triangle
    /// whose entry calls a quad or simdgroup operation over its active lanes. A gap
    /// [`Self::decided_by_case`] does not decide is met by any case.
    pub fn met_by<S: Borrow<SourceRow>>(self, checked: &CheckedCase<S>) -> bool {
        match self {
            Self::FragmentSubgroupUnderTwoQuads => true,
            Self::QuadsSplitByClipping => {
                let shader = &checked.primary;
                shader.case.stage == Stage::Fragment
                    && matches!(shader.companion, Some(Companion::PassthroughVertex(_)))
                    && reads_active_lanes(shader.entry_ll())
            }
            Self::ClippedPrimitiveSplit => {
                let shader = &checked.primary;
                shader.case.stage == Stage::Fragment
                    && shader
                        .case
                        .raster_sample_count
                        .is_some_and(|count| count > 1)
                    && matches!(shader.companion, Some(Companion::PassthroughVertex(_)))
                    && shader.interface.params.iter().any(|param| {
                        param.live
                            && !param.binding.unused
                            && matches!(
                                param.binding.role,
                                ParamRole::Builtin(
                                    Builtin::SampleMaskIn | Builtin::PostDepthCoverage
                                )
                            )
                    })
            }
        }
    }

    /// The gaps the answer of `checked` can meet.
    pub fn met_by_case<S: Borrow<SourceRow>>(checked: &CheckedCase<S>) -> BTreeSet<AnswerGap> {
        Self::ALL
            .into_iter()
            .filter(|gap| gap.met_by(checked))
            .collect()
    }
}

/// Whether the AIR text `ll` calls a quad or simdgroup operation whose answer depends on which
/// of its group's lanes are active, by the product's decode of the `air.*` name
/// (`GroupOp::reads_active_lanes`).
fn reads_active_lanes(ll: &str) -> bool {
    use metal2vulkan_air::intrinsics::AirIntrinsic;
    ll.lines()
        .filter(|line| line.contains("call "))
        .filter_map(|line| {
            let at = line.find("@air.")?;
            let name = &line[at + 1..];
            let end = name
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '_'))
                .unwrap_or(name.len());
            AirIntrinsic::parse(&name[..end])
        })
        .any(|op| matches!(op, AirIntrinsic::Group(g) if g.op.reads_active_lanes()))
}

/// A case whose answer meets an answer gap: a current mismatching row of it in a slot whose venue's
/// device has `gap`, where the case's structure meets the gap ([`AnswerGap::met_by`]), stays a
/// [`Verdict::Mismatch`] whose reason names the gap ([`OpenReason::AnswerGap`]). The entry's review
/// holds the evidence and the note; its sign-off stays `awaiting_user`, as the user decided on
/// 2026-10-06 that answer gaps stay open until an exact emulation exists, so the list refuses one
/// signed off and the report counts none toward sign-offs. A slot whose device lacks the gap, or a
/// case whose structure does not meet it or is unknown, is judged as any other.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerGapEntry {
    pub case_id: String,
    pub name: String,
    pub gap: AnswerGap,
    /// The evidence that measured the gap, the reviewer's note, and the user's sign-off.
    pub review: Review,
}

/// The reviewed list: the readings, the causes of the old engine's mismatches, the cases the
/// reference device cannot run, and the cases that meet an answer gap.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedList {
    pub readings: Vec<Reading>,
    pub old_mismatches: Vec<OldMismatch>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reference_gaps: Vec<ReferenceGapEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub answer_gaps: Vec<AnswerGapEntry>,
}

/// The committed list is `validation/cutover/c3-reviewed.json`, its entries in the order given.
impl List for ReviewedList {
    const FILE: &'static str = "c3-reviewed.json";
    const NAME: &'static str = "C3 reviewed list";

    /// Well-formed hashes, a name and a bound, each case at most once per part, each reading's
    /// slots engine slots named once, word counts that agree with the hashes (equal outputs
    /// differ in no word), and on every entry a review that passes [`Review::check`] (evidence, a
    /// note, and a well-formed sign-off).
    fn check(&self) -> Result<(), String> {
        let mut cases = BTreeSet::new();
        for reading in &self.readings {
            let at = |what: String| format!("C3 reading {}: {what}", reading.case_id);
            validate_hash("case_id", &reading.case_id).map_err(at)?;
            if !cases.insert(reading.case_id.as_str()) {
                return Err(at("listed twice".into()));
            }
            entry_name(&reading.name).map_err(at)?;
            reading.review.check().map_err(at)?;
            if reading.bound.trim().is_empty() {
                return Err(at("states no bound".into()));
            }
            if !reading.lanes.is_empty() && reading.cause != Cause::AppleReading {
                return Err(at(format!(
                    "a {} reading states no bounded lanes; only an apple_reading does",
                    reading.cause.as_str()
                )));
            }
            bound::check_lanes(&reading.lanes).map_err(at)?;
            if reading.cause == Cause::AppleReading
                && reading.review.sign_off.is_signed_off()
                && reading.lanes.is_empty()
            {
                return Err(at(
                    "is signed off, but an apple_reading is signed off only with the bounded \
                     lanes its words are checked against"
                        .into(),
                ));
            }
            if reading.observations.is_empty() {
                return Err(at("lists no observation".into()));
            }
            let mut slots = BTreeSet::new();
            for observation in &reading.observations {
                let at = |what: String| at(format!("{}: {what}", observation.environment_id));
                if !is_engine_slot(&observation.environment_id) {
                    return Err(at("not an engine slot".into()));
                }
                if !slots.insert(observation.environment_id.as_str()) {
                    return Err(at("listed twice".into()));
                }
                for (field, value) in [
                    ("spv_sha256", &observation.spv_sha256),
                    (
                        "candidate_output_sha256",
                        &observation.candidate_output_sha256,
                    ),
                    ("golden_output_sha256", &observation.golden_output_sha256),
                ] {
                    validate_hash(field, value).map_err(at)?;
                }
                if observation.values.trim().is_empty() {
                    return Err(at("states no values".into()));
                }
                let equal = observation.candidate_output_sha256 == observation.golden_output_sha256;
                if observation.words == 0
                    || observation.differing_words > observation.words
                    || equal != (observation.differing_words == 0)
                {
                    return Err(at(format!(
                        "{} differing of {} words disagrees with its output hashes",
                        observation.differing_words, observation.words
                    )));
                }
            }
        }
        let mut cases = BTreeSet::new();
        for cause in &self.old_mismatches {
            let at = |what: String| format!("C3 old mismatch {}: {what}", cause.case_id);
            validate_hash("case_id", &cause.case_id).map_err(at)?;
            if !cases.insert(cause.case_id.as_str()) {
                return Err(at("listed twice".into()));
            }
            entry_name(&cause.name).map_err(at)?;
            cause.review.check().map_err(at)?;
        }
        let mut cases = BTreeSet::new();
        for gap in &self.reference_gaps {
            let at = |what: String| format!("C3 reference gap {}: {what}", gap.case_id);
            validate_hash("case_id", &gap.case_id).map_err(at)?;
            if !cases.insert(gap.case_id.as_str()) {
                return Err(at("listed twice".into()));
            }
            entry_name(&gap.name).map_err(at)?;
            if gap.apple_family == 0 {
                return Err(at("names no Apple GPU family".into()));
            }
            gap.review.check().map_err(at)?;
        }
        let mut cases = BTreeSet::new();
        for gap in &self.answer_gaps {
            let at = |what: String| format!("C3 answer gap {}: {what}", gap.case_id);
            validate_hash("case_id", &gap.case_id).map_err(at)?;
            if !cases.insert(gap.case_id.as_str()) {
                return Err(at("listed twice".into()));
            }
            entry_name(&gap.name).map_err(at)?;
            gap.review.check().map_err(at)?;
            if gap.review.sign_off.is_signed_off() {
                return Err(at(format!(
                    "is signed off, but an answer gap ({}) stays open until an exact emulation \
                     exists (the user's decision of 2026-10-06)",
                    gap.gap.as_str()
                )));
            }
        }
        Ok(())
    }
}

impl ReviewedList {
    /// The reading that reviews `row`, if the list holds one: of the same case and, for a cause
    /// keyed by candidate, with an observation of the same slot, candidate, output, and Metal
    /// output; for a cause keyed by its cause, an engine row of any slot that shows the cause.
    fn reviews(&self, row: &CandidateObservation) -> Option<&Reading> {
        self.readings.iter().find(|reading| {
            reading.case_id == row.case_id
                && if reading.cause.keyed_by_cause() {
                    is_engine_slot(&row.environment_id) && reading.cause.shown_by(row)
                } else {
                    reading.observations.iter().any(|observation| {
                        observation.environment_id == row.environment_id
                            && observation.spv_sha256 == row.spv_sha256
                            && observation.candidate_output_sha256 == row.candidate_output_sha256
                            && observation.golden_output_sha256 == row.golden_output_sha256
                    })
                }
        })
    }
}

fn entry_name(name: &str) -> Result<(), String> {
    if name.trim().is_empty() {
        return Err("needs a name".into());
    }
    Ok(())
}

/// What the report reads: the store's cases and observations and the public fixtures' AIR.
#[derive(Clone, Debug, Default)]
pub struct C3Input {
    pub cases: Vec<AuthoredCase>,
    pub metals: Vec<MetalObservation>,
    pub moltenvk: Vec<CandidateObservation>,
    pub vulkan: Vec<CandidateObservation>,
    /// The engine's refusals of both backends (`crate::refusal`).
    pub refusals: Vec<RefusalObservation>,
    /// The AIR of every public fixture: a case over one is a public case.
    pub public_air: BTreeSet<String>,
}

impl C3Input {
    /// The committed store at `root` and the public fixtures. Reads only the case and observation
    /// shards and the fixtures, never the index or a source shard.
    pub fn read(root: &Path) -> Result<C3Input, String> {
        let store = CorpusStore::new(root);
        Ok(C3Input {
            cases: store.read_all_cases()?,
            metals: store.read_metal()?,
            moltenvk: store.read_candidates(Backend::Moltenvk)?,
            vulkan: store.read_candidates(Backend::Vulkan)?,
            refusals: [Backend::Moltenvk, Backend::Vulkan]
                .into_iter()
                .map(|backend| store.read_refusals(backend))
                .collect::<Result<Vec<_>, _>>()?
                .concat(),
            public_air: public_sources()?
                .into_iter()
                .map(|source| source.air_sha256)
                .collect(),
        })
    }
}

/// One case's verdict in one engine slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Verdict {
    /// A current row that matches Metal.
    Match,
    /// A current mismatching row the list reviewed at its candidate.
    Reviewed,
    /// A current refusal of a capability the venue's device lacks, which C1's reviewed list holds
    /// as a device gap of the venue's profile (its sign-off is C1's entry's).
    DeviceGap,
    /// A current product refusal of a family the launch leaves out of scope: counted apart, and it
    /// does not keep the slot from holding.
    OutOfScope(OutOfScope),
    /// No row or refusal, or a stale one, of a case whose current Metal row is a reference gap: no
    /// row of it can be current until a device with the family qualifies it. Counted apart, and it
    /// does not keep the slot from holding; the report's [`ReferenceGapRow`] awaits the user's
    /// sign-off instead.
    ReferenceGap,
    /// A current mismatching row the list does not review; its reason may name what keeps it
    /// open ([`OpenReason`]).
    Mismatch,
    /// A current refusal no device gap of C1's list covers.
    Refused,
    /// A row or refusal not current under this build.
    Stale,
    /// No row or refusal, for a case the slot must cover.
    Missing,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Reviewed => "reviewed",
            Self::DeviceGap => "device_gap",
            Self::OutOfScope(OutOfScope::RayTracing) => "out_of_scope_ray_tracing",
            Self::ReferenceGap => "reference_gap",
            Self::Mismatch => "mismatch",
            Self::Refused => "refused",
            Self::Stale => "stale",
            Self::Missing => "missing",
        }
    }
}

/// A case whose verdict in a slot is not [`Verdict::Match`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotRow {
    pub case_id: String,
    pub name: String,
    pub verdict: Verdict,
    /// The refusal the slot holds, for a refused case, a device gap, or an out-of-scope case.
    pub refusal: Option<Refusal>,
    /// Why the case stays open although its difference is known: `None` for an ordinary mismatch,
    /// stale row, or missing row.
    pub reason: Option<OpenReason>,
}

/// The notes of recorded near-equivalents C3 defers for the whole row (the user's decision of
/// 2026-10-03): a row that carries one ran an approximation no exact path replaces yet, so a
/// difference stays open, "deferred for the future", and is never a reviewed reading. Such a row's
/// mismatch is still counted as one; only its reason names the note, whatever word differs, so a
/// difference the note did not cause is labelled deferred too but hides nothing.
///
/// - The rasterization-rate map decoder's: Apple's GPU decoder reads fractional tile starts only
///   Apple's parameter data holds, and a moved coordinate changes what every later word of the row
///   reads, so no word's difference can be told apart from the note's.
/// - A sparse texture's filtered access (`SparseResidency`): outside Metal's measured footprint
///   rule its residency word is the device's footprint, and its filtered value is the device's
///   filter, whose sub-texel precision need not be Metal's 1/256; an emulation of Metal's measured
///   filter would be exact but is not built. Neither can be told apart word by word: a residency
///   word and a sampled value are ordinary data the function goes on to use (a branch on
///   `resident()`, a blend), so a moved footprint or weight moves whatever later words they feed,
///   and the bytes do not say which words came from a sparse access.
/// - An explicit cube gradient's footprint (`CubeGradientFootprint`): the product passes the
///   Euclidean length of the gradient's two counted components where Apple combines them by its own
///   rule, so near a level or filter boundary a sample selects another level or filter than
///   Metal's, and the sampled value is ordinary data later words use.
/// - A non-quad simdgroup operation before a discard (`QuadsCountedBeforeDiscard`), in a shape
///   whose placement by Metal's compiler is not known: its lanes include the quads the discard
///   drops, which Metal's may leave out, and the count, ballot, or reduction is ordinary data later
///   words use. Where Metal's placement was measured, the module makes it (a sum moved past the
///   discard, or counted ahead of it) and records no note.
/// - An emulated filter's arithmetic (`BicubicFilterArithmetic`): a bicubic sum (and an
///   extended-range linear, 3D or level blend) in the device's multiply and add under weights
///   computed from the fraction, where Metal sums its fixed-point weights exactly onto the
///   format's grid, and the sampled value is ordinary data later words use.
/// - A vertex function's simdgroup operation on a device that groups vertices otherwise than
///   Apple's GPUs and cannot pin the vertex stage's subgroup size (`VertexSimdgroupGrouping`): the
///   operation combines the device's group of vertices, a simdgroup Metal allows, and its value is
///   ordinary data later words use. No exact path exists on such a device.
/// - A fragment function's simdgroup operation on a device whose fragment subgroups are not
///   Metal's simdgroups (`FragmentSimdgroupGrouping`): the operation combines the pixels the
///   device's rasterizer packs together, a simdgroup Metal allows, and its value is ordinary data
///   later words use. No exact path exists on such a device.
/// - A half-float color channel under `textureWriteRoundingMode` 2 that blends with factors
///   other than one and zero (`BlendedWriteRounding`): the device's blend converts its result by
///   its own rounding where Metal rounds it to nearest even, and the stored half is ordinary data
///   later words use. Open until the function blends under the pixel interlock.
/// - A pipeline that covers by alpha under `textureWriteRoundingMode` 2 whose function rounds
///   attachment 0's alpha (`CoverageReadsRoundedAlpha`): the coverage reads the rounded alpha
///   where Metal's reads it before rounding, so a fragment near a coverage threshold may cover
///   another sample, and the stored samples are ordinary data later words use. Open until Metal's
///   coverage pattern is computed in the function, on a device whose coverage is Metal's
///   (elsewhere the device's coverage is its native answer, `DeviceAlphaCoverage`).
/// - An `ARGB8Unorm` attachment blended in none of the measured forms
///   (`RotatedBlendRounding`): Apple's blend computes a factor pair apart in its color and alpha
///   arithmetic, which the rotation swaps for Metal's alpha and blue, and the stored code is
///   ordinary data later words use. Open until the blend state has a measured form.
/// - A fragment function that runs its draw's depth test itself at more than one sample from
///   each sample's computed depth (`ComputedSampleDepth`): under perspective, a slope-scaled bias
///   or another depth range a sample's depth may be up to 2 ulp from the device's, so a close test
///   may decide otherwise and keep or drop a sample's color. Open until the sample's own depth is
///   read exactly.
///
/// A varying the device interpolates (`DeviceInterpolation`) is not deferred: every fragment
/// function that reads one records it on every device but MoltenVK's, so a row-level deferral
/// would hold any mismatch of such a function open as the interpolation's, a coverage or a
/// rate-map difference among them. Its differences stay mismatches for review. Nor is the
/// device's alpha-to-coverage (`DeviceAlphaCoverage`): the device's native answer, which no exact
/// path replaces, recorded by every pipeline that covers by alpha off MoltenVK.
///
/// Row-level only. A note whose difference can be attributed word by word (a NaN encoding, a
/// denormal, a `half` rounding) belongs in a per-word judgement of a reading's differing words,
/// beside the check of each against Apple's bound, never in this list: a row-level match would
/// reopen every signed-off reading whose row merely records the note.
pub const DEFERRED_NEAR_EQUIVALENTS: [PrecisionNote; 11] = [
    PrecisionNote::RasterizationRateMap,
    PrecisionNote::SparseResidency,
    PrecisionNote::CubeGradientFootprint,
    PrecisionNote::QuadsCountedBeforeDiscard,
    PrecisionNote::BicubicFilterArithmetic,
    PrecisionNote::VertexSimdgroupGrouping,
    PrecisionNote::FragmentSimdgroupGrouping,
    PrecisionNote::BlendedWriteRounding,
    PrecisionNote::CoverageReadsRoundedAlpha,
    PrecisionNote::RotatedBlendRounding,
    PrecisionNote::ComputedSampleDepth,
];

/// Why a case stays open in a slot although its difference is known: deferred until an exact
/// path exists, never accepted as a known difference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenReason {
    /// The row ran a recorded near-equivalent ([`DEFERRED_NEAR_EQUIVALENTS`]).
    NearEquivalent(PrecisionNote),
    /// No row (the case is missing, or its refusal is recorded): the product refuses a
    /// rasterization-rate map of the case, so no Vulkan executor carries it.
    RateMapRefused(String),
    /// The reviewed list holds the case under an answer gap the venue's device has and the case's
    /// structure meets ([`AnswerGapEntry`]), open until an exact emulation exists (the user's
    /// decision of 2026-10-06).
    AnswerGap(AnswerGap),
}

impl OpenReason {
    /// The reason as `corpus-status --c3` prints it: what was deferred, and the measured bound
    /// or the refusal.
    pub fn label(&self) -> String {
        match self {
            Self::NearEquivalent(note) => format!(
                "deferred{{near_equivalent:{note:?}:{}}}",
                near_equivalent_bound(*note)
            ),
            Self::RateMapRefused(why) => format!("deferred{{rate_map_refused:{why}}}"),
            Self::AnswerGap(gap) => format!("deferred{{answer_gap:{}}}", gap.as_str()),
        }
    }
}

/// The measured bound of a deferred near-equivalent ([`DEFERRED_NEAR_EQUIVALENTS`]).
fn near_equivalent_bound(note: PrecisionNote) -> &'static str {
    match note {
        PrecisionNote::RasterizationRateMap => {
            "tile starts at integers where Apple M3's are fractional: decoder within 0.0194 px \
             physical to screen and 0.002 px screen to physical of Apple's GPU on the maps \
             measured; a rate-mapped pass's points within 0.025 px of Metal's where the device \
             keeps the viewport's fractions, and an axis's last physical line, where its last \
             tile maps under 2 screen pixels to one, shaded where Metal may leave it"
        }
        PrecisionNote::SparseResidency => {
            "no bound: outside Metal's measured footprint rule (a runtime sampler, a mip level, \
             repeat or mirrored addressing, a texel offset) a sample's or gather's residency is the \
             device's footprint, which may count a texel of weight 0; a filtered value is the \
             device's filter at its own sub-texel precision (Apple M3 Max's is 1/256, lavapipe's \
             finer: values within 2^-9 on the cases measured, and a gather's texels one over \
             where its position crosses a texel's center), and an unmapped texel's sampled value \
             is the device's without residencyNonResidentStrict; exact path: an emulation of \
             Metal's measured filter"
        }
        PrecisionNote::CubeGradientFootprint => {
            "a cube gradient's footprint from the Euclidean length of its two counted components, \
             where Apple M3 Max's lies 2.2% below to 1.7% above it by their ratio, and the level the \
             device's own threshold above 0, where Apple's minifies from a footprint of \
             1 + 3/2048: a sample near a level or filter boundary can take another level or \
             filter"
        }
        PrecisionNote::QuadsCountedBeforeDiscard => {
            "a non-quad simdgroup operation before a discard, in a shape whose placement by \
             Metal's compiler is not known (another operation than a sum, a use of its result \
             between, a nested discard, which it was measured placing both ways), counts among \
             its lanes each quad whose every \
             thread the discard drops, where Apple M3 Max's and M4's may leave it out: its count \
             is up to the dropped quads' lanes over Metal's (16 of 32 on RADV, NVIDIA and \
             MoltenVK, 8 of 16 on ANV, 4 of 8 on x86 lavapipe, with every other quad \
             discarded); after the discard every venue measured counts Metal's lanes"
        }
        PrecisionNote::BicubicFilterArithmetic => {
            "a bicubic sum (and an extended-range linear, 3D or level blend) in the device's \
             f32 multiply and add under the Catmull-Rom weights computed from the fraction, a \
             half result the device's narrowing, where Apple M3 Max's sums its fixed-point \
             weights (each axis within 4.83e-4 of the polynomial) exactly onto the format's \
             grid: a few grid steps apart (1/4080 of an 8-bit format), up to 2.1e-3 on an \
             extended-range sRGB texture; no exact path: the device's arithmetic is the form"
        }
        PrecisionNote::VertexSimdgroupGrouping => {
            "a vertex function's simdgroup holds the device's group of vertices: 8 a subgroup, \
             interleaved (vertices 0, 2, ..., 14 and 1, 3, ..., 15), on ANV's Mesa 26.2.4, where \
             Apple M4's holds 32 consecutive vertices (rvc_simd_vertex: its sums, ballots and \
             shuffles combine those 8); no exact path: the vertex stage takes no required \
             subgroup size there"
        }
        PrecisionNote::FragmentSimdgroupGrouping => {
            "a fragment function's simdgroup holds the pixels the device's rasterizer packs into a \
             subgroup, where Apple's GPUs run an aligned 8x4 block one primitive covers whole as \
             one simdgroup and pack other quads by their own rasterizer (gsm2_fragment_lanes); no \
             exact path: no Vulkan device states or pins its packing"
        }
        PrecisionNote::BlendedWriteRounding => {
            "a half-float color channel under textureWriteRoundingMode 2 blended with factors \
             other than one and zero stores the device's rounding of the blend's result, where \
             Metal rounds it to nearest even (MoltenVK's toward zero; under a constant factor \
             Metal's result is neither rule's on 2,376 of 580,508 RGBA16Float channels); \
             exact path: the blend in the function under the pixel interlock"
        }
        PrecisionNote::CoverageReadsRoundedAlpha => {
            "a pipeline covering by alpha under textureWriteRoundingMode 2 stores Metal's \
             nearest-even half alpha, and its coverage reads that rounded alpha where Metal's \
             reads the alpha before rounding: a fragment whose alpha lies within half a half step \
             of a coverage threshold may cover one sample more or fewer (2 of 16,384 alphas on an \
             Apple M3 Max); exact path: Metal's coverage pattern computed in the function, \
             where the device covers as Metal does"
        }
        PrecisionNote::RotatedBlendRounding => {
            "an ARGB8Unorm attachment blended in none of the measured forms may store a channel \
             one code apart from Metal's: Apple's blend computes a factor pair apart in its color \
             and alpha arithmetic, and the rotation moves Metal's blue into the image's alpha \
             and Metal's alpha into its red (up to 1,767 of 65,536 tie-heavy texels over 440 \
             swept states); exact path: a measured form for the state"
        }
        PrecisionNote::ComputedSampleDepth => {
            "a fragment function that runs its draw's depth test itself at more than one sample \
             computes each sample's depth from its vertex function's window depth; under \
             perspective, a slope-scaled bias or another depth range it may be up to 2 ulp from \
             the depth the device stores, so a close test may decide \
             otherwise; exact path: the sample's own depth read exactly"
        }
        _ => "unbounded",
    }
}

/// The deferred near-equivalent `row` ran, if it records one: in its comparison class, which its
/// executed modules' precision notes derive, or as a degradation its run announced whose kind is
/// the note's ([`CandidateObservation::announced_degradations`], `Degradation::precision_note`). A
/// rate-mapped pass the runtime draws records its map only so: the runtime makes the map and
/// announces `rasterization-rate-map`, and no executed module's notes name it. An announced kind
/// that is no deferred note's (a cost, or another precision note) notes nothing.
fn deferred_note(row: &CandidateObservation) -> Option<PrecisionNote> {
    let class = match &row.comparison {
        ComparisonClass::Unbounded(notes) => Some(notes),
        _ => None,
    };
    let announced = row
        .announced_degradations()
        .filter_map(|kind| kind.precision_note())
        .collect::<BTreeSet<_>>();
    DEFERRED_NEAR_EQUIVALENTS
        .into_iter()
        .find(|note| class.is_some_and(|notes| notes.contains(note)) || announced.contains(note))
}

/// Why the product refuses one of `case`'s rasterization-rate maps (the render pass's, or a
/// buffer's), if it does.
fn rate_map_refusal(case: &AuthoredCase) -> Option<OpenReason> {
    let maps = case
        .render_pass_rate_map
        .iter()
        .chain(
            case.buffers
                .iter()
                .filter_map(|b| b.rasterization_rate_map.as_ref()),
        )
        .chain(
            case.argument_buffer_buffers
                .iter()
                .filter_map(|b| b.rasterization_rate_map.as_ref()),
        );
    maps.into_iter().find_map(|map| {
        metal2vulkan::bind::RasterizationRateMap::new(&map.descriptor())
            .err()
            .map(|error| OpenReason::RateMapRefused(error.to_string()))
    })
}

/// One engine slot of a venue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotReport {
    pub kind: VenueKind,
    /// The venue's operator environment ID.
    pub venue: String,
    pub slot: String,
    pub leg: Leg,
    /// The cases the slot must cover: the venue's [`Population`] in the legs it lists, none in
    /// another leg. Every other case
    /// with a row in the slot is judged too, and so is every other case its Metal row judges in a
    /// leg the venue lists: one the slot never recorded reads [`Verdict::Missing`].
    pub population: usize,
    pub verdicts: BTreeMap<Verdict, usize>,
    /// Every judged case whose verdict is not [`Verdict::Match`], in case order.
    pub rows: Vec<SlotRow>,
    /// Every judged case whose verdict is [`Verdict::Match`], by case ID.
    pub matched: BTreeSet<String>,
    /// Unreviewed current mismatches of cases the old engine's baseline did not record as a
    /// mismatch (C3's "zero new mismatches"; only in a venue that holds the baseline,
    /// [`Population::OldEngineMatched`]).
    pub new_mismatches: usize,
    /// Missing cases that have no Metal reference row either ([`reference::Judge::judging`]):
    /// they need Metal qualification before any executor can compare them.
    pub missing_without_metal: usize,
    /// Stale rows of cases that have no Metal reference row: whatever else the row is stale for,
    /// it cannot be current until Metal qualifies the case under this oracle. A row depends on its
    /// Metal row's content (its input and output hashes), and the oracle decides only which Metal
    /// row is current, so once Metal requalifies such a case with the same input and output, a row
    /// stale for this alone is current again with no device run. A refusal compares nothing with
    /// Metal and is never counted.
    pub stale_without_metal: usize,
    /// Stale rows or refusals that recorded a device or driver the venue does not allow
    /// ([`DeviceIdentity`], [`Venue::devices`]): recorded on another device, or before or after a
    /// driver update.
    pub stale_device: usize,
}

impl SlotReport {
    pub fn count(&self, verdict: Verdict) -> usize {
        self.verdicts.get(&verdict).copied().unwrap_or(0)
    }

    /// The verdict of the case `case_id` in this slot, or `None` when the slot did not judge it
    /// (outside its population, with nothing recorded and no Metal row to judge a record by, or in
    /// a leg the venue does not list).
    pub fn verdict_of(&self, case_id: &str) -> Option<Verdict> {
        if self.matched.contains(case_id) {
            return Some(Verdict::Match);
        }
        self.rows
            .iter()
            .find(|row| row.case_id == case_id)
            .map(|row| row.verdict)
    }

    /// Every judged case is current and matches, is reviewed, is a device gap, is out of the
    /// launch's scope, or is a reference gap (which the report holds to the user's sign-off), and
    /// the population is covered.
    pub fn holds(&self) -> bool {
        [
            Verdict::Mismatch,
            Verdict::Refused,
            Verdict::Stale,
            Verdict::Missing,
        ]
        .into_iter()
        .all(|verdict| self.count(verdict) == 0)
    }
}

/// What the store holds for one reviewed observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadingState {
    /// The slot's row has the reviewed candidate, output, and Metal output.
    Holds,
    /// The slot's row has another candidate, output, or Metal output: review it again.
    Changed,
    /// The slot holds no row of the case.
    Unobserved,
    /// The store has no such case.
    UnknownCase,
    /// The entry disagrees with the store: the case's name, or the words of the row it reviews.
    Inconsistent(String),
}

impl ReadingState {
    pub fn label(&self) -> String {
        match self {
            Self::Holds => "holds".into(),
            Self::Changed => "changed".into(),
            Self::Unobserved => "unobserved".into(),
            Self::UnknownCase => "unknown_case".into(),
            Self::Inconsistent(why) => format!("inconsistent{{{why}}}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadingRow {
    pub case_id: String,
    pub name: String,
    pub environment_id: String,
    pub cause: Cause,
    /// Whether the slot's row is current under this build (only a current row is evidence).
    pub current: bool,
    pub state: ReadingState,
    /// For an [`Cause::AppleReading`] whose slot holds the reviewed row: every word it differs
    /// from Metal in, judged against Apple's stated accuracy.
    pub bound: Option<BoundCheck>,
}

/// One old-engine row recorded as a mismatch, with the case's engine verdicts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerRow {
    pub case_id: String,
    pub name: String,
    pub backend: Backend,
    pub environment_id: String,
    pub executor_abi: String,
    /// The 32-bit words in which the row's output differs from the Metal output it was compared
    /// with, when the store still holds that Metal output.
    pub differing_words: Option<usize>,
    pub words: usize,
    /// Whether the row was compared with the case's current Metal output.
    pub against_current_metal: bool,
    /// Whether the row's output equals the case's current Metal output (it would match today).
    pub output_is_current_metal: bool,
    /// The case's verdict in the stated engine slot of the row's venue: the venue of the row's
    /// environment, else the venue holding its backend's baseline; missing without either.
    pub engine: Verdict,
    /// The status that slot's row recorded, current or not (a stale row's status is not evidence,
    /// but says what the last run saw).
    pub engine_recorded: Option<CandidateStatus>,
    pub cause: Option<Cause>,
}

/// The part of the list an entry is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ListPart {
    /// A [`Reading`].
    Reading,
    /// An [`OldMismatch`].
    OldMismatch,
}

impl ListPart {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reading => "reading",
            Self::OldMismatch => "old_mismatch",
        }
    }
}

/// One entry of the list and the user's decision on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryRow {
    pub part: ListPart,
    pub case_id: String,
    pub name: String,
    pub cause: Cause,
    pub sign_off: SignOff,
}

/// One entry of C1's reviewed list that a [`Verdict::DeviceGap`] reads: its bucket, its class,
/// the user's decision on it, and how many judged slot cases it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceGapRow {
    pub profile: String,
    pub outcome: String,
    pub key: String,
    pub class: ListClass,
    pub sign_off: SignOff,
    pub cases: usize,
}

/// A device-capability refusal C1's reviewed list does not hold for the venue's profile: the C1
/// entry to propose (awaiting the user) for the bucket, and how many judged slot cases read
/// `refused` by it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GapProposal {
    pub profile: String,
    pub outcome: String,
    pub key: String,
    pub cases: usize,
}

/// A case whose current Metal row is a reference gap, or that the list holds as one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceGapRow {
    pub case_id: String,
    pub name: String,
    /// The Apple GPU family the case's current Metal row names; `None` when the store holds no
    /// current reference gap of the case.
    pub recorded: Option<u32>,
    /// The list's entry: the family it names and the user's decision; `None` when the list does
    /// not hold the case (an entry to propose).
    pub listed: Option<(u32, SignOff)>,
    /// The judged slot cases that read [`Verdict::ReferenceGap`].
    pub slots: usize,
}

impl ReferenceGapRow {
    /// The list's entry holds the recorded gap and the user signed it off.
    pub fn is_signed_off(&self) -> bool {
        !self.is_stale()
            && self
                .listed
                .as_ref()
                .is_some_and(|(_, sign_off)| sign_off.is_signed_off())
    }

    /// A listed entry that names no gap the store records: none, or another family.
    pub fn is_stale(&self) -> bool {
        self.listed
            .as_ref()
            .is_some_and(|(family, _)| self.recorded != Some(*family))
    }

    /// `listed(signed_off(...))`, `listed(awaiting_user(...))`, `stale(...)`, or `unlisted`.
    pub fn label(&self) -> String {
        match &self.listed {
            Some((_, sign_off)) if self.is_stale() => format!("stale({})", sign_off.label()),
            Some((_, sign_off)) => format!("listed({})", sign_off.label()),
            None => "unlisted".into(),
        }
    }
}

/// An entry of the list's answer gaps: its case, its gap, and how many judged slot cases are
/// mismatches it names ([`OpenReason::AnswerGap`]). Every one is open: no answer gap is signed off.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnswerGapRow {
    pub case_id: String,
    pub name: String,
    pub gap: AnswerGap,
    pub slots: usize,
}

/// A sign-off the report refuses: a signed-off `apple_reading` with a differing word that is not
/// within Apple's stated accuracy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefusedSignOff {
    pub case_id: String,
    pub name: String,
    /// The slot and each word that keeps the reading open.
    pub words: Vec<String>,
}

/// C3's report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct C3Report {
    pub slots: Vec<SlotReport>,
    pub readings: Vec<ReadingRow>,
    pub ledger: Vec<LedgerRow>,
    /// Cases the list gives a cause for that no recorded old-engine row mismatched.
    pub stale_causes: Vec<String>,
    /// Every list entry, readings first, in the list's order, with its sign-off.
    pub entries: Vec<EntryRow>,
    /// The C1 entries the report's device gaps read, in (profile, outcome, key) order.
    pub device_gaps: Vec<DeviceGapRow>,
    /// The C1 entries the report's device-capability refusals propose, in the same order.
    pub gap_proposals: Vec<GapProposal>,
    /// Signed-off readings whose words are not all within Apple's stated accuracy.
    pub refused: Vec<RefusedSignOff>,
    /// Every case with a current reference-gap Metal row, and every one the list holds, in case
    /// order.
    pub reference_gaps: Vec<ReferenceGapRow>,
    /// The list's answer gaps, in its order.
    pub answer_gaps: Vec<AnswerGapRow>,
    /// The Metal reference environment's rows ([`mod@reference`]).
    pub reference: ReferenceReport,
}

impl C3Report {
    /// Ledger cases without a cause.
    pub fn unexplained(&self) -> BTreeSet<&str> {
        self.ledger
            .iter()
            .filter(|row| row.cause.is_none())
            .map(|row| row.case_id.as_str())
            .collect()
    }

    /// Reviewed observations that no longer describe the store.
    pub fn stale_readings(&self) -> usize {
        self.readings
            .iter()
            .filter(|row| row.state != ReadingState::Holds)
            .count()
    }

    /// List entries (readings and causes, and the C1 entries the device gaps read) and recorded
    /// reference gaps the user has signed off. Answer gaps are never signed off: they stay open
    /// mismatches ([`OpenReason::AnswerGap`]).
    pub fn signed_off(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| &entry.sign_off)
            .chain(self.device_gaps.iter().map(|gap| &gap.sign_off))
            .filter(|sign_off| sign_off.is_signed_off())
            .count()
            + self
                .reference_gaps
                .iter()
                .filter(|gap| gap.is_signed_off())
                .count()
    }

    /// List entries (readings and causes, and the C1 entries the device gaps read) and recorded
    /// reference gaps, listed or not, the user has not signed off.
    pub fn awaiting_sign_off(&self) -> usize {
        self.entries.len() + self.device_gaps.len() + self.reference_gaps.len()
            - self.stale_reference_gaps()
            - self.signed_off()
    }

    /// Listed reference gaps the store no longer records ([`ReferenceGapRow::is_stale`]).
    pub fn stale_reference_gaps(&self) -> usize {
        self.reference_gaps
            .iter()
            .filter(|gap| gap.is_stale())
            .count()
    }

    /// C3 holds: every slot holds, every reviewed observation holds, every old-engine mismatch
    /// has a cause the list still needs, every listed reference gap is recorded, the user signed
    /// off every entry and every recorded reference gap, and each sign-off is one the report does
    /// not refuse.
    pub fn holds(&self) -> bool {
        self.slots.iter().all(SlotReport::holds)
            && self.stale_readings() == 0
            && self.unexplained().is_empty()
            && self.stale_causes.is_empty()
            && self.stale_reference_gaps() == 0
            && self.awaiting_sign_off() == 0
            && self.refused.is_empty()
    }
}

fn decode(b64: &str) -> Option<Vec<u8>> {
    base64::engine::general_purpose::STANDARD.decode(b64).ok()
}

/// The 32-bit words in which two outputs of one length differ (a trailing partial word counts as
/// one), or `None` when the lengths differ.
fn differing_words(left: &[u8], right: &[u8]) -> Option<usize> {
    (left.len() == right.len()).then(|| {
        left.chunks(4)
            .zip(right.chunks(4))
            .filter(|(a, b)| a != b)
            .count()
    })
}

fn word_count(bytes: &[u8]) -> usize {
    bytes.len().div_ceil(4)
}

/// Build C3's report over `input`, with C3's reviewed list `list`, C1's reviewed list `c1` (the
/// device gaps a refusal may read), the venues, the Metal reference environment `references`, and
/// the answer gaps each case meets (`facts`).
pub fn report(
    input: &C3Input,
    list: &ReviewedList,
    c1: &C1List,
    venues: &Venues,
    keys: &VenueKeys,
    references: &References,
    facts: &dyn CaseFacts,
) -> C3Report {
    let cases: BTreeMap<&str, &AuthoredCase> = input
        .cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect();
    let (judge, reference) = Judge::new(references, &input.cases, &input.metals);
    // Each case's Metal rows current under this oracle.
    let mut current = HashMap::<&str, Vec<&MetalObservation>>::new();
    for metal in &input.metals {
        if let Some(case) = cases.get(metal.case_id.as_str()) {
            if metal.dependency_matches(case, ORACLE_ABI) {
                current
                    .entry(case.case_id.as_str())
                    .or_default()
                    .push(metal);
            }
        }
    }
    // The one Metal row each case is judged against: every Metal dependency below reads it alone.
    let metals: HashMap<&str, &MetalObservation> = cases
        .values()
        .filter_map(|case| {
            let current = current.get(case.case_id.as_str())?;
            Some((case.case_id.as_str(), judge.judging(current)?))
        })
        .collect();
    // The Apple GPU family of each case's current reference-gap Metal row, read from the primary
    // reference alone.
    let reference_gaps: BTreeMap<&str, u32> = cases
        .values()
        .filter_map(|case| {
            judge
                .reference_gap(case, &input.metals)
                .map(|gap| (case.case_id.as_str(), gap.apple_family))
        })
        .collect();
    // The row a slot holds for a case.
    let rows: HashMap<(&str, &str), &CandidateObservation> = input
        .moltenvk
        .iter()
        .chain(&input.vulkan)
        .map(|row| ((row.case_id.as_str(), row.environment_id.as_str()), row))
        .collect();
    // The refusal a slot holds for a case (a slot holds a row or a refusal, never both).
    let refusals: HashMap<(&str, &str), &RefusalObservation> = input
        .refusals
        .iter()
        .map(|refusal| {
            (
                (refusal.case_id.as_str(), refusal.environment_id.as_str()),
                refusal,
            )
        })
        .collect();
    // Each venue's engine slots, and whether a row or refusal in one recorded another device
    // or driver than its venue declares: such a record is not the venue's evidence.
    let venue_of_slot: HashMap<String, &Venue> = venues
        .venues
        .iter()
        .flat_map(|venue| {
            [Leg::Stated, Leg::RobustnessOff]
                .map(|leg| (engine_slot(&venue.environment_id, None, leg), venue))
        })
        .collect();
    let other_device = |slot: &str, environment: &serde_json::Value| {
        venue_of_slot.get(slot).is_some_and(|venue| {
            !DeviceIdentity::recorded(environment).is_some_and(|recorded| venue.allows(&recorded))
        })
    };
    let row_on_other_device =
        |row: &CandidateObservation| other_device(&row.environment_id, &row.environment);
    // The Metal row a record of `case` is judged against.
    let judged_by = |case: &AuthoredCase| -> Option<&MetalObservation> {
        metals.get(case.case_id.as_str()).copied()
    };
    // The entry of C1's reviewed list that holds `refusal` as a device gap of the venue's profile:
    // the product's typed device-capability refusal, in a bucket C1 lists as `device_lacks` or
    // `near_equivalent`. C1's list is the one source of device-gap sign-offs.
    let gap_entry = |slot: &str, refusal: &Refusal| {
        let venue = venue_of_slot.get(slot)?;
        let product = refusal
            .product
            .as_ref()
            .filter(|_| refusal.class == RefusalClass::Device)?;
        c1.entries.iter().find(|entry| {
            entry.profile == venue.profile
                && entry.outcome == product.label
                && entry.key == product.key
                && matches!(
                    entry.class,
                    ListClass::DeviceLacks | ListClass::NearEquivalent
                )
        })
    };
    // The answer gap the list holds `case` under, where the device of the venue of `slot` has it
    // and, for a gap the case's structure decides, the case meets it. A case whose structure is
    // unknown meets no such gap.
    let answer_gap = |case: &AuthoredCase, slot: &str| {
        list.answer_gaps.iter().find(|entry| {
            entry.case_id == case.case_id
                && keys.has_gap(slot, entry.gap)
                && (!entry.gap.decided_by_case()
                    || facts
                        .answer_gaps(case)
                        .is_some_and(|gaps| gaps.contains(&entry.gap)))
        })
    };
    let verdict = |case: &AuthoredCase, backend: Backend, slot: &str| -> Option<Verdict> {
        let Some(row) = rows.get(&(case.case_id.as_str(), slot)) else {
            let refusal = refusals.get(&(case.case_id.as_str(), slot))?;
            if other_device(slot, &refusal.environment)
                || !crate::index::refusal_is_current(case, backend, refusal, keys)
            {
                return Some(Verdict::Stale);
            }
            // A product refusal out of the launch's scope, by its typed outcome.
            if let Some(scope) = refusal
                .refusal
                .product
                .as_ref()
                .and_then(|product| OutOfScope::of_recorded(&product.label, &product.key))
            {
                return Some(Verdict::OutOfScope(scope));
            }
            return Some(match gap_entry(slot, &refusal.refusal) {
                Some(_) => Verdict::DeviceGap,
                None => Verdict::Refused,
            });
        };
        if row_on_other_device(row) {
            return Some(Verdict::Stale);
        }
        let judging = judged_by(case);
        Some(
            match crate::index::candidate_slot_state(case, backend, judging.as_slice(), row, keys) {
                EvidenceState::Current => Verdict::Match,
                EvidenceState::Failed
                    if list.reviews(row).is_some() && deferred_note(row).is_none() =>
                {
                    Verdict::Reviewed
                }
                EvidenceState::Failed => Verdict::Mismatch,
                // A candidate slot is never a reference gap; the case's gap is judged below.
                EvidenceState::Stale | EvidenceState::Missing | EvidenceState::ReferenceGap => {
                    Verdict::Stale
                }
            },
        )
    };
    let candidates = |backend: Backend| match backend {
        Backend::Moltenvk => &input.moltenvk,
        Backend::Vulkan => &input.vulkan,
    };
    // The old engine's recorded rows in each venue's own environment: the baseline of a venue
    // that holds it.
    let baseline = |venue: &Venue| -> HashMap<&str, CandidateStatus> {
        candidates(venue.backend())
            .iter()
            .filter(|row| row.environment_id == venue.environment_id)
            .map(|row| (row.case_id.as_str(), row.status))
            .collect()
    };

    let mut slots = Vec::new();
    let mut gap_cases = BTreeMap::<(&str, &str, &str), usize>::new();
    let mut proposals = BTreeMap::<(&str, &str, &str), usize>::new();
    for venue in &venues.venues {
        let baseline = baseline(venue);
        let holds_baseline = venue.population == Population::OldEngineMatched;
        for leg in [Leg::Stated, Leg::RobustnessOff] {
            let slot = engine_slot(&venue.environment_id, None, leg);
            let listed = venue.legs.contains(&leg);
            let required = |case: &AuthoredCase| {
                listed
                    && match venue.population {
                        Population::OldEngineMatched => {
                            baseline.get(case.case_id.as_str()) == Some(&CandidateStatus::Match)
                        }
                        Population::Public => input.public_air.contains(&case.air_sha256),
                        Population::MetalQualified => metals.contains_key(case.case_id.as_str()),
                    }
            };
            let mut report = SlotReport {
                kind: venue.kind,
                venue: venue.environment_id.clone(),
                slot: slot.clone(),
                leg,
                population: 0,
                verdicts: BTreeMap::new(),
                rows: Vec::new(),
                matched: BTreeSet::new(),
                new_mismatches: 0,
                missing_without_metal: 0,
                stale_without_metal: 0,
                stale_device: 0,
            };
            for case in cases.values() {
                let key = (case.case_id.as_str(), slot.as_str());
                let required = required(case);
                // A case outside the population that the slot never recorded is still judged in a
                // leg the venue lists once a Metal row would judge its record: it reads missing, so
                // the venue's lane, which runs what C3 reads stale or missing, records it, and no
                // new case stays unrun because the population is narrower than the corpus. It does
                // not join the population: covering it is no baseline claim.
                let judgeable = listed && judged_by(case).is_some();
                report.population += usize::from(required);
                let judged = match verdict(case, venue.backend(), &slot) {
                    Some(verdict) => verdict,
                    None if required || judgeable => Verdict::Missing,
                    None => continue,
                };
                let judged = match judged {
                    Verdict::Missing | Verdict::Stale
                        if reference_gaps.contains_key(case.case_id.as_str()) =>
                    {
                        Verdict::ReferenceGap
                    }
                    judged => judged,
                };
                *report.verdicts.entry(judged).or_default() += 1;
                if judged == Verdict::Missing && judged_by(case).is_none() {
                    report.missing_without_metal += 1;
                }
                if judged == Verdict::Stale && rows.contains_key(&key) && judged_by(case).is_none()
                {
                    report.stale_without_metal += 1;
                }
                if judged == Verdict::Stale
                    && match rows.get(&key) {
                        Some(row) => row_on_other_device(row),
                        None => refusals
                            .get(&key)
                            .is_some_and(|refusal| other_device(&slot, &refusal.environment)),
                    }
                {
                    report.stale_device += 1;
                }
                let refusal = matches!(
                    judged,
                    Verdict::Refused | Verdict::DeviceGap | Verdict::OutOfScope(_)
                )
                .then(|| refusals.get(&key).map(|refusal| &refusal.refusal))
                .flatten();
                if let Some(refusal) = refusal {
                    match (judged, gap_entry(&slot, refusal), &refusal.product) {
                        (Verdict::DeviceGap, Some(entry), _) => {
                            *gap_cases
                                .entry((&entry.profile, &entry.outcome, &entry.key))
                                .or_default() += 1;
                        }
                        // An out-of-scope refusal is no device gap: nothing is proposed.
                        (Verdict::Refused, None, Some(product))
                            if refusal.class == RefusalClass::Device
                                && !c1.entries.iter().any(|entry| {
                                    entry.profile == venue.profile
                                        && entry.outcome == product.label
                                        && entry.key == product.key
                                }) =>
                        {
                            *proposals
                                .entry((&venue.profile, &product.label, &product.key))
                                .or_default() += 1;
                        }
                        _ => {}
                    }
                }
                if judged == Verdict::Mismatch
                    && holds_baseline
                    && baseline.get(case.case_id.as_str()) != Some(&CandidateStatus::Mismatch)
                {
                    report.new_mismatches += 1;
                }
                if judged == Verdict::Match {
                    report.matched.insert(case.case_id.clone());
                } else {
                    let reason = match judged {
                        Verdict::Mismatch => rows
                            .get(&(case.case_id.as_str(), slot.as_str()))
                            .and_then(|row| deferred_note(row))
                            .map(OpenReason::NearEquivalent)
                            .or_else(|| {
                                answer_gap(case, slot.as_str())
                                    .map(|entry| OpenReason::AnswerGap(entry.gap))
                            }),
                        Verdict::Missing | Verdict::Refused => rate_map_refusal(case),
                        _ => None,
                    };
                    report.rows.push(SlotRow {
                        case_id: case.case_id.clone(),
                        name: case.name.clone(),
                        verdict: judged,
                        refusal: refusal.cloned(),
                        reason,
                    });
                }
            }
            slots.push(report);
        }
    }

    let golden_bytes = |case_id: &str, sha: &str| {
        input
            .metals
            .iter()
            .find(|metal| metal.case_id == case_id && metal.metal_output_sha256 == sha)
            .and_then(|metal| decode(&metal.output_b64))
    };

    // The profile each venue's rows are compiled for, derived once per committed profile when an
    // Apple reading first needs it: the product's notes for an operation are that profile's.
    let mut venue_profiles: HashMap<String, Result<DeviceProfile, String>> = HashMap::new();
    let mut readings = Vec::new();
    let mut refused = Vec::new();
    for reading in &list.readings {
        let case = cases.get(reading.case_id.as_str());
        let mut open_words = Vec::new();
        for observation in &reading.observations {
            let row = rows.get(&(
                reading.case_id.as_str(),
                observation.environment_id.as_str(),
            ));
            let current = match (case, row) {
                (Some(case), Some(row)) => {
                    let judging = judged_by(case);
                    !row_on_other_device(row)
                        && matches!(
                            crate::index::candidate_slot_state(
                                case,
                                row.backend,
                                judging.as_slice(),
                                row,
                                keys,
                            ),
                            EvidenceState::Current | EvidenceState::Failed
                        )
                }
                _ => false,
            };
            let mut bound = None;
            let state = match (case, row, row.and_then(|row| deferred_note(row))) {
                (None, _, _) => ReadingState::UnknownCase,
                (Some(case), _, _) if case.name != reading.name => {
                    ReadingState::Inconsistent(format!("the case is named {:?}", case.name))
                }
                (Some(_), None, _) => ReadingState::Unobserved,
                // A row that ran a deferred near-equivalent stays open: the list may not review
                // it.
                (Some(_), Some(_), Some(note)) => ReadingState::Inconsistent(format!(
                    "the row ran the deferred near-equivalent {note:?}, which stays open and is \
                     never a reviewed reading"
                )),
                // A reading keyed by its cause holds while the slot's row shows the cause, at
                // whatever candidate and words.
                (Some(_), Some(row), None) if reading.cause.keyed_by_cause() => {
                    if reading.cause.shown_by(row) {
                        ReadingState::Holds
                    } else {
                        ReadingState::Changed
                    }
                }
                (Some(_), Some(row), None)
                    if row.spv_sha256 != observation.spv_sha256
                        || row.candidate_output_sha256 != observation.candidate_output_sha256
                        || row.golden_output_sha256 != observation.golden_output_sha256 =>
                {
                    ReadingState::Changed
                }
                (Some(case), Some(row), None) => {
                    let output = decode(&row.output_b64).unwrap_or_default();
                    let words = word_count(&output);
                    match golden_bytes(&row.case_id, &row.golden_output_sha256)
                        .and_then(|golden| Some((differing_words(&golden, &output)?, golden)))
                    {
                        Some((differing, golden))
                            if differing == observation.differing_words
                                && words == observation.words =>
                        {
                            let profile =
                                (reading.cause == Cause::AppleReading).then(|| match venue_of_slot
                                    .get(observation.environment_id.as_str())
                                {
                                    Some(venue) => venue_profiles
                                        .entry(venue.profile.clone())
                                        .or_insert_with(|| {
                                            VenueProfile::committed(&venue.profile).and_then(
                                                |committed| {
                                                    crate::acceptance::derived_profile(
                                                        committed.description,
                                                    )
                                                },
                                            )
                                        })
                                        .clone(),
                                    None => Err("no venue holds the slot".into()),
                                });
                            match profile {
                                Some(Ok(profile)) => {
                                    let recorded = row
                                        .census
                                        .iter()
                                        .flat_map(|census| &census.modules)
                                        .flat_map(|module| module.precision.iter().copied())
                                        .collect();
                                    let check = bound::check(
                                        &reading.lanes,
                                        &case.buffers,
                                        &recorded,
                                        &profile,
                                        &golden,
                                        &output,
                                    );
                                    open_words.extend(check.open(UNSTATED_WORDS).map(|word| {
                                        format!(
                                            "{} word {} ({}): {} {}",
                                            observation.environment_id,
                                            word.word,
                                            word.part.as_str(),
                                            word.verdict.label(),
                                            word.verdict.detail()
                                        )
                                    }));
                                    bound = Some(check);
                                    ReadingState::Holds
                                }
                                Some(Err(error)) => ReadingState::Inconsistent(format!(
                                    "the venue's profile, which the bound reads: {error}"
                                )),
                                None => ReadingState::Holds,
                            }
                        }
                        Some((differing, _)) => ReadingState::Inconsistent(format!(
                            "the row differs in {differing} of {words} words"
                        )),
                        None => ReadingState::Inconsistent(
                            "the store holds no Metal output of the row's length".into(),
                        ),
                    }
                }
            };
            readings.push(ReadingRow {
                case_id: reading.case_id.clone(),
                name: reading.name.clone(),
                environment_id: observation.environment_id.clone(),
                cause: reading.cause,
                current,
                state,
                bound,
            });
        }
        if reading.cause == Cause::AppleReading
            && reading.review.sign_off.is_signed_off()
            && !open_words.is_empty()
        {
            refused.push(RefusedSignOff {
                case_id: reading.case_id.clone(),
                name: reading.name.clone(),
                words: open_words,
            });
        }
    }

    let causes: HashMap<&str, Cause> = list
        .old_mismatches
        .iter()
        .map(|cause| (cause.case_id.as_str(), cause.cause))
        .collect();
    let mut ledger = Vec::new();
    for row in input.moltenvk.iter().chain(&input.vulkan) {
        if is_engine_slot(&row.environment_id) || row.status != CandidateStatus::Mismatch {
            continue;
        }
        let case = cases.get(row.case_id.as_str());
        let judging = metals.get(row.case_id.as_str()).copied();
        let output = decode(&row.output_b64).unwrap_or_default();
        // The row's venue: the venue of its environment, else the venue holding its backend's
        // baseline (a probe build's row of the same device).
        let venue = venues
            .by_environment(&row.environment_id)
            .filter(|venue| venue.backend() == row.backend)
            .or_else(|| venues.baseline(row.backend));
        let (engine, engine_recorded) = match venue {
            Some(venue) => {
                let new_slot = engine_slot(&venue.environment_id, None, Leg::Stated);
                (
                    case.and_then(|case| verdict(case, venue.backend(), &new_slot))
                        .unwrap_or(Verdict::Missing),
                    rows.get(&(row.case_id.as_str(), new_slot.as_str()))
                        .map(|row| row.status),
                )
            }
            None => (Verdict::Missing, None),
        };
        ledger.push(LedgerRow {
            case_id: row.case_id.clone(),
            name: case.map_or_else(String::new, |case| case.name.clone()),
            backend: row.backend,
            environment_id: row.environment_id.clone(),
            executor_abi: row.executor_abi.clone(),
            differing_words: golden_bytes(&row.case_id, &row.golden_output_sha256)
                .and_then(|golden| differing_words(&golden, &output)),
            words: word_count(&output),
            against_current_metal: judging
                .is_some_and(|metal| metal.metal_output_sha256 == row.golden_output_sha256),
            output_is_current_metal: judging
                .is_some_and(|metal| metal.metal_output_sha256 == row.candidate_output_sha256),
            engine,
            engine_recorded,
            cause: causes.get(row.case_id.as_str()).copied(),
        });
    }
    ledger.sort_by(|left, right| {
        (&left.name, &left.case_id, &left.environment_id).cmp(&(
            &right.name,
            &right.case_id,
            &right.environment_id,
        ))
    });
    let mismatched: BTreeSet<&str> = ledger.iter().map(|row| row.case_id.as_str()).collect();
    let stale_causes = list
        .old_mismatches
        .iter()
        .filter(|cause| !mismatched.contains(cause.case_id.as_str()))
        .map(|cause| cause.case_id.clone())
        .collect();
    let entry = |part, case_id: &String, name: &String, cause, review: &Review| EntryRow {
        part,
        case_id: case_id.clone(),
        name: name.clone(),
        cause,
        sign_off: review.sign_off.clone(),
    };
    let entries = list
        .readings
        .iter()
        .map(|reading| {
            entry(
                ListPart::Reading,
                &reading.case_id,
                &reading.name,
                reading.cause,
                &reading.review,
            )
        })
        .chain(list.old_mismatches.iter().map(|cause| {
            entry(
                ListPart::OldMismatch,
                &cause.case_id,
                &cause.name,
                cause.cause,
                &cause.review,
            )
        }))
        .collect();
    let device_gaps = gap_cases
        .into_iter()
        .filter_map(|((profile, outcome, key), cases)| {
            let entry = c1.entries.iter().find(|entry| {
                entry.profile == profile && entry.outcome == outcome && entry.key == key
            })?;
            Some(DeviceGapRow {
                profile: profile.into(),
                outcome: outcome.into(),
                key: key.into(),
                class: entry.class,
                sign_off: entry.review.sign_off.clone(),
                cases,
            })
        })
        .collect();
    let gap_proposals = proposals
        .into_iter()
        .map(|((profile, outcome, key), cases)| GapProposal {
            profile: profile.into(),
            outcome: outcome.into(),
            key: key.into(),
            cases,
        })
        .collect();
    let mut gap_slots = BTreeMap::<&str, usize>::new();
    let mut answer_slots = BTreeMap::<&str, usize>::new();
    for row in slots.iter().flat_map(|slot| &slot.rows) {
        if row.verdict == Verdict::ReferenceGap {
            *gap_slots.entry(row.case_id.as_str()).or_default() += 1;
        }
        if matches!(row.reason, Some(OpenReason::AnswerGap(_))) {
            *answer_slots.entry(row.case_id.as_str()).or_default() += 1;
        }
    }
    let answer_gaps = list
        .answer_gaps
        .iter()
        .map(|gap| AnswerGapRow {
            case_id: gap.case_id.clone(),
            name: gap.name.clone(),
            gap: gap.gap,
            slots: answer_slots.get(gap.case_id.as_str()).copied().unwrap_or(0),
        })
        .collect();
    let listed_gaps: BTreeMap<&str, &ReferenceGapEntry> = list
        .reference_gaps
        .iter()
        .map(|gap| (gap.case_id.as_str(), gap))
        .collect();
    let reference_gaps = reference_gaps
        .keys()
        .chain(listed_gaps.keys())
        .copied()
        .collect::<BTreeSet<&str>>()
        .into_iter()
        .map(|case_id| ReferenceGapRow {
            case_id: case_id.into(),
            name: cases.get(case_id).map_or_else(
                || listed_gaps[case_id].name.clone(),
                |case| case.name.clone(),
            ),
            recorded: reference_gaps.get(case_id).copied(),
            listed: listed_gaps
                .get(case_id)
                .map(|gap| (gap.apple_family, gap.review.sign_off.clone())),
            slots: gap_slots.get(case_id).copied().unwrap_or(0),
        })
        .collect();
    C3Report {
        slots,
        readings,
        ledger,
        stale_causes,
        entries,
        device_gaps,
        gap_proposals,
        refused,
        reference_gaps,
        answer_gaps,
        reference,
    }
}

#[cfg(test)]
mod tests;
mod venues;

pub use venues::{
    DeviceIdentity, Population, RecordedKey, Venue, VenueCheck, VenueDevice, VenueKeys, VenueKind,
    VenueProfile, VenueRekey, Venues, VENUES_FILE, VENUE_PROFILE,
};
