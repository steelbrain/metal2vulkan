//! Replay: read a case's recording, run our side, compare, and say what it was.
//!
//! Nothing here needs Metal. The recording is a file, the shaders are committed fixtures, and the
//! device is whatever Vulkan the host has, so the whole leg runs on a Linux host with lavapipe as
//! it runs on this Mac with MoltenVK.

use crate::allowance::{Justification, Offered};
use crate::case::{inputs_node, Case, MetalSource, Side, Venue, INPUTS_SITE};
use crate::compare::{compare_run, Unfired};
use crate::recording::{self, Answer, Lookup, Recording};
use crate::transcript::{Node, Sites, Transcript};
use crate::verdict::Verdict;

/// The reserved site replay emits the degrade notices our run announced at. Metal announces none,
/// so a recording holds an empty set here and any notice of ours must be justified by naming its
/// code: the rule the suite exists for, checked by execution.
pub const NOTICES_SITE: &str = "notices";

/// What replaying one case on one venue came to.
#[derive(Clone, Debug, PartialEq)]
pub struct Replayed {
    pub case: String,
    pub venue: Venue,
    pub verdict: Verdict,
    /// How many allowances the case named, and which of them nothing on this run needed
    /// ([`crate::compare::Unfired`]). Empty where no comparison happened at all — a venue that
    /// lacks the feature fires nothing, and saying its allowances went unused would be noise.
    pub allowances: Allowances,
    /// The chip a carried answer was measured on, where that is not the oracle's
    /// ([`crate::recording::Provenance::measured_elsewhere`]). The report counts these: an answer
    /// carried from a machine that is gone is weaker evidence than one taken here, and the suite's
    /// own rule is that a recording of another oracle is stale, so the one place that rule is
    /// relaxed has to be visible rather than implied.
    pub measured_on: Option<String>,
    /// What our side observed, for a reader who wants the whole transcript after a failure.
    pub ours: Transcript,
}

/// What a case's allowances did on one row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Allowances {
    /// How many the case named.
    pub named: usize,
    /// The ones no site of this run needed.
    pub unfired: Vec<Unfired>,
}

impl Allowances {
    /// How many fired.
    pub fn fired(&self) -> usize {
        self.named.saturating_sub(self.unfired.len())
    }
}

/// Why a case could not be replayed at all. These are never verdicts: a stale recording must fail
/// loudly rather than resolve to a pass or to a difference nobody can read.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayError {
    /// The recording file could not be read.
    NoRecording { case: String, why: String },
    /// The recording is not current evidence.
    Stale { case: String, why: String },
    /// The case could not state its own inputs or shaders.
    Case { case: String, why: String },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::NoRecording { case, why } => {
                write!(out, "{case}: no recording ({why})")
            }
            ReplayError::Stale { case, why } => write!(out, "{case}: a stale recording: {why}"),
            ReplayError::Case { case, why } => write!(out, "{case}: {why}"),
        }
    }
}

/// Run `case` on `side` and compare it with its recording.
pub fn replay(case: &dyn Case, side: &mut dyn Side) -> Result<Replayed, ReplayError> {
    let name = case.name();
    let recording = recording::read(name).map_err(|why| ReplayError::NoRecording {
        case: name.to_owned(),
        why,
    })?;
    replay_against(case, side, &recording)
}

/// [`replay`] against a recording already in hand.
///
/// # A case's verdict is a function of that case alone
///
/// **Everything this reads is the case's own**: its recording, its allowances, and the transcript
/// `side` just emitted. There is no parameter through which a selection, an order or a neighbour's
/// run can reach the comparison, and that is the point — a suite whose verdict moves with what
/// else ran can report a clean run on a broken implementation, which no amount of added depth
/// catches.
///
/// It did carry one such parameter, and it cost a verdict. The codes the device had announced
/// *earlier in the run* were added to what this case's tolerances could cite, on the reasoning
/// that a notice is a fact about the device for the whole run. Measured: a case with a value
/// tolerance justified by a notice, over a run that announced nothing itself and a word that
/// genuinely differed, read `Divergent` alone and `Announced` inside a selection where any earlier
/// case had provoked that code
/// (`tests::a_case_verdict_does_not_move_with_what_a_neighbour_announced`). A real difference was
/// excused by a neighbour. A notice now justifies a difference only in the case whose own work
/// announced it, which is what the declaration claims in the first place.
pub fn replay_against(
    case: &dyn Case,
    side: &mut dyn Side,
    recording: &Recording,
) -> Result<Replayed, ReplayError> {
    let name = case.name();
    let venue = side.venue();
    let inputs = case.inputs().map_err(|why| ReplayError::Case {
        case: name.to_owned(),
        why,
    })?;

    let mut sites = Sites::new();
    sites.push(inputs_node(&inputs));
    let abort = case.run(side, &mut sites).err();
    // The venue's own gap at the site that **stopped** the case, for a case that recorded the
    // site's value by some other route than [`crate::case::Failure::recorded`], which is where
    // the fact otherwise rides ([`crate::case::lacks_recorded`]).
    //
    // It is a second reading and not the one the comparison leans on: this one is keyed by the
    // path the abort names, and a case whose site is one node but whose rows are many names the
    // row there rather than the node, so the fact landed at a site that does not exist. Kept
    // because it costs nothing and is idempotent, dropped from the comparison's critical path
    // because a path a case states twice cannot be made to agree by a comment.
    if let Some(stopped) = &abort {
        if let Some(needs) = &stopped.failure.lacks {
            sites.device_gap(&stopped.at, needs.clone());
        }
    }
    let mut announced = side.announced();
    sites.notices(NOTICES_SITE, &announced);
    let ours = sites.finish();
    announced.extend(ours.announced_notices().into_keys());

    let answer = match recording.answer() {
        Ok(answer) => answer,
        Err(Lookup::Unrecorded { release }) => {
            return Ok(Replayed {
                case: name.to_owned(),
                venue,
                verdict: Verdict::Unrecorded {
                    why: format!("no Metal answer at macOS {release}"),
                },
                allowances: Allowances::default(),
                measured_on: None,
                ours,
            })
        }
        Err(stale @ Lookup::Stale(_)) => {
            return Err(ReplayError::Stale {
                case: name.to_owned(),
                why: stale.to_string(),
            })
        }
    };

    // A run the venue stopped is still its own outcome — never a pass, never a failure, never
    // counted as coverage — and that outcome is now the **comparison's** answer rather than a
    // short circuit ahead of it.
    //
    // Why it moved: returning `LacksFeature` straight off the abort skipped the comparison
    // entirely, so a refusal our side had already recorded at a site the run **did** reach
    // vanished behind a *compliant* verdict — the worst form of a refusal going missing, since
    // the run stayed green. `verdict.rs` already states the invariant this broke ("a divergence
    // or a refusal anywhere else in the same case still fails, which is what makes this safe to
    // be compliant"). It also hid a moved input fingerprint the same way, because
    // [`inputs_differ`] is checked inside [`verdict_over`].
    //
    // Letting [`Verdict::of`] decide holds the invariant by construction: the site that stopped
    // the case records the venue's gap (above), the sites the run never reached are not
    // divergences because the comparison is told the run stopped, and the device gaps are read
    // **after** the divergences and the refusals, so anything worse outranks them.
    let (verdict, allowances) = verdict_over(case, answer, &ours, &announced, abort.is_some());
    Ok(Replayed {
        case: name.to_owned(),
        venue,
        // A venue that lacks the feature compared almost nothing, so the allowances its case
        // named went unused for a reason that is not the suite's blind spot. Saying they never
        // fired would be noise, which is what [`Replayed::allowances`] is empty for.
        allowances: match &verdict {
            Verdict::LacksFeature { .. } => Allowances::default(),
            _ => allowances,
        },
        verdict,
        measured_on: answer.measured_elsewhere().map(str::to_owned),
        ours,
    })
}

/// The verdict of `ours` against `answer`'s outcomes: the most favourable of them, because Metal
/// produced any of the set and matching one of them is matching Metal. With it, what the case's
/// allowances did against that outcome.
///
/// The allowances follow the verdict rather than being gathered over every outcome, because they
/// are a fact about the comparison the verdict came from: a racy case whose run matched the
/// outcome that needed no tolerance did not need the tolerance, however the other outcomes read.
fn verdict_over(
    case: &dyn Case,
    answer: &Answer,
    ours: &Transcript,
    announced: &[String],
    stopped: bool,
) -> (Verdict, Allowances) {
    let allowances = case.allowances();
    let carried = match case.metal_source() {
        MetalSource::Carried { from, .. } => Some(from),
        MetalSource::Run | MetalSource::DidNotReturn { .. } => None,
    };
    let mut best: Option<(Verdict, Allowances)> = None;
    for outcome in &answer.outcomes {
        // The inputs come first: an answer taken from other bytes is no answer for this run.
        if let Some(why) = inputs_differ(&outcome.transcript, ours) {
            best = Some(better(
                best,
                (
                    Verdict::Unrecorded {
                        why: match carried {
                            Some(from) => format!("{why}; the answer is carried from {from}"),
                            None => why,
                        },
                    },
                    Allowances::default(),
                ),
            ));
            continue;
        }
        let offered = Offered::new(announced, &answer.validated);
        let found = compare_run(&outcome.transcript, ours, &allowances, offered, stopped);
        let used = Allowances {
            named: found.named,
            unfired: found.unfired.clone(),
        };
        best = Some(better(best, (Verdict::of(&found, &answer.validated), used)));
    }
    best.unwrap_or_else(|| {
        (
            Verdict::Unrecorded {
                why: "the recording's answer holds no outcome".to_owned(),
            },
            Allowances::default(),
        )
    })
}

/// Why the inputs our run fed are not the ones `metal`'s answer was taken from, if they are not.
fn inputs_differ(metal: &Transcript, ours: &Transcript) -> Option<String> {
    let recorded = metal.root.child(INPUTS_SITE);
    let fed = ours.root.child(INPUTS_SITE);
    match (recorded, fed) {
        (Some(recorded), Some(fed)) if recorded == fed => None,
        (None, _) => Some("the recording states no input fingerprint".to_owned()),
        (Some(recorded), Some(fed)) => Some(format!(
            "the inputs this run fed are not the recording's: {}",
            input_diff(recorded, fed)
        )),
        (Some(_), None) => Some("this run stated no input fingerprint".to_owned()),
    }
}

/// Which inputs moved between two input fingerprints, in words.
fn input_diff(recorded: &Node, fed: &Node) -> String {
    let mut moved = Vec::new();
    for input in &recorded.children {
        match fed.child(&input.name) {
            Some(ours) if ours.observed == input.observed => {}
            Some(_) => moved.push(format!("{} differs", input.name)),
            None => moved.push(format!("{} is missing", input.name)),
        }
    }
    for input in &fed.children {
        if recorded.child(&input.name).is_none() {
            moved.push(format!("{} is extra", input.name));
        }
    }
    moved.join(", ")
}

/// The more favourable of two verdicts, with the allowances of whichever is kept.
fn better(
    held: Option<(Verdict, Allowances)>,
    found: (Verdict, Allowances),
) -> (Verdict, Allowances) {
    match held {
        Some(held) if rank(&held.0) <= rank(&found.0) => held,
        _ => found,
    }
}

/// How favourable a verdict is, lowest first.
fn rank(verdict: &Verdict) -> u8 {
    match verdict {
        Verdict::Exact => 0,
        Verdict::Announced { .. } => 1,
        Verdict::MetalNeverReturned { .. } => 2,
        Verdict::LacksFeature { .. } => 3,
        Verdict::Refused { .. } => 4,
        Verdict::Divergent { .. } => 5,
        Verdict::Unrecorded { .. } => 6,
    }
}

/// The allowance a case needs to announce `code`, for a case whose own degrade notice is the
/// difference: the notice site, loosened by the code it names.
///
/// Every notice our side announces must be named by the case, or the case is divergent at the
/// notices site ([`crate::diff::DivergenceKind::UndeclaredNotice`]). **A value tolerance justified
/// by a notice names it too, and that is deliberate** — it is not obvious, so it is written down
/// here. An `Allowance` at a value site justified by `Degradation(code)` already says "this value
/// differs because the device announced this degradation", which is a statement of the notice by
/// any honest reading; asking its author for a second allowance here would be stating one fact
/// twice, and two statements of one rule drift apart.
///
/// The failure direction is what settles it. Under this rule, when the value becomes exact and its
/// tolerance is deleted, the declaration goes with it and the case reports the notice as undeclared
/// at once — loudly, at the moment the answer may have changed. A bookkeeping declaration here
/// would instead outlive its own reason and keep asserting that somebody examined a notice nobody
/// has looked at since, which is the defect this check exists to close, one level up.
pub fn notice_allowance(code: &str) -> crate::allowance::Allowance {
    crate::allowance::Allowance::new(
        format!("case/{NOTICES_SITE}/*"),
        crate::allowance::Loosening::Unchecked,
        Justification::Degradation(code.to_owned()),
    )
}

/// The declaration a case needs for the one notice a **venue's memory model** provokes and no
/// recording of this suite could have declared: `shared-in-system-memory`, which the runtime
/// announces for a `Shared` buffer or heap on a device whose GPU does not share the CPU's memory
/// ([`metal2vulkan_runtime::Degradation::SharedInSystemMemory`]).
///
/// **Why it is the venue's and not ours.** Every recording here was taken on an Apple chip, where
/// the GPU and the CPU hold one memory, so a `Shared` resource is in device memory by construction
/// and the notice cannot fire at all — Metal recorded nothing for a case to compare it against,
/// and MoltenVK on the same chip announces nothing either. A discrete card has no such memory: the
/// same resource lies in host-cached system memory and the GPU reads it across the bus, which is
/// exactly what the notice says. So the difference belongs to the device, the notice is the
/// runtime reporting it honestly, and a case that reads its answer back through a `Shared` buffer
/// has to name it or read `Divergent` on every such card for a reason that is not a wrong answer.
///
/// **It is a declaration and never an excuse.** Like every [`notice_allowance`] it loosens the
/// notices site alone, so a case that names it holds every one of its values to Metal's exactly.
/// A case whose values also differ on such a device is a separate finding and keeps its strict
/// value comparison.
///
/// **It is still held to its evidence.** A case names it where a run on a device without unified
/// memory announced it; where no run has, the suite counts the claim as one nothing checked
/// ([`crate::report::Report::unchecked_everywhere`]), and the case that names it says in a comment
/// which run it rests on.
pub fn shared_memory_notice() -> crate::allowance::Allowance {
    notice_allowance(metal2vulkan_runtime::Degradation::SharedInSystemMemory.code())
}

/// The declaration a case needs where its kernel reaches a texture or a sampler through an
/// **argument buffer**: `set-past-layout-support`, which the runtime announces where it makes a
/// descriptor set layout its driver's `vkGetDescriptorSetLayoutSupport` refuses though the driver
/// runs it ([`metal2vulkan_runtime::Degradation::SetPastLayoutSupport`]).
///
/// **Why it is the venue's and the case's shape.** Such a kernel takes the handle heap, set 1,
/// whose arrays hold the device's update-after-bind capacity (65536 descriptors each on the M3
/// Max). MoltenVK answers the layout query by the sum of the bindings' counts against
/// `maxPerSetDescriptors` (1212 there), ignoring the update-after-bind limits it reports, so it
/// refuses every such set though it makes and runs it; Vulkan's validation layer reports each one
/// (`VUID-vkCreateDescriptorSetLayout-support-09582`), and the notice says so. lavapipe supports
/// the same layouts and announces nothing; Metal recorded nothing. The query is not ours to answer
/// otherwise: it is asked of the create info the layout is made from.
///
/// **It is a declaration and never an excuse**, like [`shared_memory_notice`]: it loosens the
/// notices site alone, so every value of a case that names it still compares exactly. A case names
/// it from its own shape (a texture or sampler in an argument buffer that a dispatch reads), never
/// from the venue, and says in a comment which run it rests on.
pub fn set_past_layout_notice() -> crate::allowance::Allowance {
    notice_allowance(metal2vulkan_runtime::Degradation::SetPastLayoutSupport.code())
}

/// The notices that state **the device's own answer** to an arithmetic or value question, which the
/// suite holds declared at the notices site for every case ([`native_note_declarations`]).
///
/// **Why the suite declares them and no case does.** The owner's rule (AGENTS.md, rule 1) makes the
/// GPU's native arithmetic the contract: where the device has a native form of an operation, that
/// form is the only one the runtime builds, and the artifact records a note where it can differ
/// from Metal's. So such a notice says what the device is, not what a case did wrong, and it fires
/// in almost every case that does arithmetic, converts a texel or samples. A case naming each one
/// would state one rule of the suite's a thousand times, and two statements of one rule drift
/// apart.
///
/// **Which notes, and from where.** A kind is here where its documentation (the `PrecisionNote`'s
/// in the translator, or the runtime's own `Degradation`'s) says the answer is the device's own
/// native form of an operation rule 1 names, and that no other form is built:
///
/// - **maths**: the native-maths class (`native-transcendentals`, `native-division`,
///   `native-fma`, `native-dot`, `native-mix`, `native-min-max`, `native-nan-encoding`), the
///   device's functions and precision bounds (`device-transcendentals`, `device-arithmetic`), the
///   device's denormals and NaN bits (`device-denormals`, `device-nan-encoding`), the device's
///   rounding of a tiny product (`float-tininess-after-rounding`), and the device's compiler's
///   grouping and fusion of the fast-math arithmetic the AIR's flags let it rewrite
///   (`fast-math-rewrites`: the module states each operation in the AIR's order and reproduces no
///   grouping of Metal's);
/// - **rounding**: the native-half class (`native-half-conversion`, `native-half-quantize`), and
///   the device's own depth conversion and depth test (`unorm-depth-rounding`, `depth-denormals`,
///   `signed-depth-test`);
/// - **texel, vertex and colour conversions**: `device-texel-conversion`,
///   `half-texel-conversion`, `srgb-texel-decode`, the device's decode of a BC texture's blocks
///   (`compressed-texel-decode`), the device's fixed-function conversion of a
///   normalized vertex attribute (`device-vertex-conversion`), `device-color-conversion`;
/// - **sampling and filtering**: `device-sampling`, `device-sample-level`, the device's filter of a
///   generated mipmap chain (`device-filtered-mipmaps`), and Metal's own sampler filtering a
///   two-plane texture on MoltenVK (`half-sample-filter`, `float-sample-filter`,
///   `ycbcr-filter-not-exact`);
/// - **blending**: the device's arithmetic of the blend a function computes into an extended-range
///   attachment, whose raw codes no blend unit reads (`extended-range-blend-source`,
///   `extended-range-float-blend-source`: `S·Fs op D·Fd` in `float`, where Metal's compiler rounds
///   each step to `half`);
/// - **resolves**: `device-rounded-resolve`;
/// - **coverage**: the device's own conversion of a fragment's alpha to its coverage under
///   alpha-to-coverage (`device-alpha-coverage`).
///
/// **What is not here**, and stays each case's to declare: a note of what the device **lacks**
/// (`device-signed-zero-inf-nan`, `device-anisotropic-filter`, `device-point-size`,
/// `restricted-depth-range`), of an emulation's own precision (`decoded-texel-precision`,
/// `grad-from-bias`, `pixel-sampler-scaling`, `bicubic-filter-arithmetic`), of a Metal form not
/// measured or not built (`blended-write-rounding`, `unmeasured-resolve`, `device-interpolation`,
/// `reduction-order`), and every performance or validity notice.
///
/// **It declares the notice, and nothing about a value.** Each entry loosens the notices site
/// alone, exactly as a case's own [`notice_allowance`] does, so every value of every case still
/// compares field by field. A value the device's answer moves is still a divergence at its own
/// site: where the note says validation derives no tolerance from it, as nearly every one of these
/// does, the suite loosens no value by it either, and the site is a `known-gaps.tsv` row that says
/// what differs and whose answer it is.
pub const NATIVE_NOTES: &[metal2vulkan_runtime::Degradation] = {
    use metal2vulkan_runtime::Degradation as D;
    &[
        D::NativeTranscendentals,
        D::NativeDivision,
        D::NativeFma,
        D::NativeDot,
        D::NativeMix,
        D::DevicePackConversion,
        D::NativeMinMax,
        D::NativeNanEncoding,
        D::DeviceTranscendentals,
        D::DeviceArithmetic,
        D::DeviceDenormals,
        D::DeviceNanEncoding,
        D::FloatTininessAfterRounding,
        D::FastMathRewrites,
        D::NativeHalfConversion,
        D::NativeHalfQuantize,
        D::UnormDepthRounding,
        D::DepthDenormals,
        D::SignedDepthTest,
        D::DeviceTexelConversion,
        D::HalfTexelConversion,
        D::SrgbTexelDecode,
        D::CompressedTexelDecode,
        D::DeviceVertexConversion,
        D::DeviceColorConversion,
        D::DeviceSampling,
        D::DeviceSampleLevel,
        D::DeviceFilteredMipmaps,
        D::HalfSampleFilter,
        D::FloatSampleFilter,
        D::YCbCrFilterNotExact,
        D::ExtendedRangeBlendSource,
        D::ExtendedRangeFloatBlendSource,
        D::DeviceRoundedResolve,
        D::DeviceAlphaCoverage,
    ]
};

/// The codes of `announced` the cost and times halves' degraded-set rule counts, sorted and once
/// each: every one but the device's native answers ([`NATIVE_NOTES`]).
///
/// **What the rule is for.** A notice can say the device lacked the exact form and the runtime did
/// something cheaper, so a count or a time taken with it is of a different amount of work, and
/// comparing it with one taken without it would read the device's gap as a saving
/// ([`crate::cost::judge`], [`crate::times::classify`]). A native note never says that: the
/// device's native form of the operation is the only form the runtime builds, under either policy,
/// so the work is the same whether the note fired or not. Counting one withheld a verdict from
/// nearly every case that computes, converts or samples, and made a record taken on a run that did
/// not provoke a note incomparable with the same work on one that did.
pub fn degrading(announced: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut codes: Vec<String> = announced
        .into_iter()
        .filter(|code| !NATIVE_NOTES.iter().any(|kind| kind.code() == code))
        .collect();
    codes.sort();
    codes.dedup();
    codes
}

/// The suite's standing declarations: one [`notice_allowance`] per [`NATIVE_NOTES`] kind, which
/// [`crate::compare::compare_run`] holds beside every case's own allowances.
///
/// They are checked by the same rules as a case's declaration — each covers the notices site
/// alone, and justifies a difference there only where its own code was announced — but they are
/// not the case's: they never count among the allowances a case named, so a native note a case's
/// run did not announce is not reported as an allowance that never fired.
pub fn native_note_declarations() -> impl Iterator<Item = crate::allowance::Allowance> {
    NATIVE_NOTES
        .iter()
        .map(|kind| notice_allowance(kind.code()))
}
