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
use metal2vulkan_runtime::MetalLevel;

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
    /// The venue answers at a Metal level no release of this runtime names.
    UnknownMetalLevel { case: String, level: String },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::NoRecording { case, why } => {
                write!(out, "{case}: no recording ({why})")
            }
            ReplayError::Stale { case, why } => write!(out, "{case}: a stale recording: {why}"),
            ReplayError::Case { case, why } => write!(out, "{case}: {why}"),
            ReplayError::UnknownMetalLevel { case, level } => {
                write!(
                    out,
                    "{case}: the venue answers at an unknown Metal level {level}"
                )
            }
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
    let level = MetalLevel::ALL
        .into_iter()
        .find(|level| level.macos() == venue.metal_level)
        .ok_or_else(|| ReplayError::UnknownMetalLevel {
            case: name.to_owned(),
            level: venue.metal_level.clone(),
        })?;
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

    let answer = match recording.answer(level) {
        Ok(answer) => answer,
        Err(Lookup::Unrecorded { level }) => {
            return Ok(Replayed {
                case: name.to_owned(),
                venue,
                verdict: Verdict::Unrecorded {
                    why: format!("no Metal answer at Metal level {level}"),
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
