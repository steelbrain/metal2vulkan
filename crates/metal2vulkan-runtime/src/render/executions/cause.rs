//! Why a pass whose executions' records the GPU may write takes the host step rather than the
//! GPU form (`gpu.rs`): the first rule that sent it there, which its notice names
//! (`Degradation::HostReplayedIndirectCommands`, [`DeferredPass::degrade`](super::DeferredPass::degrade)),
//! and, where the notice is announced, the rules past it that the pass breaks too
//! ([`Passed`], `DeferredPass::rules_past`): each judged as the decision would judge it were the
//! rules named before it kept, the rate map's as though the pass had none, so a walk sees every
//! rule that keeps the pass from the GPU form, not only the first. A rule the rest of the decision
//! takes its answer from (the commands' one primitive type, the encoder's pipeline state and its
//! kind, the buffer's size, a refusal) is never passed over: the rules past it stay unjudged.

use std::fmt;

use crate::degradation::HostStepKey;
use metal2vulkan::bind::DrawReplayRule;

/// The rule that sent a pass to the host step (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HostStepCause {
    /// A rule of the call, the buffer, the pass or the state, as the notice words it.
    Rule(&'static str),
    /// The caller's check looks for what the records may hold: draws of points (`points`), and
    /// commands under a state that may encode indirect commands (`encodes`), the needs the
    /// buffer's records did not rule out.
    Check { points: bool, encodes: bool },
    /// A rule of the buffer, its records, the inherited state or its draw that every draw replay
    /// keeps (`metal2vulkan::bind::draw_replay_primitive_type` and its neighbours).
    Replay(DrawReplayRule),
    /// The inherited state's replay variant, or the call's check of it, was refused so.
    Refused(String),
    /// What the records may hold grew after the commit's decision, before the pass's batch was
    /// ready (`DeferredPass::outgrown`).
    Outgrown,
}

impl HostStepCause {
    /// The cause its notice is announced once for (`Device::degrade_host_step`): the rule's words,
    /// the draw replay rule's variant, or the class.
    pub(crate) fn key(&self) -> HostStepKey {
        match self {
            HostStepCause::Rule(rule) => HostStepKey::Rule(rule),
            HostStepCause::Check { .. } => HostStepKey::Check,
            HostStepCause::Replay(rule) => HostStepKey::Replay(std::mem::discriminant(rule)),
            HostStepCause::Refused(_) => HostStepKey::Refused,
            HostStepCause::Outgrown => HostStepKey::Outgrown,
        }
    }
}

/// The rules a decision of the GPU form passes over (the module docs): those named already, by
/// their notice's key ([`HostStepCause::key`]).
#[derive(Default)]
pub(crate) struct Passed(Vec<HostStepKey>);

impl Passed {
    /// `Err(cause)` where the pass `breaks` the rule and the decision does not pass over it.
    pub(crate) fn rule(&self, breaks: bool, cause: HostStepCause) -> Result<(), HostStepCause> {
        match breaks && !self.over(&cause) {
            true => Err(cause),
            false => Ok(()),
        }
    }

    /// Whether the decision passes over `cause`'s rule.
    pub(crate) fn over(&self, cause: &HostStepCause) -> bool {
        self.0.contains(&cause.key())
    }

    /// Pass over `cause`'s rule from now on.
    pub(crate) fn pass(&mut self, cause: &HostStepCause) {
        self.0.push(cause.key());
    }
}

/// The words a notice names `first` and the rules past it, `also`, in (the module docs).
pub(crate) fn named(first: &HostStepCause, also: &[HostStepCause]) -> String {
    match also {
        [] => first.to_string(),
        also => {
            let past: Vec<String> = also.iter().map(HostStepCause::to_string).collect();
            format!(
                "{first}; past that rule, each judged as though the rules before it were kept, it \
                 also breaks: {}",
                past.join("; ")
            )
        }
    }
}

impl fmt::Display for HostStepCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostStepCause::Rule(rule) => f.write_str(rule),
            HostStepCause::Check { points, encodes } => {
                let needs = match (points, encodes) {
                    (true, true) => {
                        "draws of points and commands under a state that may encode \
                                     indirect commands"
                    }
                    (true, false) => "draws of points",
                    _ => "commands under a state that may encode indirect commands",
                };
                write!(
                    f,
                    "the execution's check looks for {needs}, which the buffer's records may hold"
                )
            }
            HostStepCause::Replay(rule) => write!(f, "{rule}"),
            HostStepCause::Refused(error) => {
                write!(f, "the inherited state's replay was refused: {error}")
            }
            HostStepCause::Outgrown => f.write_str(
                "what the buffer's records may hold grew after the commit, before the pass's \
                 batch was ready (a CPU setter, or a route encoded since)",
            ),
        }
    }
}
