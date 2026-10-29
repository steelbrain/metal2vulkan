//! A wait made from a command buffer's own handler: the vocabulary for the one shape Metal does not
//! come back from.
//!
//! Metal runs a command buffer's scheduled and completed handlers on threads of its own, and a wait
//! made from one of them for **that same command buffer** is a wait for a transition the handler is
//! itself in the middle of. Metal does not detect it: the call never returns, and the probe that
//! found that out ended with it.
//!
//! So there is nothing to carry and nothing to run. What a case over this pins is the one fact there
//! is — that Metal never came back, and that we answer a typed error there instead — through
//! [`MetalSource::did_not_return`](crate::case::MetalSource::did_not_return) and
//! [`Verdict::MetalNeverReturned`](crate::verdict::Verdict::MetalNeverReturned).
//!
//! **Nothing here has a Metal side, by construction.** `metal_side::MetalSide` does not
//! implement [`Side::handler_wait`](crate::case::Side::handler_wait), and must not: the call it
//! would make is the call that ends Metal's process. Record mode refuses the row before anything is
//! encoded, and the recording is written from the probe that died.
//!
//! Our own leg is safe and bounded: the runtime refuses the wait at once, so the handler returns,
//! the command buffer completes, and the case's own thread waits for it with a bound.

use crate::transcript::{Negative, Node, NodeName, Observed, Value};

/// Which of a command buffer's handlers the wait is made from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandlerKind {
    /// `addScheduledHandler:`.
    Scheduled,
    /// `addCompletedHandler:`.
    Completed,
}

impl HandlerKind {
    /// The Metal call that adds it, which is how a site names it.
    pub fn call(self) -> &'static str {
        match self {
            HandlerKind::Scheduled => "addScheduledHandler",
            HandlerKind::Completed => "addCompletedHandler",
        }
    }
}

/// Which wait the handler makes for its own command buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitKind {
    /// `waitUntilCompleted`.
    UntilCompleted,
    /// `waitUntilScheduled`.
    UntilScheduled,
}

impl WaitKind {
    /// The Metal call, which is how a site names it.
    pub fn call(self) -> &'static str {
        match self {
            WaitKind::UntilCompleted => "waitUntilCompleted",
            WaitKind::UntilScheduled => "waitUntilScheduled",
        }
    }
}

/// One wait a case asks about: made from `from`, for the handler's own command buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnWait {
    pub from: HandlerKind,
    pub wait: WaitKind,
}

impl OwnWait {
    /// The wait `wait` made from `from`.
    pub const fn new(from: HandlerKind, wait: WaitKind) -> Self {
        Self { from, wait }
    }

    /// The site the wait's answer sits at: the two calls, so a reader of a diff sees which handler
    /// made which wait without counting rows.
    ///
    /// It is `'static` rather than built, because a recording's provenance names the call Metal did
    /// not return from and must hold the same words the site does.
    pub const fn site(&self) -> &'static str {
        match (self.from, self.wait) {
            (HandlerKind::Scheduled, WaitKind::UntilCompleted) => {
                "addScheduledHandler/waitUntilCompleted"
            }
            (HandlerKind::Scheduled, WaitKind::UntilScheduled) => {
                "addScheduledHandler/waitUntilScheduled"
            }
            (HandlerKind::Completed, WaitKind::UntilCompleted) => {
                "addCompletedHandler/waitUntilCompleted"
            }
            (HandlerKind::Completed, WaitKind::UntilScheduled) => {
                "addCompletedHandler/waitUntilScheduled"
            }
        }
    }

    /// The words the wait fingerprints as, for a case's input fingerprint.
    pub fn words(&self) -> Vec<u8> {
        format!("{}:{}", self.from.call(), self.wait.call()).into_bytes()
    }
}

/// The site Metal's answer is, for a wait Metal does not come back from: the negative fact, not a
/// value.
///
/// A case states this rather than leaving the site out, which is the whole point of the record kind:
/// "Metal never returned" and "nobody looked" must read differently, and a run of ours that
/// *answers* here is a divergence
/// ([`DivergenceKind::AnsweredWhereMetalDidNotReturn`](crate::diff::DivergenceKind::AnsweredWhereMetalDidNotReturn)),
/// not a pass.
pub fn never_returned(name: impl Into<NodeName>) -> Node {
    Node::leaf(
        name,
        Observed::value(Value::Nothing(Negative::NeverReturned)),
    )
}
