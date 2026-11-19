//! What a fence orders between two queues, and what Metal's hazard tracking orders between them
//! with no fence: the vocabulary of [`crate::cases::fence_orders`].
//!
//! In every shape one queue's command buffer, the **updater**, writes a buffer X, and a command
//! buffer of another queue committed after it, the **waiter**, copies X into a buffer Y. The
//! updater's blit encoder fills X with [`WRITTEN`] and then updates a fence; the waiter's blit
//! encoder waits on the fence and then copies. What each shape reads is how many of Y's words hold
//! the updater's word once the waiter completed ([`FenceOrder::copied_words`]), the statuses, and
//! the facts of its own shape. As in [`crate::event_orders`], each shape is **one** [`Side`] call
//! that runs the whole experiment and answers what it saw: the facts are orders between two command
//! buffers, which no sequence of separate calls could state.
//!
//! The shapes come from the probe `probes/cross_queue_fence_held_update.m`, which measured them on
//! an Apple M3 Max with compute dispatches; these state them with **blit** encoders, fills and a
//! copy, so no shader is compiled and no loop runs on the GPU. That is a deliberate change of
//! encoder kind, and the recordings are what Metal answered for this form.
//!
//! [`Side`]: crate::case::Side
//!
//! # Why none of this can hang
//!
//! Every wait of the probe's own thread is bounded ([`crate::event_orders::COMPLETION`],
//! [`HELD_BOUND`]), and the one command buffer an event holds is held by a shared event the probe
//! itself raises, once [`HELD_BOUND`] has run out or the waiter completed.

use crate::event_orders::Status;
use crate::transcript::{Node, NodeName, Observed, Value};
use std::time::Duration;

/// The words X and Y each hold, four bytes a word.
pub const WORDS: u64 = 1024;

/// The bytes X and Y each hold.
pub const BYTES: u64 = WORDS * 4;

/// The byte the updater fills X with, so a word of Y holding four of it is the updater's word.
/// Neither zero, which a new buffer holds, nor anything else the probe writes.
pub const WRITTEN: u8 = 0x5a;

/// The bytes of the private buffer a long job fills, [`LONG_FILLS`] times over.
pub const LONG_FILL_BYTES: u64 = 64 << 20;

/// How many fills of [`LONG_FILL_BYTES`] a long job runs: fills alone, no loop on the GPU, so the
/// updater (or a busy command buffer ahead of it) is still running when the waiter's queue could
/// start it.
pub const LONG_FILLS: u32 = 32;

/// How long [`FenceShape::HeldUpdater`] gives the waiter to complete before the probe raises the
/// event that holds the updater. Its answer is whether the waiter completed within it.
pub const HELD_BOUND: Duration = Duration::from_secs(2);

/// Which experiment a case runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceShape {
    /// The updater runs a long job, fills X and updates the fence; the waiter waits on the fence
    /// and copies X into Y. X and Y are untracked, so only the fence orders the two.
    Ordered,
    /// As [`FenceShape::Ordered`] with no long job, but the updater's command buffer first waits
    /// for a shared event the probe raises only after [`HELD_BOUND`] or the waiter's completion.
    /// X and Y are untracked.
    HeldUpdater,
    /// The updater as in [`FenceShape::Ordered`]; the waiter's encoder only waits on the fence and
    /// copies nothing. Its completed handler counts X's words then
    /// ([`FenceOrder::source_words_when_the_waiter_completed`]). X is untracked.
    EmptyWaiter,
    /// No fence. `busy` command buffers, each a long job, then the updater, which fills X, are
    /// committed on one queue; the waiter on another copies X into Y. X and Y are tracked, so the
    /// order is Metal's hazard tracking's.
    Tracked { busy: usize },
}

impl FenceShape {
    /// Whether X and Y are made untracked (`MTLResourceHazardTrackingModeUntracked`), which is
    /// also whether the updater updates a fence the waiter waits on.
    pub fn untracked(self) -> bool {
        !matches!(self, FenceShape::Tracked { .. })
    }

    /// The command buffers, each a long job, committed on the updater's queue ahead of it.
    pub fn busy(self) -> usize {
        match self {
            FenceShape::Tracked { busy } => busy,
            _ => 0,
        }
    }

    /// The bytes a case's inputs hold for the shape: its name and what it is made of.
    pub fn words(self) -> Vec<u8> {
        let mut words = format!("{self:?}").into_bytes();
        words.push(u8::from(self.untracked()));
        words
    }
}

/// What one fence experiment observed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FenceOrder {
    /// [`FenceShape::HeldUpdater`] alone: whether the waiter completed within [`HELD_BOUND`],
    /// while the updater was still held.
    pub waiter_completed_before_the_release: Option<bool>,
    /// [`FenceShape::HeldUpdater`] alone: the updater's status when the bound ran out or the
    /// waiter completed, before the release.
    pub updater_before_the_release: Option<Status>,
    /// [`FenceShape::EmptyWaiter`] alone: how many of X's words held the updater's word when the
    /// waiter's completed handler ran; `None` where the handler never came.
    pub source_words_when_the_waiter_completed: Option<u64>,
    /// How many of Y's words held the updater's word once the waiter completed; `None` for
    /// [`FenceShape::EmptyWaiter`], which copies nothing.
    pub copied_words: Option<u64>,
    /// Both statuses once both completed.
    pub updater_at_the_end: Status,
    pub waiter_at_the_end: Status,
}

impl FenceOrder {
    /// The observations as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let mut node = Node::branch(name, Vec::new());
        if let Some(completed) = self.waiter_completed_before_the_release {
            node = node.with(Node::leaf(
                "waiter/completed-before-the-release",
                Observed::value(Value::Bool(completed)),
            ));
        }
        if let Some(status) = &self.updater_before_the_release {
            node = node.with(Node::leaf(
                "updater/status-before-the-release",
                Observed::value(status.value()),
            ));
        }
        if let Some(words) = self.source_words_when_the_waiter_completed {
            node = node.with(Node::leaf(
                "x/words-written-when-the-waiter-completed",
                Observed::value(Value::Uint(words)),
            ));
        }
        if let Some(words) = self.copied_words {
            node = node.with(Node::leaf(
                "y/words-written",
                Observed::value(Value::Uint(words)),
            ));
        }
        node.with(Node::leaf(
            "updater/status-once-both-completed",
            Observed::value(self.updater_at_the_end.value()),
        ))
        .with(Node::leaf(
            "waiter/status-once-both-completed",
            Observed::value(self.waiter_at_the_end.value()),
        ))
    }
}

/// How many of the words in `bytes` hold the updater's word, four [`WRITTEN`] bytes.
pub fn written_words(bytes: &[u8]) -> u64 {
    bytes
        .chunks_exact(4)
        .filter(|word| word.iter().all(|byte| *byte == WRITTEN))
        .count() as u64
}
