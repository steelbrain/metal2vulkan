//! A command queue's limit on uncompleted command buffers: the first case whose subject is a call
//! that **did not return**.
//!
//! Metal's `commandBuffer` blocks once a queue holds `maxCommandBufferCount` uncompleted command
//! buffers, and answers once one of them is dropped uncommitted or completes. The case records four
//! facts per queue ([`crate::queue_slots::QueueSlots`]): how many makes answered at once, that the
//! next one did not, that it answered once a slot came back, and that another queue's make went on
//! meanwhile.
//!
//! # What it measures
//!
//! The first row measures Metal's default count on the oracle: the default queue's probe makes
//! command buffers until one blocks, up to a cap above 64, so the number in the recording is the
//! number Metal answered. A release where Apple changes it fails the case.
//!
//! # Why it is safe
//!
//! Nothing is committed, so nothing reaches the GPU: the probe makes command buffers, holds them,
//! and drops them. Every make runs on a thread of its own and a blocked one is released by the
//! probe, so no wait here can hang ([`crate::queue_slots::probe`]).
//!
//! **A count of 0 is deliberately not a row.** Metal's own validation ends the process on
//! `maxCommandBufferCount:0`, so no case may ask Metal for it; the runtime refuses it typed
//! (`Refusal::NoCommandBuffers`).

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::transcript::{Node, Observed, Sites};

/// The case's name.
pub const QUEUE_SLOTS: &str = "command-queue-slots";

/// The queues the case probes: the `maxCommandBufferCount` it is made with, and how many makes the
/// probe tries before it gives up on finding a limit.
///
/// The default queue's cap is above Metal's measured 64 on purpose: the probe is **discovering** the
/// limit, not confirming one, so a release that raised it would be recorded rather than mistaken for
/// a block at 64. The stated counts need one make past their own count and no more.
const QUEUES: [(Option<u32>, u32); 4] = [(None, 70), (Some(1), 2), (Some(4), 5), (Some(65), 66)];

/// A case over the slot limits of a few command queues.
pub struct CommandQueueSlots {
    name: &'static str,
}

/// The case.
pub fn queue_slots() -> CommandQueueSlots {
    CommandQueueSlots { name: QUEUE_SLOTS }
}

/// The site one queue's row sits at.
fn row(max: Option<u32>) -> String {
    match max {
        None => "default-count".to_owned(),
        Some(max) => format!("max-{max}"),
    }
}

impl Case for CommandQueueSlots {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // What the case feeds: each queue's stated count and the probe's cap, in order, and the
        // window a make must stay unanswered to count as blocked. The window is an input because
        // it decides what the answer means.
        let mut words = Vec::new();
        for (max, cap) in QUEUES {
            words.extend(max.unwrap_or(u32::MAX).to_le_bytes());
            words.extend(cap.to_le_bytes());
        }
        Ok(vec![
            Input::bytes("queues", &words),
            Input::bytes(
                "blocked-window-millis",
                &(crate::queue_slots::BLOCKED.as_millis() as u64).to_le_bytes(),
            ),
        ])
    }

    fn record_runs(&self) -> u32 {
        // A probe that reads a timeout is reading a race, however wide the window: a make slowed by
        // the machine rather than by the queue's limit would read as blocked. More runs do not
        // remove that, but they record it — the recording holds the set of outcomes with their
        // frequencies, so a row that flaps says so instead of passing as whichever run came last.
        5
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for (max, cap) in QUEUES {
            let at = row(max);
            match side.queue_slots(max, cap) {
                Ok(slots) => sites.push(slots.node(at)),
                // A queue the side could not make, or a make that answered an error, is recorded at
                // the row and the sweep goes on: the other queues' answers stay readable, and a
                // refusal where Metal answered compares as a refusal at that one row.
                Err(failure) => sites.push(Node::leaf(at, Observed::value(failure.recorded()))),
            }
        }
        Ok(())
    }
}
