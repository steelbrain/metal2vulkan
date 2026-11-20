//! What a fence orders between two queues, measured on Metal, and what Metal's hazard tracking
//! orders between them with no fence ([`crate::fence_orders`] holds the vocabulary).
//!
//! Six cases, each one experiment of an **updater** on one queue, which fills a buffer X, and a
//! **waiter** on another, committed after it, which copies X into a buffer Y:
//!
//! - [`ORDERED`]: X and Y untracked, so only the fence orders the two. The updater runs a long job,
//!   fills X and updates the fence; the waiter waits on it and copies. Y holds the updater's words
//!   on Metal.
//! - [`HELD_UPDATER`]: X and Y untracked, and the updater's command buffer first waits for a shared
//!   event the probe raises later. Metal does not wait for the held update: the waiter completes
//!   before the release, and Y holds the words it held before.
//! - [`EMPTY_WAITER`]: the waiter's encoder only waits on the fence. It completes after the
//!   updater: X holds the updater's words when its completed handler runs.
//! - [`TRACKED`], [`TRACKED_BEHIND_ONE`], [`TRACKED_BEHIND_BUSY`]: no fence, X and Y tracked, and
//!   the updater committed behind none, one or three busy command buffers of its queue: what
//!   Metal's hazard tracking orders between two queues by itself. Behind one or three, the M3 Max
//!   ran the waiter without the updater's words: its tracking orders only against work already
//!   submitted, as measured there, not documented. Behind one, the runtime submits the updater
//!   at once, as its queue has fewer than three submissions in flight, and the waiter copies every
//!   word; [`TRACKED_BEHIND_ONE`] declares that count on its run (docs/API.md section 10).
//!
//! The first three are the answers the probe `probes/cross_queue_fence_held_update.m` measured
//! with compute dispatches on an Apple M3 Max under macOS 27.0.1; the recordings hold what Metal
//! answers for this blit form.
//!
//! Every wait is bounded and the one command buffer held is held by an event the probe raises, so
//! nothing here can hang; nothing renders, nothing dispatches and no loop runs on the GPU.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::fence_orders::{
    FenceShape, BYTES, HELD_BOUND, LONG_FILLS, LONG_FILL_BYTES, WORDS, WRITTEN,
};
use crate::transcript::{Node, Observed, Sites, Value};

/// A fence update on one queue, waited on by another's command buffer committed after it.
pub const ORDERED: &str = "fence-across-queues";

/// A fence update in a command buffer an event holds, waited on by another queue's.
pub const HELD_UPDATER: &str = "fence-held-updater";

/// A waiter whose encoder only waits on the fence.
pub const EMPTY_WAITER: &str = "fence-empty-waiter";

/// No fence: tracked buffers, the writer behind three busy command buffers of its queue.
pub const TRACKED_BEHIND_BUSY: &str = "fence-none-tracked-behind-busy";

/// No fence: tracked buffers, the writer behind one busy command buffer of its queue.
pub const TRACKED_BEHIND_ONE: &str = "fence-none-tracked-behind-one";

/// No fence: tracked buffers, nothing ahead of the writer on its queue.
pub const TRACKED: &str = "fence-none-tracked";

/// The site each case's answer sits at.
const AT: &str = "order";

/// The accepted difference of a tracked write another queue's later command buffer sees where
/// the writer's queue holds nothing back (docs/API.md section 10), by its slug.
const SEEN_ACROSS_QUEUES: &str =
    "a-write-whose-queue-holds-nothing-back-is-seen-by-another-queue-s-command-buffer-committed-after-it";

/// A case over one fence experiment.
pub struct FenceOrderCase {
    name: &'static str,
    shape: FenceShape,
}

/// The six cases.
pub fn all() -> Vec<FenceOrderCase> {
    [
        (ORDERED, FenceShape::Ordered),
        (HELD_UPDATER, FenceShape::HeldUpdater),
        (EMPTY_WAITER, FenceShape::EmptyWaiter),
        (TRACKED, FenceShape::Tracked { busy: 0 }),
        (TRACKED_BEHIND_ONE, FenceShape::Tracked { busy: 1 }),
        (TRACKED_BEHIND_BUSY, FenceShape::Tracked { busy: 3 }),
    ]
    .into_iter()
    .map(|(name, shape)| FenceOrderCase { name, shape })
    .collect()
}

impl Case for FenceOrderCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // X and Y are `Shared` buffers the answer is read back out of, and on a device without
        // unified memory the runtime announces `shared-in-system-memory` for them
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the notices
        // site alone: every word count and status still compares exactly.
        let mut allowances = vec![crate::replay::shared_memory_notice()];
        // Behind one busy command buffer, the runtime's 1024 is its structure: a writer whose
        // queue has fewer than three submissions in flight is submitted at once, so the later copy
        // on another queue reads every word, on every device and run, the order Apple describes
        // for a tracked write. Metal's 0 is measured, not documented: on the M3 Max (macOS
        // 27.0.1) the copy ran first in 5 runs of 5 (job m3-max-52432), its tracking ordering only
        // against work already submitted (docs/API.md section 10). Either count holds; the
        // statuses still compare exactly.
        if self.name == TRACKED_BEHIND_ONE {
            allowances.push(Allowance::new(
                "case/order/y/words-written",
                Loosening::AnyOf(vec![Value::Uint(0), Value::Uint(WORDS)]),
                Justification::AcceptedDifference(SEEN_ACROSS_QUEUES.to_owned()),
            ));
        }
        allowances
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut job = LONG_FILL_BYTES.to_le_bytes().to_vec();
        job.extend(LONG_FILLS.to_le_bytes());
        job.extend((HELD_BOUND.as_millis() as u64).to_le_bytes());
        Ok(vec![
            Input::bytes("shape", &self.shape.words()),
            Input::bytes("buffer-bytes-and-written-byte", &{
                let mut words = BYTES.to_le_bytes().to_vec();
                words.push(WRITTEN);
                words
            }),
            Input::bytes("long-job-and-bound", &job),
        ])
    }

    fn record_runs(&self) -> u32 {
        // Each answer is an order between two queues' command buffers, which a run can read
        // differently from the next where Metal leaves the order open. More runs make that visible:
        // the recording holds the set of outcomes with their frequencies, as the event orders'
        // recordings do.
        5
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match side.fence_order(self.shape) {
            Ok(order) => sites.push(order.node(AT)),
            // Every fact the case has is in the one call, so there is nothing after it to read.
            Err(failure) => sites.push(Node::leaf(AT, Observed::value(failure.recorded()))),
        }
        Ok(())
    }
}
