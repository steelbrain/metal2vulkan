//! Where a command buffer's scheduling sits relative to its scheduled handlers, measured on Metal.
//!
//! The order under test: a command buffer's status reads `Scheduled` just before its first
//! scheduled handler starts, and Metal's wait for the scheduling returns only once every scheduled
//! handler returned. This case takes it on the oracle, with both sides running, and compares six
//! fields:
//!
//! - the status the scheduled handler read of its own command buffer, from inside itself;
//! - the status the probe's thread read while that handler was held inside itself;
//! - whether a `waitUntilScheduled` on a third thread returned while the handler was held — the
//!   rule itself, as a fact that can be false;
//! - what that wait answered once the handler was released;
//! - the status the completed handler read of its own command buffer;
//! - the order the two handlers were called in.
//!
//! # Why holding a handler is the only way to ask
//!
//! Every field above is a statement about a moment **inside** a handler. A status read after
//! `commit` returned would be a race, not a fact: the command buffer may reach `Scheduled` between
//! the two statements, so a case that recorded it would record whichever run came last. Holding the
//! scheduled handler is what makes the window real, and so every field deterministic.
//!
//! # Why it cannot hang and holds no GPU
//!
//! The held handler blocks on a channel with a bound of its own
//! ([`crate::scheduled_handlers::HOLD_LIMIT`]), so it leaves Metal's handler thread whatever the
//! probe does, and the probe releases it as soon as it has its answer. The wait under test runs on
//! a thread of the probe's own and is never joined, so a wait that did not come back leaves that
//! thread and not the suite — the shape [`crate::queue_slots::probe`] already takes for the same
//! reason. The GPU work is one fill of [`crate::event_orders::SCRATCH`] bytes and is independent of
//! every handler thread: no pass, no dispatch, no loop.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::event_orders::{COMPLETED, SCHEDULED, SCRATCH};
use crate::scheduled_handlers::WINDOW;
use crate::transcript::{Node, Observed, Sites};

/// The case's name.
pub const SCHEDULED_HANDLER_ORDER: &str = "scheduled-handler-order";

/// The site the row sits at.
const AT: &str = "order";

/// A case over where a command buffer's scheduling sits relative to its scheduled handlers.
pub struct HandlerOrder {
    name: &'static str,
}

/// The case.
pub fn handler_order() -> HandlerOrder {
    HandlerOrder {
        name: SCHEDULED_HANDLER_ORDER,
    }
}

impl Case for HandlerOrder {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        Ok(vec![
            Input::bytes("scratch-bytes", &SCRATCH.to_le_bytes()),
            // The window is an input because it is what the negative field means: "the wait had
            // not returned" says "not within this long".
            Input::bytes("window-millis", &(WINDOW.as_millis() as u64).to_le_bytes()),
            Input::bytes(
                "handler-names",
                format!("{SCHEDULED},{COMPLETED}").as_bytes(),
            ),
        ])
    }

    fn record_runs(&self) -> u32 {
        // One field is negative and read after a window — the wait had not returned — so it is a
        // race read however wide the window is. More runs do not make it safe; they make it
        // visible, because the recording holds the set of outcomes with their frequencies.
        5
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match side.scheduled_handler_order() {
            Ok(order) => sites.push(order.node(AT)),
            // The row records the failure and the run ends: every fact the case has is in the one
            // call, so there is nothing after it to read.
            Err(failure) => sites.push(Node::leaf(AT, Observed::value(failure.recorded()))),
        }
        Ok(())
    }
}
