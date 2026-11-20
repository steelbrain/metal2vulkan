//! Where a command buffer's scheduling sits relative to its scheduled handlers: the vocabulary for
//! one Metal order.
//!
//! Metal runs a command buffer's scheduled handlers on a thread of its own, and the order is that
//! `waitUntilScheduled` returns **only once every scheduled handler returned**, while the status
//! already reads `Scheduled` from the moment the first of them starts. That is a relation between
//! three things — a handler, a status and a wait on another thread — so the whole experiment is one
//! [`Side`](crate::case::Side) call.
//!
//! # Why the shape is safe and bounded
//!
//! A scheduled handler of the probe's **blocks**, which is the only way to read a status "while a
//! scheduled handler runs". It blocks on a channel with a bound ([`HOLD_LIMIT`]), so it always
//! comes out, whatever the probe's own thread does; and it is released by the probe as soon as the
//! probe has its answer. Nothing of the GPU is held by it: the work is one small fill, which has
//! run or will run regardless of what a handler thread does.
//!
//! The wait under test runs on a thread of the probe's own, so a wait that did not return leaves
//! that thread rather than the suite. Where it does not return, the row records that
//! ([`Returned::NotWithinTheWindow`] at the end) instead of hanging.

use crate::event_orders::Status;
use crate::transcript::{Node, NodeName, Observed, Value};
use std::time::Duration;

/// The longest a held scheduled handler stays inside its own wait before it gives up and returns.
///
/// It is the handler's own safety net, not a measurement: the probe releases it in microseconds,
/// and this is what makes the shape impossible to leave a Metal handler thread stuck in.
pub const HOLD_LIMIT: Duration = Duration::from_secs(30);

/// How long `waitUntilScheduled` is watched, while the scheduled handler is held, before the probe
/// records that it had not returned.
///
/// An input of the case: "it had not returned" means "not within this long". Short, because the
/// fact is negative and the window only has to let a return that was going to happen happen.
pub const WINDOW: Duration = Duration::from_millis(150);

/// The longest the probe waits for the handler it is about to read.
pub const ARRIVAL: Duration = Duration::from_secs(30);

/// What a wait for the scheduling answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Returned {
    /// It returned, and answered `Ok`.
    Ok,
    /// It returned a typed error, as the value records it.
    Err(Value),
    /// It had not returned inside the window it was given.
    NotWithinTheWindow,
}

impl Returned {
    /// The value the site records.
    pub fn value(&self) -> Value {
        match self {
            Returned::Ok => Value::ok(Value::Absent),
            Returned::Err(value) => value.clone(),
            Returned::NotWithinTheWindow => {
                Value::Nothing(crate::transcript::Negative::NeverReturned)
            }
        }
    }
}

/// What the scheduled-handler order probe observed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledHandlerOrder {
    /// The status the scheduled handler read of its own command buffer, from inside itself.
    pub inside_the_handler: Status,
    /// The status the probe's own thread read while that handler was held inside itself.
    pub while_the_handler_ran: Status,
    /// Whether a `waitUntilScheduled` on another thread returned while the handler was held.
    ///
    /// The fact under test: Metal's does not.
    pub wait_returned_while_the_handler_ran: bool,
    /// What that wait answered once the handler was released.
    pub wait_answered: Returned,
    /// The status the completed handler read of its own command buffer, from inside itself.
    pub inside_the_completed_handler: Status,
    /// The handlers that ran, in the order they were called.
    pub handler_order: Vec<String>,
}

impl ScheduledHandlerOrder {
    /// The observations as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "status/inside-the-scheduled-handler",
                Observed::value(self.inside_the_handler.value()),
            ))
            .with(Node::leaf(
                "status/on-another-thread-while-the-scheduled-handler-ran",
                Observed::value(self.while_the_handler_ran.value()),
            ))
            .with(Node::leaf(
                "wait-until-scheduled/returned-while-the-scheduled-handler-ran",
                Observed::value(Value::Bool(self.wait_returned_while_the_handler_ran)),
            ))
            .with(Node::leaf(
                "wait-until-scheduled/answered-once-the-handler-returned",
                Observed::value(self.wait_answered.value()),
            ))
            .with(Node::leaf(
                "status/inside-the-completed-handler",
                Observed::value(self.inside_the_completed_handler.value()),
            ))
            .with(Node::leaf(
                "handler-order",
                Observed::value(Value::Lines(self.handler_order.clone())),
            ))
    }
}
