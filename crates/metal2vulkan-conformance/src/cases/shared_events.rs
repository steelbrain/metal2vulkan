//! A shared event's value only rises, from the host and from the GPU alike.
//!
//! `MTLSharedEvent` holds one 64-bit value. The host reads and writes it with `signaledValue` and
//! `setSignaledValue:`, the GPU raises it with `encodeSignalEvent:value:`, and the host waits for it
//! with `waitUntilSignaledValue:timeoutMS:`. A **lower** value, set or signalled, leaves it where it
//! was: the value is a high-water mark, not a register.
//!
//! The case walks that in one shape:
//!
//! - a new event's value;
//! - the host sets 5, then 3, then 5 — it reads 5, the 3 having changed nothing;
//! - a command buffer signals 3, then another signals 5 — it reads 5;
//! - a host wait for 3 and one for 5 answer, and one for 6 runs out;
//! - a command buffer signals `u64::MAX`, and a host wait for it answers.
//!
//! `runtime_lane::events` holds this as `a_shared_events_value_only_rises`, which asserts the
//! runtime agrees with itself. Here both sides answer and the answers are compared field by field,
//! so the numbers are Metal's rather than ours.
//!
//! # Why it is safe and quick
//!
//! The GPU's part is three command buffers that encode one signal each and nothing else: no shader,
//! no pass, no dispatch. Every wait of the case's own thread is bounded
//! ([`crate::shared_events::WAIT_MS`]), including the one that is meant to run out, and the two
//! bounded waits that are meant to answer are for values the event has already reached.
//!
//! The ordering facts of the event family — a wait that holds work until a signal arrives, and the
//! order two queues see — are not here. They need completed and scheduled handlers in the
//! vocabulary, which nothing has yet.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::shared_events::{
    HANDLE_FIRST_VALUE, HANDLE_SECOND_VALUE, HANDLE_UNREACHED_VALUE, WAIT_MS,
};
use crate::transcript::{Node, Observed, Sites};

/// The case's name.
pub const RISING_VALUE: &str = "shared-event-rising-value";

/// The handle case's name.
pub const HANDLE_ROUND_TRIP: &str = "shared-event-handle-round-trip";

/// The label the handle case sets.
const HANDLE_LABEL: &str = "conformance-handle-round-trip";

/// The site the handle case's row sits at.
const HANDLE_AT: &str = "handle-round-trip";

/// The label the case sets, so `label` has something of ours to answer with.
const LABEL: &str = "conformance-rising-value";

/// The site the row sits at.
const AT: &str = "rising-value";

/// A case over what a shared event's value does.
pub struct RisingValue {
    name: &'static str,
}

/// The case.
pub fn rising_value() -> RisingValue {
    RisingValue { name: RISING_VALUE }
}

impl Case for RisingValue {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // What the case feeds: the label it sets, the values the host sets and the GPU signals in
        // order, the values it waits for, and the bound each wait is given. The bound is an input
        // because it decides what a wait's answer means.
        let mut words = Vec::new();
        for value in [5u64, 3, 5] {
            words.extend(value.to_le_bytes());
        }
        let mut signals = Vec::new();
        for value in [3u64, 5, u64::MAX] {
            signals.extend(value.to_le_bytes());
        }
        let mut waits = Vec::new();
        for value in [3u64, 5, 6, u64::MAX] {
            waits.extend(value.to_le_bytes());
        }
        Ok(vec![
            Input::bytes("label", LABEL.as_bytes()),
            Input::bytes("host-sets", &words),
            Input::bytes("gpu-signals", &signals),
            Input::bytes("host-waits-for", &waits),
            Input::bytes("wait-bound-millis", &WAIT_MS.to_le_bytes()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match side.shared_event_rises(LABEL) {
            Ok(rise) => sites.push(rise.node(AT)),
            // The row records the failure and the run ends: every fact the case has is in the one
            // call, so there is nothing after it to read.
            Err(failure) => sites.push(Node::leaf(AT, Observed::value(failure.recorded()))),
        }
        Ok(())
    }
}

/// A case over the handle a shared event answers, and the event the device opens from it.
///
/// `MTLSharedEventHandle` is a container **no case of the suite touched**, and
/// `newSharedEventWithHandle:` one of `MTLDevice`'s untouched calls. The handle is what a Metal
/// program passes between processes, and in one process it is the way to ask for the event again:
/// the question is whether the device opens the **same** event or a copy of its value, and one
/// field settles it — the value the host sets on the reopened event, read back on the first.
///
/// The event's `label` travels with it, which is the other thing the pair says: the handle carries
/// its event's label, and the reopened event answers it too.
///
/// # What it found
///
/// **A Metal shared event answers nil for `device`**, where a plain `MTLEvent` answers the device
/// it was made on. Measured on the oracle, and confirmed by a one-off Objective-C probe outside the
/// suite: `[[dev newEvent] device]` is the device and `[[dev newSharedEvent] device]` is nil, as is
/// the event reopened from the handle. `MTLEvent`'s header declares the getter nonnull, so the
/// asymmetry is Metal's behaviour rather than its documentation, and only a case that asks both
/// events the same question finds it.
///
/// The runtime answered the device for both until this case said so; it now answers `None`
/// ([`metal2vulkan_runtime::SharedEvent::device`]), which is what Metal answers, and these two
/// sites compare exactly like the rest of the pair.
///
/// # Why it is safe
///
/// Nothing reaches the GPU: no command buffer is made, no signal is encoded, and every value is
/// set and read by the host. Both waits are bounded by [`WAIT_MS`] — one for a value the event has
/// already reached, which answers at once, and one for a value nothing in the case will ever
/// signal, which runs out.
pub struct HandleRoundTrip {
    name: &'static str,
}

/// The case.
pub fn handle_round_trip() -> HandleRoundTrip {
    HandleRoundTrip {
        name: HANDLE_ROUND_TRIP,
    }
}

impl Case for HandleRoundTrip {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // What the case feeds: the label, the value set on the first event, the value set on the
        // reopened one, the value nothing signals, and the bound each wait is given. The bound is
        // an input because it decides what a wait's answer means.
        let mut values = Vec::new();
        for value in [
            HANDLE_FIRST_VALUE,
            HANDLE_SECOND_VALUE,
            HANDLE_UNREACHED_VALUE,
        ] {
            values.extend(value.to_le_bytes());
        }
        Ok(vec![
            Input::bytes("label", HANDLE_LABEL.as_bytes()),
            Input::bytes("values", &values),
            Input::bytes("wait-bound-millis", &WAIT_MS.to_le_bytes()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match side.shared_event_handle_round_trip(HANDLE_LABEL) {
            Ok(held) => sites.push(held.node(HANDLE_AT)),
            // Every fact the case has is in the one call, so the row records the failure and there
            // is nothing after it to read.
            Err(failure) => {
                sites.push(Node::leaf(HANDLE_AT, Observed::value(failure.recorded())));
            }
        }
        Ok(())
    }
}
