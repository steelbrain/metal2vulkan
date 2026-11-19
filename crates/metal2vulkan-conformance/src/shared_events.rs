//! What a shared event's value does: the vocabulary for the event family's first case.
//!
//! `MTLSharedEvent`'s value **only rises**, from the host and from the GPU alike: a lower value set
//! or signalled leaves it where it was. A host wait for a value the event has reached answers at
//! once, and one for a value above it runs out.
//!
//! Every fact here is a value or a boolean, so the record is a small set of fields rather than
//! bytes, and the GPU's part is three command buffers that encode one signal each and nothing else.

use crate::case::field;
use crate::transcript::{Node, NodeName, Observed, Value};

/// How long a host wait is given before it counts as having run out.
///
/// Short on purpose: the waits a case makes are for values the event has **already** reached, which
/// answer at once, and for one it has not, which never will. A long bound would only make the case
/// slow.
pub const WAIT_MS: u64 = 10;

/// What one run of the rising-value probe observed. Each field is one call's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventRise {
    /// `label`, after `setLabel:`; a side whose runtime keeps no label refuses it.
    pub label: crate::case::Field<Option<String>>,
    /// `signaledValue` of a new event.
    pub initial: u64,
    /// `signaledValue` after the host set 5, then 3, then 5.
    pub after_host_sets: u64,
    /// `signaledValue` after a command buffer signalled 3, and then after one signalled 5.
    pub after_gpu_signals: [u64; 2],
    /// `waitUntilSignaledValue:timeoutMS:` for 3, for 5 and for 6, each with [`WAIT_MS`].
    pub waits: [bool; 3],
    /// `signaledValue` after a command buffer signalled `u64::MAX`.
    pub after_max_signal: u64,
    /// Whether the host wait for `u64::MAX` answered.
    pub waited_for_max: bool,
}

impl EventRise {
    /// The observations as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let label = |held: &Option<String>| match held {
            Some(label) => Value::Variant {
                name: "Some".to_owned(),
                payload: Box::new(Value::Lines(vec![label.clone()])),
            },
            None => Value::unit_variant("None"),
        };
        let mut node = Node::branch(name, Vec::new())
            .with(Node::leaf("label", field(&self.label, label)))
            .with(Node::leaf(
                "signaled-value/new",
                Observed::value(Value::Uint(self.initial)),
            ))
            .with(Node::leaf(
                "signaled-value/after-host-sets",
                Observed::value(Value::Uint(self.after_host_sets)),
            ));
        for (at, (value, signalled)) in self.after_gpu_signals.iter().zip([3u64, 5]).enumerate() {
            node = node.with(Node::leaf(
                format!("signaled-value/after-gpu-signal-{at}-of-{signalled}"),
                Observed::value(Value::Uint(*value)),
            ));
        }
        for (answered, wanted) in self.waits.iter().zip([3u64, 5, 6]) {
            node = node.with(Node::leaf(
                format!("wait-until-signaled-value/{wanted}"),
                Observed::value(Value::Bool(*answered)),
            ));
        }
        node.with(Node::leaf(
            "signaled-value/after-gpu-signal-of-u64-max",
            Observed::value(Value::Uint(self.after_max_signal)),
        ))
        .with(Node::leaf(
            "wait-until-signaled-value/u64-max",
            Observed::value(Value::Bool(self.waited_for_max)),
        ))
    }
}

/// What one run of the handle round trip observed: a shared event, the in-process handle it answers,
/// and the event the device opens from that handle.
///
/// The question is whether the handle names the **same** event rather than a copy of its value, and
/// one field settles it: [`EventHandleRoundTrip::value_on_the_first_after_the_second_was_set`] is
/// read on the first event after the host set a value on the second.
///
/// Nothing here reaches the GPU: every value is set and read by the host, and the two waits are the
/// bounded ones [`EventRise`] already uses — one for a value the event has reached, one for a value
/// nothing will ever signal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventHandleRoundTrip {
    /// `label` of the event, after `setLabel:`.
    pub label: Option<String>,
    /// Whether the event's `device` is the one the case's other objects were made on. The device
    /// object is not comparable, so this is the one portable question the getter answers.
    pub device_is_the_cases: bool,
    /// `signaledValue` of the new event, before anything set it.
    pub value_at_first: u64,
    /// `signaledValue` after the host set [`HANDLE_FIRST_VALUE`].
    pub value_after_the_host_set_it: u64,
    /// `label` of the handle `newSharedEventHandle` answered.
    pub handle_label: Option<String>,
    /// `label` of the event `newSharedEventWithHandle:` answered.
    pub reopened_label: Option<String>,
    /// Whether that event's `device` is the case's.
    pub reopened_device_is_the_cases: bool,
    /// `signaledValue` of that event: the first event's value, if the handle names the same event.
    pub reopened_value: u64,
    /// `signaledValue` of the **first** event after the host set [`HANDLE_SECOND_VALUE`] on the
    /// reopened one. One event means the first reads the new value; two would leave it where it
    /// was.
    pub value_on_the_first_after_the_second_was_set: u64,
    /// `waitUntilSignaledValue:timeoutMS:` on the reopened event for a value it has reached, and
    /// for one nothing will ever signal, each with [`WAIT_MS`].
    pub reopened_waits: [bool; 2],
}

/// The value the host sets on the event before taking its handle.
pub const HANDLE_FIRST_VALUE: u64 = 7;

/// The value the host sets on the **reopened** event, which the first event then reads if the two
/// are one event.
pub const HANDLE_SECOND_VALUE: u64 = 11;

/// The value the reopened event is asked to wait for and will never reach.
pub const HANDLE_UNREACHED_VALUE: u64 = 12;

impl EventHandleRoundTrip {
    /// The observations as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let label = |held: &Option<String>| match held {
            Some(held) => Value::Variant {
                name: "Some".to_owned(),
                payload: Box::new(Value::Lines(vec![held.clone()])),
            },
            None => Value::unit_variant("None"),
        };
        Node::branch(name, Vec::new())
            .with(Node::leaf("label", Observed::value(label(&self.label))))
            .with(Node::leaf(
                "device",
                Observed::value(Value::Bool(self.device_is_the_cases)),
            ))
            .with(Node::leaf(
                "signaled-value/new",
                Observed::value(Value::Uint(self.value_at_first)),
            ))
            .with(Node::leaf(
                "signaled-value/after-the-host-set-it",
                Observed::value(Value::Uint(self.value_after_the_host_set_it)),
            ))
            .with(Node::leaf(
                "handle/label",
                Observed::value(label(&self.handle_label)),
            ))
            .with(Node::leaf(
                "reopened/label",
                Observed::value(label(&self.reopened_label)),
            ))
            .with(Node::leaf(
                "reopened/device",
                Observed::value(Value::Bool(self.reopened_device_is_the_cases)),
            ))
            .with(Node::leaf(
                "reopened/signaled-value",
                Observed::value(Value::Uint(self.reopened_value)),
            ))
            .with(Node::leaf(
                "signaled-value/after-the-reopened-was-set",
                Observed::value(Value::Uint(
                    self.value_on_the_first_after_the_second_was_set,
                )),
            ))
            .with(Node::leaf(
                "reopened/wait-for-a-value-it-reached",
                Observed::value(Value::Bool(self.reopened_waits[0])),
            ))
            .with(Node::leaf(
                "reopened/wait-for-a-value-nothing-signals",
                Observed::value(Value::Bool(self.reopened_waits[1])),
            ))
    }
}
