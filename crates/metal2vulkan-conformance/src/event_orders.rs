//! What an event orders: the vocabulary for the four ordering facts of `runtime_lane::events`,
//! and for what Metal does with a wait whose signal is already submitted and running.
//!
//! An `MTLEvent` and an `MTLSharedEvent` hold a value that only rises, and a command buffer's
//! `encodeWaitForEvent:value:` holds that command buffer — and the ones committed behind it on its
//! own queue — until the value arrives. Everything a program can see of that is an **order of
//! events**: a status read while a command buffer is held, a handler that had not run yet, a signal
//! encoded before a wait that landed while the wait was still pending. None of it can be stated as
//! independent calls whose order nothing fixes, so each experiment here is **one** [`Side`] call
//! that runs the whole shape and answers what it saw, in the manner of
//! [`crate::queue_slots`] and [`crate::shared_events`].
//!
//! [`Side`]: crate::case::Side
//!
//! # Why none of this can hang
//!
//! Every wait of the suite's own thread is bounded, and every event a command buffer waits for is
//! one this module's own code raises:
//!
//! - a command buffer's scheduling and completion are observed through its **handlers**, which both
//!   sides have, and the probe's thread waits on a channel with a bound ([`COMPLETION`]) rather
//!   than inside a `waitUntilCompleted` Metal has no bounded form of;
//! - a host wait for a value the GPU will signal is given [`SIGNAL_WAIT`], which both
//!   `waitUntilSignaledValue:timeoutMS:` and the runtime's `wait_until_signaled_value` take as a
//!   bound and answer `false` from;
//! - a bound that runs out is **recorded** ([`Negative::NeverReturned`]) rather than retried, so a
//!   side that did not come back states that and the comparison reads it.
//!
//! # The buffers are small on purpose
//!
//! `runtime_lane::events` fills 8 MiB a command buffer, which is the size a *hazard* between two
//! queues needs to be seen at. The fact these cases pin is the event's, and the fills here are
//! [`SCRATCH`] bytes and write **disjoint** halves, so there is no hazard either way and a bytes
//! record stays far inside the recording budget.

use crate::transcript::{Negative, Node, NodeName, Observed, Value};
use std::time::Duration;

/// The bytes of each probe's scratch buffer. Small: what the cases observe is an order, not a
/// throughput, and a recording holds these bytes whole.
pub const SCRATCH: u64 = 256;

/// The bytes each of the two queues of [`CrossQueueWait`] fills: the halves of [`SCRATCH`],
/// disjoint, so neither fill is a hazard on the other and the bytes say only which ran.
pub const HALF: u64 = SCRATCH / 2;

/// The byte the waiting command buffer's fill writes, and the byte the signalling one's writes.
pub const WAITING_BYTE: u8 = 1;
pub const SIGNALLING_BYTE: u8 = 2;

/// The longest a probe's own thread waits for a command buffer's handler before it records that
/// the handler never came.
pub const COMPLETION: Duration = Duration::from_secs(30);

/// The bound a host wait for a value the GPU is about to signal is given.
///
/// It is an **input** of every case that makes such a wait, because it is what the wait's answer
/// means: `false` says "not within this long", not "never". Wide enough that a loaded machine does
/// not turn a signal that arrived into one that did not.
pub const SIGNAL_WAIT: Duration = Duration::from_secs(10);

/// How long a command buffer held by an unsignalled event is watched before its status is read.
///
/// The facts it guards are negative — this command buffer is not scheduled, its handlers have not
/// run — and a negative fact needs a window to be false in. Short, because the window is only there
/// to let a submission that was going to happen happen.
pub const HELD_FOR: Duration = Duration::from_millis(150);

/// What a command buffer's status read as, in the one vocabulary both sides answer in
/// (`metal_side`'s own `status_word` and the runtime's `Debug`), or that the side never came
/// back from the wait that was to settle it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// The status word: `NotEnqueued`, `Committed`, `Scheduled`, `Completed`, `Error`.
    Read(String),
    /// The handler the status was to be read after never came within its bound.
    NeverSettled,
}

impl Status {
    /// The status `word` read.
    pub fn read(word: impl Into<String>) -> Self {
        Self::Read(word.into())
    }

    /// The value the site records: the status as a unit variant, or the negative fact.
    pub fn value(&self) -> Value {
        match self {
            Status::Read(word) => Value::unit_variant(word.clone()),
            Status::NeverSettled => Value::Nothing(Negative::NeverReturned),
        }
    }
}

/// What the cross-queue wait probe observed: one queue's command buffer waits for an event the
/// other's signals, and the waiting one is committed **first**.
///
/// The commit order is the point. The two logical queues share the device's `VkQueue`s — one on most
/// devices — so a wait committed ahead of its signal deadlocks unless the implementation holds it
/// back rather than submitting it in commit order. Metal's answer is that it does not deadlock, and
/// this records what Metal's statuses and bytes are while it does not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossQueueWait {
    /// The waiting command buffer's status after its own commit and before the signalling one is
    /// committed.
    ///
    /// Deterministic, and that is why it is read there: the event is at 0 and nothing but the
    /// signalling command buffer — not yet committed — can raise it, so no run can find this
    /// command buffer further along.
    pub waiting_after_its_commit: Status,
    /// Its status once both completed.
    pub waiting_at_the_end: Status,
    /// The signalling command buffer's status once both completed.
    pub signalling_at_the_end: Status,
    /// The scratch buffer's [`SCRATCH`] bytes, read once both completed: the waiting command
    /// buffer's half and the signalling one's.
    pub bytes: Vec<u8>,
}

impl CrossQueueWait {
    /// The observations as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "waiting/status-after-its-own-commit",
                Observed::value(self.waiting_after_its_commit.value()),
            ))
            .with(Node::leaf(
                "waiting/status-once-both-completed",
                Observed::value(self.waiting_at_the_end.value()),
            ))
            .with(Node::leaf(
                "signalling/status-once-both-completed",
                Observed::value(self.signalling_at_the_end.value()),
            ))
            .with(Node::leaf(
                "scratch",
                Observed::bytes(self.bytes.clone(), scratch_layout()),
            ))
    }
}

/// The layout a scratch buffer's bytes are recorded under: [`SCRATCH`] bytes of one buffer, read
/// as bytes, with nothing outside them.
fn scratch_layout() -> crate::transcript::ByteLayout {
    crate::transcript::ByteLayout::buffer(0, SCRATCH, crate::transcript::Element::U8)
}

/// What the held-command-buffer probe observed: a command buffer waiting for a shared event the
/// **host** raises later, one committed behind it on its own queue, and one on another queue.
///
/// Three facts in one shape, and all three are orders:
///
/// - the held command buffer and the one behind it stay where they are, while the other queue's
///   command buffer completes — one queue's hold is one queue's;
/// - neither of the held one's handlers has run;
/// - once the host raises the event, its scheduled handler runs before its completed handler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldCommandBuffer {
    /// The other queue's command buffer's status once it completed.
    pub other_queue: Status,
    /// The held command buffer's status while its event is unsignalled, read after [`HELD_FOR`].
    pub held_while_waiting: Status,
    /// The status of the command buffer committed behind it on the same queue, read then too.
    pub behind_while_waiting: Status,
    /// The held command buffer's handlers that had run by then, in order.
    ///
    /// Empty is the fact the case is about, and it records as [`Negative::RanNothing`] rather than
    /// as an empty list: "no handler ran" and "nobody looked" must not print the same.
    pub handlers_while_waiting: Vec<String>,
    /// Its handlers once the host raised the event and both command buffers completed, in order.
    pub handlers_at_the_end: Vec<String>,
    /// Its status at the end, and the status of the one behind it.
    pub held_at_the_end: Status,
    pub behind_at_the_end: Status,
}

impl HeldCommandBuffer {
    /// The observations as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "other-queue/status-once-it-completed",
                Observed::value(self.other_queue.value()),
            ))
            .with(Node::leaf(
                "held/status-while-its-event-is-unsignalled",
                Observed::value(self.held_while_waiting.value()),
            ))
            .with(Node::leaf(
                "behind/status-while-its-event-is-unsignalled",
                Observed::value(self.behind_while_waiting.value()),
            ))
            .with(Node::leaf(
                "held/handlers-while-its-event-is-unsignalled",
                Observed::value(handlers(&self.handlers_while_waiting)),
            ))
            .with(Node::leaf(
                "held/handlers-once-the-host-signalled",
                Observed::value(handlers(&self.handlers_at_the_end)),
            ))
            .with(Node::leaf(
                "held/status-at-the-end",
                Observed::value(self.held_at_the_end.value()),
            ))
            .with(Node::leaf(
                "behind/status-at-the-end",
                Observed::value(self.behind_at_the_end.value()),
            ))
    }
}

/// The value a list of handler calls records as: the names in order, or the negative fact that none
/// ran.
fn handlers(called: &[String]) -> Value {
    if called.is_empty() {
        Value::Nothing(Negative::RanNothing)
    } else {
        Value::Lines(called.to_vec())
    }
}

/// What the signal-before-a-wait probe observed: **one** command buffer that fills, signals
/// `before`, waits for `gate`, fills again and signals `after`.
///
/// The fact is that the work ahead of the wait runs and its signal lands while the wait is still
/// pending, so a consumer can see a command buffer's progress before the whole of it goes. The
/// command buffer stays short of `Scheduled` meanwhile, and the signal behind the wait has not
/// landed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalBeforeWait {
    /// Whether the host's bounded wait for `before`'s 1 answered — the signal ahead of the wait
    /// landed while the command buffer was still held.
    pub before_landed: bool,
    /// `after`'s value while the gate is unsignalled: the signal behind the wait has not landed.
    pub after_while_pending: u64,
    /// The command buffer's status then.
    pub status_while_pending: Status,
    /// `after`'s value once the host raised the gate and the command buffer completed.
    pub after_at_the_end: u64,
    /// Its status then.
    pub status_at_the_end: Status,
}

impl SignalBeforeWait {
    /// The observations as the sites a case emits under one node, so both sides emit one shape.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "before/host-wait-answered-while-the-gate-was-unsignalled",
                Observed::value(Value::Bool(self.before_landed)),
            ))
            .with(Node::leaf(
                "after/value-while-the-gate-was-unsignalled",
                Observed::value(Value::Uint(self.after_while_pending)),
            ))
            .with(Node::leaf(
                "status-while-the-gate-was-unsignalled",
                Observed::value(self.status_while_pending.value()),
            ))
            .with(Node::leaf(
                "after/value-once-the-host-raised-the-gate",
                Observed::value(Value::Uint(self.after_at_the_end)),
            ))
            .with(Node::leaf(
                "status-at-the-end",
                Observed::value(self.status_at_the_end.value()),
            ))
    }
}

/// The bytes of the private buffer the signalling command buffer of [`WaitBehindRunningSignal`]
/// fills, [`LONG_FILLS`] times over, ahead of its signal: 256 MiB.
pub const LONG_FILL_BYTES: u64 = 256 << 20;

/// How many fills of [`LONG_FILL_BYTES`] the signalling command buffer of
/// [`WaitBehindRunningSignal`] runs before its signals: fills alone, no loop on the GPU. 48 GiB
/// written, about 158 ms of GPU time on the recording M3 Max (Metal's `GPUEndTime` less
/// `GPUStartTime` of the same fills, 156 to 163 ms over three runs), some eight times
/// [`RUNNING_WINDOW`]; the record says whether the work outlasted the reading on each side
/// ([`WaitBehindRunningSignal::landed_before_the_reading`]).
pub const LONG_FILLS: u32 = 192;

/// How long the waiting command buffer of [`WaitBehindRunningSignal`] is given, once both are
/// committed, before its status is read while the signalling one still runs: as [`HELD_FOR`], the
/// window a submission that was going to happen has had its chance in, and short enough to close
/// well before the signalling command buffer's long work does.
pub const RUNNING_WINDOW: Duration = Duration::from_millis(20);

/// What the wait-behind-a-running-signal probe observed: one queue's command buffer runs a long
/// GPU job ([`LONG_FILLS`] fills of [`LONG_FILL_BYTES`]), signals a companion shared event at 1
/// and then the event under test at 1; another queue's command buffer waits for the event at 1
/// and then fills [`SCRATCH`] bytes. The signalling one is committed first, then the waiting one;
/// [`RUNNING_WINDOW`] later the waiting one's status and handlers are read, then the signalling
/// one's status, then the companion's value.
///
/// The question is whether Metal submits a command buffer whose wait is met only by a signal
/// already submitted and still running, so the wait is on the GPU, or holds it until the signal
/// lands. Its status is the proxy for "submitted, and so checked": Metal answers `Scheduled` once
/// a command buffer is submitted to the GPU (and runs its scheduled handlers then), so
/// `Scheduled` while the signaller runs says "submitted before the signal landed", and
/// `Committed` says "held until it lands".
///
/// **The reading has an answer only while the signal is still running**, and the companion says
/// whether it was. The signalling command buffer signals it just before the event, and one
/// command buffer's signals land in order, so a companion still at 0 when read, last, says the
/// event's signal had not landed when anything before it was read. A run whose companion reads 1
/// holds no reading: [`Self::node`] then records that fact alone where the reading would be, so it
/// differs from a recording that has an answer rather than passing as one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaitBehindRunningSignal {
    /// Whether the signal had landed by the time the rest was read (the companion read 1).
    pub landed_before_the_reading: bool,
    /// The waiting command buffer's status, [`RUNNING_WINDOW`] after both commits.
    pub waiting_while_it_runs: Status,
    /// Its handlers that had run by then, in order.
    pub waiting_handlers_while_it_runs: Vec<String>,
    /// A shared event's `signaledValue` then; `None` for an `MTLEvent`, which has none.
    pub value_while_it_runs: Option<u64>,
    /// The signalling command buffer's status, read right after the waiting one's.
    pub signalling_while_it_runs: Status,
    /// The waiting one's handlers once both completed, in order.
    pub waiting_handlers_at_the_end: Vec<String>,
    /// Both statuses once both completed.
    pub waiting_at_the_end: Status,
    pub signalling_at_the_end: Status,
    /// The scratch buffer's [`SCRATCH`] bytes once both completed: the waiting one's fill.
    pub bytes: Vec<u8>,
}

impl WaitBehindRunningSignal {
    /// The observations as the sites a case emits under one node, so both sides emit one shape:
    /// the reading while the signaller runs only where the signal had not landed by then.
    pub fn node(&self, name: impl Into<NodeName>) -> Node {
        let mut node = Node::branch(name, Vec::new()).with(Node::leaf(
            "signalling/signal-landed-before-the-reading",
            Observed::value(Value::Bool(self.landed_before_the_reading)),
        ));
        if !self.landed_before_the_reading {
            node = node
                .with(Node::leaf(
                    "waiting/status-while-the-signaller-runs",
                    Observed::value(self.waiting_while_it_runs.value()),
                ))
                .with(Node::leaf(
                    "waiting/handlers-while-the-signaller-runs",
                    Observed::value(handlers(&self.waiting_handlers_while_it_runs)),
                ));
            if let Some(value) = self.value_while_it_runs {
                node = node.with(Node::leaf(
                    "event/value-while-the-signaller-runs",
                    Observed::value(Value::Uint(value)),
                ));
            }
            node = node.with(Node::leaf(
                "signalling/status-when-the-waiting-one-was-read",
                Observed::value(self.signalling_while_it_runs.value()),
            ));
        }
        node.with(Node::leaf(
            "waiting/handlers-once-both-completed",
            Observed::value(handlers(&self.waiting_handlers_at_the_end)),
        ))
        .with(Node::leaf(
            "waiting/status-once-both-completed",
            Observed::value(self.waiting_at_the_end.value()),
        ))
        .with(Node::leaf(
            "signalling/status-once-both-completed",
            Observed::value(self.signalling_at_the_end.value()),
        ))
        .with(Node::leaf(
            "scratch",
            Observed::bytes(self.bytes.clone(), scratch_layout()),
        ))
    }
}

/// The name a scheduled handler logs itself as, and the name a completed handler logs itself as.
///
/// They are the words the record holds, so both sides must spell them the same; a case's input
/// fingerprint covers them.
pub const SCHEDULED: &str = "scheduled";
pub const COMPLETED: &str = "completed";
