//! What an event orders, measured on Metal: the four ordering facts `runtime_lane::events` holds
//! as assertions of the runtime against itself.
//!
//! An event's value only rises ([`super::shared_events`]) — these cases are about what it
//! **orders**. A command buffer's `encodeWaitForEvent:value:` holds that command buffer, and the
//! ones committed behind it on its own queue, until the value arrives; a command buffer's
//! `encodeSignalEvent:value:` raises it where it is encoded, not when the command buffer ends.
//! Four cases:
//!
//! - [`CROSS_QUEUE_WAIT`]: one queue's command buffer waits for an event the other's signals, and
//!   the waiting one is committed **first**. Both run and both fills land.
//! - [`HIGHEST_SIGNAL`]: two queues signal one shared event with 7 and 4, in both submission
//!   orders. The event ends at 7 either way.
//! - [`HELD_COMMAND_BUFFER`]: a command buffer held by a shared event the host raises later holds
//!   back its own queue and not another's, neither of its handlers has run, and once the host
//!   raises the event its scheduled handler runs before its completed one.
//! - [`SIGNAL_BEFORE_WAIT`]: the work ahead of a wait inside one command buffer runs, and its
//!   signal lands, while the wait is still pending.
//!
//! # What this pulls forward
//!
//! Every one of these numbers was ours before. `runtime_lane::events` runs the shapes on the
//! runtime and asserts the answers the module's prose says Metal gave — "as measured on Metal",
//! "measured on the M3 and the M4" — with no Metal leg in the test and, for the statuses, no
//! measurement written down at all beyond the word in the comment. Here both sides run and the
//! answers are compared field by field, so the statuses, the handler order and the bytes are
//! Metal's.
//!
//! The commit order of [`CROSS_QUEUE_WAIT`] is the one that can deadlock. Two logical queues share
//! the device's `VkQueue`s — one on most devices — so a wait committed ahead of its signal
//! deadlocks unless the implementation holds it back instead of submitting in commit order. That
//! is a property of ours with a Metal answer behind it, and this is the first time the Metal answer
//! is recorded rather than described.
//!
//! # Why none of it can hang, and none of it holds a GPU
//!
//! Every event a command buffer waits for is one the probe itself raises, from the host or from a
//! command buffer already committed. Every wait of the probe's own thread is bounded: a command
//! buffer's progress is observed through its **handlers**, with a channel and a bound, because
//! Metal has no bounded `waitUntilCompleted`; a host wait for an event takes its bound as an
//! argument on both sides. A bound that runs out is recorded as a negative fact, never retried.
//!
//! The work is four fills of [`crate::event_orders::SCRATCH`] bytes. Nothing renders, nothing
//! dispatches, no loop runs on the GPU and nothing waits on a value no code here raises.
//!
//! # What is deliberately not here
//!
//! The lane's **tracked** resources across queues (`a_tracked_buffer_across_queues_follows_the_
//! commit_order` and its texture twin) are left out: the answer there is "what the order the
//! runtime schedules them in gives it", which is a fact about our scheduler with Metal's own
//! answer attached by a comment naming an M4. It is a real question and it deserves a case; it
//! needs the hazard-tracking vocabulary first, and a hazard between two queues needs work big
//! enough to overlap, which a bytes record of a small buffer cannot stand in for.
//!
//! The lane's refusals of an event command **with an encoder open** and **after the commit** are
//! left out, and must stay out: the lane's own prose says Metal asserts there, which ends Metal's
//! process. They are the shape the suite has no outcome kind for, and the round's rule is that
//! such a design is written down and left unbuilt. The runtime's typed refusals of both stay
//! `runtime_lane` assertions.

use crate::case::{Abort, Case, Input, MetalSource, Side};
use crate::event_orders::{COMPLETED, HELD_FOR, SCHEDULED, SCRATCH, SIGNAL_WAIT};
use crate::transcript::{Node, Observed, Sites, Value};

/// A wait committed before the signal that meets it.
pub const CROSS_QUEUE_WAIT: &str = "event-cross-queue-wait";

/// Two queues signalling one shared event, in both orders.
pub const HIGHEST_SIGNAL: &str = "event-highest-signal";

/// A command buffer held by a shared event the host raises later.
pub const HELD_COMMAND_BUFFER: &str = "event-held-command-buffer";

/// A signal encoded before a wait, inside one command buffer.
pub const SIGNAL_BEFORE_WAIT: &str = "event-signal-before-a-wait";

/// The values the two queues of [`HIGHEST_SIGNAL`] signal, in each submission order.
///
/// Both orders are rows of one case because the fact is the pair's: a lower signal cannot lower the
/// value, so whichever executed last the event holds the higher. One order alone would not say
/// that.
const SIGNAL_PAIRS: [[u64; 2]; 2] = [[7, 4], [4, 7]];

/// The site each row of the four cases sits at.
const AT: &str = "order";

/// The bytes every case's wait bounds fingerprint as: the window a negative fact is read after, and
/// the bound a host wait for a GPU signal is given.
///
/// Both are inputs because they are what the answers mean. A status read "while the event is
/// unsignalled" is read after the window; a host wait's `false` says "not within this long".
fn bounds() -> Vec<u8> {
    let mut words = (HELD_FOR.as_millis() as u64).to_le_bytes().to_vec();
    words.extend((SIGNAL_WAIT.as_millis() as u64).to_le_bytes());
    words
}

/// A case over one ordering fact of the event family.
pub struct EventOrder {
    name: &'static str,
    shape: Shape,
}

/// Which experiment a case runs.
#[derive(Clone, Copy, Debug)]
enum Shape {
    CrossQueueWait,
    HighestSignal,
    Held,
    SignalBeforeWait,
}

/// The four cases.
pub fn all() -> Vec<EventOrder> {
    [
        (CROSS_QUEUE_WAIT, Shape::CrossQueueWait),
        (HIGHEST_SIGNAL, Shape::HighestSignal),
        (HELD_COMMAND_BUFFER, Shape::Held),
        (SIGNAL_BEFORE_WAIT, Shape::SignalBeforeWait),
    ]
    .into_iter()
    .map(|(name, shape)| EventOrder { name, shape })
    .collect()
}

impl Case for EventOrder {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<crate::allowance::Allowance> {
        // The cross-queue row is the one row of the family that reads an answer back out of a
        // `Shared` buffer, and on a device without unified memory the runtime announces
        // `shared-in-system-memory` for it (`crate::replay::shared_memory_notice` holds the whole
        // reason). It declares the notices site alone: the scratch words still compare exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): the notice fired for
        // `event-cross-queue-wait` alone, which was divergent on that field and nothing else, and
        // the other three rows were exact (replay-nvidia.log). So the other three name nothing:
        // what they read back is a count our side holds, not a buffer the device wrote.
        match self.shape {
            Shape::CrossQueueWait => vec![crate::replay::shared_memory_notice()],
            Shape::HighestSignal | Shape::Held | Shape::SignalBeforeWait => Vec::new(),
        }
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = vec![
            Input::bytes("scratch-bytes", &SCRATCH.to_le_bytes()),
            Input::bytes("wait-bounds-millis", &bounds()),
        ];
        match self.shape {
            Shape::CrossQueueWait => {
                let mut fills = Vec::new();
                for byte in [
                    crate::event_orders::WAITING_BYTE,
                    crate::event_orders::SIGNALLING_BYTE,
                ] {
                    fills.push(byte);
                }
                inputs.push(Input::bytes("fill-bytes", &fills));
            }
            Shape::HighestSignal => {
                let mut values = Vec::new();
                for pair in SIGNAL_PAIRS {
                    for value in pair {
                        values.extend(value.to_le_bytes());
                    }
                }
                inputs.push(Input::bytes("signal-pairs", &values));
            }
            // The handler names are what the record holds, so a change to either word is a change
            // to the case's inputs.
            Shape::Held => inputs.push(Input::bytes(
                "handler-names",
                format!("{SCHEDULED},{COMPLETED}").as_bytes(),
            )),
            Shape::SignalBeforeWait => {}
        }
        Ok(inputs)
    }

    fn record_runs(&self) -> u32 {
        // Every one of these cases reads at least one **negative** fact after a window: this
        // command buffer is not scheduled, that handler has not run, that signal has not landed.
        // A negative fact read after a window is a race read, however wide the window, so more
        // runs do not make it safe — they make it visible. The recording holds the set of outcomes
        // with their frequencies, so a field that flaps says so instead of passing as whichever
        // run came last.
        5
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        match self.shape {
            Shape::CrossQueueWait => match side.event_cross_queue_wait() {
                Ok(order) => sites.push(order.node(AT)),
                // The row records the failure and the run ends: every fact the case has is in the
                // one call, so there is nothing after it to read.
                Err(failure) => sites.push(Node::leaf(AT, Observed::value(failure.recorded()))),
            },
            Shape::HighestSignal => {
                for pair in SIGNAL_PAIRS {
                    let at = format!("{AT}/{}-then-{}", pair[0], pair[1]);
                    match side.event_highest_signal(pair) {
                        Ok(value) => sites.value(at, Value::Uint(value)),
                        // One order's answer is still readable where the other refused, so the
                        // sweep goes on.
                        Err(failure) => {
                            sites.push(Node::leaf(at, Observed::value(failure.recorded())))
                        }
                    }
                }
            }
            Shape::Held => match side.held_command_buffer() {
                Ok(order) => sites.push(order.node(AT)),
                Err(failure) => sites.push(Node::leaf(AT, Observed::value(failure.recorded()))),
            },
            Shape::SignalBeforeWait => match side.signal_before_a_wait() {
                Ok(order) => sites.push(order.node(AT)),
                Err(failure) => sites.push(Node::leaf(AT, Observed::value(failure.recorded()))),
            },
        }
        Ok(())
    }
}
