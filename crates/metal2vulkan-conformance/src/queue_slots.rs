//! A command queue's limit on uncompleted command buffers, as a probe both sides run.
//!
//! Metal's `commandBuffer` **blocks** once a queue holds `maxCommandBufferCount` uncompleted
//! command buffers, and answers once one of them completes or is dropped uncommitted. It never
//! answers nil. That is a behaviour nothing in the suite could state before: the fact under test is
//! that a call *did not return*, and then that it did once a slot came back.
//!
//! A case over this probe measures the default count on the oracle, and records it keyed to the
//! oracle.
//!
//! # Why no wait here can hang
//!
//! Every make runs on a thread of its own, and the command buffers it makes go into a shared list
//! the probe's own thread can drain. So a make that blocks is always released by the probe dropping
//! one held command buffer, and the worker is told to stop as soon as the probe has its answer: the
//! worker makes at most one command buffer after the probe is done with it, and is joined. Nothing
//! is committed, so nothing reaches the GPU and no wait is for the device.
//!
//! # What is deliberately not here
//!
//! `maxCommandBufferCount:` of **0**. Metal's own validation ends the process on it, so no case
//! may ask Metal for it; the runtime refuses it typed (`Refusal::NoCommandBuffers`).

use crate::transcript::{Negative, Node, Observed, Value};
use std::time::Duration;

/// How long a make must stay unanswered to count as blocked.
///
/// Long enough that a make slowed by something other than the queue's limit — a queue's first use,
/// a loaded machine — is not read as blocked, and short enough that a case of four rows is quick.
/// A run where it is read wrong does not pass quietly: the recording holds the **set** of outcomes
/// over its runs, so a flapping row records as two outcomes and says so.
pub const BLOCKED: Duration = Duration::from_millis(400);

/// The longest the probe waits for a make it has just released.
pub const RELEASED: Duration = Duration::from_secs(30);

/// What one queue's slot probe observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueSlots {
    /// Makes that answered without blocking, from the queue's first: the queue's limit, where the
    /// next one blocked.
    pub answered_at_once: u32,
    /// Whether the make past them stayed unanswered for [`BLOCKED`].
    pub next_blocked: bool,
    /// Whether that make answered once one held command buffer was dropped uncommitted.
    ///
    /// `false` where it never blocked, since there was nothing to release.
    pub answered_once_a_slot_came_back: bool,
    /// Whether another queue's make answered while that one was blocked.
    ///
    /// `false` where it never blocked. The fact is that one queue's limit is one queue's: a
    /// runtime that took a device-wide lock would fail here.
    pub other_queue_answered: bool,
}

impl QueueSlots {
    /// The probe's observations as the sites a case emits under one node, so both sides emit one
    /// shape.
    pub fn node(&self, name: impl Into<crate::transcript::NodeName>) -> Node {
        // "Did not block" is a fact of its own, not an absent field: a make that returned and a
        // make nobody timed must read differently, and this is the record kind that says so.
        let blocked = if self.next_blocked {
            Value::unit_variant("Blocked")
        } else {
            Value::Nothing(Negative::DidNotBlock)
        };
        Node::branch(name, Vec::new())
            .with(Node::leaf(
                "answered-at-once",
                Observed::value(Value::Uint(u64::from(self.answered_at_once))),
            ))
            .with(Node::leaf("next-make", Observed::value(blocked)))
            .with(Node::leaf(
                "answered-once-a-slot-came-back",
                Observed::value(Value::Bool(self.answered_once_a_slot_came_back)),
            ))
            .with(Node::leaf(
                "another-queue-went-on",
                Observed::value(Value::Bool(self.other_queue_answered)),
            ))
    }
}

/// Run the slot probe over at most `cap` makes of one queue: `make` makes the queue's next command
/// buffer, and `other_make` makes one on a second queue and says whether it answered.
///
/// The probe is shared by both sides so neither can drift into a different experiment. `make` runs
/// on a worker thread, because the fact under test is that it does not return; the command buffers
/// it makes go into a list this thread can drain, which is how a blocked make is released. Nothing
/// is committed.
pub fn probe<C: Send + 'static>(
    cap: u32,
    make: impl Fn() -> Result<C, crate::case::Failure> + Send + 'static,
    other_make: impl Fn() -> bool,
) -> Result<QueueSlots, crate::case::Failure> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc, Mutex};

    let held: Arc<Mutex<Vec<C>>> = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let (made, answers) = mpsc::channel();
    let worker = {
        let (held, stop) = (Arc::clone(&held), Arc::clone(&stop));
        std::thread::spawn(move || {
            for _ in 0..cap {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                match make() {
                    Ok(command_buffer) => {
                        if let Ok(mut held) = held.lock() {
                            held.push(command_buffer);
                        }
                        if made.send(Ok(())).is_err() {
                            break;
                        }
                    }
                    Err(failure) => {
                        let _ = made.send(Err(failure));
                        break;
                    }
                }
            }
        })
    };
    let mut slots = QueueSlots {
        answered_at_once: 0,
        next_blocked: false,
        answered_once_a_slot_came_back: false,
        other_queue_answered: false,
    };
    let mut refused = None;
    for _ in 0..cap {
        match answers.recv_timeout(BLOCKED) {
            Ok(Ok(())) => slots.answered_at_once += 1,
            Ok(Err(failure)) => {
                refused = Some(failure);
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                slots.next_blocked = true;
                // One queue's limit is one queue's: another queue's make answers meanwhile.
                slots.other_queue_answered = other_make();
                // Release one slot by dropping a command buffer held uncommitted, which is what
                // lets the blocked make answer.
                let one = held.lock().ok().and_then(|mut held| held.pop());
                drop(one);
                slots.answered_once_a_slot_came_back =
                    matches!(answers.recv_timeout(RELEASED), Ok(Ok(())));
                break;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    stop.store(true, Ordering::SeqCst);
    // Every slot is released, so a make still in flight answers and the worker then sees the stop.
    if let Ok(mut held) = held.lock() {
        held.clear();
    }
    if slots.next_blocked && !slots.answered_once_a_slot_came_back {
        // A make the probe released and that still did not answer is a divergence, and the row
        // says so. The worker is left to finish on its own rather than joined, so no case of the
        // suite can hang waiting for it.
    } else {
        let _ = worker.join();
    }
    match refused {
        Some(failure) => Err(failure),
        None => Ok(slots),
    }
}
