//! A device's lowering thread (`render/deferral.rs`): it lowers the batches of logged calls the
//! device's render encoders hand it, while the program goes on.
//!
//! - **Order.** It takes the slots it is handed in the order they were handed, and each slot's
//!   batches in the order the program logged them, one at a time: a slot's calls are lowered in
//!   their order whichever thread lowers them, since a drain on the program's thread waits for the
//!   batch the thread is lowering and takes the rest (`LoweringSlot::drain`).
//! - **The program's calls.** Each batch is lowered as the calls it holds
//!   (`exit_guard::lowering_program_calls`): the notices it announces, the device's maps it
//!   changes and the pipelines it waits for are those calls', which the program made in that
//!   order on its own thread.
//! - **Exit.** Each batch is lowered inside a section of the exit guard, which an exit waits for.
//!   Once the process exits it lowers none: the program's thread lowers what it was handed at its
//!   next drain, or not at all.
//! - **A panic** while lowering ends the thread: the slot it was lowering refuses every later
//!   drain with that panic, on the program's thread, as the call would have panicked under
//!   `RenderLoweringPolicy::AtCall`, and from then on the program's threads lower every slot's
//!   calls themselves.
//! - **Ends bounded.** At most [`ENDS_IN_FLIGHT`] encoders' ends are handed to the thread and not
//!   yet lowered: the next waits in `end_encoding` until the thread lowers one
//!   (`WorkCounts::render_log_waits`). So a program that outruns the thread holds that many
//!   encoders' logs, command buffers and the runtime's pools for them, not a queue's every command
//!   buffer, which the thread lowers without the program, so the wait ends.
//! - **Commits.** The device's commits that wait for its render encoders' ends lie here too
//!   (`render/queued_commits.rs`): the thread that lowers such an end runs them.
//! - **Started** with the device's first encoder that logs its calls, at the user-interactive
//!   class (`exit_guard::spawn_user_interactive`), as the threads command buffers' work waits for;
//!   stopped and joined with the device (`DeviceShared`'s drop). No slot handed to it holds the
//!   device by then: an encoder holds its device until it ends or goes, and its slot lets go of
//!   its lowering then.

use super::deferral::LoweringSlot;
use super::queued_commits::QueuedCommits;
use crate::device::lock;
use crate::exit_guard;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// The encoders' ends handed to the thread and not yet lowered at which the next end waits (the
/// module docs): a few passes' lead over the thread, enough to keep it busy while the program
/// encodes the next pass, each end holding at most its encoder's batches
/// (`render/deferral.rs`).
pub(crate) const ENDS_IN_FLIGHT: usize = 4;

/// The device's lowering thread (the module docs).
pub(crate) struct LoweringWorker {
    shared: Arc<WorkerShared>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// What the thread and the encoders that hand it their slots share.
struct WorkerShared {
    queue: Mutex<Queue>,
    wake: Condvar,
    /// The thread ended by a panic (the module docs).
    ended: AtomicBool,
    /// The device's commits that wait for its render encoders' ends (the module docs).
    commits: QueuedCommits,
    /// The ends handed to the thread, or about to be, and not yet lowered (the module docs).
    ends: Mutex<usize>,
    /// Signalled as one of them is lowered.
    end_lowered: Condvar,
}

struct Queue {
    /// The slots handed to the thread, in order.
    slots: VecDeque<Arc<LoweringSlot>>,
    /// The thread waits for a slot.
    parked: bool,
    /// The device is torn down: the thread ends.
    stop: bool,
}

impl LoweringWorker {
    /// The device's lowering thread, started now; `None` where the host starts no thread, where
    /// the program's threads lower every logged call at their drains.
    pub(super) fn start() -> Option<LoweringWorker> {
        let shared = Arc::new(WorkerShared {
            queue: Mutex::new(Queue {
                slots: VecDeque::new(),
                parked: false,
                stop: false,
            }),
            wake: Condvar::new(),
            ended: AtomicBool::new(false),
            commits: QueuedCommits::new(),
            ends: Mutex::new(0),
            end_lowered: Condvar::new(),
        });
        let running = Arc::clone(&shared);
        let builder = std::thread::Builder::new().name("metal2vulkan-lowering".to_owned());
        let thread = exit_guard::spawn_user_interactive(builder, move || running.run()).ok()?;
        Some(LoweringWorker {
            shared,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Hand `slot`, which has batches handed and is in no queue, to the thread: whether it takes
    /// it (it has not ended).
    pub(super) fn hand(&self, slot: &Arc<LoweringSlot>) -> bool {
        let mut queue = lock(&self.shared.queue);
        if self.shared.ended.load(Ordering::Acquire) {
            return false;
        }
        queue.slots.push_back(Arc::clone(slot));
        if queue.parked {
            self.shared.wake.notify_one();
        }
        true
    }

    /// The device's commits that wait for its render encoders' ends (the module docs).
    pub(super) fn commits(&self) -> &QueuedCommits {
        &self.shared.commits
    }

    /// Room for an end to hand to the thread (the module docs): wait while [`ENDS_IN_FLIGHT`]
    /// ends are handed and not lowered, unless the thread ended or the process exits. Whether
    /// the caller waited.
    pub(super) fn reserve_end(&self) -> bool {
        let mut ends = lock(&self.shared.ends);
        let mut waited = false;
        while *ends >= ENDS_IN_FLIGHT && self.runs() && !exit_guard::closed() {
            waited = true;
            ends = (self.shared.end_lowered.wait_timeout(ends, exit_guard::WAKE))
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        *ends += 1;
        waited
    }

    /// An end [`reserve_end`](Self::reserve_end) made room for was lowered, or will never be:
    /// room for the next.
    pub(super) fn end_done(&self) {
        let mut ends = lock(&self.shared.ends);
        debug_assert!(*ends > 0, "an end lowered that was not handed");
        *ends -= 1;
        drop(ends);
        self.shared.end_lowered.notify_one();
    }

    /// Whether the thread lowers what it is handed: it has not ended.
    pub(super) fn runs(&self) -> bool {
        !self.shared.ended.load(Ordering::Acquire)
    }

    /// End the thread and wait for it, as the device is torn down.
    pub(super) fn stop(&self) {
        lock(&self.shared.queue).stop = true;
        self.shared.wake.notify_one();
        if let Some(thread) = lock(&self.thread).take() {
            let _ = thread.join();
        }
    }
}

/// The first of `queue`, taken off, and `queue` left room for twice as many as waited, made on
/// the thread that takes, so a push grows the queue, an allocation on the pushing thread, only
/// where more wait than twice as many as this thread ever found, as the scheduler's inbox leaves
/// its commits room (`scheduler/inbox.rs`).
fn take_leaving_room<T>(queue: &mut VecDeque<T>) -> Option<T> {
    let waited = queue.len();
    let first = queue.pop_front()?;
    queue.reserve(waited + 1);
    Some(first)
}

impl WorkerShared {
    /// The thread's loop: each slot handed lowered, until [`LoweringWorker::stop`].
    fn run(&self) {
        let mut running = Running {
            shared: self,
            slot: None,
        };
        loop {
            let slot = {
                let mut queue = lock(&self.queue);
                loop {
                    if let Some(slot) = take_leaving_room(&mut queue.slots) {
                        break slot;
                    }
                    if queue.stop {
                        return;
                    }
                    queue.parked = true;
                    queue =
                        (self.wake.wait(queue)).unwrap_or_else(std::sync::PoisonError::into_inner);
                    queue.parked = false;
                }
            };
            running.slot = Some(slot);
            if let Some(slot) = &running.slot {
                slot.lower_handed();
            }
            running.slot = None;
        }
    }
}

/// The thread's loop as it runs: a panic that unwinds it ends the thread (the module docs).
struct Running<'a> {
    shared: &'a WorkerShared,
    /// The slot it lowers now.
    slot: Option<Arc<LoweringSlot>>,
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        let waiting: Vec<Arc<LoweringSlot>> = {
            let mut queue = lock(&self.shared.queue);
            self.shared.ended.store(true, Ordering::Release);
            queue.slots.drain(..).collect()
        };
        if let Some(slot) = &self.slot {
            slot.lost_to_panic();
        }
        for slot in waiting {
            slot.left_by_thread();
        }
    }
}
