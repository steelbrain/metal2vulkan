//! A command buffer whose render encoder drew primitive type 9 directly, of 3 vertices or more,
//! at which Metal's GPU hangs (`DrawPrimitiveType::Nine`, `CommandBufferError::GpuHang`), or
//! whose render pass of no draw call over a mutable rasterization-rate map it hangs at
//! (`rate_mapped_pass.rs`), held alike.
//! Measured on the M3 Max (macOS 27.0.1, 33 hangs, queues warmed and not): it read `Scheduled`
//! 0.68 to 13.2 ms after its commit (about 1 ms, or 4.6 ms and more: its submission's own time),
//! and Metal failed it 28.3 to 46.2 ms after its commit, 27.5 to 28.9 ms after it read
//! `Scheduled` in 26 hangs (24.2 to 37.8 in all). Every command buffer of its queue that it
//! scheduled meanwhile completed with no error together with it, in the queue's order, its event
//! signals firing: those committed at once, after the hang read `Scheduled`, 5 and 20 ms on, one
//! writing what the hang's command buffer writes, and ones whose event waits the host or another
//! queue's command buffer met during the hang. Whether their work ran is a race, which the hang's
//! submission time decides most: where it read `Scheduled` within 2 ms their fills ran on the GPU
//! during the hang in 17 of 18 command buffers, and where later in 6 of 62, the rest running
//! nothing or in part (75,648 and 80,064 bytes of a 4 MiB fill); whether the queues were warmed,
//! and whether the command buffers were made before the hang's commit, decided neither. The
//! runtime answers the later submission's outcome: `Scheduled` [`SCHEDULED`] after the commit,
//! and nothing of the others run. One whose event wait was met only after the hang completed
//! ran, as did one of another queue and one committed after.
//!
//! A command buffer Metal's GPU faults at holds the same way, its timing and its queue's others
//! apart ([`Held::Fault`]): an indexed draw of type 9 (`CommandBufferError::PageFault`), an
//! indirect one (failed as a victim of the GPU's recovery, `GpuRecoveryVictim`), and a render
//! encoder's execution of indirect commands under a state that cannot inherit (`PageFault`), each
//! through `State::fault`. Measured on the M3 Max (macOS 27.0.1, 164 faulting command buffers on
//! fresh queues, 44 of them such executions): Metal completed it 6.86 to 10.3 ms after it read
//! `Scheduled`, which came as any command buffer's (0.5 to 1.5 ms after its commit, 6 to 12 ms
//! on a GPU idle before); the queue's command buffers committed meanwhile read `Scheduled` as
//! they were committed, ran, and completed after it. The encoders after the faulting one were
//! lost to the GPU's recovery where the GPU ran just before (127 of 142; every one of 34 right
//! after a hang), ran where it was idle before (the first fault after 3 s idle), and in some
//! shapes ran in part (an encoder after an event signal, a command buffer committed right
//! after); its event signals fired, 164 of 164.
//! The runtime answers the steady state: the faulting encoder and every one after it run nothing,
//! every event signal fires (`Failure::with_faults`); it reads `Scheduled` once its batch before
//! the hold is taken, and its wait goes [`FAULT_FROM_REACHED`] after that and the GPU reached the
//! encoder; the queue's command buffers committed meanwhile join its list, read `Scheduled`
//! early ([`holding`]), and run.
//!
//! Metal's GPU that faulted at an encoder goes on, once it recovered, to the later encoders that
//! fail too ([`Later`]): an indexed draw of type 9 then a direct one of 3 vertices, or another
//! indexed one. Measured on the M3 Max (macOS 27.0.1, 88 such command buffers, each pass its own
//! encoder, fresh processes and warm ones): the command buffer failed with the first fault's
//! error (`PageFault`, 88 of 88), every event signal firing. It read `Scheduled` 14.3 to 21.0 ms
//! after its commit, where one that faulted once read it 8.4 to 9.4 ms after and completed 6.1 to
//! 9.0 ms after that: once the GPU recovered from the fault. It completed 25.8 to 29.1 ms after
//! that where a hang followed the fault (56), 6.05 to 9.67 ms after it where a fault did (17), 12.7
//! to 16.5 ms where two did (5), and 31.65 to 34.97 ms where a fault then a hang did (5); past a
//! hang the GPU reached nothing (a fault after it added nothing, 27.0 to 27.2 ms, 5). The queue's
//! command buffer committed right after it ran nothing where a hang followed the fault (6 of 6),
//! and in 1 of 2 where a fault did. The encoders between the fault and the later failure ran in
//! that shape, as the ones after a single fault there did. A command buffer whose GPU hangs first
//! fails as a hang alone does (30 of 30: `GpuHang`, 27.0 to 28.1 ms after `Scheduled`, the fault
//! after the hang never reached). The runtime answers: `Scheduled` [`FAULT_FROM_REACHED`] after
//! the GPU reached the fault, its wait going [`FAULT_FROM_REACHED`] after that for each later
//! fault, then [`FROM_REACHED`] for a hang ([`Held::after_recovered`]); the queue's command
//! buffers committed meanwhile run nothing where a hang follows, as during a hang, and run where
//! faults alone do; and from the fault on nothing runs, as after a single fault.
//!
//! - **The hold.** The hanging encoder's end signals an event of the hang's own to 1 after the
//!   encoder, and adds the runtime's own wait ([`Batches::hold`]) for 2 after that, so what
//!   follows it in the command buffer (its later event signals, its completion) waits; its commit
//!   signals 3 after all its work. The signals and the wait are the runtime's, counted among no
//!   event command. The encoder and every one after it run nothing (`State::fail_from`).
//! - **`Scheduled`** [`SCHEDULED`] (the earliest Metal's read it) after the commit, not with its
//!   last batch, which the wait holds: the hang's thread makes it then, once the batch before the
//!   wait was taken, submitted or held in a group (`group.rs`), its submission handlers first
//!   ([`Scheduling::Hang`]). Where that batch is not taken within [`SIGNAL_BOUND`] (an event wait
//!   before the hang not met), or a switch of storage forms rides on its commit, which settles by
//!   its last batch's submission, it reads `Scheduled` with its last batch.
//! - **In flight** from its commit until its work signalled 3, the queue counting it
//!   (`CommandQueue::hangs`): a thread of the hang's own makes it `Scheduled`, waits for 1 (the
//!   GPU reached the hang), makes the queue's command buffers committed meanwhile `Scheduled`
//!   (below) until it lets the wait go ([`FROM_REACHED`] after it read `Scheduled` and the GPU
//!   reached the hang, the later of the two: the earliest Metal failed one after it read
//!   `Scheduled`), waits for 3, and only then stops counting
//!   it, inside a trail (`trails.rs`): a waiter of the command buffer returns once the queue no
//!   longer counts it. A command buffer that does not run its hang (its commit refused, a commit
//!   failure that fails the encoder or drops its signals, or itself discarded at its commit) sets
//!   3 at once.
//! - **The queue's others.** A command buffer committed on the queue while a hang of it is in
//!   flight, with no event wait of its own, runs nothing and completes with no error, its event
//!   signals firing, as `commit_failed` of code 0 does ([`discarded`]); it completes after the
//!   hang, in the queue's order. One with an event wait is listed with its waits ([`behind`]);
//!   the hang's thread decides each listed one just before it lets the hold go, from its events'
//!   values then: one whose every wait is met runs nothing so, as Metal ran nothing of one it
//!   scheduled during the hang, and one whose waits are not all met runs. Its first batch's
//!   submission applies the verdict (`checks.rs`), which no batch of it can reach before the hold
//!   goes: they follow the hang's last batch in the queue's order. One committed after the
//!   decision but while the hang is still counted is decided at that submission instead, by
//!   whether the hang still is.
//! - **Their `Scheduled`**, as Metal's: from the hang's on, as they are committed, each once its
//!   event waits are met. Each joins the hang's list at its commit ([`join`]); the hang's thread,
//!   once it made the hang `Scheduled`, makes each listed one `Scheduled` in the queue's order,
//!   once its waits are met, its submission handlers first and its kernel times then
//!   ([`QueueHangs::schedule_early`], [`Scheduling::Behind`]), until it lets the wait go. One
//!   whose waits are not met by then, and every one after it, read `Scheduled` with their last
//!   batch, after the hang's, as does every one where the hang's own came with its last batch, and
//!   one with a switch of storage forms riding on its commit.
//! - **The device's recovery.** As its wait goes, the hold raises the device's recovery from
//!   what it holds for: a hang's, and a fault's that a hang follows, from a hang; else from a
//!   fault. A fault's that later failures follow raised the fault's already, as it read
//!   `Scheduled`. A predicated loop running meanwhile ends there (`recovery.rs`).
//!
//! [`Batches::hold`]: super::batches::Batches::hold

use super::super::event::{EventShared, SharedEvent};
use super::super::exit_guard;
use super::super::gpu_time;
use super::super::queue::QueueGen;
use super::super::{Device, Error};
use super::recovery::Recovery;
use super::{lock, CommandBufferShared, CommitFailure, FailedEncoders, Item, State};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

/// When after its commit a command buffer its GPU hangs at reads `Scheduled`: 5.06 ms, the
/// earliest Metal's read it where its submission took 4.6 ms or more, whose queue then mostly ran
/// nothing of what it scheduled meanwhile (the module docs).
pub(super) const SCHEDULED: Duration = Duration::from_micros(5_060);

/// The earliest after it read `Scheduled` that Metal failed a command buffer its GPU hung at:
/// 27.5 ms (the module docs).
pub(super) const FROM_REACHED: Duration = Duration::from_micros(27_500);

/// The earliest after it read `Scheduled` that Metal completed a command buffer its GPU faulted
/// at, or failed as a victim of the GPU's recovery: 6.86 ms (the module docs).
pub(super) const FAULT_FROM_REACHED: Duration = Duration::from_micros(6_860);

/// How long the hang's thread waits for a signal of the GPU's (1 or 3) before it goes on as if it
/// came: a device that lost its work never signals.
const SIGNAL_BOUND: Duration = Duration::from_secs(10);

/// A queue's hangs: how many are in flight, the command buffers committed with an event wait
/// meanwhile, not yet decided, and those committed meanwhile, not yet `Scheduled` (the module
/// docs).
#[derive(Default)]
pub(crate) struct QueueHangs {
    in_flight: AtomicUsize,
    /// Its faults' holds in flight (the module docs): the queue's command buffers committed
    /// meanwhile read `Scheduled` early, and run.
    holding: AtomicUsize,
    behind: Mutex<Vec<Behind>>,
    early: Mutex<Early>,
    /// Signaled as a command buffer joins `early`.
    joined: Condvar,
}

/// The command buffers committed on a queue while a hang of it is in flight, in their order, for
/// the hang's thread to make `Scheduled` early (the module docs); each of the hang it joined
/// under, which the hang's start counts.
#[derive(Default)]
struct Early {
    hang: u64,
    joined: VecDeque<Joined>,
}

/// A command buffer that joined [`Early`]: the hang it joined under, its event waits, and what
/// makes it `Scheduled` ([`behind_scheduled`] of it, held weakly). Held as a closure: a queue's
/// hangs then name no command buffer's type, which would name the queue's again.
struct Joined {
    hang: u64,
    waits: Vec<(Arc<EventShared>, u64)>,
    schedule: Box<dyn FnOnce() + Send>,
}

/// A command buffer committed with event waits while a hang of its queue was in flight: its
/// waits, and its verdict.
struct Behind {
    waits: Vec<(Arc<EventShared>, u64)>,
    verdict: Verdict,
}

/// A command buffer's verdict behind a hang ([`behind`]): undecided, it runs, or it runs nothing.
#[derive(Clone, Default)]
pub(super) struct Verdict(Arc<AtomicU8>);

impl Verdict {
    const RUNS: u8 = 1;
    const DISCARDED: u8 = 2;

    /// Whether the command buffer runs nothing, as its first batch is about to be submitted on
    /// `queue`: as decided, or where undecided, whether a hang of the queue is in flight now.
    pub(super) fn discards(&self, queue: &QueueGen) -> bool {
        match self.0.load(Ordering::SeqCst) {
            Verdict::DISCARDED => true,
            Verdict::RUNS => false,
            _ => in_flight(queue),
        }
    }
}

impl QueueHangs {
    /// The count of holds in flight of `held`'s kind.
    fn counter(&self, held: Held) -> &AtomicUsize {
        match held.discards() {
            true => &self.in_flight,
            false => &self.holding,
        }
    }

    /// Until `until`, make the command buffers that joined [`Early`] under hang `hang`
    /// `Scheduled`, in their order, each once its event waits are met (the module docs). One
    /// whose waits are not met by then, and every one after it, read `Scheduled` with their last
    /// batch.
    fn schedule_early(&self, hang: u64, until: Instant) {
        loop {
            let next = {
                let mut early = lock(&self.early);
                loop {
                    if let Some(joined) = early.joined.pop_front() {
                        break Some(joined);
                    }
                    let left = until.saturating_duration_since(Instant::now());
                    if left.is_zero() || exit_guard::closed() {
                        break None;
                    }
                    early = (self.joined.wait_timeout(early, left))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
            };
            let Some(joined) = next else {
                return;
            };
            if joined.hang != hang {
                continue;
            }
            let met = (joined.waits.iter()).all(|(event, value)| {
                event.wait_until(*value, until.saturating_duration_since(Instant::now()))
            });
            if !met {
                sleep_until(until);
                return;
            }
            (joined.schedule)();
        }
    }

    /// Decide every listed command buffer from its events' values now: one whose every wait is
    /// met runs nothing.
    fn decide(&self) {
        let listed: Vec<Behind> = lock(&self.behind).drain(..).collect();
        for behind in listed {
            let met = behind
                .waits
                .iter()
                .all(|(event, value)| event.value() >= *value);
            let verdict = match met {
                true => Verdict::DISCARDED,
                false => Verdict::RUNS,
            };
            behind.verdict.0.store(verdict, Ordering::SeqCst);
        }
    }
}

/// What a command buffer's hold holds for (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Held {
    /// Metal's GPU hangs at its encoder (`CommandBufferError::GpuHang`).
    Hang,
    /// Metal's GPU faults at its encoder, or fails it as a victim of its recovery
    /// (`CommandBufferError::PageFault`, `GpuRecoveryVictim`), then at the later encoders
    /// [`Later`] counts, which it reaches once it recovered.
    Fault(Later),
}

/// The ended encoders after a faulting one that Metal's GPU fails too, which it reaches once it
/// recovered from the fault (the module docs): the faults (victims too) before the first hang,
/// and whether a hang follows them, past which it reaches none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Later {
    faults: u32,
    hang: bool,
}

impl Held {
    /// Whether the queue's command buffers committed meanwhile run nothing, as during a hang (a
    /// hang's hold, and a fault's that a hang follows), or run (a fault's alone, or followed by
    /// faults alone) (the module docs).
    fn discards(self) -> bool {
        matches!(self, Held::Hang | Held::Fault(Later { hang: true, .. }))
    }

    /// When after its commit the command buffer reads `Scheduled` at the earliest.
    fn scheduled(self) -> Duration {
        match self {
            Held::Hang => SCHEDULED,
            Held::Fault(_) => Duration::ZERO,
        }
    }

    /// How long after it read `Scheduled` and the GPU reached the encoder its wait goes.
    fn after_reached(self) -> Duration {
        match self {
            Held::Hang => FROM_REACHED,
            Held::Fault(_) => FAULT_FROM_REACHED,
        }
    }

    /// What the GPU recovers from as the wait goes: a hang's, and a fault's that a hang follows,
    /// from a hang; else from a fault (`recovery.rs`).
    fn recovery(self) -> Recovery {
        match self.discards() {
            true => Recovery::Hang,
            false => Recovery::Fault,
        }
    }

    /// A fault's that later failures follow: how long after it read `Scheduled`, once the GPU
    /// recovered from the fault, its wait goes: [`FAULT_FROM_REACHED`] for each later fault, then
    /// [`FROM_REACHED`] for a hang. `None` for a hang's, and a fault's alone.
    fn after_recovered(self) -> Option<Duration> {
        match self {
            Held::Fault(Later { faults, hang }) if faults > 0 || hang => {
                let hang = match hang {
                    true => FROM_REACHED,
                    false => Duration::ZERO,
                };
                Some(FAULT_FROM_REACHED * faults + hang)
            }
            _ => None,
        }
    }
}

/// A hanging or faulting command buffer's hold: the event of its signals and its wait.
pub(super) struct Hang {
    event: SharedEvent,
    /// The batch its signal of [`Hang::REACHED`] ends: the last before its wait.
    batch: usize,
    /// The ended encoder it holds after, the one Metal's GPU hangs or faults at.
    encoder: usize,
    held: Held,
}

/// A command buffer's transition to `Scheduled` around a hang of its queue (the module docs).
pub(super) enum Scheduling {
    /// The hold's own: its thread makes it once the batch before its wait (`batch`,
    /// [`Hang::batch`]) was taken, submitted or held in a group (`group.rs`), a hang's no sooner
    /// than [`SCHEDULED`] after the commit.
    Hang { batch: usize, taken: bool },
    /// One committed while a hang of its queue was in flight: the hang's thread makes it early,
    /// or its last batch does, whichever comes first.
    Behind { made: bool },
}

impl Hang {
    /// The value its signal after the hanging encoder sets: the GPU reached the hang.
    pub(super) const REACHED: u64 = 1;
    /// The value its wait waits for, which its thread sets.
    pub(super) const RELEASED: u64 = 2;
    /// The value its signal after all its work sets.
    pub(super) const DONE: u64 = 3;

    /// A hold of `device` for `held`, its event at 0, whose signal of [`Hang::REACHED`] ends
    /// batch `batch`, after ended encoder `encoder`.
    pub(super) fn new(
        device: &Device,
        (batch, encoder): (usize, usize),
        held: Held,
    ) -> Result<Hang, Error> {
        Ok(Hang {
            event: device.new_shared_event_now()?,
            batch,
            encoder,
            held,
        })
    }

    /// The ended encoder it holds after.
    pub(super) fn encoder(&self) -> usize {
        self.encoder
    }

    /// Metal's GPU fails a later ended encoder too, hanging there where `hangs`, else faulting:
    /// a fault's hold counts it, the GPU reaching it once it recovered, unless a hang came before
    /// it, past which the GPU reaches nothing (the module docs).
    pub(super) fn fails_later(&mut self, hangs: bool) {
        let Held::Fault(later) = &mut self.held else {
            return;
        };
        match (later.hang, hangs) {
            (true, _) => {}
            (false, true) => later.hang = true,
            (false, false) => later.faults += 1,
        }
    }

    /// The event of its signals and its wait.
    pub(super) fn event(&self) -> &SharedEvent {
        &self.event
    }

    /// The command buffer does not run its hang: its wait goes at once.
    pub(super) fn release(self) {
        self.event.set_signaled_value(Hang::DONE);
    }

    /// The command buffer `buffer`, of state `state`, committed at `committed`, runs its hold:
    /// its queue counts it until its work signalled [`Hang::DONE`] (a hang's, and a fault's that
    /// a hang follows, among those that discard the queue's others; another fault's among those
    /// that do not); it reads `Scheduled` [`SCHEDULED`] after the commit (a hang's) or at once (a
    /// fault's), or later once the batch before its wait was taken ([`Scheduling::Hang`]), and the
    /// queue's command buffers committed meanwhile follow ([`QueueHangs::schedule_early`]); its
    /// wait goes [`FROM_REACHED`] (a hang's) or [`FAULT_FROM_REACHED`] (a fault's) after it read
    /// `Scheduled` and the GPU reached the encoder. A fault's that later failures follow reads
    /// `Scheduled` [`FAULT_FROM_REACHED`] after the GPU reached the encoder, and its wait goes
    /// [`Held::after_recovered`] after that. One with a switch of storage forms riding on its
    /// commit, which settles by its last batch's submission, reads `Scheduled` with that batch,
    /// as do the queue's others. Where no thread can be made for that, the wait goes at once, the
    /// queue not counting it.
    pub(super) fn start(
        self,
        buffer: &Arc<CommandBufferShared>,
        state: &mut State,
        committed: Instant,
    ) {
        let queue = &buffer.queue;
        let hangs = Arc::clone(queue.hangs());
        let trails = Arc::clone(&queue.device().shared().trails);
        let recoveries = Arc::clone(&queue.device().shared().recoveries);
        let held = self.held;
        hangs.counter(held).fetch_add(1, Ordering::SeqCst);
        let hang = {
            let mut early = lock(&hangs.early);
            early.hang += 1;
            early.hang
        };
        if state.storage_form.is_none() {
            state.hang_scheduling = Some(Scheduling::Hang {
                batch: self.batch,
                taken: false,
            });
        }
        let weak = Arc::downgrade(buffer);
        let event = self.event.clone();
        let builder = std::thread::Builder::new().name("metal2vulkan-hang".to_owned());
        let ended = crate::exit_guard::spawn(builder, move || {
            sleep_until(committed + held.scheduled());
            let (scheduled, release) = match held.after_recovered() {
                // A fault that later failures follow reads `Scheduled` once the GPU recovered
                // from it, and its wait goes once the GPU recovered from them too.
                Some(after) => {
                    event.wait_until_signaled_value(Hang::REACHED, SIGNAL_BOUND);
                    std::thread::sleep(FAULT_FROM_REACHED);
                    recoveries.raise(Recovery::Fault);
                    (hang_scheduled(weak), Instant::now() + after)
                }
                None => {
                    let scheduled = hang_scheduled(weak);
                    event.wait_until_signaled_value(Hang::REACHED, SIGNAL_BOUND);
                    (scheduled, Instant::now() + held.after_reached())
                }
            };
            match scheduled {
                true => hangs.schedule_early(hang, release),
                false => sleep_until(release),
            }
            let trail = trails.open();
            if held.discards() {
                hangs.decide();
            }
            recoveries.raise(held.recovery());
            event.set_signaled_value(Hang::RELEASED);
            event.wait_until_signaled_value(Hang::DONE, SIGNAL_BOUND);
            lock(&hangs.early).joined.clear();
            hangs.counter(held).fetch_sub(1, Ordering::SeqCst);
            drop(event);
            drop(hangs);
            drop(trail);
        });
        if ended.is_err() {
            state.hang_scheduling = None;
            if held.discards() {
                queue.hangs().decide();
            }
            lock(&queue.hangs().early).joined.clear();
            queue.hangs().counter(held).fetch_sub(1, Ordering::SeqCst);
            self.event.set_signaled_value(Hang::DONE);
        }
    }
}

/// Sleep until `until`, if it is still ahead.
fn sleep_until(until: Instant) {
    if let Some(left) = until.checked_duration_since(Instant::now()) {
        std::thread::sleep(left);
    }
}

/// The hanging command buffer `buffer` holds, where something still holds it, came to its time to
/// read `Scheduled` ([`Scheduling::Hang`]): it does once the batch before its wait was taken, its
/// submission handlers first on this thread. Where that batch is not taken within
/// [`SIGNAL_BOUND`] (an event wait before the hang not met), its last batch makes it instead, and
/// none where a batch of it could not be submitted (`CommandBufferShared::failed`). Whether it was
/// made here.
fn hang_scheduled(buffer: Weak<CommandBufferShared>) -> bool {
    let Some(buffer) = buffer.upgrade() else {
        return false;
    };
    let deadline = Instant::now() + SIGNAL_BOUND;
    let made = {
        let mut state = lock(&buffer.state);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match state.hang_scheduling {
                Some(Scheduling::Hang { taken: true, .. }) => {
                    let handlers = !state.submission.is_empty();
                    state.submitting = handlers;
                    break Some(handlers);
                }
                Some(Scheduling::Hang { taken: false, .. })
                    if !left.is_zero() && !exit_guard::closed() => {}
                Some(Scheduling::Hang { taken: false, .. }) => {
                    state.hang_scheduling = None;
                    break None;
                }
                _ => break None,
            }
            state.parked += 1;
            state = (buffer
                .changed
                .wait_timeout(state, left.min(exit_guard::WAKE)))
            .unwrap_or_else(PoisonError::into_inner)
            .0;
            state.parked -= 1;
        }
    };
    if let Some(handlers) = made {
        transition(&buffer, handlers);
    }
    let_go(buffer);
    made.is_some()
}

/// The command buffer `buffer` holds, where something still holds it, committed while a hang of
/// its queue was in flight, reads `Scheduled` here, where its last batch has not made it
/// ([`Scheduling::Behind`]): its kernel times then, its submission handlers first on this
/// thread.
fn behind_scheduled(buffer: Weak<CommandBufferShared>) {
    let Some(buffer) = buffer.upgrade() else {
        return;
    };
    let made = {
        let mut state = lock(&buffer.state);
        match &mut state.hang_scheduling {
            Some(Scheduling::Behind { made }) if !*made => {
                *made = true;
                let handlers = !state.submission.is_empty();
                state.submitting = handlers;
                Some(handlers)
            }
            _ => None,
        }
    };
    if let Some(handlers) = made {
        buffer.kernel.begin();
        transition(&buffer, handlers);
    }
    let_go(buffer);
}

/// `buffer` becomes `Scheduled` here, its kernel end now: its transition, then its submission
/// handlers on this thread, which the transition waits for where `handlers` (the caller set
/// `submitting` under the state's lock).
fn transition(buffer: &Arc<CommandBufferShared>, handlers: bool) {
    buffer.kernel.end_at(gpu_time::host_nanos());
    buffer.schedule();
    if handlers {
        buffer.run_submission_handlers();
    }
}

/// Let go of `buffer` here, unless this is its last reference, which holds the device: that goes
/// in a context of its own (`Device::let_go_elsewhere`), so the hang's thread never runs the
/// device's teardown.
fn let_go(buffer: Arc<CommandBufferShared>) {
    if let Some(last) = Arc::into_inner(buffer) {
        let (contexts, context) = last.device.elsewhere();
        contexts.post(context, Item::Release(Box::new(last)));
    }
}

/// Whether batch `index` of a command buffer, its last where `last`, makes it `Scheduled` as it
/// is submitted or held in a group ([`Scheduling`]): its last, but where its hang's thread makes
/// it or the hang's thread made it early; and whether it is the hang's batch before its wait,
/// taken now, which the hang's thread waits for (on the command buffer's `changed`).
pub(super) fn schedules(
    scheduling: &mut Option<Scheduling>,
    index: usize,
    last: bool,
) -> (bool, bool) {
    match scheduling {
        None => (last, false),
        Some(Scheduling::Hang { batch, taken }) => {
            let took = index == *batch && !*taken;
            *taken |= took;
            (false, took)
        }
        Some(Scheduling::Behind { made }) => {
            let makes = last && !*made;
            *made |= makes;
            (makes, false)
        }
    }
}

/// The command buffer `buffer`, of state `state`, committed on its queue while a hang of it is in
/// flight with the event waits `waits`, joins the hang's command buffers to make `Scheduled`
/// early ([`QueueHangs::schedule_early`]); not one with a switch of storage forms riding on its
/// commit, which settles by its last batch's submission.
pub(super) fn join(
    buffer: &Arc<CommandBufferShared>,
    state: &mut State,
    waits: Vec<(Arc<EventShared>, u64)>,
) {
    if state.storage_form.is_some() {
        return;
    }
    state.hang_scheduling = Some(Scheduling::Behind { made: false });
    let hangs = buffer.queue.hangs();
    let mut early = lock(&hangs.early);
    let hang = early.hang;
    let buffer = Arc::downgrade(buffer);
    early.joined.push_back(Joined {
        hang,
        waits,
        schedule: Box::new(move || behind_scheduled(buffer)),
    });
    drop(early);
    hangs.joined.notify_all();
}

/// Whether a hang of `queue` is in flight (the module docs).
pub(super) fn in_flight(queue: &QueueGen) -> bool {
    queue.hangs().in_flight.load(Ordering::SeqCst) > 0
}

/// Whether a fault's hold of `queue` is in flight (the module docs): a command buffer committed
/// on it now reads `Scheduled` early, and runs.
pub(super) fn holding(queue: &QueueGen) -> bool {
    queue.hangs().holding.load(Ordering::SeqCst) > 0
}

/// The verdict of a command buffer committed on `queue` with the event waits `waits` gives,
/// listed for the hang's decision where a hang of the queue is in flight; `None` where none is
/// (the module docs).
pub(super) fn behind(queue: &QueueGen, waits: &[(Arc<EventShared>, u64)]) -> Option<Verdict> {
    if !in_flight(queue) {
        return None;
    }
    let verdict = Verdict::default();
    lock(&queue.hangs().behind).push(Behind {
        waits: waits.to_vec(),
        verdict: verdict.clone(),
    });
    Some(verdict)
}

/// How a command buffer committed on `queue` with `event_waits` event waits of its own fails
/// where a hang of the queue is in flight: it runs nothing and completes with no error, its
/// signals firing (the module docs). `None` where none is, or where it waits for an event, which
/// its first batch's submission decides.
pub(super) fn discarded(queue: &QueueGen, event_waits: usize) -> Option<CommitFailure> {
    (in_flight(queue) && event_waits == 0).then_some(CommitFailure {
        code: 0,
        encoders: FailedEncoders::From(0),
        signals: true,
    })
}
