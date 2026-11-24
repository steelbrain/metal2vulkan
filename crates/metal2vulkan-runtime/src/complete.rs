//! The completion thread: it retires command buffers and uploads as the GPU completes them, runs
//! their completed handlers, follows the event signals it submitted and runs listeners, and ends
//! the work in flight at a loss.

use crate::capture::CopyIn;
use crate::command::{self, CbInner, CommandBufferError, DeviceLoss, Gpu, Handlers, Pending};
use crate::counters::{Metric, HANDLER_STEPS};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::event::{self, EventInner, Listener};
use crate::fault;
use crate::submit::{Ended, Finished, Upload};
use crate::times::{self, Calibration, OwnEnd, Spans};
use ash::vk;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// How long [`Done::wait`] waits on the GPU at a time before it checks for a loss.
const LOSS_POLL: Duration = Duration::from_millis(100);

/// The completion side: what the completion thread has to do, and its two timeline semaphores.
pub(crate) struct Done {
    state: Mutex<DoneState>,
    /// Wakes the completion thread while it waits on the host.
    changed: Condvar,
    /// Each batch signals its value when it completes.
    pub(crate) progress: vk::Semaphore,
    /// The host signals it to wake the completion thread from a GPU wait.
    wake: vk::Semaphore,
    /// The highest progress value the completion thread has seen reached, or, at a loss, that
    /// the lost device's idle wait covered.
    completed: AtomicU64,
    /// For the device tests: the step of its round the completion thread last entered
    /// ([`Done::enter`], [`Done::stall`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    step: Mutex<String>,
}

struct DoneState {
    /// Work submitted, by progress value: each retires once the progress semaphore reaches it.
    flight: VecDeque<Retiree>,
    /// Command buffers that ended without anything in flight: they retire at once.
    ended: Vec<Retiree>,
    /// Events with submitted signals the completion thread has not yet seen.
    watched: Vec<Arc<EventInner>>,
    /// Events whose listeners a host value may reach.
    fire: Vec<Arc<EventInner>>,
    waiting: Waiting,
    /// The wake semaphore's last value.
    woken: u64,
    stop: bool,
    lost: bool,
    /// For the device tests: the values the completion thread's last GPU wait waits for, in its
    /// order: the progress semaphore's where it waits for it, each watched event's next pending
    /// value, then the wake semaphore's ([`Done::stall`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    waits_for: Vec<u64>,
}

/// Where the completion thread waits.
#[derive(Clone, Copy)]
enum Waiting {
    /// It does not: it reads the state before it next waits.
    Not,
    /// On the host, for a change of the state.
    Host,
    /// On the GPU; `progress` where it waits for the progress semaphore too, so that more work in
    /// flight needs no wake.
    Gpu { progress: bool },
}

struct Retiree {
    value: u64,
    /// Of a held group's submission ([`Sent::held`]): its last progress value, which its members
    /// share where it goes member by member, each with a value of its own.
    held: Option<u64>,
    /// In flight at a loss ([`Retiree::lost`]): nothing of it is read.
    lost: bool,
    item: Item,
}

/// A submission passed to `vkQueueSubmit2`: its progress value, and whether it is a group its
/// queue held without room that goes because a completion left room (`Metrics::held_starts`).
pub(crate) struct Sent {
    pub(crate) value: u64,
    pub(crate) held: bool,
}

enum Item {
    Command {
        cb: Arc<CbInner>,
        gpu: Gpu,
        handlers: Handlers,
        error: Option<CommandBufferError>,
        /// How many of its two queries a submission wrote.
        written: u32,
    },
    Upload(Upload),
    /// A Managed buffer's copy-in the submission took (`crate::capture`).
    CopyIn(CopyIn),
}

impl Done {
    /// Waits until the progress semaphore reaches `value`, a value [`Work::flush`] gave: once
    /// what that batch carries has completed.
    ///
    /// Err: [`Error::Vulkan`] (`VK_ERROR_DEVICE_LOST`) where the device is lost first.
    ///
    /// [`Work::flush`]: crate::submit::Work::flush
    pub(crate) fn wait(&self, core: &Core, value: u64) -> Result<(), Error> {
        let lost = Error::Vulkan {
            call: "vkWaitSemaphores",
            result: vk::Result::ERROR_DEVICE_LOST,
        };
        let semaphores = [self.progress];
        let values = [value];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        loop {
            // SAFETY: the progress semaphore is the device's, live while the core is.
            match unsafe {
                core.device
                    .wait_semaphores(&wait, LOSS_POLL.as_nanos() as u64)
            } {
                Ok(()) => return Ok(()),
                // A lost device's pending batches never signal: the work side ends them.
                Err(vk::Result::TIMEOUT) if core.work.lock().lost() => return Err(lost),
                Err(vk::Result::TIMEOUT) => {}
                Err(result) => {
                    return Err(Error::Vulkan {
                        call: "vkWaitSemaphores",
                        result,
                    })
                }
            }
        }
    }

    /// Makes the two semaphores.
    ///
    /// # Safety
    ///
    /// `device` is live.
    pub(crate) unsafe fn new(device: &ash::Device) -> Result<Done, Error> {
        let progress = event::timeline(device)?;
        let wake = event::timeline(device).inspect_err(|_| {
            // SAFETY: the semaphore is unused.
            unsafe { device.destroy_semaphore(progress, None) };
        })?;
        Ok(Done {
            state: Mutex::new(DoneState {
                flight: VecDeque::new(),
                ended: Vec::new(),
                watched: Vec::new(),
                fire: Vec::new(),
                waiting: Waiting::Not,
                woken: 0,
                stop: false,
                lost: false,
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                waits_for: Vec::new(),
            }),
            changed: Condvar::new(),
            progress,
            wake,
            completed: AtomicU64::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            step: Mutex::new("none yet".to_owned()),
        })
    }

    /// Destroys the semaphores.
    ///
    /// # Safety
    ///
    /// No pending work names them.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: the caller's contract.
        unsafe {
            device.destroy_semaphore(self.progress, None);
            device.destroy_semaphore(self.wake, None);
        }
    }

    fn lock(&self) -> MutexGuard<'_, DoneState> {
        lock(&self.state)
    }

    /// The highest progress value the completion thread has seen reached, or, at a loss, that
    /// the lost device's idle wait covered.
    pub(crate) fn completed(&self) -> u64 {
        self.completed.load(Ordering::SeqCst)
    }

    /// For the device tests: the completion thread enters `step` of its round, which a timed-out
    /// wait names ([`Done::stall`]) where the thread is not in a wait of its own.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn enter(&self, step: String) {
        *lock(&self.step) = step;
    }

    /// For the device tests: what the completion side holds, read at once, for a wait that timed
    /// out. The command buffers in flight (submitted and not retired) among its retirees; the
    /// oldest retiree's progress value, the progress semaphore's counter, and whether a wait of
    /// timeout 0 on that value reports it reached (the oldest's fence: signalled or not); and
    /// where the completion thread waits, with the values of its GPU wait and the wake
    /// semaphore's counter, or, in no wait of its own, the step of its round it last entered
    /// ([`Done::enter`]). Its lock is tried for a second: one held longer is named, not waited
    /// for.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn stall(&self, device: &ash::Device) -> String {
        use std::sync::TryLockError;
        let deadline = Instant::now() + Duration::from_secs(1);
        let state = loop {
            match self.state.try_lock() {
                Ok(state) => break state,
                Err(TryLockError::Poisoned(poisoned)) => break poisoned.into_inner(),
                Err(TryLockError::WouldBlock) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(TryLockError::WouldBlock) => {
                    return "the completion side's lock held by another thread for a second"
                        .to_owned();
                }
            }
        };
        // SAFETY: both are timeline semaphores of the device.
        let counter = |semaphore: vk::Semaphore| match unsafe {
            device.get_semaphore_counter_value(semaphore)
        } {
            Ok(value) => value.to_string(),
            Err(result) => format!("unread ({result:?})"),
        };
        let commands = (state.flight.iter())
            .filter(|retiree| matches!(retiree.item, Item::Command { .. }))
            .count();
        let oldest = match state.flight.front() {
            None => "nothing in flight".to_owned(),
            Some(oldest) => {
                let kind = match oldest.item {
                    Item::Command { .. } => "a command buffer",
                    Item::Upload(_) => "an upload",
                    Item::CopyIn(_) => "a copy-in",
                };
                let reached = match reached_by_wait(device, self.progress, oldest.value) {
                    Ok(reached) => reached.to_string(),
                    Err(_) => "no: the wait failed, a loss".to_owned(),
                };
                format!(
                    "the oldest, {kind}, at progress {}: the progress semaphore's counter {}, a \
                     wait of timeout 0 reports it reached: {reached}",
                    oldest.value,
                    counter(self.progress)
                )
            }
        };
        let waiting = match state.waiting {
            Waiting::Not => format!(
                "in no wait of its own: in a round, its step last entered {}",
                lock(&self.step)
            ),
            Waiting::Host => "on the host, for a change of the completion side's state".to_owned(),
            Waiting::Gpu { progress } => format!(
                "on the GPU, for any of the values {:?} ({} the progress semaphore's first, then \
                 the watched events', the wake semaphore's last), the wake semaphore's counter {}",
                state.waits_for,
                match progress {
                    true => "with",
                    false => "without",
                },
                counter(self.wake)
            ),
        };
        format!(
            "in flight {commands} command buffers among {} retirees, {} ended with nothing in \
             flight; {oldest}; the completion thread, which saw progress {} reached, waits \
             {waiting}; watched events {}, lost {}, stopping {}",
            state.flight.len(),
            state.ended.len(),
            self.completed(),
            state.watched.len(),
            state.lost,
            state.stop
        )
    }

    /// Wakes the completion thread where it waits. With `flight_only` (the change is only more
    /// work in flight), a thread that already waits for the progress semaphore is left to wait.
    fn wake(&self, state: &mut DoneState, device: &ash::Device, flight_only: bool) {
        match state.waiting {
            Waiting::Not => return,
            Waiting::Gpu { progress: true } if flight_only => return,
            Waiting::Host => self.changed.notify_one(),
            Waiting::Gpu { .. } => {
                state.woken += 1;
                let signal = vk::SemaphoreSignalInfo::default()
                    .semaphore(self.wake)
                    .value(state.woken);
                // SAFETY: only the host signals the semaphore, under this lock, each time higher.
                // A failure is a loss, which the thread's wait then reports.
                let _ = unsafe { device.signal_semaphore(&signal) };
            }
        }
        state.waiting = Waiting::Not;
    }

    /// From the submit thread: a submission passed to `vkQueueSubmit2` (`sent`). Its command
    /// buffers retire in order, each once its own progress value is reached (the submission's,
    /// or where it goes member by member its part's), each ended one with its error; its uploads
    /// and the copy-ins it took once the submission completes.
    pub(crate) fn submitted(
        &self,
        core: &Core,
        sent: Sent,
        uploads: Vec<Upload>,
        copy_ins: Vec<CopyIn>,
        finished: Vec<Finished>,
        watch: Vec<Arc<EventInner>>,
    ) {
        let Sent { value, held } = sent;
        let held = held.then_some(value);
        let mut state = self.lock();
        let flight_only = watch.is_empty();
        for upload in uploads {
            state.flight.push_back(Retiree {
                value,
                held,
                lost: false,
                item: Item::Upload(upload),
            });
        }
        for copy_in in copy_ins {
            state.flight.push_back(Retiree {
                value,
                held,
                lost: false,
                item: Item::CopyIn(copy_in),
            });
        }
        for (cb, pending, error) in finished {
            let written = pending.stamps_written();
            let Pending {
                value,
                gpu,
                handlers,
                ..
            } = pending;
            let at = state.flight.partition_point(|other| other.value <= value);
            state.flight.insert(
                at,
                Retiree {
                    value,
                    held,
                    lost: false,
                    item: Item::Command {
                        cb,
                        gpu,
                        handlers,
                        error,
                        written,
                    },
                },
            );
        }
        state.watched.extend(watch);
        self.wake(&mut state, &core.device, flight_only);
    }

    /// Command buffers that retire with no submission of their own, each with its error where
    /// it ended: each retires once what it had in flight completes, at once where it had nothing
    /// in flight.
    pub(crate) fn end_all(&self, core: &Core, ended: Vec<Ended>) {
        if ended.is_empty() {
            return;
        }
        let mut state = self.lock();
        for (cb, pending, error) in ended {
            let written = pending.stamps_written();
            let Pending {
                value,
                gpu,
                handlers,
                ..
            } = pending;
            let retiree = Retiree {
                value,
                held: None,
                lost: false,
                item: Item::Command {
                    cb,
                    gpu,
                    handlers,
                    error,
                    written,
                },
            };
            if value == 0 {
                state.ended.push(retiree);
            } else {
                let at = state.flight.partition_point(|other| other.value <= value);
                state.flight.insert(at, retiree);
            }
        }
        self.wake(&mut state, &core.device, false);
    }

    /// A host value of `event` may reach some of its listeners.
    pub(crate) fn fire(&self, core: &Core, event: Arc<EventInner>) {
        let mut state = self.lock();
        state.fire.push(event);
        self.wake(&mut state, &core.device, false);
    }

    /// Tells the completion thread to end once nothing is in flight. The submit thread tells it
    /// as it ends, after the last command buffers it ended came here, so none comes after the
    /// thread. The thread ends then whatever a stop's deadline was: one past a quiesce's deadline
    /// ends alone, once the GPU completed what was in flight and it retired that, so nothing in
    /// flight is left holding what it used, the core among it; the teardown joins it.
    pub(crate) fn stop(&self, core: &Core) {
        let mut state = self.lock();
        state.stop = true;
        self.wake(&mut state, &core.device, false);
    }
}

/// The device is lost, or a submission failed, from `loss`: the submit thread ends what it holds
/// and every later commit, and the completion thread ends what is in flight once the device is
/// idle, without waiting on the GPU again.
pub(crate) fn lose(core: &Core, loss: DeviceLoss) {
    core.work.lose(loss);
    let mut state = core.done.lock();
    state.lost = true;
    core.done.wake(&mut state, &core.device, false);
}

impl Retiree {
    /// Retires it, a command buffer by its `completion`; an upload or a copy-in with its
    /// timestamps read where it completed (`times::retire_own`), which it answers where both
    /// read.
    fn retire(self, core: &Core, completion: Completion<'_>) -> Option<(u64, u64)> {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        core.done.enter(format!(
            "retiring {} of progress {}{}",
            match self.item {
                Item::Command { .. } => "a command buffer",
                Item::Upload(_) => "an upload",
                Item::CopyIn(_) => "a copy-in",
            },
            self.value,
            if self.lost { ", lost" } else { "" }
        ));
        let end = match self.lost {
            true => OwnEnd::Lost,
            false => OwnEnd::Completed,
        };
        match self.item {
            Item::Command {
                cb,
                gpu,
                handlers,
                error,
                written,
            } => {
                command::retire(core, cb, gpu, handlers, error, written, completion);
                None
            }
            Item::Upload(upload) => upload.retire(core, end),
            Item::CopyIn(copy_in) => copy_in.retire(core, end),
        }
    }

    /// A completed command buffer's two timestamps, where it recorded both: the read does not
    /// wait (`readback::timestamps`), so a pair a cancelled submission never wrote reads as none.
    fn stamps(&self, core: &Core) -> Option<(u64, u64)> {
        match &self.item {
            Item::Command {
                cb,
                error: None,
                written: 2,
                ..
            } => cb.stamps(core),
            _ => None,
        }
    }

    /// In flight at a loss.
    fn lost(mut self) -> Retiree {
        self.lost = true;
        if let Item::Command { error, .. } = &mut self.item {
            error.get_or_insert(CommandBufferError::DeviceLost { in_flight: true });
        }
        self
    }
}

/// The host times of a round of the completion thread's steps, which split each of its command
/// buffers' spans from their GPU end to their handlers (`Metrics::gpu_end_to_handler_steps_nanos`).
pub(crate) struct Round {
    /// The thread saw the round's completions.
    seen: u64,
    /// Its read of the round's timestamps began.
    read: u64,
    /// That read ended.
    stamped: u64,
    /// Its calibration ended, where the round read one; else the read's end.
    calibrated: u64,
}

impl Round {
    /// A round that saw its completions at `seen`, read its timestamps from `read` to `stamped`,
    /// and ended its calibration at `calibrated` (`stamped` where it read none).
    pub(crate) fn at(seen: u64, read: u64, stamped: u64, calibrated: u64) -> Round {
        Round {
            seen,
            read,
            stamped,
            calibrated,
        }
    }

    /// The steps of a command buffer's span from its GPU end, `end`, to its handlers, which
    /// begin at `handled`, its retirement having begun at `retiring`: each from the earlier of
    /// its two host times, 0 where the later comes first.
    pub(crate) fn steps(&self, end: u64, retiring: u64, handled: u64) -> [u64; HANDLER_STEPS] {
        [
            self.seen.saturating_sub(end),
            self.read.saturating_sub(self.seen),
            self.stamped.saturating_sub(self.read),
            self.calibrated.saturating_sub(self.stamped),
            retiring.saturating_sub(self.calibrated),
            handled.saturating_sub(retiring),
        ]
    }
}

/// What the completion thread knows of a command buffer's completion as it retires it: its GPU
/// times where they were read, its round, and the GPU spans retired before it.
pub(crate) struct Completion<'a> {
    pub(crate) gpu_times: Option<(u64, u64)>,
    pub(crate) round: &'a Round,
    pub(crate) spans: &'a Spans,
}

/// What a round of the completion thread saw of the progress semaphore.
enum Seen {
    /// A host wait reported this value reached (0: nothing new).
    Reached(u64),
    /// The counter read this value, which a host wait then reported not reached.
    Behind(u64),
}

/// Whether a host wait on `semaphore` for `value`, with a timeout of 0, reports it reached.
///
/// Err: the loss a failed wait names.
pub(crate) fn reached_by_wait(
    device: &ash::Device,
    semaphore: vk::Semaphore,
    value: u64,
) -> Result<bool, DeviceLoss> {
    let semaphores = [semaphore];
    let values = [value];
    let wait = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    // SAFETY: the semaphore is a timeline semaphore of the device.
    match unsafe { device.wait_semaphores(&wait, 0) } {
        Ok(()) => Ok(true),
        Err(vk::Result::TIMEOUT) => Ok(false),
        Err(result) => Err(DeviceLoss::of("vkWaitSemaphores")(result)),
    }
}

/// The completion thread, until it is stopped.
pub(crate) fn run(core: Arc<Core>) {
    let done = &core.done;
    let device = &core.device;
    // The GPU clock on the host's: read as the device starts, and again at a completion once the
    // period has passed since the last reading.
    let mut calibration = Calibration::read(&core);
    // The latest end timestamp of a command buffer retired, and the value of the last held
    // group's submission whose gap was counted (`Metrics::held_start_gap_nanos`).
    let mut last_end: Option<u64> = None;
    let mut gapped = 0;
    // The latest GPU spans retired, on the host clock, under which a command buffer's span from
    // its scheduling to its GPU start may lie (`Metrics::scheduled_to_gpu_start_behind_nanos`).
    let mut spans = Spans::new();
    // A value the counter read and the host wait then reported not reached: the next wait is for
    // it, so the thread sleeps until the wait agrees rather than reading the counter again.
    let mut behind: Option<u64> = None;
    let mut state = done.lock();
    loop {
        let mut retire = std::mem::take(&mut state.ended);
        let fired = std::mem::take(&mut state.fire);
        let mut fire: Vec<(Listener, u64)> = Vec::new();
        let mut newly_lost = None;
        // The host time the thread saw this round's completions, which no GPU time of theirs
        // passes (`Calibration::place`).
        let mut seen_at = None;
        if state.lost {
            // The signals pending never come: the host waits for them end first.
            for event in state.watched.drain(..) {
                event.lose();
            }
            if let Some(last) = state.flight.back().map(|retiree| retiree.value) {
                // Nothing is pending once the device is idle, which a lost device reaches at once:
                // what was in flight before the idle wait no longer uses what it held, a lost
                // device's wait answering as its success does for that. Its values are marked
                // completed before it retires, as the GPU's are, so what waits for them goes
                // (`memory::Wait`). One submitted during the wait stays in flight for the next
                // round's own idle wait.
                state.waiting = Waiting::Not;
                drop(state);
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                done.enter("the lost device's idle wait (vkDeviceWaitIdle)".to_owned());
                core.wait_idle();
                state = done.lock();
                done.completed.fetch_max(last, Ordering::SeqCst);
                let idle = state
                    .flight
                    .partition_point(|retiree| retiree.value <= last);
                retire.extend(state.flight.drain(..idle).map(Retiree::lost));
            }
        } else {
            if let (Some(front), Some(back)) = (state.flight.front(), state.flight.back()) {
                // A retirement tells the guest its command buffer completed and lets it read
                // what the GPU wrote, which Vulkan orders after the signal only through a host
                // wait on it ("signal operations happening before host operations that wait on
                // them"); the counter read states no such order. So nothing retires until a wait
                // on its value, with a timeout of 0, says it is reached. With one value pending,
                // that wait is the one call; with more, the counter read finds the highest value
                // reached and the wait confirms it.
                let (front, back) = (front.value, back.value);
                let seen = match front == back {
                    true => reached_by_wait(device, done.progress, front)
                        .map(|by| Seen::Reached(if by { front } else { 0 })),
                    // SAFETY: the progress semaphore is the device's.
                    false => unsafe { device.get_semaphore_counter_value(done.progress) }
                        .map_err(DeviceLoss::of("vkGetSemaphoreCounterValue"))
                        .and_then(|read| match read >= front {
                            true => reached_by_wait(device, done.progress, read).map(|by| {
                                if by {
                                    Seen::Reached(read)
                                } else {
                                    Seen::Behind(read)
                                }
                            }),
                            false => Ok(Seen::Reached(read)),
                        }),
                };
                match seen {
                    Ok(Seen::Reached(reached)) => {
                        seen_at = Some(times::host_time());
                        behind = None;
                        // Before the submit thread's mark is read (`Work::completed`).
                        done.completed.fetch_max(reached, Ordering::SeqCst);
                        while state
                            .flight
                            .front()
                            .is_some_and(|retiree| retiree.value <= reached)
                        {
                            retire.extend(state.flight.pop_front());
                        }
                    }
                    Ok(Seen::Behind(read)) => {
                        // Nothing retires this round: the thread waits for the value again.
                        behind = Some(read);
                        core.counters.count(Metric::WaitsBehindCounter);
                    }
                    Err(loss) => {
                        state.lost = true;
                        newly_lost = Some(loss);
                    }
                }
            }
            let mut index = 0;
            while index < state.watched.len() {
                let event = &state.watched[index];
                if event.observe(event.counter().ok(), &mut fire) {
                    index += 1;
                } else {
                    state.watched.swap_remove(index);
                }
            }
        }
        for event in &fired {
            event.observe(None, &mut fire);
        }
        if newly_lost.is_some() || !retire.is_empty() || !fire.is_empty() {
            state.waiting = Waiting::Not;
            drop(state);
            if let Some(loss) = newly_lost {
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                done.enter(format!("the loss's fault report and work.lose ({loss:?})"));
                core.work.lose(fault::report(&core, loss));
            }
            // `completed` was raised before the submit thread's mark is read (`Work::completed`):
            // what it holds goes as the completion left room. The round sees no completion until
            // it ends, so held work waits for its end.
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            done.enter("work.completed".to_owned());
            core.work.completed();
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            done.enter("pool.retired".to_owned());
            core.pool.retired();
            // Handlers and listeners run with no lock held.
            // Each completed command buffer's timestamps are placed on the host clock by the
            // last calibration, read again first where the period has passed since it, and moved
            // earlier where it would place an end after the host time the thread saw it complete.
            // The round's steps are timed by clock reads around them (`Round`): two, and a third
            // where it calibrates.
            let seen = seen_at.unwrap_or_else(times::host_time);
            let read = times::host_time();
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            done.enter(format!("reading {} retirees' timestamps", retire.len()));
            let stamps: Vec<Option<(u64, u64)>> =
                retire.iter().map(|retiree| retiree.stamps(&core)).collect();
            let stamped = times::host_time();
            let mut calibrated = stamped;
            if stamps.iter().any(Option::is_some)
                && calibration.is_none_or(|last| {
                    stamped.saturating_sub(last.host()) >= times::CALIBRATION_PERIOD
                })
            {
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                done.enter("reading the GPU clock's calibration".to_owned());
                calibration = Calibration::read(&core).or(calibration);
                calibrated = times::host_time();
            }
            let round = Round::at(seen, read, stamped, calibrated);
            for (retiree, stamps) in retire.into_iter().zip(stamps) {
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                if let Some((start, end)) = stamps {
                    lock(&core.tick_spans).push(core.ticks.span(start, end));
                }
                if let Some((start, end)) = stamps {
                    // Retirees come in submission order: the first of a held group's command
                    // buffers with both timestamps starts the GPU's time behind it, from the
                    // latest end before it where the start is the later.
                    if let Some(group) = retiree.held.filter(|&group| group != gapped) {
                        gapped = group;
                        if let Some(last) = last_end {
                            core.counters.held_start_gap(core.ticks.span(last, start));
                        }
                    }
                    if last_end.is_none_or(|last| core.ticks.span(last, end) > 0) {
                        last_end = Some(end);
                    }
                }
                let times = stamps
                    .zip(calibration.as_mut())
                    .map(|(stamps, calibration)| calibration.place(&core.ticks, stamps, seen));
                let completion = Completion {
                    gpu_times: times,
                    round: &round,
                    spans: &spans,
                };
                // Its GPU span joins those the later ones' spans lie under: a command buffer's as
                // placed, an upload's or a copy-in's by the calibration as it stands.
                let own = retiree.retire(&core, completion);
                let own = own
                    .zip(calibration.as_ref())
                    .map(|((start, end), calibration)| {
                        (
                            calibration.at(&core.ticks, start),
                            calibration.at(&core.ticks, end),
                        )
                    });
                if let Some(span) = own.or(times) {
                    spans.push(span);
                }
            }
            // The GPU's writes before the signals the listeners see are the host's first: the
            // Shared memory the command buffers in flight may write is settled
            // (`crate::coherence`).
            if let Some(coherence) = &core.coherence {
                if !fire.is_empty() {
                    coherence.signal_seen(&core);
                }
            }
            for (listener, value) in fire {
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                done.enter(format!("a listener of the value {value}"));
                listener(value);
            }
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            done.enter("dropping the round's fired events".to_owned());
            drop(fired);
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            done.enter("taking the completion side's lock at the round's end".to_owned());
            state = done.lock();
            continue;
        }
        drop(fired);
        if state.stop && state.flight.is_empty() {
            break;
        }
        if state.lost || (state.flight.is_empty() && state.watched.is_empty()) {
            state.waiting = Waiting::Host;
            state = done
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
            state.waiting = Waiting::Not;
            continue;
        }
        let mut semaphores = Vec::new();
        let mut values = Vec::new();
        let target = (state.flight.front()).map(|retiree| retiree.value.max(behind.unwrap_or(0)));
        if let Some(value) = target {
            semaphores.push(done.progress);
            values.push(value);
        }
        for event in &state.watched {
            if let Some(value) = event.next_pending() {
                semaphores.push(event.semaphore);
                values.push(value);
            }
        }
        semaphores.push(done.wake);
        values.push(state.woken + 1);
        state.waiting = Waiting::Gpu {
            progress: target.is_some(),
        };
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        state.waits_for.clone_from(&values);
        drop(state);
        let wait = vk::SemaphoreWaitInfo::default()
            .flags(vk::SemaphoreWaitFlags::ANY)
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: every semaphore is a timeline semaphore of the device, live while the state
        // holds its event or the core lives.
        let result = unsafe { device.wait_semaphores(&wait, u64::MAX) };
        state = done.lock();
        state.waiting = Waiting::Not;
        match result {
            Err(vk::Result::TIMEOUT) | Ok(()) => {}
            Err(result) => {
                state.lost = true;
                drop(state);
                core.work.lose(fault::report(
                    &core,
                    DeviceLoss::of("vkWaitSemaphores")(result),
                ));
                state = done.lock();
            }
        }
    }
    let watched = std::mem::take(&mut state.watched);
    let fired = std::mem::take(&mut state.fire);
    drop(state);
    drop((watched, fired));
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl Core {
    /// Waits, up to `deadline`, for the batch of progress `value` and every one before it. True
    /// once they completed, or the device is lost and runs nothing more; false at the deadline.
    pub(crate) fn wait_progress(&self, value: u64, deadline: Option<Instant>) -> bool {
        let semaphores = [self.done.progress];
        let values = [value];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        let nanos = deadline.map_or(u64::MAX, |deadline| {
            nanos(deadline.saturating_duration_since(Instant::now()))
        });
        // SAFETY: the progress semaphore is the device's.
        let result = unsafe { self.device.wait_semaphores(&wait, nanos) };
        result != Err(vk::Result::TIMEOUT)
    }
}
