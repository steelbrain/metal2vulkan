//! The completion thread: it retires command buffers and uploads as the GPU completes them, runs
//! their completed handlers, follows the event signals it submitted and runs listeners, schedules
//! the command buffers held at `Committed` for a shared event's value once it is reached, and ends
//! the work in flight at a loss.

use crate::command::{self, CbInner, CommandBufferError, Gpu, Handler, Handlers, Pending};
use crate::counters::Metric;
use crate::device::{lock, Core};
use crate::error::Error;
use crate::event::{self, EventInner, Listener};
use crate::queue::QueueInner;
use crate::submit::{Ended, Finished, Upload};
use crate::times::{self, Calibration};
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
    /// The highest progress value the completion thread has seen reached.
    completed: AtomicU64,
}

struct DoneState {
    /// Work submitted, by progress value: each retires once the progress semaphore reaches it.
    flight: VecDeque<Retiree>,
    /// Command buffers that ended without anything in flight: they retire at once.
    ended: Vec<Retiree>,
    /// Events with submitted signals the completion thread has not yet seen.
    watched: Vec<Arc<EventInner>>,
    /// Events whose listeners, or whose held command buffers' waits, a host value may reach.
    fire: Vec<Arc<EventInner>>,
    /// Submitted command buffers held at `Committed` (`Pending::held`), in their submission order:
    /// each scheduled once its waits are reached and its queue's held ones before it are
    /// scheduled, or as it retires.
    held: Vec<Held>,
    waiting: Waiting,
    /// The wake semaphore's last value.
    woken: u64,
    stop: bool,
    lost: bool,
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
    item: Item,
}

impl DoneState {
    /// `cb`, held at `Committed` with `waits`, taking its scheduled handlers from `handlers`: the
    /// thread looks at it from its next round.
    fn hold(
        &mut self,
        cb: &Arc<CbInner>,
        waits: Vec<(Arc<EventInner>, u64)>,
        handlers: &mut Handlers,
    ) {
        for (event, _) in &waits {
            lock(&event.state).holds += 1;
        }
        self.held.push(Held {
            cb: Arc::clone(cb),
            waits,
            scheduled: std::mem::take(&mut handlers.scheduled),
        });
    }
}

/// A command buffer held at `Committed`: its waits for a shared event's value (none for one held
/// only behind an earlier one of its queue) and its scheduled handlers.
struct Held {
    cb: Arc<CbInner>,
    waits: Vec<(Arc<EventInner>, u64)>,
    scheduled: Vec<Handler>,
}

impl Held {
    /// Whether the value the host sees reaches each of its waits.
    fn reached(&self) -> bool {
        (self.waits.iter()).all(|(event, value)| event.reached(*value))
    }
}

/// Takes out of `held` the command buffers to schedule now, in order: those whose waits are
/// reached with no held one of their queue left before them, and those that retire in `retire`,
/// which are scheduled, where they ran, before they complete. Each event's count of held waits
/// falls with them; each queue's count of held command buffers falls once each is scheduled.
fn due(held: &mut Vec<Held>, retire: &[Retiree]) -> Vec<Held> {
    // The queues with a held command buffer left, by address: a later one of each waits for it.
    let mut blocked: Vec<*const QueueInner> = Vec::new();
    let mut due = Vec::new();
    let mut index = 0;
    while index < held.len() {
        let entry = &held[index];
        let queue = Arc::as_ptr(&entry.cb.queue);
        let retires = retire.iter().any(|retiree| retiree.is(&entry.cb));
        if retires || (entry.reached() && !blocked.contains(&queue)) {
            due.push(held.remove(index));
        } else {
            blocked.push(queue);
            index += 1;
        }
    }
    for (event, _) in due.iter().flat_map(|entry| &entry.waits) {
        lock(&event.state).holds -= 1;
    }
    due
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
                held: Vec::new(),
                waiting: Waiting::Not,
                woken: 0,
                stop: false,
                lost: false,
            }),
            changed: Condvar::new(),
            progress,
            wake,
            completed: AtomicU64::new(0),
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

    /// The highest progress value the completion thread has seen reached.
    pub(crate) fn completed(&self) -> u64 {
        self.completed.load(Ordering::SeqCst)
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

    /// From the submit thread: a submission passed to `vkQueueSubmit2` with progress `value`.
    /// Its command buffers retire in order once it completes, each ended one with its error.
    pub(crate) fn submitted(
        &self,
        core: &Core,
        value: u64,
        upload: Option<Upload>,
        finished: Vec<Finished>,
        watch: Vec<Arc<EventInner>>,
    ) {
        let mut state = self.lock();
        let mut flight_only = watch.is_empty();
        if let Some(upload) = upload {
            state.flight.push_back(Retiree {
                value,
                item: Item::Upload(upload),
            });
        }
        for (cb, pending, error) in finished {
            let written = pending.stamps_written();
            let Pending {
                gpu,
                mut handlers,
                held,
                ..
            } = pending;
            if let Some(waits) = held {
                state.hold(&cb, waits, &mut handlers);
                flight_only = false;
            }
            state.flight.push_back(Retiree {
                value,
                item: Item::Command {
                    cb,
                    gpu,
                    handlers,
                    error,
                    written,
                },
            });
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
                mut handlers,
                held,
                ..
            } = pending;
            if let Some(waits) = held {
                state.hold(&cb, waits, &mut handlers);
            }
            let retiree = Retiree {
                value,
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

    /// A host value of `event` may reach some of its listeners, or the waits of command buffers
    /// held for it.
    pub(crate) fn fire(&self, core: &Core, event: Arc<EventInner>) {
        let mut state = self.lock();
        state.fire.push(event);
        self.wake(&mut state, &core.device, false);
    }

    /// Tells the completion thread to end once nothing is in flight. The submit thread tells it
    /// as it ends, after the last command buffers it ended came here, so none comes after the
    /// thread. The thread ends then whatever a stop's deadline was: one past a quiesce's deadline
    /// ends once the GPU completed what was in flight and it retired that, so nothing in flight
    /// is left holding what it used, the core among it, and the stop joins it then.
    pub(crate) fn stop(&self, core: &Core) {
        let mut state = self.lock();
        state.stop = true;
        self.wake(&mut state, &core.device, false);
    }
}

/// The device is lost, or a submission failed: the submit thread ends what it holds and every
/// later commit, and the completion thread ends what is in flight once the device is idle,
/// without waiting on the GPU again.
pub(crate) fn lose(core: &Core) {
    core.work.lose();
    let mut state = core.done.lock();
    state.lost = true;
    core.done.wake(&mut state, &core.device, false);
}

impl Retiree {
    /// Retires it, a command buffer with its GPU times where it completed.
    fn retire(self, core: &Core, times: Option<(u64, u64)>) {
        match self.item {
            Item::Command {
                cb,
                gpu,
                handlers,
                error,
                written,
            } => command::retire(core, cb, gpu, handlers, error, times, written),
            Item::Upload(upload) => upload.retire(core),
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

    /// Whether it is `cb`.
    fn is(&self, cb: &Arc<CbInner>) -> bool {
        matches!(&self.item, Item::Command { cb: own, .. } if Arc::ptr_eq(own, cb))
    }

    /// Where it is `cb` and ends with an error: its handlers, which its scheduled handlers join.
    fn failed(&mut self, cb: &Arc<CbInner>) -> Option<&mut Handlers> {
        match &mut self.item {
            Item::Command {
                cb: own,
                handlers,
                error: Some(_),
                ..
            } if Arc::ptr_eq(own, cb) => Some(handlers),
            _ => None,
        }
    }

    /// In flight at a loss.
    fn lost(mut self) -> Retiree {
        if let Item::Command { error, .. } = &mut self.item {
            error.get_or_insert(CommandBufferError::DeviceLost { in_flight: true });
        }
        self
    }
}

/// What a round of the completion thread saw of the progress semaphore.
enum Seen {
    /// A host wait reported this value reached (0: nothing new).
    Reached(u64),
    /// The counter read this value, which a host wait then reported not reached.
    Behind(u64),
}

/// Whether a host wait on `semaphore` for `value`, with a timeout of 0, reports it reached.
pub(crate) fn reached_by_wait(
    device: &ash::Device,
    semaphore: vk::Semaphore,
    value: u64,
) -> Result<bool, vk::Result> {
    let semaphores = [semaphore];
    let values = [value];
    let wait = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    // SAFETY: the semaphore is a timeline semaphore of the device.
    match unsafe { device.wait_semaphores(&wait, 0) } {
        Ok(()) => Ok(true),
        Err(vk::Result::TIMEOUT) => Ok(false),
        Err(error) => Err(error),
    }
}

/// The completion thread, until it is stopped.
pub(crate) fn run(core: Arc<Core>) {
    let done = &core.done;
    let device = &core.device;
    // The GPU clock on the host's: read as the device starts, and again at a completion once the
    // period has passed since the last reading.
    let mut calibration = Calibration::read(&core);
    // A value the counter read and the host wait then reported not reached: the next wait is for
    // it, so the thread sleeps until the wait agrees rather than reading the counter again.
    let mut behind: Option<u64> = None;
    let mut state = done.lock();
    loop {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if !state.flight.is_empty() && core.hold_completions.try_read().is_err() {
            drop(state);
            drop(core.hold_completions.read());
            state = done.lock();
        }
        let mut retire = std::mem::take(&mut state.ended);
        let fired = std::mem::take(&mut state.fire);
        let mut fire: Vec<(Listener, u64)> = Vec::new();
        let mut newly_lost = false;
        // The host time the thread saw this round's completions, which no GPU time of theirs
        // passes (`Calibration::place`).
        let mut seen_at = None;
        if state.lost {
            // The signals pending never come: the host waits for them end first.
            for event in state.watched.drain(..) {
                event.lose();
            }
            if !state.flight.is_empty() {
                // Nothing is pending once the device is idle, which a lost device reaches at once.
                state.waiting = Waiting::Not;
                drop(state);
                core.wait_idle();
                state = done.lock();
                retire.extend(state.flight.drain(..).map(Retiree::lost));
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
                    false => unsafe { device.get_semaphore_counter_value(done.progress) }.and_then(
                        |read| match read >= front {
                            true => reached_by_wait(device, done.progress, read).map(|by| {
                                if by {
                                    Seen::Reached(read)
                                } else {
                                    Seen::Behind(read)
                                }
                            }),
                            false => Ok(Seen::Reached(read)),
                        },
                    ),
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
                    Err(_) => {
                        state.lost = true;
                        newly_lost = true;
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
        let due = due(&mut state.held, &retire);
        if newly_lost || !retire.is_empty() || !fire.is_empty() || !due.is_empty() {
            state.waiting = Waiting::Not;
            drop(state);
            if newly_lost {
                core.work.lose();
            }
            if !retire.is_empty() {
                // A batch kept while submissions were in flight may go.
                core.work.completed();
            }
            // Each held command buffer whose waits are reached, or that retires now, is scheduled
            // before any retires, so before it completes. One that retires with an error is not:
            // its scheduled handlers run as it retires, as a failed submission's do. Its queue's
            // later command buffers are held behind it until then, its handlers run.
            for Held { cb, scheduled, .. } in due {
                let queue = Arc::clone(&cb.queue);
                match retire.iter_mut().find_map(|retiree| retiree.failed(&cb)) {
                    Some(handlers) => handlers.scheduled = scheduled,
                    None => {
                        cb.scheduled(times::host_time());
                        if !scheduled.is_empty() {
                            command::run_scheduled(cb, scheduled);
                        }
                    }
                }
                queue.held.fetch_sub(1, Ordering::SeqCst);
            }
            // Handlers and listeners run with no lock held.
            // Each completed command buffer's timestamps are placed on the host clock by the
            // last calibration, read again first where the period has passed since it, and moved
            // earlier where it would place an end after the host time the thread saw it complete.
            let seen = seen_at.unwrap_or_else(times::host_time);
            let stamps: Vec<Option<(u64, u64)>> =
                retire.iter().map(|retiree| retiree.stamps(&core)).collect();
            if stamps.iter().any(Option::is_some)
                && calibration.is_none_or(|last| {
                    times::host_time().saturating_sub(last.host()) >= times::CALIBRATION_PERIOD
                })
            {
                calibration = Calibration::read(&core).or(calibration);
            }
            for (retiree, stamps) in retire.into_iter().zip(stamps) {
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                if let Some((start, end)) = stamps {
                    lock(&core.tick_spans).push(core.ticks.span(start, end));
                }
                let times = stamps
                    .zip(calibration.as_mut())
                    .map(|(stamps, calibration)| calibration.place(&core.ticks, stamps, seen));
                retiree.retire(&core, times);
            }
            for (listener, value) in fire {
                listener(value);
            }
            drop(fired);
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
        if result.is_err_and(|result| result != vk::Result::TIMEOUT) {
            state.lost = true;
            drop(state);
            core.work.lose();
            state = done.lock();
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
