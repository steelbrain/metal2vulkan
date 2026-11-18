//! The completion thread: it retires command buffers and uploads as the GPU completes them, runs
//! the handlers, follows the event signals it submitted and runs listeners, and ends the work in
//! flight at a loss.

use crate::command::{self, CbInner, CommandBufferError, Gpu, Handler, Handlers, Pending};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::event::{self, EventInner, Listener};
use crate::readback;
use crate::submit::{Ended, Finished, Joined, Upload};
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
    /// Command buffers just scheduled, with their scheduled handlers.
    scheduled: Vec<(Arc<CbInner>, Vec<Handler>)>,
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
    /// With `stop`: when the thread ends whatever is still in flight.
    deadline: Option<Instant>,
    lost: bool,
    /// The thread ended with nothing in flight.
    drained: bool,
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
                scheduled: Vec::new(),
                flight: VecDeque::new(),
                ended: Vec::new(),
                watched: Vec::new(),
                fire: Vec::new(),
                waiting: Waiting::Not,
                woken: 0,
                stop: false,
                deadline: None,
                lost: false,
                drained: false,
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

    /// From the submit thread: command buffers scheduled as they joined a submission, with their
    /// scheduled handlers, ahead of that submission.
    pub(crate) fn scheduled(&self, core: &Core, joined: Vec<Joined>) {
        if joined.is_empty() {
            return;
        }
        let mut state = self.lock();
        state.scheduled.extend(joined);
        self.wake(&mut state, &core.device, false);
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
        let flight_only = watch.is_empty();
        if let Some(upload) = upload {
            state.flight.push_back(Retiree {
                value,
                item: Item::Upload(upload),
            });
        }
        for (cb, pending, error) in finished {
            let written = pending.stamps_written();
            let Pending { gpu, handlers, .. } = pending;
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
                handlers,
                ..
            } = pending;
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

    /// A host value of `event` may reach some of its listeners.
    pub(crate) fn fire(&self, core: &Core, event: Arc<EventInner>) {
        let mut state = self.lock();
        state.fire.push(event);
        self.wake(&mut state, &core.device, false);
    }

    /// Tells the completion thread to end once nothing is in flight, or at `deadline`.
    pub(crate) fn stop(&self, core: &Core, deadline: Option<Instant>) {
        let mut state = self.lock();
        state.stop = true;
        state.deadline = deadline;
        self.wake(&mut state, &core.device, false);
    }

    /// Whether the completion thread ended with nothing in flight.
    pub(crate) fn drained(&self) -> bool {
        self.lock().drained
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

    /// A completed command buffer's two timestamps, where it wrote both: the read waits for them,
    /// and a query never written would never become available.
    fn stamps(&self, core: &Core) -> Option<(u64, u64)> {
        match &self.item {
            Item::Command {
                gpu: Gpu {
                    stamps: Some(pair), ..
                },
                error: None,
                written: 2,
                ..
            } => {
                // SAFETY: the pair's pool holds its two queries, which a submission that
                // completed wrote.
                unsafe { readback::timestamps(&core.device, pair.pool, pair.first) }
            }
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

/// The completion thread, until it is stopped.
pub(crate) fn run(core: Arc<Core>) {
    let done = &core.done;
    let device = &core.device;
    // The GPU clock on the host's: read as the device starts, and again at a completion once the
    // period has passed since the last reading.
    let mut calibration = Calibration::read(&core);
    let mut state = done.lock();
    loop {
        let scheduled = std::mem::take(&mut state.scheduled);
        let mut retire = std::mem::take(&mut state.ended);
        let fired = std::mem::take(&mut state.fire);
        let mut fire: Vec<(Listener, u64)> = Vec::new();
        let mut newly_lost = false;
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
            if !state.flight.is_empty() {
                // SAFETY: the progress semaphore is the device's.
                match unsafe { device.get_semaphore_counter_value(done.progress) } {
                    Ok(reached) => {
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
        if newly_lost || !scheduled.is_empty() || !retire.is_empty() || !fire.is_empty() {
            state.waiting = Waiting::Not;
            drop(state);
            if newly_lost {
                core.work.lose();
            }
            if !retire.is_empty() {
                // A batch kept while submissions were in flight may go.
                core.work.completed();
            }
            // Handlers and listeners run with no lock held.
            for (cb, handlers) in scheduled {
                command::run_scheduled(cb, handlers);
            }
            // Each completed command buffer's timestamps are placed on the host clock by the
            // last calibration, read again first where the period has passed since it.
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
                let times = stamps
                    .zip(calibration)
                    .map(|((start, end), calibration)| calibration.place(&core.ticks, start, end));
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
        if state.stop {
            if state.flight.is_empty() {
                state.drained = true;
                break;
            }
            if state
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                break;
            }
        }
        let timeout = state
            .deadline
            .filter(|_| state.stop)
            .map(|deadline| deadline.saturating_duration_since(Instant::now()));
        if state.lost || (state.flight.is_empty() && state.watched.is_empty()) {
            state.waiting = Waiting::Host;
            state = match timeout {
                None => done
                    .changed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(timeout) => {
                    done.changed
                        .wait_timeout(state, timeout)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
            state.waiting = Waiting::Not;
            continue;
        }
        let mut semaphores = Vec::new();
        let mut values = Vec::new();
        let target = state.flight.front().map(|retiree| retiree.value);
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
        let nanos = timeout.map_or(u64::MAX, nanos);
        // SAFETY: every semaphore is a timeline semaphore of the device, live while the state
        // holds its event or the core lives.
        let result = unsafe { device.wait_semaphores(&wait, nanos) };
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
