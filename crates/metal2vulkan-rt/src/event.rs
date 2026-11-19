//! Events: Metal's event values carried by timeline semaphores, and the host floor of a shared
//! event.

use crate::counters::Gauge;
use crate::device::{lock, Core, Device};
use crate::error::Error;
use crate::reaper::{Free, Job};
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// `MTLEvent`: a timeline semaphore whose values are Metal's values.
///
/// Metal's values only rise, so a signal not above the highest value already submitted or set
/// changes nothing: the submit thread leaves it out, and the semaphore only rises. A cheap handle:
/// clones share one event. It holds the device.
#[derive(Clone)]
pub struct Event {
    inner: Arc<EventInner>,
    device: Device,
}

/// `MTLSharedEvent`: an [`Event`] the host also reads, sets and waits on.
///
/// A host set raises a floor and never signals the semaphore, so it never conflicts with a GPU
/// signal already submitted. `newSharedEventHandle` has no counterpart: one device serves every
/// guest process.
#[derive(Clone)]
pub struct SharedEvent {
    event: Event,
}

/// What a listener of a shared event runs: the event's value when it fires.
pub(crate) type Listener = Box<dyn FnOnce(u64) + Send>;

/// The event itself: what command buffers and the completion thread hold.
pub(crate) struct EventInner {
    pub(crate) semaphore: vk::Semaphore,
    /// Made by `newSharedEvent`: a command buffer that waits for a value of it not yet reached
    /// when it is submitted is scheduled once the value is (`crate::complete`), as Metal holds it
    /// at `Committed` until then.
    pub(crate) shared: bool,
    core: Arc<Core>,
    pub(crate) state: Mutex<EventState>,
    /// Signalled whenever the value the host sees rises, and at a loss.
    changed: Condvar,
}

pub(crate) struct EventState {
    /// The highest value of a signal handed to `vkQueueSubmit2`.
    pub(crate) submitted: u64,
    /// The highest value the host set.
    pub(crate) floor: u64,
    /// The highest semaphore value the completion thread has seen.
    observed: u64,
    /// Values of submitted signals the completion thread has not yet seen, lowest first.
    pub(crate) pending: VecDeque<u64>,
    /// In the completion thread's watch list: while signals are pending.
    pub(crate) watched: bool,
    /// The waits of submitted command buffers held at `Committed` for a value of it
    /// (`crate::complete`): while there are any, a host set wakes the completion thread.
    pub(crate) holds: usize,
    listeners: Vec<(u64, Listener)>,
    /// The device was lost while signals were pending: they never come.
    lost: bool,
}

impl Device {
    /// `newEvent`.
    ///
    /// Err: [`Error::Vulkan`] where the semaphore cannot be made.
    pub fn new_event(&self) -> Result<Event, Error> {
        self.event(false)
    }

    /// An event, a shared one where `shared`.
    fn event(&self, shared: bool) -> Result<Event, Error> {
        let core = self.core();
        let semaphore = timeline(&core.device)?;
        core.counters.rise(Gauge::Events);
        Ok(Event {
            inner: Arc::new(EventInner {
                semaphore,
                shared,
                core: Arc::clone(core),
                state: Mutex::new(EventState {
                    submitted: 0,
                    floor: 0,
                    observed: 0,
                    pending: VecDeque::new(),
                    watched: false,
                    holds: 0,
                    listeners: Vec::new(),
                    lost: false,
                }),
                changed: Condvar::new(),
            }),
            device: self.clone(),
        })
    }

    /// `newSharedEvent`.
    ///
    /// A command buffer that waits for a value of it is `Scheduled`, with its scheduled handlers
    /// run, once that value is reached, never while the signal it waits for is still running on
    /// the GPU; a wait for an [`Event`]'s value is scheduled as its command buffer is submitted.
    /// Apple defines `MTLCommandBufferStatusScheduled` as "All dependencies have been resolved and
    /// the command buffer has been scheduled for execution", and Metal applies it so to
    /// shared-event waits only. Recorded on the M3 Max (`event-wait-behind-a-running-signal`, five
    /// runs alike), a command buffer committed on another queue behind one whose signal still runs
    /// (that one `Scheduled`, its signal not landed) reads `Scheduled` with its scheduled handler
    /// run where it waits for an `MTLEvent`, and `Committed` with no handler run and the value 0
    /// where it waits for an `MTLSharedEvent`.
    ///
    /// Err: [`Error::Vulkan`] where the semaphore cannot be made.
    pub fn new_shared_event(&self) -> Result<SharedEvent, Error> {
        Ok(SharedEvent {
            event: self.event(true)?,
        })
    }
}

/// A new timeline semaphore at 0.
pub(crate) fn timeline(device: &ash::Device) -> Result<vk::Semaphore, Error> {
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
    // SAFETY: the device is live, and timeline semaphores are in the floor.
    unsafe { device.create_semaphore(&info, None) }.map_err(Error::vulkan("vkCreateSemaphore"))
}

impl Event {
    /// The event, for the command buffers that wait on it and signal it.
    pub(crate) fn inner(&self) -> &Arc<EventInner> {
        &self.inner
    }

    /// No Metal counterpart: the device the event was made on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl SharedEvent {
    /// `signaledValue`: the higher of the semaphore's value and the host's floor.
    pub fn signaled_value(&self) -> u64 {
        let inner = &self.event.inner;
        let state = lock(&inner.state);
        let floor = state.observed.max(state.floor);
        // A lost device answers with what was seen before the loss.
        inner.counter().map_or(floor, |counter| counter.max(floor))
    }

    /// `setSignaledValue:`: raises the host's floor to `value`; a value not above it changes
    /// nothing. The semaphore is never signalled from the host. Held command buffers and host
    /// waits that the new value meets go on, and listeners it reaches run.
    pub fn set_signaled_value(&self, value: u64) {
        let inner = &self.event.inner;
        let reached = {
            let mut state = lock(&inner.state);
            if value <= state.floor {
                return;
            }
            state.floor = value;
            inner.changed.notify_all();
            state.holds > 0 || state.listeners.iter().any(|(at, _)| *at <= value)
        };
        let core = &inner.core;
        core.work.poke();
        if reached {
            core.done.fire(core, Arc::clone(inner));
        }
    }

    /// `waitUntilSignaledValue:timeoutMS:`: true once [`signaled_value`](Self::signaled_value)
    /// reaches `value`, false when `timeout` passes first.
    ///
    /// Also false once the device is lost while the signals the wait needs are pending. A wait
    /// that no submitted signal can meet is met only by the host.
    ///
    /// Err: [`Refusal::InScheduledHandler`](crate::Refusal::InScheduledHandler) in a scheduled
    /// handler; [`Refusal::WaitOnOwnThread`](crate::Refusal::WaitOnOwnThread) in a completed
    /// handler or a listener of its device for a value not yet reached, whatever `timeout`: no
    /// false there, since that thread observes the value.
    pub fn wait_until_signaled_value(&self, value: u64, timeout: Duration) -> Result<bool, Error> {
        crate::command::not_in_scheduled_handler()?;
        let inner = &self.event.inner;
        let own = crate::device::on_completion_thread(&inner.core);
        let deadline = Instant::now().checked_add(timeout);
        let mut state = lock(&inner.state);
        loop {
            if state.floor.max(state.observed) >= value
                || inner.counter().is_ok_and(|counter| counter >= value)
            {
                return Ok(true);
            }
            if state.lost {
                return Ok(false);
            }
            if own {
                return Err(Error::Refused(crate::Refusal::WaitOnOwnThread));
            }
            state = match deadline {
                None => inner
                    .changed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(deadline) => {
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        return Ok(false);
                    };
                    inner
                        .changed
                        .wait_timeout(state, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
    }

    /// `notifyListener:atValue:block:`: runs `listener` on the completion thread once
    /// [`signaled_value`](Self::signaled_value) reaches `value`, at once where it already has, with
    /// the event's value then. Metal's block also receives the event; a listener captures what it
    /// needs. A signal the GPU reaches in the middle of a command buffer runs it then, not at the
    /// command buffer's end.
    ///
    /// Err: [`Error::Quiesced`] after [`Device::quiesce`], when no listener runs again.
    pub fn notify_listener(
        &self,
        value: u64,
        listener: impl FnOnce(u64) + Send + 'static,
    ) -> Result<(), Error> {
        let inner = &self.event.inner;
        let core = &inner.core;
        if core.closed() {
            return Err(Error::Quiesced);
        }
        let reached = {
            let mut state = lock(&inner.state);
            state.listeners.push((value, Box::new(listener)));
            value <= state.observed.max(state.floor)
        };
        if reached {
            core.done.fire(core, Arc::clone(inner));
        }
        Ok(())
    }

    /// `MTLSharedEvent` as an `MTLEvent`, for the encode calls.
    pub fn as_event(&self) -> &Event {
        &self.event
    }

    /// For the device tests: records a GPU signal of `value` as submitted, so a wait for it waits
    /// on the semaphore, which the test then signals from the host.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn pretend_submitted_for_test(&self, value: u64) {
        self.event.pretend_submitted_for_test(value);
    }
}

impl Event {
    /// As [`SharedEvent::pretend_submitted_for_test`].
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn pretend_submitted_for_test(&self, value: u64) {
        let mut state = lock(&self.inner.state);
        state.submitted = state.submitted.max(value);
    }
}

impl EventInner {
    /// Whether the event is `core`'s.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(&self.core, core)
    }

    /// The semaphore's value now.
    pub(crate) fn counter(&self) -> Result<u64, Error> {
        // SAFETY: the semaphore is a timeline semaphore of the device, live while `self` is.
        unsafe { self.core.device.get_semaphore_counter_value(self.semaphore) }
            .map_err(Error::vulkan("vkGetSemaphoreCounterValue"))
    }

    /// On the completion thread: takes in the semaphore's value (`counter`, where read), wakes the
    /// host waits it meets, and takes out the listeners the event's value now reaches, with that
    /// value. Returns whether signals are still pending.
    pub(crate) fn observe(&self, counter: Option<u64>, fire: &mut Vec<(Listener, u64)>) -> bool {
        let mut state = lock(&self.state);
        if let Some(counter) = counter.filter(|&counter| counter > state.observed) {
            state.observed = counter;
            while state.pending.front().is_some_and(|&value| value <= counter) {
                state.pending.pop_front();
            }
            self.changed.notify_all();
        }
        let value = state.observed.max(state.floor);
        fire.extend(
            state
                .listeners
                .extract_if(.., |(at, _)| *at <= value)
                .map(|(_, listener)| (listener, value)),
        );
        if state.pending.is_empty() {
            state.watched = false;
        }
        state.watched
    }

    /// At a loss: the pending signals never come, so the host waits for them end.
    pub(crate) fn lose(&self) {
        let mut state = lock(&self.state);
        state.lost = true;
        state.pending.clear();
        state.watched = false;
        self.changed.notify_all();
    }

    /// Whether the value the host sees, its floor or the semaphore's value the completion thread
    /// has seen, reaches `value`.
    pub(crate) fn reached(&self, value: u64) -> bool {
        let state = lock(&self.state);
        state.observed.max(state.floor) >= value
    }

    /// The value the completion thread waits for next: the lowest pending signal.
    pub(crate) fn next_pending(&self) -> Option<u64> {
        lock(&self.state).pending.front().copied()
    }
}

impl Drop for EventInner {
    fn drop(&mut self) {
        // No pending GPU work names the semaphore: every command buffer that waits on it or
        // signals it holds the event until it retires, and the completion thread holds it while
        // its signals are pending. The reaper destroys it, off the completion thread.
        self.core
            .reaper
            .give(Job::Free(Free::Semaphore(self.semaphore)));
    }
}
