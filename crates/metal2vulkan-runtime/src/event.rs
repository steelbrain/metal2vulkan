//! `Event`, `SharedEvent`, `SharedEventHandle` and `SharedEventListener` (`MTLEvent`,
//! `MTLSharedEvent`, `MTLSharedEventHandle`, `MTLSharedEventListener`).
//!
//! - **One Vulkan object.** An event is a timeline semaphore. Its counter is a signal index that
//!   [`EventTimeline`] maps to Metal's values: only the GPU signals the scheduler keeps raise it,
//!   and a host set is a floor that never touches it.
//! - **A shared event's wait is never handed to Vulkan.** The host may set a shared event's value
//!   (`MTLSharedEvent`), and Metal's set releases a command buffer's wait as soon as the value is
//!   reached. No Vulkan wait a batch was submitted with can be released by the host, so a wait for
//!   such a value is held by the scheduler until the value is reached, and goes with no Vulkan
//!   wait (`event_timeline.rs`). Whatever reaches the value wakes the scheduler for it: the set,
//!   and the observation of a GPU signal ([`EventShared::reached`], [`EventShared::value`]). A
//!   plain event's (`MTLEvent`) wait, which no host call can satisfy, still chains on the
//!   `VkQueue`.
//! - **Who observes.** A completion turn sees each batch complete, and with it the event
//!   signals of the batch; reading the value reads the semaphore's counter too. Either one wakes
//!   the host's waits and posts the listeners' notifications that came due. A signal submitted
//!   while a copied host buffer is live is the completion turn's alone, which merges first
//!   (`event_timeline.rs`, `copied.rs`).
//! - **A read is observed once a host wait returned for it.** The program may read what the
//!   signalling work wrote once it sees the value, and Vulkan orders host operations after a
//!   signal only through a wait ("signal operations happening before host operations that wait
//!   on them" system-synchronize-with them, the memory model): `vkGetSemaphoreCounterValue`'s
//!   value "may immediately be out of date" and orders nothing. So a counter the read shows past
//!   the index last seen is waited for first, a wait of no time ("If the condition is satisfied
//!   when `vkWaitSemaphores` is called, then `vkWaitSemaphores` returns immediately"), as the
//!   queues' waiters do (`command_buffer/completion.rs`): one call more, and only where the
//!   read ran ahead of the completion turns.
//! - **Host waits** (`wait_until_signaled_value`) wait on the event's condition variable, which
//!   every rise signals where a host wait sleeps on it (`EventState::parked`, read under the lock
//!   the rise is made under: a wait that sleeps later looks at the value first), and read the
//!   counter again at least every [`POLL`]. A rise with no wait asleep spares the notify, a system
//!   call (`WorkCounts::wakes_event_waiters`, `WorkCounts::wakes_spared`). Once the process is
//!   exiting they answer from the value last seen ([`exit_guard`]).
//! - **Shared event handles** are in-process: a handle shares the event, and opening it gives a
//!   new `SharedEvent` of the same event. Opening it on another device is refused: Vulkan
//!   semaphores do not cross devices unless exported, which the runtime does not do.
//! - **Listeners** run notifications on a thread of their own, one after another in the order they
//!   came due, started as the first notification is asked for: a listener no notification is
//!   asked of starts no thread, as Metal's runs its blocks on a dispatch queue, whose threads
//!   come and go with the work. A notification for a value already reached comes due at once
//!   (measured on Metal).

use super::device::{lock, Device};
use super::error::{vulkan, Error, RuntimeGap};
use super::event_timeline::{EventTimeline, Signal, WaitFor};
use super::exit_guard::{self, Call};
use crate::release;
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long a host wait goes at most without reading the semaphore's counter.
const POLL: Duration = Duration::from_millis(50);

/// A Metal event (`MTLEvent`): command buffers signal it and wait for it. Cheap to clone (a
/// retain); `Send + Sync`.
#[derive(Clone)]
pub struct Event(Arc<EventShared>);

/// A Metal shared event (`MTLSharedEvent`): an event the host can also signal, read, wait for and
/// listen to. Cheap to clone (a retain); `Send + Sync`.
#[derive(Clone)]
pub struct SharedEvent(Arc<EventShared>);

/// An in-process handle of a shared event (`MTLSharedEventHandle`): opened with
/// [`Device::new_shared_event_with_handle`], it gives the same event.
#[derive(Clone)]
pub struct SharedEventHandle(Arc<EventShared>);

/// The event a command buffer signals or waits for: either kind (`id<MTLEvent>`).
#[derive(Clone, Copy, Debug)]
pub enum EventRef<'a> {
    /// An `MTLEvent`.
    Event(&'a Event),
    /// An `MTLSharedEvent`.
    Shared(&'a SharedEvent),
}

impl<'a> From<&'a Event> for EventRef<'a> {
    fn from(event: &'a Event) -> EventRef<'a> {
        EventRef::Event(event)
    }
}

impl<'a> From<&'a SharedEvent> for EventRef<'a> {
    fn from(event: &'a SharedEvent) -> EventRef<'a> {
        EventRef::Shared(event)
    }
}

impl EventRef<'_> {
    /// The event.
    pub(super) fn shared(&self) -> &Arc<EventShared> {
        match self {
            EventRef::Event(event) => &event.0,
            EventRef::Shared(event) => &event.0,
        }
    }
}

/// A shared event's notification (`MTLSharedEventNotificationBlock`): called once on its
/// listener's thread with the event and the value it waited for. A panic inside it is caught.
pub type SharedEventNotification = Box<dyn FnOnce(&SharedEvent, u64) + Send + 'static>;

/// What both kinds of event share.
pub(super) struct EventShared {
    device: Device,
    /// The timeline semaphore; its counter is the signal index.
    semaphore: vk::Semaphore,
    state: Mutex<EventState>,
    /// Signaled whenever the value rises.
    changed: Condvar,
    label: Mutex<Option<String>>,
}

struct EventState {
    /// The signal indices' Metal values, by `VkQueue` index.
    timeline: EventTimeline<usize>,
    /// The listeners' notifications not yet due, in the order they were asked for.
    notifications: Vec<Notification>,
    /// The host waits asleep on `changed`: a rise notifies it only where one is.
    parked: usize,
}

struct Notification {
    value: u64,
    listener: SharedEventListener,
    block: SharedEventNotification,
}

// One event is equal only to itself: a command buffer's batch waits for or signals each event
// once.
impl PartialEq for EventShared {
    fn eq(&self, other: &EventShared) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Drop for EventShared {
    fn drop(&mut self) {
        // Once the process is exiting, the semaphore is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        shared.interop.timeline_gone(self.semaphore);
        // SAFETY: every batch that signals or waits for the semaphore holds this event until it
        // completed, so the semaphore is not in use.
        unsafe { release::object(shared.vk(), self.semaphore) };
    }
}

impl EventShared {
    /// `settable` for a shared event (`MTLSharedEvent`), whose value the host may set.
    fn new(device: &Device, settable: bool) -> Result<Arc<EventShared>, Error> {
        if device.is_lost() {
            return Err(Error::DeviceLost);
        }
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: the device is live and enables `timelineSemaphore` (created devices enable it;
        // `Device::from_raw` requires it).
        let semaphore =
            unsafe { device.shared().vk().create_semaphore(&info, None) }.map_err(vulkan)?;
        device.shared().interop.timeline_made(semaphore);
        Ok(Arc::new(EventShared {
            device: device.clone(),
            semaphore,
            state: Mutex::new(EventState {
                timeline: EventTimeline::new(settable),
                notifications: Vec::new(),
                parked: 0,
            }),
            changed: Condvar::new(),
            label: Mutex::new(None),
        }))
    }

    /// The device.
    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    /// The timeline semaphore.
    pub(super) fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }

    /// How a batch submitted now waits for `value`; `None` while nothing submitted or set reaches
    /// it.
    pub(super) fn wait(&self, value: u64) -> Option<WaitFor> {
        lock(&self.state).timeline.wait(value)
    }

    /// How the host waits for `value` to land: reached, or the index of the submitted GPU signal
    /// that reaches it, a wait held on the host included (`EventTimeline::landing`); `None` while
    /// nothing submitted or set reaches it.
    pub(super) fn landing(&self, value: u64) -> Option<WaitFor> {
        lock(&self.state).timeline.landing(value)
    }

    /// Whether a wait for `value` is held on the host where a Vulkan wait for a submitted GPU
    /// signal's index would have served (`event_timeline.rs`).
    pub(super) fn held_for_the_host(&self, value: u64) -> bool {
        lock(&self.state).timeline.held_for_the_host(value)
    }

    /// What a signal of `value` by a batch submitted now on `VkQueue` `queue` becomes.
    pub(super) fn signal(&self, value: u64, queue: usize) -> Signal {
        lock(&self.state).timeline.signal(value, queue)
    }

    /// The batch of a kept signal was submitted.
    pub(super) fn keep(&self, index: u64, value: u64, queue: usize, behind_merge: bool) {
        lock(&self.state)
            .timeline
            .keep(index, value, queue, behind_merge);
    }

    /// The semaphore reached `index`: wake the host's waits and post the notifications that came
    /// due.
    pub(super) fn reached(self: &Arc<Self>, index: u64) {
        let due = {
            let mut state = lock(&self.state);
            if !state.timeline.observe(index) {
                return;
            }
            self.rose(&state);
            due(&mut state)
        };
        self.post(due);
        self.woke();
    }

    /// The value now, from the semaphore's counter and the host's floor. A counter the read
    /// shows past the index last seen is waited for, a wait that returns at once, before it is
    /// observed (the module docs). Once the process is exiting, the counter is not read
    /// (`exit_guard`): the value is the one last seen.
    pub(super) fn value(self: &Arc<Self>) -> u64 {
        if let Some(busy) = exit_guard::enter() {
            let raw = self.device.shared().vk();
            // SAFETY: the semaphore is the device's and live. A failure (a lost device) leaves
            // the value as last seen.
            let counter = unsafe { raw.get_semaphore_counter_value(self.semaphore) };
            exit_guard::called(Call::GetSemaphoreCounterValue);
            if let Ok(read) = counter {
                let reaches = |index: u64| {
                    let semaphores = [self.semaphore];
                    let values = [index];
                    let info = vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&values);
                    // SAFETY: as the read's above. "If timeout is zero, then vkWaitSemaphores
                    // does not wait"; a failure (a lost device) leaves the value as last seen.
                    let waited = unsafe { raw.wait_semaphores(&info, 0) };
                    exit_guard::called(Call::WaitSemaphores);
                    waited.map(|()| true).or_else(|error| match error {
                        vk::Result::TIMEOUT => Ok(false),
                        error => Err(error),
                    })
                };
                let (rose, due) = {
                    let mut state = lock(&self.state);
                    match state.timeline.observe_read(read, reaches) {
                        Ok(true) => {
                            self.rose(&state);
                            (true, due(&mut state))
                        }
                        Ok(false) | Err(_) => (false, Vec::new()),
                    }
                };
                drop(busy);
                self.post(due);
                if rose {
                    self.woke();
                }
            }
        }
        lock(&self.state).timeline.value()
    }

    /// The host set `value`: [`SharedEvent::set_signaled_value`]'s, and a command buffer's
    /// scheduled signal's (`CommandBuffer::encode_signal_event_scheduled`).
    pub(super) fn set(self: &Arc<Self>, value: u64) {
        let due = {
            let mut state = lock(&self.state);
            if !state.timeline.set(value) {
                return;
            }
            self.rose(&state);
            due(&mut state)
        };
        self.post(due);
        self.woke();
    }

    /// Wait until the value reaches `value`, at most `timeout`; a timeout past what the host's
    /// clock holds (Metal's `UINT64_MAX` milliseconds) has no deadline. Once the process is
    /// exiting, it answers at once from the value last seen: neither a completion turn nor the
    /// counter raises it any more (`exit_guard`).
    pub(super) fn wait_until(self: &Arc<Self>, value: u64, timeout: Duration) -> bool {
        let deadline = Instant::now().checked_add(timeout);
        loop {
            if self.value() >= value {
                return true;
            }
            let mut state = lock(&self.state);
            if state.timeline.value() >= value {
                return true;
            }
            let left = deadline.map_or(POLL, |deadline| {
                deadline.saturating_duration_since(Instant::now())
            });
            if left.is_zero() || exit_guard::closed() {
                return false;
            }
            state.parked += 1;
            let mut state = self
                .changed
                .wait_timeout(state, left.min(POLL))
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
            state.parked -= 1;
        }
    }

    /// Notify `listener` with `block` once the value reaches `value`.
    fn notify(
        self: &Arc<Self>,
        listener: &SharedEventListener,
        value: u64,
        block: SharedEventNotification,
    ) {
        // Its thread runs from now on, so the notification runs as it comes due.
        listener.start();
        let notification = Notification {
            value,
            listener: listener.clone(),
            block,
        };
        let due = {
            let mut state = lock(&self.state);
            state.notifications.push(notification);
            due(&mut state)
        };
        self.post(due);
    }

    /// The value rose, under the state's lock (`state`): wake the host waits asleep on it, if any
    /// (the module docs).
    fn rose(&self, state: &EventState) {
        let parked = state.parked > 0;
        if parked {
            self.changed.notify_all();
        }
        self.device.shared().counts.event_wake(parked);
    }

    /// The value rose: a batch held for it may go now, so the scheduler looks again (its waits
    /// for a value the host may set are held, never handed to Vulkan: the module docs). Called
    /// without the event's lock, which the scheduler's thread takes under its own.
    fn woke(&self) {
        self.device.shared().scheduler.wake();
    }

    /// Post `due` notifications to their listeners.
    fn post(self: &Arc<Self>, due: Vec<Notification>) {
        for notification in due {
            let event = SharedEvent(Arc::clone(self));
            let Notification {
                value,
                listener,
                block,
            } = notification;
            listener.post(Box::new(move || block(&event, value)));
        }
    }
}

/// The notifications of `state` whose value was reached, removed, in the order they were asked
/// for.
fn due(state: &mut EventState) -> Vec<Notification> {
    let value = state.timeline.value();
    let mut due = Vec::new();
    let mut at = 0;
    while at < state.notifications.len() {
        if state.notifications[at].value <= value {
            due.push(state.notifications.remove(at));
        } else {
            at += 1;
        }
    }
    due
}

impl Device {
    /// `newEvent`.
    pub fn new_event(&self) -> Result<Event, Error> {
        Ok(Event(EventShared::new(self, false)?))
    }

    /// `newSharedEvent`.
    pub fn new_shared_event(&self) -> Result<SharedEvent, Error> {
        Ok(SharedEvent(EventShared::new(self, true)?))
    }

    /// `newSharedEventWithHandle:`: the event of `handle`, made by this device. Two calls of
    /// [`Device::system_default_with`] or [`Device::all_with`] with equal options return one
    /// device, as Metal's do, so their events' handles open on each other. A handle of another
    /// device's event (another physical device's, or one made with other options or adopted) is
    /// refused (`NotImplemented`, `SharedEventAcrossDevices`).
    pub fn new_shared_event_with_handle(
        &self,
        handle: &SharedEventHandle,
    ) -> Result<SharedEvent, Error> {
        if !handle.0.device.same(self) {
            return Err(Error::NotImplemented(RuntimeGap::SharedEventAcrossDevices));
        }
        Ok(SharedEvent(Arc::clone(&handle.0)))
    }
}

impl std::fmt::Debug for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Event")
            .field("label", &self.label())
            .finish()
    }
}

impl Event {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        *lock(&self.0.label) = Some(label.to_owned());
    }

    /// `label`.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.label).clone()
    }
}

impl std::fmt::Debug for SharedEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEvent")
            .field("label", &self.label())
            .finish()
    }
}

impl SharedEvent {
    /// `device`, which a Metal **shared** event answers `nil` for, so this answers `None`.
    ///
    /// `MTLSharedEvent` inherits the getter from `MTLEvent`, whose header declares it nonnull, and
    /// a plain `MTLEvent` does answer the device it was made on ([`Event::device`]). A shared one
    /// does not: measured on an Apple M3 Max (macOS 27.0.1, plain Metal),
    /// `[[dev newSharedEvent] device]` and the event `newSharedEventWithHandle:` answers are both
    /// `nil`, while `[[dev newEvent] device]` is the device. The runtime answers what Metal
    /// answers.
    ///
    /// The event still belongs to a device, and the runtime still holds it: a handle opened on
    /// another device is refused ([`Device::new_shared_event_with_handle`]), and a signal or wait
    /// encoded on another device's command buffer is refused as a foreign object, exactly as
    /// Metal's are. What is not available is a getter for it, because Metal offers none.
    pub fn device(&self) -> Option<&Device> {
        None
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        *lock(&self.0.label) = Some(label.to_owned());
    }

    /// `label`.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.label).clone()
    }

    /// `signaledValue`. Once the process is exiting, the value last seen: the runtime makes no
    /// Vulkan call then.
    pub fn signaled_value(&self) -> u64 {
        self.0.value()
    }

    /// `setSignaledValue:`: the value becomes at least `value` at once; a value not above the
    /// current one changes nothing (measured on Metal).
    pub fn set_signaled_value(&self, value: u64) {
        self.0.set(value);
    }

    /// `waitUntilSignaledValue:timeoutMS:`: whether the value reached `value` within `timeout`.
    /// A timeout past what the host's clock holds, as Metal's `UINT64_MAX` milliseconds, waits
    /// until the value comes. Once the process is exiting, it answers at once from the value last seen (a wait already
    /// blocked, within 50 ms): the runtime reads no counter then.
    pub fn wait_until_signaled_value(&self, value: u64, timeout: Duration) -> bool {
        self.0.wait_until(value, timeout)
    }

    /// `notifyListener:atValue:block:`: call `block` on `listener`'s thread once the value
    /// reaches `value`, at once when it already has.
    pub fn notify_listener(
        &self,
        listener: &SharedEventListener,
        value: u64,
        block: SharedEventNotification,
    ) {
        self.0.notify(listener, value, block);
    }

    /// `newSharedEventHandle`: an in-process handle of this event.
    pub fn new_shared_event_handle(&self) -> SharedEventHandle {
        SharedEventHandle(Arc::clone(&self.0))
    }
}

impl std::fmt::Debug for SharedEventHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEventHandle")
            .field("label", &self.label())
            .finish()
    }
}

impl SharedEventHandle {
    /// `label`: the event's.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.label).clone()
    }
}

/// A Metal shared event listener (`MTLSharedEventListener`): a thread of its own that runs the
/// notifications posted to it, one after another, started as the first one is asked for
/// ([`SharedEvent::notify_listener`]; none for a listener never asked one). Cheap to clone (a
/// retain); `Send + Sync`. The thread ends once every handle is gone, including those that pending
/// notifications hold, and what was posted ran. Metal runs the blocks on a dispatch queue of its
/// own, which a listener made with `initWithDispatchQueue:` names: the runtime has no form of that
/// initializer.
#[derive(Clone)]
pub struct SharedEventListener(Arc<ListenerHandle>);

impl std::fmt::Debug for SharedEventListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEventListener")
            .finish_non_exhaustive()
    }
}

/// The handles' hold on the listener's thread: dropped last, it lets the thread end.
struct ListenerHandle {
    queue: Arc<ListenerQueue>,
}

impl Drop for ListenerHandle {
    fn drop(&mut self) {
        lock(&self.queue.jobs).closed = true;
        self.queue.wake.notify_all();
    }
}

pub(super) type Job = Box<dyn FnOnce() + Send>;

struct ListenerQueue {
    jobs: Mutex<Jobs>,
    wake: Condvar,
}

struct Jobs {
    posted: VecDeque<Job>,
    /// Every handle is gone: nothing more can be posted.
    closed: bool,
    /// The thread was started ([`SharedEventListener::start`]).
    started: bool,
}

impl Default for SharedEventListener {
    fn default() -> SharedEventListener {
        SharedEventListener::new()
    }
}

impl SharedEventListener {
    /// `[[MTLSharedEventListener alloc] init]`: a listener whose thread starts as its first
    /// notification is asked for. Where the host cannot start the thread then, each later
    /// notification asked for or come due tries again, and those come due wait for it.
    pub fn new() -> SharedEventListener {
        let queue = Arc::new(ListenerQueue {
            jobs: Mutex::new(Jobs {
                posted: VecDeque::new(),
                closed: false,
                started: false,
            }),
            wake: Condvar::new(),
        });
        SharedEventListener(Arc::new(ListenerHandle { queue }))
    }

    /// Start the thread unless it started.
    fn start(&self) {
        Self::start_under(&self.0.queue, &mut lock(&self.0.queue.jobs));
    }

    /// Start `queue`'s thread unless it started, under its jobs' lock (`jobs`): one thread
    /// starts, whatever thread asks.
    fn start_under(queue: &Arc<ListenerQueue>, jobs: &mut Jobs) {
        if jobs.started {
            return;
        }
        let listening = Arc::clone(queue);
        let started = exit_guard::spawn(
            std::thread::Builder::new().name("metal2vulkan-listener".into()),
            move || listen(&listening),
        );
        jobs.started = started.is_ok();
    }

    /// Post `job`, after the jobs posted before it; the thread started where it has not.
    pub(super) fn post(&self, job: Job) {
        let queue = &self.0.queue;
        let mut jobs = lock(&queue.jobs);
        jobs.posted.push_back(job);
        Self::start_under(queue, &mut jobs);
        drop(jobs);
        queue.wake.notify_all();
    }

    /// Whether its thread was started.
    #[cfg(test)]
    pub(crate) fn started(&self) -> bool {
        lock(&self.0.queue.jobs).started
    }
}

/// The listener thread's loop: run what is posted until every handle is gone and nothing is left.
fn listen(queue: &ListenerQueue) {
    loop {
        let job = {
            let mut jobs = lock(&queue.jobs);
            loop {
                if let Some(job) = jobs.posted.pop_front() {
                    break job;
                }
                if jobs.closed {
                    return;
                }
                jobs = queue
                    .wake
                    .wait(jobs)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        exit_guard::program(job);
    }
}
