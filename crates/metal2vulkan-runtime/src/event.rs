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
//! - **The caller's signals** (`Event::reserve_vk_signal`, `SharedEvent::reserve_vk_signal`,
//!   `interop.rs`). Either kind's semaphore takes signals of the caller's own submissions (a
//!   video queue's, say): the caller reserves the next signal index for a Metal value under the
//!   scheduler's lock, so no batch takes that index meanwhile, and its submission signals it
//!   (`event_timeline.rs`). A plain event's wait for it chains on the `VkQueue` from the
//!   reservation, a shared event's is held until it landed. No completion turn sees such a
//!   signal, so the device's watcher ([`CallerSignals`]) waits for each on the host, in
//!   reservation order, and observes it as a completion turn would: the value rises, host waits
//!   wake, notifications come due, and the scheduler looks again. Its thread starts at the first
//!   reservation, and ends once none was left for [`WATCH_LINGER`].
//! - **Listeners** run notifications on a thread of their own, one after another in the order they
//!   came due, started as the first notification is asked for: a listener no notification is
//!   asked of starts no thread, as Metal's runs its blocks on a dispatch queue, whose threads
//!   come and go with the work. A notification for a value already reached comes due at once
//!   (measured on Metal).

use super::command_buffer::follow::Follow;
use super::device::carry::{LostParts, MadeOn};
use super::device::{lock, Device};
use super::error::{vulkan, Error, RuntimeGap};
use super::event_timeline::{EventTimeline, Signal, WaitFor};
use super::exit_guard::{self, Call};
use crate::release;
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

/// How long a host wait goes at most without reading the semaphore's counter.
const POLL: Duration = Duration::from_millis(50);

/// How long the watcher of the caller's signals waits on the GPU at once, so it sees a lost device
/// or the process's exit.
const WATCH_SLICE: Duration = Duration::from_millis(100);

/// How long the watcher's thread lingers with no reserved signal left before it ends.
const WATCH_LINGER: Duration = Duration::from_secs(1);

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
    /// The event, as its device's serving generation holds it ([`EventShared::serving`]).
    pub(super) fn shared(&self) -> &Arc<EventShared> {
        match self {
            EventRef::Event(event) => event.0.serving(),
            EventRef::Shared(event) => event.0.serving(),
        }
    }
}

/// A shared event's notification (`MTLSharedEventNotificationBlock`): called once on its
/// listener's thread with the event and the value it waited for. A panic inside it is caught.
pub type SharedEventNotification = Box<dyn FnOnce(&SharedEvent, u64) + Send + 'static>;

/// What both kinds of event share.
pub(super) struct EventShared {
    /// Its generation of the device: let go of once the event was carried to the next one and
    /// its semaphore there went (`EventShared::release_lost`).
    device: MadeOn,
    /// The timeline semaphore; its counter is the signal index.
    semaphore: vk::Semaphore,
    /// The semaphore was made exportable as an opaque file descriptor
    /// (`exported_semaphore.rs`).
    exportable: bool,
    state: Mutex<EventState>,
    /// Signaled whenever the value rises, and as the event is carried to its device's next
    /// generation, which its host waits go on with.
    changed: Condvar,
    label: Mutex<Option<String>>,
    /// The event as its device's recovery carried it to the next generation (`device/carry.rs`):
    /// a new semaphore there, which every call of the program's goes to from then on
    /// ([`EventShared::serving`]). Set once, under the state's lock.
    next: OnceLock<Arc<EventShared>>,
    /// The host's calls on the semaphore under way ([`EventShared::host_use`]), whose last lets
    /// the semaphore of an event carried to the next generation go.
    uses: Mutex<HostUses>,
}

/// The host's calls on an event's semaphore under way (`EventShared::host_use`): a read of its
/// counter, a host wait's slice, a reservation of a caller's signal.
#[derive(Default)]
struct HostUses {
    under_way: usize,
    /// The event was carried to its device's next generation and its lost generation is at rest
    /// (`device/carry.rs`): its semaphore goes once no call is under way, and none begins again.
    lost: bool,
}

/// One host call on an event's semaphore (`EventShared::host_use`), until it goes.
struct HostUse<'a>(&'a EventShared);

impl Drop for HostUse<'_> {
    fn drop(&mut self) {
        let last = {
            let mut uses = lock(&self.0.uses);
            uses.under_way -= 1;
            uses.under_way == 0 && uses.lost
        };
        if last {
            self.0.release_lost();
        }
    }
}

/// An event's parts on its device's lost generation, which a recovery carried it from
/// (`device/carry.rs`): its semaphore and its handle of the generation.
struct LostEvent(Weak<EventShared>);

impl LostParts for LostEvent {
    fn release(self: Box<Self>) {
        let Some(event) = self.0.upgrade() else {
            return;
        };
        let now = {
            let mut uses = lock(&event.uses);
            uses.lost = true;
            uses.under_way == 0
        };
        // Else the last call under way lets them go.
        if now {
            event.release_lost();
        }
    }
}

struct EventState {
    /// The signal indices' Metal values, by `VkQueue` index.
    /// Its kept signals' command buffers are named by their follows (`command_buffer/follow.rs`).
    timeline: EventTimeline<usize, Arc<Follow>>,
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
        // Carried to the next generation, its semaphore went (`EventShared::release_lost`).
        if self.device.gone() {
            return;
        }
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
    /// `settable` for a shared event (`MTLSharedEvent`), whose value the host may set; its
    /// semaphore made exportable as `export` names where given (`exported_semaphore.rs`).
    fn new(
        device: &Device,
        settable: bool,
        export: Option<vk::ExternalSemaphoreHandleTypeFlags>,
    ) -> Result<Arc<EventShared>, Error> {
        if device.is_lost() {
            return Err(device.lost_refusal());
        }
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut exportable = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(export.unwrap_or(vk::ExternalSemaphoreHandleTypeFlags::empty()));
        let mut info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        if export.is_some() {
            info = info.push_next(&mut exportable);
        }
        // SAFETY: the device is live and enables `timelineSemaphore` (created devices enable it;
        // `Device::from_raw` requires it); an export's handle type is one the device exports a
        // timeline semaphore as (the caller's).
        let semaphore =
            unsafe { device.shared().vk().create_semaphore(&info, None) }.map_err(vulkan)?;
        device.shared().interop.timeline_made(semaphore);
        let event = Arc::new(EventShared {
            device: MadeOn::new(device.clone()),
            semaphore,
            exportable: export.is_some(),
            state: Mutex::new(EventState {
                timeline: EventTimeline::new(settable),
                notifications: Vec::new(),
                parked: 0,
            }),
            changed: Condvar::new(),
            label: Mutex::new(None),
            next: OnceLock::new(),
            uses: Mutex::default(),
        });
        // Kept for the device's recovery; one made after its recovery took this generation's
        // events is carried here, by its maker (`device/carry.rs`).
        if device.shared().carry.event(&event) {
            device.carry_late(|serving| event.carry(serving));
        }
        Ok(event)
    }

    /// The event as the device's serving generation holds it: itself, or where its device's
    /// recovery carried it (`device/carry.rs`), the event it was carried to, every call of the
    /// program's going there.
    pub(super) fn serving(self: &Arc<Self>) -> &Arc<EventShared> {
        let mut event = self;
        while let Some(next) = event.next.get() {
            event = next;
        }
        event
    }

    /// [`serving`](Self::serving), its state locked: the event a carry did not pass while the
    /// lock is held, as a carry sets `next` under the lock of the event it carries.
    fn serving_locked(self: &Arc<Self>) -> (&Arc<EventShared>, MutexGuard<'_, EventState>) {
        let mut event = self.serving();
        loop {
            let state = lock(&event.state);
            match event.next.get() {
                None => return (event, state),
                Some(next) => {
                    drop(state);
                    event = next;
                }
            }
        }
    }

    /// Carry the event to `to`, its device's next generation, by the recovery that made it
    /// (`device/carry.rs`): a new timeline semaphore there, at the value the host saw (its floor
    /// included; the lost device's signals the host never saw never land), with its label and the
    /// listeners' notifications not yet due, which come due there. Every call of the program's
    /// goes to it from then on, and the host waits under way go on with it. Refused for an event
    /// whose semaphore was made exportable (`exported_semaphore.rs`): its importers hold the lost
    /// one, whose state Vulkan leaves undefined (vkspec 5.2.3), so it stays the lost device's
    /// (`RuntimeGap::MadeBeforeRecovery` where it meets the next).
    pub(crate) fn carry(self: &Arc<Self>, to: &Device) -> Result<(), Error> {
        if self.exportable {
            return Err(Error::NotImplemented(RuntimeGap::MadeBeforeRecovery));
        }
        let settable = lock(&self.state).timeline.settable();
        let next = EventShared::new(to, settable, None)?;
        {
            let mut state = lock(&self.state);
            if self.next.get().is_some() {
                // Carried already: by its maker, as it was made after the recovery took the
                // generation's events.
                return Ok(());
            }
            let mut carried = lock(&next.state);
            carried.timeline = state.timeline.carried();
            carried.notifications.append(&mut state.notifications);
            drop(carried);
            let label = lock(&self.label).clone();
            *lock(&next.label) = label;
            let _set = self.next.set(Arc::clone(&next));
            // The host waits asleep on it go on with the event carried.
            self.changed.notify_all();
        }
        // Its semaphore here goes once the lost generation is at rest and no host call on it is
        // under way.
        let from = self.device.shared();
        from.carry
            .lost(from, Box::new(LostEvent(Arc::downgrade(self))));
        let due = due(&mut lock(&next.state));
        next.post(due);
        Ok(())
    }

    /// A host call on the semaphore, under way until the result goes: `None` once the event was
    /// carried to its device's next generation and its semaphore here went or goes as the calls
    /// under way end, which the caller then makes on the event carried, or answers without.
    fn host_use(&self) -> Option<HostUse<'_>> {
        let mut uses = lock(&self.uses);
        if uses.lost {
            return None;
        }
        uses.under_way += 1;
        Some(HostUse(self))
    }

    /// Destroy the semaphore of an event carried to its device's next generation, on the lost one
    /// at rest, once no host call on it is under way (`HostUses`), and let go of its handle of
    /// that generation: none of the generation's batches is pending (vkspec 5.2.3: a wait for one
    /// on a lost device returns in finite time), and a host wait on the semaphore that began
    /// before the loss returned ("If device loss occurs (see Lost Device) before the timeout has
    /// expired, vkWaitSemaphores must return in finite time with either VK_SUCCESS or
    /// VK_ERROR_DEVICE_LOST", vkspec 7.4.4). Left to the process's end while it exits.
    fn release_lost(&self) {
        let Some(busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        shared.interop.timeline_gone(self.semaphore);
        // SAFETY: no batch or host call uses the semaphore any more (above), and none begins.
        unsafe { release::object(shared.vk(), self.semaphore) };
        drop(busy);
        // SAFETY: every call of the program's goes to the event carried; no host call on this one
        // begins again, and none of the generation's command buffers records.
        unsafe { self.device.let_go() };
    }

    /// The device.
    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    /// The timeline semaphore.
    pub(super) fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }

    /// Whether the semaphore was made exportable (`exported_semaphore.rs`).
    pub(super) fn exportable(&self) -> bool {
        self.exportable
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

    /// The batch of a kept signal was submitted: on `VkQueue` `queue` at `batch` on its timeline;
    /// `signaller` is its command buffer where the batch ends it.
    pub(super) fn keep(
        &self,
        index: u64,
        value: u64,
        batch: (usize, u64),
        signaller: Option<Arc<Follow>>,
        behind_merge: bool,
    ) {
        (lock(&self.state).timeline).keep(index, value, batch, signaller, behind_merge);
    }

    /// What a batch that waits for `value` completes after: the batch whose kept GPU signal
    /// reaches it while that one's completion turn has not run, and that batch's command buffer
    /// while its transition to `Completed` has not run up to its completed handlers
    /// (`EventTimeline::signallers`).
    pub(super) fn signallers(&self, value: u64) -> (Option<(usize, u64)>, Option<Arc<Follow>>) {
        lock(&self.state).timeline.signallers(value)
    }

    /// The batch whose kept signal is `index` finished with a failed wait's drain, its signal not
    /// seen (`command_buffer/completion.rs`): no batch finishes after it.
    pub(super) fn drained(&self, index: u64) {
        lock(&self.state).timeline.finished(index);
    }

    /// The transition to `Completed` of the command buffer whose last batch's kept signal is
    /// `index` ran up to its completed handlers: no command buffer completes after it any more.
    pub(super) fn released(&self, index: u64) {
        lock(&self.state).timeline.released(index);
    }

    /// The semaphore reached `index`: wake the host's waits and post the notifications that came
    /// due.
    pub(super) fn reached(self: &Arc<Self>, index: u64) {
        let due = {
            let mut state = lock(&self.state);
            // Called by the completion turn of the batch that signals it alone.
            state.timeline.finished(index);
            if !state.timeline.observe(index) {
                return;
            }
            self.rose(&state);
            due(&mut state)
        };
        self.post(due);
        self.woke();
    }

    /// Reserve the next signal index for a signal of `value` by one of the caller's submissions
    /// (`event_timeline.rs`, `EventTimeline::signal_apart`), under the scheduler's lock, so no
    /// batch takes that index meanwhile; where it is kept, the device's watcher waits for it to
    /// land, and the scheduler looks again, as a batch held for the value may now wait for it on
    /// the GPU. Refused where the device is lost, or where the watcher's thread cannot start then:
    /// nothing is reserved.
    pub(super) fn reserve_signal(self: &Arc<Self>, value: u64) -> Result<Signal, Error> {
        let Some(_use) = self.host_use() else {
            // Carried: its semaphore here went.
            return self.serving().reserve_signal(value);
        };
        if self.device.is_lost() {
            return Err(self.device.lost_refusal());
        }
        let shared = self.device.shared();
        let watcher = &shared.interop.caller_signals;
        let mut watched = lock(&watcher.state);
        if !watched.running {
            let thread = Arc::clone(watcher);
            exit_guard::spawn_user_interactive(
                std::thread::Builder::new().name("metal2vulkan-caller-signals".into()),
                move || watch(&thread),
            )
            .map_err(|_| Error::OutOfMemory(crate::error::MemoryKind::Host))?;
            watched.running = true;
        }
        let signal = (shared.scheduler).under_lock(|| {
            let mut state = lock(&self.state);
            // Not where the device's recovery carried the event meanwhile (`device/carry.rs`).
            (self.next.get().is_none()).then(|| state.timeline.signal_apart(value))
        });
        let Some(signal) = signal else {
            // Reserved on the event carried, which every call goes to from now on.
            drop(watched);
            return self.serving().reserve_signal(value);
        };
        if let Signal::Kept { index, .. } = signal {
            watched.pending.push_back((Arc::clone(self), index));
            watcher.wake.notify_all();
        }
        drop(watched);
        if matches!(signal, Signal::Kept { .. }) {
            self.woke();
        }
        Ok(signal)
    }

    /// Whether the semaphore reached `index` within `slice`, on the host: `None` where it never
    /// will as far as the runtime can tell (a lost device, or the process exiting, when the
    /// runtime makes no Vulkan call).
    fn wait_landed(&self, index: u64, slice: Duration) -> Option<bool> {
        let _busy = exit_guard::enter()?;
        if self.device.is_lost() {
            return None;
        }
        let semaphores = [self.semaphore];
        let values = [index];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        let nanos = u64::try_from(slice.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: the semaphore is the device's and live while `self` is.
        let waited = unsafe { self.device.shared().vk().wait_semaphores(&info, nanos) };
        exit_guard::called(Call::WaitSemaphores);
        match waited {
            Ok(()) => Some(true),
            Err(vk::Result::TIMEOUT) => Some(false),
            Err(_) => None,
        }
    }

    /// A host wait returned for the semaphore to reach `index`, one of the caller's signals
    /// (`EventTimeline::landed`): wake the host's waits and post the notifications that came due,
    /// as [`reached`](Self::reached) does.
    fn landed(self: &Arc<Self>, index: u64) {
        let due = {
            let mut state = lock(&self.state);
            if !state.timeline.landed(index) {
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
        // Carried, its semaphore here gone: the value it was carried at.
        let host_use = self.host_use();
        if let Some(busy) = host_use.as_ref().and_then(|_| exit_guard::enter()) {
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
        let (event, due) = {
            let (event, mut state) = self.serving_locked();
            if !state.timeline.set(value) {
                return;
            }
            event.rose(&state);
            (event, due(&mut state))
        };
        event.post(due);
        event.woke();
    }

    /// Wait until the value reaches `value`, at most `timeout`; a timeout past what the host's
    /// clock holds (Metal's `UINT64_MAX` milliseconds) has no deadline. Once the process is
    /// exiting, it answers at once from the value last seen: neither a completion turn nor the
    /// counter raises it any more (`exit_guard`).
    pub(super) fn wait_until(self: &Arc<Self>, value: u64, timeout: Duration) -> bool {
        let deadline = Instant::now().checked_add(timeout);
        loop {
            // The event as the device's serving generation holds it, at each turn: a recovery
            // that carries it wakes the wait, which goes on with the event carried.
            if self.serving().value() >= value {
                return true;
            }
            let (event, mut state) = self.serving_locked();
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
            let mut state = event
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
        let (event, due) = {
            let (event, mut state) = self.serving_locked();
            state.notifications.push(notification);
            (event, due(&mut state))
        };
        event.post(due);
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

/// The caller's reserved signals one device's watcher waits for (the module docs), in reservation
/// order, each its event and index.
#[derive(Default)]
pub(super) struct CallerSignals {
    state: Mutex<Watched>,
    /// Signalled as a signal is reserved.
    wake: Condvar,
}

#[derive(Default)]
struct Watched {
    pending: VecDeque<(Arc<EventShared>, u64)>,
    /// The watcher's thread runs: a reservation starts it where it does not.
    running: bool,
}

/// The watcher's thread (the module docs): wait for each reserved signal to land, in reservation
/// order, and observe it; end once none was left for [`WATCH_LINGER`], or where none will land (a
/// lost device, the process exiting), letting go of every event it held.
fn watch(signals: &CallerSignals) {
    loop {
        let (event, index) = {
            let mut watched = lock(&signals.state);
            loop {
                if let Some((event, index)) = watched.pending.front() {
                    break (Arc::clone(event), *index);
                }
                let (now, waited) = signals
                    .wake
                    .wait_timeout(watched, WATCH_LINGER)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                watched = now;
                if waited.timed_out() && watched.pending.is_empty() {
                    watched.running = false;
                    return;
                }
            }
        };
        // A carried event's semaphore here went: as on the lost device it was carried from.
        let landed = event.host_use().and_then(|_use| {
            let landed = event.wait_landed(index, WATCH_SLICE);
            if landed == Some(true) {
                event.landed(index);
            }
            landed
        });
        match landed {
            Some(true) => {
                lock(&signals.state).pending.pop_front();
            }
            Some(false) => {}
            None => {
                let mut watched = lock(&signals.state);
                watched.pending.clear();
                watched.running = false;
                return;
            }
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
        self.at_serving(|device| device.new_event_now())
    }

    /// [`Device::new_event`] on the generation that serves (`device/line.rs`).
    pub(crate) fn new_event_now(&self) -> Result<Event, Error> {
        Ok(Event(EventShared::new(self, false, None)?))
    }

    /// `newSharedEvent`.
    pub fn new_shared_event(&self) -> Result<SharedEvent, Error> {
        self.at_serving(|device| device.new_shared_event_now())
    }

    /// [`Device::new_shared_event`] on the generation that serves (`device/line.rs`).
    pub(crate) fn new_shared_event_now(&self) -> Result<SharedEvent, Error> {
        Ok(SharedEvent(EventShared::new(self, true, None)?))
    }

    /// [`Device::new_shared_event`], its semaphore made exportable as `export`
    /// (`exported_semaphore.rs`), a handle type the device exports a timeline semaphore as.
    pub(super) fn new_exported_shared_event(
        &self,
        export: vk::ExternalSemaphoreHandleTypeFlags,
    ) -> Result<SharedEvent, Error> {
        Ok(SharedEvent(EventShared::new(self, true, Some(export))?))
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
        self.at_serving(|device| device.new_shared_event_with_handle_now(handle))
    }

    /// [`Device::new_shared_event_with_handle`] on the generation that serves (`device/line.rs`).
    pub(crate) fn new_shared_event_with_handle_now(
        &self,
        handle: &SharedEventHandle,
    ) -> Result<SharedEvent, Error> {
        if !handle.0.serving().device.same(self) {
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
        &self.0.serving().device
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        *lock(&self.0.serving().label) = Some(label.to_owned());
    }

    /// `label`.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.serving().label).clone()
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
        *lock(&self.0.serving().label) = Some(label.to_owned());
    }

    /// `label`.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.serving().label).clone()
    }

    /// `signaledValue`. Once the process is exiting, the value last seen: the runtime makes no
    /// Vulkan call then.
    pub fn signaled_value(&self) -> u64 {
        self.0.serving().value()
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
        lock(&self.0.serving().label).clone()
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
