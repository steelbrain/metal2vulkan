//! Freed memory that waits for the batches in flight, and buffers of addresses that go with no
//! submission inside its `vkQueueSubmit`, on a device whose driver fails pending work when memory
//! a device-address buffer lay in is freed
//! (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), or a submission when such a
//! buffer is destroyed as it encodes (`RuntimeDriverDefect::DestroyedAddressBufferFailsSubmission`):
//! MoltenVK, which has both. Where the driver has either, both hold.
//!
//! - **The defect.** MoltenVK makes every `VkBuffer` of `SHADER_DEVICE_ADDRESS` usage resident
//!   (`useResource:`) in every encoder whose pipeline reads device addresses
//!   (`MVKDevice::encodeGPUAddressableBuffers`), in Metal command buffers that retain none of it,
//!   and lets go of the buffer's Metal memory as its `VkDeviceMemory` is freed: a Metal command
//!   buffer encoded before the free and not yet run fails (`Invalid Resource`, Metal's code 9).
//!   Vulkan allows the free, as no submitted command refers to the memory
//!   (VUID-vkFreeMemory-memory-00677). Destroying the buffer alone, its memory kept, takes it out
//!   of the driver's list. The runtime destroys a block's buffer of addresses only once every
//!   command buffer that may reach an address in the block completed, as
//!   VUID-vkDestroyBuffer-buffer-00922 asks: each holds what it may reach there until then (a
//!   buffer's pages, `buffer.rs`'s `FixedPages`, by its commands or its table of live buffers; a
//!   pool's block, `gpu_pools.rs`'s `PoolLease`), and a block goes only once nothing lies in it.
//!   But the destroy fails a submission whose `vkQueueSubmit` encodes on another thread meanwhile
//!   (`DestroyedAddressBufferFailsSubmission`, `driver_defects.rs`): on an Apple GPU each such
//!   `VkBuffer` is a Metal buffer of its own, which the destroy releases after the encoder listed
//!   it.
//! - **The wait.** There a block of the runtime's memory that goes (`memory.rs`) has its buffer of
//!   addresses destroyed at once, under the gate's write lock (below), and its memory waits,
//!   parked, until each `VkQueue`'s timeline
//!   reaches the value of the last batch submitted there as the block went. Vulkan's semaphore
//!   signal operations "defined by vkQueueSubmit or vkQueueSubmit2 additionally include all
//!   commands that occur earlier in submission order" in their first synchronization scope, so
//!   at that value every submission there before it completed. A host import's deallocator and
//!   keep-alive, which go after its memory is freed (`buffer.rs`), wait with it, the deallocator's
//!   wait announced (`host-deallocator-delayed`).
//! - **Each batch noted.** A batch's `vkQueueSubmit` and the store of its value run under the
//!   gate's read lock, inside the queue's lock (`relay.rs`); a block's buffer of addresses is
//!   destroyed, and the values noted, under the write lock ([`Parking::note_destroying`]). So no
//!   batch is encoding as the buffer goes, a batch whose encoding listed it was submitted before
//!   and is noted, and a batch submitted after lists none.
//! - **Lock order.** The gate comes after the memory's lock, never before it: a block goes under
//!   the memory's lock (`Memory::let_go`) and then takes the write lock; a batch's or an own
//!   submission's gated section takes the read lock inside its queue's lock and runs only the
//!   `vkQueueSubmit`, the log of its fence (`relay.rs`'s `submit_logged`), the counts and the
//!   stores of its value (`device/submit.rs`), none of which takes the memory's lock. So no thread
//!   inside the gate waits for the memory's lock, and a block that goes waits only for the
//!   submissions already inside.
//! - **And the device's own submissions.** Those that run command buffers outside the batches
//!   (`device/submit.rs`) signal no queue's timeline: each takes a number before its
//!   `vkQueueSubmit` and gives it back once its submitter saw it complete ([`OwnSubmission`]),
//!   and a block waits too for every one numbered before it went. Their `vkQueueSubmit` runs under
//!   the gate's read lock too, inside the queue's lock, as a batch's does. Its sparse binds and
//!   the staging queue's copies run no pipeline, so no encoder of theirs lists a buffer.
//! - **And the program's.** The runtime sees neither what the program's own submissions list
//!   (`CommandQueue::with_vk_queue`) nor when they complete, so each use of a queue is marked
//!   ([`ProgramUses`]): under the queue's lock, before the program's code runs, the use takes the
//!   next value of a timeline of the queue's own for such marks, and after it a submission of no
//!   work signals that value, which Vulkan's signal order completes after every submission the
//!   use made. A block waits too for each queue's value taken before it went: a submission the
//!   program made before the block's buffers were destroyed lies in a use that took its value
//!   before. While marks are pending a handler thread waits for them ([`Watch`]), and posts the
//!   release of the blocks whose values were reached as each completes. A mark whose submission
//!   fails is reached once the queue is idle. The program's code runs outside the gate, which a
//!   release inside it would wait for on its own thread: a buffer of addresses that goes while
//!   the program's own `vkQueueSubmit` encodes on another thread can fail that submission.
//! - **Freed off the runtime's completion threads.** A queue's waiter that publishes a value while
//!   blocks wait posts their release to a handler thread ([`Unpark`]), which frees the blocks it
//!   reached and then lets go of what waited for them; a block whose values were reached as it
//!   went is freed at once, on the device's free thread (`free_thread.rs`) where it is no import
//!   and nothing waits for it, else where it went. At the device's teardown, its queues idle, every block
//!   left is freed (`Memory::destroy`).
//! - **Holding nothing of the device.** The watch and the releases hold only the parts of the
//!   device they work with ([`Parts`]), never a handle: posted as the program's wait returns, or
//!   after its last use of a queue, one holding a handle would hold the device's last once the
//!   program dropped its own, and the device would go on a handler thread after the program's
//!   last drop returned, where a consumer that adopted its `VkDevice` destroys it then
//!   ("Host access to `device` must be externally synchronized", and every object made from it
//!   is destroyed before it: VUID-vkDestroyDevice-device-05137). So the program's last drop tears
//!   the device down before it returns, as Metal's last release of a device ends it. The teardown
//!   closes the parking first ([`Parking::close`]) and then waits for the handler threads to end
//!   (`Contexts::stop`); one that runs after it, on the thread the teardown ran on, finds the
//!   parking closed and calls nothing.

use crate::command_buffer::Item;
use crate::contexts::Contexts;
use crate::device::functions::Functions;
use crate::device::{lock, Device};
use crate::exit_guard::{self, Call};
use crate::memory::Memory;
use ash::vk;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard};
use std::time::Duration;

/// How long the watch of the program's marks waits at a time, between its checks of the exit.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// A device's record of its batches for the blocks that wait (the module docs).
pub(crate) struct Parking {
    /// Read around each batch's submission and the store of its value, written around a block's
    /// noting of the values.
    gate: RwLock<()>,
    /// Per `VkQueue`: the value of the last batch submitted there.
    submitted: Box<[AtomicU64]>,
    /// Per `VkQueue`: the value through which every batch submitted there completed, as the host
    /// saw.
    completed: Box<[AtomicU64]>,
    /// The blocks waiting (`Memory`'s, counted here so that a waiter reads it with no lock).
    waiting: AtomicUsize,
    /// A release is posted to a handler thread and has not begun: no other is posted.
    posted: AtomicBool,
    /// The device's own submissions: the next one's number, and those in flight.
    own: Mutex<Own>,
    /// Per `VkQueue`: the program's uses of it, marked.
    program: Box<[ProgramUses]>,
    /// A [`Watch`] of the program's marks is posted or running: no other is posted.
    watching: AtomicBool,
    /// The device's teardown began ([`close`](Self::close)): a watch or release that runs from
    /// here on calls nothing.
    closed: AtomicBool,
    /// The crate's device tests' hold of the watch and the releases as they begin
    /// (`tests/teardown_threads.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) held: held::Held,
}

/// The program's uses of one `VkQueue` (`CommandQueue::with_vk_queue`), each marked by a value
/// of a timeline of their own (the module docs).
#[derive(Default)]
struct ProgramUses {
    /// The timeline the marks signal, made at the queue's first marked use.
    timeline: OnceLock<vk::Semaphore>,
    /// The value the last use took, under the queue's lock, before the program's code ran.
    taken: AtomicU64,
    /// The value through which the marks completed, as the host saw.
    completed: AtomicU64,
}

/// The device's own submissions ([`OwnSubmission`]).
#[derive(Default)]
struct Own {
    /// The next one's number.
    next: u64,
    /// The numbers of those whose submitter has not yet seen them complete.
    in_flight: BTreeSet<u64>,
}

/// What a block noted as it went: the value of each `VkQueue`'s last batch submitted, the number
/// the device's next own submission takes, and the value each queue's last use by the program
/// took.
pub(crate) struct Noted {
    values: Box<[u64]>,
    own: u64,
    program: Box<[u64]>,
}

impl Parking {
    /// The record of a device with `queues` `VkQueue`s, none submitted to.
    pub(crate) fn new(queues: usize) -> Parking {
        Parking {
            gate: RwLock::new(()),
            submitted: (0..queues).map(|_| AtomicU64::new(0)).collect(),
            completed: (0..queues).map(|_| AtomicU64::new(0)).collect(),
            waiting: AtomicUsize::new(0),
            posted: AtomicBool::new(false),
            own: Mutex::new(Own::default()),
            program: (0..queues).map(|_| ProgramUses::default()).collect(),
            watching: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            held: held::Held::default(),
        }
    }

    /// The device's teardown begins: a watch or release that runs from here on calls nothing
    /// (the module docs).
    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    /// Let the tests' hold go once the device's teardown began, or once it was let go already.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn release_once_closed(&self) {
        self.held.release_once(&self.closed);
    }

    /// The gate's read lock, held around a batch's `vkQueueSubmit` and the store of its value
    /// ([`submitted`](Self::submitted)), and around an own submission's `vkQueueSubmit`, inside
    /// the queue's lock.
    pub(crate) fn submitting(&self) -> RwLockReadGuard<'_, ()> {
        self.gate
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A batch signalling `value` was submitted to `VkQueue` `queue`, under
    /// [`submitting`](Self::submitting)'s lock.
    pub(crate) fn submitted(&self, queue: usize, value: u64) {
        self.submitted[queue].fetch_max(value, Ordering::SeqCst);
    }

    /// The host saw `VkQueue` `queue`'s timeline reach `value`.
    pub(crate) fn completed(&self, queue: usize, value: u64) {
        self.completed[queue].fetch_max(value, Ordering::SeqCst);
    }

    /// Under the gate's write lock, so no batch and no own submission is inside its
    /// `vkQueueSubmit`: `destroy` (a block's buffer of addresses going, the module docs), then the
    /// value of each `VkQueue`'s last batch submitted, the device's next own submission's number
    /// and each queue's last value taken by the program's uses.
    pub(crate) fn note_destroying<T>(&self, destroy: impl FnOnce() -> T) -> (T, Noted) {
        let _gate = self
            .gate
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let destroyed = destroy();
        let values = (self.submitted.iter())
            .map(|value| value.load(Ordering::SeqCst))
            .collect();
        let noted = Noted {
            values,
            own: lock(&self.own).next,
            program: (self.program.iter())
                .map(|uses| uses.taken.load(Ordering::SeqCst))
                .collect(),
        };
        (destroyed, noted)
    }

    /// Whether every `VkQueue`'s timeline reached the value `noted` holds for it, every own
    /// submission numbered before `noted`'s completed, and every queue's marks of the program's
    /// uses reached the value `noted` holds for them.
    pub(crate) fn reached(&self, noted: &Noted) -> bool {
        let batches = (self.completed.iter())
            .zip(&noted.values)
            .all(|(completed, noted)| completed.load(Ordering::SeqCst) >= *noted);
        let program = (self.program.iter())
            .zip(&noted.program)
            .all(|(uses, noted)| uses.completed.load(Ordering::SeqCst) >= *noted);
        batches
            && program
            && lock(&self.own)
                .in_flight
                .first()
                .is_none_or(|first| *first >= noted.own)
    }

    /// A use of `VkQueue` `queue` by the program begins, under the queue's lock, before the
    /// program's code runs: the timeline its mark signals and the value it takes; `None` where the
    /// timeline cannot be made (the use is then not waited for). Inside a section of the exit
    /// guard.
    pub(crate) fn program_use(
        &self,
        queue: usize,
        device: &ash::Device,
    ) -> Option<(vk::Semaphore, u64)> {
        let uses = &self.program[queue];
        let timeline = match uses.timeline.get() {
            Some(&timeline) => timeline,
            None => {
                let mut kind = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                    .initial_value(0);
                let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
                // SAFETY: the device is live and enables `timelineSemaphore`.
                let made = unsafe { device.create_semaphore(&info, None) };
                exit_guard::called(Call::CreateSemaphore);
                let timeline = made.ok()?;
                // Set under the queue's lock, which every use takes: never set before.
                let _ = uses.timeline.set(timeline);
                timeline
            }
        };
        Some((timeline, uses.taken.fetch_add(1, Ordering::SeqCst) + 1))
    }

    /// The marks of `VkQueue` `queue`'s uses by the program completed through `value`, as the
    /// host saw.
    pub(crate) fn program_completed(&self, queue: usize, value: u64) {
        self.program[queue]
            .completed
            .fetch_max(value, Ordering::SeqCst);
    }

    /// Each queue's timeline of marks whose last value taken is not yet seen complete, with that
    /// value.
    fn program_pending(&self) -> Vec<(usize, vk::Semaphore, u64)> {
        (self.program.iter().enumerate())
            .filter_map(|(queue, uses)| {
                let taken = uses.taken.load(Ordering::SeqCst);
                let timeline = *uses.timeline.get()?;
                (uses.completed.load(Ordering::SeqCst) < taken).then_some((queue, timeline, taken))
            })
            .collect()
    }

    /// After a use of a queue by the program: where its mark, or another, is pending and no watch
    /// is posted, post one to a handler thread of `device`'s.
    pub(crate) fn watch(&self, device: &Device) {
        if self.program_pending().is_empty() || self.watching.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(parts) = Parts::of(device) {
            parts.post(Box::new(Watch(parts.clone())));
        }
    }

    /// Destroy the timelines of the program's marks: the device is torn down, idle.
    ///
    /// # Safety
    ///
    /// The timelines are `device`'s, and no submission or wait uses them.
    pub(crate) unsafe fn release(&self, device: &ash::Device) {
        for uses in &self.program {
            if let Some(&timeline) = uses.timeline.get() {
                // SAFETY: the caller's.
                unsafe { crate::release::object(device, timeline) };
            }
        }
    }

    /// An own submission of `device`'s, from before its `vkQueueSubmit` until the guard drops,
    /// once its submitter saw it complete (or gave up on it).
    pub(crate) fn own_submission<'a>(&'a self, device: &'a Device) -> OwnSubmission<'a> {
        let mut own = lock(&self.own);
        let number = own.next;
        own.next += 1;
        own.in_flight.insert(number);
        OwnSubmission {
            parking: self,
            device,
            number,
        }
    }

    /// One more block waits.
    pub(crate) fn waits(&self) {
        self.waiting.fetch_add(1, Ordering::SeqCst);
    }

    /// `count` blocks waited and are freed.
    pub(crate) fn freed(&self, count: usize) {
        self.waiting.fetch_sub(count, Ordering::SeqCst);
    }

    /// After the host saw a value of a queue's: where blocks wait and no release is posted, post
    /// one to a handler thread of `device`'s, which frees those whose values were reached.
    pub(crate) fn after_completion(&self, device: &Device) {
        if self.unpark_due() {
            if let Some(parts) = Parts::of(device) {
                parts.post(Box::new(Unpark(parts.clone())));
            }
        }
    }

    /// Whether a release is to be posted: blocks wait, and none is posted (it is then).
    fn unpark_due(&self) -> bool {
        self.waiting.load(Ordering::SeqCst) != 0 && !self.posted.swap(true, Ordering::SeqCst)
    }
}

/// The parts of a device that its watch of the program's marks and its releases of freed memory
/// work with ([`Watch`], [`Unpark`]), none of which holds the device (the module docs): its
/// parking, its memory, a share of its functions (reached through [`Functions::get`] alone, as
/// the device's own are), and its handler threads.
#[derive(Clone)]
struct Parts {
    parking: Arc<Parking>,
    memory: Arc<Mutex<Memory>>,
    functions: Arc<Functions>,
    contexts: Contexts<Item>,
}

impl Parts {
    /// `device`'s, where it parks freed memory.
    fn of(device: &Device) -> Option<Parts> {
        let shared = device.shared();
        Some(Parts {
            parking: Arc::clone(shared.parking.get()?),
            memory: Arc::clone(&shared.memory),
            functions: shared.functions(),
            contexts: device.elsewhere().0,
        })
    }

    /// Run `held`'s drop on one of the handler threads, in a context of its own.
    fn post(&self, held: Box<dyn Send>) {
        self.contexts
            .post(self.contexts.fresh(), Item::Release(held));
    }
}

/// An own submission of the device's in flight ([`Parking::own_submission`]): as it drops, the
/// blocks that waited for it alone may go, on a handler thread.
pub(crate) struct OwnSubmission<'a> {
    parking: &'a Parking,
    device: &'a Device,
    number: u64,
}

impl Drop for OwnSubmission<'_> {
    fn drop(&mut self) {
        lock(&self.parking.own).in_flight.remove(&self.number);
        self.parking.after_completion(self.device);
    }
}

/// The watch of the program's marks ([`Parking::watch`]): on a handler thread, it waits for the
/// marks pending, a slice at a time, publishes each queue's value reached, and posts the release
/// of the blocks that waited for them, until none is pending. It ends where the device is lost
/// (the blocks then wait for the device's teardown), once its teardown began, and once the
/// process is exiting.
struct Watch(Parts);

impl Drop for Watch {
    fn drop(&mut self) {
        let parking = &*self.0.parking;
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        parking.held.pass();
        loop {
            if parking.closed.load(Ordering::SeqCst) {
                return;
            }
            let pending = parking.program_pending();
            if pending.is_empty() {
                parking.watching.store(false, Ordering::SeqCst);
                // A use that found the watch on before the store is seen here.
                if parking.program_pending().is_empty()
                    || parking.watching.swap(true, Ordering::SeqCst)
                {
                    return;
                }
                continue;
            }
            let Some(busy) = exit_guard::enter() else {
                return;
            };
            let raw = self.0.functions.get();
            let (timelines, values): (Vec<vk::Semaphore>, Vec<u64>) = pending
                .iter()
                .map(|&(_, timeline, value)| (timeline, value))
                .unzip();
            let info = vk::SemaphoreWaitInfo::default()
                .flags(vk::SemaphoreWaitFlags::ANY)
                .semaphores(&timelines)
                .values(&values);
            // SAFETY: the timelines are the device's, live until its teardown, which closed the
            // parking before and waits for this watch to end (the module docs).
            let waited = unsafe { raw.wait_semaphores(&info, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitSemaphores);
            if !matches!(waited, Ok(()) | Err(vk::Result::TIMEOUT)) {
                parking.watching.store(false, Ordering::SeqCst);
                return;
            }
            for &(queue, timeline, _) in &pending {
                // SAFETY: as above.
                if let Ok(value) = unsafe { raw.get_semaphore_counter_value(timeline) } {
                    parking.program_completed(queue, value);
                }
                exit_guard::called(Call::GetSemaphoreCounterValue);
            }
            drop(busy);
            if parking.unpark_due() {
                self.0.post(Box::new(Unpark(self.0.clone())));
            }
        }
    }
}

/// A block's memory waiting for the batches in flight as it went, and what goes after it.
pub(crate) struct Parked {
    pub(crate) noted: Noted,
    pub(crate) memory: vk::DeviceMemory,
    /// What waits for the memory to be freed (a host import's deallocators and keep-alive).
    pub(crate) after: Option<Box<dyn Send>>,
}

impl Parked {
    /// Free the memory: what waited for it, to let go of outside the exit guard's section the
    /// free runs in (it runs the program's code).
    ///
    /// # Safety
    ///
    /// The memory is `device`'s, nothing is placed in it, and every batch that may list it
    /// completed.
    pub(crate) unsafe fn release(self, device: &ash::Device) -> Option<Box<dyn Send>> {
        let Parked { memory, after, .. } = self;
        // SAFETY: the caller's.
        unsafe { crate::release::object(device, memory) };
        after
    }
}

/// The release a queue's waiter posts ([`Parking::after_completion`]): on a handler thread, the
/// blocks whose values were reached are freed, and what waited for them goes. Once the device's
/// teardown began it frees nothing: the teardown frees every block left.
struct Unpark(Parts);

impl Drop for Unpark {
    fn drop(&mut self) {
        let Parts {
            parking,
            memory,
            functions,
            ..
        } = &self.0;
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        parking.held.pass();
        // Cleared first: a value published from here on posts another release.
        parking.posted.store(false, Ordering::SeqCst);
        if parking.closed.load(Ordering::SeqCst) {
            return;
        }
        // The frees are a section of the exit guard: while the process exits, the blocks stay
        // to its end (`exit_guard`).
        let Some(busy) = exit_guard::enter() else {
            return;
        };
        let reached = lock(memory).reached();
        let after: Vec<Box<dyn Send>> = (reached.into_iter())
            // SAFETY: its values were reached, so every batch that may list it completed; the
            // device lives, its teardown waiting for this release to end (the module docs).
            .filter_map(|parked| unsafe { parked.release(functions.get()) })
            .collect();
        drop(busy);
        drop(after);
    }
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl Device {
    /// Let the memory's blocks that go from now on wait for the batches in flight, as where the
    /// driver has `FreedAddressMemoryFailsPendingWork`: the crate's tests' way to take that path
    /// on any device. Before the device's first batch.
    pub(crate) fn park_freed_memory(&self) {
        let shared = self.shared();
        let parking = shared
            .parking
            .get_or_init(|| std::sync::Arc::new(Parking::new(shared.queues.len())));
        lock(&shared.memory).park_by(std::sync::Arc::clone(parking));
    }
}

/// The crate's device tests' hold of a device's watch and releases (`tests/teardown_threads.rs`,
/// compiled under the validation crate's GPU safety cfgs alone): while held, each waits as it
/// begins, counted, until the hold is let go.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod held {
    use crate::device::lock;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Condvar, Mutex};
    use std::time::{Duration, Instant};

    /// The hold: whether held, and how many wait at it.
    #[derive(Default)]
    pub(crate) struct Held {
        state: Mutex<(bool, usize)>,
        changed: Condvar,
    }

    impl Held {
        /// Hold the watch and the releases that begin from now on.
        pub(crate) fn hold(&self) {
            lock(&self.state).0 = true;
        }

        /// Let them go.
        pub(crate) fn release(&self) {
            lock(&self.state).0 = false;
            self.changed.notify_all();
        }

        /// Let them go once `closed` is set, or once they were let go already.
        pub(crate) fn release_once(&self, closed: &AtomicBool) {
            while !closed.load(Ordering::SeqCst) && lock(&self.state).0 {
                std::thread::sleep(Duration::from_millis(1));
            }
            self.release();
        }

        /// Wait until `count` wait at the hold, failing past `within` (none past what the host's
        /// clock holds).
        pub(crate) fn until_waiting(&self, count: usize, within: Duration) {
            let deadline = Instant::now().checked_add(within);
            let mut state = lock(&self.state);
            while state.1 < count {
                let left = deadline.map_or(Duration::MAX, |deadline| {
                    deadline.saturating_duration_since(Instant::now())
                });
                assert!(!left.is_zero(), "{count} never waited at the hold");
                state = (self.changed.wait_timeout(state, left))
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
        }

        /// As a watch or release begins: wait while held, counted.
        pub(crate) fn pass(&self) {
            let mut state = lock(&self.state);
            state.1 += 1;
            self.changed.notify_all();
            while state.0 {
                state =
                    (self.changed.wait(state)).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            state.1 -= 1;
        }
    }
}
