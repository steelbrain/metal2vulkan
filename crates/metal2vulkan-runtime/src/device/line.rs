//! A device's line of generations: the one Metal device (`MTLDevice`) a program holds across its
//! GPU's recoveries, each recovery a new `VkDevice` behind it.
//!
//! Metal's `MTLDevice` survives a GPU recovery: the hung command buffer fails, the others in
//! flight fail as its victims, and the program goes on with the same device. Vulkan's device does
//! not: once a call returned `VK_ERROR_DEVICE_LOST`, the `VkDevice` serves nothing more (vkspec
//! 5.2.3, "Lost Device"), and only a new one serves again. So the runtime's device is a line of
//! generations, each a [`DeviceShared`] over its own `VkDevice`:
//!
//! - **Two kinds of handle.** A *root* handle (the device the program made, `Device::system_default`
//!   and the rest, and every clone of it) holds the line and no generation: every call made
//!   through it goes to the generation serving when it is made (the public calls'
//!   [`Device::current`]), and again to the one serving next where that one was lost under the
//!   call and refused it for that ([`Device::at_serving`]; a call that takes host memory or a
//!   range token, which it cannot take twice, is not refused for a loss once it took them,
//!   [`Device::at_serving_taking`]). A *generation* handle holds one generation: the device of
//!   every object made on it (a buffer's, a pipeline state's), through which the object's Vulkan
//!   calls reach its own `VkDevice` alone, its drop included. So no object's Vulkan handle ever
//!   reaches another generation's `VkDevice`, and an object made before a recovery is refused
//!   where it meets the new generation (`RuntimeGap::MadeBeforeRecovery`, `error.rs`).
//! - **The states.** [`State`]: the line serves one generation (`Live`); once it is lost, the
//!   first call that needs the device recovers it (`Lost`, then claimed), while every other call
//!   waits for that recovery, the first to wait past [`RECOVERY_NOTICE`] announcing it and waiting
//!   on; a line that cannot recover (a device the consumer made, `Device::from_raw`, or one whose
//!   new `VkDevice` failed) answers as the lost device does (`Failed`).
//! - **The recovery** ([`Line::recover`]), on the program's thread whose call claimed it: the lost
//!   generation's command buffers in flight fail through its completion path, the oldest
//!   incomplete batch's as the hang and the others as its victims (`command_buffer/completion.rs`);
//!   its queues waited idle, each under its lock; then a new `VkDevice` of the same physical
//!   device, options, queue families and extensions (`create.rs`, the same making as the first),
//!   whose device-level objects are made as the first generation's were, the shader cache's
//!   records kept on disk warm. The calls waiting go on with it, the command queues made before
//!   take a new place on it at their next command buffer (`queue.rs`), and it announces
//!   `Degradation::DeviceRecovered`.
//! - **Each recovery's notices are its own.** The kinds a device announced go on across its
//!   generations, each kind announced once for the device, but a recovery's notices
//!   (`Degradation::DeviceRecovered`) are announced at every recovery, whatever an earlier one
//!   announced: the recovery itself, and at most one each of a call's wait for it past
//!   [`RECOVERY_NOTICE`], its command buffers in flight taking past it to fail, and its new
//!   `VkDevice`'s failure. So a slow second recovery says so as the first did, and repeated losses
//!   announce a few notices each, never one per waiting call.
//! - **What goes on across generations**: the line's handles and holds (`holds.rs`), its counts
//!   (`WorkCounts`), the Metal address space with its reserved pools, the resource IDs and the
//!   `uniqueIdentifier`s, so nothing made after a recovery takes an address or an identity one made
//!   before holds.
//! - **At most two `VkDevice`s.** A generation is destroyed when its last handle goes: the lost
//!   one once the objects made on it went, and, where `Device::raw` of a root handle answered its
//!   handles, once a later such call answered the new one's (`Line::lend`), the new one serving
//!   meanwhile. Counted: `WorkCounts::vk_devices_live`. A second recovery while objects made
//!   before the first still live keeps their `VkDevice` as well, counted there too.

use super::create;
use super::{holds, lock, Device, DeviceOptions, DeviceShared, InstanceShared};
use crate::address::AddressSpace;
use crate::capabilities::DeviceProperties;
use crate::degradation::Announced;
use crate::error::{invalid, Error, Problem, RuntimeGap};
use crate::resource_ids::{PassingHolds, ResourceIds};
use crate::shader_cache::CacheSetup;
use crate::unique_identifier::UniqueIdentifiers;
use crate::work_counts::Counters;
use ash::vk;
use metal2vulkan::device::EnableSet;
use std::cell::Cell;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

/// How long a call waits for a recovery under way before it announces the wait
/// (`Degradation::DeviceRecovered`, once per recovery), and how long a recovery waits for its
/// command buffers in flight to fail before it announces that; each waits on past it.
pub(crate) const RECOVERY_NOTICE: Duration = Duration::from_secs(10);

/// What a line's every generation answers alike: they are made of the same physical device with
/// the same options. A root handle reads them without a generation; each generation holds a copy.
#[derive(Clone)]
pub(crate) struct Invariant {
    pub(crate) properties: DeviceProperties,
    pub(crate) compile: metal2vulkan::device::Device,
    pub(crate) enabled: EnableSet,
    pub(crate) extensions: Vec<CString>,
    /// The shader cache's directory (`Device::shader_cache_directory`).
    pub(crate) cache_directory: PathBuf,
}

/// How the line makes its next `VkDevice`: what the first was made of (`create.rs`). None for a
/// device the consumer made, which the runtime cannot make again.
pub(crate) struct Recipe {
    pub(super) instance: Arc<InstanceShared>,
    pub(super) physical: vk::PhysicalDevice,
    pub(super) options: DeviceOptions,
    pub(super) cache: CacheSetup,
}

/// What every generation of a line goes on with (the module docs).
pub(super) struct Carried {
    /// The Metal address space and its reserved pools.
    pub(super) addresses: Arc<Mutex<AddressSpace>>,
    /// The resource IDs of textures and sampler states, and the caller's reservations of them.
    pub(super) resource_ids: Arc<Mutex<ResourceIds>>,
    /// The passing holds a making waits out before it takes a resource ID: of every generation,
    /// as the IDs are (`resource_ids.rs`).
    pub(super) passing_holds: Arc<PassingHolds>,
    /// The `uniqueIdentifier`s.
    pub(super) identifiers: Arc<UniqueIdentifiers>,
    /// The counts of the commit path's work (`WorkCounts`).
    pub(super) counts: Arc<Counters>,
    /// The kinds of degradation announced, so each is announced once for the device, not once per
    /// generation (`degradation.rs`); a recovery's notices are announced at each recovery
    /// (`Device::degrade_recovered`, `Device::degrade_recovery_step`; the module docs).
    pub(super) announced: Arc<Announced>,
}

/// The generations' counts every generation shares, read without the line: which generation
/// serves, and how many `VkDevice`s the line made and keeps.
#[derive(Default)]
pub(crate) struct Lineage {
    /// The generation that serves, or that a recovery makes: a generation handle of an earlier
    /// one goes to the line for its calls ([`Device::current`]).
    current: AtomicU64,
    /// The line's `VkDevice`s alive (`WorkCounts::vk_devices_live`).
    live: AtomicU64,
    /// The line's `VkDevice`s made, the first among them (`WorkCounts::vk_devices_made`).
    made: AtomicU64,
    /// Its recoveries that made a new `VkDevice` (`WorkCounts::device_recoveries`).
    recoveries: AtomicU64,
}

impl Lineage {
    /// The generation that serves, or that a recovery makes. A relaxed load: a handle's check of
    /// its own generation against it is the one load a call pays where nothing was lost.
    pub(crate) fn current(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    /// A `VkDevice` of the line was made: its generation's number.
    pub(super) fn made(&self) -> u64 {
        self.live.fetch_add(1, Ordering::AcqRel);
        self.made.fetch_add(1, Ordering::AcqRel);
        self.current.load(Ordering::Acquire)
    }

    /// A `VkDevice` of the line was destroyed.
    pub(super) fn destroyed(&self) {
        self.live.fetch_sub(1, Ordering::AcqRel);
    }

    /// The `VkDevice`s alive, made, and the recoveries (`WorkCounts`).
    pub(crate) fn counts(&self) -> (u64, u64, u64) {
        (
            self.live.load(Ordering::Acquire),
            self.made.load(Ordering::Acquire),
            self.recoveries.load(Ordering::Acquire),
        )
    }
}

/// Where a line is (the module docs).
enum State {
    /// Its first `VkDevice` is being made: no handle of it exists yet.
    Making,
    /// It serves this generation.
    Live(Arc<DeviceShared>),
    /// This generation was lost; `claimed` once a call took its recovery on; `culprit` the
    /// command buffer its completion failed as the hang, once it named one; `waited` once a call
    /// announced its wait for this recovery past [`RECOVERY_NOTICE`], which no other call waiting
    /// for it announces again.
    Lost {
        lost: Arc<DeviceShared>,
        claimed: bool,
        culprit: Option<Culprit>,
        waited: bool,
    },
    /// It cannot recover (a device the consumer made, or one whose new `VkDevice` failed): this
    /// lost generation answers every call, as a lost device does.
    Failed(Arc<DeviceShared>),
    /// The program's last handle went.
    Gone,
}

/// The command buffer a loss named the hang (`command_buffer/completion.rs`), for the notice and
/// for the hold of its queue.
pub(crate) struct Culprit {
    /// Its command queue's part every generation of the queue shares (`queue.rs`).
    pub(crate) queue: Weak<crate::queue::QueueCommon>,
    /// Its queue's label, where it has one.
    pub(crate) label: Option<String>,
    /// The queue was picked among two or more with work in flight: an inference, the queue whose
    /// oldest incomplete batch was submitted first.
    pub(crate) inferred: bool,
    /// The command buffers that failed with it as its victims.
    pub(crate) victims: usize,
}

/// A device's line (the module docs).
pub(crate) struct Line {
    /// The registry of the handles every generation's share (`holds.rs`).
    pub(super) holds: Arc<holds::Holds>,
    pub(crate) invariant: Invariant,
    pub(super) carried: Carried,
    pub(crate) lineage: Arc<Lineage>,
    recipe: Option<Recipe>,
    state: Mutex<State>,
    /// Signalled as the line serves a generation again, or fails.
    changed: Condvar,
    /// The command queues made on the device, which each recovery has take a new place on the new
    /// generation at their next command buffer (`queue.rs`).
    queues: Mutex<Vec<Weak<crate::queue::QueueLine>>>,
    /// The generation a root handle's `Device::raw` lent (`Line::lend`).
    lent: Mutex<Lent>,
    /// What the device tests set of its recoveries ([`hooks`]), in their builds only.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) hooks: hooks::Hooks,
}

/// What a line lent a root handle's `Device::raw` ([`Line::lend`]).
enum Lent {
    /// No generation yet.
    Nothing,
    /// This generation, kept until a later call lends another.
    To(Arc<DeviceShared>),
    /// The program's last handle went ([`Line::release`]).
    Released,
}

impl Line {
    /// A new line, its first `VkDevice` being made.
    pub(super) fn new(
        holds: Arc<holds::Holds>,
        invariant: Invariant,
        announced: Announced,
        recipe: Option<Recipe>,
    ) -> Arc<Line> {
        Arc::new(Line {
            holds,
            invariant,
            carried: Carried {
                addresses: Arc::new(Mutex::new(AddressSpace::new())),
                resource_ids: Arc::default(),
                passing_holds: Arc::default(),
                identifiers: Arc::default(),
                counts: Arc::default(),
                announced: Arc::new(announced),
            },
            lineage: Arc::default(),
            recipe,
            state: Mutex::new(State::Making),
            changed: Condvar::new(),
            queues: Mutex::default(),
            lent: Mutex::new(Lent::Nothing),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            hooks: hooks::Hooks::default(),
        })
    }

    /// How long a call waits for a recovery under way, or a recovery for its command buffers in
    /// flight to fail, before it announces the wait: [`RECOVERY_NOTICE`], or in the device tests'
    /// builds what a test set (`hooks::Hooks::notice_after`).
    fn notice_after(&self) -> Duration {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if let Some(after) = *lock(&self.hooks.notice_after) {
            return after;
        }
        RECOVERY_NOTICE
    }

    /// Whether the line makes a new `VkDevice` once one is lost: a device the runtime made.
    pub(crate) fn recovers(&self) -> bool {
        self.recipe.is_some()
    }

    /// Whether the line went past `generation`, lost, to a later one it serves or recovers to:
    /// not where it cannot recover, nor where its new `VkDevice` failed.
    pub(crate) fn recovers_past(&self, generation: u64) -> bool {
        self.recovers()
            && self.lineage.current() != generation
            && !matches!(*lock(&self.state), State::Failed(_))
    }

    /// Its first generation, made: the line serves it.
    pub(super) fn first(&self, first: &Arc<DeviceShared>) {
        let mut state = lock(&self.state);
        if let State::Making = *state {
            *state = State::Live(Arc::clone(first));
        }
    }

    /// `lost`, a generation of the line, was lost: the line's next call that needs the device
    /// recovers it, where the line recovers, and each generation handle of it goes to the line
    /// for its calls from now on. Nothing where the line no longer serves it.
    pub(super) fn lose(&self, lost: &DeviceShared) {
        let mut state = lock(&self.state);
        let State::Live(serving) = &*state else {
            return;
        };
        if !std::ptr::eq(Arc::as_ptr(serving), lost) {
            return;
        }
        let serving = Arc::clone(serving);
        *state = match self.recovers() {
            true => {
                // Past the lost one: its handles go to the line, which recovers.
                (self.lineage.current).store(lost.generation + 1, Ordering::Release);
                State::Lost {
                    lost: serving,
                    claimed: false,
                    culprit: None,
                    waited: false,
                }
            }
            false => State::Failed(serving),
        };
        drop(state);
        self.changed.notify_all();
    }

    /// The generation a call of `device`'s goes to: the serving one, after the recovery of a lost
    /// one, made here where no call took it on yet, else waited for (the module docs); a line that
    /// cannot recover answers with its lost generation. A handle for the call, counted as the
    /// program's.
    pub(super) fn current(self: &Arc<Self>, device: &Device) -> Device {
        let mut began: Option<Instant> = None;
        let mut state = lock(&self.state);
        loop {
            match &mut *state {
                State::Live(serving) => {
                    // A generation the relay found lost (`relay.rs`), which no failed wait has
                    // reported yet.
                    if serving.lost() {
                        let serving = Arc::clone(serving);
                        drop(state);
                        self.lose(&serving);
                        state = lock(&self.state);
                        continue;
                    }
                    return holds::program(Arc::clone(serving));
                }
                State::Failed(lost) => return holds::program(Arc::clone(lost)),
                // A thread command buffers' completion depends on, which the recovery waits for,
                // neither recovers nor waits: its call reaches the lost generation, and answers as
                // a lost device does (`exit_guard::serving_completion`).
                State::Lost { lost, .. } if crate::exit_guard::serving_completion() => {
                    return holds::program(Arc::clone(lost));
                }
                State::Lost { lost, claimed, .. } if !*claimed => {
                    *claimed = true;
                    let lost = Arc::clone(lost);
                    drop(state);
                    return self.recover(lost);
                }
                State::Lost { lost, waited, .. } => {
                    let began = *began.get_or_insert_with(Instant::now);
                    let after = self.notice_after();
                    // The first call to wait past the bound for this recovery announces it, once
                    // per recovery whatever an earlier recovery announced (the module docs).
                    if !*waited && began.elapsed() >= after {
                        *waited = true;
                        let lost = Arc::clone(lost);
                        drop(state);
                        holds::program(lost).degrade_recovery_step(|| {
                            (
                                "a call made during the device's recovery".to_owned(),
                                format!(
                                    "it waited {:?} for the recovery under way, and waits on",
                                    began.elapsed()
                                ),
                            )
                        });
                        state = lock(&self.state);
                        continue;
                    }
                    let timeout = match *waited {
                        true => RECOVERY_NOTICE,
                        false => after.saturating_sub(began.elapsed()),
                    };
                    let (next, _) = (self.changed)
                        .wait_timeout(state, timeout)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state = next;
                }
                // No handle of a line in either exists but the caller's own: it answers.
                State::Making | State::Gone => return device.clone(),
            }
        }
    }

    /// Recover from the loss of `lost`, which this call claimed (the module docs): the generation
    /// serving next, or `lost` where the new `VkDevice` failed (the line then fails).
    fn recover(self: &Arc<Self>, lost: Arc<DeviceShared>) -> Device {
        let began = Instant::now();
        let old = holds::program(Arc::clone(&lost));
        // Its command buffers in flight fail through its completion path: the hang and its
        // victims (`command_buffer/completion.rs`).
        old.fail_in_flight();
        // Its queues idle, each under its lock: a wait on a lost device returns in finite time
        // (vkspec 5.2.3), and on one the test hook marked lost waits for the work it ran.
        old.drain_queues();
        let made = match &self.recipe {
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            Some(_) if self.hooks.making_fails.swap(false, Ordering::AcqRel) => {
                Err(Error::DeviceLost)
            }
            Some(recipe) => create::create_on(
                &recipe.instance,
                recipe.physical,
                &recipe.options,
                &recipe.cache,
                Some(self),
            ),
            None => Err(Error::DeviceLost),
        };
        let next = match made {
            Ok(next) => next,
            Err(error) => {
                *lock(&self.state) = State::Failed(lost);
                self.changed.notify_all();
                old.degrade_recovery_step(|| {
                    (
                        "the device".to_owned(),
                        format!(
                            "its new VkDevice could not be made ({error}): every later call is \
                             refused as on a lost device"
                        ),
                    )
                });
                return old;
            }
        };
        let previous = {
            let mut state = lock(&self.state);
            std::mem::replace(&mut *state, State::Live(Arc::clone(next.shared_arc())))
        };
        self.lineage.recoveries.fetch_add(1, Ordering::AcqRel);
        self.changed.notify_all();
        let culprit = match previous {
            State::Lost { culprit, .. } => culprit,
            _ => None,
        };
        // The lost generation goes once the objects made on it went.
        drop(old);
        drop(lost);
        // The command queues take a new place on the new generation at their next command buffer.
        let queues: Vec<_> = lock(&self.queues).drain(..).collect();
        let mut kept = Vec::with_capacity(queues.len());
        for queue in queues {
            if let Some(line) = queue.upgrade() {
                line.retire();
                kept.push(queue);
            }
        }
        lock(&self.queues).extend(kept);
        let took = began.elapsed();
        let generation = next.shared().generation;
        next.degrade_recovered(|| {
            let culprit = match culprit {
                Some(culprit) => {
                    let label = culprit.label.map_or_else(
                        || "a command buffer".to_owned(),
                        |label| format!("a command buffer of queue `{label}`"),
                    );
                    let inferred = match culprit.inferred {
                        true => {
                            " (an inference: work was in flight on two or more `VkQueue`s, and \
                             its queue's oldest incomplete batch was submitted first)"
                        }
                        false => "",
                    };
                    format!(
                        "{label} failed as the hang (`GpuHang`){inferred}, and {} more in flight \
                         as its victims (`GpuRecoveryVictim`)",
                        culprit.victims
                    )
                }
                None => "no command buffer was in flight".to_owned(),
            };
            (
                "the device".to_owned(),
                format!(
                    "its VkDevice was lost and made again in {took:?} (generation {generation}): \
                     {culprit}"
                ),
            )
        });
        next
    }

    /// Name the command buffer a loss failed as the hang, for the notice and the hold of its queue
    /// (`command_buffer/completion.rs`): the first named, until the recovery.
    pub(crate) fn named(&self, named: Culprit) {
        if let State::Lost { culprit, .. } = &mut *lock(&self.state) {
            culprit.get_or_insert(named);
        }
    }

    /// Whether a command buffer committed now on the queue whose shared part is `queue` is held as
    /// Metal holds the queue of a hang in flight: it runs nothing and completes with no error. So
    /// from the loss that named one of its command buffers the hang until the recovery.
    pub(crate) fn holds_hang_queue(&self, queue: &Arc<crate::queue::QueueCommon>) -> bool {
        match &*lock(&self.state) {
            State::Lost {
                culprit: Some(culprit),
                ..
            } => std::ptr::eq(culprit.queue.as_ptr(), Arc::as_ptr(queue)),
            _ => false,
        }
    }

    /// The generation that serves, lent to a root handle's [`Device::raw`]: kept until a later
    /// call lends another, so the handles that call returned stay valid until then.
    pub(super) fn lend(&self, serving: &Device) -> &DeviceShared {
        let shared = Arc::clone(serving.shared_arc());
        let pointer = Arc::as_ptr(&shared);
        let previous = {
            let mut lent = lock(&self.lent);
            match &*lent {
                Lent::To(kept) if Arc::ptr_eq(kept, &shared) => None,
                _ => Some(std::mem::replace(&mut *lent, Lent::To(shared))),
            }
        };
        drop(previous);
        // SAFETY: the line keeps the generation in `lent` until a later call lends another, which
        // `Device::raw`'s contract puts past every use of the handles it returned; the line lives
        // as long as the root handle the reference is borrowed from.
        unsafe { &*pointer }
    }

    /// A command queue made on the device, which each recovery has take a new place (`queue.rs`).
    pub(crate) fn queue_made(&self, queue: Weak<crate::queue::QueueLine>) {
        let mut queues = lock(&self.queues);
        queues.retain(|queue| queue.strong_count() > 0);
        queues.push(queue);
    }

    /// The program's last handle went (`holds.rs`): the generation it serves goes once nothing
    /// else holds it, here or as the last handle drops.
    pub(super) fn release(&self) {
        let released = std::mem::replace(&mut *lock(&self.state), State::Gone);
        drop(released);
        let lent = std::mem::replace(&mut *lock(&self.lent), Lent::Released);
        drop(lent);
    }
}

thread_local! {
    /// The generation the thread's public call runs on past its check of the generation's loss,
    /// its arguments taken ([`Device::at_serving_taking`]): the address of its `DeviceShared`.
    static RUNS_ON: Cell<usize> = const { Cell::new(0) };
}

/// The thread's public call running on a generation past its check of the generation's loss
/// ([`Device::at_serving_taking`]): while it lives, the generation is not lost to the call's own
/// checks ([`runs_on`]). What it replaced comes back as it drops.
struct RunsOn(usize);

impl RunsOn {
    fn enter(shared: &DeviceShared) -> RunsOn {
        let address = std::ptr::from_ref(shared).addr();
        RunsOn(RUNS_ON.with(|runs| runs.replace(address)))
    }
}

impl Drop for RunsOn {
    fn drop(&mut self) {
        RUNS_ON.with(|runs| runs.set(self.0));
    }
}

/// Whether the thread's public call runs on `shared` past its check of its loss ([`RunsOn`]):
/// a loss of it since follows the call, which its checks do not refuse (`Device::is_lost`).
#[cold]
pub(super) fn runs_on(shared: &DeviceShared) -> bool {
    RUNS_ON.with(|runs| runs.get() == std::ptr::from_ref(shared).addr())
}

/// The device a call goes to ([`Device::current`]): the caller's own handle, or one of the
/// generation that serves.
pub(crate) enum Current<'a> {
    This(&'a Device),
    Serving(Device),
}

impl std::ops::Deref for Current<'_> {
    type Target = Device;

    fn deref(&self) -> &Device {
        match self {
            Current::This(device) => device,
            Current::Serving(device) => device,
        }
    }
}

impl Device {
    /// No Metal counterpart: run the device's recovery from the loss of its `VkDevice` with no
    /// GPU fault, as a real loss runs it, for tests of what survives one. The command buffers in
    /// flight fail, the one of the oldest incomplete batch as the hang (`GpuHang`, counted against
    /// its queue as Metal counts a hang, so a queue's second revokes it, `AccessRevoked`) and the
    /// rest as its victims (`GpuRecoveryVictim`); the runtime makes a new `VkDevice` of the same
    /// physical device, options, queue families and extensions, and every later call of the
    /// device, and of its command queues, goes to it. An object made before is refused where it
    /// meets it (`RuntimeGap::MadeBeforeRecovery`). Announced (`Degradation::DeviceRecovered`).
    /// Returns once the new `VkDevice` serves; where it could not be made, `DeviceLost`, and the
    /// device answers as a lost one from then on.
    ///
    /// Refused on a device the consumer made (`Device::from_raw`), which the runtime cannot make
    /// again: `RuntimeGap::AdoptedDeviceRecovery`, the device left as it was. Called from a thread
    /// the device's command buffers' completion depends on (a completion-seen handler), it marks
    /// the device lost and returns: the next call from another thread recovers it.
    pub fn recover_for_test(&self) -> Result<(), Error> {
        let device = self.current();
        let line = device.line();
        if !line.recovers() {
            return Err(Error::NotImplemented(RuntimeGap::AdoptedDeviceRecovery));
        }
        device.mark_lost();
        if crate::exit_guard::serving_completion() {
            return Ok(());
        }
        let next = line.current(&device);
        match next.generation_serves() && !next.is_lost() {
            true => Ok(()),
            false => Err(Error::DeviceLost),
        }
    }

    /// The generation this call goes to (the module docs): this handle's own where it is a
    /// generation's that serves, at one relaxed load; else the line's, after its recovery.
    #[inline]
    pub(crate) fn current(&self) -> Current<'_> {
        match self.generation_serves() {
            true => Current::This(self),
            false => Current::Serving(self.line().current(self)),
        }
    }

    /// Run `call`, a public call of this handle, on the generation it goes to
    /// ([`current`](Self::current)), and again on the one serving next wherever the generation it
    /// ran on was lost under it and refused it for that ([`lost_under`](Self::lost_under)): a
    /// call of the device Metal keeps across a recovery is not refused because the generation it
    /// took was lost while it ran. Refused so, a run kept nothing it made, so the call runs whole
    /// again. Bounded as the line's waits are: each run is on a later generation than the one
    /// before, after its recovery, which [`Line::current`] waits for (announced past
    /// [`RECOVERY_NOTICE`]); a line that cannot recover answers the loss.
    pub(crate) fn at_serving<T>(
        &self,
        mut call: impl FnMut(&Device) -> Result<T, Error>,
    ) -> Result<T, Error> {
        loop {
            let device = self.current();
            match call(&*device) {
                Err(refused) if device.lost_under(&refused) => continue,
                answer => return answer,
            }
        }
    }

    /// [`at_serving`](Self::at_serving) of a call that takes `taken`, arguments it consumes (host
    /// memory and what gives it back, a range token), which a run refused for a loss could not
    /// give a later one: they go to the generation the call goes to once it is not lost, else to
    /// the one serving next, and the call then runs on it whole, a loss of it from then on
    /// following the call, as one after the call returned would. Its checks of the generation's
    /// loss answer so ([`RunsOn`]): the call is not refused for a loss once it took its
    /// arguments, and answers as on a generation lost after it (its objects of that generation,
    /// refused where they meet the next: `RuntimeGap::MadeBeforeRecovery`). A line that cannot
    /// recover refuses it as before.
    pub(crate) fn at_serving_taking<A, T>(
        &self,
        taken: A,
        call: impl FnOnce(&Device, A) -> Result<T, Error>,
    ) -> Result<T, Error> {
        loop {
            let device = self.current();
            if device.shared().lost() {
                let refused = device.lost_refusal();
                if device.lost_under(&refused) {
                    continue;
                }
                return Err(refused);
            }
            let _runs = RunsOn::enter(device.shared());
            return call(&*device, taken);
        }
    }

    /// Whether `refused`, a call's answer on this generation, is the refusal of its loss
    /// (`DeviceLost`, or `RuntimeGap::MadeBeforeRecovery` as [`lost_refusal`](Self::lost_refusal)
    /// answers it) and the line went past it: the call goes to the generation serving next
    /// ([`at_serving`](Self::at_serving)). A call refused for an object of an earlier generation
    /// while its own still serves is answered so, as is one of a thread command buffers'
    /// completion depends on, whose calls reach the lost generation ([`Line::current`]).
    #[cold]
    fn lost_under(&self, refused: &Error) -> bool {
        !crate::exit_guard::serving_completion()
            && self.line().recovers_past(self.shared().generation)
            && (*refused == Error::DeviceLost || *refused == self.lost_refusal())
    }

    /// The refusal of a call that reaches this generation after its loss: where the device's line
    /// recovers past it, the call's object was made before the recovery
    /// (`RuntimeGap::MadeBeforeRecovery`); else the lost device's (`DeviceLost`).
    #[cold]
    pub(crate) fn lost_refusal(&self) -> Error {
        match self.line().recovers_past(self.shared().generation) {
            true => Error::NotImplemented(RuntimeGap::MadeBeforeRecovery),
            false => Error::DeviceLost,
        }
    }

    /// The refusal of `call` for an object of this device that `other` met, which is not this
    /// device: where both are of one line, generations a recovery parts, the object was made
    /// before the recovery (`RuntimeGap::MadeBeforeRecovery`), or the call's object was; else the
    /// object is another device's (`Invalid`, `ForeignObject`). Out of line, so that a bind's
    /// check, whose refusal this is, inlines where the bind makes it.
    #[cold]
    #[inline(never)]
    pub(crate) fn foreign(&self, call: &'static str, other: &Device) -> Error {
        match self.same_line(other) {
            true => Error::NotImplemented(RuntimeGap::MadeBeforeRecovery),
            false => invalid(call, Problem::ForeignObject),
        }
    }

    /// Whether this is a generation handle of the generation that serves: one relaxed load of
    /// the line's generation. Its loss moves the line past it at once where the line recovers
    /// (`Line::lose`, under `Device::mark_lost`); a line that cannot recover answers with the lost
    /// generation, which so serves on.
    #[inline]
    pub(crate) fn generation_serves(&self) -> bool {
        match &self.0 {
            super::Pin::Generation(shared) => shared.lineage.current() == shared.generation,
            super::Pin::Root(..) => false,
        }
    }

    /// The device's command buffers in flight fail through its completion path, and the batches
    /// its scheduler holds through their submission, and this waits until its completion took
    /// them (`command_buffer/completion.rs`): the first step of a recovery, the device lost.
    fn fail_in_flight(&self) {
        let shared = self.shared();
        shared.completion.fail_in_flight();
        shared.scheduler.wake();
        let began = Instant::now();
        let after = self.line().notice_after();
        let mut announced = false;
        while !shared.completion.settled() {
            std::thread::sleep(Duration::from_millis(1));
            // Once per recovery: only the call that claimed it fails its command buffers.
            if !announced && began.elapsed() >= after {
                announced = true;
                self.degrade_recovery_step(|| {
                    (
                        "the device's recovery".to_owned(),
                        format!(
                            "its command buffers in flight took {:?} to fail, and it waits on",
                            began.elapsed()
                        ),
                    )
                });
            }
        }
    }
}

/// The device tests' hooks into a line's recoveries (`tests::device_recovery`), in those tests'
/// builds only: each set on one device, so a test's settings reach no other test's device.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod hooks {
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;
    use std::time::Duration;

    /// What a test set of a line's recoveries.
    #[derive(Default)]
    pub(crate) struct Hooks {
        /// How long a call waits for a recovery under way, or a recovery for its command buffers
        /// in flight to fail, before it announces the wait, in place of
        /// [`RECOVERY_NOTICE`](super::RECOVERY_NOTICE): zero announces every such wait.
        pub(crate) notice_after: Mutex<Option<Duration>>,
        /// The next recovery's new `VkDevice` fails to be made, as a driver's refusal fails it:
        /// the line fails, and answers as the lost device does from then on.
        pub(crate) making_fails: AtomicBool,
    }
}
