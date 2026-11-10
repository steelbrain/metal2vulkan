//! The `VkDeviceMemory` allocations each device has live, held to its `maxMemoryAllocationCount`
//! (`VUID-vkAllocateMemory-maxMemoryAllocationCount-04101`: "There must be less than
//! `VkPhysicalDeviceLimits::maxMemoryAllocationCount` device memory allocations currently allocated
//! on the device").
//!
//! - **Counted at the two doors.** Every allocation the runtime makes goes through
//!   `creation::allocate_memory` and every free through `release::object`, a source test holding
//!   both (`tests/exit_guard.rs`). So the count is kept there, by the device's handle, with no
//!   caller passing it: the runtime's pooled blocks, its dedicated ones, every import of host
//!   memory (no-copy objects' own, guest RAM chunks, fresh pages) and of dma-bufs. The device's
//!   priming (`device/prime.rs`) frees all it allocates before the device is registered, so it is
//!   not counted. What a consumer allocates itself on a device it hands the runtime
//!   (`Device::adopt`) is not seen.
//! - **Refused at the limit.** An allocation the limit leaves no room for is refused
//!   (`ERROR_TOO_MANY_OBJECTS`) without calling the driver, counted
//!   (`WorkCounts::memory_allocations_refused_at_limit`). A caller then frees what it keeps and
//!   tries again as after the driver's own refusal (`memory.rs`).
//! - **Posted frees count until they run.** A block whose free is posted to the device's free
//!   thread (`free_thread.rs`) stays counted until its `vkFreeMemory`; an allocation that finds
//!   the limit reached while frees are posted waits for them, then asks again, before it is
//!   refused.
//! - **A margin the runtime's own work leaves.** [`Allocations::room_ahead`]: the allocations the
//!   runtime makes ahead of any object (`host_ram.rs`'s ahead import, and the pooled blocks
//!   `memory.rs`'s `keep_room` and `memory/ready.rs` make before anything is placed in them) stop
//!   while fewer than a quarter of the limit are left, so the objects and their imports have
//!   those. Once the
//!   device's allocations reach that quarter, it gives back the chunks it imported ahead that no
//!   object took ([`Allocations::made`]), so they are gone before they could take the last.

use crate::free_thread::FreeThread;
use crate::step_aside::Activity;
use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The registered devices' counts, by handle: a few at most, so a list.
static DEVICES: Mutex<Vec<(vk::Device, Arc<Allocations>)>> = Mutex::new(Vec::new());

/// One device's live allocations and its limit.
pub(crate) struct Allocations {
    live: AtomicU32,
    /// `maxMemoryAllocationCount`, which the tests may lower (`Allocations::limit_to`).
    limit: AtomicU32,
    /// Allocations refused at the limit.
    refused: AtomicU32,
    /// Something the runtime allocated ahead may be given back: set by each such allocation,
    /// taken by the first allocation that leaves the device within the margin.
    ahead: AtomicBool,
    /// What gives the allocations made ahead back, set once the device is made.
    give_back: OnceLock<Box<dyn Fn() + Send + Sync>>,
    /// The device's free thread, whose posted frees an allocation at the limit waits for, set
    /// once the device is made.
    frees: OnceLock<Arc<FreeThread>>,
    /// The allocations and frees in progress on threads not the device's checks', which those
    /// step aside for (`step_aside.rs`).
    activity: Activity,
    /// How long each give-back's thread waits as it starts: the tests' stand-in for a loaded host
    /// that leaves it unscheduled ([`Allocations::hold_give_backs`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    give_backs_held: Mutex<std::time::Duration>,
}

impl Allocations {
    fn new(limit: u32) -> Allocations {
        Allocations {
            live: AtomicU32::new(0),
            limit: AtomicU32::new(limit),
            refused: AtomicU32::new(0),
            ahead: AtomicBool::new(false),
            give_back: OnceLock::new(),
            frees: OnceLock::new(),
            activity: Activity::new(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            give_backs_held: Mutex::default(),
        }
    }

    /// The device's allocations and frees in progress on threads not its checks'
    /// (`step_aside.rs`).
    pub(crate) fn activity(&self) -> &Activity {
        &self.activity
    }

    /// The margin of the limit the runtime's work ahead leaves: a quarter (1024 of the 4096 the
    /// spec requires at least), at least one.
    fn margin(limit: u32) -> u32 {
        (limit / 4).max(1)
    }

    /// Count one more live allocation, or refuse it where the limit is reached, once the frees
    /// posted to the device's free thread completed (the module docs).
    fn take(&self) -> bool {
        let taken = self.try_take()
            || self.frees.get().is_some_and(|frees| frees.wait()) && self.try_take();
        if !taken {
            self.refused.fetch_add(1, Ordering::Relaxed);
        }
        taken
    }

    /// Count one more live allocation where the limit leaves room for it.
    fn try_take(&self) -> bool {
        let limit = self.limit.load(Ordering::Relaxed);
        self.live
            .try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                (live < limit).then_some(live + 1)
            })
            .is_ok()
    }

    /// Wait for the frees posted to `frees` before refusing an allocation at the limit, from now
    /// on.
    pub(crate) fn wait_for_frees_on(&self, frees: Arc<FreeThread>) {
        let _ = self.frees.set(frees);
    }

    /// One counted allocation was freed, or its making failed.
    fn give(&self) {
        // Saturating: a free the count never saw (none is made today) leaves it at zero.
        let _ = (self.live).try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
            live.checked_sub(1)
        });
    }

    /// An allocation was made: where it leaves the device fewer than the margin and something
    /// was allocated ahead since the last give-back, give that back (on a thread of the
    /// device's, with no lock of the caller's held).
    fn made(&self) {
        let limit = self.limit.load(Ordering::Relaxed);
        let live = self.live.load(Ordering::Acquire);
        if live.saturating_add(Self::margin(limit)) <= limit {
            return;
        }
        if self.ahead.swap(false, Ordering::AcqRel) {
            if let Some(give_back) = self.give_back.get() {
                give_back();
            }
        }
    }

    /// Whether the runtime may allocate one more ahead of any object: one more leaves at least
    /// the margin of the limit. Counted as such ([`Allocations::made_ahead`]).
    pub(crate) fn room_ahead(&self) -> bool {
        let limit = self.limit.load(Ordering::Relaxed);
        let live = self.live.load(Ordering::Acquire);
        live.saturating_add(1).saturating_add(Self::margin(limit)) <= limit
    }

    /// An allocation was made ahead of any object, which [`Allocations::made`] gives back once
    /// the device's allocations come within the margin.
    pub(crate) fn made_ahead(&self) {
        self.ahead.store(true, Ordering::Release);
        self.made();
    }

    /// Give back what was allocated ahead with `give_back` from now on.
    pub(crate) fn give_back_with(&self, give_back: Box<dyn Fn() + Send + Sync>) {
        let _ = self.give_back.set(give_back);
    }

    /// The allocations refused at the limit so far.
    pub(crate) fn refused(&self) -> u32 {
        self.refused.load(Ordering::Relaxed)
    }

    /// The device's live allocations.
    #[cfg(test)]
    pub(crate) fn live(&self) -> u32 {
        self.live.load(Ordering::Acquire)
    }

    /// The device's limit.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn limit(&self) -> u32 {
        self.limit.load(Ordering::Relaxed)
    }

    /// Hold the device to `limit` allocations from now on: the tests' small device.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn limit_to(&self, limit: u32) {
        self.limit.store(limit, Ordering::Relaxed);
    }

    /// Hold each give-back's thread `held` as it starts, from now on.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn hold_give_backs(&self, held: std::time::Duration) {
        *self.give_backs_held.lock().unwrap() = held;
    }

    /// How long each give-back's thread is held as it starts ([`Allocations::hold_give_backs`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn give_backs_held(&self) -> std::time::Duration {
        *self.give_backs_held.lock().unwrap()
    }
}

/// A device's registration ([`register`]), let go of with the device.
pub(crate) struct Registered {
    device: vk::Device,
    pub(crate) allocations: Arc<Allocations>,
}

impl Drop for Registered {
    fn drop(&mut self) {
        let mut devices = lock();
        devices.retain(|(device, allocations)| {
            *device != self.device || !Arc::ptr_eq(allocations, &self.allocations)
        });
    }
}

/// Count `device`'s allocations from now on, against `limit` (its `maxMemoryAllocationCount`):
/// before its first allocation but its priming's. A handle a device let go of may come again:
/// the new device's count replaces the old.
pub(crate) fn register(device: vk::Device, limit: u32) -> Registered {
    let allocations = Arc::new(Allocations::new(limit));
    let mut devices = lock();
    devices.retain(|(known, _)| *known != device);
    devices.push((device, Arc::clone(&allocations)));
    Registered {
        device,
        allocations,
    }
}

/// Whether `device` may allocate one more ahead of any object ([`Allocations::room_ahead`]):
/// always where it is not registered.
pub(crate) fn room_ahead(device: vk::Device) -> bool {
    of(device).is_none_or(|allocations| allocations.room_ahead())
}

/// `device`'s count, where it is registered.
pub(crate) fn of(device: vk::Device) -> Option<Arc<Allocations>> {
    let devices = lock();
    (devices.iter())
        .find(|(known, _)| *known == device)
        .map(|(_, allocations)| Arc::clone(allocations))
}

fn lock() -> std::sync::MutexGuard<'static, Vec<(vk::Device, Arc<Allocations>)>> {
    DEVICES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One allocation of `device` about to be made: `None` where the limit leaves it no room, so
/// the caller refuses it without calling the driver; else what the caller hands
/// [`allocated`] with the driver's answer.
pub(crate) fn before_allocation(device: vk::Device) -> Option<Option<Arc<Allocations>>> {
    match of(device) {
        Some(allocations) => allocations.take().then_some(Some(allocations)),
        None => Some(None),
    }
}

/// The driver answered an allocation [`before_allocation`] admitted: `made` or not.
pub(crate) fn allocated(allocations: Option<Arc<Allocations>>, made: bool) {
    let Some(allocations) = allocations else {
        return;
    };
    if made {
        allocations.made();
    } else {
        allocations.give();
    }
}

/// One allocation of the device of `allocations` (its count, where it is registered: [`of`]) was
/// freed.
pub(crate) fn freed(allocations: Option<&Allocations>) {
    if let Some(allocations) = allocations {
        allocations.give();
    }
}

/// An allocation or free of the device of `allocations` in progress on this thread, counted for
/// its checks to step aside for (`step_aside.rs`) until this goes: none where the device is not
/// registered or this is a check's thread.
pub(crate) struct Busy<'a>(Option<&'a Allocations>);

impl<'a> Busy<'a> {
    pub(crate) fn of(allocations: Option<&'a Allocations>) -> Busy<'a> {
        Busy(allocations.filter(|allocations| allocations.activity.begin()))
    }
}

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        if let Some(allocations) = self.0 {
            allocations.activity.end();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn an_allocation_past_the_limit_is_refused_and_a_free_makes_room() {
        let count = Allocations::new(3);
        assert!(count.take() && count.take() && count.take());
        assert!(!count.take(), "the limit reached");
        assert_eq!(count.refused(), 1);
        count.give();
        assert!(count.take(), "a free made room");
        assert_eq!(count.live(), 3);
    }

    #[test]
    fn work_ahead_leaves_a_quarter_and_is_given_back_once_the_device_reaches_it() {
        let count = Allocations::new(8);
        let given = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&given);
        count.give_back_with(Box::new(move || {
            seen.fetch_add(1, Ordering::Relaxed);
        }));
        // A margin of 2 of 8: room ahead while one more leaves 2.
        let mut ahead = 0;
        while count.room_ahead() {
            assert!(count.take());
            count.made_ahead();
            ahead += 1;
        }
        assert_eq!((ahead, count.live()), (6, 6));
        assert_eq!(
            given.load(Ordering::Relaxed),
            0,
            "not within the margin yet"
        );
        // The objects' allocation inside the margin gives the work ahead back, once.
        assert!(count.take());
        count.made();
        assert_eq!(given.load(Ordering::Relaxed), 1);
        assert!(count.take());
        count.made();
        assert_eq!(given.load(Ordering::Relaxed), 1, "once per work ahead");
    }

    #[test]
    fn a_registration_counts_its_device_until_it_goes() {
        let handle = vk::Device::from_raw(0x5ee5);
        let registered = register(handle, 2);
        let admitted = before_allocation(handle).unwrap();
        allocated(admitted, true);
        assert_eq!(registered.allocations.live(), 1);
        freed(of(handle).as_deref());
        assert_eq!(registered.allocations.live(), 0);
        drop(registered);
        assert!(of(handle).is_none());
        assert!(
            matches!(before_allocation(handle), Some(None)),
            "not counted"
        );
    }
}
