//! The runtime's own holds of a device: its check of textures in place (`in_place_check.rs`) and
//! its warm pass (`warm/pass.rs`), each on a thread of its own, which hold the device while they
//! make objects on it and let it go between their steps.
//!
//! - **Two counts.** Every handle of the device is counted here, as the program's or as a
//!   holder's. A handle a holder takes ([`WeakDevice::hold`]) is a holder's, and so is each handle
//!   cloned from one (the device of every object the holder makes); every other is the program's
//!   (the device's own, its objects', and a runtime thread's upgrade of a weak handle).
//! - **The program's last drop tears the device down.** The drop that leaves the program no
//!   handle stops the holds, and on a thread of the program's waits until the holders' handles
//!   went, so the device goes with it, on that thread, before it returns. A consumer that adopted
//!   the `VkDevice` destroys it once the last drop returned, and Vulkan requires every object
//!   made from it destroyed first. On one of the runtime's own threads (a completed handler's
//!   release of the program's last command buffer) it waits for nothing: those threads serve the
//!   work the holds may wait for, and the program's work there ends where it ends already.
//! - **The wait is a step.** A holder reads the stop between its steps
//!   ([`Device::holds_go_on`]): the check between its rounds, the warm pass between its records.
//!   So the wait is at most the check's round in progress (and its release of its buffer), or the
//!   record the pass is making. It happens only at teardown.
//! - **Exact.** A program handle's drop, a holder's take and a holder's handle's clone and drop
//!   count under the registry's lock; a holder's handle is counted gone only once its share of the
//!   device went ([`Share`]), so the program's last drop, waiting for that, holds the device's last
//!   share. A program handle is counted gone before its share goes: a program handle another
//!   thread drops at once may go last, inside that drop. A program handle's clone counts without
//!   the lock (it clones a handle that is counted), and so does a runtime thread's upgrade, which
//!   keeps the device as before: one made while the last drop waits goes on that thread.
//! - **A stopped device is gone.** Once stopped, no holder takes the device, and a device made on
//!   the same physical device makes a new one (`create.rs`) rather than revive it.

use super::{lock, Device, DeviceShared, WeakDevice};
use crate::exit_guard;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// The runtime's holders of a device, each of whose steps reads the stop.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Holder {
    /// The check of textures in place, between its rounds.
    InPlaceCheck,
    /// The warm pass, between its records.
    WarmPass,
}

/// A device's registry of its handles and the runtime's own holds (the module docs).
#[derive(Default)]
pub(crate) struct Holds {
    /// The program's handles.
    program: AtomicUsize,
    /// The holders' handles, and whether the program's last drop stopped the holds.
    state: Mutex<State>,
    /// Signalled as the last holder's handle goes.
    released: Condvar,
    /// Whether the holds were stopped: read between a holder's steps without the lock.
    stopping: AtomicBool,
    /// How long the program's last drop waited for the holds, in nanoseconds: 0 where it waited
    /// for none.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    waited: std::sync::atomic::AtomicU64,
    /// The crate's device tests' holds of each holder between its steps
    /// (`tests/teardown_threads.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    held: [crate::freed_memory::held::Held; 2],
}

/// The registry's count of the holders' handles, and its stop.
#[derive(Default)]
struct State {
    /// The holders' handles: taken by a holder or cloned from one, their share of the device not
    /// yet gone.
    holders: usize,
    /// The program's last drop stopped the holds.
    stopped: bool,
}

/// A handle's place in the registry (the module docs): the registry where it is a holder's,
/// `None` where it is the program's (whose drop reaches the registry through the device).
pub(super) struct Share(Option<Arc<Holds>>);

impl Holds {
    /// Whether the program's last drop stopped the holds.
    pub(crate) fn stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    /// How long the program's last drop waited for the holds; `None` where it waited for none.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn waited(&self) -> Option<std::time::Duration> {
        match self.waited.load(Ordering::Acquire) {
            0 => None,
            nanos => Some(std::time::Duration::from_nanos(nanos)),
        }
    }

    /// The test hold of `holder`.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn held(&self, holder: Holder) -> &crate::freed_memory::held::Held {
        &self.held[holder as usize]
    }

    /// Let `holder`'s test hold go once the holds were stopped, or once it was let go already.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn release_once_stopping(&self, holder: Holder) {
        self.held(holder).release_once(&self.stopping);
    }
}

/// The program's handle of `shared`, counted.
fn program(shared: Arc<DeviceShared>) -> Device {
    shared.holds.program.fetch_add(1, Ordering::SeqCst);
    Device(shared, Share(None))
}

/// A new device's first handle, the program's.
pub(super) fn first(shared: Arc<DeviceShared>) -> Device {
    program(shared)
}

/// The registry a new device's handles share.
pub(super) fn holds() -> Arc<Holds> {
    Arc::new(Holds::default())
}

/// `weak`'s device for the program (a runtime thread's upgrade among them), counted, while it
/// lives.
pub(super) fn upgrade(weak: &WeakDevice) -> Option<Device> {
    weak.0.upgrade().map(program)
}

impl WeakDevice {
    /// The device for a step of one of the runtime's holders, counted as the holder's (the module
    /// docs): `None` where it went, or the program's last drop stopped the holds.
    pub(crate) fn hold(&self) -> Option<Device> {
        let mut state = lock(&self.1.state);
        if state.stopped {
            return None;
        }
        let shared = self.0.upgrade()?;
        state.holders += 1;
        Some(Device(shared, Share(Some(Arc::clone(&self.1)))))
    }

    /// The device for the program, as [`WeakDevice::upgrade`]: `None` where it went, or its
    /// holds were stopped, so that it goes.
    pub(super) fn revive(&self) -> Option<Device> {
        let state = lock(&self.1.state);
        if state.stopped {
            return None;
        }
        let shared = self.0.upgrade()?;
        drop(state);
        Some(program(shared))
    }
}

impl Device {
    /// Whether `holder`, which holds the device through [`WeakDevice::hold`], goes on with its
    /// next step: `false` once the program's last drop stopped the holds.
    pub(crate) fn holds_go_on(&self, holder: Holder) -> bool {
        let holds = &self.0.holds;
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        holds.held(holder).pass();
        let _ = holder;
        !holds.stopping()
    }
}

impl Clone for Device {
    fn clone(&self) -> Device {
        match &(self.1).0 {
            Some(holds) => {
                let mut state = lock(&holds.state);
                state.holders += 1;
                Device(Arc::clone(&self.0), Share(Some(Arc::clone(holds))))
            }
            None => program(Arc::clone(&self.0)),
        }
    }
}

impl Drop for Device {
    // A program handle's drop: the program's last stops the holds and waits for them. A holder's
    // handle is counted gone by its `Share`, once the device's share went before it.
    fn drop(&mut self) {
        if (self.1).0.is_some() {
            return;
        }
        let holds = &self.0.holds;
        let mut state = lock(&holds.state);
        if holds.program.fetch_sub(1, Ordering::SeqCst) > 1 {
            return;
        }
        state.stopped = true;
        holds.stopping.store(true, Ordering::SeqCst);
        if state.holders == 0 || exit_guard::on_runtime_thread() {
            return;
        }
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let waiting = std::time::Instant::now();
        while state.holders > 0 {
            state = (holds.released.wait(state)).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        {
            let nanos = u64::try_from(waiting.elapsed().as_nanos()).unwrap_or(u64::MAX);
            holds.waited.store(nanos.max(1), Ordering::Release);
        }
        // The device's share goes as this returns, the last: the teardown runs here.
    }
}

impl Drop for Share {
    // A holder's handle is counted gone here, after the device's share it held went (fields go
    // in their order): the program's last drop, which waits for this, then holds the last share.
    fn drop(&mut self) {
        let Some(holds) = &self.0 else {
            return;
        };
        let mut state = lock(&holds.state);
        state.holders -= 1;
        if state.holders == 0 {
            holds.released.notify_all();
        }
    }
}
