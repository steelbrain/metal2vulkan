//! The device's idle sweep: what the runtime keeps for a while after the program let go of it
//! goes once that while ended, on a thread of the device's own, whether or not the program makes
//! or releases anything after it.
//!
//! - **What.** Each kept empty block of the device's memory once it stayed empty for
//!   `KEPT_EMPTY_FOR` (`memory.rs`'s `Pool::stale`; a pool the runtime keeps room in for its own
//!   resources keeps one, the room its next creation call would make again,
//!   `Memory::keeps_room`), each block made ahead that went unused once its pool stopped growing
//!   for as long (`Memory::sweep_kept`), and each image and buffer parked past its time
//!   (`recycle.rs`'s `IMAGES` and `BUFFERS`), destroyed and its memory returned. The calls that
//!   let them go before still do as they come: a release in a block's pool, the device's creation
//!   calls (once a second at most, `Memory::sweep_stale`), a recycler's park or take. Without the
//!   sweep, a device whose program stopped making resources kept them until its next creation
//!   call, however long that took (`tests::idle_sweep`).
//! - **When.** Each block kept empty, each block made ahead and each object parked tells the
//!   sweep when it may go ([`Due::by`]). The thread waits until the earliest such moment, sweeps,
//!   asks what comes due next and waits for that; with nothing due it waits with no timeout. It
//!   sweeps at most once every [`EVERY`], so a program that parks objects every frame wakes it
//!   once a second at most.
//! - **Bounded.** One thread a device, started by the device's first creation call that may wait
//!   (`Device::keep_room`), stopped and joined by the device's teardown before the teardown
//!   destroys the parked objects and frees the memory ([`IdleSweep::stop`]).
//! - **Holds nothing of the device.** The thread holds the device's memory, its two recyclers, its
//!   functions and its counts ([`SweepTargets`]), as the free thread does (`free_thread.rs`), never the
//!   device: no release of the device's runs on it. A block it lets go of is posted to the free
//!   thread, which paces its frees (`step_aside::pace`).
//! - **A section of the exit guard** (`exit_guard.rs`): each sweep's Vulkan calls run in one; once
//!   the process exits, nothing is swept and the thread ends.

use crate::churn_counts::Recycled;
use crate::device::functions::Functions;
use crate::device::lock;
use crate::exit_guard;
use crate::memory::Memory;
use crate::recycle::{self, BufferMaking, ImageMaking, Recycler};
use crate::work_counts::Counters;
use ash::vk;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The least time between two sweeps: a second, the creation calls' own spacing of their sweeps
/// (`memory.rs`'s `KEPT_SWEEP_EVERY`) and the recyclers' time (`recycle.rs`).
pub(crate) const EVERY: Duration = Duration::from_secs(1);

/// When the sweep is next due: told by what keeps something for a while (the memory, the
/// recyclers' parks), read by the thread.
#[derive(Default)]
pub(crate) struct Due {
    state: Mutex<State>,
    /// An earlier moment was noted, or the thread is asked to stop.
    wake: Condvar,
    /// The sweeps the thread ran, for the crate's device tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    sweeps: std::sync::atomic::AtomicU64,
}

#[derive(Default)]
struct State {
    /// The earliest moment noted at which something may go.
    at: Option<Instant>,
    /// The earliest the next sweep runs: [`EVERY`] after the last began.
    not_before: Option<Instant>,
    stopping: bool,
}

impl Due {
    /// Something kept may go at `at`: the sweep is due then, or once [`EVERY`] passed since the
    /// last sweep began, where that is later. Wakes the thread only for a moment earlier than the
    /// one it waits for.
    pub(crate) fn by(&self, at: Instant) {
        let mut state = lock(&self.state);
        let at = state.not_before.map_or(at, |not_before| at.max(not_before));
        if state.stopping || state.at.is_some_and(|due| due <= at) {
            return;
        }
        state.at = Some(at);
        drop(state);
        self.wake.notify_one();
    }

    /// Wait until the sweep is due: `false` where the thread is asked to stop.
    fn wait(&self) -> bool {
        let mut state = lock(&self.state);
        loop {
            if state.stopping {
                return false;
            }
            let now = Instant::now();
            state = match state.at {
                Some(at) if at <= now => {
                    state.at = None;
                    state.not_before = Some(now + EVERY);
                    return true;
                }
                Some(at) => {
                    let waited = self.wake.wait_timeout(state, at - now);
                    waited.unwrap_or_else(PoisonError::into_inner).0
                }
                None => (self.wake.wait(state)).unwrap_or_else(PoisonError::into_inner),
            };
        }
    }
}

/// What the thread sweeps: the device's memory, its recyclers, and what their objects' release
/// takes.
pub(crate) struct SweepTargets {
    pub(crate) memory: Arc<Mutex<Memory>>,
    pub(crate) images: Arc<Mutex<Recycler<vk::Image, ImageMaking>>>,
    pub(crate) buffers: Arc<Mutex<Recycler<vk::Buffer, BufferMaking>>>,
    pub(crate) functions: Arc<Functions>,
    pub(crate) counts: Arc<Counters>,
}

/// A device's idle sweep (the module docs).
#[derive(Default)]
pub(crate) struct IdleSweep {
    due: Arc<Due>,
    /// A start was tried: the thread runs, or the host started none.
    tried: AtomicBool,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl IdleSweep {
    /// What tells the sweep when something may go, for the device's memory.
    pub(crate) fn due(&self) -> Arc<Due> {
        Arc::clone(&self.due)
    }

    /// Something kept may go at `at` ([`Due::by`]).
    pub(crate) fn by(&self, at: Instant) {
        self.due.by(at);
    }

    /// Start the thread over what `swept` gives, unless a start was tried or the teardown began.
    /// Where the host starts no thread, what is kept goes at the calls that let it go as they
    /// come (the module docs).
    pub(crate) fn start(&self, swept: impl FnOnce() -> SweepTargets) {
        if self.tried.load(Ordering::Acquire) {
            return;
        }
        let mut thread = lock(&self.thread);
        if self.tried.swap(true, Ordering::AcqRel) || lock(&self.due.state).stopping {
            return;
        }
        let (due, swept) = (Arc::clone(&self.due), swept());
        let builder = std::thread::Builder::new().name("metal2vulkan-idle-sweep".into());
        if let Ok(handle) = exit_guard::spawn(builder, move || run(&due, &swept)) {
            *thread = Some(handle);
        }
    }

    /// The device's teardown, before it destroys the parked objects and frees the memory: the
    /// thread ends, its sweep in progress done, joined.
    pub(crate) fn stop(&self) {
        lock(&self.due.state).stopping = true;
        self.due.wake.notify_all();
        let handle = lock(&self.thread).take();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }

    /// The sweeps the thread ran so far.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn sweeps(&self) -> u64 {
        self.due.sweeps.load(Ordering::Relaxed)
    }
}

/// The thread's loop: a sweep each time one is due, until stopped or the process exits.
fn run(due: &Due, swept: &SweepTargets) {
    while due.wait() {
        let Some(busy) = exit_guard::enter() else {
            return;
        };
        let next = sweep(swept, Instant::now());
        drop(busy);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        due.sweeps.fetch_add(1, Ordering::Relaxed);
        if let Some(next) = next {
            due.by(next);
        }
    }
}

/// Let go of what stayed kept past its time at `now`: the parked objects, then the memory's
/// blocks (each recycler's lock before the memory's, as everywhere). When the next of what stays
/// may go.
fn sweep(swept: &SweepTargets, now: Instant) -> Option<Instant> {
    let device = swept.functions.get();
    let (images, images_due) = lock(&swept.images).sweep(now);
    if !images.is_empty() {
        (swept.counts.churn).evicted(Recycled::Image, images.len());
        recycle::destroy(device, &mut lock(&swept.memory), images);
    }
    let (buffers, buffers_due) = lock(&swept.buffers).sweep(now);
    if !buffers.is_empty() {
        (swept.counts.churn).evicted(Recycled::Buffer, buffers.len());
        recycle::destroy(device, &mut lock(&swept.memory), buffers);
    }
    let memory_due = {
        let mut memory = lock(&swept.memory);
        memory.sweep_kept(device, now);
        memory.sweep_due(now)
    };
    [images_due, buffers_due, memory_due]
        .into_iter()
        .flatten()
        .min()
}
