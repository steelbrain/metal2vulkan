//! The device's begin-ahead thread: a settled spare command pool's primaries are begun here, for
//! the encoders of the command buffer that takes the pool next (`spares.rs`), off the thread that
//! completed the command buffer that gave the pool back.
//!
//! - **What is posted.** The first command pool of a command buffer that completed and ended
//!   every command buffer its pools handed out, once its completed handlers ran, before its gate
//!   opens (`command_buffer/mod.rs`, `command_buffer/gate.rs`): the completing thread posts the
//!   pool and goes on, so neither the command buffer's waiters nor its queue's next transitions
//!   wait for a begin, and a waiter's return comes after the post. On ANV a begin was 8 to 20 us
//!   (`spares.rs`), which the completing thread spent before, one a command buffer, while the
//!   queue's context was held, and only where no thread waited for the command buffer.
//! - **What it begins.** As many of the pool's primaries as encoders of the command buffer that
//!   gave it back opened their own in it, one at least and `spares::AHEAD` at most, one at a time,
//!   the pool out of the spares only while it begins one (`spares.rs`'s
//!   `BeginTargets::begin_ahead`). A command buffer that takes the pool before the thread came
//!   to it, or between two begins, takes it as it is: its encoders past those begun begin their
//!   own, counted (`WorkCounts::encoder_primaries_begun_ahead_missed`).
//! - **Bounded.** One thread a device, started by its first post. At most [`QUEUE`] pools wait;
//!   a pool posted while it waits waits once, and one that finds the queue full, the device's
//!   teardown begun, or no thread, is begun by none: its command buffer's encoders begin their
//!   own. Stopped and joined by the device's teardown, before the spares are released
//!   ([`BeginAhead::stop`]).
//! - **Holds nothing of the device**: the spares' command pools, the device's functions, its
//!   loader word and its counts (`spares.rs`'s `BeginTargets`), as the free thread does
//!   (`free_thread.rs`). No release of the device's runs on it.
//! - **A section of the exit guard** for each begin (`exit_guard.rs`); once the process exits,
//!   nothing more is begun and the thread ends.
//! - **At the user-interactive class** ([`exit_guard::spawn_user_interactive`]), as the handler
//!   threads that began the primaries before: its begins race the next command buffer's take.

use crate::device::lock;
use crate::exit_guard;
use crate::spares::BeginTargets;
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

/// The most pools that wait for the thread (the module docs).
pub(crate) const QUEUE: usize = 64;

/// A device's begin-ahead thread (the module docs).
#[derive(Default)]
pub(crate) struct BeginAhead {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

/// What the thread and the posters share.
#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    /// A pool was posted, or the thread is asked to stop.
    work: Condvar,
    /// The crate's device tests' hold of each posted pool's begins as they start
    /// (`tests/begin_ahead.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    held: crate::freed_memory::held::Held,
    /// A posted pool's begins ended, for the crate's device tests ([`BeginAhead::settle`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    idle: Condvar,
}

#[derive(Default)]
struct State {
    queue: VecDeque<vk::CommandPool>,
    /// The thread was started, or a start failed.
    started: bool,
    stopping: bool,
    /// The thread makes a posted pool's begins, for the crate's device tests.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    running: bool,
}

impl BeginAhead {
    /// Post the spare command pool `raw`, given back settled, for its primaries to be begun ahead
    /// on the thread, started unless it runs, with what `targets` gives; nothing where the pool
    /// waits already, the queue is full, the device's teardown began or the host starts no
    /// thread (the module docs).
    pub(crate) fn post(&self, raw: vk::CommandPool, targets: impl FnOnce() -> BeginTargets) {
        let mut state = lock(&self.shared.state);
        if state.stopping || state.queue.len() >= QUEUE || state.queue.contains(&raw) {
            return;
        }
        if !state.started {
            state.started = true;
            let (shared, targets) = (Arc::clone(&self.shared), targets());
            let builder = std::thread::Builder::new().name("metal2vulkan-begin-ahead".into());
            match exit_guard::spawn_user_interactive(builder, move || run(&shared, &targets)) {
                Ok(handle) => *lock(&self.thread) = Some(handle),
                Err(_) => return,
            }
        }
        if lock(&self.thread).is_none() {
            return;
        }
        state.queue.push_back(raw);
        drop(state);
        self.shared.work.notify_one();
    }

    /// The device's teardown, before it releases the spares: no pool is posted from here on, the
    /// thread ends once its begins in progress are done, joined; those still waiting are begun by
    /// none.
    pub(crate) fn stop(&self) {
        lock(&self.shared.state).stopping = true;
        self.shared.work.notify_all();
        let handle = lock(&self.thread).take();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }

    /// The tests' hold of each posted pool's begins as they start.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn held(&self) -> &crate::freed_memory::held::Held {
        &self.shared.held
    }

    /// Wait until no posted pool waits and the thread makes no begins, for the crate's device
    /// tests: what the spares hold, and the counts of the begins, are then final for the command
    /// buffers whose waits returned, as a waiter's return comes after its command buffer's post.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn settle(&self) {
        let mut state = lock(&self.shared.state);
        while state.running || !state.queue.is_empty() && lock(&self.thread).is_some() {
            state = (self.shared.idle.wait(state)).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// The thread's loop: each posted pool's begins in turn, until stopped or the process exits.
fn run(shared: &Shared, targets: &BeginTargets) {
    loop {
        let raw = {
            let mut state = lock(&shared.state);
            loop {
                if state.stopping {
                    return;
                }
                if let Some(raw) = state.queue.pop_front() {
                    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                    {
                        state.running = true;
                    }
                    break raw;
                }
                state = (shared.work.wait(state)).unwrap_or_else(PoisonError::into_inner);
            }
        };
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        shared.held.pass();
        if exit_guard::closed() {
            return;
        }
        targets.begin_ahead(raw);
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        {
            lock(&shared.state).running = false;
            shared.idle.notify_all();
        }
    }
}
