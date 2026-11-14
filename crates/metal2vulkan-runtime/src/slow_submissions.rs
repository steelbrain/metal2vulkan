//! The slow submissions log: a default-off diagnostic, under the queue times
//! (`METAL2VULKAN_QUEUE_TIMES`, `metal2vulkan::env_vars::queue_times`; `Timing::queue_times`), that
//! names where the submission of a batch spent its time when the thread that submitted it took
//! [`SLOW`] or longer from the start of the submission to the driver's call. A scheduler thread's
//! pass holds every queue's submissions while it submits one batch (`scheduler/mod.rs`), so such
//! a span holds back every later commit of the device.
//!
//! - **What it prints.** One line per such batch, through the device's message sink at `Info`
//!   severity (id name [`ID_NAME`]), else on stderr: the batch's place in its command buffer, the
//!   slot of its `VkQueue`, the whole span and its phases ([`Phase`]): the prologue (the batch's
//!   copies in again, its first submission's resolution, its segments placed, its copies out
//!   settled), its first work's take (the new resources' first work and the hold of the heap
//!   sets' slots it takes), the staging queue's flush (with the time it waited for the staging
//!   queue's lock and for one of that queue's command buffers to come free, each counted apart:
//!   [`Wait`]), and the planning (the tracker's plan, its barriers' command buffers, its waits and
//!   signals); then how long the driver's call took.
//! - **Which thread.** Whichever submits the batch: the scheduler's, or a committing or waiting
//!   thread that submits for itself (`scheduler/mod.rs`). A submission that holds its batch back
//!   (for a group, a rebind, its first submission's resolution) makes no call and prints nothing;
//!   the call that later submits it is timed from its own start.
//! - **Off, it costs a flag read a submission**: no clock is read and no line is formatted. On,
//!   it reads the host's clock at each phase's start, and changes no work.

use crate::device::Device;
use crate::driver_facts::inform_through;
use crate::gpu_time::host_nanos;
use std::cell::Cell;

/// The id name of the log's messages.
pub(crate) const ID_NAME: &str = "metal2vulkan-slow-submissions";

/// The span from a submission's start to the driver's call, in nanoseconds, from which it is
/// named: a scheduler thread held that long holds back a frame's worth of a 120 Hz program's
/// commits.
pub(crate) const SLOW: u64 = 5_000_000;

/// A phase of a batch's submission (the module docs), in the order the submission enters them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Phase {
    /// From the submission's start to its first work's take.
    Prologue,
    /// The take of its first work.
    FirstWork,
    /// The flush of the copies queued for the staging queue.
    Staging,
    /// From the flush to the driver's call.
    Planning,
}

/// How many phases there are.
const PHASES: usize = 4;

/// A wait inside a phase, counted apart (the module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Wait {
    /// For the staging queue's lock (`Device::staging_lock`).
    StagingLock,
    /// For one of the staging queue's command buffers, all pending, to complete
    /// (`device/staging.rs`, `free_buffer`).
    StagingBuffer,
}

/// How many waits there are.
const WAITS: usize = 2;

thread_local! {
    /// The waits counted on this thread while a watched submission runs on it; `None` while none
    /// does.
    static COUNTED: Cell<Option<[u64; WAITS]>> = const { Cell::new(None) };
}

/// Run `f`, counting how long it took as `wait` where a watched submission runs on this thread;
/// elsewhere `f` alone, with no clock read.
pub(crate) fn waited<T>(wait: Wait, f: impl FnOnce() -> T) -> T {
    if COUNTED.get().is_none() {
        return f();
    }
    let started = host_nanos();
    let out = f();
    let took = host_nanos().saturating_sub(started);
    if let Some(mut counted) = COUNTED.get() {
        counted[wait as usize] += took;
        COUNTED.set(Some(counted));
    }
    out
}

/// One batch's submission, timed phase by phase on the thread that submits it.
pub(crate) struct Watch {
    started: u64,
    phase: Phase,
    since: u64,
    phases: [u64; PHASES],
}

impl Watch {
    /// A watch of the submission starting now, where `on`; `None` elsewhere.
    pub(crate) fn start(on: bool) -> Option<Watch> {
        if !on {
            return None;
        }
        COUNTED.set(Some([0; WAITS]));
        let now = host_nanos();
        Some(Watch {
            started: now,
            phase: Phase::Prologue,
            since: now,
            phases: [0; PHASES],
        })
    }

    /// The submission enters `phase` now.
    pub(crate) fn enter(&mut self, phase: Phase) {
        let now = host_nanos();
        self.phases[self.phase as usize] += now.saturating_sub(self.since);
        self.phase = phase;
        self.since = now;
    }

    /// The driver's call carrying batch `index` (of `batches`) on `VkQueue` slot `queue` of
    /// `device` was made at host time `called` and returned at `returned`: the line, where the
    /// span from the start to the call was [`SLOW`] or longer (the module docs).
    pub(crate) fn called(
        mut self,
        device: &Device,
        (index, batches): (usize, usize),
        queue: usize,
        (called, returned): (u64, u64),
    ) {
        let span = called.saturating_sub(self.started);
        if span < SLOW {
            return;
        }
        let at = self.phase as usize;
        self.phases[at] += called.saturating_sub(self.since);
        let waits = COUNTED.get().unwrap_or([0; WAITS]);
        let ms = |nanos: u64| nanos as f64 / 1e6;
        let [prologue, first, staging, planning] = self.phases.map(ms);
        let [lock, buffer] = waits.map(ms);
        let text = format!(
            "metal2vulkan: slow submission: batch {} of {batches} of a command buffer, on VkQueue \
             slot {queue}, held its thread {:.3} ms from the start of its submission to the \
             driver's call: the prologue {prologue:.3} ms, its first work's take {first:.3} ms, \
             the staging queue's flush {staging:.3} ms (of it {lock:.3} ms waiting for the \
             staging queue's lock and {buffer:.3} ms for one of its command buffers to come \
             free), the planning and recording {planning:.3} ms; the call took {:.3} ms",
            index + 1,
            ms(span),
            ms(returned.saturating_sub(called)),
        );
        inform_through(device.shared().instance.sink(), ID_NAME, text);
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        COUNTED.set(None);
    }
}

/// Enter `phase` where `watch` times a submission.
pub(crate) fn enter(watch: &mut Option<Watch>, phase: Phase) {
    if let Some(watch) = watch {
        watch.enter(phase);
    }
}
