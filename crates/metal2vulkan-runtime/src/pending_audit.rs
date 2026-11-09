//! The pending audit: a default-off diagnostic, `METAL2VULKAN_PENDING_AUDIT`
//! (`metal2vulkan::env_vars`) or `DeviceOptions::pending_audit`, that checks, each time the runtime
//! frees or reuses what a `VkCommandBuffer` recorded, that the GPU completed every submission that
//! carries it, against the GPU's own timelines rather than what the host saw of them.
//!
//! - **What it checks** ([`Freeing`]). Each `vkBeginCommandBuffer` the device's functions make
//!   (`device/functions.rs`: every begin of the runtime's, a primary begun ahead in a spare pool
//!   among them, `spares.rs`), which resets the command buffer; each `vkResetCommandPool` of a
//!   spare command pool, and each command pool given back to the device's spares, for every
//!   command buffer the pool made (`Spares::command_pool`, `Spares::give_command_pool`); and each
//!   submission of a command buffer that was submitted before, unless it was begun for
//!   simultaneous use (ANV rewrites a chainable primary's last jump at each submission: Mesa,
//!   `anv_cmd_buffer_record_chain_submit`). ANV frees a command buffer's batch, dynamic and
//!   surface state, back to the device's pools, as it is reset (Mesa, `anv_cmd_buffer_reset`).
//! - **What carries a command buffer.** A batch's submission, from the moment the batch goes to its
//!   `VkQueue` or waits in the scheduler's group (`command_buffer/submit.rs`), to the queue's
//!   timeline value it signals; a secondary command buffer through each primary that executes it
//!   (each `vkCmdExecuteCommands` since that primary's last begin); the device's own submissions
//!   and the staging queue's, to their own timelines (`device/submit.rs`, `device/staging.rs`). A
//!   submission that fails is withdrawn. Other submissions carry nothing (the predicated loop's
//!   runs again, at fences, `command_buffer/rerun.rs`; the device's checks as it is made).
//! - **The answer is the GPU's.** At each check it reads each carrying timeline
//!   (`vkGetSemaphoreCounterValue`): a value not reached means the GPU may still run that
//!   submission. A read that fails (a lost device) finds nothing. A command buffer whose carrying
//!   submissions all completed leaves the ledger.
//! - **What it prints.** One line as the device is made, and one per finding, through the
//!   device's message sink at `Info` severity (id name `metal2vulkan-pending-audit`), else on
//!   stderr: what was done, the thread, the command buffer, the timeline semaphore, its counter and
//!   the value the submission signals, and for the first [`BACKTRACES`] findings the thread's
//!   backtrace. As the device goes, the checks made and the findings, by kind.
//! - **It changes nothing.** It reads counters and logs: every begin, reset, give-back and
//!   submission goes on as it would without it.
//! - **What it costs on.** A lock of the ledger at each begin, execution of secondaries, pool
//!   reset, pool given back and submission; a map entry per command buffer per submission until a
//!   check finds the submission completed; one `vkGetSemaphoreCounterValue` per carrying timeline
//!   at each check of a command buffer the ledger holds; a backtrace for each of the first
//!   [`BACKTRACES`] findings. Off, one test of an `Option` at each of those places.

use crate::device::lock;
use crate::driver_facts::inform_through;
use crate::exit_guard::{self, Call};
use crate::MessageSink;
use ash::vk::{self, Handle};
use std::backtrace::Backtrace;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The id name of the audit's messages.
const ID_NAME: &str = "metal2vulkan-pending-audit";

/// The findings that print the thread's backtrace: the first ones.
const BACKTRACES: u64 = 8;

/// What frees or reuses what a command buffer recorded (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Freeing {
    /// `vkBeginCommandBuffer`, which resets it.
    Begin,
    /// `vkResetCommandPool` of the pool that made it.
    PoolReset,
    /// The pool that made it given back to the device's spares, for the next command buffer to
    /// take.
    GiveBack,
    /// A submission of it again.
    Submit,
}

impl Freeing {
    /// Every kind, in the order of the counts' arrays.
    pub(crate) const ALL: [Freeing; 4] = [
        Freeing::Begin,
        Freeing::PoolReset,
        Freeing::GiveBack,
        Freeing::Submit,
    ];

    /// What it says in a line.
    fn says(self) -> &'static str {
        match self {
            Freeing::Begin => "a begin (vkBeginCommandBuffer)",
            Freeing::PoolReset => "a reset of its pool (vkResetCommandPool)",
            Freeing::GiveBack => "its pool given back to the device's spares",
            Freeing::Submit => "a submission of it again",
        }
    }
}

/// A command buffer a check found carried by a submission the GPU may still run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pending {
    pub(crate) buffer: vk::CommandBuffer,
    /// The timeline the submission signals, and the value.
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) value: u64,
    /// The timeline's counter, as read.
    pub(crate) counter: u64,
}

/// What carries each command buffer (the module docs).
#[derive(Default)]
pub(crate) struct Ledger {
    /// Each command buffer submitted, with the timelines whose values complete the submissions
    /// that carry it: per timeline, the highest value (a timeline's signal completes every
    /// submission before it in submission order on its queue).
    carried: HashMap<vk::CommandBuffer, Vec<(vk::Semaphore, u64)>>,
    /// The secondaries each primary executed since its last begin.
    executes: HashMap<vk::CommandBuffer, Vec<vk::CommandBuffer>>,
    /// The command buffers last begun for simultaneous use, which may be pending in several
    /// submissions at once.
    simultaneous: HashSet<vk::CommandBuffer>,
}

impl Ledger {
    /// `buffers` (and the secondaries each executes) go in a submission that signals `semaphore`
    /// to `value`.
    pub(crate) fn submitted(
        &mut self,
        (semaphore, value): (vk::Semaphore, u64),
        buffers: &[vk::CommandBuffer],
    ) {
        let Ledger {
            carried, executes, ..
        } = self;
        for buffer in buffers {
            let secondaries = executes.get(buffer).map_or(&[][..], Vec::as_slice);
            for &carried_one in std::iter::once(buffer).chain(secondaries) {
                let timelines = carried.entry(carried_one).or_default();
                match timelines.iter_mut().find(|(of, _)| *of == semaphore) {
                    Some((_, highest)) => *highest = (*highest).max(value),
                    None => timelines.push((semaphore, value)),
                }
            }
        }
    }

    /// The submission of `buffers` that would have signalled `semaphore` to `value` failed:
    /// nothing of it is pending.
    pub(crate) fn withdrawn(
        &mut self,
        (semaphore, value): (vk::Semaphore, u64),
        buffers: &[vk::CommandBuffer],
    ) {
        let Ledger {
            carried, executes, ..
        } = self;
        for buffer in buffers {
            let secondaries = executes.get(buffer).map_or(&[][..], Vec::as_slice);
            for carried_one in std::iter::once(buffer).chain(secondaries) {
                let Some(timelines) = carried.get_mut(carried_one) else {
                    continue;
                };
                timelines.retain(|&(of, highest)| (of, highest) != (semaphore, value));
                if timelines.is_empty() {
                    carried.remove(carried_one);
                }
            }
        }
    }

    /// `primary` executes `secondaries` (`vkCmdExecuteCommands`).
    pub(crate) fn executes(
        &mut self,
        primary: vk::CommandBuffer,
        secondaries: &[vk::CommandBuffer],
    ) {
        (self.executes.entry(primary).or_default()).extend_from_slice(secondaries);
    }

    /// `buffer` was begun, for simultaneous use where `simultaneous`: what it executed before is
    /// no longer its own.
    pub(crate) fn begun(&mut self, buffer: vk::CommandBuffer, simultaneous: bool) {
        self.executes.remove(&buffer);
        match simultaneous {
            true => self.simultaneous.insert(buffer),
            false => self.simultaneous.remove(&buffer),
        };
    }

    /// Of `buffers`, those a submission may be pending in many at once: begun for simultaneous
    /// use, which a submission again does not reuse.
    pub(crate) fn submitted_again(&self, buffers: &[vk::CommandBuffer]) -> Vec<vk::CommandBuffer> {
        let Ledger {
            executes,
            simultaneous,
            ..
        } = self;
        let mut again = Vec::new();
        for buffer in buffers {
            let secondaries = executes.get(buffer).map_or(&[][..], Vec::as_slice);
            let each = std::iter::once(buffer).chain(secondaries);
            again.extend(each.filter(|buffer| !simultaneous.contains(buffer)));
        }
        again
    }

    /// Of `buffers`, those carried by a submission whose timeline's counter, as `read` reads it,
    /// is below the submission's value; carrying submissions found completed leave the ledger.
    /// A timeline `read` cannot read finds nothing.
    pub(crate) fn pending(
        &mut self,
        buffers: &[vk::CommandBuffer],
        mut read: impl FnMut(vk::Semaphore) -> Option<u64>,
    ) -> Vec<Pending> {
        let mut counters: Vec<(vk::Semaphore, Option<u64>)> = Vec::new();
        let mut found = Vec::new();
        for &buffer in buffers {
            let Some(timelines) = self.carried.get_mut(&buffer) else {
                continue;
            };
            timelines.retain(|&(semaphore, value)| {
                let counter = match counters.iter().find(|(of, _)| *of == semaphore) {
                    Some(&(_, counter)) => counter,
                    None => {
                        let counter = read(semaphore);
                        counters.push((semaphore, counter));
                        counter
                    }
                };
                match counter {
                    Some(counter) if counter >= value => false,
                    Some(counter) => {
                        found.push(Pending {
                            buffer,
                            semaphore,
                            value,
                            counter,
                        });
                        true
                    }
                    None => true,
                }
            });
            if timelines.is_empty() {
                self.carried.remove(&buffer);
            }
        }
        found
    }
}

/// The checks a device's audit made and the findings, by kind ([`Freeing::ALL`]'s order).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    pub(crate) checks: [u64; 4],
    pub(crate) found: [u64; 4],
}

/// A device's pending audit, where it is on (the module docs).
pub(crate) struct PendingAudit {
    sink: Option<MessageSink>,
    ledger: Mutex<Ledger>,
    checks: [AtomicU64; 4],
    found: [AtomicU64; 4],
}

impl PendingAudit {
    /// The audit of a device whose instance reports to `sink`, where `asked`
    /// (`DeviceOptions::pending_audit`) or the variable is set; its first line said.
    pub(crate) fn new(asked: bool, sink: Option<&MessageSink>) -> Option<Arc<PendingAudit>> {
        if !asked && !metal2vulkan::env_vars::pending_audit() {
            return None;
        }
        let audit = PendingAudit {
            sink: sink.cloned(),
            ledger: Mutex::new(Ledger::default()),
            checks: Default::default(),
            found: Default::default(),
        };
        audit.line(
            "metal2vulkan: pending audit: on: each begin, pool reset, pool given back and \
             submission again of a command buffer is checked against the timelines of the \
             submissions that carry it"
                .to_owned(),
        );
        Some(Arc::new(audit))
    }

    /// `buffers` go in a submission that signals `semaphore` to `value`: each of them submitted
    /// before and not begun for simultaneous use is checked first ([`Freeing::Submit`]).
    pub(crate) fn submitted(
        &self,
        raw: &ash::Device,
        (semaphore, value): (vk::Semaphore, u64),
        buffers: &[&[vk::CommandBuffer]],
    ) {
        let mut found = Vec::new();
        {
            let mut ledger = lock(&self.ledger);
            for &buffers in buffers {
                let again = ledger.submitted_again(buffers);
                found.extend(self.check_in(&mut ledger, raw, Freeing::Submit, &again));
                ledger.submitted((semaphore, value), buffers);
            }
        }
        self.report(Freeing::Submit, found);
    }

    /// The submission of `buffers` that would have signalled `semaphore` to `value` failed.
    pub(crate) fn withdrawn(
        &self,
        (semaphore, value): (vk::Semaphore, u64),
        buffers: &[&[vk::CommandBuffer]],
    ) {
        let mut ledger = lock(&self.ledger);
        for &buffers in buffers {
            ledger.withdrawn((semaphore, value), buffers);
        }
    }

    /// `primary` executes `secondaries`.
    pub(crate) fn executes(&self, primary: vk::CommandBuffer, secondaries: &[vk::CommandBuffer]) {
        lock(&self.ledger).executes(primary, secondaries);
    }

    /// `buffer` is about to be begun with `flags`: checked ([`Freeing::Begin`]), then begun.
    pub(crate) fn begins(
        &self,
        raw: &ash::Device,
        buffer: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
    ) {
        let found = {
            let mut ledger = lock(&self.ledger);
            let found = self.check_in(&mut ledger, raw, Freeing::Begin, &[buffer]);
            let simultaneous = flags.contains(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
            ledger.begun(buffer, simultaneous);
            found
        };
        self.report(Freeing::Begin, found);
    }

    /// `what` is about to free or reuse what `buffers` recorded: checked.
    pub(crate) fn check(&self, raw: &ash::Device, what: Freeing, buffers: &[vk::CommandBuffer]) {
        let found = self.check_in(&mut lock(&self.ledger), raw, what, buffers);
        self.report(what, found);
    }

    /// The checks made and the findings so far.
    pub(crate) fn counts(&self) -> Counts {
        let read = |counts: &[AtomicU64; 4]| counts.each_ref().map(|n| n.load(Ordering::Relaxed));
        Counts {
            checks: read(&self.checks),
            found: read(&self.found),
        }
    }

    /// [`check`](Self::check) under the ledger's lock, `ledger`: what it found, said once the
    /// lock is let go of (a sink may call the runtime).
    fn check_in(
        &self,
        ledger: &mut Ledger,
        raw: &ash::Device,
        what: Freeing,
        buffers: &[vk::CommandBuffer],
    ) -> Vec<Pending> {
        if buffers.is_empty() {
            return Vec::new();
        }
        self.checks[what as usize].fetch_add(1, Ordering::Relaxed);
        ledger.pending(buffers, |semaphore| {
            // SAFETY: a timeline of the device's, which lives while the device does.
            let read = unsafe { raw.get_semaphore_counter_value(semaphore) };
            exit_guard::called(Call::GetSemaphoreCounterValue);
            read.ok()
        })
    }

    /// Count and say each of `found`, which `what` met.
    fn report(&self, what: Freeing, found: Vec<Pending>) {
        for pending in found {
            let finding = self.found[what as usize].fetch_add(1, Ordering::Relaxed) + 1;
            let total: u64 = (self.found.iter()).map(|n| n.load(Ordering::Relaxed)).sum();
            let trace = match total <= BACKTRACES {
                true => format!("\n{}", Backtrace::force_capture()),
                false => String::new(),
            };
            let thread = std::thread::current();
            let Pending {
                buffer,
                semaphore,
                value,
                counter,
            } = pending;
            self.line(format!(
                "metal2vulkan: pending audit: {} of command buffer 0x{:x} on thread {}, while a \
                 submission that carries it may run: timeline semaphore 0x{:x} at {counter}, \
                 below the submission's value {value} (finding {finding} of its kind){trace}",
                what.says(),
                buffer.as_raw(),
                thread.name().unwrap_or("unnamed"),
                semaphore.as_raw(),
            ));
        }
    }

    /// Deliver `text`, a line of the audit.
    fn line(&self, text: String) {
        inform_through(self.sink.as_ref(), ID_NAME, text);
    }
}

/// Its last line, as the device goes: the checks and findings, by kind.
impl Drop for PendingAudit {
    fn drop(&mut self) {
        let Counts { checks, found } = self.counts();
        let each = (Freeing::ALL.iter().enumerate())
            .map(|(at, what)| {
                format!(
                    "{}: {} checks, {} found",
                    what.says(),
                    checks[at],
                    found[at]
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        self.line(format!("metal2vulkan: pending audit: {each}"));
    }
}
