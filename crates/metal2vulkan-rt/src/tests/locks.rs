//! The lock budget: the locked instructions an encoder call executes, counted as the difference
//! between a recording of `2n` commands and one of `n`, over the `n` added. A locked instruction
//! between two recorded commands flushes a write-combining command stream (on a discrete Radeon,
//! about 490 ns each), so an encoder call's success path executes none (the runtime's README,
//! "Encoder calls"). The count is the whole test process's in user space: the encoding thread's,
//! the runtime's threads' and the driver's, whose locks per added command count as much.
//!
//! A node names the event that counts locked instructions on its CPU in `M2V_LOCK_EVENT`, as
//! `perf` names a raw event (`metal2vulkan_devtools::instructions::ProcessEvent`):
//! `cpu_core/r21d0` (an Intel performance core's `MEM_INST_RETIRED.LOCK_LOADS`) or `r1e25` (a
//! Zen 4 core's locked instructions). A node that names none prints that it measured nothing.
//!
//! The devices are opened on an instance without the validation layer ([`Locks::each_device`]):
//! the layer takes a lock of its own on every command it checks, 74 locked instructions a draw on
//! the Radeon RX 7900 XTX and the RTX 2080, which would be all the count read.

use crate::{Device, Instance};
use metal2vulkan_devtools::instructions::{pin_this_thread, ProcessEvent};

/// The variable naming the node's locked-instruction event.
const EVENT: &str = "M2V_LOCK_EVENT";
/// Measured pairs of a recording of `n` and one of `2n` commands, after the warm-up.
const ROUNDS: usize = 7;
/// The most locked instructions an added command may add, the least of the rounds' counts each:
/// one value on every device. A site of the runtime's that runs per command adds 1 or more, and
/// the drivers' own growth stays far below it (0.14 per added bound draw on Intel's Iris Xe, its
/// highest measured).
const BOUND: f64 = 0.25;

/// The node's locked-instruction counter, for the whole process from its opening.
pub(crate) struct Locks {
    name: String,
    event: ProcessEvent,
}

impl Locks {
    /// The counter of the event the node names, with this thread, and so every thread it starts
    /// after, pinned to the CPUs its unit counts on; `None`, printed, where the node names none.
    /// Opened before the test makes its instance, so the runtime's and the driver's threads
    /// count.
    pub(crate) fn open(test: &str) -> Option<Locks> {
        let Ok(name) = std::env::var(EVENT) else {
            println!("LOCKS-UNMEASURED {test}: the node names no {EVENT}");
            return None;
        };
        let event =
            ProcessEvent::new(&name).unwrap_or_else(|error| panic!("{EVENT}={name}: {error}"));
        if !event.cpus().is_empty() {
            pin_this_thread(event.cpus())
                .unwrap_or_else(|error| panic!("pinning to {name}'s CPUs: {error}"));
        }
        Some(Locks { name, event })
    }

    /// Each device of the cfg's kind, as `each_device` opens it, on an instance without the
    /// validation layer: the test fails where a device lists the layer active, since its locks
    /// would be the count.
    pub(crate) fn each_device(&self, test: &str, body: impl FnMut(&str, &Instance, Device)) {
        let instance = Instance::new(&[])
            .unwrap_or_else(|error| panic!("{test}: an instance at the floor: {error}"));
        for (name, physical) in crate::tests::devices_of_the_cfgs_kind(&instance) {
            let tool = crate::tests::validation::validation_tool(&instance, &name, physical.raw());
            assert!(
                tool.is_none(),
                "{test}: {name} lists {tool:?} active: the lock count is the runtime's and the \
                 driver's, without the layer"
            );
        }
        crate::tests::command::on_device::on_each_device(test, &instance, body);
    }

    /// The locked instructions the process executed while `work` ran.
    fn count(&self, work: impl FnOnce()) -> u64 {
        let before = self.event.read().expect("reading the lock counter");
        work();
        self.event.read().expect("reading the lock counter") - before
    }

    /// Records, through `record`, `2n` commands of kind `what` twice to warm every ring, list and
    /// pool to the longer recording's size, then [`ROUNDS`] pairs of `n` and `2n`; prints the
    /// locked instructions per added command, the least count of each length's against the
    /// other's, with every round's, and the matrix's `locks lock_budget/<key>` record of it per
    /// million added commands (none below 0), and asserts it is at most [`BOUND`].
    pub(crate) fn per_added(
        &self,
        device: &str,
        (what, key): (&str, &str),
        n: u32,
        mut record: impl FnMut(u32),
    ) {
        record(2 * n);
        record(2 * n);
        let mut short = Vec::with_capacity(ROUNDS);
        let mut long = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            short.push(self.count(|| record(n)));
            long.push(self.count(|| record(2 * n)));
        }
        let least = |counts: &[u64]| *counts.iter().min().expect("a round") as f64;
        let added = (least(&long) - least(&short)) / f64::from(n);
        let each: Vec<f64> = (short.iter().zip(&long))
            .map(|(&short, &long)| (long as f64 - short as f64) / f64::from(n))
            .collect();
        let spread = (each.iter().copied()).fold((f64::MAX, f64::MIN), |(low, high), round| {
            (low.min(round), high.max(round))
        });
        println!(
            "LOCKS {device}: {added:.4} locked instructions per added {what} ({} {what}s less {n}, \
             least of {ROUNDS} rounds each; the rounds' {:.4} to {:.4}; {short:?} and {long:?}; \
             {})",
            2 * n,
            spread.0,
            spread.1,
            self.name
        );
        println!(
            "M2V-MATRIX locks lock_budget/{key} {}",
            (added.max(0.0) * 1e6).round() as u64
        );
        assert!(
            added <= BOUND,
            "{device}: {added:.4} locked instructions per added {what}, over the budget's {BOUND}"
        );
    }
}
