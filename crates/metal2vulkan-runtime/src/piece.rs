//! Pieces: a command buffer's compute work goes to the device's `VkQueue` in pieces, each one
//! segment, whose dispatches are expected to run on the GPU for at most
//! [`GROUP_RUN`](crate::submit::GROUP_RUN) together, so a command buffer of another queue committed
//! while one runs waits for that piece rather than for the whole command buffer
//! (`crate::submit`). A dispatch's expected run is its kernel's rate, the run of one of its
//! threadgroups, times its threadgroups. Each rate comes from the runs of the pieces that ran the
//! kernel: a command buffer of one piece timed by its own two timestamps, which nothing of another
//! submission passes inside; the first piece of several from the command buffer's start to a
//! timestamp at the piece's end; each later piece by a pair of its own. So the work of another
//! queue's command buffer that went between two pieces is in no rate, and a command buffer that
//! is not cut writes no timestamp past its own two. The rate is scheduling data, as a queue's
//! expected run is (`QueueInner::expected_run`): it bounds where a piece ends, and never chooses
//! between two forms of the work.

use crate::kept::Uncount;
use crate::pipeline::Compiled;
use crate::queue::next_run;
use crate::submit::GROUP_RUN;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

/// A kernel's rate: the GPU run of one threadgroup of its dispatches, in picoseconds, as the
/// pieces that ran it measured, rising at once to a longer one and halving the way toward a
/// shorter one ([`next_run`]); unknown until the first such piece completed.
pub(crate) struct Rate(AtomicU64);

impl Default for Rate {
    fn default() -> Rate {
        Rate(AtomicU64::new(u64::MAX))
    }
}

impl Rate {
    /// The expected run, in nanoseconds, of `groups` threadgroups; None while the rate is
    /// unknown.
    pub(crate) fn expected(&self, groups: u64) -> Option<u64> {
        let rate = self.0.load(Relaxed);
        let nanos = u128::from(groups) * u128::from(rate) / 1000;
        (rate != u64::MAX).then(|| u64::try_from(nanos).unwrap_or(u64::MAX))
    }

    /// For the device tests: the rate is `picos` exactly, the test's own input in place of what
    /// the pieces measured, until a piece that ran the kernel completes.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn set_for_test(&self, picos: u64) {
        self.0.store(picos, Relaxed);
    }

    /// A piece measured `picos` for each threadgroup of this kernel.
    fn ran(&self, picos: u64) {
        let _ = self.0.try_update(Relaxed, Relaxed, |rate| {
            Some(next_run((rate != u64::MAX).then_some(rate), picos))
        });
    }
}

/// A kernel of a piece, with its threadgroups summed: its content as its state holds it, taking
/// no count ([`Uncount`]), since the command buffer keeps the state it bound until it retires,
/// after its pieces' runs set their rates (`crate::command::retire`).
pub(crate) struct Kernel {
    compiled: ManuallyDrop<Arc<Compiled>>,
    groups: u64,
}

/// A command buffer's pieces' kernels, piece by piece: a list its kept objects hold
/// (`crate::kept::ByKind`), whose capacity goes on to the queue's next recording, so a recording
/// allocates for them only past what an earlier one left, and frees none.
pub(crate) type Kernels = Vec<Kernel>;

/// The dispatches of one piece: each kernel once, with its threadgroups summed, in its command
/// buffer's list ([`Kernels`]), and their expected run.
#[derive(Default)]
pub(crate) struct Work {
    /// Its kernels' places in the list: the last piece's run to the list's end.
    kernels: Range<usize>,
    /// The expected GPU run of its dispatches, in nanoseconds, summed.
    pub(crate) run: u64,
    /// It holds dispatches of a kernel whose rate was unknown, and of no other kernel, so its run
    /// sets that kernel's rate.
    alone: bool,
    /// Its run tells no rate: it holds an indirect dispatch, whose threadgroups the GPU reads and
    /// which counts as one, a dispatch of a kernel of unknown rate beside others, or commands of
    /// a blit or render encoder after its first dispatch.
    blind: bool,
    /// Commands of a blit or render encoder come before its first dispatch, so a span from the
    /// command buffer's start would hold them: its own pair starts at its first dispatch.
    led: bool,
}

impl Work {
    /// Whether it holds a dispatch.
    pub(crate) fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Whether a dispatch of `kernel` expected to run `run` (None: its kernel's rate unknown;
    /// where an encoder opens, no kernel and `Some(0)`) begins a new piece: this one holds a
    /// dispatch, and holds a kernel of unknown rate (unless the dispatch is that kernel's too, of
    /// a rate still unknown), or the dispatch's rate is unknown, or the two would pass the bound.
    pub(crate) fn full(
        &self,
        list: &[Kernel],
        kernel: Option<&Arc<Compiled>>,
        run: Option<u64>,
    ) -> bool {
        let own = |kernel: &Arc<Compiled>| self.of(list).iter().all(|kept| kept.is(kernel));
        !self.is_empty()
            && match (self.alone, run) {
                (true, None) => !kernel.is_some_and(own),
                (true, Some(_)) | (false, None) => true,
                (false, Some(run)) => self.run.saturating_add(run) > GROUP_RUN,
            }
    }

    /// The expected run of a dispatch of `kernel` of `groups` threadgroups; None where its rate
    /// is unknown. An indirect dispatch (`groups` None) counts as one threadgroup, and as none of
    /// a kernel of unknown rate: its run, of threadgroups only the GPU reads, sets no rate.
    pub(crate) fn expected(kernel: &Compiled, groups: Option<u64>) -> Option<u64> {
        match groups {
            Some(groups) => kernel.rate.expected(groups),
            None => Some(kernel.rate.expected(1).unwrap_or(0)),
        }
    }

    /// Adds a dispatch of `kernel`, of the state its command buffer keeps, to the piece, the last
    /// of `list`'s: `groups` threadgroups, None for an indirect dispatch.
    pub(crate) fn add(&mut self, list: &mut Kernels, kernel: &Arc<Compiled>, groups: Option<u64>) {
        let expected = Work::expected(kernel, groups);
        let own = self.of(list).iter().all(|kept| kept.is(kernel));
        self.blind |= groups.is_none() || ((expected.is_none() || self.alone) && !own);
        self.alone |= expected.is_none();
        self.run = self.run.saturating_add(expected.unwrap_or(0));
        let groups = groups.unwrap_or(1);
        let range = self.kernels.clone();
        match (list.get_mut(range)).and_then(|mine| mine.iter_mut().find(|kept| kept.is(kernel))) {
            Some(kept) => kept.groups = kept.groups.saturating_add(groups),
            None => {
                if self.kernels.is_empty() {
                    self.kernels = list.len()..list.len();
                }
                // SAFETY: the command buffer keeps the state `kernel` is of until it retires,
                // after the list's last use (`crate::command::retire`).
                let compiled = unsafe { kernel.uncounted() };
                list.push(Kernel { compiled, groups });
                self.kernels.end = list.len();
            }
        }
    }

    /// Its kernels in `list`, its command buffer's.
    fn of<'a>(&self, list: &'a [Kernel]) -> &'a [Kernel] {
        list.get(self.kernels.clone()).unwrap_or_default()
    }

    /// Commands of another encoder go in the piece: after its dispatches, or ahead of them.
    pub(crate) fn blind(&mut self) {
        self.blind |= !self.is_empty();
        self.led |= self.is_empty();
    }

    /// Whether commands of another encoder came before its first dispatch.
    pub(crate) fn led(&self) -> bool {
        self.led
    }

    /// The piece ran `nanos` on the GPU: a kernel alone in it takes its rate from that run, and
    /// several each scale theirs by the run against what they expected.
    pub(crate) fn ran(&self, list: &[Kernel], nanos: u64) {
        if self.blind {
            return;
        }
        let kernels = self.of(list);
        let picos = u128::from(nanos) * 1000;
        let rate = |kernel: &Compiled| u128::from(kernel.rate.0.load(Relaxed));
        let expected: u128 = (kernels.iter())
            .map(|kept| rate(&kept.compiled) * u128::from(kept.groups))
            .sum();
        for kept in kernels {
            let picos = match kernels.len() {
                1 => picos / u128::from(kept.groups.max(1)),
                _ if expected == 0 => continue,
                _ => rate(&kept.compiled).saturating_mul(picos) / expected,
            };
            (kept.compiled.rate).ran(u64::try_from(picos).unwrap_or(u64::MAX - 1));
        }
    }
}

impl Kernel {
    /// Whether it is `kernel`.
    fn is(&self, kernel: &Arc<Compiled>) -> bool {
        Arc::ptr_eq(&self.compiled, kernel)
    }
}
