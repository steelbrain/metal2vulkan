//! Command queues sharing `VkQueue`s (`QueueSharingPolicy::Adaptive`, the default): which
//! `VkQueue` a command queue submits to, chosen again each time it makes a command buffer while it
//! holds no other uncompleted one, on a device whose own check finds the GPU idle when consecutive
//! work comes from another `VkQueue` (`queue_switch_check.rs`).
//!
//! - **Why.** NVIDIA's driver (615.71.09 on the RTX 5080 Laptop GPU, Linux) leaves the GPU idle
//!   56 to 62 us whenever its next work comes from another `VkQueue` (p50 56.3 us over 10,713 such
//!   switches), so a device of one `VkQueue` per command queue, round robin (`queue.rs`), ran
//!   eight threads' one-draw command buffers at 25,800 to 29,500 a second, the GPU idle at the
//!   switches 60% of its time, where one `VkQueue` for all of them ran 54,700 to 58,700.
//! - **Where.** A device whose policy is `Adaptive`, once its check found the switch cost above
//!   [`SWITCH_IDLES`] ([`QueueSwitch::shares`]); before the check answered, and everywhere else,
//!   each command queue keeps the `VkQueue` it took at its creation (the fewest live queues',
//!   below), as under `QueueSharingPolicy::PerQueue`. MoltenVK's four families of one queue each, and RADV's and
//!   lavapipe's single graphics queue, have no two `VkQueue`s of one family, so no queue ever
//!   moves there and the check makes no measurement.
//! - **When a queue moves.** Only as it makes a command buffer while it holds no other uncompleted
//!   one (`CommandQueue::slot`, under its slots' lock, the only command buffer of its queue the new
//!   one): every earlier command buffer of the queue completed, its completed handlers ran, and
//!   nothing of it is held in the scheduler, a group, a rerun or a hang hold, so the move breaks no
//!   order of the queue's own. What a later command buffer reads of an earlier one's writes it
//!   reads as of another `VkQueue`'s work seen complete (`command_buffer/submit.rs`). A command
//!   queue reads its `VkQueue` from that make to its command buffer's completion, through every
//!   step that names it (its pools' family, its submissions, its groups, its failure's value),
//!   and only one of its family is taken, so its pools' family never changes. Never a queue made
//!   for presenting, never one made with `Device::new_command_queue_unshared`, and never one that
//!   lent its `VkQueue` (`CommandQueue::with_vk_queue`, after which it is pinned for good).
//! - **Which `VkQueue`** ([`Device::pick_vk_queue`]): one pass over the round robin's `VkQueue`s
//!   of its family, atomics alone, no lock. A `VkQueue` a long queue or an unshared queue holds is
//!   never taken. Among the busy ones whose oldest batch in flight has run under [`YOUNG_HEAD`],
//!   the one with the most batches in flight, then the latest submission; else it stays, unless a
//!   long or unshared queue holds its own `VkQueue`; then the most recently used idle one; else it
//!   stays.
//! - **Long queues.** A command buffer whose GPU work ran for over [`LONG_HEAD`], by its own
//!   timestamps (its first batch's start to its last's end), marks its command queue long, in the
//!   completion turn that read them, before the turn finishes it, so before the queue's next
//!   command buffer is made (`CommandBufferShared::place_gpu_times`). Its own GPU time, not the
//!   host's view of how long it ran as its `VkQueue`'s oldest: that view also holds its wait for
//!   the GPU to start it and the waiter's lateness to see it complete, which marked a queue of
//!   one-draw command buffers long on the RTX 5080 Laptop GPU beside another process's 30 ms ones
//!   (draws of 11 and 16 us of GPU time seen 2.0 and 2.3 ms after they became their `VkQueue`'s
//!   oldest, 1.8 and 2.1 ms of it before they started). One exception: a command buffer one of
//!   whose batches waits on the GPU for work of another `VkQueue`'s that the host had not seen
//!   complete as it was submitted (an `MTLEvent`'s signal pending there, the tracker's wait for
//!   another `VkQueue`'s writes, or new resources' first work there) also marks its queue long
//!   where more than [`LONG_HEAD`] ran from that batch's submission to its own GPU end
//!   (`Times::waited_across`). Its timestamps start only once the wait was met, while its
//!   `VkQueue` held the wait, and every command buffer committed behind it there, for as long as
//!   the other work ran: a queue whose short command buffers each wait for another queue's long
//!   one would otherwise share for good, holding its `VkQueue`'s other queues behind each wait. A
//!   device whose command buffers write no timestamps the host can read
//!   (`gpu_times_from_host`, a driver whose timestamps wait for every queue's, or no calibrated
//!   timestamps) does not share ([`Verdict::Unchecked`],
//!   `WorkCounts::queue_sharing_untimed`). A long queue moves once more, at its next such make: to an
//!   idle `VkQueue` of its family that no long or unshared queue holds, the one used least
//!   recently, else it stays; then never again. So the first long command buffer of a queue that
//!   shared holds the command buffers the queues sharing its `VkQueue` committed behind it for its
//!   whole length (29 ms behind 30 ms command buffers on the RTX 5080 Laptop GPU), and those its
//!   later command buffers while the long queue holds work until its next idle point; a queue
//!   that held a long queue's `VkQueue` leaves at its own next idle point. A queue gives no sign of
//!   its first long command buffer before it runs, so no rule of joining avoids that first one.
//! - **Placement at creation**, under every policy and verdict, before the check answered too: a
//!   command queue takes the round robin's `VkQueue` that the fewest live command queues hold, of
//!   those no unshared queue holds, the first in turn among equals ([`fewest_held`]), so live
//!   queues spread over the `VkQueue`s whatever queues came and went before them (a guest's
//!   processes make and drop hundreds), and share one only where each holds one. One pass over
//!   the round robin's `VkQueue`s' counts, atomics alone, as the queue is made; nothing scans the
//!   command queues.
//! - **Unshared queues** (`Device::new_command_queue_unshared`): never moved and never joined. One
//!   takes, under every policy and verdict, a round robin's `VkQueue` that no command queue holds,
//!   the one used least recently; else one no long or unshared queue holds, the one fewest command
//!   queues hold; else, where a long or unshared queue holds every one, the one the fewest hold.
//!   One that shares as it is made is counted (`WorkCounts::unshared_queues_sharing_a_vk_queue`):
//!   a device of one graphics queue, or more queues live than `VkQueue`s. Where the device shares,
//!   a command queue on a `VkQueue` an unshared queue took leaves at its next idle point, where
//!   another of its family takes it; where it does not, the queue keeps it, and no later queue
//!   takes it while another is free of unshared queues. A queue's claim of a `VkQueue` and an
//!   unshared queue's are made so that two made at once never both succeed: each counts itself
//!   there first, then reads the other's count, both sequentially consistent.
//! - **Counted**: `WorkCounts::queue_moves`, `queue_joins`, `queues_gone_long`,
//!   `long_queue_settles`, `queue_moves_off_unshared`, `joins_withheld_from_unshared`,
//!   `picks_with_none_busy` and `joins_withheld_from_old_heads`. A choice at an idle point that
//!   moves nothing found no other `VkQueue` with batches in flight (`picks_with_none_busy`: work
//!   committed and not yet submitted is in none's flight), passed over a busy one for its oldest
//!   batch's age (`joins_withheld_from_old_heads`) or for the unshared queue that holds it, found
//!   the busy ones held by long or unshared queues, or stayed on its own busy, young one.
//!   Announced `queue-sharing` at the device's first join, a move onto a `VkQueue` with another
//!   queue's work in flight.

use crate::device::{lock, Device, QueueSlot};
use crate::in_place::DriverIdentity;
use crate::shader_cache::store::{hex, Lookup};
use metal2vulkan::device::QueueSharingPolicy;
use sha2::{Digest, Sha256};
use std::sync::atomic::{
    AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering,
};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// A command buffer whose GPU work ran for longer than this, in nanoseconds, by its own
/// timestamps, marks its command queue long: 2 ms. Measured against 10 ms on the RTX 5080 Laptop
/// GPU (each batch timed by its host-seen age as its `VkQueue`'s oldest), beside an application
/// queue of 30 ms command buffers: both mark that queue at its first
/// one, but only 2 ms also marks a queue of 2 to 10 ms command buffers, which at 10 ms the queues
/// sharing with it would stall behind again and again (the compositor's p99 4.81 ms against
/// 24.0 ms where the long queue stays home, 2.15 against 3.48 ms where it moves). A command buffer
/// the GPU runs in slices between other work counts those slices too, which only stops its queue
/// sharing: one-draw command buffers beside another process's 30 ms ones spanned up to 1.52 ms
/// there.
pub(crate) const LONG_HEAD: u64 = 2_000_000;

/// A busy `VkQueue` is joined only while its oldest batch in flight has run less than this, in
/// nanoseconds: 0.5 ms. A queue joining one whose oldest batch has run longer would likely wait
/// behind a long command buffer.
const YOUNG_HEAD: u64 = 500_000;

/// The switch cost, in nanoseconds, above which the device's check finds its GPU idle when
/// consecutive work comes from another `VkQueue`: 20 us. The RTX 5080 Laptop GPU measured 56 to
/// 62 us.
pub(crate) const SWITCH_IDLES: i64 = 20_000;

/// One `VkQueue`'s part in the sharing: when its oldest batch in flight became its oldest, its last
/// submission, and the command queues it holds.
#[derive(Default)]
pub(crate) struct SlotShare {
    /// The device shares `VkQueue`s, so the times below are kept ([`QueueSwitch::shares`]).
    watched: AtomicBool,
    /// The host time (`gpu_time::host_nanos`) its oldest batch in flight became its oldest: the
    /// host saw the batch before it complete, or it was submitted while the host saw every batch
    /// complete. 0 before the first.
    head_since: AtomicU64,
    /// The host time of its last submission; 0 before the first.
    last_submit: AtomicU64,
    /// The round robin's command queues it holds.
    homes: AtomicUsize,
    /// The long ones among them.
    long: AtomicUsize,
    /// The unshared command queues it holds.
    unshared: AtomicUsize,
}

impl SlotShare {
    /// A batch was submitted: `idle` where the host had seen every batch before it complete.
    pub(crate) fn submitted(&self, idle: bool) {
        if !self.watched.load(Ordering::Relaxed) {
            return;
        }
        let now = crate::gpu_time::host_nanos();
        if idle {
            self.head_since.store(now, Ordering::Relaxed);
        }
        self.last_submit.store(now, Ordering::Relaxed);
    }

    /// The host saw a batch complete: the next becomes the oldest in flight.
    pub(crate) fn completed(&self) {
        if self.watched.load(Ordering::Relaxed) {
            (self.head_since).store(crate::gpu_time::host_nanos(), Ordering::Relaxed);
        }
    }

    /// When its oldest batch in flight became its oldest, where known (the device shares).
    pub(crate) fn head_since(&self) -> Option<u64> {
        let since = self.head_since.load(Ordering::Relaxed);
        (self.watched.load(Ordering::Relaxed) && since != 0).then_some(since)
    }

    /// A round robin's command queue took it.
    pub(crate) fn took(&self) {
        self.homes.fetch_add(1, Ordering::SeqCst);
    }

    /// A round robin's command queue left it, long or not.
    pub(crate) fn left(&self, long: bool) {
        self.homes.fetch_sub(1, Ordering::SeqCst);
        if long {
            self.long.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// One of the command queues it holds went long.
    pub(crate) fn went_long(&self) {
        self.long.fetch_add(1, Ordering::SeqCst);
    }

    /// An unshared command queue took it.
    pub(crate) fn unshared_took(&self) {
        self.unshared.fetch_add(1, Ordering::SeqCst);
    }

    /// An unshared command queue left it.
    pub(crate) fn unshared_left(&self) {
        self.unshared.fetch_sub(1, Ordering::SeqCst);
    }

    /// Whether another command queue holds it beside the one unshared queue that took it.
    pub(crate) fn held_beside_unshared(&self) -> bool {
        self.homes.load(Ordering::SeqCst) > 0 || self.unshared.load(Ordering::SeqCst) > 1
    }
}

/// What a round robin's command queue that holds no uncompleted command buffer does
/// ([`Device::pick_vk_queue`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Choice {
    /// It keeps its `VkQueue`.
    Stay,
    /// It takes this busy `VkQueue`.
    Join(usize),
    /// It takes this idle (or, off an unshared queue's, least busy) `VkQueue`.
    Leave(usize),
}

/// What [`choose`] reads of one `VkQueue` of the family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Look {
    /// Its index among the device's `VkQueue`s.
    pub(crate) index: usize,
    /// Its batches in flight: timeline values submitted past those the host saw complete.
    pub(crate) depth: u64,
    /// How long its oldest batch in flight has been its oldest, where known.
    pub(crate) head_age: Option<u64>,
    /// When it was last submitted to (host nanoseconds; 0 before the first).
    pub(crate) last_submit: u64,
    /// A long queue holds it.
    pub(crate) long: bool,
    /// An unshared queue holds it.
    pub(crate) unshared: bool,
}

/// The choice of a round robin's command queue on `current` that holds no uncompleted command
/// buffer and is not long, among `looks`, every `VkQueue` of its family, `current` among them (the
/// module docs); with whether a busy, young `VkQueue` was passed over for the unshared queue it
/// holds alone.
pub(crate) fn choose(current: usize, looks: &[Look]) -> (Choice, bool) {
    let mut busy: Option<&Look> = None;
    let mut idle: Option<&Look> = None;
    let mut least: Option<&Look> = None;
    let mut withheld = false;
    let mut here = None;
    for look in looks {
        if look.index == current {
            here = Some(look);
        }
        if look.long {
            continue;
        }
        let young = look.head_age.is_some_and(|age| age < YOUNG_HEAD);
        if look.unshared {
            withheld |= look.depth > 0 && young;
            continue;
        }
        if least.is_none_or(|l| (look.depth, look.last_submit) < (l.depth, l.last_submit)) {
            least = Some(look);
        }
        if look.depth == 0 {
            if idle.is_none_or(|i| look.last_submit > i.last_submit) {
                idle = Some(look);
            }
        } else if young
            && busy.is_none_or(|b| (look.depth, look.last_submit) > (b.depth, b.last_submit))
        {
            busy = Some(look);
        }
    }
    let to = |look: &Look, make: fn(usize) -> Choice| match look.index == current {
        true => Choice::Stay,
        false => make(look.index),
    };
    let choice = match (busy, here) {
        (Some(busy), _) => to(busy, Choice::Join),
        (None, Some(here)) if !here.long && !here.unshared => Choice::Stay,
        (None, Some(here)) => match (idle, here.unshared) {
            (Some(idle), _) => to(idle, Choice::Leave),
            // Off an unshared queue's `VkQueue`, any other no long or unshared queue holds.
            (None, true) => least.map_or(Choice::Stay, |least| to(least, Choice::Leave)),
            (None, false) => Choice::Stay,
        },
        (None, None) => Choice::Stay,
    };
    (choice, withheld)
}

/// Whether a command queue on `current` passes over `look` for its oldest batch alone: another
/// `VkQueue` with batches in flight, which no long or unshared queue holds, whose oldest batch has
/// run [`YOUNG_HEAD`] or longer, or for a time not known.
pub(crate) fn old_busy(current: usize, look: &Look) -> bool {
    look.index != current
        && look.depth > 0
        && !look.long
        && !look.unshared
        && !look.head_age.is_some_and(|age| age < YOUNG_HEAD)
}

/// The `VkQueue` a long queue on `current` settles on, among `looks`: an idle one no long or
/// unshared queue holds, the one used least recently; `None` to stay.
pub(crate) fn settle(current: usize, looks: &[Look]) -> Option<usize> {
    looks
        .iter()
        .filter(|look| look.depth == 0 && !look.long && !look.unshared && look.index != current)
        .min_by_key(|look| look.last_submit)
        .map(|look| look.index)
}

/// The `VkQueue`s an unshared queue may take, among `looks` with each one's command queues, in the
/// order it tries them (the module docs): those no command queue holds, least recently used first;
/// then those no long or unshared queue holds, fewest command queues first. Empty for the round
/// robin's next.
pub(crate) fn unshared_order(looks: &[(Look, usize)]) -> Vec<usize> {
    let mut empty: Vec<&(Look, usize)> = looks
        .iter()
        .filter(|(look, homes)| *homes == 0 && !look.unshared)
        .collect();
    empty.sort_by_key(|(look, _)| look.last_submit);
    let mut fewest: Vec<&(Look, usize)> = looks
        .iter()
        .filter(|(look, homes)| *homes > 0 && !look.long && !look.unshared)
        .collect();
    fewest.sort_by_key(|(look, homes)| (*homes, look.depth, look.last_submit));
    empty
        .into_iter()
        .chain(fewest)
        .map(|(look, _)| look.index)
        .collect()
}

/// The round robin's `VkQueue` a command queue made now takes, among `held`, each one's live
/// command queues and whether it is barred (an unshared queue holds it, or the choice passed it
/// over): of those not barred, the one the fewest hold, the first from `start` on in turn among
/// equals; where every one is barred, the one the fewest hold, so. So live queues spread over the
/// `VkQueue`s whatever queues came and went before them, and share one only where each holds one.
pub(crate) fn fewest_held(start: usize, held: &[(usize, bool)]) -> usize {
    let count = held.len();
    (0..count)
        .map(|turn| (start + turn) % count)
        .min_by_key(|&index| (held[index].1, held[index].0))
        .unwrap_or(0)
}

/// What the device's check answered (`queue_switch_check.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Not known yet.
    Pending,
    /// The GPU idles at a switch: the switch cost measured above [`SWITCH_IDLES`].
    Idles,
    /// It does not.
    DoesNotIdle,
    /// No measurement: the device has no two `VkQueue`s of its primary family in the round
    /// robin, writes no timestamps there, calibrates none (timestamps of two `VkQueue`s compare
    /// only on a calibrated device) or answers GPU times from the host
    /// ([`queue_switch_check::timed`](crate::queue_switch_check::timed)), its policy is
    /// `PerQueue`, or a run failed.
    Unchecked,
}

impl Verdict {
    fn byte(self) -> u8 {
        match self {
            Verdict::Pending => 0,
            Verdict::Idles => 1,
            Verdict::DoesNotIdle => 2,
            Verdict::Unchecked => 3,
        }
    }

    fn of_byte(byte: u8) -> Verdict {
        match byte {
            1 => Verdict::Idles,
            2 => Verdict::DoesNotIdle,
            3 => Verdict::Unchecked,
            _ => Verdict::Pending,
        }
    }
}

/// A device's sharing (`DeviceShared::queue_switch`): its policy, its check's verdict and the switch
/// cost it measured, whether the check started and settled, and whether its thread runs, which
/// the device's teardown stops and waits for (`queue_switch_check.rs`).
pub(crate) struct QueueSwitch {
    /// The device's policy is `Adaptive`.
    adaptive: bool,
    verdict: AtomicU8,
    /// The switch cost measured, in nanoseconds (median gap between batches of two `VkQueue`s less
    /// that of one); 0 where none was.
    cost: AtomicI64,
    started: AtomicBool,
    settled: Mutex<bool>,
    wake: Condvar,
    /// The check's run in this process, in nanoseconds; 0 where none ended.
    took: AtomicU64,
    /// The check's thread runs: from before it is spawned until it let go of its objects and of
    /// its shares of the device's parts ([`began`](Self::began), [`ended`](Self::ended)).
    running: Mutex<bool>,
    /// Signalled as the check's thread ends.
    ending: Condvar,
    /// The device's teardown began: the check stops at its next step ([`stop`](Self::stop)).
    stopping: AtomicBool,
    /// The calls waiting for the verdict ([`QueueSwitch::wait`]), for which no round steps aside
    /// (`step_aside.rs`).
    waiters: AtomicU32,
    /// How long the teardown waited for the check's thread to end, in nanoseconds: 0 where it
    /// waited for none.
    stop_waited: AtomicU64,
    /// The crate's device tests' hold of the check between its rounds
    /// (`tests/teardown_threads.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) held: crate::freed_memory::held::Held,
    /// The crate's device tests' hold of the check between the two submissions of a round on one
    /// `VkQueue`, its queue's lock held (`tests/device_recovery.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) between: crate::freed_memory::held::Held,
    /// The crate's device tests' timings of the check's last two steps, in nanoseconds: the
    /// release of its Vulkan objects, and the drop of its shares of the device's parts.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    last_steps: [AtomicU64; 2],
    /// The crate's device tests' count of the fences of the check's own its held rounds waited
    /// for (`queue_switch_check.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    fences_waited: AtomicU64,
}

impl QueueSwitch {
    /// A device's of `policy`, its verdict pending.
    pub(crate) fn new(policy: QueueSharingPolicy) -> QueueSwitch {
        QueueSwitch {
            adaptive: policy == QueueSharingPolicy::Adaptive,
            verdict: AtomicU8::new(Verdict::Pending.byte()),
            cost: AtomicI64::new(0),
            started: AtomicBool::new(false),
            settled: Mutex::new(false),
            wake: Condvar::new(),
            took: AtomicU64::new(0),
            running: Mutex::new(false),
            ending: Condvar::new(),
            stopping: AtomicBool::new(false),
            waiters: AtomicU32::new(0),
            stop_waited: AtomicU64::new(0),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            held: Default::default(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            between: Default::default(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            last_steps: Default::default(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            fences_waited: AtomicU64::new(0),
        }
    }

    /// Whether the device shares `VkQueue`s now: its policy is `Adaptive` and its check found the
    /// GPU idling at a switch.
    pub(crate) fn shares(&self) -> bool {
        self.adaptive && self.verdict() == Verdict::Idles
    }

    /// The verdict now.
    pub(crate) fn verdict(&self) -> Verdict {
        Verdict::of_byte(self.verdict.load(Ordering::Acquire))
    }

    /// Note the check started; whether it had not been before.
    fn start(&self) -> bool {
        !self.started.swap(true, Ordering::AcqRel)
    }

    /// Settle the verdict as [`settle`](Self::settle) does, where it shares now with `slots`, the
    /// round robin's `VkQueue`s, keeping their times from here on ([`SlotShare::watched`]).
    pub(crate) fn settle_over(
        &self,
        slots: &[QueueSlot],
        verdict: Verdict,
        cost: i64,
        took: Option<Duration>,
    ) {
        if self.adaptive && verdict == Verdict::Idles {
            for slot in slots {
                slot.share.watched.store(true, Ordering::Relaxed);
            }
        }
        self.settle(verdict, cost, took);
    }

    /// The check's thread is about to be spawned: it runs until [`ended`](Self::ended).
    pub(crate) fn began(&self) {
        *lock(&self.running) = true;
    }

    /// The check's thread let go of its objects and of its shares of the device's parts (or was
    /// never spawned).
    pub(crate) fn ended(&self) {
        *lock(&self.running) = false;
        self.ending.notify_all();
    }

    /// Whether the device's teardown began, so the check stops at its next step.
    pub(crate) fn stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    /// Whether a call waits for the verdict ([`QueueSwitch::wait`]).
    pub(crate) fn awaited(&self) -> bool {
        self.waiters.load(Ordering::Acquire) > 0
    }

    /// The device's teardown: the check stops at its next step, and this waits until its thread
    /// ended, which holds no device handle, so never runs this teardown (`queue_switch_check.rs`).
    pub(crate) fn stop(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        let mut running = lock(&self.running);
        if !*running {
            return;
        }
        let waiting = Instant::now();
        while *running {
            running =
                (self.ending.wait(running)).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        let nanos = u64::try_from(waiting.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.stop_waited.store(nanos.max(1), Ordering::Release);
    }

    /// How long the device's teardown waited for the check's thread to end; `None` where it
    /// waited for none.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn stop_waited(&self) -> Option<Duration> {
        match self.stop_waited.load(Ordering::Acquire) {
            0 => None,
            nanos => Some(Duration::from_nanos(nanos)),
        }
    }

    /// Record that the check's last step `step` (0 its objects' release, 1 its shares' drop) ran
    /// from `from` until now.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn timed_step(&self, step: usize, from: Instant) {
        let nanos = u64::try_from(from.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.last_steps[step].store(nanos.max(1), Ordering::Release);
    }

    /// Record that a held round of the check waited for `fences` of its own.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn waited_fences(&self, fences: usize) {
        (self.fences_waited).fetch_add(fences as u64, Ordering::Relaxed);
    }

    /// The fences of its own the check's held rounds waited for so far
    /// ([`waited_fences`](Self::waited_fences)).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn fences_waited(&self) -> u64 {
        self.fences_waited.load(Ordering::Relaxed)
    }

    /// How long the check's last two steps took ([`timed_step`](Self::timed_step)).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn last_steps(&self) -> [Duration; 2] {
        (self.last_steps)
            .each_ref()
            .map(|t| Duration::from_nanos(t.load(Ordering::Acquire)))
    }

    /// Let the tests' hold of the check go once the device's teardown began, or once it was let
    /// go already.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn release_once_stopping(&self) {
        self.held.release_once(&self.stopping);
    }

    /// Settle the verdict, with the cost measured (or 0) and the check's run where it ran.
    fn settle(&self, verdict: Verdict, cost: i64, took: Option<Duration>) {
        self.cost.store(cost, Ordering::Release);
        if let Some(took) = took {
            let nanos = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX);
            self.took.store(nanos.max(1), Ordering::Release);
        }
        self.verdict.store(verdict.byte(), Ordering::Release);
        *lock(&self.settled) = true;
        self.wake.notify_all();
    }

    /// Wait until the verdict is known, or `timeout` passed: whether it is. Meanwhile the check's
    /// rounds step aside for nothing (`step_aside.rs`).
    fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        self.waiters.fetch_add(1, Ordering::AcqRel);
        let _waiting = Waiting(&self.waiters);
        let mut settled = lock(&self.settled);
        while !*settled {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            settled = match self.wake.wait_timeout(settled, left) {
                Ok((settled, _)) => settled,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        true
    }
}

/// A call waiting for the verdict ([`QueueSwitch::wait`]), counted until it returns.
struct Waiting<'a>(&'a AtomicU32);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The version of what the check measures and of how its record is named and framed: a change to
/// either bumps it, and a record of another version is never read.
pub(crate) const CHECK_VERSION: u32 = 1;

/// A record's first bytes.
const MAGIC: [u8; 8] = *b"M2VQSWT\0";

/// The domain tag of a record's name.
const NAME_TAG: &[u8] = b"metal2vulkan queue-switch check record\0";

/// What the check measures, in a record's name.
const CHECK_KIND: &[u8] = b"switch idle\0";

/// A record's length: magic, version, runtime, identity, verdict, cost, checksum.
const RECORD_BYTES: usize = 8 + 4 + 4 + 32 + 1 + 8 + 32;

/// The digest of `identity` that names its record and that the record holds again.
fn digest(identity: &DriverIdentity) -> [u8; 32] {
    identity.digest_for(NAME_TAG, CHECK_KIND, CHECK_VERSION)
}

/// The name of the record of `identity`'s verdict in the shader cache's content directory.
pub(crate) fn record_name(identity: &DriverIdentity) -> String {
    hex(&digest(identity))
}

/// The bytes of a record of `verdict` and `cost` for `identity`:
///
/// ```text
/// magic          8 bytes   "M2VQSWT\0"
/// version        u32       CHECK_VERSION
/// runtime        u32       RUNTIME_VERSION
/// identity       32 bytes  the identity's digest for this check
/// verdict        1 byte    Idles 1, DoesNotIdle 2
/// cost           i64       the switch cost measured, in nanoseconds
/// checksum       32 bytes  SHA-256 of every byte before it
/// ```
///
/// Integers are little-endian.
pub(crate) fn encode(identity: &DriverIdentity, verdict: Verdict, cost: i64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(RECORD_BYTES);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&CHECK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&metal2vulkan::cache::RUNTIME_VERSION.to_le_bytes());
    bytes.extend_from_slice(&digest(identity));
    bytes.push(verdict.byte());
    bytes.extend_from_slice(&cost.to_le_bytes());
    let checksum: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// The verdict and cost `bytes` record for `identity`: `None` for anything but a whole record of
/// this version and runtime, for this identity, whose checksum holds and whose verdict is a
/// measured answer.
pub(crate) fn decode(identity: &DriverIdentity, bytes: &[u8]) -> Option<(Verdict, i64)> {
    if bytes.len() != RECORD_BYTES {
        return None;
    }
    let (body, checksum) = bytes.split_at(RECORD_BYTES - 32);
    let computed: [u8; 32] = Sha256::digest(body).into();
    if computed[..] != *checksum {
        return None;
    }
    let (magic, rest) = body.split_at(8);
    let (version, rest) = rest.split_at(4);
    let (runtime, rest) = rest.split_at(4);
    let (identity_digest, rest) = rest.split_at(32);
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap_or([0xff; 4]));
    if magic != MAGIC
        || word(version) != CHECK_VERSION
        || word(runtime) != metal2vulkan::cache::RUNTIME_VERSION
        || identity_digest != digest(identity)
    {
        return None;
    }
    let verdict = Verdict::of_byte(rest[0]);
    let cost = i64::from_le_bytes(rest[1..9].try_into().ok()?);
    matches!(verdict, Verdict::Idles | Verdict::DoesNotIdle).then_some((verdict, cost))
}

impl Device {
    /// Whether the device shares `VkQueue`s now ([`QueueSwitch::shares`]).
    pub(crate) fn shares_queues(&self) -> bool {
        self.shared().queue_switch.shares()
    }

    /// The round robin's `VkQueue`s of the family of `VkQueue` `of`, each looked at now
    /// ([`Look`]), with how many command queues each holds.
    fn looks(&self, of: usize) -> impl Iterator<Item = (Look, usize)> + '_ {
        let shared = self.shared();
        let family = shared.queues[of].family;
        let now = crate::gpu_time::host_nanos();
        let round_robin = &shared.queues[..shared.round_robin()];
        (round_robin.iter().enumerate())
            .filter(move |(_, slot)| slot.family == family)
            .map(move |(index, slot)| look(index, slot, now))
    }

    /// Every round robin's `VkQueue`, each looked at now ([`Look`]), with how many command queues
    /// each holds: of one family where the device shares, of MoltenVK's four where it has them.
    fn round_robin_looks(&self) -> Vec<(Look, usize)> {
        let shared = self.shared();
        let now = crate::gpu_time::host_nanos();
        (shared.queues[..shared.round_robin()].iter().enumerate())
            .map(|(index, slot)| look(index, slot, now))
            .collect()
    }

    /// Each round robin's `VkQueue`'s live command queues, and whether an unshared queue holds
    /// it: what a test reads of the placement.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn round_robin_held(&self) -> Vec<(usize, bool)> {
        (self.round_robin_looks().into_iter())
            .map(|(look, homes)| (homes, look.unshared))
            .collect()
    }

    /// The round robin's `VkQueue` a command queue made now takes, counted there as one of the
    /// command queues it holds ([`SlotShare::took`]), or, `unshared`, as an unshared queue
    /// ([`SlotShare::unshared_took`]), `made` the round robin's command queues the device made
    /// before it: of the `VkQueue`s no unshared queue holds, the one the fewest live command
    /// queues hold, the first from the `made`th on in turn among equals ([`fewest_held`]); where
    /// an unshared queue holds every one, the one the fewest hold. Under every policy and verdict,
    /// as the queue is made: one pass over the round robin's `VkQueue`s, atomics alone, made again
    /// only where an unshared queue took the chosen one meanwhile (each counts itself there
    /// first, then reads the other's count, both sequentially consistent, so the two never both
    /// succeed: [`place_unshared`](Self::place_unshared)).
    pub(crate) fn take_round_robin(&self, made: u64, unshared: bool) -> usize {
        let queues = &self.shared().queues[..self.shared().round_robin()];
        let start = (made % queues.len() as u64) as usize;
        let mut passed = vec![false; queues.len()];
        loop {
            let held: Vec<(usize, bool)> = (queues.iter().zip(&passed))
                .map(|(slot, &passed)| {
                    let share = &slot.share;
                    let barred = passed || share.unshared.load(Ordering::SeqCst) > 0;
                    (share.homes.load(Ordering::SeqCst), barred)
                })
                .collect();
            let index = fewest_held(start, &held);
            let share = &queues[index].share;
            if unshared {
                share.unshared_took();
                return index;
            }
            share.took();
            if held[index].1 || share.unshared.load(Ordering::SeqCst) == 0 {
                return index;
            }
            // An unshared queue took it meanwhile: another, where one is free of them.
            share.left(false);
            passed[index] = true;
        }
    }

    /// The choice of a round robin's command queue on `current` that holds no uncompleted command
    /// buffer and is not long (the module docs): one pass over the `VkQueue`s of its family,
    /// atomics alone. Counted where a busy, young `VkQueue` was passed over for the unshared queue
    /// it holds alone (`WorkCounts::joins_withheld_from_unshared`); where no other `VkQueue` had
    /// batches in flight (`picks_with_none_busy`); else where it joined none and passed over a busy
    /// one for its oldest batch's age alone ([`old_busy`], `joins_withheld_from_old_heads`).
    pub(crate) fn pick_vk_queue(&self, current: usize) -> Choice {
        let mut looks = [Look {
            index: 0,
            depth: 0,
            head_age: None,
            last_submit: 0,
            long: false,
            unshared: false,
        }; crate::device::create::ROUND_ROBIN_CEILING as usize];
        let mut count = 0;
        for (look, _) in self.looks(current) {
            if count < looks.len() {
                looks[count] = look;
                count += 1;
            }
        }
        let looks = &looks[..count];
        let (choice, withheld) = choose(current, looks);
        let counts = &self.shared().counts;
        if withheld {
            counts.join_withheld_from_unshared();
        }
        if !looks
            .iter()
            .any(|look| look.index != current && look.depth > 0)
        {
            counts.picked_with_none_busy();
        } else if !matches!(choice, Choice::Join(_))
            && looks.iter().any(|look| old_busy(current, look))
        {
            counts.join_withheld_from_old_head();
        }
        choice
    }

    /// Whether an unshared queue holds `VkQueue` `index`.
    pub(crate) fn holds_unshared(&self, index: usize) -> bool {
        self.shared().queues[index]
            .share
            .unshared
            .load(Ordering::SeqCst)
            > 0
    }

    /// Where a long queue on `current` settles ([`settle`]): `None` to stay.
    pub(crate) fn settle_long_queue(&self, current: usize) -> Option<usize> {
        let looks: Vec<Look> = self.looks(current).map(|(look, _)| look).collect();
        settle(current, &looks)
    }

    /// Move a round robin's command queue (`long` where it is) from `VkQueue` `from` to `to`:
    /// refused, nothing changed, where an unshared queue took `to` meanwhile (the module docs).
    pub(crate) fn move_queue(&self, from: usize, to: usize, long: bool) -> bool {
        let queues = &self.shared().queues;
        let target = &queues[to].share;
        target.homes.fetch_add(1, Ordering::SeqCst);
        if target.unshared.load(Ordering::SeqCst) > 0 {
            target.homes.fetch_sub(1, Ordering::SeqCst);
            return false;
        }
        if long {
            target.went_long();
        }
        queues[from].share.left(long);
        true
    }

    /// The `VkQueue` a new unshared queue takes, counted there (the module docs), under every
    /// policy and verdict: one of the round robin's that no command queue holds, else one no long
    /// or unshared queue holds, the one the fewest hold ([`unshared_order`]); `None` where every
    /// one holds a long or unshared queue, for [`take_round_robin`](Self::take_round_robin)'s.
    pub(crate) fn place_unshared(&self) -> Option<usize> {
        let queues = &self.shared().queues;
        let looks = self.round_robin_looks();
        for index in unshared_order(&looks) {
            let share = &queues[index].share;
            let empty = looks
                .iter()
                .any(|(look, homes)| look.index == index && *homes == 0);
            if share.unshared.fetch_add(1, Ordering::SeqCst) == 0
                && (!empty || share.homes.load(Ordering::SeqCst) == 0)
            {
                return Some(index);
            }
            share.unshared.fetch_sub(1, Ordering::SeqCst);
        }
        None
    }

    /// Start the device's check on a thread of its own (`queue_switch_check.rs`), where its
    /// verdict is pending and no check started.
    pub(crate) fn start_queue_switch_check(&self) {
        let switch = &self.shared().queue_switch;
        if switch.verdict() != Verdict::Pending || !switch.start() {
            return;
        }
        if !crate::queue_switch_check::start(self) {
            switch.settle(Verdict::Unchecked, 0, None);
        }
    }

    /// As the device is made: its verdict where the check cannot measure (`Unchecked`, below) or
    /// a record holds it (settled, the round robin's `VkQueue`s then watched where it shares);
    /// else it waits for the check, which the first command queue of the round robin starts. A
    /// record that does not decode is no record. A device with two `VkQueue`s to share whose
    /// command buffers write no timestamps it reads is counted (`WorkCounts::queue_sharing_untimed`).
    pub(crate) fn recall_queue_switch_check(&self) {
        let shared = self.shared();
        let switch = &shared.queue_switch;
        if !switch.adaptive || crate::queue_switch_check::pair(self).is_none() {
            switch.settle(Verdict::Unchecked, 0, None);
            return;
        }
        if !crate::queue_switch_check::timed(self) {
            shared.counts.queue_sharing_untimed();
            switch.settle(Verdict::Unchecked, 0, None);
            return;
        }
        let identity = self.driver_identity();
        let name = record_name(&identity);
        let looked = (shared.shader_cache.content_store())
            .load_content(&name, |bytes| decode(&identity, &bytes));
        if let Lookup::Hit((verdict, cost)) = looked {
            shared.counts.queue_switch_check_recalled();
            self.settle_queue_switch(verdict, cost, None);
        }
    }

    /// Settle the device's verdict: where it shares now, the round robin's `VkQueue`s keep their
    /// times from here on ([`SlotShare::watched`]).
    pub(crate) fn settle_queue_switch(&self, verdict: Verdict, cost: i64, took: Option<Duration>) {
        let shared = self.shared();
        let slots = &shared.queues[..shared.round_robin()];
        shared.queue_switch.settle_over(slots, verdict, cost, took);
    }

    /// Start the device's check where its verdict is pending and none started, and wait until the
    /// verdict is known or `timeout` passed: whether it is. A runtime extension for the lanes and
    /// benchmarks that hold a device's command queues to the mapping they measure.
    #[doc(hidden)]
    pub fn settle_queue_switch_check(&self, timeout: Duration) -> bool {
        let current = self.current();
        current.note_sharing_read("Device::settle_queue_switch_check");
        current.settle_queue_switch_check_now(timeout)
    }

    /// [`Device::settle_queue_switch_check`] on the generation that serves (`device/line.rs`).
    pub(crate) fn settle_queue_switch_check_now(&self, timeout: Duration) -> bool {
        self.start_queue_switch_check();
        self.shared().queue_switch.wait(timeout)
    }

    /// No Metal counterpart: whether the device shares `VkQueue`s now, its policy `Adaptive` and
    /// its check's verdict `Idles` (the module docs). A runtime extension for the lanes that
    /// assert a queue joined another's `VkQueue` where the device shares.
    #[doc(hidden)]
    pub fn shares_vk_queues(&self) -> bool {
        self.current().shares_vk_queues_now()
    }

    /// [`Device::shares_vk_queues`] on the generation that serves (`device/line.rs`).
    pub(crate) fn shares_vk_queues_now(&self) -> bool {
        self.shares_queues()
    }

    /// The device's check's verdict, the switch cost it measured, how long it ran in this process
    /// (where it ran), and whether the device shares `VkQueue`s now, in words. A runtime extension
    /// for the lanes and benchmarks that report it.
    #[doc(hidden)]
    pub fn queue_switch_check_report(&self) -> String {
        let current = self.current();
        current.note_sharing_read("Device::queue_switch_check_report");
        current.queue_switch_check_report_now()
    }

    /// Where the device's queue limits changed its round robin's or presenting queues'
    /// `VkQueue`s from the queue plan's, say that its queue sharing was read through `call`
    /// (`DeviceOptions::queue_limits`).
    fn note_sharing_read(&self, call: &str) {
        let Some((taken, planned)) = self.shared().queues_limited() else {
            return;
        };
        let told = crate::degradation::Told::QueueLimitsReached;
        self.inform_once(told, crate::device::QUEUE_LIMITS_REACHED, || {
            let what = format!(
                "its queue sharing was read ({call}) with {} VkQueue(s) for the round robin and \
                 {} for presenting queues, where the plan gives {} and {}",
                taken.0, taken.1, planned.0, planned.1
            );
            crate::device::queue_limits_reached(self.name(), &what)
        });
    }

    /// [`Device::queue_switch_check_report`] on the generation that serves (`device/line.rs`).
    pub(crate) fn queue_switch_check_report_now(&self) -> String {
        let switch = &self.shared().queue_switch;
        let took = match switch.took.load(Ordering::Acquire) {
            0 => "not run here".to_owned(),
            nanos => format!("ran {:.2} ms", nanos as f64 / 1e6),
        };
        format!(
            "verdict {:?}, switch cost {:.1} us, {took}, sharing {}",
            switch.verdict(),
            switch.cost.load(Ordering::Acquire) as f64 / 1e3,
            switch.shares()
        )
    }
}

/// `slot`, at `index` among the device's `VkQueue`s, looked at `now`, with how many command
/// queues it holds.
fn look(index: usize, slot: &QueueSlot, now: u64) -> (Look, usize) {
    let share = &slot.share;
    let look = Look {
        index,
        depth: slot
            .submitted_value()
            .saturating_sub(slot.completed_value()),
        head_age: share.head_since().map(|since| now.saturating_sub(since)),
        last_submit: share.last_submit.load(Ordering::Relaxed),
        long: share.long.load(Ordering::SeqCst) > 0,
        unshared: share.unshared.load(Ordering::SeqCst) > 0,
    };
    (look, share.homes.load(Ordering::SeqCst))
}
