//! Other command queues' work held back from a `VkQueue` an unshared queue shares with them
//! (`UnsharedIsolationPolicy::Bounded`, the default): only about [`BUDGET`] of their expected GPU
//! work is submitted ahead of the unshared queue's next command buffer, or [`IDLE_BUDGET`] while
//! it is quiet.
//!
//! - **Why.** A `VkQueue` runs its submissions in order. On a device whose graphics family has
//!   one queue (RADV, ANV, lavapipe), every command queue submits to it,
//!   `Device::new_command_queue_unshared`'s (a compositor's) included, so a compositor's light
//!   command buffer waits for every batch submitted before it there. On a Radeon 780M (RADV,
//!   Mesa 26.2.6), a compositor committing one draw every 16 ms beside an application keeping
//!   three 30 ms command buffers in flight waited 76 ms at the median and 112 ms at worst, every
//!   one of its frames over 16 ms, with 7 batches in flight ahead of it on average. The kernel
//!   does not reorder within a `VkQueue`, and another `VkDevice`'s work (its own kernel context on
//!   i915) interleaves only between submissions, never inside one, whatever its priority.
//! - **Where.** A batch of a command queue not made unshared, on a `VkQueue` that an unshared
//!   queue holds (`SlotShare::unshared`), on a device whose policy is `Bounded`. Where an unshared
//!   queue has a `VkQueue` of its own, no other command queue's batch goes there, and nothing is
//!   held.
//! - **Two budgets.** While an unshared queue there committed a command buffer within [`ACTIVE`]
//!   (50 ms; `SlotShare::unshared_committed`, one atomic store at the commit call), the bound
//!   below is [`BUDGET`] (2 ms). Otherwise it is [`IDLE_BUDGET`] (16 ms): a compositor gone quiet
//!   (a still screen, a guest window not redrawn) holds back only an application that keeps 16 ms
//!   or more of expected work queued, and that one still has 16 ms queued as its next batch goes
//!   (`WorkCounts::heads_let_go_unshared_idle` counts the heads the 2 ms bound would have held).
//!   The trade: the unshared queue's first command buffer after such a stretch goes in behind
//!   every batch submitted before its commit, at most about 16 ms of expected work plus the last
//!   batch let go; the 2 ms bound applies from that commit on. The batches stay in the [`Flight`]
//!   either way. A batch already held as the 50 ms run out goes at the next look, which its gate
//!   (the oldest batch's completion) or its time brings, as before; one of the unshared queue's
//!   own command buffers that waits past [`ACTIVE`] after its commit (on an event, a host step)
//!   goes in behind what the 16 ms bound let go meanwhile.
//! - **The bound.** Such a batch, otherwise ready, is submitted only while less than the budget of
//!   expected GPU work of other queues' batches is left in flight on its `VkQueue` ([`Flight`]):
//!   the expected runs of the batches submitted there and not seen complete, less what the oldest
//!   one has run since it became the oldest, up to its own expected run. So an unshared queue's
//!   command buffer goes in behind about one batch of theirs, and the next batch goes in before
//!   the one ahead of it ends, so the GPU is not left idle between them. A held batch keeps its
//!   place in the scheduler's order (`scheduler/order.rs`): a later commit of another queue whose
//!   footprint runs in commit order with it waits behind it, as behind any held batch.
//! - **Expected runs** ([`Expected`]). Each command queue's: the host's view of how long its
//!   recent batches ran as their `VkQueue`'s oldest in flight (`queue_wait.rs`'s runs), raised at
//!   once by a longer one and halved toward a shorter one, so a queue whose batches got shorter
//!   takes its new length within a few batches, while one long batch counts at once. A run several
//!   batches were seen complete in shares among the other queues' batches among them in
//!   proportion to their expected runs (`Shares`), so a light batch seen complete with a long one
//!   keeps a light run; an unshared queue's takes no share.
//! - **Unknown length.** A batch of a queue with no run seen yet may be of any length: while one
//!   is in flight, no other batch of the other queues goes there until it is seen complete, which
//!   gives its queue a run. So an application that commits several command buffers at once does
//!   not put them all ahead of a compositor's before the first is seen; it costs the GPU one
//!   completion's turn, once per such queue.
//! - **When it goes.** A pass that holds a batch so leaves the scheduler's thread waiting for the
//!   completion of the oldest batch in flight there (a gate on its value, `scheduler/mod.rs`'s
//!   `Awaits`), and, where the oldest batch's expected run still covers it, for the time the work
//!   left ahead falls under the budget. The hold is decided on the value the host published as
//!   completed: a batch at or below it counts as left, though the completion that published it
//!   has yet to take it out of the flight ([`Flight::ahead`]), so the gate a hold leaves is one
//!   that value has not reached, and its completion wakes the thread. A committing or waiting thread's own pass leaves such a batch to the scheduler's
//!   thread, which it wakes. A pass that finds such a batch let go and an unshared queue's
//!   command buffer ready together on its `VkQueue` sends the unshared one first
//!   (`scheduler/order.rs`, "Heads put first"): a pass the host ran late, after the room came,
//!   would otherwise put the whole batch ahead of it.
//! - **Commits not yet taken in.** A pass takes the scheduler's inbox in once, at its start, then
//!   submits one batch after another (about 0.3 ms each on the Radeon 780M), so an unshared
//!   queue's commit posted meanwhile waits in the inbox until the pass ends, where "Heads put
//!   first" cannot see it. A batch whose room comes during that pass is held while an unshared
//!   queue's commit on its `VkQueue` posted before its room came is not yet in the order
//!   ([`behind_commits`], `SlotShare::unshared_commits`); the pass that takes the commit in, which
//!   the passer makes before it sleeps, sends the unshared one first where it is ready, and the
//!   held batch next. Commits posted after its room came never hold it, so a compositor committing
//!   during every pass holds it for one pass at most. Without the hold, beside a guest compositor
//!   on the Radeon 780M, an application's batch let go during such a pass went in ahead of the
//!   frame's command buffers in 106 to 129 of about 611 frames a 12 s walk, and those were 7 to 23
//!   of the 9 to 27 frames answered more than a refresh after their take. Counted:
//!   `WorkCounts::heads_held_for_unshared_commits`.
//! - **Commits on their way.** A commit taken in may still wait for the runtime before its batch
//!   goes: for its host step (its gate, work already submitted, then its recording on a steps'
//!   thread), for its submission checks, or for an event value that a signal already submitted
//!   reaches and that has not landed (a shared event's, whose wait is held on the host). A pass
//!   that finds another queue's batch's room meanwhile would send that batch ahead of it, and the
//!   unshared queue's command buffer would then wait for the whole batch. So a batch whose room
//!   came is held too while an unshared queue's commit on its `VkQueue`, posted before its room
//!   came, leads its line on its way so ([`behind_commits`]; each commit carries its place among
//!   those posted there, `SlotShare::posted_unshared`). The step's end, the checks' end or the
//!   signal's landing wakes the scheduler's thread, whose next pass sends the unshared one first
//!   and the held batch next. Each look asks again, and such a commit holds nothing while it waits
//!   for anything else: an event value no submitted signal reaches, a binary semaphore's signal, a
//!   pipeline's making, a CPU decode, a rebind's encoding, or a batch committed before it that it
//!   waits behind (which may be the held batch itself: neither then waits for the other). The
//!   bound: the steps, checks and landings of the unshared commits posted before the room came.
//!   Beside a guest compositor on the Radeon 780M (reims's traced walk ld91, 12 s), another queue's
//!   batch went in while a compositor's command buffer was so on its way ahead of 33 of the 46
//!   frames answered more than a refresh after their take: 20 at its host step, 13 waiting for
//!   the landing of its other unshared queue's signal. Counted:
//!   `WorkCounts::heads_held_for_unshared_commits`.
//! - **Passes of other threads.** A thread about to block, and a thread that just committed with
//!   `CommandBuffer::commit_and_submit`, pass for themselves, and submit only the batches that
//!   need nothing of the scheduler's thread (`scheduler/mod.rs`'s `submit_plain_ready`). Where
//!   such a pass leaves an unshared queue's head that the scheduler's own look would find ready
//!   and send first (one with a submission or scheduled handler, for a thread about to block; one
//!   a switch of storage forms rides on; one on a `VkQueue` whose calls are slow, for a committing
//!   thread, `device/submit_calls.rs`), or one on its way
//!   (above), it leaves the other queues' batches held back on that head's `VkQueue` to the
//!   scheduler's thread too, which it wakes: that thread's next pass sends the unshared one first
//!   and them next, or holds them while it is on its way. They wait for that wake and pass, or the
//!   hold above, no more. An unshared head that waits for anything else (an event value no
//!   submitted signal reaches, a binary semaphore's signal, a pipeline's making, a decode, a
//!   rebind) leaves them to the pass, as that thread's own pass lets them go then. Without the
//!   rule such a pass sent another queue's batch ahead of the unshared one
//!   (`tests::queue_share_device`).
//! - **Deferred commits** (a known limit). Under `RenderLoweringPolicy::Worker`, an unshared
//!   queue's commit left to the device's lowering thread (`render/queued_commits.rs`) counts as
//!   posted only once its own work ran and its batches reach the scheduler, so a batch whose room
//!   comes while that lowering runs goes ahead of it. Counting the commit as posted at its call
//!   would hold such a batch through the lowering of every commit queued before it, which run in
//!   commit order across all queues on one runner: the program's encoders still to lower, not one
//!   pass. With less than the budget of other queues' work left on the GPU as the room came, the
//!   GPU would wait for up to that backlog, less about 2 ms, each time the compositor committed
//!   behind one. The default `RenderLoweringPolicy::AtCall` defers no commit.
//! - **What it costs.** A batch longer than its queue's recent ones still holds the unshared
//!   queue's command buffer for its whole length: one command buffer's batch is one submission,
//!   which the GPU runs to its end. The next batch goes in about [`BUDGET`] before the one ahead
//!   ends, where `Off` keeps every ready batch queued, so the GPU waits for the scheduler's thread
//!   where the host's CPU is loaded: on the Iris Xe at a load of about 10, an application's 5 ms
//!   command buffers ran 181.8 a second against 189.3 (docs/FAST-PATHS.md); on a quiet host no
//!   throughput was lost. Every batch on such a `VkQueue` costs a place in its [`Flight`] and a
//!   lock of its own at the submission and the completion. Counted:
//!   `WorkCounts::heads_held_for_unshared_isolation` (each look that held a head so).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The expected GPU work of other queues' batches, in nanoseconds, below which another of their
/// batches is submitted to a `VkQueue` an unshared queue holds, while one there committed within
/// [`ACTIVE`]: 2 ms, the length past which a command buffer marks its queue long where queues
/// share `VkQueue`s (`queue_share::LONG_HEAD`).
pub(crate) const BUDGET: u64 = 2_000_000;

/// The budget, in nanoseconds, while no unshared queue there committed within [`ACTIVE`]: 16 ms,
/// the owner's line for an intolerable wait. The unshared queue's first command buffer after a
/// quiet stretch waits behind at most about this much expected work plus the last batch let go
/// (one batch, which runs to its end); an application that keeps less queued is not held at all
/// (a guest compute benchmark's super-resolution workload on the Radeon 780M kept at most 6.48 ms queued ahead of a
/// batch, with no hold), and one that keeps more still has 16 ms queued as its next batch goes.
pub(crate) const IDLE_BUDGET: u64 = 16_000_000;

/// How long, in nanoseconds, an unshared queue's commit keeps the hold on its `VkQueue` (the
/// module docs): 50 ms. A compositor commits when what it shows changed, so its commits follow its
/// content: every refresh on a busy screen, but 33.3 ms apart for a game capped at 30 frames a
/// second and 41.7 ms for a 24-frame video. 50 ms keeps the hold on between such commits; one
/// refresh of a 50 Hz display (20 ms) would leave a 30-frame game's compositor behind all of the
/// game's queued work, the case the hold exists for. The runtime knows no refresh rate: the
/// unshared queue's presenting is the program's.
pub(crate) const ACTIVE: u64 = 50_000_000;

/// Whether a batch whose room came is held behind an unshared queue's commit (the module docs,
/// "Commits not yet taken in" and "Commits on their way"), its `VkQueue`'s unshared commits posted
/// and taken in so far being `posted` and `taken`, and `on_their_way` the places among those
/// posted (`SlotShare::posted_unshared`) of the unshared commits leading their lines there on
/// their way: `noted` keeps the commits posted at the first look that found its room, those it
/// waits behind, while one of them is not yet taken in or is on its way; any posted later never
/// hold it.
pub(crate) fn behind_commits(
    noted: &mut Option<u64>,
    (posted, taken): (u64, u64),
    mut on_their_way: impl Iterator<Item = u64>,
) -> bool {
    let noted = *noted.get_or_insert(posted);
    taken < noted || on_their_way.any(|place| place <= noted)
}

/// A command queue's expected run of one batch (the module docs), in nanoseconds; 0 before the
/// first is seen.
#[derive(Debug, Default)]
pub(crate) struct Expected(AtomicU64);

impl Expected {
    /// The run its next batch is expected to take: none where none was seen.
    pub(crate) fn nanos(&self) -> Option<u64> {
        match self.0.load(Ordering::Relaxed) {
            0 => None,
            nanos => Some(nanos),
        }
    }

    /// One of its batches ran `run` nanoseconds: taken at once where longer, halved toward it
    /// where shorter.
    pub(crate) fn observed(&self, run: u64) {
        let run = run.max(1);
        let _ = (self.0).try_update(Ordering::Relaxed, Ordering::Relaxed, |was| {
            Some(match was {
                0 => run,
                was if run >= was => run,
                was => was - (was - run) / 2,
            })
        });
    }
}

/// How a batch taken on a `VkQueue` enters its [`Flight`].
pub(crate) enum Flies<'a> {
    /// Not at all: no unshared queue holds the `VkQueue` under `UnsharedIsolationPolicy::Bounded`,
    /// and the device's queues take no turns (`QueueTurnPolicy::CommitOrder`).
    Not,
    /// As an unshared queue's (its `QueueGen::id`): in flight, with no expected run of other
    /// queues' work, and no turn of its own.
    Unshared(u64),
    /// As another queue's: its `QueueGen::id`, and its queue's expected run.
    Other(u64, &'a Arc<Expected>),
}

/// One batch in a [`Flight`].
#[derive(Debug)]
struct InFlight {
    /// Its value of its `VkQueue`'s timeline.
    value: u64,
    /// Its command queue (`QueueGen::id`).
    queue: u64,
    /// Its expected run, counted in [`Flight::expected`]; 0 for an unshared queue's, and for one
    /// whose queue had no run seen.
    expected: u64,
    /// Its queue had no run seen at its submission: of unknown length.
    unseen: bool,
    /// Its queue's expected run, which its run updates; none for an unshared queue's.
    of: Option<Arc<Expected>>,
}

/// How a run several batches were seen complete in shares among the other queues' batches among
/// them ([`Flight::completed`]): in proportion to each one's expected run at its submission, so a
/// light batch seen complete with a long one of another queue (a compositor's draw behind an
/// application's command buffer, the two seen together) keeps a light run and the long one its
/// own. The batches of queues with no run seen take the run beyond the others' expected runs, an
/// equal part each (all of it, equally, where every one is of unknown length), the others then
/// sharing the rest. An unshared queue's batch takes no share.
#[derive(Clone, Copy, Debug)]
struct Shares {
    /// The run.
    run: u64,
    /// The expected runs at their submission of the batches of known length, summed.
    known: u64,
    /// How many are of unknown length.
    unseen: u64,
}

impl Shares {
    /// The shares of `run` among `batches`, which were seen complete together.
    fn among<'a>(batches: impl Iterator<Item = &'a InFlight>, run: u64) -> Self {
        let (known, unseen) =
            (batches.filter(|batch| batch.of.is_some())).fold((0, 0), |(known, unseen), batch| {
                match batch.unseen {
                    true => (known, unseen + 1),
                    false => (known + batch.expected, unseen),
                }
            });
        Self { run, known, unseen }
    }

    /// `batch`'s share (the type's docs).
    fn of(self, batch: &InFlight) -> u64 {
        // What the batches of known length take: the whole run where none is of unknown length.
        let known_run = match self.unseen {
            0 => self.run,
            _ => self.run.min(self.known),
        };
        match batch.unseen {
            true => (self.run - known_run) / self.unseen.max(1),
            false => {
                let share = u128::from(known_run) * u128::from(batch.expected);
                u64::try_from(share / u128::from(self.known.max(1))).unwrap_or(u64::MAX)
            }
        }
    }
}

/// The batches taken on one `VkQueue` an unshared queue holds, or whose queues take turns
/// (`queue_turns.rs`), and not seen complete (`queue_wait::Ahead::flight`), under that `VkQueue`'s
/// waits' lock.
#[derive(Debug, Default)]
pub(crate) struct Flight {
    /// In the order of their values, which rises.
    batches: VecDeque<InFlight>,
    /// The expected runs of the other queues' among them, summed.
    expected: u64,
    /// How many among them are of unknown length.
    unseen: usize,
    /// How many among them each command queue not made unshared has (`queue_turns.rs`), by its id.
    turns: HashMap<u64, usize>,
    /// The latest batch's command queue and the host time it was taken, and the same of the
    /// latest of another command queue before it (`queue_turns.rs`, **Two budgets**): 0 for none.
    latest: [(u64, u64); 2],
}

/// Whether a batch may go to a `VkQueue` now ([`Flight::ahead`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Room {
    /// Less than the budget of other queues' expected work is left in flight.
    Go,
    /// The budget or more, or a batch of unknown length is in flight: held until the oldest batch
    /// in flight, of value `through`, completes, or, where only the budget holds it, until `after`
    /// nanoseconds from now, where the oldest batch's expected run brings the work left under the
    /// budget then.
    Hold { after: Option<u64>, through: u64 },
    /// Goes under [`IDLE_BUDGET`] where [`BUDGET`] would hold it: no unshared queue there
    /// committed within [`ACTIVE`] ([`Flight::room`]).
    Idle,
}

/// Whether an unshared queue of a `VkQueue` that last committed at `last` (`gpu_time::host_nanos`;
/// 0 where none has) is active when the host's time is `now`: that commit is less than [`ACTIVE`]
/// old, one stored after `now` was read none old.
pub(crate) fn active(last: u64, now: u64) -> bool {
    last != 0 && now.saturating_sub(last) < ACTIVE
}

impl Flight {
    /// A batch took `value` at host time `now` (`gpu_time::host_nanos`), entering as `flies`
    /// says.
    pub(crate) fn taken(&mut self, value: u64, flies: Flies<'_>, now: u64) {
        let (queue, expected, of) = match flies {
            Flies::Not => return,
            Flies::Unshared(queue) => (queue, Some(0), None),
            Flies::Other(queue, of) => {
                *self.turns.entry(queue).or_default() += 1;
                (queue, of.nanos(), Some(Arc::clone(of)))
            }
        };
        match self.latest[0].0 == queue {
            true => self.latest[0].1 = now,
            false => self.latest = [(queue, now), self.latest[0]],
        }
        let unseen = expected.is_none();
        let expected = expected.unwrap_or(0);
        self.expected += expected;
        self.unseen += usize::from(unseen);
        self.batches.push_back(InFlight {
            value,
            queue,
            expected,
            unseen,
            of,
        });
    }

    /// The host saw the timeline reach `value`, `run` nanoseconds after the oldest batch in flight
    /// became so: each batch at or below `value` leaves, and each other queue's among them updates
    /// its queue's expected run with its share of `run` ([`Shares`]). An unshared queue's batch
    /// takes no share: a compositor's light batch is seen complete with the application's batch
    /// ahead of it, and a share of that batch's run would cut the application's expected one.
    pub(crate) fn completed(&mut self, value: u64, run: u64) {
        let done = self
            .batches
            .iter()
            .take_while(|batch| batch.value <= value)
            .count();
        let shares = Shares::among(self.batches.iter().take(done), run);
        for batch in self.batches.drain(..done) {
            self.expected -= batch.expected;
            self.unseen -= usize::from(batch.unseen);
            let share = shares.of(&batch);
            if let Some(of) = batch.of {
                of.observed(share);
                if let Some(count) = self.turns.get_mut(&batch.queue) {
                    *count -= 1;
                    if *count == 0 {
                        self.turns.remove(&batch.queue);
                    }
                }
            }
        }
    }

    /// Whether another queue's batch may go now, under [`BUDGET`] where an unshared queue there
    /// is `active` ([`active`]), else under [`IDLE_BUDGET`], [`Room::Idle`] where only that let it
    /// go; [`ahead`](Self::ahead)'s other arguments.
    pub(crate) fn room(&self, ran: u64, completed: u64, active: bool) -> Room {
        if active {
            return self.ahead(ran, completed, BUDGET);
        }
        match self.ahead(ran, completed, IDLE_BUDGET) {
            Room::Go if self.ahead(ran, completed, BUDGET) != Room::Go => Room::Idle,
            room => room,
        }
    }

    /// Whether another queue's batch may go now under `budget`, the oldest batch in flight having
    /// run `ran` nanoseconds since it became so (the module docs), the host having seen the
    /// timeline reach `completed`. A batch at or below `completed` counts as left: its completion
    /// was published before it takes it out ([`Flight::completed`]), so the answer follows the
    /// published value alone, which a held batch's gate is read against. The oldest batch past
    /// it became so at that completion, so none of its run counts yet.
    pub(crate) fn ahead(&self, ran: u64, completed: u64, budget: u64) -> Room {
        let left_already = self
            .batches
            .iter()
            .take_while(|batch| batch.value <= completed);
        let (gone, gone_expected, gone_unseen) =
            left_already.fold((0, 0, 0), |(gone, expected, unseen), batch| {
                (
                    gone + 1,
                    expected + batch.expected,
                    unseen + usize::from(batch.unseen),
                )
            });
        let Some(oldest) = self.batches.get(gone) else {
            return Room::Go;
        };
        let through = oldest.value;
        if self.unseen > gone_unseen {
            return Room::Hold {
                after: None,
                through,
            };
        }
        let credit = match gone {
            0 => ran.min(oldest.expected),
            _ => 0,
        };
        let left = self.expected - gone_expected - credit;
        if left < budget {
            return Room::Go;
        }
        let over = left - budget + 1;
        let after = (oldest.expected - credit >= over).then_some(over);
        Room::Hold { after, through }
    }

    /// Whether a batch of command queue `queue` may go now on its turn (`queue_turns.rs`): under
    /// [`BUDGET`] where another command queue took a batch here within [`ACTIVE`] of `now`
    /// (`gpu_time::host_nanos`), else under [`IDLE_BUDGET`], [`Room::Idle`] where only that let
    /// it go; one held under [`BUDGET`] that [`IDLE_BUDGET`] would let go is held at most until
    /// that queue's last batch here is [`ACTIVE`] old. `expected` is its queue's expected run now;
    /// `ran` and `completed` are [`ahead`](Self::ahead)'s.
    pub(crate) fn turn(
        &self,
        (queue, expected): (u64, Option<u64>),
        ran: u64,
        completed: u64,
        now: u64,
    ) -> Room {
        let [last, before] = self.latest;
        let (_, other) = match last.0 == queue {
            true => before,
            false => last,
        };
        if active(other, now) {
            // Held under the 2 ms budget where 16 ms would let it go: it goes once that queue's
            // last batch here is `ACTIVE` old, at the latest.
            return match self.own_ahead((queue, expected), ran, completed, BUDGET) {
                Room::Hold { after, through }
                    if !matches!(
                        self.own_ahead((queue, expected), ran, completed, IDLE_BUDGET),
                        Room::Hold { .. }
                    ) =>
                {
                    let quiet = (other + ACTIVE).saturating_sub(now);
                    Room::Hold {
                        after: Some(after.map_or(quiet, |after| after.min(quiet))),
                        through,
                    }
                }
                room => room,
            };
        }
        match self.own_ahead((queue, expected), ran, completed, IDLE_BUDGET) {
            Room::Go if self.own_ahead((queue, expected), ran, completed, BUDGET) != Room::Go => {
                Room::Idle
            }
            room => room,
        }
    }

    /// Whether a batch of command queue `queue`, whose expected run is `expected` now, may go
    /// under `budget` of its own queue's work left in flight (`queue_turns.rs`): each of its
    /// batches here counts that run, less what the oldest batch here ran (`ran`, as in
    /// [`ahead`](Self::ahead)) where it is one of them; one of a queue with no run seen holds the
    /// next until it completes. Held until the batch of value `through` completes
    /// ([`turn_through`]: the one whose completion lets it go, or makes the batch of its queue
    /// whose run does so the oldest), or, where only the budget holds it and its own oldest runs
    /// now, until `after`
    /// nanoseconds from now, where that batch's expected run brings its work left under the
    /// budget then.
    pub(crate) fn own_ahead(
        &self,
        (queue, expected): (u64, Option<u64>),
        ran: u64,
        completed: u64,
        budget: u64,
    ) -> Room {
        let (gone, gone_own) = (self.batches.iter())
            .take_while(|batch| batch.value <= completed)
            .fold((0, 0), |(gone, own), batch| {
                (gone + 1, own + usize::from(batch.queue == queue))
            });
        let own = self.turns.get(&queue).map_or(0, |&count| count - gone_own);
        let Some(oldest) = self.batches.get(gone) else {
            return Room::Go;
        };
        if own == 0 {
            return Room::Go;
        }
        let in_flight = self.batches.range(gone..);
        let Some(run) = expected else {
            // Its first batch here gives its queue a run once it completes.
            let first = (in_flight.clone()).find(|batch| batch.queue == queue);
            return Room::Hold {
                after: None,
                through: first.map_or(oldest.value, |batch| batch.value),
            };
        };
        let running = gone == 0 && oldest.queue == queue;
        let credit = match running {
            true => ran.min(run),
            false => 0,
        };
        let left = (own as u64).saturating_mul(run) - credit;
        if left < budget {
            return Room::Go;
        }
        let over = left - budget + 1;
        let after = (oldest.queue == queue && run - credit >= over).then_some(over);
        // Its batches that must complete before its work left falls under the budget, each at
        // `run`: all but those the budget holds.
        let holds = usize::try_from((budget - 1) / run).unwrap_or(usize::MAX);
        let must = own - holds.min(own - 1);
        Room::Hold {
            after,
            through: turn_through(in_flight, queue, must),
        }
    }

    /// The batches in flight, of every queue.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.batches.len()
    }
}

/// The value whose completion lets a held batch of command queue `queue` go, or makes the batch
/// of its queue whose run does so the oldest here (`queue_turns.rs`, **When it goes**): the batch
/// before the `must`-th of its queue's in `in_flight`, the batches not seen complete, oldest
/// first, or that batch itself where it is the oldest. So a completion wakes the scheduler's
/// thread for it only once it may go, or once its time to go can be told.
fn turn_through<'a>(in_flight: impl Iterator<Item = &'a InFlight>, queue: u64, must: usize) -> u64 {
    let (mut seen, mut before, mut last) = (0, None, 0);
    for batch in in_flight {
        if batch.queue == queue {
            seen += 1;
            if seen == must {
                return before.unwrap_or(batch.value);
            }
        }
        (before, last) = (Some(batch.value), batch.value);
    }
    last
}

impl crate::Device {
    /// Whether other queues' batches are held back from `VkQueue` `index` (the module docs): an
    /// unshared queue holds it, and the device's policy is `Bounded`.
    pub(crate) fn holds_back_for_unshared(&self, index: usize) -> bool {
        self.holds_unshared(index)
            && self.compile_device().policy().unshared_isolation
                == metal2vulkan::device::UnsharedIsolationPolicy::Bounded
    }

    /// How a batch of command queue `queue` (`unshared` where made so) taken on `VkQueue` `index`
    /// enters its flight, `expected` its queue's expected run: every batch where the device's
    /// queues take turns (`queue_turns.rs`), else where other queues' work is held back there for
    /// an unshared queue.
    pub(crate) fn flies<'a>(
        &self,
        index: usize,
        (queue, unshared): (u64, bool),
        expected: &'a Arc<Expected>,
    ) -> Flies<'a> {
        if !self.takes_turns() && !self.holds_back_for_unshared(index) {
            return Flies::Not;
        }
        match unshared {
            true => Flies::Unshared(queue),
            false => Flies::Other(queue, expected),
        }
    }
}
