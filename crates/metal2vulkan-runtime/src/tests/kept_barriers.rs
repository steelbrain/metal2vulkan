//! `command_buffer/kept_barriers.rs`: what a batch takes of a `VkQueue`'s kept barrier command
//! buffers (`QueueKept::take`) as the queue's timeline moves: a kept one only once the batch that
//! last took it completed, so none is pending twice; a list met twice in one batch, and a value a
//! failed submission left to the next batch; where a batch records its own, past the bounds; and
//! which list a new one replaces once the kept ones fill the places. One that writes a timed
//! command buffer's timestamps is kept per slot and taken again only by a later holding of the
//! slot, once its timestamps were read. The command buffers are handles the test makes: no device
//! records or submits them.

use crate::command_buffer::kept_barriers::{QueueKept, SlotStamps, Take, KEPT, LISTS, STALE};
use crate::gpu_time::{Pool, PREFIXED, SLOTS};
use crate::tracker::{Access, Barrier};
use ash::vk::{self, Handle};

/// The list of barriers numbered `n`: one barrier, into stage bit `n`.
fn list(n: u32) -> Vec<Barrier> {
    vec![Barrier {
        src: Access::NONE,
        dst: Access::new(
            vk::PipelineStageFlags2::from_raw(1 << n),
            vk::AccessFlags2::MEMORY_READ,
        ),
    }]
}

/// A queue's kept command buffers, and the handles the test records them as.
#[derive(Default)]
struct Queue {
    kept: QueueKept,
    recorded: u64,
}

/// What a batch got.
#[derive(Debug, PartialEq, Eq)]
enum Got {
    /// A kept command buffer, the `n`th recorded.
    Kept(u64),
    /// One recorded now, the `n`th, into the list at `list`.
    Recorded { n: u64, list: usize },
    /// None: the batch records its own.
    Own,
}

impl Queue {
    /// What the batch that signals `at` takes for `barriers`, the timeline completed through
    /// `completed`, as `KeptBarriers::take` does: one recorded is a spare one where there is one.
    fn take(&mut self, barriers: &[Barrier], at: u64, completed: u64) -> Got {
        self.take_stamped(barriers, None, at, completed)
    }

    /// As [`take`](Self::take), for a command buffer that writes `stamps` where given.
    fn take_stamped(
        &mut self,
        barriers: &[Barrier],
        stamps: Option<SlotStamps>,
        at: u64,
        completed: u64,
    ) -> Got {
        match self.kept.take(barriers, stamps, at, completed) {
            Take::Kept(raw) => Got::Kept(raw.as_raw()),
            Take::Past => Got::Own,
            Take::Record(list) => {
                let raw = self.kept.spare().unwrap_or_else(|| {
                    self.recorded += 1;
                    vk::CommandBuffer::from_raw(self.recorded)
                });
                self.kept.recorded(list, raw, at, stamps);
                Got::Recorded {
                    n: raw.as_raw(),
                    list,
                }
            }
        }
    }
}

/// A batch that meets one list twice takes two command buffers, and the next batch takes the
/// same two once the first completed; one taken while they are pending records a third.
#[test]
fn a_list_met_twice_in_one_batch_takes_two_command_buffers() {
    let mut queue = Queue::default();
    let barriers = list(0);
    let recorded = |n| Got::Recorded { n, list: 0 };
    assert_eq!(queue.take(&barriers, 1, 0), recorded(1));
    assert_eq!(queue.take(&barriers, 1, 0), recorded(2));
    // Batch 2 while batch 1 runs: both pending.
    assert_eq!(queue.take(&barriers, 2, 0), recorded(3));
    // Batch 3 once batch 1 completed: its two, then a fourth (batch 2's is pending).
    assert_eq!(queue.take(&barriers, 3, 1), Got::Kept(1));
    assert_eq!(queue.take(&barriers, 3, 1), Got::Kept(2));
    assert_eq!(queue.take(&barriers, 3, 1), recorded(4));
    // Every batch completed: in the order the batches last took them.
    for n in [3, 1, 2, 4] {
        assert_eq!(queue.take(&barriers, 4, 3), Got::Kept(n));
    }
    assert_eq!(
        queue.kept.lists().0[0].1,
        [3, 1, 2, 4].map(vk::CommandBuffer::from_raw)
    );
}

/// A batch whose submission failed leaves its timeline value to the queue's next batch: the
/// command buffer it took waits for that batch, beside the one the next batch takes, and both
/// are taken again once it completed.
#[test]
fn a_failed_submission_leaves_its_command_buffer_to_the_next_batch_s_value() {
    let mut queue = Queue::default();
    let barriers = list(0);
    assert_eq!(queue.take(&barriers, 1, 0), Got::Recorded { n: 1, list: 0 });
    // Batch 1 completed. Batch 2 takes the kept one, and its submission fails.
    assert_eq!(queue.take(&barriers, 2, 1), Got::Kept(1));
    // The next batch signals 2 in its place: the one batch 2 took is not taken again.
    assert_eq!(queue.take(&barriers, 2, 1), Got::Recorded { n: 2, list: 0 });
    assert_eq!(queue.take(&barriers, 3, 1), Got::Recorded { n: 3, list: 0 });
    // Once 2 completed, both are taken again, and the third once 3 completed.
    assert_eq!(queue.take(&barriers, 4, 2), Got::Kept(1));
    assert_eq!(queue.take(&barriers, 4, 2), Got::Kept(2));
    assert_eq!(queue.take(&barriers, 4, 2), Got::Recorded { n: 4, list: 0 });
    assert_eq!(queue.take(&barriers, 5, 3), Got::Kept(3));
}

/// Past [`KEPT`] command buffers of a list all pending, and past [`LISTS`] lists all in use, a
/// batch records its own; the kept ones are taken again once they completed.
#[test]
fn past_the_bounds_a_batch_records_its_own() {
    let mut queue = Queue::default();
    let barriers = list(0);
    for n in 1..=KEPT as u64 {
        assert_eq!(queue.take(&barriers, 1, 0), Got::Recorded { n, list: 0 });
    }
    assert_eq!(queue.take(&barriers, 1, 0), Got::Own);
    assert_eq!(queue.take(&barriers, 2, 1), Got::Kept(1));

    let mut queue = Queue::default();
    for n in 0..LISTS as u32 {
        let got = queue.take(&list(n), 1, 0);
        assert_eq!(
            got,
            Got::Recorded {
                n: u64::from(n) + 1,
                list: n as usize
            }
        );
    }
    let extra = list(LISTS as u32);
    assert_eq!(queue.take(&extra, 1, 0), Got::Own);
    // Completed, but taken within the last `STALE` values: still in use.
    assert_eq!(queue.take(&extra, 2, 1), Got::Own);
    assert_eq!(queue.take(&extra, STALE, 1), Got::Own);
    assert_eq!(queue.take(&list(0), STALE, 1), Got::Kept(1));
}

/// Once [`LISTS`] lists are kept, a new list replaces the one taken least lately among those no
/// batch took in the last [`STALE`] values and whose command buffers all completed, and is
/// recorded into that one's command buffers (and kept from then on); lists still in use, or with
/// a command buffer pending, stay.
#[test]
fn a_list_out_of_use_gives_way_to_a_new_one() {
    let mut queue = Queue::default();
    // A program's first batches meet `LISTS` lists, one each, the first list twice.
    assert_eq!(queue.take(&list(0), 1, 0), Got::Recorded { n: 1, list: 0 });
    assert_eq!(queue.take(&list(0), 1, 0), Got::Recorded { n: 2, list: 0 });
    for n in 1..LISTS as u32 {
        let at = u64::from(n) + 1;
        let got = queue.take(&list(n), at, at - 1);
        assert_eq!(
            got,
            Got::Recorded {
                n: at + 1,
                list: n as usize
            }
        );
    }
    // Long after: lists 1 to 3 still in use, list 4's command buffer pending.
    let at = 1000;
    for n in 1..=3 {
        assert!(matches!(queue.take(&list(n), at, at - 1), Got::Kept(_)));
    }
    assert!(matches!(queue.take(&list(4), at + 1, at), Got::Kept(_)));
    // The frame's lists, each new: the first replaces list 0 (taken least lately, at 1), and is
    // recorded into its two command buffers in turn.
    let frame = |n| list(LISTS as u32 + n);
    assert_eq!(
        queue.take(&frame(0), at + 2, at),
        Got::Recorded { n: 2, list: 0 }
    );
    assert_eq!(
        queue.take(&frame(0), at + 2, at),
        Got::Recorded { n: 1, list: 0 }
    );
    let (lists, spare) = queue.kept.lists();
    assert_eq!(lists[0].0, frame(0));
    assert!(spare.is_empty(), "{spare:?}");
    // Then lists 5, 6 and 7, each into its own command buffer, never 1 to 3 (in use) or 4
    // (pending).
    for (n, list) in [(1, 5), (2, 6), (3, 7)] {
        let got = queue.take(&frame(n), at + 2, at);
        let n = list as u64 + 2;
        assert_eq!(got, Got::Recorded { n, list }, "frame list {n}");
    }
    assert_eq!(queue.take(&frame(4), at + 2, at), Got::Own);
    // Each is kept from then on.
    for n in 0..4 {
        assert!(matches!(
            queue.take(&frame(n), at + 3, at + 2),
            Got::Kept(_)
        ));
    }
}

/// More than [`LISTS`] lists in use in turn replace none, whichever comes first in a batch: the
/// first kept stay kept, and the rest are recorded by their batches, batch after batch.
#[test]
fn lists_in_use_in_turn_replace_none() {
    let mut queue = Queue::default();
    // The ones met last in each batch are met first: the kept ones are completed, not yet taken.
    let lists: Vec<u32> = (0..LISTS as u32 + 2).rev().collect();
    let mut kept = Vec::new();
    for round in 0..4 * STALE {
        let at = round + 1;
        for &n in &lists {
            let got = queue.take(&list(n), at, at - 1);
            match got {
                Got::Recorded { .. } if round == 0 => kept.push(n),
                Got::Kept(_) if round > 0 && kept.contains(&n) => {}
                Got::Own if !kept.contains(&n) => {}
                got => panic!("round {round}, list {n}: {got:?}, kept {kept:?}"),
            }
        }
    }
    assert_eq!(kept.len(), LISTS);
}

/// A start command buffer of the slot whose first query is `first`, for the holding `lease`.
fn start(first: u32, lease: u64) -> Option<SlotStamps> {
    Some(SlotStamps {
        queries: vk::QueryPool::from_raw(7),
        first,
        lease,
        start: true,
        after: false,
        end: false,
    })
}

/// A command buffer that writes a slot's timestamps is taken again only by a later holding of the
/// slot (its timestamps read, the slot given back), once the batch that last took it completed:
/// never by the holding that took it, though its batch completed, nor while it is pending. Each
/// slot has its own; the end's are apart from the start's, and from the barriers alone.
#[test]
fn a_stamped_command_buffer_waits_for_its_slot_s_timestamps_to_be_read() {
    let mut queue = Queue::default();
    let barriers = list(0);
    // Holding 1 of slot 0: its first batch, value 1.
    assert_eq!(
        queue.take_stamped(&barriers, start(0, 1), 1, 0),
        Got::Recorded { n: 1, list: 0 }
    );
    // Its batch completed, the slot still held, its timestamps not read: not taken again.
    assert_eq!(queue.take_stamped(&barriers, start(0, 1), 2, 1), Got::Own);
    // Holding 2 of slot 0 (holding 1 read and gave it back), while value 1 is pending: not yet.
    assert_eq!(queue.take_stamped(&barriers, start(0, 2), 3, 0), Got::Own);
    // Once value 1 completed: the kept one.
    assert_eq!(
        queue.take_stamped(&barriers, start(0, 2), 3, 1),
        Got::Kept(1)
    );
    // Slot 1 has its own; so has the end of slot 0, and the barriers alone.
    assert_eq!(
        queue.take_stamped(&barriers, start(2, 3), 4, 3),
        Got::Recorded { n: 2, list: 0 }
    );
    let end = Some(SlotStamps {
        start: false,
        end: true,
        ..start(0, 2).unwrap()
    });
    assert_eq!(
        queue.take_stamped(&barriers, end, 3, 1),
        Got::Recorded { n: 3, list: 1 }
    );
    assert_eq!(queue.take(&barriers, 4, 3), Got::Recorded { n: 4, list: 0 });
    // Holding 4 of slot 0 takes the start that holding 2 last took, once value 3 completed.
    assert_eq!(queue.take_stamped(&barriers, start(0, 4), 5, 2), Got::Own);
    assert_eq!(
        queue.take_stamped(&barriers, start(0, 4), 5, 3),
        Got::Kept(1)
    );
}

/// Kept command buffers marked to be recorded again (`QueueKept::invalidate`, before a trim round
/// of the spares) are each recorded again, into the same command buffer, as a batch would take
/// it, once its last batch completed: one pending meanwhile leaves the batch to record another.
/// Those recorded since are kept as before; so is a slot's stamped one.
#[test]
fn kept_command_buffers_marked_are_recorded_again_before_a_batch_takes_them() {
    let mut queue = Queue::default();
    let barriers = list(0);
    assert_eq!(queue.take(&barriers, 1, 0), Got::Recorded { n: 1, list: 0 });
    assert_eq!(queue.take(&barriers, 2, 0), Got::Recorded { n: 2, list: 0 });
    assert_eq!(queue.take(&barriers, 3, 1), Got::Kept(1));
    assert_eq!(
        queue.take_stamped(&barriers, start(0, 1), 3, 1),
        Got::Recorded { n: 3, list: 0 }
    );
    queue.kept.invalidate();
    // 2, whose batch completed: recorded again; 1, pending: another one.
    assert_eq!(queue.take(&barriers, 4, 2), Got::Recorded { n: 2, list: 0 });
    assert_eq!(queue.take(&barriers, 4, 2), Got::Recorded { n: 4, list: 0 });
    // 1 once its batch completed: recorded again; then each is kept.
    assert_eq!(queue.take(&barriers, 5, 3), Got::Recorded { n: 1, list: 0 });
    assert_eq!(queue.take(&barriers, 6, 5), Got::Kept(2));
    assert_eq!(queue.take(&barriers, 6, 5), Got::Kept(4));
    assert_eq!(queue.take(&barriers, 6, 5), Got::Kept(1));
    // The slot's, at its next holding: recorded again, then kept.
    assert_eq!(
        queue.take_stamped(&barriers, start(0, 2), 6, 5),
        Got::Recorded { n: 3, list: 0 }
    );
    assert_eq!(
        queue.take_stamped(&barriers, start(0, 3), 7, 6),
        Got::Kept(3)
    );
}

/// The holdings of one slot rise: a slot is handed out again only once the command buffer that
/// held it dropped it, after the completion turn read its timestamps, and the next holding's
/// lease is above every earlier one's, so a kept command buffer's slot was read before it is
/// taken again.
#[test]
fn a_slot_s_next_holding_has_a_higher_lease() {
    let pool = Pool::new(vk::QueryPool::null(), SLOTS, PREFIXED);
    let held = pool.take().unwrap();
    let other = pool.take().unwrap();
    assert_ne!(held.first(), other.first());
    assert!(other.lease() > held.lease());
    let (first, lease) = (held.first(), held.lease());
    drop(held);
    let again = pool.take().unwrap();
    assert_eq!(again.first(), first);
    assert!(again.lease() > lease && again.lease() > other.lease());
}

/// Past [`KEPT`] slots' command buffers of a list, a batch of another slot records its own;
/// every slot kept is taken again by its next holding.
#[test]
fn past_kept_slots_a_batch_records_its_own() {
    let mut queue = Queue::default();
    let barriers = list(0);
    for slot in 0..KEPT as u32 {
        let got = queue.take_stamped(&barriers, start(2 * slot, 1), 1, 0);
        assert_eq!(
            got,
            Got::Recorded {
                n: u64::from(slot) + 1,
                list: 0
            }
        );
    }
    assert_eq!(
        queue.take_stamped(&barriers, start(2 * KEPT as u32, 1), 1, 0),
        Got::Own
    );
    for slot in 0..KEPT as u32 {
        let got = queue.take_stamped(&barriers, start(2 * slot, 2), 2, 1);
        assert_eq!(got, Got::Kept(u64::from(slot) + 1));
    }
}
