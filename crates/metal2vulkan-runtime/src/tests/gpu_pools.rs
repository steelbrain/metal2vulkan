//! A queue's blocks of the GPU pools (`gpu_pools.rs`): the command buffers of one queue share its
//! block, made once, round after round; another queue has a block of its own; a block no lease
//! held for [`QUIET`] goes as a later lease ends, and a queue's goes with the queue, to the
//! device's spare where it has none, which the next queue that needs a block takes.

use super::super::gpu_pools::{QueueBlocks, QUIET};
use std::time::{Duration, Instant};

/// A lease of `queue`'s block from `blocks` at `now`: the block (`made + 1` where the lease
/// makes it) and the leases holding it now.
fn lease(blocks: &mut QueueBlocks<u32>, queue: u64, made: &mut u32, now: Instant) -> (u32, usize) {
    if let Some(taken) = blocks.take(queue, now, |&block| block) {
        return taken;
    }
    *made += 1;
    let (block, leases, extra) = blocks.made(queue, *made, now, |&block| block);
    assert_eq!(extra, None, "no other lease made one meanwhile");
    (block, leases)
}

/// Two command buffers of one queue in flight at once, round after round, share one block, made
/// in the first round alone.
#[test]
fn command_buffers_of_one_queue_share_its_block() {
    let mut blocks = QueueBlocks::default();
    let mut made = 0;
    let start = Instant::now();
    for round in 0..8u32 {
        let now = start + Duration::from_millis(16) * round;
        let first = lease(&mut blocks, 7, &mut made, now);
        let second = lease(&mut blocks, 7, &mut made, now);
        assert_eq!((first, second), ((1, 1), (1, 2)), "round {round}");
        assert!(blocks.give(7, now).is_empty(), "round {round}");
        assert!(blocks.give(7, now).is_empty(), "round {round}");
    }
    assert_eq!((made, blocks.blocks()), (1, 1));
}

/// Another queue's command buffers take a block of their own.
#[test]
fn another_queue_takes_its_own_block() {
    let mut blocks = QueueBlocks::default();
    let mut made = 0;
    let now = Instant::now();
    assert_eq!(lease(&mut blocks, 1, &mut made, now), (1, 1));
    assert_eq!(lease(&mut blocks, 2, &mut made, now), (2, 1));
    assert_eq!(lease(&mut blocks, 1, &mut made, now), (1, 2));
    assert_eq!((made, blocks.blocks()), (2, 2));
}

/// A block two leases made at once is made twice: the second made becomes the device's spare,
/// and where it holds one already, it is released.
#[test]
fn a_block_made_meanwhile_is_the_spare_or_released() {
    let mut blocks = QueueBlocks::default();
    let now = Instant::now();
    let read = |&block: &u32| block;
    assert_eq!(blocks.take(3, now, read), None);
    assert_eq!(blocks.take(3, now, read), None);
    assert_eq!(blocks.made(3, 10, now, read), (10, 1, None));
    assert_eq!(blocks.made(3, 11, now, read), (10, 2, None));
    assert_eq!((blocks.blocks(), blocks.spare()), (1, true));
    assert_eq!(
        blocks.take(4, now, read),
        Some((11, 1)),
        "the next queue takes the spare"
    );
    assert_eq!(blocks.take(5, now, read), None);
    assert_eq!(blocks.take(5, now, read), None);
    assert_eq!(blocks.made(5, 12, now, read), (12, 1, None));
    assert_eq!(blocks.made(5, 13, now, read), (12, 2, None));
    assert_eq!(
        blocks.made(5, 14, now, read),
        (12, 3, Some(14)),
        "a spare is held already"
    );
    assert_eq!((blocks.blocks(), blocks.spare()), (3, true));
}

/// A queue's block no lease held for the quiet period goes as a lease of another queue ends, the
/// first to the device's spare and the next released; a block in use stays, however old, and one
/// whose last lease ended less than the quiet period before stays too. A queue that rested takes
/// the spare where its block went.
#[test]
fn a_block_no_lease_held_for_the_quiet_period_goes() {
    let mut blocks = QueueBlocks::default();
    let mut made = 0;
    let start = Instant::now();
    // Queues 1 and 4 lease once and stop; queue 2 keeps one lease held throughout; queue 3
    // leases on.
    for queue in [1, 4] {
        lease(&mut blocks, queue, &mut made, start);
        assert!(blocks.give(queue, start).is_empty());
    }
    lease(&mut blocks, 2, &mut made, start);
    let mut now = start;
    let mut released = Vec::new();
    while now < start + QUIET + Duration::from_millis(100) {
        now += Duration::from_millis(16);
        lease(&mut blocks, 3, &mut made, now);
        let gone = blocks.give(3, now);
        if now.duration_since(start) < QUIET {
            assert!(gone.is_empty(), "nothing goes before the quiet period");
        }
        released.extend(gone);
    }
    assert_eq!(
        released,
        [2],
        "queue 1's block is the spare, queue 4's goes"
    );
    assert_eq!((blocks.blocks(), blocks.spare()), (2, true));
    assert_eq!(made, 4);
    assert_eq!(
        lease(&mut blocks, 4, &mut made, now),
        (1, 1),
        "queue 4 takes the spare"
    );
    assert_eq!((made, blocks.blocks(), blocks.spare()), (4, 3, false));
}

/// A queue that goes lets its block go, to the device's spare where it has none, else to be
/// released; the device's teardown takes the rest, the spare's included.
#[test]
fn blocks_go_with_their_queue_and_with_the_device() {
    let mut blocks = QueueBlocks::default();
    let mut made = 0;
    let now = Instant::now();
    for queue in [1, 2, 3] {
        lease(&mut blocks, queue, &mut made, now);
        assert!(blocks.give(queue, now).is_empty());
    }
    assert_eq!(blocks.forget(2), None, "the spare");
    assert_eq!(blocks.forget(2), None);
    assert_eq!(blocks.forget(1), Some(1), "a spare is held already");
    assert_eq!(blocks.drain(), [3, 2]);
    assert_eq!((blocks.blocks(), blocks.spare()), (0, false));
}

/// Queues made and gone one after another, each with a command buffer, make one block: each
/// takes the one the queue before it let go.
#[test]
fn a_queue_made_later_takes_the_block_another_let_go() {
    let mut blocks = QueueBlocks::default();
    let mut made = 0;
    let start = Instant::now();
    for queue in 0..8 {
        let now = start + Duration::from_millis(16) * queue;
        assert_eq!(lease(&mut blocks, queue.into(), &mut made, now), (1, 1));
        assert!(blocks.give(queue.into(), now).is_empty());
        assert_eq!(
            blocks.forget(queue.into()),
            None,
            "queue {queue}'s block is the spare"
        );
    }
    assert_eq!((made, blocks.blocks(), blocks.spare()), (1, 0, true));
}
