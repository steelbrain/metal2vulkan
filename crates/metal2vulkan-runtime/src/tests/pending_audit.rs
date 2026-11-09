//! The pending audit's ledger (`pending_audit.rs`): what carries each command buffer, a secondary
//! through each primary that executes it, and which command buffers a check finds pending against
//! the timelines as read.

use crate::pending_audit::{Ledger, Pending};
use ash::vk::{self, Handle};

/// Command buffer `n`.
fn buffer(n: u64) -> vk::CommandBuffer {
    vk::CommandBuffer::from_raw(n)
}

/// Timeline semaphore `n`.
fn timeline(n: u64) -> vk::Semaphore {
    vk::Semaphore::from_raw(n)
}

/// A read of timelines whose counters are `counters` (semaphore, counter); none of another.
fn reader(counters: &[(u64, u64)]) -> impl FnMut(vk::Semaphore) -> Option<u64> + '_ {
    move |semaphore| {
        (counters.iter())
            .find(|&&(of, _)| timeline(of) == semaphore)
            .map(|&(_, counter)| counter)
    }
}

/// A command buffer is pending while the timeline of a submission that carries it is below that
/// submission's value, found at each check until the timeline reaches it; then it leaves the
/// ledger, and no later check reads a timeline for it. One never submitted is pending nowhere.
#[test]
fn a_command_buffer_is_pending_until_its_timeline_reaches_the_value() {
    let mut ledger = Ledger::default();
    ledger.submitted((timeline(1), 5), &[buffer(10), buffer(11)]);
    let pending = Pending {
        buffer: buffer(10),
        semaphore: timeline(1),
        value: 5,
        counter: 4,
    };
    assert_eq!(ledger.pending(&[buffer(10)], reader(&[(1, 4)])), [pending]);
    assert_eq!(ledger.pending(&[buffer(10)], reader(&[(1, 4)])), [pending]);
    let both = [buffer(10), buffer(11)];
    assert!(ledger.pending(&both, reader(&[(1, 5)])).is_empty());
    let mut reads = 0;
    let found = ledger.pending(&both, |_| {
        reads += 1;
        Some(0)
    });
    assert_eq!((found, reads), (Vec::new(), 0), "both left the ledger");
    assert!(ledger.pending(&[buffer(12)], reader(&[(1, 0)])).is_empty());
}

/// A check reads each timeline once, however many command buffers it carries.
#[test]
fn a_check_reads_each_timeline_once() {
    let mut ledger = Ledger::default();
    ledger.submitted((timeline(1), 5), &[buffer(10), buffer(11)]);
    ledger.submitted((timeline(2), 3), &[buffer(11)]);
    let mut reads = Vec::new();
    let found = ledger.pending(&[buffer(10), buffer(11)], |semaphore| {
        reads.push(semaphore);
        Some(4)
    });
    assert_eq!(reads, [timeline(1), timeline(2)]);
    assert_eq!(found.len(), 2, "both carried by the submission at 5");
}

/// A command buffer submitted again waits for its latest submission on each timeline, and for
/// each timeline apart.
#[test]
fn a_command_buffer_waits_for_its_latest_submission_on_each_timeline() {
    let mut ledger = Ledger::default();
    ledger.submitted((timeline(1), 5), &[buffer(10)]);
    ledger.submitted((timeline(1), 7), &[buffer(10)]);
    ledger.submitted((timeline(2), 3), &[buffer(10)]);
    let found = ledger.pending(&[buffer(10)], reader(&[(1, 6), (2, 2)]));
    let values: Vec<(vk::Semaphore, u64, u64)> = (found.iter())
        .map(|pending| (pending.semaphore, pending.value, pending.counter))
        .collect();
    assert_eq!(values, [(timeline(1), 7, 6), (timeline(2), 3, 2)]);
    assert!(ledger
        .pending(&[buffer(10)], reader(&[(1, 7), (2, 3)]))
        .is_empty());
}

/// A secondary command buffer is carried by each primary that executes it, through that
/// primary's submissions; a primary begun again executes nothing it did before, so its next
/// submission carries none of it.
#[test]
fn a_secondary_is_carried_through_each_primary_that_executes_it() {
    let mut ledger = Ledger::default();
    let (primary, other, secondary) = (buffer(1), buffer(2), buffer(3));
    ledger.executes(primary, &[secondary]);
    ledger.executes(other, &[secondary]);
    ledger.submitted((timeline(1), 5), &[primary]);
    ledger.submitted((timeline(2), 9), &[other]);
    let pending = Pending {
        buffer: secondary,
        semaphore: timeline(2),
        value: 9,
        counter: 8,
    };
    assert_eq!(
        ledger.pending(&[secondary], reader(&[(1, 5), (2, 8)])),
        [pending]
    );
    assert!(ledger.pending(&[secondary], reader(&[(2, 9)])).is_empty());
    ledger.begun(primary, false);
    ledger.submitted((timeline(1), 6), &[primary]);
    assert!(ledger.pending(&[secondary], reader(&[(1, 5)])).is_empty());
    assert_eq!(ledger.pending(&[primary], reader(&[(1, 5)])).len(), 1);
}

/// A submission that failed carries nothing: neither its command buffers nor the secondaries they
/// execute.
#[test]
fn a_failed_submission_carries_nothing() {
    let mut ledger = Ledger::default();
    let (primary, secondary) = (buffer(1), buffer(2));
    ledger.executes(primary, &[secondary]);
    ledger.submitted((timeline(1), 5), &[primary]);
    ledger.withdrawn((timeline(1), 5), &[primary]);
    assert!(ledger
        .pending(&[primary, secondary], reader(&[(1, 0)]))
        .is_empty());
}

/// A timeline that cannot be read (a lost device) finds nothing, and forgets nothing.
#[test]
fn a_timeline_that_cannot_be_read_finds_nothing() {
    let mut ledger = Ledger::default();
    ledger.submitted((timeline(1), 5), &[buffer(10)]);
    assert!(ledger.pending(&[buffer(10)], |_| None).is_empty());
    assert_eq!(ledger.pending(&[buffer(10)], reader(&[(1, 4)])).len(), 1);
}

/// A submission again checks the command buffers it submits and the secondaries they execute,
/// but those last begun for simultaneous use, which may be pending in several submissions.
#[test]
fn a_submission_again_checks_all_but_those_begun_for_simultaneous_use() {
    let mut ledger = Ledger::default();
    let (once, simultaneous, secondary, kept) = (buffer(1), buffer(2), buffer(3), buffer(4));
    ledger.begun(once, false);
    ledger.begun(simultaneous, true);
    ledger.begun(kept, true);
    ledger.executes(once, &[secondary, kept]);
    assert_eq!(
        ledger.submitted_again(&[once, simultaneous]),
        [once, secondary]
    );
    ledger.begun(simultaneous, false);
    assert_eq!(ledger.submitted_again(&[simultaneous]), [simultaneous]);
}
