//! Command queues sharing `VkQueue`s (`queue_share.rs`), the parts that need no device: the choice
//! a queue makes at an idle point, where a long queue settles, where an unshared queue goes, and
//! the record that keeps the device's check's verdict.
//!
//! - **The choice.** A queue joins the busy `VkQueue` whose oldest batch is young with the most
//!   batches in flight (then the latest submission), never one a long or unshared queue holds;
//!   with none it stays, unless a long or unshared queue holds its own, when it leaves for the
//!   most recently used idle one (off an unshared queue's, any other where none is idle).
//! - **Settling.** A long queue takes the least recently used idle `VkQueue` no long or unshared
//!   queue holds, or stays.
//! - **Unshared.** An unshared queue tries the `VkQueue`s no queue holds, least recently used
//!   first, then those no long or unshared queue holds, fewest queues first.
//! - **The record.** Its verdict and cost read back for its identity alone, of a measured verdict
//!   alone.

use crate::in_place::DriverIdentity;
use crate::queue_share::{
    choose, decode, encode, record_name, settle, unshared_order, Choice, Look, Verdict,
};

/// A `VkQueue` at `index`, idle, used at `last_submit`, held by no long or unshared queue.
fn idle(index: usize, last_submit: u64) -> Look {
    Look {
        index,
        depth: 0,
        head_age: Some(10_000_000),
        last_submit,
        long: false,
        unshared: false,
    }
}

/// A `VkQueue` at `index` with `depth` batches in flight, its oldest `age` ns old.
fn busy(index: usize, depth: u64, age: u64, last_submit: u64) -> Look {
    Look {
        depth,
        head_age: Some(age),
        ..idle(index, last_submit)
    }
}

#[test]
fn a_queue_joins_the_deepest_young_busy_vk_queue_and_never_a_long_or_unshared_one() {
    let looks = [
        idle(0, 5),
        busy(1, 2, 100_000, 7),
        busy(2, 5, 100_000, 6),
        // Deeper, but its oldest batch has run 0.6 ms: maybe a long command buffer.
        busy(3, 9, 600_000, 9),
        Look {
            long: true,
            ..busy(4, 12, 10_000, 9)
        },
        Look {
            unshared: true,
            ..busy(5, 20, 10_000, 9)
        },
    ];
    let (choice, withheld) = choose(0, &looks);
    assert_eq!(choice, Choice::Join(2));
    assert!(
        withheld,
        "the unshared queue's busy young VkQueue was passed over"
    );
    // Equal depths go to the latest submission.
    let tied = [idle(0, 1), busy(1, 3, 1_000, 8), busy(2, 3, 1_000, 9)];
    assert_eq!(choose(0, &tied), (Choice::Join(2), false));
    // Already on the best, it stays.
    assert_eq!(choose(2, &tied).0, Choice::Stay);
    // An unknown head age (the device just began sharing) is never young.
    let unknown = [
        idle(0, 1),
        Look {
            head_age: None,
            ..busy(1, 4, 0, 2)
        },
    ];
    assert_eq!(choose(0, &unknown).0, Choice::Stay);
}

#[test]
fn with_no_young_busy_vk_queue_a_queue_stays_unless_a_long_or_unshared_queue_holds_its_own() {
    let quiet = [idle(0, 3), idle(1, 9), busy(2, 4, 900_000, 9)];
    assert_eq!(choose(0, &quiet).0, Choice::Stay);
    // A long queue holds its own: the most recently used idle one other than it.
    let long_here = [
        Look {
            long: true,
            ..idle(0, 3)
        },
        idle(1, 9),
        idle(2, 4),
    ];
    assert_eq!(choose(0, &long_here).0, Choice::Leave(1));
    // A long queue holds its own and none other is idle: it stays.
    let long_busy = [
        Look {
            long: true,
            ..idle(0, 3)
        },
        busy(1, 1, 900_000, 9),
    ];
    assert_eq!(choose(0, &long_busy).0, Choice::Stay);
    // An unshared queue holds its own and none other is idle: the least busy other.
    let unshared_busy = [
        Look {
            unshared: true,
            ..busy(0, 3, 900_000, 3)
        },
        busy(1, 4, 900_000, 9),
        busy(2, 2, 900_000, 9),
        Look {
            long: true,
            ..idle(3, 1)
        },
    ];
    assert_eq!(choose(0, &unshared_busy).0, Choice::Leave(2));
    // Every other is long or unshared: it stays.
    let fenced = [
        Look {
            unshared: true,
            ..idle(0, 3)
        },
        Look {
            long: true,
            ..idle(1, 9)
        },
    ];
    assert_eq!(choose(0, &fenced).0, Choice::Stay);
}

#[test]
fn a_long_queue_settles_on_the_least_recently_used_idle_vk_queue_no_long_or_unshared_queue_holds() {
    let looks = [
        Look {
            long: true,
            ..busy(0, 1, 0, 9)
        },
        idle(1, 7),
        idle(2, 3),
        Look {
            unshared: true,
            ..idle(3, 1)
        },
        Look {
            long: true,
            ..idle(4, 2)
        },
        busy(5, 2, 10, 0),
    ];
    assert_eq!(settle(0, &looks), Some(2));
    assert_eq!(settle(0, &looks[..1]), None, "with no other, it stays");
    assert_eq!(settle(0, &[looks[0], looks[3], looks[4], looks[5]]), None);
}

#[test]
fn an_unshared_queue_tries_empty_vk_queues_first_then_the_least_held() {
    let looks = [
        (idle(0, 9), 2),
        (idle(1, 5), 0),
        (busy(2, 3, 10, 2), 0),
        (
            Look {
                unshared: true,
                ..idle(3, 1)
            },
            0,
        ),
        (
            Look {
                long: true,
                ..idle(4, 1)
            },
            1,
        ),
        (busy(5, 1, 10, 4), 1),
    ];
    assert_eq!(unshared_order(&looks), vec![2, 1, 5, 0]);
    assert!(unshared_order(&[looks[3]]).is_empty());
}

fn identity() -> DriverIdentity {
    DriverIdentity {
        vendor_id: 0x10de,
        device_id: 0x2c58,
        driver_version: 0x99e1_6000,
        driver_id: 4,
        driver_name: "NVIDIA".to_owned(),
        driver_info: "615.71.09".to_owned(),
        device_uuid: [7; 16],
    }
}

#[test]
fn the_record_reads_back_its_measured_verdict_for_its_identity_alone() {
    let identity = identity();
    for (verdict, cost) in [(Verdict::Idles, 57_300), (Verdict::DoesNotIdle, -1_200)] {
        let bytes = encode(&identity, verdict, cost);
        assert_eq!(decode(&identity, &bytes), Some((verdict, cost)));
        let mut changed = bytes.clone();
        changed[60] ^= 1;
        assert_eq!(decode(&identity, &changed), None, "a byte changed");
        assert_eq!(decode(&identity, &bytes[1..]), None, "cut short");
        let mut other = identity.clone();
        other.driver_info.push('x');
        assert_eq!(decode(&other, &bytes), None, "another identity");
        assert_ne!(record_name(&identity), record_name(&other));
    }
    for unmeasured in [Verdict::Pending, Verdict::Unchecked] {
        assert_eq!(decode(&identity, &encode(&identity, unmeasured, 0)), None);
    }
    // Its name is not the in-place check's of the same identity.
    assert_ne!(
        record_name(&identity),
        crate::in_place::record_name(&identity)
    );
}
