//! Command queues sharing `VkQueue`s (`queue_share.rs`), the parts that need no device: the choice
//! a queue makes at an idle point, where a long queue settles, where an unshared queue goes, and
//! the record that keeps the device's check's verdict.
//!
//! - **The choice.** A queue joins the busy `VkQueue` whose oldest batch is young with the most
//!   batches in flight (then the latest submission), never one a long or unshared queue holds;
//!   with none it stays, unless a long or unshared queue holds its own, when it leaves for the
//!   most recently used idle one (off an unshared queue's, any other where none is idle).
//! - **Passed over.** A busy `VkQueue` no long or unshared queue holds, whose oldest batch is not
//!   young, is one a queue passes over for that batch alone (counted where it joins none).
//! - **Settling.** A long queue takes an idle `VkQueue` no long or unshared queue holds, one no
//!   command queue holds where one is (else the one the fewest hold), least recently used among
//!   equals, or stays.
//! - **Unshared.** An unshared queue tries the `VkQueue`s no queue holds, least recently used
//!   first, then those no long or unshared queue holds, fewest queues first.
//! - **Placement at creation.** A new queue takes the `VkQueue` the fewest live queues hold, of
//!   those no unshared queue holds, the first in turn from the device's count among equals. Over a
//!   churn shaped like a guest's walk (about 350 queues made and dropped over 14 `VkQueue`s, some
//!   kept the whole walk, an unshared one among them), no queue lands where another lives while a
//!   `VkQueue` is empty, and the unshared one never shares, where the old rule (the device's count
//!   modulo the `VkQueue`s) put live queues together while others stood empty.
//! - **The record.** Its verdict and cost read back for its identity alone, of a measured verdict
//!   alone.

use crate::in_place::DriverIdentity;
use crate::queue_share::{
    choose, decode, encode, fewest_held, old_busy, record_name, settle, unshared_order, Choice,
    Look, Verdict,
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
fn a_busy_vk_queue_is_passed_over_for_its_oldest_batch_alone_where_that_batch_is_not_young() {
    // Another busy one whose oldest batch has run 0.5 ms or longer, or for a time not known.
    assert!(old_busy(0, &busy(1, 2, 600_000, 5)));
    assert!(old_busy(
        0,
        &Look {
            head_age: None,
            ..busy(1, 2, 0, 5)
        }
    ));
    // A young one, an idle one, its own, or one a long or unshared queue holds: not for its
    // oldest batch.
    assert!(!old_busy(0, &busy(1, 2, 100_000, 5)));
    assert!(!old_busy(0, &idle(1, 5)));
    assert!(!old_busy(1, &busy(1, 2, 600_000, 5)));
    assert!(!old_busy(
        0,
        &Look {
            long: true,
            ..busy(1, 2, 600_000, 5)
        }
    ));
    assert!(!old_busy(
        0,
        &Look {
            unshared: true,
            ..busy(1, 2, 600_000, 5)
        }
    ));
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
    let empty = |looks: &[Look]| looks.iter().map(|&look| (look, 0)).collect::<Vec<_>>();
    assert_eq!(settle(0, &empty(&looks)), Some(2));
    assert_eq!(
        settle(0, &empty(&looks[..1])),
        None,
        "with no other, it stays"
    );
    let none = [looks[0], looks[3], looks[4], looks[5]];
    assert_eq!(settle(0, &empty(&none)), None);
}

/// A long queue settles on an idle `VkQueue` no command queue holds where one is, even one used
/// more recently than an idle one a short queue holds: a compositor that has just left the long
/// queue's `VkQueue` for an idle one, its last command buffer complete, is not followed there
/// (the RTX 5080 Laptop GPU's isolation run beside a queue waiting on the GPU). Where every idle
/// one is held, the one the fewest hold, the least recently used among equals.
#[test]
fn a_long_queue_settles_on_an_idle_vk_queue_no_command_queue_holds_before_a_held_one() {
    let long = Look {
        long: true,
        ..busy(1, 1, 0, 9)
    };
    // The compositor's new `VkQueue`, used long ago until its one command buffer there; and an
    // empty one used since.
    let looks = [(long, 1), (idle(2, 1), 1), (idle(3, 5), 0)];
    assert_eq!(settle(1, &looks), Some(3));
    let held = [(long, 1), (idle(2, 1), 2), (idle(3, 5), 1), (idle(4, 3), 1)];
    assert_eq!(settle(1, &held), Some(4));
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

#[test]
fn a_new_queue_takes_the_least_held_vk_queue_free_of_unshared_ones_in_turn() {
    // (live queues, barred): VkQueue 1 is empty but barred (an unshared queue holds it).
    let held = [(2, false), (0, true), (1, false), (1, false), (3, false)];
    assert_eq!(
        fewest_held(0, &held),
        2,
        "the fewest held, first in turn from 0"
    );
    assert_eq!(
        fewest_held(3, &held),
        3,
        "first in turn from 3 among equals"
    );
    assert_eq!(fewest_held(4, &held), 2, "turn wraps");
    // Every one barred: the fewest held among them.
    let barred = [(2, true), (1, true), (1, true)];
    assert_eq!(fewest_held(0, &barred), 1);
    assert_eq!(fewest_held(2, &barred), 2);
    assert_eq!(fewest_held(0, &[(5, false)]), 0, "one VkQueue");
}

/// A churn shaped like a guest's walk on NVIDIA's round robin of 14 `VkQueue`s: 4 queues kept the
/// whole walk (a compositor's unshared one among them, made first), then 350 made one at a time,
/// each living for 1 to 8 of the makes after it (a deterministic sequence), so at most 12 live at
/// once. Each placement by `place`, `(made, live per VkQueue, the unshared one's VkQueue)`; the
/// placements that land where another queue lives while a `VkQueue` is empty, and the most live
/// at once.
fn churn(place: &dyn Fn(u64, &[usize], Option<usize>) -> usize) -> (usize, usize) {
    const VK_QUEUES: usize = 14;
    let mut homes = [0usize; VK_QUEUES];
    let mut live: Vec<(usize, u64)> = Vec::new();
    let mut together = 0;
    let mut most = 0;
    // The unshared one, on an empty VkQueue: the first.
    let unshared = 0;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    // `made`: the queues made before this one, one a step.
    for made in 0..354u64 {
        live.retain(|&(vk_queue, until)| {
            let keep = until > made;
            if !keep {
                homes[vk_queue] -= 1;
            }
            keep
        });
        let vk_queue = place(made, &homes, Some(unshared));
        let empty = (0..VK_QUEUES).any(|index| homes[index] == 0 && index != unshared);
        if (homes[vk_queue] > 0 || vk_queue == unshared) && empty {
            together += 1;
        }
        homes[vk_queue] += 1;
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let lifetime = match made {
            0..3 => u64::MAX,
            _ => 1 + (seed >> 33) % 8,
        };
        live.push((vk_queue, made.saturating_add(lifetime)));
        most = most.max(live.len() + 1);
    }
    (together, most)
}

#[test]
fn over_a_walk_s_churn_live_queues_spread_and_the_unshared_one_never_shares() {
    let fewest = |made: u64, homes: &[usize], unshared: Option<usize>| {
        let held: Vec<(usize, bool)> = (homes.iter().enumerate())
            .map(|(index, &homes)| (homes, Some(index) == unshared))
            .collect();
        fewest_held((made % homes.len() as u64) as usize, &held)
    };
    // The rule before: the device's count modulo the round robin's `VkQueue`s, past the unshared
    // one's own (it took the count's first).
    let counted =
        |made: u64, homes: &[usize], _: Option<usize>| ((made + 1) % homes.len() as u64) as usize;
    let (together, most) = churn(&fewest);
    let (before, _) = churn(&counted);
    println!(
        "CHURN: 354 queues made, at most {most} live of 14 VkQueues: {together} placed beside a \
         live queue while one stood empty, against {before} by the count"
    );
    assert!(
        most < 14,
        "the churn keeps fewer live than VkQueues: {most}"
    );
    assert_eq!(together, 0);
    assert!(before > 0, "the count put live queues together");
    // An unshared queue's order, every VkQueue held but one: the empty one first.
    let looks: Vec<(Look, usize)> = (0..14)
        .map(|index| (idle(index, 0), usize::from(index != 9)))
        .collect();
    assert_eq!(unshared_order(&looks)[0], 9);
}
