//! A device's queue limits (`DeviceOptions::queue_limits`), without a device: the plan within
//! limits (`queue_plan_within`) is the queue plan where the limits are the plan's own or past
//! them, and where a device's families have no queue to spare; the plan's round robin, presenting,
//! reserved and extra queues it is compared with where the limits change them
//! (`planned_queues`); and which placements the device notes, for each case it says so in
//! (`shares_past_limit` for a command queue, unshared or not, and a presenting queue; the
//! reserved and extra queues' places; the message's id name). `queue_limits_device.rs` holds
//! that a device sends exactly those messages, and none without limits.

use crate::device::create::{
    extra_queue_places, planned_queues, queue_limits_reached, queue_plan, queue_plan_within,
    shares_past_limit, PlannedQueues, QUEUE_LIMITS_REACHED,
};
use crate::QueueLimits;
use ash::vk;

/// Every flag a graphics family of NVIDIA's, MoltenVK's or RADV's has.
const ALL: vk::QueueFlags = vk::QueueFlags::from_raw(
    vk::QueueFlags::GRAPHICS.as_raw()
        | vk::QueueFlags::COMPUTE.as_raw()
        | vk::QueueFlags::TRANSFER.as_raw()
        | vk::QueueFlags::SPARSE_BINDING.as_raw(),
);

/// The limits the runtime lane makes its devices with: one `VkQueue` for the round robin, none
/// for presenting.
const LEAN: QueueLimits = QueueLimits {
    round_robin: 1,
    presenting: 0,
};

/// A family of `count` queues of `flags`.
fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
    vk::QueueFamilyProperties {
        queue_flags: flags,
        queue_count: count,
        timestamp_valid_bits: 64,
        min_image_transfer_granularity: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
    }
}

/// NVIDIA's shape: a graphics family of 16, a copy engine, a compute family, a decode family.
fn nvidia() -> Vec<vk::QueueFamilyProperties> {
    let copy = family(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING, 2);
    vec![
        family(ALL, 16),
        copy,
        family(ALL & !vk::QueueFlags::GRAPHICS, 8),
        family(copy.queue_flags | vk::QueueFlags::VIDEO_DECODE_KHR, 1),
    ]
}

/// The family shapes the plan's tests take: MoltenVK's four of one queue, RADV's and lavapipe's
/// one, families of six and of 24, NVIDIA's.
fn shapes() -> Vec<Vec<vk::QueueFamilyProperties>> {
    vec![
        vec![family(ALL, 1); 4],
        vec![family(ALL, 1)],
        vec![family(ALL, 6)],
        vec![family(ALL, 24)],
        nvidia(),
    ]
}

/// Limits at the plan's own (16 for the round robin, two for presenting) or past them are the
/// plan, for every shape and reservation, so a device made with them is the device made without:
/// the plan within them is the plan, and nothing is compared.
#[test]
fn limits_at_or_past_the_plan_s_are_the_plan() {
    for families in shapes() {
        for reserved in [0, 1, 2, 64] {
            let plan = queue_plan(&families, reserved);
            for limits in [
                QueueLimits {
                    round_robin: 16,
                    presenting: 2,
                },
                QueueLimits {
                    round_robin: 100,
                    presenting: 9,
                },
            ] {
                let within = queue_plan_within(&families, reserved, Some(limits));
                assert_eq!(within, plan, "{limits:?}, {reserved} reserved");
                let within = within.unwrap();
                assert_eq!(planned_queues(&families, reserved, &[], &within, &[]), None);
            }
            assert_eq!(queue_plan_within(&families, reserved, None), plan);
        }
    }
}

/// A device whose families have no queue to spare (RADV's and lavapipe's one graphics queue)
/// takes its plan under the lane's limits: nothing changes, so nothing is noted.
#[test]
fn one_queue_takes_its_plan_under_limits() {
    let families = [family(ALL, 1)];
    let mut within = queue_plan_within(&families, 1, Some(LEAN)).unwrap();
    assert_eq!(Some(&within), queue_plan(&families, 1).as_ref());
    let extra = [vk::QueueFlags::GRAPHICS];
    let places = extra_queue_places(&families, &mut within, &extra);
    assert_eq!(planned_queues(&families, 1, &extra, &within, &places), None);
}

/// Limits of zero for the round robin are one: a device always has a round robin.
#[test]
fn a_round_robin_of_none_is_one() {
    let none = QueueLimits {
        round_robin: 0,
        presenting: 0,
    };
    assert_eq!(
        queue_plan_within(&nvidia(), 0, Some(none)),
        queue_plan_within(&nvidia(), 0, Some(LEAN))
    );
}

/// The lane's limits on NVIDIA's shape: one round-robin `VkQueue` and none for presenting, the
/// staging queue as in the plan; compared with the plan's 14 and 2, so a second command queue,
/// unshared or not, shares where the plan gives it its own (up to the plan's 14th), and a
/// presenting queue shares where the plan reserves one (the first two).
#[test]
fn a_command_or_presenting_queue_past_the_limit_is_noted_where_the_plan_gives_its_own() {
    let families = nvidia();
    let within = queue_plan_within(&families, 0, Some(LEAN)).unwrap();
    let plan = queue_plan(&families, 0).unwrap();
    assert_eq!(within.queues, [(0, 0)]);
    assert!(within.present.is_empty());
    assert_eq!(within.staging, plan.staging);
    let planned = planned_queues(&families, 0, &[], &within, &[]).unwrap();
    assert_eq!(
        planned,
        PlannedQueues {
            round_robin: 14,
            presenting: 2,
            reserved: Vec::new(),
            extra: Vec::new(),
        }
    );
    // A command queue (unshared or not): the first takes its own, the second to the 14th share
    // where the plan gives each its own, the 15th shares under both.
    let round_robin: Vec<bool> = (0..16)
        .map(|made| shares_past_limit(made, within.queues.len(), planned.round_robin))
        .collect();
    assert_eq!(round_robin.iter().filter(|&&noted| noted).count(), 13);
    assert!(!round_robin[0] && round_robin[1] && round_robin[13] && !round_robin[14]);
    // A presenting queue: none reserved under the limits, two under the plan.
    let presenting: Vec<bool> = (0..3)
        .map(|asked| shares_past_limit(asked, within.present.len(), planned.presenting))
        .collect();
    assert_eq!(presenting, [true, true, false]);
    // One reserved for presenting: the second shares where the plan reserves it one.
    let one = QueueLimits {
        round_robin: 4,
        presenting: 1,
    };
    let within = queue_plan_within(&families, 0, Some(one)).unwrap();
    assert_eq!((within.queues.len(), within.present.len()), (4, 1));
    let presenting: Vec<bool> = (0..3)
        .map(|asked| shares_past_limit(asked, within.present.len(), 2))
        .collect();
    assert_eq!(presenting, [false, true, false]);
}

/// The reserved and extra queues on NVIDIA's shape under the lane's limits: the reserved queue
/// is the round robin's next where the plan reserves the family's last, and an extra graphics
/// queue takes a free one where the plan gives it presenting's last; both compared, so the device
/// notes each as it is made.
#[test]
fn reserved_and_extra_queues_placed_otherwise_are_noted() {
    let families = nvidia();
    let extra = [vk::QueueFlags::GRAPHICS];
    let mut within = queue_plan_within(&families, 1, Some(LEAN)).unwrap();
    let places = extra_queue_places(&families, &mut within, &extra);
    assert_eq!(within.reserved, [(0, 1)]);
    assert_eq!(places, [Some((0, 2))]);
    let planned = planned_queues(&families, 1, &extra, &within, &places).unwrap();
    assert_eq!(planned.reserved, [(0, 15)]);
    assert_eq!(planned.extra, [Some((0, 14))]);
    assert_eq!((planned.round_robin, planned.presenting), (13, 1));
    // The message names its id, as the lane's sink reads it.
    let text = queue_limits_reached("a device", "the reserved queues moved");
    assert!(
        text.ends_with(&format!("[{QUEUE_LIMITS_REACHED}]")),
        "{text}"
    );
    assert_eq!(QUEUE_LIMITS_REACHED, "metal2vulkan-queue-limits-reached");
}
