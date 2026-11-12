//! A device's spare command pools and descriptor pools (`spares.rs`): which command buffers a pool
//! hands out again, and what the bounds keep. The last tests run on a device: device tests compile
//! only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use crate::descriptor_arena::{Room, SetNeeds};
use crate::spares::{
    keep_command_pool, keep_descriptor_pool, keep_idle_command_pool, past_held, take_command_pool,
    take_command_pool_holding, take_descriptor_pool, take_idle, CommandPool, CommandPools,
    DescriptorPool, DESCRIPTORS, DESCRIPTOR_POOLS, HELD, IDLE_FOR, RECENT_FOR, RECENT_USES,
};
use ash::vk::{self, Handle};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Needs of `count` storage buffers, with `flags`.
fn storage(count: u32, flags: vk::DescriptorPoolCreateFlags) -> SetNeeds {
    SetNeeds {
        counts: vec![(vk::DescriptorType::STORAGE_BUFFER.as_raw(), count)],
        flags,
    }
}

/// Descriptor pool `n` of `sets` sets of `needs`.
fn descriptor_pool(n: u64, needs: &SetNeeds, sets: u32) -> DescriptorPool {
    DescriptorPool {
        raw: vk::DescriptorPool::from_raw(n),
        room: Room::of(needs, sets),
    }
}

/// A spare command pool that a command buffer took hands out its command buffers again, in the
/// order they were made, each level apart, before any new one is made.
#[test]
fn a_taken_pool_hands_out_its_command_buffers_again() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let secondary = vk::CommandBufferLevel::SECONDARY;
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    assert_eq!(pool.reuse(primary), None);
    pool.made(primary, vk::CommandBuffer::from_raw(10));
    pool.made(secondary, vk::CommandBuffer::from_raw(20));
    pool.made(primary, vk::CommandBuffer::from_raw(11));
    assert_eq!(
        pool.reuse(primary),
        None,
        "each made one is handed out already"
    );
    let mut pools = Vec::new();
    keep_command_pool(&mut pools, pool);
    let mut pool = pools.pop().expect("kept");
    pool.taken();
    assert_eq!(pool.reuse(primary), Some(vk::CommandBuffer::from_raw(10)));
    assert_eq!(pool.reuse(secondary), Some(vk::CommandBuffer::from_raw(20)));
    assert_eq!(pool.reuse(primary), Some(vk::CommandBuffer::from_raw(11)));
    assert_eq!(pool.reuse(primary), None);
    assert_eq!(pool.reuse(secondary), None);
}

/// A spare command pool that made no secondary command buffer wants one; stocked with one, the
/// next command buffer that takes it hands that one out before making any; one that made a
/// secondary wants none.
#[test]
fn a_pool_without_a_secondary_is_stocked_with_one_for_its_next_taker() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let secondary = vk::CommandBufferLevel::SECONDARY;
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    pool.made(primary, vk::CommandBuffer::from_raw(10));
    assert!(pool.wants_secondary(), "only a primary made");
    pool.stock_secondary(vk::CommandBuffer::from_raw(20));
    assert!(!pool.wants_secondary(), "stocked");
    let mut pools = Vec::new();
    keep_command_pool(&mut pools, pool);
    let mut pool = pools.pop().expect("kept");
    pool.taken();
    assert_eq!(pool.reuse(secondary), Some(vk::CommandBuffer::from_raw(20)));
    assert_eq!(pool.reuse(secondary), None);
    let mut made = CommandPool::new(vk::CommandPool::from_raw(2), 0);
    made.made(secondary, vk::CommandBuffer::from_raw(30));
    assert!(!made.wants_secondary(), "one made already");
}

/// A spare command pool's primary begun ahead goes to the first encoder of the command buffer
/// that takes the pool, handed out among the pool's command buffers (so the pool settles only
/// once it ended too), never to `reuse`, which hands out the others; a pool its taker resets (one
/// given back recording) holds it as an ordinary primary again.
#[test]
fn a_primary_begun_ahead_goes_to_the_first_encoder_alone() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    pool.made(primary, vk::CommandBuffer::from_raw(10));
    pool.begun = Some(vk::CommandBuffer::from_raw(11));
    pool.taken();
    assert_eq!(pool.reuse(primary), Some(vk::CommandBuffer::from_raw(10)));
    assert_eq!(pool.reuse(primary), None, "the begun one is not reused");
    assert_eq!(pool.take_begun(), Some(vk::CommandBuffer::from_raw(11)));
    assert_eq!(pool.take_begun(), None, "taken once");
    assert_eq!(pool.handed_out(), 2, "the begun one counts as handed out");
    pool.taken();
    assert_eq!(pool.reuse(primary), Some(vk::CommandBuffer::from_raw(10)));
    assert_eq!(
        pool.reuse(primary),
        Some(vk::CommandBuffer::from_raw(11)),
        "taken, it is an ordinary primary of the pool"
    );
    let mut reset = CommandPool::new(vk::CommandPool::from_raw(2), 0);
    reset.begun = Some(vk::CommandBuffer::from_raw(20));
    reset.recording = true;
    reset.taken();
    assert_eq!(
        reset.take_begun(),
        None,
        "the reset left it in the initial state"
    );
    assert_eq!(reset.reuse(primary), Some(vk::CommandBuffer::from_raw(20)));
}

/// A command buffer takes a spare command pool of its queue's family that holds a primary begun
/// ahead, else one of that family, the newest of those holding the fewest command buffers.
#[test]
fn a_spare_command_pool_holding_a_begun_primary_is_taken_first() {
    let mut pools = Vec::new();
    for (raw, begun) in [(1, false), (2, true), (3, false)] {
        let mut pool = CommandPool::new(vk::CommandPool::from_raw(raw), 0);
        if begun {
            pool.begun = Some(vk::CommandBuffer::from_raw(raw * 10));
        }
        keep_command_pool(&mut pools, pool);
    }
    let raw = |pool: Option<CommandPool>| pool.map(|pool| pool.raw);
    for expected in [2, 3, 1] {
        assert_eq!(
            raw(take_command_pool(&mut pools, 0)),
            Some(vk::CommandPool::from_raw(expected))
        );
    }
}

/// A command buffer takes the spare holding the fewest command buffers of those holding a primary
/// begun ahead, then of the rest; one past its first pool takes the spare holding the most of the
/// level it lacks, one holding no primary begun ahead first, and none where none holds one.
#[test]
fn a_taker_takes_the_fewest_and_one_past_its_first_pool_the_most() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let secondary = vk::CommandBufferLevel::SECONDARY;
    let mut pools = vec![
        holding(1, 5, 1),
        holding(2, 40, 1),
        holding(3, 2, 1),
        holding(4, 40, 0),
        holding(5, 9, 1),
    ];
    pools[0].begun = Some(vk::CommandBuffer::from_raw(10));
    pools[1].begun = Some(vk::CommandBuffer::from_raw(20));
    let raws = |pools: &[CommandPool]| pools.iter().map(|p| p.raw.as_raw()).collect::<Vec<_>>();
    let taken = |pool: Option<CommandPool>| pool.map(|pool| pool.raw.as_raw());
    assert_eq!(
        taken(take_command_pool_holding(&mut pools, 0, primary)),
        Some(4)
    );
    assert_eq!(
        taken(take_command_pool_holding(&mut pools, 0, primary)),
        Some(2)
    );
    assert_eq!(
        taken(take_command_pool_holding(&mut pools, 0, secondary)),
        Some(5)
    );
    assert_eq!(
        taken(take_command_pool_holding(&mut pools, 1, primary)),
        None
    );
    assert_eq!(raws(&pools), [1, 3]);
    let mut pools = vec![
        holding(1, 5, 1),
        holding(2, 40, 1),
        holding(3, 2, 1),
        holding(4, 0, 0),
        holding(5, 9, 1),
    ];
    pools[0].begun = Some(vk::CommandBuffer::from_raw(10));
    pools[1].begun = Some(vk::CommandBuffer::from_raw(20));
    for expected in [1, 2, 4, 3, 5] {
        assert_eq!(taken(take_command_pool(&mut pools, 0)), Some(expected));
    }
    let mut empty = vec![holding(6, 0, 0)];
    assert_eq!(
        taken(take_command_pool_holding(&mut empty, 0, secondary)),
        None
    );
    assert_eq!(empty.len(), 1);
}

/// A command buffer takes the newest spare command pool made for its queue's family, never one
/// of another family.
#[test]
fn a_spare_command_pool_is_taken_only_for_its_family() {
    let mut pools = Vec::new();
    for (raw, family) in [(1, 0), (2, 1), (3, 0), (4, 2)] {
        keep_command_pool(
            &mut pools,
            CommandPool::new(vk::CommandPool::from_raw(raw), family),
        );
    }
    let raw = |pool: Option<CommandPool>| pool.map(|pool| pool.raw);
    assert_eq!(
        raw(take_command_pool(&mut pools, 0)),
        Some(vk::CommandPool::from_raw(3))
    );
    assert_eq!(
        raw(take_command_pool(&mut pools, 0)),
        Some(vk::CommandPool::from_raw(1))
    );
    assert_eq!(raw(take_command_pool(&mut pools, 0)), None);
    assert_eq!(raw(take_command_pool(&mut pools, 3)), None);
    assert_eq!(
        raw(take_command_pool(&mut pools, 2)),
        Some(vk::CommandPool::from_raw(4))
    );
    assert_eq!(pools.len(), 1);
}

/// Every command pool given back is kept, past the former bounds too (8 spares, and a pool that
/// made more than 64 command buffers, each released on the completing handler thread before),
/// and each is taken again, newest first: the runtime never releases one while the device lives.
#[test]
fn every_command_pool_given_back_is_kept() {
    let mut pools = Vec::new();
    let mut large = CommandPool::new(vk::CommandPool::from_raw(1000), 0);
    for n in 0..200u64 {
        large.made(
            vk::CommandBufferLevel::PRIMARY,
            vk::CommandBuffer::from_raw(n + 1),
        );
    }
    keep_command_pool(&mut pools, large);
    for n in 0..100u64 {
        keep_command_pool(
            &mut pools,
            CommandPool::new(vk::CommandPool::from_raw(n + 1), 0),
        );
    }
    assert_eq!(pools.len(), 101);
    for n in (0..100u64).rev() {
        let pool = take_command_pool(&mut pools, 0).expect("kept");
        assert_eq!(pool.raw, vk::CommandPool::from_raw(n + 1));
    }
    let mut large = take_command_pool(&mut pools, 0).expect("the large pool kept");
    assert_eq!(large.raw, vk::CommandPool::from_raw(1000));
    large.taken();
    assert_eq!(
        large.reuse(vk::CommandBufferLevel::PRIMARY),
        Some(vk::CommandBuffer::from_raw(1)),
        "its command buffers handed out again"
    );
    assert!(pools.is_empty());
}

/// A set takes the newest spare descriptor pool with room for it, of its flags (a fitted arena's,
/// one of at most the descriptors it would open); past the bounds
/// the oldest spares go first, and a pool that alone holds more descriptors than the bound is not
/// kept.
#[test]
fn spare_descriptor_pools_are_bounded_and_taken_by_room() {
    let plain = vk::DescriptorPoolCreateFlags::empty();
    let after_bind = vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
    let mut pools = VecDeque::new();
    assert!(
        keep_descriptor_pool(&mut pools, descriptor_pool(1, &storage(2, plain), 16)).is_empty()
    );
    assert!(keep_descriptor_pool(&mut pools, descriptor_pool(2, &storage(1, plain), 1)).is_empty());
    assert!(
        keep_descriptor_pool(&mut pools, descriptor_pool(3, &storage(2, after_bind), 16))
            .is_empty()
    );
    // Two storage buffers: pool 2 holds one in all (a pool's room is its totals), and pool 3 has
    // other flags.
    // A fitted arena's bound: pool 1 holds 32 descriptors.
    assert!(take_descriptor_pool(&mut pools, &storage(2, plain), Some(31)).is_none());
    let taken = take_descriptor_pool(&mut pools, &storage(2, plain), Some(32)).expect("fits");
    assert_eq!(taken.raw, vk::DescriptorPool::from_raw(1));
    assert!(keep_descriptor_pool(&mut pools, taken).is_empty());
    let taken = take_descriptor_pool(&mut pools, &storage(2, plain), None).expect("pool 1 fits");
    assert_eq!(taken.raw, vk::DescriptorPool::from_raw(1));
    assert!(take_descriptor_pool(&mut pools, &storage(4, plain), None).is_none());
    assert_eq!(pools.len(), 2);

    let mut pools = VecDeque::new();
    for n in 0..DESCRIPTOR_POOLS as u64 {
        assert!(
            keep_descriptor_pool(&mut pools, descriptor_pool(n + 1, &storage(1, plain), 16))
                .is_empty()
        );
    }
    let past = keep_descriptor_pool(&mut pools, descriptor_pool(99, &storage(1, plain), 16));
    assert_eq!(past, [vk::DescriptorPool::from_raw(1)]);
    assert_eq!(pools.len(), DESCRIPTOR_POOLS);

    // By descriptors: three pools of a third of the bound each, then a fourth.
    let third = (DESCRIPTORS / 3) as u32;
    let mut pools = VecDeque::new();
    for n in 0..3 {
        assert!(keep_descriptor_pool(
            &mut pools,
            descriptor_pool(n + 1, &storage(third, plain), 1)
        )
        .is_empty());
    }
    let past = keep_descriptor_pool(&mut pools, descriptor_pool(4, &storage(third, plain), 1));
    assert_eq!(past, [vk::DescriptorPool::from_raw(1)]);
    let whole = (DESCRIPTORS + 1) as u32;
    let past = keep_descriptor_pool(&mut pools, descriptor_pool(5, &storage(whole, plain), 1));
    assert_eq!(past, [vk::DescriptorPool::from_raw(5)]);
    assert_eq!(pools.len(), 3);
}

/// A spare command pool a completing thread begins a primary ahead in is out of the spares while
/// it begins, so a taker meanwhile takes another, and never waits for the begin; a pool that may
/// not begin one (recording, or holding one begun) stays; kept again, it is the newest.
#[test]
fn a_spare_command_pool_begins_ahead_out_of_the_spares() {
    let pool = |n: u64| CommandPool::new(vk::CommandPool::from_raw(n), 0);
    let mut pools = CommandPools {
        spares: vec![pool(1), pool(2)],
        out: 0,
    };
    let taken = pools.take_to_begin(vk::CommandPool::from_raw(2)).unwrap();
    assert_eq!(taken.raw, vk::CommandPool::from_raw(2));
    let other = take_command_pool(&mut pools.spares, 0).unwrap();
    assert_eq!(
        other.raw,
        vk::CommandPool::from_raw(1),
        "a taker takes another"
    );
    assert!(
        take_command_pool(&mut pools.spares, 0).is_none(),
        "and finds none being begun"
    );
    keep_command_pool(&mut pools.spares, other);
    keep_command_pool(&mut pools.spares, taken);
    let raws: Vec<_> = pools.spares.iter().map(|pool| pool.raw.as_raw()).collect();
    assert_eq!(raws, [1, 2], "kept again, the newest");
    pools.spares[1].begun = Some(vk::CommandBuffer::from_raw(20));
    assert!(pools.take_to_begin(vk::CommandPool::from_raw(2)).is_none());
    pools.spares[0].recording = true;
    assert!(pools.take_to_begin(vk::CommandPool::from_raw(1)).is_none());
    assert!(pools.take_to_begin(vk::CommandPool::from_raw(3)).is_none());
    assert_eq!(pools.spares.len(), 2, "a pool that may not begin one stays");
}

/// A spare command pool holding no primary, whose begin ahead makes one, goes out to begin only
/// where the spares, with what the pools out of them count, hold fewer than [`HELD`] command
/// buffers, and counts the one it makes while out, so that a give-back meanwhile keeps the bound
/// with it; one holding a primary begins it whatever the spares hold.
#[test]
fn a_begin_ahead_makes_its_primary_only_within_the_bound() {
    let empty = vk::CommandPool::from_raw(2);
    // One short of the bound: the pool holding only a secondary goes out to make its primary.
    let mut pools = CommandPools {
        spares: vec![holding(1, HELD as u64 - 3, 1), holding(2, 0, 1)],
        out: 0,
    };
    let taken = pools.take_to_begin(empty).expect("room for one");
    assert_eq!(
        (taken.held(), pools.out),
        (1, 2),
        "counted out with the primary it makes"
    );
    // A give-back of one command buffer meanwhile finds the bound reached with it.
    let mut given = holding(3, 0, 1);
    let (oldest, past) = past_held(&mut pools.spares, pools.out, &mut given);
    let raws: Vec<_> = oldest.iter().map(|pool| pool.raw.as_raw()).collect();
    assert_eq!(raws, [1], "the oldest spare sheds its command buffers");
    assert!(past.is_empty(), "the pool given back keeps its own");
    // At the bound: it stays, and a pool holding a primary begins it, making none.
    let mut pools = CommandPools {
        spares: vec![holding(1, HELD as u64 - 2, 1), holding(2, 0, 1)],
        out: 0,
    };
    assert!(
        pools.take_to_begin(empty).is_none(),
        "no room for its primary"
    );
    assert_eq!(pools.spares.len(), 2, "it stays among the spares");
    let first = pools.take_to_begin(vk::CommandPool::from_raw(1));
    let first = first.expect("a pool holding a primary begins it");
    assert_eq!(pools.out, first.held(), "counted with what it holds alone");
}

/// A pool given back keeps, of each level, the command buffers the most one of its last
/// [`RECENT_USES`] uses handed out, a secondary at least, and gives up the rest to be freed: a
/// command buffer of many encoders leaves its pool holding them, once two spans of
/// [`RECENT_FOR`] passed, only until that many smaller ones used it.
#[test]
fn a_pool_given_back_keeps_what_its_recent_uses_handed_out() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let secondary = vk::CommandBufferLevel::SECONDARY;
    let mut now = Instant::now();
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    for n in 0..100u64 {
        pool.made(primary, vk::CommandBuffer::from_raw(n + 1));
    }
    pool.made(secondary, vk::CommandBuffer::from_raw(1000));
    pool.made(secondary, vk::CommandBuffer::from_raw(1001));
    assert!(pool.given_back(now).is_empty(), "its own use's are kept");
    assert_eq!(pool.held(), 102);
    // Smaller uses, past the large one's spans: three primaries each, no secondary.
    now += 2 * RECENT_FOR;
    let mut small = |pool: &mut CommandPool| {
        pool.taken();
        for _ in 0..3 {
            assert!(pool.reuse(primary).is_some());
        }
        now += Duration::from_millis(1);
        pool.given_back(now)
    };
    for use_ in 1..RECENT_USES {
        assert!(
            small(&mut pool).is_empty(),
            "use {use_}: the large one is among the last"
        );
    }
    // The large use leaves the last ones: every command buffer past them goes.
    let past = small(&mut pool);
    let mut expected: Vec<_> = (4..=100u64).map(vk::CommandBuffer::from_raw).collect();
    expected.push(vk::CommandBuffer::from_raw(1001));
    assert_eq!(
        past, expected,
        "past the most of its last uses, a secondary kept"
    );
    assert_eq!(pool.held(), 4);
    pool.taken();
    assert_eq!(pool.reuse(primary), Some(vk::CommandBuffer::from_raw(1)));
    assert_eq!(
        pool.reuse(secondary),
        Some(vk::CommandBuffer::from_raw(1000))
    );
}

/// A pool given back keeps, of each level, the command buffers the most one of its uses in its
/// current and its last span of [`RECENT_FOR`] handed out, past its last [`RECENT_USES`] uses:
/// a large use's stay until a use two spans after the one it fell in.
#[test]
fn a_pool_given_back_keeps_what_its_uses_in_its_last_spans_handed_out() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let start = Instant::now();
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    for n in 0..4u64 {
        pool.made(primary, vk::CommandBuffer::from_raw(n + 1));
    }
    assert!(pool.given_back(start).is_empty());
    let small = |pool: &mut CommandPool, at: Duration| {
        pool.taken();
        assert!(pool.reuse(primary).is_some());
        pool.given_back(start + at)
    };
    // Many uses of one primary, in the large one's span and the next.
    for use_ in 1..4 * RECENT_USES as u32 {
        let at = RECENT_FOR * use_ / (2 * RECENT_USES as u32);
        assert!(small(&mut pool, at).is_empty(), "use {use_}, at {at:?}");
    }
    let last = 2 * RECENT_FOR - Duration::from_millis(1);
    assert!(small(&mut pool, last).is_empty(), "the span before still");
    let past: Vec<_> = (2..=4).map(vk::CommandBuffer::from_raw).collect();
    assert_eq!(small(&mut pool, 2 * RECENT_FOR), past, "two spans on");
    assert_eq!(pool.held(), 1);
}

/// A steady workload whose every `EVERY`th use takes a second primary, a use every `APART`,
/// makes each command buffer once: the second stays through the uses between (the module docs).
/// Kept by its last [`RECENT_USES`] uses alone, the pool freed it after as many and made it again
/// at the next such use.
#[test]
fn a_steady_workload_makes_its_command_buffers_once() {
    const EVERY: u32 = 15;
    const APART: Duration = Duration::from_millis(2);
    let primary = vk::CommandBufferLevel::PRIMARY;
    let start = Instant::now();
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    let (mut made, mut freed) = (0u64, 0);
    // Five spans of uses.
    let uses = 5 * RECENT_FOR.as_millis() as u32 / APART.as_millis() as u32;
    for use_ in 0..uses {
        pool.taken();
        for _ in 0..1 + u32::from(use_ % EVERY == EVERY - 1) {
            if pool.reuse(primary).is_none() {
                made += 1;
                pool.made(primary, vk::CommandBuffer::from_raw(made));
            }
        }
        freed += pool.given_back(start + APART * use_).len();
    }
    assert_eq!(
        (made, freed),
        (2, 0),
        "{uses} uses made {made} command buffers and freed {freed}"
    );
}

/// The oldest spare no command buffer took for [`IDLE_FOR`] since it was given back, and that
/// holds a command buffer, is taken out of the spares to shed them all, the primary begun ahead
/// first, and goes back as the oldest.
#[test]
fn a_spare_left_idle_sheds_its_command_buffers_and_goes_back_the_oldest() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let start = Instant::now();
    let pool = |n: u64, given: Instant| {
        let mut pool = CommandPool::new(vk::CommandPool::from_raw(n), 0);
        pool.made(primary, vk::CommandBuffer::from_raw(n * 10));
        pool.made(primary, vk::CommandBuffer::from_raw(n * 10 + 1));
        assert!(pool.given_back(given).is_empty());
        pool
    };
    let mut pools = vec![pool(1, start), pool(2, start + IDLE_FOR / 2)];
    pools[0].begun = Some(vk::CommandBuffer::from_raw(19));
    let mut empty = CommandPool::new(vk::CommandPool::from_raw(3), 0);
    assert!(empty.given_back(start).is_empty());
    keep_command_pool(&mut pools, empty);
    assert!(take_idle(&mut pools, start + IDLE_FOR / 2).is_none());
    let mut pool = take_idle(&mut pools, start + IDLE_FOR).expect("the pool given back first");
    assert_eq!(pool.raw, vk::CommandPool::from_raw(1));
    assert_eq!(
        pool.shed(),
        [
            vk::CommandBuffer::from_raw(19),
            vk::CommandBuffer::from_raw(10),
            vk::CommandBuffer::from_raw(11)
        ],
        "the begun primary first"
    );
    assert_eq!(pool.held(), 0);
    keep_idle_command_pool(&mut pools, pool);
    let raws: Vec<_> = pools.iter().map(|pool| pool.raw.as_raw()).collect();
    assert_eq!(raws, [1, 2, 3], "the oldest");
    let idle = take_idle(&mut pools, start + 2 * IDLE_FOR).expect("the second");
    assert_eq!(
        idle.raw,
        vk::CommandPool::from_raw(2),
        "a pool that holds none is never idle"
    );
}

/// Command pool `n` holding `primaries` and `secondaries` command buffers, given back.
fn holding(n: u64, primaries: u64, secondaries: u64) -> CommandPool {
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(n), 0);
    for (level, count, base) in [
        (vk::CommandBufferLevel::PRIMARY, primaries, 0),
        (vk::CommandBufferLevel::SECONDARY, secondaries, 1 << 20),
    ] {
        for at in 0..count {
            pool.made(
                level,
                vk::CommandBuffer::from_raw((n << 32) + base + at + 1),
            );
        }
    }
    assert!(pool.given_back(Instant::now()).is_empty());
    pool
}

/// Past [`HELD`] command buffers in the spares and a pool given back, the oldest spares that hold
/// any are taken out to shed theirs, until the rest hold at most that many; a pool given back that
/// alone holds more gives up its last made past it, secondaries first and one kept, then
/// primaries, then that secondary and its primary begun ahead, where the bound leaves no room
/// for them.
#[test]
fn the_spares_and_a_pool_given_back_hold_at_most_held_command_buffers() {
    let third = HELD as u64 / 3;
    let mut pools = vec![
        holding(1, third, third),
        CommandPool::new(vk::CommandPool::from_raw(2), 0),
        holding(3, third, 0),
        holding(4, third, 0),
    ];
    let mut given = holding(5, 0, third);
    let (oldest, past) = past_held(&mut pools, 0, &mut given);
    let raws = |pools: &[CommandPool]| {
        pools
            .iter()
            .map(|pool| pool.raw.as_raw())
            .collect::<Vec<_>>()
    };
    assert_eq!(raws(&oldest), [1], "the oldest that holds any");
    assert!(past.is_empty(), "the pool given back is kept whole");
    assert_eq!(
        raws(&pools),
        [2, 3, 4],
        "the rest stay, the empty one among them"
    );
    let held = |pools: &[CommandPool]| pools.iter().map(CommandPool::held).sum::<usize>();
    assert!(held(&pools) + given.held() <= HELD);
    // Within the bound, nothing is taken out.
    let (oldest, past) = past_held(&mut pools, 0, &mut given);
    assert!(oldest.is_empty() && past.is_empty());
    // A pool given back of more than the bound alone.
    let over = HELD as u64 + 10;
    let mut given = holding(6, 4, over);
    given.begun = Some(vk::CommandBuffer::from_raw(99));
    let (oldest, past) = past_held(&mut pools, 0, &mut given);
    assert_eq!(raws(&oldest), [3, 4], "every spare that holds any");
    assert_eq!(given.held(), HELD, "the pool given back keeps the bound");
    assert_eq!(past.len() as u64, 4 + over + 1 - HELD as u64);
    let last = (6 << 32) + (1 << 20) + over;
    assert_eq!(
        past.first(),
        Some(&vk::CommandBuffer::from_raw(last - past.len() as u64 + 1)),
        "its last made secondaries"
    );
    assert_eq!(given.begun, Some(vk::CommandBuffer::from_raw(99)));
    // Past the bound in primaries and the begun one, a secondary is kept.
    let mut given = holding(7, HELD as u64 + 5, 3);
    given.begun = Some(vk::CommandBuffer::from_raw(98));
    let (_, past) = past_held(&mut Vec::new(), 0, &mut given);
    assert_eq!(
        past.len(),
        5 + 3 + 1,
        "two secondaries, then seven primaries"
    );
    assert_eq!(given.held(), HELD);
    given.taken();
    assert!(given.reuse(vk::CommandBufferLevel::SECONDARY).is_some());
    // What the pools out of the spares hold counts: the pool given back keeps the rest.
    let mut given = holding(8, 200, 100);
    let (_, past) = past_held(&mut Vec::new(), HELD - 100, &mut given);
    assert_eq!((past.len(), given.held()), (200, 100));
    // Room for one: the primaries go, then the secondary, the begun primary kept.
    let mut given = holding(9, 2, 1);
    given.begun = Some(vk::CommandBuffer::from_raw(97));
    let (_, past) = past_held(&mut Vec::new(), HELD - 1, &mut given);
    let last = (9 << 32) + (1 << 20) + 1;
    assert_eq!(past.len(), 3, "the primaries, then the secondary");
    assert_eq!(past.last(), Some(&vk::CommandBuffer::from_raw(last)));
    assert_eq!(given.begun, Some(vk::CommandBuffer::from_raw(97)));
    // The pools out of the spares count the bound: the pool given back keeps none, the begun
    // primary neither.
    let (_, past) = past_held(&mut Vec::new(), HELD, &mut given);
    assert_eq!(past, [vk::CommandBuffer::from_raw(97)]);
    assert_eq!(given.held(), 0);
    let mut given = holding(10, 2, 1);
    let (_, past) = past_held(&mut Vec::new(), HELD, &mut given);
    assert_eq!(
        (past.len(), given.held()),
        (3, 0),
        "no secondary kept past the bound"
    );
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod device {
    use super::{HELD, RECENT_FOR, RECENT_USES};
    use crate::command_buffer::kept_barriers::KEPT;
    use crate::free_thread::FreeThread;
    use crate::spares::FREED_AT_ONCE;
    use crate::tests::freed_memory::device;
    use crate::{Buffer, CommandBufferStatus, CommandQueue, Device, ResourceOptions};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// The blit encoders of the burst's command buffer.
    const BURST: usize = 64;

    /// The blit encoders of the trims test's burst, each filling a buffer of its own, before
    /// those filling one buffer, each after a barrier, and after them.
    const LEAD: usize = 320;
    const TAIL: usize = 768;

    /// The encoders filling buffers of their own that each command buffer after the trims test's
    /// frees adds to the one before.
    const STEP: usize = 16;

    /// The command buffers alive at once of the held test, each in a pool of its own, and the
    /// blit encoders of each: two together past the spares' bound.
    const APART: usize = 3;
    const EACH: usize = HELD / 2 + 64;

    /// The mixed load's command buffers committed at once, the blit encoders of each command
    /// buffer of its cycle, its cycles, and those it runs before its counts are read.
    const MIXED_IN_FLIGHT: usize = 64;
    const MIXED: [usize; 10] = [200, 2, 2, 2, 2, 2, 2, 2, 2, 2];
    const MIXED_CYCLES: usize = 100;
    const MIXED_WARM: usize = 40;

    /// The longest a command buffer, or the spares' count, is waited for.
    const WAIT: Duration = Duration::from_secs(30);

    /// The device's free thread (`free_thread.rs`).
    const FREE_THREAD: &str = "metal2vulkan-frees";

    /// Commit a command buffer of `queue` of `device` of `encoders` blit encoders, each filling
    /// `target`, and wait for it, then for the frees its pool's give-back posted to the device's
    /// free thread, so that the next command buffer takes the pool again.
    fn fills(
        device: &Device,
        queue: &CommandQueue,
        target: &Buffer,
        encoders: usize,
    ) -> CommandBufferStatus {
        fills_of(device, queue, &vec![target; encoders])
    }

    /// [`fills`] of one blit encoder for each of `targets`, filling its first 256 bytes; the
    /// command buffer's status once it completed or failed.
    fn fills_of(device: &Device, queue: &CommandQueue, targets: &[&Buffer]) -> CommandBufferStatus {
        let status = committed(queue, targets);
        device.shared().frees.drain();
        status
    }

    /// A command buffer of `queue` of one blit encoder for each of `targets`, filling its first
    /// 256 bytes, committed and waited for, not the frees its pool's give-back posted; its status
    /// once it completed or failed.
    fn committed(queue: &CommandQueue, targets: &[&Buffer]) -> CommandBufferStatus {
        let command_buffer = queue.command_buffer().unwrap();
        for (at, target) in targets.iter().enumerate() {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(target, 0..256, at as u8).unwrap();
            blit.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        command_buffer.status()
    }

    /// The device's free thread held at each free until this drops, a failing case's unwinding
    /// too, so that the device's teardown, which waits for the frees, never waits for a hold left
    /// behind.
    struct Hold(Arc<FreeThread>);

    impl Hold {
        fn new(frees: &Arc<FreeThread>) -> Hold {
            frees.held().hold();
            Hold(Arc::clone(frees))
        }
    }

    impl Drop for Hold {
        fn drop(&mut self) {
            self.0.held().release();
        }
    }

    /// The command buffers of a load that grew to a burst go once the load fell back, the
    /// pool's recent uses and the spans of its uses no longer reaching them, freed on the
    /// device's free thread, never on the thread that saw a command buffer complete. Each count
    /// is read once [`fills`] returned, which is final: a command buffer's wait returns once its
    /// completion's transition ran, its pool's give-back and begin ahead among it
    /// (`command_buffer/gate.rs`), and the drain once the frees that give-back posted ran.
    #[test]
    fn command_buffers_the_spares_hold_fall_back_after_a_burst_freed_on_the_free_thread() {
        let scratch = std::env::temp_dir().join(format!("m2v-spares-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let device = device(&scratch);
        let queue = device.new_command_queue().unwrap();
        let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let held = || device.shared().spares.command_buffers_held();
        fills(&device, &queue, &target, 1);
        let before = held();
        let mut grown = Vec::new();
        for encoders in (1..=BURST.ilog2()).map(|power| 1 << power) {
            fills(&device, &queue, &target, encoders);
            grown.push(held());
        }
        let burst = *grown.last().unwrap();
        // The load idles past the spans of the burst's use, which then leaves the pool's recent
        // uses after as many more, and the give-back frees what it made.
        std::thread::sleep(2 * RECENT_FOR);
        let more = RECENT_USES + 1;
        for _ in 0..more {
            fills(&device, &queue, &target, 1);
        }
        let after = held();
        let freed_on = device.shared().spares.freed_on();
        println!(
            "SPARES {}: command buffers the spares hold: {before} after a command buffer of one \
             blit encoder, {grown:?} after ones of 2 to {BURST}, {after} after {more} more of \
             one; freed on {freed_on:?}",
            device.name()
        );
        assert!(
            !freed_on.is_empty() && freed_on.iter().all(|name| name == FREE_THREAD),
            "the spares' command buffers were freed on {freed_on:?}, not on {FREE_THREAD} alone"
        );
        assert!(
            burst >= before + BURST / 2,
            "the burst's command buffer left {burst} where one encoder's left {before}"
        );
        assert!(
            after <= before + 2,
            "{after} held after {more} command buffers of one encoder, {before} before the burst"
        );
        // The device's counts say what the spares freed and why (`WorkCounts`).
        let (churn, counts) = (device.shared().spares.churn(), device.work_counts());
        let freed = [
            counts.spare_command_buffers_freed_past_recent_uses,
            counts.spare_command_buffers_freed_past_held,
            counts.spare_command_buffers_shed,
        ];
        assert_eq!(freed, [churn.past_recent_uses, churn.past_held, churn.shed]);
        assert!(
            freed[0] > 0,
            "the burst's command buffers were freed past recent uses"
        );
        let trims = match device.shared().spares.trims() {
            true => freed[0].div_ceil(FREED_AT_ONCE as u64),
            false => 0,
        };
        assert!(
            counts.spare_command_pool_trims >= trims,
            "{} trims after {freed:?} freed",
            counts.spare_command_pool_trims
        );
        drop((target, queue, device));
        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// A steady workload whose every `EVERY`th command buffer has two blit encoders, the rest
    /// one, makes no command buffer once each kind ran: what one use takes stays through the
    /// uses between (`spares.rs`). Kept by its last [`RECENT_USES`] uses alone, a pool freed
    /// what the larger command buffer took after as many uses, and made it again at the next.
    #[test]
    fn a_steady_workload_makes_no_command_buffer_once_warm() {
        const EVERY: usize = 30;
        const ROUNDS: usize = 10;
        let scratch =
            std::env::temp_dir().join(format!("m2v-spares-steady-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let device = device(&scratch);
        let queue = device.new_command_queue().unwrap();
        let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let round = || {
            for at in 0..EVERY {
                let encoders = 1 + usize::from(at == EVERY - 1);
                let status = fills(&device, &queue, &target, encoders);
                assert_eq!(status, CommandBufferStatus::Completed);
            }
        };
        let allocated = || -> u64 { device.shared().spares.churn().allocated.values().sum() };
        let start = Instant::now();
        round();
        round();
        let (warm, held) = (allocated(), device.shared().spares.command_buffers_held());
        for _ in 0..ROUNDS {
            round();
        }
        let churn = device.shared().spares.churn();
        let made = allocated() - warm;
        println!(
            "SPARES-STEADY {}: {ROUNDS} rounds of {EVERY} command buffers, the last of two blit \
             encoders, in {:?}, made {made} command buffers after the first two made {warm}; the \
             spares hold {held} then {}; freed past a pool's recent uses {}, past the bound {}, \
             shed {}",
            device.name(),
            start.elapsed(),
            device.shared().spares.command_buffers_held(),
            churn.past_recent_uses,
            churn.past_held,
            churn.shed
        );
        assert_eq!(
            made, 0,
            "the steady workload made {made} command buffers once warm"
        );
        drop((target, queue, device));
        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// A load of command buffers of mixed sizes, [`MIXED_IN_FLIGHT`] committed at once, the same
    /// each cycle, frees no command buffer once warm, and makes fewer than its large command
    /// buffer's encoders (none, or a few where a large one found every pool holding the most out):
    /// its larger command buffers take the spares holding the most past their first pools, so the
    /// pools' command buffers follow what the load holds at once (`spares.rs`). Where each pool kept the most its own uses
    /// handed out, and a taker took the newest, every pool came to keep the larger command
    /// buffer's, past [`HELD`] together, and each give-back then shed the oldest spare, whose next
    /// taker made its command buffers again.
    #[test]
    fn command_buffers_of_mixed_sizes_in_flight_make_and_free_none_once_warm() {
        let scratch = std::env::temp_dir().join(format!("m2v-spares-mixed-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let device = device(&scratch);
        let queue = device.new_command_queue().unwrap();
        let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let mut flying = std::collections::VecDeque::new();
        let mut warm = None;
        for cycle in 0..MIXED_CYCLES {
            if cycle == MIXED_WARM {
                warm = Some((device.work_counts(), device.shared().spares.churn()));
            }
            for &encoders in &MIXED {
                let command_buffer = queue.command_buffer().unwrap();
                for at in 0..encoders {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(&target, 0..256, at as u8).unwrap();
                    blit.end_encoding().unwrap();
                }
                command_buffer.commit().unwrap();
                flying.push_back(command_buffer);
                if flying.len() == MIXED_IN_FLIGHT {
                    let oldest = flying.pop_front().unwrap();
                    oldest.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(oldest.status(), CommandBufferStatus::Completed);
                }
            }
        }
        for command_buffer in flying.drain(..) {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        device.shared().frees.drain();
        let (warm, churn_warm) = warm.unwrap();
        let (counts, churn) = (device.work_counts(), device.shared().spares.churn());
        let allocated =
            churn.allocated.values().sum::<u64>() - churn_warm.allocated.values().sum::<u64>();
        let made_at: Vec<_> = (churn.allocated.iter())
            .map(|(at, &n)| (at, n - churn_warm.allocated.get(at).copied().unwrap_or(0)))
            .filter(|&(_, n)| n > 0)
            .collect();
        let freed = [
            churn.past_recent_uses - churn_warm.past_recent_uses,
            churn.past_held - churn_warm.past_held,
            churn.shed - churn_warm.shed,
        ];
        let past_first =
            counts.spare_command_pools_taken_past_first - warm.spare_command_pools_taken_past_first;
        let trims = counts.spare_command_pool_trims - warm.spare_command_pool_trims;
        println!(
            "SPARES-MIXED {}: {MIXED_IN_FLIGHT} in flight of {MIXED:?}, {} cycles once warm: made \
             {allocated} command buffers ({made_at:?}), freed {freed:?} (past recent uses, past the bound, \
             shed), {trims} trims; {past_first} spares taken past a first pool; the spares hold \
             {} of {} begun",
            device.name(),
            MIXED_CYCLES - MIXED_WARM,
            device.shared().spares.command_buffers_held(),
            counts.command_buffers_begun - warm.command_buffers_begun,
        );
        assert_eq!(
            (freed, trims),
            ([0; 3], 0),
            "the load freed command buffers once warm, and made {allocated}"
        );
        assert!(
            allocated <= MIXED[0] as u64,
            "the load made {allocated} command buffers once warm"
        );
        assert!(
            past_first > 0,
            "no command buffer took a spare past its first pool"
        );
        drop((target, queue, device));
        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// Command buffers of many encoders alive at once, each in a pool of its own, leave the
    /// spares holding at most [`HELD`] command buffers once each was given back, where each pool
    /// kept its peak before: the oldest spares shed theirs. On ANV below Xe-HP each command
    /// buffer a spare kept held a block of the device's binding-table pool (`spares.rs`).
    #[test]
    fn command_buffers_alive_at_once_leave_the_spares_holding_at_most_held() {
        let scratch = std::env::temp_dir().join(format!("m2v-spares-held-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let device = device(&scratch);
        let queue = device.new_command_queue().unwrap();
        let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let begun = device.work_counts().command_buffers_begun;
        let command_buffers: Vec<_> = (0..APART)
            .map(|_| {
                let command_buffer = queue.command_buffer().unwrap();
                for at in 0..EACH {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(&target, 0..256, at as u8).unwrap();
                    blit.end_encoding().unwrap();
                }
                command_buffer
            })
            .collect();
        for command_buffer in &command_buffers {
            command_buffer.commit().unwrap();
        }
        for command_buffer in &command_buffers {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
        drop(command_buffers);
        device.shared().frees.drain();
        let begun = device.work_counts().command_buffers_begun - begun;
        let held = device.shared().spares.command_buffers_held();
        let freed_on = device.shared().spares.freed_on();
        println!(
            "SPARES-HELD {}: {APART} command buffers of {EACH} blit encoders alive at once began \
             {begun} command buffers; the spares hold {held}, frees ran {} times",
            device.name(),
            freed_on.len()
        );
        assert!(
            begun as usize > HELD,
            "the command buffers began {begun}, within the bound"
        );
        assert!(held <= HELD, "the spares hold {held}, past {HELD}");
        drop((target, queue, device));
        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// A pool given back while a pool trimmed to [`HELD`] is out of the spares, its frees waiting
    /// on the device's free thread, keeps none of its command buffers, its secondary neither, and
    /// a begin ahead in it then makes no primary: the spares hold [`HELD`] once both came back,
    /// and still after the begin. The free thread is held at each free (the crate's hook) until
    /// both pools were given back: the order a loaded suite met by chance in
    /// [`command_buffers_alive_at_once_leave_the_spares_holding_at_most_held`]. A pool holding a
    /// primary still begins it ahead at the bound, making none.
    #[test]
    fn a_pool_given_back_while_the_bound_is_out_freeing_keeps_none_past_it() {
        let scratch = std::env::temp_dir().join(format!("m2v-spares-out-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let device = device(&scratch);
        let shared = device.shared();
        let queue = device.new_command_queue().unwrap();
        let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        let hold = Hold::new(&shared.frees);
        let large = committed(&queue, &vec![&target; HELD + 64]);
        let trimmed = shared.spares.churn().past_held;
        let freed = shared.spares.freed_on();
        let small = committed(&queue, &[&target]);
        drop(hold);
        shared.frees.drain();
        let pools = shared.spares.command_pools_held();
        let held: usize = pools.iter().map(|(_, held)| held).sum();
        let (newest, last) = *pools.last().unwrap();
        let begun = device.work_counts().primaries_begun_ahead;
        shared.spares.begin_ahead(shared, newest);
        let after = shared.spares.command_pools_held();
        let (largest, _) = *pools.iter().max_by_key(|(_, held)| *held).unwrap();
        shared.spares.begin_ahead(shared, largest);
        let begun = device.work_counts().primaries_begun_ahead - begun;
        println!(
            "SPARES-OUT {}: the large command buffer's pool freed {trimmed} past the bound, \
             {} frees ran before the small one's give-back; the spares hold {pools:?} once both \
             came back, {after:?} after the begin ahead in the newest; {begun} begun ahead",
            device.name(),
            freed.len()
        );
        let completed = CommandBufferStatus::Completed;
        assert_eq!((large, small), (completed, completed));
        assert!(
            trimmed > 0,
            "the large command buffer's pool was not trimmed"
        );
        assert!(
            freed.is_empty(),
            "its frees ran before the small one's give-back"
        );
        assert_eq!(held, HELD, "the spares hold {held} once both came back");
        assert_eq!(last, 0, "the pool given back last holds {last}");
        let after: usize = after.iter().map(|(_, held)| held).sum();
        assert_eq!(after, HELD, "the spares hold {after} after its begin ahead");
        assert_eq!(begun, 1, "the pool holding primaries began one ahead");
        assert_eq!(shared.spares.command_buffers_held(), HELD);
        drop((target, queue, device));
        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// The kept barriers a burst's encoders came after (`command_buffer/kept_barriers.rs`) are
    /// submitted again, and complete, once the pool's recent uses no longer reach the burst and
    /// its command buffers were freed. Each of the burst's middle [`KEPT`] encoders fills one
    /// buffer after a barrier, so a kept command buffer comes before each, and the encoders
    /// before and after them fill buffers of their own, with no barrier, so the command buffers
    /// those barriers came before are neither the pool's first nor its last. Then each command
    /// buffer takes the kept barriers again and makes [`STEP`] command buffers more than the one
    /// before, so that the driver takes back the memory of those freed while it makes few of its
    /// own. On ANV, which names in each submission of a kept command buffer every command buffer
    /// that ever came next after it, such a submission failed and the device was lost where the
    /// pool was trimmed after the frees
    /// (`RuntimeDriverDefect::TrimmedCommandBufferFailsResubmission`): Mesa's slab allocator takes
    /// freed slices back only once an allocation finds its current slab full
    /// (`pb_slab_alloc_reclaimed`), releasing each slab they empty, and a submission fails only
    /// while no buffer object made since holds the handle released.
    #[test]
    fn kept_barriers_run_again_after_the_command_buffers_after_them_were_freed() {
        let scratch = std::env::temp_dir().join(format!("m2v-spares-trims-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let device = device(&scratch);
        let queue = device.new_command_queue().unwrap();
        let options = ResourceOptions::private();
        let shared = device.new_buffer(256, options).unwrap();
        let own: Vec<Buffer> = (0..LEAD + TAIL)
            .map(|_| device.new_buffer(256, options).unwrap())
            .collect();
        let (lead, tail) = own.split_at(LEAD);
        let burst: Vec<&Buffer> = lead
            .iter()
            .chain(std::iter::repeat_n(&shared, KEPT))
            .chain(tail)
            .collect();
        let mut statuses = vec![fills(&device, &queue, &shared, 1)];
        statuses.push(fills_of(&device, &queue, &burst));
        // Past the spans of the burst's use, it leaves the pool's recent uses after as many more,
        // and the give-back then frees what it made.
        std::thread::sleep(2 * RECENT_FOR);
        for _ in 0..=RECENT_USES {
            statuses.push(fills(&device, &queue, &shared, 2));
        }
        let freed = device.shared().spares.freed_on().len();
        for step in 1..=TAIL / STEP {
            let again: Vec<&Buffer> = std::iter::repeat_n(&shared, KEPT)
                .chain(&own[..step * STEP])
                .collect();
            statuses.push(fills_of(&device, &queue, &again));
        }
        let failed = statuses
            .iter()
            .filter(|status| **status != CommandBufferStatus::Completed)
            .count();
        let first = statuses
            .iter()
            .position(|status| *status != CommandBufferStatus::Completed);
        let recoveries = device.work_counts().device_recoveries;
        println!(
            "SPARES-TRIMS {}: trims {}; {freed} frees ran; {failed} of {} command buffers not \
             completed, the first at {first:?}; {recoveries} recoveries",
            device.name(),
            device.shared().spares.trims(),
            statuses.len()
        );
        assert!(freed > 0, "the burst's command buffers were never freed");
        assert_eq!(failed, 0, "command buffers did not complete: {statuses:?}");
        assert_eq!(recoveries, 0, "the device recovered from a loss");
        drop((own, shared, queue, device));
        let _ = std::fs::remove_dir_all(&scratch);
    }
}
