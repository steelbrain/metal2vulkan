//! A device's spare command pools and descriptor pools (`spares.rs`): which command buffers a pool
//! hands out again, and what the bounds keep. The last tests run on a device: device tests compile
//! only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use crate::descriptor_arena::{Room, SetNeeds};
use crate::spares::{
    keep_command_pool, keep_descriptor_pool, keep_idle_command_pool, take_command_pool,
    take_descriptor_pool, take_idle, take_to_begin, CommandPool, DescriptorPool, DESCRIPTORS,
    DESCRIPTOR_POOLS, IDLE_FOR, RECENT_USES,
};
use ash::vk::{self, Handle};
use std::collections::VecDeque;
use std::time::Instant;

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

/// A command buffer takes the newest spare command pool of its queue's family that holds a
/// primary begun ahead, else the newest of that family.
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
    let mut pools = vec![pool(1), pool(2)];
    let taken = take_to_begin(&mut pools, vk::CommandPool::from_raw(2)).unwrap();
    assert_eq!(taken.raw, vk::CommandPool::from_raw(2));
    let other = take_command_pool(&mut pools, 0).unwrap();
    assert_eq!(
        other.raw,
        vk::CommandPool::from_raw(1),
        "a taker takes another"
    );
    assert!(
        take_command_pool(&mut pools, 0).is_none(),
        "and finds none being begun"
    );
    keep_command_pool(&mut pools, other);
    keep_command_pool(&mut pools, taken);
    let raws: Vec<_> = pools.iter().map(|pool| pool.raw.as_raw()).collect();
    assert_eq!(raws, [1, 2], "kept again, the newest");
    pools[1].begun = Some(vk::CommandBuffer::from_raw(20));
    assert!(take_to_begin(&mut pools, vk::CommandPool::from_raw(2)).is_none());
    pools[0].recording = true;
    assert!(take_to_begin(&mut pools, vk::CommandPool::from_raw(1)).is_none());
    assert!(take_to_begin(&mut pools, vk::CommandPool::from_raw(3)).is_none());
    assert_eq!(pools.len(), 2, "a pool that may not begin one stays");
}

/// A pool given back keeps, of each level, the command buffers the most one of its last
/// [`RECENT_USES`] uses handed out, a secondary at least, and gives up the rest to be freed: a
/// command buffer of many encoders leaves its pool holding them only until that many smaller ones
/// used it.
#[test]
fn a_pool_given_back_keeps_what_its_recent_uses_handed_out() {
    let primary = vk::CommandBufferLevel::PRIMARY;
    let secondary = vk::CommandBufferLevel::SECONDARY;
    let now = Instant::now();
    let mut pool = CommandPool::new(vk::CommandPool::from_raw(1), 0);
    for n in 0..100u64 {
        pool.made(primary, vk::CommandBuffer::from_raw(n + 1));
    }
    pool.made(secondary, vk::CommandBuffer::from_raw(1000));
    pool.made(secondary, vk::CommandBuffer::from_raw(1001));
    assert!(pool.given_back(now).is_empty(), "its own use's are kept");
    assert_eq!(pool.held(), 102);
    // Smaller uses: three primaries each, no secondary.
    let small = |pool: &mut CommandPool| {
        pool.taken();
        for _ in 0..3 {
            assert!(pool.reuse(primary).is_some());
        }
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

/// The oldest spare no command buffer took for [`IDLE_FOR`] since it was given back, and that
/// holds a command buffer, is taken out of the spares to shed them all, the primary begun ahead
/// first, and goes back as the oldest, so that a taker takes a spare that still holds its command
/// buffers first.
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

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod device {
    use super::RECENT_USES;
    use crate::command_buffer::kept_barriers::KEPT;
    use crate::tests::freed_memory::device;
    use crate::{Buffer, CommandBufferStatus, CommandQueue, Device, ResourceOptions};
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
        let command_buffer = queue.command_buffer().unwrap();
        for (at, target) in targets.iter().enumerate() {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(target, 0..256, at as u8).unwrap();
            blit.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        device.shared().frees.drain();
        command_buffer.status()
    }

    /// The command buffers the device's spare command pools hold once `done` holds of it, or
    /// after [`WAIT`]: a pool is given back as its command buffer's completion is processed.
    fn held_once(device: &Device, done: impl Fn(usize) -> bool) -> usize {
        let start = Instant::now();
        loop {
            let held = device.shared().spares.command_buffers_held();
            if done(held) || start.elapsed() > WAIT {
                return held;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A burst's command buffers go once the pool's recent uses no longer reach them, freed on
    /// the device's free thread, never on the thread that saw a command buffer complete.
    #[test]
    fn command_buffers_the_spares_hold_fall_back_after_a_burst_freed_on_the_free_thread() {
        let scratch = std::env::temp_dir().join(format!("m2v-spares-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        let device = device(&scratch);
        let queue = device.new_command_queue().unwrap();
        let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
        fills(&device, &queue, &target, 1);
        let before = held_once(&device, |held| held > 0);
        fills(&device, &queue, &target, BURST);
        let burst = held_once(&device, |held| held >= before + BURST / 2);
        // The burst's use leaves the pool's recent uses after as many more, and the give-back
        // then frees what it made.
        let more = RECENT_USES + 1;
        for _ in 0..more {
            fills(&device, &queue, &target, 1);
        }
        let after = held_once(&device, |held| held <= before + 2);
        let freed_on = device.shared().spares.freed_on();
        println!(
            "SPARES {}: command buffers the spares hold: {before} after a command buffer of one \
             blit encoder, {burst} after one of {BURST}, {after} after {more} more of one; freed \
             on {freed_on:?}",
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
        // The burst's use leaves the pool's recent uses after as many more, and the give-back
        // then frees what it made.
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
