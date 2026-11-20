//! Texture blocks: a block's free ranges give the first aligned room and join again as places are
//! freed; and, under a device cfg, on every device of the cfg's kind, textures are placed in
//! blocks of the pool with no allocation of their own: a block's worth of 1 MiB textures takes
//! one block and then a second, releasing them gives one back to the pool and keeps the other,
//! textures made and released in that block allocate nothing, and every gauge reads zero after
//! teardown; the pool's allocations are 16 MiB of either class, a take is a range of one rounded
//! up to 64 KiB, a longer take an allocation of its own, and a texture of 32 MiB takes a texture
//! block of its own length; and the gauges count the texture blocks and the consumer's Private
//! allocations apart, each held whole, what is placed in them, and the sparse ones.

use crate::places::Ranges;

/// The free ranges are sorted, apart from one another and inside the block.
fn check(ranges: &Ranges, len: u64) {
    let free = ranges.free();
    for (index, &(start, end)) in free.iter().enumerate() {
        assert!(start < end && end <= len, "range {start}..{end}");
        if let Some(&(next, _)) = free.get(index + 1) {
            assert!(end < next, "{start}..{end} meets the range at {next}");
        }
    }
}

#[test]
fn ranges_take_the_first_aligned_room_and_join_as_places_are_freed() {
    let mut ranges = Ranges::new(1024);
    assert_eq!(ranges.take(100, 1), Some(0));
    assert_eq!(ranges.take(10, 64), Some(128), "after the first, aligned");
    assert_eq!(ranges.free(), &[(100, 128), (138, 1024)]);
    assert_eq!(ranges.take(28, 4), Some(100), "the room before the second");
    assert_eq!(ranges.free(), &[(138, 1024)]);
    assert_eq!(ranges.take(2000, 1), None, "longer than the block");
    ranges.give(100, 28);
    assert_eq!(ranges.free(), &[(100, 128), (138, 1024)]);
    ranges.give(128, 10);
    assert_eq!(ranges.free(), &[(100, 1024)], "joined on both sides");
    ranges.give(0, 100);
    assert_eq!(ranges.free(), &[(0, 1024)]);
    assert_eq!(ranges.take(1024, 1024), Some(0), "the whole block");
    assert_eq!(ranges.free(), &[]);
    assert_eq!(ranges.take(1, 1), None, "a full block");
    ranges.give(0, 1024);
    assert_eq!(ranges.free(), &[(0, 1024)]);
}

/// A long run of places of many lengths and alignments, freed out of order, against the
/// invariants after every step: no two places overlap, and once all are freed the block is one
/// free range again.
#[test]
fn ranges_under_load_never_overlap_and_end_whole() {
    let len = 1 << 20;
    let mut ranges = Ranges::new(len);
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut taken: Vec<(u64, u64)> = Vec::new();
    let mut placed = 0;
    for _ in 0..20_000 {
        if next() % 3 != 0 || taken.is_empty() {
            let want = 1 + next() % (len / 16);
            let align = 1 << (next() % 13);
            if let Some(at) = ranges.take(want, align) {
                assert_eq!(at % align, 0, "{want} bytes at {at}, aligned to {align}");
                for &(other, other_len) in &taken {
                    assert!(
                        at + want <= other || other + other_len <= at,
                        "{at}+{want} overlaps {other}+{other_len}"
                    );
                }
                taken.push((at, want));
                placed += 1;
            }
        } else {
            let index = (next() % taken.len() as u64) as usize;
            let (at, want) = taken.swap_remove(index);
            ranges.give(at, want);
        }
        check(&ranges, len);
    }
    assert!(placed > 1_000, "the run placed {placed}");
    for (at, want) in taken {
        ranges.give(at, want);
    }
    assert_eq!(ranges.free(), &[(0, len)]);
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::texture::TextureObject;
    use crate::{
        Device, DeviceOptions, Gauges, IndexClass, MemoryClass, PixelFormat, TextureDescriptor,
        TextureType,
    };
    use std::thread;
    use std::time::{Duration, Instant};

    fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(120);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn every_device_places_textures_in_blocks() {
        const MIB: u64 = 1 << 20;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            core.pool.await_internals();
            wait_until("a wholly free unit in each class", || {
                core.pool.stocked(MemoryClass::Private) && core.pool.stocked(MemoryClass::Shared)
            });
            let base = counters.gauges().blocks;
            let made = counters.metrics().blocks_made;
            let most = 2 * MemoryClass::Private.block_len() / MIB;
            let grant = device
                .grant_indices(IndexClass::Texture, 4096, most as u32)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 512, 512);

            // Textures of 1 MiB until a second block is taken.
            let mut textures = Vec::new();
            while core.places.blocks() < 2 {
                assert!(
                    (textures.len() as u64) < most,
                    "{name}: {} textures took {} blocks",
                    textures.len(),
                    core.places.blocks()
                );
                let index = 4096 + textures.len() as u32;
                textures.push(
                    (device.new_texture(&d, Some(index)))
                        .unwrap_or_else(|error| panic!("{name}: texture {index}: {error}")),
                );
            }
            let in_first = textures.len() as u64 - 1;
            assert!(
                in_first > MemoryClass::Private.block_len() / MIB / 2,
                "{name}: the first block held {in_first} textures of 1 MiB"
            );
            // Each block an allocation whole, after which the block-maker thread made the next.
            wait_until("the pool's next two allocations", || {
                counters.metrics().blocks_made == made + 2
            });
            assert_eq!(counters.gauges().blocks, base + 2, "{name}: two blocks");

            // Released: one block back to the pool, a free range of its allocation at once; one
            // kept for textures.
            drop(textures);
            wait_until("the textures freed", || {
                counters.gauges().textures == 0 && core.places.blocks() == 1
            });
            // One idle allocation more than the class's reserve, freed at its idle age.
            wait_until("the released block's allocation freed", || {
                counters.gauges().blocks == base + 1
            });

            // Textures made and released in the kept block allocate nothing.
            let made = counters.metrics().blocks_made;
            let small =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 32, 32);
            for round in 0..1_000 {
                let texture = (device.new_texture(&small, Some(4096)))
                    .unwrap_or_else(|error| panic!("{name}: round {round}: {error}"));
                drop(texture);
            }
            wait_until("the small textures freed", || {
                counters.gauges().textures == 0
            });
            assert_eq!(
                (counters.metrics().blocks_made, core.places.blocks()),
                (made, 1),
                "{name}: made and released in the kept block"
            );
            assert_eq!(
                counters.gauges().blocks,
                base + 1,
                "{name}: after the rounds"
            );

            device
                .release_indices(grant)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
            println!(
                "places: {name}: {in_first} textures of 1 MiB in the first block; metrics {:?}",
                counters.metrics()
            );
        }
    }

    /// The pool's allocations are 16 MiB of either class, on every device: a take of a byte is a
    /// range of one rounded up to 64 KiB (to `bufferImageGranularity` where that is larger) and
    /// allocates nothing on the taker's thread, where the block-maker thread then makes the
    /// class's next unit, the first no longer wholly free; a take longer than an allocation makes
    /// one of its own length, rounded up the same way; and a texture of 32 MiB is placed in a
    /// texture block of its own length.
    #[test]
    fn every_device_takes_blocks_from_16_mib_allocations() {
        const MIB: u64 = 1 << 20;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            core.pool.await_internals();
            for class in [MemoryClass::Shared, MemoryClass::Private] {
                assert_eq!(class.block_len(), 16 * MIB, "{name}: {class:?}");
                let made = counters.metrics().blocks_made;
                let small = (device.take_block(class, 1))
                    .unwrap_or_else(|error| panic!("{name}: a small {class:?} block: {error}"));
                assert!(
                    small.len.is_multiple_of(64 << 10) && small.len < MIB,
                    "{name}: a small {class:?} block of {} bytes",
                    small.len
                );
                wait_until("the class's next unit", || {
                    counters.metrics().blocks_made == made + 1 && core.pool.stocked(class)
                });
                let long = (device.take_block(class, 16 * MIB + 1))
                    .unwrap_or_else(|error| panic!("{name}: a long {class:?} block: {error}"));
                assert!(
                    long.len > 16 * MIB && long.len.is_multiple_of(64 << 10),
                    "{name}: a long {class:?} block of {} bytes",
                    long.len
                );
                assert_eq!(
                    counters.metrics().blocks_made,
                    made + 2,
                    "{name}: the long {class:?} block's allocation alone"
                );
                for block in [small, long] {
                    (device.release_block(block))
                        .unwrap_or_else(|error| panic!("{name}: a release: {error}"));
                }
            }
            let grant = device
                .grant_indices(IndexClass::Texture, 4096, 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4096, 2048);
            let texture = (device.new_texture(&d, Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: a texture of 32 MiB: {error}"));
            assert_eq!(
                core.places.blocks(),
                1,
                "{name}: a texture of 32 MiB in a texture block"
            );
            drop(texture);
            wait_until("the texture freed", || counters.gauges().textures == 0);
            device
                .release_indices(grant)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        }
    }

    /// The consumer's Private allocations' gauges: each held whole, its blocks' bytes, and the
    /// sparse ones, holding less than a sixteenth of their length.
    fn takes(gauges: &Gauges) -> [u64; 4] {
        [
            gauges.private_take_held_bytes,
            gauges.private_take_bytes,
            gauges.private_take_sparse_bytes,
            gauges.private_take_sparse_allocations,
        ]
    }

    /// The texture blocks' gauges, as [`takes`] for the consumer's allocations.
    fn texture_blocks(gauges: &Gauges) -> [u64; 4] {
        [
            gauges.texture_block_held_bytes,
            gauges.texture_block_bytes,
            gauges.texture_block_sparse_bytes,
            gauges.texture_block_sparse_blocks,
        ]
    }

    /// The Private units that hold a block, of either kind, and the bytes in use inside them.
    fn units(gauges: &Gauges) -> [u64; 2] {
        [gauges.private_units, gauges.private_unit_used_bytes]
    }

    /// The gauges count the texture blocks and the consumer's Private allocations apart, on every
    /// device: a small block alone in its unit makes the unit sparse until a sixteenth of it is
    /// placed; a small texture alone in its texture block likewise; neither kind is counted in
    /// the other's gauges; and once all is released only the texture block kept for the next
    /// place is held. The units gauges count both kinds' units together, with the blocks' and
    /// the textures' bytes in them, and no long allocation.
    #[test]
    fn every_device_gauges_texture_blocks_and_private_takes_apart() {
        let unit = MemoryClass::Private.block_len();
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            core.pool.await_internals();
            let gauges = counters.gauges();
            assert_eq!(
                (takes(&gauges), texture_blocks(&gauges), units(&gauges)),
                ([0; 4], [0; 4], [0; 2]),
                "{name}: at the start"
            );

            // A small block alone in its unit, then a sixteenth of the unit beside it.
            let small = (device.take_block(MemoryClass::Private, 1))
                .unwrap_or_else(|error| panic!("{name}: a small block: {error}"));
            let gauges = counters.gauges();
            assert_eq!(
                (takes(&gauges), units(&gauges)),
                ([unit, small.len, unit, 1], [1, small.len]),
                "{name}: a small block alone"
            );
            let more = (device.take_block(MemoryClass::Private, unit / 16))
                .unwrap_or_else(|error| panic!("{name}: a sixteenth: {error}"));
            assert_eq!(more.allocation, small.allocation, "{name}: one unit");
            let blocks = small.len + more.len;
            let gauges = counters.gauges();
            assert_eq!(
                (takes(&gauges), units(&gauges)),
                ([unit, blocks, 0, 0], [1, blocks]),
                "{name}: a sixteenth placed"
            );

            // A small texture alone in its texture block, then a sixteenth of the block beside it.
            let grant = device
                .grant_indices(IndexClass::Texture, 4096, 2)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let tiny = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 32, 32);
            let first = (device.new_texture(&tiny, Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: a small texture: {error}"));
            let size = first.image_size();
            let gauges = counters.gauges();
            assert_eq!(
                (texture_blocks(&gauges), units(&gauges)),
                ([unit, size, unit, 1], [2, blocks + size]),
                "{name}: a small texture alone"
            );
            let d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 512, 512);
            let second = (device.new_texture(&d, Some(4097)))
                .unwrap_or_else(|error| panic!("{name}: a texture of 1 MiB: {error}"));
            let placed = size + second.image_size();
            let gauges = counters.gauges();
            assert_eq!(
                (texture_blocks(&gauges), takes(&gauges), units(&gauges)),
                (
                    [unit, placed, 0, 0],
                    [unit, blocks, 0, 0],
                    [2, blocks + placed]
                ),
                "{name}: a sixteenth placed in the texture block, the takes unchanged"
            );
            assert_eq!(
                gauges.private_placed_bytes,
                gauges.private_take_bytes + gauges.texture_block_held_bytes,
                "{name}: the Private blocks out are the takes and the texture blocks"
            );

            // A take longer than a unit: an allocation of its own length, no unit.
            let long = (device.take_block(MemoryClass::Private, unit + 1))
                .unwrap_or_else(|error| panic!("{name}: a long block: {error}"));
            assert_eq!(
                units(&counters.gauges()),
                [2, blocks + placed],
                "{name}: a long allocation is no unit"
            );
            (device.release_block(long))
                .unwrap_or_else(|error| panic!("{name}: a release: {error}"));

            drop((first, second));
            wait_until("the textures freed", || counters.gauges().textures == 0);
            let gauges = counters.gauges();
            assert_eq!(
                (texture_blocks(&gauges), units(&gauges)),
                ([unit, 0, 0, 0], [2, blocks]),
                "{name}: the texture block kept for the next place"
            );
            for block in [small, more] {
                (device.release_block(block))
                    .unwrap_or_else(|error| panic!("{name}: a release: {error}"));
            }
            wait_until("the takes given back", || {
                takes(&counters.gauges()) == [0; 4]
            });
            assert_eq!(
                units(&counters.gauges()),
                [1, 0],
                "{name}: the kept texture block alone holds its unit"
            );
            device
                .release_indices(grant)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
        }
    }

    /// A texture longer than a block holds its texture block alone, on every device: a small
    /// texture made while the long block has free bytes past the long texture goes in a block of
    /// the block length, and the long block's allocation is freed at its idle age after the long
    /// texture's release, while the small texture lives.
    #[test]
    fn every_device_frees_a_long_texture_block_with_its_texture() {
        let unit = MemoryClass::Private.block_len();
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            core.pool.await_internals();
            let grant = device
                .grant_indices(IndexClass::Texture, 4096, 2)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // A texture longer than a block whose block, of its length rounded up to the pool's
            // granule, has room for a small texture past it: the first of a few shapes that
            // leaves 16 KiB. A device whose images leave none never meets the case.
            let mut found = None;
            for (width, height) in [(4000, 1100), (4000, 1101), (3000, 1500), (4100, 1025)] {
                let d = TextureDescriptor::new(
                    TextureType::Type2D,
                    PixelFormat::RGBA8Unorm,
                    width,
                    height,
                );
                let long = (device.new_texture(&d, Some(4096)))
                    .unwrap_or_else(|error| panic!("{name}: a long texture: {error}"));
                let held = counters.gauges().texture_block_held_bytes;
                assert!(held > unit, "{name}: a long texture block of {held} bytes");
                if held - long.image_size() >= 16 << 10 {
                    found = Some((long, held));
                    break;
                }
                drop(long);
                wait_until("the shape's texture freed", || {
                    counters.gauges().textures == 0
                });
            }
            let Some((long, held)) = found else {
                println!("places: {name}: no long texture leaves room past it in its block");
                continue;
            };
            // The shapes' long blocks freed at their idle age.
            wait_until("the shapes' blocks freed", || {
                let gauges = counters.gauges();
                gauges.private_idle_bytes == gauges.private_reserve_bytes
            });
            let free = held - long.image_size();
            let tiny = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 32, 32);
            let small = (device.new_texture(&tiny, Some(4097)))
                .unwrap_or_else(|error| panic!("{name}: a small texture: {error}"));
            let size = small.image_size();
            println!(
                "places: {name}: a long texture of {} bytes in a block of {held}, {free} free \
                 past it; a small texture of {size} bytes",
                long.image_size()
            );
            let (long_block, _) = long.place().expect("the long texture's place");
            let (small_block, _) = small.place().expect("the small texture's place");
            assert_ne!(
                small_block, long_block,
                "{name}: the small texture of {size} bytes in the long block's {free} free bytes"
            );
            assert_eq!(
                texture_blocks(&counters.gauges()),
                [held + unit, long.image_size() + size, unit, 1],
                "{name}: the long block, and the small texture's block of the block length"
            );
            // The small texture took the reserve: its next unit, so the allocations settle.
            wait_until("the next unit", || core.pool.stocked(MemoryClass::Private));
            let allocated = counters.gauges().private_allocation_bytes;
            let frees = core.pool.frees().len();
            drop(long);
            wait_until("the long block's allocation freed", || {
                counters.gauges().private_allocation_bytes == allocated - held
            });
            let freed = core.pool.frees();
            assert_eq!(freed.len(), frees + 1, "{name}: one free");
            let (since, began, _) = freed[frees];
            let age = began - since;
            assert!(
                (crate::memory::IDLE_AGE_NANOS..crate::memory::IDLE_AGE_NANOS + 1_000_000_000)
                    .contains(&age),
                "{name}: the long block freed {age} ns after it came back"
            );
            assert_eq!(
                (
                    counters.gauges().textures,
                    texture_blocks(&counters.gauges())
                ),
                (1, [unit, size, unit, 1]),
                "{name}: the small texture still live in its block"
            );
            drop(small);
            wait_until("the small texture freed", || {
                counters.gauges().textures == 0
            });
            device
                .release_indices(grant)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
        }
    }

    /// An idle device returns to its fixed Private memory within a second of the release of all
    /// it held, with no call, on every device: textures of many lengths, some longer than a
    /// block, and Private blocks of many lengths, some longer than a unit, all released, leave the
    /// class's reserve and the texture block kept for the next place, 16 MiB each, and no other
    /// Private allocation.
    #[test]
    fn every_device_returns_to_its_fixed_private_memory_within_a_second_of_going_idle() {
        let unit = MemoryClass::Private.block_len();
        let count = 160;
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            core.pool.await_internals();
            let grant = device
                .grant_indices(IndexClass::Texture, 4096, count)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut state = 0x2545_F491_4F6C_DD1D_u64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut textures = Vec::new();
            for index in 0..count {
                // Sides of 64 to 2,048 texels, and every fortieth longer than a block.
                let (width, height) = match index % 40 {
                    0 => (4096, 1025 + (next() % 1024) as u32),
                    _ => (64 << (next() % 6), 64 << (next() % 6)),
                };
                let d = TextureDescriptor::new(
                    TextureType::Type2D,
                    PixelFormat::RGBA8Unorm,
                    width,
                    height,
                );
                textures.push(
                    (device.new_texture(&d, Some(4096 + index)))
                        .unwrap_or_else(|error| panic!("{name}: texture {index}: {error}")),
                );
            }
            let mut blocks = Vec::new();
            for take in 0..64 {
                let len = (64 << 10) << (next() % 7);
                blocks.push(
                    (device.take_block(MemoryClass::Private, len))
                        .unwrap_or_else(|error| panic!("{name}: block {take}: {error}")),
                );
            }
            blocks.push(
                (device.take_block(MemoryClass::Private, unit + 1))
                    .unwrap_or_else(|error| panic!("{name}: a long block: {error}")),
            );
            let most = counters.gauges().private_allocation_bytes;
            drop(textures);
            for block in blocks {
                (device.release_block(block))
                    .unwrap_or_else(|error| panic!("{name}: a release: {error}"));
            }
            let released = Instant::now();
            // The allocations, the idle ones, the reserve, and the texture blocks.
            let fixed = [2 * unit, unit, unit, unit];
            let read = || {
                let gauges = counters.gauges();
                [
                    gauges.private_allocation_bytes,
                    gauges.private_idle_bytes,
                    gauges.private_reserve_bytes,
                    gauges.texture_block_held_bytes,
                ]
            };
            while read() != fixed && released.elapsed() < Duration::from_secs(1) {
                thread::sleep(Duration::from_millis(1));
            }
            let took = released.elapsed();
            println!(
                "places: {name}: {most} bytes of Private allocations at the most, {:?} {took:?} \
                 after the release",
                read()
            );
            assert_eq!(
                read(),
                fixed,
                "{name}: a second after the release of {most} bytes of Private allocations"
            );
            let gauges = counters.gauges();
            assert_eq!(
                (gauges.textures, takes(&gauges), texture_blocks(&gauges)),
                (0, [0; 4], [unit, 0, 0, 0]),
                "{name}: nothing placed"
            );
            device
                .release_indices(grant)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
        }
    }
}
