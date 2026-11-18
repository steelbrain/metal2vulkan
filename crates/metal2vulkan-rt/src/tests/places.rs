//! Texture blocks: a block's free ranges give the first aligned room and join again as places are
//! freed; and, under a device cfg, on every device of the cfg's kind, textures are placed in
//! blocks of the pool with no allocation of their own: a block's worth of 1 MiB textures takes
//! one block and then a second, releasing them gives one back to the pool and keeps the other,
//! textures made and released in that block allocate nothing, and every gauge reads zero after
//! teardown.

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
    use crate::{
        Device, DeviceOptions, Gauges, IndexClass, Instance, MemoryClass, PixelFormat,
        TextureDescriptor, TextureType, BLOCK_LEN,
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
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            wait_until("a ready block of each class", || {
                core.pool.idle(MemoryClass::Private) == 1
                    && core.pool.idle(MemoryClass::Shared) == 1
            });
            let base = counters.gauges().blocks;
            let made = counters.metrics().blocks_made;
            let most = 2 * BLOCK_LEN / MIB;
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
                in_first > BLOCK_LEN / MIB / 2,
                "{name}: the first block held {in_first} textures of 1 MiB"
            );
            // Each block the pool's ready one, which the pool made again.
            wait_until("the pool's two blocks made again", || {
                counters.metrics().blocks_made == made + 2
            });
            assert_eq!(counters.gauges().blocks, base + 2, "{name}: two blocks");

            // Released: one block back to the pool, which keeps it idle; one kept for textures.
            drop(textures);
            wait_until("the textures freed", || {
                counters.gauges().textures == 0 && core.places.blocks() == 1
            });
            wait_until("the block back in the pool", || {
                core.pool.idle(MemoryClass::Private) == 2
            });
            assert_eq!(counters.gauges().blocks, base + 2, "{name}: after release");

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
                base + 2,
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
}
