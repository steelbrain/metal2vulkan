//! The upload that goes ahead of later commits: under a device cfg, on every device of the cfg's
//! kind, with no command buffer committed and nothing waiting on the host but the calls
//! themselves:
//!
//! - textures made and released are freed at once: their moves to a home layout wait for a
//!   submission, and hold nothing until one;
//! - `replace_region` calls of twice the staging region's bytes into four textures stage every
//!   copy in the region: no call allocates, the ring wraps, the upload goes each time its copies
//!   take half the region, and each texture reads its last call's bytes; once the reads have
//!   waited for every copy, the next copy's span starts at the region's start;
//! - textures released while the upload holds them for a copy, and a block released while it
//!   holds it for a zero fill, count against the bound: a block of the bound's length alone keeps
//!   the upload pending, and past the bound the next call sends it and frees them;
//! - zero fills past [`UPLOAD_ENTRIES`] submit it;
//!
//! and every gauge reads zero after teardown.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::staging::REGION_LEN;
    use crate::submit::{HOMES_PRUNE, UPLOAD_BYTES, UPLOAD_ENTRIES};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        Device, DeviceOptions, Gauges, IndexClass, Instance, MemoryClass, PixelFormat, Region,
        StorageMode, TextureDescriptor, TextureType,
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

    /// A Shared RGBA8 texture of `side` texels square.
    fn square(side: u32) -> TextureDescriptor {
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, side, side);
        d.storage_mode = StorageMode::Shared;
        d
    }

    fn whole(side: u32) -> Region {
        Region {
            origin: [0; 3],
            size: [side, side, 1],
        }
    }

    #[test]
    fn every_device_bounds_the_pending_upload() {
        const MIB: u64 = 1 << 20;
        let instance = Instance::new(&[]).expect("an instance at the floor");
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let core = device.core();
            let counters = device.counters();
            // The pool's first allocations, which the device's making makes.
            wait_until("8 MiB free in each class", || {
                core.pool.stocked(MemoryClass::Private) && core.pool.stocked(MemoryClass::Shared)
            });
            device
                .grant_indices(IndexClass::Texture, 4096, 8)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let submissions = || counters.metrics().submissions;
            // The first texture's texture block takes the first Private allocation whole, and the
            // block-maker thread makes the next, zeroing it in a submission of its own.
            drop(
                (device.new_texture(&square(256), Some(4096)))
                    .unwrap_or_else(|error| panic!("{name}: {error}")),
            );
            wait_until("the next Private allocation", || {
                core.pool.stocked(MemoryClass::Private) && counters.gauges().textures == 0
            });
            let base = counters.gauges().blocks;

            // Textures made and released: nothing holds them, and nothing is submitted.
            let before = submissions();
            for _ in 0..4 * HOMES_PRUNE {
                let texture = device
                    .new_texture(&square(256), Some(4096))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                drop(texture);
                let (_, _, homes) = core.work.pending();
                assert!(
                    homes <= HOMES_PRUNE,
                    "{name}: {homes} textures await a move"
                );
            }
            wait_until("the released textures freed", || {
                let gauges = counters.gauges();
                (gauges.textures, gauges.blocks) == (0, base)
            });
            assert_eq!(submissions(), before, "{name}: textures made and released");

            // Whole-texture copies of 4,000,000 bytes, twice the staging region's length, into four
            // textures in turn: the region is the only allocation, the textures being placed in
            // the texture block, and the upload goes each time its copies take half the region.
            let side = 1000;
            let call = u64::from(side * side * 4);
            let calls = 2 * REGION_LEN / call + 3;
            let per_upload = (REGION_LEN / 2).div_ceil(call);
            let targets: Vec<_> = (0..4)
                .map(|index| {
                    device
                        .new_texture(&square(side), Some(4097 + index))
                        .unwrap_or_else(|error| panic!("{name}: {error}"))
                })
                .collect();
            let ceiling = base + 1;
            let pattern = |seed: u64| -> Vec<u8> {
                (0..call)
                    .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed as u8))
                    .collect()
            };
            let waits = counters.metrics().upload_waits;
            let before = submissions();
            for seed in 0..calls {
                let target = &targets[(seed % 4) as usize];
                target
                    .replace_region(whole(side), 0, 0, &pattern(seed), u64::from(side * 4), call)
                    .unwrap_or_else(|error| panic!("{name}: copy {seed}: {error}"));
                let blocks = counters.gauges().blocks;
                assert!(
                    blocks <= ceiling,
                    "{name}: copy {seed}: {blocks} allocations live, at most {ceiling} expected"
                );
                let (_, bytes, _) = core.work.pending();
                assert_eq!(bytes, 0, "{name}: copy {seed}: staging of its own");
            }
            let waited = counters.metrics().upload_waits - waits;
            assert!(
                submissions() - before >= calls / per_upload,
                "{name}: {calls} copies of {call} bytes went in {} submissions",
                submissions() - before
            );
            for (index, target) in targets.iter().enumerate() {
                let last = calls - 1 - (calls - 1 - index as u64) % 4;
                let mut read = vec![0; call as usize];
                target
                    .get_bytes(&mut read, u64::from(side * 4), call, whole(side), 0, 0)
                    .unwrap_or_else(|error| panic!("{name}: get_bytes: {error}"));
                assert!(
                    read == pattern(last),
                    "{name}: texture {index}: copy {last}'s bytes"
                );
            }
            assert!(
                counters.gauges().blocks <= ceiling,
                "{name}: after the reads"
            );
            // The reads waited for every copy, so the ring is empty, and starts again at 0.
            targets[0]
                .replace_region(whole(side), 0, 0, &pattern(0), u64::from(side * 4), call)
                .unwrap_or_else(|error| panic!("{name}: the copy after the reads: {error}"));
            let spans = core.work.staging_spans();
            assert_eq!(
                spans
                    .iter()
                    .map(|&(start, end, _)| (start, end))
                    .collect::<Vec<_>>(),
                [(0, call)],
                "{name}: the copy after the reads"
            );
            // A read of one texel sends it and waits, so the upload starts the next part empty.
            let texel = Region {
                origin: [0; 3],
                size: [1, 1, 1],
            };
            targets[0]
                .get_bytes(&mut [0; 4], 4, 4, texel, 0, 0)
                .unwrap_or_else(|error| panic!("{name}: get_bytes: {error}"));

            // Textures of 4 MiB released after a copy of 64 bytes: the upload holds them alone.
            let big = 1024;
            let rounds = 3 * UPLOAD_BYTES / (4 * MIB);
            let before = submissions();
            for round in 0..rounds {
                let texture = device
                    .new_texture(&square(big), Some(4101))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                texture
                    .replace_region(whole(4), 0, 0, &[round as u8; 64], 16, 64)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                drop(texture);
                let (_, bytes, _) = core.work.pending();
                assert!(
                    bytes <= UPLOAD_BYTES + 8 * MIB,
                    "{name}: round {round}: the upload keeps {bytes} bytes"
                );
            }
            assert!(
                submissions() >= before + 2,
                "{name}: {rounds} released textures submitted {} times",
                submissions() - before
            );
            // The copy targets, and at most one bound's worth of released textures.
            let kept = targets.len() as u64 + UPLOAD_BYTES / (4 * MIB);
            wait_until("the released textures freed down to one upload's", || {
                counters.gauges().textures <= kept
            });

            // A Private block released after a zero fill counts against the bound. The block is
            // the bound's length, so the upload starts this part empty (the read of a texel sends
            // it and waits) whatever the textures above left in it, which differs by device.
            targets[0]
                .get_bytes(&mut [0; 4], 4, 4, texel, 0, 0)
                .unwrap_or_else(|error| panic!("{name}: get_bytes: {error}"));
            let released = device
                .take_block(MemoryClass::Private, MemoryClass::Private.block_len())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = device
                .new_buffer(released.id, 0, 256, true)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(buffer);
            let released_id = released.id;
            device
                .release_block(released)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (_, bytes, _) = core.work.pending();
            assert_eq!(
                bytes,
                MemoryClass::Private.block_len(),
                "{name}: the released block is counted"
            );
            // At the bound and not past it, a fill keeps the upload pending: the released block
            // still counts, where a send would have started the count again. The upload's own
            // count, not the device's submissions: the block-maker thread's zero fills of the
            // blocks it makes and is given back submit beside it, when they come (on llvmpipe,
            // one came inside this step).
            let fills = device
                .take_block(MemoryClass::Private, MemoryClass::Private.block_len())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = device
                .new_buffer(fills.id, 0, 256, true)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(buffer);
            let (_, bytes, _) = core.work.pending();
            assert_eq!(
                bytes,
                MemoryClass::Private.block_len(),
                "{name}: a fill at the bound"
            );
            assert!(!core.pool.holds(released_id), "{name}: the block held");
            // A texture released after a copy takes it past the bound: the next fill sends the
            // upload, and the block comes back to the pool.
            let small = device
                .new_texture(&square(64), Some(4102))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            small
                .replace_region(whole(4), 0, 0, &[7; 64], 16, 64)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(small);
            let buffer = device
                .new_buffer(fills.id, 256, 256, true)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(buffer);
            let (_, bytes, _) = core.work.pending();
            assert!(
                bytes < MemoryClass::Private.block_len(),
                "{name}: a fill past the bound keeps {bytes} bytes pending"
            );
            wait_until("the released block back in the pool", || {
                core.pool.holds(released_id)
            });

            // Zero fills past the upload's calls.
            let before = submissions();
            for fill in 0..UPLOAD_ENTRIES as u64 + 8 {
                let buffer = device
                    .new_buffer(fills.id, fill * 256, 4, true)
                    .unwrap_or_else(|error| panic!("{name}: fill {fill}: {error}"));
                drop(buffer);
                let (entries, _, _) = core.work.pending();
                assert!(
                    entries <= UPLOAD_ENTRIES,
                    "{name}: fill {fill}: {entries} calls"
                );
            }
            assert!(submissions() > before, "{name}: the fills went");

            drop(targets);
            device
                .release_block(fills)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
            println!(
                "upload: {name}: {calls} copies waited {waited} times, {rounds} released textures; \
                 metrics {:?}",
                counters.metrics()
            );
        }
    }
}
