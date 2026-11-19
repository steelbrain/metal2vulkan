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
//! - the staging gauge counts the region once a copy made it, and a copy larger than the region's
//!   staging buffer of its own until the upload that copies through it is gone;
//! - a copy that finds the region full behind a `get_bytes`'s span sleeps until the read ends:
//!   once, with no spin, waking at the read's end, and a quiesce ends the sleep;
//! - a quiesce past its deadline, a `get_bytes`'s copy in flight behind busy work, leaves the
//!   completion thread to end alone once the copy completes, and the device then frees itself;
//!
//! and every gauge reads zero after teardown.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::staging::REGION_LEN;
    use crate::submit::{HOMES_PRUNE, UPLOAD_BYTES, UPLOAD_ENTRIES};
    use crate::tests::command::on_device::{Busy, PATIENCE};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        Device, DeviceOptions, Error, Gauges, IndexClass, MemoryClass, PixelFormat, Region,
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
        let instance = crate::tests::validation::instance(&[]);
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
            // block-maker thread makes the next.
            drop(
                (device.new_texture(&square(256), Some(4096)))
                    .unwrap_or_else(|error| panic!("{name}: {error}")),
            );
            wait_until("the next Private allocation", || {
                core.pool.stocked(MemoryClass::Private) && counters.gauges().textures == 0
            });
            // The allocations in use: the idle ones are the pool's, kept for its window. A texture
            // made while earlier ones are still being freed can take a second texture block, which
            // goes back to the pool once empty and stays there idle. Counted in the pool, not from
            // the gauges, which also count an allocation the block-maker thread is making or
            // freeing at the time.
            let in_use = || core.pool.in_use() as u64;
            let base = in_use();

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
                counters.gauges().textures == 0
            });
            wait_until("their allocations idle", || in_use() == base);
            println!(
                "upload: {name}: after the released textures, {:?}",
                counters.gauges()
            );
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
                    .replace_region(
                        whole(side),
                        0,
                        0,
                        crate::HostBytes::from(&pattern(seed)[..]),
                        u64::from(side * 4),
                        call,
                    )
                    .unwrap_or_else(|error| panic!("{name}: copy {seed}: {error}"));
                let blocks = in_use();
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
            assert!(in_use() <= ceiling, "{name}: after the reads");
            // The reads waited for every copy, so the ring is empty, and starts again at 0.
            targets[0]
                .replace_region(
                    whole(side),
                    0,
                    0,
                    crate::HostBytes::from(&pattern(0)[..]),
                    u64::from(side * 4),
                    call,
                )
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
                    .replace_region(
                        whole(4),
                        0,
                        0,
                        crate::HostBytes::from(&[round as u8; 64][..]),
                        16,
                        64,
                    )
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
                .take_block(MemoryClass::Private, UPLOAD_BYTES)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = device
                .new_buffer(released.id, 0, 256, Some(0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(buffer);
            let released_id = released.id;
            device
                .release_block(released)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (_, bytes, _) = core.work.pending();
            assert_eq!(bytes, UPLOAD_BYTES, "{name}: the released block is counted");
            // At the bound and not past it, a fill keeps the upload pending: the released block
            // still counts, where a send would have started the count again. The upload's own
            // count, not the device's submissions: the block-maker thread's zero fills of the
            // blocks it makes and is given back submit beside it, when they come (on llvmpipe,
            // one came inside this step).
            let fills = device
                .take_block(MemoryClass::Private, UPLOAD_BYTES)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = device
                .new_buffer(fills.id, 0, 256, Some(0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(buffer);
            let (_, bytes, _) = core.work.pending();
            assert_eq!(bytes, UPLOAD_BYTES, "{name}: a fill at the bound");
            assert!(!core.pool.holds(released_id), "{name}: the block held");
            // A texture released after a copy takes it past the bound: the next fill sends the
            // upload, and the block comes back to the pool.
            let small = device
                .new_texture(&square(64), Some(4102))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            small
                .replace_region(whole(4), 0, 0, crate::HostBytes::from(&[7; 64][..]), 16, 64)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(small);
            let buffer = device
                .new_buffer(fills.id, 256, 256, Some(0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            drop(buffer);
            let (_, bytes, _) = core.work.pending();
            assert!(
                bytes < UPLOAD_BYTES,
                "{name}: a fill past the bound keeps {bytes} bytes pending"
            );
            wait_until("the released block back in the pool", || {
                core.pool.holds(released_id)
            });

            // Zero fills past the upload's calls.
            let before = submissions();
            for fill in 0..UPLOAD_ENTRIES as u64 + 8 {
                let buffer = device
                    .new_buffer(fills.id, fill * 256, 4, Some(0))
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

    /// A `get_bytes` of the region's whole 16 MiB holds the ring's one span, read, while its copy
    /// waits behind busy work the host holds ([`Busy`]); a `replace_region` on another thread then
    /// finds no room behind it. The write sleeps once (`staging_sleeps` rises by one and no more,
    /// so the loop turned once to sleep and once on waking) and returns only after the read ends.
    /// In the second round a quiesce on a third thread ends the sleep instead: the write is
    /// refused, and the quiesce goes idle once the busy work opens.
    #[test]
    fn every_device_sleeps_a_staging_call_behind_a_read_until_it_ends() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        let side = 2048;
        let len = u64::from(side) * u64::from(side) * 4;
        assert_eq!(len, REGION_LEN, "the read's span takes the region");
        for (name, physical) in &devices {
            for quiesce in [false, true] {
                let device = Device::new(physical, DeviceOptions::default())
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let counters = device.counters();
                device
                    .grant_indices(IndexClass::Texture, 4096, 2)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let read = (device.new_texture(&square(side), Some(4096)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let write = (device.new_texture(&square(4), Some(4097)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                // What the device and the textures' making left pending goes and completes, so
                // the next submission is the read's.
                let core = device.core();
                let value = core.work.flush();
                (core.done.wait(core, value)).unwrap_or_else(|error| panic!("{name}: {error}"));
                let submitted = counters.metrics().submissions;
                let busy = Busy::new(&device);
                let reader = thread::spawn(move || {
                    let mut bytes = vec![0; len as usize];
                    read.get_bytes(&mut bytes, u64::from(side * 4), len, whole(side), 0, 0)
                });
                // Submitted behind the busy work: the read's span stays the ring's front, read,
                // until the read ends.
                wait_until("the read's submission", || {
                    counters.metrics().submissions > submitted
                });
                let sleeps = counters.metrics().staging_sleeps;
                let writer = thread::spawn(move || {
                    write.replace_region(
                        whole(4),
                        0,
                        0,
                        crate::HostBytes::from(&[7; 64][..]),
                        16,
                        64,
                    )
                });
                wait_until("the write's sleep", || {
                    counters.metrics().staging_sleeps > sleeps
                });
                thread::sleep(Duration::from_millis(50));
                assert!(
                    !writer.is_finished(),
                    "{name}: the write returned while the read held the region"
                );
                assert_eq!(
                    counters.metrics().staging_sleeps - sleeps,
                    1,
                    "{name}: the write slept once, with no spin"
                );
                if quiesce {
                    // The quiesce closes the device at once, and then waits for the GPU, which
                    // the busy work holds until the write has answered.
                    let quiescing = {
                        let device = device.clone();
                        thread::spawn(move || device.quiesce(PATIENCE))
                    };
                    let written = writer.join().expect("the writer");
                    assert!(
                        matches!(written, Err(Error::Quiesced)),
                        "{name}: the quiesce ended the sleep with {written:?}"
                    );
                    busy.open();
                    assert!(
                        quiescing.join().expect("the quiesce"),
                        "{name}: the device went idle once the busy work opened"
                    );
                    let _ = reader.join().expect("the reader");
                    drop(busy);
                    drop(device);
                    continue;
                }
                busy.open();
                (reader.join().expect("the reader"))
                    .unwrap_or_else(|error| panic!("{name}: get_bytes: {error}"));
                (writer.join().expect("the writer"))
                    .unwrap_or_else(|error| panic!("{name}: replace_region: {error}"));
                assert_eq!(
                    counters.metrics().staging_sleeps - sleeps,
                    1,
                    "{name}: the write woke at the read's end and slept no more"
                );
                drop(busy);
                drop(device);
                wait_until("the teardown", || counters.gauges() == Gauges::default());
            }
        }
    }

    /// A quiesce that answers past its deadline while busy work the host holds ([`Busy`]) keeps
    /// the GPU, with a `get_bytes` copy submitted behind it, leaves the completion thread to end
    /// alone: once the busy work opens, it retires the copy, which held what it used (the core
    /// among it), and ends. The device's last handle then frees everything: every gauge reads
    /// zero, and the instance goes within the validation guard's patience.
    #[test]
    fn every_device_frees_itself_after_a_quiesce_past_its_deadline() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        let side = 256;
        let len = u64::from(side) * u64::from(side) * 4;
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            device
                .grant_indices(IndexClass::Texture, 4096, 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let read = (device.new_texture(&square(side), Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // What the device and the texture's making left pending goes and completes, so the
            // next submission is the read's.
            let core = device.core();
            let value = core.work.flush();
            (core.done.wait(core, value)).unwrap_or_else(|error| panic!("{name}: {error}"));
            let submitted = counters.metrics().submissions;
            let busy = Busy::new(&device);
            let reader = thread::spawn(move || {
                let mut bytes = vec![0; len as usize];
                read.get_bytes(&mut bytes, u64::from(side * 4), len, whole(side), 0, 0)
            });
            wait_until("the read's submission", || {
                counters.metrics().submissions > submitted
            });
            assert!(
                !device.quiesce(Duration::from_millis(100)),
                "{name}: the quiesce answers past its deadline while the busy work holds the GPU"
            );
            busy.open();
            let _ = reader.join().expect("the reader");
            drop(busy);
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
        }
    }

    #[test]
    fn every_device_counts_its_staging_bytes() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let device = Device::new(physical, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            device
                .grant_indices(IndexClass::Texture, 4096, 8)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let staged = || counters.gauges().staging_bytes;
            assert_eq!(staged(), 0, "{name}: before any copy");
            // A copy of 256 KiB: the region, made at the first copy.
            let small = 256;
            let texture = (device.new_texture(&square(small), Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let bytes = vec![7u8; (small * small * 4) as usize];
            texture
                .replace_region(
                    whole(small),
                    0,
                    0,
                    crate::HostBytes::from(&bytes[..]),
                    u64::from(small * 4),
                    bytes.len() as u64,
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(staged(), REGION_LEN, "{name}: the region");
            // A copy of 2304 texels square, 21,233,664 bytes, larger than the region: a staging
            // buffer of its own besides, until the upload that copies through it is gone.
            let large = 2304;
            let big = (device.new_texture(&square(large), Some(4097)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let bytes: Vec<u8> = (0..large * large * 4).map(|i| (i % 251) as u8).collect();
            let len = bytes.len() as u64;
            big.replace_region(
                whole(large),
                0,
                0,
                crate::HostBytes::from(&bytes[..]),
                u64::from(large * 4),
                len,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                staged(),
                REGION_LEN + len,
                "{name}: a staging buffer of its own"
            );
            let mut read = vec![0; len as usize];
            big.get_bytes(&mut read, u64::from(large * 4), len, whole(large), 0, 0)
                .unwrap_or_else(|error| panic!("{name}: get_bytes: {error}"));
            assert!(read == bytes, "{name}: the large copy's bytes");
            wait_until("the staging buffers gone", || staged() == REGION_LEN);
            drop((texture, big));
            drop(device);
            wait_until("the teardown", || counters.gauges() == Gauges::default());
        }
    }
}
