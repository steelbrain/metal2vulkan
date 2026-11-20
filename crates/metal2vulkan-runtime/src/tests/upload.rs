//! The upload that goes ahead of later commits: under a device cfg, on every device of the cfg's
//! kind, with no command buffer committed and nothing waiting on the host but the calls
//! themselves, but in the last two:
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
//! - a `replace_region` and a new texture made after a commit held at an event wait go in the
//!   submission of the next commit, so the held one copies out the bytes from before the call;
//! - two copies into one texture in two uploads of one submission leave the later one's bytes;
//!
//! and every gauge reads zero after teardown.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::staging::REGION_LEN;
    use crate::submit::{HOMES_PRUNE, UPLOAD_BYTES, UPLOAD_ENTRIES};
    use crate::tests::command::on_device::{
        behind_busy, each_device, finished, within, Busy, PATIENCE,
    };
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        Buffer, BufferLayout, CommandBuffer, CommandBufferStatus, Device, DeviceOptions, Error,
        Gauges, IndexClass, MemoryClass, PixelFormat, Region, StorageMode, Texture,
        TextureDescriptor, TexturePlace, TextureType,
    };
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
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
            wait_until("a wholly free unit in each class", || {
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
            // The region, which the block-maker thread makes as its first work.
            device.core().pool.await_internals();
            assert_eq!(staged(), REGION_LEN, "{name}: before any copy");
            // A copy of 256 KiB, through the region alone.
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

    /// The uploads are cut at commits: a `replace_region` and a new texture made after a commit
    /// go in the submission that carries the next commit, not in that of the one before, though
    /// the submit thread holds that one at an event wait until after the calls. The first command
    /// buffer copies out the texture's bytes from before the later call, the next its bytes after
    /// it, and the new texture's move home completes with the next one's submission.
    #[test]
    fn every_device_puts_a_call_after_a_commit_in_the_next_commit_s_submission() {
        each_device("upload cut", |name, _, device| {
            let core = device.core();
            device
                .grant_indices(IndexClass::Texture, 4096, 2)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let side = 16;
            let len = u64::from(side * side * 4);
            let block = (device.take_block(MemoryClass::Shared, 4 * len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let out = |offset| {
                (device.new_buffer(block.id, offset, len, Some(0)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let (first_out, next_out) = (out(0), out(2 * len));
            let texture = (device.new_texture(&square(side), Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let event = device.new_shared_event().expect("an event");

            replace(name, &texture, side, 0xA1);
            let before = core.work.ready_point();
            let first = queue.command_buffer().expect("a command buffer");
            (first.encode_wait_for_event(event.as_event(), 1)).expect("a wait");
            copy_out(name, &first, &texture, side, &first_out);
            first.commit().expect("a commit");
            // After the commit, the first command buffer held at its wait.
            replace(name, &texture, side, 0xB2);
            let made = (device.new_texture(&square(side), Some(4097)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let after = core.work.ready_point();
            let next = queue.command_buffer().expect("a command buffer");
            copy_out(name, &next, &texture, side, &next_out);
            next.commit().expect("a commit");
            assert_eq!(
                first.status(),
                CommandBufferStatus::Committed,
                "{name}: the first command buffer is held at its wait"
            );
            event.set_signaled_value(1);
            finished(name, &next);
            finished(name, &first);
            assert_eq!(first.error(), None, "{name}");
            assert_eq!(next.error(), None, "{name}");

            let (carried_first, carried_next) = (
                before.load(Ordering::Relaxed),
                after.load(Ordering::Relaxed),
            );
            println!(
                "upload cut: {name}: the first commit's upload went in submission {carried_first}, \
                 the next commit's in {carried_next}, of {}",
                core.work.progress()
            );
            assert!(
                carried_first > 0 && carried_first < carried_next,
                "{name}: the calls after the first commit went in its submission \
                 ({carried_first}, {carried_next})"
            );
            assert_eq!(
                carried_next,
                core.work.progress(),
                "{name}: the calls after the first commit go with the next one"
            );
            assert!(
                read(&first_out) == vec![0xA1; len as usize],
                "{name}: the first command buffer reads the bytes from before the later call"
            );
            assert!(
                read(&next_out) == vec![0xB2; len as usize],
                "{name}: the next command buffer reads the bytes of the later call"
            );
            drop((
                first, next, event, queue, made, texture, first_out, next_out,
            ));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// Two copies into one texture in two uploads of one submission: the first made before the
    /// commit of a command buffer its queue holds in a group behind busy work, the second after
    /// it, before the commit of the next, which joins that group and copies the texture out. The
    /// two calls record into two uploads, the submission of the group, sent once both command
    /// buffers are staged into it and before the busy work ends (`behind_busy`), carries both,
    /// and the later
    /// copy's bytes are the texture's: each copy orders itself after every earlier write of the
    /// image with a barrier of its own, as command buffer boundaries add no order.
    #[test]
    fn every_device_orders_a_copy_after_a_copy_in_two_uploads_of_one_submission() {
        each_device("upload copy after copy", |name, _, device| {
            let core = device.core();
            device
                .grant_indices(IndexClass::Texture, 4096, 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let side = 16;
            let len = u64::from(side * side * 4);
            let block = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let out = (device.new_buffer(block.id, 0, len, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let texture = (device.new_texture(&square(side), Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // The busy work's command buffers and the test's two are made and not completed at
            // once: a queue of 4 would hold the second `command_buffer` until a busy one
            // completed, which only `busy.open` after it lets happen.
            let queue = device.new_command_queue(8).expect("a queue");
            let mut points = Vec::new();
            let mut held = Vec::new();
            let mut run = None;
            // Each step's submissions counted, progress seen and the completion thread's round,
            // which a failed check names.
            let mut trace = Vec::new();
            let mut step = |label: &str| {
                trace.push(format!(
                    "{label}: {} sent, {} seen, busy {}",
                    device.counters().metrics().submissions,
                    core.done.completed(),
                    core.done.busy()
                ))
            };
            behind_busy(name, &device, &queue, |busy, _, _| {
                step("behind the busy work");
                replace(name, &texture, side, 0xC3);
                points.push(core.work.ready_point());
                step("first replace");
                let first = queue.command_buffer().expect("a command buffer");
                (first.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
                first.commit().expect("a commit");
                step("first commit");
                replace(name, &texture, side, 0xD4);
                points.push(core.work.ready_point());
                step("second replace");
                let next = queue.command_buffer().expect("a command buffer");
                copy_out(name, &next, &texture, side, &out);
                next.commit().expect("a commit");
                step("next commit");
                // Both staged into the group the queue holds before it is sent: a send while the
                // submit thread, woken late on a loaded host, has staged neither sends the first
                // as it joins a group of its own, before the next joins it.
                for command_buffer in [&first, &next] {
                    within(name, || {
                        command_buffer.status() != CommandBufferStatus::Committed
                    });
                }
                step("both staged");
                // The group goes before the busy work ends, at the run `behind_busy` set: sent
                // once a busy one retires, it would go at the run their own runs leave, member by
                // member where a loaded host measured them long.
                run = next.queue().expected_run();
                let sent = device.counters().metrics().submissions;
                queue.send_now(&next);
                within(name, || device.counters().metrics().submissions > sent);
                step("sent");
                busy.open();
                held.extend([first, next]);
            });
            for command_buffer in &held {
                finished(name, command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}");
            }
            assert!(
                !Arc::ptr_eq(&points[0], &points[1]),
                "{name}: the calls before and after the commit recorded into one upload"
            );
            let carried = points.iter().map(|point| point.load(Ordering::Relaxed));
            let carried: Vec<u64> = carried.collect();
            println!("upload copy after copy: {name}: the uploads went in submissions {carried:?}");
            assert_eq!(
                carried[0], carried[1],
                "{name}: the two uploads went in one submission: they went in submissions \
                 {carried:?}, the queue's expected run {run:?} ns as the group went; {trace:?}"
            );
            assert!(
                read(&out) == vec![0xD4; len as usize],
                "{name}: the later copy's bytes are the texture's"
            );
            drop((held, points, queue, texture, out));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// Writes `byte` over the whole of `texture`, `side` texels square, through the upload.
    fn replace(name: &str, texture: &Texture, side: u32, byte: u8) {
        let len = u64::from(side * side * 4);
        let bytes = vec![byte; len as usize];
        texture
            .replace_region(
                whole(side),
                0,
                0,
                crate::HostBytes::from(&bytes[..]),
                u64::from(side * 4),
                len,
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }

    /// Records into `command_buffer` a copy of the whole of `texture`, `side` texels square, into
    /// `to`.
    fn copy_out(
        name: &str,
        command_buffer: &CommandBuffer,
        texture: &Texture,
        side: u32,
        to: &Buffer,
    ) {
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(side * 4),
            bytes_per_image: 0,
        };
        (encoder.copy_from_texture_to_buffer(
            texture,
            TexturePlace::default(),
            [side, side, 1],
            to,
            layout,
        ))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        encoder.end_encoding();
    }

    /// A Shared buffer's bytes, as the host reads them.
    fn read(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive, `length` bytes long, and no GPU work uses it.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }
}
