//! The runtime's fixed-size internals (`crate::memory::Internal`), under a device cfg on every
//! device of the cfg's kind: the block-maker thread makes the zero source, the parameter ring's
//! first block and the staging region as its first work, so the device's making and its first
//! copy of a texture's bytes, first encoder that writes parameters and first fill that copies from
//! the zero source allocate none of them on the caller's thread; each such call made before the
//! block-maker thread's make waits for it, allocates nothing and is counted in
//! `Metrics::maker_waits`; and a device dropped before the block-maker thread made them, or while
//! it makes one, frees what it made, every gauge at zero.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::lock;
    use crate::memory::MakerTest;
    use crate::staging::REGION_LEN;
    use crate::tests::command::on_device::{each_device, finished, within, Arena, Release};
    use crate::tests::compute::on_device::{add, WORDS};
    use crate::tests::devices_of_the_cfgs_kind;
    use crate::{
        Device, DeviceOptions, DispatchType, Gauges, HostBytes, MemoryClass, Metrics, PixelFormat,
        Region, SiteAllocations, StorageMode, TextureDescriptor, TextureType, TextureUsage,
    };
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    /// How long the block-maker thread stays parked while the calls that need its internals wait.
    const HOLD: Duration = Duration::from_millis(250);

    /// A call that needs one of the internals, made ready with all else it needs.
    type Call = (&'static str, Box<dyn FnOnce() + Send>);

    /// The consumer's-thread sites of the internals in `metrics`, by name.
    fn sites(metrics: &Metrics) -> [(&'static str, SiteAllocations); 3] {
        [
            ("zero source", metrics.zero_source_allocations),
            ("ring block", metrics.ring_block_allocations),
            ("staging region", metrics.staging_region_allocations),
        ]
    }

    /// The sites of `metrics` that counted an allocation.
    fn allocated(metrics: &Metrics) -> Vec<(&'static str, SiteAllocations)> {
        (sites(metrics).into_iter())
            .filter(|(_, site)| *site != SiteAllocations::default())
            .collect()
    }

    /// The device's first calls that need an internal, each made ready on this thread with all
    /// else it needs, none of which needs one: the first copy of a texture's bytes (the staging
    /// region), the first encoder that writes parameters (the ring) and the first fill that
    /// copies from the zero source.
    fn first_calls(name: &str, device: &Device) -> Vec<Call> {
        let mut calls: Vec<Call> = Vec::new();

        // A render target, which no shader reads, so it takes no texture index.
        let mut d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 64, 64);
        d.usage = TextureUsage::RENDER_TARGET;
        d.storage_mode = StorageMode::Shared;
        let texture =
            (device.new_texture_for_views(&d)).unwrap_or_else(|error| panic!("{name}: {error}"));
        calls.push((
            "a copy of a texture's bytes",
            Box::new(move || {
                let bytes = vec![0x5au8; 64 * 64 * 4];
                let whole = Region {
                    origin: [0; 3],
                    size: [64, 64, 1],
                };
                (texture.replace_region(whole, 0, 0, HostBytes::from(&bytes[..]), 256, 0))
                    .expect("a replace");
            }),
        ));

        let mut arena = Arena::new(device);
        let (a, o) = (arena.buffer(), arena.buffer());
        let state = add(device);
        let queue = device.new_command_queue(1).expect("a queue");
        let label = name.to_string();
        calls.push((
            "an encoder that writes parameters",
            Box::new(move || {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&a), 0, 0).expect("a buffer");
                encoder.set_bytes(&[7u8; WORDS * 4], 1).expect("bytes");
                encoder.set_buffer(Some(&o), 0, 2).expect("a buffer");
                (encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])).expect("a dispatch");
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(&label, &command_buffer);
                drop(arena);
            }),
        ));

        let private = (device.take_block(MemoryClass::Private, 4096))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let filler = device.clone();
        calls.push((
            "a fill that copies from the zero source",
            Box::new(move || {
                // Bytes 1..9: the words 4..8 filled, the bytes beside them copied.
                (filler.zero_block_range(private.id, 1, 8)).expect("a fill");
                drop(private);
            }),
        ));
        calls
    }

    /// The device's making allocates none of the internals on the caller's thread, and the
    /// block-maker thread makes the zero source, the ring's first block and the staging region.
    /// Once it settled, the device's first copy, encoder that writes parameters and fill that
    /// copies from the zero source allocate nothing on the caller's thread and wait for nothing.
    #[test]
    fn every_device_makes_its_internals_off_the_caller_s_thread() {
        each_device("internals_off_the_caller", |name, _, device| {
            let core = device.core();
            let counters = device.counters();
            // What the device's making and the block-maker thread's first work left, read before
            // any call needs an internal.
            let made = counters.metrics();
            let staging_bytes = counters.gauges().staging_bytes;
            let zero_source = core.filler.get().is_some();
            let ring_blocks = core.ring.chunks().map(|(blocks, _)| blocks);

            let calls = first_calls(name, &device);
            let before = counters.metrics();
            let what: Vec<&str> = calls.iter().map(|&(what, _)| what).collect();
            for (_, call) in calls {
                call();
            }
            let after = counters.metrics();
            let on_caller = allocated(&after);
            println!(
                "ahead: {name}: {what:?}: consumer-thread sites {:?}, maker waits {}",
                sites(&after),
                after.maker_waits
            );
            assert!(
                on_caller.is_empty(),
                "{name}: allocated on the caller's thread: {on_caller:?}"
            );
            assert_eq!(
                made.consumer_allocations, 0,
                "{name}: the making allocated on the caller's thread"
            );
            assert!(
                made.maker_allocation_nanos > 0,
                "{name}: the block-maker thread's makes"
            );
            assert!(zero_source, "{name}: the zero source");
            assert_eq!(ring_blocks, Some(1), "{name}: the ring's first block");
            assert_eq!(staging_bytes, REGION_LEN, "{name}: the staging region");
            assert_eq!(
                after.maker_waits, 0,
                "{name}: waited for the block-maker thread"
            );
            assert!(
                after.ring_writes > before.ring_writes,
                "{name}: the encoder wrote the ring"
            );
        });
    }

    /// With the block-maker thread parked before its first work, each of the device's first calls
    /// that needs an internal waits for it: none returns while the thread is parked, and none
    /// allocates on its own thread. Released, the thread makes them, every call returns, and each
    /// counts one wait; the longest is at least the time the thread stayed parked once a call was
    /// waiting, as a wait's time starts when its call begins to wait.
    #[test]
    fn every_device_s_call_before_the_make_waits_for_the_block_maker_thread() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            let park = Arc::new(Barrier::new(2));
            let options = DeviceOptions {
                maker: MakerTest::Park(Arc::clone(&park)),
                ..DeviceOptions::default()
            };
            let device =
                Device::new(physical, options).unwrap_or_else(|error| panic!("{name}: {error}"));
            let counters = device.counters();
            // Parked before its first work.
            park.wait();
            let mut release = Release::of(&park);
            let core = device.core();
            assert!(
                core.filler.get().is_none() && counters.gauges().staging_bytes == 0,
                "{name}: an internal made before the block-maker thread's first work"
            );
            assert_eq!(core.ring.chunks(), Some((0, 0)), "{name}: a ring block");
            let calls = first_calls(name, &device);
            let what: Vec<&str> = calls.iter().map(|&(what, _)| what).collect();
            let running: Vec<thread::JoinHandle<()>> = (calls.into_iter())
                .map(|(_, call)| thread::spawn(call))
                .collect();
            // The time parked counts from the moment a call first waits: its internal stays asked
            // while the thread is parked, so that call waits past the release. Not every call
            // need be waiting then, as one may wait behind another's.
            within("a call's wait", || core.pool.waiting() > 0);
            let waiting_from = crate::host_time();
            thread::sleep(HOLD);
            for (what, call) in what.iter().zip(&running) {
                assert!(
                    !call.is_finished(),
                    "{name}: {what} returned before the block-maker thread made its internal"
                );
            }
            let parked = counters.metrics();
            let on_caller = allocated(&parked);
            assert!(
                on_caller.is_empty(),
                "{name}: allocated on a caller's thread while parked: {on_caller:?}"
            );
            assert_eq!(parked.maker_waits, 0, "{name}: a wait ended while parked");
            let waiting = core.pool.waiting();
            let parked_nanos = crate::host_time().saturating_sub(waiting_from);
            release.now();
            for call in running {
                call.join().expect("a call");
            }
            let after = counters.metrics();
            let on_caller = allocated(&after);
            assert!(
                on_caller.is_empty(),
                "{name}: allocated on a caller's thread: {on_caller:?}"
            );
            assert_eq!(
                after.maker_waits,
                what.len() as u64,
                "{name}: one wait for each call"
            );
            assert!(
                after.maker_wait_max_nanos >= parked_nanos,
                "{name}: the longest wait, {} ns, under the {parked_nanos} ns the thread stayed \
                 parked once a call waited",
                after.maker_wait_max_nanos
            );
            println!(
                "ahead: {name}: {what:?} waited: maker waits {}, {waiting} waiting at once while \
                 parked, longest {} ns of {parked_nanos} ns parked once a call waited, \
                 consumer-thread sites {:?}",
                after.maker_waits,
                after.maker_wait_max_nanos,
                sites(&after)
            );
            drop(device);
            within("the teardown", || counters.gauges() == Gauges::default());
        }
    }

    /// A device dropped while the block-maker thread is parked before its first work, released
    /// once the drop has stopped it, makes none of the internals; one dropped while the thread
    /// is inside the staging region's make, after the zero source and the ring's first block,
    /// frees all three. Every gauge reads zero after either.
    #[test]
    fn every_device_dropped_before_or_during_the_maker_s_first_work_frees_what_it_made() {
        let instance = crate::tests::validation::instance(&[]);
        let devices = devices_of_the_cfgs_kind(&instance);
        assert!(!devices.is_empty(), "no device of this cfg's kind");
        for (name, physical) in &devices {
            for during in [false, true] {
                let park = Arc::new(Barrier::new(2));
                let options = DeviceOptions {
                    maker: MakerTest::Park(Arc::clone(&park)),
                    ..DeviceOptions::default()
                };
                let device = (Device::new(physical, options))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let counters = device.counters();
                let core = Arc::clone(device.core());
                park.wait();
                let made = counters.metrics().blocks_made;
                // The barrier the thread meets inside the staging region's make, or the first.
                let release = match during {
                    false => park,
                    true => {
                        let region = Arc::new(Barrier::new(2));
                        *lock(&core.park_region) = Some(Arc::clone(&region));
                        park.wait();
                        region.wait();
                        region
                    }
                };
                let before_drop = counters.metrics().blocks_made;
                // Released once the drop has told the thread to stop; the last reference to the
                // core is this thread's, which frees the device as it ends.
                let releaser = thread::spawn(move || {
                    within("the drop's stop", || core.pool.stopped());
                    release.wait();
                    drop(core);
                });
                drop(device);
                releaser.join().expect("the releaser");
                within("the teardown", || counters.gauges() == Gauges::default());
                let blocks_made = counters.metrics().blocks_made;
                match during {
                    false => assert_eq!(
                        blocks_made, made,
                        "{name}: an internal made after the drop's stop"
                    ),
                    // The zero source and the ring's block: the staging region is not a block.
                    true => assert_eq!(
                        before_drop,
                        made + 2,
                        "{name}: the internals made before the staging region"
                    ),
                }
                println!(
                    "ahead: {name}: dropped {} the first work: blocks made {made} -> \
                     {blocks_made}, every gauge zero",
                    if during { "during" } else { "before" }
                );
            }
        }
    }
}
