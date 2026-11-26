//! The per-site allocation counters (`Metrics::texture_allocations` and the other sites'): under a
//! device cfg, on every device of the cfg's kind, each site's counter moves, by one call of at
//! least the site's bytes, when the site allocates on the consumer's thread, and no other site's
//! moves with it; a call that reuses what a site made moves none. The device's block-maker thread
//! fails every make of the runtime's internals (`MakerTest::Fail`), so the calls that need them
//! make them, at their sites, as they do where its makes fail.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::fill::ZEROS;
    use crate::memory::MakerTest;
    use crate::ring::{Writes, CHUNK};
    use crate::tests::command::on_device::{each_device_in_both_forms_with, within};
    use crate::{
        Device, DeviceOptions, HostBytes, MemoryClass, Metrics, PixelFormat, Region,
        SiteAllocations, StorageMode, Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use std::sync::atomic::Ordering::Relaxed;

    /// Each site's counter in `metrics`, by name.
    fn sites(metrics: &Metrics) -> [(&'static str, SiteAllocations); 8] {
        [
            ("texture", metrics.texture_allocations),
            ("texture block", metrics.texture_block_allocations),
            ("ring block", metrics.ring_block_allocations),
            ("staging region", metrics.staging_region_allocations),
            ("staging buffer", metrics.staging_buffer_allocations),
            ("zero source", metrics.zero_source_allocations),
            ("shared take", metrics.shared_take_allocations),
            ("private take", metrics.private_take_allocations),
        ]
    }

    /// The sites whose allocations moved from `before` to `after`, each with the calls and bytes
    /// it added and its longest call after; a longest call never falls. A take that landed in a
    /// unit made ahead allocated nothing, so it moves no site here.
    fn moved(before: &Metrics, after: &Metrics) -> Vec<(&'static str, u64, u64, u64)> {
        (sites(before).into_iter())
            .zip(sites(after))
            .inspect(|((name, was), (_, now))| {
                assert!(
                    now.longest_nanos >= was.longest_nanos,
                    "{name}: a longest call fell"
                );
            })
            .filter(|((_, was), (_, now))| {
                (was.count, was.bytes, was.longest_nanos)
                    != (now.count, now.bytes, now.longest_nanos)
            })
            .map(|((name, was), (_, now))| {
                (
                    name,
                    now.count - was.count,
                    now.bytes - was.bytes,
                    now.longest_nanos,
                )
            })
            .collect()
    }

    /// Runs `run` on `device` and checks that the sites `expected` moved, each by one call of at
    /// least its bytes, and no other; prints what moved.
    fn step(name: &str, device: &Device, what: &str, expected: &[(&str, u64)], run: impl FnOnce()) {
        let before = device.counters().metrics();
        run();
        let moved = moved(&before, &device.counters().metrics());
        let names: Vec<&str> = moved.iter().map(|&(site, ..)| site).collect();
        let wanted: Vec<&str> = expected.iter().map(|&(site, _)| site).collect();
        assert_eq!(names, wanted, "{name}: {what}: the sites that moved");
        for (&(site, count, bytes, longest), &(_, least)) in moved.iter().zip(expected) {
            assert_eq!(count, 1, "{name}: {what}: {site}: one call");
            assert!(
                bytes >= least,
                "{name}: {what}: {site}: {bytes} bytes of {least}"
            );
            assert!(longest > 0, "{name}: {what}: {site}: timed");
        }
        println!("sites: {name}: {what}: moved {moved:?}");
    }

    /// A Shared RGBA8 texture of `width` by `height`, rendered to and copied.
    fn texture(name: &str, device: &Device, width: u32, height: u32) -> Texture {
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, width, height);
        d.usage = TextureUsage::RENDER_TARGET;
        d.storage_mode = StorageMode::Shared;
        (device.new_texture_for_views(&d))
            .unwrap_or_else(|error| panic!("{name}: a texture: {error}"))
    }

    /// Writes the whole of `texture`, of `width` by `height`, from the host.
    fn replace(name: &str, texture: &Texture, width: u32, height: u32) {
        let row = u64::from(width) * 4;
        let bytes = vec![0x5au8; (row * u64::from(height)) as usize];
        let whole = Region {
            origin: [0; 3],
            size: [width, height, 1],
        };
        (texture.replace_region(whole, 0, 0, HostBytes::from(&bytes[..]), row, 0))
            .unwrap_or_else(|error| panic!("{name}: a replace: {error}"));
    }

    #[test]
    fn every_device_counts_each_allocation_site_apart() {
        let options = || DeviceOptions {
            maker: MakerTest::Fail,
            ..DeviceOptions::default()
        };
        each_device_in_both_forms_with("allocation_sites", options, |name, _, device| {
            count_each_site(name, &device);
        });
    }

    /// Every site, on `device`, whose block-maker thread makes none of the internals.
    fn count_each_site(name: &str, device: &Device) {
        let core = device.core();
        const MIB: u64 = 1 << 20;

        // The device's making allocates at no site.
        let made = device.counters().metrics();
        let at_making: Vec<(&str, SiteAllocations)> = (sites(&made).into_iter())
            .filter(|(_, site)| *site != SiteAllocations::default())
            .collect();
        assert!(
            at_making.is_empty(),
            "{name}: the making's sites: {at_making:?}"
        );

        // The first fill that copies from the zero source makes it; the next reuses it.
        let private = (device.take_block(MemoryClass::Private, 4096))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        step(
            name,
            device,
            "the first fill beside whole words",
            &[("zero source", ZEROS)],
            || {
                (device.zero_block_range(private.id, 1, 8))
                    .unwrap_or_else(|error| panic!("{name}: a fill: {error}"));
            },
        );
        step(
            name,
            device,
            "a second fill beside whole words",
            &[],
            || {
                (device.zero_block_range(private.id, 9, 8))
                    .unwrap_or_else(|error| panic!("{name}: a fill: {error}"));
            },
        );

        // A texture of 24 MiB, past every allocation of the pool, whose texture block's
        // device-local allocation fails: an allocation of its own. A spill of the block into
        // another type the device offers is the texture block's. The failure is set once the
        // block-maker thread holds a wholly free unit of each class, so no unit it makes takes it.
        within("a wholly free unit in each class", || {
            core.pool.stocked(MemoryClass::Private) && core.pool.stocked(MemoryClass::Shared)
        });
        core.fail_private.store(1, Relaxed);
        let before = device.counters().metrics();
        let own = texture(name, device, 2048, 3072);
        assert_eq!(
            core.fail_private.load(Relaxed),
            0,
            "{name}: the failure ran"
        );
        let moved = moved(&before, &device.counters().metrics());
        let texture_moved = moved.iter().find(|&&(site, ..)| site == "texture");
        assert!(
            texture_moved.is_some_and(|&(_, count, bytes, longest)| count == 1
                && bytes >= 24 * MIB
                && longest > 0),
            "{name}: a texture of its own: {moved:?}"
        );
        assert!(
            (moved.iter()).all(|&(site, ..)| site == "texture" || site == "texture block"),
            "{name}: a texture of its own: {moved:?}"
        );
        println!("sites: {name}: a texture of its own: moved {moved:?}");

        // A texture of 32 MiB, longer than any free range: its texture block's take
        // allocates.
        let mut large = None;
        step(
            name,
            device,
            "a 32 MiB texture",
            &[("texture block", 32 * MIB)],
            || {
                large = Some(texture(name, device, 2048, 4096));
            },
        );
        let large = large.expect("the 32 MiB texture");

        // A texture whose texture block has room allocates nothing.
        let small = texture(name, device, 64, 64);
        let mut placed = None;
        step(name, device, "a texture in a block with room", &[], || {
            placed = Some(texture(name, device, 64, 64));
        });
        let placed = placed.expect("the second small texture");

        // The device's first copy of 16 MiB or less makes the staging region; the next reuses
        // it.
        step(
            name,
            device,
            "the first small replace",
            &[("staging region", 16 * MIB)],
            || {
                replace(name, &small, 64, 64);
            },
        );
        step(name, device, "a second small replace", &[], || {
            replace(name, &placed, 64, 64);
        });

        // A copy larger than the region has a staging buffer of its own.
        step(
            name,
            device,
            "a 32 MiB replace",
            &[("staging buffer", 32 * MIB)],
            || {
                replace(name, &large, 2048, 4096);
            },
        );

        // A ring write with no chunk free makes a ring block; one after the chunks are given
        // back takes one of them.
        let free = core.ring.chunks().expect("the ring's lock is free").1;
        let bytes = vec![7u8; CHUNK as usize];
        let mut writes = Writes::default();
        step(
            name,
            device,
            "a ring write past its chunks",
            &[("ring block", 2 * MIB)],
            || {
                for _ in 0..=free {
                    writes.write(core, &bytes).expect("a ring write");
                }
            },
        );
        writes.give_back(core);
        let mut again = Writes::default();
        step(
            name,
            device,
            "a ring write into a chunk given back",
            &[],
            || {
                again.write(core, &bytes).expect("a ring write");
            },
        );
        again.give_back(core);

        // A take longer than any free range allocates; a short one fits a free range.
        let mut taken = Vec::new();
        step(
            name,
            device,
            "a 64 MiB take",
            &[("shared take", 64 * MIB)],
            || {
                taken.push(
                    device
                        .take_block(MemoryClass::Shared, 64 * MIB)
                        .expect("a take"),
                );
            },
        );
        step(name, device, "a 4 KiB take", &[], || {
            taken.push(
                device
                    .take_block(MemoryClass::Shared, 4096)
                    .expect("a take"),
            );
        });

        // A Managed buffer's first named range allocates nothing, with a device copy or without:
        // the commit copies it from the consumer's block.
        let two = !device.has_unified_memory();
        let copy = two.then(|| {
            (device.take_block(MemoryClass::Private, 4096 + 4096))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
        });
        let bytes_block = (device.take_block(MemoryClass::Shared, 4096))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let at = copy.as_ref().map(|copy| (copy.id, 4096));
        let managed = (device.new_managed_buffer((bytes_block.id, 0), at, 4096, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        step(
            name,
            device,
            "a Managed buffer's first named range",
            &[],
            || {
                (managed.did_modify_range(0, 4096))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            },
        );

        drop((managed, own, large, small, placed));
        for block in taken.into_iter().chain(copy).chain([bytes_block, private]) {
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        }
        println!(
            "sites: {name}: every site counted apart (a device copy of Managed buffers: {two})"
        );
    }
}
