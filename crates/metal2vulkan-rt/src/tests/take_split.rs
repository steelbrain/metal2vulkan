//! The take split and the byte gauges: under a device cfg, on every device of the cfg's kind, a
//! Shared take longer than the 16 MiB unit is one take of its bytes, one wait for the pool's lock
//! and one make of an allocation of its own on the caller's thread, and adds its bytes to the
//! placed Shared bytes; a buffer zeroed over it is one zero write of its bytes and adds its length
//! to the Shared buffer bytes; a take that fits a free range makes nothing; a placed texture adds
//! its image's bytes to the texture block bytes; and each gauge falls back as its object goes.
//! Each call's total time holds its longest, and the buckets over 1, 5 and 20 ms each hold no more
//! calls than the one below.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, within};
    use crate::{
        MemoryClass, PixelFormat, StorageMode, TextureDescriptor, TextureType, TextureUsage, Timed,
    };

    /// The calls `after` adds to `before`, with their bytes.
    fn added(before: Timed, after: Timed) -> (u64, u64) {
        (after.count - before.count, after.bytes - before.bytes)
    }

    /// A part's counts hold together: its total holds its longest call where it has one, and each
    /// bucket holds no more than the one below it.
    fn whole(name: &str, part: &str, t: Timed) {
        assert!(
            t.count == 0 || t.total_nanos >= t.longest_nanos,
            "{name}: {part}: {t:?}"
        );
        assert!(
            t.count >= t.over_1_ms && t.over_1_ms >= t.over_5_ms && t.over_5_ms >= t.over_20_ms,
            "{name}: {part}: {t:?}"
        );
    }

    #[test]
    fn every_device_times_a_takes_parts_and_gauges_its_bytes() {
        each_device("take_split", |name, _, device| {
            settled(&device);
            let counters = device.counters();
            let len = MemoryClass::Shared.block_len() * 3 / 2;
            let (before, gauged) = (counters.metrics(), counters.gauges());
            let block = (device.take_block(MemoryClass::Shared, len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let taken = counters.metrics();
            assert_eq!(
                added(before.takes, taken.takes),
                (1, block.len),
                "{name}: the take"
            );
            assert_eq!(
                added(before.take_lock_waits, taken.take_lock_waits),
                (1, 0),
                "{name}"
            );
            let (makes, made) = added(before.take_makes, taken.take_makes);
            assert!(
                makes == 1 && made >= len,
                "{name}: the take's own allocation: {makes} of {made} bytes"
            );
            let now = counters.gauges();
            assert_eq!(
                now.shared_placed_bytes - gauged.shared_placed_bytes,
                block.len,
                "{name}: the placed Shared bytes"
            );
            let buffer = (device.new_buffer(block.id, 0, len, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let zeroed = counters.metrics();
            assert_eq!(
                added(taken.zero_writes, zeroed.zero_writes),
                (1, len),
                "{name}: the buffer's zeros"
            );
            assert_eq!(
                counters.gauges().shared_buffer_bytes - gauged.shared_buffer_bytes,
                len,
                "{name}: the Shared buffer bytes"
            );
            drop(buffer);
            assert_eq!(
                counters.gauges().shared_buffer_bytes,
                gauged.shared_buffer_bytes,
                "{name}: the buffer gone"
            );
            drop(block);
            assert_eq!(
                counters.gauges().shared_placed_bytes,
                gauged.shared_placed_bytes,
                "{name}: the block given back"
            );

            // A take of a unit's quarter fits the wholly free unit the device keeps.
            settled(&device);
            let before = counters.metrics();
            let small = (device.take_block(MemoryClass::Shared, len / 6))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let after = counters.metrics();
            assert_eq!(added(before.takes, after.takes), (1, small.len), "{name}");
            assert_eq!(added(before.take_makes, after.take_makes), (0, 0), "{name}");
            drop(small);

            // A placed texture's bytes.
            let gauged = counters.gauges();
            let mut td =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 64, 64);
            td.usage = TextureUsage::RENDER_TARGET;
            td.storage_mode = StorageMode::Private;
            let texture =
                (device.new_texture(&td, None)).unwrap_or_else(|error| panic!("{name}: {error}"));
            let placed = counters.gauges().texture_block_bytes - gauged.texture_block_bytes;
            let own = counters.gauges().texture_own_bytes - gauged.texture_own_bytes;
            assert!(
                (placed >= 64 * 64 * 4 && own == 0) || (placed == 0 && own >= 64 * 64 * 4),
                "{name}: the texture's bytes: placed {placed}, own {own}"
            );
            drop(texture);
            within("the texture freed", || {
                let now = counters.gauges();
                (now.texture_block_bytes, now.texture_own_bytes)
                    == (gauged.texture_block_bytes, gauged.texture_own_bytes)
            });
            let m = counters.metrics();
            for (part, t) in [
                ("takes", m.takes),
                ("lock waits", m.take_lock_waits),
                ("makes", m.take_makes),
                ("maps", m.take_maps),
                ("maker makes", m.maker_makes),
                ("zero writes", m.zero_writes),
            ] {
                whole(name, part, t);
            }
            assert!(
                m.maker_makes.count > 0,
                "{name}: the allocations made ahead"
            );
            println!(
                "{name}: take {:?}; lock {:?}; make {:?}; zero {:?}; texture bytes placed {placed}, \
                 own {own}",
                m.takes, m.take_lock_waits, m.take_makes, m.zero_writes
            );
        });
    }
}
