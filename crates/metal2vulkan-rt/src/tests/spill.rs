//! The Private spill: under a device cfg, on every device of the cfg's kind, with the Private
//! type's allocations made to fail as out of device memory (the tests' stand-in for a full
//! device): a Private block goes to the next memory type its buffer takes, reports itself not
//! device-local and counts its bytes as spilled until it is freed; where the buffer takes no other
//! type, the take fails with the allocation's error. While the pool's Private blocks spill, a
//! texture takes no texture block (an allocation of its own, itself spilled where the image takes
//! another type, else the allocation's error), and Metal's two-copy rule holds for a Managed
//! buffer: where the device is not unified, its device copy, which the consumer places in a Private
//! block it takes while the pool's Private blocks spill, is in another memory type, its bytes
//! counted in `Gauges::managed_spilled_bytes` while it lives. Every gauge reads zero after
//! teardown.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::compute::on_device::WORDS;
    use crate::tests::managed::on_device::two_copy_rule;
    use crate::{MemoryClass, PixelFormat, TextureDescriptor, TextureType, TextureUsage};
    use std::sync::atomic::Ordering::Relaxed;
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
    fn every_device_spills_private_memory_out_of_device_memory() {
        each_device("spill", |name, _, device| {
            let core = device.core();
            let counters = device.counters();
            wait_until("a wholly free unit in each class", || {
                core.pool.stocked(MemoryClass::Private) && core.pool.stocked(MemoryClass::Shared)
            });

            // A Private block longer than an allocation, which makes its own in the call, whose
            // device-local allocation fails.
            core.fail_private.store(1, Relaxed);
            let spilled =
                device.take_block(MemoryClass::Private, MemoryClass::Private.block_len() + 1);
            assert_eq!(
                core.fail_private.load(Relaxed),
                0,
                "{name}: the failure ran"
            );
            let spills = match spilled {
                Ok(block) => {
                    assert!(!block.device_local, "{name}: a spilled block");
                    assert_eq!(counters.gauges().spilled_bytes, block.len, "{name}");
                    (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
                    wait_until("the spilled block freed", || {
                        counters.gauges().spilled_bytes == 0
                    });
                    true
                }
                Err(error) if error.is_out_of_device_memory() => false,
                Err(error) => panic!("{name}: a spilled block: {error}"),
            };
            println!("spill: {name}: a Private block spills: {spills}");
            if !spills {
                return;
            }

            // Every Private-type allocation fails from here: the first allocation, made before, is
            // device-local; the next, which the block-maker thread makes once a block takes the
            // first whole, spills.
            core.fail_private.store(u32::MAX, Relaxed);
            let plain = (device.take_block(MemoryClass::Private, MemoryClass::Private.block_len()))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(plain.device_local, "{name}: the first allocation");
            wait_until("the pool's spilled allocation", || {
                core.pool.stocked(MemoryClass::Private)
                    && counters.gauges().spilled_bytes >= MemoryClass::Private.block_len()
            });

            // A texture takes no texture block in it.
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 512, 512);
            d.usage = TextureUsage::RENDER_TARGET;
            let texture = match device.new_texture_for_views(&d) {
                Ok(texture) => Some(texture),
                Err(error) if error.is_out_of_device_memory() => None,
                Err(error) => panic!("{name}: a texture: {error}"),
            };
            assert_eq!(core.places.blocks(), 0, "{name}: no texture block");

            // A Managed buffer keeps two copies where the device is not unified, its device copy
            // spilled and counted.
            let spilled = if device.has_unified_memory() {
                0
            } else {
                (WORDS * 4) as u64
            };
            two_copy_rule(name, &device, |_| {
                assert_eq!(
                    counters.gauges().managed_spilled_bytes,
                    spilled,
                    "{name}: the Managed buffer's spilled bytes"
                );
            });
            wait_until("the Managed buffer's device copy freed", || {
                counters.gauges().managed_spilled_bytes == 0
            });
            core.fail_private.store(0, Relaxed);
            println!(
                "spill: {name}: a texture of its own while blocks spill: {}; gauges {:?}",
                texture.is_some(),
                counters.gauges()
            );
            drop(texture);
            device
                .release_block(plain)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }
}
