//! Heap textures: under a device cfg, on every device of the cfg's kind, the heap size-and-align
//! answer is the device's image's own size rounded to its buffer-image granularity; textures
//! placed first-fit by that answer in one heap range of a Private block take no allocation and no
//! texture block, and each keeps its own bytes; a texture may alias another; a placement past the
//! heap's end, one off the image's alignment, a heap range past its block and a heap in a Shared
//! block are refused; and every gauge reads zero after teardown.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::{
        Error, Gap, HeapRange, MemoryClass, PixelFormat, Refusal, Region, StorageMode,
        TextureDescriptor, TextureType, TextureUsage,
    };
    use std::thread;
    use std::time::{Duration, Instant};

    const SIDE: u32 = 256;
    const MIB: u64 = 1 << 20;

    fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(120);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn every_device_places_heap_textures_at_the_consumer_s_offsets() {
        each_device("heap", |name, _, device| {
            let core = device.core();
            let counters = device.counters();
            let block = (device.take_block(MemoryClass::Private, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            wait_until("a ready block of each class", || {
                core.pool.idle(MemoryClass::Private) == 1
                    && core.pool.idle(MemoryClass::Shared) == 1
            });
            let blocks = counters.gauges().blocks;
            let heap = HeapRange {
                block: block.id,
                offset: MIB,
                len: 4 * MIB,
            };
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, SIDE, SIDE);
            d.usage = TextureUsage::RENDER_TARGET;
            d.storage_mode = StorageMode::Shared;
            let fit = (device.heap_texture_size_and_align(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let granularity = core.granularity.max(1);
            assert!(
                fit.align.is_power_of_two()
                    && fit.align.is_multiple_of(granularity)
                    && fit.size.is_multiple_of(granularity)
                    && fit.size >= u64::from(SIDE * SIDE * 4),
                "{name}: {fit:?}, granularity {granularity}"
            );

            // First fit by the answer, as the consumer's allocator places them.
            let mut textures = Vec::new();
            let mut offset = 0;
            while offset + fit.size <= heap.len {
                let texture = (device.new_heap_texture(&d, None, heap, offset))
                    .unwrap_or_else(|error| panic!("{name}: at {offset}: {error}"));
                assert_eq!(
                    texture.image_size().next_multiple_of(granularity),
                    fit.size,
                    "{name}: the answer is the image's own size"
                );
                textures.push(texture);
                offset = (offset + fit.size).next_multiple_of(fit.align);
            }
            assert!(textures.len() >= 2, "{name}: {fit:?} in {} bytes", heap.len);
            assert_eq!(
                (counters.gauges().blocks, core.places.blocks()),
                (blocks, 0),
                "{name}: heap textures allocate nothing"
            );

            // Each keeps its own bytes.
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let len = (SIDE * SIDE * 4) as usize;
            let row = u64::from(SIDE * 4);
            for (at, texture) in textures.iter().enumerate() {
                (texture.replace_region(whole, 0, 0, &vec![at as u8 + 1; len], row, 0))
                    .unwrap_or_else(|error| panic!("{name}: texture {at}: {error}"));
            }
            for (at, texture) in textures.iter().enumerate() {
                let mut read = vec![0; len];
                (texture.get_bytes(&mut read, row, len as u64, whole, 0, 0))
                    .unwrap_or_else(|error| panic!("{name}: texture {at}: {error}"));
                assert!(
                    read.iter().all(|&byte| byte == at as u8 + 1),
                    "{name}: texture {at} kept its bytes"
                );
            }

            // Aliasing, as a Metal heap's aliasable resources do.
            let alias = (device.new_heap_texture(&d, None, heap, 0))
                .unwrap_or_else(|error| panic!("{name}: an alias: {error}"));

            // Placements that do not lie in the heap.
            let refused = |heap: HeapRange, offset: u64, what: &str| match device
                .new_heap_texture(&d, None, heap, offset)
            {
                Err(Error::Refused(Refusal::HeapTexturePlace {
                    offset: at,
                    size,
                    align,
                    heap_len,
                })) => assert_eq!(
                    (at, size, align, heap_len),
                    (offset, fit.size, fit.align, heap.len),
                    "{name}: {what}"
                ),
                Err(error) => panic!("{name}: {what}: {error}"),
                Ok(_) => panic!("{name}: {what}: made"),
            };
            let past = (heap.len - fit.size + 1).next_multiple_of(fit.align);
            refused(heap, past, "past the heap's end");
            if fit.align > 1 {
                refused(heap, fit.align / 2, "off the alignment");
            }
            let outside = HeapRange {
                offset: block.len - MIB,
                ..heap
            };
            refused(outside, 0, "a heap past its block");
            let shared = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let in_shared = HeapRange {
                block: shared.id,
                ..heap
            };
            match device.new_heap_texture(&d, None, in_shared, 0) {
                Err(Error::Unsupported(Gap::HeapTexture { pixel_format })) => {
                    assert_eq!(pixel_format, PixelFormat::RGBA8Unorm, "{name}")
                }
                Err(error) => panic!("{name}: a Shared heap: {error}"),
                Ok(_) => panic!("{name}: a Shared heap's texture was made"),
            }
            println!(
                "heap: {name}: {} textures of {fit:?} in {} bytes, granularity {granularity}",
                textures.len(),
                heap.len
            );
            drop((textures, alias));
            for block in [block, shared] {
                device
                    .release_block(block)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }
}
