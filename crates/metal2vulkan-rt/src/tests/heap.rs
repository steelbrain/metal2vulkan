//! Heap textures: under a device cfg, on every device of the cfg's kind, the heap size-and-align
//! answer is the device's image's own size rounded to its buffer-image granularity; textures
//! placed first-fit by that answer in one heap range of a Private block take no allocation and no
//! texture block, and each keeps its own bytes; a texture may alias another; a placement past the
//! heap's end, one off the image's alignment, a heap range past its block and a heap in a Shared
//! block are refused; and every gauge reads zero after teardown. A compute encoder's `use_heap`
//! holds the heap's block for its command buffer once, past the consumer's release, until the
//! command buffer retires, and a released block's number names nothing. A texture for views
//! placed in a heap takes no allocation, and its view writes and reads the heap's bytes. On no
//! device: a heap
//! buffer's size and alignment are Metal's, its length at a multiple of 256.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::texture::TextureObject;
    use crate::{
        DispatchType, Error, Gap, HeapRange, MemoryClass, PixelFormat, Refusal, Region,
        StorageMode, TextureDescriptor, TextureType, TextureUsage,
    };
    use std::sync::Arc;
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
            let block = (device.take_block(MemoryClass::Private, MemoryClass::Private.block_len()))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            core.pool.await_internals();
            wait_until("a wholly free unit in each class", || {
                core.pool.stocked(MemoryClass::Private) && core.pool.stocked(MemoryClass::Shared)
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
                (texture.replace_region(
                    whole,
                    0,
                    0,
                    crate::HostBytes::from(&vec![at as u8 + 1; len][..]),
                    row,
                    0,
                ))
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

    /// A texture for views placed in a heap range of a Private block: its make allocates nothing
    /// (no texture's allocation of its own, no texture block, no block), a view of it reads what
    /// the texture's writes left and the texture what the view's writes left, and the heap rule's
    /// refusals are the indexed texture's: a placement off the image's alignment, and a heap in a
    /// Shared block.
    #[test]
    fn every_device_places_a_texture_for_views_in_a_heap() {
        each_device("heap_views", |name, _, device| {
            let core = device.core();
            let counters = device.counters();
            let block = (device.take_block(MemoryClass::Private, MemoryClass::Private.block_len()))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // The take left the class no wholly free unit: the block-maker thread makes the next,
            // which the gauge read below must not count.
            core.pool.await_internals();
            wait_until("a wholly free unit in each class", || {
                core.pool.stocked(MemoryClass::Private) && core.pool.stocked(MemoryClass::Shared)
            });
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
            let offset = fit.size.next_multiple_of(fit.align);
            let before = (counters.metrics(), counters.gauges().blocks);
            let texture = (device.new_heap_texture_for_views(&d, heap, offset))
                .unwrap_or_else(|error| panic!("{name}: a texture for views in a heap: {error}"));
            let after = (counters.metrics(), counters.gauges().blocks);
            assert_eq!(
                (
                    after.0.texture_allocations,
                    after.0.texture_block_allocations,
                    after.1,
                    core.places.blocks(),
                ),
                (
                    before.0.texture_allocations,
                    before.0.texture_block_allocations,
                    before.1,
                    0
                ),
                "{name}: the make allocates nothing"
            );
            assert_eq!(texture.gpu_resource_id().index, 0, "{name}: no index");

            // Through a view and back.
            let view = (texture.new_texture_view(
                PixelFormat::RGBA8Unorm,
                TextureType::Type2D,
                0..1,
                0..1,
                None,
            ))
            .unwrap_or_else(|error| panic!("{name}: a view: {error}"));
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let len = (SIDE * SIDE * 4) as usize;
            let row = u64::from(SIDE * 4);
            let written: Vec<u8> = (0..len).map(|at| (at % 251) as u8).collect();
            (view.replace_region(whole, 0, 0, crate::HostBytes::from(&written[..]), row, 0))
                .unwrap_or_else(|error| panic!("{name}: a write through the view: {error}"));
            let mut read = vec![0; len];
            (texture.get_bytes(&mut read, row, len as u64, whole, 0, 0))
                .unwrap_or_else(|error| panic!("{name}: a read of the texture: {error}"));
            assert!(
                read == written,
                "{name}: the texture reads the view's write"
            );
            let again: Vec<u8> = written.iter().map(|byte| byte ^ 0xa5).collect();
            (texture.replace_region(whole, 0, 0, crate::HostBytes::from(&again[..]), row, 0))
                .unwrap_or_else(|error| panic!("{name}: a write of the texture: {error}"));
            (view.get_bytes(&mut read, row, len as u64, whole, 0, 0))
                .unwrap_or_else(|error| panic!("{name}: a read through the view: {error}"));
            assert!(read == again, "{name}: the view reads the texture's write");

            // The heap rule's refusals.
            if fit.align > 1 {
                match device.new_heap_texture_for_views(&d, heap, fit.align / 2) {
                    Err(Error::Refused(Refusal::HeapTexturePlace { .. })) => {}
                    Err(error) => panic!("{name}: off the alignment: {error}"),
                    Ok(_) => panic!("{name}: off the alignment: made"),
                }
            }
            let shared = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let in_shared = HeapRange {
                block: shared.id,
                ..heap
            };
            match device.new_heap_texture_for_views(&d, in_shared, 0) {
                Err(Error::Unsupported(Gap::HeapTexture { .. })) => {}
                Err(error) => panic!("{name}: a Shared heap: {error}"),
                Ok(_) => panic!("{name}: a Shared heap's texture was made"),
            }
            println!(
                "heap views: {name}: a texture for views at {offset} in a heap of {} bytes \
                 ({fit:?}): texture allocations {:?}, texture block allocations {:?} at the make; \
                 written and read through a view",
                heap.len, after.0.texture_allocations, after.0.texture_block_allocations
            );
            drop((view, texture));
            for block in [block, shared] {
                device
                    .release_block(block)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }

    #[test]
    fn every_device_holds_a_used_heap_until_its_command_buffer_retires() {
        each_device("use_heap", |name, _, device| {
            let block = (device.take_block(MemoryClass::Private, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let id = block.id;
            let memory = Arc::downgrade(&device.block(id).expect("the claimed block"));
            assert_eq!(
                memory.strong_count(),
                1,
                "{name}: the claim alone holds the block"
            );
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            for _ in 0..2 {
                (encoder.use_heap(id)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            assert_eq!(
                memory.strong_count(),
                2,
                "{name}: the command buffer holds the block once"
            );
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                memory.strong_count(),
                1,
                "{name}: the command buffer holds the block past the consumer's release"
            );
            assert!(
                matches!(encoder.use_heap(id), Err(Error::NoSuchBlock(at)) if at == id),
                "{name}: a released block's number names nothing"
            );
            encoder.end_encoding();
            (command_buffer.commit()).unwrap_or_else(|error| panic!("{name}: {error}"));
            command_buffer
                .wait_until_completed()
                .expect("a wait outside a scheduled handler");
            wait_until("the command buffer to retire", || {
                memory.strong_count() == 0
            });
            assert_eq!(command_buffer.error(), None, "{name}");
            drop((command_buffer, queue));
        });
    }
}

/// `heapBufferSizeAndAlignWithLength:options:` answers the length itself at a multiple of 256,
/// as Metal answers on an Apple M3 Max under macOS 27.0.1 (the conformance recording
/// `device-heap-buffer-sizes`: 1, 15, 255, 256, 257, 511, 4096 and 1 MiB), and past it.
#[test]
fn a_heap_buffer_is_its_length_at_a_multiple_of_256() {
    use crate::buffer::heap_buffer_size_and_align;
    use crate::{SizeAndAlign, MAX_BUFFER_LENGTH};
    for length in [
        0,
        1,
        15,
        255,
        256,
        257,
        511,
        4096,
        1 << 20,
        MAX_BUFFER_LENGTH,
        u64::MAX,
    ] {
        assert_eq!(
            heap_buffer_size_and_align(length),
            SizeAndAlign {
                size: length,
                align: 256
            },
            "{length}"
        );
    }
}
