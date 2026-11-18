//! Managed buffers: changed ranges merge where they overlap or touch; and, under a device cfg, on
//! every device of the cfg's kind, in the form the device's memory-type query picks: with a device
//! copy where Shared memory is not device-local, the copy holds the consumer's first bytes and
//! zeros after them, a changed range reaches it at the next submission and a range not marked
//! changed does not, and changes past the upload's calls submit it; without one, shaders read the
//! consumer's bytes as they are and a changed range submits nothing. A range past the buffer and a
//! Managed buffer over a Private block are refused, and every gauge reads zero after teardown.
//!
//! Managed textures, on every device: the image takes the named bytes of a region at its upload
//! and nothing outside it, `get_bytes` reads the named bytes, `replace_region` writes them and
//! uploads them, and places that do not lie as a copy reads them are refused.

use crate::submit::merge;

#[test]
fn changed_ranges_merge_where_they_overlap_or_touch() {
    let mut ranges = Vec::new();
    merge(&mut ranges, 100, 200);
    merge(&mut ranges, 300, 400);
    assert_eq!(ranges, [(100, 200), (300, 400)]);
    merge(&mut ranges, 0, 50);
    assert_eq!(ranges, [(0, 50), (100, 200), (300, 400)], "before, apart");
    merge(&mut ranges, 200, 250);
    assert_eq!(
        ranges,
        [(0, 50), (100, 250), (300, 400)],
        "touching the end"
    );
    merge(&mut ranges, 290, 300);
    assert_eq!(
        ranges,
        [(0, 50), (100, 250), (290, 400)],
        "touching the start"
    );
    merge(&mut ranges, 120, 130);
    assert_eq!(ranges, [(0, 50), (100, 250), (290, 400)], "inside one");
    merge(&mut ranges, 40, 295);
    assert_eq!(ranges, [(0, 400)], "across three");
    merge(&mut ranges, 500, 600);
    merge(&mut ranges, 0, 1000);
    assert_eq!(ranges, [(0, 1000)], "over all");
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::submit::UPLOAD_ENTRIES;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Buffer, BytesPlace, Device, Error, MemoryClass, PixelFormat, Refusal, Region, StorageMode,
        TextureDescriptor, TextureType, TextureUsage,
    };
    use ash::vk;

    const LEN: u64 = 64 << 10;

    /// The bytes shaders reach at `managed`'s address, copied by the GPU into `into`, a Shared
    /// buffer as long.
    fn device_bytes(device: &Device, managed: &Buffer, into: &Buffer) -> Vec<u8> {
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let (source, from) = managed.device_range();
        let (target, to) = into.device_range();
        command_buffer
            .record_for_test(&[managed, into], |device, command| {
                let region = vk::BufferCopy {
                    src_offset: from,
                    dst_offset: to,
                    size: LEN,
                };
                // SAFETY: the command buffer is recording; both ranges are the buffers', which it
                // keeps.
                unsafe { device.cmd_copy_buffer(command, source, target, &[region]) };
            })
            .expect("a copy");
        command_buffer.commit().expect("a commit");
        finished("the copy", &command_buffer);
        let contents = into.contents().expect("a Shared buffer");
        // SAFETY: the copy into the buffer completed, and its writes reach the host.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), LEN as usize) }.to_vec()
    }

    /// Writes `bytes` at `at` in the consumer's bytes of `managed`.
    fn write(managed: &Buffer, at: u64, bytes: &[u8]) {
        let contents = managed.contents().expect("the consumer's bytes");
        // SAFETY: `at..at + bytes.len()` lies in the buffer, and nothing reads it meanwhile.
        unsafe {
            bytes
                .as_ptr()
                .copy_to_nonoverlapping(contents.as_ptr().add(at as usize), bytes.len())
        };
    }

    #[test]
    fn every_device_copies_a_managed_buffer_s_changed_ranges() {
        each_device("managed", |name, _, device| {
            let core = device.core();
            let copied = !core.shared_is_device_local();
            let counters = device.counters();
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let into = (device.new_buffer(cpu.id, LEN, LEN, true))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let first: Vec<u8> = (0..LEN).map(|i| (i % 251) as u8).collect();
            let start = (device.new_buffer(cpu.id, 0, LEN, false))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            write(&start, 0, &first);
            drop(start);

            // The first half copied, the rest zero.
            let managed = (device.new_managed_buffer(cpu.id, 0, LEN, LEN / 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut expected = first.clone();
            if copied {
                expected[LEN as usize / 2..].fill(0);
            }
            assert!(
                device_bytes(&device, &managed, &into) == expected,
                "{name}: the bytes at creation (device copy: {copied})"
            );

            // Changed ranges, two of them overlapping; and a change not marked.
            for (at, len) in [(1000, 2000), (2000, 3000), (40_000, 1)] {
                write(&managed, at, &vec![0xA5; len as usize]);
                managed
                    .did_modify_range(at, len)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                expected[at as usize..(at + len) as usize].fill(0xA5);
            }
            write(&managed, 20_000, &[0x5A; 100]);
            if !copied {
                expected[20_000..20_100].fill(0x5A);
            }
            assert!(
                device_bytes(&device, &managed, &into) == expected,
                "{name}: the bytes after the changes (device copy: {copied})"
            );

            // Changes past the upload's calls submit it; without a copy nothing is submitted.
            let before = counters.metrics().submissions;
            for change in 0..UPLOAD_ENTRIES as u64 + 8 {
                managed
                    .did_modify_range((change * 8) % LEN, 4)
                    .unwrap_or_else(|error| panic!("{name}: change {change}: {error}"));
            }
            let submitted = counters.metrics().submissions - before;
            match copied {
                true => assert!(submitted >= 1, "{name}: the changes went"),
                false => assert_eq!(submitted, 0, "{name}: nothing to copy"),
            }

            match managed.did_modify_range(LEN - 4, 8) {
                Err(Error::Refused(Refusal::ModifiedRange {
                    start,
                    length: 8,
                    buffer_length: LEN,
                })) if start == LEN - 4 => {}
                other => panic!("{name}: a range past the buffer: {other:?}"),
            }
            let private = (device.take_block(MemoryClass::Private, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            match device.new_managed_buffer(private.id, 0, 256, 0) {
                Err(Error::Refused(Refusal::ManagedBlockNotShared { block })) => {
                    assert_eq!(block, private.id, "{name}")
                }
                Err(error) => panic!("{name}: a Managed buffer over a Private block: {error}"),
                Ok(_) => panic!("{name}: a Managed buffer over a Private block was made"),
            }
            println!(
                "managed: {name}: device copy {copied}; metrics {:?}",
                counters.metrics()
            );
            drop((managed, into));
            for block in [cpu.id, private.id] {
                device
                    .release_block(block)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }

    fn whole(side: u32) -> Region {
        Region {
            origin: [0; 3],
            size: [side, side, 1],
        }
    }

    /// The square RGBA8 level `level` of a texture of `SIDE`, as it lies in `named`: rows `PITCH`
    /// apart at the level's place, packed.
    fn level_bytes(named: &[u8], places: &[BytesPlace], level: u32) -> Vec<u8> {
        let side = (SIDE >> level) as usize;
        let place = places[level as usize];
        (0..side)
            .flat_map(|row| {
                let at = (place.offset + row as u64 * place.bytes_per_row) as usize;
                named[at..at + side * 4].iter().copied()
            })
            .collect()
    }

    /// Writes `value` over `region` of the RGBA8 level at `place` in `named`.
    fn fill_region(named: &mut [u8], place: BytesPlace, region: Region, value: u8) {
        let [x, y, _] = region.origin.map(u64::from);
        let [width, height, _] = region.size.map(u64::from);
        for row in y..y + height {
            let at = (place.offset + row * place.bytes_per_row + x * 4) as usize;
            named[at..at + width as usize * 4].fill(value);
        }
    }

    const SIDE: u32 = 64;
    const PITCH: u64 = 256;
    const NAMED: u64 = 24_576;

    #[test]
    fn every_device_copies_a_managed_texture_s_named_regions() {
        each_device("managed texture", |name, _, device| {
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Level 0's rows packed at 0, level 1's 128-byte rows 256 apart after it.
            let places = [
                BytesPlace {
                    offset: 0,
                    bytes_per_row: PITCH,
                    bytes_per_image: 0,
                },
                BytesPlace {
                    offset: 16_384,
                    bytes_per_row: PITCH,
                    bytes_per_image: 0,
                },
            ];
            let named = (device.new_buffer(cpu.id, 4096, NAMED, true))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut now: Vec<u8> = (0..NAMED).map(|i| (i % 251) as u8).collect();
            write(&named, 0, &now);
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, SIDE, SIDE);
            d.mipmap_level_count = 2;
            d.usage = TextureUsage::RENDER_TARGET;
            let texture = (device.new_managed_texture(&d, None, &named, &places))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(texture.descriptor().storage_mode, StorageMode::Managed);
            let image = |level: u32| {
                let side = SIDE >> level;
                let row = u64::from(side) * 4;
                let mut bytes = vec![0; (row * u64::from(side)) as usize];
                (texture.image_bytes(&mut bytes, row, whole(side), level, 0))
                    .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
                bytes
            };

            // Both levels uploaded whole.
            for level in 0..2 {
                (texture.upload_region(whole(SIDE >> level), level, 0))
                    .unwrap_or_else(|error| panic!("{name}: level {level}: {error}"));
            }
            for level in 0..2 {
                assert!(
                    image(level) == level_bytes(&now, &places, level),
                    "{name}: level {level} after its upload"
                );
            }

            // A region changed and uploaded; a change outside it, not uploaded.
            let mut seen = now.clone();
            let region = Region {
                origin: [8, 4, 0],
                size: [16, 8, 1],
            };
            fill_region(&mut now, places[0], region, 0xA5);
            fill_region(&mut seen, places[0], region, 0xA5);
            let outside = Region {
                origin: [0, 40, 0],
                size: [SIDE, 1, 1],
            };
            fill_region(&mut now, places[0], outside, 0x5A);
            write(&named, 0, &now);
            (texture.upload_region(region, 0, 0)).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(
                image(0) == level_bytes(&seen, &places, 0),
                "{name}: level 0 after the region's upload"
            );
            // get_bytes reads the named bytes, the change not uploaded too.
            let mut read = vec![0; 16_384];
            (texture.get_bytes(&mut read, PITCH, 16_384, whole(SIDE), 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(
                read == level_bytes(&now, &places, 0),
                "{name}: get_bytes reads the named bytes"
            );

            // replace_region writes the named bytes, and the image takes them.
            let small = Region {
                origin: [4, 4, 0],
                size: [4, 4, 1],
            };
            (texture.replace_region(small, 1, 0, &[0x11; 64], 16, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            fill_region(&mut now, places[1], small, 0x11);
            fill_region(&mut seen, places[1], small, 0x11);
            let contents = named.contents().expect("Shared");
            // SAFETY: the buffer's bytes, which nothing writes meanwhile.
            let held = unsafe { std::slice::from_raw_parts(contents.as_ptr(), NAMED as usize) };
            assert!(
                held == now.as_slice(),
                "{name}: replace_region wrote the named bytes"
            );
            assert!(
                image(1) == level_bytes(&seen, &places, 1),
                "{name}: level 1 after replace_region"
            );

            // Places a copy does not read, and a texture without named bytes.
            let refused = |places: &[BytesPlace], bytes: &Buffer, what: &str| -> (u32, u32) {
                match device.new_managed_texture(&d, None, bytes, places) {
                    Err(Error::Refused(Refusal::ManagedTextureBytes { level, slice })) => {
                        (level, slice)
                    }
                    Err(error) => panic!("{name}: {what}: {error}"),
                    Ok(_) => panic!("{name}: {what}: made"),
                }
            };
            let with = |offset: u64, bytes_per_row: u64| {
                let mut wrong = places;
                wrong[1] = BytesPlace {
                    offset,
                    bytes_per_row,
                    bytes_per_image: 0,
                };
                wrong
            };
            assert_eq!(refused(&places[..1], &named, "one place"), (2, 0));
            assert_eq!(refused(&with(16_386, PITCH)[..], &named, "offset"), (1, 0));
            assert_eq!(refused(&with(16_384, 130)[..], &named, "pitch"), (1, 0));
            assert_eq!(refused(&with(16_384, 64)[..], &named, "short rows"), (1, 0));
            assert_eq!(
                refused(&with(NAMED - 4096, PITCH)[..], &named, "past"),
                (1, 0)
            );
            let private = (device.take_block(MemoryClass::Private, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let unmapped = (device.new_buffer(private.id, 0, NAMED, false))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(refused(&places, &unmapped, "Private"), (0, 0));
            d.storage_mode = StorageMode::Managed;
            let plain =
                (device.new_texture(&d, None)).unwrap_or_else(|error| panic!("{name}: {error}"));
            match plain.upload_region(whole(SIDE), 0, 0) {
                Err(Error::Refused(Refusal::ManagedTextureBytes { level: 0, slice: 0 })) => {}
                Err(error) => panic!("{name}: an upload without named bytes: {error}"),
                Ok(()) => panic!("{name}: an upload without named bytes went"),
            }
            match texture.upload_region(whole(1), 2, 0) {
                Err(Error::Refused(Refusal::TextureRegion { level: 2, slice: 0 })) => {}
                Err(error) => panic!("{name}: a level past the texture: {error}"),
                Ok(()) => panic!("{name}: a level past the texture went"),
            }
            drop((texture, plain, named, unmapped));
            for block in [cpu.id, private.id] {
                device
                    .release_block(block)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
        });
    }
}
