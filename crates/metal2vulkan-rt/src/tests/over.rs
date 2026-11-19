//! Textures over a buffer: under a device cfg, on every device of the cfg's kind, the linear
//! alignment answer is a power of two from the format's texel bytes to 4,096, the same when asked
//! again, and the count of formats that answer 16 is printed; at each width tried, a texture over a Shared buffer whose
//! pitch is its row rounded up to the answer, at a place of the linear image's memory alignment,
//! is in place; an RGBA8Unorm texture over a Shared buffer at its row rounded up to the answer
//! (longer than its texels where the device pads rows), at such a place, is made in place: it allocates nothing (live
//! allocations counted once the block-maker thread holds a ready block of each class, the count
//! that a plain texture's allocation moves) and stages nothing, its image's bytes are the host's
//! in the buffer as they change, `replace_region` and `get_bytes` write and read the
//! buffer, and a blit's write to the texture reaches the host in the buffer. Over a Managed buffer
//! the image is the bytes shaders reach, which `replace_region` changes. Rows past the buffer,
//! closer than a row or off a texel, and an offset off a texel, are refused; a 3D texture or one
//! of two levels over a buffer is the gap; and every gauge reads zero after teardown. A render
//! target over a buffer whose subresource the device advertises past its own rows is copied (on
//! lavapipe one of 3 rows; one of 4 stays in place), its copies counted, and a pass that clears
//! it leaves every byte but its texels as the host wrote them; a sampled texture over the same
//! buffer stays in place.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::Core;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::texture::TextureObject;
    use crate::{
        Buffer, ClearColor, Error, Gap, IndexClass, LoadAction, MemoryClass, PixelFormat, Refusal,
        Region, RenderPassDescriptor, StorageMode, StoreAction, Texture, TextureDescriptor,
        TexturePlace, TextureType, TextureUsage,
    };
    use std::thread;
    use std::time::{Duration, Instant};

    const WIDTH: u32 = 100;
    const HEIGHT: u32 = 10;
    const ROW: u64 = WIDTH as u64 * 4;
    /// The buffer's offset in its block: a multiple of every answer and memory alignment.
    const AT: u64 = 65_536;
    const LEN: u64 = 65_536;
    /// The formats whose answers are printed, each with its texel bytes.
    const FORMATS: [(PixelFormat, u64); 5] = [
        (PixelFormat::R8Unorm, 1),
        (PixelFormat::RGBA8Unorm, 4),
        (PixelFormat::R32Float, 4),
        (PixelFormat::RGBA16Float, 8),
        (PixelFormat::RGBA32Float, 16),
    ];

    fn whole() -> Region {
        Region {
            origin: [0; 3],
            size: [WIDTH, HEIGHT, 1],
        }
    }

    /// The buffer's bytes.
    fn held(buffer: &Buffer) -> &[u8] {
        let contents = buffer.contents().expect("Shared");
        // SAFETY: the buffer's bytes, which no command writes while the test reads them.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }
    }

    /// Writes `bytes` over the buffer's.
    fn write(buffer: &Buffer, bytes: &[u8]) {
        let contents = buffer.contents().expect("Shared");
        // SAFETY: the buffer holds `bytes.len()` bytes, and no command reads them meanwhile.
        unsafe {
            contents
                .as_ptr()
                .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len())
        };
    }

    /// The texture's rows as they lie in `bytes`, `pitch` apart from `offset`, packed.
    fn rows(bytes: &[u8], offset: u64, pitch: u64) -> Vec<u8> {
        (0..u64::from(HEIGHT))
            .flat_map(|row| {
                let at = (offset + row * pitch) as usize;
                bytes[at..at + ROW as usize].iter().copied()
            })
            .collect()
    }

    /// The device's live allocations once the block-maker thread has nothing to make (8 MiB free
    /// in each class), and the texture blocks among them.
    fn settled(name: &str, core: &Core) -> (u64, usize) {
        let deadline = Instant::now() + Duration::from_secs(120);
        while !core.pool.stocked(MemoryClass::Private) || !core.pool.stocked(MemoryClass::Shared) {
            assert!(
                Instant::now() < deadline,
                "{name}: 8 MiB free in each class"
            );
            thread::sleep(Duration::from_millis(1));
        }
        (core.counters.gauges().blocks, core.places.blocks())
    }

    /// The texture's image's bytes, read by the device.
    fn image(name: &str, texture: &Texture) -> Vec<u8> {
        let mut bytes = vec![0; (ROW * u64::from(HEIGHT)) as usize];
        (texture.image_bytes(&mut bytes, ROW, whole(), 0, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        bytes
    }

    #[test]
    fn every_device_holds_a_texture_over_a_buffer_in_place() {
        each_device("over", |name, _, device| {
            let mut answers = Vec::new();
            for (pf, texel) in FORMATS {
                let answer = (device.minimum_linear_texture_alignment(pf))
                    .unwrap_or_else(|error| panic!("{name}: {pf:?}: {error}"));
                assert!(
                    answer.is_power_of_two() && (texel..=4096).contains(&answer),
                    "{name}: {pf:?}: {answer}"
                );
                answers.push((pf, answer));
            }
            let sixteen = answers.iter().filter(|&&(_, answer)| answer == 16).count();
            println!(
                "over: {name}: minimum linear texture alignments {answers:?}; {sixteen} of {} \
                 formats answer 16",
                FORMATS.len()
            );
            let align = answers[1].1;
            let again = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(again, align, "{name}: asked again");
            // The image's memory alignment, which the answer leaves out: a place of it.
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            d.usage = TextureUsage::SHADER_READ;
            let memory = (device.linear_memory_alignment(&d)).expect("a linear image");
            let offset = align.next_multiple_of(memory);

            let counters = device.counters();
            let core = device.core();
            let cpu = (device.take_block(MemoryClass::Shared, 2 * AT + LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = (device.new_buffer(cpu.id, AT, LEN, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut now: Vec<u8> = (0..LEN).map(|i| (i % 253) as u8).collect();
            write(&buffer, &now);
            // At each width tried, the row rounded up to the answer is the device's own pitch.
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 1)).expect("a grant");
            for width in [1, 3, 17, 64, 65, 100] {
                let row = u64::from(width) * 4;
                let mut w = d;
                w.width = width;
                let memory = (device.linear_memory_alignment(&w)).expect("a linear image");
                let at = align.next_multiple_of(memory);
                let pitch = row.next_multiple_of(align);
                let texture = (buffer.new_texture(&w, Some(first), at, pitch))
                    .unwrap_or_else(|error| panic!("{name}: width {width}: {error}"));
                assert!(
                    texture.copied_over().is_none(),
                    "{name}: width {width} at the answer's pitch is in place"
                );
            }
            let pitch = ROW.next_multiple_of(align);
            // Read and written by shaders, at indices of a grant.
            (device.grant_indices(IndexClass::Texture, 2, 4)).expect("a grant");
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.storage_mode = StorageMode::Private;
            // Counted with the block-maker thread settled on both sides, so a refill of the
            // ready block the take above used is not the texture's, and a block the texture took
            // from the pool shows as the refill it causes.
            let (blocks, copied) = (settled(name, core), counters.metrics().bytes_copied);
            assert_eq!(blocks.1, 0, "{name}: no texture block yet");
            let texture = (buffer.new_texture(&d, Some(1), offset, pitch))
                .unwrap_or_else(|error| panic!("{name}: at {offset}, pitch {pitch}: {error}"));
            assert_eq!(texture.descriptor().storage_mode, StorageMode::Shared);
            assert_eq!(
                settled(name, core),
                blocks,
                "{name}: a texture over a buffer allocates nothing"
            );

            // The image is the buffer's bytes, as they change.
            assert!(
                image(name, &texture) == rows(&now, offset, pitch),
                "{name}: the image at creation"
            );
            now.iter_mut().for_each(|byte| *byte = byte.wrapping_mul(7));
            write(&buffer, &now);
            assert!(
                image(name, &texture) == rows(&now, offset, pitch),
                "{name}: the image after the host's write"
            );

            // replace_region and get_bytes write and read the buffer, staging nothing.
            let small = Region {
                origin: [3, 2, 0],
                size: [5, 4, 1],
            };
            let before = counters.metrics().bytes_copied;
            (texture.replace_region(small, 0, 0, crate::HostBytes::from(&[0xA5; 80][..]), 20, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for row in 2..6 {
                let at = (offset + row * pitch + 12) as usize;
                now[at..at + 20].fill(0xA5);
            }
            assert!(held(&buffer) == now.as_slice(), "{name}: replace_region");
            let mut read = vec![0; (ROW * u64::from(HEIGHT)) as usize];
            (texture.get_bytes(&mut read, ROW, 0, whole(), 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(read == rows(&now, offset, pitch), "{name}: get_bytes");
            assert_eq!(
                counters.metrics().bytes_copied,
                before,
                "{name}: nothing staged"
            );
            assert!(
                image(name, &texture) == rows(&now, offset, pitch),
                "{name}: the image after replace_region"
            );

            // A blit's write to the texture reaches the host in the buffer.
            let mut plain_d = d;
            plain_d.usage = TextureUsage::RENDER_TARGET;
            plain_d.storage_mode = StorageMode::Shared;
            let plain = (device.new_texture(&plain_d, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // The count sees a texture's allocation: a plain texture's.
            assert_ne!(
                settled(name, core),
                blocks,
                "{name}: a plain texture allocates"
            );
            let source: Vec<u8> = (0..ROW * u64::from(HEIGHT))
                .map(|i| (i % 241) as u8 ^ 0x3C)
                .collect();
            (plain.replace_region(whole(), 0, 0, crate::HostBytes::from(&source[..]), ROW, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let reaches = counters.host_reaches();
            let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
            encoder
                .copy_from_texture(
                    &plain,
                    TexturePlace::default(),
                    [WIDTH, HEIGHT, 1],
                    &texture,
                    TexturePlace::default(),
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            assert!(
                counters.host_reaches() > reaches,
                "{name}: the blit's segment reaches the host"
            );
            for row in 0..u64::from(HEIGHT) {
                let at = (offset + row * pitch) as usize;
                let from = (row * ROW) as usize;
                now[at..at + ROW as usize].copy_from_slice(&source[from..from + ROW as usize]);
            }
            assert!(held(&buffer) == now.as_slice(), "{name}: the blit's write");
            assert_eq!(
                counters.metrics().bytes_copied - copied,
                // The plain texture's upload, and each read of the image above.
                ROW * u64::from(HEIGHT) * 4,
                "{name}: nothing else copied"
            );

            // A render target over the buffer, which a device may not hold in place.
            let mut r = d;
            r.usage = TextureUsage::RENDER_TARGET;
            let target = buffer.new_texture(&r, None, offset, pitch).map(|_| ());
            println!("over: {name}: a render target over a buffer: {target:?}");

            // What Metal's layout rules refuse, and what is not held in place.
            let d = r;
            let refused = |d: &TextureDescriptor, offset: u64, pitch: u64, what: &str| match buffer
                .new_texture(d, None, offset, pitch)
            {
                Err(Error::Refused(Refusal::BufferTextureLayout {
                    offset: o,
                    bytes_per_row: p,
                    len,
                })) => assert_eq!((o, p, len), (offset, pitch, LEN), "{name}: {what}"),
                Err(error) => panic!("{name}: {what}: {error}"),
                Ok(_) => panic!("{name}: {what}: made"),
            };
            refused(&d, LEN - pitch, pitch, "rows past the buffer");
            refused(&d, offset, ROW - 4, "rows closer than a row");
            refused(&d, offset, pitch + 2, "rows off a texel");
            refused(&d, offset + 2, pitch, "an offset off a texel");
            let gap = |d: &TextureDescriptor, what: &str| match buffer
                .new_texture(d, None, offset, pitch)
            {
                Err(Error::Unsupported(Gap::TextureOverBuffer { pixel_format })) => {
                    assert_eq!(pixel_format, d.pixel_format, "{name}: {what}")
                }
                Err(error) => panic!("{name}: {what}: {error}"),
                Ok(_) => panic!("{name}: {what}: made"),
            };
            let mut deep = d;
            deep.texture_type = TextureType::Type3D;
            deep.depth = 2;
            gap(&deep, "a 3D texture");
            let mut levels = d;
            levels.mipmap_level_count = 2;
            gap(&levels, "two levels");

            // Over a Managed buffer the image lies in the bytes shaders reach: its device copy,
            // where it has one, which replace_region's rows reach as did_modify_range's do.
            let managed =
                (crate::tests::managed::on_device::new_managed(&device, cpu.id, 2 * AT, LEN, LEN))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut bytes: Vec<u8> = (0..LEN).map(|i| (i % 239) as u8).collect();
            write(&managed, &bytes);
            (managed.did_modify_range(0, LEN)).unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut m = d;
            m.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let over = (managed.new_texture(&m, Some(2), offset, pitch))
                .unwrap_or_else(|error| panic!("{name}: over a Managed buffer: {error}"));
            assert_ne!(over.descriptor().storage_mode, StorageMode::Private);
            assert!(
                image(name, &over) == rows(&bytes, offset, pitch),
                "{name}: the image over a Managed buffer"
            );
            (over.replace_region(small, 0, 0, crate::HostBytes::from(&[0x5A; 80][..]), 20, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for row in 2..6 {
                let at = (offset + row * pitch + 12) as usize;
                bytes[at..at + 20].fill(0x5A);
            }
            assert!(
                held(&managed) == bytes.as_slice(),
                "{name}: the Managed bytes"
            );
            assert!(
                image(name, &over) == rows(&bytes, offset, pitch),
                "{name}: the image over a Managed buffer after replace_region (device copy: {})",
                !core.unified_memory()
            );

            drop((texture, plain, buffer, command_buffer, queue, over, managed));
            (device.release_block(cpu)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }

    /// A render target over a buffer is in place only where the subresource the device
    /// advertises ends within its own rows and the buffer: an RGBA8Unorm render target 16 texels
    /// wide at the device's own pitch, 3 and then 4 rows high, in a buffer that ends at its rows'
    /// end, every byte of the span to 64 KiB past it holding a sentinel. Where its subresource
    /// passes its rows it is copied; on lavapipe, whose llvmpipe advertises a height rounded up to
    /// 4 rows, the 3-row one is copied and the 4-row one in place. A pass that clears it and
    /// stores leaves its texels the clear's and every other byte of the span the sentinel, and
    /// where it is copied, its copies are counted. A sampled texture over the same buffer stays in
    /// place on every device, at either height.
    #[test]
    fn a_render_target_over_a_buffer_is_in_place_only_within_its_rows() {
        const SIDE: u32 = 16;
        const SIDE_ROW: u64 = SIDE as u64 * 4;
        const TAIL: u64 = 65_536;
        const SENTINEL: u8 = 0xC3;
        const RED: [u8; 4] = [255, 0, 0, 255];
        each_device("over reach", |name, _, device| {
            let answer = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let pitch = SIDE_ROW.next_multiple_of(answer);
            let counters = device.counters();
            let queue = device.new_command_queue(4).expect("a queue");
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 1)).expect("a grant");
            for height in [3, 4] {
                let rows = u64::from(height) * pitch;
                let block = (device.take_block(MemoryClass::Shared, AT + rows + TAIL))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let span = (device.new_buffer(block.id, AT, rows + TAIL, None))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let buffer = (device.new_buffer(block.id, AT, rows, None))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let mut now = vec![SENTINEL; (rows + TAIL) as usize];
                write(&span, &now);
                let mut d = TextureDescriptor::new(
                    TextureType::Type2D,
                    PixelFormat::RGBA8Unorm,
                    SIDE,
                    height,
                );
                d.usage = TextureUsage::SHADER_READ;
                let sampled = (buffer.new_texture(&d, Some(first), 0, pitch))
                    .unwrap_or_else(|error| panic!("{name}: {height} rows: {error}"));
                assert!(
                    sampled.copied_over().is_none(),
                    "{name}: a sampled texture of {height} rows is in place"
                );
                d.usage = TextureUsage::RENDER_TARGET;
                let size = device.linear_subresource_size(&d);
                let target = (buffer.new_texture(&d, None, 0, pitch))
                    .unwrap_or_else(|error| panic!("{name}: {height} rows: {error}"));
                let copied = target.copied_over().is_some();
                let past = size.is_some_and(|size| size > rows);
                assert!(
                    copied || !past,
                    "{name}: {height} rows advertised as {size:?} bytes past its {rows} is copied"
                );
                if cfg!(m2v_cpu_icd) {
                    assert_eq!(
                        copied,
                        height == 3,
                        "{name}: on lavapipe, {height} rows advertised as {size:?} bytes"
                    );
                }

                let copies = counters.metrics().over_copies;
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut pass = RenderPassDescriptor::default();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(target.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor {
                    red: 1.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0,
                };
                let encoder = (command_buffer.render_command_encoder(&pass))
                    .unwrap_or_else(|error| panic!("{name}: {height} rows: {error}"));
                encoder.end_encoding();
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: {height} rows");
                let moved = counters.metrics().over_copies - copies;
                assert_eq!(
                    moved > 0,
                    copied,
                    "{name}: {height} rows: {moved} copies counted"
                );
                for y in 0..u64::from(height) {
                    let at = (y * pitch) as usize;
                    let (texels, _) = now[at..at + SIDE_ROW as usize].as_chunks_mut::<4>();
                    texels.fill(RED);
                }
                assert!(
                    held(&span) == now.as_slice(),
                    "{name}: {height} rows: the texels cleared, every other byte the sentinel"
                );
                println!(
                    "over reach: {name}: {height} rows at pitch {pitch}, advertised as {size:?} \
                     bytes against {rows}: {}, {moved} copies counted; the sampled one in place",
                    match copied {
                        true => "copied",
                        false => "in place",
                    }
                );
                drop((target, sampled, command_buffer, buffer, span));
                (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            drop(queue);
        });
    }
}
