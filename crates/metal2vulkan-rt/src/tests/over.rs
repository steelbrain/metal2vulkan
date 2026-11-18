//! Textures over a buffer: under a device cfg, on every device of the cfg's kind, the linear
//! alignment answer is a power of two from 16 to 4,096, the same when asked again; an RGBA8Unorm
//! texture over a Shared buffer at an offset and pitch of the answer, its rows longer than their
//! texels, is made in place: it allocates nothing and stages nothing, its image's bytes are the
//! host's in the buffer as they change, `replace_region` and `get_bytes` write and read the
//! buffer, and a blit's write to the texture reaches the host in the buffer. Over a Managed buffer
//! the image is the bytes shaders reach, which `replace_region` changes. Rows past the buffer,
//! closer than a row or off a texel, and an offset off a texel, are refused; a 3D texture or one
//! of two levels over a buffer is the gap; and every gauge reads zero after teardown.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Buffer, Error, Gap, IndexClass, MemoryClass, PixelFormat, Refusal, Region, StorageMode,
        Texture, TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };

    const WIDTH: u32 = 100;
    const HEIGHT: u32 = 10;
    const ROW: u64 = WIDTH as u64 * 4;
    /// The buffer's offset in its block: a multiple of every answer.
    const AT: u64 = 65_536;
    const LEN: u64 = 65_536;

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
            for pf in [
                PixelFormat::R8Unorm,
                PixelFormat::RGBA8Unorm,
                PixelFormat::R32Float,
                PixelFormat::RGBA16Float,
                PixelFormat::RGBA32Float,
            ] {
                let answer = (device.minimum_linear_texture_alignment(pf))
                    .unwrap_or_else(|error| panic!("{name}: {pf:?}: {error}"));
                assert!(
                    answer.is_power_of_two() && (16..=4096).contains(&answer),
                    "{name}: {pf:?}: {answer}"
                );
                answers.push((pf, answer));
            }
            println!("over: {name}: minimum linear texture alignments {answers:?}");
            let align = answers[1].1;
            let again = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(again, align, "{name}: asked again");

            let counters = device.counters();
            let core = device.core();
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = (device.new_buffer(cpu.id, AT, LEN, true))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut now: Vec<u8> = (0..LEN).map(|i| (i % 253) as u8).collect();
            write(&buffer, &now);
            let pitch = (ROW + 1).next_multiple_of(align);
            let offset = align;
            // Read and written by shaders, at indices of a grant.
            (device.grant_indices(IndexClass::StorageTexture, 1, 4)).expect("a grant");
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.storage_mode = StorageMode::Private;
            let (blocks, copied) = (counters.gauges().blocks, counters.metrics().bytes_copied);
            let texture = (buffer.new_texture(&d, Some(1), offset, pitch))
                .unwrap_or_else(|error| panic!("{name}: at {offset}, pitch {pitch}: {error}"));
            assert_eq!(texture.descriptor().storage_mode, StorageMode::Shared);
            assert_eq!(
                (counters.gauges().blocks, core.places.blocks()),
                (blocks, 0),
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
            (texture.replace_region(small, 0, 0, &[0xA5; 80], 20, 0))
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
            let source: Vec<u8> = (0..ROW * u64::from(HEIGHT))
                .map(|i| (i % 241) as u8 ^ 0x3C)
                .collect();
            (plain.replace_region(whole(), 0, 0, &source, ROW, 0))
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
            let managed = (device.new_managed_buffer(cpu.id, 2 * AT, LEN, LEN))
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
            (over.replace_region(small, 0, 0, &[0x5A; 80], 20, 0))
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
                !core.shared_is_device_local()
            );

            drop((texture, plain, buffer, command_buffer, queue, over, managed));
            (device.release_block(cpu.id)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }
}
