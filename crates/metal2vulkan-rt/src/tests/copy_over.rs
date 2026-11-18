//! Textures over a buffer in an image of their own (`crate::over`): under a device cfg, on every
//! device of the cfg's kind, an RGBA8Unorm render target over a Shared buffer whose rows lie one
//! alignment past the device's own pitch, which no linear image takes. Each command buffer that
//! uses the texture, alone in its submission, copies it in once, the host's writes between them
//! reaching the image; a use after a fill of the buffer takes the filled bytes, in the fill's
//! command buffer or the next one's; a blit's write to the texture is in the buffer for the host
//! once the command buffer completes.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Buffer, CommandBuffer, CommandQueue, Device, MemoryClass, PixelFormat, Region, StorageMode,
        Texture, TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };

    const WIDTH: u32 = 16;
    const HEIGHT: u32 = 4;
    const ROW: u64 = WIDTH as u64 * 4;
    const LEN: u64 = 8192;
    const TEXELS: usize = (ROW * HEIGHT as u64) as usize;

    fn whole() -> Region {
        Region {
            origin: [0; 3],
            size: [WIDTH, HEIGHT, 1],
        }
    }

    fn descriptor() -> TextureDescriptor {
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
        d.usage = TextureUsage::RENDER_TARGET;
        d
    }

    /// One alignment past the device's own pitch for the texture's rows: no linear image takes it.
    fn pitch(device: &Device) -> u64 {
        let align = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
            .expect("an alignment");
        ROW.next_multiple_of(align) + align
    }

    /// The buffer's consumer bytes.
    fn held(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("host bytes");
        // SAFETY: the buffer's bytes, which no command writes while the test reads them.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    /// Writes `bytes` over the buffer's consumer bytes.
    fn write(buffer: &Buffer, bytes: &[u8]) {
        let contents = buffer.contents().expect("host bytes");
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

    /// Lays `texels` (packed) into `bytes` at the texture's rows.
    fn lay(bytes: &mut [u8], texels: &[u8], offset: u64, pitch: u64) {
        for row in 0..u64::from(HEIGHT) {
            let at = (offset + row * pitch) as usize;
            let from = (row * ROW) as usize;
            bytes[at..at + ROW as usize].copy_from_slice(&texels[from..from + ROW as usize]);
        }
    }

    /// A plain texture's image's bytes, read by the device.
    fn image(name: &str, texture: &Texture) -> Vec<u8> {
        let mut bytes = vec![0; TEXELS];
        (texture.image_bytes(&mut bytes, ROW, whole(), 0, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        bytes
    }

    /// A plain Shared texture holding `texels`.
    fn plain(name: &str, device: &Device, texels: &[u8]) -> Texture {
        let mut d = descriptor();
        d.storage_mode = StorageMode::Shared;
        let texture = (device.new_texture(&d, None)).unwrap_or_else(|e| panic!("{name}: {e}"));
        (texture.replace_region(whole(), 0, 0, texels, ROW, 0))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        texture
    }

    /// Texels of `seed`.
    fn texels(seed: u8) -> Vec<u8> {
        (0..TEXELS)
            .map(|i| (i as u8).wrapping_mul(7) ^ seed)
            .collect()
    }

    /// Runs one command buffer that `record` fills, alone in its submission, to completion.
    fn run(name: &str, queue: &CommandQueue, record: impl FnOnce(&CommandBuffer)) {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        record(&command_buffer);
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// A blit of `from` into `to`, whole.
    fn blit(command_buffer: &CommandBuffer, from: &Texture, to: &Texture) {
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        let at = TexturePlace::default();
        (encoder.copy_from_texture(from, at, [WIDTH, HEIGHT, 1], to, at)).expect("a copy");
        encoder.end_encoding();
    }

    fn copies(device: &Device) -> u64 {
        device.counters().metrics().over_copies
    }

    #[test]
    fn every_device_copies_a_texture_over_a_shared_buffer_once_a_submission() {
        each_device("over shared", |name, _, device| {
            let pitch = pitch(&device);
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let buffer =
                (device.new_buffer(cpu.id, 0, LEN, true)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut now: Vec<u8> = (0..LEN).map(|i| (i % 251) as u8).collect();
            write(&buffer, &now);
            let texture = (buffer.new_texture(&descriptor(), None, 0, pitch))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                texture.copied_over().is_some(),
                "{name}: an image of its own"
            );
            let scratch = plain(name, &device, &texels(0));
            let queue = device.new_command_queue(4).expect("a queue");
            for round in 0..3 {
                let before = copies(&device);
                run(name, &queue, |cb| blit(cb, &texture, &scratch));
                assert_eq!(copies(&device) - before, 1, "{name}: round {round}");
                assert!(
                    image(name, &scratch) == rows(&now, 0, pitch),
                    "{name}: round {round}: the image takes the host's bytes"
                );
                now.iter_mut().for_each(|byte| *byte = byte.wrapping_add(3));
                write(&buffer, &now);
            }

            // A fill of the buffer before a use of the texture in one command buffer: the use
            // takes the filled bytes, copied in after the fill, with no leading copy.
            let before = copies(&device);
            run(name, &queue, |cb| {
                let mut encoder = cb.blit_command_encoder().expect("an encoder");
                (encoder.fill_buffer(&buffer, 0..LEN, 0xC3)).expect("a fill");
                let at = TexturePlace::default();
                (encoder.copy_from_texture(&texture, at, [WIDTH, HEIGHT, 1], &scratch, at))
                    .expect("a copy");
                encoder.end_encoding();
            });
            assert_eq!(copies(&device) - before, 1, "{name}: in after the fill");
            assert!(
                image(name, &scratch) == vec![0xC3; TEXELS],
                "{name}: the use after the fill takes the filled bytes"
            );

            // A fill in one command buffer and a use of the texture in the next, committed with no
            // wait between: the use's copy in follows the fill, in one submission or two.
            let before = copies(&device);
            let fill = queue.command_buffer().expect("a command buffer");
            let mut encoder = fill.blit_command_encoder().expect("an encoder");
            (encoder.fill_buffer(&buffer, 0..LEN, 0x3C)).expect("a fill");
            encoder.end_encoding();
            let use_ = queue.command_buffer().expect("a command buffer");
            blit(&use_, &texture, &scratch);
            fill.commit().expect("a commit");
            use_.commit().expect("a commit");
            finished(name, &use_);
            assert_eq!(
                copies(&device) - before,
                1,
                "{name}: in once, after the fill"
            );
            assert!(
                image(name, &scratch) == vec![0x3C; TEXELS],
                "{name}: the next command buffer's use takes the filled bytes"
            );
            now.fill(0x3C);

            // A blit's write goes back to the buffer at the segment's end: one copy in, one back.
            let source = plain(name, &device, &texels(0x5A));
            let before = copies(&device);
            run(name, &queue, |cb| blit(cb, &source, &texture));
            assert_eq!(copies(&device) - before, 2, "{name}: in and back");
            lay(&mut now, &texels(0x5A), 0, pitch);
            assert!(
                held(&buffer) == now,
                "{name}: the texture's write in the buffer"
            );
            drop((texture, scratch, source, buffer, queue));
            (device.release_block(cpu.id)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }
}
