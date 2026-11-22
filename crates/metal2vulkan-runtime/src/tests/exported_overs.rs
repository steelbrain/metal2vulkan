//! A texture over a buffer in an exported block, under a device cfg, on every device of the cfg's
//! kind: copied in once after each event that changes its bytes (`crate::over::ExportedOvers`),
//! counted in `Metrics::over_copies` and `Metrics::over_copy_bytes`. After a fill of the buffer and
//! the block's acquire, four submissions that read the texture copy it in once; a fill between
//! makes one copy more, and a zero fill of the block's range one more; reads with no event between
//! copy nothing. Each read finds the bytes of the last event.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Buffer, BufferLayout, CommandQueue, Device, ExportHandle, IndexClass, MemoryClass,
        PixelFormat, StorageMode, Texture, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage,
    };

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 8;
    const ROW: u64 = WIDTH as u64 * 4;
    const LEN: u64 = 1 << 16;

    /// Runs `encode` in a command buffer of its own on `queue`, and waits for it.
    fn run(name: &str, queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        encode(&mut encoder);
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// `count` submissions that each copy the texture into `out`, every byte of which then is
    /// `byte`: the copies in they made, and their bytes.
    fn reads(
        name: &str,
        (device, queue): (&Device, &CommandQueue),
        (texture, out): (&Texture, &Buffer),
        count: usize,
        byte: u8,
    ) -> (u64, u64) {
        let before = device.counters().metrics();
        let contents = out.contents().expect("Shared");
        for read in 0..count {
            // SAFETY: no pending command reaches the buffer's bytes.
            unsafe { contents.as_ptr().write_bytes(0xee, out.length() as usize) };
            let layout = BufferLayout {
                offset: 0,
                bytes_per_row: ROW,
                bytes_per_image: 0,
            };
            run(name, queue, |encoder| {
                (encoder.copy_from_texture_to_buffer(
                    texture,
                    TexturePlace::default(),
                    [WIDTH, HEIGHT, 1],
                    out,
                    layout,
                ))
                .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
            });
            // SAFETY: the read has completed; nothing else reaches the bytes.
            let bytes =
                unsafe { std::slice::from_raw_parts(contents.as_ptr(), out.length() as usize) };
            let wrong = bytes.iter().position(|&b| b != byte);
            assert_eq!(wrong, None, "{name}: read {read}: byte {byte:#x}");
        }
        let after = device.counters().metrics();
        (
            after.over_copies - before.over_copies,
            after.over_copy_bytes - before.over_copy_bytes,
        )
    }

    #[test]
    fn an_exported_blocks_texture_is_copied_in_once_a_change() {
        each_device("exported_overs", |name, _, device| {
            let Some(handle) = (ExportHandle::ALL.into_iter())
                .find(|&handle| device.new_exported_block(handle, LEN).is_ok())
            else {
                println!("exported_overs: {name}: exports no handle");
                return;
            };
            let (block, _export) = (device.new_exported_block(handle, LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = (device.new_buffer(block.id, 0, LEN, None)).expect("a buffer");
            let pitch = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .max(ROW);
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            d.usage = TextureUsage::SHADER_READ;
            d.storage_mode = StorageMode::Private;
            let first = IndexClass::Texture.range().start;
            let _grant = (device.grant_indices(IndexClass::Texture, first, 1)).expect("a grant");
            let texture = (buffer.new_texture(&d, Some(first), 0, pitch))
                .unwrap_or_else(|error| panic!("{name}: a texture over the export: {error}"));
            let out_block = (device.take_block(MemoryClass::Shared, ROW * u64::from(HEIGHT)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let out = (device.new_buffer(out_block.id, 0, ROW * u64::from(HEIGHT), Some(0)))
                .expect("a buffer");
            let queue = device.new_command_queue(4).expect("a queue");
            let plane = ROW * u64::from(HEIGHT);
            let fill = |value: u8| {
                run(name, &queue, |encoder| {
                    (encoder.fill_buffer(&buffer, 0..LEN, value))
                        .unwrap_or_else(|error| panic!("{name}: the fill: {error}"));
                });
            };
            let at = (&device, &queue);
            let over = (&texture, &out);
            fill(0x5a);
            (block.external_writes_done()).unwrap_or_else(|error| panic!("{name}: {error}"));
            let acquired = reads(name, at, over, 4, 0x5a);
            assert_eq!(acquired, (1, plane), "{name}: an acquire, four reads");
            fill(0xb7);
            let filled = reads(name, at, over, 3, 0xb7);
            assert_eq!(filled, (1, plane), "{name}: a fill, three reads");
            (device.zero_block_range(block.id, 0, LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let zeroed = reads(name, at, over, 2, 0);
            assert_eq!(zeroed, (1, plane), "{name}: a zero fill, two reads");
            let unchanged = reads(name, at, over, 2, 0);
            assert_eq!(unchanged, (0, 0), "{name}: two reads, no event");
            println!(
                "exported_overs: {name}: {handle:?}: copies in after an acquire, a fill, a zero \
                 fill and no event: {} {} {} {}",
                acquired.0, filled.0, zeroed.0, unchanged.0
            );
        });
    }
}
