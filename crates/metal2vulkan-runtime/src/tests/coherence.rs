//! Shared memory's coherence (`crate::coherence`). Under a device cfg, on every device of the
//! cfg's kind: the zero source is flushed at its make; a fill into a Shared buffer is flushed over
//! the buffer's atoms before its submission, then flushed and invalidated there before its status
//! reads completed, and its bytes are the fill's; a fill into a Private buffer makes no call; the
//! bytes a texture's `replace_region` stages are flushed once, and those its `get_bytes` reads
//! flushed and invalidated once. Each call is counted with its bytes, and none fails.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, finished, within};
    use crate::{
        CommandBuffer, CommandQueue, HostBytes, MemoryClass, Metrics, PixelFormat, Region,
        TextureDescriptor, TextureType,
    };

    /// From `before` to `after`: the flushes, their bytes, the invalidates, their bytes, and the
    /// calls that failed.
    fn calls(before: &Metrics, after: &Metrics) -> [u64; 5] {
        [
            after.shared_flushes - before.shared_flushes,
            after.shared_flushed_bytes - before.shared_flushed_bytes,
            after.shared_invalidates - before.shared_invalidates,
            after.shared_invalidated_bytes - before.shared_invalidated_bytes,
            after.shared_coherence_failures - before.shared_coherence_failures,
        ]
    }

    /// A command buffer of `queue` with one fill of `range` of `buffer` to `value`, committed and
    /// complete with no error.
    fn fill(
        name: &str,
        queue: &CommandQueue,
        buffer: &crate::Buffer,
        range: std::ops::Range<u64>,
        value: u8,
    ) {
        let command_buffer: CommandBuffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        (encoder.fill_buffer(buffer, range, value))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        encoder.end_encoding();
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// The flushes and invalidates of a fill into a Shared buffer, of one into a Private buffer,
    /// and of a texture's copies through staging.
    #[test]
    fn every_device_flushes_and_invalidates_the_shared_memory_work_reaches() {
        const OFFSET: u64 = 256;
        const LEN: u64 = 64;
        each_device("coherence", |name, _, device| {
            let atom = device.core().coherence.atom;
            let counters = device.counters();
            settled(&device);
            within("the zero source", || device.core().filler.get().is_some());
            let made = counters.metrics();
            println!(
                "coherence: {name}: atom {atom}; at the zero source's make: flushes {}, bytes {}",
                made.shared_flushes, made.shared_flushed_bytes
            );
            assert!(
                made.shared_flushes >= 1,
                "{name}: the zero source was flushed at its make"
            );

            let shared = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
            let buffer = (device.new_buffer(shared.id, OFFSET, LEN, Some(0))).expect("a buffer");
            let private = (device.take_block(MemoryClass::Private, 4096)).expect("a Private block");
            let unseen = (device.new_buffer(private.id, 0, LEN, None)).expect("a buffer");
            let queue = device.new_command_queue(4).expect("a queue");
            // The first fill of each, so the second meets no first use.
            fill(name, &queue, &buffer, 0..LEN, 1);
            fill(name, &queue, &unseen, 0..LEN, 1);

            // The buffer's bytes in its block, widened to whole atoms; a block starts at a
            // multiple of 64 KiB of its allocation.
            let widened = (OFFSET + LEN).next_multiple_of(atom) - (OFFSET - OFFSET % atom);
            let before = counters.metrics();
            fill(name, &queue, &buffer, 0..LEN, 0x5a);
            let after = counters.metrics();
            let into_shared = calls(&before, &after);
            let contents = buffer.contents().expect("a Shared buffer");
            // SAFETY: the buffer is alive, `LEN` bytes long, and its work completed.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), LEN as usize) };

            let before = counters.metrics();
            fill(name, &queue, &unseen, 0..LEN, 0x5a);
            let into_private = calls(&before, &counters.metrics());

            let d = TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
            // A texture with no index: only its copies reach it.
            let texture = (device.new_texture_for_views(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let whole = Region {
                origin: [0; 3],
                size: [4, 4, 1],
            };
            let texels: Vec<u8> = (0..64).collect();
            let before = counters.metrics();
            (texture.replace_region(whole, 0, 0, HostBytes::from(&texels[..]), 16, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let replaced = counters.metrics();
            let mut read = vec![0u8; 64];
            (texture.get_bytes(&mut read, 16, 0, whole, 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let staged_in = calls(&before, &replaced);
            let staged_out = calls(&replaced, &counters.metrics());
            println!(
                "coherence: {name}: flushes, flushed bytes, invalidates, invalidated bytes, \
                 failures: a fill into Shared {into_shared:?} (atoms {widened} bytes), into \
                 Private {into_private:?}, replace_region {staged_in:?}, get_bytes {staged_out:?}"
            );
            assert_eq!(
                into_shared,
                [2, 2 * widened, 1, widened, 0],
                "{name}: a fill into a Shared buffer: its submission's flush, then its completion's \
                 flush and invalidate, each over the buffer's atoms"
            );
            assert!(bytes.iter().all(|&byte| byte == 0x5a), "{name}: the fill");
            assert_eq!(into_private, [0; 5], "{name}: a fill into a Private buffer");
            assert_eq!(
                [staged_in[0], staged_in[2], staged_in[4]],
                [1, 0, 0],
                "{name}: replace_region's flushes, invalidates and failures"
            );
            assert!(
                staged_in[1] >= 64,
                "{name}: replace_region flushed its 64 bytes"
            );
            assert_eq!(
                [staged_out[0], staged_out[2], staged_out[4]],
                [1, 1, 0],
                "{name}: get_bytes's flushes, invalidates and failures"
            );
            assert!(
                staged_out[1] >= 64 && staged_out[3] >= 64,
                "{name}: get_bytes flushed and invalidated its 64 bytes"
            );
            assert_eq!(read, texels, "{name}: the texture's bytes");
        });
    }
}
