//! Shared memory's coherence (`crate::coherence`). Under a device cfg, on every device of the
//! cfg's kind: the device holds a coherence state exactly where its Shared type, as it reads it,
//! is not host-coherent, and makes no flush or invalidate where it holds none. Where it holds one
//! (every GPU read through the refused device's view, `crate::tests::cached_types_incoherent`):
//! the zero source is flushed at its make; a fill into a Shared buffer is flushed over the
//! buffer's atoms before its submission, then flushed and invalidated there before its status
//! reads completed, and its bytes are the fill's; a fill into a Private buffer makes no call; the
//! bytes a texture's `replace_region` stages are flushed once, and those its `get_bytes` reads
//! flushed and invalidated once. Each call is counted with its bytes, and none fails.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    #[cfg(m2v_gpu)]
    use crate::tests::command::on_device::each_device_with;
    use crate::tests::command::on_device::{finished, on_each_device, within};
    use crate::{
        CommandBuffer, CommandQueue, Device, HostBytes, MemoryClass, Metrics, PixelFormat, Region,
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

    /// The calls of the zero source's make, then of a fill into a Shared buffer, of one into a
    /// Private buffer, and of a texture's copies through staging, with `atom` the device's
    /// `nonCoherentAtomSize` where it holds a coherence state, else `None`. Asserts the fill's
    /// bytes and the texture's.
    fn calls_of_the_work(name: &str, device: &Device, atom: Option<u64>) {
        const OFFSET: u64 = 256;
        const LEN: u64 = 64;
        let counters = device.counters();
        settled(device);
        within("the zero source", || device.core().filler.get().is_some());
        let made = counters.metrics();
        println!(
            "coherence: {name}: atom {atom:?}; at the zero source's make: flushes {}, bytes {}",
            made.shared_flushes, made.shared_flushed_bytes
        );

        let shared = (device.take_block(MemoryClass::Shared, 4096)).expect("a Shared block");
        let buffer = (device.new_buffer(shared.id, OFFSET, LEN, Some(0))).expect("a buffer");
        let private = (device.take_block(MemoryClass::Private, 4096)).expect("a Private block");
        let unseen = (device.new_buffer(private.id, 0, LEN, None)).expect("a buffer");
        let queue = device.new_command_queue(4).expect("a queue");
        let recording: CommandBuffer = queue.command_buffer().expect("a command buffer");
        assert_eq!(
            (recording.holds_coherence_for_test())
                .unwrap_or_else(|error| panic!("{name}: {error}")),
            atom.is_some(),
            "{name}: a command buffer holds the coherence state exactly where its device does"
        );
        drop(recording);
        // The first fill of each, so the second meets no first use.
        fill(name, &queue, &buffer, 0..LEN, 1);
        fill(name, &queue, &unseen, 0..LEN, 1);

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
        let texture =
            (device.new_texture_for_views(&d)).unwrap_or_else(|error| panic!("{name}: {error}"));
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
             failures: a fill into Shared {into_shared:?}, into Private {into_private:?}, \
             replace_region {staged_in:?}, get_bytes {staged_out:?}"
        );
        assert!(bytes.iter().all(|&byte| byte == 0x5a), "{name}: the fill");
        assert_eq!(read, texels, "{name}: the texture's bytes");
        assert_eq!(into_private, [0; 5], "{name}: a fill into a Private buffer");
        let Some(atom) = atom else {
            assert_eq!(
                (made.shared_flushes, made.shared_invalidates),
                (0, 0),
                "{name}: the zero source's make, with no coherence state"
            );
            for (work, calls) in [
                ("a fill into a Shared buffer", into_shared),
                ("replace_region", staged_in),
                ("get_bytes", staged_out),
            ] {
                assert_eq!(calls, [0; 5], "{name}: {work}, with no coherence state");
            }
            return;
        };
        assert!(
            made.shared_flushes >= 1,
            "{name}: the zero source was flushed at its make"
        );
        // The buffer's bytes in its block, widened to whole atoms; a block starts at a multiple
        // of 64 KiB of its allocation.
        let widened = (OFFSET + LEN).next_multiple_of(atom) - (OFFSET - OFFSET % atom);
        assert_eq!(
            into_shared,
            [2, 2 * widened, 1, widened, 0],
            "{name}: a fill into a Shared buffer: its submission's flush, then its completion's \
             flush and invalidate, each over the buffer's atoms ({widened} bytes)"
        );
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
    }

    /// Every device of the cfg's kind, made as the device tests make it: it holds a coherence
    /// state exactly where its Shared type lacks `HOST_COHERENT` as it reads it, and its work's
    /// calls follow. Under `m2v_shared_view` the device reads its types through the suite's view,
    /// which leaves no cached type coherent, so there every device builds the state.
    #[test]
    fn every_device_flushes_and_invalidates_only_where_its_shared_type_is_not_coherent() {
        use ash::vk::MemoryPropertyFlags as F;
        let runtime = crate::tests::validation::instance(&[]);
        let (_, vk_instance) = runtime.raw();
        on_each_device("coherence", &runtime, |name, _, device| {
            let physical = device.raw().physical_device;
            // SAFETY: the device's physical device is one of the instance's.
            let mut memory = unsafe { vk_instance.get_physical_device_memory_properties(physical) };
            if cfg!(m2v_shared_view) {
                crate::tests::cached_types_incoherent(&mut memory);
            }
            let index = device.memory_types().shared;
            let shared = memory.memory_types_as_slice()[index as usize].property_flags;
            let atom = (device.core().coherence.as_ref()).map(|coherence| coherence.atom);
            let state = if atom.is_some() { "built" } else { "none" };
            let flags = crate::floor::flag_names(shared);
            println!(
                "coherence: {name}: Shared type {index} with {flags}; coherence state {state}"
            );
            assert_eq!(
                atom.is_some(),
                !shared.contains(F::HOST_COHERENT),
                "{name}: the coherence state is built exactly where Shared's type is not coherent"
            );
            assert!(
                atom.is_some() || !cfg!(m2v_shared_view),
                "{name}: under m2v_shared_view every device builds the coherence state"
            );
            calls_of_the_work(name, &device, atom);
        });
    }

    /// Every GPU of the cfg's kind, read through the refused device's view
    /// (`crate::tests::cached_types_incoherent`): its Shared type is cached and not coherent, so
    /// it holds the coherence state and makes the calls, on memory that is coherent in fact.
    /// Lavapipe's one memory type holds the parameter ring too, which the view leaves with no
    /// coherent type, so the floor refuses it there (`tests::floor`).
    #[cfg(m2v_gpu)]
    #[test]
    fn every_gpu_read_as_the_refused_device_flushes_and_invalidates_the_shared_memory_work_reaches()
    {
        let view = || crate::DeviceOptions {
            memory: Some(crate::tests::cached_types_incoherent),
            ..crate::DeviceOptions::default()
        };
        each_device_with("coherence", view, |name, _, device| {
            let atom = (device.core().coherence.as_ref()).map(|coherence| coherence.atom);
            assert!(atom.is_some(), "{name}: the view built the coherence state");
            calls_of_the_work(name, &device, atom);
        });
    }
}
