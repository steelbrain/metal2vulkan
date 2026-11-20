//! Textures over a buffer in an image of their own (`crate::over`): under a device cfg, on every
//! device of the cfg's kind, RGBA8Unorm render targets over a buffer whose rows lie one alignment
//! past the device's own pitch, which no linear image takes.
//!
//! - Over a Shared buffer, once a submission: each command buffer that uses the texture, alone in
//!   its submission, copies it in once, the host's writes between them reaching the image; a use
//!   after a fill of the buffer takes the filled bytes, in the fill's command buffer or the next
//!   one's; a blit's write to the texture is in the buffer for the host once the command buffer
//!   completes.
//! - Over a Managed buffer's device copy, where the device keeps one, only at Metal's Managed
//!   events: 100 command buffers using the texture copy it in once; a didModifyRange over part of
//!   its range copies it again once, and one outside it not at all; a kernel that writes the
//!   buffer owes one copy in the next submission, one that only reads it none; a texture written
//!   and then read through its buffer by a blit shows the written texels; a synchronize after a
//!   texture write puts its texels in the consumer's bytes; of two textures over one range, the
//!   second read after the first is written sees the first's texels; a texture written and
//!   released copies nothing back once its buffer is released, its image freed then; one written
//!   and released goes back before a kernel that reads its buffer, which sees its texels.
//! - A kernel that declares a Managed buffer read-write and never touches it owes the texture over
//!   its device copy no copy: the compiled interface's uses decide, not the declaration.
//!
//! `llvm-as` assembles the last test's kernel; without it that test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::tests::compute::on_device::{add, step, WORDS};
    use crate::texture::TextureObject;
    use crate::{
        Buffer, BufferLayout, CommandBuffer, CommandQueue, ComputePipelineDescriptor, Device,
        DispatchType, FunctionDescriptor, IndexClass, Library, MemoryClass, PixelFormat, Region,
        Resource, StageSource, StorageMode, Texture, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage,
    };
    use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};
    use std::thread;
    use std::time::{Duration, Instant};

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
        let texture = (device.new_texture_for_views(&d)).unwrap_or_else(|e| panic!("{name}: {e}"));
        (texture.replace_region(whole(), 0, 0, texels.into(), ROW, 0))
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

    fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(120);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn every_device_copies_a_texture_over_a_shared_buffer_once_a_submission() {
        each_device("over shared", |name, _, device| {
            let pitch = pitch(&device);
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let buffer = (device.new_buffer(cpu.id, 0, LEN, Some(0)))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut now: Vec<u8> = (0..LEN).map(|i| (i % 251) as u8).collect();
            write(&buffer, &now);
            let copied_over = || device.counters().gauges().textures_copied_over;
            let before = copied_over();
            // A colour render target takes an index.
            (device.grant_indices(IndexClass::Texture, 4096, 1)).expect("a grant");
            let texture = (buffer.new_texture(&descriptor(), Some(4096), 0, pitch))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                texture.copied_over().is_some(),
                "{name}: an image of its own"
            );
            assert_eq!(copied_over(), before + 1, "{name}: the gauge counts it");
            let scratch = plain(name, &device, &texels(0));
            let queue = device.new_command_queue(4).expect("a queue");
            for round in 0..3 {
                let before = device.counters().metrics();
                run(name, &queue, |cb| blit(cb, &texture, &scratch));
                let after = device.counters().metrics();
                assert_eq!(
                    after.over_copies - before.over_copies,
                    1,
                    "{name}: round {round}"
                );
                // The copy's bytes, the texture's texels, count in both byte metrics.
                let bytes = TEXELS as u64;
                assert_eq!(
                    after.over_copy_bytes - before.over_copy_bytes,
                    bytes,
                    "{name}: round {round}: its bytes"
                );
                assert_eq!(
                    after.bytes_copied - before.bytes_copied,
                    bytes,
                    "{name}: round {round}: its bytes copied"
                );
                // The buffer is in a block of the runtime's own: no imported copy.
                assert_eq!(
                    (after.over_copies_imported, after.over_copy_bytes_imported),
                    (before.over_copies_imported, before.over_copy_bytes_imported),
                    "{name}: round {round}: not imported"
                );
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
            (device.release_block(cpu)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }

    #[test]
    fn every_device_copies_a_texture_over_a_managed_buffer_at_its_events() {
        each_device("over managed", |name, _, device| {
            if device.core().unified_memory() {
                println!("over managed: {name}: no device copy, the Shared rule");
                return;
            }
            let pitch = pitch(&device);
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let managed = |offset: u64| {
                (crate::tests::managed::on_device::new_managed(&device, cpu.id, offset, LEN, 0))
                    .unwrap_or_else(|e| panic!("{name}: {e}"))
            };
            // A colour render target takes an index: each texture over a buffer its own.
            (device.grant_indices(IndexClass::Texture, 4096, 4)).expect("a grant");
            let next = std::cell::Cell::new(4096);
            let over = |buffer: &Buffer| {
                let index = next.replace(next.get() + 1);
                let texture = (buffer.new_texture(&descriptor(), Some(index), 0, pitch))
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                assert!(
                    texture.managed_over().is_some(),
                    "{name}: over the device copy"
                );
                assert_eq!(texture.descriptor().storage_mode, StorageMode::Managed);
                texture
            };
            let queue = device.new_command_queue(4).expect("a queue");
            let scratch = plain(name, &device, &texels(0));
            let buffer = managed(0);
            let mut bytes: Vec<u8> = (0..LEN).map(|i| (i % 249) as u8).collect();
            write(&buffer, &bytes);
            (buffer.did_modify_range(0, LEN)).unwrap_or_else(|e| panic!("{name}: {e}"));
            // What the device copy holds.
            let mut side = bytes.clone();
            let texture = over(&buffer);
            let used = |what: &str, expected: u64| {
                let before = copies(&device);
                run(name, &queue, |cb| blit(cb, &texture, &scratch));
                assert_eq!(copies(&device) - before, expected, "{name}: {what}");
            };

            // 100 command buffers with no event: one copy in.
            let before = copies(&device);
            for _ in 0..100 {
                run(name, &queue, |cb| blit(cb, &texture, &scratch));
            }
            assert_eq!(copies(&device) - before, 1, "{name}: 100 uses");
            assert!(
                image(name, &scratch) == rows(&side, 0, pitch),
                "{name}: first copy"
            );

            // A didModifyRange over part of the range, then one outside it.
            bytes[pitch as usize..pitch as usize + 16].fill(0xEE);
            write(&buffer, &bytes);
            (buffer.did_modify_range(pitch, 16)).unwrap_or_else(|e| panic!("{name}: {e}"));
            side[pitch as usize..pitch as usize + 16].fill(0xEE);
            used("a modify over the range", 1);
            assert!(
                image(name, &scratch) == rows(&side, 0, pitch),
                "{name}: modified"
            );
            bytes[LEN as usize - 64..].fill(0x11);
            write(&buffer, &bytes);
            (buffer.did_modify_range(LEN - 64, 64)).unwrap_or_else(|e| panic!("{name}: {e}"));
            side[LEN as usize - 64..].fill(0x11);
            used("a modify outside the range", 0);

            // A kernel that writes the buffer owes a copy; one that only reads it, none.
            let state = step(&device);
            run(name, &queue, |cb| {
                let mut encoder =
                    (cb.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&buffer), 0, 0).expect("a buffer");
                (encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])).expect("a dispatch");
                encoder.end_encoding();
            });
            for word in side[..WORDS * 4].as_chunks_mut::<4>().0 {
                *word = u32::from_le_bytes(*word)
                    .wrapping_mul(2)
                    .wrapping_add(1)
                    .to_le_bytes();
            }
            used("a kernel's write", 1);
            assert!(
                image(name, &scratch) == rows(&side, 0, pitch),
                "{name}: kernel's write"
            );
            let mut arena = Arena::new(&device);
            let (zero, out) = (arena.buffer(), arena.buffer());
            let sum = add(&device);
            let read = |name: &str, from: &Buffer| {
                run(name, &queue, |cb| {
                    let mut encoder =
                        (cb.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
                    encoder.set_compute_pipeline_state(&sum).expect("a state");
                    encoder.set_buffer(Some(from), 0, 0).expect("a buffer");
                    encoder.set_buffer(Some(&zero), 0, 1).expect("a buffer");
                    encoder.set_buffer(Some(&out), 0, 2).expect("a buffer");
                    let grid = [WORDS as u32, 1, 1];
                    (encoder.dispatch_threads(grid, [16, 1, 1])).expect("a dispatch");
                    encoder.end_encoding();
                });
            };
            read(name, &buffer);
            used("a kernel's read", 0);

            // A texture written, then its buffer read by a blit: the written texels.
            let written = plain(name, &device, &texels(0x21));
            run(name, &queue, |cb| blit(cb, &written, &texture));
            let target = plain(name, &device, &texels(0));
            let before = copies(&device);
            run(name, &queue, |cb| {
                let mut encoder = cb.blit_command_encoder().expect("an encoder");
                let layout = BufferLayout {
                    offset: 0,
                    bytes_per_row: pitch,
                    bytes_per_image: 0,
                };
                let at = TexturePlace::default();
                (encoder.copy_from_buffer(&buffer, layout, [WIDTH, HEIGHT, 1], &target, at))
                    .expect("a copy");
                encoder.end_encoding();
            });
            assert_eq!(copies(&device) - before, 1, "{name}: one copy back");
            assert!(
                image(name, &target) == texels(0x21),
                "{name}: the buffer's texels"
            );
            lay(&mut side, &texels(0x21), 0, pitch);

            // A synchronize after a texture write: the texels in the consumer's bytes.
            let synced = plain(name, &device, &texels(0x33));
            run(name, &queue, |cb| blit(cb, &synced, &texture));
            let before = copies(&device);
            run(name, &queue, |cb| {
                let mut encoder = cb.blit_command_encoder().expect("an encoder");
                (encoder.synchronize_resource(Resource::Buffer(&buffer))).expect("a synchronize");
                encoder.end_encoding();
            });
            assert_eq!(
                copies(&device) - before,
                1,
                "{name}: back before the synchronize"
            );
            lay(&mut side, &texels(0x33), 0, pitch);
            assert!(held(&buffer) == side, "{name}: synchronized");

            // Two textures over one range: the second, read after the first is written, sees it.
            let second = over(&buffer);
            run(name, &queue, |cb| blit(cb, &second, &scratch));
            let first = plain(name, &device, &texels(0x44));
            run(name, &queue, |cb| blit(cb, &first, &texture));
            let before = copies(&device);
            run(name, &queue, |cb| blit(cb, &second, &scratch));
            assert_eq!(
                copies(&device) - before,
                2,
                "{name}: the first back, the second in"
            );
            assert!(
                image(name, &scratch) == texels(0x44),
                "{name}: the first's texels"
            );

            // Written, released, its buffer released: nothing goes back, the image is freed.
            let gone = managed(2 * LEN);
            let lost = over(&gone);
            run(name, &queue, |cb| blit(cb, &first, &lost));
            let (before, textures) = (copies(&device), device.counters().gauges().textures);
            drop(lost);
            assert_eq!(
                device.counters().gauges().textures,
                textures,
                "{name}: held"
            );
            drop(gone);
            wait_until("the held image freed", || {
                device.counters().gauges().textures < textures
            });
            assert_eq!(copies(&device), before, "{name}: nothing copied back");

            // Written and released, then a kernel reads the buffer: it sees the texels.
            let kept = managed(4 * LEN);
            let released = over(&kept);
            run(name, &queue, |cb| blit(cb, &written, &released));
            drop(released);
            let before = copies(&device);
            read(name, &kept);
            assert_eq!(copies(&device) - before, 1, "{name}: back before the read");
            let words = held(&out);
            assert!(
                words[..ROW as usize] == texels(0x21)[..ROW as usize],
                "{name}: the kernel reads the texels"
            );

            drop((texture, second, scratch, written, target, synced, first));
            drop((buffer, kept, zero, out, arena, state, sum, queue));
            (device.release_block(cpu)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }

    /// A kernel that declares buffer 0 `device uint*` (read-write) and never touches it, and
    /// writes each thread's position into word `gid` of buffer 1.
    const UNTOUCHED: &str = "; Owned synthetic kernel.
source_filename = \"untouched.metal\"

define void @untouched(ptr addrspace(1) %b0, ptr addrspace(1) %b1, i32 %gid) {
entry:
  %w = zext i32 %gid to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %w
  store i32 %gid, ptr addrspace(1) %p, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @untouched, !1, !2}
!1 = !{}
!2 = !{!10, !11, !12}
!10 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b0\"}
!11 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b1\"}
!12 = !{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}
";

    #[test]
    fn every_device_owes_no_copy_for_a_kernel_that_leaves_a_managed_buffer_untouched() {
        let bitcode = match llvm_assemble_text(UNTOUCHED) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("untouched: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let function = library.function("untouched").expect("the kernel");
        let kernel = ComputePipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(function),
        ));
        each_device("over untouched", |name, _, device| {
            if device.core().unified_memory() {
                println!("over untouched: {name}: no device copy, the Shared rule");
                return;
            }
            let state = (device
                .new_compute_pipeline_state(&kernel, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            let pitch = pitch(&device);
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let buffer =
                (crate::tests::managed::on_device::new_managed(&device, cpu.id, 0, LEN, 0))
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
            // A colour render target takes an index.
            (device.grant_indices(IndexClass::Texture, 4096, 1)).expect("a grant");
            let texture = (buffer.new_texture(&descriptor(), Some(4096), 0, pitch))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                texture.managed_over().is_some(),
                "{name}: over the device copy"
            );
            let mut arena = Arena::new(&device);
            let out = arena.buffer();
            let scratch = plain(name, &device, &texels(0));
            let queue = device.new_command_queue(4).expect("a queue");
            let used = |what: &str, expected: u64| {
                let before = copies(&device);
                run(name, &queue, |cb| blit(cb, &texture, &scratch));
                assert_eq!(copies(&device) - before, expected, "{name}: {what}");
            };
            used("the first use", 1);
            run(name, &queue, |cb| {
                let mut encoder =
                    (cb.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
                encoder.set_compute_pipeline_state(&state).expect("a state");
                encoder.set_buffer(Some(&buffer), 0, 0).expect("a buffer");
                encoder.set_buffer(Some(&out), 0, 1).expect("a buffer");
                (encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1])).expect("a dispatch");
                encoder.end_encoding();
            });
            let ran: Vec<u8> = (0..WORDS as u32).flat_map(u32::to_le_bytes).collect();
            assert!(held(&out)[..WORDS * 4] == ran, "{name}: the kernel ran");
            used("a use after the kernel that leaves the buffer", 0);
            drop((texture, scratch, buffer, out, arena, state, queue));
            (device.release_block(cpu)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }
}
