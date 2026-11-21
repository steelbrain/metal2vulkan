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
//! - Over a Shared buffer, what a draw or a dispatch reaches decides, read at the command, the
//!   counted copies read at once:
//!   - a draw that binds the texture after a fill of its buffer in the command buffer copies it in
//!     first, at the draw, and reads the filled texel; a second draw of the binding copies
//!     nothing; a dispatch that only declares it (`use_resource`) after a fill copies it in
//!     first too, and one that does not reach it copies nothing;
//!   - a draw whose vertex function stores into the buffer makes the image stale: the next use
//!     copies it in again, at the use, and takes the stores, whether its first use in the command
//!     buffer came before the draw or after;
//!   - after a blit wrote the texture, a draw that reads no buffer under it copies nothing, a draw
//!     that reads the buffer as a vertex buffer takes the image back first, at the draw, and a
//!     draw that reads through an argument buffer (an address outside its resources) takes every
//!     such image back first, at the draw, as the whole segment's answer;
//!   - 512 such textures, bound once and unbound in the command buffer before its draws, cost the
//!     draws and dispatches that bind none of them nothing: user-space instructions a draw and a
//!     dispatch within a tenth of those of a command buffer that keeps none, the textures alive in
//!     both, printed where the kernel offers the counter (Linux).
//!
//! `llvm-as` assembles the kernels and the draws' functions; without it those tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished, Arena};
    use crate::tests::compute::on_device::{add, step, WORDS};
    use crate::tests::draw::on_device as draw;
    use crate::tests::storage_element::on_device::{place_id, ARGUMENT_FRAGMENT};
    use crate::texture::TextureObject;
    use crate::{
        Block, Buffer, BufferLayout, ColorAttachmentDescriptor, CommandBuffer, CommandQueue,
        ComputePipelineDescriptor, Device, DispatchType, Error, FunctionDescriptor, IndexClass,
        Library, MemoryClass, PixelFormat, PrimitiveType, Region, RenderPipelineDescriptor,
        Resource, ResourceUsage, StageSource, StorageMode, Texture, TextureDescriptor,
        TexturePlace, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::{engine_dir, fixtures_dir, llvm_assemble_text, AssembleError};
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

    /// A Shared buffer of `LEN` bytes in a block of its own, and a texture of `usage` over it at
    /// index 4096, one alignment past the device's own pitch: an image of its own.
    fn copied(name: &str, device: &Device, usage: TextureUsage) -> (Block, Buffer, Texture) {
        let cpu =
            (device.take_block(MemoryClass::Shared, 1)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let buffer =
            (device.new_buffer(cpu.id, 0, LEN, Some(0))).unwrap_or_else(|e| panic!("{name}: {e}"));
        (device.grant_indices(IndexClass::Texture, 4096, 1)).expect("a grant");
        let mut d = descriptor();
        d.usage = usage;
        let texture = (buffer.new_texture(&d, Some(4096), 0, pitch(device)))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            texture.copied_over().is_some(),
            "{name}: an image of its own"
        );
        (cpu, buffer, texture)
    }

    /// Records a fill of the whole of `buffer` with `byte`.
    fn fill_with(command_buffer: &CommandBuffer, buffer: &Buffer, byte: u8) {
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        (encoder.fill_buffer(buffer, 0..buffer.length(), byte)).expect("a fill");
        encoder.end_encoding();
    }

    /// A draw that binds a texture over a Shared buffer in an image of its own after a fill of
    /// the buffer copies it in at the draw and reads the filled texel; the next draw of the
    /// binding copies nothing. A dispatch that does not reach it after a fill copies nothing, and
    /// one that declares it (`use_resource`) copies it in at the dispatch.
    #[test]
    fn every_device_copies_in_a_texture_over_a_buffer_a_command_reaches_after_a_fill() {
        let Some(library) = draw::examples() else {
            return;
        };
        let d = draw::descriptor(&library, "first_texel_fragment", PixelFormat::RGBA8Unorm);
        each_device("over reached", |name, _, device| {
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let state = draw::state(name, &device, &d);
            let (cpu, buffer, texture) = copied(name, &device, TextureUsage::SHADER_READ);
            let mut buffers = draw::Buffers::new(&device);
            let (positions, colors) = draw::vertices(&device, &mut buffers);
            let color = draw::target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill_with(&command_buffer, &buffer, 0xC3);
            let mut encoder = (command_buffer.render_command_encoder(&draw::pass(&color, None)))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            ok(encoder.set_render_pipeline_state(&state));
            ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            ok(encoder.set_fragment_texture(Some(&texture), 0));
            let before = copies(&device);
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            assert_eq!(
                copies(&device) - before,
                1,
                "{name}: in at the draw, after the fill"
            );
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            assert_eq!(
                copies(&device) - before,
                1,
                "{name}: nothing at the next draw"
            );
            encoder.end_encoding();
            draw::run(name, &command_buffer);
            // The first texel, 0xC3 a channel, plus the draws' blue.
            let pixels = draw::read(&color, 4);
            assert!(
                pixels
                    .chunks(4)
                    .all(|pixel| pixel == [0xC3, 0xC3, 0xFF, 0xFF]),
                "{name}: the filled texel read: {:?}",
                &pixels[..4]
            );

            let mut arena = Arena::new(&device);
            let words = arena.buffer();
            let kernel = step(&device);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill_with(&command_buffer, &buffer, 0x3C);
            let mut encoder =
                (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
            ok(encoder.set_compute_pipeline_state(&kernel));
            ok(encoder.set_buffer(Some(&words), 0, 0));
            let before = copies(&device);
            ok(encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1]));
            assert_eq!(
                copies(&device) - before,
                0,
                "{name}: nothing at a dispatch that does not reach it"
            );
            ok(encoder.use_resource(Resource::Texture(&texture), ResourceUsage::READ));
            ok(encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1]));
            assert_eq!(
                copies(&device) - before,
                1,
                "{name}: in at the dispatch that declares it"
            );
            encoder.end_encoding();
            draw::run(name, &command_buffer);
            drop((
                texture, buffer, words, arena, kernel, positions, colors, buffers,
            ));
            drop((color, state, queue));
            (device.release_block(cpu)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }

    /// A draw that reaches a texture over a Shared buffer in an image of its own only through the
    /// render encoder's `use_resource` takes the texture's copies as a bound one does. Declared
    /// read after a fill of the buffer, it is copied in at the encoder's next draw, which does not
    /// bind it, and not again at the draw after; a draw before the declaration copies nothing.
    /// Declared written, the texel a draw's fragment function writes through its resource ID (an
    /// argument buffer's `texture2d<float, write>`) is in the buffer for the host once the
    /// command buffer completes, over the bytes a fill in the command buffer left.
    #[test]
    fn every_device_copies_a_texture_over_a_buffer_a_draw_declares() {
        let Some(library) = draw::examples() else {
            return;
        };
        let fragment = match llvm_assemble_text(ARGUMENT_FRAGMENT) {
            Ok(bitcode) => Library::from_air(&bitcode).expect("the library"),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("the argument fragment: {error:?}"),
        };
        let plain = draw::descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let mut writing = RenderPipelineDescriptor::new(StageSource::Function(
            FunctionDescriptor::new(library.function("colored_vertex").expect("the function")),
        ));
        writing.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            (fragment.function("argument_write_fragment")).expect("the function"),
        )));
        writing.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        each_device("over declared", |name, _, device| {
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let plain = draw::state(name, &device, &plain);
            let writing = draw::state(name, &device, &writing);
            let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let (cpu, buffer, texture) = copied(name, &device, usage);
            let mut buffers = draw::Buffers::new(&device);
            let (positions, colors) = draw::vertices(&device, &mut buffers);
            let argument = buffers.of(&device, &[0; 8]);
            place_id(&argument, &texture);
            let color = draw::target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");

            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill_with(&command_buffer, &buffer, 0xC3);
            let mut encoder = (command_buffer.render_command_encoder(&draw::pass(&color, None)))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            ok(encoder.set_render_pipeline_state(&plain));
            ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            let before = copies(&device);
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            assert_eq!(
                copies(&device) - before,
                0,
                "{name}: nothing at a draw before the declaration"
            );
            ok(encoder.use_resource(Resource::Texture(&texture), ResourceUsage::READ));
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            assert_eq!(
                copies(&device) - before,
                1,
                "{name}: in at the draw after the declaration"
            );
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            assert_eq!(
                copies(&device) - before,
                1,
                "{name}: nothing at the next draw"
            );
            encoder.end_encoding();
            draw::run(name, &command_buffer);

            let command_buffer = queue.command_buffer().expect("a command buffer");
            fill_with(&command_buffer, &buffer, 0x11);
            let mut encoder = (command_buffer.render_command_encoder(&draw::pass(&color, None)))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            ok(encoder.use_resource(Resource::Texture(&texture), ResourceUsage::WRITE));
            ok(encoder.set_render_pipeline_state(&writing));
            ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            ok(encoder.set_fragment_buffer(Some(&argument), 0, 0));
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            draw::run(name, &command_buffer);
            let bytes = held(&buffer);
            let mut want = vec![0x11; ROW as usize];
            want[..4].copy_from_slice(&[255, 0, 255, 255]);
            assert_eq!(
                bytes[..ROW as usize],
                want[..],
                "{name}: the written texel over the filled row, in the buffer"
            );
            println!("{name}: a declared texture copied in at its draw and back after its write");
            drop((texture, buffer, argument, positions, colors, buffers));
            drop((color, plain, writing, queue));
            (device.release_block(cpu)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }

    /// A draw whose vertex function stores into a Shared buffer (the owned fixture
    /// `engine/rvc_draw_bases.ll`'s `draw_bases_in_a_vertex_function`, rasterization disabled)
    /// makes the image of a texture over it stale: the next use copies it in again, at the use,
    /// and takes the stores, whether the texture's first use in the command buffer came before
    /// the draw or after it.
    #[test]
    fn every_device_copies_in_again_a_texture_over_a_buffer_a_draw_stores_into() {
        let Some(engine) = draw::library(&engine_dir().join("rvc_draw_bases.ll")) else {
            return;
        };
        let function = (engine.function("draw_bases_in_a_vertex_function")).expect("the function");
        let mut d =
            RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.rasterization_enabled = false;
        each_device("over stored", |name, _, device| {
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let state = draw::state(name, &device, &d);
            let (cpu, buffer, texture) = copied(name, &device, TextureUsage::SHADER_READ);
            let pitch = pitch(&device);
            let scratch = plain(name, &device, &texels(0));
            let color = draw::target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            for (round, first) in [(0u8, true), (1, false)] {
                let mut now: Vec<u8> = (0..LEN).map(|i| (i % 251) as u8 ^ round).collect();
                write(&buffer, &now);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                if first {
                    blit(&command_buffer, &texture, &scratch);
                }
                let mut encoder = (command_buffer
                    .render_command_encoder(&draw::pass(&color, None)))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
                ok(encoder.set_render_pipeline_state(&state));
                ok(encoder.set_vertex_buffer(Some(&buffer), 0, 0));
                // Three vertices of one instance from 0: slots 0 to 2, (vid, 0, 0, 0) each.
                ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                let before = copies(&device);
                blit(&command_buffer, &texture, &scratch);
                assert_eq!(
                    copies(&device) - before,
                    1,
                    "{name}: round {round}: in at the use after the draw"
                );
                command_buffer.commit().expect("a commit");
                finished(name, &command_buffer);
                assert_eq!(command_buffer.error(), None, "{name}: round {round}");
                for vid in 0..3u32 {
                    let slot = vid as usize * 16;
                    let stored: Vec<u8> = [vid, 0, 0, 0]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect();
                    now[slot..slot + 16].copy_from_slice(&stored);
                }
                assert!(
                    held(&buffer) == now,
                    "{name}: round {round}: the draw's stores in the buffer"
                );
                assert!(
                    image(name, &scratch) == rows(&now, 0, pitch),
                    "{name}: round {round}: the use after the draw takes its stores"
                );
            }
            drop((texture, scratch, buffer, color, state, queue));
            (device.release_block(cpu)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }

    /// After a blit wrote a texture over a Shared buffer in an image of its own: a draw that reads
    /// no buffer under it copies nothing; one that reads the buffer as a vertex buffer takes the
    /// image back first, at the draw, once; and, after a second blit wrote it, a draw that reads
    /// through an argument buffer (the owned fixture `examples/heap_draws.ll`'s
    /// `read_argument_fragment`, an address outside its resources) takes it back first, at the
    /// draw, the whole segment's answer. The buffer holds the second blit's texels.
    #[test]
    fn every_device_takes_back_a_texture_over_a_buffer_before_a_draw_reads_its_buffer() {
        let Some(library) = draw::examples() else {
            return;
        };
        let Some(heap) = draw::library(&fixtures_dir().join("examples/heap_draws.ll")) else {
            return;
        };
        let colored = draw::descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                heap.function(name).expect("the function"),
            ))
        };
        let mut unbound = RenderPipelineDescriptor::new(function("cover_vertex"));
        unbound.fragment = Some(function("read_argument_fragment"));
        unbound.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        each_device("over read", |name, _, device| {
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let colored = draw::state(name, &device, &colored);
            let unbound = draw::state(name, &device, &unbound);
            let (cpu, buffer, texture) = copied(name, &device, TextureUsage::SHADER_READ);
            let pitch = pitch(&device);
            let mut now: Vec<u8> = (0..LEN).map(|i| (i % 251) as u8).collect();
            write(&buffer, &now);
            let sources = [
                plain(name, &device, &texels(0x11)),
                plain(name, &device, &texels(0x22)),
            ];
            let mut buffers = draw::Buffers::new(&device);
            let (positions, colors) = draw::vertices(&device, &mut buffers);
            let color = draw::target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            blit(&command_buffer, &sources[0], &texture);
            let mut encoder = (command_buffer.render_command_encoder(&draw::pass(&color, None)))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            ok(encoder.set_render_pipeline_state(&colored));
            ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            let before = copies(&device);
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            assert_eq!(
                copies(&device) - before,
                0,
                "{name}: nothing at a draw that reads no buffer under it"
            );
            ok(encoder.set_vertex_buffer(Some(&buffer), 0, 1));
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            assert_eq!(
                copies(&device) - before,
                1,
                "{name}: back at the draw that reads its buffer"
            );
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            assert_eq!(copies(&device) - before, 1, "{name}: back once");
            encoder.end_encoding();
            blit(&command_buffer, &sources[1], &texture);
            let mut encoder = (command_buffer.render_command_encoder(&draw::pass(&color, None)))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            ok(encoder.set_render_pipeline_state(&unbound));
            // The argument buffer's texture: a nil resource ID, which reads zero.
            ok(encoder.set_fragment_bytes(&[0; 8], 0));
            let before = copies(&device);
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            assert_eq!(
                copies(&device) - before,
                1,
                "{name}: back at the draw that reads outside its resources"
            );
            encoder.end_encoding();
            draw::run(name, &command_buffer);
            lay(&mut now, &texels(0x22), 0, pitch);
            assert!(
                held(&buffer) == now,
                "{name}: the second blit's texels in the buffer"
            );
            drop((texture, buffer, sources, positions, colors, buffers, color));
            drop((colored, unbound, queue));
            (device.release_block(cpu)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }

    /// User-space instructions a draw and a dispatch that bind no texture over a buffer in an
    /// image of their own, in a command buffer that keeps none of them and in one that keeps 512,
    /// bound once and unbound before the draws, the textures alive in both: equal within a
    /// tenth, as the draws and dispatches look at none of them. Printed where the kernel offers
    /// the counter (Linux).
    #[test]
    fn every_device_draws_and_dispatches_past_kept_textures_over_buffers_at_no_cost() {
        use metal2vulkan_devtools::instructions::InstructionCounter;
        const KEPT: u32 = 512;
        const COUNTED: u32 = 512;
        let Some(library) = draw::examples() else {
            return;
        };
        let d = draw::descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("over kept", |name, _, device| {
            let Ok(mut counter) = InstructionCounter::new() else {
                println!("{name}: no instruction counter here");
                return;
            };
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let state = draw::state(name, &device, &d);
            let kernel = add(&device);
            let cpu = (device.take_block(MemoryClass::Shared, 1))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let buffer = (device.new_buffer(cpu.id, 0, LEN, Some(0)))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            (device.grant_indices(IndexClass::Texture, 4096, KEPT)).expect("a grant");
            let mut sd = descriptor();
            sd.usage = TextureUsage::SHADER_READ;
            let pitch = pitch(&device);
            let kept: Vec<Texture> = (0..KEPT)
                .map(|at| {
                    (buffer.new_texture(&sd, Some(4096 + at), 0, pitch))
                        .unwrap_or_else(|e| panic!("{name}: {e}"))
                })
                .collect();
            assert!(
                kept.iter().all(|texture| texture.copied_over().is_some()),
                "{name}: images of their own"
            );
            let mut buffers = draw::Buffers::new(&device);
            let (positions, colors) = draw::vertices(&device, &mut buffers);
            let mut arena = Arena::new(&device);
            let words = [arena.buffer(), arena.buffer(), arena.buffer()];
            let color = draw::target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let mut per: Vec<[u64; 2]> = Vec::new();
            for keeps in [false, true] {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer
                    .render_command_encoder(&draw::pass(&color, None)))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
                ok(encoder.set_render_pipeline_state(&state));
                ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                if keeps {
                    for (at, texture) in kept.iter().enumerate() {
                        ok(encoder.set_fragment_texture(Some(texture), at as u32 % 128));
                    }
                    for slot in 0..128 {
                        ok(encoder.set_fragment_texture(None, slot));
                    }
                }
                // One draw first, so the counted ones record what changed alone.
                ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
                let ((), drawn) = (counter.measure(|| {
                    for _ in 0..COUNTED {
                        ok(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
                    }
                }))
                .expect("a count");
                encoder.end_encoding();
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                ok(encoder.set_compute_pipeline_state(&kernel));
                for (slot, buffer) in words.iter().enumerate() {
                    ok(encoder.set_buffer(Some(buffer), 0, slot as u32));
                }
                ok(encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1]));
                let ((), dispatched) = (counter.measure(|| {
                    for _ in 0..COUNTED {
                        ok(encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1]));
                    }
                }))
                .expect("a count");
                encoder.end_encoding();
                draw::run(name, &command_buffer);
                per.push([drawn, dispatched].map(|count| count / u64::from(COUNTED)));
            }
            let (none, keeping) = (per[0], per[1]);
            println!(
                "{name}: user-space instructions a draw and a dispatch: {} and {} keeping no \
                 texture over a buffer, {} and {} keeping {KEPT} and binding none",
                none[0], none[1], keeping[0], keeping[1]
            );
            for (what, at) in [("draw", 0), ("dispatch", 1)] {
                assert!(
                    keeping[at] * 10 <= none[at] * 11,
                    "{name}: a {what} past {KEPT} kept textures over buffers: {} instructions \
                     against {}",
                    keeping[at],
                    none[at]
                );
            }
            drop((
                kept, buffer, words, arena, positions, colors, buffers, color,
            ));
            drop((state, kernel, queue));
            (device.release_block(cpu)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }
    /// While a texture over a buffer in an image of its own lives, a dispatch that writes a
    /// buffer costs no more after its segment wrote 4096 other buffers (a fill each) than after
    /// it wrote none: what a segment wrote is a set, not a list each write searches.
    #[test]
    fn every_device_writes_past_many_written_buffers_at_no_cost() {
        use metal2vulkan_devtools::instructions::InstructionCounter;
        const WRITTEN: u64 = 4096;
        const COUNTED: u32 = 512;
        each_device("over written", |name, _, device| {
            let Ok(mut counter) = InstructionCounter::new() else {
                println!("{name}: no instruction counter here");
                return;
            };
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let kernel = add(&device);
            let (cpu, buffer, texture) = copied(name, &device, TextureUsage::SHADER_READ);
            let small = (device.take_block(MemoryClass::Shared, WRITTEN * 16))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let written: Vec<Buffer> = (0..WRITTEN)
                .map(|at| {
                    (device.new_buffer(small.id, at * 16, 16, None))
                        .unwrap_or_else(|e| panic!("{name}: {e}"))
                })
                .collect();
            let mut arena = Arena::new(&device);
            let words = [arena.buffer(), arena.buffer(), arena.buffer()];
            let queue = device.new_command_queue(4).expect("a queue");
            let mut per: Vec<u64> = Vec::new();
            for many in [false, true] {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                if many {
                    let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
                    for buffer in &written {
                        ok(encoder.fill_buffer(buffer, 0..16, 0x5a));
                    }
                    encoder.end_encoding();
                }
                let mut encoder = (command_buffer.compute_command_encoder(DispatchType::Serial))
                    .expect("an encoder");
                ok(encoder.set_compute_pipeline_state(&kernel));
                for (slot, buffer) in words.iter().enumerate() {
                    ok(encoder.set_buffer(Some(buffer), 0, slot as u32));
                }
                // One dispatch first, so the counted ones record what changed alone.
                ok(encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1]));
                let ((), dispatched) = (counter.measure(|| {
                    for _ in 0..COUNTED {
                        ok(encoder.dispatch_threads([WORDS as u32, 1, 1], [16, 1, 1]));
                    }
                }))
                .expect("a count");
                encoder.end_encoding();
                draw::run(name, &command_buffer);
                per.push(dispatched / u64::from(COUNTED));
            }
            let (none, many) = (per[0], per[1]);
            println!(
                "{name}: user-space instructions a dispatch while a texture over a buffer lives: \
                 {none} after no write in its segment, {many} after {WRITTEN} buffers written"
            );
            assert!(
                many * 10 <= none * 11,
                "{name}: a dispatch past {WRITTEN} buffers written: {many} instructions against \
                 {none}"
            );
            drop((texture, buffer, written, words, arena, kernel, queue));
            (device.release_block(small)).unwrap_or_else(|e| panic!("{name}: {e}"));
            (device.release_block(cpu)).unwrap_or_else(|e| panic!("{name}: {e}"));
        });
    }
}
