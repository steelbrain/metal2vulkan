//! Textures over a buffer in a dma-buf's import, on every Linux device of the cfg's kind: in place
//! where the device's external image query takes the linear image made for the dma-buf, its move
//! home the acquire of the image and the buffer's range under it from the foreign queue family
//! (one counted in `Metrics::external_acquires`, no copy counted); else the counted copy (copies
//! in `Metrics::over_copies`, all of them in `Metrics::over_copies_imported`). Either way:
//!
//! - a texture whose texels the CPU wrote into the memfd is read by a kernel exact, a kernel then
//!   reads the buffer under it by its address exact, and both read the CPU's next write exact;
//! - a render target over the buffer, cleared and drawn, holds the draw's texels in the memfd,
//!   every other byte as the CPU wrote it.
//!
//! `llvm-as` assembles the kernels' and the draw's library; without it the tests are skipped,
//! visibly, as are devices that import no dma-buf and hosts without `/dev/udmabuf`.

#[cfg(all(target_os = "linux", any(m2v_gpu, m2v_cpu_icd)))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, pass, run, state, texels, vertices, Buffers, BLACK, GREEN, SIDE,
    };
    use crate::tests::import::on_device::{memfd, releasing, udmabuf, wait_until};
    use crate::texture::TextureObject;
    use crate::{
        Block, Buffer, CommandQueue, ComputePipelineDescriptor, ComputePipelineState, Device,
        DispatchType, Error, FunctionDescriptor, IndexClass, Library, Limit, MemoryClass, Metrics,
        PixelFormat, PrimitiveType, Region, StageSource, StorageMode, Texture, TextureDescriptor,
        TextureType, TextureUsage,
    };
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 8;
    const ROW: u64 = WIDTH as u64 * 4;

    /// A dma-buf of a memfd of `bytes`' length, the CPU's `bytes` written into the memfd first,
    /// imported: the block, the memfd and the release's flag. `None`, printed, where the host
    /// has no `/dev/udmabuf`, the device imports no dma-buf, or its rule takes no type for it.
    fn imported(
        test: &str,
        name: &str,
        device: &Device,
        bytes: &[u8],
    ) -> Option<(Block, File, Arc<AtomicBool>)> {
        let len = bytes.len() as u64;
        let memfd = memfd(len);
        let Some(fd) = udmabuf(&memfd, 0, len) else {
            println!("{test}: {name}: no /dev/udmabuf here; not run");
            return None;
        };
        let file = File::from(memfd);
        file.write_all_at(bytes, 0).expect("the CPU's bytes");
        if device.core().imports.dma_buf.is_none() {
            println!("{test}: {name}: imports no dma-buf; not run");
            return None;
        }
        let (release, released) = releasing();
        match device.import_dma_buf(fd, len, release) {
            Ok((block, _)) => Some((block, file, released)),
            Err(Error::Limit(Limit::ImportType { .. })) => {
                println!("{test}: {name}: no type taken for the dma-buf; not run");
                None
            }
            Err(error) => panic!("{name}: the import: {error}"),
        }
    }

    /// The memfd's `len` bytes.
    fn pages(file: &File, len: u64) -> Vec<u8> {
        let mut bytes = vec![0; len as usize];
        file.read_exact_at(&mut bytes, 0)
            .expect("the memfd's bytes");
        bytes
    }

    fn kernel(
        name: &str,
        device: &Device,
        library: &Library,
        function: &str,
    ) -> ComputePipelineState {
        let source = StageSource::Function(FunctionDescriptor::new(
            library.function(function).expect("the function"),
        ));
        (device.new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(source),
            crate::tests::no_constexpr_samplers,
        ))
        .unwrap_or_else(|error| panic!("{name}: {function}: {error}"))
    }

    /// Checks the form `texture` took against the counters moved from `before` to `after`: in
    /// place, no copy and one acquire; copied, copies, every one over the import, and no acquire.
    /// Returns the form's name.
    fn form(name: &str, texture: &Texture, before: &Metrics, after: &Metrics) -> &'static str {
        let copies = after.over_copies - before.over_copies;
        let imported = after.over_copies_imported - before.over_copies_imported;
        let acquires = after.external_acquires - before.external_acquires;
        match texture.copied_over() {
            None => {
                assert_eq!(
                    (copies, acquires),
                    (0, 1),
                    "{name}: in place: copies, acquires"
                );
                "in place"
            }
            Some(_) => {
                assert!(copies > 0, "{name}: copied, the copies counted");
                assert_eq!(
                    (imported, acquires),
                    (copies, 0),
                    "{name}: copied: copies over the import, acquires"
                );
                "copied"
            }
        }
    }

    /// The float the memfd holds at float `index` in `round`.
    fn float(index: u64, round: u32) -> f32 {
        (index + u64::from(round) * 1000) as f32
    }

    /// The memfd's `len` bytes in `round`: float `i` at byte `4 * i`.
    fn round_bytes(len: u64, round: u32) -> Vec<u8> {
        (0..len / 4)
            .flat_map(|i| float(i, round).to_le_bytes())
            .collect()
    }

    /// One command buffer: `invert_texels` from `texture` into `inverted`, then
    /// `scale_and_offset` (scale 1) from the buffer's first `floats` floats, by its address, into
    /// `sums`.
    fn dispatches(
        name: &str,
        queue: &CommandQueue,
        (invert, sum): (&ComputePipelineState, &ComputePipelineState),
        (texture, inverted): (&Texture, &Texture),
        (buffer, sums): (&Buffer, &Buffer),
        floats: u32,
    ) {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        let done = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        done(encoder.set_compute_pipeline_state(invert));
        done(encoder.set_texture(Some(texture), 0));
        done(encoder.set_texture(Some(inverted), 1));
        done(encoder.dispatch_threads([WIDTH, HEIGHT, 1], [8, 1, 1]));
        done(encoder.set_compute_pipeline_state(sum));
        done(encoder.set_buffer(Some(buffer), 0, 0));
        done(encoder.set_buffer(Some(sums), 0, 1));
        done(encoder.set_bytes(&1.0f32.to_le_bytes(), 2));
        done(encoder.dispatch_threads([floats, 1, 1], [64, 1, 1]));
        encoder.end_encoding();
        run(name, &command_buffer);
    }

    /// A texture over a dma-buf's buffer whose texels the CPU wrote is read exact by a kernel,
    /// and the buffer under it by a second kernel through its address after the texture's
    /// acquire; after the CPU writes the memfd again, both read the new bytes exact. In place, no
    /// copy is counted and one acquire; copied, the copies are counted over the import.
    #[test]
    fn every_device_reads_a_texture_over_a_dma_buf_and_the_buffer_under_it_exact() {
        let Some(library) = examples() else {
            return;
        };
        each_device("import_overs", |name, _, device| {
            let answer = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let pitch = ROW.next_multiple_of(answer);
            let len = (pitch * u64::from(HEIGHT)).next_multiple_of(4096);
            let Some((block, file, released)) =
                imported("import_overs", name, &device, &round_bytes(len, 0))
            else {
                return;
            };
            let invert = kernel(name, &device, &library, "invert_texels");
            let sum = kernel(name, &device, &library, "scale_and_offset");
            let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 2)).expect("a grant");
            let counters = device.counters();
            let before = counters.metrics();
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            d.usage = TextureUsage::SHADER_READ;
            let texture = (buffer.new_texture(&d, Some(first), 0, pitch))
                .unwrap_or_else(|error| panic!("{name}: a texture over the dma-buf: {error}"));
            assert_eq!(
                texture.descriptor().storage_mode,
                StorageMode::Shared,
                "{name}"
            );
            let mut w = d;
            w.usage = TextureUsage::SHADER_WRITE;
            w.storage_mode = StorageMode::Shared;
            let inverted = (device.new_texture(&w, Some(first + 1)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let floats = (pitch * u64::from(HEIGHT) / 4) as u32;
            let sums_block = (device.take_block(MemoryClass::Shared, u64::from(floats) * 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let sums = (device.new_buffer(sums_block.id, 0, u64::from(floats) * 4, Some(0)))
                .expect("a buffer");
            let queue = device.new_command_queue(4).expect("a queue");
            for round in 0..2 {
                if round > 0 {
                    // The CPU's next write, after the acquire: the image reads it as the buffer.
                    (file.write_all_at(&round_bytes(len, round), 0)).expect("the CPU's bytes");
                }
                dispatches(
                    name,
                    &queue,
                    (&invert, &sum),
                    (&texture, &inverted),
                    (&buffer, &sums),
                    floats,
                );
                let host = round_bytes(len, round);
                let want: Vec<u8> = (0..u64::from(HEIGHT))
                    .flat_map(|y| (0..ROW).map(move |x| (y * pitch + x) as usize))
                    .enumerate()
                    .map(|(at, byte)| match at % 4 {
                        3 => host[byte],
                        _ => 255 - host[byte],
                    })
                    .collect();
                let mut got = vec![0; want.len()];
                let region = Region {
                    origin: [0; 3],
                    size: [WIDTH, HEIGHT, 1],
                };
                (inverted.get_bytes(&mut got, ROW, ROW * u64::from(HEIGHT), region, 0, 0))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                assert!(
                    got == want,
                    "{name}: round {round}: the kernel read the CPU's texels"
                );
                let contents = sums.contents().expect("Shared");
                // SAFETY: the buffer's bytes, which no command writes any more.
                let read =
                    unsafe { std::slice::from_raw_parts(contents.as_ptr(), floats as usize * 4) };
                let expected: Vec<u8> = (0..u64::from(floats))
                    .flat_map(|i| (float(i, round) + i as f32).to_le_bytes())
                    .collect();
                assert!(
                    read == expected.as_slice(),
                    "{name}: round {round}: the kernel read the buffer under the texture"
                );
            }
            let after = counters.metrics();
            let taken = form(name, &texture, &before, &after);
            drop((queue, texture, inverted, sums, sums_block, buffer, block));
            wait_until(&device, "the release", || released.load(Ordering::SeqCst));
            println!(
                "import_overs: {name}: a texture over a dma-buf at pitch {pitch}: {taken}, {} \
                 copies, {} acquires; two rounds read exact through the texture and the buffer",
                after.over_copies - before.over_copies,
                after.external_acquires - before.external_acquires,
            );
        });
    }

    /// A render target over a dma-buf's buffer, cleared and drawn (its top half green), holds the
    /// draw's texels in the memfd, every byte that is not a texel as the CPU wrote it. In place,
    /// no copy is counted and one acquire; copied, the copies are counted over the import.
    #[test]
    fn every_device_draws_into_a_render_target_over_a_dma_buf_exact() {
        const SENTINEL: u8 = 0xC3;
        let Some(library) = examples() else {
            return;
        };
        let draw = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("import_overs draw", |name, _, device| {
            let answer = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let row = u64::from(SIDE) * 4;
            let pitch = row.next_multiple_of(answer);
            let len = (pitch * u64::from(SIDE)).next_multiple_of(4096);
            let Some((block, file, released)) = imported(
                "import_overs draw",
                name,
                &device,
                &vec![SENTINEL; len as usize],
            ) else {
                return;
            };
            let state = state(name, &device, &draw);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 1)).expect("a grant");
            let counters = device.counters();
            let before = counters.metrics();
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, SIDE, SIDE);
            d.usage = TextureUsage::RENDER_TARGET;
            let target = (buffer.new_texture(&d, Some(first), 0, pitch)).unwrap_or_else(|error| {
                panic!("{name}: a render target over the dma-buf: {error}")
            });
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&target, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let done = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            done(encoder.set_render_pipeline_state(&state));
            done(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            done(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            // The top half, green, over the clear's black.
            done(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            let taken = form(name, &target, &before, &after);
            let drawn = texels(|_, y| match y < SIDE / 2 {
                true => GREEN,
                false => BLACK,
            });
            let mut want = vec![SENTINEL; len as usize];
            for y in 0..u64::from(SIDE) {
                let (at, from) = ((y * pitch) as usize, (y * row) as usize);
                want[at..at + row as usize].copy_from_slice(&drawn[from..from + row as usize]);
            }
            assert!(
                pages(&file, len) == want,
                "{name}: {taken}: the memfd holds the draw's texels, every other byte the CPU's"
            );
            drop((command_buffer, queue, target, buffer, block));
            wait_until(&device, "the release", || released.load(Ordering::SeqCst));
            println!(
                "import_overs draw: {name}: a render target over a dma-buf at pitch {pitch}: \
                 {taken}, {} copies, {} acquires; the memfd holds the draw exact",
                after.over_copies - before.over_copies,
                after.external_acquires - before.external_acquires,
            );
        });
    }
}
