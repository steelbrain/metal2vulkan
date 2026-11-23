//! Heap textures over one memory, each written whole through a view of its own and read through
//! itself in turn: under a device cfg, on every device of the cfg's kind. Six textures of
//! different sizes and formats lie at offset 0 of one heap. Each is written through its view
//! (`rows_write`) and then read through itself (`rows_read`), and the next is written only after
//! that read: in one command buffer, and with each write and each read a command buffer of its
//! own. A third run takes a program's steps in separate serial compute encoders of one command
//! buffer: heap H1 holds A and C at offset 0, heap H2 holds B0 and B at offset 0; B0 is written,
//! then A through its view, then A is read while B is written through its view (`invert_texels`),
//! then C through its view, then C and B are read. A fourth and fifth take a program's shapes:
//! 2D arrays of `RGBA16Float` at offset 0 of one heap, eight of 1 to 312 layers each written and
//! read through a 2D view of its first slice, then a 512 by 512 array of one layer written through
//! a 2D view and read through another, in one command buffer and across two. Each read is the
//! texels its texture's last write wrote, as on a Mac, whose heap textures alias by the same rules.
//!
//! Vulkan's memory aliasing asks that an image over bytes another image wrote move from
//! `VK_IMAGE_LAYOUT_UNDEFINED` before it is used; a device that keeps a compression state beside
//! an image's bytes may read another image's state without that move. `llvm-as` assembles the
//! kernels' libraries; without it the tests are skipped, visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::draw::on_device::{examples, library};
    use crate::{
        Block, Buffer, CommandBuffer, ComputePipelineDescriptor, ComputePipelineState, Device,
        DispatchType, FunctionDescriptor, HeapRange, IndexClass, Library, MemoryClass, PixelFormat,
        StageSource, StorageMode, Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::fixtures_dir;

    /// Each Shared buffer's place in its block: past the largest buffer, at a multiple of every
    /// memory alignment.
    const AT: u64 = 1 << 20;

    /// As [`AT`], past a 512 by 512 texture's input.
    const BIG: u64 = 4 << 20;

    /// The six textures at offset 0 of one heap: each one's format, width and height.
    const SIX: [(PixelFormat, u32, u32); 6] = [
        (PixelFormat::RGBA16Float, 128, 64),
        (PixelFormat::RGBA32Float, 64, 64),
        (PixelFormat::R32Float, 200, 96),
        (PixelFormat::RG32Float, 48, 72),
        (PixelFormat::RGBA8Unorm, 256, 40),
        (PixelFormat::RGBA16Float, 96, 160),
    ];

    /// The kernels: `rows_write` and `rows_read` of `examples/texel_rows`, and `invert_texels` of
    /// the runtime's examples.
    struct Kernels {
        write: ComputePipelineState,
        read: ComputePipelineState,
        invert: ComputePipelineState,
    }

    /// The libraries, or None where `llvm-as` is missing.
    fn libraries() -> Option<(Library, Library)> {
        let rows = library(&fixtures_dir().join("examples/texel_rows.ll"))?;
        Some((rows, examples()?))
    }

    fn kernels(name: &str, device: &Device, (rows, examples): &(Library, Library)) -> Kernels {
        let state = |library: &Library, function: &str| {
            let source = StageSource::Function(FunctionDescriptor::new(
                library.function(function).expect("the function"),
            ));
            (device.new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(source),
                crate::tests::no_constexpr_samplers,
            ))
            .unwrap_or_else(|error| panic!("{name}: {function}: {error}"))
        };
        Kernels {
            write: state(rows, "rows_write"),
            read: state(rows, "rows_read"),
            invert: state(examples, "invert_texels"),
        }
    }

    /// Channel `c`'s byte of texel `(x, y)` of the texture numbered `t`.
    fn byte(t: u32, x: u32, y: u32, c: u32) -> u32 {
        (x * 7 + y * 13 + c * 61 + 5 + t * 29) % 256
    }

    /// The value written to channel `c` of texel `(x, y)` of texture `t` of `format`: the byte
    /// over 255 for `RGBA8Unorm`, over 64 for a float format, so each format holds it exactly.
    fn value(format: PixelFormat, t: u32, x: u32, y: u32, c: u32) -> f32 {
        let b = byte(t, x, y, c) as f32;
        match format {
            PixelFormat::RGBA8Unorm => b / 255.0,
            _ => b / 64.0,
        }
    }

    /// What a read of channel `c` of a texel whose written channels are `written` returns: the
    /// channel where the format holds it, 0 for a missing green or blue, 1 for a missing alpha.
    fn held(format: PixelFormat, written: [f32; 4], c: usize) -> f32 {
        let channels = match format {
            PixelFormat::R32Float => 1,
            PixelFormat::RG32Float => 2,
            _ => 4,
        };
        match (c < channels, c) {
            (true, _) => written[c],
            (false, 3) => 1.0,
            (false, _) => 0.0,
        }
    }

    /// Whether a read `got` is the `want` the format holds: the same byte for `RGBA8Unorm`, the
    /// same float otherwise.
    fn same(format: PixelFormat, got: f32, want: f32) -> bool {
        match format {
            PixelFormat::RGBA8Unorm => (got * 255.0).round() == (want * 255.0).round(),
            _ => got.to_bits() == want.to_bits(),
        }
    }

    /// Shared buffers in a block of their own, each `at` bytes apart.
    struct Buffers {
        block: Block,
        at: u64,
        next: u64,
    }

    impl Buffers {
        fn new(name: &str, device: &Device, count: u64) -> Buffers {
            Buffers::apart(name, device, count, AT)
        }

        fn apart(name: &str, device: &Device, count: u64, at: u64) -> Buffers {
            let block = (device.take_block(MemoryClass::Shared, count * at))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            Buffers { block, at, next: 0 }
        }

        /// The next buffer, holding `floats`.
        fn of(&mut self, name: &str, device: &Device, floats: &[f32]) -> Buffer {
            let len = (floats.len() * 4) as u64;
            assert!(len <= self.at, "{name}: a buffer past its place");
            let buffer = (device.new_buffer(self.block.id, self.next, len, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            self.next += self.at;
            let contents = buffer.contents().expect("Shared");
            // SAFETY: the buffer holds `len` bytes, and no command uses it yet.
            unsafe {
                (contents.as_ptr().cast::<f32>())
                    .copy_from_nonoverlapping(floats.as_ptr(), floats.len())
            };
            buffer
        }
    }

    /// The floats `buffer` holds.
    fn floats(buffer: &Buffer) -> Vec<f32> {
        let contents = buffer.contents().expect("Shared");
        let len = buffer.length() as usize / 4;
        // SAFETY: the buffer's bytes, which no command writes while the test reads them.
        unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<f32>(), len) }.to_vec()
    }

    /// A heap texture: its number, format and extent, the texture and its view.
    struct Placed {
        t: u32,
        format: PixelFormat,
        width: u32,
        height: u32,
        texture: Texture,
        view: Texture,
    }

    /// Texture `t` of `format`, `width` by `height`, at offset 0 of `heap`, at index `index`, and
    /// its view at `index + 1`.
    fn place(
        name: &str,
        device: &Device,
        heap: HeapRange,
        (t, format, width, height): (u32, PixelFormat, u32, u32),
        index: u32,
    ) -> Placed {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, width, height);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        d.storage_mode = StorageMode::Private;
        let fit = (device.heap_texture_size_and_align(&d))
            .unwrap_or_else(|error| panic!("{name}: texture {t}: {error}"));
        assert!(fit.size <= heap.len, "{name}: texture {t}: {fit:?}");
        let texture = (device.new_heap_texture(&d, Some(index), heap, 0))
            .unwrap_or_else(|error| panic!("{name}: texture {t}: {error}"));
        let view =
            (texture.new_texture_view(format, TextureType::Type2D, 0..1, 0..1, Some(index + 1)))
                .unwrap_or_else(|error| panic!("{name}: texture {t}'s view: {error}"));
        Placed {
            t,
            format,
            width,
            height,
            texture,
            view,
        }
    }

    /// The input `rows_write` writes into `p`: each texel's four values.
    fn input(p: &Placed) -> Vec<f32> {
        let mut floats = Vec::new();
        for y in 0..p.height {
            for x in 0..p.width {
                floats.extend((0..4).map(|c| value(p.format, p.t, x, y, c)));
            }
        }
        floats
    }

    /// Encodes, in a serial compute encoder of its own, `rows_write` of `input` into `p` through
    /// its view.
    fn write(name: &str, cb: &CommandBuffer, kernels: &Kernels, p: &Placed, input: &Buffer) {
        let done = |result: Result<(), crate::Error>| {
            result.unwrap_or_else(|error| panic!("{name}: texture {}: {error}", p.t))
        };
        let mut e = (cb.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        done(e.set_compute_pipeline_state(&kernels.write));
        done(e.set_buffer(Some(input), 0, 0));
        done(e.set_bytes(&p.width.to_ne_bytes(), 1));
        done(e.set_texture(Some(&p.view), 0));
        done(e.dispatch_threads([p.width, p.height, 1], [8, 8, 1]));
        e.end_encoding();
    }

    /// Encodes, in a serial compute encoder of its own, `rows_read` of `p` through itself into
    /// `out`.
    fn read(name: &str, cb: &CommandBuffer, kernels: &Kernels, p: &Placed, out: &Buffer) {
        let done = |result: Result<(), crate::Error>| {
            result.unwrap_or_else(|error| panic!("{name}: texture {}: {error}", p.t))
        };
        let mut e = (cb.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        done(e.set_compute_pipeline_state(&kernels.read));
        done(e.set_texture(Some(&p.texture), 0));
        done(e.set_buffer(Some(out), 0, 0));
        done(e.set_bytes(&p.width.to_ne_bytes(), 1));
        done(e.dispatch_threads([p.width, p.height, 1], [8, 8, 1]));
        e.end_encoding();
    }

    /// Where `out`, `p`'s read, is not `want`, each texel's four values: a line naming the
    /// texels that differ and the first.
    fn wrong(what: &str, p: &Placed, out: &Buffer, want: &[f32]) -> Option<String> {
        let got = floats(out);
        let differs = |at: usize| !same(p.format, got[at], want[at]);
        let first = (0..want.len()).find(|&at| differs(at))?;
        let texels = (0..want.len() / 4)
            .filter(|&texel| (0..4).any(|c| differs(texel * 4 + c)))
            .count();
        let texel = first / 4;
        Some(format!(
            "{what}: texture {} ({:?} {}x{}): {texels} of {} texels differ; first ({}, {}) channel \
             {}: {} where {} was written",
            p.t,
            p.format,
            p.width,
            p.height,
            want.len() / 4,
            texel as u32 % p.width,
            texel as u32 / p.width,
            first % 4,
            got[first],
            want[first],
        ))
    }

    /// What a read of `p` returns where its last write was `rows_write` of `written`.
    fn reads(p: &Placed, written: &[f32]) -> Vec<f32> {
        (written.as_chunks::<4>().0.iter())
            .flat_map(|texel| (0..4).map(|c| held(p.format, *texel, c)))
            .collect()
    }

    /// The six textures at offset 0 of one heap, each written through its view and read through
    /// itself in turn: in one command buffer where `one`, else each write and each read in a
    /// command buffer of its own. A line for each texture whose read is not its write.
    fn six(name: &str, device: &Device, kernels: &Kernels, one: bool) -> Vec<String> {
        let len = MemoryClass::Private.block_len();
        let block = (device.take_block(MemoryClass::Private, len))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let heap = HeapRange {
            block: block.id,
            offset: 0,
            len,
        };
        let first = 4200;
        let _grant = (device.grant_indices(IndexClass::Texture, first, 12))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let placed: Vec<Placed> = (SIX.iter().enumerate())
            .map(|(t, &(format, width, height))| {
                let at = first + 2 * t as u32;
                place(name, device, heap, (t as u32, format, width, height), at)
            })
            .collect();
        let mut buffers = Buffers::new(name, device, 12);
        let inputs: Vec<(Vec<f32>, Buffer)> = (placed.iter())
            .map(|p| {
                let floats = input(p);
                let buffer = buffers.of(name, device, &floats);
                (floats, buffer)
            })
            .collect();
        let outs: Vec<Buffer> = (inputs.iter())
            .map(|(floats, _)| buffers.of(name, device, &vec![f32::NAN; floats.len()]))
            .collect();
        let queue = device.new_command_queue(16).expect("a queue");
        let mut cbs = Vec::new();
        let mut cb = queue.command_buffer().expect("a command buffer");
        for (p, ((_, input), out)) in placed.iter().zip(inputs.iter().zip(&outs)) {
            write(name, &cb, kernels, p, input);
            if !one {
                cb.commit().expect("a commit");
                cbs.push(cb);
                cb = queue.command_buffer().expect("a command buffer");
            }
            read(name, &cb, kernels, p, out);
            if !one {
                cb.commit().expect("a commit");
                cbs.push(cb);
                cb = queue.command_buffer().expect("a command buffer");
            }
        }
        cb.commit().expect("a commit");
        cbs.push(cb);
        for cb in &cbs {
            finished(name, cb);
            assert_eq!(cb.error(), None, "{name}");
        }
        let what = match one {
            true => "one command buffer",
            false => "a command buffer each",
        };
        let wrong: Vec<String> = (placed.iter().zip(inputs.iter().zip(&outs)))
            .filter_map(|(p, ((floats, _), out))| wrong(what, p, out, &reads(p, floats)))
            .collect();
        drop((placed, inputs, outs, cbs, queue));
        (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        wrong
    }

    /// A program's steps over one memory in separate serial compute encoders of one command
    /// buffer, all `RGBA16Float`: heap H1 holds A and C at offset 0, heap H2 holds B0 and B at
    /// offset 0. B0 is written; then A through its view; then A is read while B is written through
    /// its view (`invert_texels`); then C through its view; then C and B are read. A line for
    /// each read that is not what its texture's last write wrote.
    fn steps(name: &str, device: &Device, kernels: &Kernels) -> Vec<String> {
        let len = MemoryClass::Private.block_len();
        let block = (device.take_block(MemoryClass::Private, len))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let heaps = [0, len / 2].map(|offset| HeapRange {
            block: block.id,
            offset,
            len: len / 2,
        });
        let first = 4300;
        let _grant = (device.grant_indices(IndexClass::Texture, first, 8))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let format = PixelFormat::RGBA16Float;
        let b0 = place(name, device, heaps[1], (10, format, 128, 128), first);
        let b = place(name, device, heaps[1], (11, format, 64, 64), first + 2);
        let a = place(name, device, heaps[0], (12, format, 64, 64), first + 4);
        let c = place(name, device, heaps[0], (13, format, 96, 48), first + 6);
        let mut buffers = Buffers::new(name, device, 7);
        let written = [&b0, &a, &c].map(|p| {
            let floats = input(p);
            let buffer = buffers.of(name, device, &floats);
            (floats, buffer)
        });
        let [(_, b0_in), (a_in, a_buffer), (c_in, c_buffer)] = &written;
        let unread = |p: &Placed| vec![f32::NAN; (4 * p.width * p.height) as usize];
        let (c_out, b_out) = (
            buffers.of(name, device, &unread(&c)),
            buffers.of(name, device, &unread(&b)),
        );
        let queue = device.new_command_queue(4).expect("a queue");
        let cb = queue.command_buffer().expect("a command buffer");
        write(name, &cb, kernels, &b0, b0_in);
        write(name, &cb, kernels, &a, a_buffer);
        // A read while B is written through its view.
        let done = |result: Result<(), crate::Error>| {
            result.unwrap_or_else(|error| panic!("{name}: A into B: {error}"))
        };
        let mut e = (cb.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        done(e.set_compute_pipeline_state(&kernels.invert));
        done(e.set_texture(Some(&a.texture), 0));
        done(e.set_texture(Some(&b.view), 1));
        done(e.dispatch_threads([b.width, b.height, 1], [8, 8, 1]));
        e.end_encoding();
        write(name, &cb, kernels, &c, c_buffer);
        read(name, &cb, kernels, &c, &c_out);
        read(name, &cb, kernels, &b, &b_out);
        cb.commit().expect("a commit");
        finished(name, &cb);
        assert_eq!(cb.error(), None, "{name}");
        // B holds A's texels inverted: `invert_texels` writes 1 - rgb and keeps alpha.
        let inverted: Vec<f32> = (a_in.as_chunks::<4>().0.iter())
            .flat_map(|&[r, g, b, alpha]| [1.0 - r, 1.0 - g, 1.0 - b, alpha])
            .collect();
        let what = "separate encoders";
        let wrong = [
            wrong(what, &c, &c_out, &reads(&c, c_in)),
            wrong(what, &b, &b_out, &reads(&b, &inverted)),
        ];
        drop((b0, b, a, c, written, c_out, b_out, cb, queue));
        (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        wrong.into_iter().flatten().collect()
    }

    /// The 2D arrays at offset 0 of one heap ahead of the last: each one's width, height and
    /// layers.
    const ARRAYS: [(u32, u32, u32); 8] = [
        (256, 256, 4),
        (128, 128, 8),
        (64, 64, 16),
        (32, 32, 32),
        (16, 16, 52),
        (16, 16, 312),
        (1, 1, 312),
        (1, 1, 8),
    ];

    /// A 2D array of `RGBA16Float` numbered `t`, `width` by `height` by `layers`, at offset 0 of
    /// `heap`, at index `index`, with two 2D views of its first slice at `index + 1` (written)
    /// and `index + 2` (read): the array and the views as a [`Placed`].
    fn array(
        name: &str,
        device: &Device,
        heap: HeapRange,
        (t, width, height, layers): (u32, u32, u32, u32),
        index: u32,
    ) -> (Texture, Placed) {
        let format = PixelFormat::RGBA16Float;
        let mut d = TextureDescriptor::new(TextureType::Type2DArray, format, width, height);
        d.array_length = layers;
        d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        d.storage_mode = StorageMode::Private;
        let fit = (device.heap_texture_size_and_align(&d))
            .unwrap_or_else(|error| panic!("{name}: array {t}: {error}"));
        assert!(fit.size <= heap.len, "{name}: array {t}: {fit:?}");
        let texture = (device.new_heap_texture(&d, Some(index), heap, 0))
            .unwrap_or_else(|error| panic!("{name}: array {t}: {error}"));
        let view = |at: u32| {
            (texture.new_texture_view(format, TextureType::Type2D, 0..1, 0..1, Some(at)))
                .unwrap_or_else(|error| panic!("{name}: array {t}'s view: {error}"))
        };
        let placed = Placed {
            t,
            format,
            width,
            height,
            texture: view(index + 2),
            view: view(index + 1),
        };
        (texture, placed)
    }

    /// A program's shapes over one memory: the [`ARRAYS`] at offset 0 of one heap, each written
    /// through a 2D view of its first slice and read through another, then a 512 by 512 array of
    /// one layer at the same offset written and read the same way; in one command buffer where
    /// `one`, else the arrays' steps in one and the last array's in a second. A line for each
    /// read that is not its write.
    fn arrays(name: &str, device: &Device, kernels: &Kernels, one: bool) -> Vec<String> {
        let len = MemoryClass::Private.block_len();
        let block = (device.take_block(MemoryClass::Private, len))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let heap = HeapRange {
            block: block.id,
            offset: 0,
            len,
        };
        let first = 4400;
        let count = ARRAYS.len() as u32 + 1;
        let _grant = (device.grant_indices(IndexClass::Texture, first, 3 * count))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let shapes = (ARRAYS.iter().copied()).chain([(512, 512, 1)]);
        let (arrays, placed): (Vec<Texture>, Vec<Placed>) = (shapes.enumerate())
            .map(|(t, (width, height, layers))| {
                let at = first + 3 * t as u32;
                array(
                    name,
                    device,
                    heap,
                    (20 + t as u32, width, height, layers),
                    at,
                )
            })
            .unzip();
        let mut buffers = Buffers::apart(name, device, 2 * u64::from(count), BIG);
        let inputs: Vec<(Vec<f32>, Buffer)> = (placed.iter())
            .map(|p| {
                let floats = input(p);
                let buffer = buffers.of(name, device, &floats);
                (floats, buffer)
            })
            .collect();
        let outs: Vec<Buffer> = (inputs.iter())
            .map(|(floats, _)| buffers.of(name, device, &vec![f32::NAN; floats.len()]))
            .collect();
        let queue = device.new_command_queue(4).expect("a queue");
        let mut cbs = Vec::new();
        let mut cb = queue.command_buffer().expect("a command buffer");
        let last = placed.len() - 1;
        for (at, (p, ((_, input), out))) in placed.iter().zip(inputs.iter().zip(&outs)).enumerate()
        {
            if at == last && !one {
                cb.commit().expect("a commit");
                cbs.push(cb);
                cb = queue.command_buffer().expect("a command buffer");
            }
            write(name, &cb, kernels, p, input);
            read(name, &cb, kernels, p, out);
        }
        cb.commit().expect("a commit");
        cbs.push(cb);
        for cb in &cbs {
            finished(name, cb);
            assert_eq!(cb.error(), None, "{name}");
        }
        let what = match one {
            true => "arrays in one command buffer",
            false => "arrays, the last in a second command buffer",
        };
        let wrong: Vec<String> = (placed.iter().zip(inputs.iter().zip(&outs)))
            .filter_map(|(p, ((floats, _), out))| wrong(what, p, out, &reads(p, floats)))
            .collect();
        drop((placed, arrays, inputs, outs, cbs, queue));
        (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        wrong
    }

    /// Six heap textures at offset 0 of one heap, each written through its view and read through
    /// itself in turn in one command buffer: each read is its write.
    #[test]
    fn every_device_reads_each_aliased_heap_texture_s_write_in_one_command_buffer() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("heap_alias_one", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let wrong = six(name, &device, &kernels, true);
            assert!(wrong.is_empty(), "{name}: {wrong:#?}");
            println!("heap alias: {name}: six aliased writes read back in one command buffer");
        });
    }

    /// As in one command buffer, with each write and each read in a command buffer of its own.
    #[test]
    fn every_device_reads_each_aliased_heap_texture_s_write_across_command_buffers() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("heap_alias_across", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let wrong = six(name, &device, &kernels, false);
            assert!(wrong.is_empty(), "{name}: {wrong:#?}");
            println!("heap alias: {name}: six aliased writes read back across command buffers");
        });
    }

    /// A program's steps over two heaps' aliased textures in separate encoders of one command
    /// buffer: C reads its write, and B reads A's texels inverted.
    #[test]
    fn every_device_reads_a_program_s_aliased_steps_in_separate_encoders() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("heap_alias_steps", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let wrong = steps(name, &device, &kernels);
            assert!(wrong.is_empty(), "{name}: {wrong:#?}");
            println!("heap alias: {name}: the aliased steps read back in separate encoders");
        });
    }

    /// A program's shapes, 2D arrays over one memory, in one command buffer: each read is its
    /// write.
    #[test]
    fn every_device_reads_each_aliased_array_s_write_in_one_command_buffer() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("heap_alias_arrays_one", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let wrong = arrays(name, &device, &kernels, true);
            assert!(wrong.is_empty(), "{name}: {wrong:#?}");
            println!("heap alias: {name}: the aliased arrays read back in one command buffer");
        });
    }

    /// As in one command buffer, with the last array's steps in a second.
    #[test]
    fn every_device_reads_each_aliased_array_s_write_across_command_buffers() {
        let Some(libraries) = libraries() else {
            return;
        };
        each_device("heap_alias_arrays_across", |name, _, device| {
            let kernels = kernels(name, &device, &libraries);
            let wrong = arrays(name, &device, &kernels, false);
            assert!(wrong.is_empty(), "{name}: {wrong:#?}");
            println!("heap alias: {name}: the aliased arrays read back across command buffers");
        });
    }
}
