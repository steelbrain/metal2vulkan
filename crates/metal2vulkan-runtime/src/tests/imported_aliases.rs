//! Two textures over one buffer in host memory's import, as a drawer and a compositor hold a shared
//! surface: T1, a render target shaders also read, and T2, a texture shaders read, both over the
//! same bytes, each in an image of its own (host memory's import takes no image in place, on every
//! device). Under a device cfg, on every device of the cfg's kind that imports host memory, in
//! BGRA8Unorm and then in RGBA16Float, 32 by 16 texels at 256-byte rows. Three command buffers, each
//! read exact:
//!
//! - the CPU writes the pages, T1 unused, and a kernel reads T2: the CPU's texels;
//! - the CPU writes the pages again, and a kernel reads T2: the new texels;
//! - a pass clears T1 and draws its top half, then a kernel reads T2: the draw's texels, which
//!   the pages then hold at their pitch, every other byte as the CPU wrote it.
//!
//! `llvm-as` assembles the kernel's and the draw's libraries; without it the test is skipped,
//! visibly, as are devices that import no host memory.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{descriptor, examples, library, pass, run, state};
    use crate::tests::draw::on_device::{floats, Buffers};
    use crate::tests::import::on_device::{free, host_memory, releasing, wait_until};
    use crate::texture::TextureObject;
    use crate::{
        Buffer, CommandQueue, ComputePipelineDescriptor, ComputePipelineState, Device,
        DispatchType, Error, FunctionDescriptor, IndexClass, MemoryClass, PixelFormat,
        PrimitiveType, RenderPipelineState, StageSource, Texture, TextureDescriptor, TextureType,
        TextureUsage,
    };
    use metal2vulkan_devtools::fixtures_dir;
    use std::ptr::NonNull;
    use std::sync::atomic::Ordering;

    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 16;
    const PITCH: u64 = 256;

    /// A format the test runs in. A texel is four channel levels: integers below 256, the value
    /// itself in RGBA16Float (each exact in a half) and the level over 255 in BGRA8Unorm.
    #[derive(Clone, Copy, Debug)]
    enum Format {
        Half,
        Bgra8,
    }

    impl Format {
        fn pixel(self) -> PixelFormat {
            match self {
                Format::Half => PixelFormat::RGBA16Float,
                Format::Bgra8 => PixelFormat::BGRA8Unorm,
            }
        }

        /// A texel's bytes.
        fn unit(self) -> u64 {
            match self {
                Format::Half => 8,
                Format::Bgra8 => 4,
            }
        }

        /// The level of 1.0.
        fn one(self) -> u32 {
            match self {
                Format::Half => 1,
                Format::Bgra8 => 255,
            }
        }

        /// The bytes of the texel of `levels`, red first.
        fn bytes(self, [r, g, b, a]: [u32; 4]) -> Vec<u8> {
            match self {
                Format::Half => [r, g, b, a]
                    .iter()
                    .flat_map(|&level| half(level).to_le_bytes())
                    .collect(),
                Format::Bgra8 => [b, g, r, a].iter().map(|&level| level as u8).collect(),
            }
        }

        /// The level a kernel's read of a channel returned, `None` for a value no level is (a NaN
        /// among them).
        fn level(self, value: f32) -> Option<u32> {
            let level = match self {
                Format::Half => value,
                Format::Bgra8 => (value * 255.0).round(),
            };
            (level.is_finite() && level.fract() == 0.0 && (0.0..256.0).contains(&level))
                .then_some(level as u32)
        }
    }

    /// The half of `level`, an integer below 2048.
    fn half(level: u32) -> u16 {
        if level == 0 {
            return 0;
        }
        let exponent = 31 - level.leading_zeros();
        let mantissa = ((level << 10) >> exponent) & 0x3ff;
        (((exponent + 15) << 10) | mantissa) as u16
    }

    /// The CPU's texel `(x, y)` in `round`.
    fn cpu(x: u32, y: u32, round: u32) -> [u32; 4] {
        [0, 1, 2, 3].map(|c| (x * 7 + y * 13 + c * 61 + round * 29 + 5) % 256)
    }

    /// The pass's texel `(x, y)`: its top half drawn green over the clear's black.
    fn drawn(format: Format, _x: u32, y: u32) -> [u32; 4] {
        let one = format.one();
        match y < HEIGHT / 2 {
            true => [0, one, 0, one],
            false => [0, 0, 0, one],
        }
    }

    /// The pages: `texel(x, y)` at each texel's place, `pitch` bytes a row, the rest of each row
    /// `under`'s.
    fn pages(
        format: Format,
        under: &[u8],
        pitch: u64,
        texel: impl Fn(u32, u32) -> [u32; 4],
    ) -> Vec<u8> {
        let mut bytes = under.to_vec();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let at = (u64::from(y) * pitch + u64::from(x) * format.unit()) as usize;
                let texel = format.bytes(texel(x, y));
                bytes[at..at + texel.len()].copy_from_slice(&texel);
            }
        }
        bytes
    }

    /// The device's kernel that reads a texture's texels into a buffer.
    fn reader(name: &str, device: &Device) -> Option<ComputePipelineState> {
        let rows = library(&fixtures_dir().join("examples/texel_rows.ll"))?;
        let source = StageSource::Function(FunctionDescriptor::new(
            rows.function("rows_read").expect("the function"),
        ));
        let state = (device.new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(source),
            crate::tests::no_constexpr_samplers,
        ))
        .unwrap_or_else(|error| panic!("{name}: rows_read: {error}"));
        Some(state)
    }

    /// What one command buffer holds before its read of T2: a pass that clears T1 and draws its
    /// top half, or nothing.
    struct Draw<'a> {
        state: &'a RenderPipelineState,
        positions: &'a Buffer,
        colors: &'a Buffer,
    }

    /// One command buffer: `draw` into `t1` where given, then a kernel's read of `t2` into `out`;
    /// committed and complete. The levels read, a texel's four in a row.
    fn read(
        name: &str,
        queue: &CommandQueue,
        reader: &ComputePipelineState,
        (t1, t2): (&Texture, &Texture),
        draw: Option<Draw<'_>>,
        (format, out): (Format, &Buffer),
    ) -> Vec<Option<u32>> {
        let done = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        let command_buffer = queue.command_buffer().expect("a command buffer");
        if let Some(draw) = draw {
            let mut encoder = (command_buffer.render_command_encoder(&pass(t1, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            done(encoder.set_render_pipeline_state(draw.state));
            done(encoder.set_vertex_buffer(Some(draw.positions), 0, 0));
            done(encoder.set_vertex_buffer(Some(draw.colors), 0, 1));
            done(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
        }
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        done(encoder.set_compute_pipeline_state(reader));
        done(encoder.set_texture(Some(t2), 0));
        done(encoder.set_buffer(Some(out), 0, 0));
        done(encoder.set_bytes(&WIDTH.to_ne_bytes(), 1));
        done(encoder.dispatch_threads([WIDTH, HEIGHT, 1], [8, 8, 1]));
        encoder.end_encoding();
        run(name, &command_buffer);
        let contents = out.contents().expect("Shared");
        // SAFETY: the buffer's bytes, which no command writes any more.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), out.length() as usize) };
        (bytes.as_chunks::<4>().0.iter())
            .map(|value| format.level(f32::from_ne_bytes(*value)))
            .collect()
    }

    /// The levels of `texel(x, y)`, row by row.
    fn levels(texel: impl Fn(u32, u32) -> [u32; 4]) -> Vec<Option<u32>> {
        (0..HEIGHT)
            .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
            .flat_map(|(x, y)| texel(x, y).map(Some))
            .collect()
    }

    /// Panics where `got` is not `want`, naming the first channel that differs and the count.
    fn exact<T: PartialEq + std::fmt::Debug>(what: &str, got: &[T], want: &[T]) {
        assert_eq!(got.len(), want.len(), "{what}: lengths");
        if let Some(at) = (0..want.len()).find(|&at| got[at] != want[at]) {
            let wrong = (0..want.len()).filter(|&at| got[at] != want[at]).count();
            panic!(
                "{what}: element {at} is {:?}, not {:?}; {wrong} of {} differ",
                got[at],
                want[at],
                want.len()
            );
        }
    }

    /// The three command buffers in `format` on `device`, over host memory it imports.
    fn shape(name: &str, device: &Device, reader: &ComputePipelineState, format: Format) {
        let Some((_, alignment)) = device.core().imports.host.as_ref() else {
            println!("imported aliases: {name}: imports no host memory; not run");
            return;
        };
        let alignment = *alignment;
        let name = &format!("{name}: {format:?}");
        let answer = (device.minimum_linear_texture_alignment(format.pixel()))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let pitch = PITCH.next_multiple_of(answer);
        let len = (u64::from(HEIGHT) * pitch).next_multiple_of(alignment);
        let (pointer, mapping) = host_memory(len, alignment);
        let write = |bytes: &[u8]| {
            // SAFETY: the test's allocation of `len` bytes, which no command reaches meanwhile.
            unsafe {
                pointer
                    .as_ptr()
                    .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len())
            };
        };
        let held = |pointer: NonNull<u8>| {
            // SAFETY: as above.
            unsafe { std::slice::from_raw_parts(pointer.as_ptr(), len as usize) }.to_vec()
        };
        let (release, released) = releasing();
        // SAFETY: the memory is the test's, allocated until it frees it after the import's
        // release.
        let (import, _) = unsafe { device.import_host_memory(pointer, len, release) }
            .unwrap_or_else(|error| panic!("{name}: the import: {error}"));
        let buffer = (device.new_buffer(import.id, 0, len, None)).expect("a buffer");
        let first = IndexClass::Texture.range().start;
        let mut d = TextureDescriptor::new(TextureType::Type2D, format.pixel(), WIDTH, HEIGHT);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let t1 = (buffer.new_texture(&d, Some(first), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: T1: {error}"));
        d.usage = TextureUsage::SHADER_READ;
        let t2 = (buffer.new_texture(&d, Some(first + 1), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: T2: {error}"));
        assert!(
            t1.copied_over().is_some() && t2.copied_over().is_some(),
            "{name}: host memory's import takes images of their own"
        );
        let texels = u64::from(WIDTH) * u64::from(HEIGHT);
        let out_block = (device.take_block(MemoryClass::Shared, 16 * texels))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let out = (device.new_buffer(out_block.id, 0, 16 * texels, Some(0))).expect("a buffer");
        let library = examples().expect("the draw's library, which llvm-as assembled");
        let draw_state = state(
            name,
            device,
            &descriptor(&library, "colored_fragment", format.pixel()),
        );
        let mut buffers = Buffers::new(device);
        // The top half of clip space (y up) in green, as `vertices` draws it.
        let positions = buffers.of(device, &floats(&[-1.0, 0.0, 3.0, 0.0, -1.0, 4.0]));
        let colors = buffers.of(device, &floats(&[0.0, 1.0, 0.0, 1.0].repeat(3)));
        let queue = device.new_command_queue(4).expect("a queue");
        let counters = device.counters();
        let before = counters.metrics();

        let mut under = held(pointer);
        for round in 0..2 {
            under = pages(format, &under, pitch, |x, y| cpu(x, y, round));
            write(&under);
            let got = read(name, &queue, reader, (&t1, &t2), None, (format, &out));
            let what = format!("{name}: round {round}: T2's read of the CPU's texels");
            exact(&what, &got, &levels(|x, y| cpu(x, y, round)));
        }
        let draw = Draw {
            state: &draw_state,
            positions: &positions,
            colors: &colors,
        };
        let got = read(name, &queue, reader, (&t1, &t2), Some(draw), (format, &out));
        exact(
            &format!("{name}: T2's read of T1's draw"),
            &got,
            &levels(|x, y| drawn(format, x, y)),
        );
        exact(
            &format!("{name}: the pages after T1's draw"),
            &held(pointer),
            &pages(format, &under, pitch, |x, y| drawn(format, x, y)),
        );
        let after = counters.metrics();
        println!(
            "imported aliases: {name}: pitch {pitch}: {} copies, {} over the import; the CPU's \
             texels twice and T1's draw read exact through T2",
            after.over_copies - before.over_copies,
            after.over_copies_imported - before.over_copies_imported,
        );
        drop((queue, t1, t2, buffer, import, out, out_block, buffers));
        wait_until("the release", || released.load(Ordering::SeqCst));
        free(mapping);
    }

    #[test]
    fn two_textures_over_imported_host_memory_read_the_cpus_texels_and_each_others_draw() {
        if examples().is_none() {
            return;
        }
        each_device("imported aliases", |name, _, device| {
            let Some(reader) = reader(name, &device) else {
                return;
            };
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 2)).expect("a grant");
            // BGRA8Unorm first: the RGBA16Float images then take places the first run's freed.
            for format in [Format::Bgra8, Format::Half] {
                shape(name, &device, &reader, format);
            }
        });
    }
}
