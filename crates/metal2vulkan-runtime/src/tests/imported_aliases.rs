//! Two textures over the same bytes of host memory's import, as a drawer and a compositor hold a
//! shared surface: T1, a render target shaders also read, and T2, a texture shaders read, made
//! second and first, over one buffer, and apart, each over a buffer of its own over the same bytes
//! of the import; each in an image of its own (host memory's import takes no image in place, on
//! every device). Under a device cfg, on every device of the cfg's kind that imports host memory, in
//! BGRA8Unorm and then in RGBA16Float, 32 by 16 texels at 256-byte rows. Three command buffers, each
//! read exact:
//!
//! - the CPU writes the pages, T1 unused, and a kernel reads T2: the CPU's texels;
//! - the CPU writes the pages again, and a kernel reads T2: the new texels;
//! - a pass clears T1 and draws its top half, then a kernel reads T2: the draw's texels, which
//!   the pages then hold at their pitch, every other byte as the CPU wrote it.
//!
//! And a compositor's process, in the order it makes things: a composite in BGRA8Unorm whose
//! textures all go, then one in RGBA16Float whose own images may take places the first freed,
//! alone and beside a thread that makes, clears and drops render targets on a queue of its own.
//! No two live textures' places meet, each target reads the CPU's texels through the texture
//! over the import, and in the order the queue runs the command buffers every image moves home
//! once, before any command names it.
//!
//! `llvm-as` assembles the kernel's and the draw's libraries; without it the tests are skipped,
//! visibly, as are devices that import no host memory.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::device::lock;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::tests::draw::on_device::{descriptor, examples, library, pass, run, state};
    use crate::tests::draw::on_device::{floats, Buffers};
    use crate::tests::import::on_device::{free, host_memory, releasing, wait_until};
    use crate::tests::layouts::{moves, Recording};
    use crate::texture::TextureObject;
    use crate::{
        BlockId, Buffer, ClearColor, CommandBuffer, CommandQueue, ComputePipelineDescriptor,
        ComputePipelineState, Device, DispatchType, Error, FunctionDescriptor, IndexClass,
        MemoryClass, PixelFormat, PrimitiveType, RenderPipelineState, StageSource, StorageMode,
        Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::fixtures_dir;
    use std::collections::VecDeque;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::thread;

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
        rows(name, &command_buffer, reader, (t2, out), [WIDTH, HEIGHT]);
        run(name, &command_buffer);
        read_out(format, out)
    }

    /// A kernel's read of `texture`'s `size` texels into the buffer at index 0, `out`, in
    /// `command_buffer`.
    fn rows(
        name: &str,
        command_buffer: &CommandBuffer,
        reader: &ComputePipelineState,
        (texture, out): (&Texture, &Buffer),
        [width, height]: [u32; 2],
    ) {
        let done = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut encoder =
            (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
        done(encoder.set_compute_pipeline_state(reader));
        done(encoder.set_texture(Some(texture), 0));
        done(encoder.set_buffer(Some(out), 0, 0));
        done(encoder.set_bytes(&width.to_ne_bytes(), 1));
        done(encoder.dispatch_threads([width, height, 1], [8, 8, 1]));
        encoder.end_encoding();
    }

    /// The levels a kernel's read wrote to `out`, a texel's four in a row.
    fn read_out(format: Format, out: &Buffer) -> Vec<Option<u32>> {
        let contents = out.contents().expect("Shared");
        // SAFETY: the buffer's bytes, which no command writes any more.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), out.length() as usize) };
        (bytes.as_chunks::<4>().0.iter())
            .map(|value| format.level(f32::from_ne_bytes(*value)))
            .collect()
    }

    /// The levels of `texel(x, y)`, row by row.
    fn levels(texel: impl Fn(u32, u32) -> [u32; 4]) -> Vec<Option<u32>> {
        levels_of([WIDTH, HEIGHT], texel)
    }

    /// The levels of `texel(x, y)` over `size`, row by row.
    fn levels_of(
        [width, height]: [u32; 2],
        texel: impl Fn(u32, u32) -> [u32; 4],
    ) -> Vec<Option<u32>> {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
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

    /// The three command buffers in `format` on `device`, over host memory it imports: the
    /// textures over one buffer, or `apart`, each over a buffer of its own.
    fn shape(
        name: &str,
        device: &Device,
        reader: &ComputePipelineState,
        (format, apart): (Format, bool),
    ) {
        let Some((_, alignment)) = device.core().imports.host.as_ref() else {
            println!("imported aliases: {name}: imports no host memory; not run");
            return;
        };
        let alignment = *alignment;
        let name = &format!("{name}: {format:?}, apart {apart}");
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
        // T2's buffer, and T1's: the same one, or, apart, a second over the same bytes.
        let b2 = (device.new_buffer(import.id, 0, len, None)).expect("a buffer");
        let b1 = match apart {
            true => (device.new_buffer(import.id, 0, len, None)).expect("a buffer"),
            false => b2.clone(),
        };
        let first = IndexClass::Texture.range().start;
        let mut d = TextureDescriptor::new(TextureType::Type2D, format.pixel(), WIDTH, HEIGHT);
        d.usage = TextureUsage::SHADER_READ;
        let t2 = (b2.new_texture(&d, Some(first + 1), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: T2: {error}"));
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let t1 = (b1.new_texture(&d, Some(first), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: T1: {error}"));
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
        drop((queue, t1, t2, b1, b2, import, out, out_block, buffers));
        wait_until(&device.counters(), "the release", || {
            released.load(Ordering::SeqCst)
        });
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
            for apart in [false, true] {
                for format in [Format::Bgra8, Format::Half] {
                    shape(name, &device, &reader, (format, apart));
                }
            }
        });
    }

    /// The targets' side.
    const SIDE: u32 = 8;

    /// The passes' clear colour.
    fn backdrop() -> ClearColor {
        ClearColor::new(0.0, 0.125, 0.25, 1.0)
    }

    /// A texture's place: its block and its bytes there, `start..end`.
    type Range = (BlockId, u64, u64);

    /// Whether two places meet.
    fn meet((block, start, end): Range, (other, from, to): Range) -> bool {
        block == other && start < to && from < end
    }

    /// The places of the live textures of a device, each under a number of its own.
    #[derive(Default)]
    struct Live {
        places: Mutex<Vec<(usize, Range)>>,
        next: AtomicUsize,
        /// The places entered.
        entered: AtomicUsize,
    }

    impl Live {
        /// Enters `texture` under a new number, which it returns, and its place where it has one;
        /// panics where the place meets a live texture's.
        fn enter(&self, name: &str, texture: &Texture) -> usize {
            let id = self.next.fetch_add(1, Ordering::Relaxed);
            let Some((block, start)) = texture.place() else {
                return id;
            };
            let range = (block, start, start + texture.image_size());
            let mut places = lock(&self.places);
            if let Some(&(other, met)) = places.iter().find(|&&(_, met)| meet(met, range)) {
                panic!("{name}: texture {id}'s place {range:?} meets texture {other}'s {met:?}");
            }
            places.push((id, range));
            self.entered.fetch_add(1, Ordering::Relaxed);
            id
        }

        /// The place of texture `id`, where it has one.
        fn range(&self, id: usize) -> Option<Range> {
            let places = lock(&self.places);
            (places.iter()).find_map(|&(other, range)| (other == id).then_some(range))
        }

        /// Texture `id` is gone, every command buffer that used it complete.
        fn leave(&self, id: usize) {
            lock(&self.places).retain(|&(other, _)| other != id);
        }
    }

    /// The places one composite's textures held: the texture over the compositor's buffer's,
    /// and every one's.
    struct Held {
        read: Option<Range>,
        all: Vec<Range>,
    }

    /// A compositor's composite in `format` on `device`, over host memory it imports holding the
    /// CPU's texels: T75, a texture shaders read over the compositor's buffer, then T11, a
    /// render target shaders also read over the drawer's buffer over the same bytes, never used,
    /// each in an image of its own;
    /// two Private targets, a reference in place over a block of its own, and one command buffer
    /// whose two passes, over the backdrop, each draw T75's texel at each pixel into a target,
    /// the first target first. Each target then reads the CPU's texels. Each texture's place is
    /// entered in `live` as it is made, and left once nothing uses it.
    fn composite(
        name: &str,
        device: &Device,
        reader: &ComputePipelineState,
        (live, first): (&Live, u32),
        format: Format,
    ) -> Held {
        let alignment = (device.core().imports.host.as_ref()).map_or(1, |&(_, a)| a);
        let name = &format!("{name}: composite in {format:?}");
        let done = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        let answer = (device.minimum_linear_texture_alignment(format.pixel()))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let pitch = PITCH.next_multiple_of(answer);
        let len = (u64::from(HEIGHT) * pitch).next_multiple_of(alignment);
        let (pointer, mapping) = host_memory(len, alignment);
        // SAFETY: the test's allocation of `len` bytes, which no command reaches yet.
        let under = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), len as usize) };
        let bytes = pages(format, under, pitch, |x, y| cpu(x, y, 0));
        // SAFETY: as above.
        unsafe { (pointer.as_ptr()).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len()) };
        let (release, released) = releasing();
        // SAFETY: the memory is the test's, allocated until it frees it after the import's
        // release.
        let (import, _) = unsafe { device.import_host_memory(pointer, len, release) }
            .unwrap_or_else(|error| panic!("{name}: the import: {error}"));
        let compositor = (device.new_buffer(import.id, 0, len, None)).expect("a buffer");
        let drawer = (device.new_buffer(import.id, 0, len, None)).expect("a buffer");
        let mut d = TextureDescriptor::new(TextureType::Type2D, format.pixel(), WIDTH, HEIGHT);
        d.usage = TextureUsage::SHADER_READ;
        let read = (compositor.new_texture(&d, Some(first), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: T75: {error}"));
        let read_id = live.enter(name, &read);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let drawn = (drawer.new_texture(&d, Some(first + 4), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: T11: {error}"));
        let drawn_id = live.enter(name, &drawn);
        assert!(
            read.copied_over().is_some() && drawn.copied_over().is_some(),
            "{name}: host memory's import takes images of their own"
        );
        let library = examples().expect("the draw's library, which llvm-as assembled");
        let reading = state(
            name,
            device,
            &descriptor(&library, "own_pixel_fragment", format.pixel()),
        );
        let mut buffers = Buffers::new(device);
        let positions = buffers.of(device, &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]));
        let colors = buffers.of(device, &floats(&[0.0; 12]));
        let mut t = TextureDescriptor::new(TextureType::Type2D, format.pixel(), SIDE, SIDE);
        t.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        t.storage_mode = StorageMode::Private;
        let targets = [first + 1, first + 2].map(|index| {
            (device.new_texture(&t, Some(index)))
                .unwrap_or_else(|error| panic!("{name}: a target: {error}"))
        });
        let target_ids = targets.each_ref().map(|target| live.enter(name, target));
        // The reference, in place over a block of its own.
        let reference_len = u64::from(SIDE) * pitch;
        let reference_block = (device.take_block(MemoryClass::Shared, reference_len))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let reference_buffer =
            (device.new_buffer(reference_block.id, 0, reference_len, Some(0))).expect("a buffer");
        let mut r = t;
        r.usage = TextureUsage::SHADER_READ;
        r.storage_mode = StorageMode::Shared;
        let reference = (reference_buffer.new_texture(&r, Some(first + 3), 0, pitch))
            .unwrap_or_else(|error| panic!("{name}: the reference: {error}"));
        let reference_id = live.enter(name, &reference);
        let queue = device.new_command_queue(4).expect("a queue");

        // One command buffer, both passes, the first target first.
        let command_buffer = queue.command_buffer().expect("a command buffer");
        for target in &targets {
            let mut p = pass(target, None);
            p.color_attachments[0].clear_color = backdrop();
            let mut encoder = (command_buffer.render_command_encoder(&p))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            done(encoder.set_render_pipeline_state(&reading));
            done(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            done(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            done(encoder.set_fragment_texture(Some(&read), 0));
            done(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
        }
        run(name, &command_buffer);
        let texels = u64::from(SIDE * SIDE);
        let out_block = (device.take_block(MemoryClass::Shared, 16 * texels))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let out = (device.new_buffer(out_block.id, 0, 16 * texels, Some(0))).expect("a buffer");
        for (index, target) in targets.iter().enumerate() {
            let command_buffer = queue.command_buffer().expect("a command buffer");
            rows(name, &command_buffer, reader, (target, &out), [SIDE, SIDE]);
            run(name, &command_buffer);
            exact(
                &format!("{name}: target {index}'s texels, T75's at each pixel"),
                &read_out(format, &out),
                &levels_of([SIDE, SIDE], |x, y| cpu(x, y, 0)),
            );
        }
        let ids = [
            read_id,
            drawn_id,
            target_ids[0],
            target_ids[1],
            reference_id,
        ];
        let held = Held {
            read: live.range(read_id),
            all: ids.iter().filter_map(|&id| live.range(id)).collect(),
        };
        for id in ids {
            live.leave(id);
        }
        drop((command_buffer, queue, read, drawn, targets, reference));
        drop(reference_buffer);
        drop((compositor, drawer, import, out, out_block, buffers));
        (device.release_block(reference_block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        wait_until(&device.counters(), "the release", || {
            released.load(Ordering::SeqCst)
        });
        free(mapping);
        held
    }

    /// Until `stop`, on a queue of its own: Private render targets of three sides in turn, each
    /// cleared to the backdrop by a pass, at most four in flight, each gone once its command
    /// buffer completes; their places entered in `live`. The count made.
    fn churn(name: &str, device: &Device, live: &Live, stop: &AtomicBool) -> usize {
        let name = &format!("{name}: churn");
        let queue = device.new_command_queue(8).expect("a queue");
        let mut flight = VecDeque::new();
        let mut made = 0;
        let land = |(id, target, command_buffer): (usize, Texture, CommandBuffer)| {
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            live.leave(id);
            drop(target);
        };
        while !stop.load(Ordering::Relaxed) {
            let side = [SIDE, WIDTH, 64][made % 3];
            let format = [PixelFormat::RGBA16Float, PixelFormat::BGRA8Unorm][made % 2];
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, side, side);
            d.usage = TextureUsage::RENDER_TARGET;
            d.storage_mode = StorageMode::Private;
            let target = (device.new_texture_for_views(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let id = live.enter(name, &target);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut p = pass(&target, None);
            p.color_attachments[0].clear_color = backdrop();
            let encoder = (command_buffer.render_command_encoder(&p))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            encoder.end_encoding();
            command_buffer.commit().expect("a commit");
            flight.push_back((id, target, command_buffer));
            made += 1;
            if flight.len() > 4 {
                land(flight.pop_front().expect("one in flight"));
            }
        }
        flight.into_iter().for_each(land);
        made
    }

    /// A compositor's process: the composite in BGRA8Unorm, every texture of it gone, sixteen
    /// 1-by-1 BGRA8Unorm textures, then the composite in RGBA16Float, whose own images may take
    /// places the first freed; alone, and then beside a second thread that makes, clears and
    /// drops render targets on a queue of its own. On every device of the cfg's kind that imports
    /// host memory, with the block-maker thread running:
    ///
    /// - no two live textures' places meet (the places rt recorded for their images);
    /// - each target reads the CPU's texels through T75, in each format;
    /// - in the order the queue runs the command buffers, every image the device made moves home
    ///   once, from the undefined or the preinitialized layout, and no command names it before
    ///   that move: a copy into an own image among them (`tests::layouts::moves`).
    ///
    /// Whether T75's RGBA16Float image takes a place the BGRA8Unorm composite held follows the
    /// driver's sizes and alignments, so it is printed, with the places, and never asserted.
    #[test]
    fn own_images_over_imported_memory_move_home_once_before_their_copies_in_places_of_their_own() {
        if examples().is_none() {
            return;
        }
        let _recording = Recording::start();
        each_device("own image moves", |name, _, device| {
            let Some(reader) = reader(name, &device) else {
                return;
            };
            if device.core().imports.host.is_none() {
                println!("own image moves: {name}: imports no host memory; not run");
                return;
            }
            let handle = device.core().device.handle();
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 5)).expect("a grant");
            let live = Live::default();
            let textures = || device.counters().gauges().textures;
            for beside in [false, true] {
                let stop = AtomicBool::new(false);
                let made = thread::scope(|scope| {
                    let churned =
                        beside.then(|| scope.spawn(|| churn(name, &device, &live, &stop)));
                    let before = textures();
                    let freed = composite(name, &device, &reader, (&live, first), Format::Bgra8);
                    if !beside {
                        wait_until(&device.counters(), "the composite's frees", || {
                            textures() == before
                        });
                    }
                    let mut d =
                        TextureDescriptor::new(TextureType::Type2D, PixelFormat::BGRA8Unorm, 1, 1);
                    d.usage = TextureUsage::RENDER_TARGET;
                    let checks: Vec<(usize, Texture)> = (0..16)
                        .map(|_| {
                            let check = (device.new_texture_for_views(&d))
                                .unwrap_or_else(|error| panic!("{name}: {error}"));
                            (live.enter(name, &check), check)
                        })
                        .collect();
                    let held = composite(name, &device, &reader, (&live, first), Format::Half);
                    // Where the driver's sizes and alignments put it is the places' policy, not
                    // an invariant: printed, never asserted.
                    let reused = (held.read)
                        .is_some_and(|read| freed.all.iter().any(|&range| meet(range, read)));
                    eprintln!(
                        "own image moves: {name}: beside a churn {beside}: T75's RGBA16Float \
                         image at {:?}, in a place the BGRA8Unorm composite held: {}; it held {:?}",
                        held.read,
                        if reused { "yes" } else { "no" },
                        freed.all
                    );
                    for (id, check) in checks {
                        live.leave(id);
                        drop(check);
                    }
                    stop.store(true, Ordering::Relaxed);
                    churned.map_or(0, |churned| churned.join().expect("the churn"))
                });
                println!(
                    "own image moves: {name}: beside a churn {beside}: {made} targets churned"
                );
            }
            let moves = moves(handle);
            // Printed before the checks, so a failing run's log carries it too.
            eprintln!(
                "own image moves: {name}: {} places entered, none met; {} moves home, {} other \
                 steps, {} copies in; {} steps out of order, the first {:?}",
                live.entered.load(Ordering::Relaxed),
                moves.homes,
                moves.uses,
                moves.copies_in,
                moves.broken.len(),
                moves.broken.first()
            );
            assert!(
                moves.broken.is_empty(),
                "{name}: {} steps out of order, the first {:?}",
                moves.broken.len(),
                moves.broken.first()
            );
            assert!(
                moves.homes > 0 && moves.copies_in > 0,
                "{name}: the queue's order was taken: {moves:?}"
            );
        });
    }
}
