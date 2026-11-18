//! A texture over a buffer in place and the bytes around its texels: under a device cfg, on every
//! device of the cfg's kind. The image and the buffer are host-accessible aliases (the image in the
//! general layout through every use), so a write through the image touches only its texels' bytes.
//! Every byte of a span around the texture that is not one of its texels holds a sentinel: the
//! row remainders, the rows past its height, the tail, and a next buffer. The span covers the
//! image's advertised subresource range and its bound range, whose ends each run prints. Each of
//! these leaves every sentinel as the host wrote it, and the texels as the write says:
//!
//! - a clear by a load action;
//! - a draw covering the target;
//! - a blit copy into it;
//! - a fill of the buffer over exactly its texels, which a read of the image returns;
//! - a fill of a row remainder, after which a read of the image returns the texels unchanged;
//! - a kernel's writes through a texture of a format with storage over the same bytes.
//!
//! The texture is a `BGRA8Unorm` 1,000 by 3 texture, read and rendered to (the usage of a stream's
//! textures over buffers) where the device's linear tiling takes the render usage in place, else
//! read only (the clear and the draw then run on no texture in place, as the test prints). It
//! stays in place, with no copy counted. Two places:
//!
//! - (a) rows of a taller surface at the device's own pitch, from a row whose place meets the
//!   image's memory alignment;
//! - (b) a texture that ends at its buffer's last texel, with a Shared buffer at the next 4 KiB
//!   after it.
//!
//! Over (a)'s surface, a window at twice the pitch (wider than its row, as a stream's windows are)
//! is copied, its copies counted, and its blit leaves the same sentinels. `llvm-as` assembles the
//! draw's and the kernel's library; without it those two writes are skipped, visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Buffer, ClearColor, ColorAttachmentDescriptor, CommandBuffer, CommandQueue,
        ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType, FunctionDescriptor,
        IndexClass, Library, LoadAction, MemoryClass, PixelFormat, PrimitiveType, Region,
        RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState, StageSource,
        StorageMode, StoreAction, Texture, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage,
    };
    use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};

    const WIDTH: u32 = 1000;
    const HEIGHT: u32 = 3;
    const ROW: usize = WIDTH as usize * 4;
    /// The span's first place in its block: a multiple of every memory alignment.
    const AT: u64 = 65_536;
    /// The span's bytes past the last buffer: room for any bound range's tail.
    const TAIL: u64 = 65_536;
    /// The next buffer's length, in place (b).
    const NEXT: u64 = 16_384;

    /// The examples' library, or None where `llvm-as` is missing.
    fn examples() -> Option<Library> {
        let ll = fixtures_dir().join("examples/runtime_examples.ll");
        match llvm_assemble(&ll) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: the draw and the kernel: {why}");
                None
            }
            Err(error) => panic!("{}: {error:?}", ll.display()),
        }
    }

    /// The draw's state (`colored_vertex`, `colored_fragment` into one `BGRA8Unorm` attachment) and
    /// the kernel's (`invert_texels`).
    struct Pipelines {
        draw: RenderPipelineState,
        invert: ComputePipelineState,
    }

    fn pipelines(name: &str, device: &Device, library: &Library) -> Pipelines {
        let function = |name: &str| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("colored_vertex"));
        d.fragment = Some(function("colored_fragment"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        let draw = (device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let kernel = ComputePipelineDescriptor::new(function("invert_texels"));
        let invert = (device
            .new_compute_pipeline_state(&kernel, crate::tests::no_constexpr_samplers))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        Pipelines { draw, invert }
    }

    /// The bytes of `buffer`, read by the host.
    fn held(buffer: &Buffer) -> Vec<u8> {
        let contents = buffer.contents().expect("Shared");
        // SAFETY: the buffer's bytes, which no command writes while the test reads them.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
    }

    /// Writes `bytes` over `buffer`'s from its start.
    fn write(buffer: &Buffer, bytes: &[u8]) {
        let contents = buffer.contents().expect("Shared");
        // SAFETY: the buffer holds `bytes.len()` bytes, and no command reads them meanwhile.
        unsafe {
            contents
                .as_ptr()
                .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len())
        };
    }

    /// `record`'s commands in a command buffer of `queue`, committed and complete with no error.
    fn run(name: &str, queue: &CommandQueue, record: impl FnOnce(&CommandBuffer)) {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        record(&command_buffer);
        command_buffer.commit().expect("a commit");
        finished(name, &command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    fn whole() -> Region {
        Region {
            origin: [0; 3],
            size: [WIDTH, HEIGHT, 1],
        }
    }

    /// The texture's texels, rows packed, read by the device from its image.
    fn image(name: &str, texture: &Texture) -> Vec<u8> {
        let mut bytes = vec![0; ROW * HEIGHT as usize];
        (texture.image_bytes(&mut bytes, ROW as u64, whole(), 0, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        bytes
    }

    /// What the span should hold, with the texels' places: `at` in the span, rows `pitch` apart.
    struct Expected {
        bytes: Vec<u8>,
        at: usize,
        pitch: usize,
    }

    impl Expected {
        /// Each texel's bytes from `texel(x, y)`.
        fn texels(&mut self, texel: impl Fn(usize, usize) -> [u8; 4]) {
            for y in 0..HEIGHT as usize {
                for x in 0..WIDTH as usize {
                    let at = self.at + y * self.pitch + x * 4;
                    self.bytes[at..at + 4].copy_from_slice(&texel(x, y));
                }
            }
        }

        /// Whether `i` of the span is one of the texels' bytes.
        fn is_texel(&self, i: usize) -> bool {
            i >= self.at
                && (i - self.at) / self.pitch < HEIGHT as usize
                && (i - self.at) % self.pitch < ROW
        }

        /// The span holds what it should after `what`: every sentinel, and the texels.
        fn check(&self, name: &str, what: &str, span: &Buffer) {
            let now = held(span);
            if let Some(i) = (0..now.len()).find(|&i| now[i] != self.bytes[i]) {
                let differ = (0..now.len()).filter(|&i| now[i] != self.bytes[i]);
                let sentinels = differ.clone().filter(|&i| !self.is_texel(i)).count();
                panic!(
                    "{name}: after {what}: byte {i} of the span ({}) is {:#04x}, not {:#04x}; \
                     {} bytes differ, {sentinels} of them sentinels",
                    match self.is_texel(i) {
                        true => "a texel's",
                        false => "a sentinel",
                    },
                    now[i],
                    self.bytes[i],
                    differ.count()
                );
            }
        }
    }

    /// A sentinel pattern of `len` bytes, no run of it a texel's colour of 0 and 255 alone.
    fn sentinels(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 7 + 3) % 251) as u8 ^ 0xA5).collect()
    }

    /// A plain Shared texture of `format` that shaders read, at index `index`, holding `bytes`.
    fn plain(
        name: &str,
        device: &Device,
        format: PixelFormat,
        index: u32,
        bytes: &[u8],
    ) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, WIDTH, HEIGHT);
        d.usage = TextureUsage::SHADER_READ;
        d.storage_mode = StorageMode::Shared;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        (texture.replace_region(whole(), 0, 0, bytes, ROW as u64, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        texture
    }

    /// One place's run: the span `span` from the block's `AT`, the texture over `buffer` at
    /// `offset` (the span's byte `at`), rows `pitch` apart.
    struct Place<'a> {
        label: &'static str,
        span: &'a Buffer,
        buffer: &'a Buffer,
        offset: u64,
        at: usize,
        pitch: u64,
    }

    /// The writes on one place, checked after each; the textures stay in place, no copy counted.
    fn writes(name: &str, device: &Device, pipelines: Option<&Pipelines>, place: &Place<'_>) {
        let label = place.label;
        let first = IndexClass::Texture.range().start;
        let mut d =
            TextureDescriptor::new(TextureType::Type2D, PixelFormat::BGRA8Unorm, WIDTH, HEIGHT);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let made = |d: &TextureDescriptor, index: u32| {
            (place
                .buffer
                .new_texture(d, Some(index), place.offset, place.pitch))
            .unwrap_or_else(|error| panic!("{name}: {label}: {error}"))
        };
        let mut texture = made(&d, first);
        let renders = texture.copied_over().is_none();
        if !renders {
            drop(texture);
            d.usage = TextureUsage::SHADER_READ;
            texture = made(&d, first);
            println!(
                "over aliases: {name}: {label}: no render target over a buffer in place (the \
                 linear tiling's answer): the clear and the draw are not run"
            );
        }
        assert!(texture.copied_over().is_none(), "{name}: {label}: in place");
        // The image's advertised range and bound range, from the texture's first byte.
        let core = device.core();
        let color = ash::vk::ImageSubresource {
            aspect_mask: ash::vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        };
        // SAFETY: the texture's image, linear, of one level and layer of a colour format.
        let (layout, bound) = unsafe {
            (
                core.device
                    .get_image_subresource_layout(texture.image(), color),
                core.device
                    .get_image_memory_requirements(texture.image())
                    .size,
            )
        };
        let advertised = place.at as u64 + layout.offset + layout.size;
        let bound = place.at as u64 + bound;
        let (rows, buffer_end) = (
            place.at as u64 + u64::from(HEIGHT) * place.pitch,
            place.at as u64 - place.offset + place.buffer.length(),
        );
        println!(
            "over aliases: {name}: {label}: {:?} at pitch {}: its buffer ends at {buffer_end} of \
             the span, its rows at {rows}, its advertised range at {advertised}, its bound range \
             at {bound}; the span holds {}",
            d.usage,
            place.pitch,
            place.span.length()
        );
        assert!(
            advertised.max(bound) <= place.span.length(),
            "{name}: {label}: the span holds both ranges"
        );
        let mut expected = Expected {
            bytes: sentinels(place.span.length() as usize),
            at: place.at,
            pitch: place.pitch as usize,
        };
        expected.texels(|x, y| [x as u8, y as u8, (x >> 8) as u8, 0x80]);
        write(place.span, &expected.bytes);
        expected.check(name, "the host's write", place.span);
        let over_copies = || device.counters().metrics().over_copies;
        let copies = over_copies();
        let queue = device.new_command_queue(4).expect("a queue");

        if renders {
            // A clear by the load action, to blue (BGRA bytes 255, 0, 0, 255).
            run(name, &queue, |cb| {
                let mut pass = RenderPassDescriptor::default();
                let c = &mut pass.color_attachments[0];
                c.texture = Some(texture.clone());
                c.load_action = LoadAction::Clear;
                c.store_action = StoreAction::Store;
                c.clear_color = ClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 1.0,
                    alpha: 1.0,
                };
                let encoder = (cb.render_command_encoder(&pass))
                    .unwrap_or_else(|error| panic!("{name}: {label}: {error}"));
                encoder.end_encoding();
            });
            expected.texels(|_, _| [255, 0, 0, 255]);
            expected.check(name, "a clear by the load action", place.span);
        }
        if let (true, Some(pipelines)) = (renders, pipelines) {
            // A triangle covering the target, in green (BGRA bytes 0, 255, 0, 255).
            let block = (device.take_block(MemoryClass::Shared, 4096)).expect("a block");
            let positions = (device.new_buffer(block.id, 0, 24, None)).expect("a buffer");
            let colors = (device.new_buffer(block.id, 256, 48, None)).expect("a buffer");
            let floats = |values: &[f32]| -> Vec<u8> {
                values.iter().flat_map(|v| v.to_le_bytes()).collect()
            };
            write(&positions, &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]));
            write(&colors, &floats(&[0.0, 1.0, 0.0, 1.0].repeat(3)));
            run(name, &queue, |cb| {
                let mut pass = RenderPassDescriptor::default();
                let c = &mut pass.color_attachments[0];
                c.texture = Some(texture.clone());
                c.load_action = LoadAction::Load;
                c.store_action = StoreAction::Store;
                let mut encoder = (cb.render_command_encoder(&pass))
                    .unwrap_or_else(|error| panic!("{name}: {label}: {error}"));
                let done = |result: Result<(), crate::Error>| {
                    result.unwrap_or_else(|error| panic!("{name}: {label}: {error}"))
                };
                done(encoder.set_render_pipeline_state(&pipelines.draw));
                done(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                done(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                done(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
            });
            expected.texels(|_, _| [0, 255, 0, 255]);
            expected.check(name, "a draw covering the target", place.span);
            drop((positions, colors));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        }

        // A blit copy into it from a plain texture.
        let source: Vec<u8> = (0..ROW * HEIGHT as usize)
            .map(|i| ((i * 13 + 5) % 241) as u8)
            .collect();
        let from = plain(name, device, PixelFormat::BGRA8Unorm, first + 1, &source);
        run(name, &queue, |cb| {
            let mut blit = cb.blit_command_encoder().expect("an encoder");
            (blit.copy_from_texture(
                &from,
                TexturePlace::default(),
                [WIDTH, HEIGHT, 1],
                &texture,
                TexturePlace::default(),
            ))
            .unwrap_or_else(|error| panic!("{name}: {label}: {error}"));
            blit.end_encoding();
        });
        expected.texels(|x, y| {
            let at = y * ROW + x * 4;
            [source[at], source[at + 1], source[at + 2], source[at + 3]]
        });
        expected.check(name, "a blit copy into it", place.span);

        // A fill of the buffer over exactly the texels, which the image reads back.
        let pitch = place.pitch;
        run(name, &queue, |cb| {
            let mut blit = cb.blit_command_encoder().expect("an encoder");
            for y in 0..u64::from(HEIGHT) {
                let start = place.offset + y * pitch;
                (blit.fill_buffer(place.buffer, start..start + ROW as u64, 0x3C))
                    .unwrap_or_else(|error| panic!("{name}: {label}: {error}"));
            }
            blit.end_encoding();
        });
        expected.texels(|_, _| [0x3C; 4]);
        expected.check(name, "a fill over the texels", place.span);
        assert!(
            image(name, &texture) == vec![0x3C; ROW * HEIGHT as usize],
            "{name}: {label}: the image reads the fill's texels"
        );

        // A fill of the first row's remainder: the texels unchanged.
        match pitch > ROW as u64 {
            true => {
                let start = place.offset + ROW as u64;
                run(name, &queue, |cb| {
                    let mut blit = cb.blit_command_encoder().expect("an encoder");
                    (blit.fill_buffer(place.buffer, start..place.offset + pitch, 0x77))
                        .unwrap_or_else(|error| panic!("{name}: {label}: {error}"));
                    blit.end_encoding();
                });
                let remainder = place.at + ROW..place.at + pitch as usize;
                expected.bytes[remainder].fill(0x77);
                expected.check(name, "a fill of a row remainder", place.span);
                assert!(
                    image(name, &texture) == vec![0x3C; ROW * HEIGHT as usize],
                    "{name}: {label}: the image's texels after a fill beside them"
                );
            }
            false => println!(
                "over aliases: {name}: {label}: rows packed (pitch {pitch}): no row remainder"
            ),
        }

        // A kernel's writes, through a texture of a format with storage over the same bytes.
        if let Some(pipelines) = pipelines {
            let mut w =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            w.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let written = made(&w, first + 2);
            assert!(
                written.copied_over().is_none(),
                "{name}: {label}: the kernel's texture in place"
            );
            let read = plain(name, device, PixelFormat::RGBA8Unorm, first + 3, &source);
            run(name, &queue, |cb| {
                let mut encoder =
                    (cb.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
                let done = |result: Result<(), crate::Error>| {
                    result.unwrap_or_else(|error| panic!("{name}: {label}: {error}"))
                };
                done(encoder.set_compute_pipeline_state(&pipelines.invert));
                done(encoder.set_texture(Some(&read), 0));
                done(encoder.set_texture(Some(&written), 1));
                done(encoder.dispatch_threads([WIDTH, HEIGHT, 1], [8, 1, 1]));
                encoder.end_encoding();
            });
            expected.texels(|x, y| {
                let at = y * ROW + x * 4;
                let inverted = |c: u8| 255 - c;
                [
                    inverted(source[at]),
                    inverted(source[at + 1]),
                    inverted(source[at + 2]),
                    source[at + 3],
                ]
            });
            expected.check(name, "a kernel's writes", place.span);
            assert!(
                written.copied_over().is_none(),
                "{name}: {label}: the kernel's texture still in place"
            );
            drop((written, read));
        }
        assert!(
            texture.copied_over().is_none(),
            "{name}: {label}: still in place"
        );
        assert_eq!(over_copies(), copies, "{name}: {label}: no copy counted");
        println!(
            "over aliases: {name}: {label}: every sentinel kept through the writes; in place, no \
             copy counted"
        );
        drop((texture, from, queue));
    }

    /// The memory alignment of the textures the writes make over a buffer.
    fn alignment(device: &Device) -> u64 {
        let mut most = 1;
        for (format, usage) in [
            (
                PixelFormat::BGRA8Unorm,
                TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
            ),
            (PixelFormat::BGRA8Unorm, TextureUsage::SHADER_READ),
            (
                PixelFormat::RGBA8Unorm,
                TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            ),
        ] {
            let mut d = TextureDescriptor::new(TextureType::Type2D, format, WIDTH, HEIGHT);
            d.usage = usage;
            if let Some(align) = device.linear_memory_alignment(&d) {
                most = most.max(align);
            }
        }
        assert!(AT.is_multiple_of(most), "an alignment of {most}");
        most
    }

    fn gcd(a: u64, b: u64) -> u64 {
        match b {
            0 => a,
            _ => gcd(b, a % b),
        }
    }

    #[test]
    fn a_texture_over_a_buffer_in_place_writes_only_its_texels() {
        let library = examples();
        each_device("over aliases", |name, _, device| {
            let pipelines = library.as_ref().map(|l| pipelines(name, &device, l));
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 4)).expect("a grant");
            let answer = (device.minimum_linear_texture_alignment(PixelFormat::BGRA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let pitch = (ROW as u64).next_multiple_of(answer);
            let memory = alignment(&device);

            // (a) Rows of a taller surface, from a row whose place meets the memory alignment.
            let skip = memory / gcd(pitch, memory);
            let surface_rows = skip + u64::from(HEIGHT) + 4;
            let surface_len = surface_rows * pitch;
            let block = (device.take_block(MemoryClass::Shared, AT + surface_len + TAIL))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let span = (device.new_buffer(block.id, AT, surface_len + TAIL, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let surface = (device.new_buffer(block.id, AT, surface_len, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let place = Place {
                label: "rows of a taller surface",
                span: &span,
                buffer: &surface,
                offset: skip * pitch,
                at: (skip * pitch) as usize,
                pitch,
            };
            writes(name, &device, pipelines.as_ref(), &place);

            // Over the surface, a window at twice the pitch: copied, counted, the same sentinels.
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::BGRA8Unorm, WIDTH, HEIGHT);
            d.usage = TextureUsage::SHADER_READ;
            let window = (surface.new_texture(&d, Some(first), place.offset, 2 * pitch))
                .unwrap_or_else(|error| panic!("{name}: the window: {error}"));
            assert!(
                window.copied_over().is_some(),
                "{name}: the window is copied"
            );
            let mut expected = Expected {
                bytes: held(&span),
                at: place.at,
                pitch: 2 * pitch as usize,
            };
            let source: Vec<u8> = (0..ROW * HEIGHT as usize)
                .map(|i| ((i * 11 + 1) % 239) as u8)
                .collect();
            let from = plain(name, &device, PixelFormat::BGRA8Unorm, first + 1, &source);
            let copies = device.counters().metrics().over_copies;
            let queue = device.new_command_queue(4).expect("a queue");
            run(name, &queue, |cb| {
                let mut blit = cb.blit_command_encoder().expect("an encoder");
                (blit.copy_from_texture(
                    &from,
                    TexturePlace::default(),
                    [WIDTH, HEIGHT, 1],
                    &window,
                    TexturePlace::default(),
                ))
                .unwrap_or_else(|error| panic!("{name}: the window: {error}"));
                blit.end_encoding();
            });
            expected.texels(|x, y| {
                let at = y * ROW + x * 4;
                [source[at], source[at + 1], source[at + 2], source[at + 3]]
            });
            expected.check(name, "a blit copy into the window", &span);
            let moved = device.counters().metrics().over_copies - copies;
            assert!(moved > 0, "{name}: the window's copies counted");
            println!(
                "over aliases: {name}: a window at pitch {}: copied, {moved} copies counted, every \
                 sentinel kept",
                2 * pitch
            );
            drop((window, from, queue, surface, span));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));

            // (b) A texture that ends at its buffer's last texel, a Shared buffer at the next
            // 4 KiB after it.
            let len = (u64::from(HEIGHT) - 1) * pitch + ROW as u64;
            let next_at = len.next_multiple_of(4096);
            let span_len = next_at + NEXT + TAIL;
            let block = (device.take_block(MemoryClass::Shared, AT + span_len))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let span = (device.new_buffer(block.id, AT, span_len, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let ending = (device.new_buffer(block.id, AT, len, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let next = (device.new_buffer(block.id, AT + next_at, NEXT, None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            println!("over aliases: {name}: the next buffer at {next_at} of the span");
            let place = Place {
                label: "ending at its buffer's last texel",
                span: &span,
                buffer: &ending,
                offset: 0,
                at: 0,
                pitch,
            };
            writes(name, &device, pipelines.as_ref(), &place);
            drop((next, ending, span, pipelines));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }
}
