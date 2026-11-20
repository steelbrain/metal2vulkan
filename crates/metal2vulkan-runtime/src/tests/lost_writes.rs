//! A fragment's memory writes in the shapes in which a GPU was measured to lose them or to run them
//! too often, under a device cfg on every device of the cfg's kind, each compared with Metal's
//! answer. The owned fixture `examples/lost_writes.ll` (its source `lost_writes.metal`) holds the
//! functions; `lw_cover` draws one triangle over a `WIDTH` by `HEIGHT` target.
//!
//! - An early-tests fragment's texture writes at 1 and 4 samples: `lw_early_write` declares early
//!   fragment tests, writes `(x, y, 0.25, 1)` at its pixel into an `RGBA32Float` texture, so
//!   every texel holds its own coordinates, and counts its runs; `lw_late_write`, the same function
//!   without early tests, is the control. Measured in Metal itself on an Apple M3 Max (macOS
//!   27.0.1): at 2 and 4 samples such a function's every texture write was lost while it ran, and
//!   the control's landed. A Vulkan driver built on Metal hands Metal the attribute and loses them
//!   the same way, out of Vulkan's spec (the runtime README's known gaps of render pipeline
//!   states): the test accepts exactly that outcome as the known gap and prints it, and any other
//!   loss fails.
//! - A discarding fragment's texture writes after a texture sample: `lw_dropped` samples a
//!   texture, discards where `(x + 3 y) % 5` is 0, samples again, runs float arithmetic the
//!   module keeps apart in a branch only the discarded fragments enter, and writes `(41, x, y, 1)`
//!   at its pixel into an `RGBA32Uint` texture of `u32::MAX` words: every kept pixel holds its
//!   code and every discarded one the sentinel. Measured in plain Metal on an Apple M4, with the
//!   sampled texture held by an argument buffer: the shape lost the writes of every live thread
//!   of a simdgroup holding a discarded one. The test samples a texture slot: the identity
//!   domains lower no sample of an argument buffer's texture (a typed refusal, the README's known
//!   gaps of pipeline states), and the argument buffer's form joins the test once they do.
//! - A memory-writing fragment at 4 samples: `lw_count` reads `[[sample_mask]]`, adds 1 to its
//!   pixel's run count and ORs the mask into its pixel's word, drawn by `lw_half` over the
//!   target's lower-left half. Metal runs such a function once per covered sample, each run
//!   reading the pixel's whole coverage: each pixel's count is its mask's bits, every pixel
//!   inside the triangle is all four samples, and some on its diagonal fewer. Measured on Apple's
//!   GPU through a driver that also ran each such shader per covered sample: a looping function
//!   counted each covered sample once per covered sample.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{library, pass, run, state, Buffers};
    use crate::{
        ColorAttachmentDescriptor, CullMode, Device, Error, FunctionDescriptor, IndexClass,
        Library, NumberedSampler, PixelFormat, PrimitiveType, Region, RenderPipelineDescriptor,
        StageSource, StorageMode, Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::fixtures_dir;

    /// The target's width and height: 4 by 4 groups of 8 by 4 pixels.
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 16;
    /// The first texture index the tests grant, and the constexpr sampler's index.
    const TEXTURES: u32 = 4096;
    const SAMPLER: u32 = 3;

    fn fixture() -> Option<Library> {
        library(&fixtures_dir().join("examples/lost_writes.ll"))
    }

    /// `vertex` and `fragment` of `library`, drawing one `RGBA8Unorm` attachment at `samples`.
    fn descriptor(
        library: &Library,
        vertex: &str,
        fragment: &str,
        samples: u32,
    ) -> RenderPipelineDescriptor {
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function(vertex));
        d.fragment = Some(function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.raster_sample_count = samples;
        d
    }

    /// The `RGBA8Unorm` attachment of `samples` samples, with no index of its own.
    fn color(name: &str, device: &Device, samples: u32) -> Texture {
        let ty = match samples {
            1 => TextureType::Type2D,
            _ => TextureType::Type2DMultisample,
        };
        let mut d = TextureDescriptor::new(ty, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
        d.sample_count = samples;
        d.usage = TextureUsage::RENDER_TARGET;
        d.storage_mode = StorageMode::Private;
        (device.new_texture_for_views(&d)).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// A Shared 2D texture of `format`, `side` texels by `rows`, at texture index `index`, its
    /// bytes `bytes` row by row (`unit` a texel).
    fn texture(
        name: &str,
        device: &Device,
        format: PixelFormat,
        (side, rows, unit): (u32, u32, u64),
        usage: TextureUsage,
        index: u32,
        bytes: &[u8],
    ) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, side, rows);
        d.usage = usage;
        d.storage_mode = StorageMode::Shared;
        let texture =
            (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"));
        let row = u64::from(side) * unit;
        (texture.replace_region(
            region(side, rows),
            0,
            0,
            crate::HostBytes::from(bytes),
            row,
            row * u64::from(rows),
        ))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        texture
    }

    fn region(side: u32, rows: u32) -> Region {
        Region {
            origin: [0; 3],
            size: [side, rows, 1],
        }
    }

    /// The `WIDTH` by `HEIGHT` texels of `texture`, four 32-bit words each.
    fn words(texture: &Texture) -> Vec<[u32; 4]> {
        let row = WIDTH as usize * 16;
        let mut bytes = vec![0; row * HEIGHT as usize];
        (texture.get_bytes(
            &mut bytes,
            row as u64,
            (row * HEIGHT as usize) as u64,
            region(WIDTH, HEIGHT),
            0,
            0,
        ))
        .expect("get_bytes");
        bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|t| {
                std::array::from_fn(|i| u32::from_le_bytes(t[i * 4..i * 4 + 4].try_into().unwrap()))
            })
            .collect()
    }

    /// Every pixel's coordinates, row by row.
    fn pixels() -> impl Iterator<Item = (u32, u32)> {
        (0..HEIGHT).flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
    }

    /// What one draw of a `lw_cover` triangle with fragment function `fragment` at `samples` left:
    /// the texels of its `RGBA32Float` texture holding their own coordinates, those still holding
    /// the sentinel, and its run count.
    struct Writes {
        written: usize,
        untouched: usize,
        runs: u32,
    }

    fn draw_writes(
        name: &str,
        device: &Device,
        library: &Library,
        fragment: &str,
        samples: u32,
    ) -> Writes {
        let d = descriptor(library, "lw_cover", fragment, samples);
        let state = state(name, device, &d);
        let sentinel = vec![0xffu8; (WIDTH * HEIGHT * 16) as usize];
        let out = texture(
            name,
            device,
            PixelFormat::RGBA32Float,
            (WIDTH, HEIGHT, 16),
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
            TEXTURES,
            &sentinel,
        );
        let mut buffers = Buffers::new(device);
        let runs = buffers.of(device, &[0; 4]);
        let target = color(name, device, samples);
        let queue = device.new_command_queue(1).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = (command_buffer.render_command_encoder(&pass(&target, None)))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        draw(encoder.set_render_pipeline_state(&state));
        encoder.set_cull_mode(CullMode::None);
        draw(encoder.set_fragment_texture(Some(&out), 0));
        draw(encoder.set_fragment_buffer(Some(&runs), 0, 0));
        draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
        encoder.end_encoding();
        run(name, &command_buffer);
        let got = words(&out);
        let written = pixels()
            .zip(&got)
            .filter(|((x, y), g)| **g == [*x as f32, *y as f32, 0.25, 1.0].map(f32::to_bits))
            .count();
        let untouched = got.iter().filter(|g| **g == [u32::MAX; 4]).count();
        let contents = runs.contents().expect("a Shared buffer");
        // SAFETY: the buffer holds one word, which the finished draw wrote and whose writes reach
        // the host.
        let runs = unsafe { contents.as_ptr().cast::<u32>().read_unaligned() };
        Writes {
            written,
            untouched,
            runs,
        }
    }

    /// An early-tests fragment's texture writes land at 1 and at 4 samples: every texel of the
    /// texture holds its own coordinates, as the control's (no early tests) does, and both ran at
    /// least once per pixel. The one other outcome accepted is the known gap's exact signature
    /// at more than one sample: the function ran, the control wrote every texel, and the
    /// early-tests function wrote none, every texel still the sentinel.
    #[test]
    fn every_device_keeps_an_early_tests_fragments_texture_writes_at_four_samples() {
        let Some(library) = fixture() else {
            return;
        };
        let pixels = (WIDTH * HEIGHT) as usize;
        each_device("lost_writes_early_tests", |name, _, device| {
            (device.grant_indices(IndexClass::Texture, TEXTURES, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for samples in [1, 4] {
                let name = &format!("{name}, {samples} samples");
                let early = draw_writes(name, &device, &library, "lw_early_write", samples);
                let late = draw_writes(name, &device, &library, "lw_late_write", samples);
                assert!(
                    early.runs as usize >= pixels && late.runs as usize >= pixels,
                    "{name}: {} and {} runs (early tests, control) over {pixels} pixels",
                    early.runs,
                    late.runs
                );
                assert_eq!(
                    late.written, pixels,
                    "{name}: the control's texels not written as Metal writes them"
                );
                if early.written == pixels {
                    println!("{name}: every texel written");
                    continue;
                }
                let known_gap = samples > 1 && early.written == 0 && early.untouched == pixels;
                assert!(
                    known_gap,
                    "{name}: texels not written as Metal writes them: {} of {pixels} written, {} \
                     untouched",
                    early.written, early.untouched
                );
                println!(
                    "{name}: known gap: early-tests storage writes dropped at more than one sample \
                     (README)"
                );
            }
        });
    }

    /// A discarding fragment's texture writes after a texture sample: every kept pixel holds its
    /// code and coordinates, and every discarded one the sentinel.
    #[test]
    fn every_device_keeps_the_texture_writes_beside_a_discard_after_a_texture_sample() {
        let Some(library) = fixture() else {
            return;
        };
        let d = descriptor(&library, "lw_cover", "lw_dropped", 1);
        each_device("lost_writes_discard", |name, _, device| {
            for (class, first, count) in [
                (IndexClass::Texture, TEXTURES, 2),
                (IndexClass::Sampler, 1, 4),
            ] {
                (device.grant_indices(class, first, count))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            let state = (device.new_render_pipeline_state(&d, |samplers| {
                // Its constexpr sampler, then its texture write's.
                assert!(
                    matches!(
                        samplers,
                        [NumberedSampler::Constexpr(_), NumberedSampler::Write]
                    ),
                    "{name}: {samplers:?}"
                );
                Ok(vec![SAMPLER, 0])
            }))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            // A 4x4 texture of one texel, (40, 80, 120, 255): every sample reads 40 / 255 in red.
            let noise = texture(
                name,
                &device,
                PixelFormat::RGBA8Unorm,
                (4, 4, 4),
                TextureUsage::SHADER_READ,
                TEXTURES,
                &[40u8, 80, 120, 255].repeat(16),
            );
            let sentinel = vec![0xffu8; (WIDTH * HEIGHT * 16) as usize];
            let out = texture(
                name,
                &device,
                PixelFormat::RGBA32Uint,
                (WIDTH, HEIGHT, 16),
                TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
                TEXTURES + 1,
                &sentinel,
            );
            let globals: Vec<u8> = [1.0f32, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
            let target = color(name, &device, 1);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&target, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            encoder.set_cull_mode(CullMode::None);
            draw(encoder.set_fragment_bytes(&globals, 1));
            draw(encoder.set_fragment_texture(Some(&out), 0));
            draw(encoder.set_fragment_texture(Some(&noise), 1));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let got = words(&out);
            let (mut lost, mut extra) = (Vec::new(), 0);
            for ((x, y), texel) in pixels().zip(&got) {
                match (x + 3 * y) % 5 {
                    0 => extra += usize::from(*texel != [u32::MAX; 4]),
                    _ if *texel != [41, x, y, 1] => lost.push((x, y)),
                    _ => {}
                }
            }
            assert_eq!(extra, 0, "{name}: discarded pixels written");
            assert!(lost.is_empty(), "{name}: kept pixels lost: {lost:?}");
            println!("{name}: every kept pixel's write landed, no discarded one's");
        });
    }

    /// A memory-writing fragment at 4 samples runs once per covered sample, each run reading the
    /// pixel's whole coverage: each pixel's run count is its mask's bits.
    #[test]
    fn every_device_runs_a_memory_writing_fragment_once_per_covered_sample() {
        let Some(library) = fixture() else {
            return;
        };
        let d = descriptor(&library, "lw_half", "lw_count", 4);
        each_device("lost_writes_per_sample", |name, _, device| {
            let state = state(name, &device, &d);
            let pixels = (WIDTH * HEIGHT) as usize;
            let mut buffers = Buffers::new(&device);
            let runs = buffers.of(&device, &vec![0; pixels * 4]);
            let masks = buffers.of(&device, &vec![0; pixels * 4]);
            let target = color(name, &device, 4);
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&target, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            encoder.set_cull_mode(CullMode::None);
            draw(encoder.set_fragment_buffer(Some(&runs), 0, 0));
            draw(encoder.set_fragment_buffer(Some(&masks), 0, 1));
            draw(encoder.set_fragment_bytes(&WIDTH.to_le_bytes(), 2));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let read = |buffer: &crate::Buffer| -> Vec<u32> {
                let contents = buffer.contents().expect("a Shared buffer");
                // SAFETY: the buffer holds `pixels` words, which the finished draw wrote and whose
                // writes reach the host.
                unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), pixels) }
                    .to_vec()
            };
            let (runs, masks) = (read(&runs), read(&masks));
            let wrong: Vec<(usize, u32, u32)> = (0..pixels)
                .filter(|&p| runs[p] != masks[p].count_ones() || masks[p] > 0xf)
                .map(|p| (p, runs[p], masks[p]))
                .collect();
            assert!(
                wrong.is_empty(),
                "{name}: (pixel, runs, mask) not once per covered sample: {wrong:?}"
            );
            let full = masks.iter().filter(|&&m| m == 0xf).count();
            let partial = masks.iter().filter(|&&m| m != 0 && m != 0xf).count();
            let empty = masks.iter().filter(|&&m| m == 0).count();
            assert!(
                full > 0 && partial > 0 && empty > 0,
                "{name}: {full} covered pixels, {partial} partly, {empty} not"
            );
            println!(
                "{name}: {full} covered pixels ran 4 times, {partial} on the edge once per covered sample"
            );
        });
    }
}
