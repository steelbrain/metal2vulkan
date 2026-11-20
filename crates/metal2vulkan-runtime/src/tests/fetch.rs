//! Framebuffer fetch (`[[color(n)]]`) on every device of the cfg's kind: a pass of N draws of one
//! triangle over the whole target, each reading the colour under it and writing a function of
//! it, which Metal orders draw after draw, so the target holds that function applied N times.
//! The fetch reads the attachment's texture at the fragment's own pixel: through the attachment
//! feedback loop, one barrier before each fetching draw after the first, or through the image's
//! shadow, one copy before each. Both forms are counted, and a device without the attachment
//! reads refuses the draw.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, library, read, readable, run, state, Buffers, SIDE,
    };
    use crate::texture::TextureObject;
    use crate::{
        BufferLayout, ClearColor, CommandQueue, Device, DrawKind, Error, ExportHandle,
        FunctionDescriptor, Gap, IndexClass, Library, LoadAction, MemoryClass, PixelFormat,
        PlaneSpec, PrimitiveType, RenderPassDescriptor, RenderPipelineDescriptor, StageSource,
        SurfaceForm, SurfaceSpec, Texture, TextureDescriptor, TexturePlace, TextureType,
        TextureUsage,
    };
    use metal2vulkan_devtools::engine_dir;

    /// The draw counts of the passes, each over a target of its own.
    const DRAWS: [u32; 2] = [1, 4];

    /// The opaque handle of this host.
    const OPAQUE: ExportHandle = match cfg!(windows) {
        true => ExportHandle::OpaqueWin32,
        false => ExportHandle::OpaqueFd,
    };

    /// The handles a surface plane is made over, where the device exports them.
    #[cfg(unix)]
    const HANDLES: [ExportHandle; 2] = [OPAQUE, ExportHandle::DmaBuf];
    #[cfg(not(unix))]
    const HANDLES: [ExportHandle; 1] = [OPAQUE];

    /// `colored_vertex` with `fragment` of the owned fixture `engine/<fixture>.ll`, into one
    /// colour attachment of `format`; `None` where `llvm-as` is missing.
    fn fetching(
        fixture: &str,
        fragment: &str,
        format: PixelFormat,
    ) -> Option<RenderPipelineDescriptor> {
        let vertex: Library = examples()?;
        let engine = library(&engine_dir().join(format!("{fixture}.ll")))?;
        let mut d = descriptor(&vertex, "colored_fragment", format);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            (engine.function(fragment)).expect("the function"),
        )));
        Some(d)
    }

    /// What a pass of draws changed: barriers, draws through the loop, copies into shadows and
    /// their bytes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Counted {
        barriers: u64,
        loop_reads: u64,
        copies: u64,
        copy_bytes: u64,
    }

    /// A pass over `color`, cleared to `clear`, of `draws` draws of the triangle over the whole
    /// target through `state`: its result, and what it counted. The new target's move home is
    /// recorded first, in a submission of its own.
    fn pass_of(
        (name, device, queue): (&str, &Device, &CommandQueue),
        state: &crate::RenderPipelineState,
        color: &Texture,
        clear: ClearColor,
        draws: u32,
    ) -> (Result<(), Error>, Counted) {
        let mut buffers = Buffers::new(device);
        let whole = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let positions = buffers.of(device, &floats(&whole));
        let colors = buffers.of(device, &floats(&[0.0; 12]));
        let upload = queue.command_buffer().expect("a command buffer");
        (upload.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
        run(name, &upload);
        let counters = device.counters();
        settled(device);
        let before = counters.metrics();
        let mut d = RenderPassDescriptor::default();
        let c = &mut d.color_attachments[0];
        c.texture = Some(color.clone());
        c.load_action = LoadAction::Clear;
        c.clear_color = clear;
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = (command_buffer.render_command_encoder(&d))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let set = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        set(encoder.set_vertex_buffer(Some(&positions), 0, 0));
        set(encoder.set_vertex_buffer(Some(&colors), 0, 1));
        set(encoder.set_render_pipeline_state(state));
        let mut drew = Ok(());
        for _ in 0..draws {
            drew = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
            if drew.is_err() {
                break;
            }
        }
        encoder.end_encoding();
        run(name, &command_buffer);
        let after = counters.metrics();
        let counted = Counted {
            barriers: after.barriers - before.barriers,
            loop_reads: after.attachment_loop_reads - before.attachment_loop_reads,
            copies: after.attachment_read_copies - before.attachment_read_copies,
            copy_bytes: after.attachment_read_copy_bytes - before.attachment_read_copy_bytes,
        };
        (drew, counted)
    }

    /// What a pass of `draws` fetching draws counts beyond a pass of one, through the loop
    /// (`looped`) or through copies of `texel` bytes a texel, and the loop reads it counts itself.
    fn more(looped: bool, draws: u32, texel: u64) -> (Counted, u64) {
        let later = u64::from(draws - 1);
        let bytes = u64::from(SIDE * SIDE) * texel;
        match looped {
            // One barrier before each fetching draw after the first; every draw through the loop.
            true => (
                Counted {
                    barriers: later,
                    loop_reads: later,
                    copies: 0,
                    copy_bytes: 0,
                },
                u64::from(draws),
            ),
            // One copy of the whole target before each, between two barriers of its own.
            false => (
                Counted {
                    barriers: 2 * later,
                    loop_reads: 0,
                    copies: later,
                    copy_bytes: later * bytes,
                },
                0,
            ),
        }
    }

    /// The texels of `texture`, `texel` bytes each, through a blit into a Shared buffer: a
    /// surface plane's bytes, which the host reads no other way.
    fn blitted(
        name: &str,
        device: &Device,
        queue: &CommandQueue,
        texture: &Texture,
        texel: u64,
    ) -> Vec<u8> {
        let len = u64::from(SIDE * SIDE) * texel;
        let block = (device.take_block(MemoryClass::Shared, len)).expect("a block");
        let out = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(SIDE) * texel,
            bytes_per_image: 0,
        };
        (encoder.copy_from_texture_to_buffer(
            texture,
            TexturePlace::default(),
            [SIDE, SIDE, 1],
            &out,
            layout,
        ))
        .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
        encoder.end_encoding();
        run(name, &command_buffer);
        let contents = out.contents().expect("a Shared buffer");
        // SAFETY: the buffer is alive, `len` bytes long, and no GPU work uses it.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), len as usize) }.to_vec()
    }

    /// The `half` of `2^e`, a normal number: its exponent field alone.
    fn half_power(e: i32) -> u16 {
        ((e + 15) as u16) << 10
    }

    /// The difference of two passes' counts.
    fn less(a: Counted, b: Counted) -> Counted {
        Counted {
            barriers: a.barriers - b.barriers,
            loop_reads: a.loop_reads - b.loop_reads,
            copies: a.copies - b.copies,
            copy_bytes: a.copy_bytes - b.copy_bytes,
        }
    }

    /// `exf_fetch_count` reads its `RGBA32Uint` attachment and writes `count * 3 + (1, 2, 3, 4)`:
    /// after N draws over a target cleared to `(0, 0, 0, 1)` every texel holds
    /// `c * 3^N + (k + 1)(3^N - 1) / 2` in channel `k`, exact only where each draw read what the
    /// one before it wrote. Through the loop, a pass of N draws records N - 1 barriers more than
    /// a pass of one and counts N draws through the loop; through the shadow, N - 1 copies of the
    /// target more, each between two barriers. A device without the attachment reads refuses the
    /// first draw, `Gap::Draw { FetchedAttachment }`.
    #[test]
    fn every_device_orders_each_fetching_draw_after_the_draws_before_it() {
        let Some(d) = fetching(
            "exf_fetch_count",
            "exf_fetch_count",
            PixelFormat::RGBA32Uint,
        ) else {
            return;
        };
        each_device("fetch", |name, _, device| {
            (device.grant_indices(IndexClass::Texture, 4096, DRAWS.len() as u32))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let reads = device.core().attachment_reads;
            let looped = device.core().feedback.is_some();
            let state = state(name, &device, &d);
            let queue = device.new_command_queue(4).expect("a queue");
            let clear = ClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            };
            let mut counts = Vec::new();
            for (at, &draws) in DRAWS.iter().enumerate() {
                let color = readable(&device, PixelFormat::RGBA32Uint, 1, 4096 + at as u32);
                let (drew, counted) =
                    pass_of((name, &device, &queue), &state, &color, clear, draws);
                if !reads {
                    assert_eq!(
                        format!("{drew:?}"),
                        format!(
                            "{:?}",
                            Err::<(), _>(Error::Unsupported(Gap::Draw {
                                kind: DrawKind::FetchedAttachment
                            }))
                        ),
                        "{name}: a device without the attachment reads"
                    );
                    continue;
                }
                drew.unwrap_or_else(|error| panic!("{name}: {draws} draws: {error}"));
                let three = 3u32.pow(draws);
                let want: Vec<u8> = (0..SIDE * SIDE)
                    .flat_map(|_| {
                        [0u32, 0, 0, 1]
                            .into_iter()
                            .enumerate()
                            .map(move |(k, c)| c * three + (k as u32 + 1) * (three - 1) / 2)
                    })
                    .flat_map(u32::to_le_bytes)
                    .collect();
                assert_eq!(read(&color, 16), want, "{name}: {draws} draws, every texel");
                counts.push(counted);
            }
            if !reads {
                println!("{name}: no attachment reads; the fetching draw refused");
                return;
            }
            let draws = DRAWS[1];
            let (want, loop_reads) = more(looped, draws, 16);
            assert_eq!(
                less(counts[1], counts[0]),
                want,
                "{name}: {draws} draws against 1"
            );
            assert_eq!(
                counts[1].loop_reads, loop_reads,
                "{name}: {draws} draws through the loop"
            );
            println!(
                "{name}: {draws} fetching draws exact through {}: {:?} against one draw's {:?}",
                if looped {
                    "the feedback loop"
                } else {
                    "copies into the shadow"
                },
                counts[1],
                counts[0],
            );
        });
    }

    /// A surface plane fetched as a texture of an image of its own is: for each handle the device
    /// exports and each format the compositor's fetches carry (`RGBA16Float`, `BGRA8Unorm` and
    /// `RGBA8Unorm`), a one-plane surface of a shader-read render target, whose image takes the
    /// feedback usage where the device has the loop and the handle's query offers it (printed
    /// with its form and pitch). On the `RGBA16Float` plane, `fetch_ordered` halves the colour
    /// under it in each of N draws over a target cleared to `(1, 0.5, 0.25, 1)`: every texel
    /// `2^-N` of it, exact in `half`, through the loop or the plane's shadow, counted as on a
    /// texture of its own image.
    #[test]
    fn every_device_fetches_a_surface_plane_through_the_loop_or_its_shadow() {
        let Some(d) = fetching("fragment_fetch", "fetch_ordered", PixelFormat::RGBA16Float) else {
            return;
        };
        each_device("fetch_surface", |name, _, device| {
            let first = IndexClass::Texture.range().start;
            let _grant = (device.grant_indices(IndexClass::Texture, first, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let reads = device.core().attachment_reads;
            let state = state(name, &device, &d);
            let queue = device.new_command_queue(4).expect("a queue");
            let clear = ClearColor {
                red: 1.0,
                green: 0.5,
                blue: 0.25,
                alpha: 1.0,
            };
            for handle in HANDLES {
                if !device.core().exports.offers(handle) {
                    println!("fetch-surface: {name}: {handle:?} not exported");
                    continue;
                }
                for format in [
                    PixelFormat::RGBA16Float,
                    PixelFormat::BGRA8Unorm,
                    PixelFormat::RGBA8Unorm,
                ] {
                    let label = format!("{name}: {handle:?} {format:?}");
                    let mut descriptor =
                        TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
                    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
                    let planes = [PlaneSpec {
                        descriptor,
                        index: Some(first),
                        rows_below: 0,
                    }];
                    let spec = SurfaceSpec {
                        handle,
                        pitch_multiple: 4,
                        start_multiple: 1,
                        planes: &planes,
                    };
                    let (surface, export) = (device.new_exported_surface(&spec))
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                    let plane = &surface.planes()[0];
                    // In place, the plane's image took the feedback usage where it reads through
                    // the loop; in the copy form the plane is a texture over a buffer.
                    let in_place = export.form == SurfaceForm::InPlace;
                    let looped = plane.looped();
                    let readable = reads && plane.attachment_reads();
                    println!(
                        "fetch-surface: {label}: form {:?}, pitch {}, the feedback usage in place \
                         {}, read through {}",
                        export.form,
                        export.planes[0].row_pitch,
                        in_place && looped,
                        match (readable, looped) {
                            (false, _) => "nothing (refused)",
                            (true, true) => "the feedback loop",
                            (true, false) => "its shadow",
                        },
                    );
                    if format != PixelFormat::RGBA16Float {
                        continue;
                    }
                    // Both counted passes draw into this one plane. Where it reads through its
                    // shadow, its first read makes the shadow, whose move home the next
                    // submission's upload records with barriers of its own: one uncounted pass
                    // first, so the counted ones differ only by their draws.
                    if readable {
                        let (drew, _) =
                            pass_of((label.as_str(), &device, &queue), &state, plane, clear, 1);
                        drew.unwrap_or_else(|error| panic!("{label}: the first pass: {error}"));
                    }
                    let mut counts = Vec::new();
                    for draws in DRAWS {
                        let (drew, counted) = pass_of(
                            (label.as_str(), &device, &queue),
                            &state,
                            plane,
                            clear,
                            draws,
                        );
                        if !readable {
                            assert!(
                                matches!(
                                    drew,
                                    Err(Error::Unsupported(Gap::Draw {
                                        kind: DrawKind::FetchedAttachment
                                    }))
                                ),
                                "{label}: a plane the draw cannot read through: {drew:?}"
                            );
                            break;
                        }
                        drew.unwrap_or_else(|error| panic!("{label}: {draws} draws: {error}"));
                        // The clear's channels are 2^0, 2^-1, 2^-2 and 2^0, each halved per draw.
                        let n = draws as i32;
                        let texel: Vec<u8> = [0, -1, -2, 0]
                            .into_iter()
                            .flat_map(|e| half_power(e - n).to_le_bytes())
                            .collect();
                        let want = texel.repeat((SIDE * SIDE) as usize);
                        assert_eq!(
                            blitted(&label, &device, &queue, plane, 8),
                            want,
                            "{label}: {draws} draws, every texel"
                        );
                        counts.push(counted);
                    }
                    if !readable {
                        continue;
                    }
                    let draws = DRAWS[1];
                    let (want, loop_reads) = more(looped, draws, 8);
                    assert_eq!(
                        less(counts[1], counts[0]),
                        want,
                        "{label}: {draws} draws against 1"
                    );
                    assert_eq!(
                        counts[1].loop_reads, loop_reads,
                        "{label}: through the loop"
                    );
                    println!(
                        "fetch-surface: {label}: {draws} fetching draws exact: {:?} against one \
                         draw's {:?}",
                        counts[1], counts[0],
                    );
                }
            }
        });
    }

    /// A colour target made for nothing but rendering (`RenderTarget` alone), at an index, as a
    /// colour render target takes one, is fetched as a shader-read target is: `exf_fetch_count`
    /// over N draws, exact, through the feedback loop where the device's image format query
    /// offers the sampled and feedback usages beside the colour usage (printed), its home the
    /// attachment layout, else through the image's shadow, counted as on a shader-read target. A
    /// device without the attachment reads, or whose query offers no such image, refuses the
    /// first draw, `Gap::Draw { FetchedAttachment }`.
    #[test]
    fn every_device_fetches_a_target_made_for_rendering_alone() {
        use ash::vk;
        let Some(d) = fetching(
            "exf_fetch_count",
            "exf_fetch_count",
            PixelFormat::RGBA32Uint,
        ) else {
            return;
        };
        each_device("fetch_render_target", |name, _, device| {
            (device.grant_indices(IndexClass::Texture, 4096, DRAWS.len() as u32))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let reads = device.core().attachment_reads;
            let feedback = device.core().feedback.is_some();
            // The feedback usage is the loop's extension's, so only a device with the loop is
            // asked it (`VUID-VkPhysicalDeviceImageFormatInfo2-usage-parameter`); the answer is
            // read only there.
            let mut usage = vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT;
            if feedback {
                usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
            }
            let info = vk::PhysicalDeviceImageFormatInfo2::default()
                .format(vk::Format::R32G32B32A32_UINT)
                .ty(vk::ImageType::TYPE_2D)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage);
            let offered = device.core().image_format_properties(&info).is_some();
            println!(
                "fetch-render-target: {name}: attachment reads {reads}, feedback loop {feedback}; \
                 R32G32B32A32_UINT optimal 2D with transfer, sampled and colour usage{}: {}",
                if feedback {
                    " and the feedback usage"
                } else {
                    ""
                },
                if offered { "offered" } else { "refused" },
            );
            let state = state(name, &device, &d);
            let queue = device.new_command_queue(4).expect("a queue");
            let clear = ClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            };
            let mut counts = Vec::new();
            let mut looped = false;
            for (&draws, index) in DRAWS.iter().zip(4096..) {
                let mut t = TextureDescriptor::new(
                    TextureType::Type2D,
                    PixelFormat::RGBA32Uint,
                    SIDE,
                    SIDE,
                );
                t.usage = TextureUsage::RENDER_TARGET;
                t.storage_mode = crate::StorageMode::Shared;
                let color = (device.new_texture(&t, Some(index)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                assert_eq!(color.gpu_resource_id().index, index, "{name}: its index");
                looped = color.looped();
                let readable = reads && color.attachment_reads();
                if reads && feedback && offered {
                    assert!(looped, "{name}: the image takes the feedback usage");
                }
                let (drew, counted) =
                    pass_of((name, &device, &queue), &state, &color, clear, draws);
                if !readable {
                    assert!(
                        matches!(
                            drew,
                            Err(Error::Unsupported(Gap::Draw {
                                kind: DrawKind::FetchedAttachment
                            }))
                        ),
                        "{name}: a target the draw cannot read through: {drew:?}"
                    );
                    println!("fetch-render-target: {name}: the fetching draw refused");
                    return;
                }
                drew.unwrap_or_else(|error| panic!("{name}: {draws} draws: {error}"));
                let three = 3u32.pow(draws);
                let want: Vec<u8> = (0..SIDE * SIDE)
                    .flat_map(|_| {
                        [0u32, 0, 0, 1]
                            .into_iter()
                            .enumerate()
                            .map(move |(k, c)| c * three + (k as u32 + 1) * (three - 1) / 2)
                    })
                    .flat_map(u32::to_le_bytes)
                    .collect();
                assert_eq!(read(&color, 16), want, "{name}: {draws} draws, every texel");
                counts.push(counted);
            }
            let draws = DRAWS[1];
            let (want, loop_reads) = more(looped, draws, 16);
            assert_eq!(
                less(counts[1], counts[0]),
                want,
                "{name}: {draws} draws against 1"
            );
            assert_eq!(
                counts[1].loop_reads, loop_reads,
                "{name}: {draws} draws through the loop"
            );
            println!(
                "fetch-render-target: {name}: {draws} fetching draws exact through {}: {:?} \
                 against one draw's {:?}",
                if looped {
                    "the feedback loop"
                } else {
                    "copies into the shadow"
                },
                counts[1],
                counts[0],
            );
        });
    }

    /// A target made for shader writes (`ShaderRead | ShaderWrite | RenderTarget`) or of an
    /// unknown usage, whose image has a storage view where the device stores `RGBA32Uint`
    /// through a storage image without a format, its home general, is fetched as a shader-read
    /// target is: `exf_fetch_count` over N draws, exact, through the feedback loop where the
    /// device's image format query offers the feedback usage beside the storage, sampled and
    /// colour usages (printed), else through the image's shadow, counted as on a shader-read
    /// target. A device without the attachment reads, or whose query offers no such image,
    /// refuses the first draw, `Gap::Draw { FetchedAttachment }`.
    #[test]
    fn every_device_fetches_a_target_made_for_shader_writes() {
        use ash::vk;
        let Some(d) = fetching(
            "exf_fetch_count",
            "exf_fetch_count",
            PixelFormat::RGBA32Uint,
        ) else {
            return;
        };
        let usages = [
            (
                "shader writes",
                TextureUsage::SHADER_READ
                    | TextureUsage::SHADER_WRITE
                    | TextureUsage::RENDER_TARGET,
            ),
            ("an unknown usage", TextureUsage::empty()),
        ];
        each_device("fetch_storage", |name, _, device| {
            let count = (usages.len() * DRAWS.len()) as u32;
            (device.grant_indices(IndexClass::Texture, 4096, count))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let reads = device.core().attachment_reads;
            let feedback = device.core().feedback.is_some();
            // The feedback usage is the loop's extension's, so only a device with the loop is
            // asked it (`VUID-VkPhysicalDeviceImageFormatInfo2-usage-parameter`); the answer is
            // read only there.
            let mut usage = vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT;
            if feedback {
                usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
            }
            let info = vk::PhysicalDeviceImageFormatInfo2::default()
                .format(vk::Format::R32G32B32A32_UINT)
                .ty(vk::ImageType::TYPE_2D)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage);
            let offered = device.core().image_format_properties(&info).is_some();
            println!(
                "fetch-storage: {name}: attachment reads {reads}, feedback loop {feedback}; \
                 R32G32B32A32_UINT optimal 2D with transfer, sampled, storage and colour usage{}: \
                 {}",
                if feedback {
                    " and the feedback usage"
                } else {
                    ""
                },
                if offered { "offered" } else { "refused" },
            );
            let state = state(name, &device, &d);
            let queue = device.new_command_queue(4).expect("a queue");
            let clear = ClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            };
            let mut index = 4096;
            for (what, usage) in usages {
                let label = format!("{name}: {what}");
                let mut counts = Vec::new();
                let mut looped = false;
                let mut readable = false;
                for &draws in &DRAWS {
                    let mut t = TextureDescriptor::new(
                        TextureType::Type2D,
                        PixelFormat::RGBA32Uint,
                        SIDE,
                        SIDE,
                    );
                    t.usage = usage;
                    t.storage_mode = crate::StorageMode::Shared;
                    let color = (device.new_texture(&t, Some(index)))
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                    index += 1;
                    assert!(
                        color.object().has_storage_view(),
                        "{label}: the image has a storage view, its home general"
                    );
                    looped = color.looped();
                    readable = reads && color.attachment_reads();
                    if reads && feedback && offered {
                        // Through the loop where the query offers the feedback usage beside the
                        // storage usage (an unknown usage's image also lists its view formats).
                        assert!(looped, "{label}: the image takes the feedback usage");
                    }
                    let (drew, counted) = pass_of(
                        (label.as_str(), &device, &queue),
                        &state,
                        &color,
                        clear,
                        draws,
                    );
                    if !readable {
                        assert!(
                            matches!(
                                drew,
                                Err(Error::Unsupported(Gap::Draw {
                                    kind: DrawKind::FetchedAttachment
                                }))
                            ),
                            "{label}: a target the draw cannot read through: {drew:?}"
                        );
                        break;
                    }
                    drew.unwrap_or_else(|error| panic!("{label}: {draws} draws: {error}"));
                    let three = 3u32.pow(draws);
                    let want: Vec<u8> = (0..SIDE * SIDE)
                        .flat_map(|_| {
                            [0u32, 0, 0, 1]
                                .into_iter()
                                .enumerate()
                                .map(move |(k, c)| c * three + (k as u32 + 1) * (three - 1) / 2)
                        })
                        .flat_map(u32::to_le_bytes)
                        .collect();
                    assert_eq!(
                        read(&color, 16),
                        want,
                        "{label}: {draws} draws, every texel"
                    );
                    counts.push(counted);
                }
                if !readable {
                    println!("fetch-storage: {label}: the fetching draw refused");
                    continue;
                }
                let draws = DRAWS[1];
                let (want, loop_reads) = more(looped, draws, 16);
                assert_eq!(
                    less(counts[1], counts[0]),
                    want,
                    "{label}: {draws} draws against 1"
                );
                assert_eq!(
                    counts[1].loop_reads, loop_reads,
                    "{label}: {draws} draws through the loop"
                );
                println!(
                    "fetch-storage: {label}: {draws} fetching draws exact through {}: {:?} \
                     against one draw's {:?}",
                    if looped {
                        "the feedback loop"
                    } else {
                        "copies into the shadow"
                    },
                    counts[1],
                    counts[0],
                );
            }
        });
    }
}
