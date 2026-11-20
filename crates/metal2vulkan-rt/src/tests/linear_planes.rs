//! Surface planes that render as linear colour attachments alone, under a device cfg, on every
//! device of the cfg's kind that exports the opaque handle: a surface of two `RGBA8Unorm` planes
//! is marked exactly where it lies in place and the device's linear tiling of the format lacks
//! the colour attachment feature, so it renders through `LINEAR_COLOR_ATTACHMENT_NV`, and the
//! device then enabled `VK_NV_linear_color_attachment`. On every device a draw into one plane,
//! cut by a scissor, and a clear of both planes as two colour attachments of one pass render,
//! each texel read back through a blit and compared. A plane beside another colour attachment,
//! beside a depth attachment, and a multisample attachment resolving into a plane are each
//! refused as the pass begins where the plane is marked, `Gap::LinearPlanePass` of its kind, and
//! render, compared, where it is not. Every device test runs under the validation layer, which
//! fails it on any message. `llvm-as` assembles the draw's fixture (`examples/runtime_examples.ll`);
//! without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{descriptor, examples, run, state, vertices, Buffers};
    use crate::{
        BufferLayout, ClearColor, CommandQueue, Device, Error, ExportHandle, Gap,
        LinearPlanePassKind, LoadAction, MemoryClass, PixelFormat, PlaneSpec, PrimitiveType,
        RenderPassDescriptor, ScissorRect, StorageMode, StoreAction, SurfaceForm, SurfaceSpec,
        Texture, TextureDescriptor, TexturePlace, TextureType, TextureUsage,
    };
    use ash::vk;

    /// The planes' side.
    const SIDE: u32 = 64;
    /// The draw's scissor, from the origin, each side.
    const DRAWN: u32 = 32;

    /// The opaque handle of this host.
    const OPAQUE: ExportHandle = match cfg!(windows) {
        true => ExportHandle::OpaqueWin32,
        false => ExportHandle::OpaqueFd,
    };

    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];

    fn clear(texel: [u8; 4]) -> ClearColor {
        let [r, g, b, a] = texel.map(|c| f64::from(c) / 255.0);
        ClearColor::new(r, g, b, a)
    }

    /// A Private render target of `format`, `SIDE` a side, of `samples`.
    fn texture(device: &Device, format: PixelFormat, samples: u32) -> Texture {
        let kind = match samples {
            1 => TextureType::Type2D,
            _ => TextureType::Type2DMultisample,
        };
        let mut d = TextureDescriptor::new(kind, format, SIDE, SIDE);
        d.sample_count = samples;
        d.usage = TextureUsage::RENDER_TARGET;
        d.storage_mode = StorageMode::Private;
        (device.new_texture(&d, None)).unwrap_or_else(|error| panic!("{format:?}: {error}"))
    }

    /// `texture`'s texels, read through a blit into a Shared buffer.
    fn read(name: &str, device: &Device, queue: &CommandQueue, texture: &Texture) -> Vec<u8> {
        let len = u64::from(SIDE * SIDE * 4);
        let block = (device.take_block(MemoryClass::Shared, len)).expect("a block");
        let buffer = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = command_buffer.blit_command_encoder().expect("an encoder");
        let layout = BufferLayout {
            offset: 0,
            bytes_per_row: u64::from(SIDE * 4),
            bytes_per_image: 0,
        };
        (encoder.copy_from_texture_to_buffer(
            texture,
            TexturePlace::default(),
            [SIDE, SIDE, 1],
            &buffer,
            layout,
        ))
        .unwrap_or_else(|error| panic!("{name}: the read: {error}"));
        encoder.end_encoding();
        run(name, &command_buffer);
        let contents = buffer.contents().expect("Shared");
        // SAFETY: the buffer's bytes, which no pending command reaches.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), len as usize) }.to_vec()
    }

    /// Asserts each texel of `got` is `want(x, y)`, naming how many differ and the first.
    fn texels_are(name: &str, what: &str, got: &[u8], want: impl Fn(u32, u32) -> [u8; 4]) {
        let wrong: Vec<(u32, u32)> = (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let at = ((y * SIDE + x) * 4) as usize;
                got[at..at + 4] != want(x, y)
            })
            .collect();
        if let Some(&(x, y)) = wrong.first() {
            let at = ((y * SIDE + x) * 4) as usize;
            panic!(
                "{name}: {what}: {} of {} texels differ, the first at ({x}, {y}): {:?}, not {:?}",
                wrong.len(),
                SIDE * SIDE,
                &got[at..at + 4],
                want(x, y)
            );
        }
    }

    /// Begins `pass` and ends it at once, or answers the refusal as it begins.
    fn begin(
        name: &str,
        queue: &CommandQueue,
        pass: &RenderPassDescriptor,
    ) -> Result<(), LinearPlanePassKind> {
        let command_buffer = queue.command_buffer().expect("a command buffer");
        match command_buffer.render_command_encoder(pass) {
            Ok(encoder) => {
                encoder.end_encoding();
                run(name, &command_buffer);
                Ok(())
            }
            Err(Error::Unsupported(Gap::LinearPlanePass { kind })) => Err(kind),
            Err(error) => panic!("{name}: {error}"),
        }
    }

    #[test]
    fn every_device_renders_a_plane_alone_and_holds_a_marked_one_to_its_rule() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("linear_planes", |name, _, device| {
            if !device.core().exports.offers(OPAQUE) {
                println!("linear planes: {name}: {OPAQUE:?} not exported");
                return;
            }
            let specs = [0; 2].map(|_| {
                let mut descriptor = TextureDescriptor::new(
                    TextureType::Type2D,
                    PixelFormat::RGBA8Unorm,
                    SIDE,
                    SIDE,
                );
                descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
                PlaneSpec {
                    descriptor,
                    index: None,
                    rows_below: 0,
                }
            });
            let spec = SurfaceSpec {
                handle: OPAQUE,
                pitch_multiple: 4,
                start_multiple: 1,
                planes: &specs,
            };
            let (surface, export) = (device.new_exported_surface(&spec))
                .unwrap_or_else(|error| panic!("{name}: the surface: {error}"));
            let (linear, _) = device.core().format_features(vk::Format::R8G8B8A8_UNORM);
            let plain = linear.contains(vk::FormatFeatureFlags2::COLOR_ATTACHMENT);
            let nv = linear.contains(vk::FormatFeatureFlags2::LINEAR_COLOR_ATTACHMENT_NV);
            let enabled = (device.extensions().iter())
                .any(|n| n.as_c_str() == ash::nv::linear_color_attachment::NAME);
            let marked = export.form == SurfaceForm::InPlace && !plain;
            println!(
                "linear planes: {name}: form {:?}; linear RGBA8 features {:#x}: colour attachment \
                 {plain}, linear colour attachment {nv}; VK_NV_linear_color_attachment enabled \
                 {enabled}; marked {marked}",
                export.form,
                linear.as_raw()
            );
            assert!(
                !marked || (nv && enabled),
                "{name}: a marked plane's device"
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let [one, two] = [&surface.planes()[0], &surface.planes()[1]];

            // A draw into one plane alone: cleared red, a quarter drawn blue.
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let mut pass = RenderPassDescriptor::default();
            let c = &mut pass.color_attachments[0];
            c.texture = Some(one.clone());
            c.load_action = LoadAction::Clear;
            c.store_action = StoreAction::Store;
            c.clear_color = clear(RED);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass))
                .unwrap_or_else(|error| panic!("{name}: one plane: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            encoder.set_scissor_rect(ScissorRect {
                x: 0,
                y: 0,
                width: u64::from(DRAWN),
                height: u64::from(DRAWN),
            });
            // The fixture's second triangle covers the whole of clip space, in blue.
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let quarter = |x: u32, y: u32| match x < DRAWN && y < DRAWN {
                true => BLUE,
                false => RED,
            };
            texels_are(
                name,
                "one plane drawn",
                &read(name, &device, &queue, one),
                quarter,
            );
            println!("linear planes: {name}: a draw into one plane renders, exact");

            // Both planes of the picture as two colour attachments of one pass.
            let mut pass = RenderPassDescriptor::default();
            for (at, (plane, texel)) in [(one, GREEN), (two, BLUE)].into_iter().enumerate() {
                let c = &mut pass.color_attachments[at];
                c.texture = Some(plane.clone());
                c.load_action = LoadAction::Clear;
                c.store_action = StoreAction::Store;
                c.clear_color = clear(texel);
            }
            begin(name, &queue, &pass)
                .unwrap_or_else(|kind| panic!("{name}: two planes: {kind:?}"));
            texels_are(
                name,
                "plane 0 of two",
                &read(name, &device, &queue, one),
                |_, _| GREEN,
            );
            texels_are(
                name,
                "plane 1 of two",
                &read(name, &device, &queue, two),
                |_, _| BLUE,
            );
            println!("linear planes: {name}: both planes in one pass render, exact");

            // A plane beside another colour attachment, beside a depth attachment, and a
            // multisample attachment resolving into it: each cleared red into the plane.
            let other = texture(&device, PixelFormat::RGBA8Unorm, 1);
            let depth = texture(&device, PixelFormat::Depth32Float, 1);
            let samples = texture(&device, PixelFormat::RGBA8Unorm, 4);
            let plane_pass = || {
                let mut pass = RenderPassDescriptor::default();
                let c = &mut pass.color_attachments[0];
                c.texture = Some(one.clone());
                c.load_action = LoadAction::Clear;
                c.store_action = StoreAction::Store;
                c.clear_color = clear(RED);
                pass
            };
            let mut beside = plane_pass();
            let c = &mut beside.color_attachments[1];
            c.texture = Some(other.clone());
            c.load_action = LoadAction::Clear;
            c.clear_color = clear(GREEN);
            let mut depthed = plane_pass();
            depthed.depth_attachment.texture = Some(depth.clone());
            depthed.depth_attachment.load_action = LoadAction::Clear;
            depthed.depth_attachment.clear_depth = 1.0;
            let mut resolved = RenderPassDescriptor::default();
            let c = &mut resolved.color_attachments[0];
            c.texture = Some(samples.clone());
            c.load_action = LoadAction::Clear;
            c.store_action = StoreAction::MultisampleResolve;
            c.clear_color = clear(RED);
            c.resolve_texture = Some(one.clone());
            for (shape, pass, kind) in [
                (
                    "beside another colour",
                    beside,
                    LinearPlanePassKind::OtherColor,
                ),
                (
                    "beside a depth attachment",
                    depthed,
                    LinearPlanePassKind::DepthStencil,
                ),
                ("resolved into", resolved, LinearPlanePassKind::Resolve),
            ] {
                // The plane starts green, so a pass that renders leaves red.
                begin(name, &queue, &{
                    let mut green = plane_pass();
                    green.color_attachments[0].clear_color = clear(GREEN);
                    green
                })
                .unwrap_or_else(|kind| panic!("{name}: the plane alone: {kind:?}"));
                match (begin(name, &queue, &pass), marked) {
                    (Err(refused), true) => {
                        assert_eq!(refused, kind, "{name}: a marked plane {shape}");
                        texels_are(name, shape, &read(name, &device, &queue, one), |_, _| GREEN);
                        println!(
                            "linear planes: {name}: a marked plane {shape}: refused, {kind:?}"
                        );
                    }
                    (Ok(()), false) => {
                        texels_are(name, shape, &read(name, &device, &queue, one), |_, _| RED);
                        println!("linear planes: {name}: a plane {shape}: renders, exact");
                    }
                    (answer, marked) => {
                        panic!("{name}: a plane {shape}, marked {marked}: {answer:?}")
                    }
                }
            }
        });
    }
}
