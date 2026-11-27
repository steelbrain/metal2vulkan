//! A pipeline state that names colours its pass has no attachment for
//! (`RenderCompiled::lacked_past`), under a device cfg on every device of the cfg's kind.
//!
//! - A state of colour 0 `BGRA8Unorm` and colour 1 `RG16Float` over `rt6_pair_two` (engine
//!   `rt6_pair`: colour 0 from fragment buffer 0 at the fragment's texel, colour 1 from buffer 2)
//!   draws in a pass of colour 0 alone, as plain Metal draws it (the conformance case
//!   `pass-pipeline-color-unattached`): through the state of its descriptor without colour 1,
//!   whose content the first such set makes and whose `VkPipeline` the first draw makes, a
//!   first-use wait. Colour 0 holds the fragment's codes in the scissor's rows and the clear in the
//!   others. A second encoder's set takes the content the device holds: no translation and no
//!   pipeline. Draws of the state in a pass of both colours, before it, make no such state at all.
//! - A state of three colours over `cg_five` (engine `color_groups`) in a pass of colours 0 and 2
//!   is refused, `DrawKind::AttachmentFormats`: there plain Metal writes colour 1's value into
//!   colour 2's attachment (`pass-pipeline-color-unattached-between`).
//! - The lacking state is refused in a pass whose colour 0 is another format, in a pass with a
//!   depth attachment, and at a raster sample count other than the pass's.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::render::WITHOUT_COLORS;
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{floats, library, read, run, state, target, Buffers, SIDE};
    use crate::{
        ColorAttachmentDescriptor, DrawKind, Error, FunctionDescriptor, Gap, Library, LoadAction,
        PixelFormat, PrimitiveType, RenderPassDescriptor, RenderPipelineDescriptor, ScissorRect,
        StageSource, StorageMode, Texture, TextureDescriptor, TextureType, TextureUsage,
    };
    use metal2vulkan_devtools::engine_dir;

    /// The codes colour 0's buffer gives texel `at`, `(r, g, b, a)`.
    fn codes(at: u32) -> [u32; 4] {
        [64 + at, 128 + at, 192 + at, at]
    }

    /// The clear colour 0 takes, as codes.
    const CLEAR: [u32; 4] = [3, 5, 7, 250];

    /// The rows the draw is confined to: the top half.
    fn rows() -> ScissorRect {
        ScissorRect {
            x: 0,
            y: 0,
            width: u64::from(SIDE),
            height: u64::from(SIDE / 2),
        }
    }

    /// A state of `vertex` and `fragment` of `library` over `colors`, by index, at `samples`.
    fn descriptor(
        library: &Library,
        (vertex, fragment): (&str, &str),
        colors: &[Option<PixelFormat>],
        samples: u32,
    ) -> RenderPipelineDescriptor {
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function(vertex));
        d.fragment = Some(function(fragment));
        for (at, format) in colors.iter().enumerate() {
            d.color_attachments[at] = format.map(ColorAttachmentDescriptor::new);
        }
        d.raster_sample_count = samples;
        d
    }

    /// A pass of `colors`, by index (`None` unset), each cleared to [`CLEAR`].
    fn pass(colors: &[Option<&Texture>]) -> RenderPassDescriptor {
        let mut d = RenderPassDescriptor::default();
        for (at, texture) in colors.iter().enumerate() {
            let c = &mut d.color_attachments[at];
            c.texture = texture.cloned();
            c.load_action = LoadAction::Clear;
            c.clear_color = crate::ClearColor {
                red: f64::from(CLEAR[0]) / 255.0,
                green: f64::from(CLEAR[1]) / 255.0,
                blue: f64::from(CLEAR[2]) / 255.0,
                alpha: f64::from(CLEAR[3]) / 255.0,
            };
        }
        d
    }

    #[test]
    fn every_device_draws_a_state_whose_last_colour_its_pass_lacks_and_refuses_the_rest() {
        let (Some(pair), Some(groups)) = (
            library(&engine_dir().join("rt6_pair.ll")),
            library(&engine_dir().join("color_groups.ll")),
        ) else {
            return;
        };
        let two = [Some(PixelFormat::BGRA8Unorm), Some(PixelFormat::RG16Float)];
        let pair_fns = ("rt6_pair_vertex", "rt6_pair_two");
        let lacking = descriptor(&pair, pair_fns, &two, 1);
        let multisampled = descriptor(&pair, pair_fns, &two, 4);
        let three = [
            Some(PixelFormat::BGRA8Unorm),
            Some(PixelFormat::RG16Float),
            Some(PixelFormat::RGBA8Unorm),
        ];
        let between = descriptor(&groups, ("cg_cover", "cg_five"), &three, 1);
        each_device("unattached_colors", |name, _, device| {
            let state = state(name, &device, &lacking);
            let between = self::state(name, &device, &between);
            let mut buffers = Buffers::new(&device);
            let zeros: Vec<f32> = (0..SIDE * SIDE)
                .flat_map(|at| codes(at).map(|code| code as f32 / 255.0))
                .collect();
            let zeros = buffers.of(&device, &floats(&zeros));
            let width = buffers.of(&device, &SIDE.to_le_bytes());
            let ones = buffers.of(&device, &floats(&[0.5; (4 * SIDE * SIDE) as usize]));
            let colour = |format| target(&device, format, TextureUsage::RENDER_TARGET);
            let queue = device.new_command_queue(4).expect("a queue");
            let fail = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let refused = Err::<(), _>(Error::Unsupported(Gap::Draw {
                kind: DrawKind::AttachmentFormats,
            }));
            let same = |got: Result<(), Error>, case: &str| {
                assert_eq!(format!("{got:?}"), format!("{refused:?}"), "{name}: {case}");
            };
            let counters = device.counters();
            let calls = || WITHOUT_COLORS.with(std::cell::Cell::get);

            // Draws that render into their pass: no state without colours is asked for.
            let (zero, one) = (
                colour(PixelFormat::BGRA8Unorm),
                colour(PixelFormat::RG16Float),
            );
            let before = (counters.metrics(), calls());
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer
                .render_command_encoder(&pass(&[Some(&zero), Some(&one)])))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            for _ in 0..4 {
                fail(encoder.set_render_pipeline_state(&state));
                fail(encoder.set_fragment_buffer(Some(&zeros), 0, 0));
                fail(encoder.set_fragment_buffer(Some(&width), 0, 1));
                fail(encoder.set_fragment_buffer(Some(&ones), 0, 2));
                fail(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            }
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = (counters.metrics(), calls());
            assert_eq!(
                (
                    after.1 - before.1,
                    after.0.translations - before.0.translations,
                    after.0.pipelines_created - before.0.pipelines_created,
                ),
                (0, 0, 0),
                "{name}: matching draws: states without colours asked for, translations, \
                 pipelines made"
            );

            // The pass of colour 0 alone, twice, each in an encoder of its own.
            let targets = [
                colour(PixelFormat::BGRA8Unorm),
                colour(PixelFormat::BGRA8Unorm),
            ];
            let mut each = Vec::new();
            for target in &targets {
                let before = (counters.metrics(), calls());
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&[Some(target)])))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                fail(encoder.set_render_pipeline_state(&state));
                fail(encoder.set_fragment_buffer(Some(&zeros), 0, 0));
                fail(encoder.set_fragment_buffer(Some(&width), 0, 1));
                fail(encoder.set_fragment_buffer(Some(&ones), 0, 2));
                encoder.set_scissor_rect(rows());
                fail(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                fail(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let after = (counters.metrics(), calls());
                each.push((
                    after.1 - before.1,
                    after.0.translations > before.0.translations,
                    after.0.pipelines_created - before.0.pipelines_created,
                    after.0.first_use_unfinished_waits - before.0.first_use_unfinished_waits,
                ));
            }
            assert_eq!(
                each,
                vec![(1, true, 1, 1), (1, false, 0, 0)],
                "{name}: each lacking pass: states without colours asked for, whether it lowered a \
                 function, pipelines made, first-use waits"
            );
            let want: Vec<u8> = (0..SIDE * SIDE)
                .flat_map(|at| {
                    let [r, g, b, a] = match at < SIDE * SIDE / 2 {
                        true => codes(at),
                        false => CLEAR,
                    };
                    [b, g, r, a].map(|code| code as u8)
                })
                .collect();
            for (at, target) in targets.iter().enumerate() {
                assert_eq!(read(target, 4), want, "{name}: pass {at}: colour 0");
            }

            // Colour 1 unset between colours 0 and 2: refused.
            let (zero, two) = (
                colour(PixelFormat::BGRA8Unorm),
                colour(PixelFormat::RGBA8Unorm),
            );
            let before = calls();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder =
                (command_buffer.render_command_encoder(&pass(&[Some(&zero), None, Some(&two)])))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            fail(encoder.set_render_pipeline_state(&between));
            same(
                encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
                "colour 1 unset between colours 0 and 2",
            );
            encoder.end_encoding();
            assert_eq!(calls() - before, 0, "{name}: between: asked for");

            // The lacking state where the rest is not the pass's.
            let other = colour(PixelFormat::RGBA8Unorm);
            let mut depth_desc =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::Depth32Float, SIDE, SIDE);
            depth_desc.usage = TextureUsage::RENDER_TARGET;
            depth_desc.storage_mode = StorageMode::Private;
            let depth = (device.new_texture_for_views(&depth_desc)).expect("a depth target");
            let mut with_depth = pass(&[Some(&zero)]);
            with_depth.depth_attachment.texture = Some(depth.clone());
            let multisampled = self::state(name, &device, &multisampled);
            for (case, descriptor, drawn) in [
                ("colour 0 another format", pass(&[Some(&other)]), &state),
                ("a depth attachment the state lacks", with_depth, &state),
                (
                    "another raster sample count",
                    pass(&[Some(&zero)]),
                    &multisampled,
                ),
            ] {
                let before = calls();
                let mut encoder = (command_buffer.render_command_encoder(&descriptor))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                fail(encoder.set_render_pipeline_state(drawn));
                same(
                    encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
                    case,
                );
                encoder.end_encoding();
                assert_eq!(calls() - before, 0, "{name}: {case}: asked for");
            }
            run(name, &command_buffer);
            println!(
                "{name}: colour 0 drawn in a pass that lacks colour 1, through 1 state made at the \
                 first set and 1 pipeline at its first draw, none at the second pass's; the \
                 between shape, another format, a depth attachment and another sample count \
                 refused"
            );
        });
    }
}
