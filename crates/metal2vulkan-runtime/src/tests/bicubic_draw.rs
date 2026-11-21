//! A draw whose function samples a 2D texture through a bicubic constexpr sampler at normalized
//! coordinates, in each stage: the fragment function of the owned fixture `engine/bcu_fragment`
//! (drawn by `colored_vertex`) and the vertex function of `engine/bcu_vertex`, under a device cfg
//! on every device of the cfg's kind. Each draw pushes, for the function's slot, the word a
//! dispatch pushes for a kernel's (`tests::bicubic`): how Metal's bicubic filter treats the format
//! bound there, for each of six formats, and the unbound word for a slot that holds no texture. A
//! draw of an `RGBA8Unorm` texture whose texels are all one value renders that value at every
//! pixel, as the filter's weights sum to one; a draw whose slot holds an `RGBA8Unorm_sRGB`
//! texture is refused, naming the slot and the format. A content with no bicubic slot has none
//! listed.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::pipeline::FieldValue;
    use crate::render::{DrawPlace, FRAGMENT, VERTEX};
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, library, pass, read, run, target, Buffers, SIDE,
    };
    use crate::{
        ColorAttachmentDescriptor, Device, Error, FunctionDescriptor, Gap, HostBytes, IndexClass,
        NumberedSampler, PixelFormat, PrimitiveType, Region, RenderCommandEncoder,
        RenderPipelineDescriptor, RenderPipelineState, StageSource, Texture, TextureDescriptor,
        TextureType, TextureUsage,
    };
    use metal2vulkan::metal::format::{bicubic_texels_word, BICUBIC_TEXELS_UNBOUND};
    use metal2vulkan::metal::TextureSwizzleChannels;
    use metal2vulkan_devtools::engine_dir;

    /// The constexpr sampler's index the test gives the functions.
    const CONSTEXPR: u32 = 1;
    /// The first texture index the test grants.
    const FIRST: u32 = 1;
    /// Each format the test binds, as `tests::bicubic` does.
    const FORMATS: [PixelFormat; 6] = [
        PixelFormat::R32Float,
        PixelFormat::RGBA8Unorm,
        PixelFormat::RGBA8Snorm,
        PixelFormat::RGBA16Float,
        PixelFormat::RGBA16Unorm,
        PixelFormat::R8Unorm,
    ];
    /// The texel of the drawn texture: codes whose `RGBA8Unorm` value a sum within a few ULPs of
    /// it rounds back to.
    const TEXEL: [u8; 4] = [64, 128, 192, 255];

    /// A 2D texture of `format`, 4 texels a side, of shader-read usage at `index`.
    fn texture(name: &str, device: &Device, format: PixelFormat, index: u32) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, 4, 4);
        d.usage = TextureUsage::SHADER_READ;
        (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// The state of `d`, whose functions sample through the one constexpr sampler.
    fn state(name: &str, device: &Device, d: &RenderPipelineDescriptor) -> RenderPipelineState {
        (device.new_render_pipeline_state(d, |samplers| {
            assert!(
                matches!(samplers, [NumberedSampler::Constexpr(_)]),
                "{name}: {samplers:?}"
            );
            Ok(vec![CONSTEXPR])
        }))
        .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// One stage's draw: its descriptor, which of the draw's tables samples (`VERTEX` or
    /// `FRAGMENT`), and how a draw binds its texture and its coordinates' bytes.
    struct Form {
        stage: &'static str,
        descriptor: RenderPipelineDescriptor,
        table: usize,
        bind: fn(&mut RenderCommandEncoder, Option<&Texture>) -> Result<(), Error>,
        coordinates: Vec<f32>,
    }

    #[test]
    fn every_device_draws_a_bicubic_sample_in_each_stage() {
        let (Some(examples), Some(fragments), Some(vertices)) = (
            examples(),
            library(&engine_dir().join("bcu_fragment.ll")),
            library(&engine_dir().join("bcu_vertex.ll")),
        ) else {
            return;
        };
        let function = |library: &crate::Library, name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut fragment = descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm);
        fragment.fragment = Some(function(&fragments, "bcu_fragment"));
        let mut vertex = RenderPipelineDescriptor::new(function(&vertices, "bcu_vertex"));
        vertex.fragment = Some(function(&vertices, "bcu_vertex_color"));
        vertex.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let forms = [
            Form {
                stage: "fragment",
                descriptor: fragment,
                table: FRAGMENT,
                bind: |encoder, texture| encoder.set_fragment_texture(texture, 0),
                // `pos.xy * p.xy + p.zw`: the target's pixels at a sixteenth of a unit each.
                coordinates: vec![1.0 / 16.0, 1.0 / 16.0, 0.0, 0.0],
            },
            Form {
                stage: "vertex",
                descriptor: vertex,
                table: VERTEX,
                bind: |encoder, texture| encoder.set_vertex_texture(texture, 0),
                // `p[vid]`, one coordinate a vertex.
                coordinates: vec![0.25, 0.25, 0.5, 0.75, 0.875, 0.125],
            },
        ];
        each_device("bicubic_draw", |name, _, device| {
            let _grants = [
                (IndexClass::Texture, FIRST, FORMATS.len() as u32 + 2),
                (IndexClass::Sampler, CONSTEXPR, 1),
            ]
            .map(|(class, first, count)| {
                (device.grant_indices(class, first, count))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let textures: Vec<Texture> = (FORMATS.iter().zip(FIRST..))
                .map(|(&format, index)| texture(name, &device, format, index))
                .collect();
            let next = FIRST + FORMATS.len() as u32;
            let srgb = texture(name, &device, PixelFormat::RGBA8UnormSRGB, next);
            let drawn = texture(name, &device, PixelFormat::RGBA8Unorm, next + 1);
            let bytes: Vec<u8> = TEXEL.repeat(16);
            let whole = Region {
                origin: [0; 3],
                size: [4, 4, 1],
            };
            (drawn.replace_region(whole, 0, 0, HostBytes::from(&bytes[..]), 16, 64))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            // `colored_vertex`'s attributes: a triangle that covers the target, and its colours.
            let positions = buffers.of(&device, &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]));
            let colors = buffers.of(&device, &floats(&[0.0; 12]));
            let queue = device.new_command_queue(4).expect("a queue");
            for form in &forms {
                let name = &format!("{name}: {}", form.stage);
                let state = state(name, &device, &form.descriptor);
                // The content lists its one bicubic slot, and its field is in the push range.
                let draws = &state.inner.compiled.draws;
                assert_eq!(draws.bicubic, [(form.table, 0)], "{name}");
                let at = (draws.fields.iter())
                    .find_map(|&(place, table, value)| match (place, value) {
                        (DrawPlace::Push(offset), FieldValue::BicubicTexels(0))
                            if table == form.table =>
                        {
                            Some(offset as usize)
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| panic!("{name}: no pushed field of the slot"));
                let call = |result: Result<(), Error>| {
                    result.unwrap_or_else(|error| panic!("{name}: {error}"))
                };
                let bytes = floats(&form.coordinates);
                let begin = |command_buffer: &crate::CommandBuffer, color: &Texture| {
                    let mut encoder = (command_buffer.render_command_encoder(&pass(color, None)))
                        .unwrap_or_else(|error| panic!("{name}: {error}"));
                    call(encoder.set_render_pipeline_state(&state));
                    call(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                    call(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                    call(match form.table {
                        FRAGMENT => encoder.set_fragment_bytes(&bytes, 0),
                        _ => encoder.set_vertex_bytes(&bytes, 0),
                    });
                    encoder
                };
                // Each format's word and the unbound word, from draws never committed.
                let rgba8 = || {
                    target(
                        &device,
                        PixelFormat::RGBA8Unorm,
                        TextureUsage::RENDER_TARGET,
                    )
                };
                let words = queue.command_buffer().expect("a command buffer");
                let mut encoder = begin(&words, &rgba8());
                let word = |encoder: &RenderCommandEncoder| {
                    u32::from_le_bytes(encoder.pushed()[at..at + 4].try_into().unwrap())
                };
                call((form.bind)(&mut encoder, None));
                call(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                assert_eq!(word(&encoder), BICUBIC_TEXELS_UNBOUND, "{name}: unbound");
                for (texture, &format) in textures.iter().zip(&FORMATS) {
                    call((form.bind)(&mut encoder, Some(texture)));
                    call(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                    let want = bicubic_texels_word(format, TextureSwizzleChannels::DEFAULT);
                    assert_eq!(Some(word(&encoder)), want, "{name}: {format:?}");
                }
                // An sRGB texture is refused, the slot and the format named.
                call((form.bind)(&mut encoder, Some(&srgb)));
                let refused = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
                let Err(Error::Unsupported(gap)) = refused else {
                    panic!("{name}: an sRGB texture's bicubic sample: {refused:?}");
                };
                assert_eq!(
                    gap,
                    Gap::BicubicTexels {
                        slot: 0,
                        pixel_format: PixelFormat::RGBA8UnormSRGB
                    },
                    "{name}"
                );
                let text = gap.to_string();
                assert!(
                    text.contains("texture slot 0") && text.contains("RGBA8UnormSRGB"),
                    "{name}: {text}"
                );
                encoder.end_encoding();
                // A texture of one texel value renders it at every pixel.
                let color = rgba8();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = begin(&command_buffer, &color);
                call((form.bind)(&mut encoder, Some(&drawn)));
                call(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let texels = read(&color, 4);
                for (i, texel) in texels.as_chunks::<4>().0.iter().enumerate() {
                    let [x, y] = [i as u32 % SIDE, i as u32 / SIDE];
                    assert_eq!(*texel, TEXEL, "{name}: pixel [{x}, {y}]");
                }
                println!(
                    "bicubic_draw: {name}: each format's word pushed, sRGB refused, {} pixels \
                     read the texel",
                    texels.len() / 4
                );
            }
            // A content with no bicubic slot lists none.
            let plain = descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm);
            let plain = crate::tests::draw::on_device::state(name, &device, &plain);
            assert!(plain.inner.compiled.draws.bicubic.is_empty(), "{name}");
        });
    }
}
