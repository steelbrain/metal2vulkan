//! A fragment function's `[[stencil]]` output (`floor`'s `OPTIONAL_EXTENSIONS`), under a device cfg
//! on every device of the cfg's kind: where the device lists `VK_EXT_shader_stencil_export`,
//! device creation enables it, the translator's profile has it, the state is made, and a draw
//! whose depth-stencil state replaces the stencil stores each fragment's output, its low 8 bits as
//! Metal takes them, at every pixel; elsewhere the device names no such extension and creation
//! refuses the state naming `StencilExport`, the pixel interlock choosing no other form under the
//! identity domains.
//!
//! `colored_vertex` (the runtime's examples) draws the fullscreen triangle; `stx_stencil_by_pixel`
//! (engine `stx_stencil`) returns word `y * width + x` of its buffer 0 at pixel `(x, y)`.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, library, pass, read, run, target, vertices, Buffers, SIDE,
    };
    use crate::texture::TextureObject;
    use crate::{
        DepthStencilDescriptor, Error, FunctionDescriptor, PixelFormat, PrimitiveType, StageSource,
        StencilOperation, TextureUsage,
    };
    use metal2vulkan::error::Feature;
    use metal2vulkan_devtools::engine_dir;

    /// The word pixel `i` (row-major) of the fragment function's buffer returns: bits above the
    /// low 8 on two of every three pixels, which Metal's stencil reference drops.
    fn value(i: u32) -> u32 {
        0x100 * (i % 3) + (i * 29 + 7) % 256
    }

    #[test]
    fn every_device_stores_a_fragment_functions_stencil_output_where_it_lists_the_export() {
        let Some(examples) = examples() else {
            return;
        };
        let Some(stencils) = library(&engine_dir().join("stx_stencil.ll")) else {
            return;
        };
        let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            stencils
                .function("stx_stencil_by_pixel")
                .expect("the function"),
        )));
        d.stencil_format = Some(PixelFormat::Stencil8);
        each_device("stencil_export", |name, _, device| {
            let export = ash::ext::shader_stencil_export::NAME;
            let listed = device.extensions().iter().any(|e| e.as_c_str() == export);
            let profile = device.core().translator.profile().fragment().stencil_export;
            assert_eq!(profile, listed, "{name}: the translator's stencil export");
            let made = device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers);
            let state = match (listed, made) {
                (true, Ok(state)) => state,
                (true, Err(error)) => panic!("{name}: {error}"),
                (false, Ok(_)) => panic!("{name}: a state made without the export"),
                (
                    false,
                    Err(Error::Translation {
                        error: metal2vulkan::error::Error::Unsupported(u),
                        ..
                    }),
                ) => {
                    assert_eq!(u.feature, Feature::StencilExport, "{name}: {u:?}");
                    println!("{name}: no stencil export listed; refused naming StencilExport");
                    return;
                }
                (false, Err(error)) => panic!("{name}: {error}"),
            };
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let words: Vec<u8> = (0..SIDE * SIDE)
                .flat_map(|i| value(i).to_le_bytes())
                .collect();
            let values = buffers.of(&device, &words);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let stencil = target(&device, PixelFormat::Stencil8, TextureUsage::RENDER_TARGET);
            let mut replace = DepthStencilDescriptor::default();
            for face in [
                &mut replace.front_face_stencil,
                &mut replace.back_face_stencil,
            ] {
                face.depth_stencil_pass_operation = StencilOperation::Replace;
            }
            let replace = device.new_depth_stencil_state(&replace);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer
                .render_command_encoder(&pass(&color, Some(&stencil))))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_fragment_buffer(Some(&values), 0, 0));
            draw(encoder.set_fragment_bytes(&SIDE.to_le_bytes(), 1));
            // The encoder's reference, which the fragment's output replaces.
            encoder.set_stencil_reference_value(0x5a);
            draw(encoder.set_depth_stencil_state(Some(&replace)));
            // The second triangle covers the whole target.
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            if stencil.unit().is_some() {
                let want: Vec<u8> = (0..SIDE * SIDE).map(|i| value(i) as u8).collect();
                assert_eq!(read(&stencil, 1), want, "{name}: the stencil");
            }
            println!(
                "{name}: VK_EXT_shader_stencil_export enabled; {} pixels stored their fragment's \
                 stencil output",
                SIDE * SIDE
            );
        });
    }
}
