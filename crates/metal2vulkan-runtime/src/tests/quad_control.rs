//! A fragment function whose group operations read other lanes (`floor`'s `QUAD_CONTROL`), under a
//! device cfg on every device of the cfg's kind: where device creation enabled
//! `VK_KHR_shader_quad_control`, the translator's profile has `shaderQuadControl`, the state is
//! made and the draw answers Metal's values; elsewhere creation refuses the state, naming
//! `SubgroupOperations`.
//!
//! `fragment_quad_bitwise` (engine `gsm2_quad_ops`) writes its quad's AND, OR and XOR of a word of
//! each pixel's position, drawn by a triangle whose edge cuts the diagonal quads, so each of those
//! quads runs one helper. Metal runs the helpers, and they take part (`lower::simd`'s
//! `full_quads`), so every drawn pixel reads all four of its quad's words.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, library, pass, read, run, target, Buffers, SIDE,
    };
    use crate::{Error, FunctionDescriptor, PixelFormat, PrimitiveType, StageSource, TextureUsage};
    use metal2vulkan::device::VkFeature;
    use metal2vulkan::error::Feature;
    use metal2vulkan_devtools::engine_dir;

    /// The word `fragment_quad_bitwise` takes of pixel `[x, y]`.
    fn word([x, y]: [u32; 2]) -> u32 {
        (x * 7 + y * 13) ^ 0x5a
    }

    /// Where Vulkan offers `shaderQuadControl`, a fragment function's quad operations read their
    /// quad's helpers, as Metal's do; elsewhere the state is refused naming the feature.
    #[test]
    fn every_device_runs_quads_whole_where_it_offers_quad_control() {
        let (Some(examples), Some(quads)) =
            (examples(), library(&engine_dir().join("gsm2_quad_ops.ll")))
        else {
            return;
        };
        let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA32Uint);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            quads
                .function("fragment_quad_bitwise")
                .expect("the function"),
        )));
        each_device("quad_control", |name, _, device| {
            // The translator's profile has the feature only where device creation enabled its
            // extension.
            let listed = (device.extensions().iter())
                .any(|enabled| enabled.as_c_str() == ash::khr::shader_quad_control::NAME);
            let core = device.core();
            let enabled = &core.translator.profile().enabled().features;
            let on = enabled.contains(&VkFeature::ShaderQuadControl);
            assert!(
                listed || !on,
                "{name}: shaderQuadControl without its extension"
            );
            let state =
                match device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers) {
                    Ok(state) => state,
                    Err(Error::Translation {
                        error: metal2vulkan::error::Error::Unsupported(u),
                        ..
                    }) if !on => {
                        assert_eq!(u.feature, Feature::SubgroupOperations, "{name}: {u:?}");
                        println!("quad_control: {name}: no quad control; the state is refused");
                        return;
                    }
                    Err(error) => panic!("{name}: {error}"),
                };
            assert!(on, "{name}: a state made without quad control");
            let mut buffers = Buffers::new(&device);
            // Pixel centres with x <= y are inside: the edge x + y = 0.125 (clip space, y up)
            // passes no centre, and cuts each diagonal quad at its top right pixel.
            let positions = buffers.of(&device, &floats(&[-1.0, -1.0, 1.125, -1.0, -1.0, 1.125]));
            let colors = buffers.of(&device, &floats(&[0.0; 12]));
            let color = target(
                &device,
                PixelFormat::RGBA32Uint,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let ok = |result: Result<(), Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(encoder.set_render_pipeline_state(&state));
            ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let texels = read(&color, 16);
            let mut drawn = 0;
            for y in 0..SIDE {
                for x in (0..SIDE).filter(|&x| x <= y) {
                    let quad = [[0, 0], [1, 0], [0, 1], [1, 1]]
                        .map(|[dx, dy]| word([(x & !1) + dx, (y & !1) + dy]));
                    let want = [
                        quad.iter().fold(!0, |all, w| all & (w | 0x100)),
                        quad.iter().fold(0, |any, w| any | w),
                        quad.iter().fold(0, |odd, w| odd ^ w),
                    ];
                    let at = ((y * SIDE + x) * 16) as usize;
                    let got: Vec<u32> = (texels[at..at + 12].as_chunks::<4>().0.iter())
                        .map(|w| u32::from_le_bytes(*w))
                        .collect();
                    assert_eq!(got, want, "{name}: pixel [{x}, {y}]");
                    drawn += 1;
                }
            }
            println!(
                "quad_control: {name}: shaderQuadControl enabled; {drawn} pixels read their \
                 quads whole, helpers included"
            );
        });
    }
}
