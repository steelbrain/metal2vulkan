//! The private alpha test of OpenGL on Metal (`PrivateRenderState::alpha_test_enabled` and
//! `alpha_test_function`), with the encoder's reference
//! (`RenderCommandEncoder::set_alpha_test_reference_value`), under a device cfg on every device of
//! the cfg's kind. The expected texels are Metal's rule as probed on an Apple M3 Max under macOS
//! 27.0.1 (413 probed rows): color 0's alpha and the reference are
//! each compared as 8-bit unorm codes, and an encoder compares against 1 until it sets the
//! reference, a reference set in an earlier encoder not kept.
//!
//! The fragment writes green at alpha 0.6, code 153, under `Greater`: the default reference (code
//! 255) and 0.75 (code 191) kill it, 0.25 (code 64) and 0 (code 0) keep it. One draw a quadrant,
//! through scissor rectangles, in two encoders.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, pass, read, run, state, target, Buffers, SIDE,
    };
    use crate::{LoadAction, PixelFormat, PrimitiveType, ScissorRect, TextureUsage};

    /// `MTLCompareFunctionGreater`.
    const GREATER: u64 = 4;
    /// The fragment's color: green at alpha 0.6, code 153 (the exact product with 255).
    const COLOR: [f32; 4] = [0.0, 1.0, 0.0, 0.6];
    /// The drawn texel, and the clear's.
    const DRAWN: [u8; 4] = [0, 255, 0, 153];
    const CLEAR: [u8; 4] = [0, 0, 0, 255];

    /// The scissor rectangle of the quadrant at `[x, y]`, each 0 or 1, y down.
    fn quadrant([x, y]: [u64; 2]) -> ScissorRect {
        let half = u64::from(SIDE / 2);
        ScissorRect {
            x: x * half,
            y: y * half,
            width: half,
            height: half,
        }
    }

    /// A state with the alpha test on reads the encoder's reference, which holds for the encoder's
    /// later draws and starts at 1 in each encoder: of four quadrants drawn in one encoder, the
    /// one drawn before any reference and the one under 0.75 keep the clear, and the two under
    /// 0.25 are drawn; a second encoder's draw of the whole target before it sets a reference
    /// changes nothing, and its draw under 0 fills the quadrant the first encoder's 0.75 kept.
    #[test]
    fn every_device_alpha_tests_against_the_encoder_s_reference() {
        let Some(library) = examples() else {
            return;
        };
        let mut d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.private.alpha_test_enabled = true;
        d.private.alpha_test_function = GREATER;
        each_device("alpha_test", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let positions = buffers.of(&device, &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]));
            let colors = buffers.of(&device, &floats(&[COLOR, COLOR, COLOR].concat()));
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let ok = |result: Result<(), crate::Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let mut loads = pass(&color, None);
            loads.color_attachments[0].load_action = LoadAction::Load;
            for (at, passed) in [pass(&color, None), loads].iter().enumerate() {
                let mut encoder = (command_buffer.render_command_encoder(passed))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                ok(encoder.set_render_pipeline_state(&state));
                ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                // Each step: the reference set first where one is given, then a draw through the
                // quadrant's rectangle, or the whole target's.
                let steps: &[(Option<f32>, Option<[u64; 2]>)] = match at {
                    0 => &[
                        (None, Some([0, 0])),
                        (Some(0.75), Some([1, 0])),
                        (Some(0.25), Some([0, 1])),
                        (None, Some([1, 1])),
                    ],
                    _ => &[(None, None), (Some(0.0), Some([1, 0]))],
                };
                for &(reference, rect) in steps {
                    if let Some(reference) = reference {
                        encoder.set_alpha_test_reference_value(reference);
                    }
                    encoder.set_scissor_rect(rect.map_or(
                        ScissorRect {
                            x: 0,
                            y: 0,
                            width: u64::from(SIDE),
                            height: u64::from(SIDE),
                        },
                        quadrant,
                    ));
                    ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                }
                encoder.end_encoding();
            }
            run(name, &command_buffer);
            let half = SIDE / 2;
            let want: Vec<u8> = (0..SIDE)
                .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
                .flat_map(|(x, y)| match x < half && y < half {
                    true => CLEAR,
                    false => DRAWN,
                })
                .collect();
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!(
                "alpha_test: {name}: the default and 0.75 kept the clear, 0.25 and 0 drew, a second \
                 encoder started at 1"
            );
        });
    }
}
