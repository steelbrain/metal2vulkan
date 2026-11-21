//! A fragment function that reads `[[barycentric_coord]]` (`floor`'s `BARYCENTRICS`), under a
//! device cfg on every device of the cfg's kind: where device creation enabled
//! `VK_KHR_fragment_shader_barycentric` with its feature, the device names the extension, the
//! translator's profile has `fragmentShaderBarycentric`, the state is made and each drawn pixel
//! reads Metal's weights; elsewhere the device names neither and creation refuses the state,
//! naming `Barycentrics`.
//!
//! `bw_vertex` (engine `bary_weights`) places a triangle whose three vertices have clip-space `w`
//! of 1, 2 and 4, so the perspective-correct weights (`bw_perspective`) and the linear ones
//! (`bw_linear`) differ at every drawn pixel; each writes its three weights.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{floats, library, pass, read, run, target, Buffers, SIDE};
    use crate::{
        ColorAttachmentDescriptor, Error, FunctionDescriptor, PixelFormat, PrimitiveType,
        RenderPipelineDescriptor, StageSource, TextureUsage,
    };
    use metal2vulkan::device::VkFeature;
    use metal2vulkan::error::Feature;
    use metal2vulkan_devtools::engine_dir;

    /// The triangle's vertices in normalized device coordinates (y up), and each one's `w`. Its
    /// long edge, `x + y = 0.125`, passes no pixel centre.
    const CORNERS: [[f64; 2]; 3] = [[-1.0, -1.0], [1.125, -1.0], [-1.0, 1.125]];
    const W: [f64; 3] = [1.0, 2.0, 4.0];

    /// The weights at pixel `[x, y]`'s centre, perspective-correct or linear, or None where the
    /// centre is outside the triangle.
    fn weights([x, y]: [u32; 2], perspective: bool) -> Option<[f64; 3]> {
        let side = f64::from(SIDE);
        let px = (2.0 * f64::from(x) + 1.0) / side - 1.0;
        let py = 1.0 - (2.0 * f64::from(y) + 1.0) / side;
        let [[ax, ay], [bx, _], [_, cy]] = CORNERS;
        let b = (px - ax) / (bx - ax);
        let c = (py - ay) / (cy - ay);
        let linear = [1.0 - b - c, b, c];
        if linear.iter().any(|&weight| weight < 0.0) {
            return None;
        }
        if !perspective {
            return Some(linear);
        }
        let over = [0, 1, 2].map(|at| linear[at] / W[at]);
        let sum: f64 = over.iter().sum();
        Some(over.map(|weight| weight / sum))
    }

    /// Where Vulkan offers `fragmentShaderBarycentric`, a fragment function's weights are Metal's,
    /// perspective-correct and linear; elsewhere the state is refused naming the feature, and the
    /// device names no barycentric extension.
    #[test]
    fn every_device_reads_metal_s_weights_where_it_offers_barycentrics() {
        let Some(weights_library) = library(&engine_dir().join("bary_weights.ll")) else {
            return;
        };
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                weights_library.function(name).expect("the function"),
            ))
        };
        let descriptor = |fragment| {
            let mut d = RenderPipelineDescriptor::new(function("bw_vertex"));
            d.fragment = Some(function(fragment));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
            d
        };
        let clip: Vec<f32> = (CORNERS.iter().zip(W))
            .flat_map(|(&[x, y], w)| [x * w, y * w, 0.0, w].map(|v| v as f32))
            .collect();
        each_device("barycentrics", |name, _, device| {
            // The device names the extension only where device creation enabled its feature, and
            // the translator's profile has the feature exactly there.
            let listed = (device.extensions().iter())
                .any(|enabled| enabled.as_c_str() == ash::khr::fragment_shader_barycentric::NAME);
            let core = device.core();
            let enabled = &core.translator.profile().enabled().features;
            let on = enabled.contains(&VkFeature::FragmentShaderBarycentric);
            assert_eq!(
                listed, on,
                "{name}: the extension named and the feature enabled"
            );
            let mut drawn = 0;
            for (fragment, perspective) in [("bw_perspective", true), ("bw_linear", false)] {
                let state = match device.new_render_pipeline_state(
                    &descriptor(fragment),
                    crate::tests::no_constexpr_samplers,
                ) {
                    Ok(state) => state,
                    Err(Error::Translation {
                        error: metal2vulkan::error::Error::Unsupported(u),
                        ..
                    }) if !on => {
                        assert_eq!(u.feature, Feature::Barycentrics, "{name}: {u:?}");
                        println!(
                            "barycentrics: {name}: no fragmentShaderBarycentric; {fragment}'s \
                             state is refused"
                        );
                        continue;
                    }
                    Err(error) => panic!("{name}: {fragment}: {error}"),
                };
                assert!(on, "{name}: {fragment}'s state made without the feature");
                let mut buffers = Buffers::new(&device);
                let positions = buffers.of(&device, &floats(&clip));
                let color = target(
                    &device,
                    PixelFormat::RGBA32Float,
                    TextureUsage::RENDER_TARGET,
                );
                let queue = device.new_command_queue(4).expect("a queue");
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let ok = |result: Result<(), Error>| {
                    result.unwrap_or_else(|error| panic!("{name}: {fragment}: {error}"))
                };
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                ok(encoder.set_render_pipeline_state(&state));
                ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                ok(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let texels = read(&color, 16);
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let Some(want) = weights([x, y], perspective) else {
                            continue;
                        };
                        let at = ((y * SIDE + x) * 16) as usize;
                        let got: Vec<f32> = (texels[at..at + 12].as_chunks::<4>().0.iter())
                            .map(|w| f32::from_le_bytes(*w))
                            .collect();
                        assert!(
                            (got.iter().zip(want))
                                .all(|(&got, want)| (f64::from(got) - want).abs() < 1e-4),
                            "{name}: {fragment}: pixel [{x}, {y}] reads {got:?}, Metal's weights \
                             are {want:?}"
                        );
                        drawn += 1;
                    }
                }
            }
            if on {
                println!(
                    "barycentrics: {name}: fragmentShaderBarycentric enabled; {drawn} pixels read \
                     Metal's weights, perspective-correct and linear"
                );
            }
        });
    }
}
