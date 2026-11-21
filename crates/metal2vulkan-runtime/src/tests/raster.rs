//! The render encoder's depth clip mode and triangle fill mode (`render::Raster`), each a
//! `VkPipeline` of the draw's class that the content's first draw under it makes, under a device
//! cfg on every device of the cfg's kind. `cfb_depth_vertex` (engine `cfb_sample_depth`) returns
//! the `float4` of its buffer 0 for each vertex, and `cfb_depth_white` colours white.
//!
//! - A triangle over the whole target in front of the near plane (z -0.5) draws nothing under
//!   `Clip` and every texel under `Clamp`, which keeps it at the near plane (the conformance case
//!   `render-state-depth-clip`).
//! - Under `Lines`, where the device has `fillModeNonSolid`, the triangle at z 0.5 through a
//!   viewport 0.4 pixels right of and 0.3 above the target's corner draws the two edges that lie
//!   inside the target, column 0 and row 7, and nothing inside them; without the feature the mode
//!   is `DrawKind::FillMode` and the encoder's mode stays `Fill`. Each edge crosses its pixels'
//!   diamonds off their centers and corners, and both start strictly inside the corner pixel's, so
//!   Bresenham's diamond-exit rule draws them on every device: a line through texel centers ties
//!   at each diamond's corners, where lavapipe drew row 7 not at all (m3-max-62049).
//! - The first draw under each raster state makes one pipeline, a first-use wait of the raster
//!   site; a later encoder's draw under the same state makes none.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        floats, library, pass, read, run, state, target, texels, Buffers, BLACK, SIDE,
    };
    use crate::{
        Buffer, ColorAttachmentDescriptor, DepthClipMode, DrawKind, Error, FunctionDescriptor, Gap,
        PixelFormat, PrimitiveType, RenderPipelineDescriptor, StageSource, TextureUsage,
        TriangleFillMode, Viewport,
    };
    use metal2vulkan::device::VkFeature;
    use metal2vulkan_devtools::engine_dir;

    const WHITE: [f32; 4] = [1.0; 4];

    /// The triangle `(-1, -1)`, `(3, -1)`, `(-1, 3)` of clip space, which covers the clip square,
    /// at depth `z`: `cfb_depth_vertex`'s positions.
    fn triangle(z: f32) -> Vec<u8> {
        floats(&[-1.0, -1.0, z, 1.0, 3.0, -1.0, z, 1.0, -1.0, 3.0, z, 1.0])
    }

    #[test]
    fn every_device_draws_clamped_and_edge_triangles_through_pipelines_made_at_first_use() {
        let Some(library) = library(&engine_dir().join("cfb_sample_depth.ll")) else {
            return;
        };
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("cfb_depth_vertex"));
        d.fragment = Some(function("cfb_depth_white"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        each_device("raster", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let near = buffers.of(&device, &triangle(-0.5));
            let inside = buffers.of(&device, &triangle(0.5));
            let non_solid = (device.core().translator.profile().enabled().features)
                .contains(&VkFeature::FillModeNonSolid);
            let queue = device.new_command_queue(4).expect("a queue");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            // Each pass in an encoder of its own, so each sets its modes from Metal's defaults.
            let passes: [(DepthClipMode, TriangleFillMode, &Buffer); 4] = [
                (DepthClipMode::Clip, TriangleFillMode::Fill, &near),
                (DepthClipMode::Clamp, TriangleFillMode::Fill, &near),
                (DepthClipMode::Clamp, TriangleFillMode::Fill, &near),
                (DepthClipMode::Clip, TriangleFillMode::Lines, &inside),
            ];
            let colors: Vec<_> = (passes.iter())
                .map(|_| {
                    target(
                        &device,
                        PixelFormat::RGBA8Unorm,
                        TextureUsage::RENDER_TARGET,
                    )
                })
                .collect();
            // The upload that moves the new targets to their home layout, in a submission apart.
            let first = queue.command_buffer().expect("a command buffer");
            (first.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
            run(name, &first);
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            for ((clip, fill, positions), color) in passes.iter().zip(&colors) {
                let mut encoder = (command_buffer.render_command_encoder(&pass(color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                draw(encoder.set_render_pipeline_state(&state));
                draw(encoder.set_vertex_buffer(Some(*positions), 0, 0));
                draw(encoder.set_depth_clip_mode(*clip));
                let lines = *fill == TriangleFillMode::Lines;
                match (encoder.set_triangle_fill_mode(*fill), lines && !non_solid) {
                    (Ok(()), false) => {}
                    (Err(Error::Unsupported(Gap::Draw { kind })), true) => {
                        assert_eq!(kind, DrawKind::FillMode, "{name}");
                    }
                    (other, _) => panic!("{name}: {fill:?}: {other:?}"),
                }
                if lines && non_solid {
                    // Clip space's left and bottom edges at x 0.4 and y 7.7, inside column 0's
                    // and row 7's diamonds.
                    draw(encoder.set_viewport(Viewport {
                        origin_x: 0.4,
                        origin_y: -0.3,
                        width: f64::from(SIDE),
                        height: f64::from(SIDE),
                        znear: 0.0,
                        zfar: 1.0,
                    }));
                }
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
            }
            run(name, &command_buffer);
            let after = counters.metrics();
            let made = 1 + u64::from(non_solid);
            assert_eq!(
                (
                    after.pipelines_created - before.pipelines_created,
                    after.first_use_raster_waits - before.first_use_raster_waits,
                    after.first_use_waits - before.first_use_waits,
                ),
                (made, made, made),
                "{name}: pipelines made, raster first-use waits, first-use waits"
            );
            let all = |c| texels(|_, _| c);
            assert_eq!(read(&colors[0], 4), all(BLACK), "{name}: clipped");
            assert_eq!(read(&colors[1], 4), all(WHITE), "{name}: clamped");
            assert_eq!(read(&colors[2], 4), all(WHITE), "{name}: clamped again");
            let edges = match non_solid {
                true => texels(|x, y| match x == 0 || y == SIDE - 1 {
                    true => WHITE,
                    false => BLACK,
                }),
                // The mode refused, the triangle fills the target through the whole viewport.
                false => all(WHITE),
            };
            assert_eq!(read(&colors[3], 4), edges, "{name}: the edges");
            println!(
                "{name}: a near triangle clipped, then clamped through 1 pipeline made at first \
                 use; fillModeNonSolid {non_solid}: {} under Lines",
                match non_solid {
                    true => "its two edges inside the target drawn, 1 pipeline more",
                    false => "refused (FillMode), filled",
                }
            );
        });
    }
}
