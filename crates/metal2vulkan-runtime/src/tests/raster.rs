//! The render encoder's depth clip mode and triangle fill mode (`render::Raster`), each a
//! `VkPipeline` of the draw's class that the content's first draw under it makes, under a device
//! cfg on every device of the cfg's kind. `cfb_depth_vertex` (engine `cfb_sample_depth`) returns
//! the `float4` of its buffer 0 for each vertex, and `cfb_depth_white` colours white.
//!
//! - A triangle over the whole target in front of the near plane (z -0.5) draws nothing under
//!   `Clip` and every texel under `Clamp`, which keeps it at the near plane (the conformance case
//!   `render-state-depth-clip`).
//! - Under `Lines`, where the device has `fillModeNonSolid`, a triangle at z 0.5 whose left and
//!   bottom edges lie inside the target, at x 1.4 and y 6.7, and strictly inside the clip volume
//!   and the viewport, draws those two edges and nothing inside them; without the feature the mode
//!   is `DrawKind::FillMode`, the encoder's mode stays `Fill`, and the triangle over the whole
//!   target fills it. Each edge crosses its pixels' diamonds off their centers and corners (a line
//!   through texel centers ties at each diamond's corners, where lavapipe drew such a row not at
//!   all, m3-max-62049), so the diamond-exit rule, Metal's, draws column 1 and row 6, and the test
//!   holds exactly those, printing the target where a device draws otherwise. The corner where the
//!   edges meet is not read: there the two lines' ends are each the device's.
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
        TriangleFillMode,
    };
    use metal2vulkan::device::VkFeature;
    use metal2vulkan_devtools::engine_dir;

    const WHITE: [f32; 4] = [1.0; 4];

    /// Holds that the Lines pass's two edges in `texels` (the target's RGBA8 bytes) are the
    /// diamond-exit rule's, Metal's: column 1 white in every row above row 5, row 6 white in every
    /// column from 3, and nothing else white outside the corner where they meet. Panics naming
    /// `name`, with the target drawn, otherwise.
    fn lines_at(name: &str, texels: &[u8]) {
        let white = |x: u32, y: u32| {
            let i = ((y * SIDE + x) * 4) as usize;
            match texels[i..i + 4] {
                [255, 255, 255, 255] => true,
                [0, 0, 0, 255] => false,
                ref other => panic!("{name}: ({x}, {y}) reads {other:?}"),
            }
        };
        // The target's rows top first, `#` drawn, for a failure's message.
        let drawn: Vec<String> = (0..SIDE)
            .map(|y| {
                (0..SIDE)
                    .map(|x| if white(x, y) { '#' } else { '.' })
                    .collect()
            })
            .collect();
        for y in 0..SIDE {
            for x in 0..SIDE {
                let corner = x < 3 && y >= 5;
                let edge = (x == 1 && y < 5) || (y == 6 && x >= 3);
                assert!(
                    corner || edge == white(x, y),
                    "{name}: ({x}, {y}) {}; the diamond-exit rule draws column 1 above row 5 and \
                     row 6 from column 3: {drawn:?}",
                    if edge { "not drawn" } else { "drawn" }
                );
            }
        }
    }

    /// The triangle `(-1, -1)`, `(3, -1)`, `(-1, 3)` of clip space, which covers the clip square,
    /// at depth `z`: `cfb_depth_vertex`'s positions.
    fn triangle(z: f32) -> Vec<u8> {
        floats(&[-1.0, -1.0, z, 1.0, 3.0, -1.0, z, 1.0, -1.0, 3.0, z, 1.0])
    }

    /// The triangle `(-0.65, -0.675)`, `(3, -0.675)`, `(-0.65, 3)` of clip space at depth `z`:
    /// through the target's viewport, its left edge at x 1.4 and its bottom edge at y 6.7, each
    /// inside the clip volume, and its third edge past the target's top left.
    fn edge_triangle(z: f32) -> Vec<u8> {
        floats(&[
            -0.65, -0.675, z, 1.0, 3.0, -0.675, z, 1.0, -0.65, 3.0, z, 1.0,
        ])
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
            let non_solid = (device.core().translator.profile().enabled().features)
                .contains(&VkFeature::FillModeNonSolid);
            // The edges where the device draws them; the whole target where it fills instead.
            let inside = match non_solid {
                true => buffers.of(&device, &edge_triangle(0.5)),
                false => buffers.of(&device, &triangle(0.5)),
            };
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
            let edges = read(&colors[3], 4);
            let drawn = match non_solid {
                true => {
                    lines_at(name, &edges);
                    "its two edges drawn, column 1 and row 6 (the diamond-exit rule's), 1 pipeline \
                     more"
                        .to_owned()
                }
                false => {
                    // The mode refused, the triangle fills the target through the whole viewport.
                    assert_eq!(edges, all(WHITE), "{name}: filled");
                    "refused (FillMode), filled".to_owned()
                }
            };
            println!(
                "{name}: a near triangle clipped, then clamped through 1 pipeline made at first \
                 use; fillModeNonSolid {non_solid}: {drawn} under Lines"
            );
        });
    }
}
