//! Several viewports and scissor rectangles (`setViewports:count:`, `setScissorRects:count:`), and
//! a vertex function's `[[viewport_array_index]]`, under a device cfg on every device of the cfg's
//! kind. `rt6_select` (engine `rt6_viewports`) draws Metal's fullscreen triangle through the
//! viewport of its instance, and `rt6_viewport_read` colours red the index it reads, green 1.
//!
//! Two viewports, the target's left and right halves, and two scissors, the whole target and its
//! top half: instance 0 fills the left half, instance 1 the right half's top, cut by its scissor,
//! and instance 2, past the count, draws through viewport 0 as Metal does, its fragment function
//! reading the 2 its vertex function wrote. No pipeline is made for the count, and none or 17
//! viewports or rectangles are `DrawKind::Viewports`.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{floats, library, pass, read, run, state, target, SIDE};
    use crate::{
        ColorAttachmentDescriptor, DrawKind, Error, FunctionDescriptor, Gap, PixelFormat,
        PrimitiveType, RenderPipelineDescriptor, ScissorRect, StageSource, TextureUsage, Viewport,
    };
    use metal2vulkan_devtools::engine_dir;

    /// The viewport over columns `[x, x + width)` of the target, every row, depth `[0, 1]`.
    fn columns(x: u32, width: u32) -> Viewport {
        Viewport {
            origin_x: f64::from(x),
            origin_y: 0.0,
            width: f64::from(width),
            height: f64::from(SIDE),
            znear: 0.0,
            zfar: 1.0,
        }
    }

    #[test]
    fn every_device_selects_a_viewport_and_its_scissor_per_primitive() {
        let Some(library) = library(&engine_dir().join("rt6_viewports.ll")) else {
            return;
        };
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("rt6_select"));
        d.fragment = Some(function("rt6_viewport_read"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        each_device("viewports", |name, _, device| {
            let state = state(name, &device, &d);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            // The upload that moves the new target to its home layout, in a submission apart.
            let first = queue.command_buffer().expect("a command buffer");
            (first.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
            run(name, &first);
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let refused = |result: Result<(), Error>, what: &str| match result {
                Err(Error::Unsupported(Gap::Draw {
                    kind: DrawKind::Viewports,
                })) => {}
                other => panic!("{name}: {what}: {other:?}"),
            };
            let half = SIDE / 2;
            refused(encoder.set_viewports(&[]), "no viewport");
            refused(
                encoder.set_viewports(&[columns(0, half); 17]),
                "17 viewports",
            );
            refused(encoder.set_scissor_rects(&[]), "no scissor rectangle");
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_bytes(&floats(&[0.5, 0.0]), 0));
            draw(encoder.set_viewports(&[columns(0, half), columns(half, half)]));
            let whole = ScissorRect {
                x: 0,
                y: 0,
                width: u64::from(SIDE),
                height: u64::from(SIDE),
            };
            let top = ScissorRect {
                height: u64::from(half),
                ..whole
            };
            draw(encoder.set_scissor_rects(&[whole, top]));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 3, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                after.pipelines_created - before.pipelines_created,
                0,
                "{name}: pipelines made"
            );
            let want: Vec<u8> = (0..SIDE)
                .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
                .flat_map(|(x, y)| match (x < half, y < half) {
                    (true, _) => [2, 255, 0, 255],
                    (false, true) => [1, 255, 0, 255],
                    (false, false) => [0, 0, 0, 255],
                })
                .collect();
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!(
                "{name}: instance 0 and 2 through viewport 0, instance 1 through viewport 1 cut by \
                 its scissor; no pipeline made"
            );
        });
    }
}
