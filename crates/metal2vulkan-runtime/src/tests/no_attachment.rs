//! A render pass with no attachment, `renderTargetWidth` by `renderTargetHeight` pixels at
//! `defaultRasterSampleCount` samples, under a device cfg on every device of the cfg's kind. The
//! owned fixture `engine/rt6_attachmentless.ll` holds the functions a Metal probe drew in such a
//! pass: `rt6_cover` draws the fullscreen triangle, `rt6_mark_sample` ORs its `[[sample_id]]`'s
//! bit into its pixel's word of a 2 by 2 area and counts itself in word 4, and
//! `rt6_mark_coverage` ORs its `[[sample_mask]]` there instead.
//!
//! - At 1, 2 and 4 samples, a pass of 2 by 2 pixels runs `rt6_mark_sample` once per sample of
//!   each pixel, as reading the sample index asks (Vulkan's sample shading): every pixel's word
//!   holds every sample and the count is 4 times the samples, Metal's answer. `rt6_mark_coverage`
//!   runs as the device runs it: every pixel's word holds every sample, and it ran at least once
//!   a pixel.
//! - A pass of no area or of no sample count is `Refusal::NoAttachmentPass`, one past the device's
//!   `maxFramebufferWidth` is `Gap::RenderPass` of `NoAttachment`, and a draw of a pipeline state
//!   of another sample count than the pass's is `DrawKind::AttachmentFormats`.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{library, run, state, Buffers};
    use crate::{
        Buffer, DrawKind, Error, FunctionDescriptor, Gap, PrimitiveType, Refusal,
        RenderPassDescriptor, RenderPassKind, RenderPipelineDescriptor, StageSource,
    };
    use metal2vulkan_devtools::engine_dir;

    /// The pass with no attachment of `width` by `height` pixels at `samples`.
    fn pass(width: u32, height: u32, samples: u32) -> RenderPassDescriptor {
        RenderPassDescriptor {
            render_target_width: width,
            render_target_height: height,
            default_raster_sample_count: samples,
            ..RenderPassDescriptor::default()
        }
    }

    /// The five words of `buffer`: the four pixels' and the count.
    fn words(buffer: &Buffer) -> [u32; 5] {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer holds five words, which the finished draw wrote and whose writes
        // reach the host.
        let words = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), 5) };
        words.try_into().expect("five words")
    }

    #[test]
    fn every_device_draws_a_pass_with_no_attachment_at_its_default_raster_sample_count() {
        let Some(library) = library(&engine_dir().join("rt6_attachmentless.ll")) else {
            return;
        };
        let descriptor = |fragment: &str, samples: u32| {
            let function = |name| {
                StageSource::Function(FunctionDescriptor::new(
                    library.function(name).expect("the function"),
                ))
            };
            let mut d = RenderPipelineDescriptor::new(function("rt6_cover"));
            d.fragment = Some(function(fragment));
            d.raster_sample_count = samples;
            d
        };
        each_device("no_attachment", |name, _, device| {
            let queue = device.new_command_queue(4).expect("a queue");
            let mut buffers = Buffers::new(&device);
            let mut runs = Vec::new();
            for samples in [1, 2, 4] {
                let every = (1u32 << samples) - 1;
                for fragment in ["rt6_mark_sample", "rt6_mark_coverage"] {
                    let state = state(name, &device, &descriptor(fragment, samples));
                    let marks = buffers.of(&device, &[0; 20]);
                    let command_buffer = queue.command_buffer().expect("a command buffer");
                    let mut encoder = (command_buffer.render_command_encoder(&pass(2, 2, samples)))
                        .unwrap_or_else(|error| panic!("{name}: {samples} samples: {error}"));
                    let draw = |result: Result<(), Error>| {
                        result.unwrap_or_else(|e| panic!("{name}: {e}"))
                    };
                    draw(encoder.set_render_pipeline_state(&state));
                    draw(encoder.set_fragment_buffer(Some(&marks), 0, 0));
                    draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                    encoder.end_encoding();
                    run(name, &command_buffer);
                    let [pixels @ .., count] = words(&marks);
                    assert_eq!(
                        pixels, [every; 4],
                        "{name}: {fragment} at {samples} samples"
                    );
                    match fragment {
                        "rt6_mark_sample" => {
                            assert_eq!(count, 4 * samples, "{name}: {fragment} at {samples}")
                        }
                        _ => assert!(count >= 4, "{name}: {fragment} at {samples}: {count} runs"),
                    }
                    runs.push((fragment, samples, count));
                }
            }
            // A pipeline state of 4 samples in a pass of 1.
            let state = state(name, &device, &descriptor("rt6_mark_sample", 4));
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(2, 2, 1)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            (encoder.set_render_pipeline_state(&state)).unwrap_or_else(|e| panic!("{name}: {e}"));
            match encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0) {
                Err(Error::Unsupported(Gap::Draw {
                    kind: DrawKind::AttachmentFormats,
                })) => {}
                other => panic!("{name}: a 4-sample state in a 1-sample pass: {other:?}"),
            }
            encoder.end_encoding();
            let refused = [
                (pass(0, 2, 1), "no width"),
                (pass(2, 0, 1), "no height"),
                (pass(2, 2, 0), "Metal's default count"),
                (pass(2, 2, 3), "3 samples"),
            ];
            for (d, what) in refused {
                match command_buffer.render_command_encoder(&d) {
                    Err(Error::Refused(Refusal::NoAttachmentPass {
                        width,
                        height,
                        samples,
                    })) => assert_eq!(
                        [width, height, samples],
                        [
                            d.render_target_width,
                            d.render_target_height,
                            d.default_raster_sample_count
                        ],
                        "{name}: {what}"
                    ),
                    Err(other) => panic!("{name}: {what}: {other:?}"),
                    Ok(_) => panic!("{name}: {what} was not refused"),
                }
            }
            match command_buffer.render_command_encoder(&pass(u32::MAX, 2, 1)) {
                Err(Error::Unsupported(Gap::RenderPass {
                    kind: RenderPassKind::NoAttachment,
                })) => {}
                Err(other) => panic!("{name}: past maxFramebufferWidth: {other:?}"),
                Ok(_) => panic!("{name}: past maxFramebufferWidth was not refused"),
            }
            run(name, &command_buffer);
            println!(
                "{name}: (fragment, samples, runs) {runs:?}; every pixel's every sample marked; \
                 another state's count, no area, no count and past the device's width refused"
            );
        });
    }
}
