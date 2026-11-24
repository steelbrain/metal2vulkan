//! A draw that reaches its pass's colour attachment through an argument buffer, by the
//! `gpuResourceID` the guest wrote there, rather than through a texture slot. Its texture's
//! sampled element of set 1 names the image's home layout, as every element does, and the draw
//! reads it while the pass holds the image in another layout: the attachment layout, or the
//! feedback layout once an earlier draw of the pass read the attachment through a slot
//! (`crate::render_encoder`'s feedback moves). Vulkan asks that the layout an image descriptor
//! names match the layout the image is in when a draw accesses it
//! (`VUID-vkCmdDraw-imageLayout-00344`, `VUID-vkCmdDraw-None-09600`).
//!
//! `examples/runtime_examples`' `own_pixel_fragment` reads the attachment through a slot.
//! `engine/resident_draws`' `rd_entry_vertex` and `rd_entry_fragment` place one small triangle at
//! the position the argument-table entry points to and sample, through the sampler the entry
//! names, the texture the entry names: the attachment's resource ID.
//!
//! `llvm-as` assembles the functions; without it the test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::pass::TARGET_WALKS;
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, library, pass, read, readable, run, state, Buffers, SIDE,
    };
    use crate::{
        ColorAttachmentDescriptor, FunctionDescriptor, HostBytes, IndexClass, PixelFormat,
        PrimitiveType, Region, RenderCommandEncoder, RenderPipelineDescriptor, Resource,
        ResourceUsage, SamplerDescriptor, StageSource,
    };
    use metal2vulkan_devtools::engine_dir;
    use std::cell::Cell;

    /// The attachment's index (the control texture's the next), and the sampler's.
    const TEXTURE: u32 = 4096;
    const SAMPLER: u32 = 7;

    /// A pass over a shader-read render target, cleared, in which a draw reaches the target
    /// through an argument buffer: after a draw that read it through a slot, which moves it to the
    /// feedback layout on a device with the loop, and in a pass of its own, where it is in the
    /// attachment layout. Each counts one in `Metrics::unslotted_attachment_reads`, the known gap
    /// in the runtime's README, and each pass's loop reads (`Metrics::attachment_loop_reads`) are
    /// one in the first pass through the loop, none in the second. A control pass first draws the
    /// same way through an entry naming a green texture, which is no attachment: it counts none,
    /// and the target's first pixel is green, so the entry's triangle covers it and the fragment
    /// function runs and reads its texture. The layer's messages, where it has any, fail the test
    /// as any other does; it does not check a run-time-indexed element's layout.
    #[test]
    fn every_device_reads_its_attachment_through_an_argument_buffer_in_its_own_pass() {
        let (Some(examples), Some(engine)) =
            (examples(), library(&engine_dir().join("resident_draws.ll")))
        else {
            return;
        };
        let own = descriptor(&examples, "own_pixel_fragment", PixelFormat::RGBA8Unorm);
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                engine.function(name).expect("the function"),
            ))
        };
        let mut held = RenderPipelineDescriptor::new(function("rd_entry_vertex"));
        held.fragment = Some(function("rd_entry_fragment"));
        held.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        each_device("held_loop_reads", |name, _, device| {
            let core = device.core();
            let (reads, looped) = (core.attachment_reads, core.feedback_loop);
            println!("{name}: attachment reads {reads}, feedback loop {looped}");
            if !reads {
                println!("{name}: SKIPPED: the device reads no attachment through a texture");
                return;
            }
            let _grants = [
                (IndexClass::Texture, TEXTURE, 2),
                (IndexClass::Sampler, SAMPLER, 1),
            ]
            .map(|(class, first, count)| {
                (device.grant_indices(class, first, count))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let own = state(name, &device, &own);
            let held = state(name, &device, &held);
            let color = readable(&device, PixelFormat::RGBA8Unorm, 1, TEXTURE);
            let green = readable(&device, PixelFormat::RGBA8Unorm, 1, TEXTURE + 1);
            let texel = [0u8, 255, 0, 255];
            let row = u64::from(SIDE) * 4;
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let bytes = texel.repeat((SIDE * SIDE) as usize);
            (green.replace_region(
                whole,
                0,
                0,
                HostBytes::from(&bytes[..]),
                row,
                row * u64::from(SIDE),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let sampler = (device.new_sampler_state(&SamplerDescriptor::new(), SAMPLER))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            // One triangle over the whole target, no colour added.
            let whole = buffers.of(&device, &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]));
            let colors = buffers.of(&device, &floats(&[0.0; 12]));
            // The entry's triangle covers the centre of the target's first pixel, its top left
            // (Metal's clip space has y up), from just below and left of it.
            let centre = 1.0 - 1.0 / SIDE as f32;
            let at = [-centre - 0.005, centre - 0.005, 0.0, 1.0];
            let position = buffers.of(&device, &floats(&at));
            let mut entry = |texture: &crate::Texture| {
                let words = [
                    position.gpu_address(),
                    texture.gpu_resource_id().to_bits(),
                    sampler.gpu_resource_id().to_bits(),
                ];
                let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                buffers.of(&device, &bytes)
            };
            let (held_color, held_green) = (entry(&color), entry(&green));
            let queue = device.new_command_queue(4).expect("a queue");
            // The upload that moves the new target to its home layout, in a submission apart.
            let upload = queue.command_buffer().expect("a command buffer");
            (upload.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
            run(name, &upload);
            let cases = [
                ("control", &held_green, &green, false),
                ("after a slot's read", &held_color, &color, true),
                ("alone", &held_color, &color, false),
            ];
            for (case, entry, reached, through_slot) in cases {
                let counters = device.counters();
                settled(&device);
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let draw = |result: Result<(), crate::Error>| {
                    result.unwrap_or_else(|error| panic!("{name} {case}: {error}"))
                };
                if through_slot {
                    draw(encoder.set_vertex_buffer(Some(&whole), 0, 0));
                    draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                    draw(encoder.set_fragment_texture(Some(&color), 0));
                    draw(encoder.set_render_pipeline_state(&own));
                    draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                }
                draw(encoder.use_resource(Resource::Texture(reached), ResourceUsage::READ));
                draw(encoder.use_resource(Resource::Buffer(&position), ResourceUsage::READ));
                draw(encoder.set_vertex_buffer(Some(entry), 0, 0));
                draw(encoder.set_fragment_buffer(Some(entry), 0, 0));
                draw(encoder.set_render_pipeline_state(&held));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let after = counters.metrics();
                let loop_reads = after.attachment_loop_reads - before.attachment_loop_reads;
                let copies = after.attachment_read_copies - before.attachment_read_copies;
                let unslotted =
                    after.unslotted_attachment_reads - before.unslotted_attachment_reads;
                println!(
                    "held-loop-reads: {name} {case}: unslotted attachment reads {unslotted}, loop \
                     reads {loop_reads}, shadow copies {copies}"
                );
                let want = u64::from(through_slot && looped);
                assert_eq!(loop_reads, want, "{name} {case}: the pass's loop reads");
                let want = u64::from(case != "control");
                assert_eq!(
                    unslotted, want,
                    "{name} {case}: the pass's unslotted attachment reads"
                );
                if case == "control" {
                    let first = read(&color, 4)[..4].to_vec();
                    assert_eq!(
                        first, texel,
                        "{name}: the entry's triangle drew the first pixel"
                    );
                }
            }
        });
    }

    /// The draws after the first since a change walk none of their pass's attachments
    /// (`crate::pass::TARGET_WALKS`): whether a draw counts in
    /// `Metrics::unslotted_attachment_reads` is settled as the encoder's declarations and its
    /// pass's moves are made, and a draw tests it alone. In a pass over a shader-read render
    /// target, which the pass holds in the attachment layout: draws of a pipeline state whose
    /// functions read nothing beyond their slots, before the target is declared read and after,
    /// count none; draws of the argument-buffer pipeline state, which may read beyond its slots,
    /// count one each.
    #[test]
    fn every_device_walks_no_attachment_at_a_draw_after_no_change() {
        const DRAWS: u64 = 64;
        let (Some(examples), Some(engine)) =
            (examples(), library(&engine_dir().join("resident_draws.ll")))
        else {
            return;
        };
        let plain = descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm);
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                engine.function(name).expect("the function"),
            ))
        };
        let mut held = RenderPipelineDescriptor::new(function("rd_entry_vertex"));
        held.fragment = Some(function("rd_entry_fragment"));
        held.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        each_device("held_read_walks", |name, _, device| {
            if !device.core().attachment_reads {
                println!("{name}: SKIPPED: the device reads no attachment through a texture");
                return;
            }
            let _grants = [
                (IndexClass::Texture, TEXTURE, 1),
                (IndexClass::Sampler, SAMPLER, 1),
            ]
            .map(|(class, first, count)| {
                (device.grant_indices(class, first, count))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let plain = state(name, &device, &plain);
            let held = state(name, &device, &held);
            let color = readable(&device, PixelFormat::RGBA8Unorm, 1, TEXTURE);
            let sampler = (device.new_sampler_state(&SamplerDescriptor::new(), SAMPLER))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let whole = buffers.of(&device, &floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]));
            let colors = buffers.of(&device, &floats(&[0.0; 12]));
            let position = buffers.of(&device, &floats(&[0.0, 0.0, 0.0, 1.0]));
            let words = [
                position.gpu_address(),
                color.gpu_resource_id().to_bits(),
                sampler.gpu_resource_id().to_bits(),
            ];
            let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            let entry = buffers.of(&device, &bytes);
            let queue = device.new_command_queue(4).expect("a queue");
            // The upload that moves the new target to its home layout, in a submission apart.
            let upload = queue.command_buffer().expect("a command buffer");
            (upload.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
            run(name, &upload);
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), crate::Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            // A draw, then `DRAWS` more with nothing changed, and the attachment walks of those.
            let after_first = |encoder: &mut RenderCommandEncoder| {
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                let walks = || TARGET_WALKS.with(Cell::get);
                let start = walks();
                for _ in 0..DRAWS {
                    draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                }
                walks() - start
            };
            draw(encoder.set_vertex_buffer(Some(&whole), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_render_pipeline_state(&plain));
            let undeclared = after_first(&mut encoder);
            draw(encoder.use_resource(Resource::Texture(&color), ResourceUsage::READ));
            draw(encoder.use_resource(Resource::Buffer(&position), ResourceUsage::READ));
            let declared = after_first(&mut encoder);
            draw(encoder.set_vertex_buffer(Some(&entry), 0, 0));
            draw(encoder.set_fragment_buffer(Some(&entry), 0, 0));
            draw(encoder.set_render_pipeline_state(&held));
            let unbound = after_first(&mut encoder);
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            let unslotted = after.unslotted_attachment_reads - before.unslotted_attachment_reads;
            println!(
                "held-read-walks: {name}: attachment walks of {DRAWS} draws after the first: \
                 {undeclared} before the target's read declaration, {declared} after it, \
                 {unbound} of the argument-buffer pipeline state; unslotted attachment reads \
                 {unslotted}"
            );
            assert_eq!(
                [undeclared, declared, unbound],
                [0; 3],
                "{name}: the attachments walked by {DRAWS} draws after the first: before the \
                 target's read declaration, after it, and of the argument-buffer pipeline state"
            );
            assert_eq!(
                unslotted,
                DRAWS + 1,
                "{name}: the unslotted attachment reads, one a draw of the argument-buffer state"
            );
        });
    }
}
