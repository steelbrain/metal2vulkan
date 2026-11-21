//! Visibility results (`setVisibilityResultMode:offset:`), under a device cfg on every device of the
//! cfg's kind, each pass an 8 by 8 `RGBA8Unorm` target that `colored_vertex` and
//! `colored_fragment` draw into, its visibility result buffer starting as `0xaa` bytes:
//!
//! - `Counting` and `Boolean` write the samples their draws pass, the CPU's count of the pixels
//!   each triangle covers at one sample: 64 for the whole target, 32 for its top half, three
//!   instances three times that, 0 for a triangle that covers no pixel and for a mode with no
//!   draw; `Disabled` counts nothing, and an offset no mode took keeps its bytes, a second pass
//!   rewriting only its own offset.
//! - A command buffer takes its occlusion queries from pools the device lends it and takes back as
//!   it retires, or as it drops uncommitted: 17 held at once are 17 pools, of which the device
//!   keeps 16 free and destroys the rest; a later command buffer of 300 modes takes two of them
//!   again, and makes none.
//! - 200 command buffers, one after another, each a pass that sets a `Counting` mode and draws,
//!   take one pool again and again: the pools read 1 after the 100th and the 200th, and 0 once
//!   the device drops.
//! - What Metal refuses is refused, the encoder's mode unchanged: a mode with no visibility result
//!   buffer, at an offset off a multiple of 8 or past the buffer, at an offset the encoder took
//!   before, and at a byte of the buffer's allocation off a multiple of 8.
//! - A mode in an `Accumulate` pass is `DrawKind::AccumulatedVisibilityResults`, counted, and the
//!   buffer keeps its bytes.
//! - While a mode's query counts the draws, a memory barrier after a draw that wrote memory, and a
//!   draw that reads its pass's attachment through a texture, which would end the rendering, are
//!   `DrawKind::VisibilityResultInterrupt`, counted; the same commands after a mode set and
//!   before its first draw run.
//!
//! `llvm-as` assembles the fixtures; without it a device test skips visibly.

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::command::on_device::{each_device, within};
    use crate::tests::draw::on_device::{
        descriptor, examples, floats, library, pass, readable, run, state, target, Buffers, BLUE,
    };
    use crate::{
        Buffer, ColorAttachmentDescriptor, DrawKind, Error, FunctionDescriptor, Gap, IndexClass,
        PixelFormat, PrimitiveType, Refusal, RenderCommandEncoder, RenderPipelineDescriptor,
        StageSource, TextureUsage, VisibilityResultMode, VisibilityResultType,
    };
    use metal2vulkan_devtools::engine_dir;

    /// A word no mode wrote.
    const UNWRITTEN: u64 = 0xaaaa_aaaa_aaaa_aaaa;

    /// The fixture's triangles for `colored_vertex`: the top half of clip space, the whole of it,
    /// and one of no area, three vertices each, all blue.
    fn triangles(device: &crate::Device, buffers: &mut Buffers) -> (Buffer, Buffer) {
        let top = [-1.0, 0.0, 3.0, 0.0, -1.0, 4.0];
        let whole = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let none = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let positions: Vec<f32> = [top, whole, none].concat();
        let colors: Vec<f32> = [BLUE; 9].concat();
        (
            buffers.of(device, &floats(&positions)),
            buffers.of(device, &floats(&colors)),
        )
    }

    /// The first vertex of each triangle of [`triangles`].
    const TOP: u32 = 0;
    const WHOLE: u32 = 3;
    const NONE: u32 = 6;

    /// The 64-bit words of `buffer`, `count` of them.
    fn words(buffer: &Buffer, count: usize) -> Vec<u64> {
        let contents = buffer.contents().expect("a Shared buffer");
        // SAFETY: the buffer holds `count` words, which the finished command buffers wrote and
        // whose writes reach the host.
        let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 8 * count) };
        (bytes.as_chunks::<8>().0.iter())
            .map(|word| u64::from_le_bytes(*word))
            .collect()
    }

    /// `result` as text, for comparing errors.
    fn text(result: Result<(), Error>) -> String {
        format!("{result:?}")
    }

    /// A refusal of `refusal`, as text.
    fn refused(refusal: Refusal) -> String {
        text(Err(Error::Refused(refusal)))
    }

    /// A gap of `kind`, as text.
    fn gap(kind: DrawKind) -> String {
        text(Err(Error::Unsupported(Gap::Draw { kind })))
    }

    /// `Counting` and `Boolean` count the samples their draws pass; `Disabled` counts none; an
    /// offset no mode took keeps its bytes; offsets 0 to 32, one run, and 56, another, are copied
    /// at the pass's end; a second pass rewrites its own offset alone.
    #[test]
    fn every_device_counts_the_samples_each_mode_s_draws_pass() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("visibility_counts", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = triangles(&device, &mut buffers);
            let visibility = buffers.of(&device, &[0xaa; 64]);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let encoder = |command_buffer: &crate::CommandBuffer| {
                let mut d = pass(&color, None);
                d.visibility_result_buffer = Some(visibility.clone());
                let mut encoder = (command_buffer.render_command_encoder(&d))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                ok(encoder.set_render_pipeline_state(&state));
                ok(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                ok(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                encoder
            };
            let draw = |encoder: &mut RenderCommandEncoder, first: u32, instances: u32| {
                ok(encoder.draw_primitives(PrimitiveType::Triangle, first, 3, instances, 0))
            };
            let (counting, boolean, disabled) = (
                VisibilityResultMode::Counting,
                VisibilityResultMode::Boolean,
                VisibilityResultMode::Disabled,
            );
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut e = encoder(&command_buffer);
            ok(e.set_visibility_result_mode(counting, 0));
            draw(&mut e, WHOLE, 1);
            ok(e.set_visibility_result_mode(counting, 8));
            draw(&mut e, TOP, 1);
            draw(&mut e, WHOLE, 1);
            ok(e.set_visibility_result_mode(boolean, 16));
            draw(&mut e, WHOLE, 1);
            ok(e.set_visibility_result_mode(boolean, 24));
            draw(&mut e, NONE, 1);
            ok(e.set_visibility_result_mode(counting, 32));
            ok(e.set_visibility_result_mode(disabled, 40));
            draw(&mut e, WHOLE, 1);
            ok(e.set_visibility_result_mode(counting, 56));
            draw(&mut e, WHOLE, 3);
            e.end_encoding();
            run(name, &command_buffer);
            // `Boolean` writes the count, nonzero exactly where a sample passed.
            let first = [64, 96, 64, 0, 0, UNWRITTEN, UNWRITTEN, 192];
            assert_eq!(
                words(&visibility, 8),
                first,
                "{name}: the first pass's words"
            );
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut e = encoder(&command_buffer);
            ok(e.set_visibility_result_mode(counting, 48));
            draw(&mut e, TOP, 1);
            e.end_encoding();
            run(name, &command_buffer);
            let second = [64, 96, 64, 0, 0, UNWRITTEN, 32, 192];
            assert_eq!(
                words(&visibility, 8),
                second,
                "{name}: the second pass's words"
            );
            println!("{name}: visibility results {second:x?}");
        });
    }

    /// A command buffer's pools come from the device and go back as it retires or drops: 17
    /// command buffers that each set a mode, held at once, hold 17 pools; dropped uncommitted,
    /// the device keeps 16 free and destroys one. A command buffer of 300 modes in one encoder,
    /// whose queries fill one pool and start a second, takes two of the free ones, makes none,
    /// and writes 0 at each of the 300 offsets, a mode with no draw counting none; after it
    /// retires the device holds 16 again, and none once it drops.
    #[test]
    fn every_device_takes_back_and_bounds_its_occlusion_pools() {
        each_device("visibility_pools", |name, _, device| {
            let mut buffers = Buffers::new(&device);
            let visibility = buffers.of(&device, &[0xaa; 8 * 301]);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(32).expect("a queue");
            let counters = device.counters();
            let pools = || counters.gauges().occlusion_pools;
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let encode = |modes: u64| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut d = pass(&color, None);
                d.visibility_result_buffer = Some(visibility.clone());
                let mut encoder = (command_buffer.render_command_encoder(&d))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                for at in 0..modes {
                    ok(encoder.set_visibility_result_mode(VisibilityResultMode::Counting, 8 * at));
                }
                encoder.end_encoding();
                command_buffer
            };
            assert_eq!(pools(), 0, "{name}: a new device's pools");
            let held: Vec<_> = (0..17).map(|_| encode(1)).collect();
            assert_eq!(pools(), 17, "{name}: one pool for each command buffer");
            drop(held);
            assert_eq!(pools(), 16, "{name}: the pools kept free");
            let command_buffer = encode(300);
            assert_eq!(pools(), 16, "{name}: 300 queries from two free pools");
            run(name, &command_buffer);
            drop(command_buffer);
            within("the pools' return", || pools() == 16);
            let mut want = vec![0; 300];
            want.push(UNWRITTEN);
            assert_eq!(words(&visibility, 301), want, "{name}: the 300 words");
            println!("{name}: 17 pools held, 16 kept free, 300 queries from two of them");
        });
    }

    /// 200 command buffers, each a pass that sets a `Counting` mode and draws the whole target,
    /// run one after another: each takes the pool the last gave back as it retired, so the
    /// device's pools read 1 after the 100th and after the 200th, the word 64; and 0 once the
    /// device drops.
    #[test]
    fn every_device_s_occlusion_pools_stay_flat_over_200_passes_and_end_with_it() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("visibility_pools_flat", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = triangles(&device, &mut buffers);
            let visibility = buffers.of(&device, &[0xaa; 8]);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut at = Vec::new();
            for iteration in 1..=200 {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut d = pass(&color, None);
                d.visibility_result_buffer = Some(visibility.clone());
                let mut e = (command_buffer.render_command_encoder(&d))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                ok(e.set_render_pipeline_state(&state));
                ok(e.set_vertex_buffer(Some(&positions), 0, 0));
                ok(e.set_vertex_buffer(Some(&colors), 0, 1));
                ok(e.set_visibility_result_mode(VisibilityResultMode::Counting, 0));
                ok(e.draw_primitives(PrimitiveType::Triangle, WHOLE, 3, 1, 0));
                e.end_encoding();
                run(name, &command_buffer);
                if iteration == 100 || iteration == 200 {
                    at.push(counters.gauges().occlusion_pools);
                }
            }
            assert_eq!(
                at,
                [1, 1],
                "{name}: the pools after the 100th and the 200th"
            );
            assert_eq!(words(&visibility, 1), [64], "{name}: the word");
            drop((state, buffers, positions, colors, visibility, color, queue));
            drop(device);
            assert_eq!(
                counters.gauges().occlusion_pools,
                0,
                "{name}: the pools at the device's drop"
            );
            println!("{name}: 200 passes flat at 1 pool after the 100th and the 200th; 0 at drop");
        });
    }

    /// Metal's refusals of a mode, each leaving the encoder's mode as it was: no visibility result
    /// buffer, an offset off a multiple of 8, past the buffer, taken before in the encoder, and
    /// one at a byte of the buffer's allocation off a multiple of 8. `Disabled` takes any offset
    /// with no buffer. The mode set after them counts the encoder's draw.
    #[test]
    fn every_device_refuses_what_metal_refuses_of_a_visibility_mode() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("visibility_refusals", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = triangles(&device, &mut buffers);
            let visibility = buffers.of(&device, &[0xaa; 16]);
            let skewed = buffers.at(&device, &[0xaa; 16], 4);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let counting = VisibilityResultMode::Counting;
            // No buffer: `Disabled` at any offset, and no other mode.
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut e = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(e.set_visibility_result_mode(VisibilityResultMode::Disabled, 1 << 40));
            assert_eq!(
                text(e.set_visibility_result_mode(VisibilityResultMode::Boolean, 0)),
                refused(Refusal::NoVisibilityResultBuffer),
                "{name}: no buffer"
            );
            e.end_encoding();
            run(name, &command_buffer);
            // A buffer whose allocation byte is off a multiple of 8.
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut d = pass(&color, None);
            d.visibility_result_buffer = Some(skewed.clone());
            let mut e = (command_buffer.render_command_encoder(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let (_, at) = crate::buffer::BufferObject::device_range(&skewed);
            assert_eq!(
                text(e.set_visibility_result_mode(counting, 8)),
                refused(Refusal::ArgumentPlace {
                    place: at + 8,
                    alignment: 8
                }),
                "{name}: a skewed buffer"
            );
            e.end_encoding();
            run(name, &command_buffer);
            // Offsets off 8, past the buffer and taken again.
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut d = pass(&color, None);
            d.visibility_result_buffer = Some(visibility.clone());
            let mut e = (command_buffer.render_command_encoder(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(e.set_render_pipeline_state(&state));
            ok(e.set_vertex_buffer(Some(&positions), 0, 0));
            ok(e.set_vertex_buffer(Some(&colors), 0, 1));
            for offset in [4, 16, u64::MAX - 7] {
                assert_eq!(
                    text(e.set_visibility_result_mode(counting, offset)),
                    refused(Refusal::VisibilityResultOffset { offset, length: 16 }),
                    "{name}: offset {offset}"
                );
            }
            ok(e.set_visibility_result_mode(counting, 0));
            ok(e.draw_primitives(PrimitiveType::Triangle, WHOLE, 3, 1, 0));
            ok(e.set_visibility_result_mode(VisibilityResultMode::Disabled, 0));
            assert_eq!(
                text(e.set_visibility_result_mode(VisibilityResultMode::Boolean, 0)),
                refused(Refusal::VisibilityResultOffsetTaken { offset: 0 }),
                "{name}: offset 0 again"
            );
            ok(e.set_visibility_result_mode(counting, 8));
            ok(e.draw_primitives(PrimitiveType::Triangle, TOP, 3, 1, 0));
            e.end_encoding();
            run(name, &command_buffer);
            assert_eq!(words(&visibility, 2), [64, 32], "{name}: the words");
            println!("{name}: Metal's refusals of a visibility mode refused");
        });
    }

    /// A `Boolean` or `Counting` mode in an `Accumulate` pass is not built, and counted; the pass
    /// draws, and the buffer keeps its bytes.
    #[test]
    fn every_device_refuses_an_accumulated_mode_and_counts_it() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("visibility_accumulate", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = triangles(&device, &mut buffers);
            let visibility = buffers.of(&device, &[0xaa; 16]);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();
            let before = counters.metrics().accumulated_visibility_refused;
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut d = pass(&color, None);
            d.visibility_result_buffer = Some(visibility.clone());
            d.visibility_result_type = VisibilityResultType::Accumulate;
            let mut e = (command_buffer.render_command_encoder(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(e.set_render_pipeline_state(&state));
            ok(e.set_vertex_buffer(Some(&positions), 0, 0));
            ok(e.set_vertex_buffer(Some(&colors), 0, 1));
            for mode in [
                VisibilityResultMode::Boolean,
                VisibilityResultMode::Counting,
            ] {
                assert_eq!(
                    text(e.set_visibility_result_mode(mode, 0)),
                    gap(DrawKind::AccumulatedVisibilityResults),
                    "{name}: {mode:?}"
                );
            }
            ok(e.set_visibility_result_mode(VisibilityResultMode::Disabled, 0));
            ok(e.draw_primitives(PrimitiveType::Triangle, WHOLE, 3, 1, 0));
            e.end_encoding();
            run(name, &command_buffer);
            assert_eq!(
                counters.metrics().accumulated_visibility_refused - before,
                2,
                "{name}: the refusals counted"
            );
            assert_eq!(
                words(&visibility, 2),
                [UNWRITTEN; 2],
                "{name}: the buffer's bytes"
            );
            println!("{name}: an accumulated mode refused and counted");
        });
    }

    /// While a mode's query counts the draws, a memory barrier after a draw that wrote memory is
    /// refused and counted, the encoder as it was; after the next mode set, before its first
    /// draw, the same barrier runs. The draws store and rasterize nothing, so both words are 0.
    #[test]
    fn every_device_refuses_a_memory_barrier_while_a_mode_counts() {
        let Some(engine) = library(&engine_dir().join("rvc_draw_bases.ll")) else {
            return;
        };
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            (engine.function("draw_bases_in_a_vertex_function")).expect("the function"),
        )));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.rasterization_enabled = false;
        each_device("visibility_barrier", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let out = buffers.of(&device, &[0u8; 32 * 16]);
            let visibility = buffers.of(&device, &[0xaa; 16]);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();
            let before = counters.metrics();
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let counting = VisibilityResultMode::Counting;
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut d = pass(&color, None);
            d.visibility_result_buffer = Some(visibility.clone());
            let mut e = (command_buffer.render_command_encoder(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(e.set_render_pipeline_state(&state));
            ok(e.set_vertex_buffer(Some(&out), 0, 0));
            ok(e.set_visibility_result_mode(counting, 0));
            ok(e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            assert_eq!(
                text(e.memory_barrier()),
                gap(DrawKind::VisibilityResultInterrupt),
                "{name}: a barrier while the query counts"
            );
            ok(e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            ok(e.set_visibility_result_mode(counting, 8));
            ok(e.memory_barrier());
            ok(e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            e.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                after.visibility_interrupts_refused - before.visibility_interrupts_refused,
                1,
                "{name}: the refusal counted"
            );
            assert_eq!(words(&visibility, 2), [0, 0], "{name}: the words");
            println!("{name}: a memory barrier refused while a mode counted, run before one");
        });
    }

    /// While a mode's query counts the draws, a draw that reads its pass's attachment through a
    /// texture, which ends the rendering first (a move to the feedback layout, or a copy into the
    /// attachment's shadow), is refused and counted; after the next mode set it draws, ending the
    /// rendering before its query begins. A device without the attachment reads refuses the read
    /// as `DrawKind::AttachmentRead` in either place.
    #[test]
    fn every_device_refuses_an_attachment_read_while_a_mode_counts() {
        let Some(library) = examples() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let own = descriptor(&library, "own_pixel_fragment", PixelFormat::RGBA8Unorm);
        each_device("visibility_attachment_read", |name, _, device| {
            (device.grant_indices(IndexClass::Texture, 4096, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let reads = device.core().attachment_reads;
            let plain = state(name, &device, &plain);
            let own = state(name, &device, &own);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = triangles(&device, &mut buffers);
            let visibility = buffers.of(&device, &[0xaa; 16]);
            let color = readable(&device, PixelFormat::RGBA8Unorm, 1, 4096);
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();
            let before = counters.metrics().visibility_interrupts_refused;
            let ok = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let counting = VisibilityResultMode::Counting;
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut d = pass(&color, None);
            d.visibility_result_buffer = Some(visibility.clone());
            let mut e = (command_buffer.render_command_encoder(&d))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            ok(e.set_vertex_buffer(Some(&positions), 0, 0));
            ok(e.set_vertex_buffer(Some(&colors), 0, 1));
            ok(e.set_fragment_texture(Some(&color), 0));
            ok(e.set_render_pipeline_state(&plain));
            ok(e.set_visibility_result_mode(counting, 0));
            ok(e.draw_primitives(PrimitiveType::Triangle, WHOLE, 3, 1, 0));
            ok(e.set_render_pipeline_state(&own));
            let read = |e: &mut RenderCommandEncoder| {
                text(e.draw_primitives(PrimitiveType::Triangle, WHOLE, 3, 1, 0))
            };
            let want = match reads {
                true => gap(DrawKind::VisibilityResultInterrupt),
                false => gap(DrawKind::AttachmentRead),
            };
            assert_eq!(read(&mut e), want, "{name}: a read while the query counts");
            ok(e.set_visibility_result_mode(counting, 8));
            let want = match reads {
                true => text(Ok(())),
                false => gap(DrawKind::AttachmentRead),
            };
            assert_eq!(read(&mut e), want, "{name}: a read before the query begins");
            e.end_encoding();
            run(name, &command_buffer);
            let refusals = counters.metrics().visibility_interrupts_refused - before;
            let (count, words_want) = match reads {
                true => (1, [64, 64]),
                false => (0, [64, 0]),
            };
            assert_eq!(refusals, count, "{name}: the refusals counted");
            assert_eq!(words(&visibility, 2), words_want, "{name}: the words");
            println!("{name}: an attachment read refused while a mode counted, run before one");
        });
    }
}
