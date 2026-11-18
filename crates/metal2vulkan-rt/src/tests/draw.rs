//! Draws: the render encoder's state and its draws.
//!
//! - A viewport is Vulkan's with its height negated, from `y + height`; one of no width or height
//!   is one pixel wide or high and covers no pixel; the pixels a viewport covers are its own,
//!   fractional edges widened, cut to the render area.
//! - Under a device cfg, on every device of the cfg's kind, of the owned fixtures
//!   `examples/runtime_examples.ll` and `engine/fca_vertex_unread_attribute.ll`:
//!   - triangles whose vertex function reads its positions and colors from buffers by vertex ID
//!     land where Metal's clip space puts them (y up): a direct draw, an indexed draw of 16-bit
//!     indices through a scissor rectangle, and a direct draw through a viewport, which cuts it
//!     to its pixels. The pass is one submission, its two barriers and the segment's end, makes no
//!     pipeline and copies no byte of the runtime's own; a memory barrier after draws that wrote
//!     no memory records nothing.
//!   - a draw of lines makes its content's line `VkPipeline` once, a first-use wait, and draws
//!     its row.
//!   - a stencil state that replaces the stencil marks what a draw covers, and one that tests it
//!     keeps a later draw to those pixels.
//!   - a fragment function samples a texture through its constexpr sampler at the index the
//!     consumer gave it: each pixel reads its own texel.
//!   - a vertex function's `[[stage_in]]` attributes come from a vertex buffer, and from
//!     `set_vertex_bytes`' copy, fetched in fixed function; an unnormalized signed-integer
//!     attribute (the owned fixture `engine/icb_merged_attributes.ll`'s `short` mark) is
//!     sign-extended into a variable declared `int`, and a layout stepped per instance at rate 2
//!     is the translator's refusal; a layout at Metal's constant step reads its first element
//!     for every vertex and instance, from a base instance of 1.
//!   - a draw whose vertex buffer is bound off its argument's alignment is refused before
//!     anything is recorded, and the same buffers bound aligned draw.
//!   - 32-bit indices 2 bytes past a multiple of 256 in their block are refused before anything
//!     is recorded, and 16-bit indices there draw.
//!   - what Metal refuses is refused, and what a draw does not build is a gap: no pipeline state,
//!     a pipeline state of other formats than the pass's, points, a depth test, a stencil test
//!     without a stencil attachment, a misaligned index offset, a non-finite viewport and a
//!     mirrored one.
//!
//! `llvm-as` assembles the fixtures; without it a device test skips visibly.

use crate::render_encoder::{viewport_pixels, vk_viewport};
use crate::Viewport;

/// A viewport of `x`, `y`, `width` and `height`, depths 0 to 1.
fn viewport(x: f64, y: f64, width: f64, height: f64) -> Viewport {
    Viewport {
        origin_x: x,
        origin_y: y,
        width,
        height,
        znear: 0.0,
        zfar: 1.0,
    }
}

/// Vulkan's viewport is Metal's flipped; one of no extent is a pixel that covers nothing.
#[test]
fn a_viewport_is_vulkans_flipped_and_covers_its_own_pixels() {
    let v = vk_viewport(&viewport(2.0, 4.0, 8.0, 6.0));
    assert_eq!([v.x, v.y, v.width, v.height], [2.0, 10.0, 8.0, -6.0]);
    // Metal's own flip is a positive Vulkan height.
    let v = vk_viewport(&viewport(0.0, 8.0, 4.0, -8.0));
    assert_eq!([v.x, v.y, v.width, v.height], [0.0, 0.0, 4.0, 8.0]);
    let v = vk_viewport(&viewport(1.0, 1.0, 0.0, 0.0));
    assert_eq!([v.width, v.height], [1.0, -1.0]);
    // Fractional edges widen to whole pixels, cut to the area; no extent covers nothing.
    assert_eq!(
        viewport_pixels(&viewport(0.5, 1.25, 3.0, 2.5), [8, 8]),
        [[0, 4], [1, 4]]
    );
    assert_eq!(
        viewport_pixels(&viewport(-4.0, 6.0, 16.0, 8.0), [8, 8]),
        [[0, 8], [6, 8]]
    );
    assert_eq!(
        viewport_pixels(&viewport(0.0, 8.0, 4.0, -8.0), [8, 8]),
        [[0, 4], [0, 8]]
    );
    assert_eq!(
        viewport_pixels(&viewport(2.0, 2.0, 0.0, 4.0), [8, 8]),
        [[2, 2], [2, 2]]
    );
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        Block, Buffer, ColorAttachmentDescriptor, CommandBuffer, CompareFunction, CullMode,
        DepthStencilDescriptor, Device, DrawKind, Error, FunctionDescriptor, Gap, IndexBuffer,
        IndexClass, IndexType, Library, MemoryClass, NumberedSampler, PixelFormat, PrimitiveType,
        Refusal, Region, RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState,
        ScissorRect, StageSource, StencilOperation, StorageMode, Texture, TextureDescriptor,
        TextureType, TextureUsage, Viewport,
    };
    use metal2vulkan::error::Feature;
    use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
    use metal2vulkan::metal::{DataType, VertexFormat, VertexStepFunction};
    use metal2vulkan::pipeline::{
        VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
    };
    use metal2vulkan_devtools::{engine_dir, fixtures_dir, llvm_assemble, AssembleError};
    use std::path::Path;

    /// The side of the targets.
    const SIDE: u32 = 8;
    /// The constexpr sampler's index the test gives `sampled_fragment`.
    const CONSTEXPR: u32 = 3;

    /// The library of `ll`, or None where `llvm-as` is missing.
    fn library(ll: &Path) -> Option<Library> {
        match llvm_assemble(ll) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{}: {error:?}", ll.display()),
        }
    }

    fn examples() -> Option<Library> {
        library(&fixtures_dir().join("examples/runtime_examples.ll"))
    }

    /// `colored_vertex` with `fragment`, drawing into one colour attachment of `color`.
    fn descriptor(
        library: &Library,
        fragment: &str,
        color: PixelFormat,
    ) -> RenderPipelineDescriptor {
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("colored_vertex"));
        d.fragment = Some(function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(color));
        d
    }

    fn state(name: &str, device: &Device, d: &RenderPipelineDescriptor) -> RenderPipelineState {
        (device.new_render_pipeline_state(d, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// Shared buffers in one block, 256 bytes apart.
    struct Buffers {
        block: Block,
        next: u64,
    }

    impl Buffers {
        fn new(device: &Device) -> Buffers {
            Buffers {
                block: (device.take_block(MemoryClass::Shared, 65536)).expect("a block"),
                next: 0,
            }
        }

        /// A buffer of `bytes`.
        fn of(&mut self, device: &Device, bytes: &[u8]) -> Buffer {
            self.at(device, bytes, 0)
        }

        /// A buffer of `bytes`, `skew` bytes past the place [`of`](Self::of) would take.
        fn at(&mut self, device: &Device, bytes: &[u8], skew: u64) -> Buffer {
            let len = bytes.len() as u64;
            let at = self.next + skew;
            let buffer = (device.new_buffer(self.block.id, at, len, Some(0))).expect("a buffer");
            self.next += (skew + len).next_multiple_of(256);
            let contents = buffer.contents().expect("a Shared buffer");
            // SAFETY: the range lies in the buffer, which is alive, and no GPU work uses it.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
            buffer
        }
    }

    fn floats(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A 2D render target of `format`, `SIDE` a side.
    fn target(device: &Device, format: PixelFormat, usage: TextureUsage) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
        d.usage = usage;
        d.storage_mode = StorageMode::Shared;
        (device.new_texture(&d, None)).unwrap_or_else(|error| panic!("{format:?}: {error}"))
    }

    /// The texels of `texture`, `unit` bytes each.
    fn read(texture: &Texture, unit: usize) -> Vec<u8> {
        let row = SIDE as usize * unit;
        let mut bytes = vec![0; row * SIDE as usize];
        let region = Region {
            origin: [0; 3],
            size: [SIDE, SIDE, 1],
        };
        (texture.get_bytes(
            &mut bytes,
            row as u64,
            (row * SIDE as usize) as u64,
            region,
            0,
            0,
        ))
        .expect("get_bytes");
        bytes
    }

    /// A pass that clears `color` to opaque black, with `stencil` cleared to 0 where given.
    fn pass(color: &Texture, stencil: Option<&Texture>) -> RenderPassDescriptor {
        let mut d = RenderPassDescriptor::default();
        let c = &mut d.color_attachments[0];
        c.texture = Some(color.clone());
        c.load_action = crate::LoadAction::Clear;
        if let Some(stencil) = stencil {
            d.stencil_attachment.texture = Some(stencil.clone());
        }
        d
    }

    /// `command_buffer` committed and complete with no error.
    fn run(name: &str, command_buffer: &CommandBuffer) {
        command_buffer.commit().expect("a commit");
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// The RGBA8 texel `color` of a float colour of 0 or 1 channels.
    fn rgba8(color: [f32; 4]) -> [u8; 4] {
        color.map(|c| (c * 255.0) as u8)
    }

    const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
    const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    const WHITE: [f32; 4] = [1.0; 4];

    /// The fixture's vertices, positions and colors: the top half of clip space (y up) in green,
    /// then the whole of it in blue and in white, three each.
    fn vertices(device: &Device, buffers: &mut Buffers) -> (Buffer, Buffer) {
        let top = [-1.0, 0.0, 3.0, 0.0, -1.0, 4.0];
        let whole = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let positions: Vec<f32> = [top, whole, whole].concat();
        let colors: Vec<f32> = [GREEN, BLUE, WHITE]
            .iter()
            .flat_map(|c| [*c; 3].concat())
            .collect();
        (
            buffers.of(device, &floats(&positions)),
            buffers.of(device, &floats(&colors)),
        )
    }

    /// The texels of `SIDE` rows of `SIDE` texels, `texel(x, y)` each.
    fn texels(texel: impl Fn(u32, u32) -> [f32; 4]) -> Vec<u8> {
        (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
            .flat_map(|(x, y)| rgba8(texel(x, y)))
            .collect()
    }

    /// Triangles land where Metal's clip space puts them, through a scissor rectangle and a
    /// viewport; the pass is three barriers, no pipeline made and no byte copied.
    #[test]
    fn every_device_draws_triangles_where_metal_puts_them() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("draw_triangles", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let indices = buffers.of(&device, &[3, 0, 4, 0, 5, 0, 0, 0]);
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
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            // The top half, green.
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            draw(encoder.memory_barrier());
            // The whole target in blue, through the bottom-right quarter's rectangle.
            encoder.set_scissor_rect(ScissorRect {
                x: 4,
                y: 4,
                width: 4,
                height: 40,
            });
            let indexed = IndexBuffer {
                buffer: &indices,
                offset: 0,
                index_type: IndexType::UInt16,
            };
            draw(encoder.draw_indexed_primitives(PrimitiveType::Triangle, 3, indexed, 1, 0, 0));
            // The whole target in white, through the bottom-left quarter's viewport, which cuts
            // it to its pixels.
            encoder.set_scissor_rect(ScissorRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            });
            draw(encoder.set_viewport(Viewport {
                origin_x: 0.0,
                origin_y: 4.0,
                width: 4.0,
                height: 4.0,
                znear: 0.0,
                zfar: 1.0,
            }));
            encoder.set_cull_mode(CullMode::None);
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 6, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.passes - before.passes,
                    after.pipelines_created - before.pipelines_created,
                    after.bytes_copied - before.bytes_copied,
                ),
                (1, 1, 1, 0, 0),
                "{name}: submissions, barriers, passes, pipelines made, bytes copied"
            );
            let want = texels(|x, y| match (x, y) {
                (_, 0..4) => GREEN,
                (0..4, _) => WHITE,
                _ => BLUE,
            });
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!("{name}: three draws in one pass, 1 barrier, no pipeline made");
        });
    }

    /// A draw of lines makes the line pipeline once and draws its row.
    #[test]
    fn every_device_makes_its_line_pipeline_at_the_first_draw_of_lines() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("draw_lines", |name, _, device| {
            let state = state(name, &device, &d);
            // A line across row 2's centres, Metal's y up.
            let y = 1.0 - 2.5 / 4.0;
            let mut buffers = Buffers::new(&device);
            let positions = buffers.of(&device, &floats(&[-1.0, y, 1.0, y]));
            let colors = buffers.of(&device, &floats(&[GREEN, GREEN].concat()));
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let counters = device.counters();
            for round in 0..2 {
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let draw =
                    |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
                draw(encoder.set_render_pipeline_state(&state));
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                draw(encoder.draw_primitives(PrimitiveType::Line, 0, 2, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let after = counters.metrics();
                let made = u64::from(round == 0);
                assert_eq!(
                    (
                        after.pipelines_created - before.pipelines_created,
                        after.first_use_waits - before.first_use_waits,
                    ),
                    (made, made),
                    "{name}: round {round}: pipelines made, first-use waits"
                );
                let texels = read(&color, 4);
                let at = |x: u32, y: u32| {
                    let i = ((y * SIDE + x) * 4) as usize;
                    [texels[i], texels[i + 1], texels[i + 2], texels[i + 3]]
                };
                for x in 1..SIDE - 1 {
                    assert_eq!(at(x, 2), rgba8(GREEN), "{name}: ({x}, 2)");
                    assert_eq!(at(x, 5), rgba8(BLACK), "{name}: ({x}, 5)");
                }
            }
            println!("{name}: one line pipeline, made at the first draw of lines");
        });
    }

    /// A stencil state that replaces marks the top half; one that tests keeps a later draw there.
    #[test]
    fn every_device_marks_and_tests_the_stencil() {
        let Some(library) = examples() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let mut d = plain.clone();
        d.stencil_format = Some(PixelFormat::Stencil8);
        each_device("draw_stencil", |name, _, device| {
            let stenciled = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
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
            let mut equal = DepthStencilDescriptor::default();
            for face in [&mut equal.front_face_stencil, &mut equal.back_face_stencil] {
                face.stencil_compare_function = CompareFunction::Equal;
            }
            let (replace, equal) = (
                device.new_depth_stencil_state(&replace),
                device.new_depth_stencil_state(&equal),
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer
                .render_command_encoder(&pass(&color, Some(&stencil))))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&stenciled));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            // 0x1a5 saturates to 255 as Metal stores it; the test compares what the first draw
            // stored.
            encoder.set_stencil_reference_value(0x1a5);
            draw(encoder.set_depth_stencil_state(Some(&replace)));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            draw(encoder.set_depth_stencil_state(Some(&equal)));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let want = texels(|_, y| match y {
                0..4 => BLUE,
                _ => BLACK,
            });
            assert_eq!(read(&color, 4), want, "{name}: the tested draw");
            if stencil.unit().is_some() {
                let want: Vec<u8> = (0..SIDE * SIDE)
                    .map(|i| if i < SIDE * SIDE / 2 { 255 } else { 0 })
                    .collect();
                assert_eq!(read(&stencil, 1), want, "{name}: the stencil");
            }

            // A stencil test in a pass without a stencil attachment is refused.
            let plain = state(name, &device, &plain);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_render_pipeline_state(&plain));
            draw(encoder.set_depth_stencil_state(Some(&equal)));
            assert!(
                matches!(
                    encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
                    Err(Error::Refused(Refusal::StencilAttachment))
                ),
                "{name}: a stencil test without a stencil attachment"
            );
            encoder.end_encoding();
            run(name, &command_buffer);
            println!("{name}: the stencil marks and tests");
        });
    }

    /// Each pixel samples its own texel through the constexpr sampler at the consumer's index.
    #[test]
    fn every_device_samples_a_texture_through_a_constexpr_sampler_in_a_draw() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "sampled_fragment", PixelFormat::RGBA16Float);
        each_device("draw_sampled", |name, _, device| {
            for (class, first) in [(IndexClass::Texture, 4096), (IndexClass::Sampler, 1)] {
                (device.grant_indices(class, first, 4))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            let state = (device.new_render_pipeline_state(&d, |samplers| {
                assert!(
                    matches!(samplers, [NumberedSampler::Constexpr(_)]),
                    "{name}: {samplers:?}"
                );
                Ok(vec![CONSTEXPR])
            }))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut sd =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA16Float, SIDE, SIDE);
            sd.usage = TextureUsage::SHADER_READ;
            let source = (device.new_texture(&sd, Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            // Each texel's channels in sixteenths, which a half holds exactly: x, y, x + y, 1.
            let half = |k: u32| -> [u8; 2] {
                match k {
                    0 => [0, 0],
                    _ => {
                        let bits = (k as f32 / 16.0).to_bits();
                        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
                        (((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16).to_le_bytes()
                    }
                }
            };
            let bytes: Vec<u8> = (0..SIDE)
                .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
                .flat_map(|(x, y)| [x, y, x + y, 16].into_iter().flat_map(half))
                .collect();
            let whole = Region {
                origin: [0; 3],
                size: [SIDE, SIDE, 1],
            };
            let row = u64::from(SIDE * 8);
            (source.replace_region(whole, 0, 0, &bytes, row, row * u64::from(SIDE)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA16Float,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_fragment_texture(Some(&source), 0));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            assert_eq!(read(&color, 8), bytes, "{name}: each pixel its own texel");
            println!("{name}: a constexpr sampler at index {CONSTEXPR} sampled each texel");
        });
    }

    /// `[[stage_in]]` attributes come from a vertex buffer, and from `set_vertex_bytes`' copy.
    #[test]
    fn every_device_fetches_vertex_attributes_from_vertex_buffers_and_bytes() {
        let Some(library) = library(&engine_dir().join("fca_vertex_unread_attribute.ll")) else {
            return;
        };
        let constants = FunctionConstantValues::defaults()
            .with_index(0, FunctionConstantValue::new(DataType::Bool, [1]));
        let function = |name| {
            let mut f = FunctionDescriptor::new(library.function(name).expect("the function"));
            f.constants = constants.clone();
            f.specialized = true;
            StageSource::Function(f)
        };
        let mut d = RenderPipelineDescriptor::new(function("fca_vertex_unread_attribute"));
        d.fragment = Some(function("fca_fragment_unread_attribute"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let mut vertex = VertexDescriptor::new();
        for a in 0..2 {
            let attribute = VertexAttributeDescriptor::new(VertexFormat::Float4, 16 * a, 0);
            vertex.attributes.insert(a, attribute);
        }
        let layout = VertexBufferLayoutDescriptor::new(Some(32));
        vertex.layouts.insert(0, layout);
        d.vertex_descriptor = Some(vertex);
        // Three `{ position, color }` vertices that cover the target, `color` (1, 0, 0, 1),
        // which the vertex function scales by 0.5 and biases by 0.25.
        let red = [1.0, 0.0, 0.0, 1.0];
        let records: Vec<f32> = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]
            .iter()
            .flat_map(|[x, y]| [[*x, *y, 0.0, 1.0], red].concat())
            .collect();
        let records = floats(&records);
        each_device("draw_stage_in", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let vertices = buffers.of(&device, &records);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            for bytes in [false, true] {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let draw =
                    |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
                draw(encoder.set_render_pipeline_state(&state));
                draw(match bytes {
                    false => encoder.set_vertex_buffer(Some(&vertices), 0, 0),
                    true => encoder.set_vertex_bytes(&records, 0),
                });
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                // Metal's (191, 64, 64, 191), each channel within one code: near a tie the
                // device may store the other code nearest it.
                for texel in read(&color, 4).chunks(4) {
                    let near = (texel.iter().zip([191u8, 64, 64, 191]))
                        .all(|(&got, want)| got.abs_diff(want) <= 1);
                    assert!(near, "{name}: bytes {bytes}: {texel:?}");
                }
            }
            println!("{name}: attributes from a vertex buffer and from bytes");
        });
    }

    /// An unnormalized signed-integer attribute is fetched in fixed function into a variable
    /// declared `int`: `ima_vertex`'s mark, a `short` of -2, reaches `ima_fragment`'s unsigned
    /// output as its sign-extended bits.
    #[test]
    fn every_device_fetches_a_signed_attribute_in_fixed_function() {
        let Some(library) = library(&engine_dir().join("icb_merged_attributes.ll")) else {
            return;
        };
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("ima_vertex"));
        d.fragment = Some(function("ima_fragment"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        let mut vertex = VertexDescriptor::new();
        let attributes = [(VertexFormat::Float2, 8), (VertexFormat::Short, 4)];
        for (index, (format, stride)) in (0u32..).zip(attributes) {
            let attribute = VertexAttributeDescriptor::new(format, 0, index);
            vertex.attributes.insert(index, attribute);
            vertex
                .layouts
                .insert(index, VertexBufferLayoutDescriptor::new(Some(stride)));
        }
        d.vertex_descriptor = Some(vertex);
        // `ima_vertex` maps pixel positions to clip space by the frame's width and height: a
        // triangle twice the target's side covers it.
        let side = SIDE as f32;
        let positions = floats(&[0.0, 0.0, 2.0 * side, 0.0, 0.0, 2.0 * side]);
        let frame = floats(&[side, side, side, 0.0]);
        // Three marks of -2, each a `short` at the start of its 4-byte element.
        let marks: Vec<u8> = [(-2i16).to_le_bytes(), [0; 2]].concat().repeat(3);
        each_device("draw_signed_attribute", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let xy = buffers.of(&device, &positions);
            let mark = buffers.of(&device, &marks);
            let color = target(
                &device,
                PixelFormat::RGBA32Uint,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&xy), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&mark), 0, 1));
            draw(encoder.set_vertex_bytes(&frame, 2));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            for texel in read(&color, 16).chunks(16) {
                let word = |at: usize| u32::from_le_bytes([0, 1, 2, 3].map(|i| texel[at + i]));
                assert_eq!((word(0), word(12)), (0xFFFF_FFFE, 1), "{name}: {texel:?}");
            }
            println!("{name}: a short of -2 read as int");
        });
    }

    /// `ima_vertex` into `ima_fragment`'s `RGBA32Uint` output, with positions at layout 0 and a
    /// `uint` mark at layout 1 stepped by `step` at `rate`.
    fn marked(library: &Library, step: VertexStepFunction, rate: u32) -> RenderPipelineDescriptor {
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("ima_vertex"));
        d.fragment = Some(function("ima_fragment"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        let mut vertex = VertexDescriptor::new();
        let attributes = [(VertexFormat::Float2, 8), (VertexFormat::UInt, 4)];
        for (index, (format, stride)) in (0u32..).zip(attributes) {
            let attribute = VertexAttributeDescriptor::new(format, 0, index);
            vertex.attributes.insert(index, attribute);
            vertex
                .layouts
                .insert(index, VertexBufferLayoutDescriptor::new(Some(stride)));
        }
        let marks = vertex.layouts.get_mut(&1).expect("the marks' layout");
        marks.step_function = step;
        marks.step_rate = rate;
        d.vertex_descriptor = Some(vertex);
        d
    }

    /// Two instances of `state`'s triangle ([`marked`]), drawn from vertex 1 at `base_instance`
    /// over the marks `0xA0` to `0xA3`: each texel's four words, or the draw's error.
    fn two_instances(
        name: &str,
        device: &Device,
        state: &RenderPipelineState,
        base_instance: u32,
    ) -> Result<Vec<[u32; 4]>, Error> {
        // `ima_vertex` maps pixel positions to clip space by the frame's width and height and
        // moves instance `iid` down `iid` rows of the frame's row stride, 4. Vertices 1 to 3 are a
        // triangle twice the target's side, starting 4 rows above it: instance 1 covers the whole
        // target, instance 2 its lower half and instance 0 less than all of it, and no pixel
        // centre lies on an edge. Vertex 0 is not drawn.
        let side = SIDE as f32;
        let positions = floats(&[
            side,
            side,
            0.0,
            -4.0,
            2.0 * side,
            -4.0,
            0.0,
            2.0 * side - 4.0,
        ]);
        let frame = floats(&[side, side, 4.0, 0.0]);
        let marks: Vec<u8> = [0xA0u32, 0xA1, 0xA2, 0xA3]
            .iter()
            .flat_map(|m| m.to_le_bytes())
            .collect();
        let mut buffers = Buffers::new(device);
        let xy = buffers.of(device, &positions);
        let mark = buffers.of(device, &marks);
        let color = target(device, PixelFormat::RGBA32Uint, TextureUsage::RENDER_TARGET);
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        draw(encoder.set_render_pipeline_state(state));
        draw(encoder.set_vertex_buffer(Some(&xy), 0, 0));
        draw(encoder.set_vertex_buffer(Some(&mark), 0, 1));
        draw(encoder.set_vertex_bytes(&frame, 2));
        let drawn = encoder.draw_primitives(PrimitiveType::Triangle, 1, 3, 2, base_instance);
        encoder.end_encoding();
        run(name, &command_buffer);
        drawn?;
        Ok((read(&color, 16).chunks(16))
            .map(|texel| {
                [0, 1, 2, 3].map(|i| u32::from_le_bytes([0, 1, 2, 3].map(|b| texel[4 * i + b])))
            })
            .collect())
    }

    /// The instance whose triangle covers texel `at` last when instances 1 and 2 are drawn
    /// ([`two_instances`] at base instance 1): 1 in the target's upper half, 2 in its lower.
    fn upper_or_lower(at: usize) -> u32 {
        match at / SIDE as usize {
            0..4 => 1,
            _ => 2,
        }
    }

    /// Metal's constant step reads the layout's first element for every vertex and instance: a
    /// vertex-rate binding of stride 0, with no divisor, on every device. `ima_vertex`'s `uint`
    /// mark comes from a layout at the constant step over four distinct marks; two instances of a
    /// triangle are drawn from vertex 1 at base instance 1, each instance moved down 4 rows, the
    /// second covering the lower half of the target over the first. Every texel holds the first
    /// mark with the provoking vertex's ID, 1, and its instance: 1 in the upper half, 2 in the
    /// lower. A vertex-rate binding at the layout's stride would read the second mark there, and
    /// any instance step the second and third.
    #[test]
    fn every_device_fetches_a_constant_step_attribute_for_every_vertex_and_instance() {
        let Some(library) = library(&engine_dir().join("icb_merged_attributes.ll")) else {
            return;
        };
        let d = marked(&library, VertexStepFunction::Constant, 0);
        each_device("draw_constant_step", |name, _, device| {
            let state = state(name, &device, &d);
            let texels = two_instances(name, &device, &state, 1)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            for (at, words) in texels.iter().enumerate() {
                assert_eq!(
                    *words,
                    [0xA0, 1, upper_or_lower(at), 1],
                    "{name}: texel {at}"
                );
            }
            println!("{name}: the first mark at every vertex of instances 1 and 2");
        });
    }

    /// Metal's per-instance step at rate 2 fetches one element for every two instances, counted
    /// from the draw's base instance (the conformance case `vertex-step-rate-instances`): a
    /// binding's divisor where the device has the step rates (Floor, Step rates).
    ///
    /// - From base instance 0, instances 0 and 1 both read the first mark; instance 1 covers the
    ///   target, so every texel holds it. A binding without the divisor would read the second.
    /// - From base instance 1, where the device reports `supportsNonZeroFirstInstance`, instances
    ///   1 and 2 both read the second mark: without the divisor the lower half would read the
    ///   third, and a division of the instance index from 0 would give the upper half the first.
    ///   Where the device does not, that draw is refused (`DrawKind::DividedBaseInstance`) and
    ///   records nothing.
    ///
    /// Without the step rates the state is the translator's refusal (`VertexStepRate`).
    #[test]
    fn every_device_with_the_step_rates_fetches_per_instance_at_rate_2_from_the_base_instance() {
        let Some(library) = library(&engine_dir().join("icb_merged_attributes.ll")) else {
            return;
        };
        let d = marked(&library, VertexStepFunction::PerInstance, 2);
        each_device("draw_step_rate", |name, _, device| {
            let core = device.core();
            let profile = core.translator.profile();
            let divides = profile.fragment().vertex_attribute_divisor
                && profile.limits().max_vertex_attrib_divisor >= 2;
            let nonzero = profile.limits().supports_non_zero_first_instance;
            let state = match device
                .new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers)
            {
                Ok(state) => state,
                Err(Error::Translation(metal2vulkan::error::Error::Unsupported(u))) if !divides => {
                    assert_eq!(u.feature, Feature::VertexStepRate, "{name}: {u:?}");
                    println!("{name}: no step rates; a step rate of 2 refused");
                    return;
                }
                Err(other) => panic!("{name}: {other}"),
            };
            assert!(divides, "{name}: a step rate of 2 made a state");
            let from_zero = two_instances(name, &device, &state, 0)
                .unwrap_or_else(|error| panic!("{name}: from base instance 0: {error}"));
            for (at, words) in from_zero.iter().enumerate() {
                assert_eq!(*words, [0xA0, 1, 1, 1], "{name}: from 0: texel {at}");
            }
            match two_instances(name, &device, &state, 1) {
                Ok(texels) => {
                    assert!(nonzero, "{name}: a base instance of 1 drew");
                    for (at, words) in texels.iter().enumerate() {
                        assert_eq!(
                            *words,
                            [0xA1, 1, upper_or_lower(at), 1],
                            "{name}: from 1: texel {at}"
                        );
                    }
                    println!("{name}: at rate 2, the first mark from base instance 0 and the second from 1");
                }
                Err(Error::Unsupported(Gap::Draw {
                    kind: DrawKind::DividedBaseInstance,
                })) if !nonzero => {
                    println!("{name}: at rate 2, the first mark from base instance 0; base instance 1 refused");
                }
                Err(other) => panic!("{name}: from base instance 1: {other}"),
            }
        });
    }

    /// A draw whose vertex function takes a bound buffer's address to be a multiple of its
    /// argument's alignment (`colored_vertex`'s `float2` positions, 8, and `float4` colours, 16)
    /// that the address, the buffer's `gpuAddress` plus the offset, is not is refused before
    /// anything is recorded, as Metal's API validation refuses the bind offset
    /// (`Refusal::BufferOffsetAlignment`); the same buffers bound 16 bytes in then draw, and the
    /// pass is its two barriers and the segment's end.
    #[test]
    fn every_device_refuses_a_draw_whose_buffer_is_off_its_alignment() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("draw_misaligned_address", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            // The top half's triangle in green, 16 bytes into each buffer.
            let positions = floats(&[0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 3.0, 0.0, -1.0, 4.0]);
            let colors: Vec<f32> = [[0.0; 4], GREEN, GREEN, GREEN].concat();
            let (positions, colors) = (
                buffers.of(&device, &positions),
                buffers.of(&device, &floats(&colors)),
            );
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
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
            draw(encoder.set_render_pipeline_state(&state));
            for (positions_at, colors_at, index, alignment) in [(20, 16, 0, 8), (16, 24, 1, 16)] {
                draw(encoder.set_vertex_buffer(Some(&positions), positions_at, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), colors_at, 1));
                let at = [positions_at, colors_at][index as usize];
                match encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0) {
                    Err(Error::Refused(Refusal::BufferOffsetAlignment {
                        index: refused,
                        offset,
                        alignment: align,
                    })) => assert_eq!(
                        (refused, offset, align),
                        (index, at, alignment),
                        "{name}: buffer {index} bound at {at}"
                    ),
                    other => panic!("{name}: buffer {index} bound at {at}: {other:?}"),
                }
            }
            draw(encoder.set_vertex_buffer(Some(&colors), 16, 1));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.passes - before.passes,
                ),
                (1, 1, 1),
                "{name}: submissions, barriers, passes"
            );
            let want = texels(|_, y| match y {
                0..4 => GREEN,
                _ => BLACK,
            });
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!("{name}: binds off 8 and 16 refused before recording; aligned binds drew");
        });
    }

    /// Vulkan binds indices only at a multiple of their size in the memory: in a buffer the
    /// consumer placed 2 bytes past a multiple of 256 in its block, 32-bit indices at offset 0,
    /// which Metal takes, are refused before anything is recorded (`Refusal::ArgumentPlace`), and
    /// 16-bit indices there draw the top half's triangle from where they lie; the pass is its two
    /// barriers and the segment's end.
    #[test]
    fn every_device_takes_indices_only_at_a_multiple_of_their_size_in_their_block() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("draw_index_place", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            // 2 bytes past a multiple of 256 in the block.
            let at = buffers.next + 2;
            let indices = buffers.at(&device, &[0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0], 2);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
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
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            let indexed = |index_type| IndexBuffer {
                buffer: &indices,
                offset: 0,
                index_type,
            };
            let triangle = PrimitiveType::Triangle;
            match encoder.draw_indexed_primitives(triangle, 3, indexed(IndexType::UInt32), 1, 0, 0)
            {
                Err(Error::Refused(Refusal::ArgumentPlace { place, alignment })) => {
                    assert_eq!((place, alignment), (at, 4), "{name}: 32-bit indices");
                }
                other => panic!("{name}: 32-bit indices at byte {at}: {other:?}"),
            }
            draw(encoder.draw_indexed_primitives(triangle, 3, indexed(IndexType::UInt16), 1, 0, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.passes - before.passes,
                ),
                (1, 1, 1),
                "{name}: submissions, barriers, passes"
            );
            let want = texels(|_, y| match y {
                0..4 => GREEN,
                _ => BLACK,
            });
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!("{name}: 32-bit indices at byte 2 refused before recording; 16-bit ones drew");
        });
    }

    /// A shader-read render target of `format`, `SIDE` a side and `levels` levels, at `index`.
    fn readable(device: &Device, format: PixelFormat, levels: u32, index: u32) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.storage_mode = StorageMode::Shared;
        d.mipmap_level_count = levels;
        (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{format:?}: {error}"))
    }

    /// A draw reads, through a texture, the attachment its own pass renders, as Metal reads a
    /// render target: after a memory barrier, what an earlier draw of the pass rendered, at the
    /// fragment's own pixel and at another pixel, the rendering ended and begun again around one
    /// barrier. The pass's first such draw moves the image to the feedback layout, a barrier of
    /// its own. A device without the attachment reads refuses the draw,
    /// `Gap::Draw { AttachmentRead }`, and records nothing for it.
    #[test]
    fn every_device_reads_its_attachment_through_a_texture_after_a_memory_barrier() {
        let Some(library) = examples() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let own = descriptor(&library, "own_pixel_fragment", PixelFormat::RGBA8Unorm);
        let first = descriptor(&library, "first_texel_fragment", PixelFormat::RGBA8Unorm);
        each_device("attachment_reads", |name, _, device| {
            (device.grant_indices(IndexClass::Texture, 4096, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let reads = device.core().feedback.is_some();
            let plain = state(name, &device, &plain);
            let mut buffers = Buffers::new(&device);
            // Three triangles over the whole target, their colors none, blue and green.
            let whole = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
            let positions = buffers.of(&device, &floats(&[whole; 3].concat()));
            let colors: Vec<f32> = [[0.0; 4], BLUE, [0.0, 1.0, 0.0, 0.0]]
                .iter()
                .flat_map(|c| [*c; 3].concat())
                .collect();
            let colors = buffers.of(&device, &floats(&colors));
            let queue = device.new_command_queue(4).expect("a queue");
            for (case, (reading, index)) in
                [("own pixel", (&own, 4096)), ("first texel", (&first, 4097))]
            {
                let reading = state(name, &device, reading);
                let color = readable(&device, PixelFormat::RGBA8Unorm, 1, index);
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
                let draw = |result: Result<(), Error>| {
                    result.unwrap_or_else(|e| panic!("{name} {case}: {e}"))
                };
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                draw(encoder.set_fragment_texture(Some(&color), 0));
                draw(encoder.set_render_pipeline_state(&reading));
                // The pass's first read of its attachment: the clear's black, and none added.
                let moved = encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0);
                if !reads {
                    assert_eq!(
                        format!("{moved:?}"),
                        format!(
                            "{:?}",
                            Err::<(), _>(Error::Unsupported(Gap::Draw {
                                kind: DrawKind::AttachmentRead
                            }))
                        ),
                        "{name} {case}: a device without the attachment reads"
                    );
                    encoder.end_encoding();
                    run(name, &command_buffer);
                    continue;
                }
                draw(moved);
                // Blue over every pixel, then the barrier, then each pixel read plus green.
                draw(encoder.set_render_pipeline_state(&plain));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
                draw(encoder.memory_barrier());
                draw(encoder.set_render_pipeline_state(&reading));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 6, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let after = counters.metrics();
                assert_eq!(
                    (after.barriers - before.barriers, after.passes - before.passes),
                    (5, 1),
                    "{name} {case}: barriers (the pass's two, the move, the memory barrier's, the segment's end), passes"
                );
                let want = texels(|_, _| [0.0, 1.0, 1.0, 1.0]);
                assert_eq!(
                    read(&color, 4),
                    want,
                    "{name} {case}: every pixel blue read plus green"
                );
            }
            match reads {
                true => println!("{name}: an attachment read after a memory barrier, at its own pixel and at another"),
                false => println!("{name}: no attachment reads; the reading draw refused"),
            }
        });
    }

    /// A pass that renders a texture's second level reads its first through a view of both
    /// levels, which reaches the rendered one: the draw reads through the feedback loop, its
    /// whole image moved, and samples the level an earlier pass rendered.
    #[test]
    fn every_device_samples_a_lower_level_while_its_pass_renders_the_next() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "first_level_fragment", PixelFormat::RGBA8Unorm);
        each_device("attachment_levels", |name, _, device| {
            if device.core().feedback.is_none() {
                println!("{name}: no attachment reads");
                return;
            }
            for (class, first) in [(IndexClass::Texture, 4096), (IndexClass::Sampler, 1)] {
                (device.grant_indices(class, first, 4))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            let state = (device.new_render_pipeline_state(&d, |_| Ok(vec![CONSTEXPR])))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let chain = readable(&device, PixelFormat::RGBA8Unorm, 2, 4096);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            // The first level cleared to blue by a pass of no draw.
            let mut first = pass(&chain, None);
            first.color_attachments[0].clear_color = crate::ClearColor::new(0.0, 0.0, 1.0, 1.0);
            (command_buffer.render_command_encoder(&first))
                .unwrap_or_else(|error| panic!("{name}: {error}"))
                .end_encoding();
            let mut second = pass(&chain, None);
            second.color_attachments[0].level = 1;
            let mut encoder = (command_buffer.render_command_encoder(&second))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_fragment_texture(Some(&chain), 0));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let side = SIDE / 2;
            let mut bytes = vec![0; (side * side * 4) as usize];
            let region = Region {
                origin: [0; 3],
                size: [side, side, 1],
            };
            (chain.get_bytes(
                &mut bytes,
                u64::from(side * 4),
                u64::from(side * side * 4),
                region,
                1,
                0,
            ))
            .expect("get_bytes");
            let want: Vec<u8> = (0..side * side).flat_map(|_| rgba8(BLUE)).collect();
            assert_eq!(
                bytes, want,
                "{name}: the second level, each texel the first's blue"
            );
            println!("{name}: a lower level sampled while its pass renders the next");
        });
    }

    /// What Metal refuses is refused, and what a draw does not build is a gap.
    #[test]
    fn every_device_refuses_the_draws_it_does_not_build() {
        let Some(library) = examples() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let other = descriptor(&library, "colored_fragment", PixelFormat::BGRA8Unorm);
        each_device("draw_refused", |name, _, device| {
            let state = state(name, &device, &plain);
            let other = self::state(name, &device, &other);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let indices = buffers.of(&device, &[3, 0, 4, 0, 5, 0, 0, 0]);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut depth = DepthStencilDescriptor::default();
            depth.depth_compare_function = CompareFunction::Less;
            let depth = device.new_depth_stencil_state(&depth);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let triangle = PrimitiveType::Triangle;
            let gap = |kind| Err::<(), _>(Error::Unsupported(Gap::Draw { kind }));
            let same = |got: Result<(), Error>, want: Result<(), Error>, case: &str| {
                assert_eq!(format!("{got:?}"), format!("{want:?}"), "{name}: {case}");
            };
            same(
                encoder.draw_primitives(triangle, 0, 3, 1, 0),
                Err(Error::Refused(Refusal::NoPipeline)),
                "no pipeline state",
            );
            encoder.set_render_pipeline_state(&other).expect("a state");
            same(
                encoder.draw_primitives(triangle, 0, 3, 1, 0),
                gap(DrawKind::AttachmentFormats),
                "another format",
            );
            encoder.set_render_pipeline_state(&state).expect("a state");
            same(
                encoder.draw_primitives(PrimitiveType::Point, 0, 3, 1, 0),
                gap(DrawKind::Points),
                "points",
            );
            encoder
                .set_depth_stencil_state(Some(&depth))
                .expect("a state");
            same(
                encoder.draw_primitives(triangle, 0, 3, 1, 0),
                gap(DrawKind::DepthTest),
                "a depth test",
            );
            encoder.set_depth_stencil_state(None).expect("no state");
            let viewport = Viewport {
                origin_x: 0.0,
                origin_y: 0.0,
                width: 4.0,
                height: 4.0,
                znear: 0.0,
                zfar: 1.0,
            };
            same(
                encoder.set_viewport(Viewport {
                    height: f64::INFINITY,
                    ..viewport
                }),
                Err(Error::Refused(Refusal::ViewportNotFinite)),
                "an infinite height",
            );
            same(
                encoder.set_viewport(Viewport {
                    width: -4.0,
                    ..viewport
                }),
                gap(DrawKind::Viewport),
                "a mirrored viewport",
            );
            encoder
                .set_vertex_buffer(Some(&positions), 0, 0)
                .expect("a buffer");
            encoder
                .set_vertex_buffer(Some(&colors), 0, 1)
                .expect("a buffer");
            let misaligned = IndexBuffer {
                buffer: &indices,
                offset: 1,
                index_type: IndexType::UInt16,
            };
            same(
                encoder.draw_indexed_primitives(triangle, 3, misaligned, 1, 0, 0),
                Err(Error::Refused(Refusal::IndexBufferOffset {
                    offset: 1,
                    length: 8,
                })),
                "a misaligned index offset",
            );
            encoder.end_encoding();
            run(name, &command_buffer);
            println!("{name}: each draw refused by kind");
        });
    }
}
