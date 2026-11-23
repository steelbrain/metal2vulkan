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
//!   - 2,000 draws in one encoder move none of the device's counters while it is open; its end
//!     adds the pass, the push and the 1,999 pushes held.
//!   - on a GPU, the lock budget (`crate::tests::locks`): 2,000 draws more in a recording, and
//!     2,000 more each with a vertex buffer and a fragment texture set before it, add no locked
//!     instruction.
//!   - a draw of lines makes its content's line `VkPipeline` once, a first-use wait, and draws
//!     its row; a state whose descriptor names the line class makes it at its
//!     finish_pipeline, and its first draw of lines makes nothing and waits for nothing.
//!   - lines draw at width 1 alone: at width 2 a draw of lines or a line strip is
//!     `DrawKind::LineWidth` and a triangle draws; at 1 again, and in a later encoder, lines draw.
//!   - points (the owned fixture `engine/rt6_render_state.ll`'s `rt6_point`, of
//!     `[[point_size]]` 4) cover the pixels whose centres their square holds, the content's
//!     point `VkPipeline` made at its first draw of points, a first-use wait; a state whose
//!     descriptor names the point class makes that pipeline at its finish_pipeline, and its
//!     points of a function without `[[point_size]]` are 1 pixel each; under another class
//!     such points rasterize nothing, and a triangle drawn after them draws through the
//!     encoder's scissor.
//!   - a stencil state that replaces the stencil marks what a draw covers, and one that tests it
//!     keeps a later draw to those pixels.
//!   - a pass clears, loads and stores a shader-read `Depth32Float` depth attachment that its
//!     draws, under Metal's default depth-stencil state, neither test nor write: one submission,
//!     three barriers and one pass each; a state without the pass's depth format, or with one in
//!     a pass without a depth attachment, is the formats' gap, and a depth test is its own.
//!   - a fragment function samples a texture through its constexpr sampler at the index the
//!     consumer gave it: each pixel reads its own texel.
//!   - a fragment function's second output for colour attachment 0 (the owned fixture
//!     `engine/stage_io.ll`'s `stage_io_fragment_dual_source`) is the blend's second source.
//!   - a vertex function stores to device memory (the owned fixture `engine/rvc_draw_bases.ll`'s
//!     `draw_bases_in_a_vertex_function`), its state's rasterization disabled: each vertex of each
//!     instance writes its slot.
//!   - four samples (the owned fixture `engine/psc_sample_coverage.ll`'s `psc_read_samples`): a
//!     triangle drawn by a state of `rasterSampleCount` 4 into a 4-sample target covers, per
//!     sample as a kernel reads them, every pixel on one side of its edge, none on the other and,
//!     on it, Metal's 4x positions below it, which are Vulkan's standard locations the floor
//!     requires;
//!     a 1-sample state in the pass is the formats' gap, and a pass of a 4-sample and a 1-sample
//!     attachment `RenderPassKind::SampleCounts`.
//!   - a fragment function that returns `get_num_samples()` (the owned fixture
//!     `engine/spc_num_samples.ll`) writes 1 to every pixel of a 1-sample `R32Uint` target through
//!     a state of `rasterSampleCount` 1, and 4 to every sample of a 4-sample one through a state
//!     of 4, where the device renders 4 samples of `R32Uint`.
//!   - a vertex function that reads `[[base_instance]]` (the owned fixture
//!     `engine/icb_merged_draws.ll`'s `icb_merged_vertex`) draws, and every pixel it covers reads
//!     the draw's base instance and its instance ID from it.
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
//!     a pipeline state of other formats than the pass's, a depth test, a stencil test without a
//!     stencil attachment, a misaligned index offset, a non-finite viewport and a mirrored one.
//!   - a pass with a visibility result buffer draws, its command buffer keeps the buffer until it
//!     retires, and its bytes stay as they were; `Disabled` at any offset records nothing, and the
//!     pass takes the submission, barrier and pass one without the buffer takes (`Boolean` and
//!     `Counting` are `tests::visibility`'s).
//!   - an indexed draw's index buffer and an indexed indirect draw's, each the one holder of its
//!     Shared block, dropped once encoded and before the commit, are kept with their blocks until
//!     the command buffer retires: the draws read their own indices, not those of blocks taken
//!     after the drops, and neither block is back in the pool before the retirement.
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
pub(crate) mod on_device {
    use crate::tests::blit::on_device::settled;
    use crate::tests::command::on_device::{each_device, finished, within};
    use crate::texture::TextureObject;
    use crate::{
        Block, Buffer, ColorAttachmentDescriptor, CommandBuffer, CompareFunction, CullMode,
        DepthStencilDescriptor, Device, DrawKind, Error, FunctionDescriptor, Gap, IndexBuffer,
        IndexClass, IndexType, Library, MemoryClass, NumberedSampler, PixelFormat, PrimitiveType,
        Refusal, RefusalBasis, Region, RenderPassDescriptor, RenderPipelineDescriptor,
        RenderPipelineState, ScissorRect, StageSource, StencilOperation, StorageMode, Texture,
        TextureDescriptor, TextureType, TextureUsage, Viewport, VisibilityResultMode,
    };
    use metal2vulkan::error::Feature;
    use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
    use metal2vulkan::metal::{DataType, VertexFormat, VertexStepFunction};
    use metal2vulkan::pipeline::{
        VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
    };
    use metal2vulkan_devtools::{
        engine_dir, fixtures_dir, llvm_assemble, llvm_assemble_text, AssembleError,
    };
    use std::path::Path;

    /// The side of the targets.
    pub(crate) const SIDE: u32 = 8;
    /// The constexpr sampler's index the test gives `sampled_fragment`.
    pub(crate) const CONSTEXPR: u32 = 3;

    /// The library of `ll`, or None where `llvm-as` is missing.
    pub(crate) fn library(ll: &Path) -> Option<Library> {
        match llvm_assemble(ll) {
            Ok(bitcode) => Some(Library::from_air(&bitcode).expect("the library")),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(error) => panic!("{}: {error:?}", ll.display()),
        }
    }

    pub(crate) fn examples() -> Option<Library> {
        library(&fixtures_dir().join("examples/runtime_examples.ll"))
    }

    /// `colored_vertex` with `fragment`, drawing into one colour attachment of `color`.
    pub(crate) fn descriptor(
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

    /// The state of `d` on `device`, its main pipelines then finished, as a consumer finishes
    /// them once it has answered.
    pub(crate) fn state(
        name: &str,
        device: &Device,
        d: &RenderPipelineDescriptor,
    ) -> RenderPipelineState {
        let state = unfinished(name, device, d);
        (state.finish_pipeline()).unwrap_or_else(|error| panic!("{name}: finished: {error}"));
        state
    }

    /// The state of `d` on `device`, its main pipelines not made yet.
    pub(crate) fn unfinished(
        name: &str,
        device: &Device,
        d: &RenderPipelineDescriptor,
    ) -> RenderPipelineState {
        (device.new_render_pipeline_state(d, crate::tests::no_constexpr_samplers))
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// Shared buffers in one block, 256 bytes apart.
    pub(crate) struct Buffers {
        block: Block,
        next: u64,
    }

    impl Buffers {
        pub(crate) fn new(device: &Device) -> Buffers {
            Buffers::sized(device, 65536)
        }

        /// Buffers from a block of `bytes`.
        pub(crate) fn sized(device: &Device, bytes: u64) -> Buffers {
            Buffers {
                block: (device.take_block(MemoryClass::Shared, bytes)).expect("a block"),
                next: 0,
            }
        }

        /// A buffer of `bytes`.
        pub(crate) fn of(&mut self, device: &Device, bytes: &[u8]) -> Buffer {
            self.at(device, bytes, 0)
        }

        /// A buffer of `bytes`, `skew` bytes past the place [`of`](Self::of) would take.
        pub(crate) fn at(&mut self, device: &Device, bytes: &[u8], skew: u64) -> Buffer {
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

    pub(crate) fn floats(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A 2D render target of `format`, `SIDE` a side, with no index of its own
    /// ([`Device::new_texture_for_views`]).
    pub(crate) fn target(device: &Device, format: PixelFormat, usage: TextureUsage) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
        d.usage = usage;
        d.storage_mode = StorageMode::Shared;
        (device.new_texture_for_views(&d)).unwrap_or_else(|error| panic!("{format:?}: {error}"))
    }

    /// The texels of `texture`, `unit` bytes each.
    pub(crate) fn read(texture: &Texture, unit: usize) -> Vec<u8> {
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
    pub(crate) fn pass(color: &Texture, stencil: Option<&Texture>) -> RenderPassDescriptor {
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
    pub(crate) fn run(name: &str, command_buffer: &CommandBuffer) {
        command_buffer.commit().expect("a commit");
        finished(name, command_buffer);
        assert_eq!(command_buffer.error(), None, "{name}");
    }

    /// The RGBA8 texel `color` of a float colour of 0 or 1 channels.
    fn rgba8(color: [f32; 4]) -> [u8; 4] {
        color.map(|c| (c * 255.0) as u8)
    }

    pub(crate) const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
    pub(crate) const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    pub(crate) const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    const WHITE: [f32; 4] = [1.0; 4];

    /// The fixture's vertices, positions and colors: the top half of clip space (y up) in green,
    /// then the whole of it in blue and in white, three each.
    pub(crate) fn vertices(device: &Device, buffers: &mut Buffers) -> (Buffer, Buffer) {
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
    pub(crate) fn texels(texel: impl Fn(u32, u32) -> [f32; 4]) -> Vec<u8> {
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

    /// What an encoder's commands count reaches the device's counters once, at the encoder's
    /// end: 2,000 draws move none of them while the encoder is open, and its end adds the pass,
    /// the push and the 1,999 pushes held.
    #[test]
    fn every_device_counts_an_encoders_draws_at_its_end() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("draw_counts", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
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
            let counted =
                |m: &crate::Metrics| (m.passes, m.barriers, m.pushes, m.pushes_held, m.ring_writes);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            for _ in 0..2_000 {
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            }
            let open = counters.metrics();
            assert_eq!(
                counted(&open),
                counted(&before),
                "{name}: passes, barriers, pushes, pushes held and ring writes while it is open"
            );
            encoder.end_encoding();
            let ended = counters.metrics();
            assert_eq!(
                (
                    ended.passes - before.passes,
                    ended.pushes - before.pushes,
                    ended.pushes_held - before.pushes_held,
                ),
                (1, 1, 1_999),
                "{name}: passes, pushes and pushes held at its end"
            );
            run(name, &command_buffer);
            let want = texels(|_, y| match y {
                0..4 => GREEN,
                _ => BLACK,
            });
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!("{name}: 2,000 draws counted once, at the encoder's end");
        });
    }

    /// The lock budget of draws (`crate::tests::locks`): a recording of 4,000 draws executes no
    /// more locked instructions than one of 2,000.
    #[test]
    #[cfg(m2v_gpu)]
    fn every_gpu_adds_no_locked_instruction_per_draw() {
        use crate::tests::locks::Locks;
        let Some(library) = examples() else {
            return;
        };
        let Some(locks) = Locks::open("draw_locks", "draw") else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        locks.each_device("draw_locks", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            locks.per_added(name, ("draw", "draw"), 2_000, |count| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                draw(encoder.set_render_pipeline_state(&state));
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                for _ in 0..count {
                    draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                }
                encoder.end_encoding();
                run(name, &command_buffer);
            });
        });
    }

    /// The lock budget of binds (`crate::tests::locks`): a recording of 4,000 draws, each with a
    /// vertex buffer and a fragment texture set before it, both alternating between two objects
    /// so that every set changes its slot, executes no more locked instructions than one of
    /// 2,000.
    #[test]
    #[cfg(m2v_gpu)]
    fn every_gpu_adds_no_locked_instruction_per_bind() {
        use crate::tests::locks::Locks;
        let Some(library) = examples() else {
            return;
        };
        let Some(locks) = Locks::open("bind_locks", "bound_draw") else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        locks.each_device("bind_locks", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let (other, _) = vertices(&device, &mut buffers);
            let positions = [positions, other];
            (device.grant_indices(IndexClass::Texture, 4096, 2))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let textures = [4096, 4097].map(|index| {
                let mut d =
                    TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, 4, 4);
                d.usage = TextureUsage::SHADER_READ;
                (device.new_texture(&d, Some(index)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            });
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            locks.per_added(name, ("bound draw", "bound_draw"), 2_000, |count| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                draw(encoder.set_render_pipeline_state(&state));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                for at in 0..count as usize {
                    draw(encoder.set_vertex_buffer(Some(&positions[at % 2]), 0, 0));
                    draw(encoder.set_fragment_texture(Some(&textures[at % 2]), 0));
                    draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                }
                encoder.end_encoding();
                run(name, &command_buffer);
            });
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

    /// A plain state draws its lines by Bresenham's rule, and one that enables alpha to coverage or
    /// alpha to one by the device's default rule, which Vulkan's Bresenham mode leaves for such a
    /// state. Neither state carries `PrecisionNote::DeviceLineRasterization`: the draw of lines
    /// does, on every device (`RenderCommandEncoder::LINE_NOTE`). Each draws its line across row 2.
    #[test]
    fn every_device_draws_lines_under_alpha_to_coverage_or_one() {
        let Some(library) = examples() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let mut coverage = plain.clone();
        coverage.alpha_to_coverage = true;
        let mut one = plain.clone();
        one.alpha_to_one = true;
        let note = crate::PrecisionNote::DeviceLineRasterization;
        assert_eq!(crate::RenderCommandEncoder::LINE_NOTE, note);
        each_device("draw_line_rule", |name, _, device| {
            let y = 1.0 - 2.5 / 4.0;
            let mut buffers = Buffers::new(&device);
            let positions = buffers.of(&device, &floats(&[-1.0, y, 1.0, y]));
            let colors = buffers.of(&device, &floats(&[GREEN, GREEN].concat()));
            let queue = device.new_command_queue(4).expect("a queue");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let alpha_to_one = (device.core().translator.profile().enabled().features)
                .contains(&metal2vulkan::device::VkFeature::AlphaToOne);
            for (what, d) in [
                ("plain", &plain),
                ("alpha to coverage", &coverage),
                ("alpha to one", &one),
            ] {
                if d.alpha_to_one && !alpha_to_one {
                    // A device without `alphaToOne`: the translator's typed refusal.
                    let refusal = metal2vulkan::error::Error::Unsupported(
                        metal2vulkan::error::Unsupported::needs(
                            metal2vulkan::error::Feature::AlphaToOne,
                            metal2vulkan::error::Capability::Feature(
                                metal2vulkan::device::VkFeature::AlphaToOne,
                            ),
                        ),
                    );
                    match device.new_render_pipeline_state(d, crate::tests::no_constexpr_samplers) {
                        Err(Error::Translation { error, .. }) => {
                            assert_eq!(error, refusal, "{name}: {what}")
                        }
                        made => panic!("{name}: {what}: {:?}", made.err()),
                    }
                    continue;
                }
                let state = state(name, &device, d);
                assert!(
                    !state.precision_notes().contains(&note),
                    "{name}: {what}: the line note on the state"
                );
                let color = target(
                    &device,
                    PixelFormat::RGBA8Unorm,
                    TextureUsage::RENDER_TARGET,
                );
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                draw(encoder.set_render_pipeline_state(&state));
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                draw(encoder.draw_primitives(PrimitiveType::Line, 0, 2, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let texels = read(&color, 4);
                for x in 1..SIDE - 1 {
                    let i = ((2 * SIDE + x) * 4) as usize;
                    assert_eq!(texels[i..i + 4], rgba8(GREEN), "{name}: {what}: ({x}, 2)");
                }
            }
            println!("{name}: lines drawn plain, under alpha to coverage and under alpha to one");
        });
    }

    /// A state whose descriptor names the line class (`inputPrimitiveTopology`) makes its line
    /// pipeline as a main one, its `finish_pipeline` making it beside the triangle one: its
    /// first draw of lines makes nothing and waits for nothing, and draws its row.
    #[test]
    fn every_device_makes_a_line_class_state_s_line_pipeline_at_its_make() {
        use metal2vulkan::metal::PrimitiveTopologyClass;

        let Some(library) = examples() else {
            return;
        };
        let mut d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.input_primitive_topology = PrimitiveTopologyClass::Line;
        each_device("draw_line_class", |name, _, device| {
            let counters = device.counters();
            let made = counters.metrics();
            let state = state(name, &device, &d);
            let created = counters.metrics();
            assert_eq!(
                created.pipelines_created - made.pipelines_created,
                2,
                "{name}: the triangle and the line pipelines, made by finish_pipeline"
            );
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
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.draw_primitives(PrimitiveType::Line, 0, 2, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.pipelines_created - created.pipelines_created,
                    after.first_use_waits - created.first_use_waits,
                ),
                (0, 0),
                "{name}: the first draw of lines: pipelines made, first-use waits"
            );
            let texels = read(&color, 4);
            for x in 1..SIDE - 1 {
                let i = ((2 * SIDE + x) * 4) as usize;
                assert_eq!(texels[i..i + 4], rgba8(GREEN), "{name}: ({x}, 2)");
            }
            println!("{name}: the line class's pipeline made by finish_pipeline; its lines drew");
        });
    }

    /// A render state answers before its main pipeline is made: its creation makes no
    /// `VkPipeline` and answers what Metal's state answers, every answer the same after the finish
    /// as before it. Its `finish_pipeline` makes the
    /// triangle class's after the answer, counted as finished, a second call makes nothing, and a
    /// draw then waits for nothing. A state of another content that is never finished draws the
    /// same texels: its first draw makes the pipeline, a first-use wait on an unfinished
    /// pipeline, and a `finish_pipeline` after it makes nothing.
    #[test]
    fn every_device_answers_a_render_state_before_its_main_pipeline_is_made() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        // Another content drawing the same: blending at its default factors keeps the source.
        let mut blended = d.clone();
        if let Some(color) = &mut blended.color_attachments[0] {
            color.blending_enabled = true;
        }
        let want = texels(|_, y| if y < 4 { GREEN } else { BLACK });
        each_device("render_answered_early", |name, _, device| {
            let counters = device.counters();
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let queue = device.new_command_queue(4).expect("a queue");
            // The texels of the fixture's top half drawn green by `state` into a new target.
            let drawn = |state: &RenderPipelineState| {
                let color = target(
                    &device,
                    PixelFormat::RGBA8Unorm,
                    TextureUsage::RENDER_TARGET,
                );
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let draw =
                    |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
                draw(encoder.set_render_pipeline_state(state));
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                read(&color, 4)
            };
            // The pipelines made, finished, failed to finish and waited for unfinished.
            let moved = |before: &crate::Metrics, after: &crate::Metrics| {
                (
                    after.pipelines_created - before.pipelines_created,
                    after.finished_pipelines - before.finished_pipelines,
                    after.failed_finishes - before.failed_finishes,
                    after.first_use_unfinished_waits - before.first_use_unfinished_waits,
                )
            };
            // Every answer of the state.
            let answers = |state: &RenderPipelineState| {
                (
                    state.precision_notes(),
                    state.texture_accesses(crate::Stage::Vertex),
                    state.texture_accesses(crate::Stage::Fragment),
                    state.imageblock_sample_length().ok(),
                    state
                        .imageblock_memory_length_for_dimensions([8, 4, 3])
                        .ok(),
                    state.support_indirect_command_buffers(),
                )
            };

            let before = counters.metrics();
            let state = unfinished(name, &device, &d);
            let answered = counters.metrics();
            assert_eq!(
                moved(&before, &answered),
                (0, 0, 0, 0),
                "{name}: no pipeline at the answer"
            );
            let early = answers(&state);
            assert_eq!(
                state.imageblock_sample_length().ok(),
                Some(4),
                "{name}: Metal's answer"
            );
            state.finish_pipeline().expect("finished");
            assert_eq!(
                answers(&state),
                early,
                "{name}: every answer is the same before the finish and after it"
            );
            let finished_at = counters.metrics();
            assert_eq!(
                moved(&answered, &finished_at),
                (1, 1, 0, 0),
                "{name}: the triangle class's pipeline, made after the answer"
            );
            assert!(
                finished_at.finished_pipeline_nanos > answered.finished_pipeline_nanos,
                "{name}: the finish's time"
            );
            state.finish_pipeline().expect("finished again");
            assert_eq!(drawn(&state), want, "{name}: the finished state's texels");
            let after = counters.metrics();
            assert_eq!(
                (moved(&finished_at, &after), after.first_use_waits),
                ((0, 0, 0, 0), finished_at.first_use_waits),
                "{name}: a second finish and a draw make nothing and wait for nothing"
            );

            let before = counters.metrics();
            let never = unfinished(name, &device, &blended);
            assert_eq!(drawn(&never), want, "{name}: the unfinished state's texels");
            let drawn_at = counters.metrics();
            assert_eq!(
                moved(&before, &drawn_at),
                (1, 0, 0, 1),
                "{name}: the draw made the unfinished pipeline and waited for it"
            );
            never.finish_pipeline().expect("finished after the draw");
            assert_eq!(
                moved(&drawn_at, &counters.metrics()),
                (0, 0, 0, 0),
                "{name}: a finish after the draw makes nothing"
            );
            println!(
                "{name}: answered with no pipeline made; finished after the answer in {} ns; an \
                 unfinished state's draw made its pipeline in {} ns",
                finished_at.finished_pipeline_nanos - answered.finished_pipeline_nanos,
                drawn_at.first_use_unfinished_wait_nanos - before.first_use_unfinished_wait_nanos,
            );
        });
    }

    /// Lines draw at width 1 alone (`set_line_width`, Apple's `setLineWidth:`): at width 2 a draw
    /// of lines or of a line strip is `DrawKind::LineWidth` and records nothing, a draw of no
    /// vertices records nothing as at any width, and a triangle draws; at 1 again lines draw, and
    /// a later encoder starts at 1 whatever the earlier one set.
    #[test]
    fn every_device_draws_lines_at_width_1_alone() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("line_width", |name, _, device| {
            let state = state(name, &device, &d);
            // Lines across rows 2 and 0 through their centres, Metal's y up, then a triangle over
            // rows 4 to 7.
            let row = |y: f32| 1.0 - (y + 0.5) / 4.0;
            let (two, zero) = (row(2.0), row(0.0));
            let positions = [
                -1.0, two, 1.0, two, -1.0, zero, 1.0, zero, -1.0, 0.0, 3.0, 0.0, -1.0, -4.0,
            ];
            let mut buffers = Buffers::new(&device);
            let positions = buffers.of(&device, &floats(&positions));
            let colors = [GREEN, GREEN, GREEN, GREEN, WHITE, WHITE, WHITE].concat();
            let colors = buffers.of(&device, &floats(&colors));
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let refused = |got: Result<(), Error>, case: &str| {
                let want = Error::Unsupported(Gap::Draw {
                    kind: DrawKind::LineWidth,
                });
                assert_eq!(
                    format!("{got:?}"),
                    format!("{:?}", Err::<(), _>(want)),
                    "{name}: {case}"
                );
            };
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            encoder.set_line_width(2.0);
            refused(
                encoder.draw_primitives(PrimitiveType::Line, 0, 2, 1, 0),
                "lines at width 2",
            );
            refused(
                encoder.draw_primitives(PrimitiveType::LineStrip, 0, 2, 1, 0),
                "a line strip at width 2",
            );
            draw(encoder.draw_primitives(PrimitiveType::Line, 0, 0, 1, 0));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 4, 3, 1, 0));
            encoder.set_line_width(1.0);
            draw(encoder.draw_primitives(PrimitiveType::Line, 0, 2, 1, 0));
            encoder.set_line_width(0.5);
            encoder.end_encoding();
            let mut load = pass(&color, None);
            load.color_attachments[0].load_action = crate::LoadAction::Load;
            let mut encoder = (command_buffer.render_command_encoder(&load))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.draw_primitives(PrimitiveType::Line, 2, 2, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let texels = read(&color, 4);
            let at = |x: u32, y: u32| {
                let i = ((y * SIDE + x) * 4) as usize;
                [texels[i], texels[i + 1], texels[i + 2], texels[i + 3]]
            };
            for x in 1..SIDE - 1 {
                for (y, want) in [(0, GREEN), (1, BLACK), (2, GREEN), (3, BLACK)] {
                    assert_eq!(at(x, y), rgba8(want), "{name}: ({x}, {y})");
                }
            }
            for y in 4..SIDE {
                for x in 0..SIDE {
                    assert_eq!(at(x, y), rgba8(WHITE), "{name}: ({x}, {y})");
                }
            }
            println!("{name}: lines at width 1, refused at 2 where a triangle draws");
        });
    }

    /// The clip-space position (Metal's y up) of pixel `(x, y)`'s centre in a target `SIDE` a
    /// side.
    fn centre(x: u32, y: u32) -> [f32; 2] {
        let half = SIDE as f32 / 2.0;
        [(x as f32 + 0.5) / half - 1.0, 1.0 - (y as f32 + 0.5) / half]
    }

    /// Points of `[[point_size]]` 4 (`rt6_point`, one at the target's centre) cover the 4 by 4
    /// pixels whose centres their square holds; the first draw of points makes the content's
    /// point pipeline, a first-use wait, and a second makes nothing. A state whose descriptor
    /// names the point class makes its point pipeline at its finish_pipeline, and draws a point
    /// of a function without `[[point_size]]` (`colored_vertex`) 1 pixel wide, its first draw
    /// making and waiting for nothing. Under no class such points rasterize nothing, as
    /// Metal's, and a triangle drawn after them in the encoder draws through its scissor again.
    #[test]
    fn every_device_draws_points_at_their_size_and_a_point_class_s_at_1() {
        use metal2vulkan::metal::PrimitiveTopologyClass;

        let (Some(library), Some(engine)) = (
            examples(),
            library(&engine_dir().join("rt6_render_state.ll")),
        ) else {
            return;
        };
        let function = |library: &Library, name: &str| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut sized = RenderPipelineDescriptor::new(function(&engine, "rt6_point"));
        sized.fragment = Some(function(&engine, "rt6_white"));
        sized.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let mut classed = plain.clone();
        classed.input_primitive_topology = PrimitiveTopologyClass::Point;
        let marks = [(1, 1), (6, 2), (3, 6)];
        each_device("draw_points", |name, _, device| {
            let counters = device.counters();
            let sized = state(name, &device, &sized);
            let plain = state(name, &device, &plain);
            let made = counters.metrics();
            let classed = state(name, &device, &classed);
            let created = counters.metrics();
            assert_eq!(
                created.pipelines_created - made.pipelines_created,
                2,
                "{name}: the point class's state: the triangle and the point pipelines"
            );
            let mut buffers = Buffers::new(&device);
            // The three marks as points, then a triangle over rows 4 to 7.
            let mut positions: Vec<f32> = marks.iter().flat_map(|&(x, y)| centre(x, y)).collect();
            positions.extend([-1.0, 0.0, 3.0, 0.0, -1.0, -4.0]);
            let positions = buffers.of(&device, &floats(&positions));
            let colors = [GREEN, GREEN, GREEN, WHITE, WHITE, WHITE].concat();
            let colors = buffers.of(&device, &floats(&colors));
            let depth = floats(&[0.0, 0.0]);
            let queue = device.new_command_queue(4).expect("a queue");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            // Each case's target, read after its command buffer completes.
            let pixels = |case: &str, encode: &dyn Fn(&mut crate::RenderCommandEncoder)| {
                let color = target(
                    &device,
                    PixelFormat::RGBA8Unorm,
                    TextureUsage::RENDER_TARGET,
                );
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {case}: {error}"));
                encode(&mut encoder);
                encoder.end_encoding();
                run(name, &command_buffer);
                let texels = read(&color, 4);
                move |x: u32, y: u32| {
                    let i = ((y * SIDE + x) * 4) as usize;
                    [texels[i], texels[i + 1], texels[i + 2], texels[i + 3]]
                }
            };
            for round in 0..2 {
                let before = counters.metrics();
                let at = pixels("sized", &|encoder: &mut crate::RenderCommandEncoder| {
                    draw(encoder.set_render_pipeline_state(&sized));
                    draw(encoder.set_vertex_bytes(&depth, 0));
                    draw(encoder.draw_primitives(PrimitiveType::Point, 0, 1, 1, 0));
                });
                let after = counters.metrics();
                let made = u64::from(round == 0);
                assert_eq!(
                    (
                        after.pipelines_created - before.pipelines_created,
                        after.first_use_point_waits - before.first_use_point_waits,
                    ),
                    (made, made),
                    "{name}: round {round}: pipelines made, first-use point waits"
                );
                for y in 0..SIDE {
                    for x in 0..SIDE {
                        let inside = (2..6).contains(&x) && (2..6).contains(&y);
                        let want = if inside { WHITE } else { BLACK };
                        assert_eq!(at(x, y), rgba8(want), "{name}: size 4: ({x}, {y})");
                    }
                }
            }
            let before = counters.metrics();
            let at = pixels(
                "point class",
                &|encoder: &mut crate::RenderCommandEncoder| {
                    draw(encoder.set_render_pipeline_state(&classed));
                    draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                    draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                    draw(encoder.draw_primitives(PrimitiveType::Point, 0, 3, 1, 0));
                },
            );
            let after = counters.metrics();
            assert_eq!(
                (
                    after.pipelines_created - before.pipelines_created,
                    after.first_use_waits - before.first_use_waits,
                ),
                (0, 0),
                "{name}: the point class's first draw of points: pipelines made, first-use waits"
            );
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let want = if marks.contains(&(x, y)) {
                        GREEN
                    } else {
                        BLACK
                    };
                    assert_eq!(at(x, y), rgba8(want), "{name}: size 1: ({x}, {y})");
                }
            }
            let at = pixels("no class", &|encoder: &mut crate::RenderCommandEncoder| {
                draw(encoder.set_render_pipeline_state(&plain));
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                draw(encoder.draw_primitives(PrimitiveType::Point, 0, 3, 1, 0));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            });
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let want = if y >= 4 { WHITE } else { BLACK };
                    assert_eq!(at(x, y), rgba8(want), "{name}: no class: ({x}, {y})");
                }
            }
            println!("{name}: points at size 4, at 1 under the point class, none under no class");
        });
    }

    /// Points whose `[[point_size]]` is not greater than 0, a zero, a negative and a NaN
    /// (`sized_vertex`), draw 1 pixel wide, the size the module writes for each, and their
    /// command buffer completes: Vulkan's `PointSize` takes a size greater than zero alone.
    #[test]
    fn every_device_draws_a_point_of_no_positive_size_1_pixel_wide() {
        let Some(library) = examples() else {
            return;
        };
        let mut d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.vertex = StageSource::Function(FunctionDescriptor::new(
            library.function("sized_vertex").expect("the function"),
        ));
        let marks = [(1, 1), (6, 2), (3, 6)];
        let sizes = [0.0, -1.0, f32::NAN];
        each_device("draw_point_sizes", |name, _, device| {
            let sized = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let positions: Vec<f32> = marks.iter().flat_map(|&(x, y)| centre(x, y)).collect();
            let positions = buffers.of(&device, &floats(&positions));
            let colors = buffers.of(&device, &floats(&GREEN.repeat(3)));
            let sizes = buffers.of(&device, &floats(&sizes));
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&sized));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_vertex_buffer(Some(&sizes), 0, 2));
            draw(encoder.draw_primitives(PrimitiveType::Point, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let texels = read(&color, 4);
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let i = ((y * SIDE + x) * 4) as usize;
                    let want = if marks.contains(&(x, y)) {
                        GREEN
                    } else {
                        BLACK
                    };
                    assert_eq!(
                        texels[i..i + 4],
                        rgba8(want),
                        "{name}: sizes 0, -1 and NaN: ({x}, {y})"
                    );
                }
            }
            println!("{name}: points of size 0, -1 and NaN drew 1 pixel wide");
        });
    }

    /// A state of a vertex function that writes `[[point_size]]` (`rt6_point`) is refused under
    /// the line and the triangle class (`Refusal::PointSizeTopology`), as Metal's creation is, and
    /// made under the unspecified and the point class.
    #[test]
    fn every_device_refuses_a_point_size_function_under_the_line_and_triangle_class() {
        use metal2vulkan::metal::PrimitiveTopologyClass as Class;

        let Some(engine) = library(&engine_dir().join("rt6_render_state.ll")) else {
            return;
        };
        let function = |name: &str| {
            StageSource::Function(FunctionDescriptor::new(
                engine.function(name).expect("the function"),
            ))
        };
        let mut sized = RenderPipelineDescriptor::new(function("rt6_point"));
        sized.fragment = Some(function("rt6_white"));
        sized.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        each_device("draw_point_size_class", |name, _, device| {
            for class in [
                Class::Unspecified,
                Class::Point,
                Class::Line,
                Class::Triangle,
            ] {
                let mut d = sized.clone();
                d.input_primitive_topology = class;
                let made =
                    device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers);
                match (class, made) {
                    (Class::Line | Class::Triangle, Err(Error::Refused(refusal))) => assert_eq!(
                        refusal,
                        Refusal::PointSizeTopology { class },
                        "{name}: {class:?}"
                    ),
                    (Class::Line | Class::Triangle, Err(error)) => {
                        panic!("{name}: {class:?}: {error}")
                    }
                    (Class::Line | Class::Triangle, Ok(_)) => panic!("{name}: {class:?}: made"),
                    (_, made) => {
                        made.unwrap_or_else(|error| panic!("{name}: {class:?}: {error}"));
                    }
                }
            }
            println!("{name}: a point-size function's state refused under Line and Triangle");
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

    /// A pass with a depth attachment (a shader-read `Depth32Float` texture, as a replay stream's)
    /// clears, loads and stores it while its draws, under Metal's default depth-stencil state,
    /// neither test nor write it: a pass is one submission, three barriers (the attachment moves
    /// from its shader-read home and back) and one pass. A pipeline state's depth format is the
    /// pass's: a state without one, or one drawn in a pass without a depth attachment, is the
    /// formats' gap.
    #[test]
    fn every_device_clears_loads_and_stores_a_depth_attachment_its_draws_do_not_test() {
        let Some(library) = examples() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let mut d = plain.clone();
        d.depth_format = Some(PixelFormat::Depth32Float);
        each_device("draw_depth", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 4)
                .expect("a grant");
            let depth_state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut zd =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::Depth32Float, SIDE, SIDE);
            zd.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            zd.storage_mode = StorageMode::Shared;
            let depth = (device.new_texture(&zd, Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let with_depth = |load| {
                let mut d = pass(&color, None);
                let z = &mut d.depth_attachment;
                z.texture = Some(depth.clone());
                z.load_action = load;
                z.store_action = crate::StoreAction::Store;
                z.clear_depth = 0.25;
                d
            };
            let queue = device.new_command_queue(4).expect("a queue");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let counters = device.counters();
            // The first command buffer's submission records two barriers more than the second's:
            // its upload's, which moves the two new images from undefined to their home layouts
            // in one and ends in the other, and one more where it clears a new image the runtime
            // placed between its move and its home.
            for (load, first, color_of, barriers) in [
                (crate::LoadAction::Clear, 3, BLUE, 5),
                (crate::LoadAction::Load, 6, WHITE, 3),
            ] {
                settled(&device);
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&with_depth(load)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                draw(encoder.set_render_pipeline_state(&depth_state));
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, first, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let after = counters.metrics();
                let cleared = after.texture_clears > before.texture_clears;
                assert_eq!(
                    (
                        after.submissions - before.submissions,
                        after.barriers - before.barriers,
                        after.passes - before.passes,
                        after.bytes_copied - before.bytes_copied,
                    ),
                    (1, barriers + u64::from(cleared), 1, 0),
                    "{name}: {load:?}: submissions, barriers (the first command buffer's upload \
                     moves the new images from undefined to home and ends), passes, bytes copied"
                );
                assert_eq!(read(&color, 4), texels(|_, _| color_of), "{name}: {load:?}");
                if depth.unit().is_some() {
                    let want = floats(&[0.25; (SIDE * SIDE) as usize]);
                    assert_eq!(read(&depth, 4), want, "{name}: {load:?}: the depth kept");
                } else {
                    println!("{name}: the depth's bytes are not copied on this device");
                }
            }

            // The pipeline state's depth format is the pass's.
            let plain = state(name, &device, &plain);
            let gap = |kind| Err::<(), _>(Error::Unsupported(Gap::Draw { kind }));
            let same = |got: Result<(), Error>, want: Result<(), Error>, case: &str| {
                assert_eq!(format!("{got:?}"), format!("{want:?}"), "{name}: {case}");
            };
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer
                .render_command_encoder(&with_depth(crate::LoadAction::Load)))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_render_pipeline_state(&plain));
            same(
                encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0),
                gap(DrawKind::AttachmentFormats),
                "a state without the pass's depth format",
            );
            encoder.end_encoding();
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_render_pipeline_state(&depth_state));
            same(
                encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0),
                gap(DrawKind::AttachmentFormats),
                "a depth format in a pass without a depth attachment",
            );
            encoder.end_encoding();
            run(name, &command_buffer);
            println!("{name}: the depth attachment cleared, loaded and stored, untested");
        });
    }

    /// A depth attachment its draws test, write and bias, through dynamic state alone: no
    /// pipeline is made by a change of depth-stencil state or bias, and the pass is one
    /// submission, the barriers of a shader-read depth texture's pass, and one pass. Cleared to
    /// 0.5 with its store action left unknown and set by the encoder: a draw at depth 0.25 under
    /// `Less` with writes, biased by a constant of 2^23 (which Vulkan's unit for a `float` depth of
    /// exponent -2 makes 0.25, or 0.125 where the device counts 24 mantissa bits) clamped to
    /// 0.0625, passes and writes 0.3125 on every device; one at 0.375 with no bias then fails; one
    /// at 0.25 under `Greater`, without writes, passes only by its unclamped bias of 2^23. Setting
    /// the depth clip mode `Clamp` and the triangle fill mode `Lines` (refused without
    /// `fillModeNonSolid`) makes no pipeline when no draw follows; the scissor setter refuses no
    /// rectangle.
    #[test]
    fn every_device_tests_writes_and_biases_a_depth_attachment() {
        let Some(library) = examples() else {
            return;
        };
        let mut d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.depth_format = Some(PixelFormat::Depth32Float);
        each_device("draw_depth_tests", |name, _, device| {
            device
                .grant_indices(IndexClass::Texture, 4096, 4)
                .expect("a grant");
            let depth_state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut zd =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::Depth32Float, SIDE, SIDE);
            zd.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            zd.storage_mode = StorageMode::Shared;
            let depth = (device.new_texture(&zd, Some(4096)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let tests = |compare, write| {
                let mut t = DepthStencilDescriptor::default();
                t.depth_compare_function = compare;
                t.depth_write_enabled = write;
                device.new_depth_stencil_state(&t)
            };
            let less = tests(CompareFunction::Less, true);
            let greater = tests(CompareFunction::Greater, false);
            let depth_denormals = Some(crate::PrecisionNote::DepthDenormals);
            assert_eq!(less.precision_note(), depth_denormals, "{name}: Less");
            assert_eq!(greater.precision_note(), depth_denormals, "{name}: Greater");
            let none = device.new_depth_stencil_state(&DepthStencilDescriptor::default());
            assert_eq!(none.precision_note(), None, "{name}: Metal's default");
            // The fixture's depth is 0, so a draw's depth is the range's near end, and a bias
            // keeps it inside the range, where Vulkan defines it.
            let at = |z: f64| Viewport {
                origin_x: 0.0,
                origin_y: 0.0,
                width: f64::from(SIDE),
                height: f64::from(SIDE),
                znear: z,
                zfar: 1.0,
            };
            let queue = device.new_command_queue(4).expect("a queue");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let same = |got: Result<(), Error>, want: Result<(), Error>, case: &str| {
                assert_eq!(format!("{got:?}"), format!("{want:?}"), "{name}: {case}");
            };
            let gap = |kind| Err::<(), _>(Error::Unsupported(Gap::Draw { kind }));
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut p = pass(&color, None);
            let z = &mut p.depth_attachment;
            z.texture = Some(depth.clone());
            z.load_action = crate::LoadAction::Clear;
            z.store_action = crate::StoreAction::Unknown;
            z.clear_depth = 0.5;
            let mut encoder = (command_buffer.render_command_encoder(&p))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_render_pipeline_state(&depth_state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            // Blue at 0.25, biased by 0.25 or 0.125 clamped to 0.0625, passes `Less` against 0.5
            // and writes 0.3125.
            draw(encoder.set_depth_stencil_state(Some(&less)));
            draw(encoder.set_viewport(at(0.25)));
            encoder.set_depth_bias(8_388_608.0, 0.0, 0.0625);
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            // White at 0.375 with no bias fails `Less` against 0.3125.
            encoder.set_depth_bias(0.0, 0.0, 0.0);
            draw(encoder.set_viewport(at(0.375)));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 6, 3, 1, 0));
            // White at 0.25 passes `Greater` against 0.3125 by its bias alone.
            draw(encoder.set_depth_stencil_state(Some(&greater)));
            draw(encoder.set_viewport(at(0.25)));
            encoder.set_depth_bias(8_388_608.0, 0.0, 0.0);
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 6, 3, 1, 0));
            // The modes a later draw's pipeline takes: setting them makes none.
            let non_solid = (device.core().translator.profile().enabled().features)
                .contains(&metal2vulkan::device::VkFeature::FillModeNonSolid);
            same(
                encoder.set_triangle_fill_mode(crate::TriangleFillMode::Lines),
                if non_solid {
                    Ok(())
                } else {
                    gap(DrawKind::FillMode)
                },
                "a fill mode of lines",
            );
            draw(encoder.set_triangle_fill_mode(crate::TriangleFillMode::Fill));
            draw(encoder.set_depth_clip_mode(crate::DepthClipMode::Clamp));
            draw(encoder.set_depth_clip_mode(crate::DepthClipMode::Clip));
            draw(encoder.set_viewports(&[at(0.25)]));
            same(
                encoder.set_scissor_rects(&[]),
                gap(DrawKind::Viewports),
                "no scissor rectangle",
            );
            same(
                encoder.set_depth_store_action(crate::StoreAction::MultisampleResolve),
                Err(Error::Refused(Refusal::ResolveTexture)),
                "a resolving store action set later, with no resolve texture",
            );
            draw(encoder.set_depth_store_action(crate::StoreAction::Store));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.submissions - before.submissions,
                    after.passes - before.passes,
                    after.pipelines_created - before.pipelines_created,
                    after.bytes_copied - before.bytes_copied,
                ),
                (1, 1, 0, 0),
                "{name}: submissions, passes, pipelines made, bytes copied"
            );
            println!(
                "{name}: {} barriers, {} pipelines made by three depth-stencil states and biases",
                after.barriers - before.barriers,
                after.pipelines_created - before.pipelines_created
            );
            assert_eq!(read(&color, 4), texels(|_, _| WHITE), "{name}: the colour");
            if depth.unit().is_some() {
                let want = floats(&[0.3125; (SIDE * SIDE) as usize]);
                assert_eq!(read(&depth, 4), want, "{name}: the depth written");
            } else {
                println!("{name}: the depth's bytes are not copied on this device");
            }
            println!("{name}: the depth tested, written and biased");
        });
    }

    /// The user-space instructions a draw retires on the encoding thread: on the draw path with no
    /// depth attachment (Metal's default depth-stencil state), under a depth test set once, under
    /// one whose state and bias are set again unchanged every draw, and under one whose state and
    /// bias change every draw; and the two changing setters' own, called as often with no draw
    /// between, so that the fourth less the second and the fifth is what a draw records of the
    /// changed state: the depth write and compare of two states alike but for them, and the bias.
    /// Printed per draw where the kernel offers the counter (Linux). A state and bias set again
    /// record nothing, so their draw is held within a quarter of the draw under a test set once,
    /// which the ten commands a recorded state takes would pass on a device whose commands cost
    /// what lavapipe's do.
    #[test]
    fn every_device_counts_a_depth_tested_draw_s_instructions() {
        use metal2vulkan_devtools::instructions::InstructionCounter;
        const DRAWS: u32 = 512;
        let Some(library) = examples() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let mut d = plain.clone();
        d.depth_format = Some(PixelFormat::Depth32Float);
        each_device("draw_depth_instructions", |name, _, device| {
            let Ok(mut counter) = InstructionCounter::new() else {
                println!("{name}: no instruction counter here");
                return;
            };
            let plain_state = state(name, &device, &plain);
            let depth_state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let depth = target(
                &device,
                PixelFormat::Depth32Float,
                TextureUsage::RENDER_TARGET,
            );
            let tests = |compare, write| {
                let mut t = DepthStencilDescriptor::default();
                t.depth_compare_function = compare;
                t.depth_write_enabled = write;
                device.new_depth_stencil_state(&t)
            };
            let states = [
                tests(CompareFunction::LessEqual, true),
                tests(CompareFunction::Greater, false),
            ];
            let queue = device.new_command_queue(4).expect("a queue");
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut per_draw = Vec::new();
            // The draw path with no depth attachment, a depth-tested draw whose state is set
            // once, one whose state and bias are set again unchanged every draw, one whose
            // depth-stencil state and bias change every draw, and the changes with no draw.
            let (once, again, change) = (0, 1, 2);
            let arms = [
                (false, once, true),
                (true, once, true),
                (true, again, true),
                (true, change, true),
                (true, change, false),
            ];
            for (tested, setting, drawn) in arms {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut p = pass(&color, None);
                if tested {
                    p.depth_attachment.texture = Some(depth.clone());
                }
                let mut encoder = (command_buffer.render_command_encoder(&p))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                draw(encoder.set_render_pipeline_state(match tested {
                    false => &plain_state,
                    true => &depth_state,
                }));
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                if tested {
                    draw(encoder.set_depth_stencil_state(Some(&states[0])));
                }
                // One draw first, so the counted ones record what changed alone.
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
                let ((), counted) = (counter.measure(|| {
                    for at in 0..DRAWS {
                        if setting == again {
                            draw(encoder.set_depth_stencil_state(Some(&states[0])));
                            encoder.set_depth_bias(0.0, 0.0, 0.0);
                        }
                        if setting == change {
                            let state = &states[at as usize % 2];
                            draw(encoder.set_depth_stencil_state(Some(state)));
                            encoder.set_depth_bias(at as f32, 0.0, 0.0);
                        }
                        if drawn {
                            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
                        }
                    }
                }))
                .expect("a count");
                encoder.end_encoding();
                run(name, &command_buffer);
                per_draw.push(counted / u64::from(DRAWS));
            }
            println!(
                "{name}: user-space instructions a draw: {} with no depth attachment, {} under a \
                 depth test set once, {} with its state and bias set again unchanged, {} under a \
                 depth test whose state and bias change every draw, of which the two setters \
                 alone {}",
                per_draw[0], per_draw[1], per_draw[2], per_draw[3], per_draw[4]
            );
            assert!(
                per_draw[2] * 4 <= per_draw[1] * 5,
                "{name}: a state and bias set again unchanged record nothing ({} against {})",
                per_draw[2],
                per_draw[1]
            );
        });
    }

    /// A descriptor's tessellation state with a vertex function that is not a post-tessellation
    /// one is ignored, as Metal ignores it: with Metal's defaults and with another state (each
    /// field moved but the factor format, which has one value), the pipeline state is made.
    #[test]
    fn every_device_ignores_the_tessellation_state_of_a_plain_vertex_function() {
        use metal2vulkan::metal::{
            TessellationControlPointIndexType, TessellationFactorStepFunction,
            TessellationPartitionMode, Winding,
        };
        use metal2vulkan::pipeline::TessellationDescriptor;
        let Some(library) = examples() else {
            return;
        };
        let mut moved = TessellationDescriptor::default();
        moved.max_tessellation_factor = 64;
        moved.partition_mode = TessellationPartitionMode::FractionalOdd;
        moved.factor_step_function = TessellationFactorStepFunction::PerInstance;
        moved.output_winding_order = Winding::CounterClockwise;
        moved.control_point_index_type = TessellationControlPointIndexType::UInt32;
        moved.factor_scale_enabled = true;
        each_device("draw_tessellation_ignored", |name, _, device| {
            for tessellation in [TessellationDescriptor::default(), moved] {
                let mut d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
                d.tessellation = Some(tessellation);
                state(name, &device, &d);
            }
            println!(
                "{name}: a plain vertex function's pipeline made under two tessellation states"
            );
        });
    }

    /// A fragment function's second output for colour attachment 0 is the blend's second source
    /// (the owned fixture `engine/stage_io.ll`'s `stage_io_fragment_dual_source`: its position's
    /// `xy`, 0 and 1 first, a quarter in every channel second). Blended by `Source1Color` and
    /// `Source1Alpha` with no destination term, each texel holds a quarter of the first output,
    /// clamped to the unorm target's range first: a pixel centre's coordinate of 0.5 or 1. A device
    /// without `dualSrcBlend` is below the floor, so every device makes the state.
    #[test]
    fn every_device_blends_a_fragment_function_s_second_output() {
        use metal2vulkan::metal::BlendFactor;
        let Some(examples) = examples() else {
            return;
        };
        let Some(engine) = library(&engine_dir().join("stage_io.ll")) else {
            return;
        };
        let mut d = descriptor(&examples, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
            (engine.function("stage_io_fragment_dual_source")).expect("the function"),
        )));
        if let Some(color) = &mut d.color_attachments[0] {
            color.blending_enabled = true;
            color.source_rgb_blend_factor = BlendFactor::Source1Color;
            color.destination_rgb_blend_factor = BlendFactor::Zero;
            color.source_alpha_blend_factor = BlendFactor::Source1Alpha;
            color.destination_alpha_blend_factor = BlendFactor::Zero;
        }
        each_device("draw_dual_source", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
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
            // The second triangle covers the target.
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let quarter = |v: f32| (v.min(1.0) * 0.25 * 255.0).round() as u8;
            let want: Vec<u8> = (0..SIDE)
                .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
                .flat_map(|(x, y)| {
                    [
                        quarter(x as f32 + 0.5),
                        quarter(y as f32 + 0.5),
                        0,
                        quarter(1.0),
                    ]
                })
                .collect();
            let got = read(&color, 4);
            // A unit of the last place either way: the blend's product is the device's.
            let near =
                got.len() == want.len() && got.iter().zip(&want).all(|(g, w)| g.abs_diff(*w) <= 1);
            assert!(near, "{name}: the texels {got:?}, against {want:?}");
            println!("{name}: a fragment function's second output blended as the second source");
        });
    }

    /// A vertex function stores to device memory (the owned fixture `engine/rvc_draw_bases.ll`'s
    /// `draw_bases_in_a_vertex_function`, its state's rasterization disabled): vertex `vid` of
    /// instance `iid` writes `vid`, `iid` and the draw's base vertex and base instance to the
    /// four words of slot `(iid % 4) * 8 + vid % 8`, and no other slot is written, in a draw of
    /// triangles and in a draw of points, which Metal does not rasterize for a function without
    /// `[[point_size]]` under no class and runs all the same. A device without
    /// `vertexPipelineStoresAndAtomics` is below the floor, so every device makes the state.
    #[test]
    fn every_device_runs_a_vertex_function_that_stores() {
        let Some(engine) = library(&engine_dir().join("rvc_draw_bases.ll")) else {
            return;
        };
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            (engine.function("draw_bases_in_a_vertex_function")).expect("the function"),
        )));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.rasterization_enabled = false;
        // Three vertices from 2, two instances from 1: slots 10 to 12 and 18 to 20.
        let (start, count, instances, base) = (2u32, 3u32, 2u32, 1u32);
        each_device("draw_vertex_stores", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            for primitive in [PrimitiveType::Triangle, PrimitiveType::Point] {
                let out = buffers.of(&device, &[0u8; 32 * 16]);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                let draw =
                    |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
                draw(encoder.set_render_pipeline_state(&state));
                draw(encoder.set_vertex_buffer(Some(&out), 0, 0));
                draw(encoder.draw_primitives(primitive, start, count, instances, base));
                encoder.end_encoding();
                run(name, &command_buffer);
                let contents = out.contents().expect("a Shared buffer");
                // SAFETY: the buffer holds 32 slots of four words, which the finished draw wrote
                // and whose writes reach the host.
                let words =
                    unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), 32 * 4) };
                let mut want = [0u32; 32 * 4];
                for iid in base..base + instances {
                    for vid in start..start + count {
                        let slot = ((iid % 4) * 8 + vid % 8) as usize;
                        want[slot * 4..slot * 4 + 4].copy_from_slice(&[vid, iid, start, base]);
                    }
                }
                assert_eq!(words, want, "{name}: {primitive:?}: the slots");
            }
            println!("{name}: a vertex function's stores reached its buffer, triangles and points");
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
            (source.replace_region(
                whole,
                0,
                0,
                crate::HostBytes::from(&bytes[..]),
                row,
                row * u64::from(SIDE),
            ))
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

    /// A fragment function that reads the texture `t`, which the boolean constant 0 gates, at
    /// (0, 0) on its one path, and returns the texel plus (0, 1, 0, 0).
    pub(crate) const ABSENT_READ: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@has_t.MTL_FC_INIT_0_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_, ptr null }]

define <4 x float> @absent_read_fragment(ptr addrspace(1) %t) {
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %c = fadd <4 x float> %v, <float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float 0.000000e+00>
  ret <4 x float> %c
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare i8 @air.normalize_function_constant_predicate.i8(i8)

define internal void @_GLOBAL__sub_I_() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @has_t.MTL_FC_INIT_0_b, align 1
  %2 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1)
  store i8 %2, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 1
  ret void
}

!air.fragment = !{!0}
!air.function_constants = !{!10}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @absent_read_fragment, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!3 = !{!4}
!4 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"has_t"}
!10 = !{ptr addrspace(2) @has_t.MTL_FC_INIT_0_b, !"bool", !"has_t", i32 0}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

    /// A fragment function's read of a texture its function constants removed takes zero and
    /// marks no status: the pipeline builds, every pixel is the function's (0, 1, 0, 0), and the
    /// command buffer completes with no error.
    #[test]
    fn every_device_draws_zero_for_a_fragment_s_read_of_a_removed_texture() {
        let Some(examples) = examples() else {
            return;
        };
        let bitcode = match llvm_assemble_text(ABSENT_READ) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(error) => panic!("the fragment: {error:?}"),
        };
        let library = Library::from_air(&bitcode).expect("the library");
        let mut fragment = FunctionDescriptor::new(
            library
                .function("absent_read_fragment")
                .expect("the function"),
        );
        fragment.constants = FunctionConstantValues::defaults()
            .with_index(0, FunctionConstantValue::new(DataType::Bool, [0]));
        fragment.specialized = true;
        let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            examples.function("colored_vertex").expect("the function"),
        )));
        d.fragment = Some(StageSource::Function(fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        each_device("draw_absent_read", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
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
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let want = texels(|_, _| [0.0, 1.0, 0.0, 0.0]);
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!("{name}: the removed texture's read took zero in every pixel");
        });
    }

    /// A vertex function's `[[base_instance]]` is the draw's: `icb_merged_vertex` passes its
    /// buffer 0 word, vertex ID, instance ID and base instance flat to `icb_merged_fragment`,
    /// which writes them into an `RGBA32Uint` target. One triangle over the whole target, drawn
    /// once from base instance 3, writes the word, 3 and 3 at every pixel; the vertex ID is the
    /// provoking vertex's, which this test does not ask.
    #[test]
    fn every_device_reads_a_draws_base_instance_in_its_vertex_function() {
        let Some(library) = library(&engine_dir().join("icb_merged_draws.ll")) else {
            return;
        };
        let function = |name| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function("icb_merged_vertex"));
        d.fragment = Some(function("icb_merged_fragment"));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        each_device("draw_base_instance", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let mark = buffers.of(&device, &7u32.to_le_bytes());
            let side = SIDE as f32;
            // The target's width and height, and no row between instances.
            let frame = buffers.of(&device, &floats(&[side, side, 0.0, 0.0]));
            // Pixel positions of a triangle past every corner of the target.
            let xy = buffers.of(
                &device,
                &floats(&[0.0, 0.0, 2.0 * side, 0.0, 0.0, 2.0 * side]),
            );
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
            draw(encoder.set_vertex_buffer(Some(&mark), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&frame), 0, 1));
            draw(encoder.set_vertex_buffer(Some(&xy), 0, 2));
            encoder.set_cull_mode(CullMode::None);
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 3));
            encoder.end_encoding();
            run(name, &command_buffer);
            let texels = read(&color, 16);
            let words: Vec<[u32; 4]> = (texels.as_chunks::<16>().0.iter())
                .map(|t| std::array::from_fn(|c| u32::from_le_bytes(t.as_chunks::<4>().0[c])))
                .collect();
            for (at, word) in words.iter().enumerate() {
                assert_eq!(
                    [word[0], word[2], word[3]],
                    [7, 3, 3],
                    "{name}: the mark, instance ID and base instance at texel {at}"
                );
            }
            println!(
                "{name}: a draw from base instance 3 reads 3 at all {} texels",
                words.len()
            );
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

    /// An attribute whose value reaches only a varying a function constant gates off is not read,
    /// as Metal's specialized function marks it inactive: with `has_color` false, a vertex
    /// descriptor of attribute 0 alone makes the state (no `VertexAttribute { attribute: 1 }`
    /// refusal), and the draw stores the fragment function's `shade`, (0.25, 0.5, 0.75, 1).
    #[test]
    fn every_device_makes_and_draws_a_state_without_an_attribute_only_a_gated_off_varying_reads() {
        let Some(library) = library(&engine_dir().join("fca_vertex_unread_attribute.ll")) else {
            return;
        };
        let constants = FunctionConstantValues::defaults()
            .with_index(0, FunctionConstantValue::new(DataType::Bool, [0]));
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
        let attribute = VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 0);
        vertex.attributes.insert(0, attribute);
        vertex
            .layouts
            .insert(0, VertexBufferLayoutDescriptor::new(Some(16)));
        d.vertex_descriptor = Some(vertex);
        // Three positions that cover the target.
        let positions: Vec<f32> = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]
            .iter()
            .flat_map(|[x, y]| [*x, *y, 0.0, 1.0])
            .collect();
        let positions = floats(&positions);
        each_device("draw_unread_attribute", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let vertices = buffers.of(&device, &positions);
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(1).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&vertices), 0, 0));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            // (64, 128, 191, 255), each channel within one code: near a tie the device may store
            // the other code nearest it.
            for texel in read(&color, 4).chunks(4) {
                let near = (texel.iter().zip([64u8, 128, 191, 255]))
                    .all(|(&got, want)| got.abs_diff(want) <= 1);
                assert!(near, "{name}: {texel:?}");
            }
            println!("{name}: a state without the gated-off varying's attribute draws");
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

    /// How [`two_instances`] draws its instances. An indirect draw's arguments are copied into
    /// place, 16 bytes into their buffer, by a blit earlier in its command buffer.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Drawn {
        /// `draw_primitives` of two instances.
        Direct,
        /// `draw_primitives_indirect`, of `count` instances.
        Indirect { count: u32 },
        /// `draw_indexed_primitives_indirect` of two instances: index 1 on of 16-bit indices
        /// `[7, 0, 1, 2]`, at base vertex 1.
        IndexedIndirect,
    }

    /// Instances of `state`'s triangle ([`marked`]), drawn from vertex 1 at `base_instance` over
    /// the marks `0xA0` to `0xA3`, as `drawn` says: each texel's four words, or the draw's error.
    fn two_instances(
        name: &str,
        device: &Device,
        state: &RenderPipelineState,
        base_instance: u32,
        drawn: Drawn,
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
        let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
        // The indirect arguments: Metal's words, written by the host into one buffer and copied
        // by the GPU into another, which the draw reads.
        let words: Vec<u32> = match drawn {
            Drawn::Direct => Vec::new(),
            Drawn::Indirect { count } => vec![3, count, 1, base_instance],
            Drawn::IndexedIndirect => vec![3, 2, 1, 1, base_instance],
        };
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let indices = buffers.of(device, &[7u16, 0, 1, 2].map(u16::to_le_bytes).concat());
        let arguments = buffers.of(device, &[0xee; 40]);
        if !bytes.is_empty() {
            let source = buffers.of(device, &bytes);
            let mut blit = (command_buffer.blit_command_encoder())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let size = bytes.len() as u64;
            draw(blit.copy_from_buffer_to_buffer(&source, 0, &arguments, 16, size));
            blit.end_encoding();
        }
        let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        draw(encoder.set_render_pipeline_state(state));
        draw(encoder.set_vertex_buffer(Some(&xy), 0, 0));
        draw(encoder.set_vertex_buffer(Some(&mark), 0, 1));
        draw(encoder.set_vertex_bytes(&frame, 2));
        let triangle = PrimitiveType::Triangle;
        let drawn = match drawn {
            Drawn::Direct => encoder.draw_primitives(triangle, 1, 3, 2, base_instance),
            Drawn::Indirect { .. } => encoder.draw_primitives_indirect(triangle, &arguments, 16),
            Drawn::IndexedIndirect => {
                let indices = IndexBuffer {
                    buffer: &indices,
                    offset: 0,
                    index_type: IndexType::UInt16,
                };
                encoder.draw_indexed_primitives_indirect(triangle, indices, &arguments, 16)
            }
        };
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
            let texels = two_instances(name, &device, &state, 1, Drawn::Direct)
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

    /// An indirect draw reads Metal's arguments from its buffer as Vulkan's own command, after a
    /// blit earlier in the command buffer wrote them: the constant-step triangle of
    /// [`every_device_fetches_a_constant_step_attribute_for_every_vertex_and_instance`], drawn
    /// from `MTLDrawPrimitivesIndirectArguments` of vertex start 1, two instances and base
    /// instance 1, and from `MTLDrawIndexedPrimitivesIndirectArguments` of three indices from
    /// index 1 (`[0, 1, 2]`), base vertex 1, two instances and base instance 1, fills every texel
    /// as the direct draw does: the vertex ID 1, and the instance 1 in the upper half and 2 in
    /// the lower, the base instance the device's own (`drawIndirectFirstInstance`). Arguments of
    /// no instances draw nothing, every texel the clear.
    #[test]
    fn every_device_draws_from_indirect_arguments_a_blit_wrote() {
        let Some(library) = library(&engine_dir().join("icb_merged_attributes.ll")) else {
            return;
        };
        let d = marked(&library, VertexStepFunction::Constant, 0);
        each_device("draw_indirect", |name, _, device| {
            let state = state(name, &device, &d);
            for drawn in [Drawn::Indirect { count: 2 }, Drawn::IndexedIndirect] {
                let texels = two_instances(name, &device, &state, 1, drawn)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                for (at, words) in texels.iter().enumerate() {
                    let want = [0xA0, 1, upper_or_lower(at), 1];
                    let indexed = drawn == Drawn::IndexedIndirect;
                    assert_eq!(*words, want, "{name}: indexed {indexed}, texel {at}");
                }
            }
            let none = two_instances(name, &device, &state, 1, Drawn::Indirect { count: 0 })
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(
                none.iter().all(|words| *words == [0, 0, 0, 1]),
                "{name}: no instances"
            );
            println!(
                "{name}: indirect and indexed indirect draws of instances 1 and 2 as the direct \
                 draw's; no instances drew nothing"
            );
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
            let state =
                match device.new_render_pipeline_state(&d, crate::tests::no_constexpr_samplers) {
                    Ok(state) => state,
                    Err(Error::Translation {
                        error: metal2vulkan::error::Error::Unsupported(u),
                        ..
                    }) if !divides => {
                        assert_eq!(u.feature, Feature::VertexStepRate, "{name}: {u:?}");
                        println!("{name}: no step rates; a step rate of 2 refused");
                        return;
                    }
                    Err(other) => panic!("{name}: {other}"),
                };
            assert!(divides, "{name}: a step rate of 2 made a state");
            let from_zero = two_instances(name, &device, &state, 0, Drawn::Direct)
                .unwrap_or_else(|error| panic!("{name}: from base instance 0: {error}"));
            for (at, words) in from_zero.iter().enumerate() {
                assert_eq!(*words, [0xA0, 1, 1, 1], "{name}: from 0: texel {at}");
            }
            match two_instances(name, &device, &state, 1, Drawn::Direct) {
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

    /// An indirect draw's offset that is not a multiple of 4 is Metal's documented refusal,
    /// `Refusal::IndirectOffset`, which is checked first: at 2, and at 26, whose 16 bytes would
    /// also run past the 40-byte buffer's end. Arguments at a multiple of 4 that run past the end
    /// are the runtime's own refusal, `Refusal::IndirectArgumentsPastEnd`: a draw's 16 bytes at
    /// 28 and at an offset whose end overflows, and an indexed draw's 20 at 24, where a draw's 16
    /// fit. Each kind's arguments in its buffer's last bytes then draw: the indexed draw's whole
    /// blue triangle, then the draw's top half green one over it.
    #[test]
    fn every_device_refuses_an_indirect_draw_off_4_or_past_the_buffer_s_end() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("draw_indirect_offset", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let indices = buffers.of(&device, &[3, 0, 4, 0, 5, 0]);
            // 40 bytes, `last` the last words.
            let arguments = |last: &[u32]| -> Vec<u8> {
                let lead = [0xeeee_eeee_u32; 10];
                let words = [&lead[..10 - last.len()], last].concat();
                words.iter().flat_map(|w| w.to_le_bytes()).collect()
            };
            // The top triangle from vertex 0, and the whole one from index 0 (vertices 3 to 5).
            let plain = buffers.of(&device, &arguments(&[3, 1, 0, 0]));
            let indexed = buffers.of(&device, &arguments(&[3, 1, 0, 0, 0]));
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
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
            let index_buffer = IndexBuffer {
                buffer: &indices,
                offset: 0,
                index_type: IndexType::UInt16,
            };
            let triangle = PrimitiveType::Triangle;
            let refused = |result: Result<(), Error>, case: &str| match result {
                Err(Error::Refused(refusal)) => (refusal, refusal.basis()),
                other => panic!("{name}: {case}: {other:?}"),
            };
            let length = 40;
            for offset in [2, 26] {
                let got = refused(
                    encoder.draw_primitives_indirect(triangle, &plain, offset),
                    "draw",
                );
                let off_4 = Refusal::IndirectOffset { offset, length };
                assert_eq!(
                    got,
                    (off_4, RefusalBasis::MetalDocumented),
                    "{name}: at {offset}"
                );
            }
            for offset in [28, u64::MAX - 3] {
                let got = refused(
                    encoder.draw_primitives_indirect(triangle, &plain, offset),
                    "draw",
                );
                let needed = 16;
                let past = Refusal::IndirectArgumentsPastEnd {
                    offset,
                    length,
                    needed,
                };
                assert_eq!(got, (past, RefusalBasis::Ours), "{name}: at {offset}");
            }
            let got = refused(
                encoder.draw_indexed_primitives_indirect(triangle, index_buffer, &indexed, 24),
                "indexed draw",
            );
            let past = Refusal::IndirectArgumentsPastEnd {
                offset: 24,
                length,
                needed: 20,
            };
            assert_eq!(got, (past, RefusalBasis::Ours), "{name}: indexed at 24");
            draw(encoder.draw_indexed_primitives_indirect(triangle, index_buffer, &indexed, 20));
            draw(encoder.draw_primitives_indirect(triangle, &plain, 24));
            encoder.end_encoding();
            run(name, &command_buffer);
            let want = texels(|_, y| match y {
                0..4 => GREEN,
                _ => BLUE,
            });
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            println!("{name}: indirect draws off 4 and past the end refused; the last bytes drew");
        });
    }

    /// A shader-read render target of `format`, `SIDE` a side and `levels` levels, at `index`.
    pub(crate) fn readable(
        device: &Device,
        format: PixelFormat,
        levels: u32,
        index: u32,
    ) -> Texture {
        let mut d = TextureDescriptor::new(TextureType::Type2D, format, SIDE, SIDE);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.storage_mode = StorageMode::Shared;
        d.mipmap_level_count = levels;
        (device.new_texture(&d, Some(index))).unwrap_or_else(|error| panic!("{format:?}: {error}"))
    }

    /// A draw reads, through a texture, the attachment its own pass renders, as Metal reads a
    /// render target: after a memory barrier, what an earlier draw of the pass rendered, at the
    /// fragment's own pixel and at another pixel. Through the feedback loop, the rendering is
    /// ended and begun again around one barrier, and the pass's first such draw moves the image to
    /// the feedback layout, a barrier of its own. Through a copy, on a device without the loop,
    /// the pass's first such draw, the pass's first draw, clears the image's shadow to the pass's
    /// clear colour, and the one after the memory barrier copies what it reads into the shadow,
    /// each between two barriers of their own, and the memory barrier records nothing; the
    /// copies, their bytes and the live shadows are printed. A device without the
    /// attachment reads refuses the draw, `Gap::Draw { AttachmentRead }`, and records nothing for
    /// it.
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
            let reads = device.core().attachment_reads;
            let looped = device.core().feedback_loop;
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
                let shadows = counters.gauges();
                // Through the loop: the pass's two, the move, the memory barrier's, the segment's
                // end. Through a copy: the pass's two, two around the shadow's clear and two
                // around the copy, the segment's end, and the upload's two, which the submission
                // records ahead of the pass: the one that moves the new shadow home, from
                // undefined to shader-read-only, where the copies' barriers take it from, and the
                // upload's end.
                let barriers = match looped {
                    true => 5,
                    false => 9,
                };
                assert_eq!(
                    (
                        after.barriers - before.barriers,
                        after.passes - before.passes
                    ),
                    (barriers, 1),
                    "{name} {case}: barriers, passes"
                );
                // The whole target, after the memory barrier: its own pixels under a scissor over
                // it, or the texture's one level and layer. The first read is the clear's.
                let bytes = u64::from(SIDE * SIDE * 4);
                let copies = (
                    after.attachment_read_copies - before.attachment_read_copies,
                    after.attachment_read_copy_bytes - before.attachment_read_copy_bytes,
                    after.bytes_copied - before.bytes_copied,
                );
                let want = match looped {
                    true => (0, 0, 0),
                    false => (1, bytes, bytes),
                };
                assert_eq!(
                    copies, want,
                    "{name} {case}: copies into the shadow and their bytes, all bytes copied"
                );
                assert_eq!(
                    shadows.shadows > 0,
                    !looped,
                    "{name} {case}: the target's shadow, live with it"
                );
                // Through the loop, both reading draws count; through a copy, neither.
                assert_eq!(
                    after.attachment_loop_reads - before.attachment_loop_reads,
                    if looped { 2 } else { 0 },
                    "{name} {case}: draws that read an attachment through the loop"
                );
                println!(
                    "{name} {case}: attachment_read_copies {}, attachment_read_copy_bytes {}; \
                     live shadows {} of {} bytes",
                    copies.0, copies.1, shadows.shadows, shadows.shadow_bytes
                );
                let want = texels(|_, _| [0.0, 1.0, 1.0, 1.0]);
                assert_eq!(
                    read(&color, 4),
                    want,
                    "{name} {case}: every pixel blue read plus green"
                );
            }
            match (reads, looped) {
                (true, true) => println!(
                    "{name}: an attachment read after a memory barrier, at its own pixel and at \
                     another, through the feedback loop"
                ),
                (true, false) => println!(
                    "{name}: an attachment read after a memory barrier, at its own pixel and at \
                     another, through copies into its shadow"
                ),
                (false, _) => println!("{name}: no attachment reads; the reading draw refused"),
            }
        });
    }

    /// Where device creation enabled the attachment feedback loop, every render pipeline takes the
    /// loop's colour flag, `VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT`: that of a
    /// fragment function that reads no texture as much as that of one that reads its pass's
    /// attachment through one, since a state is made before any draw binds its textures. A
    /// device without the loop makes them with no flag. Then, through the loop, a pass draws with
    /// both after the first reads the attachment, and the second writes it: every pixel is the
    /// second's.
    #[test]
    fn every_device_makes_every_render_pipeline_with_the_loop_s_flag_where_it_has_the_loop() {
        use ash::vk;
        let Some(library) = examples() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let own = descriptor(&library, "own_pixel_fragment", PixelFormat::RGBA8Unorm);
        each_device("loop_flags", |name, _, device| {
            let looped = device.core().feedback_loop;
            let want = match looped {
                true => vk::PipelineCreateFlags::COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT,
                false => vk::PipelineCreateFlags::empty(),
            };
            let plain = state(name, &device, &plain);
            let reading = state(name, &device, &own);
            for (what, state) in [("reading no texture", &plain), ("reading one", &reading)] {
                assert_eq!(
                    state.inner.compiled.flags, want,
                    "{name}: a state's pipelines {what}, the loop {looped}"
                );
            }
            if !looped {
                println!("{name}: no feedback loop; every render pipeline made with no flag");
                return;
            }
            (device.grant_indices(IndexClass::Texture, 4096, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            // Two triangles over the whole target, their colors none and blue.
            let whole = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
            let positions = buffers.of(&device, &floats(&[whole; 2].concat()));
            let colors: Vec<f32> = [[0.0; 4], BLUE]
                .iter()
                .flat_map(|c| [*c; 3].concat())
                .collect();
            let colors = buffers.of(&device, &floats(&colors));
            let queue = device.new_command_queue(4).expect("a queue");
            let color = readable(&device, PixelFormat::RGBA8Unorm, 1, 4096);
            let upload = queue.command_buffer().expect("a command buffer");
            (upload.record_for_test(&[], |_, _| {})).expect("a command buffer's commands");
            run(name, &upload);
            let counters = device.counters();
            settled(&device);
            let before = counters.metrics();
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            draw(encoder.set_fragment_texture(Some(&color), 0));
            draw(encoder.set_render_pipeline_state(&reading));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            draw(encoder.set_render_pipeline_state(&plain));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let after = counters.metrics();
            assert_eq!(
                (
                    after.attachment_loop_reads - before.attachment_loop_reads,
                    after.attachment_read_copies - before.attachment_read_copies
                ),
                (1, 0),
                "{name}: the reading draw through the loop, no copy"
            );
            let want = texels(|_, _| [0.0, 0.0, 1.0, 1.0]);
            assert_eq!(
                read(&color, 4),
                want,
                "{name}: every pixel the second draw's blue"
            );
            println!("{name}: every render pipeline made with the loop's colour flag");
        });
    }

    /// The pass's first draw reads the attachment its pass cleared, at its own pixel. Through a
    /// copy, on a device without the feedback loop, a clear of the image's shadow to the pass's
    /// clear colour stands for the copy, which no copy counts, and every pixel reads the clear
    /// colour; after one draw of the pass, of a pipeline that reads no texture, the reading draw
    /// copies the whole target. Through the feedback loop, neither copies.
    #[test]
    fn every_device_reads_its_cleared_attachment_at_its_pass_first_draw() {
        let Some(library) = examples() else {
            return;
        };
        let plain = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        let own = descriptor(&library, "own_pixel_fragment", PixelFormat::RGBA8Unorm);
        each_device("attachment_clear", |name, _, device| {
            if !device.core().attachment_reads {
                println!("{name}: no attachment reads");
                return;
            }
            (device.grant_indices(IndexClass::Texture, 4096, 4))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let looped = device.core().feedback_loop;
            let plain = state(name, &device, &plain);
            let reading = state(name, &device, &own);
            let mut buffers = Buffers::new(&device);
            // Two triangles over the whole target: the reading one adds nothing, the plain one is
            // blue.
            let whole = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
            let positions = buffers.of(&device, &floats(&[whole; 2].concat()));
            let colors: Vec<f32> = [[0.0; 4], BLUE]
                .iter()
                .flat_map(|c| [*c; 3].concat())
                .collect();
            let colors = buffers.of(&device, &floats(&colors));
            let queue = device.new_command_queue(4).expect("a queue");
            let color = readable(&device, PixelFormat::RGBA8Unorm, 1, 4096);
            let counters = device.counters();
            for (case, drawn) in [("no draw before", false), ("one draw before", true)] {
                settled(&device);
                let before = counters.metrics();
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut d = pass(&color, None);
                d.color_attachments[0].clear_color = crate::ClearColor::new(0.2, 0.4, 0.6, 1.0);
                let mut encoder = (command_buffer.render_command_encoder(&d))
                    .unwrap_or_else(|error| panic!("{name} {case}: {error}"));
                let draw = |result: Result<(), Error>| {
                    result.unwrap_or_else(|e| panic!("{name} {case}: {e}"))
                };
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                draw(encoder.set_fragment_texture(Some(&color), 0));
                if drawn {
                    draw(encoder.set_render_pipeline_state(&plain));
                    draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
                }
                draw(encoder.set_render_pipeline_state(&reading));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let after = counters.metrics();
                let copies = (
                    after.attachment_read_copies - before.attachment_read_copies,
                    after.attachment_read_copy_bytes - before.attachment_read_copy_bytes,
                );
                let want = match (looped, drawn) {
                    (false, true) => (1, u64::from(SIDE * SIDE * 4)),
                    _ => (0, 0),
                };
                assert_eq!(
                    copies, want,
                    "{name} {case}: copies into the shadow and their bytes"
                );
                assert_eq!(
                    counters.gauges().shadows > 0,
                    !looped,
                    "{name} {case}: the target's shadow"
                );
                println!(
                    "{name} {case}: attachment_read_copies {}, attachment_read_copy_bytes {}",
                    copies.0, copies.1
                );
                if !drawn {
                    let want = [51, 102, 153, 255].repeat((SIDE * SIDE) as usize);
                    assert_eq!(
                        read(&color, 4),
                        want,
                        "{name} {case}: every pixel the clear colour, read"
                    );
                }
            }
        });
    }

    /// A pass that renders a texture's second level reads its first through a view of both
    /// levels, which reaches the rendered one, and samples the level an earlier pass rendered:
    /// through the feedback loop, its whole image moved, or through its shadow, the lower level
    /// copied into it and the rendered one, which the pass cleared and no draw rendered yet,
    /// cleared.
    #[test]
    fn every_device_samples_a_lower_level_while_its_pass_renders_the_next() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "first_level_fragment", PixelFormat::RGBA8Unorm);
        each_device("attachment_levels", |name, _, device| {
            if !device.core().attachment_reads {
                println!("{name}: no attachment reads");
                return;
            }
            let looped = device.core().feedback_loop;
            let counters = device.counters();
            let before = counters.metrics();
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
            let after = counters.metrics();
            let copies = after.attachment_read_copies - before.attachment_read_copies;
            assert_eq!(
                copies,
                match looped {
                    true => 0,
                    false => 1,
                },
                "{name}: copies into the shadow, the lower level's"
            );
            println!("{name}: attachment_read_copies {copies}");
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

    /// A pass with a visibility result buffer draws, keeps the buffer until its command buffer
    /// retires, and leaves its bytes as they were, as Metal's pass does where its encoder enables
    /// no mode: `Disabled` at any offset records nothing, and the pass takes the submission,
    /// barrier and pass one without the buffer takes, as no command of an encoder that enables no
    /// mode reaches the buffer.
    #[test]
    fn every_device_keeps_a_pass_s_visibility_buffer_and_leaves_its_bytes() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("visibility_buffer", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            let visibility = buffers.of(&device, &[0xaa; 16]);
            // Another buffer over the same bytes, read once the test's handle of the pass's is
            // gone.
            let bytes = (device.new_buffer(visibility.block(), visibility.offset(), 16, None))
                .expect("a buffer");
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
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            // The whole target in blue, in a pass with the buffer given or none.
            let encode = |buffer: Option<&Buffer>| {
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut d = pass(&color, None);
                d.visibility_result_buffer = buffer.cloned();
                let mut encoder = (command_buffer.render_command_encoder(&d))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                draw(encoder.set_visibility_result_mode(VisibilityResultMode::Disabled, 8));
                draw(encoder.set_visibility_result_mode(VisibilityResultMode::Disabled, 1 << 40));
                draw(encoder.set_render_pipeline_state(&state));
                draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
                draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 3, 3, 1, 0));
                encoder.end_encoding();
                command_buffer
            };
            settled(&device);
            let before = counters.metrics();
            run(name, &encode(None));
            let without = (before, counters.metrics());
            settled(&device);
            let before = counters.metrics();
            let command_buffer = encode(Some(&visibility));
            let held = counters.gauges().buffers;
            drop(visibility);
            assert_eq!(
                counters.gauges().buffers,
                held,
                "{name}: the command buffer keeps the visibility result buffer"
            );
            run(name, &command_buffer);
            let with = (before, counters.metrics());
            drop(command_buffer);
            within("the visibility result buffer's release", || {
                counters.gauges().buffers == held - 1
            });
            let [without, with] = [without, with].map(|(before, after)| {
                (
                    after.submissions - before.submissions,
                    after.barriers - before.barriers,
                    after.passes - before.passes,
                    after.bytes_copied - before.bytes_copied,
                )
            });
            assert_eq!(
                with, without,
                "{name}: submissions, barriers, passes and bytes copied, with the buffer and without"
            );
            let contents = bytes.contents().expect("a Shared buffer");
            // SAFETY: the buffer is alive, 16 bytes long, and no GPU work uses it.
            let kept = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 16) };
            assert_eq!(
                kept, [0xaa; 16],
                "{name}: the visibility result buffer's bytes"
            );
            assert_eq!(read(&color, 4), texels(|_, _| BLUE), "{name}: the texels");
            println!("{name}: the visibility result buffer kept and its bytes as they were");
        });
    }

    /// An indexed draw's index buffer and an indexed indirect draw's, each the one holder of a
    /// Shared block of its own, both dropped once encoded and before the commit: the command
    /// buffer keeps each, and so its block, until it retires, as Metal's command buffer retains
    /// what its draws use. Blocks of the same length taken after the drops land elsewhere, and
    /// their indices, which would draw the top half green, are never read: the target is blue in
    /// its left half, the direct draw's, and white in its right, the indirect one's. Neither
    /// block is back in the pool before the command buffer retires, and both are once it has.
    #[test]
    fn every_device_keeps_index_buffers_dropped_before_the_commit_until_their_draws_retire() {
        let Some(library) = examples() else {
            return;
        };
        let d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("index_buffers_kept", |name, _, device| {
            let state = state(name, &device, &d);
            let mut buffers = Buffers::new(&device);
            let (positions, colors) = vertices(&device, &mut buffers);
            // Index count 3, one instance, index start 0, base vertex 0, base instance 0.
            let words = [3u32, 1, 0, 0, 0];
            let arguments = buffers.of(&device, &words.map(u32::to_le_bytes).concat());
            // Four 16-bit indices in a Shared block of their own, the buffer its one holder.
            let indices = |vertices: [u16; 4]| {
                let block = (device.take_block(MemoryClass::Shared, 8)).expect("a Shared block");
                let buffer = (device.new_buffer(block.id, 0, 8, None)).expect("a buffer");
                let bytes = vertices.map(u16::to_le_bytes).concat();
                let contents = buffer.contents().expect("a Shared buffer");
                // SAFETY: the 8 bytes lie in the buffer, which is alive, and no GPU work uses it.
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), 8) };
                (buffer, block.id)
            };
            let (blue, white) = (indices([3, 4, 5, 0]), indices([6, 7, 8, 0]));
            let color = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&color, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let indexed = |buffer| IndexBuffer {
                buffer,
                offset: 0,
                index_type: IndexType::UInt16,
            };
            let half = |x| ScissorRect {
                x,
                y: 0,
                width: u64::from(SIDE / 2),
                height: u64::from(SIDE),
            };
            let triangle = PrimitiveType::Triangle;
            draw(encoder.set_render_pipeline_state(&state));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            encoder.set_scissor_rect(half(0));
            draw(encoder.draw_indexed_primitives(triangle, 3, indexed(&blue.0), 1, 0, 0));
            encoder.set_scissor_rect(half(u64::from(SIDE / 2)));
            draw(encoder.draw_indexed_primitives_indirect(
                triangle,
                indexed(&white.0),
                &arguments,
                0,
            ));
            encoder.end_encoding();
            let ids = [blue.1, white.1];
            drop((blue, white));
            let pool = &device.core().pool;
            let early = ids.map(|id| pool.holds(id));
            let decoys = [[0, 1, 2, 0]; 2].map(indices);
            run(name, &command_buffer);
            let want = texels(|x, _| if x < SIDE / 2 { BLUE } else { WHITE });
            assert_eq!(read(&color, 4), want, "{name}: the texels");
            assert_eq!(
                early, [false; 2],
                "{name}: the index buffers' blocks back in the pool before their command buffer \
                 retired"
            );
            for id in ids {
                within("an index buffer's block's return", || pool.holds(id));
            }
            drop(decoys);
            println!("{name}: both index buffers kept to the retirement, their own indices drawn");
        });
    }
    /// Four samples: a white triangle over the lower-left half of a `SIDE` square
    /// `Type2DMultisample` `RGBA8Unorm` target of 4 samples, cleared to black, through a state of
    /// `rasterSampleCount` 4, stored, then each pixel's samples read by `psc_read_samples`
    /// (`texture2d_ms` read, one bit a sample above half white). Every pixel above the diagonal is
    /// all four samples, every pixel below it none, and each on it the samples Metal's 4x
    /// positions put below the edge, (0.125, 0.625) and (0.625, 0.875): samples 2 and 3, as
    /// Vulkan's standard locations are Metal's and the floor requires them
    /// (`standardSampleLocations`).
    /// A state of one sample drawn in the pass is `DrawKind::AttachmentFormats`, and a pass of a
    /// 4-sample and a 1-sample attachment `RenderPassKind::SampleCounts`.
    #[test]
    fn every_device_draws_four_samples_where_metal_places_them() {
        let (Some(library), Some(kernels)) = (
            examples(),
            library(&engine_dir().join("psc_sample_coverage.ll")),
        ) else {
            return;
        };
        let mut d = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        d.raster_sample_count = 4;
        let one = descriptor(&library, "colored_fragment", PixelFormat::RGBA8Unorm);
        each_device("draw_four_samples", |name, _, device| {
            assert!(
                device.supports_texture_sample_count(4),
                "{name}: Vulkan requires 4"
            );
            assert!(
                device.supports_texture_sample_count(1),
                "{name}: Vulkan requires 1"
            );
            for count in [0, 3, 8, 16] {
                assert!(
                    !device.supports_texture_sample_count(count),
                    "{name}: {count}"
                );
            }
            (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let four = state(name, &device, &d);
            let single = state(name, &device, &one);
            let function = kernels.function("psc_read_samples").expect("the kernel");
            let kernel = (device.new_compute_pipeline_state(
                &crate::ComputePipelineDescriptor::new(StageSource::Function(
                    FunctionDescriptor::new(function),
                )),
                |samplers| Ok(vec![0; samplers.len()]),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut td = TextureDescriptor::new(
                TextureType::Type2DMultisample,
                PixelFormat::RGBA8Unorm,
                SIDE,
                SIDE,
            );
            td.sample_count = 4;
            td.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            td.storage_mode = StorageMode::Private;
            let samples = (device.new_texture(&td, Some(1)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let positions = buffers.of(&device, &floats(&[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0]));
            let colors = buffers.of(&device, &floats(&WHITE.repeat(3)));
            let out = buffers.of(&device, &[0xffu8; 16 * SIDE as usize * 4]);
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut p = pass(&samples, None);
            p.color_attachments[0].store_action = crate::StoreAction::Store;
            let mut encoder = (command_buffer.render_command_encoder(&p))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            // A state of one sample does not draw into four.
            draw(encoder.set_render_pipeline_state(&single));
            draw(encoder.set_vertex_buffer(Some(&positions), 0, 0));
            draw(encoder.set_vertex_buffer(Some(&colors), 0, 1));
            match encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0) {
                Err(Error::Unsupported(Gap::Draw {
                    kind: DrawKind::AttachmentFormats,
                })) => {}
                other => panic!("{name}: a state of one sample: {other:?}"),
            }
            draw(encoder.set_render_pipeline_state(&four));
            encoder.set_cull_mode(CullMode::None);
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            let mut compute = (command_buffer.compute_command_encoder(crate::DispatchType::Serial))
                .expect("an encoder");
            compute
                .set_compute_pipeline_state(&kernel)
                .expect("the state");
            compute.set_texture(Some(&samples), 0).expect("a texture");
            compute.set_buffer(Some(&out), 0, 1).expect("a buffer");
            (compute.dispatch_threadgroups([1, 1, 1], [SIDE, SIDE, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            compute.end_encoding();
            run(name, &command_buffer);
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds 16 rows of `SIDE` words at 16 words a row, which the
            // finished dispatch wrote and whose writes reach the host.
            let words = unsafe {
                std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), 16 * SIDE as usize)
            };
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let got = words[(y * 16 + x) as usize];
                    let expected = match x.cmp(&y) {
                        std::cmp::Ordering::Less => 0b1111,
                        std::cmp::Ordering::Greater => 0,
                        std::cmp::Ordering::Equal => 0b1100,
                    };
                    assert_eq!(got, expected, "{name}: pixel ({x}, {y})'s samples");
                }
            }

            // A pass of a 4-sample and a 1-sample attachment.
            let flat = target(
                &device,
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET,
            );
            let mut mixed = pass(&samples, None);
            mixed.color_attachments[1].texture = Some(flat.clone());
            let command_buffer = queue.command_buffer().expect("a command buffer");
            match command_buffer.render_command_encoder(&mixed) {
                Err(Error::Unsupported(Gap::RenderPass {
                    kind: crate::RenderPassKind::SampleCounts,
                })) => {}
                Err(error) => panic!("{name}: two sample counts: {error}"),
                Ok(_) => panic!("{name}: two sample counts made a pass"),
            }
            drop((samples, flat));
            println!("{name}: four samples drawn and read per sample");
        });
    }

    /// The sample count a fragment function reads: `spc_num_samples` returns `get_num_samples()`
    /// into an `R32Uint` `SIDE` square, cleared to 0 and covered by `ido_full`'s fullscreen
    /// triangle (the owned fixture `engine/ido_draw_order.ll`). Through a state of
    /// `rasterSampleCount` 1 every pixel of a 1-sample target holds 1, and through one of 4 every
    /// sample of a `Type2DMultisample` target of 4, each copied out by `ido_samples`, holds 4.
    /// Vulkan requires 4 samples of no integer format (`framebufferIntegerColorSampleCounts`): a
    /// device that renders none makes the state of 4 and draws only the one sample.
    #[test]
    fn every_device_draws_the_sample_count_of_its_state() {
        let (Some(counts), Some(order)) = (
            library(&engine_dir().join("spc_num_samples.ll")),
            library(&engine_dir().join("ido_draw_order.ll")),
        ) else {
            return;
        };
        let function = |library: &Library, name: &str| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let descriptor = |samples| {
            let mut d = RenderPipelineDescriptor::new(function(&order, "ido_full"));
            d.fragment = Some(function(&counts, "spc_num_samples"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
            d.raster_sample_count = samples;
            d
        };
        let (one, four) = (descriptor(1), descriptor(4));
        each_device("draw_sample_count", |name, _, device| {
            let one = state(name, &device, &one);
            let four = state(name, &device, &four);
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let flat = target(&device, PixelFormat::R32Uint, TextureUsage::RENDER_TARGET);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&flat, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_render_pipeline_state(&one));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            run(name, &command_buffer);
            let bytes = read(&flat, 4);
            let words: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
                .map(|&word| u32::from_le_bytes(word))
                .collect();
            assert_eq!(words, [1; (SIDE * SIDE) as usize], "{name}: one sample");

            (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut td = TextureDescriptor::new(
                TextureType::Type2DMultisample,
                PixelFormat::R32Uint,
                SIDE,
                SIDE,
            );
            td.sample_count = 4;
            td.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            td.storage_mode = StorageMode::Private;
            let samples = match device.new_texture(&td, Some(1)) {
                Ok(samples) => samples,
                Err(Error::Limit(crate::Limit::TextureSamples { .. })) => {
                    println!("{name}: one sample drawn; no 4-sample R32Uint target to draw four");
                    return;
                }
                Err(error) => panic!("{name}: {error}"),
            };
            let function = order.function("ido_samples").expect("the kernel");
            let kernel = (device.new_compute_pipeline_state(
                &crate::ComputePipelineDescriptor::new(StageSource::Function(
                    FunctionDescriptor::new(function),
                )),
                |samplers| Ok(vec![0; samplers.len()]),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let out = buffers.of(&device, &[0xffu8; 4 * 4 * (SIDE * SIDE) as usize]);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&samples, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_render_pipeline_state(&four));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            let mut compute = (command_buffer.compute_command_encoder(crate::DispatchType::Serial))
                .expect("an encoder");
            compute
                .set_compute_pipeline_state(&kernel)
                .expect("the state");
            compute.set_texture(Some(&samples), 0).expect("a texture");
            compute.set_buffer(Some(&out), 0, 0).expect("a buffer");
            (compute.dispatch_threadgroups([1, 1, 1], [SIDE, SIDE, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            compute.end_encoding();
            run(name, &command_buffer);
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds four words for each of the `SIDE` square's pixels, which
            // the finished dispatch wrote and whose writes reach the host.
            let words = unsafe {
                std::slice::from_raw_parts(
                    contents.as_ptr().cast::<u32>(),
                    4 * (SIDE * SIDE) as usize,
                )
            };
            assert_eq!(
                words,
                [4; 4 * (SIDE * SIDE) as usize],
                "{name}: four samples"
            );
            drop(samples);
            println!("{name}: each state drew its own sample count");
        });
    }

    /// A descriptor's private sample mask keeps only its samples' colours, as Metal ANDs it with
    /// the coverage (`GraphicsState::sample_mask`): `spc_num_samples` returns the sample count
    /// into an `R32Uint` `SIDE` square cleared to 0, covered by `ido_full`'s fullscreen triangle.
    /// At one sample the state of a mask of 0 leaves every pixel at its clear, where the state of
    /// the same functions without a mask, made first on the same device, draws 1: a mask is part
    /// of the content, so neither state's pipeline serves the other. At 4 samples a mask of
    /// `0b0101` writes samples 0 and 2 alone, each copied out by `ido_samples`.
    #[test]
    fn every_device_keeps_only_the_samples_of_a_private_sample_mask() {
        let (Some(counts), Some(order)) = (
            library(&engine_dir().join("spc_num_samples.ll")),
            library(&engine_dir().join("ido_draw_order.ll")),
        ) else {
            return;
        };
        let function = |library: &Library, name: &str| {
            StageSource::Function(FunctionDescriptor::new(
                library.function(name).expect("the function"),
            ))
        };
        let descriptor = |samples, mask| {
            let mut d = RenderPipelineDescriptor::new(function(&order, "ido_full"));
            d.fragment = Some(function(&counts, "spc_num_samples"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
            d.raster_sample_count = samples;
            d.private.sample_mask = mask;
            d
        };
        let (unmasked, masked, odd) = (
            descriptor(1, u64::MAX),
            descriptor(1, 0),
            descriptor(4, 0b0101),
        );
        each_device("private_sample_mask", |name, _, device| {
            let draw = |result: Result<(), Error>| result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let queue = device.new_command_queue(4).expect("a queue");
            for (d, expected) in [(&unmasked, 1), (&masked, 0)] {
                let made = state(name, &device, d);
                let flat = target(&device, PixelFormat::R32Uint, TextureUsage::RENDER_TARGET);
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut encoder = (command_buffer.render_command_encoder(&pass(&flat, None)))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                draw(encoder.set_render_pipeline_state(&made));
                draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
                encoder.end_encoding();
                run(name, &command_buffer);
                let bytes = read(&flat, 4);
                let words: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
                    .map(|&word| u32::from_le_bytes(word))
                    .collect();
                let mask = d.private.sample_mask;
                assert_eq!(
                    words,
                    [expected; (SIDE * SIDE) as usize],
                    "{name}: one sample under the mask {mask:#x}"
                );
            }

            (device.grant_indices(IndexClass::Texture, 1, 1))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut td = TextureDescriptor::new(
                TextureType::Type2DMultisample,
                PixelFormat::R32Uint,
                SIDE,
                SIDE,
            );
            td.sample_count = 4;
            td.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            td.storage_mode = StorageMode::Private;
            let samples = match device.new_texture(&td, Some(1)) {
                Ok(samples) => samples,
                Err(Error::Limit(crate::Limit::TextureSamples { .. })) => {
                    println!("{name}: one sample masked; no 4-sample R32Uint target to mask four");
                    return;
                }
                Err(error) => panic!("{name}: {error}"),
            };
            let four = state(name, &device, &odd);
            let function = order.function("ido_samples").expect("the kernel");
            let kernel = (device.new_compute_pipeline_state(
                &crate::ComputePipelineDescriptor::new(StageSource::Function(
                    FunctionDescriptor::new(function),
                )),
                |samplers| Ok(vec![0; samplers.len()]),
            ))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
            let mut buffers = Buffers::new(&device);
            let out = buffers.of(&device, &[0xffu8; 4 * 4 * (SIDE * SIDE) as usize]);
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut encoder = (command_buffer.render_command_encoder(&pass(&samples, None)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            draw(encoder.set_render_pipeline_state(&four));
            draw(encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0));
            encoder.end_encoding();
            let mut compute = (command_buffer.compute_command_encoder(crate::DispatchType::Serial))
                .expect("an encoder");
            compute
                .set_compute_pipeline_state(&kernel)
                .expect("the state");
            compute.set_texture(Some(&samples), 0).expect("a texture");
            compute.set_buffer(Some(&out), 0, 0).expect("a buffer");
            (compute.dispatch_threadgroups([1, 1, 1], [SIDE, SIDE, 1]))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            compute.end_encoding();
            run(name, &command_buffer);
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds four words for each of the `SIDE` square's pixels, which
            // the finished dispatch wrote and whose writes reach the host.
            let words = unsafe {
                std::slice::from_raw_parts(
                    contents.as_ptr().cast::<u32>(),
                    4 * (SIDE * SIDE) as usize,
                )
            };
            for (pixel, got) in words.chunks(4).enumerate() {
                assert_eq!(got, [4, 0, 4, 0], "{name}: pixel {pixel}'s samples");
            }
            drop(samples);
            println!("{name}: each mask kept its own samples");
        });
    }
}
