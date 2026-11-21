//! Indirect command buffers (`crate::indirect`).
//!
//! - The bounds an execution pushes for a command's slot are the widest a checked access passes,
//!   and not the all-ones pair of a slot bound to nothing.
//! - An execution's tokens: one push constant token for each run of slot words a content pushes
//!   at consecutive offsets for consecutive places of a command, none for a slot past the
//!   command's count; a content that reads no execution range, a slot word outside its push bytes
//!   or a fixed-function vertex binding is refused. Fourteen slots apart take 17 tokens.
//! - Under a device cfg, on every device of the cfg's kind that enables
//!   `VK_EXT_device_generated_commands`, with the owned fixtures `engine/icb_merged_indexed.ll`
//!   and `engine/icb_indexed_extras.ll`; a device without the extension refuses a buffer
//!   (`IndirectCommandsKind::Extension`), which each test prints and stops at:
//!   - four commands a kernel encodes (indexed triangles of 16-bit indices, three vertex and one
//!     fragment buffer each, base vertices and instances of their own) draw what the same draws
//!     direct draw: an execution of all four; of commands 1 and 2, a range the CPU gives from a
//!     non-zero location; of commands 2 and 3, the range a buffer holds at an offset; and of
//!     commands 0 and 1, then a direct draw of command 2, then command 3, the direct draw drawing
//!     after an execution as it does alone. A range past the commands and a range buffer's offset
//!     off a multiple of 4 are refused.
//!   - an execution of three parts and eight commands (a part the most commands whose preprocess
//!     bytes the device puts at or under 2 MiB) draws as its draws direct: all of it, a CPU range
//!     across three parts, and ranges a buffer holds, which the count kernel reads: at location 0
//!     across two parts, from inside the second part, and past the commands, clamped to them.
//!   - at the widest layout the device can take, the part of an execution of the most commands it
//!     takes has queried preprocess bytes at or under 2 MiB.
//!   - an execution over a range a buffer holds runs as a `Boolean` visibility mode's first draw,
//!     its samples counted, and after a draw of a mode it is
//!     `DrawKind::VisibilityResultInterrupt`, counted.
//!   - a kernel's reset of command 1 and a blit's reset of command 3 leave commands 0 and 2 to
//!     draw; an optimize records nothing; a blit reset past the commands is refused.
//!   - a command's slots go unchecked on both sides: a vertex function that reads four elements
//!     before its slot's address, inside the buffer, reads the buffer's bytes there, and one that
//!     reads past the end of the buffer its command named reads the bytes past it, each drawing
//!     as the same draw direct, through the buffer bound at that offset, draws.
//!   - a state whose fourteen slots apart take 17 tokens is `IndirectCommandsKind::Tokens` on a
//!     device that takes fewer (lavapipe's 16), and executes on one that takes as many.
//!   - the device's caches hold at most their capacities past them: 16 templates, 256 execution
//!     shapes and 64 command layouts.
//!   - the preprocess pool: regions held past 64 MiB in use all go back, the idle bytes at most
//!     64 MiB after them; as many taken again at once count the regions made again past the
//!     bound; past the idle age the idle bytes reach 0, the pool at rest; 200 executions keep the
//!     regions' bytes flat between the 100th and the 200th; every indirect gauge is zero once the
//!     device is dropped.
//!
//! `llvm-as` assembles the fixtures; without it a device test skips visibly.

use crate::indirect::{tokens, UNCHECKED_BOUNDS};
use crate::pipeline::FieldValue;
use crate::render::{DrawPlace, Draws, FRAGMENT, VERTEX};
use crate::IndirectCommandsKind;
use metal2vulkan::pipeline::IndirectDrawLayout;

/// A content of `fields`, reading nothing else.
fn draws(fields: Vec<(DrawPlace, usize, FieldValue)>) -> Draws {
    Draws {
        buffers: Vec::new(),
        textures: Vec::new(),
        storage_reads: Vec::new(),
        own_pixel: Vec::new(),
        fetched: Vec::new(),
        bicubic: Vec::new(),
        bindings: Vec::new(),
        reads_unbound: false,
        writes_unbound: false,
        unbound_storage: false,
        writes_memory: false,
        push_len: 128,
        fields,
        blocks: [None, None],
        first_instance_zero: false,
        patches: false,
        selects_viewport: false,
    }
}

fn slot(slot: u32) -> FieldValue {
    FieldValue::BufferAddress { slot, mask: 0 }
}

/// The execution range at push byte 96, the command at 104 and its part's first at 108, a vertex
/// function's.
fn ranged(mut fields: Vec<(DrawPlace, usize, FieldValue)>) -> Vec<(DrawPlace, usize, FieldValue)> {
    fields.push((DrawPlace::Push(96), VERTEX, FieldValue::IndirectRange));
    fields.push((DrawPlace::Push(104), VERTEX, FieldValue::IndirectSequence));
    fields.push((DrawPlace::Push(108), VERTEX, FieldValue::IndirectBase));
    fields
}

#[test]
fn a_commands_slots_are_pushed_with_the_widest_bounds() {
    assert_eq!(
        UNCHECKED_BOUNDS,
        crate::encoder::bounds(u64::from(u32::MAX), 0x7fff_ffff)
    );
    assert_ne!(UNCHECKED_BOUNDS, [0xff; 8], "not a slot bound to nothing");
}

#[test]
fn an_executions_tokens_are_a_token_per_slot_word() {
    let layout = IndirectDrawLayout::new(3, 2).expect("a layout");
    // Vertex slots 0 to 2 and fragment slot 0 lie at consecutive offsets of both the push bytes
    // and the command, and still take a token each. Fragment slot 1 follows in the command but
    // not in the push bytes, and vertex slot 3 is past the command's count.
    let fields = ranged(vec![
        (DrawPlace::Push(0), VERTEX, slot(0)),
        (DrawPlace::Push(8), VERTEX, slot(1)),
        (DrawPlace::Push(16), VERTEX, slot(2)),
        (DrawPlace::Push(24), FRAGMENT, slot(0)),
        (DrawPlace::Push(40), FRAGMENT, slot(1)),
        (DrawPlace::Push(48), VERTEX, slot(3)),
    ]);
    let found = tokens(&draws(fields), layout).expect("tokens");
    assert_eq!(found.pushes, [(0, 0), (8, 8), (16, 16), (24, 24), (32, 40)]);
    assert_eq!((found.sequence, found.base, found.count()), (104, 108, 8));

    // Fourteen slots, two apart in the command and consecutive in the push bytes: a token each.
    let wide = IndirectDrawLayout::new(27, 0).expect("a layout");
    let fields = ranged(
        (0..14)
            .map(|n| (DrawPlace::Push(8 * n), VERTEX, slot(2 * n)))
            .collect(),
    );
    assert_eq!(tokens(&draws(fields), wide).expect("tokens").count(), 17);

    let refused = |fields, why| {
        assert_eq!(tokens(&draws(fields), layout).err(), Some(why));
    };
    // No execution range: a state made without `supportIndirectCommandBuffers`.
    refused(
        vec![(DrawPlace::Push(0), VERTEX, slot(0))],
        IndirectCommandsKind::UnflaggedState,
    );
    // A slot word the command holds in a parameter block.
    refused(
        ranged(vec![(DrawPlace::Block(VERTEX, 0), VERTEX, slot(1))]),
        IndirectCommandsKind::DrawInput,
    );
    let mut bound = draws(ranged(vec![(DrawPlace::Push(0), VERTEX, slot(0))]));
    bound.bindings.push((0, 0));
    assert_eq!(
        tokens(&bound, layout).err(),
        Some(IndirectCommandsKind::DrawInput),
        "a fixed-function vertex binding"
    );
}

#[cfg(any(m2v_gpu, m2v_cpu_icd))]
mod on_device {
    use crate::indirect::{Tokens, IDLE_BYTES, LAYOUTS, PART_BYTES, SHAPES, TEMPLATES};
    use crate::render::Raster;
    use crate::tests::command::on_device::each_device;
    use crate::tests::draw::on_device::{library, pass, read, run, state, target, Buffers};
    use crate::{
        Buffer, ColorAttachmentDescriptor, ComputePipelineDescriptor, ComputePipelineState,
        CullMode, Device, DispatchType, DrawKind, Error, FunctionDescriptor, Gap, IndexBuffer,
        IndexType, IndirectCommandBuffer, IndirectCommandBufferDescriptor, IndirectCommandType,
        IndirectCommandsKind, Library, MemoryClass, PixelFormat, PrimitiveType, Refusal,
        RenderCommandEncoder, RenderPipelineDescriptor, RenderPipelineKind, RenderPipelineState,
        Resource, ResourceUsage, StageSource, TextureUsage, VisibilityResultMode,
    };
    use ash::vk;
    use metal2vulkan::metal::VertexFormat;
    use metal2vulkan::pipeline::{
        IndirectDrawLayout, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
        VertexDescriptor,
    };
    use metal2vulkan_devtools::engine_dir;
    use std::time::{Duration, Instant};

    /// The commands the scene's kernel encodes.
    const N: u32 = 4;
    /// Each command's mark and tint: its index plus these.
    const MARK: u32 = 10;
    const TINT: u32 = 20;

    fn ok<T>(name: &str, result: Result<T, Error>) -> T {
        result.unwrap_or_else(|error| panic!("{name}: {error}"))
    }

    /// Commands of indexed draws that inherit the pipeline state and bind `vertex` vertex and
    /// `fragment` fragment buffers each.
    fn descriptor(vertex: u32, fragment: u32) -> IndirectCommandBufferDescriptor {
        let mut d = IndirectCommandBufferDescriptor::new();
        d.command_types = IndirectCommandType::DRAW_INDEXED;
        d.inherit_pipeline_state = true;
        d.max_vertex_buffer_bind_count = vertex;
        d.max_fragment_buffer_bind_count = fragment;
        d
    }

    /// A buffer of `count` commands of `d`, or `None` (printed) on a device without the extension,
    /// which refuses it by name.
    fn commands(
        name: &str,
        device: &Device,
        d: &IndirectCommandBufferDescriptor,
        count: u32,
    ) -> Option<IndirectCommandBuffer> {
        match device.new_indirect_command_buffer(d, count) {
            Ok(commands) => Some(commands),
            Err(Error::Unsupported(Gap::IndirectCommandBuffers {
                kind: IndirectCommandsKind::Extension,
            })) => {
                assert!(device.core().indirect.is_none(), "{name}");
                println!("{name}: no VK_EXT_device_generated_commands: refused by name");
                None
            }
            Err(error) => panic!("{name}: {error}"),
        }
    }

    fn function(library: &Library, name: &str) -> StageSource {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("the function"),
        ))
    }

    /// A state of `vertex` and `fragment` of `library` into an `RGBA32Uint` target, made for
    /// indirect command buffers.
    fn flagged(
        name: &str,
        device: &Device,
        library: &Library,
        (vertex, fragment): (&str, &str),
    ) -> RenderPipelineState {
        let mut d = RenderPipelineDescriptor::new(function(library, vertex));
        d.fragment = Some(function(library, fragment));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        d.support_indirect_command_buffers = true;
        let made = state(name, device, &d);
        assert!(made.support_indirect_command_buffers(), "{name}");
        made
    }

    fn kernel(name: &str, device: &Device, library: &Library, entry: &str) -> ComputePipelineState {
        let d = ComputePipelineDescriptor::new(function(library, entry));
        ok(
            name,
            device.new_compute_pipeline_state(&d, |samplers| Ok(vec![0; samplers.len()])),
        )
    }

    fn words(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn floats(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The pixel corners of the quad over rows `2 * band` and `2 * band + 1` of the target.
    fn quad(band: u32) -> [f32; 8] {
        let (top, bottom) = ((2 * band) as f32, (2 * band + 2) as f32);
        [0.0, top, 8.0, top, 0.0, bottom, 8.0, bottom]
    }

    /// The scene's buffers: the argument buffer holding the commands' `gpuResourceID`, each
    /// command's draw (first index, count, instances, base vertex, base instance), mark and tint,
    /// the frame, the four quads' corners and two copies of a quad's six 16-bit indices.
    struct Scene {
        arguments: Buffer,
        draws: Buffer,
        marks: Buffer,
        frame: Buffer,
        xy: Buffer,
        tints: Buffer,
        indices: Buffer,
    }

    /// Command `i`'s first index: the second copy of the indices for an odd one.
    fn first(i: u32) -> u32 {
        6 * (i % 2)
    }

    impl Scene {
        fn new(device: &Device, buffers: &mut Buffers, commands: &IndirectCommandBuffer) -> Scene {
            Scene::of(device, buffers, commands, N)
        }

        /// The scene of `count` commands, command `i` drawing band `i % 4` with mark `MARK + i`
        /// and tint `TINT + i`.
        fn of(
            device: &Device,
            buffers: &mut Buffers,
            commands: &IndirectCommandBuffer,
            count: u32,
        ) -> Scene {
            let draws: Vec<u32> = (0..count)
                .flat_map(|i| [first(i), 6, 1, 4 * i, i])
                .collect();
            let xy: Vec<f32> = (0..count).flat_map(|i| quad(i % 4)).collect();
            let marks: Vec<u32> = (0..count).map(|i| MARK + i).collect();
            let tints: Vec<u32> = (0..count).map(|i| TINT + i).collect();
            let indices: Vec<u8> = [0u16, 1, 2, 2, 1, 3]
                .repeat(2)
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            Scene {
                arguments: buffers.of(device, &commands.gpu_resource_id().to_le_bytes()),
                draws: buffers.of(device, &words(&draws)),
                marks: buffers.of(device, &words(&marks)),
                frame: buffers.of(device, &floats(&[8.0, 8.0, 0.0, 0.0])),
                xy: buffers.of(device, &floats(&xy)),
                tints: buffers.of(device, &words(&tints)),
                indices: buffers.of(device, &indices),
            }
        }
    }

    /// `threads` threads of `kernel` (`icb_indexed_16`) encode the scene's commands into
    /// `commands`, the vertex positions those of `xy` at `xy_offset`.
    fn encode(
        name: &str,
        device: &Device,
        kernel: &ComputePipelineState,
        commands: &IndirectCommandBuffer,
        (s, xy, xy_offset): (&Scene, &Buffer, u64),
        threads: u32,
    ) {
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut e = ok(
            name,
            command_buffer.compute_command_encoder(DispatchType::Serial),
        );
        ok(name, e.set_compute_pipeline_state(kernel));
        let bound = [
            &s.arguments,
            &s.draws,
            &s.marks,
            &s.frame,
            xy,
            &s.tints,
            &s.indices,
        ];
        for (index, buffer) in bound.into_iter().enumerate() {
            let offset = if index == 4 { xy_offset } else { 0 };
            ok(name, e.set_buffer(Some(buffer), offset, index as u32));
        }
        ok(
            name,
            e.use_resource(
                Resource::IndirectCommandBuffer(commands),
                ResourceUsage::WRITE,
            ),
        );
        ok(
            name,
            e.dispatch_threads([threads, 1, 1], [threads.min(64), 1, 1]),
        );
        e.end_encoding();
        run(name, &command_buffer);
    }

    /// Command `i` drawn direct, its positions those of `xy` at `xy_offset` from base vertex
    /// `base`.
    fn direct(
        name: &str,
        e: &mut RenderCommandEncoder,
        (s, xy, xy_offset): (&Scene, &Buffer, u64),
        i: u32,
        base: i32,
    ) {
        ok(
            name,
            e.set_vertex_buffer(Some(&s.marks), 4 * u64::from(i), 0),
        );
        ok(name, e.set_vertex_buffer(Some(&s.frame), 0, 1));
        ok(name, e.set_vertex_buffer(Some(xy), xy_offset, 2));
        ok(
            name,
            e.set_fragment_buffer(Some(&s.tints), 4 * u64::from(i), 0),
        );
        let indices = IndexBuffer {
            buffer: &s.indices,
            offset: 2 * u64::from(first(i)),
            index_type: IndexType::UInt16,
        };
        ok(
            name,
            e.draw_indexed_primitives(PrimitiveType::Triangle, 6, indices, 1, base, i),
        );
    }

    /// The `RGBA32Uint` texels a pass of `state` that `body` records draws, from opaque black.
    fn drawn(
        name: &str,
        device: &Device,
        state: &RenderPipelineState,
        body: impl FnOnce(&mut RenderCommandEncoder),
    ) -> Vec<u8> {
        let color = target(device, PixelFormat::RGBA32Uint, TextureUsage::RENDER_TARGET);
        let queue = device.new_command_queue(4).expect("a queue");
        let command_buffer = queue.command_buffer().expect("a command buffer");
        let mut e = ok(
            name,
            command_buffer.render_command_encoder(&pass(&color, None)),
        );
        ok(name, e.set_render_pipeline_state(state));
        e.set_cull_mode(CullMode::None);
        body(&mut e);
        e.end_encoding();
        run(name, &command_buffer);
        read(&color, 16)
    }

    /// The mark of each row of `texels`, every pixel of a row holding the same mark (its first
    /// word; the second is the provoking vertex's `vid`, which differs between a quad's two
    /// triangles).
    fn marks(name: &str, texels: &[u8]) -> Vec<u32> {
        let mark = |pixel: &[u8]| u32::from_le_bytes(pixel[..4].try_into().expect("a word"));
        (texels.chunks(16 * 8).enumerate())
            .map(|(row, pixels)| {
                assert!(
                    pixels.chunks(16).all(|p| mark(p) == mark(pixels)),
                    "{name}: row {row} holds one mark"
                );
                mark(pixels)
            })
            .collect()
    }

    /// The marks of rows whose commands in `drawn` drew, 0 where they did not.
    fn want(drawn: &[u32]) -> Vec<u32> {
        (0..2 * N)
            .map(|row| {
                let i = row / 2;
                if drawn.contains(&i) {
                    MARK + i
                } else {
                    0
                }
            })
            .collect()
    }

    #[test]
    fn every_device_draws_encoded_commands_as_their_draws_direct() {
        let Some(indexed) = library(&engine_dir().join("icb_merged_indexed.ll")) else {
            return;
        };
        each_device("icb_encoded_draws", |name, _, device| {
            let Some(commands) = commands(name, &device, &descriptor(3, 1), N) else {
                return;
            };
            let state = flagged(
                name,
                &device,
                &indexed,
                ("icb_indexed_vertex", "icb_indexed_fragment"),
            );
            let kernel = kernel(name, &device, &indexed, "icb_indexed_16");
            let mut buffers = Buffers::new(&device);
            let s = Scene::new(&device, &mut buffers, &commands);
            let at = (&s, &s.xy, 0);
            encode(name, &device, &kernel, &commands, at, N);
            let direct_of = |range: std::ops::Range<u32>| {
                drawn(name, &device, &state, |e| {
                    for i in range {
                        direct(name, e, at, i, 4 * i as i32);
                    }
                })
            };
            let all = drawn(name, &device, &state, |e| {
                ok(name, e.execute_commands_in_buffer(&commands, 0..N))
            });
            assert_eq!(all, direct_of(0..N), "{name}: all four");
            assert_eq!(marks(name, &all), want(&[0, 1, 2, 3]), "{name}");
            let middle = drawn(name, &device, &state, |e| {
                ok(name, e.execute_commands_in_buffer(&commands, 1..3))
            });
            assert_eq!(middle, direct_of(1..3), "{name}: commands 1 and 2");
            assert_eq!(marks(name, &middle), want(&[1, 2]), "{name}");
            // The range a buffer holds 8 bytes in: location 2, length 2.
            let range = buffers.of(&device, &words(&[7, 7, 2, 2]));
            let held = drawn(name, &device, &state, |e| {
                ok(
                    name,
                    e.execute_commands_in_buffer_with_range(&commands, &range, 8),
                )
            });
            assert_eq!(held, direct_of(2..N), "{name}: the range the buffer holds");
            assert_eq!(marks(name, &held), want(&[2, 3]), "{name}");
            // A direct draw between two executions pushes and binds its own.
            let mixed = drawn(name, &device, &state, |e| {
                ok(name, e.execute_commands_in_buffer(&commands, 0..2));
                direct(name, e, at, 2, 8);
                ok(name, e.execute_commands_in_buffer(&commands, 3..N));
            });
            assert_eq!(mixed, all, "{name}: a direct draw after an execution");
            drawn(name, &device, &state, |e| {
                match e.execute_commands_in_buffer(&commands, 3..N + 1) {
                    Err(Error::Refused(Refusal::CommandRange)) => {}
                    other => panic!("{name}: a range past the commands: {other:?}"),
                }
                match e.execute_commands_in_buffer_with_range(&commands, &range, 6) {
                    Err(Error::Refused(Refusal::CommandRange)) => {}
                    other => panic!("{name}: a range buffer offset of 6: {other:?}"),
                }
            });
            println!(
                "{name}: four encoded commands, commands 1 and 2, the range 2 and 3 a buffer \
                 holds, and executions around a direct draw each draw as their draws direct"
            );
        });
    }

    /// An execution of more commands than one part holds runs its parts in order, each command
    /// tested against the range as its own, against the same draws direct: whole, over a CPU range
    /// across three parts, and over ranges a buffer holds, which the count kernel reads: one at
    /// location 0 across two parts, one from a location inside a part (the part's commands before
    /// it run and draw nothing), and one past the commands, clamped to them.
    #[test]
    fn every_device_draws_an_execution_of_several_parts_as_its_draws_direct() {
        let Some(indexed) = library(&engine_dir().join("icb_merged_indexed.ll")) else {
            return;
        };
        each_device("icb_parts", |name, _, device| {
            let state = flagged(
                name,
                &device,
                &indexed,
                ("icb_indexed_vertex", "icb_indexed_fragment"),
            );
            // The device's part for this state, from a buffer of as many commands as it takes up
            // to 2^16.
            let most = (device.core().indirect.as_ref()).map_or(1, |i| i.limits.sequences);
            let Some(sized) = commands(name, &device, &descriptor(3, 1), most.min(1 << 16)) else {
                return;
            };
            let mut part = 0;
            drawn(name, &device, &state, |e| {
                part = ok(name, e.execution_part(&sized));
            });
            drop(sized);
            let many = (3 * part + 8).min(most.min(1 << 16));
            let commands = commands(name, &device, &descriptor(3, 1), many).expect("a buffer");
            let kernel = kernel(name, &device, &indexed, "icb_indexed_16");
            let mut buffers = Buffers::new(&device);
            let s = Scene::of(&device, &mut buffers, &commands, many);
            let at = (&s, &s.xy, 0);
            encode(name, &device, &kernel, &commands, at, many);
            let direct_of = |range: std::ops::Range<u32>| {
                drawn(name, &device, &state, |e| {
                    for i in range {
                        direct(name, e, at, i, 4 * i as i32);
                    }
                })
            };
            // Each band's mark is the last command of `range` that draws it.
            let last = |range: std::ops::Range<u32>| -> Vec<u32> {
                (0..8)
                    .map(|row| {
                        let band = row / 2;
                        range
                            .clone()
                            .rev()
                            .find(|i| i % 4 == band)
                            .map_or(0, |i| MARK + i)
                    })
                    .collect()
            };
            let check = |what: &str, texels: Vec<u8>, range: std::ops::Range<u32>| {
                assert_eq!(texels, direct_of(range.clone()), "{name}: {what}");
                assert_eq!(marks(name, &texels), last(range), "{name}: {what}");
            };
            let all = drawn(name, &device, &state, |e| {
                ok(name, e.execute_commands_in_buffer(&commands, 0..many))
            });
            check("all", all, 0..many);
            let across = part / 2..(2 * part + part / 2).min(many);
            let cpu = drawn(name, &device, &state, |e| {
                ok(
                    name,
                    e.execute_commands_in_buffer(&commands, across.clone()),
                )
            });
            check("a CPU range across three parts", cpu, across);
            let mut held = |location: u32, length: u32| {
                let range = buffers.of(&device, &words(&[location, length]));
                drawn(name, &device, &state, |e| {
                    ok(
                        name,
                        e.execute_commands_in_buffer_with_range(&commands, &range, 0),
                    )
                })
            };
            check(
                "a held range at location 0 across two parts",
                held(0, part + 5),
                0..part + 5,
            );
            let inside = part + part / 2;
            check(
                "a held range from inside the second part",
                held(inside, part),
                inside..(inside + part).min(many),
            );
            check(
                "a held range past the commands",
                held(many - 6, 100),
                many - 6..many,
            );
            println!(
                "{name}: {many} encoded commands in parts of {part}: all, a CPU range across three \
                 parts and ranges a buffer holds (at 0, inside the second part, past the \
                 commands) each draw as their draws direct"
            );
        });
    }

    /// An execution over a range a buffer holds, as a `Boolean` mode's first draw, runs and its
    /// samples count: its count kernel ends the rendering before the mode's query begins. After a
    /// draw of a mode, whose query then counts, it is `DrawKind::VisibilityResultInterrupt`,
    /// counted, and the encoder draws on.
    #[test]
    fn every_device_runs_a_held_range_first_in_a_visibility_mode_and_refuses_it_after_a_draw() {
        let Some(indexed) = library(&engine_dir().join("icb_merged_indexed.ll")) else {
            return;
        };
        each_device("icb_visibility", |name, _, device| {
            let Some(commands) = commands(name, &device, &descriptor(3, 1), N) else {
                return;
            };
            let state = flagged(
                name,
                &device,
                &indexed,
                ("icb_indexed_vertex", "icb_indexed_fragment"),
            );
            let kernel = kernel(name, &device, &indexed, "icb_indexed_16");
            let mut buffers = Buffers::new(&device);
            let s = Scene::new(&device, &mut buffers, &commands);
            let at = (&s, &s.xy, 0);
            encode(name, &device, &kernel, &commands, at, N);
            let range = buffers.of(&device, &words(&[0, N]));
            let visibility = buffers.of(&device, &[0xaa; 16]);
            let counters = device.counters();
            let before = counters.metrics().visibility_interrupts_refused;
            let color = target(
                &device,
                PixelFormat::RGBA32Uint,
                TextureUsage::RENDER_TARGET,
            );
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut d = pass(&color, None);
            d.visibility_result_buffer = Some(visibility.clone());
            let mut e = ok(name, command_buffer.render_command_encoder(&d));
            ok(name, e.set_render_pipeline_state(&state));
            e.set_cull_mode(CullMode::None);
            let boolean = VisibilityResultMode::Boolean;
            ok(name, e.set_visibility_result_mode(boolean, 0));
            ok(
                name,
                e.execute_commands_in_buffer_with_range(&commands, &range, 0),
            );
            ok(name, e.set_visibility_result_mode(boolean, 8));
            direct(name, &mut e, at, 0, 0);
            let refused = e.execute_commands_in_buffer_with_range(&commands, &range, 0);
            assert!(
                matches!(
                    refused,
                    Err(Error::Unsupported(Gap::Draw {
                        kind: DrawKind::VisibilityResultInterrupt
                    }))
                ),
                "{name}: an execution while the query counts: {refused:?}"
            );
            direct(name, &mut e, at, 1, 4);
            e.end_encoding();
            run(name, &command_buffer);
            let refusals = counters.metrics().visibility_interrupts_refused - before;
            assert_eq!(refusals, 1, "{name}: the refusal counted");
            let contents = visibility.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds 16 bytes, which the finished command buffer wrote and whose
            // writes reach the host.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 16) };
            let counts: Vec<u64> = (bytes.as_chunks::<8>().0.iter())
                .map(|word| u64::from_le_bytes(*word))
                .collect();
            // Each command draws a band of 16 pixels: the execution's four, the two direct draws'.
            assert_eq!(counts, [64, 32], "{name}: each mode's samples");
            println!(
                "{name}: an execution over a held range as a Boolean mode's first draw counted 64 \
                 samples; after a draw of the next mode it was refused and counted"
            );
        });
    }

    #[test]
    fn every_device_resets_commands_by_kernel_and_by_blit() {
        let Some(indexed) = library(&engine_dir().join("icb_merged_indexed.ll")) else {
            return;
        };
        let Some(extras) = library(&engine_dir().join("icb_indexed_extras.ll")) else {
            return;
        };
        each_device("icb_resets", |name, _, device| {
            let Some(commands) = commands(name, &device, &descriptor(3, 1), N) else {
                return;
            };
            let state = flagged(
                name,
                &device,
                &indexed,
                ("icb_indexed_vertex", "icb_indexed_fragment"),
            );
            let encoder = kernel(name, &device, &indexed, "icb_indexed_16");
            let reset = kernel(name, &device, &extras, "icb_extras_reset");
            let mut buffers = Buffers::new(&device);
            let s = Scene::new(&device, &mut buffers, &commands);
            let at = (&s, &s.xy, 0);
            encode(name, &device, &encoder, &commands, at, N);
            let first = buffers.of(&device, &words(&[1]));
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut e = ok(
                name,
                command_buffer.compute_command_encoder(DispatchType::Serial),
            );
            ok(name, e.set_compute_pipeline_state(&reset));
            ok(name, e.set_buffer(Some(&s.arguments), 0, 0));
            ok(name, e.set_buffer(Some(&first), 0, 1));
            ok(name, e.dispatch_threadgroups([1, 1, 1], [1, 1, 1]));
            e.end_encoding();
            let mut blit = ok(name, command_buffer.blit_command_encoder());
            ok(name, blit.reset_commands_in_buffer(&commands, 3..N));
            ok(name, blit.optimize_indirect_command_buffer(&commands, 0..N));
            match blit.reset_commands_in_buffer(&commands, 3..N + 1) {
                Err(Error::Refused(Refusal::CommandRange)) => {}
                other => panic!("{name}: a reset past the commands: {other:?}"),
            }
            blit.end_encoding();
            run(name, &command_buffer);
            let after = drawn(name, &device, &state, |e| {
                ok(name, e.execute_commands_in_buffer(&commands, 0..N))
            });
            let direct = drawn(name, &device, &state, |e| {
                for i in [0, 2] {
                    direct(name, e, at, i, 4 * i as i32);
                }
            });
            assert_eq!(after, direct, "{name}: commands 0 and 2");
            assert_eq!(marks(name, &after), want(&[0, 2]), "{name}");
            println!(
                "{name}: a kernel's reset of command 1 and a blit's of command 3 leave 0 and 2"
            );
        });
    }

    /// Two buffers over one block's bytes: `whole` of `bytes`, and the first `part` bytes of it.
    fn aliased(device: &Device, bytes: &[u8], part: u64) -> (crate::Block, Buffer, Buffer) {
        let block = (device.take_block(MemoryClass::Shared, 65536)).expect("a block");
        let len = bytes.len() as u64;
        let whole = (device.new_buffer(block.id, 0, len, None)).expect("a buffer");
        let first = (device.new_buffer(block.id, 0, part, None)).expect("a buffer");
        let contents = whole.contents().expect("a Shared buffer");
        // SAFETY: the range lies in the buffer, which is alive, and no GPU work uses it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        (block, whole, first)
    }

    /// How the commands that draw nothing in a part are made ([`empties_in_a_part`]).
    #[derive(Clone, Copy, Debug)]
    enum Empty {
        /// A blit's reset of the encoded commands.
        Reset,
        /// Encoded with 0 instances and their indices.
        NoInstances,
        /// Encoded with their instances and 0 indices.
        NoIndices,
    }

    /// One cell of the parts' empties: the kernel encodes 64 commands, each drawing band `i % 4`
    /// with its own mark, those of `empties` made by `empty` to draw nothing; one execution of
    /// `executed` (at most a part) leaves each band with the mark of its last command that
    /// draws, or none. Each cell is a test of its own, one process each, so a device fault stops
    /// at its cell. On NVIDIA's driver (620.70.06, an RTX 2080) runs of slot words in one push
    /// constant token lost the draws after leading empties (`crate::indirect::Tokens`).
    fn empties_in_a_part(
        tag: &str,
        empty: Empty,
        empties: std::ops::Range<u32>,
        executed: std::ops::Range<u32>,
    ) {
        const COUNT: u32 = 64;
        let Some(indexed) = library(&engine_dir().join("icb_merged_indexed.ll")) else {
            return;
        };
        each_device(tag, |name, _, device| {
            let Some(commands) = commands(name, &device, &descriptor(3, 1), COUNT) else {
                return;
            };
            let state = flagged(
                name,
                &device,
                &indexed,
                ("icb_indexed_vertex", "icb_indexed_fragment"),
            );
            let kernel = kernel(name, &device, &indexed, "icb_indexed_16");
            let mut buffers = Buffers::new(&device);
            let mut s = Scene::of(&device, &mut buffers, &commands, COUNT);
            if !matches!(empty, Empty::Reset) {
                let draws: Vec<u32> = (0..COUNT)
                    .flat_map(|i| {
                        let inside = empties.contains(&i);
                        let count = match (inside, empty) {
                            (true, Empty::NoIndices) => 0,
                            _ => 6,
                        };
                        let instances = match (inside, empty) {
                            (true, Empty::NoInstances) => 0,
                            _ => 1,
                        };
                        [first(i), count, instances, 4 * i, i]
                    })
                    .collect();
                s.draws = buffers.of(&device, &words(&draws));
            }
            encode(name, &device, &kernel, &commands, (&s, &s.xy, 0), COUNT);
            if matches!(empty, Empty::Reset) && !empties.is_empty() {
                let queue = device.new_command_queue(4).expect("a queue");
                let command_buffer = queue.command_buffer().expect("a command buffer");
                let mut blit = ok(name, command_buffer.blit_command_encoder());
                ok(
                    name,
                    blit.reset_commands_in_buffer(&commands, empties.clone()),
                );
                blit.end_encoding();
                run(name, &command_buffer);
            }
            let texels = drawn(name, &device, &state, |e| {
                ok(
                    name,
                    e.execute_commands_in_buffer(&commands, executed.clone()),
                )
            });
            let got = marks(name, &texels);
            let want: Vec<u32> = (0..8)
                .map(|row| {
                    (executed.clone().rev())
                        .find(|i| i % 4 == row / 2 && !empties.contains(i))
                        .map_or(0, |i| MARK + i)
                })
                .collect();
            let bands = |m: &[u32]| -> Vec<i64> {
                (m.iter().step_by(2))
                    .map(|&v| i64::from(v) - i64::from(MARK))
                    .collect()
            };
            let verdict = if got == want { "right" } else { "WRONG" };
            println!(
                "MATRIX {name}: {empty:?} {empties:?}, executed {executed:?}: {verdict}; bands {:?}, \
                 want {:?}",
                bands(&got),
                bands(&want)
            );
            assert_eq!(
                got, want,
                "{name}: {empty:?} {empties:?}, executed {executed:?}"
            );
        });
    }

    #[test]
    fn every_device_draws_a_part_of_no_empties() {
        empties_in_a_part("icb_empties_none", Empty::Reset, 0..0, 0..64);
    }

    #[test]
    fn every_device_draws_a_part_after_its_first_half_reset() {
        empties_in_a_part("icb_empties_reset_0_31", Empty::Reset, 0..32, 0..64);
    }

    #[test]
    fn every_device_draws_a_part_after_its_first_quarter_reset() {
        empties_in_a_part("icb_empties_reset_0_15", Empty::Reset, 0..16, 0..64);
    }

    #[test]
    fn every_device_draws_a_part_after_its_first_command_reset() {
        empties_in_a_part("icb_empties_reset_0", Empty::Reset, 0..1, 0..64);
    }

    #[test]
    fn every_device_draws_a_part_around_its_second_quarter_reset() {
        empties_in_a_part("icb_empties_reset_16_31", Empty::Reset, 16..32, 0..64);
    }

    #[test]
    fn every_device_draws_a_part_before_its_second_half_reset() {
        empties_in_a_part("icb_empties_reset_32_63", Empty::Reset, 32..64, 0..64);
    }

    #[test]
    fn every_device_draws_a_part_of_32_after_its_first_half_reset() {
        empties_in_a_part("icb_empties_part_32", Empty::Reset, 0..16, 0..32);
    }

    #[test]
    fn every_device_draws_a_part_of_16_after_its_first_half_reset() {
        empties_in_a_part("icb_empties_part_16", Empty::Reset, 0..8, 0..16);
    }

    #[test]
    fn every_device_draws_a_part_after_commands_of_no_instances() {
        empties_in_a_part("icb_empties_no_instances", Empty::NoInstances, 0..32, 0..64);
    }

    #[test]
    fn every_device_draws_a_part_after_commands_of_no_indices() {
        empties_in_a_part("icb_empties_no_indices", Empty::NoIndices, 0..32, 0..64);
    }

    #[test]
    fn every_device_reads_a_commands_slots_unchecked_on_both_sides() {
        let Some(indexed) = library(&engine_dir().join("icb_merged_indexed.ll")) else {
            return;
        };
        let Some(extras) = library(&engine_dir().join("icb_indexed_extras.ll")) else {
            return;
        };
        each_device("icb_unchecked_slots", |name, _, device| {
            let Some(before) = commands(name, &device, &descriptor(3, 1), N) else {
                return;
            };
            let after = (commands(name, &device, &descriptor(3, 1), N)).expect("a second buffer");
            let kernel = kernel(name, &device, &indexed, "icb_indexed_16");
            let mut buffers = Buffers::new(&device);
            // Command 0's quad, then corners far outside the target.
            let far = [1000.0f32; 8];
            let quad_then_far = floats(&[quad(0), far].concat());
            let far_then_quad = floats(&[far, quad(0)].concat());

            // Before: buffer 2 starts 32 bytes into the corners, and the vertex function reads
            // four elements before it, which the buffer holds.
            let back = flagged(
                name,
                &device,
                &extras,
                ("icb_extras_back_vertex", "icb_extras_back_fragment"),
            );
            let s = Scene::new(&device, &mut buffers, &before);
            let corners = buffers.of(&device, &quad_then_far);
            encode(name, &device, &kernel, &before, (&s, &corners, 32), 1);
            let encoded = drawn(name, &device, &back, |e| {
                ok(name, e.execute_commands_in_buffer(&before, 0..1))
            });
            let direct_back = drawn(name, &device, &back, |e| {
                direct(name, e, (&s, &corners, 32), 0, 0)
            });
            assert_eq!(
                encoded, direct_back,
                "{name}: a read before the slot's address"
            );
            assert_eq!(marks(name, &encoded), want(&[0]), "{name}: before");

            // Past: buffer 2 is the first 32 bytes of the corners, and the draw's base vertex of
            // 4 reads the 32 bytes past its end, which the block holds.
            let state = flagged(
                name,
                &device,
                &indexed,
                ("icb_indexed_vertex", "icb_indexed_fragment"),
            );
            let s = Scene::new(&device, &mut buffers, &after);
            let (_block, whole, part) = aliased(&device, &far_then_quad, 32);
            // The kernel's draw 0 takes base vertex 4.
            let draws = buffers.of(&device, &words(&[0, 6, 1, 4, 0]));
            let s = Scene { draws, ..s };
            encode(name, &device, &kernel, &after, (&s, &part, 0), 1);
            let encoded = drawn(name, &device, &state, |e| {
                ok(name, e.execute_commands_in_buffer(&after, 0..1))
            });
            let direct_past = drawn(name, &device, &state, |e| {
                direct(name, e, (&s, &whole, 0), 0, 4)
            });
            assert_eq!(
                encoded, direct_past,
                "{name}: a read past the slot's buffer"
            );
            assert_eq!(marks(name, &encoded), want(&[0]), "{name}: past");
            println!(
                "{name}: a command's slot reads the buffer's bytes 32 before its address and \
                 32 past the end of the buffer it named, as the draws direct do"
            );
        });
    }

    #[test]
    fn every_device_refuses_a_state_of_more_tokens_than_it_takes() {
        let Some(extras) = library(&engine_dir().join("icb_indexed_extras.ll")) else {
            return;
        };
        each_device("icb_tokens", |name, _, device| {
            let Some(commands) = commands(name, &device, &descriptor(27, 0), 1) else {
                return;
            };
            let wide = flagged(
                name,
                &device,
                &extras,
                ("icb_extras_wide_vertex", "icb_extras_fragment"),
            );
            let tokens = (device.core().indirect.as_ref())
                .expect("the extension")
                .limits
                .tokens;
            drawn(name, &device, &wide, |e| {
                match (e.execute_commands_in_buffer(&commands, 0..1), tokens < 17) {
                    (
                        Err(Error::Unsupported(Gap::IndirectCommandBuffers {
                            kind: IndirectCommandsKind::Tokens,
                        })),
                        true,
                    )
                    | (Ok(()), false) => {}
                    (other, _) => panic!("{name}: 17 tokens on a device of {tokens}: {other:?}"),
                }
            });
            println!("{name}: a state of 17 tokens on a device that takes {tokens}");
        });
    }

    /// A state made for indirect command buffers fetches its vertex function's `[[stage_in]]`
    /// attributes in the shader, as a command binds its vertex buffers by address alone
    /// (`icb_render_stride_template`, rasterization disabled): a direct draw of vertices 1 to 3
    /// reads each vertex's word at the layout's stride of 8 bytes past its buffer's bind offset of
    /// 4, and writes it with its vertex id to `out[vid]`. The same state with its layout at
    /// `MTLBufferLayoutStrideDynamic` is refused by name: its fetch reads the stride each bind
    /// sets from a push field the encoder does not fill.
    #[test]
    fn every_device_pulls_a_flagged_states_stage_in_and_refuses_a_stride_per_bind() {
        let Some(fetched) = library(&engine_dir().join("icb_render_stride_template.ll")) else {
            return;
        };
        let descriptor = |stride| {
            let mut d =
                RenderPipelineDescriptor::new(function(&fetched, "icb_render_stride_template"));
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
            d.rasterization_enabled = false;
            d.support_indirect_command_buffers = true;
            let mut vertex = VertexDescriptor::new();
            let attribute = VertexAttributeDescriptor::new(VertexFormat::UInt, 0, 1);
            vertex.attributes.insert(0, attribute);
            vertex
                .layouts
                .insert(1, VertexBufferLayoutDescriptor::new(stride));
            d.vertex_descriptor = Some(vertex);
            d
        };
        // Word 1 + 2v holds 0xb0 + v, a filler between: a stride of 4 or no offset reads a filler.
        let mut data = [0x5555_5555u32; 16];
        for v in 0..6 {
            data[1 + 2 * v] = 0xb0 + v as u32;
        }
        each_device("icb_stage_in", |name, _, device| {
            let refused = (device
                .new_render_pipeline_state(&descriptor(None), crate::tests::no_constexpr_samplers))
            .err();
            assert!(
                matches!(
                    refused,
                    Some(Error::Unsupported(Gap::RenderPipeline {
                        kind: RenderPipelineKind::DrawInput
                    }))
                ),
                "{name}: a stride per bind: {refused:?}"
            );
            let state = state(name, &device, &descriptor(Some(8)));
            assert!(state.support_indirect_command_buffers(), "{name}");
            let mut buffers = Buffers::new(&device);
            let attributes = buffers.of(&device, &words(&data));
            let out = buffers.of(&device, &[0u8; 8 * 8]);
            drawn(name, &device, &state, |e| {
                ok(name, e.set_vertex_buffer(Some(&out), 0, 0));
                ok(name, e.set_vertex_buffer(Some(&attributes), 4, 1));
                ok(name, e.draw_primitives(PrimitiveType::Triangle, 1, 3, 1, 0));
            });
            let contents = out.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds 16 words, which the finished draw wrote and whose writes
            // reach the host.
            let got = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), 16) };
            let mut want = [0u32; 16];
            for v in 1..4u32 {
                want[2 * v as usize..2 * v as usize + 2].copy_from_slice(&[0xb0 + v, v]);
            }
            assert_eq!(got, want, "{name}: each vertex's word and id");
            println!("{name}: an attribute pulled at its stride; a stride per bind refused");
        });
    }

    #[test]
    fn every_device_bounds_its_indirect_caches() {
        let Some(indexed) = library(&engine_dir().join("icb_merged_indexed.ll")) else {
            return;
        };
        each_device("icb_caches", |name, _, device| {
            let Some(_commands) = commands(name, &device, &descriptor(3, 1), 1) else {
                return;
            };
            let state = flagged(
                name,
                &device,
                &indexed,
                ("icb_indexed_vertex", "icb_indexed_fragment"),
            );
            let core = device.core();
            let indirect = (core.indirect.as_ref()).expect("the extension");
            let compiled = &state.inner.compiled;
            let pipeline = ok(
                name,
                compiled.pipeline(vk::PrimitiveTopology::TRIANGLE_LIST, Raster::default()),
            );
            let layout =
                |vertex, fragment| IndirectDrawLayout::new(vertex, fragment).expect("a layout");
            // A pipeline of another raster state sizes its own shape (the preprocess requirements
            // are queried with the execution's pipeline).
            let clamped = Raster {
                clamp: true,
                lines: false,
            };
            let other = ok(
                name,
                compiled.pipeline(vk::PrimitiveTopology::TRIANGLE_LIST, clamped),
            );
            let before = indirect.held().1;
            ok(
                name,
                indirect.shape(
                    core,
                    (compiled, Raster::default(), pipeline),
                    layout(3, 1),
                    1,
                ),
            );
            ok(
                name,
                indirect.shape(core, (compiled, clamped, other), layout(3, 1), 1),
            );
            assert_eq!(
                indirect.held().1,
                before + 2,
                "{name}: a shape for each raster state"
            );
            for vertex in 0..=TEMPLATES as u32 {
                ok(name, indirect.template(core, layout(vertex, 0)));
            }
            for capacity in 1..=SHAPES as u32 + 1 {
                ok(
                    name,
                    indirect.shape(
                        core,
                        (compiled, Raster::default(), pipeline),
                        layout(3, 1),
                        capacity,
                    ),
                );
            }
            let layouts =
                (3..=31).flat_map(|vertex| (1..=3).map(move |fragment| (vertex, fragment)));
            for (vertex, fragment) in layouts.take(LAYOUTS + 1) {
                ok(
                    name,
                    indirect.shape(
                        core,
                        (compiled, Raster::default(), pipeline),
                        layout(vertex, fragment),
                        1,
                    ),
                );
            }
            assert_eq!(indirect.held(), (TEMPLATES, SHAPES, LAYOUTS), "{name}");
            println!("{name}: {TEMPLATES} templates, {SHAPES} shapes and {LAYOUTS} layouts held past them");
        });
    }

    /// The part's bound, checked on the device: at the widest layout the device can take (as many
    /// 8-byte slot words as its tokens allow, at most the push bytes less the word of the sequence
    /// index and the part's first command), the part of an execution of the most commands the
    /// device takes has queried preprocess bytes at or under `PART_BYTES`.
    #[test]
    fn every_device_keeps_the_part_of_its_widest_layout_under_the_bound() {
        let Some(indexed) = library(&engine_dir().join("icb_merged_indexed.ll")) else {
            return;
        };
        each_device("icb_part_bytes", |name, _, device| {
            let Some(_commands) = commands(name, &device, &descriptor(3, 1), 1) else {
                return;
            };
            let state = flagged(
                name,
                &device,
                &indexed,
                ("icb_indexed_vertex", "icb_indexed_fragment"),
            );
            let core = device.core();
            let indirect = (core.indirect.as_ref()).expect("the extension");
            let pipeline = ok(
                name,
                (state.inner.compiled)
                    .pipeline(vk::PrimitiveTopology::TRIANGLE_LIST, Raster::default()),
            );
            let layout = IndirectDrawLayout::new(31, 31).expect("a layout");
            let room = crate::pipeline::PUSH_BYTES / 8 - 1;
            let words = room.min(indirect.limits.tokens.saturating_sub(3));
            let slots = (0..31)
                .map(|slot| layout.vertex_slot(slot))
                .chain((0..31).map(|slot| layout.fragment_slot(slot)));
            let tokens = Tokens {
                layout,
                pushes: slots
                    .take(words as usize)
                    .zip((0..).map(|w| 8 * w))
                    .collect(),
                sequence: 8 * room,
                base: 8 * room + 4,
            };
            let commands = ok(name, indirect.layout(core, &tokens));
            let most = indirect.limits.sequences;
            let (part, need) = indirect.part(core, &commands, pipeline, most);
            assert!(
                part >= 1 && need.size <= PART_BYTES,
                "{name}: {} tokens: a part of {part} commands takes {} preprocess bytes",
                tokens.count(),
                need.size
            );
            println!(
                "{name}: {} tokens: a part of {part} of the most {most} commands takes {} \
                 preprocess bytes, at most {PART_BYTES}",
                tokens.count(),
                need.size
            );
        });
    }

    #[test]
    fn every_device_bounds_its_preprocess_pool_and_drops_it() {
        let Some(indexed) = library(&engine_dir().join("icb_merged_indexed.ll")) else {
            return;
        };
        each_device("icb_pool", |name, _, device| {
            let Some(commands) = commands(name, &device, &descriptor(3, 1), N) else {
                return;
            };
            let counters = device.counters();
            let state = flagged(
                name,
                &device,
                &indexed,
                ("icb_indexed_vertex", "icb_indexed_fragment"),
            );
            {
                let core = device.core();
                let indirect = (core.indirect.as_ref()).expect("the extension");
                let compiled = &state.inner.compiled;
                let pipeline = ok(
                    name,
                    compiled.pipeline(vk::PrimitiveTopology::TRIANGLE_LIST, Raster::default()),
                );
                let layout = IndirectDrawLayout::new(3, 1).expect("a layout");
                let shape = |capacity| {
                    ok(
                        name,
                        indirect.shape(
                            core,
                            (compiled, Raster::default(), pipeline),
                            layout,
                            capacity,
                        ),
                    )
                };
                // Regions of about 8 MiB, so a few dozen pass the bound: parts of a shape of 1024
                // commands, each at most `PART_BYTES`.
                let shape = shape(1024);
                let parts = ((8 << 20) / shape.part_stride().max(1)).max(1) as u32;
                let mut held = Vec::new();
                while counters.gauges().preprocess_used_bytes <= IDLE_BYTES {
                    held.push(ok(name, indirect.executed(core, &shape, parts)).0);
                }
                let regions = held.len();
                drop(held);
                // The regions past the bound go on the block-maker thread, their bytes idle until
                // then.
                let settled = |done: &dyn Fn(u64) -> bool| {
                    let deadline = Instant::now() + Duration::from_secs(5);
                    loop {
                        let idle = counters.gauges().preprocess_idle_bytes;
                        if done(idle) || Instant::now() > deadline {
                            return idle;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                };
                let kept = settled(&|idle| idle <= IDLE_BYTES);
                assert_eq!(counters.gauges().preprocess_used_bytes, 0, "{name}");
                assert!(0 < kept && kept <= IDLE_BYTES, "{name}: {kept} idle bytes");
                // Those freed past the bound are freed ranges of a lost device's report.
                let freed = (core.spans.freed_kept().iter())
                    .filter(|range| range.kind == crate::RangeKind::Preprocess)
                    .count();
                assert!(
                    freed > 0,
                    "{name}: no preprocess region among the freed ranges"
                );
                let before = counters.metrics().preprocess_regions_remade;
                let again: Vec<_> = (0..regions)
                    .map(|_| ok(name, indirect.executed(core, &shape, parts)).0)
                    .collect();
                let remade = counters.metrics().preprocess_regions_remade - before;
                assert!(remade > 0, "{name}: regions made again past the bound");
                drop(again);
                let since = Instant::now();
                let rest = settled(&|idle| idle == 0);
                assert_eq!(rest, 0, "{name}: the pool at rest");
                println!(
                    "{name}: {regions} regions of {parts} parts held past {IDLE_BYTES} bytes \
                     went back, {kept} bytes kept idle; {remade} made again past the bound; idle \
                     bytes 0 {} ms after the last went back",
                    since.elapsed().as_millis()
                );
            }
            let mut buffers = Buffers::new(&device);
            let kernel = kernel(name, &device, &indexed, "icb_indexed_16");
            let s = Scene::new(&device, &mut buffers, &commands);
            encode(name, &device, &kernel, &commands, (&s, &s.xy, 0), N);
            let regions = || {
                let g = counters.gauges();
                (
                    g.preprocess_used_bytes + g.preprocess_idle_bytes,
                    g.indirect_command_buffers,
                )
            };
            let mut at = Vec::new();
            for iteration in 1..=200 {
                drawn(name, &device, &state, |e| {
                    ok(name, e.execute_commands_in_buffer(&commands, 0..N))
                });
                if iteration == 100 || iteration == 200 {
                    at.push(regions());
                }
            }
            assert_eq!(
                at[0], at[1],
                "{name}: the regions' bytes and the buffers, flat"
            );
            drop((commands, state, kernel, s, buffers));
            drop(device);
            let g = counters.gauges();
            assert_eq!(
                (
                    g.indirect_command_buffers,
                    g.preprocess_used_bytes,
                    g.preprocess_idle_bytes
                ),
                (0, 0, 0),
                "{name}: at the device's drop"
            );
            println!(
                "{name}: 200 executions flat between the 100th and the 200th; zero at the drop"
            );
        });
    }
}
