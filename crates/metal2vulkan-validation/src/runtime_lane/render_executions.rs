//! Executions of indirect command buffers' render commands in a render encoder
//! (`execute_commands_in_buffer`) on the device the cfg selects, under the validation layer. The
//! commands draw the owned `icb_render_replay_template` fixture's vertex function (rasterization
//! disabled: vertex `vid` of instance `iid` writes `params[0] * (vid + 1) + params[1] + 1000 * iid`
//! to word `16 * iid + vid` of `out`) in a 1x1 pass without attachments, over 256-byte windows of
//! `params` and `out`, and draw as Metal's (an M4):
//!
//! - as their records stand at commit, a CPU write after the call included; a command with no
//!   draw or no pipeline state runs nothing, and its neighbours run; the encoder's draws before
//!   and after the execution run too;
//! - an inheriting buffer's with the encoder's pipeline state and buffers as the call found them,
//!   a later binding being the encoder's own; an inherited state that is unset or made without
//!   `supportIndirectCommandBuffers` is refused at the call;
//! - an address resolves against what any encoder of the command buffer made resident, before
//!   the execution or after it, and every live buffer of its own on the device; one in a heap no
//!   encoder of the command buffer used misses, counted in the execution report;
//! - commands a kernel writes (`icb_render_replay_encode_draw_only`) are drawn by a host step,
//!   announced once;
//! - a record the GPU wrote naming a state the CPU set on none of the executed commands draws it,
//!   its `VkPipeline` made by the host step and counted;
//! - what Metal's validation stops, and a buffer of compute commands, is refused at the call;
//! - indexed commands draw their indices as the encoder's indexed draws do, and patch commands (the
//!   owned `rt6_patch_index` fixture) with their own factor buffer and their pipeline's control
//!   points; their buffers resolve as the buffers do;
//! - patch commands a kernel writes multiply their factors by the scale they hold, though their
//!   pipeline state never enables the scale;
//! - (Metal probes) patch commands the host encodes read a factor offset of 2 from that byte, step
//!   an instance stride 2 past a multiple of 4 by the multiple below, and draw 33 control points
//!   with their pipeline's;
//! - a command whose barrier is set draws after the commands before it, their
//!   writes visible to it;
//! - a host step drawing a state made after the commit, on the variant for a mapped texture, makes
//!   and counts a sampler set of its own;
//! - a host step drawing a state made after the commit, whose fragment samples through its argument
//!   buffer, binds the neutrals the state's creation made, and makes none.

use super::degrade_notices::{
    assert_made_once, forced, made_render_kinds, on_device, on_device_with, shared_in_system_memory,
};
use super::indirect_command_buffers::{render_template_descriptor, render_template_state};
use super::indirect_host_steps::{self, complete, param, HOST_STEPS};
use super::patch_indices::Lane;
use super::patches::{self, options_of, Path};
use super::render_arguments::{output, Members, SAMPLED};
use super::render_state::texture;
use super::renders::public_render;
use super::resource_ids::{distinct_sampler_states, UNLISTED};
use super::{buffer_with, pipelines_settled, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use crate::candidate::engine::graphics::Assembly;
use crate::candidate::engine::resources::pixel_format;
use crate::candidate::runtime_case::render_pipeline_descriptor;
use crate::case::indirect_commands::metal_primitive_type;
use metal2vulkan::bind::DrawCall;
use metal2vulkan::bind::ReplayFaults;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::metal::{
    IndexType, IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage,
    TessellationControlPointIndexType, TessellationFactorStepFunction,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, RuntimeFact, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble_text};
use metal2vulkan_runtime::{
    Attachment, Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, Degradation,
    Device, DrawPrimitiveType, DrawProblem, Error, EventRef, ExecutedPipeline, Execution, Heap,
    HeapDescriptor, HeapType, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    IndirectCommandBufferProblem, IndirectRenderCommand, Invalid, PipelineOptions, Problem, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap,
    Size,
};
use metal2vulkan_runtime::{LoadAction, StoreAction, TextureDescriptor};

/// The call an execution's refusals name.
const CALL: &str = "execute_commands_in_buffer";

/// The words of one window of `params` and of `out`.
pub(super) const WINDOW: usize = 64;

/// The windows of `params` and `out`.
const WINDOWS: usize = 4;

/// The template's state, `params` (each window's words as `indirect_host_steps::param` gives
/// them) and `out` (zero).
pub(super) struct Setup {
    pub(super) state: RenderPipelineState,
    pub(super) params: Buffer,
    pub(super) out: Buffer,
}

impl Setup {
    pub(super) fn new(device: &Device) -> Setup {
        let params: Vec<u8> = (0..WINDOWS * WINDOW)
            .map(|word| param(word / WINDOW, word % WINDOW))
            .flat_map(u32::to_le_bytes)
            .collect();
        Setup {
            state: render_template_state(device, true),
            params: buffer_with(device, &params).unwrap(),
            out: buffer_with(device, &vec![0; WINDOWS * WINDOW * 4]).unwrap(),
        }
    }

    /// Command `index` of `icb` with the template's state and window `window` of `params` and
    /// `out` as vertex buffers 0 and 1.
    fn command(
        &self,
        icb: &IndirectCommandBuffer,
        index: u64,
        window: usize,
    ) -> IndirectRenderCommand {
        let command = icb.indirect_render_command_at(index).unwrap();
        command.set_render_pipeline_state(&self.state).unwrap();
        let offset = (window * WINDOW * 4) as u64;
        command
            .set_vertex_buffer(Some(&self.params), offset, 0)
            .unwrap();
        command
            .set_vertex_buffer(Some(&self.out), offset, 1)
            .unwrap();
        command
    }

    /// Bind window `window` of `params` and `out` as `encoder`'s vertex buffers 0 and 1.
    fn bind(&self, encoder: &mut RenderCommandEncoder, window: usize) {
        let offset = (window * WINDOW * 4) as u64;
        encoder
            .set_vertex_buffer(Some(&self.params), offset, 0)
            .unwrap();
        encoder
            .set_vertex_buffer(Some(&self.out), offset, 1)
            .unwrap();
    }

    /// Make `params` and `out` resident for `encoder`'s vertex stage.
    pub(super) fn resident(&self, encoder: &mut RenderCommandEncoder) {
        encoder
            .use_resource(
                self.params.as_resource(),
                ResourceUsage::READ,
                RenderStages::VERTEX,
            )
            .unwrap();
        encoder
            .use_resource(
                self.out.as_resource(),
                ResourceUsage::WRITE,
                RenderStages::VERTEX,
            )
            .unwrap();
    }

    /// `out`, read back.
    pub(super) fn output(&self) -> Vec<u32> {
        read_buffer(&self.out, 0, self.out.length())
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    }

    /// [`Setup::new`]'s, with `params` and `out` placed in a heap of their own ([`shared_heap`]),
    /// which it returns.
    pub(super) fn in_heap(device: &Device) -> (Setup, Heap) {
        let own = Setup::new(device);
        let heap = shared_heap(device);
        let place =
            |buffer: &Buffer| placed(&heap, &read_buffer(buffer, 0, buffer.length()).unwrap());
        let setup = Setup {
            params: place(&own.params),
            out: place(&own.out),
            state: own.state,
        };
        (setup, heap)
    }
}

/// A `Shared` automatic heap of 1 MiB.
fn shared_heap(device: &Device) -> Heap {
    device
        .new_heap(&HeapDescriptor::new(
            HeapType::Automatic,
            1 << 20,
            ResourceOptions::shared(),
        ))
        .unwrap()
}

/// A `Shared` buffer placed in `heap`, holding `bytes`.
fn placed(heap: &Heap, bytes: &[u8]) -> Buffer {
    let buffer = heap
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().expect("a Shared buffer");
    // SAFETY: the buffer holds `bytes.len()` bytes at `contents`, and no command uses them.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// A draw over a window: `count` vertices from `start`, `instances` instances from `base`.
pub(super) struct Drawn {
    pub(super) window: usize,
    pub(super) start: usize,
    pub(super) count: usize,
    pub(super) instances: usize,
    pub(super) base: usize,
}

/// `out` after `draws`, zero where none wrote.
pub(super) fn expected(draws: &[Drawn]) -> Vec<u32> {
    let mut out = vec![0; WINDOWS * WINDOW];
    for draw in draws {
        let (p0, p1) = (param(draw.window, 0), param(draw.window, 1));
        for iid in draw.base..draw.base + draw.instances {
            for vid in draw.start..draw.start + draw.count {
                out[draw.window * WINDOW + 16 * iid + vid] =
                    p0 * (vid as u32 + 1) + p1 + 1000 * iid as u32;
            }
        }
    }
    out
}

/// A buffer of `count` render commands as `descriptor` describes.
pub(super) fn icb(
    device: &Device,
    descriptor: &IndirectCommandBufferDescriptor,
    count: u64,
) -> IndirectCommandBuffer {
    device
        .new_indirect_command_buffer(descriptor, count, ResourceOptions::shared())
        .unwrap()
}

/// Draw commands of two vertex buffers and one fragment buffer, nothing inherited.
fn descriptor() -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = 1;
    descriptor
}

/// A 1x1 pass without attachments.
pub(super) fn pass() -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    pass
}

/// The kinds a test's steps force beside its own, where the device has not announced them: the
/// makings of the pipeline states of `made`, and the `Shared` buffers' system memory.
pub(super) fn forced_with(
    device: &Device,
    made: &[metal2vulkan::pipeline::RenderPipelineDescriptor],
) -> std::collections::BTreeSet<Degradation> {
    let mut kinds = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(device),
    )]);
    for descriptor in made {
        kinds.extend(made_render_kinds(device, descriptor));
    }
    kinds
}

/// The descriptor of the state the host steps of `render_arguments::SAMPLED` draw, made with
/// `supportIndirectCommandBuffers`.
fn sampled_descriptor(
    device: &Device,
    case: &super::renders::PublicRender,
) -> metal2vulkan::pipeline::RenderPipelineDescriptor {
    let mut descriptor = render_pipeline_descriptor(device, &case.inputs()).unwrap();
    descriptor.support_indirect_command_buffers = true;
    descriptor
}

fn refused(problem: Problem) -> Option<Error> {
    Some(Error::Invalid(Invalid::new(CALL, problem)))
}

/// Whether `execution`, a draw, reaches some buffer slot at its device address: its variant
/// takes the large-buffer form.
fn addressed(execution: &Execution) -> bool {
    match execution.pipeline() {
        ExecutedPipeline::Render(pipeline) => pipeline
            .dependencies
            .iter()
            .any(|dependency| matches!(dependency.fact, RuntimeFact::DeviceAddresses { .. })),
        ExecutedPipeline::Compute(_) => false,
    }
}

/// Commands draw as their records stand at commit: one the CPU wrote after the call (Metal's
/// execution sees it) included. A command with a state but no draw, and one with a draw but no
/// state, run nothing, and their neighbours run. The encoder's own draws before and after the
/// execution run, and its execution record holds every draw. Nothing is replayed on the host.
#[test]
fn commands_draw_as_their_records_stand_at_commit() {
    on_device(|device, messages| {
        let setup = Setup::new(device);
        let icb = icb(device, &descriptor(), 4);
        setup
            .command(&icb, 0, 0)
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        setup.command(&icb, 1, 1);
        let stateless = icb.indirect_render_command_at(2).unwrap();
        stateless
            .set_vertex_buffer(Some(&setup.out), (WINDOW * 4) as u64, 1)
            .unwrap();
        stateless
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        setup.resident(&mut encoder);
        encoder.set_render_pipeline_state(&setup.state).unwrap();
        setup.bind(&mut encoder, 3);
        encoder
            .draw_primitives(PrimitiveType::Point, 0, 2, 1, 0)
            .unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..4).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Point, 2, 2, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        setup
            .command(&icb, 3, 2)
            .draw_primitives(PrimitiveType::Point, 2, 3, 2, 1)
            .unwrap();
        complete(&command_buffer);
        let draws = [
            Drawn {
                window: 0,
                start: 0,
                count: 4,
                instances: 1,
                base: 0,
            },
            Drawn {
                window: 2,
                start: 2,
                count: 3,
                instances: 2,
                base: 1,
            },
            Drawn {
                window: 3,
                start: 0,
                count: 4,
                instances: 1,
                base: 0,
            },
        ];
        assert_eq!(setup.output(), expected(&draws));
        assert_eq!(command_buffer.executions().len(), 4);
        assert_made_once(messages, &[], &forced_with(device, &[]));
    });
}

/// An inheriting buffer's commands draw with the encoder's pipeline state and buffers as the
/// call found them; a state and buffers bound after it are the encoder's own. Refused at the
/// call, changing nothing, as Metal's validation stops it and plain Metal's process ends: a range
/// that is not empty with an inherited state that is unset. An empty range, at the start or at the
/// size, runs nothing whatever the state, and the encoder goes on, as plain Metal's. One under a
/// state made without `supportIndirectCommandBuffers` faults its encoder
/// (`an_execution_under_a_state_without_support_faults_its_encoder`).
#[test]
fn inheriting_commands_draw_with_the_encoder_s_state_at_the_call() {
    on_device(|device, _| {
        let setup = Setup::new(device);
        let mut inheriting = IndirectCommandBufferDescriptor::new();
        inheriting.command_types = IndirectCommandType::DRAW;
        inheriting.inherit_pipeline_state = true;
        inheriting.inherit_buffers = true;
        let icb = icb(device, &inheriting, 1);
        icb.indirect_render_command_at(0)
            .unwrap()
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        let unsupported = render_template_state(device, false);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        setup.resident(&mut encoder);
        encoder.execute_commands_in_buffer(&icb, 0..0).unwrap();
        encoder.execute_commands_in_buffer(&icb, 1..1).unwrap();
        assert_eq!(
            encoder.execute_commands_in_buffer(&icb, 0..1).err(),
            refused(Problem::NoPipelineState)
        );
        encoder.set_render_pipeline_state(&unsupported).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..0).unwrap();
        encoder.execute_commands_in_buffer(&icb, 1..1).unwrap();
        encoder.set_render_pipeline_state(&setup.state).unwrap();
        setup.bind(&mut encoder, 0);
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        setup.bind(&mut encoder, 1);
        encoder.set_render_pipeline_state(&unsupported).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Point, 4, 2, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let draws = [
            Drawn {
                window: 0,
                start: 0,
                count: 4,
                instances: 1,
                base: 0,
            },
            Drawn {
                window: 1,
                start: 4,
                count: 2,
                instances: 1,
                base: 0,
            },
        ];
        assert_eq!(setup.output(), expected(&draws));
    });
}

/// An execution of commands that inherit the encoder's pipeline state, made without
/// `supportIndirectCommandBuffers`, faults its encoder, as Metal's GPU faults it (an M4, the M3
/// Max): the call is taken, for a range given and for one read from a buffer, `{0, 0}` included;
/// the encoder runs nothing, its own draw before the call either, and nor do the encoders after
/// it, as Metal's GPU loses them where it ran just before (`CommandBufferError::PageFault`); the
/// encoder before it runs and both event signals fire; the command buffer completes with status
/// `Error` and `CommandBufferError::PageFault`, Metal's code 3; and the next command buffer of the
/// queue completes, its work landed. The next one is committed while the faulting one is in
/// flight: Metal runs such a command buffer as it would, committed at any point from the faulting
/// one's commit to its completion, and on another queue (an M4). Window 0 is the pass's before,
/// window 1 the faulting passes' draws, windows 2 and 3 the pass's after and the next command
/// buffer's.
#[test]
fn an_execution_under_a_state_without_support_faults_its_encoder() {
    on_device(|device, _| {
        let setup = Setup::new(device);
        let mut inheriting = IndirectCommandBufferDescriptor::new();
        inheriting.command_types = IndirectCommandType::DRAW;
        inheriting.inherit_pipeline_state = true;
        inheriting.inherit_buffers = true;
        let icb = icb(device, &inheriting, 1);
        icb.indirect_render_command_at(0)
            .unwrap()
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        let unsupported = render_template_state(device, false);
        let empty = range_at_8(device, 0, 0);
        let queue = device.new_command_queue().unwrap();
        let before = device.new_shared_event().unwrap();
        let after = device.new_shared_event().unwrap();
        let draw = |command_buffer: &CommandBuffer, window, execute: Option<bool>| {
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            setup.resident(&mut encoder);
            encoder.set_render_pipeline_state(&setup.state).unwrap();
            setup.bind(&mut encoder, window);
            encoder
                .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
                .unwrap();
            if let Some(indirect) = execute {
                encoder.set_render_pipeline_state(&unsupported).unwrap();
                match indirect {
                    true => encoder.execute_commands_in_buffer_indirect(&icb, &empty, 8),
                    false => encoder.execute_commands_in_buffer(&icb, 0..1),
                }
                .unwrap();
            }
            encoder.end_encoding().unwrap();
        };
        let command_buffer = queue.command_buffer().unwrap();
        draw(&command_buffer, 0, None);
        command_buffer
            .encode_signal_event(EventRef::from(&before), 1)
            .unwrap();
        draw(&command_buffer, 1, Some(false));
        draw(&command_buffer, 1, Some(true));
        draw(&command_buffer, 2, None);
        command_buffer
            .encode_signal_event(EventRef::from(&after), 1)
            .unwrap();
        let next = queue.command_buffer().unwrap();
        draw(&next, 3, None);
        command_buffer.commit().unwrap();
        next.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert_eq!(command_buffer.error(), Some(CommandBufferError::PageFault));
        assert_eq!(CommandBufferError::PageFault.metal_code(), 3);
        assert_eq!([before.signaled_value(), after.signaled_value()], [1, 1]);
        next.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(next.status(), CommandBufferStatus::Completed);
        assert_eq!(next.error(), None);
        let drawn = |window| Drawn {
            window,
            start: 0,
            count: 4,
            instances: 1,
            base: 0,
        };
        assert_eq!(
            setup.output(),
            expected(&[drawn(0), drawn(3)]),
            "on {}",
            device.name()
        );
    });
}

/// Metal's private primitive types 5 to 8, and how many of six vertices (`0..6`, or the indices
/// [`PERMUTED`]) each type's list reaches: every one of a line loop, a fan and a quad strip, the
/// first four of quads.
const PRIVATE: [(DrawPrimitiveType, usize); 4] = [
    (DrawPrimitiveType::LineLoop, 6),
    (DrawPrimitiveType::TriangleFan, 6),
    (DrawPrimitiveType::Quads, 4),
    (DrawPrimitiveType::QuadStrip, 6),
];

/// Six 16-bit indices: a permutation of `0..6`.
const PERMUTED: [u16; 6] = [5, 3, 0, 1, 4, 2];

/// A shared buffer of [`PERMUTED`].
fn permuted(device: &Device) -> Buffer {
    let halves: Vec<u8> = PERMUTED.iter().flat_map(|h| h.to_le_bytes()).collect();
    buffer_with(device, &halves).unwrap()
}

/// Draw commands of both kinds, of two vertex buffers and one fragment buffer, nothing inherited.
fn both_draws() -> IndirectCommandBufferDescriptor {
    let mut both = descriptor();
    both.command_types = IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED);
    both
}

/// Commands of Metal's private primitive types 5 to 8 draw as the encoder's draws of them do, the
/// vertices of the list each type's program writes, direct and indexed (Metal's: on the M3 Max,
/// macOS 27.0.1, each drew the same texels through an indirect command buffer as the encoder's
/// draw). Windows 0 and 1 are the commands', 2 and 3 the encoder's own draws after them.
#[test]
fn commands_of_private_types_draw_as_the_encoder_s_draws() {
    on_device(|device, _| {
        let indices = permuted(device);
        for (primitive_type, reached) in PRIVATE {
            let setup = Setup::new(device);
            let icb = icb(device, &both_draws(), 2);
            setup
                .command(&icb, 0, 0)
                .draw_primitives(primitive_type, 0, 6, 1, 0)
                .unwrap();
            setup
                .command(&icb, 1, 1)
                .draw_indexed_primitives(primitive_type, 6, IndexType::UInt16, &indices, 0, 1, 0, 0)
                .unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            setup.resident(&mut encoder);
            encoder
                .use_resource(
                    indices.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::VERTEX,
                )
                .unwrap();
            encoder.set_render_pipeline_state(&setup.state).unwrap();
            encoder.execute_commands_in_buffer(&icb, 0..2).unwrap();
            setup.bind(&mut encoder, 2);
            encoder.draw_primitives(primitive_type, 0, 6, 1, 0).unwrap();
            setup.bind(&mut encoder, 3);
            encoder
                .draw_indexed_primitives(primitive_type, 6, IndexType::UInt16, &indices, 0, 1, 0, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let direct: Vec<usize> = (0..reached).collect();
            let indexed: Vec<usize> = PERMUTED[..reached]
                .iter()
                .map(|&i| usize::from(i))
                .collect();
            let drawn = expected_vertices(&[
                (0, &direct, 0..1),
                (1, &indexed, 0..1),
                (2, &direct, 0..1),
                (3, &indexed, 0..1),
            ]);
            assert_eq!(
                setup.output(),
                drawn,
                "{primitive_type:?} on {}",
                device.name()
            );
        }
    });
}

/// A command of primitive type 9 among a range given at the call fails its command buffer as the
/// encoder's draw of it does (Metal's: on the M3 Max, macOS 27.0.1, through an indirect command
/// buffer as directly): a direct one of 3 vertices hangs it (`GpuHang`), an indexed one of any
/// count faults it (`PageFault`), its encoder and every one after it running nothing, the command
/// before it included, and the encoder before running; a direct one of 2 vertices draws nothing,
/// and the rest runs. Window 3 is the encoder's before, 0 the command before the draw, 1 the
/// draw's, 2 the encoder's after; each on a queue of its own (a queue's second fault revokes it).
#[test]
fn a_command_of_type_nine_fails_its_command_buffer_as_the_encoder_s_draw_does() {
    on_device(|device, _| {
        let indices = permuted(device);
        let nine = DrawPrimitiveType::Nine;
        let cases = [
            (
                "a direct draw of 3",
                None,
                Some(CommandBufferError::GpuHang),
            ),
            (
                "an indexed draw of 3",
                Some(3),
                Some(CommandBufferError::PageFault),
            ),
            (
                "an indexed draw of none",
                Some(0),
                Some(CommandBufferError::PageFault),
            ),
            ("a direct draw of 2", None, None),
        ];
        for (label, indexed, error) in cases {
            let setup = Setup::new(device);
            let icb = icb(device, &both_draws(), 2);
            setup
                .command(&icb, 0, 0)
                .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
                .unwrap();
            let command = setup.command(&icb, 1, 1);
            match (indexed, error) {
                (Some(count), _) => command.draw_indexed_primitives(
                    nine,
                    count,
                    IndexType::UInt16,
                    &indices,
                    0,
                    1,
                    0,
                    0,
                ),
                (None, Some(_)) => command.draw_primitives(nine, 0, 3, 1, 0),
                (None, None) => command.draw_primitives(nine, 0, 2, 1, 0),
            }
            .unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let points = |window: usize, execute: bool| {
                let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
                setup.resident(&mut encoder);
                encoder
                    .use_resource(
                        indices.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )
                    .unwrap();
                encoder.set_render_pipeline_state(&setup.state).unwrap();
                if execute {
                    encoder.execute_commands_in_buffer(&icb, 0..2).unwrap();
                }
                setup.bind(&mut encoder, window);
                encoder
                    .draw_primitives(PrimitiveType::Point, 0, 2, 1, 0)
                    .unwrap();
                encoder.end_encoding().unwrap();
            };
            points(3, false);
            points(2, true);
            points(2, false);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let label = format!("{label} on {}", device.name());
            assert_eq!(command_buffer.error(), error, "{label}");
            let window = |window, count| Drawn {
                window,
                start: 0,
                count,
                instances: 1,
                base: 0,
            };
            let drawn = match error {
                Some(_) => {
                    assert_eq!(
                        command_buffer.status(),
                        CommandBufferStatus::Error,
                        "{label}"
                    );
                    expected(&[window(3, 2)])
                }
                None => {
                    assert_eq!(
                        command_buffer.status(),
                        CommandBufferStatus::Completed,
                        "{label}"
                    );
                    expected(&[window(3, 2), window(0, 4), window(2, 2)])
                }
            };
            assert_eq!(setup.output(), drawn, "{label}");
        }
    });
}

/// A command's buffer address in no resource the command buffer's encoders made resident, and
/// in no buffer of its own (a buffer placed in a heap no encoder of the command buffer used, which
/// Metal's M4 did not reach either), misses: the slot reads zero and drops writes, so `out` keeps
/// its bytes, and the execution report counts one miss per slot.
#[test]
fn addresses_outside_the_resident_resources_miss() {
    on_device(|device, _| {
        let (setup, _heap) = Setup::in_heap(device);
        let icb = icb(device, &descriptor(), 1);
        setup
            .command(&icb, 0, 0)
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(setup.output(), expected(&[]));
        let report = command_buffer.execution_report().unwrap();
        assert!(report.resident_miss);
        assert_eq!(report.resident_misses, 2);
    });
}

/// Residency is the command buffer's (an M4: an indirect render command's vertex
/// buffer that only a later compute or render encoder made resident was reached, every run):
/// a command's buffers that only a later encoder made resident, or an earlier one, draw as
/// those its own encoder made resident, with no miss; and so do buffers of their own that no
/// encoder made resident, and buffers placed in a heap a later encoder used.
#[test]
fn commands_reach_what_any_encoder_of_their_command_buffer_made_resident() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        for when in [
            "earlier compute",
            "later compute",
            "later render",
            "none",
            "later heap",
        ] {
            let (setup, heap) = match when {
                "later heap" => {
                    let (setup, heap) = Setup::in_heap(device);
                    (setup, Some(heap))
                }
                _ => (Setup::new(device), None),
            };
            let icb = icb(device, &descriptor(), 1);
            setup
                .command(&icb, 0, 0)
                .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let compute = || {
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                let resident = [setup.params.as_resource(), setup.out.as_resource()];
                encoder
                    .use_resources(&resident, ResourceUsage::READ | ResourceUsage::WRITE)
                    .unwrap();
                encoder.end_encoding().unwrap();
            };
            if when == "earlier compute" {
                compute();
            }
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            match (when, &heap) {
                ("later compute", _) => compute(),
                ("later render", _) => {
                    let mut later = command_buffer.render_command_encoder(&pass()).unwrap();
                    setup.resident(&mut later);
                    later.end_encoding().unwrap();
                }
                (_, Some(heap)) => {
                    let mut later = command_buffer.compute_command_encoder().unwrap();
                    later.use_heap(heap).unwrap();
                    later.end_encoding().unwrap();
                }
                _ => {}
            }
            complete(&command_buffer);
            let drawn = Drawn {
                window: 0,
                start: 0,
                count: 4,
                instances: 1,
                base: 0,
            };
            assert_eq!(setup.output(), expected(&[drawn]), "{when}");
            let report = command_buffer.execution_report().unwrap();
            assert!(!report.resident_miss, "{when}");
        }
    });
}

/// The CPU's draw of primitives into a buffer whose command types lack `DRAW` (indexed draws,
/// patch draws, indexed patch draws), whose records hold it, draws as on a buffer of its own
/// kind, as plain Metal draws it, where Metal's validation stops the setter.
#[test]
fn a_draw_into_a_buffer_whose_records_hold_it_draws_as_on_its_own() {
    on_device(|device, _| {
        for types in [
            IndirectCommandType::DRAW_INDEXED,
            IndirectCommandType::DRAW_PATCHES,
            IndirectCommandType::DRAW_INDEXED_PATCHES,
        ] {
            let setup = Setup::new(device);
            let mut held = descriptor();
            held.command_types = types;
            let icb = icb(device, &held, 1);
            setup
                .command(&icb, 0, 1)
                .draw_primitives(PrimitiveType::Point, 1, 3, 2, 0)
                .unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            setup.resident(&mut encoder);
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let drawn = Drawn {
                window: 1,
                start: 1,
                count: 3,
                instances: 2,
                base: 0,
            };
            assert_eq!(setup.output(), expected(&[drawn]), "{types:?}");
            let report = command_buffer.execution_report().unwrap();
            assert!(!report.replay_fault, "{types:?}");
        }
    });
}

/// Commands a kernel writes (`icb_render_replay_encode_draw_only`: thread `tid` sets command
/// `tid`'s draw, 4 points from `4 * tid`, onto the state and buffers the CPU set) are drawn once
/// the kernel completed, by a host step the device announces once. Once the pipelines the first
/// run began making settled, a run of twice the draws looks the draws' `VkPipeline` up by key no
/// more often (`WorkCounts::graphics_pipelines_looked_up`, the commit's settle of each primitive
/// type included): a step's draws of one state and key past the first find the entry the state's
/// last draws found made.
#[test]
fn commands_a_kernel_writes_are_drawn_by_a_host_step() {
    on_device(|device, messages| {
        let setup = Setup::new(device);
        let writer =
            indirect_host_steps::state(device, "icb_render_replay_encode_draw_only", false);
        let commands = 8;
        let icb = icb(device, &descriptor(), commands);
        for index in 0..commands {
            setup.command(&icb, index, 0);
        }
        let arguments = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let mut looked_up = Vec::new();
        for executed in [commands / 2, commands / 2, commands] {
            pipelines_settled(device);
            let command_buffer = queue.command_buffer().unwrap();
            let mut writing = command_buffer.compute_command_encoder().unwrap();
            writing.set_compute_pipeline_state(&writer).unwrap();
            writing.set_buffer(Some(&arguments), 0, 0).unwrap();
            writing.set_buffer(Some(&setup.params), 0, 1).unwrap();
            writing.set_buffer(Some(&setup.out), 0, 2).unwrap();
            writing
                .use_resource(icb.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            let threads = Size::new(commands, 1, 1);
            writing.dispatch_threads(threads, threads).unwrap();
            writing.end_encoding().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            setup.resident(&mut encoder);
            encoder
                .execute_commands_in_buffer(&icb, 0..executed)
                .unwrap();
            encoder.end_encoding().unwrap();
            let before = device.work_counts().graphics_pipelines_looked_up;
            complete(&command_buffer);
            looked_up.push(device.work_counts().graphics_pipelines_looked_up - before);
            let drawn = Drawn {
                window: 0,
                start: 0,
                count: 4 * executed as usize,
                instances: 1,
                base: 0,
            };
            assert_eq!(setup.output(), expected(&[drawn]), "{executed} commands");
        }
        assert!(looked_up[2] <= looked_up[1], "{looked_up:?}");
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &[]));
    });
}

/// A host step draws on what its commit compiled: commands a kernel writes, drawn by a host step
/// that waits for an event the test signals once the commit returned, draw as Metal's on a
/// variant in the large-buffer form (one that reaches the commands' buffers at their device
/// addresses), and the shader cache counts no lookup from the commit's return to the
/// completion, so the step compiled nothing; the commit settled the draws' pipeline with no
/// buffer of the commands bound. Nor does it count one across the commit (the coordinator's ruling
/// of 2026-10-07): the state's creation compiled the variant, and the commit only looked it up. The
/// same commands the CPU wrote, drawn at commit, take the bound form.
#[test]
fn a_host_step_draws_on_what_its_commit_compiled() {
    on_device(|device, messages| {
        let setup = Setup::new(device);
        let writer =
            indirect_host_steps::state(device, "icb_render_replay_encode_draw_only", false);
        let commands = 4;
        let drawn = || Drawn {
            window: 0,
            start: 0,
            count: 4 * commands as usize,
            instances: 1,
            base: 0,
        };
        let queue = device.new_command_queue().unwrap();
        // Whether each command's draw reached its buffers at their device addresses.
        let large = |command_buffer: &CommandBuffer| {
            let executions = command_buffer.executions();
            let draws = &executions[executions.len() - commands as usize..];
            draws.iter().map(addressed).collect::<Vec<_>>()
        };

        // The CPU's records, drawn at commit.
        let written = icb(device, &descriptor(), commands);
        for index in 0..commands {
            setup.command(&written, index, 0);
            written
                .indirect_render_command_at(index)
                .unwrap()
                .draw_primitives(PrimitiveType::Point, 4 * index, 4, 1, 0)
                .unwrap();
        }
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        setup.resident(&mut encoder);
        encoder
            .execute_commands_in_buffer(&written, 0..commands)
            .unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected(&[drawn()]), "drawn at commit");
        assert_eq!(large(&command_buffer), vec![false; commands as usize]);

        // The kernel's records, drawn by the host step once the event is signaled.
        let setup = Setup::new(device);
        let icb = icb(device, &descriptor(), commands);
        for index in 0..commands {
            setup.command(&icb, index, 0);
        }
        let arguments = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut writing = command_buffer.compute_command_encoder().unwrap();
        writing.set_compute_pipeline_state(&writer).unwrap();
        writing.set_buffer(Some(&arguments), 0, 0).unwrap();
        writing.set_buffer(Some(&setup.params), 0, 1).unwrap();
        writing.set_buffer(Some(&setup.out), 0, 2).unwrap();
        writing
            .use_resource(icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        let threads = Size::new(commands, 1, 1);
        writing.dispatch_threads(threads, threads).unwrap();
        writing.end_encoding().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        setup.resident(&mut encoder);
        encoder
            .execute_commands_in_buffer(&icb, 0..commands)
            .unwrap();
        encoder.end_encoding().unwrap();
        let encoded = device.shader_cache_statistics();
        command_buffer.commit().unwrap();
        let committed = device.shader_cache_statistics();
        assert_eq!(
            (committed.hits, committed.misses),
            (encoded.hits, encoded.misses),
            "the commit compiled"
        );
        event.set_signaled_value(1);
        indirect_host_steps::wait(&command_buffer);
        let completed = device.shader_cache_statistics();
        assert_eq!(
            (completed.hits, completed.misses),
            (committed.hits, committed.misses),
            "the host step compiled"
        );
        assert_eq!(
            setup.output(),
            expected(&[drawn()]),
            "drawn by the host step"
        );
        assert_eq!(large(&command_buffer), vec![true; commands as usize]);
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &[]));
    });
}

/// A state of [`render_template_state`]'s function under another name, made with
/// `supportIndirectCommandBuffers`: it draws as the template's, from a variant of its own, so no
/// `VkPipeline` of the template's serves its draws.
fn renamed_template_state(device: &Device) -> RenderPipelineState {
    const NAME: &str = "icb_render_replay_template";
    let text = std::fs::read_to_string(engine_dir().join(format!("{NAME}.ll"))).unwrap();
    let renamed = format!("{NAME}_renamed");
    let bitcode = llvm_assemble_text(&text.replace(NAME, &renamed)).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(&renamed).unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    descriptor.support_indirect_command_buffers = true;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A record the GPU wrote naming a state the CPU set on none of the executed commands: the commit
/// settles the states the CPU set (command 0's), and the host step makes the other state's
/// `VkPipeline` itself, on the steps' thread, counted in `WorkCounts::host_step_pipelines_made`
/// with the step in `host_steps_recorded` and `host_step_nanos`. A blit of an earlier command
/// buffer copies another buffer's command, of its own state over window 1, into command 1. Both
/// draw as Metal's, and the host step is announced once.
#[test]
fn a_host_step_makes_the_pipeline_of_a_state_only_a_gpu_written_record_names() {
    on_device(|device, messages| {
        let setup = Setup::new(device);
        let other = renamed_template_state(device);
        let queue = device.new_command_queue().unwrap();
        let source = icb(device, &descriptor(), 1);
        let copied = setup.command(&source, 0, 1);
        copied.set_render_pipeline_state(&other).unwrap();
        copied
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        let target = icb(device, &descriptor(), 2);
        setup
            .command(&target, 0, 0)
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        let copying = queue.command_buffer().unwrap();
        let mut blit = copying.blit_command_encoder().unwrap();
        blit.copy_indirect_command_buffer(&source, 0..1, &target, 1)
            .unwrap();
        blit.end_encoding().unwrap();
        complete(&copying);
        pipelines_settled(device);
        let before = device.work_counts();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        setup.resident(&mut encoder);
        encoder.execute_commands_in_buffer(&target, 0..2).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let after = device.work_counts();
        let drawn = |window| Drawn {
            window,
            start: 0,
            count: 4,
            instances: 1,
            base: 0,
        };
        assert_eq!(setup.output(), expected(&[drawn(0), drawn(1)]));
        assert_eq!(after.host_steps_recorded - before.host_steps_recorded, 1);
        assert!(after.host_step_nanos > before.host_step_nanos);
        assert!(
            after.host_step_pipelines_made > before.host_step_pipelines_made,
            "the step made the other state's pipeline: {} then {}",
            before.host_step_pipelines_made,
            after.host_step_pipelines_made
        );
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &[]));
        drop(other);
    });
}

/// A command whose barrier is set draws after the commands before it, their writes visible to
/// it, as plain Metal orders it where its validation stops `setBarrier` (an M4): command 0 writes
/// window 0 of `out`, and command 1, behind its barrier, reads that window as its `params` and
/// writes window 1, so window 1 holds what command 0's words make. A barrier set after the encoder
/// that executes it ended, whose pass the encoder made without runs, orders the draw the same.
#[test]
fn a_command_s_barrier_orders_its_draw_after_the_earlier_ones() {
    on_device(|device, _| {
        let setup = Setup::new(device);
        let queue = device.new_command_queue().unwrap();
        // Command 0 over window 0; command 1 reading window 0 of `out`, writing window 1.
        let commands = |icb: &IndirectCommandBuffer| {
            setup
                .command(icb, 0, 0)
                .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
                .unwrap();
            let second = icb.indirect_render_command_at(1).unwrap();
            second.set_render_pipeline_state(&setup.state).unwrap();
            second.set_vertex_buffer(Some(&setup.out), 0, 0).unwrap();
            second
                .set_vertex_buffer(Some(&setup.out), (WINDOW * 4) as u64, 1)
                .unwrap();
            second
                .draw_primitives(PrimitiveType::Point, 0, 2, 1, 0)
                .unwrap();
            second
        };
        let execute = |command_buffer: &CommandBuffer, icb: &IndirectCommandBuffer| {
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            encoder
                .use_resource(
                    setup.params.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::VERTEX,
                )
                .unwrap();
            encoder
                .use_resource(
                    setup.out.as_resource(),
                    ResourceUsage::READ | ResourceUsage::WRITE,
                    RenderStages::VERTEX,
                )
                .unwrap();
            encoder.execute_commands_in_buffer(icb, 0..2).unwrap();
            encoder.end_encoding().unwrap();
        };
        let ordered = icb(device, &descriptor(), 2);
        commands(&ordered).set_barrier().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        execute(&command_buffer, &ordered);
        complete(&command_buffer);
        let mut want = expected(&[Drawn {
            window: 0,
            start: 0,
            count: 4,
            instances: 1,
            base: 0,
        }]);
        let (a0, a1) = (want[0], want[1]);
        for vid in 0..2 {
            want[WINDOW + vid] = a0.wrapping_mul(vid as u32 + 1).wrapping_add(a1);
        }
        assert_eq!(setup.output(), want);

        let contents = setup.out.contents().expect("a Shared buffer");
        // SAFETY: `out` holds `out.length()` bytes at `contents`, and no command uses them now.
        unsafe { std::ptr::write_bytes(contents.as_ptr(), 0, setup.out.length() as usize) };
        let late = icb(device, &descriptor(), 2);
        let second = commands(&late);
        let command_buffer = queue.command_buffer().unwrap();
        execute(&command_buffer, &late);
        second.set_barrier().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), want);
    });
}

/// Refused at the call, changing nothing: a range that ends past the size (Metal's validation
/// stops it), and a buffer of dispatches alone. An empty range runs nothing, and so does a buffer
/// of no command types, or of dispatches beside draws of primitives, as plain Metal's.
#[test]
fn what_metal_stops_is_refused_at_the_call() {
    on_device(|device, _| {
        let icb = icb(device, &descriptor(), 2);
        let mut compute = IndirectCommandBufferDescriptor::new();
        compute.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        let mut mixed = compute;
        mixed.command_types = mixed.command_types.union(IndirectCommandType::DRAW);
        let mixed = self::icb(device, &mixed, 2);
        let none = self::icb(device, &IndirectCommandBufferDescriptor::new(), 2);
        let compute = self::icb(device, &compute, 1);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        assert_eq!(
            encoder.execute_commands_in_buffer(&icb, 1..3).err(),
            refused(Problem::IndirectCommandBuffer(
                IndirectCommandBufferProblem::Range {
                    location: 1,
                    length: 2,
                    size: 2,
                }
            ))
        );
        assert_eq!(
            encoder.execute_commands_in_buffer(&compute, 0..1).err(),
            Some(Error::NotImplemented(RuntimeGap::IndirectRenderCommands))
        );
        encoder.execute_commands_in_buffer(&icb, 2..2).unwrap();
        encoder.execute_commands_in_buffer(&mixed, 0..2).unwrap();
        encoder.execute_commands_in_buffer(&none, 0..2).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert!(command_buffer.executions().is_empty());
    });
}

/// `out` after indexed draws: each `(window, vertices, instances)` writes each vertex of each
/// instance, as [`expected`] does.
fn expected_vertices(draws: &[(usize, &[usize], std::ops::Range<usize>)]) -> Vec<u32> {
    let mut out = vec![0; WINDOWS * WINDOW];
    for (window, vertices, instances) in draws {
        let (p0, p1) = (param(*window, 0), param(*window, 1));
        for iid in instances.clone() {
            for &vid in *vertices {
                out[window * WINDOW + 16 * iid + vid] =
                    p0 * (vid as u32 + 1) + p1 + 1000 * iid as u32;
            }
        }
    }
    out
}

/// A shared buffer of `words`.
fn words(device: &Device, words: &[u32]) -> Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer_with(device, &bytes).unwrap()
}

/// The encoder's indexed draw reads its indices from its offset's low 32 bits, as Metal reads
/// them (an M4: an offset of 2^32 + 16 into a buffer past 4 GiB read the index
/// at 16): an offset of 2^32 + 16 into a 20-byte buffer draws the index at 16.
#[test]
fn an_indexed_draw_reads_from_its_offset_s_low_32_bits() {
    on_device(|device, _| {
        let setup = Setup::new(device);
        let indices = words(device, &[0, 1, 2, 7, 3]);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        encoder.set_render_pipeline_state(&setup.state).unwrap();
        setup.bind(&mut encoder, 0);
        encoder
            .draw_indexed_primitives(
                PrimitiveType::Point,
                1,
                IndexType::UInt32,
                &indices,
                (1 << 32) + 16,
                1,
                0,
                0,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let mut expected = vec![0; WINDOW];
        expected[3] = param(0, 0) * 4 + param(0, 1);
        assert_eq!(setup.output()[..WINDOW], expected);
    });
}

/// Indexed commands draw their indices as the encoder's indexed draws do: from the offset, each
/// plus the signed base vertex, `ushort` and `uint`, over their instances from the base instance,
/// beside a draw of primitives. An index buffer no encoder made resident draws, as Metal's did
/// there (a buffer of its own, which the command buffer's resident set holds); one placed in a heap
/// no encoder of the command buffer used misses: that command runs nothing, its neighbour runs, and
/// the execution report counts it.
#[test]
fn indexed_commands_draw_their_indices() {
    on_device(|device, messages| {
        let setup = Setup::new(device);
        let mut descriptor = descriptor();
        descriptor.command_types =
            IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED);
        let icb = icb(device, &descriptor, 3);
        let uints = words(device, &[99, 6, 2, 4]);
        let halves: Vec<u8> = [3u16, 0, 7, 0]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let ushorts = buffer_with(device, &halves).unwrap();
        setup
            .command(&icb, 0, 0)
            .draw_indexed_primitives(
                PrimitiveType::Point,
                3,
                IndexType::UInt32,
                &uints,
                4,
                2,
                -1,
                1,
            )
            .unwrap();
        setup
            .command(&icb, 1, 1)
            .draw_indexed_primitives(
                PrimitiveType::Point,
                3,
                IndexType::UInt16,
                &ushorts,
                0,
                1,
                2,
                0,
            )
            .unwrap();
        setup
            .command(&icb, 2, 2)
            .draw_primitives(PrimitiveType::Point, 0, 2, 1, 0)
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        setup.resident(&mut encoder);
        for indices in [&uints, &ushorts] {
            encoder
                .use_resource(
                    indices.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::VERTEX,
                )
                .unwrap();
        }
        encoder.execute_commands_in_buffer(&icb, 0..3).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let drawn = expected_vertices(&[
            (0, &[5, 1, 3], 1..3),
            (1, &[5, 2, 9], 0..1),
            (2, &[0, 1], 0..1),
        ]);
        assert_eq!(setup.output(), drawn);

        // The index buffer in a heap no encoder used: the command misses, its neighbour runs.
        let stray_words: Vec<u8> = [0u32, 1, 2].iter().flat_map(|w| w.to_le_bytes()).collect();
        let heap = shared_heap(device);
        let run_stray = |stray: &Buffer| {
            setup
                .command(&icb, 0, 3)
                .draw_indexed_primitives(
                    PrimitiveType::Point,
                    3,
                    IndexType::UInt32,
                    stray,
                    0,
                    1,
                    0,
                    0,
                )
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            setup.resident(&mut encoder);
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.execute_commands_in_buffer(&icb, 2..3).unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            command_buffer.execution_report().unwrap()
        };
        let report = run_stray(&placed(&heap, &stray_words));
        assert_eq!(setup.output(), drawn);
        assert!(report.resident_miss);
        assert_eq!(report.resident_misses, 1);
        // One of its own, which no encoder made resident: the command draws.
        let report = run_stray(&buffer_with(device, &stray_words).unwrap());
        let drawn = expected_vertices(&[
            (0, &[5, 1, 3], 1..3),
            (1, &[5, 2, 9], 0..1),
            (2, &[0, 1], 0..1),
            (3, &[0, 1, 2], 0..1),
        ]);
        assert_eq!(setup.output(), drawn);
        assert!(!report.resident_miss);
        assert_made_once(messages, &[], &forced_with(device, &[]));
    });
}

/// Patch commands draw with their own tessellation factor buffer, never the encoder's (whose
/// factors of 0 would cull every patch), and with their pipeline's control points, whatever the
/// record holds, as Metal's did: the `rt6_patch_index` fixture's marks are those of its encoder
/// draw with the same patch index buffer (`patch_indices.rs`'s `pib` case, Metal's marks). A factor
/// buffer of its own that no encoder made resident is reached, as the command buffer's resident set
/// holds it; one placed in a heap no encoder of the command buffer used misses: the command runs
/// nothing. A tessellation pipeline that scales its factors is refused with
/// `supportIndirectCommandBuffers`, as Metal's validation refuses it. A device without tessellation
/// shaders refuses the pipeline, reported as its refusal.
#[test]
fn patch_commands_draw_with_their_own_factors() {
    patch_commands_draw_with_their_own_factors_on(Path::Native);
}

/// [`patch_commands_draw_with_their_own_factors`], emulated.
#[test]
fn emulated_patch_commands_draw_with_their_own_factors() {
    patch_commands_draw_with_their_own_factors_on(Path::Emulated);
}

fn patch_commands_draw_with_their_own_factors_on(path: Path) {
    on_device_with(options_of(path), |device, _| {
        let lane = Lane::new(device.clone());
        let mut descriptor = lane.descriptor(TessellationControlPointIndexType::None);
        descriptor.support_indirect_command_buffers = true;
        let made = device.new_render_pipeline_state(&descriptor, &PipelineOptions::default());
        let state = match made {
            Ok(state) => state,
            Err(Error::Unsupported(u)) if path == Path::Native => {
                eprintln!(
                    "patch commands on {}: the pipeline is refused, {u:?}",
                    device.name()
                );
                return;
            }
            Err(error) => panic!("the pipeline: {error}"),
        };
        let mut scaled = descriptor.clone();
        if let Some(tessellation) = scaled.tessellation.as_mut() {
            tessellation.factor_scale_enabled = true;
        }
        assert_eq!(
            device
                .new_render_pipeline_state(&scaled, &PipelineOptions::default())
                .err(),
            Some(Error::Invalid(Invalid::new(
                "new_render_pipeline_state",
                Problem::IndirectScaledTessellation
            )))
        );
        let points = lane.words(&(0..12).collect::<Vec<u32>>(), 0);
        let tags = lane.words(&(0..16).map(|p| 0xa0 + p).collect::<Vec<u32>>(), 0);
        // Half factors (three edges, one inside) of 1 per patch, but patch 3's 0; the
        // encoder's are all 0.
        let factor_bytes: Vec<u8> = (0..16)
            .flat_map(|p| [if p == 3 { 0u16 } else { 0x3c00 }; 4])
            .flat_map(u16::to_le_bytes)
            .collect();
        let factors = buffer_with(device, &factor_bytes).unwrap();
        let culled = buffer_with(device, &[0; 128]).unwrap();
        let shuffled = lane.words(&[1, 3, 0, 2], 0);
        let mut inheriting = IndirectCommandBufferDescriptor::new();
        inheriting.command_types = IndirectCommandType::DRAW_PATCHES;
        inheriting.inherit_buffers = true;
        let icb = icb(device, &inheriting, 1);
        let command = icb.indirect_render_command_at(0).unwrap();
        command.set_render_pipeline_state(&state).unwrap();
        command
            .draw_patches(0, 0, 4, Some(&shuffled), 0, 1, 0, &factors, 0, 0)
            .unwrap();
        let run = |resident: &[&Buffer]| {
            let marks = lane.words(&[0; 4], 0);
            let command_buffer = lane.queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            encoder.set_vertex_buffer(Some(&points), 0, 0).unwrap();
            encoder.set_vertex_buffer(Some(&tags), 0, 1).unwrap();
            encoder.set_vertex_buffer(Some(&marks), 0, 2).unwrap();
            encoder
                .set_tessellation_factor_buffer(Some(&culled), 0, 0)
                .unwrap();
            for buffer in resident {
                encoder
                    .use_resource(
                        buffer.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )
                    .unwrap();
            }
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let bytes = read_buffer(&marks, 0, 16).unwrap();
            let marks: Vec<String> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| format!("{:x}", u32::from_le_bytes(*w)))
                .collect();
            (marks, command_buffer.execution_report().unwrap())
        };
        let drawn = ["a0000012", "a1010345", "0", "a30309ab"];
        let (marks, report) = run(&[&factors, &shuffled]);
        assert_eq!(marks, drawn);
        assert_eq!(report.resident_misses, 0);
        let (marks, report) = run(&[&shuffled]);
        assert_eq!(marks, drawn);
        assert_eq!(report.resident_misses, 0);
        let heap = shared_heap(device);
        let unused = placed(&heap, &factor_bytes);
        command
            .draw_patches(0, 0, 4, Some(&shuffled), 0, 1, 0, &unused, 0, 0)
            .unwrap();
        let (marks, report) = run(&[&shuffled]);
        assert_eq!(marks, ["0"; 4]);
        assert_eq!(report.resident_misses, 1);
    });
}

/// Patch commands a kernel writes with a factor scale (`icb_render_patch_scale`: thread `tid` sets
/// command `tid`'s draw of one patch of the owned `k7_tess_triangle_domain` fixture, its triangle
/// factors 1, scaled by `scales[tid]`), executed one at a time by an encoder whose pipeline state,
/// made with `supportIndirectCommandBuffers` and so never enabling the scale, and buffers the
/// commands inherit. As Metal's (an M4): a scale of 2 marks what factors of 2 mark, 0.5 what
/// factors of 1 mark (integer partitioning rounds 0.5 up to 1), and 0, -1 and NaN mark nothing.
/// Metal rounds the scale to `half` first (the M3 Max and the M4 alike): 1.0004 marks what 1 marks,
/// 1.3 and 2.0005 what 2 marks, 10^-6 and 2^-24 (subnormal `half`s) what factors of 1 mark, 70000
/// and `+∞` (infinite `half`s) what the maximum factor, 64, marks, and 2^-30 (0 as a `half`)
/// nothing. The marks for factors of 1 and 2 are `patches.rs`'s (a Metal probe on an M4), those of
/// the maximum factor its recorded marks (`patches::maximum`; its grid is the
/// fixture's: its marks of factors 1, 2 and 3 are those of `patches.rs`'s). A device without
/// tessellation shaders refuses the pipeline, reported as its refusal.
#[test]
fn patch_commands_scale_their_factors_as_metal_does() {
    on_device(|device, messages| {
        let lane = patches::Lane::new(device.clone());
        let mut descriptor = lane.descriptor(TessellationFactorStepFunction::PerPatch, false);
        descriptor.support_indirect_command_buffers = true;
        let made = device.new_render_pipeline_state(&descriptor, &PipelineOptions::default());
        let state = match made {
            Ok(state) => state,
            Err(Error::Unsupported(u)) => {
                eprintln!(
                    "scaled patch commands on {}: the pipeline is refused, {u:?}",
                    device.name()
                );
                return;
            }
            Err(error) => panic!("the pipeline: {error}"),
        };
        let (none, corners, twos) = (&[][..], patches::CORNERS, patches::TWOS);
        let maximum = patches::maximum();
        let maximum = &maximum[..];
        let cases: [(f32, &[u32]); 15] = [
            (1.0, corners),
            (2.0, twos),
            (0.5, corners),
            (0.0, none),
            (-1.0, none),
            (f32::NAN, none),
            (1.3, twos),
            (1.0004, corners),
            (2.0005, twos),
            (1.0e-6, corners),
            (2f32.powi(-24), corners),
            (2f32.powi(-30), none),
            (64.0, maximum),
            (70000.0, maximum),
            (f32::INFINITY, maximum),
        ];
        let factors = lane.factors(&[patches::ONE]);
        let scales: Vec<u8> = cases
            .iter()
            .flat_map(|(scale, _)| scale.to_le_bytes())
            .collect();
        let scales = lane.buffer(&scales);
        let mut inheriting = IndirectCommandBufferDescriptor::new();
        inheriting.command_types = IndirectCommandType::DRAW_PATCHES;
        inheriting.inherit_pipeline_state = true;
        inheriting.inherit_buffers = true;
        let commands = cases.len() as u64;
        let icb = icb(device, &inheriting, commands);
        let writer = indirect_host_steps::state(device, "icb_render_patch_scale", false);
        let arguments = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut writing = command_buffer.compute_command_encoder().unwrap();
        writing.set_compute_pipeline_state(&writer).unwrap();
        writing.set_buffer(Some(&arguments), 0, 0).unwrap();
        writing.set_buffer(Some(&factors), 0, 1).unwrap();
        writing.set_buffer(Some(&scales), 0, 2).unwrap();
        writing
            .use_resource(icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        let threads = Size::new(commands, 1, 1);
        writing.dispatch_threads(threads, threads).unwrap();
        writing.end_encoding().unwrap();
        complete(&command_buffer);
        for (index, (scale, expected)) in (0..commands).zip(cases) {
            let marks = lane.buffer(&vec![0; patches::MARK_WORDS * 4]);
            let command_buffer = lane
                .pass(&state, &marks, |encoder| {
                    encoder.use_resource(
                        factors.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )?;
                    encoder.execute_commands_in_buffer(&icb, index..index + 1)
                })
                .unwrap();
            if let Some(unlike) = patches::unlike(&patches::cells(&marks), expected) {
                panic!("scale {scale}: {unlike}");
            }
            let report = command_buffer.execution_report().unwrap();
            assert_eq!(
                report.replay_faults,
                ReplayFaults::default(),
                "scale {scale}"
            );
            assert_eq!(report.resident_misses, 0, "scale {scale}");
        }
        let mut first = forced_with(device, std::slice::from_ref(&descriptor));
        // Every command sets a factor scale, which the state, made with
        // `supportIndirectCommandBuffers`, never enables: the execution makes the variant that
        // reads it, whose tessellation stage records `TessellationFactorScale`
        // (`link::tessellation::control`'s notes, `render/executions.rs`).
        first.insert(Degradation::TessellationFactorScale);
        assert_made_once(messages, &HOST_STEPS, &first);
    });
}

/// Patch commands the host encodes read their factors as Metal's did where its validation stops
/// the call (Metal probes, an M4 with macOS 27.0.1, a command and the encoder's draw alike): a
/// factor offset of 2 from that byte, an instance stride 2 past a multiple of 4 by the multiple
/// below, and 33 control points with the pipeline's own; and odd offsets and strides as our probes
/// read them, from the byte below and by the multiple of 4 below. Each
/// command draws one patch of `patches.rs`'s fixture, its factors read per instance from `half`s
/// then 16 bytes of zeros, and marks what `patches.rs`'s matching encoder case marks: instance 0
/// reads an edge of 0 and culls, but at the offset of 2; instance 1 reads at the rounded stride,
/// where a read at the stride itself would draw for 2 and 14 and cull for 6 and 10. A device
/// without tessellation shaders refuses the pipeline, reported as its refusal.
#[test]
fn patch_commands_read_their_factors_as_metal_s_did() {
    patch_commands_read_their_factors_as_metal_s_did_on(Path::Native);
}

/// [`patch_commands_read_their_factors_as_metal_s_did`], emulated.
#[test]
fn emulated_patch_commands_read_their_factors_as_metal_s_did() {
    patch_commands_read_their_factors_as_metal_s_did_on(Path::Emulated);
}

fn patch_commands_read_their_factors_as_metal_s_did_on(path: Path) {
    use patches::{CORNERS, ONE, TWO, TWOS};
    on_device_with(options_of(path), |device, _| {
        let lane = patches::Lane::new(device.clone());
        let mut descriptor = lane.descriptor(TessellationFactorStepFunction::PerInstance, false);
        descriptor.support_indirect_command_buffers = true;
        let made = device.new_render_pipeline_state(&descriptor, &PipelineOptions::default());
        let state = match made {
            Ok(state) => state,
            Err(Error::Unsupported(u)) if path == Path::Native => {
                eprintln!(
                    "patch commands' factors on {}: the pipeline is refused, {u:?}",
                    device.name()
                );
                return;
            }
            Err(error) => panic!("the pipeline: {error}"),
        };
        // Each command's name, control points, factor `half`s, offset, stride and instances,
        // and the cells Metal marked.
        type Command = (
            &'static str,
            u64,
            &'static [u16],
            u64,
            u64,
            u64,
            &'static [u32],
        );
        let commands: [Command; 10] = [
            ("offset_2", 1, &[0, TWO, TWO, TWO, TWO], 2, 0, 1, TWOS),
            ("offset_1", 1, &[0, TWO, TWO, TWO, TWO], 1, 0, 1, &[]),
            ("offset_3", 1, &[0, TWO, TWO, TWO, TWO], 3, 0, 1, TWOS),
            ("stride_7", 1, &[0, TWO, TWO, TWO, TWO, TWO], 0, 7, 2, TWOS),
            (
                "stride_15",
                1,
                &[0, TWO, TWO, TWO, TWO, TWO, 0, TWO, TWO, TWO, TWO],
                0,
                15,
                2,
                &[],
            ),
            (
                "control_points_33",
                33,
                &[ONE, ONE, ONE, ONE],
                0,
                0,
                1,
                CORNERS,
            ),
            ("stride_2", 1, &[0, TWO, TWO, TWO, TWO], 0, 2, 2, &[]),
            ("stride_6", 1, &[0, TWO, TWO, TWO, TWO, TWO], 0, 6, 2, TWOS),
            (
                "stride_10",
                1,
                &[0, TWO, TWO, TWO, TWO, TWO, TWO, TWO],
                0,
                10,
                2,
                TWOS,
            ),
            (
                "stride_14",
                1,
                &[0, TWO, TWO, TWO, TWO, TWO, 0, TWO, TWO, TWO, TWO],
                0,
                14,
                2,
                &[],
            ),
        ];
        let mut inheriting = IndirectCommandBufferDescriptor::new();
        inheriting.command_types = IndirectCommandType::DRAW_PATCHES;
        inheriting.inherit_pipeline_state = true;
        inheriting.inherit_buffers = true;
        let icb = icb(device, &inheriting, commands.len() as u64);
        let mut factors = Vec::new();
        for (index, &(_, points, halves, offset, stride, instances, _)) in
            (0..).zip(commands.iter())
        {
            let mut bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
            bytes.extend([0; 16]);
            let buffer = lane.buffer(&bytes);
            icb.indirect_render_command_at(index)
                .unwrap()
                .draw_patches(points, 0, 1, None, 0, instances, 0, &buffer, offset, stride)
                .unwrap();
            factors.push(buffer);
        }
        for ((index, (name, .., expected)), factors) in (0..).zip(commands).zip(&factors) {
            let marks = lane.buffer(&vec![0; patches::MARK_WORDS * 4]);
            let command_buffer = lane
                .pass(&state, &marks, |encoder| {
                    encoder.use_resource(
                        factors.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )?;
                    encoder.execute_commands_in_buffer(&icb, index..index + 1)
                })
                .unwrap();
            assert_eq!(
                patches::cells(&marks),
                expected,
                "{name} on {}",
                device.name()
            );
            let report = command_buffer.execution_report().unwrap();
            assert_eq!(report.replay_faults, ReplayFaults::default(), "{name}");
            assert_eq!(report.resident_misses, 0, "{name}");
        }
    });
}

/// A buffer of 16 bytes: zero, then the execution range `location`, `length` at 8.
fn range_at_8(device: &Device, location: u32, length: u32) -> Buffer {
    let mut bytes = vec![0u8; 8];
    bytes.extend(location.to_le_bytes());
    bytes.extend(length.to_le_bytes());
    buffer_with(device, &bytes).unwrap()
}

/// `out` after commands `windows` of four, command `w` drawing 4 points over window `w`.
fn drawn_windows(windows: std::ops::Range<usize>) -> Vec<u32> {
    let draws: Vec<Drawn> = windows
        .map(|window| Drawn {
            window,
            start: 0,
            count: 4,
            instances: 1,
            base: 0,
        })
        .collect();
    expected(&draws)
}

/// Four commands, command `w` drawing 4 points over window `w`, run through the range at byte 8
/// of `range`, after `before` encoded what it encodes: the setup and the committed command
/// buffer, completed.
fn run_indirect(
    device: &Device,
    range: &Buffer,
    before: impl FnOnce(&CommandBuffer),
) -> (Setup, CommandBuffer) {
    let setup = Setup::new(device);
    let icb = icb(device, &descriptor(), 4);
    for index in 0..4 {
        setup
            .command(&icb, index, index as usize)
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
    }
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    before(&command_buffer);
    let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
    setup.resident(&mut encoder);
    encoder
        .execute_commands_in_buffer_indirect(&icb, range, 8)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    (setup, command_buffer)
}

/// A render execution reads its range from a buffer as Metal's does (a probe, an M3 Max and an M4):
/// the commands from the location, clamped at the size; nothing for a length of 0 or a location at
/// the size; nothing for a location past the size or a range past `2^32 - 1` (Metal's GPU faults
/// the command buffer), the execution report stating `RANGE`; and the range a blit of an earlier
/// encoder wrote. A host step records the pass, announced once.
#[test]
fn a_render_execution_reads_its_range_from_a_buffer_as_metal_s_does() {
    on_device(|device, messages| {
        for (location, length, ran) in [
            (0u32, 4u32, 0..4),
            (1, 2, 1..3),
            (0, 0, 0..0),
            (2, 0, 2..2),
            (4, 1, 4..4),
            (2, 5, 2..4),
            (0, u32::MAX, 0..4),
        ] {
            let range = range_at_8(device, location, length);
            let (setup, command_buffer) = run_indirect(device, &range, |_| {});
            assert_eq!(setup.output(), drawn_windows(ran), "{location}, {length}");
            let report = command_buffer.execution_report().unwrap();
            assert!(!report.replay_fault, "{location}, {length}");
        }
        for (location, length) in [(5u32, 1u32), (1, u32::MAX)] {
            let range = range_at_8(device, location, length);
            let (setup, command_buffer) = run_indirect(device, &range, |_| {});
            assert_eq!(setup.output(), drawn_windows(0..0), "{location}, {length}");
            let report = command_buffer.execution_report().unwrap();
            assert!(report.replay_fault, "{location}, {length}");
            assert_eq!(report.replay_faults, ReplayFaults::RANGE);
        }
        // The range a blit of an earlier encoder wrote.
        let source = range_at_8(device, 1, 2);
        let range = range_at_8(device, 0, 0);
        let (setup, _) = run_indirect(device, &range, |command_buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&source, 8, &range, 8, 8)
                .unwrap();
            blit.end_encoding().unwrap();
        });
        assert_eq!(setup.output(), drawn_windows(1..3), "a range a blit wrote");
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &[]));
    });
}

/// Refused at the call, recording nothing (a probe, an M3 Max and an M4): a range at an offset
/// that is not a multiple of 4 bytes, which Metal's GPU faults the command buffer over where the
/// range read there runs a command, or whose 8 bytes run past its buffer, which Metal reads from
/// whatever follows it — plain Metal takes both calls, so both are the runtime's own gaps, not
/// misuse; a buffer of compute commands; and a range a draw of the encoder before the call wrote,
/// which Metal's execution reads and the runtime's copy before the pass would not.
#[test]
fn a_render_execution_s_range_is_refused_where_metal_s_read_is_not_the_copy_s() {
    on_device(|device, _| {
        let setup = Setup::new(device);
        let icb = icb(device, &descriptor(), 2);
        let mut compute = IndirectCommandBufferDescriptor::new();
        compute.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        let compute = self::icb(device, &compute, 1);
        let range = range_at_8(device, 0, 2);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        for offset in [2, 6] {
            assert_eq!(
                encoder
                    .execute_commands_in_buffer_indirect(&icb, &range, offset)
                    .err(),
                Some(Error::NotImplemented(RuntimeGap::ExecutionRangeOffFour {
                    offset
                })),
                "offset {offset}"
            );
        }
        for (offset, length) in [(12, 16), (u64::MAX - 3, 16)] {
            assert_eq!(
                encoder
                    .execute_commands_in_buffer_indirect(&icb, &range, offset)
                    .err(),
                Some(Error::NotImplemented(
                    RuntimeGap::ExecutionRangePastBuffer { offset, length }
                )),
                "offset {offset}"
            );
        }
        assert_eq!(
            encoder
                .execute_commands_in_buffer_indirect(&compute, &range, 8)
                .err(),
            Some(Error::NotImplemented(RuntimeGap::IndirectRenderCommands))
        );
        // A draw that writes the range's bytes: the template writes words 0 and 1 of its `out`.
        encoder.set_render_pipeline_state(&setup.state).unwrap();
        encoder
            .set_vertex_buffer(Some(&setup.params), 0, 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(&range), 8, 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Point, 0, 2, 1, 0)
            .unwrap();
        assert_eq!(
            encoder
                .execute_commands_in_buffer_indirect(&icb, &range, 8)
                .err(),
            Some(Error::NotImplemented(RuntimeGap::IndirectRenderCommands))
        );
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(command_buffer.executions().len(), 1);
    });
}

/// One command of [`indexed_patch_commands_draw_as_metal_s_did`]: the pipeline's index type
/// (`None`: no index type), whether it is indexed (with its control-point index buffer's words,
/// 16-bit where `half`, the bytes before them and its offset, `None` for a null buffer) or plain,
/// its control points, patch start and count, and its patch index buffer's words and the bytes
/// before them, its offset.
struct PatchCommand {
    index_type: TessellationControlPointIndexType,
    points: Option<Option<(&'static [u32], bool, u64)>>,
    control_points: u64,
    start: u64,
    count: u64,
    patch_indices: Option<&'static [u32]>,
    patch_pad: u64,
}

impl PatchCommand {
    /// An indexed command of a 32-bit pipeline: two patches through `points` from 0.
    const fn indexed(points: &'static [u32]) -> PatchCommand {
        PatchCommand {
            index_type: TessellationControlPointIndexType::UInt32,
            points: Some(Some((points, false, 0))),
            control_points: 3,
            start: 0,
            count: 2,
            patch_indices: None,
            patch_pad: 0,
        }
    }
}

/// Patch `p`'s control points are those of patch `3 - p` (`patch_indices.rs`'s).
const REVERSED: &[u32] = &[9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2];

/// Indexed patch commands draw as Metal's did (probes, an M4), the `rt6_patch_index` fixture's
/// marks those Metal stored: control point `k` of the patch whose index is `p` is the index at
/// `3p + k` of the control-point index buffer from its offset, of the pipeline's index type, its 3
/// control points whatever the command names (4 here); with a patch start, a patch index buffer, a
/// culled patch; an index past the buffer's end reads 0, and so does every index of a null buffer,
/// one at the buffer's end, and a plain patch command's on a pipeline with an index type; an
/// indexed command on a pipeline without one draws the patches' own control points; a plain command
/// and an indexed one execute side by side. The commands draw with their own factor buffer, never
/// the encoder's (whose factors of 0 would cull every patch). An offset that is not a multiple of
/// the index size reads the indices from the offset rounded down to a multiple of it (an M3 Max and
/// an M4 alike), and a patch index offset that is not a multiple of 4 reads them from the offset
/// rounded down to one. A control-point index buffer of its own that no encoder made resident is
/// reached; one placed in a heap no encoder of the command buffer used misses. A device without
/// tessellation shaders refuses the pipeline, reported as its refusal.
#[test]
fn indexed_patch_commands_draw_as_metal_s_did() {
    indexed_patch_commands_draw_as_metal_s_did_on(Path::Native);
}

/// [`indexed_patch_commands_draw_as_metal_s_did`], emulated.
#[test]
fn emulated_indexed_patch_commands_draw_as_metal_s_did() {
    indexed_patch_commands_draw_as_metal_s_did_on(Path::Emulated);
}

fn indexed_patch_commands_draw_as_metal_s_did_on(path: Path) {
    use TessellationControlPointIndexType as T;
    on_device_with(options_of(path), |device, _| {
        let lane = Lane::new(device.clone());
        let mut states = Vec::new();
        for index_type in [T::None, T::UInt16, T::UInt32] {
            let mut descriptor = lane.descriptor(index_type);
            descriptor.support_indirect_command_buffers = true;
            match device.new_render_pipeline_state(&descriptor, &PipelineOptions::default()) {
                Ok(state) => states.push((index_type, state)),
                Err(Error::Unsupported(u)) if path == Path::Native => {
                    eprintln!(
                        "indexed patch commands on {}: the pipeline is refused, {u:?}",
                        device.name()
                    );
                    return;
                }
                Err(error) => panic!("the pipeline: {error}"),
            }
        }
        let state = |index_type| {
            states
                .iter()
                .find(|(t, _)| *t == index_type)
                .map(|(_, state)| state.clone())
                .unwrap()
        };
        let points = lane.words(&(0..12).collect::<Vec<u32>>(), 0);
        let tags = lane.words(&(0..16).map(|p| 0xa0 + p).collect::<Vec<u32>>(), 0);
        // Half factors (three edges, one inside) of 1 per patch, but patch 3's 0; the
        // encoder's are all 0.
        let factors: Vec<u8> = (0..16)
            .flat_map(|p| [if p == 3 { 0u16 } else { 0x3c00 }; 4])
            .flat_map(u16::to_le_bytes)
            .collect();
        let factors = buffer_with(device, &factors).unwrap();
        let culled = buffer_with(device, &[0; 128]).unwrap();
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types =
            IndirectCommandType::DRAW_PATCHES.union(IndirectCommandType::DRAW_INDEXED_PATCHES);
        descriptor.inherit_buffers = true;
        // Run `commands` in one execution, `resident` saying whether their buffers are made
        // resident, their control-point index buffers placed in `heap` where it is given: the
        // marks and the execution report.
        let run = |commands: &[PatchCommand], resident: bool, heap: Option<&Heap>| {
            let icb = icb(device, &descriptor, commands.len() as u64);
            let mut buffers = vec![factors.clone()];
            for (index, c) in commands.iter().enumerate() {
                let command = icb.indirect_render_command_at(index as u64).unwrap();
                command
                    .set_render_pipeline_state(&state(c.index_type))
                    .unwrap();
                let patch_indices = c.patch_indices.map(|words| lane.words(words, c.patch_pad));
                buffers.extend(patch_indices.clone());
                match c.points {
                    None => command
                        .draw_patches(
                            c.control_points,
                            c.start,
                            c.count,
                            patch_indices.as_ref(),
                            c.patch_pad,
                            1,
                            0,
                            &factors,
                            0,
                            0,
                        )
                        .unwrap(),
                    Some(points) => {
                        let points = points.map(|(words, half, pad)| {
                            let mut bytes = vec![0u8; pad as usize];
                            for &word in words {
                                if half {
                                    bytes.extend((word as u16).to_le_bytes());
                                } else {
                                    bytes.extend(word.to_le_bytes());
                                }
                            }
                            let buffer = match heap {
                                Some(heap) => placed(heap, &bytes),
                                None => buffer_with(device, &bytes).unwrap(),
                            };
                            (buffer, pad)
                        });
                        command
                            .draw_indexed_patches(
                                c.control_points,
                                c.start,
                                c.count,
                                patch_indices.as_ref(),
                                c.patch_pad,
                                points.as_ref().map(|(b, _)| b),
                                points.as_ref().map_or(0, |(_, pad)| *pad),
                                1,
                                0,
                                &factors,
                                0,
                                0,
                            )
                            .unwrap();
                        buffers.extend(points.map(|(b, _)| b));
                    }
                }
            }
            let marks = lane.words(&[0; 4], 0);
            let command_buffer = lane.queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            encoder.set_vertex_buffer(Some(&points), 0, 0).unwrap();
            encoder.set_vertex_buffer(Some(&tags), 0, 1).unwrap();
            encoder.set_vertex_buffer(Some(&marks), 0, 2).unwrap();
            encoder
                .set_tessellation_factor_buffer(Some(&culled), 0, 0)
                .unwrap();
            for buffer in buffers.iter().filter(|_| resident) {
                encoder
                    .use_resource(
                        buffer.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )
                    .unwrap();
            }
            if !resident {
                encoder
                    .use_resource(
                        factors.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )
                    .unwrap();
            }
            encoder
                .execute_commands_in_buffer(&icb, 0..commands.len() as u64)
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let bytes = read_buffer(&marks, 0, 16).unwrap();
            let marks: Vec<String> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| format!("{:x}", u32::from_le_bytes(*w)))
                .collect();
            (marks, command_buffer.execution_report().unwrap())
        };
        let cases: Vec<(&str, Vec<PatchCommand>, [&str; 4])> = vec![
            (
                "u32",
                vec![PatchCommand::indexed(REVERSED)],
                ["0", "0", "a1010678", "a00009ab"],
            ),
            (
                "u16 at 2",
                vec![PatchCommand {
                    index_type: T::UInt16,
                    points: Some(Some((REVERSED, true, 2))),
                    ..PatchCommand::indexed(REVERSED)
                }],
                ["0", "0", "a1010678", "a00009ab"],
            ),
            (
                "the command's 4 control points",
                vec![PatchCommand {
                    control_points: 4,
                    ..PatchCommand::indexed(REVERSED)
                }],
                ["0", "0", "a1010678", "a00009ab"],
            ),
            (
                "start 1, patch 3 culled",
                vec![PatchCommand {
                    start: 1,
                    count: 3,
                    ..PatchCommand::indexed(REVERSED)
                }],
                ["0", "a2020345", "a1010678", "0"],
            ),
            (
                "patch indices",
                vec![PatchCommand {
                    patch_indices: Some(&[3, 1]),
                    ..PatchCommand::indexed(REVERSED)
                }],
                ["a3030012", "0", "a1010678", "0"],
            ),
            (
                "an index past the buffer",
                vec![PatchCommand::indexed(&[9, 10, 11, 6, 7])],
                ["0", "0", "a1010670", "a00009ab"],
            ),
            (
                "no buffer",
                vec![PatchCommand {
                    points: Some(None),
                    start: 1,
                    count: 1,
                    ..PatchCommand::indexed(REVERSED)
                }],
                ["a1010000", "0", "0", "0"],
            ),
            (
                "at the buffer's end",
                vec![PatchCommand {
                    points: Some(Some((&[], false, 48))),
                    count: 1,
                    ..PatchCommand::indexed(REVERSED)
                }],
                ["a0000000", "0", "0", "0"],
            ),
            (
                "a pipeline without an index type",
                vec![PatchCommand {
                    index_type: T::None,
                    ..PatchCommand::indexed(REVERSED)
                }],
                ["a0000012", "a1010345", "0", "0"],
            ),
            (
                "a plain command on a 32-bit pipeline",
                vec![PatchCommand {
                    points: None,
                    start: 1,
                    count: 1,
                    ..PatchCommand::indexed(REVERSED)
                }],
                ["a1010000", "0", "0", "0"],
            ),
            (
                "plain and indexed",
                vec![
                    PatchCommand {
                        index_type: T::None,
                        points: None,
                        start: 2,
                        count: 1,
                        ..PatchCommand::indexed(REVERSED)
                    },
                    PatchCommand {
                        count: 1,
                        ..PatchCommand::indexed(REVERSED)
                    },
                ],
                ["0", "0", "a2020678", "a00009ab"],
            ),
            // Patch indices at an offset of 2, which Metal's validation stops and plain Metal
            // reads from offset 0: after the 2 zero bytes, the words 0x2_0000 and 0x1_0000 read as
            // patches 0 and 2, where read at the offset itself they would name patches past the
            // buffers.
            (
                "patch indices at 2",
                vec![PatchCommand {
                    index_type: T::None,
                    points: None,
                    patch_indices: Some(&[0x2_0000, 0x1_0000]),
                    patch_pad: 2,
                    ..PatchCommand::indexed(REVERSED)
                }],
                ["a0000012", "0", "a2020678", "0"],
            ),
        ];
        for (name, commands, expected) in cases {
            let (marks, report) = run(&commands, true, None);
            assert_eq!(marks, expected, "{name}");
            assert!(!report.replay_fault, "{name}");
            assert_eq!(report.resident_misses, 0, "{name}");
        }
        // An offset that is not a multiple of the index size: the 32-bit indices 0, 9, 10, 11, 6,
        // 7 and 8 written as 16-bit halves after 2 bytes of zeros, from offset 2, which Metal
        // reads from offset 0 (the offset rounded down to a multiple of 4), where read unaligned
        // they would be out of the points' range.
        let misaligned = PatchCommand {
            points: Some(Some((&[0, 9, 0, 10, 0, 11, 0, 6, 0, 7, 0, 8], true, 2))),
            ..PatchCommand::indexed(REVERSED)
        };
        let (marks, report) = run(&[misaligned], true, None);
        assert_eq!(marks, ["a000009a", "0", "0", "a1010b67"]);
        assert!(!report.replay_fault);
        assert_eq!(report.resident_misses, 0);
        // A control-point index buffer of its own that no encoder made resident is reached; one
        // placed in a heap no encoder used misses.
        let (marks, report) = run(&[PatchCommand::indexed(REVERSED)], false, None);
        assert_eq!(marks, ["0", "0", "a1010678", "a00009ab"]);
        assert_eq!(report.resident_misses, 0);
        let heap = shared_heap(device);
        let (marks, report) = run(&[PatchCommand::indexed(REVERSED)], false, Some(&heap));
        assert_eq!(marks, ["0"; 4]);
        assert_eq!(report.resident_misses, 1);
    });
}

/// One case of indexed patch commands a kernel writes: its name, its pipeline's control-point
/// index type, the command (the fixture's thread) it runs, the control-point index bytes, the
/// patch indices, its patch start and count, the offsets of its patch indices (in `uint`s) and
/// control-point indices (in bytes), and the marks Metal stored.
struct KernelPatchCase {
    name: &'static str,
    index_type: TessellationControlPointIndexType,
    command: u64,
    points: Vec<u8>,
    patch_indices: &'static [u32],
    start: u32,
    count: u32,
    patch_offset: u32,
    point_offset: u32,
    marks: [u32; 4],
}

impl KernelPatchCase {
    /// A case of thread 2's form (no patch index buffer, device control-point indices) on a
    /// 32-bit pipeline: two patches through `points` from 0.
    fn indexed(name: &'static str, points: &[u32], marks: [u32; 4]) -> KernelPatchCase {
        KernelPatchCase {
            name,
            index_type: TessellationControlPointIndexType::UInt32,
            command: 2,
            points: points.iter().flat_map(|p| p.to_le_bytes()).collect(),
            patch_indices: &[0],
            start: 0,
            count: 2,
            patch_offset: 0,
            point_offset: 0,
            marks,
        }
    }
}

/// Indexed patch commands a kernel writes (`icb_render_ipatch_encode`: thread 0's patch indices,
/// control-point indices and factors in device memory, thread 1's in constant memory, thread 2
/// with no patch index buffer, thread 3 with neither index buffer) into an inheriting buffer of
/// indexed patch draws draw as Metal drew them (the M3 Max and the M4 alike, each as the host's
/// command of the same values): the `rt6_patch_index` fixture's marks for 32- and 16-bit
/// control-point indices, their offsets (one not a multiple of the index size, read from the offset
/// rounded down to one), a patch start, a culled patch, patch indices from an offset, constant
/// memory, a null control-point index buffer, an index past the buffer's end, and a pipeline
/// without an index type. A host step records the commands, announced once. A device without
/// tessellation shaders refuses the pipeline, reported as its refusal.
#[test]
fn indexed_patch_commands_a_kernel_writes_draw_as_metal_s_did() {
    use TessellationControlPointIndexType as T;
    on_device(|device, messages| {
        let lane = Lane::new(device.clone());
        let mut states = Vec::new();
        for index_type in [T::None, T::UInt16, T::UInt32] {
            let mut descriptor = lane.descriptor(index_type);
            descriptor.support_indirect_command_buffers = true;
            match device.new_render_pipeline_state(&descriptor, &PipelineOptions::default()) {
                Ok(state) => states.push((index_type, state)),
                Err(Error::Unsupported(u)) => {
                    eprintln!(
                        "kernel indexed patch commands on {}: the pipeline is refused, {u:?}",
                        device.name()
                    );
                    return;
                }
                Err(error) => panic!("the pipeline: {error}"),
            }
        }
        let halves = |indices: &[u32], pad: usize| -> Vec<u8> {
            let mut bytes = vec![0u8; pad];
            bytes.extend(indices.iter().flat_map(|&i| (i as u16).to_le_bytes()));
            bytes
        };
        let words = |indices: &[u32], pad: usize| -> Vec<u8> {
            let mut bytes = vec![0u8; pad];
            bytes.extend(indices.iter().flat_map(|i| i.to_le_bytes()));
            bytes
        };
        let both = [0, 0, 0xa101_0678, 0xa000_09ab];
        let cases = [
            KernelPatchCase::indexed("u32", REVERSED, both),
            KernelPatchCase {
                index_type: T::UInt16,
                points: halves(REVERSED, 0),
                ..KernelPatchCase::indexed("u16", REVERSED, both)
            },
            KernelPatchCase {
                points: words(REVERSED, 8),
                point_offset: 8,
                ..KernelPatchCase::indexed("u32 at 8", REVERSED, both)
            },
            KernelPatchCase {
                index_type: T::UInt16,
                points: halves(REVERSED, 2),
                point_offset: 2,
                ..KernelPatchCase::indexed("u16 at 2", REVERSED, both)
            },
            // Offsets that are not a multiple of the index size read from the offset rounded down
            // to one: 32-bit indices after 4 bytes of zeros at 2 read 0, 9, 10, 11, 6, 7, and at 6
            // the indices; 16-bit after 2 bytes at 3 read the indices. Read unaligned, each would
            // be out of the points' range.
            KernelPatchCase {
                points: words(REVERSED, 4),
                point_offset: 2,
                ..KernelPatchCase::indexed("u32 at 2", REVERSED, [0xa000_009a, 0, 0, 0xa101_0b67])
            },
            KernelPatchCase {
                points: words(REVERSED, 4),
                point_offset: 6,
                ..KernelPatchCase::indexed("u32 at 6", REVERSED, both)
            },
            KernelPatchCase {
                index_type: T::UInt16,
                points: halves(REVERSED, 2),
                point_offset: 3,
                ..KernelPatchCase::indexed("u16 at 3", REVERSED, both)
            },
            KernelPatchCase {
                start: 1,
                ..KernelPatchCase::indexed("start 1", REVERSED, [0, 0xa202_0345, 0xa101_0678, 0])
            },
            KernelPatchCase {
                start: 1,
                count: 3,
                ..KernelPatchCase::indexed(
                    "start 1, patch 3 culled",
                    REVERSED,
                    [0, 0xa202_0345, 0xa101_0678, 0],
                )
            },
            KernelPatchCase {
                command: 0,
                patch_indices: &[3, 1],
                ..KernelPatchCase::indexed(
                    "patch indices",
                    REVERSED,
                    [0xa303_0012, 0, 0xa101_0678, 0],
                )
            },
            KernelPatchCase {
                command: 0,
                patch_indices: &[9, 2, 3, 1],
                patch_offset: 1,
                start: 1,
                ..KernelPatchCase::indexed(
                    "patch indices from 1, start 1",
                    REVERSED,
                    [0xa303_0012, 0, 0xa101_0678, 0],
                )
            },
            KernelPatchCase {
                command: 1,
                patch_indices: &[3, 1],
                ..KernelPatchCase::indexed(
                    "constant memory",
                    REVERSED,
                    [0xa303_0012, 0, 0xa101_0678, 0],
                )
            },
            KernelPatchCase {
                command: 3,
                count: 1,
                ..KernelPatchCase::indexed(
                    "no control-point index buffer",
                    REVERSED,
                    [0xa000_0000, 0, 0, 0],
                )
            },
            KernelPatchCase::indexed(
                "an index past the buffer",
                &[9, 10, 11, 6, 7],
                [0, 0, 0xa101_0670, 0xa000_09ab],
            ),
            KernelPatchCase {
                index_type: T::None,
                ..KernelPatchCase::indexed(
                    "a pipeline without an index type",
                    REVERSED,
                    [0xa000_0012, 0xa101_0345, 0, 0],
                )
            },
        ];
        let writer = indirect_host_steps::state(device, "icb_render_ipatch_encode", false);
        let points = lane.words(&(0..12).collect::<Vec<u32>>(), 0);
        let tags = lane.words(&(0..16).map(|p| 0xa0 + p).collect::<Vec<u32>>(), 0);
        // Half factors (three edges, one inside) of 1 per patch, but patch 3's 0; the
        // encoder's are all 0.
        let factors: Vec<u8> = (0..16)
            .flat_map(|p| [if p == 3 { 0u16 } else { 0x3c00 }; 4])
            .flat_map(u16::to_le_bytes)
            .collect();
        let factors = buffer_with(device, &factors).unwrap();
        let culled = buffer_with(device, &[0; 128]).unwrap();
        let unread = buffer_with(device, &[0; 16]).unwrap();
        let mut inheriting = IndirectCommandBufferDescriptor::new();
        inheriting.command_types = IndirectCommandType::DRAW_INDEXED_PATCHES;
        inheriting.inherit_pipeline_state = true;
        inheriting.inherit_buffers = true;
        for case in &cases {
            let name = case.name;
            // Every thread's row: 3 control points, the case's patches, one instance from 0, the
            // case's offsets, the factors from structure 0, a scale of 1.
            let row: Vec<u8> = [
                3,
                case.start,
                case.count,
                1,
                0,
                case.patch_offset,
                case.point_offset,
                0,
                1.0f32.to_bits(),
            ]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
            let draws = buffer_with(device, &row.repeat(4)).unwrap();
            let patch_indices = lane.words(case.patch_indices, 0);
            let point_indices = buffer_with(device, &case.points).unwrap();
            let icb = icb(device, &inheriting, 4);
            let arguments = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
            let command_buffer = lane.queue.command_buffer().unwrap();
            let mut writing = command_buffer.compute_command_encoder().unwrap();
            writing.set_compute_pipeline_state(&writer).unwrap();
            for (index, buffer) in [
                &arguments,
                &patch_indices,
                &point_indices,
                &factors,
                &draws,
                &patch_indices,
                &point_indices,
                &factors,
                &unread,
            ]
            .into_iter()
            .enumerate()
            {
                writing.set_buffer(Some(buffer), 0, index as u32).unwrap();
            }
            writing
                .use_resource(icb.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            let threads = Size::new(4, 1, 1);
            writing.dispatch_threads(threads, threads).unwrap();
            writing.end_encoding().unwrap();
            complete(&command_buffer);
            let state = states
                .iter()
                .find(|(t, _)| *t == case.index_type)
                .map(|(_, state)| state)
                .unwrap();
            let marks = lane.words(&[0; 4], 0);
            let command_buffer = lane.queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            encoder.set_render_pipeline_state(state).unwrap();
            encoder.set_vertex_buffer(Some(&points), 0, 0).unwrap();
            encoder.set_vertex_buffer(Some(&tags), 0, 1).unwrap();
            encoder.set_vertex_buffer(Some(&marks), 0, 2).unwrap();
            encoder
                .set_tessellation_factor_buffer(Some(&culled), 0, 0)
                .unwrap();
            for buffer in [&factors, &patch_indices, &point_indices] {
                encoder
                    .use_resource(
                        buffer.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )
                    .unwrap();
            }
            encoder
                .execute_commands_in_buffer(&icb, case.command..case.command + 1)
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "{name}"
            );
            let bytes = read_buffer(&marks, 0, 16).unwrap();
            let got: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            assert_eq!(got, case.marks, "{name}");
            let report = command_buffer.execution_report().unwrap();
            assert!(!report.replay_fault, "{name}");
            assert_eq!(report.resident_misses, 0, "{name}");
        }
        let made: Vec<_> = [T::None, T::UInt16, T::UInt32]
            .into_iter()
            .map(|index_type| {
                let mut descriptor = lane.descriptor(index_type);
                descriptor.support_indirect_command_buffers = true;
                descriptor
            })
            .collect();
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &made));
    });
}

/// The template's state with a color attachment of `format` (rasterization disabled, as the
/// template's).
fn colored_template_state(device: &Device, format: PixelFormat) -> RenderPipelineState {
    let mut descriptor = render_template_descriptor(device);
    descriptor.support_indirect_command_buffers = true;
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A command whose pipeline state the pass refuses (an `RGBA8Unorm` color in a `BGRA8Unorm`
/// pass, as `set_render_pipeline_state` refuses it) is refused with one typed error wherever its
/// records are recorded: `commit` returns it where the CPU wrote them (the call checks no
/// record, as Metal lets the CPU write them until the commit); where a blit reset them on the
/// GPU and the CPU wrote them again, the host step that records them fails the command buffer
/// `CommandBufferError::Refused`, `refusal` holding the same error, never `Internal`.
#[test]
fn a_command_the_pass_refuses_is_refused_alike_at_commit_and_host_step() {
    on_device(|device, messages| {
        let state = colored_template_state(device, PixelFormat::RGBA8Unorm);
        let target = texture(device, PixelFormat::BGRA8Unorm, 1);
        let mut pass = pass();
        pass.color_attachments[0].texture = Some(target);
        // Cleared: a `DontCare` load of a pass that draws announces its tiles' near-equivalent
        // (`dont-care-load-tiles`, `clear_reach`'s lanes), which the refusal here does not test.
        pass.color_attachments[0].load_action = LoadAction::Clear;
        let icb = icb(device, &descriptor(), 1);
        let encode = || {
            let command = icb.indirect_render_command_at(0).unwrap();
            command.set_render_pipeline_state(&state).unwrap();
            command
                .draw_primitives(PrimitiveType::Point, 0, 1, 1, 0)
                .unwrap();
        };
        let refusal = Error::Invalid(Invalid::new(
            CALL,
            Problem::Draw(DrawProblem::AttachmentFormat(Attachment::Color(0))),
        ));
        let queue = device.new_command_queue().unwrap();

        encode();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        assert_eq!(command_buffer.commit().err(), Some(refusal.clone()));

        let resetting = queue.command_buffer().unwrap();
        let mut blit = resetting.blit_command_encoder().unwrap();
        blit.reset_commands_in_buffer(&icb, 0..1).unwrap();
        blit.end_encoding().unwrap();
        complete(&resetting);
        encode();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert_eq!(command_buffer.error(), Some(CommandBufferError::Refused));
        assert_eq!(command_buffer.refusal(), Some(refusal));
        // It makes no `Shared` buffer (the indirect command buffer's records are not one), so
        // nothing is announced in system memory on any device.
        assert_made_once(messages, &HOST_STEPS, &std::collections::BTreeSet::new());
    });
}

/// A command buffer that fails before its last batch is submitted runs no submission handler and
/// no scheduled handler: its first batch (a fill, then an event signal) is submitted at the
/// commit, and the host step of its last (an execution whose command the pass refuses, as above)
/// fails it `Refused` before that batch is submitted. Its completed handler runs, seeing `Error`.
#[test]
fn a_command_buffer_that_fails_before_its_last_batch_runs_no_submission_handler() {
    on_device(|device, messages| {
        let state = colored_template_state(device, PixelFormat::RGBA8Unorm);
        let target = texture(device, PixelFormat::BGRA8Unorm, 1);
        let mut pass = pass();
        pass.color_attachments[0].texture = Some(target);
        // Cleared: a `DontCare` load of a pass that draws announces its tiles' near-equivalent
        // (`dont-care-load-tiles`, `clear_reach`'s lanes), which the refusal here does not test.
        pass.color_attachments[0].load_action = LoadAction::Clear;
        let icb = icb(device, &descriptor(), 1);
        let queue = device.new_command_queue().unwrap();
        // The CPU writes the records after a blit reset them on the GPU: a host step records them.
        let resetting = queue.command_buffer().unwrap();
        let mut blit = resetting.blit_command_encoder().unwrap();
        blit.reset_commands_in_buffer(&icb, 0..1).unwrap();
        blit.end_encoding().unwrap();
        complete(&resetting);
        let command = icb.indirect_render_command_at(0).unwrap();
        command.set_render_pipeline_state(&state).unwrap();
        command
            .draw_primitives(PrimitiveType::Point, 0, 1, 1, 0)
            .unwrap();
        let event = device.new_shared_event().unwrap();
        let scratch = device.new_buffer(256, ResourceOptions::private()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&scratch, 0..256, 1).unwrap();
        blit.end_encoding().unwrap();
        command_buffer
            .encode_signal_event(EventRef::from(&event), 1)
            .unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let note = |name: &'static str| {
            let calls = std::sync::Arc::clone(&calls);
            Box::new(move |own: &CommandBuffer| calls.lock().unwrap().push((name, own.status())))
        };
        command_buffer
            .add_submission_handler(note("submitted"))
            .unwrap();
        command_buffer
            .add_scheduled_handler(note("scheduled"))
            .unwrap();
        command_buffer
            .add_completed_handler(note("completed"))
            .unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert_eq!(command_buffer.error(), Some(CommandBufferError::Refused));
        // Its first batch was submitted and ran: its signal fired.
        assert!(event.wait_until_signaled_value(1, WAIT));
        assert_eq!(
            *calls.lock().unwrap(),
            [("completed", CommandBufferStatus::Error)]
        );
        // Its buffer is private and its records are not a `Shared` buffer: nothing is announced
        // in system memory on any device.
        assert_made_once(messages, &HOST_STEPS, &std::collections::BTreeSet::new());
    });
}

/// The host step, in the form Metal allows a state that supports indirect command buffers
/// (`teardown_holders::host_step_s_shared_set`): a host step that draws a state made after its
/// command buffer's commit, whose fragment function samples its argument buffer's texture through a
/// constexpr sampler, shares the sampler set the state's creation made and makes none of its own
/// (`WorkCounts::host_step_sampler_sets_made` stays). Metal refuses the state's direct texture
/// argument that once reached the step's own set, and the runtime refuses a
/// mapped texture an argument buffer can reach (`MappedSampling::Handles`), so no step's variant
/// bakes a sampler the state's did not. The draw writes the source's texels, under the validation
/// layer, announcing the host step once.
#[test]
fn a_host_step_s_variant_of_a_state_made_after_the_commit_shares_the_state_s_sampler_set() {
    on_device(|device, messages| {
        let (state, command_buffer, target, texels) =
            super::teardown_holders::host_step_s_shared_set(device);
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let mut written = vec![0u8; texels.len()];
        target
            .get_bytes(
                &mut written,
                8 * 4,
                8 * 8 * 4,
                Region::new_2d(0, 0, 8, 8),
                0,
                0,
            )
            .unwrap();
        assert_eq!(written, texels, "each pixel sampled its own texel");
        let made = [super::teardown_holders::shared_set_descriptor(device)];
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &made));
        drop(state);
    });
}

/// A host step that draws a state made after its command buffer's commit, whose fragment function
/// samples a texture through a sampler state its argument buffer holds (the committed public case
/// `render_arguments::SAMPLED`), binds the device's neutrals its plan names (its set 1's heap
/// array's neutral element and its argument samplers' neutrals), which the state's creation made
/// on the program's thread: the step, on the steps' thread inside the exit guard's section,
/// makes no image, memory or submission of its own. The buffer's records are the GPU's to write (an earlier blit copied one),
/// so its execution is a host step's; the command buffer waits for an event, and once the commit
/// returned the test makes the state, encodes command 0 with it and the argument buffer on the
/// CPU, and signals. The draw writes Metal's texels, under the validation layer, announcing the
/// host step once.
#[test]
fn a_host_step_s_draw_of_a_state_made_after_the_commit_binds_a_neutral_made_before_it() {
    let case = public_render(SAMPLED);
    let metal = case.metal_output.clone().unwrap();
    on_device(|device, messages| {
        let render = &case.render;
        let fragment = render.fragment.as_ref().expect("a fragment case");
        let members = Members::of(device, fragment);
        let fragment_buffers = members.owners.keys().map(|&index| index + 1).max();
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW;
        commands.max_fragment_buffer_bind_count = fragment_buffers.unwrap_or(0);
        let icb = icb(device, &commands, 2);
        let queue = device.new_command_queue().unwrap();
        // The GPU writes the records: a blit copies command 0 to command 1.
        let copying = queue.command_buffer().unwrap();
        let mut blit = copying.blit_command_encoder().unwrap();
        blit.copy_indirect_command_buffer(&icb, 0..1, &icb, 1)
            .unwrap();
        blit.end_encoding().unwrap();
        complete(&copying);
        let color = &render.color_targets[&0];
        let [width, height] = color.dimensions;
        let mut target = TextureDescriptor::texture_2d(
            pixel_format(color.format),
            width.into(),
            height.into(),
            false,
        );
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let target = device.new_texture(&target).unwrap();
        let row = u64::from(width) * color.format.bytes_per_pixel() as u64;
        let region = Region::new_2d(0, 0, width.into(), height.into());
        target
            .replace_region(region, 0, 0, &color.bytes, row, row * u64::from(height))
            .unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder
            .use_resources(
                &members.resources(),
                ResourceUsage::READ,
                RenderStages::FRAGMENT,
            )
            .unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        // After the commit: the state, and command 0 drawing with it.
        let mut descriptor = render_pipeline_descriptor(device, &case.inputs()).unwrap();
        descriptor.support_indirect_command_buffers = true;
        let state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let command = icb.indirect_render_command_at(0).unwrap();
        command.set_render_pipeline_state(&state).unwrap();
        for (&index, owner) in &members.owners {
            command.set_fragment_buffer(Some(owner), 0, index).unwrap();
        }
        let DrawCall::Primitives {
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } = render.draw
        else {
            panic!("{SAMPLED}: a draw of primitives");
        };
        let &Assembly::Primitives(primitive) = &render.assembly else {
            panic!("{SAMPLED}: a draw of primitives");
        };
        command
            .draw_primitives(
                metal_primitive_type(primitive),
                vertex_start.into(),
                vertex_count.into(),
                instance_count.into(),
                base_instance.into(),
            )
            .unwrap();
        event.set_signaled_value(1);
        indirect_host_steps::wait(&command_buffer);
        assert_eq!(output(&case, &target, &members), metal, "Metal's texels");
        let made = [sampled_descriptor(device, &case)];
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &made));
        drop(state);
    });
}

/// The most sampler states [`a_host_step_s_draw_past_its_sampler_arrays_runs_with_the_states_it_lists`]
/// makes to crowd its variant's set-1 sampler arrays (`resource_ids`' crowd).
const HOST_CROWD: u64 = 2048;

/// A command a host step records after its command buffer's commit that leaves a live sampler
/// state out of its argument samplers, more states living than its variant's set-1 sampler arrays
/// hold (`heap_sets::fit_samplers`), runs with the states it lists, announced once (`UNLISTED`)
/// and counted once
/// (`WorkCounts::unlisted_argument_sampler_buffers`): the command buffer completes, and a function
/// reading a state left out would read the neutral sampler. The host step draws, as in
/// [`a_host_step_s_draw_of_a_state_made_after_the_commit_binds_a_neutral_made_before_it`], a state
/// made after the commit whose fragment function samples a texture through a sampler state its
/// argument buffer holds (`render_arguments::SAMPLED`). With 20 more states made after the commit,
/// within the arrays, the draw writes Metal's texels and nothing fails or is counted. The line
/// `SAMPLER-CAPACITY` reports the step's variant's arrays; where they hold at most [`HOST_CROWD`]
/// elements, as many more states made after the commit leave one out, and the draw, whose state
/// was made before them and is listed, still writes Metal's texels. Where they hold more (65536
/// on lavapipe and the M4), filling them takes more Vulkan samplers than lavapipe allows
/// (`maxSamplerAllocationCount` 32768), so that half runs only on such a device.
#[test]
fn a_host_step_s_draw_past_its_sampler_arrays_runs_with_the_states_it_lists() {
    let case = public_render(SAMPLED);
    let metal = case.metal_output.clone().unwrap();
    on_device(|device, messages| {
        let unlisted = || device.work_counts().unlisted_argument_sampler_buffers;
        let before = unlisted();
        let render = &case.render;
        let fragment = render.fragment.as_ref().expect("a fragment case");
        let members = Members::of(device, fragment);
        let fragment_buffers = members.owners.keys().map(|&index| index + 1).max();
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW;
        commands.max_fragment_buffer_bind_count = fragment_buffers.unwrap_or(0);
        let queue = device.new_command_queue().unwrap();
        let color = &render.color_targets[&0];
        let [width, height] = color.dimensions;
        let row = u64::from(width) * color.format.bytes_per_pixel() as u64;
        let region = Region::new_2d(0, 0, width.into(), height.into());
        let DrawCall::Primitives {
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } = render.draw
        else {
            panic!("{SAMPLED}: a draw of primitives");
        };
        let &Assembly::Primitives(primitive) = &render.assembly else {
            panic!("{SAMPLED}: a draw of primitives");
        };
        // A host step's draw of a state made after the commit, `more` sampler states made after
        // the commit too: its command buffer, once completed, and its target.
        let step = |more: u64| {
            let icb = icb(device, &commands, 2);
            // The GPU writes the records: a blit copies command 0 to command 1.
            let copying = queue.command_buffer().unwrap();
            let mut blit = copying.blit_command_encoder().unwrap();
            blit.copy_indirect_command_buffer(&icb, 0..1, &icb, 1)
                .unwrap();
            blit.end_encoding().unwrap();
            complete(&copying);
            let mut target = TextureDescriptor::texture_2d(
                pixel_format(color.format),
                width.into(),
                height.into(),
                false,
            );
            target.usage = TextureUsage::RENDER_TARGET;
            target.resource_options = ResourceOptions::shared();
            let target = device.new_texture(&target).unwrap();
            target
                .replace_region(region, 0, 0, &color.bytes, row, row * u64::from(height))
                .unwrap();
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let event = device.new_shared_event().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder
                .use_resources(
                    &members.resources(),
                    ResourceUsage::READ,
                    RenderStages::FRAGMENT,
                )
                .unwrap();
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            // After the commit: the state, the sampler states, and command 0 drawing.
            let mut descriptor = render_pipeline_descriptor(device, &case.inputs()).unwrap();
            descriptor.support_indirect_command_buffers = true;
            let state = device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap();
            let late = distinct_sampler_states(device, more);
            let command = icb.indirect_render_command_at(0).unwrap();
            command.set_render_pipeline_state(&state).unwrap();
            for (&index, owner) in &members.owners {
                command.set_fragment_buffer(Some(owner), 0, index).unwrap();
            }
            command
                .draw_primitives(
                    metal_primitive_type(primitive),
                    vertex_start.into(),
                    vertex_count.into(),
                    instance_count.into(),
                    base_instance.into(),
                )
                .unwrap();
            event.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            drop((late, state));
            (command_buffer, target)
        };
        let (command_buffer, target) = step(20);
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(output(&case, &target, &members), metal, "Metal's texels");
        assert_eq!(unlisted(), before, "20 more states");
        let made = [sampled_descriptor(device, &case)];
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &made));
        let executions = command_buffer.executions();
        let Some(ExecutedPipeline::Render(pipeline)) = executions.last().map(Execution::pipeline)
        else {
            panic!("the host step's draw");
        };
        let set = &pipeline.layout.sets[1];
        let capacity = set.bindings.iter().map(|b| b.count).min().unwrap();
        assert!(
            set.bindings
                .iter()
                .any(|b| b.ty == metal2vulkan::vk::DescriptorType::SAMPLER),
            "{SAMPLED} reads argument samplers"
        );
        eprintln!(
            "SAMPLER-CAPACITY {}: the host step's variant's set-1 capacity {capacity}, crowd \
             {HOST_CROWD}",
            device.name()
        );
        if u64::from(capacity) <= HOST_CROWD {
            let (command_buffer, target) = step(capacity.into());
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(command_buffer.error(), None);
            assert_eq!(output(&case, &target, &members), metal, "past the arrays");
            assert_eq!(unlisted(), before + 1, "counted once");
            let steps = HOST_STEPS.into_iter().collect();
            assert_made_once(messages, &UNLISTED, &steps);
        }
    });
}
