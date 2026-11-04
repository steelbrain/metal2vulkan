//! Draws of a primitive type outside the pipeline's input primitive topology class, which Metal's
//! validation stops, on the device the cfg selects, under the validation layer. The owned fixture
//! `cfb_depth_never`'s counting functions draw into 4x4 passes of one `RGBA8Unorm` attachment,
//! cleared and stored: the fullscreen triangle's corners, the vertex function counting its runs
//! (word 1), the fragment function its runs (word 0) and returning red. Neither function writes a
//! point size.
//!
//! Every answer is plain Metal's (Metal probes, an M4 with macOS 27.0.1):
//!
//! - Lines and triangles draw as the draw's type, whatever the class: what a pipeline of no
//!   class (`Unspecified`) draws (probe 1, every class against every type: the texels alike).
//! - Points rasterize nothing under the triangle, the line and no class, from a vertex function
//!   that writes no point size (probe 1 for the triangle and line classes, probe 2 for all
//!   three): the vertex function runs for every point (probe 2's counter: 4 of 4), the
//!   fragment function never, and the texels keep the clear. Metal's answer there is undefined
//!   (the consumer, on an M3 Max: most draws draw nothing, some a few points or a box), so each
//!   such draw announces `unsized-points`, at its call, and for an execution's command where the
//!   runtime records it ([`UNSIZED_POINTS`]).

use super::degrade_notices::counts;
use super::indirect_host_steps::encoding_state_set;
use super::render_state::{texels, texture};
use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveTopologyClass, PrimitiveType, RenderStages,
    ResourceUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandQueue, Degradation, Device, DeviceOptions,
    IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, StoreAction,
};

/// The kinds this lane's draws of points no state sizes announce (`degrade_notices.rs`).
pub(super) const UNSIZED_POINTS: [Degradation; 1] = [Degradation::UnsizedPoints];

/// The passes' clear color, (0.2, 0.4, 0.6, 1), as `RGBA8Unorm` stores it.
const CLEARED: [u8; 4] = [51, 102, 153, 255];

/// What a pass left: its texels, and the fragment and vertex functions' runs.
#[derive(Debug, PartialEq)]
struct Left {
    color: Vec<u8>,
    fragments: u32,
    vertices: u32,
}

/// The lane's device, queue and the fixture's library.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new() -> (Messages, ScratchCache, Lane) {
        Lane::with(|_| {})
    }

    /// [`Lane::new`] with the device options `adjust` changes.
    fn with(adjust: impl FnOnce(&mut DeviceOptions)) -> (Messages, ScratchCache, Lane) {
        let messages = Messages::default();
        let cache = ScratchCache::new().unwrap();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        let bitcode = llvm_assemble(&engine_dir().join("cfb_depth_never.ll")).unwrap();
        let lane = Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        };
        (messages, cache, lane)
    }

    /// The counting functions' pipeline under the input primitive topology class `class`.
    fn pipeline(&self, class: PrimitiveTopologyClass) -> RenderPipelineState {
        self.pipeline_with(class, false)
    }

    /// [`Lane::pipeline`], made with `supportIndirectCommandBuffers` as `indirect` says.
    fn pipeline_with(&self, class: PrimitiveTopologyClass, indirect: bool) -> RenderPipelineState {
        let function = |name| self.library.new_function_with_name(name).unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
            function("cfb_dn_count_vertices").function_descriptor(),
        ));
        descriptor.fragment = Some(StageSource::Function(
            function("cfb_dn_count").function_descriptor(),
        ));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.input_primitive_topology = class;
        descriptor.support_indirect_command_buffers = indirect;
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// A pass of what `encode` encodes after `state` and the counter are set, committed: what it
    /// left, and how many notices of `unsized-points` the device sent before the encoder ended,
    /// between then and `commit`'s return, and from then until the command buffer completed.
    fn pass(
        &self,
        messages: &Messages,
        (state, encoding): (&RenderPipelineState, bool),
        encode: impl FnOnce(&mut RenderCommandEncoder, &Buffer),
    ) -> (Left, [usize; 3]) {
        let announced = || {
            counts(&messages.take_notices())
                .get(&Degradation::UnsizedPoints)
                .copied()
                .unwrap_or(0)
        };
        let color = texture(&self.device, PixelFormat::RGBA8Unorm, 4);
        let counter = buffer_with(&self.device, &[0; 8]).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.2, 0.4, 0.6, 1.0);
        target.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        if encoding {
            encoding_state_set(&command_buffer);
        }
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(state).unwrap();
        e.set_fragment_buffer(Some(&counter), 0, 0).unwrap();
        e.set_vertex_buffer(Some(&counter), 0, 1).unwrap();
        encode(&mut e, &counter);
        let encoded = announced();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let committed = announced();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let completed = announced();
        let words = read_buffer(&counter, 0, 8).unwrap();
        let word = |i: usize| u32::from_le_bytes(words[4 * i..4 * i + 4].try_into().unwrap());
        let left = Left {
            color: texels(&color, 4),
            fragments: word(0),
            vertices: word(1),
        };
        (left, [encoded, committed, completed])
    }

    /// A pass of one draw of `count` vertices of `ty` with the pipeline of `class`.
    fn draw(&self, class: PrimitiveTopologyClass, ty: PrimitiveType, count: u32) -> Left {
        let color = texture(&self.device, PixelFormat::RGBA8Unorm, 4);
        let counter = buffer_with(&self.device, &[0; 8]).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.2, 0.4, 0.6, 1.0);
        target.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(&self.pipeline(class)).unwrap();
        e.set_fragment_buffer(Some(&counter), 0, 0).unwrap();
        e.set_vertex_buffer(Some(&counter), 0, 1).unwrap();
        e.draw_primitives(ty, 0, count, 1, 0).unwrap();
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let words = read_buffer(&counter, 0, 8).unwrap();
        let word = |i: usize| u32::from_le_bytes(words[4 * i..4 * i + 4].try_into().unwrap());
        Left {
            color: texels(&color, 4),
            fragments: word(0),
            vertices: word(1),
        }
    }
}

/// Lines and triangles of every class draw what a pipeline of no class draws (probe 1), and
/// points from a vertex function without a point size draw nothing under the triangle, the line
/// and no class, their vertex function run for every point (probes 1 and 2).
#[test]
fn a_draw_outside_the_pipeline_s_class_draws_as_plain_metal() {
    use PrimitiveTopologyClass as Class;
    let (messages, _cache, lane) = Lane::new();
    let name = lane.device.name().to_owned();
    for (ty, count) in [
        (PrimitiveType::Line, 2),
        (PrimitiveType::LineStrip, 3),
        (PrimitiveType::Triangle, 3),
        (PrimitiveType::TriangleStrip, 4),
    ] {
        let unspecified = lane.draw(Class::Unspecified, ty, count);
        for class in [Class::Point, Class::Line, Class::Triangle] {
            assert_eq!(
                lane.draw(class, ty, count),
                unspecified,
                "{ty:?} under {class:?} on {name}"
            );
        }
    }
    for class in [Class::Unspecified, Class::Line, Class::Triangle] {
        assert_eq!(
            lane.draw(class, PrimitiveType::Point, 3),
            Left {
                color: CLEARED.repeat(16),
                fragments: 0,
                vertices: 3,
            },
            "points under {class:?} on {name}"
        );
    }
    eprintln!("draws outside their pipeline's class on {name}: Metal's");
    drop(lane);
    layer_reported_nothing_at_all(&messages);
}

/// `unsized-points`, asked for at every occurrence, is announced where each draw of points the
/// pipeline sizes none of is decided, and each draws nothing, its vertex function run for every
/// point: a direct draw at its call; an execution of an indirect command buffer's records the
/// CPU wrote in `commit`, before it returns; an execution of a buffer the GPU may write
/// (`gpu_written`: made resident for writing where a state that may encode is set) in its host
/// step, after the commit, and one made resident for writing with no such state in `commit`. A
/// draw of no vertices, a draw
/// under the point class, and an execution of either, announce nothing.
#[test]
fn unsized_points_are_announced_at_the_draw_the_commit_and_the_host_step() {
    use PrimitiveTopologyClass as Class;
    let (messages, _cache, lane) = Lane::with(|options| {
        options.announce_each = UNSIZED_POINTS.into_iter().collect();
    });
    let name = lane.device.name().to_owned();
    let nothing = |vertices| Left {
        color: CLEARED.repeat(16),
        fragments: 0,
        vertices,
    };
    let point = PrimitiveType::Point;
    // Direct draws.
    for (class, count, announced) in [
        (Class::Triangle, 3, 1),
        (Class::Unspecified, 3, 1),
        (Class::Triangle, 0, 0),
        (Class::Point, 3, 0),
    ] {
        let state = lane.pipeline(class);
        let (left, notices) = lane.pass(&messages, (&state, false), |e, _| {
            e.draw_primitives(point, 0, count, 1, 0).unwrap();
        });
        assert_eq!(notices, [announced, 0, 0], "{class:?} {count} on {name}");
        if class != Class::Point {
            assert_eq!(left, nothing(count), "{class:?} {count} on {name}");
        }
    }
    // Executions of commands that inherit the encoder's state and buffers.
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.inherit_pipeline_state = true;
    descriptor.inherit_buffers = true;
    let commands = |count: u64| {
        let icb = lane
            .device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
            .unwrap();
        icb.indirect_render_command_at(0)
            .unwrap()
            .draw_primitives(point, 0, count, 1, 0)
            .unwrap();
        icb
    };
    // An execution of a buffer made resident for writing, where a state that may encode is set in
    // the command buffer (an encoder of its own before the pass), is recorded by the host step;
    // without one, the GPU cannot write the records, and the commit records it, as one of a
    // buffer made resident for nothing.
    for (class, count, (writable, encoding), announced) in [
        (Class::Triangle, 3, (false, false), [0, 1, 0]),
        (Class::Triangle, 0, (false, false), [0, 0, 0]),
        (Class::Point, 3, (false, false), [0, 0, 0]),
        (Class::Triangle, 3, (true, true), [0, 0, 1]),
        (Class::Triangle, 3, (true, false), [0, 1, 0]),
    ] {
        let gpu_written = writable && encoding;
        let state = lane.pipeline_with(class, true);
        let icb = commands(count);
        let (left, mut notices) = lane.pass(&messages, (&state, encoding), |e, _| {
            if writable {
                e.use_resource(
                    icb.as_resource(),
                    ResourceUsage::READ.union(ResourceUsage::WRITE),
                    RenderStages::VERTEX,
                )
                .unwrap();
            }
            e.execute_commands_in_buffer(&icb, 0..1).unwrap();
        });
        let at = format!(
            "an execution: {class:?} {count}, writable {writable}, encoding {encoding}, on {name}"
        );
        assert_eq!(icb.gpu_written(), gpu_written, "{at}");
        if gpu_written {
            // The host step runs on the steps' thread once the work before it completed,
            // which may be before `commit` returns: after the commit either way.
            notices = [notices[0], 0, notices[1] + notices[2]];
        }
        assert_eq!(notices, announced, "{at}");
        if class != Class::Point {
            assert_eq!(left, nothing(count as u32), "{at}");
        }
    }
    eprintln!(
        "CBF-MARK {name}: unsized points were announced at the draw, the commit and the host \
         step, and drew nothing"
    );
    layer_reported_nothing_at_all(&messages);
}
