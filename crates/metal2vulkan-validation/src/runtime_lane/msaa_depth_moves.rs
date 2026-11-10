//! A multisampled depth attachment keeps its texels through the layouts it is rendered and read
//! in, on the device the cfg selects, under the validation layer, whose layout tracking follows
//! every command buffer and queue. On a device with `DriverFact::MultisampledDepthFasterOutOfGeneral`
//! the runtime renders such a depth in the attachment layout: a render encoder that touches it only
//! as its attachment holds it there through its instances and between the encoders that hold it
//! (`texture/held_depth.rs` in the runtime), and any other moves it there around each instance
//! (`render_target::depth_layout`). The cases count both (`WorkCounts::depth_instances_held`,
//! `depth_attachments_moved`) and the moves back to `GENERAL` before an encoder that names the
//! depth otherwise (`held_depths_released`), so each run shows which path ran; on any other device
//! every count is zero.
//!
//! The depth is a 4x `Depth32Float` target of [`SIDE`]², `Private`, each pass drawing
//! `rt6_render_state`'s fullscreen triangle (`rt6_full`) shaded white under `Less` with writes at
//! one of [`DEPTHS`]: a pass that clears to 1.0 and draws at 0.25 leaves 0.25, one that loads it
//! and draws at 0.5 leaves it, one that draws at 0.125 leaves 0.125. Each case reads every sample
//! back and checks it: through a compute function's texture slot (`rt6_read_depth`), through an
//! argument buffer the encoder makes resident with `use_resource` ([`READ_DMS`]), or through a
//! blit's copy into a buffer.

use super::{
    at_plan, buffer_with, layer_reported_nothing, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_samples::encode;
use crate::executor_contract::multisample::SampleRead;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BarrierScope, CompareFunction, IndirectCommandType, PixelFormat, PrimitiveType, RenderStages,
    ResourceUsage, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::DeviceOptions;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandQueue, ComputePipelineState, DepthStencilState,
    Device, EventRef, IndirectCommandBufferDescriptor, LoadAction, MultisampleDepthResolveFilter,
    Origin, PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SharedEvent, Size, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::path::Path;

/// The targets' side.
const SIDE: u32 = 16;

/// The samples of every attachment.
const SAMPLES: u64 = 4;

/// The bytes between two depths in the depth buffer (`Rt6Depth`, two `float`s), as wide as any
/// device's constant buffer offset alignment.
const STRIDE: u64 = 256;

/// The depths the passes draw at, by index into the depth buffer.
const DEPTHS: [f32; 3] = [0.25, 0.5, 0.125];

/// The index of each depth in [`DEPTHS`].
const QUARTER: usize = 0;
const HALF: usize = 1;
const EIGHTH: usize = 2;

/// Synthetic AIR written for this lane: `read_dms` reads sample `p.z` of texel `p.xy` of the
/// `depth2d_ms<float>` its argument buffer (buffer 0, `{ depth2d_ms<float, access::read> d; }`)
/// holds, storing it to buffer 1's `float` `(16 p.y + p.x) 4 + p.z`, one thread per sample.
const READ_DMS: &str = r#"; Owned synthetic kernel.
source_filename = "msaa_depth_moves.metal"

%struct.R = type { ptr addrspace(1) }

define void @read_dms(ptr addrspace(2) %m, ptr addrspace(1) noalias %out, <3 x i32> %p) {
entry:
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %rs = tail call ptr addrspace(2) @air.get_read_sampler()
  %xy = shufflevector <3 x i32> %p, <3 x i32> poison, <2 x i32> <i32 0, i32 1>
  %s = extractelement <3 x i32> %p, i64 2
  %r = tail call { float, i8 } @air.read_depth_2d_ms.f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, i32 1, <2 x i32> %xy, i32 %s, i32 1)
  %x = extractvalue { float, i8 } %r, 0
  %py = extractelement <3 x i32> %p, i64 1
  %row = shl i32 %py, 4
  %px = extractelement <3 x i32> %p, i64 0
  %texel = add i32 %row, %px
  %base = shl i32 %texel, 2
  %i = add i32 %base, %s
  %j = zext i32 %i to i64
  %q = getelementptr inbounds float, ptr addrspace(1) %out, i64 %j
  store float %x, ptr addrspace(1) %q, align 4
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { float, i8 } @air.read_depth_2d_ms.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.version = !{!30}
!air.language_version = !{!31}
!0 = !{ptr @read_dms, !1, !2}
!1 = !{}
!2 = !{!3, !6, !7}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"m"}
!4 = !{i32 0, i32 8, i32 0, !"depth2d_ms<float, read>", !"d", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"depth2d_ms<float, read>", !"air.arg_name", !"d"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"p"}
!30 = !{i32 2, i32 8, i32 0}
!31 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The device's pipelines, targets, depth test and depth buffer.
struct Kit {
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    target: Texture,
    depth: Texture,
    less: DepthStencilState,
    depths: Buffer,
    /// Whether the device has the fact, and so renders the depth in the attachment layout.
    fact: bool,
}

impl Kit {
    fn new(device: &Device) -> Kit {
        let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public/engine/rt6_render_state.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("rt6_full"));
        descriptor.fragment = Some(function("rt6_white"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
        descriptor.raster_sample_count = SAMPLES as u32;
        descriptor.support_indirect_command_buffers = true;
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let mut less = DepthStencilDescriptor::default();
        less.depth_compare_function = CompareFunction::Less;
        less.depth_write_enabled = true;
        let mut bytes = vec![0u8; STRIDE as usize * DEPTHS.len()];
        for (i, z) in DEPTHS.iter().enumerate() {
            let at = i * STRIDE as usize;
            bytes[at..at + 4].copy_from_slice(&z.to_le_bytes());
        }
        Kit {
            queue: device.new_command_queue().unwrap(),
            pipeline,
            target: multisampled(device, PixelFormat::RGBA8Unorm, TextureUsage::RENDER_TARGET),
            depth: depth(device),
            less: device.new_depth_stencil_state(&less).unwrap(),
            depths: buffer_with(device, &bytes).unwrap(),
            fact: (device.properties().driver_facts)
                .contains(&DriverFact::MultisampledDepthFasterOutOfGeneral),
        }
    }

    /// A pass descriptor that loads `depth` by `load` (cleared to 1.0) and stores it.
    fn descriptor(&self, depth: &Texture, load: LoadAction) -> RenderPassDescriptor {
        let mut descriptor = RenderPassDescriptor::new();
        let target = &mut descriptor.color_attachments[0];
        target.texture = Some(self.target.clone());
        target.load_action = load;
        target.store_action = StoreAction::Store;
        descriptor.depth_attachment.texture = Some(depth.clone());
        descriptor.depth_attachment.load_action = load;
        descriptor.depth_attachment.clear_depth = 1.0;
        descriptor.depth_attachment.store_action = StoreAction::Store;
        descriptor
    }

    /// Bind the pipeline, the depth test and depth `at`'s vertex buffer, and draw the triangle.
    fn draw(&self, encoder: &mut RenderCommandEncoder, at: usize) {
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        encoder.set_depth_stencil_state(&self.less).unwrap();
        encoder
            .set_vertex_buffer(Some(&self.depths), at as u64 * STRIDE, 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }

    /// One encoder of `descriptor` in `command_buffer`: `before` on it, then a draw at depth
    /// `at`.
    fn pass_of(
        &self,
        command_buffer: &CommandBuffer,
        descriptor: &RenderPassDescriptor,
        at: usize,
        before: impl FnOnce(&mut RenderCommandEncoder),
    ) {
        let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
        before(&mut encoder);
        self.draw(&mut encoder, at);
        encoder.end_encoding().unwrap();
    }

    /// One pass into the depth, loaded by `load`, drawing at depth `at`, in `command_buffer`.
    fn pass_in(&self, command_buffer: &CommandBuffer, load: LoadAction, at: usize) {
        self.pass_of(
            command_buffer,
            &self.descriptor(&self.depth, load),
            at,
            |_| {},
        );
    }

    /// [`pass_in`](Self::pass_in) a command buffer of its own, committed and waited for.
    fn pass(&self, load: LoadAction, at: usize) {
        let command_buffer = self.queue.command_buffer().unwrap();
        self.pass_in(&command_buffer, load, at);
        complete(&command_buffer);
    }

    /// Every sample of `texture` read through a compute function's texture slot, in a command
    /// buffer of its own.
    fn sample_depths_of(&self, device: &Device, texture: &Texture) -> Vec<f32> {
        let command_buffer = self.queue.command_buffer().unwrap();
        let extent = [SIDE, SIDE];
        let read = encode(
            device,
            &command_buffer,
            texture,
            SampleRead::Depth,
            extent,
            SAMPLES as u32,
        )
        .unwrap();
        complete(&command_buffer);
        floats(&read.bytes().unwrap())
    }

    /// [`sample_depths_of`](Self::sample_depths_of) the depth.
    fn sample_depths(&self, device: &Device) -> Vec<f32> {
        self.sample_depths_of(device, &self.depth)
    }
}

/// A multisampled `Private` texture of the targets' size, sample count and `format`.
fn multisampled(device: &Device, format: PixelFormat, usage: TextureUsage) -> Texture {
    let side = u64::from(SIDE);
    let mut d = TextureDescriptor::texture_2d(format, side, side, false);
    d.texture_type = TextureType::Type2DMultisample;
    d.sample_count = SAMPLES;
    d.usage = usage;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

/// A depth like the kit's: rendered and read.
fn depth(device: &Device) -> Texture {
    let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    multisampled(device, PixelFormat::Depth32Float, usage)
}

/// Commit `command_buffer`, wait for it, and check it ran without an error.
fn complete(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The `float`s of `bytes`.
fn floats(bytes: &[u8]) -> Vec<f32> {
    let words = bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    words.collect()
}

/// Check every one of `samples` (every sample of the target) is `expected`.
fn assert_all(samples: &[f32], expected: f32, what: &str) {
    let count = u64::from(SIDE * SIDE) * SAMPLES;
    assert_eq!(samples.len() as u64, count, "{what}");
    let wrong = samples.iter().filter(|&&d| d != expected).count();
    assert_eq!(
        wrong,
        0,
        "{what}: {wrong} of {count} samples not {expected}, first {:?}",
        &samples[..4]
    );
}

/// The counts a case reads: instances held, instances moved, moves back to `GENERAL`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
    held: u64,
    moved: u64,
    released: u64,
}

impl Counts {
    fn of(counts: &WorkCounts) -> Counts {
        Counts {
            held: counts.depth_instances_held,
            moved: counts.depth_attachments_moved,
            released: counts.held_depths_released,
        }
    }

    /// What ran since `before`.
    fn since(device: &Device, before: Counts) -> Counts {
        let now = Counts::of(&device.work_counts());
        Counts {
            held: now.held - before.held,
            moved: now.moved - before.moved,
            released: now.released - before.released,
        }
    }
}

/// What a case's counts must be on a device with the fact (elsewhere every count is zero).
enum Expect {
    /// Each count.
    Exactly(Counts),
    /// Only the instances held or moved, this many: a pass may record later, its draw's pipeline
    /// for the pass still being made (`render.rs`), and such a pass keeps the moves around its
    /// instance.
    Instances(u64),
}

/// Run `case` on a validated device with a kit, and check the counts it ran against `expected`
/// and that the validation layer reported nothing.
fn run(name: &str, expected: Expect, case: impl FnOnce(&Device, &Kit)) {
    run_with(|_| {}, name, expected, case)
}

/// [`run`], `adjust` applied to the options of each device it makes (`at_plan` gives them
/// their queue plan).
fn run_with(
    adjust: fn(&mut DeviceOptions),
    name: &str,
    expected: Expect,
    case: impl FnOnce(&Device, &Kit),
) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
    let device_name = device.name().to_owned();
    let kit = Kit::new(&device);
    let before = Counts::of(&device.work_counts());
    case(&device, &kit);
    let ran = Counts::since(&device, before);
    let fact = kit.fact;
    eprintln!(
        "MSAA-DEPTH-MOVES {name} on {device_name}: fact {fact}, instances held {}, moved {}, \
         released {}",
        ran.held, ran.moved, ran.released
    );
    let none = Counts {
        held: 0,
        moved: 0,
        released: 0,
    };
    let what = format!("{name} on {device_name} (fact {fact})");
    match (fact, expected) {
        (false, _) => assert_eq!(ran, none, "{what}"),
        (true, Expect::Exactly(counts)) => assert_eq!(ran, counts, "{what}"),
        (true, Expect::Instances(count)) => assert_eq!(ran.held + ran.moved, count, "{what}"),
    }
    drop(kit);
    drop(device);
    layer_reported_nothing(&messages, &format!("{name} on {device_name}"));
}

/// Three passes, each its own command buffer and each read back by a compute function after it:
/// 0.25, 0.25, then 0.125. Each pass holds the depth, and each read moves it back.
#[test]
fn a_multisampled_depth_keeps_its_texels_across_the_layout_moves() {
    let counts = Counts {
        held: 3,
        moved: 0,
        released: 3,
    };
    run("read-after-each", Expect::Exactly(counts), |device, kit| {
        let steps = [
            (LoadAction::Clear, QUARTER, 0.25f32),
            (LoadAction::Load, HALF, 0.25),
            (LoadAction::Load, EIGHTH, 0.125),
        ];
        for (pass, (load, at, expected)) in steps.into_iter().enumerate() {
            kit.pass(load, at);
            assert_all(
                &kit.sample_depths(device),
                expected,
                &format!("pass {pass}"),
            );
        }
    });
}

/// Frames that only render the depth leave it in the attachment layout: four passes in four
/// command buffers, then four in one with the read in it too, each followed by one read. Only the
/// reads move it back.
#[test]
fn a_held_depth_rests_in_the_attachment_layout_between_frames() {
    let counts = Counts {
        held: 8,
        moved: 0,
        released: 2,
    };
    run("frames", Expect::Exactly(counts), |device, kit| {
        let frames = [
            (LoadAction::Clear, QUARTER),
            (LoadAction::Load, HALF),
            (LoadAction::Load, EIGHTH),
            (LoadAction::Load, HALF),
        ];
        for (load, at) in frames {
            kit.pass(load, at);
        }
        assert_all(&kit.sample_depths(device), 0.125, "four command buffers");
        let command_buffer = kit.queue.command_buffer().unwrap();
        for (load, at) in frames {
            kit.pass_in(&command_buffer, load, at);
        }
        let read = encode(
            device,
            &command_buffer,
            &kit.depth,
            SampleRead::Depth,
            [SIDE, SIDE],
            SAMPLES as u32,
        )
        .unwrap();
        complete(&command_buffer);
        assert_all(&floats(&read.bytes().unwrap()), 0.125, "one command buffer");
    });
}

/// A blit between passes takes the depth out of the attachment layout before it, in one command
/// buffer: a pass leaves 0.25, a blit copies the depth into another and into a buffer, a pass
/// leaves 0.125. The copy and the buffer hold 0.25, the depth 0.125. The blit and the read move
/// the held depth back; the copy, never held, is in `GENERAL` already.
#[test]
fn a_blit_between_passes_reads_a_held_depth_after_its_move() {
    let counts = Counts {
        held: 2,
        moved: 0,
        released: 2,
    };
    run("blit", Expect::Exactly(counts), |device, kit| {
        let copy = depth(device);
        let length = u64::from(SIDE * SIDE) * SAMPLES * 4;
        let bytes = device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap();
        let command_buffer = kit.queue.command_buffer().unwrap();
        kit.pass_in(&command_buffer, LoadAction::Clear, QUARTER);
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_texture(&kit.depth, &copy).unwrap();
        let side = u64::from(SIDE);
        blit.copy_from_texture_to_buffer(
            &kit.depth,
            0,
            0,
            Origin { x: 0, y: 0, z: 0 },
            Size::new(side, side, 1),
            &bytes,
            0,
            side * SAMPLES * 4,
            length,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        kit.pass_in(&command_buffer, LoadAction::Load, EIGHTH);
        complete(&command_buffer);
        let copied = floats(&read_buffer(&bytes, 0, length).unwrap());
        assert_all(&copied, 0.25, "the blit's buffer");
        assert_all(&kit.sample_depths_of(device, &copy), 0.25, "the copy");
        assert_all(&kit.sample_depths(device), 0.125, "the depth");
    });
}

/// An encoder that names its depth otherwise too holds it in `GENERAL` and moves it around its
/// instance: after held passes of 0.25 and 0.5 (the second making the depth resident with
/// `use_resource`, which no draw reads through, so nothing names it), a pass that binds it as a
/// fragment texture its pipeline does not read moves it back first, then draws 0.125; the read
/// finds it in `GENERAL` already.
#[test]
fn an_encoder_that_names_its_depth_otherwise_moves_it_around_its_instance() {
    let counts = Counts {
        held: 2,
        moved: 1,
        released: 1,
    };
    run("named", Expect::Exactly(counts), |device, kit| {
        kit.pass(LoadAction::Clear, QUARTER);
        let command_buffer = kit.queue.command_buffer().unwrap();
        let descriptor = kit.descriptor(&kit.depth, LoadAction::Load);
        kit.pass_of(&command_buffer, &descriptor, HALF, |encoder| {
            let fragment = RenderStages::FRAGMENT;
            let resource = kit.depth.as_resource();
            (encoder.use_resource(resource, ResourceUsage::READ, fragment)).unwrap();
        });
        kit.pass_of(&command_buffer, &descriptor, EIGHTH, |encoder| {
            (encoder.set_fragment_texture(Some(&kit.depth), 0)).unwrap();
        });
        complete(&command_buffer);
        assert_all(&kit.sample_depths(device), 0.125, "the depth");
    });
}

/// A pass split by a memory barrier holds the depth through both its instances: 0.5, then 0.25
/// over it.
#[test]
fn a_split_pass_holds_its_depth_through_every_instance() {
    let counts = Counts {
        held: 2,
        moved: 0,
        released: 1,
    };
    run("split", Expect::Exactly(counts), |device, kit| {
        let command_buffer = kit.queue.command_buffer().unwrap();
        let descriptor = kit.descriptor(&kit.depth, LoadAction::Clear);
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        kit.draw(&mut encoder, HALF);
        let (after, before) = (RenderStages::FRAGMENT, RenderStages::VERTEX);
        encoder.memory_barrier_with_scope(BarrierScope::BUFFERS, after, before);
        kit.draw(&mut encoder, QUARTER);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_all(&kit.sample_depths(device), 0.25, "the depth");
    });
}

/// An execution of render commands keeps its pass in `GENERAL` between held passes: a held pass
/// of 0.25, a pass whose indirect command draws 0.125, a held pass of 0.5 it hides. The execution's
/// encoder and the read each move the depth back.
#[test]
fn an_execution_keeps_its_pass_in_general_between_held_passes() {
    let counts = Counts {
        held: 2,
        moved: 0,
        released: 2,
    };
    run("execution", Expect::Exactly(counts), |device, kit| {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW;
        descriptor.max_vertex_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
            .unwrap();
        let command = icb.indirect_render_command_at(0).unwrap();
        command.set_render_pipeline_state(&kit.pipeline).unwrap();
        let offset = EIGHTH as u64 * STRIDE;
        (command.set_vertex_buffer(Some(&kit.depths), offset, 0)).unwrap();
        (command.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)).unwrap();
        let command_buffer = kit.queue.command_buffer().unwrap();
        kit.pass_in(&command_buffer, LoadAction::Clear, QUARTER);
        let loaded = kit.descriptor(&kit.depth, LoadAction::Load);
        let mut encoder = command_buffer.render_command_encoder(&loaded).unwrap();
        let resource = kit.depths.as_resource();
        let vertex = RenderStages::VERTEX;
        (encoder.use_resource(resource, ResourceUsage::READ, vertex)).unwrap();
        encoder.set_depth_stencil_state(&kit.less).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        kit.pass_in(&command_buffer, LoadAction::Load, HALF);
        complete(&command_buffer);
        assert_all(&kit.sample_depths(device), 0.125, "the depth");
    });
}

/// Passes that resolve the depth into a single-sample texture, by sample 0 and by the maximum,
/// each twice: every resolved texel 0.25, whichever layout each pass renders the depth in (the
/// first of each records later where its draw's pipeline for the resolving pass is still being
/// made).
#[test]
fn a_depth_resolve_reads_the_depth_in_its_pass() {
    run("resolve", Expect::Instances(4), |device, kit| {
        let side = u64::from(SIDE);
        let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, side, side, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        let length = side * side * 4;
        for filter in [
            MultisampleDepthResolveFilter::Sample0,
            MultisampleDepthResolveFilter::Sample0,
            MultisampleDepthResolveFilter::Max,
            MultisampleDepthResolveFilter::Max,
        ] {
            let resolved = device.new_texture(&d).unwrap();
            let bytes = device
                .new_buffer(length, ResourceOptions::shared())
                .unwrap();
            let command_buffer = kit.queue.command_buffer().unwrap();
            let mut descriptor = kit.descriptor(&kit.depth, LoadAction::Clear);
            let attachment = &mut descriptor.depth_attachment;
            attachment.resolve_texture = Some(resolved.clone());
            attachment.store_action = StoreAction::StoreAndMultisampleResolve;
            attachment.depth_resolve_filter = filter;
            kit.pass_of(&command_buffer, &descriptor, QUARTER, |_| {});
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_buffer(
                &resolved,
                0,
                0,
                Origin { x: 0, y: 0, z: 0 },
                Size::new(side, side, 1),
                &bytes,
                0,
                side * 4,
                length,
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            complete(&command_buffer);
            let texels = floats(&read_buffer(&bytes, 0, length).unwrap());
            let wrong = texels.iter().filter(|&&d| d != 0.25).count();
            assert_eq!(wrong, 0, "{filter:?}: resolved texels not 0.25");
        }
        assert_all(&kit.sample_depths(device), 0.25, "the depth");
    });
}

/// Two queues, the second's command buffer committed first and waiting for an event the first's
/// signals after its pass: the read runs after the pass, its move back after the pass on the other
/// queue. Twice, at the event's next value: 0.25, then 0.125.
#[test]
fn a_queue_waiting_for_an_event_reads_a_held_depth_after_its_move() {
    let counts = Counts {
        held: 2,
        moved: 0,
        released: 2,
    };
    run_with(
        at_plan,
        "two-queues",
        Expect::Exactly(counts),
        |device, kit| {
            let other = device.new_command_queue().unwrap();
            let event: SharedEvent = device.new_shared_event().unwrap();
            let extent = [SIDE, SIDE];
            for (round, (load, at, expected)) in [
                (LoadAction::Clear, QUARTER, 0.25f32),
                (LoadAction::Load, EIGHTH, 0.125),
            ]
            .into_iter()
            .enumerate()
            {
                let value = round as u64 + 1;
                let reading = other.command_buffer().unwrap();
                reading
                    .encode_wait_for_event(EventRef::from(&event), value)
                    .unwrap();
                let read = encode(
                    device,
                    &reading,
                    &kit.depth,
                    SampleRead::Depth,
                    extent,
                    SAMPLES as u32,
                )
                .unwrap();
                reading.commit().unwrap();
                let rendering = kit.queue.command_buffer().unwrap();
                kit.pass_in(&rendering, load, at);
                rendering
                    .encode_signal_event(EventRef::from(&event), value)
                    .unwrap();
                complete(&rendering);
                reading.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(reading.error(), None);
                let samples = floats(&read.bytes().unwrap());
                assert_all(&samples, expected, &format!("round {round}"));
            }
        },
    );
}

/// A compute function reads the depth through an argument buffer the encoder makes resident with
/// `use_resource`: the residency names the depth, which moves it back first. Every sample 0.25.
#[test]
fn an_argument_buffer_read_moves_a_held_depth_through_use_resource() {
    let counts = Counts {
        held: 1,
        moved: 0,
        released: 1,
    };
    run("argument-buffer", Expect::Exactly(counts), |device, kit| {
        let library = device
            .new_library_with_air(&llvm_assemble_text(READ_DMS).unwrap())
            .unwrap();
        let function = library.new_function_with_name("read_dms").unwrap();
        let state: ComputePipelineState = device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let id = kit.depth.gpu_resource_id().0.to_le_bytes();
        let argument = buffer_with(device, &id).unwrap();
        let length = u64::from(SIDE * SIDE) * SAMPLES * 4;
        let out = buffer_with(device, &vec![0xab; length as usize]).unwrap();
        kit.pass(LoadAction::Clear, QUARTER);
        let command_buffer = kit.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        let resource = kit.depth.as_resource();
        encoder.use_resource(resource, ResourceUsage::READ).unwrap();
        encoder.set_buffer(Some(&argument), 0, 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        let side = u64::from(SIDE);
        encoder
            .dispatch_threads(Size::new(side, side, SAMPLES), Size::new(side, side, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let samples = floats(&read_buffer(&out, 0, length).unwrap());
        assert_all(&samples, 0.25, "through the argument buffer");
    });
}
