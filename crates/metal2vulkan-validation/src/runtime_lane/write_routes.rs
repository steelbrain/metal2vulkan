//! An indirect command buffer a kernel writes, executed by a later encoder of the same command
//! buffer, on the device the cfg selects, under the validation layer: by where the command buffer
//! makes the buffer resident for writing ([`Resident`]), what the execution runs is what Metal's
//! runs ([`METAL`], read from the probe
//! `crates/metal2vulkan-conformance/probes/icb_routed_writes.m` on an M3 Max, macOS 27.0.1, which
//! runs the same shapes), but where no encoder made it resident for writing ([`APART`]).
//!
//! The owned fixture `icb_routed_writes`: `irw_encode_draw` writes render command 0 (its pipeline
//! state, taken from the argument buffer that names the command buffer, its vertex buffer 0 and a
//! draw of a triangle over the target, which `irw_fragment` colours green); `irw_encode_point`
//! writes render command 0 of a buffer that inherits the encoder's state (its vertex buffer 0 and
//! a draw of one point, 4 pixels wide, at the target's centre), beside command 1, which the CPU
//! wrote (a triangle in the target's bottom-left corner); `irw_encode_dispatch` writes compute
//! command 0 (its state, `irw_mark`'s, from the argument buffer, its kernel buffer 0 and a dispatch
//! of 4 threads, each writing `0x100` plus its index). The CPU writes none of the kernels' records,
//! so a run of the CPU's records runs none of them: the target's centre keeps its red clear
//! colour, and the buffer its zeros.
//!
//! The point is where the write summary (`runtime/written_commands.rs`) shows: an execution of a
//! buffer that inherits the encoder's state may run on the GPU, as draws of the one primitive type
//! the summary allows (`bind::draw_replay_primitive_type`), the CPU's triangle here, unless the
//! summary holds the kernel's point too.
//!
//! [`a_refused_command_another_queue_writes_again_runs_what_it_wrote`]: a command the runtime
//! refuses at commit, which a later command buffer of another queue writes again before the
//! execution runs, runs what that one wrote, as Metal's does.

use super::degrade_notices::on_device;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, ComputePipelineState, Device, EventRef,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions,
    Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};

/// Where the command buffer makes the indirect command buffer resident for writing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resident {
    /// In the kernel's encoder: Metal's documented contract.
    Same,
    /// Only in a compute encoder after the kernel's, which runs nothing.
    After,
    /// Only in a compute encoder before the kernel's, which runs nothing.
    Before,
    /// Nowhere.
    Nowhere,
}

/// The target's width and height.
const SIDE: u64 = 16;

/// The fixture's functions on a device.
struct Scene {
    device: Device,
    library: Library,
}

impl Scene {
    fn new(device: &Device) -> Scene {
        let bitcode = llvm_assemble(&engine_dir().join("icb_routed_writes.ll")).unwrap();
        Scene {
            device: device.clone(),
            library: device.new_library_with_air(&bitcode).unwrap(),
        }
    }

    fn function(&self, name: &str) -> StageSource {
        let function = self.library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    }

    fn kernel(&self, name: &str, indirect: bool) -> ComputePipelineState {
        let mut descriptor = ComputePipelineDescriptor::new(self.function(name));
        descriptor.support_indirect_command_buffers = indirect;
        (self.device)
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// The table the kernels take: the command buffer's ID, then the state's.
    fn table(&self, icb: &IndirectCommandBuffer, state: u64) -> Buffer {
        let ids = [icb.gpu_resource_id().0, state];
        buffer_with(&self.device, &ids.map(u64::to_le_bytes).concat()).unwrap()
    }

    /// The kernel `writer` over `table` and `argument`, with `icb` made resident for writing as
    /// `resident` says.
    fn write(
        command_buffer: &CommandBuffer,
        resident: Resident,
        (writer, table, argument): (&ComputePipelineState, &Buffer, &Buffer),
        icb: &IndirectCommandBuffer,
    ) {
        let alone = || {
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder
                .use_resource(icb.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            encoder.end_encoding().unwrap();
        };
        if let Resident::Before = resident {
            alone();
        }
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        if let Resident::Same = resident {
            encoder
                .use_resource(icb.as_resource(), ResourceUsage::WRITE)
                .unwrap();
        }
        encoder.set_compute_pipeline_state(writer).unwrap();
        encoder.set_buffer(Some(table), 0, 0).unwrap();
        encoder.set_buffer(Some(argument), 0, 1).unwrap();
        let one = Size::new(1, 1, 1);
        encoder.dispatch_threads(one, one).unwrap();
        encoder.end_encoding().unwrap();
        if let Resident::After = resident {
            alone();
        }
    }

    /// A render state of `vertex` and `irw_fragment` that indirect command buffers may run.
    fn render_state(&self, vertex: &str) -> RenderPipelineState {
        self.render_state_in(vertex, PixelFormat::RGBA8Unorm)
    }

    /// [`render_state`](Self::render_state), drawing into `format`.
    fn render_state_in(&self, vertex: &str, format: PixelFormat) -> RenderPipelineState {
        let mut descriptor = RenderPipelineDescriptor::new(self.function(vertex));
        descriptor.fragment = Some(self.function("irw_fragment"));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
        descriptor.support_indirect_command_buffers = true;
        (self.device)
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// A pass of `command_buffer` over a new target cleared red, which `encode` fills, committed
    /// and waited for: the target's texels at `at`.
    fn pass<const N: usize>(
        &self,
        command_buffer: &CommandBuffer,
        encode: impl FnOnce(&mut RenderCommandEncoder),
        at: [(u64, u64); N],
    ) -> [[u8; 4]; N] {
        let (target, pass) = self.target();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encode(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        texels(&target, at)
    }

    /// A new target cleared red, and a pass over it.
    fn target(&self) -> (Texture, RenderPassDescriptor) {
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&target).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(1.0, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        (target, pass)
    }

    /// The render command's run, as `resident` says: the target's centre texel.
    fn render(&self, resident: Resident) -> [u8; 4] {
        let device = &self.device;
        let state = self.render_state("irw_vertex");
        let writer = self.kernel("irw_encode_draw", false);
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW;
        commands.max_vertex_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&commands, 1, ResourceOptions::shared())
            .unwrap();
        let corners = floats(device, &[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
        let table = self.table(&icb, state.gpu_resource_id().0);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        Scene::write(&command_buffer, resident, (&writer, &table, &corners), &icb);
        let [centre] = self.pass(
            &command_buffer,
            |encoder| {
                let vertex = RenderStages::VERTEX;
                (encoder.use_resource(corners.as_resource(), ResourceUsage::READ, vertex)).unwrap();
                encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            },
            [(SIDE / 2, SIDE / 2)],
        );
        centre
    }

    /// The inherited state's run of the point command and the CPU's triangle, as `resident`
    /// says: the target's centre texel (the point's) and its bottom-left corner's (the
    /// triangle's).
    fn points(&self, resident: Resident) -> [[u8; 4]; 2] {
        let device = &self.device;
        let state = self.render_state("irw_point_vertex");
        let writer = self.kernel("irw_encode_point", false);
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW;
        commands.inherit_pipeline_state = true;
        commands.max_vertex_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&commands, 2, ResourceOptions::shared())
            .unwrap();
        let corner = floats(device, &[-1.0, -1.0, -0.75, -1.0, -1.0, -0.75]);
        let centre = floats(device, &[0.0, 0.0]);
        let cpu = icb.indirect_render_command_at(1).unwrap();
        cpu.set_vertex_buffer(Some(&corner), 0, 0).unwrap();
        cpu.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        let table = self.table(&icb, state.gpu_resource_id().0);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        Scene::write(&command_buffer, resident, (&writer, &table, &centre), &icb);
        self.pass(
            &command_buffer,
            |encoder| {
                encoder.set_render_pipeline_state(&state).unwrap();
                let vertex = RenderStages::VERTEX;
                for buffer in [&corner, &centre] {
                    (encoder.use_resource(buffer.as_resource(), ResourceUsage::READ, vertex))
                        .unwrap();
                }
                encoder.execute_commands_in_buffer(&icb, 0..2).unwrap();
            },
            [(SIDE / 2, SIDE / 2), (0, SIDE - 1)],
        )
    }

    /// The compute command's run, as `resident` says: the buffer's four words.
    fn compute(&self, resident: Resident) -> [u32; 4] {
        let device = &self.device;
        let state = self.kernel("irw_mark", true);
        let writer = self.kernel("irw_encode_dispatch", false);
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        commands.max_kernel_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&commands, 1, ResourceOptions::shared())
            .unwrap();
        let out = buffer_with(device, &[0u8; 16]).unwrap();
        let table = self.table(&icb, state.gpu_resource_id().0);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        Scene::write(&command_buffer, resident, (&writer, &table, &out), &icb);
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .use_resource(out.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let words = read_buffer(&out, 0, 16).unwrap();
        [0, 1, 2, 3].map(|at| {
            let word = &words[4 * at..4 * at + 4];
            u32::from_le_bytes([word[0], word[1], word[2], word[3]])
        })
    }
}

/// `target`'s texels at `at`.
fn texels<const N: usize>(target: &Texture, at: [(u64, u64); N]) -> [[u8; 4]; N] {
    at.map(|(x, y)| {
        let mut texel = [0u8; 4];
        let region = Region::new_2d(x, y, 1, 1);
        target.get_bytes(&mut texel, 4, 0, region, 0, 0).unwrap();
        texel
    })
}

/// Points `(x, y)` as `float4(x, y, 0, 1)`, in a new buffer.
fn floats(device: &Device, points: &[f32]) -> Buffer {
    let bytes: Vec<u8> = (points.chunks(2))
        .flat_map(|point| [point[0], point[1], 0.0, 1.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    buffer_with(device, &bytes).unwrap()
}

/// What Metal's run gave for each shape (the probe, on an M3 Max, macOS 27.0.1): whether the
/// command the kernel wrote ran, for a render command, an inherited state's point and a compute
/// command. Where no encoder of the command buffer made the buffer resident for writing, outside
/// Metal's contract for a write, Metal's render executions ran the CPU's records and its compute
/// execution the kernel's.
const METAL: [(Resident, [bool; 3]); 4] = [
    (Resident::Same, [true, true, true]),
    (Resident::After, [true, true, true]),
    (Resident::Before, [true, true, true]),
    (Resident::Nowhere, [false, false, true]),
];

/// Where the runtime's run is not Metal's, by shape and kind (0 render, 1 point, 2 compute): a
/// kernel's write to a buffer no encoder of its command buffer made resident for writing meets no
/// route (`IndirectCommandBuffer::gpu_written`), so the execution runs the CPU's records.
const APART: [(Resident, usize); 1] = [(Resident::Nowhere, 2)];

/// Each shape's execution runs what Metal's runs, but for [`APART`]: the command the kernel
/// wrote, with the state it took from its argument buffer, or the CPU's records alone.
#[test]
fn an_execution_after_its_writer_runs_what_metal_runs() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let (green, red) = ([0, 255, 0, 255], [255, 0, 0, 255]);
        let drawn = |texel: [u8; 4], what: &str| match texel {
            texel if texel == green => true,
            texel if texel == red => false,
            texel => panic!("{what}: {texel:?}"),
        };
        let marked = [0x100, 0x101, 0x102, 0x103];
        let mut seen = Vec::new();
        let mut steps = Vec::new();
        for (resident, _) in METAL {
            let render = drawn(scene.render(resident), "the render command");
            let before = device.work_counts().host_steps_recorded;
            let [centre, corner] = scene.points(resident);
            steps.push((resident, device.work_counts().host_steps_recorded - before));
            assert!(drawn(corner, "the CPU's triangle"), "{resident:?}");
            let point = drawn(centre, "the point");
            let words = scene.compute(resident);
            assert!(
                words == marked || words == [0; 4],
                "{resident:?}: {words:#x?}"
            );
            seen.push((resident, [render, point, words == marked]));
        }
        let mut expected = METAL;
        for (resident, kind) in APART {
            let (_, ran) = (expected.iter_mut())
                .find(|(shape, _)| *shape == resident)
                .unwrap();
            ran[kind] = !ran[kind];
        }
        let steps = format!("the inherited state's host steps: {steps:?}");
        assert_eq!(seen, expected, "{steps}");
    });
}

/// A render command the CPU wrote with a state that draws into another color format than the
/// pass's target, which the runtime refuses at commit, executed by a command buffer that waits for
/// an event and is committed first, its render encoder making the buffer resident for reading and
/// writing; then, on another queue, a command buffer whose kernel writes the command again (the
/// valid state, taken from its argument buffer) and which signals the event. Metal runs the
/// kernel's command (the probe `crates/metal2vulkan-conformance/probes/icb_cross_queue_writes.m`,
/// `written`, on an M3 Max, macOS 27.0.1: both complete, the target's centre green), so the first
/// commit holds its refusal for the host step (`WorkCounts::commit_refusals_held`), which records
/// the kernel's command once the writer completed: both complete, the centre green.
#[test]
fn a_refused_command_another_queue_writes_again_runs_what_it_wrote() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let state = scene.render_state("irw_vertex");
        let other = scene.render_state_in("irw_vertex", PixelFormat::RGBA16Float);
        let writer = scene.kernel("irw_encode_draw", false);
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW;
        commands.max_vertex_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&commands, 1, ResourceOptions::shared())
            .unwrap();
        let corners = floats(device, &[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
        let cpu = icb.indirect_render_command_at(0).unwrap();
        cpu.set_render_pipeline_state(&other).unwrap();
        cpu.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        cpu.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        let table = scene.table(&icb, state.gpu_resource_id().0);
        let (executing, writing) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let event = device.new_shared_event().unwrap();
        let before = device.work_counts().commit_refusals_held;
        let executes = executing.command_buffer().unwrap();
        executes
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let (target, pass) = scene.target();
        let mut encoder = executes.render_command_encoder(&pass).unwrap();
        let both = ResourceUsage::READ | ResourceUsage::WRITE;
        let stages = RenderStages::VERTEX | RenderStages::FRAGMENT;
        (encoder.use_resource(icb.as_resource(), both, stages)).unwrap();
        let vertex = RenderStages::VERTEX;
        (encoder.use_resource(corners.as_resource(), ResourceUsage::READ, vertex)).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        executes.commit().unwrap();
        let held = device.work_counts().commit_refusals_held - before;
        let writes = writing.command_buffer().unwrap();
        Scene::write(&writes, Resident::Same, (&writer, &table, &corners), &icb);
        writes
            .encode_signal_event(EventRef::from(&event), 1)
            .unwrap();
        writes.commit().unwrap();
        for command_buffer in [&writes, &executes] {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
        assert_eq!(held, 1, "the commit held its refusal for the host step");
        let [centre] = texels(&target, [(SIDE / 2, SIDE / 2)]);
        assert_eq!(centre, [0, 255, 0, 255], "the kernel's command ran");
    });
}
