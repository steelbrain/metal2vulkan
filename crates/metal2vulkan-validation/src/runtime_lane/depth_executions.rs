//! A GPU-driven depth-only pass on the device the cfg selects, under the validation layer: the
//! shape of a culling renderer's depth prepass and of its shadow cascades. A kernel (the owned
//! fixture `icb_depth_indexed_encode`) resets an execution range, then keeps the chunks a mask
//! names, each taking the next command by an atomic increment of the range's length and encoding
//! there its own vertex buffers (the positions, and the address of its constant camera argument)
//! and an indexed triangle-list draw of its quad; a blit reset the commands before. The render
//! encoder inherits its pipeline state to the commands (a vertex function fetching a `float3`
//! attribute, no fragment function, a `Depth32Float` attachment), sets a depth/stencil state and a
//! depth bias, and executes the commands through the range the kernel wrote
//! (`execute_commands_in_buffer_indirect`). The depth the pass stores is each visible chunk's quad
//! at its depth, and the clear depth elsewhere: into a 2D texture, into slice 1 of a 2D array,
//! beside a later encoder of the command buffer that makes a shadowed texture resident, and over a
//! rasterization rate map. And a frame's two passes that execute the commands after the culling
//! wait for the culling alone: not for another queue's unrelated work submitted before them, nor
//! the second for the first.

use super::degrade_notices::on_device;
use super::rate_maps::descriptor as descriptor_of_rate_map;
use super::render_state::texels;
use super::{buffer_with, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, IndirectCommandType, PixelFormat, RenderStages, ResourceUsage, TextureType,
    TextureUsage, VertexFormat,
};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputePipelineState, DepthStencilState, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions,
    Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The chunks, one quad each over a quadrant of the target.
const CHUNKS: u64 = 4;

/// The target's width and height.
const SIZE: u64 = 4;

/// The chunks the kernel keeps: 0, 1 and 3.
const VISIBLE: u32 = 0b1011;

/// The camera's depth scale and offset: chunk `t`'s quad, at raw depth `t / 4`, lands at
/// `0.25 + t / 8`.
const CAMERA: [f32; 2] = [0.5, 0.25];

/// The depth state the pass tests and writes under.
fn less(device: &Device) -> DepthStencilState {
    let mut less = DepthStencilDescriptor::default();
    less.depth_compare_function = CompareFunction::Less;
    less.depth_write_enabled = true;
    device.new_depth_stencil_state(&less).unwrap()
}

/// The fixture's library.
fn library(device: &Device) -> Library {
    let bitcode = llvm_assemble(&engine_dir().join("icb_depth_indexed_encode.ll")).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// The state of the fixture's kernel `name`.
fn kernel(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
    let function = library.new_function_with_name(name).unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The depth-only state of `dio_vertex`: attribute 0 a `float3` of buffer 0's 12-byte layout,
/// no fragment function, a `Depth32Float` attachment, made for indirect command buffers.
fn depth_only(device: &Device, library: &Library) -> RenderPipelineState {
    let function = library.new_function_with_name("dio_vertex").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let mut vertex = VertexDescriptor::new();
    vertex
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(12)));
    vertex.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float3, 0, 0),
    );
    descriptor.vertex_descriptor = Some(vertex);
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    descriptor.support_indirect_command_buffers = true;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The positions: chunk `t`'s quad over quadrant (`t % 2`, `t / 2`) from the top left, its four
/// corners at raw depth `t / 4`.
fn positions() -> Vec<u8> {
    (0..CHUNKS)
        .flat_map(|t| {
            let (col, row) = ((t % 2) as f32, (t / 2) as f32);
            let (x0, top) = (col - 1.0, 1.0 - row);
            let z = t as f32 / 4.0;
            [
                [x0, top - 1.0, z],
                [x0 + 1.0, top - 1.0, z],
                [x0, top, z],
                [x0 + 1.0, top, z],
            ]
        })
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The 16-bit indices: chunk `t`'s two triangles from 6 * `t`.
fn indices() -> Vec<u8> {
    (0..CHUNKS as u16)
        .flat_map(|t| {
            let v = 4 * t;
            [v, v + 1, v + 2, v + 2, v + 1, v + 3]
        })
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// The depth the pass stores at pixel (`x`, `y`) from the top left: its chunk's where the kernel
/// keeps it, else the clear depth.
fn expected() -> Vec<f32> {
    (0..SIZE * SIZE)
        .map(|p| {
            let (x, y) = (p % SIZE, p / SIZE);
            let t = (y / 2) * 2 + x / 2;
            match VISIBLE >> t & 1 {
                1 => 0.25 + t as f32 / 8.0,
                _ => 1.0,
            }
        })
        .collect()
}

/// A `Shared` `Depth32Float` render target of `SIZE` by `SIZE`: a 2D texture, or a 2D array of
/// `slices`.
fn target(device: &Device, slices: Option<u64>) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::Depth32Float, SIZE, SIZE, false);
    if let Some(slices) = slices {
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = slices;
    }
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    device.new_texture(&descriptor).unwrap()
}

/// Slice `slice` of `texture`'s depth, as floats.
fn depths(texture: &Texture, slice: u64) -> Vec<f32> {
    let bytes = match slice {
        0 => texels(texture, 4),
        _ => {
            let mut out = vec![0u8; (SIZE * SIZE * 4) as usize];
            texture
                .get_bytes(
                    &mut out,
                    SIZE * 4,
                    SIZE * SIZE * 4,
                    Region::new_2d(0, 0, SIZE, SIZE),
                    0,
                    slice,
                )
                .unwrap();
            out
        }
    };
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// What a run varies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    /// A 2D target.
    Plain,
    /// Slice 1 of a 2D array of 3, as a shadow cascade's.
    Slice,
    /// A 2D target, and a later encoder of the command buffer that makes a shadowed texture
    /// resident, as a pass after the depth prepass makes its textures resident.
    LaterShadow,
    /// A 2D target over a rasterization rate map of half quality.
    RateMap,
}

/// A shadowed texture: an 8 by 8 `RGBA8Unorm` texture over a buffer whose rows lie 48 bytes
/// apart, 16 bytes past each row's texels.
fn shadowed(device: &Device) -> Texture {
    let truth = buffer_with(device, &[0; 8 * 48]).unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 8, 8, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    truth.new_texture(&descriptor, 0, 48).unwrap()
}

/// The fixture's states, the commands, and the buffers the kernel and the records name.
struct Workload {
    reset: ComputePipelineState,
    encode: ComputePipelineState,
    state: RenderPipelineState,
    less: DepthStencilState,
    icb: IndirectCommandBuffer,
    encoder_args: Buffer,
    positions: Buffer,
    indices: Buffer,
    camera: Buffer,
    visible: Buffer,
    range: Buffer,
}

impl Workload {
    fn new(device: &Device) -> Workload {
        let library = library(device);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW_INDEXED;
        descriptor.inherit_pipeline_state = true;
        descriptor.inherit_buffers = false;
        descriptor.max_vertex_buffer_bind_count = 2;
        let icb = device
            .new_indirect_command_buffer(&descriptor, CHUNKS, ResourceOptions::shared())
            .unwrap();
        let encoder_args = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let camera: Vec<u8> = CAMERA.iter().flat_map(|f| f.to_le_bytes()).collect();
        Workload {
            reset: kernel(device, &library, "dio_reset"),
            encode: kernel(device, &library, "dio_encode"),
            state: depth_only(device, &library),
            less: less(device),
            icb,
            encoder_args,
            positions: buffer_with(device, &positions()).unwrap(),
            indices: buffer_with(device, &indices()).unwrap(),
            camera: buffer_with(device, &camera).unwrap(),
            visible: buffer_with(device, &VISIBLE.to_le_bytes()).unwrap(),
            range: buffer_with(device, &[0xFF; 8]).unwrap(),
        }
    }

    /// The culling, into `command_buffer`: a blit resets the commands, then a kernel resets the
    /// range and one encodes the kept chunks' commands, taking each its place in the range.
    fn cull(&self, command_buffer: &CommandBuffer) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.reset_commands_in_buffer(&self.icb, 0..CHUNKS).unwrap();
        blit.end_encoding().unwrap();
        let mut writing = command_buffer.compute_command_encoder().unwrap();
        writing.set_compute_pipeline_state(&self.reset).unwrap();
        writing.set_buffer(Some(&self.range), 0, 0).unwrap();
        let one = Size::new(1, 1, 1);
        writing.dispatch_threads(one, one).unwrap();
        writing.set_compute_pipeline_state(&self.encode).unwrap();
        writing.set_buffer(Some(&self.encoder_args), 0, 0).unwrap();
        writing.set_buffer(Some(&self.positions), 0, 1).unwrap();
        writing.set_buffer(Some(&self.indices), 0, 2).unwrap();
        writing.set_buffer(Some(&self.camera), 0, 3).unwrap();
        writing.set_buffer(Some(&self.range), 4, 4).unwrap();
        writing.set_buffer(Some(&self.visible), 0, 5).unwrap();
        writing
            .use_resource(self.icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        let chunks = Size::new(CHUNKS, 1, 1);
        writing.dispatch_threads(chunks, chunks).unwrap();
        writing.end_encoding().unwrap();
    }

    /// A pass of `pass` into `command_buffer` that executes the commands through the range under
    /// the depth-only state.
    fn draw(&self, command_buffer: &CommandBuffer, pass: &RenderPassDescriptor) {
        let mut encoder = command_buffer.render_command_encoder(pass).unwrap();
        encoder.set_render_pipeline_state(&self.state).unwrap();
        encoder.set_depth_stencil_state(&self.less).unwrap();
        encoder.set_depth_bias(0.0, 2.0, 0.0).unwrap();
        for buffer in [&self.positions, &self.indices, &self.camera] {
            encoder
                .use_resource(
                    buffer.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::VERTEX,
                )
                .unwrap();
        }
        encoder
            .execute_commands_in_buffer_indirect(&self.icb, &self.range, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
    }
}

/// A pass that clears `target`'s depth at slice `slice` and stores it.
fn depth_pass(target: &Texture, slice: u64) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    pass.depth_attachment.texture = Some(target.clone());
    pass.depth_attachment.slice = slice;
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 1.0;
    pass.depth_attachment.store_action = StoreAction::Store;
    pass
}

/// The commands a kernel encodes run from the range it wrote, each drawing its chunk's quad with
/// the buffers it set onto the encoder's depth-only state: the depth the pass stores is each kept
/// chunk's, and the clear depth where the kernel kept none, in a 2D texture, in slice 1 of a 2D
/// array, beside a later encoder's shadowed texture, and over a rate map.
#[test]
fn a_kernel_s_depth_only_commands_draw_through_the_range_it_wrote() {
    on_device(|device, _| {
        let work = Workload::new(device);
        let queue = device.new_command_queue().unwrap();
        let mut runs = Vec::new();
        for variant in [
            Variant::Plain,
            Variant::Slice,
            Variant::LaterShadow,
            Variant::RateMap,
        ] {
            let (slices, slice) = match variant {
                Variant::Slice => (Some(3), 1),
                _ => (None, 0),
            };
            let target = target(device, slices);
            let command_buffer = queue.command_buffer().unwrap();
            work.cull(&command_buffer);
            let mut pass = depth_pass(&target, slice);
            if variant == Variant::RateMap {
                let map = descriptor_of_rate_map(SIZE as u32, SIZE as u32, &[(0.5, 0.5)]);
                pass.rasterization_rate_map =
                    Some(device.new_rasterization_rate_map(&map).unwrap());
            }
            work.draw(&command_buffer, &pass);
            let later = (variant == Variant::LaterShadow).then(|| shadowed(device));
            if let Some(texture) = &later {
                let mut after = command_buffer.compute_command_encoder().unwrap();
                after
                    .use_resource(texture.as_resource(), ResourceUsage::READ)
                    .unwrap();
                after.end_encoding().unwrap();
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let run = (
                variant,
                command_buffer.status(),
                command_buffer.refusal(),
                depths(&target, slice),
            );
            eprintln!("{run:?}");
            runs.push(run);
        }
        for (variant, status, refusal, depths) in runs {
            assert_eq!(
                (status, refusal),
                (CommandBufferStatus::Completed, None),
                "{variant:?}"
            );
            if variant != Variant::RateMap {
                assert_eq!(depths, expected(), "{variant:?}");
            }
        }
    });
}

/// A frame's executions after the culling wait for the culling alone. One queue commits the
/// culling; another commits unrelated work (a fill of a buffer of its own), submitted before the
/// frame is committed; then the first commits the frame: two passes, each into a target of its
/// own, that execute the commands through the range. Each pass's host step waits for the batches
/// that wrote the records and the range, so both gates leave work submitted before them out (the
/// unrelated fill, and for the second the first pass's batch): `host_step_gates_narrowed` counts
/// two, against none where a gate is every queue's last submission. Both targets hold each kept
/// chunk's depth.
#[test]
fn a_frame_s_executions_wait_for_the_culling_alone() {
    on_device(|device, _| {
        let work = Workload::new(device);
        let queue = device.new_command_queue().unwrap();
        let other = device.new_command_queue().unwrap();
        let culling = queue.command_buffer().unwrap();
        work.cull(&culling);
        culling.commit().unwrap();
        let filled = buffer_with(device, &[0; 1 << 16]).unwrap();
        let unrelated = other.command_buffer().unwrap();
        let mut fill = unrelated.blit_command_encoder().unwrap();
        fill.fill_buffer(&filled, 0..1 << 16, 0x5A).unwrap();
        fill.end_encoding().unwrap();
        unrelated.commit().unwrap();
        unrelated.wait_until_scheduled().unwrap();
        // The culling's own host steps (a dispatch whose pipeline was still being made) are
        // behind it once it is scheduled.
        culling.wait_until_scheduled().unwrap();
        let before = device.work_counts();
        let targets = [target(device, None), target(device, None)];
        let frame = queue.command_buffer().unwrap();
        for target in &targets {
            work.draw(&frame, &depth_pass(target, 0));
        }
        frame.commit().unwrap();
        frame.wait_until_completed_for(WAIT).unwrap();
        let after = device.work_counts();
        unrelated.wait_until_completed_for(WAIT).unwrap();
        let narrowed = after.host_step_gates_narrowed - before.host_step_gates_narrowed;
        let steps = after.host_steps_recorded - before.host_steps_recorded;
        let drawn: Vec<Vec<f32>> = targets.iter().map(|target| depths(target, 0)).collect();
        let statuses = [culling.status(), unrelated.status(), frame.status()];
        eprintln!(
            "statuses {statuses:?}, refusal {:?}, steps {steps}, gates narrowed {narrowed}, \
             depths {drawn:?}",
            frame.refusal()
        );
        assert_eq!(statuses, [CommandBufferStatus::Completed; 3]);
        assert_eq!(frame.refusal(), None);
        assert_eq!(drawn, [expected(), expected()]);
        assert_eq!((steps, narrowed), (2, 2), "host steps, gates narrowed");
    });
}
