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
//! at its depth, and the clear depth elsewhere, into a 2D texture and into slice 1 of a 2D array.

use super::degrade_notices::on_device;
use super::render_state::texels;
use super::{buffer_with, WAIT};
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
    CommandBufferStatus, ComputePipelineState, DepthStencilState, Device,
    IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
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

/// The commands a kernel encodes run from the range it wrote, each drawing its chunk's quad with
/// the buffers it set onto the encoder's depth-only state: the depth the pass stores is each kept
/// chunk's, and the clear depth where the kernel kept none, in a 2D texture and in slice 1 of a 2D
/// array.
#[test]
fn a_kernel_s_depth_only_commands_draw_through_the_range_it_wrote() {
    on_device(|device, _| {
        let library = library(device);
        let reset = kernel(device, &library, "dio_reset");
        let encode = kernel(device, &library, "dio_encode");
        let state = depth_only(device, &library);
        let less = less(device);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW_INDEXED;
        descriptor.inherit_pipeline_state = true;
        descriptor.inherit_buffers = false;
        descriptor.max_vertex_buffer_bind_count = 2;
        let icb = device
            .new_indirect_command_buffer(&descriptor, CHUNKS, ResourceOptions::shared())
            .unwrap();
        let encoder_args = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let positions = buffer_with(device, &positions()).unwrap();
        let indices = buffer_with(device, &indices()).unwrap();
        let camera: Vec<u8> = CAMERA.iter().flat_map(|f| f.to_le_bytes()).collect();
        let camera = buffer_with(device, &camera).unwrap();
        let visible = buffer_with(device, &VISIBLE.to_le_bytes()).unwrap();
        let range = buffer_with(device, &[0xFF; 8]).unwrap();
        let queue = device.new_command_queue().unwrap();
        for (slices, slice) in [(None, 0), (Some(3), 1)] {
            let target = target(device, slices);
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.reset_commands_in_buffer(&icb, 0..CHUNKS).unwrap();
            blit.end_encoding().unwrap();
            let mut writing = command_buffer.compute_command_encoder().unwrap();
            writing.set_compute_pipeline_state(&reset).unwrap();
            writing.set_buffer(Some(&range), 0, 0).unwrap();
            let one = Size::new(1, 1, 1);
            writing.dispatch_threads(one, one).unwrap();
            writing.set_compute_pipeline_state(&encode).unwrap();
            writing.set_buffer(Some(&encoder_args), 0, 0).unwrap();
            writing.set_buffer(Some(&positions), 0, 1).unwrap();
            writing.set_buffer(Some(&indices), 0, 2).unwrap();
            writing.set_buffer(Some(&camera), 0, 3).unwrap();
            writing.set_buffer(Some(&range), 4, 4).unwrap();
            writing.set_buffer(Some(&visible), 0, 5).unwrap();
            writing
                .use_resource(icb.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            let chunks = Size::new(CHUNKS, 1, 1);
            writing.dispatch_threads(chunks, chunks).unwrap();
            writing.end_encoding().unwrap();
            let mut pass = RenderPassDescriptor::new();
            pass.depth_attachment.texture = Some(target.clone());
            pass.depth_attachment.slice = slice;
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = 1.0;
            pass.depth_attachment.store_action = StoreAction::Store;
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_depth_stencil_state(&less).unwrap();
            encoder.set_depth_bias(0.0, 2.0, 0.0).unwrap();
            for buffer in [&positions, &indices, &camera] {
                encoder
                    .use_resource(
                        buffer.as_resource(),
                        ResourceUsage::READ,
                        RenderStages::VERTEX,
                    )
                    .unwrap();
            }
            encoder
                .execute_commands_in_buffer_indirect(&icb, &range, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                (command_buffer.status(), command_buffer.refusal()),
                (CommandBufferStatus::Completed, None),
                "slices {slices:?}"
            );
            assert_eq!(depths(&target, slice), expected(), "slices {slices:?}");
        }
    });
}
