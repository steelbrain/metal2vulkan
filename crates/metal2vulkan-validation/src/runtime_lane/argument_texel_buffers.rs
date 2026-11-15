//! Texture buffers read through argument buffers, on the device the cfg selects, under the
//! validation layer: descriptor sets encoded as tier-2 argument buffers in the shape SPIRV-Cross
//! writes for a Vulkan layer that keeps each set in one (the owned fixture
//! `abt_descriptor_sets`), whose texel buffers are `texture_buffer` members. Such a member's
//! handle is a set-1 element of a texel-buffer class (`lower::texture`'s heap handles), which
//! every live texture buffer's texel buffer view holds at its resource ID (`heap_sets.rs`).

use super::degrade_notices::on_device_with;
use super::{at_plan, buffer_with, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBufferStatus, Device, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, ResourceOptions, SamplerStateOptions, StoreAction, TextureDescriptor,
};

/// The target's side, in pixels.
const SIDE: u64 = 4;

/// The texel buffer's texels: texel `i` is `(i, 10 + i, 20 + i, 30 + i)`.
const TEXELS: u64 = 8;

/// The frame's `offset.x`: the vertex function reads texel `FIRST + vid`, and the index it hands
/// on is `FIRST + 3 + vid`.
const FIRST: u64 = 1;

/// The frame's tint.
const TINT: [f32; 4] = [0.5, 0.25, 2.0, 1.0];

/// The albedo texture's one texel, `RGBA8Unorm`: `(1, 0, 1, 0)`.
const ALBEDO: [u8; 4] = [255, 0, 255, 0];

/// Texel `i` of the texel buffer.
fn texel(i: u64) -> [f32; 4] {
    let i = i as f32;
    [i, 10.0 + i, 20.0 + i, 30.0 + i]
}

/// A buffer of `words`, little-endian.
fn words<const N: usize>(device: &Device, words: &[[u8; N]]) -> Buffer {
    buffer_with(device, &words.concat()).unwrap()
}

/// `abt_vert_bones` reads a texel of set 0's texel buffer through its argument buffer into flat
/// varyings, and `abt_frag_bones` reads another of it, with set 1's 2D texture through the
/// sampler beside it: the pipeline is made (no `ArgumentBufferTextures` refusal), and every pixel
/// is texel[FIRST] + texel[FIRST + 3] · tint + albedo, the texels the texture buffer holds (the
/// first vertex provokes, as Metal's does).
#[test]
fn a_texture_buffer_in_an_argument_buffer_is_read_in_both_stages() {
    on_device_with(at_plan, |device, _| {
        let ll = engine_dir().join("abt_descriptor_sets.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(function("abt_vert_bones"));
        descriptor.fragment = Some(function("abt_frag_bones"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
        let state = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap_or_else(|e| panic!("the pipeline of a texel buffer argument: {e}"));

        // Set 0's resources: the frame (tint, offset), and the texture buffer over its texels.
        let frame: Vec<[u8; 4]> = TINT
            .into_iter()
            .chain([FIRST as f32, 0.0, 0.0, 0.0])
            .map(f32::to_le_bytes)
            .collect();
        let frame = words(device, &frame);
        let texels: Vec<[u8; 4]> = (0..TEXELS).flat_map(texel).map(f32::to_le_bytes).collect();
        let texels = words(device, &texels);
        let bones = TextureDescriptor::texture_buffer(
            PixelFormat::RGBA32Float,
            TEXELS,
            texels.resource_options(),
            TextureUsage::SHADER_READ,
        );
        let bones = texels.new_texture(&bones, 0, 16 * TEXELS).unwrap();

        // Set 1's resources: the albedo texture and the sampler beside it.
        let mut albedo = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
        albedo.usage = TextureUsage::SHADER_READ;
        let albedo = device.new_texture(&albedo).unwrap();
        albedo
            .replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &ALBEDO, 4, 0)
            .unwrap();
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let sampler = device
            .new_sampler_state_with_options(&SamplerDescriptor::new(), options)
            .unwrap();

        // The argument buffers, each member's address or resource ID at its field: set 0's
        // `{frame, shadow map, its sampler, bones, padding}` and set 1's `{albedo, its sampler,
        // env, its sampler, layers[8], materials, scratch}`, the members neither stage reads nil.
        let id = |n: u64| n.to_le_bytes();
        let set0 = words(
            device,
            &[
                id(frame.gpu_address().0),
                id(0),
                id(0),
                id(bones.gpu_resource_id().0),
                id(0),
            ],
        );
        let mut set1 = vec![id(0); 14];
        set1[0] = id(albedo.gpu_resource_id().0);
        set1[1] = id(sampler.gpu_resource_id().0);
        let set1 = words(device, &set1);

        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, SIDE, SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let target = device.new_texture(&target).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(-1.0, -1.0, -1.0, -1.0);

        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&set0), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&set0), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(&set1), 0, 1).unwrap();
        let both = RenderStages::VERTEX.union(RenderStages::FRAGMENT);
        encoder
            .use_resource(frame.as_resource(), ResourceUsage::READ, both)
            .unwrap();
        encoder
            .use_resource(bones.as_resource(), ResourceUsage::READ, both)
            .unwrap();
        encoder
            .use_resource(
                albedo.as_resource(),
                ResourceUsage::READ,
                RenderStages::FRAGMENT,
            )
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);

        let mut bytes = vec![0u8; (16 * SIDE * SIDE) as usize];
        target
            .get_bytes(
                &mut bytes,
                16 * SIDE,
                0,
                Region::new_2d(0, 0, SIDE, SIDE),
                0,
                0,
            )
            .unwrap();
        let (bone, read) = (texel(FIRST), texel(FIRST + 3));
        let want: [f32; 4] =
            std::array::from_fn(|c| bone[c] + read[c] * TINT[c] + f32::from(ALBEDO[c]) / 255.0);
        for (i, pixel) in bytes.as_chunks::<16>().0.iter().enumerate() {
            let got: [f32; 4] = std::array::from_fn(|c| {
                f32::from_le_bytes([
                    pixel[4 * c],
                    pixel[4 * c + 1],
                    pixel[4 * c + 2],
                    pixel[4 * c + 3],
                ])
            });
            assert_eq!(got, want, "pixel {i}");
        }
    });
}
