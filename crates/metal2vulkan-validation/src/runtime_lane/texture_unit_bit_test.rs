//! A texture unit a `switch` picks and a bit test of the unit that every unit it can pick passes,
//! the shape GL's AIR takes where it emulates a cube texture as a 2D array, on the device the cfg
//! selects, under the validation layer: the owned fixture `kernel_texture_unit_bit_test` samples
//! the texture of the unit each thread's `switch` picks as a 2D array, and answers what its header
//! derives. Its cube sample, on the way no unit takes, is not in the pipeline.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, Device, PipelineOptions, Region, ResourceOptions, Size, Texture,
    TextureDescriptor,
};

/// Each thread's `which`: the default (0 and 7), case 5 and case 29.
const WHICH: [u32; 4] = [0, 5, 29, 7];

/// What each thread answers: texture 0, 1, 2 and 0's texel (0, 1), folded (the fixture's header).
const ANSWERS: [f32; 4] = [137641.0, 248741.0, 359841.0, 137641.0];

/// Texture `k`: 2x2, one slice, texel (x, y) of channels 100(k + 1) + 10(2y + x) + c, c in 1..4.
fn texture(device: &Device, k: u32) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, 2, 2, false);
    d.texture_type = TextureType::Type2DArray;
    d.array_length = 1;
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    let texels: Vec<u8> = (0..4u32)
        .flat_map(|t| (1..=4u32).map(move |c| 100 * (k + 1) + 10 * t + c))
        .flat_map(|v| (v as f32).to_le_bytes())
        .collect();
    texture
        .replace_region(Region::new_2d(0, 0, 2, 2), 0, 0, &texels, 32, 0)
        .unwrap();
    texture
}

/// Each thread picks its unit's texture and samples it as a 2D array, as Metal runs the kernel.
#[test]
fn the_unit_a_switch_picks_samples_its_texture_as_a_2d_array() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let ll = fixtures_dir().join("kernel_texture_unit_bit_test.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = library
            .new_function_with_name("kernel_texture_unit_bit_test")
            .unwrap();
        let pipeline = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let textures = [0, 1, 2].map(|k| texture(&device, k));
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let which: Vec<u8> = WHICH.iter().flat_map(|w| w.to_le_bytes()).collect();
        let which = buffer_with(&device, &which).unwrap();
        let out = buffer_with(&device, &[0; 4 * WHICH.len()]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        for (texture, k) in textures.iter().zip(0u32..) {
            encoder.set_texture(Some(texture), k).unwrap();
        }
        encoder.set_sampler_state(Some(&sampler), 0).unwrap();
        encoder.set_buffer(Some(&which), 0, 0).unwrap();
        encoder.set_buffer(Some(&out), 0, 1).unwrap();
        let n = WHICH.len() as u64;
        encoder
            .dispatch_threads(Size::new(n, 1, 1), Size::new(n, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let words: Vec<f32> = read_buffer(&out, 0, 4 * n)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        assert_eq!(words, ANSWERS);
    }
    layer_reported_nothing_at_all(&messages);
}
