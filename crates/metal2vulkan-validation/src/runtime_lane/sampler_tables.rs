//! A program-scope table of constexpr samplers read at an index only the run knows, on the device
//! the cfg selects, under the validation layer: the owned fixture `sampler_table`'s
//! `sampler_table_slot` (a texture slot) and `sampler_table_held` (a texture held in an argument
//! buffer) answer what the M3 Max answers (`sampler_table.m`).
//!
//! The texture is 4 by 1 `RGBA8Unorm`: red, green, blue, white. Thread `i = 4c + s` samples at
//! `x = X[c]` (texel units -0.5, 0.75, 3.75, 4.5 and 2.5: outside the texture on both sides, and
//! a quarter of the way between texel centers) through `samplers[s]` (repeat nearest, clamp
//! nearest, repeat linear, clamp linear), so each thread's texel says which sampler sampled. Every
//! answer is a multiple of a quarter, which a linear filter's weights of a quarter give exactly.

use super::{
    buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with, Messages,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region, ResourceOptions,
    Size, Texture, TextureDescriptor,
};
use std::path::Path;

/// The threads: five coordinates through four samplers.
const THREADS: usize = 20;

/// The coordinates, in normalized units.
const X: [f32; 5] = [-0.125, 0.1875, 0.9375, 1.125, 0.625];

const ONE: u32 = 0x3f80_0000;
const ZERO: u32 = 0;
const QUARTER: u32 = 0x3e80_0000;
const THREE_QUARTERS: u32 = 0x3f40_0000;

/// What the M3 Max answers (macOS 27.0.1), thread by thread, as words: both kernels alike.
#[rustfmt::skip]
const METAL: [[u32; 4]; THREADS] = [
    [ONE, ONE, ONE, ONE],
    [ONE, ZERO, ZERO, ONE],
    [ONE, ONE, ONE, ONE],
    [ONE, ZERO, ZERO, ONE],
    [ONE, ZERO, ZERO, ONE],
    [ONE, ZERO, ZERO, ONE],
    [THREE_QUARTERS, QUARTER, ZERO, ONE],
    [THREE_QUARTERS, QUARTER, ZERO, ONE],
    [ONE, ONE, ONE, ONE],
    [ONE, ONE, ONE, ONE],
    [ONE, THREE_QUARTERS, THREE_QUARTERS, ONE],
    [ONE, ONE, ONE, ONE],
    [ONE, ZERO, ZERO, ONE],
    [ONE, ONE, ONE, ONE],
    [ONE, ZERO, ZERO, ONE],
    [ONE, ONE, ONE, ONE],
    [ZERO, ZERO, ONE, ONE],
    [ZERO, ZERO, ONE, ONE],
    [ZERO, ZERO, ONE, ONE],
    [ZERO, ZERO, ONE, ONE],
];

/// The fixture's `entry`, made into a pipeline.
fn pipeline(device: &Device, entry: &str) -> ComputePipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/sampler_table.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = library.new_function_with_name(entry).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// The 4 by 1 texture: red, green, blue, white.
fn texture(device: &Device) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 1, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    let texels = [
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];
    texture
        .replace_region(Region::new_2d(0, 0, 4, 1), 0, 0, &texels, 16, 0)
        .unwrap();
    texture
}

/// Each thread's four words that `pipeline` writes, its texture at slot 0 and held in buffer 3.
fn dispatched(device: &Device, pipeline: &ComputePipelineState) -> Vec<[u32; 4]> {
    let texture = texture(device);
    let which: Vec<u8> = (0..THREADS as u32)
        .flat_map(|i| (i % 4).to_le_bytes())
        .collect();
    let xs: Vec<u8> = (0..THREADS).flat_map(|i| X[i / 4].to_le_bytes()).collect();
    let out = buffer_with(device, &[0x5a; 16 * THREADS]).unwrap();
    let which = buffer_with(device, &which).unwrap();
    let xs = buffer_with(device, &xs).unwrap();
    let held = buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline).unwrap();
    for (buffer, index) in [(&out, 0), (&which, 1), (&xs, 2), (&held, 3)] {
        encoder.set_buffer(Some(buffer), 0, index).unwrap();
    }
    encoder.set_texture(Some(&texture), 0).unwrap();
    encoder
        .use_resource(texture.as_resource(), ResourceUsage::READ)
        .unwrap();
    encoder
        .dispatch_threads(
            Size::new(THREADS as u64, 1, 1),
            Size::new(THREADS as u64, 1, 1),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, 16 * THREADS as u64)
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|v| std::array::from_fn(|k| u32::from_le_bytes(v[4 * k..][..4].try_into().unwrap())))
        .collect()
}

/// Both kernels sample through the sampler the run's index picks from the table, as Metal does,
/// for a texture slot and for a texture held in an argument buffer.
#[test]
fn a_sampler_table_read_at_a_dynamic_index_samples_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        for entry in ["sampler_table_slot", "sampler_table_held"] {
            let words = dispatched(&device, &pipeline(&device, entry));
            for (i, (got, want)) in words.iter().zip(METAL.iter()).enumerate() {
                assert_eq!(got, want, "{entry}: thread {i}");
            }
        }
    }
    layer_reported_nothing_at_all(&messages);
}
