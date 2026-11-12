//! A sampler state's `lodBias` on the device the cfg selects, under the validation layer: Metal
//! samples nothing differently for it on the families the runtime claims (an Apple M3 Max on macOS
//! 27.0.1, Apple9: every sample kind, biases from -1000 to 1000, NaN and the infinities), and makes
//! another state for another bias, one state for `0` and `-0`, and one for two NaNs of one
//! encoding.
//!
//! The kernel is `mip_level_sampling`'s `mip_bias_kernel` (`validation/fixtures/public/engine`):
//! each thread samples the runtime sampler at slot 0 at its own coordinate with its own shader
//! bias, a kernel's default level 0 shifted by it. The texture is `r32Float`, of five levels whose
//! level `L` holds `L` in every texel, so a linear mip filter's sample states the level it read.

use super::mip_levels::{texture, LEVELS};
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{SamplerMinMagFilter, SamplerMipFilter, TextureType};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ComputePipelineState, Device, PipelineOptions, ResourceOptions, SamplerState, Size, Texture,
};

/// The shader biases each thread samples at, one thread each: whole levels, levels between them,
/// below the first and past the last.
const SHADER_BIASES: [f32; 9] = [0.0, 0.5, 1.0, 1.25, 2.0, 3.0, 4.0, 6.0, -1.0];

/// The sampler biases, each a state of its own beside the unbiased one: small ones of both signs,
/// Apple's stated range's ends and past them, values past every device's `maxSamplerLodBias`, a
/// NaN and an infinity.
const SAMPLER_BIASES: [f32; 10] = [
    1.0,
    -1.0,
    2.5,
    15.999,
    -16.0,
    17.0,
    -17.0,
    100.0,
    f32::NAN,
    f32::INFINITY,
];

/// The kernel on `device`.
fn pipeline(device: &Device) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("mip_level_sampling.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("mip_bias_kernel").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A sampler state of nearest filters, mip filter `mip` and LOD bias `bias`.
fn sampler(device: &Device, mip: SamplerMipFilter, bias: f32) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = SamplerMinMagFilter::Nearest;
    descriptor.mag_filter = SamplerMinMagFilter::Nearest;
    descriptor.mip_filter = mip;
    descriptor.lod_bias = bias;
    device.new_sampler_state(&descriptor).unwrap()
}

/// What each thread reads of `texture` through `sampler` at its shader bias.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: &SamplerState,
) -> Vec<f32> {
    let params: Vec<u8> = SHADER_BIASES
        .iter()
        .flat_map(|&b| [0.3f32, 0.6, b, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let params = buffer_with(device, &params).unwrap();
    let length = (SHADER_BIASES.len() * 16) as u64;
    let output = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(sampler), 0).unwrap();
    encoder.set_buffer(Some(&params), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    let threads = SHADER_BIASES.len() as u64;
    encoder
        .dispatch_threads(Size::new(threads, 1, 1), Size::new(threads, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    // The red channel of each thread's `float4`.
    read_buffer(&output, 0, length)
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(c[..4].try_into().unwrap()))
        .collect()
}

/// Every sample through a state with a LOD bias reads what the unbiased state reads, bit for bit,
/// under a linear and a nearest mip filter: the level the shader bias alone selects, which the
/// unbiased state reads at each whole level. A state of another bias is another state, `-0` is
/// `0`'s, and a NaN bias's state is the next NaN's. Prints the samples checked.
#[test]
fn a_sampler_s_lod_bias_reads_the_unbiased_level() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let levels = texture(&device, TextureType::Type2D, |l, _, _, _| l as f32);
        let state = pipeline(&device);
        let mut checked = 0;
        for mip in [SamplerMipFilter::Linear, SamplerMipFilter::Nearest] {
            let unbiased = sampler(&device, mip, 0.0);
            let want = run(&device, &state, &levels, &unbiased);
            // The kernel reads levels: each whole shader bias's level, the texture's levels
            // bounding it.
            for (&b, &w) in SHADER_BIASES.iter().zip(&want) {
                if b.fract() == 0.0 {
                    let level = b.clamp(0.0, (LEVELS - 1) as f32);
                    assert_eq!(w, level, "{name}: {mip:?}, shader bias {b}");
                }
            }
            for bias in SAMPLER_BIASES {
                let biased = sampler(&device, mip, bias);
                assert_ne!(
                    biased.gpu_resource_id(),
                    unbiased.gpu_resource_id(),
                    "{name}: bias {bias} is another state"
                );
                let got = run(&device, &state, &levels, &biased);
                for ((&b, g), w) in SHADER_BIASES.iter().zip(&got).zip(&want) {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "{name}: {mip:?}, sampler bias {bias}, shader bias {b}: {g} against \
                         the unbiased {w}"
                    );
                    checked += 1;
                }
            }
            let zero = sampler(&device, mip, -0.0);
            assert_eq!(zero.gpu_resource_id(), unbiased.gpu_resource_id());
            let (nan, again) = (
                sampler(&device, mip, f32::NAN),
                sampler(&device, mip, f32::NAN),
            );
            assert_eq!(nan.gpu_resource_id(), again.gpu_resource_id());
        }
        println!("LOD-BIAS {name}: {checked} biased samples read the unbiased level");
    }
    layer_reported_nothing_at_all(&messages);
}
