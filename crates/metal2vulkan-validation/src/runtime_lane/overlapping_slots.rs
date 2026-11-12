//! Two texture arguments that function constants put on one slot, on the device the cfg selects,
//! under the validation layer: the owned fixture `overlapping_slots`, whose write texture `dst` is
//! at texture(1) and whose sampled texture `src` is at texture(n), n function constant 0.
//!
//! - n = 1 puts both on texture slot 1. Metal refuses the function when it makes it ("texture
//!   argument src has invalid location (<1,1>)"); the runtime makes the function, which checks
//!   constants when a pipeline compiles it, and refuses the compute pipeline state
//!   (`Compile(InvalidDescriptor(OverlappingArgument))`, naming `src`, parameter 1, at slot 1 of
//!   count 1).
//! - n = 0 makes the kernel: 4 threads over a 4 by 1 `R32Float` `src` holding 1 2 3 4, through
//!   Metal's default sampler state, write 1 2 3 4 into `dst`, as on the M3 Max
//!   (`overlapping_slots.m`, macOS 27.0.1: n = 1 refused at the function, n = 0 writes 1 2 3 4).

use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::error::{DescriptorError, Error as CompileError};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{DataType, PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, Error, PipelineOptions, Region,
    ResourceOptions, Size, Texture, TextureDescriptor,
};

/// The texels of `src`, and what Metal writes into `dst` for n = 0.
const TEXELS: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

/// The kernel's compute pipeline state with function constant 0 set to `n`, or its refusal.
fn state(device: &Device, n: u32) -> Result<ComputePipelineState, Error> {
    let bitcode = llvm_assemble(&engine_dir().join("overlapping_slots.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, n.to_le_bytes()),
    );
    let function = library
        .new_function_with_name_constant_values("overlapping_slots", &constants)
        .unwrap();
    device.new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
        &PipelineOptions::default(),
    )
}

/// A 4 by 1 `R32Float` texture holding `texels`.
fn texture(device: &Device, texels: [f32; 4]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::R32Float, 4, 1, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    let bytes: Vec<u8> = texels.iter().flat_map(|t| t.to_le_bytes()).collect();
    texture
        .replace_region(Region::new_2d(0, 0, 4, 1), 0, 0, &bytes, 16, 0)
        .unwrap();
    texture
}

/// n = 1 is refused as Metal refuses it, naming the later argument; n = 0 samples as Metal does.
#[test]
fn arguments_on_one_slot_are_refused_and_apart_answer_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let Err(Error::Compile(CompileError::InvalidDescriptor(refused))) = state(&device, 1)
        else {
            panic!("n = 1: made, though both textures are on slot 1");
        };
        assert_eq!(
            refused,
            DescriptorError::OverlappingArgument {
                param: 1,
                index: 1,
                count: 1
            }
        );
        let pipeline = state(&device, 0).unwrap();
        let src = texture(&device, TEXELS);
        let dst = texture(&device, [0.0; 4]);
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_texture(Some(&dst), 1).unwrap();
        encoder.set_texture(Some(&src), 0).unwrap();
        encoder.set_sampler_state(Some(&sampler), 0).unwrap();
        encoder
            .dispatch_threads(Size::new(4, 1, 1), Size::new(4, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let mut bytes = [0u8; 16];
        dst.get_bytes(&mut bytes, 16, 0, Region::new_2d(0, 0, 4, 1), 0, 0)
            .unwrap();
        let got: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        assert_eq!(got, TEXELS);
    }
    layer_reported_nothing_at_all(&messages);
}
