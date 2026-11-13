//! What a device's recovery carries of its sampler states (`sampler.rs`): a state made before it is
//! the same state on the new `VkDevice` (an equal request gets it, a slot bind samples through it,
//! and an argument buffer naming its resource ID reads it), as the recovery's notice counts, and
//! the lost `VkDevice` goes while the state lives.
//!
//! The state repeats on every axis, which the neutral sampler, clamped to the edge, does not: a
//! read past the texture's edge tells them apart.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).
//! The kernels are `fpv_swz_sample` and `argument_sampler_sample` of the owned fixtures
//! `engine/fpv_swizzled_reads.ll` and `engine/tail_argument_samplers.ll` (`llvm-as`).

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch};
use super::device_recovery_carry::live_vk_devices;
use super::encoder_holds::library;
use crate::{
    CommandBufferStatus, CommandQueue, ComputePipelineState, Device, Error, PipelineOptions,
    Region, ResourceOptions, SamplerState, SamplerStateOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, SamplerAddressMode, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The texture's side, in texels.
const SIDE: u64 = 4;

/// The threads of a dispatch: thread `g` samples at `(c, c)`, `c = 0.25 g + 0.125`, the centre of
/// texel `g` along each axis, the last four past the edge.
const THREADS: u64 = 8;

/// What a repeating sampler reads at each thread's position: texel `(i, i)`, `i = g % 4`, which
/// holds `11 i`. The neutral sampler reads texel `(3, 3)` past the edge.
const REPEATED: [f32; 8] = [0.0, 11.0, 22.0, 33.0, 0.0, 11.0, 22.0, 33.0];

/// How a dispatch reaches the sampler state.
#[derive(Clone, Copy, Debug)]
enum Route {
    /// Bound in sampler slot 0 (`fpv_swz_sample`).
    Slot,
    /// Named by its resource ID in an argument buffer beside the texture's
    /// (`argument_sampler_sample`).
    Argument,
}

/// A nearest sampler state repeating on every axis, which an argument buffer may name.
fn repeating(device: &Device) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.s_address_mode = SamplerAddressMode::Repeat;
    descriptor.t_address_mode = SamplerAddressMode::Repeat;
    descriptor.r_address_mode = SamplerAddressMode::Repeat;
    let options = SamplerStateOptions {
        support_argument_buffers: true,
        ..SamplerStateOptions::default()
    };
    device
        .new_sampler_state_with_options(&descriptor, options)
        .unwrap()
}

/// A `Shared` `R32Float` texture of [`SIDE`] by [`SIDE`] texels, texel `(x, y)` holding `10 y + x`.
fn texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Float, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|at| ((10 * (at / SIDE) + at % SIDE) as f32).to_le_bytes())
        .collect();
    (texture.replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &bytes, 4 * SIDE, 0)).unwrap();
    texture
}

/// The compute state of `kernel` of the fixture at `path`.
fn kernel(device: &Device, path: &str, kernel: &str) -> ComputePipelineState {
    let function = library(device, path)
        .new_function_with_name(kernel)
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    (device.new_compute_pipeline_state(&descriptor, &PipelineOptions::default())).unwrap()
}

/// A shared buffer of `device` holding `bytes`.
fn holding(device: &Device, bytes: &[u8]) -> crate::Buffer {
    (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap()
}

/// What `queue`'s dispatch of [`THREADS`] threads reads of `texture` through `sampler` by `route`,
/// completed: each thread's red. Refused as the encoding refuses the state.
fn sampled(
    queue: &CommandQueue,
    route: Route,
    texture: &Texture,
    sampler: &SamplerState,
) -> Result<Vec<f32>, Error> {
    let device = queue.device();
    let out = device.new_buffer(16 * THREADS, ResourceOptions::shared())?;
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder()?;
    match route {
        Route::Slot => {
            let state = kernel(device, "engine/fpv_swizzled_reads.ll", "fpv_swz_sample");
            let positions: Vec<u8> = (0..THREADS)
                .map(|g| 0.25 * g as f32 + 0.125)
                .flat_map(|c| [c, c])
                .flat_map(f32::to_le_bytes)
                .collect();
            encoder.set_compute_pipeline_state(&state)?;
            encoder.set_texture(Some(texture), 0)?;
            encoder.set_sampler_state(Some(sampler), 0)?;
            encoder.set_buffer(Some(&out), 0, 0)?;
            encoder.set_buffer(Some(&holding(device, &positions)), 0, 1)?;
        }
        Route::Argument => {
            let state = kernel(
                device,
                "engine/tail_argument_samplers.ll",
                "argument_sampler_sample",
            );
            let ids = [texture.gpu_resource_id().0, sampler.gpu_resource_id().0];
            let ids: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            encoder.set_compute_pipeline_state(&state)?;
            encoder.use_resource(texture.as_resource(), ResourceUsage::READ)?;
            encoder.set_buffer(Some(&holding(device, &ids)), 0, 0)?;
            encoder.set_buffer(Some(&out), 0, 1)?;
        }
    }
    encoder.dispatch_threads(Size::new(THREADS, 1, 1), Size::new(THREADS, 1, 1))?;
    encoder.end_encoding()?;
    command_buffer.commit()?;
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None),
        "{route:?}"
    );
    let pointer = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed work wrote and nothing else uses.
    let bytes = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), (16 * THREADS) as usize) };
    Ok((bytes.as_chunks::<16>().0.iter())
        .map(|texel| f32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]))
        .collect())
}

/// A sampler state made before a recovery, which an argument buffer named there (so the lost
/// generation's heap sets hold its Vulkan samplers), is the same state after it: an equal request
/// gets it, its resource ID is kept, and a slot bind and an argument buffer of the new generation
/// read through it what they read before, as the recovery's notice counts. The lost `VkDevice`
/// goes while the state lives.
#[test]
fn a_recovery_carries_a_sampler_state() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let queue = device.new_command_queue().unwrap();
    let sampler = repeating(&device);
    let id = sampler.gpu_resource_id();
    {
        let before = texture(&device);
        for route in [Route::Slot, Route::Argument] {
            assert_eq!(sampled(&queue, route, &before, &sampler).unwrap(), REPEATED);
        }
    }
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(said[0].contains(", and 1 sampler states;"), "{said:?}");
    assert!(sampler.device().generation_serves());
    assert_eq!(repeating(&device), sampler, "an equal request's state");
    assert_eq!(sampler.gpu_resource_id(), id);
    let after = texture(&device);
    for route in [Route::Slot, Route::Argument] {
        assert_eq!(
            sampled(&queue, route, &after, &sampler).unwrap(),
            REPEATED,
            "{route:?}"
        );
    }
    drop(after);
    assert_eq!(live_vk_devices(&device, 1), 1);
}
