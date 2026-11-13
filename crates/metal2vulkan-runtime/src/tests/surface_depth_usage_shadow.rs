//! A surface depth texture's usage shadow (`texture/usage_shadow.rs`), on the device the GPU cfg
//! selects, under the validation layer: a `Private` 16x16 `Depth32Float_Stencil8` texture over
//! plane 0 of a surface's bytes, made without `ShaderRead`, on a device where `SAMPLED` costs its
//! shadow something (`DeviceShared::sampled_costs_forced`), so its shadow lacks it and a kernel's
//! read of its depth binds a usage shadow of the bytes. The plane holds the depth alone: the usage
//! shadow copies that aspect, one aspect a region as Vulkan copies
//! (`VUID-VkBufferImageCopy-aspectMask-00212`), and the kernel reads the plane's depths. One usage
//! shadow is made (`DeviceShared::usage_shadows_made`), a read of the texture's `X32_Stencil8`
//! view reads 0, and the layer reports nothing. Printed (`SURFACE-DEPTH-USAGE-SHADOW`): what the
//! reads found.

use super::blit_holds::{options, Scratch, WAIT};
use super::encoder_holds::library;
use super::layer_claim::Watched;
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, DeviceOptions, IoSurfacePlane,
    PipelineOptions, ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::Ordering;

/// The texture's side.
const SIDE: u64 = 16;

/// The depth the plane holds at texel `n` (row-major).
fn depth(n: u64) -> f32 {
    (n + 1) as f32 / 512.0
}

/// What `kernel` writes reading `texture` at each texel, `bytes` a texel, into a new buffer.
fn read(device: &Device, kernel: &ComputePipelineState, texture: &Texture, bytes: u64) -> Vec<u8> {
    let out = (device.new_buffer(SIDE * SIDE * bytes, ResourceOptions::shared())).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(kernel).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    (encoder.dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    bytes_of(&out)
}

/// A copy of `buffer`'s bytes, which no command uses now.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let length = buffer.length() as usize;
    // SAFETY: a `Shared` buffer's contents span its length, and no command uses them now.
    unsafe { std::slice::from_raw_parts(buffer.contents().unwrap().as_ptr(), length) }.to_vec()
}

#[test]
fn a_surface_depth_texture_s_usage_shadow_copies_the_plane_s_aspect_alone() {
    let scratch = Scratch::new("surface-depth-usage-shadow");
    let watched = Watched::default();
    let options = DeviceOptions {
        layers: watched.layers(),
        messages: Some(watched.sink(|_| {})),
        ..options(&scratch.0)
    };
    let device = Device::system_default_with(&options).unwrap();
    (device.shared().sampled_costs_forced).store(true, Ordering::Relaxed);
    let scout = library(&device, "engine/format_scout.ll");
    let kernel = |name: &str| {
        let function = scout.new_function_with_name(name).unwrap();
        (device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default()))
            .unwrap()
    };
    let (read_d, read_u) = (kernel("scout_read_d"), kernel("scout_read_u"));
    let surface = device.new_buffer(16384, ResourceOptions::shared()).unwrap();
    let depths: Vec<f32> = (0..SIDE * SIDE).map(depth).collect();
    // SAFETY: the new buffer's mapped bytes, which no command uses yet.
    unsafe {
        let at = surface.contents().unwrap().as_ptr();
        std::ptr::copy_nonoverlapping(depths.as_ptr().cast::<u8>(), at, depths.len() * 4);
    }
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::Depth32FloatStencil8, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let plane = IoSurfacePlane::new(0, SIDE * 4, true);
    let texture = (surface.new_texture_with_iosurface_plane(&descriptor, plane)).unwrap();
    let made = || (device.shared().usage_shadows_made).load(Ordering::Acquire);
    let before = made();
    let seen: Vec<f32> = (read(&device, &read_d, &texture, 4).chunks(4))
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let shadows = made() - before;
    let view = texture.new_texture_view(PixelFormat::X32Stencil8).unwrap();
    let stencils: Vec<u32> = (read(&device, &read_u, &view, 16).chunks(16))
        .map(|texel| u32::from_le_bytes(texel[..4].try_into().unwrap()))
        .collect();
    let depths_read = seen == depths;
    let zero = stencils.iter().all(|&stencil| stencil == 0);
    println!(
        "SURFACE-DEPTH-USAGE-SHADOW {}: usage shadows made {shadows}, depths the plane's \
         {depths_read} (first {:?}), stencils zero {zero}",
        device.name(),
        &seen[..4]
    );
    assert_eq!(shadows, 1, "the read took no usage shadow");
    assert!(depths_read, "the depths read are not the plane's");
    assert!(zero, "a stencil read is not zero: {:?}", &stencils[..8]);
    drop((view, texture, surface, device));
    watched.layer_reported_nothing("a surface depth texture's usage shadow");
}
